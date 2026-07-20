/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Manager Owner Actor
//!
//! Controllerless-singleton coordinator that fronts the per-proc `rdma_manager`
//! service. Given a target proc mesh it get-or-spawns a mesh of
//! [`RdmaManagerActor`]s (creating or reusing the per-proc actor on each proc),
//! waits until every rank has finished `init()` and posted a [`ReadyAck`], then
//! answers the caller on a reply port with `Result<(), RdmaInitError>`.
//!
//! ## Topology
//!
//! Three distinct objects, kept separate throughout this module:
//!
//! 1. **Owner entry** — one [`MeshState`], identified by a small owner-local
//!    [`EntryId`], per requested `ProcMeshRef` view (`by_mesh` maps the view
//!    to its `EntryId`). Two overlapping `ProcMeshRef` views are *distinct*
//!    entries even when they share procs.
//! 2. **Manager mesh + controller** — the first `EnsureRdmaManager` for a view
//!    calls `spawn_service`, which materializes a new
//!    `ActorMesh<RdmaManagerActor>` view plus one `ActorMeshController`, and
//!    *creates or reuses* the per-proc actor on each proc in the view. The
//!    entry retains the mesh to cast to.
//! 3. **Per-proc service actor** — on each proc the `rdma_manager` service is a
//!    per-proc singleton (by reserved name). Overlapping views reuse the same
//!    physical actor on a shared proc; the physical actors are shared while the
//!    owner entries and mesh views stay distinct.
//!
//! ## Correlation key
//!
//! A `ProcMeshRef` carries every rank ref, so hashing or serializing one is
//! O(N) in the rank count. To keep readiness O(N) overall rather than O(N²),
//! the owner hashes the `ProcMeshRef` only on the ensure path — one lookup
//! hash per ensure, plus one insertion hash when a new view is created —
//! never on the ack path, and thereafter correlates by the small [`EntryId`]:
//! `RdmaManagerReady` and `ReadyAck` carry the `EntryId`, so the N per-rank
//! messages neither carry nor hash a full `ProcMeshRef`, and the
//! per-`EntryId` `entries` map is the hot path.
//!
//! ## Invariants (RMO-*)
//!
//! - **RMO-1** (single owner): Exactly one owner, living in the client process
//!   (the Python client root) where it is spawned with the stable client-root
//!   context; not caller-owned; torn down only when that process exits.
//! - **RMO-2** (terminal-absorbing): `Ready`/`Failed` are terminal — an
//!   `EnsureRdmaManager` for a resolved view replies from cache, and `ReadyAck`
//!   / [`RdmaManagerOwnerActor::transition`] no-op on a non-`Pending` entry.
//! - **RMO-3** (single-transition guard): [`RdmaManagerOwnerActor::transition`]
//!   is the sole `Pending -> terminal` mutator; it drains `waiters` on resolve.
//! - **RMO-4** (never handler-`Err`; owner never fails): A failure resolves the
//!   affected entry through its `Result` reply rather than terminating the
//!   owner. Spawn and cast failures resolve the entry to `Failed`; handlers
//!   return `Ok`.
//! - **RMO-5** (non-returning replies): Every reply ref gets
//!   `return_undeliverable(false)` before it is posted or stashed.
//! - **RMO-6** (readiness-after-init): `Ready` is set only once every rank's
//!   post-`init()` `ReadyAck` has arrived (`remaining_acks == 0`).
//! - **RMO-8** (manager per-proc-shared): Each target proc has at most one
//!   `rdma_manager` service actor. Overlapping proc-mesh views reuse the
//!   per-proc actor on shared procs while retaining distinct logical owner
//!   entries.
//! - **RMO-10** (bounded-spawn is the only block): `spawn_service().await` is
//!   the sole await; the ack gather is message-driven.
//! - **RMO-11** (retain-to-message, not teardown): The retained manager mesh is
//!   kept to cast to; its `Drop` is inert, so retention is not teardown.
//! - **RMO-13** (controller topology): The owner service is controllerless;
//!   every manager mesh it creates is controller-ful, with its controller owned
//!   as a child of the owner.
//! - **RMR-1** (ack-after-init): Each per-proc manager posts `ReadyAck` from
//!   its `RdmaManagerReady` handler, which runs strictly after `init()`.
//!
//! The owner implements `Handler<MeshFailure>` because it calls
//! `spawn_service` to spawn or reuse a mesh of `RdmaManagerActor`s, one per
//! target proc. `spawn_service` creates an `ActorMeshController` for that mesh,
//! and its caller must handle the supervision events that controller forwards.
//! The owner itself is spawned controllerless, so spawning the owner imposes no
//! such bound.

use std::collections::HashMap;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorAddr;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::Endpoint;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::RemoteSpawn;
use hyperactor_config::Flattrs;
use hyperactor_mesh::ActorMesh;
use hyperactor_mesh::ProcMeshRef;
use hyperactor_mesh::supervision::MeshFailure;
use ndslice::view::Ranked;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::errors::RdmaInitError;
use crate::rdma_manager_actor::RdmaManagerActor;

/// Owner-local handle for a `ProcMeshRef`.
///
/// A `ProcMeshRef` contains O(N) rank references. Carrying it in each of N
/// readiness acknowledgements would make correlation O(N²). The ensure path
/// hashes the `ProcMeshRef` once to obtain this id; the N per-rank
/// `RdmaManagerReady`/`ReadyAck` messages then carry only this small id, so
/// readiness correlation stays O(N).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntryId(u64);

/// Owner -> manager-mesh readiness cast, delivered to every per-proc
/// `RdmaManagerActor`. Carries the owner ref so each rank can post [`ReadyAck`]
/// back after its own `init()` (RMR-1).
#[derive(
    Handler,
    HandleClient,
    RefClient,
    Debug,
    Serialize,
    Deserialize,
    Named,
    Clone
)]
pub struct RdmaManagerReady {
    pub owner: ActorRef<RdmaManagerOwnerActor>,
    pub entry: EntryId,
}
wirevalue::register_type!(RdmaManagerReady);

/// Per-proc manager -> owner readiness acknowledgement, one per rank.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct ReadyAck {
    pub entry: EntryId,
}
wirevalue::register_type!(ReadyAck);

/// Caller -> owner request to ensure a proc mesh's per-proc managers are ready.
///
/// `reply` is a plain port field (not `#[reply]`) so the handler can stash
/// it on a `Pending` entry and post the result once the mesh is ready.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct EnsureRdmaManager {
    pub proc_mesh: ProcMeshRef,
    pub reply: OncePortRef<Result<(), RdmaInitError>>,
}
wirevalue::register_type!(EnsureRdmaManager);

/// Per-view coordination state, held in `entries` keyed by [`EntryId`].
#[derive(Debug)]
enum MeshState {
    Pending {
        /// The manager-mesh view returned by `spawn_service`, containing one
        /// actor reference per target proc. Overlapping views may reference the
        /// same per-proc service actors. Retained for the readiness cast and
        /// `Ready` state; `Drop` is inert and does not control teardown
        /// (RMO-11).
        manager_mesh: ActorMesh<RdmaManagerActor>,
        /// Controller `ActorAddr` for this mesh view, captured at spawn.
        /// Readiness is correlated by the small `EntryId`; this address is the
        /// separate key that attributes a `MeshFailure` to its sending
        /// controller.
        controller_addr: ActorAddr,
        remaining_acks: usize,
        waiters: Vec<OncePortRef<Result<(), RdmaInitError>>>,
    },
    Ready {
        manager_mesh: ActorMesh<RdmaManagerActor>,
    },
    Failed(RdmaInitError),
}

#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        EnsureRdmaManager,
        ReadyAck,
        MeshFailure,
    ],
)]
#[hyperactor::spawnable]
pub struct RdmaManagerOwnerActor {
    /// Maps each requested view to its entry id. The `ProcMeshRef` is hashed
    /// here (one lookup per ensure, plus one insertion for a new view), never
    /// on the ack path.
    by_mesh: HashMap<ProcMeshRef, EntryId>,
    /// Per-entry coordination state, keyed by the small `EntryId`.
    entries: HashMap<EntryId, MeshState>,
    /// Monotonic source of `EntryId`s.
    next_entry_id: u64,
}

impl RdmaManagerOwnerActor {
    fn alloc_entry_id(&mut self) -> EntryId {
        let id = EntryId(self.next_entry_id);
        self.next_entry_id = self
            .next_entry_id
            .checked_add(1)
            .expect("owner entry id space exhausted");
        id
    }

    /// The single `Pending -> terminal` mutator (RMO-3). Resolves a `Pending`
    /// entry to `Ready`/`Failed`, draining `waiters` with `result`. No-op when
    /// the entry is absent or already terminal (RMO-2).
    fn transition(
        &mut self,
        cx: &Context<Self>,
        entry: EntryId,
        result: Result<(), RdmaInitError>,
    ) {
        // Act only on Pending; absent/terminal entries are ignored
        // (RMO-2/RMO-3).
        if !matches!(self.entries.get(&entry), Some(MeshState::Pending { .. })) {
            return;
        }
        let Some(MeshState::Pending {
            manager_mesh,
            controller_addr,
            waiters,
            ..
        }) = self.entries.remove(&entry)
        else {
            unreachable!("checked Pending immediately above");
        };

        tracing::debug!(
            entry = entry.0,
            controller = %controller_addr,
            ok = result.is_ok(),
            waiters = waiters.len(),
            "RdmaManagerOwnerActor draining Pending owner entry",
        );

        // Every waiter was made non-returning before storage (RMO-5), so a
        // departed caller drops its reply instead of bouncing it to the owner.
        for reply in waiters {
            reply.post(cx, result.clone());
        }

        let next = match result {
            Ok(()) => MeshState::Ready { manager_mesh },
            Err(e) => MeshState::Failed(e),
        };
        self.entries.insert(entry, next);
        debug_assert!(
            !matches!(self.entries.get(&entry), Some(MeshState::Pending { .. })),
            "RMO-3: entry must be terminal after a transition",
        );
    }
}

#[async_trait]
impl RemoteSpawn for RdmaManagerOwnerActor {
    type Params = ();

    async fn new(_params: (), _environment: Flattrs) -> Result<Self, anyhow::Error> {
        Ok(Self {
            by_mesh: HashMap::new(),
            entries: HashMap::new(),
            next_entry_id: 0,
        })
    }
}

#[async_trait]
impl Actor for RdmaManagerOwnerActor {}

// Implements the get-or-initialize barrier for one proc-mesh view. Existing
// entries reply from terminal state or join the pending waiter set. New entries
// spawn or reuse the per-proc service actors, record `Pending` before casting
// readiness, and resolve only after every rank acknowledges. Operational
// failures are returned through `reply`, never propagated from this handler.
#[async_trait]
impl Handler<EnsureRdmaManager> for RdmaManagerOwnerActor {
    async fn handle(
        &mut self,
        cx: &Context<Self>,
        msg: EnsureRdmaManager,
    ) -> Result<(), anyhow::Error> {
        let EnsureRdmaManager {
            proc_mesh,
            mut reply,
        } = msg;
        // RMO-5: never bounce an undeliverable reply back to the owner. Set
        // before the ref is posted or enters `waiters`.
        reply.return_undeliverable(false);

        // Hash the (O(N)) `ProcMeshRef` here (one lookup; plus one insertion
        // below for a new view) — never on the ack path. `EntryId: Copy`, so
        // this borrow of `by_mesh` ends before `entries` is touched.
        if let Some(&entry) = self.by_mesh.get(&proc_mesh) {
            match self.entries.get_mut(&entry) {
                Some(MeshState::Ready { manager_mesh }) => {
                    debug_assert!(
                        manager_mesh.controller().is_some(),
                        "RMO-13: a Ready manager mesh is controller-ful",
                    );
                    reply.post(cx, Ok(()));
                }
                Some(MeshState::Failed(e)) => {
                    reply.post(cx, Err(e.clone()));
                }
                Some(MeshState::Pending { waiters, .. }) => {
                    waiters.push(reply);
                }
                None => {
                    // `by_mesh` and `entries` are inserted together, so this is
                    // unreachable; reply rather than hang the caller if it ever
                    // desyncs.
                    debug_assert!(false, "by_mesh points to a missing entries slot");
                    reply.post(
                        cx,
                        Err(RdmaInitError::SpawnFailed("owner index desync".to_string())),
                    );
                }
            }
            return Ok(());
        }

        // Absent view: allocate its entry id, then materialize a
        // controller-ful manager mesh. `spawn_service` applies the singleton
        // `rdma_manager` resource id on each target proc, creating missing
        // actors and reusing existing actors from overlapping views (RMO-8).
        // This is the sole blocking await (RMO-10).
        let entry = self.alloc_entry_id();
        let mesh = match proc_mesh
            .spawn_service::<RdmaManagerActor, _>(cx, "rdma_manager", &None)
            .await
        {
            Ok(mesh) => mesh,
            Err(e) => {
                // Spawn failure is cached terminally (RMO-2): no `Pending`
                // exists yet, so insert `Failed` directly and reply inline.
                let err = RdmaInitError::SpawnFailed(e.to_string());
                self.entries.insert(entry, MeshState::Failed(err.clone()));
                self.by_mesh.insert(proc_mesh, entry);
                reply.post(cx, Err(err));
                return Ok(());
            }
        };
        // Capture the controller address for this mesh view. A controller-ful
        // `spawn_service` always sets one (RMO-13), so `expect` here asserts
        // that invariant. It is the key that attributes a `MeshFailure` to its
        // sending controller.
        let controller_addr = mesh
            .controller()
            .as_ref()
            .expect("controller-ful spawn_service always sets a controller")
            .actor_addr()
            .clone();
        let remaining_acks = mesh.region().num_ranks();
        let manager_for_cast = mesh.clone();

        // Record `Pending` BEFORE casting so a cast failure has an entry to
        // drain terminally (RMO-2).
        self.entries.insert(
            entry,
            MeshState::Pending {
                manager_mesh: mesh,
                controller_addr,
                remaining_acks,
                waiters: vec![reply],
            },
        );
        self.by_mesh.insert(proc_mesh, entry);

        // 0-rank mesh: no ack will ever come, so resolve immediately.
        if remaining_acks == 0 {
            self.transition(cx, entry, Ok(()));
            return Ok(());
        }

        // Cast readiness to every rank; each rank acks post-`init()` (RMR-1).
        let owner = cx.bind::<RdmaManagerOwnerActor>();
        if let Err(e) = manager_for_cast.cast(cx, RdmaManagerReady { owner, entry }) {
            self.transition(cx, entry, Err(RdmaInitError::SpawnFailed(e.to_string())));
        }
        Ok(())
    }
}

// Counts the post-`init()` acknowledgement from each manager rank. Unknown or
// terminal entries absorb late acknowledgements; only the final acknowledgement
// resolves a `Pending` entry through the guarded transition path.
#[async_trait]
#[hyperactor::handle(ReadyAck)]
impl ReadyAckHandler for RdmaManagerOwnerActor {
    async fn ready_ack(&mut self, cx: &Context<Self>, entry: EntryId) -> Result<(), anyhow::Error> {
        // Decrement under a scoped borrow; resolve after it is released.
        let resolved = if let Some(MeshState::Pending { remaining_acks, .. }) =
            self.entries.get_mut(&entry)
        {
            debug_assert!(
                *remaining_acks > 0,
                "ReadyAck on a Pending entry with 0 remaining (0-rank resolves before any ack can arrive)",
            );
            *remaining_acks -= 1;
            *remaining_acks == 0
        } else {
            // Resolved (Ready/Failed) or absent -> ignore (RMO-2 late-signal
            // absorption).
            false
        };
        if resolved {
            // RMO-3/RMO-6: the final acknowledgement performs the entry's sole
            // `Pending -> Ready` transition through the centralized guard.
            self.transition(cx, entry, Ok(()));
        }
        Ok(())
    }
}

// `spawn_service` creates an `ActorMeshController` for each manager-mesh view
// as a child of this owner and registers the owner's `MeshFailure` port as its
// failure sink. The handler logs and ignores the event, returning `Ok` so a
// supervision report does not terminate the owner.
#[async_trait]
impl Handler<MeshFailure> for RdmaManagerOwnerActor {
    async fn handle(
        &mut self,
        _cx: &Context<Self>,
        message: MeshFailure,
    ) -> Result<(), anyhow::Error> {
        tracing::warn!(?message, "RdmaManagerOwnerActor ignoring MeshFailure");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use async_trait::async_trait;
    use hyperactor::ActorHandle;
    use hyperactor::ActorId;
    use hyperactor::ActorRef;
    use hyperactor::Context;
    use hyperactor::Endpoint;
    use hyperactor::Handler;
    use hyperactor::RemoteSpawn;
    use hyperactor_config::Flattrs;
    use hyperactor_mesh::ActorMesh;
    use hyperactor_mesh::ProcMeshRef;
    use hyperactor_mesh::context;
    use hyperactor_mesh::host_mesh::HostMesh;
    use ndslice::ViewExt;
    use ndslice::view::Ranked;
    use tokio::sync::oneshot;
    use tokio::time::timeout;

    use super::EnsureRdmaManagerClient;
    use super::MeshState;
    use super::RdmaManagerOwnerActor;
    use crate::RdmaManagerActor;
    use crate::errors::RdmaInitError;

    /// Happy path: ensuring a manager on a two-proc mesh replies `Ok` only once
    /// every rank posts its post-`init()` `ReadyAck` (RMO-6, RMO-8, RMR-1). TCP
    /// fallback lets the manager `init()` succeed without RDMA hardware.
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn ensure_resolves_ok_after_all_ranks_ack() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 2),
                None,
                None,
            )
            .await?;
        let proc_mesh_ref: ProcMeshRef = (*proc_mesh).clone();

        let owner = RdmaManagerOwnerActor::new((), Flattrs::default()).await?;
        let owner_handle = client.spawn(owner);

        let (reply, rx) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, proc_mesh_ref, reply.bind())
            .await?;

        let result = rx.recv().await?;
        assert_eq!(result, Ok(()));

        owner_handle.stop("test complete")?;
        let _ = owner_handle.await;
        host_mesh.shutdown(client).await?;
        Ok(())
    }

    // Test-only local probe messages, delivered via the owner's ActorHandle.

    #[derive(Debug, PartialEq, Eq)]
    enum EntryStateTag {
        Absent,
        Pending,
        Ready,
        Failed,
    }

    #[derive(Debug)]
    struct ProbeResult {
        state: EntryStateTag,
        entry_count: usize,
        waiter_count: usize,
        remaining_acks: Option<usize>,
        manager_ids: Vec<ActorId>,
    }

    /// Reports a view's entry state, the owner's total entry count, and (when
    /// `Pending`) its waiter and remaining-ack counts.
    #[derive(Debug)]
    struct Probe {
        key: ProcMeshRef,
        reply: oneshot::Sender<ProbeResult>,
    }

    #[async_trait]
    impl Handler<Probe> for RdmaManagerOwnerActor {
        async fn handle(&mut self, _cx: &Context<Self>, msg: Probe) -> Result<(), anyhow::Error> {
            let entry_count = self.entries.len();
            let (state, waiter_count, remaining_acks, manager_ids) = match self
                .by_mesh
                .get(&msg.key)
                .copied()
                .and_then(|id| self.entries.get(&id))
            {
                None => (EntryStateTag::Absent, 0, None, Vec::new()),
                Some(MeshState::Pending {
                    waiters,
                    remaining_acks,
                    manager_mesh,
                    controller_addr: _,
                }) => (
                    EntryStateTag::Pending,
                    waiters.len(),
                    Some(*remaining_acks),
                    manager_mesh_ids(manager_mesh),
                ),
                Some(MeshState::Ready { manager_mesh }) => (
                    EntryStateTag::Ready,
                    0,
                    None,
                    manager_mesh_ids(manager_mesh),
                ),
                Some(MeshState::Failed(_)) => (EntryStateTag::Failed, 0, None, Vec::new()),
            };
            let _ = msg.reply.send(ProbeResult {
                state,
                entry_count,
                waiter_count,
                remaining_acks,
                manager_ids,
            });
            Ok(())
        }
    }

    async fn probe(
        handle: &ActorHandle<RdmaManagerOwnerActor>,
        cx: &impl hyperactor::context::Actor,
        key: ProcMeshRef,
    ) -> ProbeResult {
        let (tx, rx) = oneshot::channel();
        handle.post(cx, Probe { key, reply: tx });
        rx.await.expect("owner replied to Probe")
    }

    /// The member `ActorId`s of a manager mesh (one per proc in the view).
    fn manager_mesh_ids(mesh: &ActorMesh<RdmaManagerActor>) -> Vec<ActorId> {
        let ranks = Ranked::region(&**mesh).num_ranks();
        (0..ranks)
            .filter_map(|rank| {
                Ranked::get(&**mesh, rank).map(|actor| actor.actor_addr().id().clone())
            })
            .collect()
    }

    /// Once an entry is `Ready`, a duplicate ensure replies from the cached
    /// terminal state without adding another owner entry (RMO-2).
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn duplicate_ensure_replies_from_ready_cache() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 2),
                None,
                None,
            )
            .await?;
        let key: ProcMeshRef = (*proc_mesh).clone();

        let owner = RdmaManagerOwnerActor::new((), Flattrs::default()).await?;
        let owner_handle = client.spawn(owner);

        let (r1, rx1) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, key.clone(), r1.bind())
            .await?;
        assert_eq!(rx1.recv().await?, Ok(()));

        let (r2, rx2) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, key.clone(), r2.bind())
            .await?;
        assert_eq!(rx2.recv().await?, Ok(()));

        let observed = probe(&owner_handle, client, key.clone()).await;
        assert_eq!(observed.state, EntryStateTag::Ready);
        assert_eq!(
            observed.entry_count, 1,
            "a duplicate ensure for the same view must not add an entry",
        );

        owner_handle.stop("test complete")?;
        let _ = owner_handle.await;
        host_mesh.shutdown(client).await?;
        Ok(())
    }

    /// Blocks the owner's message loop: signals `entered`, then awaits
    /// `release`. Lets a test enqueue messages behind it to control ordering.
    #[derive(Debug)]
    struct Pause {
        entered: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    }

    #[async_trait]
    impl Handler<Pause> for RdmaManagerOwnerActor {
        async fn handle(&mut self, _cx: &Context<Self>, msg: Pause) -> Result<(), anyhow::Error> {
            let _ = msg.entered.send(());
            let _ = msg.release.await;
            Ok(())
        }
    }

    /// A second ensure for the same view arriving while the first is still
    /// `Pending` joins that entry as an extra waiter rather than spawning
    /// again; both waiters then resolve `Ok` together (RMO-2, RMO-3).
    /// `Pause` + mailbox FIFO make the in-flight overlap deterministic.
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn duplicate_ensure_joins_pending_entry() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 2),
                None,
                None,
            )
            .await?;
        let key: ProcMeshRef = (*proc_mesh).clone();

        let owner = RdmaManagerOwnerActor::new((), Flattrs::default()).await?;
        let owner_handle = client.spawn(owner);

        // Block the owner, then enqueue (FIFO, behind the pause) two ensures
        // for the same key and a Probe.
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        owner_handle.post(
            client,
            Pause {
                entered: entered_tx,
                release: release_rx,
            },
        );
        entered_rx.await.expect("owner entered Pause");

        let (r1, rx1) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, key.clone(), r1.bind())
            .await?;
        let (r2, rx2) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, key.clone(), r2.bind())
            .await?;

        let (ptx, prx) = oneshot::channel();
        owner_handle.post(
            client,
            Probe {
                key: key.clone(),
                reply: ptx,
            },
        );

        // Release: FIFO runs ensure 1 (records `Pending` + casts), ensure 2
        // (joins as a waiter), then the Probe -- all before any `ReadyAck`.
        let _ = release_tx.send(());

        let observed = prx.await.expect("owner replied to Probe");
        assert_eq!(observed.state, EntryStateTag::Pending);
        assert_eq!(observed.entry_count, 1, "both ensures share one entry");
        assert_eq!(observed.waiter_count, 2, "second ensure joined as a waiter");
        assert_eq!(
            observed.remaining_acks,
            Some(2),
            "two-rank barrier initialized before any ack",
        );

        assert_eq!(rx1.recv().await?, Ok(()));
        assert_eq!(rx2.recv().await?, Ok(()));

        owner_handle.stop("test complete")?;
        let _ = owner_handle.await;
        host_mesh.shutdown(client).await?;
        Ok(())
    }

    /// While one view's readiness acks are outstanding, the owner keeps
    /// servicing other messages: two ensures for different views both reach
    /// `Pending` before any ack, so the gather is message-driven, not a
    /// blocking await (RMO-10).
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn ensure_gather_is_non_blocking() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 2),
                None,
                None,
            )
            .await?;
        let view_a: ProcMeshRef = proc_mesh.range("procs", 0..1)?;
        let view_b: ProcMeshRef = proc_mesh.range("procs", 1..2)?;

        let owner = RdmaManagerOwnerActor::new((), Flattrs::default()).await?;
        let owner_handle = client.spawn(owner);

        // Block the owner, then enqueue (FIFO, behind the pause) ensures for
        // two distinct views and a Probe of each.
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        owner_handle.post(
            client,
            Pause {
                entered: entered_tx,
                release: release_rx,
            },
        );
        entered_rx.await.expect("owner entered Pause");

        let (ra, rxa) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, view_a.clone(), ra.bind())
            .await?;
        let (rb, rxb) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, view_b.clone(), rb.bind())
            .await?;

        let (pta, prxa) = oneshot::channel();
        owner_handle.post(
            client,
            Probe {
                key: view_a.clone(),
                reply: pta,
            },
        );
        let (ptb, prxb) = oneshot::channel();
        owner_handle.post(
            client,
            Probe {
                key: view_b.clone(),
                reply: ptb,
            },
        );

        let _ = release_tx.send(());

        // Both views are Pending before any ack: view B was serviced without
        // blocking on view A's outstanding acks.
        let a = prxa.await.expect("probe view A");
        let b = prxb.await.expect("probe view B");
        assert_eq!(a.state, EntryStateTag::Pending);
        assert_eq!(b.state, EntryStateTag::Pending);
        assert_eq!(a.entry_count, 2, "owner tracks both views concurrently");

        assert_eq!(rxa.recv().await?, Ok(()));
        assert_eq!(rxb.recv().await?, Ok(()));

        owner_handle.stop("test complete")?;
        let _ = owner_handle.await;
        host_mesh.shutdown(client).await?;
        Ok(())
    }

    /// The owner is spawned as a controllerless, ProcAgent-managed singleton
    /// (as production does via `spawn_controllerless_service`): its mesh has no
    /// controller (RMO-13), and after the returned `ActorMesh` is dropped the
    /// owner still serves via the retained rank-0 `ActorRef` (RMO-1).
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn owner_is_controllerless_singleton() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 1),
                None,
                None,
            )
            .await?;
        let target: ProcMeshRef = (*proc_mesh).clone();

        // Spawn the owner the way production does: a controllerless singleton.
        let owner_mesh = proc_mesh
            .spawn_controllerless_service::<RdmaManagerOwnerActor, _>(
                client,
                "rdma_manager_owner",
                &(),
            )
            .await?;
        assert!(
            owner_mesh.controller().is_none(),
            "owner is spawned controllerless",
        );
        let owner_ref: ActorRef<RdmaManagerOwnerActor> =
            owner_mesh.values().next().expect("owner rank 0").clone();

        // Drop the returned mesh; the ProcAgent-managed singleton persists.
        drop(owner_mesh);

        // The retained ref still reaches the owner, which serves the request.
        let (r, rx) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_ref
            .ensure_rdma_manager(client, target, r.bind())
            .await?;
        assert_eq!(rx.recv().await?, Ok(()));

        host_mesh.shutdown(client).await?;
        Ok(())
    }

    /// Spawning the owner service twice yields the same singleton actor: the
    /// controllerless-singleton owner is deduped by its reserved name, so there
    /// is exactly one per process (RMO-1).
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn owner_service_is_singleton() -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 1),
                None,
                None,
            )
            .await?;

        let first = proc_mesh
            .spawn_controllerless_service::<RdmaManagerOwnerActor, _>(
                client,
                "rdma_manager_owner",
                &(),
            )
            .await?;
        let second = proc_mesh
            .spawn_controllerless_service::<RdmaManagerOwnerActor, _>(
                client,
                "rdma_manager_owner",
                &(),
            )
            .await?;

        let addr1 = first
            .values()
            .next()
            .expect("owner rank 0")
            .actor_addr()
            .clone();
        let addr2 = second
            .values()
            .next()
            .expect("owner rank 0")
            .actor_addr()
            .clone();
        assert_eq!(
            addr1, addr2,
            "the owner service is a singleton, deduped by name"
        );

        host_mesh.shutdown(client).await?;
        Ok(())
    }

    /// Overlapping views share the per-proc `rdma_manager`: the shared proc
    /// runs one physical actor (identical `ActorId`) referenced by both views'
    /// manager meshes, while each view keeps a distinct owner entry (RMO-8).
    /// Verified in both ensure orders (whole-first and slice-first).
    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn overlapping_views_share_per_proc_manager_whole_first() -> anyhow::Result<()> {
        overlapping_views_share_per_proc_manager(true).await
    }

    #[timed_test::async_timed_test(timeout_secs = 30)]
    async fn overlapping_views_share_per_proc_manager_slice_first() -> anyhow::Result<()> {
        overlapping_views_share_per_proc_manager(false).await
    }

    async fn overlapping_views_share_per_proc_manager(whole_first: bool) -> anyhow::Result<()> {
        let config = hyperactor_config::global::lock();
        let _guard = config.override_key(crate::config::RDMA_ALLOW_TCP_FALLBACK, true);

        let cx = context().await;
        let client = cx.actor_instance;

        let mut host_mesh = HostMesh::local_in_process().await?;
        let proc_mesh = host_mesh
            .spawn(
                client,
                "rdma_procs",
                hyperactor_mesh::extent!(procs = 2),
                None,
                None,
            )
            .await?;
        let whole: ProcMeshRef = (*proc_mesh).clone();
        let slice: ProcMeshRef = proc_mesh.range("procs", 1..2)?;

        let owner = RdmaManagerOwnerActor::new((), Flattrs::default()).await?;
        let owner_handle = client.spawn(owner);

        let (first, second) = if whole_first {
            (whole.clone(), slice.clone())
        } else {
            (slice.clone(), whole.clone())
        };

        // The reuse path crosses three independently-blockable stages: the
        // second `spawn_service`, the reused actor's second `ReadyAck`, and the
        // teardown of two overlapping controllers. Bound each separately so a
        // timeout names its stage rather than tripping the whole-test deadline.

        // First view resolves against fresh per-proc managers.
        let (r1, rx1) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, first, r1.bind())
            .await?;
        let first_reply = timeout(Duration::from_secs(15), rx1.recv())
            .await
            .expect("first ensure reply elapsed")?;
        assert_eq!(first_reply, Ok(()));

        // Second view overlaps the first. Probe before awaiting its reply: if
        // the owner is blocked inside the second `spawn_service`, it cannot
        // service this Probe and the wait elapses (isolates a spawn hang).
        let (r2, rx2) = client.open_once_port::<Result<(), RdmaInitError>>();
        owner_handle
            .ensure_rdma_manager(client, second.clone(), r2.bind())
            .await?;
        let after_spawn = timeout(
            Duration::from_secs(15),
            probe(&owner_handle, client, second.clone()),
        )
        .await
        .expect("second spawn_service blocked: probe elapsed");
        assert!(
            matches!(
                after_spawn.state,
                EntryStateTag::Pending | EntryStateTag::Ready
            ),
            "second ensure recorded no entry: {after_spawn:?}",
        );

        // spawn_service returned; a hang here is the reused actor failing to
        // deliver the second view's `ReadyAck`.
        let second_reply = timeout(Duration::from_secs(15), rx2.recv())
            .await
            .expect("second ensure reply elapsed: shared actor did not re-ack")?;
        assert_eq!(second_reply, Ok(()));

        let whole_entry = probe(&owner_handle, client, whole.clone()).await;
        let slice_entry = probe(&owner_handle, client, slice.clone()).await;

        // Distinct owner entries, one per view.
        assert_eq!(whole_entry.entry_count, 2, "each view keeps its own entry");
        assert_eq!(whole_entry.state, EntryStateTag::Ready);
        assert_eq!(slice_entry.state, EntryStateTag::Ready);

        // The shared proc runs one physical manager, referenced by both views.
        let shared = whole_entry
            .manager_ids
            .iter()
            .filter(|id| slice_entry.manager_ids.contains(id))
            .count();
        assert_eq!(
            whole_entry.manager_ids.len(),
            2,
            "whole view spans two procs"
        );
        assert_eq!(
            slice_entry.manager_ids.len(),
            1,
            "slice view spans one proc"
        );
        assert_eq!(shared, 1, "views share the one manager on the common proc");

        // Teardown drops two overlapping controllers over shared procs.
        timeout(Duration::from_secs(15), async move {
            owner_handle.stop("test complete")?;
            let _ = owner_handle.await;
            host_mesh.shutdown(client).await?;
            anyhow::Ok(())
        })
        .await
        .expect("shutdown elapsed: overlapping controller cleanup")?;
        Ok(())
    }
}
