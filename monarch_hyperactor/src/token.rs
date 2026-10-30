/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Connect to an existing Python actor mesh using a token string.
//!
//! An actor publishes a token for a Python actor mesh. Another process
//! uses it to obtain a reference and call the mesh's endpoints.
//! Connecting does not add actors to the mesh.
//!
//! The actor whose instance publishes the token owns its rendezvous actor
//! and receives each connecting actor's address. It need not be an actor
//! in the published mesh. Once the publishing actor stops, the token no
//! longer accepts connections.

use std::sync::Arc;
use std::time::Duration;

use hyperactor::Actor;
use hyperactor::ActorAddr;
use hyperactor::Endpoint as _;
use hyperactor::Instance;
use hyperactor::PortReceiver;
use hyperactor::PortRef;
use hyperactor::actor::ActorStatus;
use hyperactor_mesh::actor_mesh::ActorMeshRef;
use hyperactor_remote::Token;
use hyperactor_remote::TokenOptions;
use hyperactor_remote::token;
use ndslice::view::Ranked;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTimeoutError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::actor::PythonActor;
use crate::actor_mesh::AsyncActorMesh;
use crate::actor_mesh::PythonActorMesh;
use crate::actor_mesh::PythonActorMeshImpl;
use crate::context::PyInstance;
use crate::handle::PyHandle;
use crate::runtime::get_tokio_runtime;
use crate::shape::PyRegion;

/// The publishing actor gives each connecting actor a Python actor mesh
/// reference and receives that actor's address in return.
type ActorMeshToken = Token<ActorMeshRef<PythonActor>, ActorAddr>;

/// The result of publishing: a token string and its connection notifications.
///
/// Share `token` with other processes. `next_peer()` returns a `Handle`
/// for the actor address supplied by the next connecting actor.
/// The actor represented by `create_token()`'s `instance` owns the
/// rendezvous actor; this Python object does not own it or keep it running.
#[pyclass(
    frozen,
    name = "PublishedToken",
    module = "monarch._rust_bindings.monarch_hyperactor.token"
)]
pub struct PyPublishedToken {
    /// Token string other processes use to request the published mesh reference.
    token: String,
    /// Receiver for a port opened on the actor that publishes the token.
    /// The rendezvous sends each connecting actor's address to this port.
    /// This object and the Tokio tasks started by `next_peer()` share the
    /// receiver through the `Arc`. The mutex serializes receives, so each
    /// notification is consumed once, not delivered to every task.
    /// A task retains the receiver until its receive finishes, even if
    /// Python drops this `PublishedToken` object.
    connection_notifications: Arc<tokio::sync::Mutex<PortReceiver<token::Joined<ActorAddr>>>>,
}

#[pymethods]
impl PyPublishedToken {
    /// The token string to hand to peers.
    #[getter]
    fn token(&self) -> String {
        self.token.clone()
    }

    /// Start receiving one connection notification and return its `PyHandle`.
    /// The result is the connecting actor's address as a string.
    fn next_peer(&self) -> PyHandle {
        let connection_notifications = Arc::clone(&self.connection_notifications);
        PyHandle::spawn(async move {
            let connected = connection_notifications
                .lock()
                .await
                .recv()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(connected.peer.to_string())
        })
    }
}

/// Whether an actor with this status may still gain a child.
fn accepts_children(status: &ActorStatus) -> bool {
    !matches!(status, ActorStatus::Stopping(_)) && !status.is_terminal()
}

/// Create the rendezvous actor behind a token as a child of `owner`.
/// `owner` is the actor instance publishing the token, not necessarily
/// an actor in the mesh that the Python binding publishes.
/// Each connecting actor receives `value_from_owner`. The value it
/// supplies in return is sent to `connection_notifications`.
///
/// Generic over the owner's actor type and the two exchanged values.
/// The Python binding uses `ActorMeshToken`; Rust lifecycle tests exchange
/// actor addresses instead, testing ownership without building a Python mesh.
///
/// Reject publication if the owner is stopping or has ended. Otherwise,
/// hold its status read lock until the new child is linked to it. This
/// prevents the owner from stopping between the check and child creation,
/// which could leave the token running after its owner has stopped.
///
/// `before_create` lets tests request a stop inside that protected interval.
/// Production callers pass a no-op.
fn create_owned<A, C, J>(
    owner: &Instance<A>,
    value_from_owner: C,
    connection_notifications: PortRef<token::Joined<J>>,
    before_create: impl FnOnce(),
) -> anyhow::Result<Token<C, J>>
where
    A: Actor,
    C: token::TokenPeer + Clone,
    J: token::TokenPeer,
{
    let status_receiver = owner.status();
    // Read the current status and hold its read lock until this function returns.
    let status_guard = status_receiver.borrow();
    anyhow::ensure!(
        accepts_children(&status_guard),
        "the token owner {} is {}",
        owner.self_addr(),
        *status_guard
    );
    // Tests can request a competing stop here, while we still hold the lock.
    before_create();

    // `token::create()` spawns a rendezvous actor as `owner`'s child and
    // returns a token pointing to it. When another actor connects, the
    // rendezvous sends that actor's supplied value to
    // `connection_notifications`, then replies with `value_from_owner`.
    // Keep the status lock held until the child is registered with `owner`,
    // so the owner's shutdown cannot miss it.
    token::create(
        owner,
        value_from_owner,
        connection_notifications,
        TokenOptions::default(),
    )
}

/// Publish a reference to `mesh` and return a `PublishedToken`.
///
/// Called by Python's `publish(mesh)` on the publishing side. `instance`
/// represents the calling actor, or the process's root client when called
/// outside an actor. The rendezvous runs as that actor's child, so stopping
/// it prevents further connections. Publishing does not change mesh ownership.
///
/// Other processes pass the returned object's `token` string to Python's
/// `connect()`. Its `next_peer()` method reports the address supplied by
/// each connecting actor or root client.
///
/// The mesh must already exist; this function does not wait for creation.
/// A pending mesh raises `ValueError`: wait for `initialized` first.
/// A failed mesh propagates its creation error. If the publishing actor
/// is stopping or has ended, raise `RuntimeError`.
#[pyfunction]
fn create_token(instance: &PyInstance, mesh: &PythonActorMesh) -> PyResult<PyPublishedToken> {
    // Obtain the existing mesh's serializable reference without waiting
    // or starting its creation.
    let mesh_ref = mesh.get_inner().ready_mesh_ref()?;

    // Make Monarch's Tokio runtime current on this thread. Creating the
    // rendezvous actor below schedules work through that runtime.
    let _runtime = get_tokio_runtime().enter();

    // Give the rendezvous a port to notify before accepting connections.
    // Each notification carries the connecting actor's address.
    let (notification_port, connection_notifications) =
        instance.open_port::<token::Joined<ActorAddr>>();

    // Publish under this actor instance's status lock. The no-op is the
    // test hook; production needs no action between check and spawn.
    let token: ActorMeshToken =
        create_owned(&**instance, mesh_ref, notification_port.bind(), || {})
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Return the shareable token string and retain the notification receiver.
    // `next_peer()` tasks share this receiver and serialize access to it.
    Ok(PyPublishedToken {
        token: token.to_string(),
        connection_notifications: Arc::new(tokio::sync::Mutex::new(connection_notifications)),
    })
}

/// Request the mesh reference published through `token`.
///
/// Called by Python's `connect(token, Class, ...)` on the connecting side.
/// `instance` represents that caller's actor or root client, not the actor
/// that published the token. Send its address to the token's rendezvous
/// actor and return a `PyHandle` for the published mesh and its region.
/// Python's `connect()` waits on this Handle and wraps the mesh using `Class`.
///
/// Raise `ValueError` for an invalid timeout or token, or a rejected
/// connection. If no answer arrives within `timeout_secs`, including when the
/// token's owner has stopped, close the reply port and raise `TimeoutError`
/// through the Handle.
#[pyfunction]
fn connect_token(instance: &PyInstance, token: &str, timeout_secs: f64) -> PyResult<PyHandle> {
    // Reject invalid durations before opening a port or sending a request.
    let timeout = Duration::try_from_secs_f64(timeout_secs)
        .map_err(|e| PyValueError::new_err(format!("invalid timeout {timeout_secs}: {e}")))?;

    // Decode the rendezvous reference and check the token's exchanged Rust
    // types. This does not check the Python actor class passed to `connect()`.
    let token: ActorMeshToken = token
        .parse()
        .map_err(|e: anyhow::Error| PyValueError::new_err(format!("invalid token: {e}")))?;

    // Use Monarch's runtime for the port operations and the task spawned below.
    let _runtime = get_tokio_runtime().enter();

    // Open a reply port on the connecting actor. `result.bind()` gives the
    // rendezvous an address to reply to; `results` receives its answer here.
    let (result, mut results) =
        instance.open_port::<token::JoinResult<ActorMeshRef<PythonActor>>>();

    // Send the connecting actor's address and the reply-port reference now,
    // without waiting for the answer. If the token's owner has stopped, its
    // rendezvous actor is gone: drop the request rather than return it to
    // `instance`, where it would fail the caller's root actor and end its
    // process. The caller then times out, as when the owner's process exits.
    //
    // TODO: `Token::join` returns an undeliverable request to the caller, so
    // this posts `Join` on the rendezvous's port directly. Give `join` a
    // policy for undeliverable requests (or make dropping them its only
    // behavior) and call it here instead.
    let mut join_port = token
        .rendezvous()
        .port::<token::Join<ActorMeshRef<PythonActor>, ActorAddr>>();
    join_port.return_undeliverable(false);
    join_port.post(
        &**instance,
        token::Join {
            joiner: instance.self_addr().clone(),
            result: result.bind(),
        },
    );

    // Start a Tokio task waiting for the answer and return its `PyHandle`.
    // Dropping the Handle does not cancel this task.
    Ok(PyHandle::spawn(async move {
        // This deadline bounds the connection's reply wait, independently
        // of any later `Handle.get()` timeout. Returning drops `results`
        // and closes the reply port; it does not retract the sent request.
        let reply = tokio::time::timeout(timeout, results.recv())
            .await
            .map_err(|_| {
                PyTimeoutError::new_err("no answer from the token's owner; is it running?")
            })?
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match reply {
            token::JoinResult::Joined { peer } => {
                // `peer` is the mesh reference supplied by the publishing actor,
                // not that actor's address. Recover its shape from the region
                // and build local wrappers; no actors are spawned here.
                let region = PyRegion::from(peer.region().clone());
                let mesh = AsyncActorMesh::from_impl(Arc::new(PythonActorMeshImpl::new_ref(peer)));
                Ok((PythonActorMesh::from_impl(Arc::new(mesh)), region))
            }
            // Report an explicit rejection as this Handle's stored exception.
            token::JoinResult::Rejected { reason } => Err(PyValueError::new_err(format!(
                "the token's owner rejected the connection: {reason}"
            ))),
        }
    }))
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPublishedToken>()?;
    for function in [
        wrap_pyfunction!(create_token, module)?,
        wrap_pyfunction!(connect_token, module)?,
    ] {
        function.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_hyperactor.token",
        )?;
        module.add_function(function)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use async_trait::async_trait;
    use hyperactor::Proc;

    use super::*;

    const WAIT: Duration = Duration::from_secs(30);

    /// Hands a clone of its instance to the test, the way the root client's
    /// instance is used from a thread outside its serving loop, and reports
    /// when its cleanup runs.
    #[derive(Debug)]
    struct Owner {
        instance: tokio::sync::mpsc::UnboundedSender<Instance<Owner>>,
        cleaned_up: mpsc::SyncSender<()>,
    }

    #[async_trait]
    impl Actor for Owner {
        async fn init(&mut self, this: &Instance<Self>) -> anyhow::Result<()> {
            self.instance
                .send(this.clone_for_py())
                .expect("the test is waiting for the instance");
            Ok(())
        }

        async fn cleanup(
            &mut self,
            _this: &Instance<Self>,
            _err: Option<&hyperactor::actor::ActorError>,
        ) -> anyhow::Result<()> {
            let _ = self.cleaned_up.send(());
            Ok(())
        }
    }

    #[test]
    fn stopping_and_stopped_owners_accept_no_children() {
        assert!(accepts_children(&ActorStatus::Idle));
        assert!(accepts_children(&ActorStatus::Client));
        assert!(!accepts_children(&ActorStatus::stopping()));
        assert!(!accepts_children(&ActorStatus::zombie("test")));
        assert!(!accepts_children(&ActorStatus::Stopped("done".into())));
        assert!(!accepts_children(&ActorStatus::generic_failure("failed")));
    }

    #[tokio::test]
    async fn a_stopped_owner_gets_no_token() {
        let proc = Proc::isolated();
        let client = proc.client("client");
        let (instance_tx, mut instance_rx) = tokio::sync::mpsc::unbounded_channel();
        let (cleaned_up, _) = mpsc::sync_channel(1);
        let handle = proc.spawn(Owner {
            instance: instance_tx,
            cleaned_up,
        });
        let owner = tokio::time::timeout(WAIT, instance_rx.recv())
            .await
            .expect("the owner starts")
            .expect("the owner sends its instance");
        handle.drain_and_stop("test").unwrap();
        tokio::time::timeout(WAIT, handle.clone())
            .await
            .expect("the owner stops");

        let (peer_port, _peers) = client.open_port::<token::Joined<ActorAddr>>();
        let created: anyhow::Result<Token<ActorAddr, ActorAddr>> =
            create_owned(&owner, client.self_addr().clone(), peer_port.bind(), || {});
        let error = created.expect_err("a stopped owner must not get a token");
        assert!(error.to_string().contains("is stopped"), "{error}");
    }

    // A stop requested between the check and the child's creation cannot take
    // effect until the child is linked, so the owner's teardown stops it.
    // Without the status guard, teardown finishes during the wait below and
    // the rendezvous outlives its owner.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_stop_racing_publication_still_stops_the_token() {
        let proc = Proc::isolated();
        let client = proc.client("client");
        let (instance_tx, mut instance_rx) = tokio::sync::mpsc::unbounded_channel();
        let (cleaned_up_tx, cleaned_up) = mpsc::sync_channel(1);
        let handle = proc.spawn(Owner {
            instance: instance_tx,
            cleaned_up: cleaned_up_tx,
        });
        let owner = tokio::time::timeout(WAIT, instance_rx.recv())
            .await
            .expect("the owner starts")
            .expect("the owner sends its instance");

        let (peer_port, _peers) = client.open_port::<token::Joined<ActorAddr>>();
        let creator = client.self_addr().clone();
        let stopper = handle.clone();
        let (token, cleaned_up_during_create) = tokio::task::spawn_blocking(move || {
            let mut cleaned_up_during_create = false;
            let token: Token<ActorAddr, ActorAddr> =
                create_owned(&owner, creator, peer_port.bind(), || {
                    stopper.drain_and_stop("racing publication").unwrap();
                    // Do not read the owner's status here: that read would
                    // queue behind the stop's pending status write.
                    cleaned_up_during_create =
                        cleaned_up.recv_timeout(Duration::from_millis(500)).is_ok();
                })
                .expect("the owner was running when publication checked");
            (token, cleaned_up_during_create)
        })
        .await
        .unwrap();
        assert!(
            !cleaned_up_during_create,
            "the owner must not finish stopping while publication holds its status"
        );

        tokio::time::timeout(WAIT, handle)
            .await
            .expect("the owner stops after publication");
        // A stopped actor leaves the proc's registry; one still registered
        // must be on its way to a terminal status. Without the status guard,
        // the rendezvous stays registered and running, so this wait fails.
        if let Some(rendezvous) = proc.get_instance(token.rendezvous().actor_addr()) {
            let mut status = rendezvous.status().clone();
            tokio::time::timeout(WAIT, status.wait_for(ActorStatus::is_terminal))
                .await
                .expect("the owner's teardown stops the rendezvous")
                .unwrap();
        }
    }
}
