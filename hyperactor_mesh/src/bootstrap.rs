/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::VecDeque;
use std::env::VarError;
use std::fs::OpenOptions;
use std::future;
use std::io;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::SystemTime;

use async_trait::async_trait;
use base64::prelude::*;
use humantime::format_duration;
use hyperactor::ActorRef;
use hyperactor::Named;
use hyperactor::ProcId;
use hyperactor::channel;
use hyperactor::channel::ChannelAddr;
use hyperactor::channel::ChannelTransport;
use hyperactor::channel::Rx;
use hyperactor::channel::Tx;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor::host;
use hyperactor::host::Host;
use hyperactor::host::HostError;
use hyperactor::host::ProcManager;
use hyperactor::mailbox::MailboxServer;
use hyperactor::proc::Proc;
use serde::Deserialize;
use serde::Serialize;
use tokio::process::Child;
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::sync::watch;

use crate::logging::create_log_writers;
use crate::proc_mesh::mesh_agent::ProcMeshAgent;
use crate::v1;
use crate::v1::host_mesh::mesh_agent::HostAgentMode;
use crate::v1::host_mesh::mesh_agent::HostMeshAgent;

pub const BOOTSTRAP_ADDR_ENV: &str = "HYPERACTOR_MESH_BOOTSTRAP_ADDR";
pub const BOOTSTRAP_INDEX_ENV: &str = "HYPERACTOR_MESH_INDEX";
pub const CLIENT_TRACE_ID_ENV: &str = "MONARCH_CLIENT_TRACE_ID";
/// A channel used by each process to receive its own stdout and stderr
/// Because stdout and stderr can only be obtained by the parent process,
/// they need to be streamed back to the process.
pub(crate) const BOOTSTRAP_LOG_CHANNEL: &str = "BOOTSTRAP_LOG_CHANNEL";

/// Messages sent from the process to the allocator. This is an envelope
/// containing the index of the process (i.e., its "address" assigned by
/// the allocator), along with the control message in question.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) struct Process2Allocator(pub usize, pub Process2AllocatorMessage);

/// Control messages sent from processes to the allocator.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) enum Process2AllocatorMessage {
    /// Initialize a process2allocator session. The process is
    /// listening on the provided channel address, to which
    /// [`Allocator2Process`] messages are sent.
    Hello(ChannelAddr),

    /// A proc with the provided ID was started. Its mailbox is
    /// served at the provided channel address. Procs are started
    /// after instruction by the allocator through the corresponding
    /// [`Allocator2Process`] message.
    StartedProc(ProcId, ActorRef<ProcMeshAgent>, ChannelAddr),

    Heartbeat,
}

/// Messages sent from the allocator to a process.
#[derive(Debug, Clone, Serialize, Deserialize, Named)]
pub(crate) enum Allocator2Process {
    /// Request to start a new proc with the provided ID, listening
    /// to an address on the indicated channel transport.
    StartProc(ProcId, ChannelTransport),

    /// A request for the process to shut down its procs and exit the
    /// process with the provided code.
    StopAndExit(i32),

    /// A request for the process to immediately exit with the provided
    /// exit code
    Exit(i32),
}

async fn exit_if_missed_heartbeat(bootstrap_index: usize, bootstrap_addr: ChannelAddr) {
    let tx = match channel::dial(bootstrap_addr.clone()) {
        Ok(tx) => tx,

        Err(err) => {
            tracing::error!(
                "Failed to establish heartbeat connection to allocator, exiting! (addr: {:?}): {}",
                bootstrap_addr,
                err
            );
            std::process::exit(1);
        }
    };
    tracing::info!(
        "Heartbeat connection established to allocator (idx: {bootstrap_index}, addr: {bootstrap_addr:?})",
    );
    loop {
        RealClock.sleep(Duration::from_secs(5)).await;

        let result = tx
            .send(Process2Allocator(
                bootstrap_index,
                Process2AllocatorMessage::Heartbeat,
            ))
            .await;

        if let Err(err) = result {
            tracing::error!(
                "Heartbeat failed to allocator, exiting! (addr: {:?}): {}",
                bootstrap_addr,
                err
            );
            std::process::exit(1);
        }
    }
}

#[macro_export]
macro_rules! ok {
    ($expr:expr $(,)?) => {
        match $expr {
            Ok(value) => value,
            Err(e) => return ::anyhow::Error::from(e),
        }
    };
}

async fn halt<R>() -> R {
    future::pending::<()>().await;
    unreachable!()
}

/// Bootstrap configures the bootstrap behavior of a binary.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bootstrap {
    /// "v1" proc bootstrap
    Proc {
        /// The ProcId of the proc to be bootstrapped.
        proc_id: ProcId,
        /// The backend address to which messages are forwarded.
        /// See [`hyperactor::host`] for channel topology details.
        backend_addr: ChannelAddr,
        /// The callback address used to indicate successful spawning.
        callback_addr: ChannelAddr,
    },

    /// Host bootstrap. This sets up a new `Host`, managed by a
    /// [`crate::v1::host_mesh::mesh_agent::HostMeshAgent`].
    Host {
        /// The address on which to serve the host.
        addr: ChannelAddr,
    },

    #[default]
    V0ProcMesh, // pass through to the v0 allocator
}

impl Bootstrap {
    /// Serialize the mode into a environment-variable-safe string by
    /// base64-encoding its JSON representation.
    fn to_env_safe_string(&self) -> v1::Result<String> {
        Ok(BASE64_STANDARD.encode(serde_json::to_string(&self)?))
    }

    /// Deserialize the mode from the representation returned by [`to_env_safe_string`].
    fn from_env_safe_string(str: &str) -> v1::Result<Self> {
        let data = BASE64_STANDARD.decode(str)?;
        let data = std::str::from_utf8(&data)?;
        Ok(serde_json::from_str(data)?)
    }

    /// Get a bootstrap configuration from the environment; returns `None`
    /// if the environment does not specify a boostrap config.
    pub fn get_from_env() -> anyhow::Result<Option<Self>> {
        match std::env::var("HYPERACTOR_MESH_BOOTSTRAP_MODE") {
            Ok(mode) => match Bootstrap::from_env_safe_string(&mode) {
                Ok(mode) => Ok(Some(mode)),
                Err(e) => {
                    Err(anyhow::Error::from(e).context("parsing HYPERACTOR_MESH_BOOTSTRAP_MODE"))
                }
            },
            Err(VarError::NotPresent) => Ok(None),
            Err(e) => Err(anyhow::Error::from(e).context("reading HYPERACTOR_MESH_BOOTSTRAP_MODE")),
        }
    }

    /// Inject this bootstrap configuration into the environment of the provided command.
    pub fn to_env(&self, cmd: &mut Command) {
        cmd.env(
            "HYPERACTOR_MESH_BOOTSTRAP_MODE",
            self.to_env_safe_string().unwrap(),
        );
    }

    /// Bootstrap this binary according to this configuration.
    /// This either runs forever, or returns an error.
    pub async fn bootstrap(self) -> anyhow::Error {
        tracing::info!(
            "bootstrapping mesh process: {}",
            serde_json::to_string(&self).unwrap()
        );

        if Debug::is_active() {
            let mut buf = Vec::new();
            writeln!(&mut buf, "bootstrapping {}:", std::process::id()).unwrap();
            writeln!(
                &mut buf,
                "\tconfig: {}",
                serde_json::to_string(&self).unwrap()
            )
            .unwrap();
            match std::env::current_exe() {
                Ok(path) => writeln!(&mut buf, "\tcurrent_exe: {}", path.display()).unwrap(),
                Err(e) => writeln!(&mut buf, "\tcurrent_exe: error<{}>", e).unwrap(),
            }
            writeln!(&mut buf, "\targs:").unwrap();
            for arg in std::env::args() {
                writeln!(&mut buf, "\t\t{}", arg).unwrap();
            }
            writeln!(&mut buf, "\tenv:").unwrap();
            for (key, val) in std::env::vars() {
                writeln!(&mut buf, "\t\t{}={}", key, val).unwrap();
            }
            let _ = Debug.write(&buf);
        }

        match self {
            Bootstrap::Proc {
                proc_id,
                backend_addr,
                callback_addr,
            } => {
                let result =
                    host::spawn_proc(proc_id, backend_addr, callback_addr, |proc| async move {
                        ProcMeshAgent::boot_v1(proc).await
                    })
                    .await;
                match result {
                    Ok(_proc) => halt().await,
                    Err(e) => e.into(),
                }
            }
            Bootstrap::Host { addr } => {
                let manager = ok!(BootstrapProcManager::new_current_exe());
                let (host, _handle) = ok!(Host::serve(manager, addr).await);
                let addr = host.addr().clone();
                let host_mesh_agent = ok!(host
                    .system_proc()
                    .clone()
                    .spawn::<HostMeshAgent>("agent", HostAgentMode::Process(host))
                    .await);

                tracing::info!(
                    "serving host at {}, agent: {}",
                    addr,
                    host_mesh_agent.bind::<HostMeshAgent>()
                );
                halt().await
            }
            Bootstrap::V0ProcMesh => bootstrap_v0_proc_mesh().await,
        }
    }

    /// A variant of [`bootstrap`] that logs the error and exits the process
    /// if bootstrapping fails.
    pub async fn bootstrap_or_die(self) -> ! {
        let err = self.bootstrap().await;
        tracing::error!("failed to bootstrap mesh process: {}", err);
        std::process::exit(1)
    }
}

/// Represents the lifecycle state of a **proc as hosted in an OS
/// process** managed by `BootstrapProcManager`.
///
/// Note: This type is deliberately distinct from [`ProcState`] and
/// [`ProcStopReason`] (see `alloc.rs`). Those types model allocator
/// *events* - e.g. "a proc was Created/Running/Stopped" - and are
/// consumed from an event stream during allocation. By contrast,
/// [`ProcStatus`] is a **live, queryable view**: it reflects the
/// current observed status of a running proc, as seen through the
/// [`BootstrapProcHandle`] API (stop, kill, status).
///
/// In short:
/// - `ProcState`/`ProcStopReason`: historical / event-driven model
/// - `ProcStatus`: immediate status surface for lifecycle control
#[derive(Debug, Clone)]
pub enum ProcStatus {
    /// The OS process has been spawned but is not yet fully running.
    /// (Process-level: child handle exists, no confirmation yet.)
    Starting,
    /// The OS process is alive and considered running.
    /// (Process-level: `pid` is known; Proc-level: bootstrap
    /// may still be running.)
    Running { pid: u32, started_at: SystemTime },
    /// Ready means boostrap has completed and the proc is serving.
    /// (Process-level: `pid` is known; Proc-level: bootstrap
    /// completed.)
    Ready {
        pid: u32,
        started_at: SystemTime,
        addr: ChannelAddr,
        agent: ActorRef<ProcMeshAgent>,
    },
    /// A stop has been requested (SIGTERM, graceful shutdown, etc.),
    /// but the OS process has not yet fully exited. (Proc-level:
    /// shutdown in progress; Process-level: still running.)
    Stopping,
    /// The process exited with a normal exit code. (Process-level:
    /// exit observed.)
    Stopped { exit_code: i32 },
    /// The process was killed by a signal (e.g. SIGKILL).
    /// (Process-level: abnormal termination.)
    Killed { signal: i32, core_dumped: bool },
    /// The proc or its process failed for some other reason
    /// (bootstrap error, unexpected condition, etc.). (Both levels:
    /// catch-all failure.)
    Failed { reason: String },
}

impl ProcStatus {
    /// Returns `true` if the proc is in a terminal (exited) state:
    /// [`ProcStatus::Stopped`], [`ProcStatus::Killed`], or
    /// [`ProcStatus::Failed`].
    #[inline]
    pub fn is_exit(&self) -> bool {
        matches!(
            self,
            ProcStatus::Stopped { .. } | ProcStatus::Killed { .. } | ProcStatus::Failed { .. }
        )
    }
}

impl std::fmt::Display for ProcStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcStatus::Starting => write!(f, "Starting"),
            ProcStatus::Running { pid, started_at } => {
                let uptime = started_at
                    .elapsed()
                    .map(|d| format!(" up {}", format_duration(d)))
                    .unwrap_or_default();
                write!(f, "Running[{pid}]{uptime}")
            }
            ProcStatus::Ready {
                pid,
                started_at,
                addr,
                ..
            } => {
                let uptime = started_at
                    .elapsed()
                    .map(|d| format!(" up {}", format_duration(d)))
                    .unwrap_or_default();
                write!(f, "Ready[{pid}] at {addr}{uptime}")
            }
            ProcStatus::Stopping => write!(f, "Stopping"),
            ProcStatus::Stopped { exit_code } => write!(f, "Stopped(exit={exit_code})"),
            ProcStatus::Killed {
                signal,
                core_dumped,
            } => {
                if *core_dumped {
                    write!(f, "Killed(sig={signal}, core)")
                } else {
                    write!(f, "Killed(sig={signal})")
                }
            }
            ProcStatus::Failed { reason } => write!(f, "Failed({reason})"),
        }
    }
}

/// Error returned by [`BootstrapProcHandle::ready`].
#[derive(Debug, Clone)]
pub enum ReadyError {
    /// The proc reached a terminal state before `Ready`.
    Terminal(ProcStatus),
    /// The internal watch channel closed unexpectedly.
    ChannelClosed,
}

impl std::fmt::Display for ReadyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadyError::Terminal(st) => write!(f, "proc terminated before running: {st:?}"),
            ReadyError::ChannelClosed => write!(f, "status channel closed"),
        }
    }
}
impl std::error::Error for ReadyError {}

/// A handle to a proc launched by [`BootstrapProcManager`].
///
/// `BootstrapProcHandle` is a lightweight supervisor for an external
/// process: it tracks and broadcasts lifecycle state, and exposes a
/// small control/observation surface. While it may temporarily hold a
/// `tokio::process::Child` (shared behind a mutex) so the exit
/// monitor can `wait()` it, it is **not** the unique owner of the OS
/// process, and dropping a `BootstrapProcHandle` does not by itself
/// terminate the process.
///
/// What it pairs together:
/// - the **logical proc identity** (`ProcId`)
/// - the **live status surface** ([`ProcStatus`]), available both as
///   a synchronous snapshot (`status()`) and as an async stream via a
///   `tokio::sync::watch` channel (`watch()` / `changed()`)
///
/// Responsibilities:
/// - Retain the child handle only until the exit monitor claims it,
///   so the OS process can be awaited and its terminal status
///   recorded.
/// - Update status via the `mark_*` transitions and broadcast changes
///   over the watch channel so tasks can `await` lifecycle
///   transitions without polling.
/// - Provide the foundation for higher-level APIs like `wait()`
///   (await terminal) and, later, `terminate()` / `kill()`.
///
/// Notes:
/// - Manager-level cleanup happens in [`BootstrapProcManager::drop`]:
///   it SIGKILLs any still-recorded PIDs; we do not rely on
///   `Child::kill_on_drop`.
///
/// Relationship to types:
/// - [`ProcStatus`]: live status surface, updated by this handle.
/// - [`ProcState`]/[`ProcStopReason`] (in `alloc.rs`):
///   allocator-facing, historical event log; not directly updated by
///   this type.
#[derive(Clone, Debug)]
pub struct BootstrapProcHandle {
    /// Logical identity of the proc in the mesh.
    proc_id: ProcId,
    /// Live lifecycle snapshot (see [`ProcStatus`]). Kept in a mutex
    /// so [`BootstrapProcHandle::status`] can return a synchronous
    /// copy. All mutations now flow through
    /// [`BootstrapProcHandle::transition`], which updates this field
    /// under the lock and then broadcasts on the watch channel.
    status: Arc<std::sync::Mutex<ProcStatus>>,
    /// Underlying OS child handle. Held only until the exit monitor
    /// claims it (consumed by `wait()` there). Not relied on for
    /// teardown; manager `Drop` handles best-effort SIGKILL.
    child: Arc<std::sync::Mutex<Option<Child>>>,
    /// Watch sender for status transitions. Every `mark_*` goes
    /// through [`BootstrapProcHandle::transition`], which updates the
    /// snapshot under the lock and then `send`s the new
    /// [`ProcStatus`].
    tx: tokio::sync::watch::Sender<ProcStatus>,
    /// Watch receiver seed. `watch()` clones this so callers can
    /// `borrow()` the current status and `changed().await` future
    /// transitions independently.
    rx: tokio::sync::watch::Receiver<ProcStatus>,
}

impl BootstrapProcHandle {
    /// Construct a new [`BootstrapProcHandle`] for a freshly spawned
    /// OS process hosting a proc.
    ///
    /// - Initializes the status to [`ProcStatus::Starting`] since the
    ///   child process has been created but not yet confirmed running.
    /// - Wraps the provided [`Child`] handle in an
    ///   `Arc<Mutex<Option<Child>>>` so lifecycle methods (`wait`,
    ///   `terminate`, etc.) can consume it later.
    ///
    /// This is the canonical entry point used by
    /// `BootstrapProcManager` when it launches a proc into a new
    /// process.
    pub fn new(proc_id: ProcId, child: Child) -> Self {
        let (tx, rx) = watch::channel(ProcStatus::Starting);
        Self {
            proc_id,
            status: Arc::new(std::sync::Mutex::new(ProcStatus::Starting)),
            child: Arc::new(std::sync::Mutex::new(Some(child))),
            tx,
            rx,
        }
    }

    /// Return the logical proc identity in the mesh.
    #[inline]
    pub fn proc_id(&self) -> &ProcId {
        &self.proc_id
    }

    /// Create a new subscription to this proc's status stream.
    ///
    /// Each call returns a fresh [`watch::Receiver`] tied to this
    /// handle's internal [`ProcStatus`] channel. The receiver can be
    /// awaited on (`rx.changed().await`) to observe lifecycle
    /// transitions as they occur.
    ///
    /// Notes:
    /// - Multiple subscribers can exist simultaneously; each sees
    ///   every status update in order.
    /// - Use [`BootstrapProcHandle::status`] for a one-off snapshot;
    ///   use `watch()` when you need to await changes over time.
    #[inline]
    pub fn watch(&self) -> tokio::sync::watch::Receiver<ProcStatus> {
        self.rx.clone()
    }

    /// Wait until this proc's status changes.
    ///
    /// This is a convenience wrapper around
    /// [`watch::Receiver::changed`]: it subscribes internally via
    /// [`BootstrapProcHandle::watch`] and awaits the next transition.
    /// If no subscribers exist or the channel is closed, this returns
    /// without error.
    ///
    /// Typical usage:
    /// ```ignore
    /// handle.changed().await;
    /// match handle.status() {
    ///     ProcStatus::Running { .. } => { /* now running */ }
    ///     ProcStatus::Stopped { .. } => { /* exited */ }
    ///     _ => {}
    /// }
    /// ```
    #[inline]
    pub async fn changed(&self) {
        let _ = self.watch().changed().await;
    }

    /// Return the OS process ID (`pid`) for this proc.
    ///
    /// If the proc is currently `Running`, this returns the cached
    /// pid even if the `Child` has already been taken by the exit
    /// monitor. Otherwise, it falls back to `Child::id()` if the
    /// handle is still present. Returns `None` once the proc has
    /// exited or if the handle has been consumed.
    #[inline]
    pub fn pid(&self) -> Option<u32> {
        match *self.status.lock().expect("status mutex poisoned") {
            ProcStatus::Running { pid, .. } | ProcStatus::Ready { pid, .. } => Some(pid),
            _ => self
                .child
                .lock()
                .expect("child mutex poisoned")
                .as_ref()
                .and_then(|c| c.id()),
        }
    }

    /// Return a snapshot of the current [`ProcStatus`] for this proc.
    ///
    /// This is a *live view* of the lifecycle state as tracked by
    /// [`BootstrapProcManager`]. It reflects what is currently known
    /// about the underlying OS process (e.g., `Starting`, `Running`,
    /// `Stopping`, etc.).
    ///
    /// Internally this reads the mutex-guarded status. Use this when
    /// you just need a synchronous snapshot; use
    /// [`BootstrapProcHandle::watch`] or
    /// [`BootstrapProcHandle::changed`] if you want to await
    /// transitions asynchronously.
    #[must_use]
    pub fn status(&self) -> ProcStatus {
        // Source of truth for now is the mutex. We broadcast via
        // `watch` in `transition`, but callers that want a
        // synchronous snapshot should read the guarded value.
        self.status.lock().expect("status mutex poisoned").clone()
    }

    /// Atomically apply a state transition while holding the status
    /// lock, and send the updated value on the watch channel **while
    /// still holding the lock**. This guarantees the mutex state and
    /// the broadcast value stay in sync and avoids reordering between
    /// concurrent transitions.
    fn transition<F>(&self, f: F) -> bool
    where
        F: FnOnce(&mut ProcStatus) -> bool,
    {
        let mut guard = self.status.lock().expect("status mutex poisoned");
        let _before = guard.clone();
        let changed = f(&mut guard);
        if changed {
            // Publish while still holding the lock to preserve order.
            let _ = self.tx.send(guard.clone());
        }
        changed
    }

    /// Transition this proc into the [`ProcStatus::Running`] state.
    ///
    /// Called internally once the child OS process has been spawned
    /// and we can observe a valid `pid`. Records the `pid` and the
    /// `started_at` timestamp so that callers can query them later
    /// via [`BootstrapProcHandle::status`] or
    /// [`BootstrapProcHandle::pid`].
    ///
    /// This is a best-effort marker: it reflects that the process
    /// exists at the OS level, but does not guarantee that the proc
    /// has completed bootstrap or is fully ready.
    pub(crate) fn mark_running(&self, pid: u32, started_at: SystemTime) -> bool {
        self.transition(|st| match *st {
            ProcStatus::Starting => {
                *st = ProcStatus::Running { pid, started_at };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Running; leaving status unchanged",
                    *st
                );
                false
            }
        })
    }

    /// Attempt to transition this proc into the [`ProcStatus::Ready`]
    /// state.
    ///
    /// This records the process ID, start time, listening address,
    /// and agent once the proc has successfully started and is ready
    /// to serve.
    ///
    /// Returns `true` if the transition succeeded (from `Starting` or
    /// `Running`), or `false` if the current state did not allow
    /// moving to `Ready`. In the latter case the state is left
    /// unchanged and a warning is logged.
    pub(crate) fn mark_ready(
        &self,
        pid: u32,
        started_at: SystemTime,
        addr: ChannelAddr,
        agent: ActorRef<ProcMeshAgent>,
    ) -> bool {
        self.transition(|st| match st {
            ProcStatus::Starting | ProcStatus::Running { .. } => {
                *st = ProcStatus::Ready {
                    pid,
                    started_at,
                    addr,
                    agent,
                };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Ready; leaving status unchanged",
                    st
                );
                false
            }
        })
    }

    /// Record that a stop has been requested for the proc (e.g. a
    /// graceful shutdown via SIGTERM), but the underlying process has
    /// not yet fully exited.
    pub(crate) fn mark_stopping(&self) -> bool {
        self.transition(|st| match *st {
            ProcStatus::Starting | ProcStatus::Running { .. } | ProcStatus::Ready { .. } => {
                *st = ProcStatus::Stopping;
                true
            }
            _ => {
                tracing::debug!(
                    "illegal transition: {:?} -> Stopping; leaving status unchanged",
                    *st
                );
                false
            }
        })
    }

    /// Record that the process has exited normally with the given
    /// exit code.
    pub(crate) fn mark_stopped(&self, exit_code: i32) -> bool {
        self.transition(|st| match *st {
            ProcStatus::Starting
            | ProcStatus::Running { .. }
            | ProcStatus::Ready { .. }
            | ProcStatus::Stopping => {
                *st = ProcStatus::Stopped { exit_code };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Stopped; leaving status unchanged",
                    *st
                );
                false
            }
        })
    }

    /// Record that the process was killed by the given signal (e.g.
    /// SIGKILL, SIGTERM).
    pub(crate) fn mark_killed(&self, signal: i32, core_dumped: bool) -> bool {
        self.transition(|st| match *st {
            ProcStatus::Starting
            | ProcStatus::Running { .. }
            | ProcStatus::Ready { .. }
            | ProcStatus::Stopping => {
                *st = ProcStatus::Killed {
                    signal,
                    core_dumped,
                };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Killed; leaving status unchanged",
                    *st
                );
                false
            }
        })
    }

    /// Record that the proc or its process failed for an unexpected
    /// reason (bootstrap error, spawn failure, etc.).
    pub(crate) fn mark_failed<S: Into<String>>(&self, reason: S) -> bool {
        self.transition(|st| match *st {
            ProcStatus::Starting | ProcStatus::Running { .. } | ProcStatus::Ready { .. } => {
                *st = ProcStatus::Failed {
                    reason: reason.into(),
                };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Failed; leaving status unchanged",
                    *st
                );
                false
            }
        })
    }

    /// Wait until the proc has reached a terminal state and return
    /// it.
    ///
    /// Terminal means [`ProcStatus::Stopped`],
    /// [`ProcStatus::Killed`], or [`ProcStatus::Failed`]. If the
    /// current status is already terminal, returns immediately.
    ///
    /// Non-consuming: `BootstrapProcHandle` is a supervisor, not the
    /// owner of the OS process, so you can call `wait()` from
    /// multiple tasks concurrently.
    ///
    /// Implementation detail: listens on this handle's `watch`
    /// channel. It snapshots the current status, and if not terminal
    /// awaits the next change. If the channel closes unexpectedly,
    /// returns the last observed status.
    ///
    /// Mirrors `tokio::process::Child::wait()`, but yields the
    /// higher-level [`ProcStatus`] instead of an `ExitStatus`.
    #[must_use]
    pub async fn wait(&self) -> ProcStatus {
        let mut rx = self.watch();
        loop {
            let st = rx.borrow().clone();
            if st.is_exit() {
                return st;
            }
            // If the channel closes, return the last observed value.
            if rx.changed().await.is_err() {
                return st;
            }
        }
    }

    /// Wait until the proc reaches the [`ProcStatus::Ready`] state.
    ///
    /// If the proc hits a terminal state ([`ProcStatus::Stopped`],
    /// [`ProcStatus::Killed`], or [`ProcStatus::Failed`]) before ever
    /// becoming `Ready`, this returns
    /// `Err(ReadyError::Terminal(status))`. If the internal watch
    /// channel closes unexpectedly, this returns
    /// `Err(ReadyError::ChannelClosed)`. Otherwise it returns
    /// `Ok(())` when `Ready` is first observed.
    ///
    /// Non-consuming: `BootstrapProcHandle` is a supervisor, not the
    /// owner; multiple tasks may await `ready()` concurrently.
    /// `Stopping` is not treated as terminal here; we continue
    /// waiting until `Ready` or a terminal state is seen.
    ///
    /// Companion to [`BootstrapProcHandle::wait`]: `wait()` resolves
    /// on exit; `ready()` resolves on startup.
    pub async fn ready(&self) -> Result<(), ReadyError> {
        let mut rx = self.watch();
        loop {
            let st = rx.borrow().clone();
            match &st {
                ProcStatus::Ready { .. } => return Ok(()),
                s if s.is_exit() => return Err(ReadyError::Terminal(st)),
                _non_terminal => {
                    if rx.changed().await.is_err() {
                        return Err(ReadyError::ChannelClosed);
                    }
                }
            }
        }
    }
}

impl hyperactor::host::ProcHandle for BootstrapProcHandle {
    type Agent = ProcMeshAgent;

    #[inline]
    fn proc_id(&self) -> &ProcId {
        &self.proc_id
    }

    #[inline]
    fn addr(&self) -> Option<ChannelAddr> {
        match &*self.status.lock().expect("status mutex poisoned") {
            ProcStatus::Ready { addr, .. } => Some(addr.clone()),
            _ => None,
        }
    }

    #[inline]
    fn agent_ref(&self) -> Option<ActorRef<Self::Agent>> {
        match &*self.status.lock().expect("status mutex poisoned") {
            ProcStatus::Ready { agent, .. } => Some(agent.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Named, Serialize, Deserialize, Clone)]
pub struct BootstrapProcManagerParams {
    pub program: std::path::PathBuf,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

/// A process manager for launching and supervising **bootstrap
/// processes** (via the [`bootstrap`] entry point).
///
/// `BootstrapProcManager` is the host-side runtime for external procs
/// in a hyperactor mesh. It spawns the configured bootstrap binary,
/// forwards each child's stdout/stderr into the logging channel,
/// tracks lifecycle state through [`BootstrapProcHandle`] /
/// [`ProcStatus`], and ensures best-effort teardown on drop.
///
/// Internally it maintains two maps:
/// - [`children`]: async map from [`ProcId`] to
///   [`BootstrapProcHandle`], used for lifecycle queries (`status`) and
///   exit monitoring.
/// - [`pid_table`]: synchronous map from [`ProcId`] to raw PIDs, used
///   in [`Drop`] to synchronously send `SIGKILL` to any still-running
///   children.
///
/// Together these provide both a queryable, real-time status surface
/// and a deterministic cleanup path, so no child processes are left
/// orphaned if the manager itself is dropped.
#[derive(Debug)]
pub struct BootstrapProcManager {
    /// Path to the bootstrap binary that this manager will launch for
    /// each proc.
    program: std::path::PathBuf,
    /// argv[0], if specified
    arg0: Option<String>,
    /// argv[1..]
    args: Vec<String>,
    /// Async registry of running children, keyed by [`ProcId`]. Holds
    /// [`BootstrapProcHandle`]s so callers can query or monitor
    /// status.
    children: Arc<tokio::sync::Mutex<HashMap<ProcId, BootstrapProcHandle>>>,
    /// Synchronous table of raw PIDs, keyed by [`ProcId`]. Used
    /// exclusively in the [`Drop`] impl to send `SIGKILL` without
    /// needing async context.
    pid_table: Arc<std::sync::Mutex<HashMap<ProcId, u32>>>,
    env: HashMap<String, String>,
}

impl Drop for BootstrapProcManager {
    /// Drop implementation for [`BootstrapProcManager`].
    ///
    /// Ensures that when the manager itself is dropped, any child
    /// processes it spawned are also terminated. This is a
    /// best-effort cleanup mechanism: the manager iterates over its
    /// recorded PID table and sends `SIGKILL` to each one.
    ///
    /// Notes:
    /// - Uses a synchronous `Mutex` so it can be locked safely in
    ///   `Drop` without async context.
    /// - Safety: sending `SIGKILL` is low-level but safe here because
    ///   it only instructs the OS to terminate the process. The only
    ///   risk is PID reuse, in which case an unrelated process might
    ///   be signaled. This is accepted for shutdown semantics.
    /// - This makes `BootstrapProcManager` robust against leaks:
    ///   dropping it should not leave stray child processes running.
    fn drop(&mut self) {
        if let Ok(table) = self.pid_table.lock() {
            for (proc_id, pid) in table.iter() {
                // SAFETY: We own these PIDs because they were spawned and recorded
                // by this manager. Sending SIGKILL is safe here since it does not
                // dereference any memory; it only instructs the OS to terminate the
                // process. The worst-case outcome is that the PID has already exited
                // and been reused, in which case the signal may affect an unrelated
                // process. We accept that risk in Drop semantics because this path
                // is only used for cleanup at shutdown.
                unsafe {
                    libc::kill(*pid as i32, libc::SIGKILL);
                }
                tracing::info!(
                    "BootstrapProcManager::drop: sent SIGKILL to pid {} for {:?}",
                    pid,
                    proc_id
                );
            }
        }
    }
}

impl BootstrapProcManager {
    /// Construct a new [`BootstrapProcManager`] that will launch
    /// procs using the given program binary.
    ///
    /// This is the general entry point when you want to manage procs
    /// backed by a specific binary path (e.g. a bootstrap
    /// trampoline).
    #[allow(dead_code)]
    pub(crate) fn new(program: std::path::PathBuf) -> Self {
        Self {
            program,
            arg0: None,
            args: Vec::new(),
            children: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            pid_table: Arc::new(std::sync::Mutex::new(HashMap::new())),
            env: HashMap::new(),
        }
    }

    /// Convenience constructor that resolves the current executable
    /// (`std::env::current_exe`) and uses that as the bootstrap
    /// binary. The program arguments are also captured and used to
    /// configure child processes.
    ///
    /// Useful when the proc manager should re-exec itself as the
    /// child program. Returns an `io::Result` since querying the
    /// current executable path can fail.
    pub(crate) fn new_current_exe() -> io::Result<Self> {
        // Ok(Self::new(std::env::current_exe()?))
        let mut args: VecDeque<String> = std::env::args().collect();
        let arg0 = args.pop_front();

        Ok(Self {
            program: std::env::current_exe()?,
            arg0,
            args: args.into(),
            children: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            pid_table: Arc::new(std::sync::Mutex::new(HashMap::new())),
            env: HashMap::new(),
        })
    }

    pub(crate) fn from_params(params: BootstrapProcManagerParams) -> Self {
        Self {
            program: params.program,
            arg0: None,
            args: params.args,
            children: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            pid_table: Arc::new(std::sync::Mutex::new(HashMap::new())),
            env: params.env,
        }
    }

    /// Test-only constructor that uses the Buck-built
    /// `monarch/hyperactor_mesh/bootstrap` binary.
    ///
    /// Intended for integration tests where we need to spawn real
    /// bootstrap processes under proc manager control. Not available
    /// outside of test builds.
    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self::new(buck_resources::get("monarch/hyperactor_mesh/bootstrap").unwrap())
    }

    /// Return the current [`ProcStatus`] for the given [`ProcId`], if
    /// the proc is known to this manager.
    ///
    /// This queries the live [`BootstrapProcHandle`] stored in the
    /// manager's internal map. It provides an immediate snapshot of
    /// lifecycle state (`Starting`, `Running`, `Stopping`, `Stopped`,
    /// etc.).
    ///
    /// Returns `None` if the manager has no record of the proc (e.g.
    /// never spawned here, or entry already removed).
    pub async fn status(&self, proc_id: &ProcId) -> Option<ProcStatus> {
        self.children.lock().await.get(proc_id).map(|h| h.status())
    }

    fn spawn_exit_monitor(&self, proc_id: ProcId, handle: BootstrapProcHandle) {
        let pid_table = Arc::clone(&self.pid_table);

        let maybe_child = {
            let mut guard = handle.child.lock().expect("child mutex");
            let taken = guard.take();
            debug_assert!(guard.is_none(), "no Child should remain in handles");

            taken
        };

        let Some(mut child) = maybe_child else {
            tracing::debug!("proc {proc_id}: child was already taken; skipping wait");
            return;
        };

        tokio::spawn(async move {
            match child.wait().await {
                Ok(status) => {
                    if let Some(sig) = status.signal() {
                        let _ = handle.mark_killed(sig, status.core_dumped());
                        if let Ok(mut table) = pid_table.lock() {
                            table.remove(&proc_id);
                        }
                    } else if let Some(code) = status.code() {
                        let _ = handle.mark_stopped(code);
                        if let Ok(mut table) = pid_table.lock() {
                            table.remove(&proc_id);
                        }
                    } else {
                        debug_assert!(
                            false,
                            "unreachable: process terminated with neither signal nor exit code"
                        );
                        tracing::error!(
                            "proc {proc_id}: unreachable exit status (no code, no signal)"
                        );
                        let _ = handle.mark_failed("process exited with unknown status");
                        if let Ok(mut table) = pid_table.lock() {
                            table.remove(&proc_id);
                        }
                    }
                }
                Err(e) => {
                    let _ = handle.mark_failed(format!("wait() failed: {e}"));
                    if let Ok(mut table) = pid_table.lock() {
                        table.remove(&proc_id);
                    }
                }
            }
        });
    }
}

#[async_trait]
impl ProcManager for BootstrapProcManager {
    type Handle = BootstrapProcHandle;

    /// Return the [`ChannelTransport`] used by this proc manager.
    ///
    /// For `BootstrapProcManager` this is always
    /// [`ChannelTransport::Unix`], since all procs are spawned
    /// locally on the same host and communicate over Unix domain
    /// sockets.
    fn transport(&self) -> ChannelTransport {
        ChannelTransport::Unix
    }

    /// Launch a new proc under this [`BootstrapProcManager`].
    ///
    /// Spawns the configured bootstrap binary (`self.program`) in a
    /// fresh child process, passing environment variables that
    /// describe the [`BootstrapMode::Proc`] (proc ID, backend
    /// address, callback address).
    ///
    /// Responsibilities performed here:
    /// - Create a one-shot callback channel so the child can confirm
    ///   successful bootstrap and return its mailbox address plus agent
    ///   reference.
    /// - Spawn the OS process with stdout/stderr piped.
    /// - Stamp the new [`BootstrapProcHandle`] as
    ///   [`ProcStatus::Running`] once a PID is observed.
    /// - Wire stdout/stderr pipes into local writers and forward them
    ///   over the logging channel (`BOOTSTRAP_LOG_CHANNEL`).
    /// - Insert the handle into the manager's children map and start
    ///   an exit monitor to track process termination.
    ///
    /// Returns a [`BootstrapProcHandle`] that exposes the child
    /// process's lifecycle (status, wait/ready, termination). Errors
    /// are surfaced as [`HostError`].
    ///
    /// Note: graceful shutdown (SIGTERM → wait → SIGKILL) is not yet
    /// implemented; see the `terminate_all` TODO.
    async fn spawn(
        &self,
        proc_id: ProcId,
        backend_addr: ChannelAddr,
    ) -> Result<Self::Handle, HostError> {
        let (callback_addr, mut callback_rx) =
            channel::serve(ChannelAddr::any(ChannelTransport::Unix))?;

        let mode = Bootstrap::Proc {
            proc_id: proc_id.clone(),
            backend_addr,
            callback_addr,
        };
        let mut cmd = Command::new(&self.program);
        if let Some(arg0) = &self.arg0 {
            cmd.arg0(arg0);
        }
        for arg in &self.args {
            cmd.arg(arg);
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd.env(
            "HYPERACTOR_MESH_BOOTSTRAP_MODE",
            mode.to_env_safe_string()
                .map_err(|e| HostError::ProcessConfigurationFailure(proc_id.clone(), e.into()))?,
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

        let log_channel = ChannelAddr::any(ChannelTransport::Unix);
        cmd.env(BOOTSTRAP_LOG_CHANNEL, log_channel.to_string());

        let child = cmd
            .spawn()
            .map_err(|e| HostError::ProcessSpawnFailure(proc_id.clone(), e))?;
        let pid = child.id().unwrap_or_default();

        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        if let Some(pid) = handle.pid() {
            handle.mark_running(pid, hyperactor::clock::RealClock.system_time_now());
            if let Ok(mut table) = self.pid_table.lock() {
                table.insert(proc_id.clone(), pid);
            }
        }

        // Writers: tee to local (stdout/stderr or file) + send over
        // channel
        let (mut out_writer, mut err_writer) = create_log_writers(0, log_channel.clone(), pid)
            .unwrap_or_else(|_| (Box::new(tokio::io::stdout()), Box::new(tokio::io::stderr())));

        // Take the pipes from the child.
        {
            let mut guard = handle.child.lock().expect("child mutex poisoned");

            if let Some(child) = guard.as_mut() {
                if let Some(mut out) = child.stdout.take() {
                    tokio::spawn(async move {
                        let _ = tokio::io::copy(&mut out, &mut out_writer).await;
                    });
                }
                if let Some(mut err) = child.stderr.take() {
                    tokio::spawn(async move {
                        let _ = tokio::io::copy(&mut err, &mut err_writer).await;
                    });
                }
            } else {
                tracing::debug!("proc {proc_id}: child already taken before wiring IO");
            }
        }

        // Retain handle for lifecycle mgt.
        {
            let mut children = self.children.lock().await;
            children.insert(proc_id.clone(), handle.clone());
        }

        // Kick off an exit monitor that updates ProcStatus when the
        // OS process ends
        self.spawn_exit_monitor(proc_id.clone(), handle.clone());

        let h = handle.clone();
        let pid_table = Arc::clone(&self.pid_table);
        tokio::spawn(async move {
            match callback_rx.recv().await {
                Ok((addr, agent)) => {
                    let pid = match h.pid() {
                        Some(p) => p,
                        None => {
                            tracing::warn!("mark_ready called with missing pid; using 0");
                            0
                        }
                    };
                    let started_at = RealClock.system_time_now();
                    let _ = h.mark_ready(pid, started_at, addr, agent);
                }
                Err(e) => {
                    // Child never called back; record failure.
                    let _ = h.mark_failed(format!("bootstrap callback failed: {e}"));
                    // Cleanup pid table entry if it was set.
                    if let Ok(mut table) = pid_table.lock() {
                        table.remove(&proc_id);
                    }
                }
            }
        });

        // Callers do `handle.read().await` for mesh readiness.
        Ok(handle)
    }
}

/// Entry point to processes managed by hyperactor_mesh. Any process that is part
/// of a hyperactor_mesh program should call [`bootstrap`], which then configures
/// the process according to how it is invoked.
///
/// If bootstrap returns any error, it is defunct from the point of view of hyperactor_mesh,
/// and the process should likely exit:
///
/// ```ignore
/// let err = hyperactor_mesh::bootstrap().await;
/// tracing::error("could not bootstrap mesh process: {}", err);
/// std::process::exit(1);
/// ```
///
/// Use [`bootstrap_or_die`] to implement this behavior directly.
pub async fn bootstrap() -> anyhow::Error {
    let boot = ok!(Bootstrap::get_from_env()).unwrap_or_else(Bootstrap::default);
    boot.bootstrap().await
}

/// Bootstrap a v0 proc mesh. This launches a control process that responds to
/// Allocator2Process messages, conveying its own state in Process2Allocator messages.
///
/// The bootstrapping process is controlled by the
/// following environment variables:
///
/// - `HYPERACTOR_MESH_BOOTSTRAP_ADDR`: the channel address to which Process2Allocator messages
///   should be sent.
/// - `HYPERACTOR_MESH_INDEX`: an index used to identify this process to the allocator.
async fn bootstrap_v0_proc_mesh() -> anyhow::Error {
    pub async fn go() -> Result<(), anyhow::Error> {
        let procs = Arc::new(tokio::sync::Mutex::new(Vec::<Proc>::new()));
        let procs_for_cleanup = procs.clone();
        let _cleanup_guard = hyperactor::register_signal_cleanup_scoped(Box::pin(async move {
            for proc_to_stop in procs_for_cleanup.lock().await.iter_mut() {
                if let Err(err) = proc_to_stop
                    .destroy_and_wait::<()>(Duration::from_millis(10), None)
                    .await
                {
                    tracing::error!(
                        "error while stopping proc {}: {}",
                        proc_to_stop.proc_id(),
                        err
                    );
                }
            }
        }));

        let bootstrap_addr: ChannelAddr = std::env::var(BOOTSTRAP_ADDR_ENV)
            .map_err(|err| anyhow::anyhow!("read `{}`: {}", BOOTSTRAP_ADDR_ENV, err))?
            .parse()?;
        let bootstrap_index: usize = std::env::var(BOOTSTRAP_INDEX_ENV)
            .map_err(|err| anyhow::anyhow!("read `{}`: {}", BOOTSTRAP_INDEX_ENV, err))?
            .parse()?;
        let listen_addr = ChannelAddr::any(bootstrap_addr.transport());
        let (serve_addr, mut rx) = channel::serve(listen_addr)?;
        let tx = channel::dial(bootstrap_addr.clone())?;

        let (rtx, mut return_channel) = oneshot::channel();
        tx.try_post(
            Process2Allocator(bootstrap_index, Process2AllocatorMessage::Hello(serve_addr)),
            rtx,
        )?;
        tokio::spawn(exit_if_missed_heartbeat(bootstrap_index, bootstrap_addr));

        let mut the_msg;

        tokio::select! {
            msg = rx.recv() => {
                the_msg = msg;
            }
            returned_msg = &mut return_channel => {
                match returned_msg {
                    Ok(msg) => {
                        return Err(anyhow::anyhow!("Hello message was not delivered:{:?}", msg));
                    }
                    Err(_) => {
                        the_msg = rx.recv().await;
                    }
                }
            }
        }
        loop {
            let _ = hyperactor::tracing::info_span!("wait_for_next_message_from_mesh_agent");
            match the_msg? {
                Allocator2Process::StartProc(proc_id, listen_transport) => {
                    let (proc, mesh_agent) = ProcMeshAgent::bootstrap(proc_id.clone()).await?;
                    let (proc_addr, proc_rx) = channel::serve(ChannelAddr::any(listen_transport))?;
                    let handle = proc.clone().serve(proc_rx);
                    drop(handle); // linter appeasement; it is safe to drop this future
                    tx.send(Process2Allocator(
                        bootstrap_index,
                        Process2AllocatorMessage::StartedProc(
                            proc_id.clone(),
                            mesh_agent.bind(),
                            proc_addr,
                        ),
                    ))
                    .await?;
                    procs.lock().await.push(proc);
                }
                Allocator2Process::StopAndExit(code) => {
                    tracing::info!("stopping procs with code {code}");
                    {
                        for proc_to_stop in procs.lock().await.iter_mut() {
                            if let Err(err) = proc_to_stop
                                .destroy_and_wait::<()>(Duration::from_millis(10), None)
                                .await
                            {
                                tracing::error!(
                                    "error while stopping proc {}: {}",
                                    proc_to_stop.proc_id(),
                                    err
                                );
                            }
                        }
                    }
                    tracing::info!("exiting with {code}");
                    std::process::exit(code);
                }
                Allocator2Process::Exit(code) => {
                    tracing::info!("exiting with {code}");
                    std::process::exit(code);
                }
            }
            the_msg = rx.recv().await;
        }
    }

    go().await.unwrap_err()
}

/// A variant of [`bootstrap`] that logs the error and exits the process
/// if bootstrapping fails.
pub async fn bootstrap_or_die() -> ! {
    let err = bootstrap().await;
    let _ = writeln!(Debug, "failed to bootstrap mesh process: {}", err);
    tracing::error!("failed to bootstrap mesh process: {}", err);
    std::process::exit(1)
}

#[derive(enum_as_inner::EnumAsInner)]
enum DebugSink {
    File(std::fs::File),
    Sink,
}

impl DebugSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            DebugSink::File(f) => f.write(buf),
            DebugSink::Sink => Ok(buf.len()),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            DebugSink::File(f) => f.flush(),
            DebugSink::Sink => Ok(()),
        }
    }
}

fn debug_sink() -> &'static Mutex<DebugSink> {
    static DEBUG_SINK: OnceLock<Mutex<DebugSink>> = OnceLock::new();
    DEBUG_SINK.get_or_init(|| {
        let debug_path = {
            let mut p = std::env::temp_dir();
            if let Ok(user) = std::env::var("USER") {
                p.push(user);
            }
            std::fs::create_dir_all(&p).ok();
            p.push("monarch-bootstrap-debug.log");
            p
        };
        let sink = if debug_path.exists() {
            match OpenOptions::new()
                .append(true)
                .create(true)
                .open(debug_path.clone())
            {
                Ok(f) => DebugSink::File(f),
                Err(e) => {
                    eprintln!(
                        "failed to open {} for bootstrap debug logging",
                        debug_path.display()
                    );
                    DebugSink::Sink
                }
            }
        } else {
            DebugSink::Sink
        };
        Mutex::new(sink)
    })
}

/// A bootstrap specific debug writer. If the file /tmp/monarch-bootstrap-debug.log
/// exists, then the writer's destination is that file; otherwise it discards all writes.
struct Debug;

impl Debug {
    fn is_active() -> bool {
        debug_sink().lock().unwrap().is_file()
    }
}

impl Write for Debug {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        debug_sink().lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        debug_sink().lock().unwrap().flush()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Stdio;

    use hyperactor::ActorId;
    use hyperactor::ActorRef;
    use hyperactor::ProcId;
    use hyperactor::WorldId;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::clock::RealClock;
    use hyperactor::id;
    use tokio::process::Command;

    use super::*;

    // Helper: Avoid repeating
    // `ChannelAddr::any(ChannelTransport::Unix)`.
    fn any_addr_for_test() -> ChannelAddr {
        ChannelAddr::any(ChannelTransport::Unix)
    }

    #[test]
    fn test_bootstrap_mode_env_string() {
        let values = [
            Bootstrap::default(),
            Bootstrap::Proc {
                proc_id: id!(foo[0]),
                backend_addr: ChannelAddr::any(ChannelTransport::Tcp),
                callback_addr: ChannelAddr::any(ChannelTransport::Unix),
            },
        ];

        for value in values {
            let safe = value.to_env_safe_string().unwrap();
            assert_eq!(value, Bootstrap::from_env_safe_string(&safe).unwrap());
        }
    }

    #[tokio::test]
    async fn test_child_terminated_on_manager_drop() {
        use std::path::PathBuf;
        use std::process::Stdio;

        use tokio::process::Command;
        use tokio::time::Duration;

        // Manager; program path is irrelevant for this test.
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/true"));

        // Spawn a long-running child process (sleep 30) with
        // kill_on_drop(true).
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let child = cmd.spawn().expect("spawn sleep");
        let pid = child.id().expect("pid");

        // Insert into the manager's children map (simulates a spawned
        // proc).
        let proc_id = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "test".to_string());
        {
            let mut children = manager.children.lock().await;
            children.insert(
                proc_id.clone(),
                BootstrapProcHandle::new(proc_id.clone(), child),
            );
        }

        // (Linux-only) Verify the process exists before drop.
        #[cfg(target_os = "linux")]
        {
            let path = format!("/proc/{}", pid);
            assert!(
                std::fs::metadata(&path).is_ok(),
                "expected /proc/{pid} to exist before drop"
            );
        }

        // Drop the manager. We don't rely on `Child::kill_on_drop`.
        // The exit monitor owns the Child; the manager's Drop uses
        // `pid_table` to best-effort SIGKILL any recorded PIDs. This
        // should terminate the child.
        drop(manager);

        // Poll until the /proc entry disappears OR the state is
        // Zombie.
        #[cfg(target_os = "linux")]
        {
            let deadline = std::time::Instant::now() + Duration::from_millis(1500);
            let proc_dir = format!("/proc/{}", pid);
            let status_file = format!("{}/status", proc_dir);

            let mut ok = false;
            while std::time::Instant::now() < deadline {
                match std::fs::read_to_string(&status_file) {
                    Ok(s) => {
                        if let Some(state_line) = s.lines().find(|l| l.starts_with("State:")) {
                            if state_line.contains('Z') {
                                // Only 'Z' (Zombie) is acceptable.
                                ok = true;
                                break;
                            } else {
                                // Still alive (e.g. 'R', 'S', etc.). Give it more time.
                            }
                        }
                    }
                    Err(_) => {
                        // /proc/<pid>/status no longer exists -> fully gone
                        ok = true;
                        break;
                    }
                }
                RealClock.sleep(Duration::from_millis(100)).await;
            }

            assert!(ok, "expected /proc/{pid} to be gone or zombie after drop");
        }

        // On non-Linux, absence of panics/hangs is the signal; PID
        // probing is platform-specific.
    }

    #[tokio::test]
    async fn test_v1_child_logging() {
        use hyperactor::channel;
        use hyperactor::data::Serialized;
        use hyperactor::mailbox::BoxedMailboxSender;
        use hyperactor::mailbox::DialMailboxRouter;
        use hyperactor::mailbox::MailboxServer;
        use hyperactor::proc::Proc;

        use crate::bootstrap::BOOTSTRAP_LOG_CHANNEL;
        use crate::logging::LogClientActor;
        use crate::logging::LogClientMessageClient;
        use crate::logging::LogForwardActor;
        use crate::logging::LogMessage;
        use crate::logging::OutputTarget;
        use crate::logging::test_tap;

        let router = DialMailboxRouter::new();
        let (proc_addr, proc_rx) =
            channel::serve(ChannelAddr::any(ChannelTransport::Unix)).unwrap();
        let proc = Proc::new(id!(client[0]), BoxedMailboxSender::new(router.clone()));
        proc.clone().serve(proc_rx);
        router.bind(id!(client[0]).into(), proc_addr.clone());
        let (client, _handle) = proc.instance("client").unwrap();

        let (tap_tx, mut tap_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        test_tap::install(tap_tx);

        let log_channel = ChannelAddr::any(ChannelTransport::Unix);
        // SAFETY: unit-test scoped env var
        unsafe {
            std::env::set_var(BOOTSTRAP_LOG_CHANNEL, log_channel.to_string());
        }

        // Spawn the log client and disable aggregation (immediate
        // print + tap push).
        let log_client: ActorRef<LogClientActor> =
            proc.spawn("log_client", ()).await.unwrap().bind();
        log_client.set_aggregate(&client, None).await.unwrap();

        // Spawn the forwarder in this proc (it will serve BOOTSTRAP_LOG_CHANNEL).
        let _log_forwarder: ActorRef<LogForwardActor> = proc
            .spawn("log_forwarder", log_client.clone())
            .await
            .unwrap()
            .bind();

        // Send a fake log message as if it came from the proc
        // manager's writer.
        let tx = channel::dial::<LogMessage>(log_channel.clone()).unwrap();
        tx.post(LogMessage::Log {
            hostname: "testhost".into(),
            pid: 12345,
            output_target: OutputTarget::Stdout,
            payload: Serialized::serialize(&"hello from child".to_string()).unwrap(),
        });

        // Assert we see it via the tap.
        // Give it up to 2 seconds to travel through forwarder ->
        // client -> print_log_line -> tap.
        let line = RealClock
            .timeout(Duration::from_secs(2), tap_rx.recv())
            .await
            .expect("timed out waiting for log line")
            .expect("tap channel closed unexpectedly");
        assert!(
            line.contains("hello from child"),
            "log line did not appear via LogClientActor; got: {line}"
        );
    }

    mod proc_handle {

        use hyperactor::ActorId;
        use hyperactor::ActorRef;
        use hyperactor::ProcId;
        use hyperactor::WorldId;
        use hyperactor::host::ProcHandle;
        use tokio::process::Command;

        use super::super::*;
        use super::any_addr_for_test;

        // Helper: build a ProcHandle with a short-lived child
        // process. We don't rely on the actual process; we only
        // exercise the status transitions.
        fn handle_for_test() -> BootstrapProcHandle {
            // Spawn a trivial child that exits immediately.
            let child = Command::new("sh")
                .arg("-c")
                .arg("exit 0")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("failed to spawn test child process");

            let proc_id = ProcId::Ranked(WorldId("test".into()), 0);
            BootstrapProcHandle::new(proc_id, child)
        }

        async fn starting_to_running_ok() {
            let h = handle_for_test();
            assert!(matches!(h.status(), ProcStatus::Starting));
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            match h.status() {
                ProcStatus::Running { pid, started_at } => {
                    assert_eq!(pid, child_pid);
                    assert_eq!(started_at, child_started_at);
                }
                other => panic!("expected Running, got {other:?}"),
            }
        }

        #[tokio::test]
        async fn running_to_stopping_to_stopped_ok() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(h.mark_stopping());
            assert!(matches!(h.status(), ProcStatus::Stopping));
            assert!(h.mark_stopped(0));
            assert!(matches!(h.status(), ProcStatus::Stopped { exit_code: 0 }));
        }

        #[tokio::test]
        async fn running_to_killed_ok() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(h.mark_killed(9, true));
            assert!(matches!(
                h.status(),
                ProcStatus::Killed {
                    signal: 9,
                    core_dumped: true
                }
            ));
        }

        #[tokio::test]
        async fn running_to_failed_ok() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(h.mark_failed("bootstrap error"));
            match h.status() {
                ProcStatus::Failed { reason } => {
                    assert_eq!(reason, "bootstrap error");
                }
                other => panic!("expected Failed(\"bootstrap error\"), got {other:?}"),
            }
        }

        #[tokio::test]
        async fn illegal_transitions_are_rejected() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            // Starting -> Running is fine; second Running should be rejected.
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(!h.mark_running(child_pid, RealClock.system_time_now()));
            match h.status() {
                ProcStatus::Running { pid, .. } => assert_eq!(pid, child_pid),
                other => panic!("expected Running, got {other:?}"),
            }
            // Once Stopped, we can't go to Running/Killed/Failed/etc.
            assert!(h.mark_stopping());
            assert!(h.mark_stopped(0));
            assert!(!h.mark_running(child_pid, child_started_at));
            assert!(!h.mark_killed(9, false));
            assert!(!h.mark_failed("nope"));

            assert!(matches!(h.status(), ProcStatus::Stopped { exit_code: 0 }));
        }

        #[tokio::test]
        async fn transitions_from_ready_are_legal() {
            let h = handle_for_test();
            let addr = any_addr_for_test();
            // Mark Running.
            let pid = h.pid().expect("child should have a pid");
            let t0 = RealClock.system_time_now();
            assert!(h.mark_running(pid, t0));
            // Build a consistent AgentRef for Ready using the
            // handle's ProcId.
            let proc_id = <BootstrapProcHandle as ProcHandle>::proc_id(&h);
            let actor_id = ActorId(proc_id.clone(), "agent".into(), 0);
            let agent_ref: ActorRef<ProcMeshAgent> = ActorRef::attest(actor_id);
            // Ready -> Stopping -> Stopped should be legal.
            assert!(h.mark_ready(pid, t0, addr, agent_ref));
            assert!(h.mark_stopping());
            assert!(h.mark_stopped(0));
        }

        #[tokio::test]
        async fn ready_to_killed_is_legal() {
            let h = handle_for_test();
            let addr = any_addr_for_test();
            // Starting -> Running
            let pid = h.pid().expect("child should have a pid");
            let t0 = RealClock.system_time_now();
            assert!(h.mark_running(pid, t0));
            // Build a consistent AgentRef for Ready using the
            // handle's ProcId.
            let proc_id = <BootstrapProcHandle as ProcHandle>::proc_id(&h);
            let actor_id = ActorId(proc_id.clone(), "agent".into(), 0);
            let agent: ActorRef<ProcMeshAgent> = ActorRef::attest(actor_id);
            // Running -> Ready
            assert!(h.mark_ready(pid, t0, addr, agent));
            // Ready -> Killed
            assert!(h.mark_killed(9, false));
        }
    }

    #[tokio::test]
    async fn test_exit_monitor_updates_status_on_clean_exit() {
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/true"));

        // Spawn a fast-exiting child.
        let mut cmd = Command::new("/bin/true");
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        let child = cmd.spawn().expect("spawn true");

        let proc_id = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "clean".into());
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Put into manager & start monitor.
        {
            let mut children = manager.children.lock().await;
            children.insert(proc_id.clone(), handle.clone());
        }
        manager.spawn_exit_monitor(proc_id.clone(), handle.clone());
        {
            let guard = handle.child.lock().expect("child mutex");
            assert!(
                guard.is_none(),
                "expected Child to be taken by exit monitor"
            );
        }

        let st = handle.wait().await;
        assert!(matches!(st, ProcStatus::Stopped { .. }), "status={st:?}");
    }

    #[tokio::test]
    async fn test_exit_monitor_updates_status_on_kill() {
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/sleep"));

        // Spawn a process that will live long enough to kill.
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("5").stdout(Stdio::null()).stderr(Stdio::null());
        let child = cmd.spawn().expect("spawn sleep");
        let pid = child.id().expect("pid") as i32;

        // Register with the manager and start the exit monitor.
        let proc_id = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "killed".into());
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);
        {
            let mut children = manager.children.lock().await;
            children.insert(proc_id.clone(), handle.clone());
        }
        manager.spawn_exit_monitor(proc_id.clone(), handle.clone());
        {
            let guard = handle.child.lock().expect("child mutex");
            assert!(
                guard.is_none(),
                "expected Child to be taken by exit monitor"
            );
        }
        // Send SIGKILL to the process.
        #[cfg(unix)]
        // SAFETY: We own the child process we just spawned and have
        // its PID.
        // Sending SIGKILL is safe here because:
        //  - the PID comes directly from `child.id()`
        //  - the process is guaranteed to be in our test scope
        //  - we don't touch any memory via this call, only signal the
        //    OS
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }

        let st = handle.wait().await;
        match st {
            ProcStatus::Killed { signal, .. } => assert_eq!(signal, libc::SIGKILL),
            other => panic!("expected Killed(SIGKILL), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn watch_notifies_on_status_changes() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn");

        let proc_id = ProcId::Ranked(WorldId("test".into()), 1);
        let handle = BootstrapProcHandle::new(proc_id, child);
        let mut rx = handle.watch();

        // Starting -> Running
        let pid = handle.pid().unwrap_or(0);
        let now = RealClock.system_time_now();
        assert!(handle.mark_running(pid, now));
        rx.changed().await.ok(); // Observe the transition.
        match &*rx.borrow() {
            ProcStatus::Running { pid: p, started_at } => {
                assert_eq!(*p, pid);
                assert_eq!(*started_at, now);
            }
            s => panic!("expected Running, got {s:?}"),
        }

        // Running -> Stopped
        assert!(handle.mark_stopped(0));
        rx.changed().await.ok(); // Observe the transition.
        assert!(matches!(
            &*rx.borrow(),
            ProcStatus::Stopped { exit_code: 0 }
        ));
    }

    #[tokio::test]
    async fn ready_errs_if_process_exits_before_running() {
        // Child exits immediately.
        let child = Command::new("sh")
            .arg("-c")
            .arg("exit 7")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn");

        let proc_id = ProcId::Direct(
            ChannelAddr::any(ChannelTransport::Unix),
            "early-exit".into(),
        );
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Simulate the exit monitor doing its job directly here.
        // (Equivalent outcome: terminal state before Running.)
        assert!(handle.mark_stopped(7));

        // `ready()` should return Err with the terminal status.
        match handle.ready().await {
            Ok(()) => panic!("ready() unexpectedly succeeded"),
            Err(ReadyError::Terminal(ProcStatus::Stopped { exit_code })) => {
                assert_eq!(exit_code, 7)
            }
            Err(other) => panic!("expected Stopped(7), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn status_unknown_proc_is_none() {
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/true"));
        let unknown = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "nope".into());
        assert!(manager.status(&unknown).await.is_none());
    }

    #[tokio::test]
    async fn exit_monitor_child_already_taken_leaves_status_unchanged() {
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/sleep"));

        // Long-ish child so it's alive while we "steal" it.
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("5").stdout(Stdio::null()).stderr(Stdio::null());
        let child = cmd.spawn().expect("spawn sleep");

        let proc_id = ProcId::Direct(
            ChannelAddr::any(ChannelTransport::Unix),
            "already-taken".into(),
        );
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Insert, then deliberately consume the Child before starting
        // the monitor.
        {
            let mut children = manager.children.lock().await;
            children.insert(proc_id.clone(), handle.clone());
        }
        {
            let _ = handle.child.lock().expect("child mutex").take();
        }

        // This should not panic; monitor will see None and bail.
        manager.spawn_exit_monitor(proc_id.clone(), handle.clone());

        // Status should remain whatever it was (Starting in this
        // setup).
        assert!(matches!(
            manager.status(&proc_id).await,
            Some(ProcStatus::Starting)
        ));
    }

    #[tokio::test]
    async fn pid_none_after_exit_monitor_takes_child() {
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/sleep"));

        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("5").stdout(Stdio::null()).stderr(Stdio::null());
        let child = cmd.spawn().expect("spawn sleep");

        let proc_id = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "pid-none".into());
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Before monitor: we should still be able to read a pid.
        assert!(handle.pid().is_some());
        {
            let mut children = manager.children.lock().await;
            children.insert(proc_id.clone(), handle.clone());
        }
        // `spawn_exit_monitor` takes the Child synchronously before
        // spawning the task.
        manager.spawn_exit_monitor(proc_id.clone(), handle.clone());

        // Immediately after, the handle should no longer expose a pid
        // (Child is gone).
        assert!(handle.pid().is_none());
    }

    #[tokio::test]
    async fn starting_may_directly_be_marked_stopped() {
        // Unit-level transition check for the "process exited very
        // fast" case.
        let child = Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn true");

        let proc_id = ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "fast-exit".into());
        let handle = BootstrapProcHandle::new(proc_id, child);

        // Take the child (don't hold the mutex across an await).
        let mut c = {
            let mut guard = handle.child.lock().expect("child mutex");
            guard.take()
        }
        .expect("child already taken");

        let status = c.wait().await.expect("wait");
        let code = status.code().unwrap_or(0);
        assert!(handle.mark_stopped(code));

        assert!(matches!(
            handle.status(),
            ProcStatus::Stopped { exit_code: 0 }
        ));
    }

    #[tokio::test]
    async fn handle_ready_allows_waiters() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let proc_id = ProcId::Ranked(WorldId("test".into()), 42);
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        let pid = handle.pid().expect("child should have a pid");
        let started_at = RealClock.system_time_now();
        assert!(handle.mark_running(pid, started_at));

        let actor_id = ActorId(proc_id.clone(), "agent".into(), 0);
        let agent_ref: ActorRef<ProcMeshAgent> = ActorRef::attest(actor_id);

        // Pick any addr to carry in Ready (what the child would have
        // called back with).
        let ready_addr = any_addr_for_test();

        // Stamp Ready and assert ready().await unblocks.
        assert!(handle.mark_ready(pid, started_at, ready_addr.clone(), agent_ref));
        handle
            .ready()
            .await
            .expect("ready() should complete after Ready");

        // Sanity-check the Ready fields we control
        // (pid/started_at/addr).
        match handle.status() {
            ProcStatus::Ready {
                pid: p,
                started_at: t,
                addr: a,
                ..
            } => {
                assert_eq!(p, pid);
                assert_eq!(t, started_at);
                assert_eq!(a, ready_addr);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pid_behavior_across_states_running_ready_then_stopped() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn");

        let proc_id = ProcId::Ranked(WorldId("test".into()), 0);
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Running → pid() Some
        let pid = handle.pid().expect("initial Child::id");
        let t0 = RealClock.system_time_now();
        assert!(handle.mark_running(pid, t0));
        assert_eq!(handle.pid(), Some(pid));

        // Ready → pid() still Some even if Child taken
        let addr = any_addr_for_test();
        let agent = {
            let actor_id = ActorId(proc_id.clone(), "agent".into(), 0);
            ActorRef::<ProcMeshAgent>::attest(actor_id)
        };
        assert!(handle.mark_ready(pid, t0, addr, agent));
        {
            let _ = handle.child.lock().expect("child mutex").take();
        }
        assert_eq!(handle.pid(), Some(pid));

        // Terminal (Stopped) → pid() None
        assert!(handle.mark_stopped(0));
        assert_eq!(handle.pid(), None, "pid() should be None once terminal");
    }

    #[tokio::test]
    async fn pid_is_available_in_ready_even_after_child_taken() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn");

        let proc_id = ProcId::Ranked(WorldId("test".into()), 99);
        let handle = BootstrapProcHandle::new(proc_id.clone(), child);

        // Mark Running.
        let pid = handle.pid().expect("child should have pid (via Child::id)");
        let started_at = RealClock.system_time_now();
        assert!(handle.mark_running(pid, started_at));

        // Stamp Ready with synthetic addr/agent (attested).
        let addr = any_addr_for_test();
        let agent = {
            let actor_id = ActorId(proc_id.clone(), "agent".into(), 0);
            ActorRef::<ProcMeshAgent>::attest(actor_id)
        };
        assert!(handle.mark_ready(pid, started_at, addr, agent));

        // Simulate exit monitor taking the Child (so Child::id() is
        // no longer available).
        {
            let _ = handle.child.lock().expect("child mutex").take();
        }

        // **Regression check**: pid() must still return the cached
        // pid in Ready.
        assert_eq!(handle.pid(), Some(pid), "pid() should be cached in Ready");
    }

    #[test]
    fn display_running_includes_pid_and_uptime() {
        let started_at = RealClock.system_time_now() - Duration::from_secs(42);
        let st = ProcStatus::Running {
            pid: 1234,
            started_at,
        };

        let s = format!("{}", st);
        assert!(s.contains("1234"));
        assert!(s.contains("Running"));
        assert!(s.contains("42s"));
    }

    #[test]
    fn display_ready_includes_pid_and_addr() {
        let started_at = RealClock.system_time_now() - Duration::from_secs(5);
        let addr = ChannelAddr::any(ChannelTransport::Unix);
        let agent =
            ActorRef::attest(ProcId::Direct(addr.clone(), "proc".into()).actor_id("agent", 0));

        let st = ProcStatus::Ready {
            pid: 4321,
            started_at,
            addr: addr.clone(),
            agent,
        };

        let s = format!("{}", st);
        assert!(s.contains("4321")); // pid
        assert!(s.contains(&addr.to_string())); // addr
        assert!(s.contains("Ready"));
    }

    #[test]
    fn display_stopped_includes_exit_code() {
        let st = ProcStatus::Stopped { exit_code: 7 };
        let s = format!("{}", st);
        assert!(s.contains("Stopped"));
        assert!(s.contains("7"));
    }

    #[test]
    fn display_other_variants_does_not_panic() {
        let samples = vec![
            ProcStatus::Starting,
            ProcStatus::Stopping,
            ProcStatus::Ready {
                pid: 42,
                started_at: RealClock.system_time_now(),
                addr: ChannelAddr::any(ChannelTransport::Unix),
                agent: ActorRef::attest(
                    ProcId::Direct(ChannelAddr::any(ChannelTransport::Unix), "x".into())
                        .actor_id("agent", 0),
                ),
            },
            ProcStatus::Killed {
                signal: 9,
                core_dumped: false,
            },
            ProcStatus::Failed {
                reason: "boom".into(),
            },
        ];

        for st in samples {
            let _ = format!("{}", st); // Just make sure it doesn't panic.
        }
    }
}
