/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::env::VarError;
use std::future;
use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;

use async_trait::async_trait;
use base64::prelude::*;
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
use hyperactor::host::HostError;
use hyperactor::host::ProcManager;
use hyperactor::mailbox::MailboxServer;
use hyperactor::proc::Proc;
use serde::Deserialize;
use serde::Serialize;
use tokio::process::Child;
use tokio::process::Command;
use tokio::sync::oneshot;

use crate::logging::create_log_writers;
use crate::proc_mesh::mesh_agent::ProcMeshAgent;
use crate::v1;

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

/// The bootstrap mode configures the behavior of the bootstrap process.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapMode {
    // "v1" proc bootstrap
    Proc {
        /// The ProcId of the proc to be bootstrapped.
        proc_id: ProcId,
        /// The backend address to which messages are forwarded.
        /// See [`hyperactor::host`] for channel topology details.
        backend_addr: ChannelAddr,
        /// The callback address used to indicate successful spawning.
        callback_addr: ChannelAddr,
    },

    #[default]
    V0ProcMesh, // pass through to the v0 allocator
}

impl BootstrapMode {
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
}

/// Represents the lifecycle state of a **proc as hosted in an OS
/// process** managed by `BootstrapProcManager`.
///
/// Note: This type is deliberately distinct from [`ProcState`] and
/// [`ProcStopReason`] (see `alloc.rs`). Those types model allocator
/// *events* - e.g. "a proc was Created/Running/Stopped" - and are
/// consumed from an event stream during allocation. By contrast,
/// [`ProcStatus`] is a **live, queryable view**: it reflects the
/// current observed status of a running proc, as seen through a
/// [`ProcHandle`] API (stop, kill, status).
///
/// In short:
/// - `ProcState`/`ProcStopReason`: historical / event-driven model
/// - `ProcStatus`: immediate status surface for lifecycle control
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcStatus {
    /// The OS process has been spawned but is not yet fully running.
    /// (Process-level: child handle exists, no confirmation yet.)
    Starting,
    /// The OS process is alive and the proc is considered running.
    /// (Process-level: `pid` is known; Proc-level: bootstrap
    /// completed.)
    Running { pid: u32, started_at: SystemTime },
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

/// A handle to a proc launched by [`BootstrapProcManager`].
///
/// `ProcHandle` is the owner-facing API for controlling and observing
/// the lifecycle of a proc. It pairs the **logical proc identity**
/// (`ProcId`) with the underlying **OS process handle**
/// (`tokio::process::Child`) and a shared, queryable [`ProcStatus`].
///
/// Responsibilities:
/// - Retains the child process handle so the OS process can be
///   terminated, waited on, or killed when the handle is dropped.
/// - Tracks the current [`ProcStatus`] in an `Arc<Mutex<...>>`,
///   allowing concurrent observation and mutation from lifecycle
///   control paths.
/// - Provides the foundation for higher-level APIs such as
///   `terminate()`, `kill()`, `wait()`, and `status()`.
///
/// Relationship to types:
/// - [`ProcStatus`]: live status surface, updated by this handle.
/// - [`ProcState`]/[`ProcStopReason`] (in `alloc.rs`): event-driven
///   allocator view; not directly updated by this type.
#[derive(Clone)]
pub struct ProcHandle {
    /// Logical identity of the proc in the mesh.
    proc_id: ProcId,
    /// Current lifecycle status of the proc (see [`ProcStatus`]).
    /// Shared/mutable so observers and controllers can read/update.
    status: Arc<std::sync::Mutex<ProcStatus>>,
    /// Underlying OS process handle. Retained so the proc can be
    /// signaled (SIGTERM/SIGKILL) or awaited for exit. Wrapped in
    /// `Option` because ownership is consumed on wait.
    child: Arc<std::sync::Mutex<Option<Child>>>,
}

impl ProcHandle {
    /// Construct a new [`ProcHandle`] for a freshly spawned OS
    /// process hosting a proc.
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
        Self {
            proc_id,
            status: Arc::new(std::sync::Mutex::new(ProcStatus::Starting)),
            child: Arc::new(std::sync::Mutex::new(Some(child))),
        }
    }

    /// Return the logical proc identity in the mesh.
    #[inline]
    pub fn proc_id(&self) -> &ProcId {
        &self.proc_id
    }

    /// Return the OS process ID (`pid`) of the underlying child
    /// process, if it is still alive.
    ///
    /// This is a *process-level* identifier (from
    /// [`tokio::process::Child::id`]), not a logical [`ProcId`]. It
    /// may return `None` if the child handle has already been
    /// consumed (e.g. via `wait()`) or the process has exited.
    #[inline]
    pub fn pid(&self) -> Option<u32> {
        self.child
            .lock()
            .expect("child mutex poisoned")
            .as_ref()
            .and_then(|c| c.id())
    }

    /// TODO: status is currently backed by `Arc<Mutex<ProcStatus>>`
    /// with mark_* mutators. This is intentionally simple for now. In
    /// future, consider replacing with `tokio::sync::watch` so
    /// callers can both poll (`status()`) and subscribe to changes
    /// (`watch`). That would provide async lifecycle observation
    /// without changing the external API.
    /// Return a snapshot of the current [`ProcStatus`] for this proc.
    ///
    /// This is a *live view* of the lifecycle state as tracked by
    /// [`BootstrapProcManager`]. It reflects what is currently known
    /// about the underlying OS process (e.g., `Starting`, `Running`,
    /// `Stopping`, etc.).
    ///
    /// Note: this is a synchronous snapshot. Future improvements may
    /// allow async subscriptions (e.g. via `tokio::sync::watch`) to
    /// observe status transitions as they happen.
    #[must_use]
    pub fn status(&self) -> ProcStatus {
        self.status.lock().expect("status mutex poisoned").clone()
    }

    /// Transition this proc into the [`ProcStatus::Running`] state.
    ///
    /// Called internally once the child OS process has successfully
    /// started and the proc is considered live. Records the `pid` and
    /// the `started_at` timestamp so that callers can query them
    /// later via [`ProcHandle::status`] or [`ProcHandle::pid`].
    pub(crate) fn mark_running(&self, pid: u32, started_at: SystemTime) -> bool {
        let mut status = self.status.lock().unwrap();
        match *status {
            ProcStatus::Starting => {
                *status = ProcStatus::Running { pid, started_at };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Running; leaving status unchanged",
                    *status
                );
                false
            }
        }
    }

    /// Record that a stop has been requested for the proc (e.g. a
    /// graceful shutdown via SIGTERM), but the underlying process has
    /// not yet fully exited.
    pub(crate) fn mark_stopping(&self) -> bool {
        let mut status = self.status.lock().expect("status mutex poisoned");
        match *status {
            ProcStatus::Running { .. } | ProcStatus::Starting => {
                *status = ProcStatus::Stopping;
                true
            }
            _ => {
                tracing::debug!(
                    "illegal transition: {:?} -> Stopping; leaving status unchanged",
                    *status
                );
                false
            }
        }
    }

    /// Record that the process has exited normally with the given
    /// exit code.
    pub(crate) fn mark_stopped(&self, exit_code: i32) -> bool {
        let mut status = self.status.lock().expect("status mutex poisoned");
        match *status {
            ProcStatus::Stopping | ProcStatus::Running { .. } | ProcStatus::Starting => {
                *status = ProcStatus::Stopped { exit_code };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Stopped; leaving status unchanged",
                    *status
                );
                false
            }
        }
    }

    /// Record that the process was killed by the given signal (e.g.
    /// SIGKILL, SIGTERM).
    pub(crate) fn mark_killed(&self, signal: i32, core_dumped: bool) -> bool {
        let mut status = self.status.lock().expect("status mutex poisoned");
        match *status {
            ProcStatus::Running { .. } | ProcStatus::Stopping | ProcStatus::Starting => {
                *status = ProcStatus::Killed {
                    signal,
                    core_dumped,
                };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Killed; leaving status unchanged",
                    *status
                );
                false
            }
        }
    }

    /// Record that the proc or its process failed for an unexpected
    /// reason (bootstrap error, spawn failure, etc.).
    pub(crate) fn mark_failed<S: Into<String>>(&self, reason: S) -> bool {
        let mut status = self.status.lock().expect("status mutex poisoned");
        match *status {
            ProcStatus::Starting | ProcStatus::Running { .. } | ProcStatus::Stopping => {
                *status = ProcStatus::Failed {
                    reason: reason.into(),
                };
                true
            }
            _ => {
                tracing::warn!(
                    "illegal transition: {:?} -> Failed; leaving status unchanged",
                    *status
                );
                false
            }
        }
    }
}

/// A proc manager that launches procs using the [`bootstrap`]
/// function as an entry point.
#[derive(Debug)]
pub struct BootstrapProcManager {
    program: std::path::PathBuf,
    children: Arc<tokio::sync::Mutex<HashMap<ProcId, Child>>>,
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
            children: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Convenience constructor that resolves the current executable
    /// (`std::env::current_exe`) and uses that as the bootstrap
    /// binary.
    ///
    /// Useful when the proc manager should re-exec itself as the
    /// child program. Returns an `io::Result` since querying the
    /// current executable path can fail.
    pub(crate) fn new_current_exe() -> io::Result<Self> {
        Ok(Self::new(std::env::current_exe()?))
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
}

#[async_trait]
impl ProcManager for BootstrapProcManager {
    type Agent = ProcMeshAgent;

    fn transport(&self) -> ChannelTransport {
        ChannelTransport::Unix
    }

    async fn spawn(
        &self,
        proc_id: ProcId,
        backend_addr: ChannelAddr,
    ) -> Result<(ChannelAddr, ActorRef<Self::Agent>), HostError> {
        let (callback_addr, mut callback_rx) =
            channel::serve(ChannelAddr::any(ChannelTransport::Unix)).await?;

        let mode = BootstrapMode::Proc {
            proc_id: proc_id.clone(),
            backend_addr,
            callback_addr,
        };
        let mut cmd = Command::new(&self.program);
        cmd.env(
            "HYPERACTOR_MESH_BOOTSTRAP_MODE",
            mode.to_env_safe_string()
                .map_err(|e| HostError::ProcessConfigurationFailure(proc_id.clone(), e.into()))?,
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
        // TODO: add graceful shutdown (SIGTERM -> wait -> SIGKILL) via
        // terminate_all().

        let log_channel = ChannelAddr::any(ChannelTransport::Unix);
        cmd.env(BOOTSTRAP_LOG_CHANNEL, log_channel.to_string());

        let mut child = cmd
            .spawn()
            .map_err(|e| HostError::ProcessSpawnFailure(proc_id.clone(), e))?;
        let pid = child.id().unwrap_or_default();

        // Writers: tee to local (stdout/stderr or file) + send over
        // channel
        let (mut out_writer, mut err_writer) = create_log_writers(0, log_channel.clone(), pid)
            .unwrap_or_else(|_| (Box::new(tokio::io::stdout()), Box::new(tokio::io::stderr())));

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

        // Retain handle for lifecycle mgt.
        {
            let mut children = self.children.lock().await;
            children.insert(proc_id.clone(), child);
        }

        // Now wait for the callback, providing the address (proc
        // listen addr + agent).
        Ok(callback_rx.recv().await?)
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
    let mode = match std::env::var("HYPERACTOR_MESH_BOOTSTRAP_MODE") {
        Ok(mode) => match BootstrapMode::from_env_safe_string(&mode) {
            Ok(mode) => mode,
            Err(e) => {
                return anyhow::Error::from(e).context("parsing HYPERACTOR_MESH_BOOTSTRAP_MODE");
            }
        },
        Err(VarError::NotPresent) => BootstrapMode::default(),
        Err(e) => return anyhow::Error::from(e).context("reading HYPERACTOR_MESH_BOOTSTRAP_MODE"),
    };

    match mode {
        BootstrapMode::Proc {
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
                Ok(_proc) => {
                    future::pending::<()>().await;
                    unreachable!()
                }
                Err(e) => e.into(),
            }
        }
        BootstrapMode::V0ProcMesh => bootstrap_v0_proc_mesh().await,
    }
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
        let (serve_addr, mut rx) = channel::serve(listen_addr).await?;
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
                    let (proc_addr, proc_rx) =
                        channel::serve(ChannelAddr::any(listen_transport)).await?;
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
    tracing::error!("failed to bootstrap mesh process: {}", err);
    std::process::exit(1)
}

#[cfg(test)]
mod tests {
    use hyperactor::ProcId;
    use hyperactor::channel::ChannelAddr;
    use hyperactor::channel::ChannelTransport;
    use hyperactor::clock::RealClock;
    use hyperactor::id;

    use super::*;

    #[test]
    fn test_bootstrap_mode_env_string() {
        let values = [
            BootstrapMode::default(),
            BootstrapMode::Proc {
                proc_id: id!(foo[0]),
                backend_addr: ChannelAddr::any(ChannelTransport::Tcp),
                callback_addr: ChannelAddr::any(ChannelTransport::Unix),
            },
        ];

        for value in values {
            let safe = value.to_env_safe_string().unwrap();
            assert_eq!(value, BootstrapMode::from_env_safe_string(&safe).unwrap());
        }
    }

    #[tokio::test]
    async fn test_children_killed_on_manager_drop() {
        use std::path::PathBuf;
        use std::process::Stdio;

        use tokio::process::Command;
        use tokio::time::Duration;

        // Manager; program path is irrelevant for this test.
        let manager = BootstrapProcManager::new(PathBuf::from("/bin/true"));

        // Spawn a long-running child process (sleep 30) with
        // kill_on_drop(true).
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg("sleep 30")
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
            children.insert(proc_id, child);
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

        // Drop the manager - this drops the Child handles; with
        // kill_on_drop(true) the OS should send SIGKILL to the child
        // process.
        drop(manager);

        // Allow a moment for the signal to be delivered and the
        // process to exit.
        RealClock.sleep(Duration::from_millis(400)).await;

        // (Linux-only) Assert the process is gone.
        #[cfg(target_os = "linux")]
        {
            let path = format!("/proc/{}", pid);
            assert!(
                std::fs::metadata(&path).is_err(),
                "expected /proc/{pid} to be gone after drop"
            );
        }

        // On non-Linux, absence of panics/hangs is the signal; PID
        // probing is platform-specific.
    }

    #[tokio::test]
    async fn test_v1_child_logging() {
        use std::time::Duration;

        use hyperactor::ActorRef;
        use hyperactor::channel::ChannelAddr;
        use hyperactor::channel::ChannelTransport;
        use hyperactor::channel::{self};
        use hyperactor::data::Serialized;
        use hyperactor::id;
        use hyperactor::mailbox::BoxedMailboxSender;
        use hyperactor::mailbox::DialMailboxRouter;
        use hyperactor::mailbox::MailboxServer;
        use hyperactor::proc::Proc;
        use tokio::time::timeout;

        use crate::bootstrap::BOOTSTRAP_LOG_CHANNEL;
        use crate::logging::LogClientActor;
        use crate::logging::LogClientMessageClient;
        use crate::logging::LogForwardActor;
        use crate::logging::LogMessage;
        use crate::logging::OutputTarget;
        use crate::logging::test_tap;

        let router = DialMailboxRouter::new();
        let (proc_addr, proc_rx) = channel::serve(ChannelAddr::any(ChannelTransport::Unix))
            .await
            .unwrap();
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
        let line = timeout(Duration::from_secs(2), tap_rx.recv())
            .await
            .expect("timed out waiting for log line")
            .expect("tap channel closed unexpectedly");
        assert!(
            line.contains("hello from child"),
            "log line did not appear via LogClientActor; got: {line}"
        );
    }

    mod proc_handle {

        use tokio::process::Command;

        use super::super::*;

        // Helper: build a ProcHandle with a short-lived child
        // process. We don't rely on the actual process; we only
        // exercise the status transitions.
        fn handle_for_test() -> ProcHandle {
            // Spawn a trivial child that exits immediately.
            let child = Command::new("sh")
                .arg("-c")
                .arg("exit 0")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("failed to spawn test child process");

            let proc_id = hyperactor::ProcId::Ranked(hyperactor::WorldId("test".into()), 0);
            ProcHandle::new(proc_id, child)
        }

        #[tokio::test]
        async fn starting_to_running_ok() {
            let h = handle_for_test();
            assert_eq!(h.status(), ProcStatus::Starting);
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
            assert_eq!(h.status(), ProcStatus::Stopping);
            assert!(h.mark_stopped(0));
            assert_eq!(h.status(), ProcStatus::Stopped { exit_code: 0 });
        }

        #[tokio::test]
        async fn running_to_killed_ok() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(h.mark_killed(9, true));
            assert_eq!(
                h.status(),
                ProcStatus::Killed {
                    signal: 9,
                    core_dumped: true
                }
            );
        }

        #[tokio::test]
        async fn running_to_failed_ok() {
            let h = handle_for_test();
            let child_pid = h.pid().expect("child should have a pid");
            let child_started_at = RealClock.system_time_now();
            assert!(h.mark_running(child_pid, child_started_at));
            assert!(h.mark_failed("bootstrap error"));
            assert_eq!(
                h.status(),
                ProcStatus::Failed {
                    reason: "bootstrap error".into()
                }
            );
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

            assert_eq!(h.status(), ProcStatus::Stopped { exit_code: 0 });
        }
    }
}
