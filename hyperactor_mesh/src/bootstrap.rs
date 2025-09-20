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
use tokio::process::Command;
use tokio::sync::Mutex;
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

/// A proc manager that launches procs using the [`bootstrap`]
/// function as an entry point.
#[derive(Debug)]
pub struct BootstrapProcManager {
    program: std::path::PathBuf,
    children: Arc<tokio::sync::Mutex<HashMap<ProcId, tokio::process::Child>>>,
}

impl BootstrapProcManager {
    #[allow(dead_code)]
    pub(crate) fn new(program: std::path::PathBuf) -> Self {
        Self {
            program,
            children: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    pub(crate) fn new_current_exe() -> io::Result<Self> {
        Ok(Self::new(std::env::current_exe()?))
    }

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
        // TODO: add graceful shutdown (SIGTERM → wait → SIGKILL) via
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
        let procs = Arc::new(Mutex::new(Vec::<Proc>::new()));
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

        // Drop the manager — this drops the Child handles; with
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
}
