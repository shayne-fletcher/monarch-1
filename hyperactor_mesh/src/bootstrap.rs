/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env::VarError;
use std::future;
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

/// A proc manager that launches procs using the [`bootstrap`] function as an entry point.
pub struct BootstrapProcManager {
    program: std::path::PathBuf,
}

impl BootstrapProcManager {
    #[allow(dead_code)]
    pub(crate) fn new(program: std::path::PathBuf) -> Self {
        Self { program }
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
        );

        // TODO: retain, manage lifecycle
        let _process = cmd
            .spawn()
            .map_err(|e| HostError::ProcessSpawnFailure(proc_id, e))?;

        // Now wait for the callback, providing the address:
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
                    .destroy_and_wait(Duration::from_millis(10), None)
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
                                .destroy_and_wait(Duration::from_millis(10), None)
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
}
