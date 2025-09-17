/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The mesh agent actor that manages a host.

use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::Handler;
use hyperactor::Named;
use hyperactor::ProcId;
use hyperactor::host::Host;
use hyperactor::host::HostError;
use serde::Deserialize;
use serde::Serialize;

use crate::bootstrap::BootstrapProcManager;
use crate::proc_mesh::mesh_agent::ProcMeshAgent;
use crate::resource;
use crate::v1::Name;

/// A mesh agent is responsible for managing a host in a [`HostMesh`],
/// through the resource behaviors defined in [`crate::resource`].
#[hyperactor::export(handlers=[resource::CreateOrUpdate<()>, resource::GetState<ProcState>])]
pub struct HostMeshAgent {
    host: Host<BootstrapProcManager>,
    created: HashMap<Name, Result<(ProcId, ActorRef<ProcMeshAgent>), HostError>>,
}

impl fmt::Debug for HostMeshAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostMeshAgent")
            .field("host", &"..")
            .field("created", &self.created)
            .finish()
    }
}

#[async_trait]
impl Actor for HostMeshAgent {
    type Params = Host<BootstrapProcManager>;

    async fn new(params: Self::Params) -> anyhow::Result<Self> {
        Ok(Self {
            host: params,
            created: HashMap::new(),
        })
    }
}

#[async_trait]
impl Handler<resource::CreateOrUpdate<()>> for HostMeshAgent {
    async fn handle(
        &mut self,
        cx: &Context<Self>,
        create_or_update: resource::CreateOrUpdate<()>,
    ) -> anyhow::Result<()> {
        if self.created.contains_key(&create_or_update.name) {
            // There is no update.
            return Ok(());
        }

        let ok = self
            .created
            .insert(
                create_or_update.name.clone(),
                self.host
                    .spawn(create_or_update.name.clone().to_string())
                    .await,
            )
            .is_none();

        create_or_update.reply.send(cx, ok)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Named, Serialize, Deserialize)]
pub struct ProcState {
    pub proc_id: ProcId,
    pub mesh_agent: ActorRef<ProcMeshAgent>,
}

#[async_trait]
impl Handler<resource::GetState<ProcState>> for HostMeshAgent {
    async fn handle(
        &mut self,
        cx: &Context<Self>,
        get_state: resource::GetState<ProcState>,
    ) -> anyhow::Result<()> {
        let state = match self.created.get(&get_state.name) {
            Some(Ok((proc_id, mesh_agent))) => resource::State {
                name: get_state.name.clone(),
                status: resource::Status::Running,
                state: Some(ProcState {
                    proc_id: proc_id.clone(),
                    mesh_agent: mesh_agent.clone(),
                }),
            },
            Some(Err(e)) => resource::State {
                name: get_state.name.clone(),
                status: resource::Status::Failed(e.to_string()),
                state: None,
            },
            None => resource::State {
                name: get_state.name.clone(),
                status: resource::Status::NotExist,
                state: None,
            },
        };

        get_state.reply.send(cx, state)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use hyperactor::Proc;
    use hyperactor::channel::ChannelTransport;

    use super::*;
    use crate::resource::CreateOrUpdateClient;
    use crate::resource::GetStateClient;

    #[tokio::test]
    async fn test_basic() {
        let (host, _handle) = Host::serve(
            BootstrapProcManager::new_for_test(),
            ChannelTransport::Unix.any(),
        )
        .await
        .unwrap();

        let host_addr = host.addr().clone();
        let system_proc = host.system_proc().clone();
        let host_agent = system_proc
            .spawn::<HostMeshAgent>("agent", host)
            .await
            .unwrap();

        let client_proc = Proc::direct(ChannelTransport::Unix.any(), "client".to_string())
            .await
            .unwrap();
        let (client, _client_handle) = client_proc.instance("client").unwrap();

        let name = Name::new("proc1");

        // First, create the proc, then query its state:

        assert!(
            host_agent
                .create_or_update(&client, name.clone(), ())
                .await
                .unwrap()
        );
        assert_eq!(
            host_agent.get_state(&client, name.clone()).await.unwrap(),
            resource::State {
                name: name.clone(),
                status: resource::Status::Running,
                state: Some(ProcState {
                    // The proc itself should be direct addressed, with its name directly.
                    proc_id: ProcId::Direct(host_addr.clone(), name.to_string()),
                    // The mesh agent should run in the same proc, under the name
                    // "agent".
                    mesh_agent: ActorRef::attest(
                        ProcId::Direct(host_addr.clone(), name.to_string()).actor_id("agent", 0)
                    ),
                }),
            }
        );
    }
}
