/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use chrysalis::Locator;
use chrysalis::NamespaceConfig;
use chrysalis::Node;
use chrysalis::NodeConfig;
use chrysalis::ParentEndpoint;
use chrysalis::ParentManagerStatus;
use chrysalis::Pid;
use chrysalis::PidPrefix;
use chrysalis::TransportConfig;
use chrysalis::UdpSocket;
use chrysalis_sqlite::Replica;
use chrysalis_sqlite::ReplicationTopology;

use crate::development_identity;

#[derive(Clone, Debug)]
pub(crate) struct JoinToken {
    pid: Pid,
    address: SocketAddr,
}

impl fmt::Display for JoinToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "udp://{}?authority={}",
            self.address,
            format_pid(self.pid)
        )
    }
}

impl FromStr for JoinToken {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = value
            .strip_prefix("udp://")
            .context("join token must start with udp://")?;
        let (address, pid) = value
            .split_once("?authority=")
            .context("join token must contain ?authority=PID")?;
        if pid.contains('&') {
            bail!("join token only supports the authority query");
        }
        let pid = pid
            .parse::<PidPrefix>()?
            .as_pid()
            .context("join token requires a complete PID")?;
        if pid.is_link_local() {
            bail!("link-local PID is reserved");
        }
        Ok(Self {
            pid,
            address: address.parse()?,
        })
    }
}

pub(crate) async fn run(replica: Replica, parent: Option<JoinToken>) -> Result<()> {
    let identity = development_identity::generate().await?;
    let pid = identity.pid();
    let socket = Arc::new(
        UdpSocket::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .context("bind loopback UDP carrier")?,
    );
    let address = socket.address();
    let mut config =
        NodeConfig::new(TransportConfig::new(socket, identity)).with_locators(vec![Locator {
            address: UdpSocket::datagram_addr(address),
            priority: 0,
        }]);
    if let Some(parent) = parent {
        config = config.with_parent(NamespaceConfig::try_new(
            parent.pid,
            vec![ParentEndpoint::new(UdpSocket::datagram_addr(
                parent.address,
            ))],
        )?);
    }
    let config = ReplicationTopology::new(replica).configure(config);
    let node = Node::create(config).context("create Chrysalis node")?;
    let result = async {
        wait_until_ready(&node).await?;
        println!("{}", JoinToken { pid, address });
        std::io::stdout().flush().context("flush join token")?;
        tokio::signal::ctrl_c().await.context("wait for interrupt")
    }
    .await;
    node.shutdown();
    node.join().await;
    result
}

async fn wait_until_ready(node: &Node) -> Result<()> {
    let Some(mut parent) = node.subscribe_parent() else {
        return Ok(());
    };
    loop {
        match parent.borrow().clone() {
            ParentManagerStatus::Connected { .. } => return Ok(()),
            ParentManagerStatus::Stopped => bail!("parent manager stopped while joining"),
            ParentManagerStatus::Failed { error } => bail!(error),
            ParentManagerStatus::Connecting => {}
        }
        parent
            .changed()
            .await
            .context("parent manager stopped while joining")?;
    }
}

fn format_pid(pid: Pid) -> String {
    pid.as_bytes()
        .iter()
        .fold(String::with_capacity(32), |mut output, byte| {
            write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
            output
        })
}
