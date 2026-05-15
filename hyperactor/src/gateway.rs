/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Connectivity layer for Hyperactor procs.
//!
//! A proc by itself is an isolated actor runtime. It owns local actor
//! lifecycle and mailboxes, but it communicates with other procs by
//! attaching to a gateway. The gateway encapsulates the proc's connectivity
//! layer: it gives attached procs an advertised location, accepts inbound
//! traffic for that location, and forwards outbound traffic to destinations
//! outside the proc.
//!
//! This separation lets us compose different topologies without changing
//! proc identity. A host can attach all of its procs to one gateway, so the
//! gateway multiplexes ingress to those procs and routes egress on their
//! behalf. A proc from a foreign host can also attach through another host's
//! gateway, inheriting that host's advertised location while still retaining
//! its own proc id. Gateways can also act as pure proxies when they do not
//! own any local procs.
//!
//! From the channel/connectivity perspective, each location has one gateway.
//! Operationally, a gateway is both a proc multiplexer for ingress and a
//! router for egress.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;

use crate::Location;
use crate::ProcAddr;
use crate::ProcId;
use crate::channel;
use crate::channel::ChannelAddr;
use crate::channel::ChannelError;
use crate::channel::ChannelTransport;
use crate::mailbox::BoxedMailboxSender;
use crate::mailbox::DeliveryError;
use crate::mailbox::DialMailboxRouter;
use crate::mailbox::IntoBoxedMailboxSender as _;
use crate::mailbox::MailboxSender as _;
use crate::mailbox::MailboxServer as _;
use crate::mailbox::MailboxServerHandle;
use crate::mailbox::MessageEnvelope;
use crate::mailbox::PortHandle;
use crate::mailbox::Undeliverable;
use crate::mailbox::UnroutableMailboxSender;
use crate::proc::Proc;
use crate::proc::WeakProc;

/// Connectivity boundary for one or more procs.
#[derive(Clone)]
pub struct Gateway {
    inner: Arc<GatewayState>,
}

struct GatewayState {
    /// The location to use when no server is active.
    fallback_location: Location,

    /// The location used when constructing routeable addresses for
    /// newly bound refs.
    default_location: RwLock<Location>,

    /// Sender used to forward messages outside of the proc.
    forwarder: BoxedMailboxSender,

    /// Procs attached to this gateway, keyed by runtime identity.
    procs: RwLock<HashMap<ProcId, WeakProc>>,

    /// Locations currently served by this gateway. The last location
    /// is the default advertised location.
    active_servers: RwLock<Vec<Location>>,
}

impl Gateway {
    /// Create a fresh unserved gateway with dial-based forwarding.
    pub fn new() -> Self {
        Self::configured(
            channel::reserve_local_addr().into(),
            DialMailboxRouter::new().into_boxed(),
        )
    }

    /// Create a fresh unserved local-only gateway.
    pub fn isolated() -> Self {
        Self::configured(
            channel::reserve_local_addr().into(),
            BoxedMailboxSender::new(UnroutableMailboxSender),
        )
    }

    /// Return the process-wide global gateway.
    pub fn global() -> &'static Self {
        static GLOBAL_GATEWAY: OnceLock<Gateway> = OnceLock::new();
        GLOBAL_GATEWAY.get_or_init(Self::new)
    }

    pub(crate) fn configured(default_location: Location, forwarder: BoxedMailboxSender) -> Self {
        Self {
            inner: Arc::new(GatewayState {
                fallback_location: default_location.clone(),
                default_location: RwLock::new(default_location),
                forwarder,
                procs: RwLock::new(HashMap::new()),
                active_servers: RwLock::new(Vec::new()),
            }),
        }
    }

    /// The gateway's default advertised location.
    pub fn default_location(&self) -> Location {
        self.inner.default_location.read().unwrap().clone()
    }

    /// Set the gateway's default advertised location.
    pub fn set_default_location(&self, location: Location) {
        *self.inner.default_location.write().unwrap() = location;
    }

    /// Construct a routeable proc address using this gateway's default location.
    pub fn proc_addr(&self, proc_id: &ProcId) -> ProcAddr {
        ProcAddr::new(proc_id.clone(), self.default_location())
    }

    pub(crate) fn forwarder(&self) -> &BoxedMailboxSender {
        &self.inner.forwarder
    }

    pub(crate) fn attach(&self, proc: &Proc) {
        let proc_id = proc.proc_id().clone();
        if let Some(existing) = self
            .inner
            .procs
            .write()
            .unwrap()
            .insert(proc_id.clone(), proc.downgrade())
            && existing.upgrade().is_some()
        {
            panic!("gateway already has a live proc with id {}", proc_id)
        }
    }

    pub(crate) fn serve_rx(
        &self,
        rx: impl channel::Rx<MessageEnvelope> + Send + 'static,
    ) -> MailboxServerHandle {
        WeakGateway::new(self).serve(rx)
    }

    /// Serve this gateway on the provided channel address.
    ///
    /// When serving the first local [`ChannelAddr::any`] address, the gateway
    /// binds the local address that was reserved when the gateway was created.
    /// Local reservation is separate from local binding so a gateway can have a
    /// stable location before it has a runtime available to run a server.
    /// Later local `any` serves allocate fresh local ports, so the gateway can
    /// have multiple active local servers.
    ///
    /// Serving updates the gateway's default location to the newly served
    /// address. When that server stops, the default location falls back to the
    /// previous active server, or to the reserved fallback location when no
    /// server remains.
    pub fn serve(&self, addr: ChannelAddr) -> Result<GatewayServeHandle, ChannelError> {
        let (location, handle) = self.serve_inner(addr)?;
        Ok(GatewayServeHandle {
            gateway: self.clone(),
            location,
            handle,
            stopped: Arc::new(AtomicBool::new(false)),
        })
    }

    fn serve_inner(
        &self,
        addr: ChannelAddr,
    ) -> Result<(Location, MailboxServerHandle), ChannelError> {
        let addr = self.resolve_serve_addr(addr);
        let (addr, rx) = channel::serve(addr)?;
        let location = Location::from(addr);
        self.add_server(location.clone());
        Ok((location, self.serve_rx(rx)))
    }

    fn resolve_serve_addr(&self, addr: ChannelAddr) -> ChannelAddr {
        // The first local-any serve activates the address that was reserved at
        // construction time. Subsequent local-any serves should allocate new
        // ports, so multiple local servers can coexist for the same gateway.
        if addr == ChannelAddr::any(ChannelTransport::Local)
            && self.inner.active_servers.read().unwrap().is_empty()
            && matches!(self.inner.fallback_location.addr(), ChannelAddr::Local(_))
        {
            return self.inner.fallback_location.addr().clone();
        }
        addr
    }

    fn add_server(&self, location: Location) {
        let mut active_servers = self.inner.active_servers.write().unwrap();
        active_servers.push(location.clone());
        *self.inner.default_location.write().unwrap() = location;
    }

    fn remove_server(&self, location: &Location) {
        let mut active_servers = self.inner.active_servers.write().unwrap();
        if let Some(index) = active_servers.iter().rposition(|active| active == location) {
            active_servers.remove(index);
        }
        let default_location = active_servers
            .last()
            .cloned()
            .unwrap_or_else(|| self.inner.fallback_location.clone());
        *self.inner.default_location.write().unwrap() = default_location;
    }

    /// Flush pending gateway traffic.
    ///
    /// This first flushes the muxers for all live procs attached to the
    /// gateway, then flushes the gateway's forwarder. Flushing the proc muxers
    /// drains local delivery and any return paths rooted in attached procs;
    /// flushing the forwarder drains outbound traffic that the gateway has
    /// routed away from those procs.
    ///
    /// The live proc set is snapshotted before awaiting, so we do not hold the
    /// proc map while flushing. Procs that have already been dropped are
    /// ignored. Concurrent posts may race with this operation; `flush` only
    /// guarantees that each flushed sender observes its usual sender-level
    /// flush semantics at the time it is flushed.
    pub(crate) async fn flush(&self) -> Result<(), anyhow::Error> {
        let procs = self
            .inner
            .procs
            .read()
            .unwrap()
            .values()
            .filter_map(WeakProc::upgrade)
            .collect::<Vec<_>>();
        for proc in procs {
            proc.muxer().flush().await?;
        }
        self.inner.forwarder.flush().await
    }
}

impl fmt::Debug for Gateway {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gateway")
            .field("default_location", &self.default_location())
            .finish()
    }
}

/// A running gateway server.
#[derive(Debug)]
pub struct GatewayServeHandle {
    gateway: Gateway,
    location: Location,
    handle: MailboxServerHandle,
    stopped: Arc<AtomicBool>,
}

impl GatewayServeHandle {
    /// Signal the gateway server to stop.
    pub fn stop(&self, reason: &str) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            self.handle.stop(reason);
            self.gateway.remove_server(&self.location);
        }
    }
}

impl Future for GatewayServeHandle {
    type Output = <MailboxServerHandle as Future>::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `handle` is structurally pinned with `GatewayServeHandle`:
        // this type is `!Unpin` because `MailboxServerHandle` is `!Unpin`, it
        // has no `Drop` impl that moves `handle`, and no method moves `handle`
        // out of a pinned `GatewayServeHandle`.
        let handle = unsafe {
            self.as_mut()
                .map_unchecked_mut(|container| &mut container.handle)
        };
        let result = handle.poll(cx);
        if result.is_ready() {
            // SAFETY: We only mutate unpinned bookkeeping fields after polling
            // the pinned `handle`; this does not move `handle` or any other
            // pinned field out of `self`.
            let this = unsafe { self.get_unchecked_mut() };
            if !this.stopped.swap(true, Ordering::AcqRel) {
                this.gateway.remove_server(&this.location);
            }
        }
        result
    }
}

#[derive(Clone, Debug)]
struct WeakGateway(Weak<GatewayState>);

impl WeakGateway {
    fn new(gateway: &Gateway) -> Self {
        Self(Arc::downgrade(&gateway.inner))
    }

    fn upgrade(&self) -> Option<Gateway> {
        self.0.upgrade().map(|inner| Gateway { inner })
    }
}

#[async_trait]
impl crate::mailbox::MailboxSender for WeakGateway {
    fn post_unchecked(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        match self.upgrade() {
            Some(gateway) => gateway.post(envelope, return_handle),
            None => envelope.undeliverable(
                DeliveryError::BrokenLink("failed to upgrade WeakGateway".to_string()),
                return_handle,
            ),
        }
    }

    async fn flush(&self) -> Result<(), anyhow::Error> {
        match self.upgrade() {
            Some(gateway) => Gateway::flush(&gateway).await,
            None => Ok(()),
        }
    }
}

#[async_trait]
impl crate::mailbox::MailboxSender for Gateway {
    fn post_unchecked(
        &self,
        envelope: MessageEnvelope,
        return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
    ) {
        let dest_proc = envelope.dest().actor_addr().proc_addr();
        let weak_proc = self
            .inner
            .procs
            .read()
            .unwrap()
            .get(dest_proc.id())
            .cloned();
        let proc = weak_proc.as_ref().and_then(WeakProc::upgrade);

        if weak_proc.is_some() && proc.is_none() {
            let mut procs = self.inner.procs.write().unwrap();
            if procs
                .get(dest_proc.id())
                .and_then(WeakProc::upgrade)
                .is_none()
            {
                procs.remove(dest_proc.id());
            }
        }

        if let Some(proc) = proc {
            if proc.is_local_delivery_target(&dest_proc) {
                proc.muxer().post(envelope, return_handle);
                return;
            }
        }

        self.inner.forwarder.post(envelope, return_handle)
    }

    async fn flush(&self) -> Result<(), anyhow::Error> {
        Gateway::flush(self).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use async_trait::async_trait;
    use hyperactor_config::Flattrs;
    use tokio::time;

    use super::*;
    use crate::Label;
    use crate::mailbox::MailboxSender;
    use crate::mailbox::PortLocation;
    use crate::mailbox::monitored_return_handle;
    use crate::port::Port;
    use crate::proc::Proc;
    use crate::testing::ids::test_actor_id;
    use crate::testing::pingpong::PingPongActor;
    use crate::testing::pingpong::PingPongMessage;

    /// `Gateway::post_unchecked` demuxes inbound envelopes by
    /// destination `ProcId` to the matching attached proc's muxer,
    /// and falls through to the configured forwarder for unknown
    /// destinations. Attached procs only receive envelopes addressed
    /// to them — a stranger-addressed envelope does not leak to local
    /// receivers.
    #[tokio::test]
    async fn test_gateway_post_demuxes_by_proc_id() {
        #[derive(Clone)]
        struct CountingSender(Arc<AtomicUsize>);

        #[async_trait]
        impl MailboxSender for CountingSender {
            fn post_unchecked(
                &self,
                _envelope: MessageEnvelope,
                _return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
            ) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let forwarded = Arc::new(AtomicUsize::new(0));
        let gateway = Gateway::configured(
            channel::reserve_local_addr().into(),
            BoxedMailboxSender::new(CountingSender(forwarded.clone())),
        );

        let alpha = Proc::builder()
            .proc_id(ProcId::instance(Label::strip("alpha")))
            .shared_gateway(gateway.clone())
            .build()
            .unwrap();
        let beta = Proc::builder()
            .proc_id(ProcId::instance(Label::strip("beta")))
            .shared_gateway(gateway.clone())
            .build()
            .unwrap();

        let (alpha_client, _) = alpha.instance("client").unwrap();
        let (alpha_port, mut alpha_rx) = alpha_client.bind_handler_port::<u64>();
        let PortLocation::Bound(alpha_dest) = alpha_port.location() else {
            panic!("alpha handler port must be bound");
        };

        let (beta_client, _) = beta.instance("client").unwrap();
        let (beta_port, mut beta_rx) = beta_client.bind_handler_port::<u64>();
        let PortLocation::Bound(beta_dest) = beta_port.location() else {
            panic!("beta handler port must be bound");
        };

        let sender = test_actor_id("sender", "client");

        gateway.post(
            MessageEnvelope::serialize(sender.clone(), alpha_dest.clone(), &111u64, Flattrs::new())
                .unwrap(),
            monitored_return_handle(),
        );
        let received = time::timeout(Duration::from_secs(5), alpha_rx.recv())
            .await
            .expect("alpha_rx timed out")
            .expect("alpha_rx closed");
        assert_eq!(received, 111);
        assert_eq!(forwarded.load(Ordering::SeqCst), 0);

        gateway.post(
            MessageEnvelope::serialize(sender.clone(), beta_dest.clone(), &222u64, Flattrs::new())
                .unwrap(),
            monitored_return_handle(),
        );
        let received = time::timeout(Duration::from_secs(5), beta_rx.recv())
            .await
            .expect("beta_rx timed out")
            .expect("beta_rx closed");
        assert_eq!(received, 222);
        assert_eq!(forwarded.load(Ordering::SeqCst), 0);

        let stranger_proc = ProcAddr::unique(ChannelAddr::Local(9999), "stranger");
        let stranger_dest = stranger_proc
            .actor_addr("ghost")
            .port_addr(Port::from(0u64));
        gateway.post(
            MessageEnvelope::serialize(sender, stranger_dest, &333u64, Flattrs::new()).unwrap(),
            monitored_return_handle(),
        );
        assert_eq!(forwarded.load(Ordering::SeqCst), 1);
        assert!(
            time::timeout(Duration::from_millis(50), alpha_rx.recv())
                .await
                .is_err(),
            "alpha_rx received a message after stranger post",
        );
        assert!(
            time::timeout(Duration::from_millis(50), beta_rx.recv())
                .await
                .is_err(),
            "beta_rx received a message after stranger post",
        );
    }

    /// Ping-pong between two `PingPongActor`s on two procs that share
    /// one served gateway. Each cross-proc hop goes out via the
    /// gateway's forwarder (`DialMailboxRouter`) and comes back in
    /// via the gateway's demux.
    #[tokio::test]
    async fn test_ping_pong_across_shared_gateway() {
        let gateway = Gateway::new();

        let alpha = Proc::builder()
            .proc_id(ProcId::instance(Label::strip("alpha")))
            .shared_gateway(gateway.clone())
            .build()
            .unwrap();
        let beta = Proc::builder()
            .proc_id(ProcId::instance(Label::strip("beta")))
            .shared_gateway(gateway.clone())
            .build()
            .unwrap();

        // Serve the gateway on its reserved local address
        // (first-serve rule). Outbound from one proc to the other
        // goes Proc::post_unchecked → forwarder (DialMailboxRouter) →
        // dial the served address → gateway's channel server →
        // Gateway::post_unchecked demux → destination proc's muxer.
        let _serve = Gateway::serve(&gateway, ChannelAddr::any(ChannelTransport::Local)).unwrap();

        let (client, _) = alpha.instance("client").unwrap();
        let (undeliverable_msg_tx, mut undeliverable_rx) =
            client.open_port::<Undeliverable<MessageEnvelope>>();

        let ping_actor = PingPongActor::new(Some(undeliverable_msg_tx.bind()), None, None);
        let pong_actor = PingPongActor::new(Some(undeliverable_msg_tx.bind()), None, None);
        let ping_handle = alpha.spawn::<PingPongActor>("ping", ping_actor).unwrap();
        let pong_handle = beta.spawn::<PingPongActor>("pong", pong_actor).unwrap();

        let (local_port, local_receiver) = client.open_once_port();

        ping_handle
            .send(
                &client,
                PingPongMessage(10, pong_handle.bind(), local_port.bind()),
            )
            .unwrap();

        let received = time::timeout(Duration::from_secs(5), local_receiver.recv())
            .await
            .expect("local_receiver timed out")
            .expect("ping pong did not complete");
        assert!(received);

        // Every hop should have made it through dial+demux+delivery.
        // An undeliverable here would indicate a dial failure, a
        // misroute by the gateway's demux, or a missing attached
        // proc.
        assert!(
            time::timeout(Duration::from_millis(50), undeliverable_rx.recv())
                .await
                .is_err(),
            "unexpected undeliverable during cross-proc ping-pong",
        );
    }

    /// `Gateway::post_unchecked` removes stale `WeakProc` entries
    /// lazily on the post path. Dropping a proc leaves a dead
    /// `WeakProc` in the gateway's `procs` map; the next post to that
    /// proc's id falls through to the forwarder and removes the stale
    /// entry.
    #[tokio::test]
    async fn test_gateway_post_removes_stale_weak_procs() {
        #[derive(Clone)]
        struct CountingSender(Arc<AtomicUsize>);

        #[async_trait]
        impl MailboxSender for CountingSender {
            fn post_unchecked(
                &self,
                _envelope: MessageEnvelope,
                _return_handle: PortHandle<Undeliverable<MessageEnvelope>>,
            ) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let forwarded = Arc::new(AtomicUsize::new(0));
        let gateway = Gateway::configured(
            channel::reserve_local_addr().into(),
            BoxedMailboxSender::new(CountingSender(forwarded.clone())),
        );

        let proc = Proc::builder()
            .proc_id(ProcId::instance(Label::strip("alpha")))
            .shared_gateway(gateway.clone())
            .build()
            .unwrap();
        let proc_id = proc.proc_id().clone();
        let dest = proc
            .proc_addr()
            .actor_addr("worker")
            .port_addr(Port::from(0u64));

        // Sanity: proc is attached.
        assert_eq!(gateway.inner.procs.read().unwrap().len(), 1);

        drop(proc);

        // The entry is still present, but it is now a *stale* WeakProc:
        // upgrade must fail.
        {
            let procs = gateway.inner.procs.read().unwrap();
            assert_eq!(procs.len(), 1);
            assert!(
                procs.get(&proc_id).and_then(WeakProc::upgrade).is_none(),
                "expected dropped proc to remain only as a stale WeakProc entry",
            );
        }

        gateway.post(
            MessageEnvelope::serialize(
                test_actor_id("sender", "client"),
                dest,
                &42u64,
                Flattrs::new(),
            )
            .unwrap(),
            monitored_return_handle(),
        );

        // Envelope fell through to the forwarder; stale entry
        // removed.
        assert_eq!(forwarded.load(Ordering::SeqCst), 1);
        {
            let procs = gateway.inner.procs.read().unwrap();
            assert_eq!(procs.len(), 0);
            assert!(procs.get(&proc_id).is_none());
        }
    }
}
