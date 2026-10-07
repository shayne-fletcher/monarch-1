/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Message and control channels for an async Python actor.
//!
//! Each plane is an independent `pympsc` channel with its own asyncio wake
//! event. `_dispatch_loop` consumes messages and `_callback_loop` consumes
//! control work as separate tasks on the actor's event loop.

use crate::pympsc::PyReceiver;
use crate::pympsc::Sender;
use crate::pympsc::channel;

/// The complete inbox created for one async actor.
pub(in super::super) struct Inbox {
    /// Enqueues messages for `_dispatch_loop`.
    pub(in super::super) message_sender: Sender,
    /// Enqueues callback work for `_callback_loop`.
    pub(in super::super) control_sender: Sender,
    /// The two receiving halves retained until `Actor::init` starts the loops.
    pub(in super::super) receiver: Receiver,
}

/// Receiving halves of an async actor's message and control planes.
#[derive(Debug)]
pub(in super::super) struct Receiver {
    /// Passed to `_dispatch_loop`.
    pub(in super::super) messages: PyReceiver,
    /// Passed to `_callback_loop`.
    pub(in super::super) control: PyReceiver,
}

/// Create the two independent channels consumed on an async actor's event loop.
pub(in super::super) fn new() -> Result<Inbox, nix::Error> {
    let (message_sender, messages) = channel()?;
    let (control_sender, control) = channel()?;
    Ok(Inbox {
        message_sender,
        control_sender,
        receiver: Receiver { messages, control },
    })
}
