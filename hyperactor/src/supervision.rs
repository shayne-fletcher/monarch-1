/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Messages used in supervision.
//!
//! ## Supervision invariants (SV-*)
//!
//! - **SV-1 (root-cause attribution):** For an
//!   `UnhandledSupervisionEvent` chain, `actually_failing_actor()`
//!   returns the event that should be treated as the root cause
//!   for structured failure attribution. In particular, if a
//!   failed parent wraps a stopped child event, the stopped child
//!   remains the root cause.

use std::fmt;
use std::fmt::Debug;
use std::fmt::Write;
use std::time::SystemTime;

use derivative::Derivative;
use hyperactor_config::Flattrs;
use indenter::indented;
use serde::Deserialize;
use serde::Serialize;

use crate::actor::ActorErrorKind;
use crate::actor::ActorStatus;
use crate::reference;

/// This is the local actor supervision event. Child actor will propagate this event to its parent.
#[derive(Clone, Debug, Derivative, Serialize, Deserialize, typeuri::Named)]
#[derivative(PartialEq, Eq)]
pub struct ActorSupervisionEvent {
    /// The actor id of the child actor where the event is triggered.
    pub actor_id: reference::ActorId,
    /// Friendly display name, if the actor class customized it.
    pub display_name: Option<String>,
    /// The time when the event is triggered.
    #[derivative(PartialEq = "ignore")]
    pub occurred_at: SystemTime,
    /// Status of the child actor.
    pub actor_status: ActorStatus,
    /// If this event is associated with a message, the message headers.
    #[derivative(PartialEq = "ignore")]
    pub message_headers: Option<Flattrs>,
}
wirevalue::register_type!(ActorSupervisionEvent);

impl ActorSupervisionEvent {
    /// Create a new supervision event. Timestamp is set to the current time.
    pub fn new(
        actor_id: reference::ActorId,
        display_name: Option<String>,
        actor_status: ActorStatus,
        message_headers: Option<Flattrs>,
    ) -> Self {
        Self {
            actor_id,
            display_name,
            occurred_at: std::time::SystemTime::now(),
            actor_status,
            message_headers,
        }
    }

    fn actor_name(&self) -> String {
        self.display_name
            .clone()
            .unwrap_or_else(|| self.actor_id.to_string())
    }

    /// Walk the `UnhandledSupervisionEvent` chain to find the root-cause
    /// actor that originally failed.
    pub fn actually_failing_actor(&self) -> &ActorSupervisionEvent {
        let mut event = self;
        while let ActorStatus::Failed(ActorErrorKind::UnhandledSupervisionEvent(e)) =
            &event.actor_status
        {
            event = e;
        }
        event
    }

    /// This event is for a a supervision error.
    pub fn is_error(&self) -> bool {
        self.actor_status.is_failed()
    }
}

impl std::error::Error for ActorSupervisionEvent {}

fn fmt_status<'a>(
    actor_id: &reference::ActorId,
    status: &'a ActorStatus,
    f: &mut fmt::Formatter<'_>,
) -> Result<Option<&'a ActorSupervisionEvent>, fmt::Error> {
    let mut f = indented(f).with_str(" ");

    match status {
        ActorStatus::Stopped(_)
            if actor_id.name() == "host_agent" || actor_id.name() == "proc_agent" =>
        {
            // Host agent stopped - use simplified message from D86984496
            let name = actor_id.proc_id().addr().to_string();
            write!(
                f,
                "The process {} owned by this actor became unresponsive and is assumed dead, check the log on the host for details",
                name
            )?;
            Ok(None)
        }
        ActorStatus::Failed(ActorErrorKind::ErrorDuringHandlingSupervision(
            msg,
            during_handling_of,
        )) => {
            write!(f, "{}", msg)?;
            Ok(Some(during_handling_of))
        }
        ActorStatus::Failed(ActorErrorKind::Generic(msg)) => {
            write!(f, "{}", msg)?;
            Ok(None)
        }
        status => {
            write!(f, "{}", status)?;
            Ok(None)
        }
    }
}

impl fmt::Display for ActorSupervisionEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let actor_name = self.actor_name();
        writeln!(
            f,
            "The actor {} and all its descendants have failed.",
            actor_name
        )?;
        let failing_event = self.actually_failing_actor();
        let failing_actor = failing_event.actor_name();
        let its_name = if failing_actor == actor_name {
            "itself"
        } else {
            &failing_actor
        };
        writeln!(f, "This occurred because the actor {} failed.", its_name)?;
        writeln!(f, "The error was:")?;
        let during_handling_of =
            fmt_status(&failing_event.actor_id, &failing_event.actor_status, f)?;
        if let Some(event) = during_handling_of {
            writeln!(
                f,
                "This error occurred during the handling of another failure:"
            )?;
            fmt::Display::fmt(event, f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelAddr;
    use crate::reference::ProcId;

    /// Exercises SV-1 (see module doc): for a parent wrapping a
    /// stopped child in `UnhandledSupervisionEvent`,
    /// `actually_failing_actor()` returns the stopped child as
    /// root cause for structured failure attribution.
    #[test]
    fn test_sv1_actually_failing_actor_returns_stopped_child() {
        let proc_id = ProcId::with_name(ChannelAddr::Local(0), "test_proc");
        let child_id = proc_id.actor_id("proc_agent", 0);
        let parent_id = proc_id.actor_id("controller", 0);

        let child_event = ActorSupervisionEvent::new(
            child_id.clone(),
            Some("proc_agent".into()),
            ActorStatus::Stopped("host died".into()),
            None,
        );
        let parent_event = ActorSupervisionEvent::new(
            parent_id,
            Some("controller".into()),
            ActorStatus::Failed(ActorErrorKind::UnhandledSupervisionEvent(Box::new(
                child_event,
            ))),
            None,
        );

        // SV-1: root cause is the stopped child, not the parent.
        let root = parent_event.actually_failing_actor();
        assert_eq!(root.actor_id, child_id);
        assert!(
            matches!(root.actor_status, ActorStatus::Stopped(_)),
            "root cause should be the stopped child, got: {:?}",
            root.actor_status,
        );
    }
}
