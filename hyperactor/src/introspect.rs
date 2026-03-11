/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Introspection protocol for hyperactor actors.
//!
//! Every actor has a dedicated introspect task that handles
//! [`IntrospectMessage`] by reading [`InstanceCell`] state directly,
//! without going through the actor's message loop. This means:
//!
//! - Stuck actors can be introspected (the task runs independently).
//! - Introspection does not perturb observed state (no Heisenberg).
//! - Live status is reported accurately.
//!
//! Infrastructure actors publish domain-specific metadata via
//! [`PublishedProperties`], which the introspect task reads for
//! Entity-view queries. Non-addressable children (e.g., system procs)
//! are resolved via a callback registered on [`InstanceCell`].
//!
//! Callers navigate topology by fetching a [`NodePayload`] and
//! following its `children` references.
//!
//! # Design Invariants
//!
//! The introspection subsystem maintains eleven invariants (S1--S11).
//! Each is documented at the code site that enforces it.
//!
//! - **S1.** Introspection must not depend on actor responsiveness --
//!   a wedged actor can still be introspected (runtime task, not
//!   actor loop).
//! - **S2.** Introspection must not perturb observed state -- reading
//!   `InstanceCell` never sets `last_message_handler` to
//!   `IntrospectMessage`.
//! - **S3.** Sender routing is unchanged -- senders target the same
//!   `PortId` (`IntrospectMessage::port()`) across processes.
//! - **S4.** `IntrospectMessage` never produces a `WorkCell` --
//!   pre-registration via `open_message_port` gives the introspect
//!   port its own channel, independent of the actor's work queue.
//! - **S5.** Replies never use `PanickingMailboxSender` -- the
//!   introspect task replies via `Mailbox::serialize_and_send_once`.
//! - **S6.** View semantics are stable -- Actor view uses live
//!   structural state + supervision children; Entity view uses
//!   published properties + domain children.
//! - **S7.** `QueryChild` must work without actor handlers -- system
//!   procs are resolved via a per-actor callback on `InstanceCell`.
//! - **S8.** Published properties are constrained -- actors cannot
//!   publish `Root` or `Error` payloads (only `Host` and `Proc`
//!   variants).
//! - **S9.** Port binding is single source of truth -- the introspect
//!   port is bound exactly once via `bind_actor_port()` in
//!   `Instance::new()`.
//! - **S10.** Introspect receiver lifecycle -- created in
//!   `Instance::new()`, spawned in `start()`, dropped in
//!   `child_instance()`.
//! - **S11.** Terminated snapshots do not keep actors resolvable --
//!   `store_terminated_snapshot` writes to the proc's snapshot map,
//!   not the instances map. `resolve_actor_ref` checks terminal
//!   status independently and is unaffected by snapshot storage.

use std::time::SystemTime;

use hyperactor_config::INTROSPECT;
use hyperactor_config::IntrospectAttr;
use hyperactor_config::declare_attrs;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::InstanceCell;
use crate::reference;

// Introspection attr keys — actor-runtime concepts.
//
// These keys are populated by the introspect handler from
// InstanceCell data. Mesh-topology keys (node_type, addr, num_procs,
// etc.) are declared in hyperactor_mesh::introspect_keys.
//
// Naming convention:
//
// - Attr names are node-type-agnostic. The `node_type` attr (from the
//   mesh layer) identifies what kind of node it is; individual attr
//   names don't repeat that. So `status`, not `actor_status`.
// - Related attrs share a prefix to form a group. The `failure_*`
//   keys decompose failure info into flat attrs — the `failure_`
//   prefix groups them semantically.
// - `actor_type` is an exception: the `actor_` prefix disambiguates
//   it from `node_type` (mesh-layer concept). `actor_type` is the
//   Rust actor type name; `node_type` is the topology role.
// - Use real types where possible (e.g. SystemTime for timestamps),
//   not String. Serialization format is a presentation concern.
// - Internal key names are fully-qualified by `declare_attrs!`
//   (module_path + attr constant), e.g.
//   `hyperactor::introspect::status`.
// - HTTP/schema public key names come from `@meta(INTROSPECT =
//   IntrospectAttr { name, desc })`. Keep `name` explicit so API
//   stability is decoupled from internal refactors.
//
// Invariants:
//
// - **IK-1 (metadata completeness):** Every actor-runtime
//   introspection key must carry `@meta(INTROSPECT = ...)` with
//   non-empty `name` and `desc`. Enforced by
//   `test_introspect_keys_are_tagged`.
// - **IK-2 (short-name uniqueness):** No two introspection keys
//   may share the same `IntrospectAttr.name`. Duplicates would break
//   the FQ→short HTTP remap and schema output. Enforced by
//   `test_introspect_short_names_are_globally_unique` within this
//   test binary; full cross-crate coverage requires an integration
//   test that links all introspection key crates.
declare_attrs! {
    /// Actor lifecycle status: "running", "stopped", "failed".
    ///
    /// Together with `STATUS_REASON`, these two attrs replace the
    /// former `actor_status` prefix protocol (`"stopped:reason"`,
    /// `"failed:reason"`) with structured fields, eliminating string
    /// prefix parsing in consumers.
    @meta(INTROSPECT = IntrospectAttr {
        name: "status".into(),
        desc: "Actor lifecycle status: running, stopped, failed".into(),
    })
    pub attr STATUS: String;

    /// Reason for stop/failure (absent when running).
    @meta(INTROSPECT = IntrospectAttr {
        name: "status_reason".into(),
        desc: "Reason for stop/failure (absent when running)".into(),
    })
    pub attr STATUS_REASON: String;

    /// Fully-qualified actor type name.
    @meta(INTROSPECT = IntrospectAttr {
        name: "actor_type".into(),
        desc: "Fully-qualified actor type name".into(),
    })
    pub attr ACTOR_TYPE: String;

    /// Number of messages processed by this actor.
    @meta(INTROSPECT = IntrospectAttr {
        name: "messages_processed".into(),
        desc: "Number of messages processed by this actor".into(),
    })
    pub attr MESSAGES_PROCESSED: u64 = 0;

    /// Timestamp when this actor was created.
    @meta(INTROSPECT = IntrospectAttr {
        name: "created_at".into(),
        desc: "Timestamp when this actor was created".into(),
    })
    pub attr CREATED_AT: SystemTime;

    /// Name of the last message handler invoked.
    @meta(INTROSPECT = IntrospectAttr {
        name: "last_handler".into(),
        desc: "Name of the last message handler invoked".into(),
    })
    pub attr LAST_HANDLER: String;

    /// Total CPU time in message handlers (microseconds).
    @meta(INTROSPECT = IntrospectAttr {
        name: "total_processing_time_us".into(),
        desc: "Total CPU time in message handlers (microseconds)".into(),
    })
    pub attr TOTAL_PROCESSING_TIME_US: u64 = 0;

    /// Flight recorder JSON (recent trace events).
    @meta(INTROSPECT = IntrospectAttr {
        name: "flight_recorder".into(),
        desc: "Flight recorder JSON (recent trace events)".into(),
    })
    pub attr FLIGHT_RECORDER: String;

    /// Whether this actor is infrastructure/system.
    @meta(INTROSPECT = IntrospectAttr {
        name: "is_system".into(),
        desc: "Whether this actor is infrastructure/system".into(),
    })
    pub attr IS_SYSTEM: bool = false;

    // Failure attrs — decomposition of FailureInfo into flat attrs.
    //
    // - **FI-A1 (presence):** failure_* attrs are present iff
    //   status == "failed"; absent otherwise. (Attr-level restatement
    //   of FI-3.)
    // - **FI-A2 (propagation):** failure_is_propagated == true iff
    //   failure_root_cause_actor != this actor's id. (Attr-level
    //   restatement of FI-4.)
    // FI-1, FI-2 (write ordering) are enforced in proc.rs serve()
    // and are unaffected by the representation change.
    // FI-5, FI-6 are proc/mesh-level and unaffected.

    /// Failure error message.
    @meta(INTROSPECT = IntrospectAttr {
        name: "failure_error_message".into(),
        desc: "Failure error message".into(),
    })
    pub attr FAILURE_ERROR_MESSAGE: String;

    /// Actor that caused the failure (root cause).
    @meta(INTROSPECT = IntrospectAttr {
        name: "failure_root_cause_actor".into(),
        desc: "Actor that caused the failure (root cause)".into(),
    })
    pub attr FAILURE_ROOT_CAUSE_ACTOR: String;

    /// Name of root cause actor.
    @meta(INTROSPECT = IntrospectAttr {
        name: "failure_root_cause_name".into(),
        desc: "Name of root cause actor".into(),
    })
    pub attr FAILURE_ROOT_CAUSE_NAME: String;

    /// Timestamp when failure occurred.
    @meta(INTROSPECT = IntrospectAttr {
        name: "failure_occurred_at".into(),
        desc: "Timestamp when failure occurred".into(),
    })
    pub attr FAILURE_OCCURRED_AT: SystemTime;

    /// Whether the failure was propagated from a child.
    @meta(INTROSPECT = IntrospectAttr {
        name: "failure_is_propagated".into(),
        desc: "Whether the failure was propagated from a child".into(),
    })
    pub attr FAILURE_IS_PROPAGATED: bool = false;
}

/// Structured failure information extracted from an
/// [`ActorSupervisionEvent`](crate::supervision::ActorSupervisionEvent)
/// at introspection time. Provides root-cause provenance without
/// carrying the recursive event type across the wire.
///
/// # Invariants
///
/// The failure introspection pipeline (from `serve()` in `proc.rs`
/// through this struct to the TUI) maintains six invariants. Each
/// is documented at its enforcement site with an `INV-N` comment.
///
/// - **FI-1 (event-before-status):** All `InstanceCell` state that
///   [`live_actor_payload`] reads must be written BEFORE
///   `change_status()` transitions to terminal. Enforced in
///   `proc.rs` `serve()`.
/// - **FI-2 (write-once):** `InstanceCellState::supervision_event`
///   is written at most once per actor lifetime. Enforced in
///   `proc.rs` `serve()` terminal paths.
/// - **FI-3 (failure_info ↔ actor_status):**
///   `failure_info.is_some()` iff
///   `actor_status.starts_with("failed:")`. Enforced in
///   [`live_actor_payload`].
/// - **FI-4 (is_propagated ↔ root_cause_actor):**
///   `is_propagated == true` iff `root_cause_actor !=
///   this_actor_id`. Enforced in [`live_actor_payload`].
/// - **FI-5 (is_poisoned ↔ failed_actor_count):**
///   `is_poisoned == true` iff `failed_actor_count > 0`. Enforced
///   in `mesh_agent.rs` `publish_introspect_properties()`.
/// - **FI-6 (clean stop = no artifacts):** When an actor stops
///   cleanly, `supervision_event` is `None`, `failure_info` is
///   `None`, and the actor does not contribute to
///   `failed_actor_count`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
pub struct FailureInfo {
    /// The error message (from `ActorStatus::Failed` display).
    pub error_message: String,
    /// The actor that originally caused the failure (root cause).
    /// Same as this actor if the failure originated here.
    pub root_cause_actor: String,
    /// Display name of the root cause actor, if set.
    pub root_cause_name: Option<String>,
    /// When the failure occurred (formatted timestamp).
    pub occurred_at: String,
    /// `true` if this actor is propagating a child's failure,
    /// `false` if the failure originated in this actor.
    pub is_propagated: bool,
}
wirevalue::register_type!(FailureInfo);

/// Kept "wire-friendly" (no `serde_json::Value`) so it can be encoded
/// via wirevalue's bincode path, while the HTTP layer can still
/// expose structured JSON via `Serialize`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
pub enum NodeProperties {
    /// Synthetic mesh root node (not a real actor/proc).
    Root {
        /// Number of hosts registered with the mesh admin agent.
        num_hosts: usize,
        /// When the mesh was started (ISO-8601 timestamp).
        started_at: String,
        /// Username who started the mesh.
        started_by: String,
        /// Children that are infrastructure-owned (system procs,
        /// admin procs) and should be hidden by default.
        system_children: Vec<String>,
    },

    /// A host in the mesh, represented by its `HostAgent`.
    Host {
        /// Host address (e.g. `127.0.0.1:12345`).
        addr: String,
        /// Number of procs currently reported on this host.
        num_procs: usize,
        /// References of children that are system/infrastructure
        /// procs.
        system_children: Vec<String>,
    },

    /// Properties describing a proc running on a host.
    Proc {
        /// Human-readable proc identifier.
        proc_name: String,
        /// Number of actors currently hosted by this proc.
        num_actors: usize,
        /// References of children that are system/infrastructure
        /// actors. Allows the TUI to filter lazily without fetching
        /// each child individually.
        system_children: Vec<String>,
        /// References of children that are stopped or failed.
        /// Populated from terminated snapshots so the TUI can
        /// filter/gray without per-child fetches.
        stopped_children: Vec<String>,
        /// Maximum number of terminated snapshots retained.
        /// When `stopped_children.len() >= stopped_retention_cap`,
        /// the list is at capacity and older entries were evicted.
        stopped_retention_cap: usize,
        /// Whether this proc is refusing new spawns because at least
        /// one actor has an unhandled supervision event.
        /// FI-5: `is_poisoned` iff `failed_actor_count > 0`.
        is_poisoned: bool,
        /// Number of actors with unhandled supervision events.
        failed_actor_count: usize,
    },

    /// Runtime metadata for a single actor instance.
    Actor {
        /// Current lifecycle/status of the actor (e.g. "Running",
        /// "Stopped").
        actor_status: String,
        /// Concrete actor type name.
        actor_type: String,
        /// Total number of messages processed by this actor so far.
        messages_processed: u64,
        /// Actor creation time, as an ISO-8601 timestamp string.
        created_at: String,
        /// Name of the most recent message handler run by the actor,
        /// if known.
        last_message_handler: Option<String>,
        /// Cumulative time spent processing messages, in
        /// microseconds.
        total_processing_time_us: u64,
        /// Serialized flight-recorder events for the actor, if
        /// enabled/available.
        flight_recorder: Option<String>,
        /// Whether this actor is infrastructure-owned (e.g.
        /// ProcAgent, HostAgent) rather than user-created.
        is_system: bool,
        /// Structured failure information, present only for failed
        /// actors. `None` for running or cleanly stopped actors.
        /// FI-3: `failure_info.is_some()` iff
        /// `actor_status.starts_with("failed:")`.
        failure_info: Option<FailureInfo>,
    },

    /// Error sentinel returned when a child reference cannot be
    /// resolved.
    Error {
        /// Machine-readable error code (e.g. "not_found").
        code: String,
        /// Human-readable error message.
        message: String,
    },
}
wirevalue::register_type!(NodeProperties);

/// Uniform response for any node in the mesh topology.
///
/// Every addressable entity (root, host, proc, actor) is represented
/// as a `NodePayload`. The client navigates the mesh by fetching a
/// node and following its `children` references.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
pub struct NodePayload {
    /// Canonical reference string for this node.
    pub identity: String,
    /// Node-specific metadata (type, status, metrics, etc.).
    pub properties: NodeProperties,
    /// Reference strings the client can GET next to descend the
    /// tree.
    pub children: Vec<String>,
    /// Parent node reference for upward navigation.
    pub parent: Option<String>,
    /// ISO 8601 timestamp indicating when this data was captured.
    pub as_of: String,
}
wirevalue::register_type!(NodePayload);

/// Context for introspection query - what aspect of the actor to
/// describe.
///
/// Infrastructure actors (e.g., ProcAgent, HostAgent)
/// have dual nature: they manage entities (Proc, Host) while also
/// being actors themselves. IntrospectView allows callers to
/// specify which aspect to query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Named)]
pub enum IntrospectView {
    /// Return managed-entity properties (Proc, Host, etc.) for
    /// infrastructure actors.
    Entity,
    /// Return standard actor properties (status, messages_processed,
    /// flight_recorder).
    Actor,
}
wirevalue::register_type!(IntrospectView);

/// Introspection query sent to any actor.
///
/// `Query` asks the actor to describe itself. `QueryChild` asks the
/// actor to describe one of its non-addressable children — an entity
/// that appears in the navigation tree but has no mailbox of its own
/// (e.g. a system proc owned by a host). The parent actor answers on
/// the child's behalf.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Named)]
pub enum IntrospectMessage {
    /// "Describe yourself."
    Query {
        /// View context - Entity or Actor.
        view: IntrospectView,
        /// Reply port receiving the actor's self-description.
        reply: reference::OncePortRef<NodePayload>,
    },
    /// "Describe one of your children."
    QueryChild {
        /// Reference identifying the child to describe.
        child_ref: reference::Reference,
        /// Reply port receiving the child's description.
        reply: reference::OncePortRef<NodePayload>,
    },
}
wirevalue::register_type!(IntrospectMessage);

/// Structured tracing event from the actor-local flight recorder.
///
/// Deserialization target for the `flight_recorder` JSON string in
/// [`NodeProperties::Actor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedEvent {
    /// ISO 8601 timestamp of the event.
    pub timestamp: String,
    /// Monotonic sequence number for ordering.
    #[serde(default)]
    pub seq: usize,
    /// Event level (INFO, DEBUG, etc.).
    pub level: String,
    /// Event target (module path).
    #[serde(default)]
    pub target: String,
    /// Event name.
    pub name: String,
    /// Event fields as JSON.
    pub fields: serde_json::Value,
}

/// Domain-specific properties an actor may publish for introspection.
///
/// Infrastructure actors (HostAgent, ProcAgent) push these to
/// make their managed-entity metadata available to the introspection
/// runtime without going through the actor's message handler. The
/// runtime handler reads the last-published value and merges it into
/// the [`NodePayload`] response for Entity-view queries.
///
/// Values may be arbitrarily stale for stuck actors — they reflect
/// whatever the actor last published before it stopped making
/// progress. The `published_at` timestamp makes staleness visible to
/// tooling.
#[derive(Debug, Clone)]
pub struct PublishedProperties {
    /// When these properties were last published.
    pub published_at: SystemTime,
    /// Domain-specific metadata.
    pub kind: PublishedPropertiesKind,
}

/// The domain-specific metadata variants that an actor may publish.
///
/// Only `Host` and `Proc` variants are available — actors cannot
/// publish `Root` or `Error` payloads.
#[derive(Debug, Clone)]
pub enum PublishedPropertiesKind {
    /// A host in the mesh.
    Host {
        /// Host address (e.g. `127.0.0.1:12345`).
        addr: String,
        /// Number of procs currently reported on this host.
        num_procs: usize,
        /// Custom children list (system procs + user procs).
        children: Vec<String>,
        /// Children that are system/infrastructure procs.
        system_children: Vec<String>,
    },
    /// A proc running on a host.
    Proc {
        /// Human-readable proc identifier.
        proc_name: String,
        /// Number of actors currently hosted by this proc.
        num_actors: usize,
        /// Custom children list (all actors in the proc).
        children: Vec<String>,
        /// Children that are system/infrastructure actors, reported
        /// by the proc so the TUI can filter without fetching each
        /// child.
        system_children: Vec<String>,
        /// Children that are stopped or failed. Populated from
        /// terminated snapshots so the TUI can filter/gray without
        /// per-child fetches.
        stopped_children: Vec<String>,
        /// Maximum number of terminated snapshots retained.
        stopped_retention_cap: usize,
        /// Whether this proc is refusing new spawns due to actor
        /// failures.
        is_poisoned: bool,
        /// Number of actors with unhandled supervision events.
        failed_actor_count: usize,
    },
}

/// Format a [`SystemTime`] as an ISO 8601 timestamp with millisecond
/// precision.
pub fn format_timestamp(time: SystemTime) -> String {
    humantime::format_rfc3339_millis(time).to_string()
}

/// Build a [`NodePayload`] from live [`InstanceCell`] state.
///
/// Reads the current live status and last handler directly from
/// the cell. Used by the introspect task (which runs outside
/// the actor's message loop) and by `Instance::introspect_payload`.
pub fn live_actor_payload(cell: &InstanceCell) -> NodePayload {
    let actor_id = cell.actor_id();
    let status = cell.status().borrow().clone();
    let last_handler = cell.last_message_handler();

    let children: Vec<String> = cell
        .child_actor_ids()
        .into_iter()
        .map(|id| id.to_string())
        .collect();

    let events = cell.recording().tail();
    let flight_recorder_events: Vec<RecordedEvent> = events
        .into_iter()
        .map(|event| RecordedEvent {
            timestamp: format_timestamp(event.time),
            seq: event.seq,
            level: event.metadata.level().to_string(),
            target: event.metadata.target().to_string(),
            name: event.metadata.name().to_string(),
            fields: event.json_value(),
        })
        .collect();

    let flight_recorder = if flight_recorder_events.is_empty() {
        None
    } else {
        serde_json::to_string(&flight_recorder_events).ok()
    };

    let supervisor = cell.parent().map(|p| p.actor_id().to_string());

    // FI-3: failure_info is computed from the same status value as
    // actor_status, ensuring they agree on whether the actor failed.
    let failure_info = if status.is_failed() {
        cell.supervision_event().map(|event| {
            let root = event.actually_failing_actor();
            // FI-4: is_propagated iff root_cause_actor != this actor.
            let is_propagated = root.actor_id != *actor_id;
            FailureInfo {
                error_message: event.actor_status.to_string(),
                root_cause_actor: root.actor_id.to_string(),
                root_cause_name: root.display_name.clone(),
                occurred_at: format_timestamp(event.occurred_at),
                is_propagated,
            }
        })
    } else {
        None
    };

    NodePayload {
        identity: actor_id.to_string(),
        properties: NodeProperties::Actor {
            actor_status: status.to_string(),
            actor_type: cell.actor_type_name().to_string(),
            messages_processed: cell.num_processed_messages(),
            created_at: format_timestamp(cell.created_at()),
            last_message_handler: last_handler.map(|info| info.to_string()),
            total_processing_time_us: cell.total_processing_time_us(),
            flight_recorder,
            is_system: cell.is_system()
                || cell.published_properties().is_some_and(|p| {
                    matches!(
                        p.kind,
                        PublishedPropertiesKind::Host { .. } | PublishedPropertiesKind::Proc { .. }
                    )
                }),
            failure_info,
        },
        children,
        parent: supervisor,
        as_of: format_timestamp(std::time::SystemTime::now()),
    }
}

/// Introspect task: runs on a dedicated tokio task per actor,
/// handling [`IntrospectMessage`] by reading [`InstanceCell`]
/// directly and replying via the actor's [`Mailbox`].
///
/// The actor's message loop never sees these messages.
///
/// # Invariants
///
/// - **S1:** Introspection does not depend on actor responsiveness --
///   this task runs independently; a wedged actor is still introspectable.
/// - **S2:** Introspection does not perturb observed state -- reads
///   `InstanceCell` directly, never sets `last_message_handler`.
/// - **S4:** `IntrospectMessage` never produces a `WorkCell` -- the
///   introspect port has its own channel, separate from the work queue.
/// - **S5:** Replies never use `PanickingMailboxSender` -- replies go
///   through `Mailbox::serialize_and_send_once`.
/// - **S6:** View semantics -- Actor view uses live structural state +
///   supervision children; Entity view uses published properties +
///   domain children.
/// - **S11:** Terminated snapshots do not keep actors resolvable --
///   `store_terminated_snapshot` writes to the proc's snapshot map,
///   not the instances map. `resolve_actor_ref` checks terminal status
///   independently and is unaffected by snapshot storage.
pub async fn serve_introspect(
    cell: InstanceCell,
    mailbox: crate::mailbox::Mailbox,
    mut receiver: crate::mailbox::PortReceiver<IntrospectMessage>,
) {
    use crate::actor::ActorStatus;
    use crate::mailbox::PortSender as _;

    // Watch for terminal status so we can break the reference cycle:
    // InstanceCellState → Ports → introspect sender → keeps receiver
    // open → this task holds InstanceCell → InstanceCellState.
    // Without this, a stopped actor's InstanceCellState is never
    // dropped and the actor lingers in the proc's instances map.
    let mut status = cell.status().clone();

    loop {
        let msg = tokio::select! {
            msg = receiver.recv() => {
                match msg {
                    Ok(msg) => msg,
                    Err(_) => {
                        // Channel closed. If the actor reached a
                        // terminal state, snapshot it before exiting
                        // so it remains queryable post-mortem.
                        if cell.status().borrow().is_terminal() {
                            let snapshot = live_actor_payload(&cell);
                            cell.store_terminated_snapshot(snapshot);
                        }
                        break;
                    }
                }
            }
            _ = status.wait_for(ActorStatus::is_terminal) => {
                // Snapshot for post-mortem introspection before
                // dropping our InstanceCell reference.
                let snapshot = live_actor_payload(&cell);
                cell.store_terminated_snapshot(snapshot);
                break;
            }
        };

        let result = match msg {
            IntrospectMessage::Query { view, reply } => {
                let payload = match view {
                    IntrospectView::Entity => match cell.published_properties() {
                        Some(props) => {
                            let published_at = props.published_at;
                            let children = match &props.kind {
                                PublishedPropertiesKind::Host { children, .. } => children.clone(),
                                PublishedPropertiesKind::Proc { children, .. } => children.clone(),
                            };
                            let properties = match props.kind {
                                PublishedPropertiesKind::Host {
                                    addr,
                                    num_procs,
                                    system_children,
                                    ..
                                } => NodeProperties::Host {
                                    addr,
                                    num_procs,
                                    system_children,
                                },
                                PublishedPropertiesKind::Proc {
                                    proc_name,
                                    num_actors,
                                    system_children,
                                    stopped_children,
                                    stopped_retention_cap,
                                    is_poisoned,
                                    failed_actor_count,
                                    ..
                                } => NodeProperties::Proc {
                                    proc_name,
                                    num_actors,
                                    system_children,
                                    stopped_children,
                                    stopped_retention_cap,
                                    is_poisoned,
                                    failed_actor_count,
                                },
                            };
                            NodePayload {
                                identity: cell.actor_id().to_string(),
                                properties,
                                children,
                                parent: cell.parent().map(|p| p.actor_id().to_string()),
                                as_of: format_timestamp(published_at),
                            }
                        }
                        None => live_actor_payload(&cell),
                    },
                    IntrospectView::Actor => live_actor_payload(&cell),
                };
                mailbox.serialize_and_send_once(
                    reply,
                    payload,
                    crate::mailbox::monitored_return_handle(),
                )
            }
            IntrospectMessage::QueryChild { child_ref, reply } => {
                let payload = cell.query_child(&child_ref).unwrap_or_else(|| NodePayload {
                    identity: String::new(),
                    properties: NodeProperties::Error {
                        code: "not_found".into(),
                        message: format!("child {} not found (no callback registered)", child_ref),
                    },
                    children: Vec::new(),
                    parent: None,
                    as_of: humantime::format_rfc3339_millis(std::time::SystemTime::now())
                        .to_string(),
                });
                mailbox.serialize_and_send_once(
                    reply,
                    payload,
                    crate::mailbox::monitored_return_handle(),
                )
            }
        };
        if let Err(e) = result {
            tracing::debug!("introspect reply failed: {e}");
        }
    }
    tracing::debug!(
        actor_id = %cell.actor_id(),
        "introspect task exiting"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enforces IK-1 (metadata completeness) for all actor-runtime
    /// introspection keys.
    #[test]
    fn test_introspect_keys_are_tagged() {
        let cases = vec![
            ("status", STATUS.attrs()),
            ("status_reason", STATUS_REASON.attrs()),
            ("actor_type", ACTOR_TYPE.attrs()),
            ("messages_processed", MESSAGES_PROCESSED.attrs()),
            ("created_at", CREATED_AT.attrs()),
            ("last_handler", LAST_HANDLER.attrs()),
            ("total_processing_time_us", TOTAL_PROCESSING_TIME_US.attrs()),
            ("flight_recorder", FLIGHT_RECORDER.attrs()),
            ("is_system", IS_SYSTEM.attrs()),
            ("failure_error_message", FAILURE_ERROR_MESSAGE.attrs()),
            ("failure_root_cause_actor", FAILURE_ROOT_CAUSE_ACTOR.attrs()),
            ("failure_root_cause_name", FAILURE_ROOT_CAUSE_NAME.attrs()),
            ("failure_occurred_at", FAILURE_OCCURRED_AT.attrs()),
            ("failure_is_propagated", FAILURE_IS_PROPAGATED.attrs()),
        ];

        for (expected_name, meta) in &cases {
            // IK-1: every key must have INTROSPECT with non-empty
            // name and desc.
            let introspect = meta
                .get(INTROSPECT)
                .unwrap_or_else(|| panic!("{expected_name}: missing INTROSPECT meta-attr"));
            assert_eq!(
                introspect.name, *expected_name,
                "short name mismatch for {expected_name}"
            );
            assert!(
                !introspect.desc.is_empty(),
                "{expected_name}: desc should not be empty"
            );
        }

        // Exhaustiveness: verify cases covers all INTROSPECT-tagged
        // keys declared in this module.
        use hyperactor_config::attrs::AttrKeyInfo;
        let registry_count = inventory::iter::<AttrKeyInfo>()
            .filter(|info| {
                info.name.starts_with("hyperactor::introspect::")
                    && info.meta.get(INTROSPECT).is_some()
            })
            .count();
        assert_eq!(
            cases.len(),
            registry_count,
            "test must cover all INTROSPECT-tagged keys in this module"
        );
    }

    /// Enforces IK-2 (short-name uniqueness) and metadata quality
    /// across all crates linked into this test binary. Iterates
    /// the global `declare_attrs!` inventory and checks that no two
    /// keys tagged with `INTROSPECT` share the same short name, and
    /// that all tagged keys have non-empty name and desc.
    #[test]
    fn test_introspect_short_names_are_globally_unique() {
        use hyperactor_config::attrs::AttrKeyInfo;

        let mut seen = std::collections::HashMap::new();
        for info in inventory::iter::<AttrKeyInfo>() {
            let Some(introspect) = info.meta.get(INTROSPECT) else {
                continue;
            };
            // Metadata quality: every tagged key must have
            // non-empty name and desc.
            assert!(
                !introspect.name.is_empty(),
                "INTROSPECT key {:?} has empty name",
                info.name
            );
            assert!(
                !introspect.desc.is_empty(),
                "INTROSPECT key {:?} has empty desc",
                info.name
            );
            if let Some(prev_fq) = seen.insert(introspect.name.clone(), info.name) {
                panic!(
                    "IK-2 violation: duplicate short name {:?} declared by both {:?} and {:?}",
                    introspect.name, prev_fq, info.name
                );
            }
        }
    }
}
