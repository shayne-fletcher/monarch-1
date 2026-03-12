/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Introspection attr keys — mesh-topology concepts.
//!
//! These keys are published by `HostMeshAgent`, `ProcAgent`, and
//! `MeshAdminAgent` to describe mesh topology (hosts, procs, root).
//! Actor-runtime keys (status, actor_type, messages_processed, etc.)
//! are declared in `hyperactor::introspect`.
//!
//! See `hyperactor::introspect` for naming convention, invariant
//! labels, and the `IntrospectAttr` meta-attribute pattern.

use hyperactor_config::INTROSPECT;
use hyperactor_config::IntrospectAttr;
use hyperactor_config::declare_attrs;

// Invariants:
//
// - **MK-1 (metadata completeness):** Every mesh-topology
//   introspection key must carry `@meta(INTROSPECT = ...)` with
//   non-empty `name` and `desc`. Enforced by
//   `test_mesh_introspect_keys_are_tagged`.
// - **MK-2 (short-name uniqueness):** Covered by
//   `test_introspect_short_names_are_globally_unique` in
//   `hyperactor::introspect` (which iterates all linked crates). Full
//   cross-crate coverage requires a test binary that links both
//   `hyperactor` and `hyperactor_mesh`.
declare_attrs! {
    /// Topology role of this node: "root", "host", "proc", "error".
    @meta(INTROSPECT = IntrospectAttr {
        name: "node_type".into(),
        desc: "Topology role: root, host, proc, error".into(),
    })
    pub attr NODE_TYPE: String;

    /// Host network address (e.g. "10.0.0.1:8080").
    @meta(INTROSPECT = IntrospectAttr {
        name: "addr".into(),
        desc: "Host network address".into(),
    })
    pub attr ADDR: String;

    /// Number of procs on a host.
    @meta(INTROSPECT = IntrospectAttr {
        name: "num_procs".into(),
        desc: "Number of procs on a host".into(),
    })
    pub attr NUM_PROCS: usize = 0;

    /// Human-readable proc name.
    @meta(INTROSPECT = IntrospectAttr {
        name: "proc_name".into(),
        desc: "Human-readable proc name".into(),
    })
    pub attr PROC_NAME: String;

    /// Number of actors in a proc.
    @meta(INTROSPECT = IntrospectAttr {
        name: "num_actors".into(),
        desc: "Number of actors in a proc".into(),
    })
    pub attr NUM_ACTORS: usize = 0;

    /// References of system/infrastructure children.
    @meta(INTROSPECT = IntrospectAttr {
        name: "system_children".into(),
        desc: "References of system/infrastructure children".into(),
    })
    pub attr SYSTEM_CHILDREN: Vec<String>;

    /// References of stopped children (proc only).
    @meta(INTROSPECT = IntrospectAttr {
        name: "stopped_children".into(),
        desc: "References of stopped children".into(),
    })
    pub attr STOPPED_CHILDREN: Vec<String>;

    /// Cap on stopped children retention.
    @meta(INTROSPECT = IntrospectAttr {
        name: "stopped_retention_cap".into(),
        desc: "Maximum number of stopped children retained".into(),
    })
    pub attr STOPPED_RETENTION_CAP: usize = 0;

    /// Whether this proc is refusing new spawns due to actor
    /// failures.
    @meta(INTROSPECT = IntrospectAttr {
        name: "is_poisoned".into(),
        desc: "Whether this proc is poisoned (refusing new spawns)".into(),
    })
    pub attr IS_POISONED: bool = false;

    /// Count of failed actors in a proc.
    @meta(INTROSPECT = IntrospectAttr {
        name: "failed_actor_count".into(),
        desc: "Number of failed actors in this proc".into(),
    })
    pub attr FAILED_ACTOR_COUNT: usize = 0;

    /// Timestamp when the mesh was started.
    @meta(INTROSPECT = IntrospectAttr {
        name: "started_at".into(),
        desc: "Timestamp when the mesh was started".into(),
    })
    pub attr STARTED_AT: std::time::SystemTime;

    /// Username who started the mesh.
    @meta(INTROSPECT = IntrospectAttr {
        name: "started_by".into(),
        desc: "Username who started the mesh".into(),
    })
    pub attr STARTED_BY: String;

    /// Number of hosts in the mesh (root only).
    @meta(INTROSPECT = IntrospectAttr {
        name: "num_hosts".into(),
        desc: "Number of hosts in the mesh".into(),
    })
    pub attr NUM_HOSTS: usize = 0;

    /// Error code for error nodes.
    @meta(INTROSPECT = IntrospectAttr {
        name: "error_code".into(),
        desc: "Machine-readable error code (e.g. not_found)".into(),
    })
    pub attr ERROR_CODE: String;

    /// Error message for error nodes.
    @meta(INTROSPECT = IntrospectAttr {
        name: "error_message".into(),
        desc: "Human-readable error message".into(),
    })
    pub attr ERROR_MESSAGE: String;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enforces MK-1 (metadata completeness) for all mesh-topology
    /// introspection keys.
    #[test]
    fn test_mesh_introspect_keys_are_tagged() {
        let cases = vec![
            ("node_type", NODE_TYPE.attrs()),
            ("addr", ADDR.attrs()),
            ("num_procs", NUM_PROCS.attrs()),
            ("proc_name", PROC_NAME.attrs()),
            ("num_actors", NUM_ACTORS.attrs()),
            ("system_children", SYSTEM_CHILDREN.attrs()),
            ("stopped_children", STOPPED_CHILDREN.attrs()),
            ("stopped_retention_cap", STOPPED_RETENTION_CAP.attrs()),
            ("is_poisoned", IS_POISONED.attrs()),
            ("failed_actor_count", FAILED_ACTOR_COUNT.attrs()),
            ("started_at", STARTED_AT.attrs()),
            ("started_by", STARTED_BY.attrs()),
            ("num_hosts", NUM_HOSTS.attrs()),
            ("error_code", ERROR_CODE.attrs()),
            ("error_message", ERROR_MESSAGE.attrs()),
        ];

        for (expected_name, meta) in &cases {
            // MK-1: every key must have INTROSPECT with non-empty
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
                info.name.starts_with("hyperactor_mesh::introspect::")
                    && info.meta.get(INTROSPECT).is_some()
            })
            .count();
        assert_eq!(
            cases.len(),
            registry_count,
            "test must cover all INTROSPECT-tagged keys in this module"
        );
    }
}
