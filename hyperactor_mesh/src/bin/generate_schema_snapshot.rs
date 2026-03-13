/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Generates JSON Schema snapshot files for the mesh admin API.
//!
//! Produces raw `schemars::schema_for!` output (without `$id`) as
//! canonical pretty-printed JSON. The snapshot test in
//! `introspect::tests` compares against these files; the serve
//! endpoints add `$id` at runtime (SC-4).
//!
//! ## Usage
//!
//! Buck:
//! ```sh
//! buck run fbcode//monarch/hyperactor_mesh:generate_schema_snapshot \
//!   @fbcode//mode/dev-nosan -- \
//!   fbcode/monarch/hyperactor_mesh/src/testdata
//! ```
//!
//! Cargo:
//! ```sh
//! cargo run -p hyperactor_mesh --bin generate_schema_snapshot -- \
//!   hyperactor_mesh/src/testdata
//! ```
//!
//! Writes:
//! - `node_payload_schema.json` — `NodePayload` schema (SC-2)
//! - `error_schema.json` — `ApiErrorEnvelope` schema

use hyperactor_mesh::introspect::NodePayload;
use hyperactor_mesh::mesh_admin::ApiErrorEnvelope;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "src/testdata".into());

    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    write_schema::<NodePayload>(&out_dir, "node_payload_schema.json");
    write_schema::<ApiErrorEnvelope>(&out_dir, "error_schema.json");
}

fn write_schema<T: schemars::JsonSchema>(out_dir: &str, filename: &str) {
    let schema = schemars::schema_for!(T);
    let json = serde_json::to_string_pretty(&schema).expect("schema must be serializable");
    let path = format!("{out_dir}/{filename}");
    std::fs::write(&path, format!("{json}\n")).expect("failed to write schema file");
    eprintln!("wrote {path}");
}
