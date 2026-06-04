/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Exercises MIT-79 (execution surface, real Python hooks end-to-end)
//! by spawning the `execution_workload` Python binary and steering it
//! through the stdin-command / stdout-sentinel handshake
//! (`WorkloadFixture::send_command` / `wait_for_stdout`). NO sleeps:
//! every poll happens only after the relevant `EXEC_ENTERED` /
//! `EXEC_ACK` sentinel, so it proves mesh-admin truth rather than
//! winning a timing race.
//!
//! Unlike the `ExecutionRegistry` unit tests (idempotence, truncation,
//! aggregation, oldest-rollover) and the DTO json-shape tests, this test
//! proves the *real* `_Actor.handle` hooks increment the `execution`
//! surface on entry and decrement on exit (completion and exception),
//! across direct AND queue dispatch, visible end-to-end through
//! `GET /v1/{actor}` and the typed `NodeProperties::Actor { execution, .. }`.
//! Cancellation is actor-fatal in the current runtime (`CancelledError`
//! is a `BaseException` -> the `except BaseException` arm re-raises -> a
//! kill signal), so it cannot be observed as `count -> 0` on a surviving
//! actor; the framework `finally` still decrements first (no leak -- the
//! cell is torn down with the actor anyway), and that decrement path is
//! covered by the `ExecutionRegistry` unit tests.
//!
//! Coverage:
//!   - direct dispatch: one held invocation -> count 1, populated row.
//!   - same-name aggregation: two held `hold`s -> count 2, ONE row,
//!     `active_count == 2`.
//!   - completion decrement: release each -> count drops; after the last
//!     release the FULL zero-shape returns (post-activity clear).
//!   - exception decrement: raise a held invocation (via `call_one`, so
//!     the handler error returns as `ActorError` and the actor survives)
//!     -> count drops back to the full zero-shape.
//!   - queue dispatch: one held invocation -> count 1 (hook fires in
//!     queue mode too).
//!   - idle sibling: count 0.

use std::time::Duration;
use std::time::Instant;

use hyperactor_mesh::introspect::ExecutionInfo;
use hyperactor_mesh::introspect::NodePayload;
use hyperactor_mesh::introspect::NodeProperties;

use crate::harness;
use crate::harness::WorkloadFixture;

/// Per-sentinel wait budget. Generous: the workload only needs to spawn
/// a task and reach the first handler line.
const SENTINEL_TIMEOUT: Duration = Duration::from_secs(30);
/// Budget for the async `finally` decrement to become visible after the
/// `control` reply returns. The Python `finally` runs after the reply,
/// so we poll rather than assume immediate.
const DECREMENT_POLL_TIMEOUT: Duration = Duration::from_secs(15);
const DECREMENT_POLL_INTERVAL: Duration = Duration::from_millis(100);

fn enc(s: &str) -> String {
    urlencoding::encode(s).into_owned()
}

/// Walk root -> hosts -> procs -> actors and return the ref of the first
/// actor whose label equals `label`. ActorAddr.to_string() is
/// `{actor_uid}.{proc_id}@{location}` where actor_uid is `label<base58>`;
/// match the label exactly so we don't pick up the controller actor
/// (`actor_mesh_controller_<label><...>`).
async fn find_actor_by_label(fixture: &WorkloadFixture, label: &str) -> String {
    let root: NodePayload = fixture
        .get_node_payload("/v1/root")
        .await
        .expect("MIT-79: GET /v1/root failed");
    for host_ref in &root.children {
        let host: NodePayload = fixture
            .get_node_payload(&format!("/v1/{}", enc(&host_ref.to_string())))
            .await
            .unwrap_or_else(|e| panic!("MIT-79: GET /v1/{host_ref} failed: {e:#}"));
        for proc_ref in &host.children {
            let proc: NodePayload = fixture
                .get_node_payload(&format!("/v1/{}", enc(&proc_ref.to_string())))
                .await
                .unwrap_or_else(|e| panic!("MIT-79: GET /v1/{proc_ref} failed: {e:#}"));
            for actor_ref in &proc.children {
                let s = actor_ref.to_string();
                if s.split('<').next().unwrap_or("") == label {
                    return s;
                }
            }
        }
    }
    panic!("MIT-79: actor with label {label:?} not found in topology");
}

/// Fetch the typed `ExecutionInfo` for `actor_ref`.
async fn execution_of(fixture: &WorkloadFixture, actor_ref: &str) -> ExecutionInfo {
    let path = format!("/v1/{}", enc(actor_ref));
    let actor: NodePayload = fixture
        .get_node_payload(&path)
        .await
        .unwrap_or_else(|e| panic!("MIT-79: GET {path} failed: {e:#}"));
    match actor.properties {
        NodeProperties::Actor { execution, .. } => *execution,
        other => panic!("MIT-79: expected Actor variant for {actor_ref}, got {other:?}"),
    }
}

/// Poll `/v1/{actor}` until `pred(execution)` holds or the budget
/// elapses. Bounded retries, never sleep-then-assume: the async Python
/// `finally` decrement is observed, not waited out blindly. Returns the
/// last observed snapshot (matching or not) for the failure message.
async fn poll_until(
    fixture: &WorkloadFixture,
    actor_ref: &str,
    pred: impl Fn(&ExecutionInfo) -> bool,
) -> ExecutionInfo {
    let deadline = Instant::now() + DECREMENT_POLL_TIMEOUT;
    loop {
        let exec = execution_of(fixture, actor_ref).await;
        if pred(&exec) || Instant::now() >= deadline {
            return exec;
        }
        tokio::time::sleep(DECREMENT_POLL_INTERVAL).await;
    }
}

/// Assert the full idle zero-shape.
fn assert_zero_shape(exec: &ExecutionInfo, ctx: &str) {
    assert_eq!(
        exec.active_handler_count, 0,
        "MIT-79 ({ctx}): active_handler_count must be 0; got {exec:?}",
    );
    assert_eq!(
        exec.total_handler_names, 0,
        "MIT-79 ({ctx}): total_handler_names must be 0 (no stale hidden-name count); got {exec:?}",
    );
    assert!(
        exec.oldest_active_handler.is_none(),
        "MIT-79 ({ctx}): oldest_active_handler must be None; got {exec:?}",
    );
    assert!(
        exec.oldest_active_since.is_none(),
        "MIT-79 ({ctx}): oldest_active_since must be None; got {exec:?}",
    );
    assert!(
        exec.active_handlers.is_empty(),
        "MIT-79 ({ctx}): active_handlers must be empty; got {exec:?}",
    );
    assert!(
        !exec.active_handlers_truncated,
        "MIT-79 ({ctx}): active_handlers_truncated must be false; got {exec:?}",
    );
}

async fn run_inner(fixture: &WorkloadFixture) {
    let busy = find_actor_by_label(fixture, "busy_actor").await;
    let idle = find_actor_by_label(fixture, "idle_actor").await;
    let queue = find_actor_by_label(fixture, "queue_actor").await;

    // --- Direct dispatch: one held invocation -> count 1. ---
    fixture
        .send_command("HOLD busy a")
        .await
        .expect("MIT-79: send HOLD busy a");
    fixture
        .wait_for_stdout("EXEC_ENTERED a", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ENTERED a");

    let exec = execution_of(fixture, &busy).await;
    assert_eq!(
        exec.active_handler_count, 1,
        "MIT-79: one held invocation must report active_handler_count == 1; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers.len(),
        1,
        "MIT-79: one held invocation must report exactly one active_handlers row; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers[0].name, "hold",
        "MIT-79: the active row name must be the Python method name `hold`; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers[0].active_count, 1,
        "MIT-79: the active row active_count must be 1; got {exec:?}",
    );
    assert_eq!(
        exec.oldest_active_handler.as_deref(),
        Some("hold"),
        "MIT-79: oldest_active_handler must be `hold`; got {exec:?}",
    );
    assert!(
        exec.oldest_active_since.is_some(),
        "MIT-79: oldest_active_since must be populated; got {exec:?}",
    );

    // --- Same-name aggregation: two held `hold`s -> count 2, ONE row. ---
    fixture
        .send_command("HOLD busy b")
        .await
        .expect("MIT-79: send HOLD busy b");
    fixture
        .wait_for_stdout("EXEC_ENTERED b", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ENTERED b");

    let exec = execution_of(fixture, &busy).await;
    assert_eq!(
        exec.active_handler_count, 2,
        "MIT-79: two held invocations must report active_handler_count == 2; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers.len(),
        1,
        "MIT-79: two invocations of the SAME endpoint must aggregate into ONE row \
         (a one-row-per-invocation bug fails here); got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers[0].active_count, 2,
        "MIT-79: the aggregated row active_count must be 2; got {exec:?}",
    );
    assert_eq!(
        exec.oldest_active_handler.as_deref(),
        Some("hold"),
        "MIT-79: oldest_active_handler must remain `hold`; got {exec:?}",
    );

    // --- Completion decrement: release a -> count drops to 1. ---
    fixture
        .send_command("RELEASE busy a")
        .await
        .expect("MIT-79: send RELEASE busy a");
    fixture
        .wait_for_stdout("EXEC_ACK release a", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ACK release a");

    let exec = poll_until(fixture, &busy, |e| e.active_handler_count == 1).await;
    assert_eq!(
        exec.active_handler_count, 1,
        "MIT-79: after releasing one of two, count must drop to 1 (the finally ran); got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers.len(),
        1,
        "MIT-79: one invocation should remain in a single row; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers[0].active_count, 1,
        "MIT-79: the remaining row active_count must be 1; got {exec:?}",
    );

    // --- Post-activity clear: release b -> FULL zero-shape returns. ---
    fixture
        .send_command("RELEASE busy b")
        .await
        .expect("MIT-79: send RELEASE busy b");
    fixture
        .wait_for_stdout("EXEC_ACK release b", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ACK release b");

    let exec = poll_until(fixture, &busy, |e| e.active_handler_count == 0).await;
    assert_zero_shape(&exec, "post-activity clear (completion)");

    // --- Exception decrement: raise a held invocation -> count drops to 0. ---
    fixture
        .send_command("HOLD busy c")
        .await
        .expect("MIT-79: send HOLD busy c");
    fixture
        .wait_for_stdout("EXEC_ENTERED c", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ENTERED c");
    let exec = execution_of(fixture, &busy).await;
    assert_eq!(
        exec.active_handler_count, 1,
        "MIT-79: held invocation `c` must report count 1 before raise; got {exec:?}",
    );

    fixture
        .send_command("RAISE busy c")
        .await
        .expect("MIT-79: send RAISE busy c");
    fixture
        .wait_for_stdout("EXEC_ACK raise c", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ACK raise c");
    let exec = poll_until(fixture, &busy, |e| e.active_handler_count == 0).await;
    assert_zero_shape(&exec, "exception (finally ran on exception)");
    // The raise is delivered via call_one, so the actor must SURVIVE the
    // exception (not be supervision-killed) -- otherwise "exception -> 0 on a
    // live actor" is hollow. Prove survival the strong way: the actor still
    // accepts and runs a NEW invocation after the exception.
    fixture
        .send_command("HOLD busy e")
        .await
        .expect("MIT-79: send HOLD busy e");
    fixture
        .wait_for_stdout("EXEC_ENTERED e", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ENTERED e");
    let exec = execution_of(fixture, &busy).await;
    assert_eq!(
        exec.active_handler_count, 1,
        "MIT-79: busy actor must still accept a new invocation after the \
         exception (count 1); got {exec:?}",
    );
    fixture
        .send_command("RELEASE busy e")
        .await
        .expect("MIT-79: send RELEASE busy e");
    fixture
        .wait_for_stdout("EXEC_ACK release e", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ACK release e");
    let exec = poll_until(fixture, &busy, |e| e.active_handler_count == 0).await;
    assert_zero_shape(&exec, "after post-exception interaction");

    // Cancellation is deliberately NOT exercised here: it is actor-fatal
    // in the current runtime (see the module doc), so `count -> 0` is not
    // observable on a surviving actor. The decrement-on-exit guarantee is
    // covered by the `ExecutionRegistry` unit tests.

    // --- Queue dispatch: one held invocation -> count 1 (hook fires). ---
    // Queue dispatch is serialized: we hold a single invocation and only
    // assert count == 1, never releasing it through a second endpoint
    // call (that would deadlock the dispatch loop). The proc tears down
    // at fixture shutdown.
    fixture
        .send_command("HOLD queue q")
        .await
        .expect("MIT-79: send HOLD queue q");
    fixture
        .wait_for_stdout("EXEC_ENTERED q", SENTINEL_TIMEOUT)
        .await
        .expect("MIT-79: wait EXEC_ENTERED q");
    let exec = execution_of(fixture, &queue).await;
    assert_eq!(
        exec.active_handler_count, 1,
        "MIT-79: queue-dispatch held invocation must report count 1 \
         (the hook fires in queue mode too); got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers.len(),
        1,
        "MIT-79: queue-dispatch held invocation must report one row; got {exec:?}",
    );
    assert_eq!(
        exec.active_handlers[0].name, "hold",
        "MIT-79: queue-dispatch active row name must be `hold`; got {exec:?}",
    );

    // --- Idle sibling: count 0 (never invoked). ---
    let exec = execution_of(fixture, &idle).await;
    assert_zero_shape(&exec, "idle sibling");
}

/// Drop guard for structural cleanup. See `dining::ShutdownGuard`.
struct ShutdownGuard<'a>(#[allow(dead_code)] &'a WorkloadFixture);

impl ShutdownGuard<'_> {
    fn disarm(self) {
        std::mem::forget(self);
    }
}

impl Drop for ShutdownGuard<'_> {
    fn drop(&mut self) {
        // Panic path: WorkloadFixture::Drop calls start_kill()
        // synchronously.
    }
}

/// MIT-79 entry point invoked from `main.rs`.
pub async fn run_execution_workload() {
    let bin = harness::execution_workload_binary();
    let fixture = harness::start_workload(&bin, &[], Duration::from_secs(60))
        .await
        .expect("MIT-79: failed to start execution_workload");
    let guard = ShutdownGuard(&fixture);
    run_inner(&fixture).await;
    guard.disarm();
    fixture.shutdown().await;
}
