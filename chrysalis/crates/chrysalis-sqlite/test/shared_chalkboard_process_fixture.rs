/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::time::Duration;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("spawn-descendant-and-exit") {
        spawn_descendant();
        return;
    }
    std::thread::sleep(Duration::from_secs(60));
}

#[expect(
    clippy::zombie_processes,
    reason = "the fixture must exit without waiting so the harness can prove group cleanup"
)]
fn spawn_descendant() {
    std::process::Command::new(std::env::current_exe().expect("locate process fixture"))
        .spawn()
        .expect("spawn process-fixture descendant");
}
