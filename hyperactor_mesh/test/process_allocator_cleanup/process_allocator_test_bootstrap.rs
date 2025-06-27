/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// A simple bootstrap binary that writes logs out to a file. This is useful for
/// debugging, as normally the ProcessAllocator children logs are piped back to
/// ProcessAllocator. When we are testing what happens when we sigkill
/// ProcessAllocator, we want to see what is happening on the children.
#[tokio::main]
async fn main() {
    // Initialize tracing to a separate log file per child
    let pid = std::process::id();
    let log_file_path = format!("/tmp/child_log{}", pid);
    let log_file = std::fs::File::create(&log_file_path).expect("Failed to create log file");

    tracing_subscriber::fmt()
        .with_writer(log_file)
        .with_ansi(false) // No color codes in file
        .init();

    // Let the user know where to find our logs
    eprintln!("CHILD_LOG_FILE:{}: {}", pid, log_file_path);

    hyperactor_mesh::bootstrap_or_die().await;
}
