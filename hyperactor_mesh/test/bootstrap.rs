/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// This is an "empty shell" bootstrap process,
/// simply invoking [`hyperactor_mesh::bootstrap_or_die`].
#[tokio::main]
async fn main() {
    hyperactor::initialize_with_current_runtime();
    hyperactor_mesh::bootstrap_or_die().await;
}
