/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use tracing::Level;

/// Set up a tracing subscriber with a filter, so we can print tracing logs
/// with >= level by using *buck run*.
///
/// Note this function does not work with *buck test*.
//
/// This is better than the traced_test macro when logs_contain and logs_assert
/// are not needed, because that macro prints TRACE level logs, which is too
/// verbose.
pub fn set_tracing_env_filter(level: Level) {
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(level.as_str()))
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
}
