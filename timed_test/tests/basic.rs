/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use timed_test::async_timed_test;

#[async_timed_test(timeout_secs = 5)]
async fn good() {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
}

#[async_timed_test(timeout_secs = 1)]
#[should_panic]
async fn bad() {
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
}
