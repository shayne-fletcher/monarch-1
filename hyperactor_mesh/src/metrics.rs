/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_telemetry::*;

declare_static_timer!(
    ACTOR_MESH_CAST_DURATION,
    "actor_mesh_cast_duration",
    TimeUnit::Micros
);
