#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dotslash

dotslash.export_fbcode_build(
    target="fbcode//monarch/hyper:hyper",
    oncall="monarch",
)
