# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import before monarch to pre-load torch DSOs as, in exploded wheel flows,
# our RPATHs won't correctly find them.
import torch  # isort:skip
