# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.warn(
    "monarch.proc_mesh is deprecated, please import from monarch.actor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from monarch._src.actor.proc_mesh import *  # noqa
