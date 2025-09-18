# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

del sys.path[0]
from monarch.actor import this_host  # noqa


this_host().spawn_procs().initialized.get()
