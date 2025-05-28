# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import contextlib

META_VAL = []


@contextlib.contextmanager
def set_meta(new_value):
    # Sets the metadata for any tasks created under this
    global META_VAL
    META_VAL.append(new_value)
    try:
        yield
    finally:
        META_VAL.pop()
