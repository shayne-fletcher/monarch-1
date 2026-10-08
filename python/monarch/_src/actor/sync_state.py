# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import threading
from contextlib import contextmanager
from typing import Generator

# How many `fake_sync_state()` contexts are active on each thread.
_depth = threading.local()


@contextmanager
def fake_sync_state() -> Generator[None, None, None]:
    prev_loop = asyncio.events._get_running_loop()
    asyncio._set_running_loop(None)
    _depth.value = getattr(_depth, "value", 0) + 1
    try:
        yield
    finally:
        _depth.value -= 1
        asyncio._set_running_loop(prev_loop)


def in_fake_sync_state() -> bool:
    """Whether this thread is inside a loop that `fake_sync_state()` hides, as an
    async actor's sync `__supervise__` or `__cleanup__` is."""
    return getattr(_depth, "value", 0) > 0
