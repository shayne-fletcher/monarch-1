# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import Generator, Generic, TypeVar

R = TypeVar("R")


# TODO: consolidate with monarch.common.future
class ActorFuture(Generic[R]):
    def __init__(self, impl, blocking_impl=None):
        self._impl = impl
        self._blocking_impl = blocking_impl

    def get(self) -> R:
        if self._blocking_impl is not None:
            return self._blocking_impl()
        return asyncio.run(self._impl())

    def __await__(self) -> Generator[R, None, R]:
        return self._impl().__await__()
