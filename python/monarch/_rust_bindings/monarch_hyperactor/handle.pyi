# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
from typing import Any, Generator, Generic, Optional, TypeVar

T = TypeVar("T")

class Handle(Generic[T]):
    """
    An observe-only handle to a background task. It resolves once and stays
    observable by any number of later observers. Unlike Shared, a Handle never
    drives a Python coroutine.
    """

    def get(self, timeout: Optional[float] = None) -> T:
        """
        Block the calling thread until the handle resolves and return its value.
        Behavior is keyed to the calling context, not to whether the value is
        ready: in a tokio runtime context it always raises WouldBlockRuntime, even
        for a ready value (blocking there would panic the runtime) -- use poll() or
        as_asyncio(); on a running asyncio loop it warns, since get() can freeze
        the loop; on a sync thread it blocks until resolved. On timeout it raises
        TimeoutError without cancelling the handle, so a later get()/poll()/await
        still observes completion.
        """
        ...

    def poll(self) -> Optional[T]:
        """
        If the handle has resolved, return the value; otherwise return None.
        Non-consuming: the value stays observable by later observers.
        """
        ...

    def as_asyncio(self) -> "asyncio.Future[T]":
        """
        Return a standard asyncio.Future that resolves when the handle does.
        Requires a running event loop; off a loop it raises RuntimeError.
        """
        ...

    def __await__(self) -> Generator[Any, Any, T]:
        """
        Await the handle on a running asyncio loop, delegating to as_asyncio().
        """
        ...

class WouldBlockRuntime(RuntimeError):
    """
    Raised when a synchronous API refuses to enter or block on Tokio from an
    existing Tokio runtime context.

    Two raisers today: ``Handle.get()``, and a fresh root-client bootstrap
    (``context()`` with no actor context and no initialized client, or
    ``attach()``). Reusing an already-initialized client does not block and so
    does not raise.
    """

    ...
