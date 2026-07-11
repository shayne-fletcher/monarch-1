# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import warnings
from typing import (
    Any,
    cast,
    Coroutine,
    Generator,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
)

from monarch._rust_bindings.monarch_hyperactor.pytokio import (
    Handle,
    is_tokio_thread,
    PythonTask,
    Shared,
)
from monarch._src.actor.telemetry import log_with_tracing

R = TypeVar("R")


async def _aincomplete(impl: Any, self: Any) -> Any:
    try:
        return self._set_result(await impl())
    except Exception as e:
        self._set_exception(e)
        raise


# Future is our generic mechanism for providing both a synchronous and asynchronous API for
# Monarch Future objects.

# We treat all code as running in one of two contexts: synchronous (asyncio._get_running_loop() is None)
# or asynchronous.

# Inside of asynchronous code, clients of our API must use `await` to wait for monarch Futures to prevent
# blocking the surrounding event loop.

# In synchronous code users must call get() because the call is comming from an non-async function so
# await is not allowed.

# [avoiding async code duplication]
# Because we allow for two modes, it is tempting as developers of Monarch to start to write two copies of
# of code for each mode. However, this results in a lot of confusing code duplication.
# To avoid this, we utilize the fact that synchronous code is allowed to start/complete an asyncio event loop
# via asyncio.run in order to complete the `get()` operation. So we can just write the async version and use
# it to implement the synchronoous version.

# However, starting and running an event loop is somewhat expensive. For simple messages, using an event loop
# is about 4x slower than just directly waiting on the tokio result. To avoid this slow down we perform an
# optimization. For any case where the `impl` coroutine of a future calls `await` only on PythonFuture
# (a Tokio future returning a Python value) objects, we pass requires_loop=False to the Future. In this mode,
# the future will just run the coroutine manually, and the PythonFuture object will recognize it is being awaited
# without an event loop (search [avoiding code duplication]) and simply do a blocking wait. By avoiding the event
# loop machinery, this gives it the same throughput as if we ran it synchronously.


class _Unawaited(NamedTuple):
    coro: PythonTask[Any]


class _Complete(NamedTuple):
    value: Any


class _Exception(NamedTuple):
    exe: Exception


class _Handle(NamedTuple):
    handle: Handle[Any]


class _Tokio(NamedTuple):
    shared: Shared[Any]


class _Taken(NamedTuple):
    pass


_Status = _Unawaited | _Complete | _Exception | _Handle | _Tokio | _Taken


# The `_Tokio` state means the task was bridged to a pytokio `Shared` on a
# tokio-thread await; `get()` and `as_asyncio()` both refuse it because that
# value is only awaitable from a PythonTask coroutine.
_ALREADY_TOKIO_MSG = (
    "already converted into a pytokio.Shared object, use 'await' from a "
    "PythonTask coroutine to get the value."
)


class Future(Generic[R]):
    """
    The Future class wraps a PythonTask, which is a handle to a asyncio coroutine running on the Tokio event loop.
    These coroutines do not use asyncio or asyncio.Future; instead, they are executed directly on the Tokio runtime.
    The Future class provides both synchronous (.get()) and asynchronous APIs (await) for interacting with these tasks.

    Args:
        coro (Coroutine[Any, Any, R] | PythonTask[R]): The coroutine or PythonTask representing
            the asynchronous computation.

    """

    def __init__(self, *, coro: "Coroutine[Any, Any, R] | PythonTask[R]") -> None:
        self._status: _Status = _Unawaited(
            coro if isinstance(coro, PythonTask) else PythonTask.from_coroutine(coro)
        )

    def _take_inner(self) -> "PythonTask[R]":
        """Surrender the underlying ``PythonTask`` to a caller that needs to drive
        it directly (for example to await the raw Rust future without the GIL).
        Only valid on a Future that has not yet been awaited or resolved; the
        Future is spent afterward (a second take, or any get()/await, raises).
        """
        match self._status:
            case _Unawaited(coro=coro):
                self._status = _Taken()
                return cast("PythonTask[R]", coro)
            case _:
                raise ValueError("Future has already been awaited or resolved.")

    def get(self, timeout: Optional[float] = None) -> R:
        """Get the result of the Future.

        Caveats:

        This method is designed to be used in places where event loops are not available. Besides that, you should
        avoid using this method if possible. Instead, use `as_asyncio()` (or `await`). This is because when Future.get() is called from
        within an active event loop, it blocks synchronously and does not yield control. That may degrade performance
        by preventing other tasks from running, and can potentially cause deadlocks if this future depends on them.

        A `timeout` never consumes the Future: on `TimeoutError` the underlying task keeps running, so a later `get()`/`await` still observes its result.

        examples:

        This is not recommended because `fut.get()` blocks the event loop and might lead to issues explained above.
        ```
        def inner_func(fut):
            result = fut.get()
            # ...

        async def out_func(fut):
            inner_func(fut)
        ```

        This is okay because everything is running synchronously.
        ```
        def inner_func(fut):
            result = fut.get()
            # ...

        def main():
            # ...
            inner_func(fut)
        ```
        """
        in_asyncio = asyncio._get_running_loop() is not None
        in_tokio = is_tokio_thread()
        if in_asyncio or in_tokio:
            # Forward the event to Rust tracing for every in-loop/tokio caller,
            # including non-actor driver processes where no `TracingForwarder`
            # handler is on the Python logging chain. The UserWarning (separate
            # from this trace) is emitted at most once, never doubled: the inline
            # `warnings.warn` below for the no-timeout `_Unawaited` path, else
            # `Handle.get()` for the `get(timeout=...)`/`_Handle` paths -- which
            # on a Tokio thread raise `WouldBlockRuntime` before warning, so no
            # UserWarning fires there.
            log_with_tracing(
                logging.WARNING,
                "Future.get() called from within an active event loop",
                extra={"context": "asyncio" if in_asyncio else "tokio"},
                stacklevel=2,
            )
        match self._status:
            case _Unawaited(coro=coro):
                if timeout is not None:
                    # A timeout must not destroy the Future. Observe the task
                    # through a Handle, which is non-cancelling on timeout, so a
                    # later get()/poll()/await still resolves. Handle.get()
                    # applies the context policy and emits the warning itself.
                    handle = coro.spawn_handle()
                    self._status = _Handle(handle)
                    return cast("R", handle.get(timeout))
                if in_asyncio:
                    warnings.warn(
                        "Future.get() was called from within an active event loop. Because this method blocks "
                        "synchronously and does not yield control, it may degrade performance by preventing "
                        "other tasks from running, and can potentially cause deadlocks if this future depends "
                        "on them. It is encouraged to use as_asyncio() (or await) instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif in_tokio:
                    # as_asyncio() needs an asyncio loop, which a Tokio worker
                    # thread lacks; the only in-context option is to await inside
                    # a PythonTask coroutine (block_on() below then raises).
                    warnings.warn(
                        "Future.get() was called from within a Tokio runtime; it cannot block there "
                        "and will raise. Await the Future inside a PythonTask coroutine instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                try:
                    v = coro.block_on()
                    self._status = _Complete(v)
                    return cast("R", v)
                except Exception as e:
                    self._status = _Exception(e)
                    raise e from None
            case _Handle(handle=handle):
                # Observe the shared Handle. `Handle.get()` applies its own
                # context policy (warn on a live loop, `WouldBlockRuntime` in a
                # Tokio runtime context, block on a sync thread) and emits the
                # Python warning itself, so we do not warn again here.
                return cast("R", handle.get(timeout))
            case _Complete(value=value):
                return cast("R", value)
            case _Exception(exe=exe):
                raise exe
            case _Tokio(_):
                raise ValueError(_ALREADY_TOKIO_MSG)
            case _Taken():
                raise ValueError("Future was consumed.")
            case _:
                raise RuntimeError("unknown status")

    def __await__(self) -> Generator[Any, Any, R]:
        if asyncio._get_running_loop() is not None:
            # Asyncio callers observe through the Handle; `__await__` delegates
            # to `as_asyncio()`.
            return self.as_asyncio().__await__()
        elif is_tokio_thread():
            match self._status:
                case _Unawaited(coro=coro):
                    shared = coro.spawn()
                    self._status = _Tokio(shared)
                    return shared.__await__()
                case _Tokio(shared=shared):
                    return shared.__await__()
                case _Handle(_):
                    raise ValueError(
                        "Future is backed by a Handle and is not awaitable on a tokio thread; "
                        "use get() or as_asyncio() from a sync/asyncio context."
                    )
                case _Taken():
                    raise ValueError("Future was consumed.")
                case _:
                    raise ValueError(
                        "already converted into a synchronous future, use 'get' to get the value."
                    )
        else:
            raise ValueError(
                "__await__ with no active event loop (either asyncio or tokio)"
            )

    def as_asyncio(self) -> "asyncio.Future[R]":
        """Return a standard ``asyncio.Future`` that resolves when this Future
        does.

        Requires a running event loop; off a loop it raises ``RuntimeError``
        **without** consuming the underlying task (the Future stays unawaited, so
        a later ``get()`` still drives it). Observation is non-consuming:
        repeated ``as_asyncio()``/``await`` each return a fresh loop-local future
        observing the same result.
        """
        loop = asyncio._get_running_loop()
        if loop is None:
            raise RuntimeError("as_asyncio() requires a running asyncio event loop.")
        match self._status:
            case _Unawaited(coro=coro):
                # The loop is confirmed above, so spawning is safe: an off-loop
                # call raised already, leaving the Future in `_Unawaited`.
                handle = coro.spawn_handle()
                self._status = _Handle(handle)
                return handle.as_asyncio()
            case _Handle(handle=handle):
                return handle.as_asyncio()
            case _Complete(value=value):
                done = loop.create_future()
                done.set_result(value)
                return cast("asyncio.Future[R]", done)
            case _Exception(exe=exe):
                failed = loop.create_future()
                # asyncio.Future.set_exception rejects StopIteration (PEP 479);
                # surface it as `raise StopIteration` already does out of a
                # coroutine, matching the Handle observer path.
                failed.set_exception(
                    RuntimeError("coroutine raised StopIteration")
                    if isinstance(exe, StopIteration)
                    else exe
                )
                return cast("asyncio.Future[R]", failed)
            case _Tokio(_):
                raise ValueError(_ALREADY_TOKIO_MSG)
            case _Taken():
                raise ValueError("Future was consumed.")
            case _:
                raise RuntimeError("unknown status")

    # compatibility with old tensor engine Future objects
    # hopefully we do not need done(), add_callback because
    # they are harder to implement right.
    def result(self, timeout: Optional[float] = None) -> R:
        return self.get(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        try:
            self.get(timeout)
            return None
        except Exception as e:
            return e
