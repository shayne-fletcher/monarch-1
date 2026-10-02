# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an ``async def`` as a Monarch ``Future`` on the caller's own event loop.

``@returns_future`` turns an ``async def`` into a function that returns a
``monarch.actor.Future``. Synchronous callers use ``.get()`` and asyncio
callers use ``await``, so one implementation supports both kinds of caller.

The body runs where it is first observed:

* ``await`` or ``as_asyncio()`` runs it as a task on the awaiting loop;
* ``.get()`` from synchronous code runs it on the calling thread's own event
  loop until it finishes, as ``asyncio.run`` does; this creates no OS threads.

``.get()`` made while the calling thread is already inside a running event
loop, as in a synchronous endpoint, runs the body on a short-lived helper
thread instead of running a second loop inside it (RF-7).

Calling the function does not start the body (RF-1, RF-2). Side effects
intended to occur when the API is called, such as submitting or broadcasting
work, should therefore run in a synchronous wrapper before it calls the
decorated function, so they are committed even if the returned Future is never
observed::

    def accumulate(self, *args):
        return self._fold(self._endpoint.stream(*args))  # broadcasts now

    @returns_future
    async def _fold(self, results):
        ...  # folds when first observed

Invariants. Enforcement sites and tests cite them, and
``test_returns_future.py`` maps each to its witnesses.

- **RF-1 (call-time preparation).** Calling a decorated function runs
  Monarch's ``context()`` on the caller's thread and copies the caller's
  context variables; if either fails, the coroutine is closed unstarted. The
  first call in a process therefore bootstraps the client outside any loop
  this module creates. The body runs in that copy, so it sees the caller's
  values and its own writes do not leak back. Enforced by the wrapper in
  ``returns_future`` and ``_start``.
- **RF-2 (the first observer owns the body).** The first observation takes the
  body and starts it; the body runs at most once. A Future dropped before it is
  observed closes its body unstarted, running none of its code. Enforced by
  ``_BodyCell``.
- **RF-3 (``await`` runs on the awaiting loop).** An owning ``await`` or
  ``as_asyncio()`` runs the body as a task on that loop. Every ``await``,
  including that one, observes the body through its Handle, so cancelling an
  observer cancels only that observer and the body continues. A started task is
  held until it finishes, since asyncio holds tasks only weakly. Enforced by
  ``_BodyCell.as_asyncio`` and ``_start``.
- **RF-4 (``.get()`` behaves like ``asyncio.run``).** An owning ``.get()``
  runs the body on the calling thread's loop until it finishes. A timeout or
  interrupt cancels the body and drives its cleanup, then raises
  ``TimeoutError`` or re-raises the interrupt. Only an interrupt that arrives
  while ``.get()`` waits on the loop is handled; one that lands between its own
  setup or publication steps can leave the Future unsettled. The deadline
  starts the cancellation but does not bound its cleanup. Before returning,
  ``.get()`` cancels the tasks left running on the loop and drives them to
  completion; a task started during that cleanup survives to the next call. A
  loop whose cleanup is interrupted is dropped instead of reused. The loop's
  async generators and default executor stay usable across calls. When the
  thread exits, the loop shuts them down on that thread, then closes.
  Enforced by ``_BodyCell.get``, ``_loop``, ``_cancel_leftovers`` and
  ``_ThreadLoop``.
- **RF-5 (one terminal outcome).** The first outcome published wins and every
  observer receives it. A timeout or interrupt publishes ``TimeoutError`` or
  ``CancelledError`` before cancelling the body, so whatever the body returns
  or raises during cleanup cannot replace it. A body cancelled by its own loop,
  as when that loop shuts down, publishes ``CancelledError``. Nothing resumes.
  Native work the body already submitted may finish on its own, but cannot
  change the outcome. Enforced by ``_Settler``.
- **RF-6 (other observers wait on the Handle).** Every observer after the
  owner waits on the body's Handle, from any thread or loop; its own timeout or
  cancellation leaves the body alone. Only an owning ``.get()`` cancels the
  body (RF-4). ``Future._as_handle()`` raises, since
  waiting on the Handle of a body nobody has started would never finish.
  Enforced by ``_BodyCell.get``, ``_BodyCell.as_asyncio`` and
  ``Future._as_handle``.
- **RF-7 (``.get()`` inside a running loop).** An owning ``.get()`` made while
  the calling thread is inside a running event loop, whether asyncio reports it
  or ``fake_sync_state()`` hides it, runs the body under RF-4 on a short-lived
  daemon thread with a loop of its own, and waits for that thread, including
  its loop's shutdown as ``asyncio.run`` does. A ``.get()`` made by code a
  ``.get()`` is driving, and one that would wait for a body on a loop the
  calling thread is running, raise ``WouldBlockRuntime`` instead, unless that
  body has already settled. An interrupt that lands while the helper thread is
  starting fails the Future with ``CancelledError`` and is re-raised at once;
  the body may run on to completion on the helper. This bridges synchronous
  endpoints, which still run on their actor's loop. Enforced by
  ``_BodyCell.get`` and ``_BodyCell._drive_on_helper``.
- **RF-8 (forked children).** A Future inherited through ``fork()`` is not
  supported in the child. The child makes its own loop and keeps every loop it
  inherited open for as long as it runs: closing one would unregister the
  parent's descriptors from the epoll instance they share, and the parent's
  loop would stop waking. A child that exits through normal interpreter
  shutdown can still close them, and is not supported; one that ends with
  ``os._exit()``, as multiprocessing workers do, is. Enforced by ``_loop`` and
  ``_ThreadLoop``.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from monarch._rust_bindings.monarch_hyperactor.handle import (
    _HandleCompleter,
    _new_handle_pair,
    Handle,
    WouldBlockRuntime,
)
from monarch._src.actor.future import _GET_ON_LOOP_WARNING, Future
from monarch._src.actor.sync_state import in_fake_sync_state

P = ParamSpec("P")
T = TypeVar("T")

# RF-3: every started body task, until it finishes.
_running: set[asyncio.Task[Any]] = set()

# RF-8: the loops a forked child inherited, never closed.
_INHERITED: list[asyncio.AbstractEventLoop] = []


class _Settler:
    """Publishes one body's outcome to its Handle, first-wins (RF-5)."""

    def __init__(self, completer: _HandleCompleter[Any]) -> None:
        self.completer: _HandleCompleter[Any] = completer
        # Guards `settled`; publishers can race across threads.
        self.lock: threading.Lock = threading.Lock()
        self.settled: bool = False

    def publish(self, exc: BaseException | None, value: Any = None) -> bool:
        """Publish `exc`, or `value` if `exc` is `None`, unless an outcome was
        already published. Return whether this call published."""
        with self.lock:
            if self.settled:
                return False
            self.settled = True
        if exc is None:
            self.completer.set_result(value)
        else:
            self.completer.set_exception(exc)
        return True

    def publish_task(self, task: asyncio.Task[Any]) -> None:
        """Publish the outcome of the body's finished task."""
        if task.cancelled():
            self.publish(asyncio.CancelledError())
            return
        exc = task.exception()
        self.publish(exc, None if exc is not None else task.result())


class _BodyCell:
    """The state behind a Future returned by a ``@returns_future`` function."""

    def __init__(
        self, coro: Coroutine[Any, Any, Any], ctx: contextvars.Context
    ) -> None:
        # Guards `unstarted`.
        self.lock: threading.Lock = threading.Lock()
        # The body and the caller's context copy, until the first observer
        # takes them (RF-2).
        self.unstarted: tuple[Coroutine[Any, Any, Any], contextvars.Context] | None = (
            coro,
            ctx,
        )
        handle, completer = _new_handle_pair()
        # Every observer reads the outcome from `handle` (RF-6).
        self.handle: Handle[Any] = handle
        self.settler: _Settler = _Settler(completer)
        # The loop the body runs on, once started.
        self.loop: asyncio.AbstractEventLoop | None = None

    def take(self) -> tuple[Coroutine[Any, Any, Any], contextvars.Context] | None:
        """Take the body if this is its first observation (RF-2)."""
        with self.lock:
            taken, self.unstarted = self.unstarted, None
        return taken

    def __del__(self) -> None:
        # RF-2: closing an unstarted coroutine runs none of its code and
        # suppresses the "never awaited" warning.
        if self.unstarted is not None:
            self.unstarted[0].close()

    def get(self, timeout: float | None) -> Any:
        """Synchronously observe a ``@returns_future`` self.

        ``Future.get()`` has already rejected a Tokio caller and validated
        ``timeout``.
        """
        hidden = in_fake_sync_state()
        visible = asyncio.events._get_running_loop() is not None
        if (hidden or visible) and getattr(_THIS_THREAD, "driving", False):
            raise WouldBlockRuntime(
                "Future.get() cannot block inside a returns_future body; await the "
                "Future instead"
            )
        if visible and self.unstarted is not None:
            # Before take(): a warning raised as an error must leave the body unclaimed.
            warnings.warn(_GET_ON_LOOP_WARNING, UserWarning, stacklevel=3)
        taken = self.take()
        if taken is None:
            loop = self.loop
            if (
                loop is not None
                and getattr(loop, "_thread_id", None) == threading.get_ident()
                and not self.settler.settled
            ):
                # RF-7: blocking here would stop the loop the body needs.
                raise WouldBlockRuntime(
                    "Future.get() cannot block on an event loop this thread is "
                    "running; await the Future instead"
                )
            # RF-6: a timed-out wait raises without affecting the self.
            return self.handle.get(timeout)
        if hidden or visible:
            return self._drive_on_helper(taken, timeout)
        return self._drive(taken, timeout)

    def as_asyncio(self, loop: asyncio.AbstractEventLoop) -> asyncio.Future[Any]:
        """Return an asyncio Future on the running `loop` that observes this self,
        starting the body on `loop` if this is its first observation (RF-3, RF-6)."""
        taken = self.take()
        if taken is not None:
            _start(self, taken, loop)
        return self.handle.as_asyncio()

    def _drive(
        self,
        taken: tuple[Coroutine[Any, Any, Any], contextvars.Context],
        timeout: float | None,
    ) -> Any:
        """Run the body on this thread's loop, as ``asyncio.run`` does (RF-4)."""
        loop = _owning_loop(self, taken, _loop)
        _THIS_THREAD.driving = True
        try:
            task = _start(self, taken, loop)
            try:
                done, _ = loop.run_until_complete(asyncio.wait({task}, timeout=timeout))
            except BaseException:
                _cancel(loop, task, self.settler, asyncio.CancelledError())
                raise
            if not done:
                error = TimeoutError("returns_future operation did not finish in time")
                _cancel(loop, task, self.settler, error)
                raise error
            return task.result()
        finally:
            _THIS_THREAD.driving = False
            try:
                _cancel_leftovers(loop)
            except BaseException:
                _THIS_THREAD.loop = None
                raise

    def _drive_on_helper(
        self,
        taken: tuple[Coroutine[Any, Any, Any], contextvars.Context],
        timeout: float | None,
    ) -> Any:
        """Run the body as `_drive` does, on a short-lived daemon thread with a loop
        of its own, and wait for that thread (RF-7)."""
        started = threading.Event()
        # Set once the helper has closed its loop. Waited on instead of
        # `Thread.join()`, which before Python 3.13 returns at once after an
        # interrupted join while the thread still runs.
        finished = threading.Event()
        loop = _owning_loop(self, taken, asyncio.new_event_loop)
        box: dict[str, Any] = {"loop": loop}

        def run() -> None:
            _THIS_THREAD.driving = True
            try:
                box["task"] = _start(self, taken, loop)
                started.set()
                try:
                    loop.run_until_complete(asyncio.wait({box["task"]}))
                except BaseException:
                    _cancel(loop, box["task"], self.settler, asyncio.CancelledError())
                    raise
            except BaseException as exc:  # noqa: B036 - re-raised by the caller
                box.setdefault("error", exc)
            finally:
                started.set()
                try:
                    _close(loop)
                finally:
                    finished.set()

        helper = threading.Thread(target=run, name="returns_future", daemon=True)
        try:
            helper.start()
        except RuntimeError as exc:
            # The thread could not start, so the body never ran.
            loop.close()
            taken[0].close()
            self.settler.publish(exc)
            raise
        except BaseException as exc:
            # Interrupted while the thread starts; it may already run the body.
            self.settler.publish(_shared(exc))
            raise
        try:
            finished.wait(timeout)
        except BaseException:
            self._cancel_on_helper(finished, started, box, asyncio.CancelledError())
            raise
        if not finished.is_set():
            error = TimeoutError("returns_future operation did not finish in time")
            if self._cancel_on_helper(finished, started, box, error):
                raise error
        if "error" in box:
            raise box["error"]
        return box["task"].result()

    def _cancel_on_helper(
        self,
        finished: threading.Event,
        started: threading.Event,
        box: dict[str, Any],
        outcome: BaseException,
    ) -> bool:
        """Publish `outcome`, then cancel the helper's body and wait for the helper
        to finish (RF-4, RF-5). Return whether `outcome` won; if the body finished
        first, its own outcome stands."""
        won = self.settler.publish(outcome)
        if won:
            started.wait()
            task = box.get("task")
            if task is not None:
                try:
                    box["loop"].call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # The loop already closed: the body has finished.
                    pass
        finished.wait()
        return won


def _shared(exc: BaseException) -> BaseException:
    """The outcome other observers get when starting the body fails: the error
    itself, or `CancelledError` for an interrupt (RF-5)."""
    return exc if isinstance(exc, Exception) else asyncio.CancelledError()


def _start(
    body: _BodyCell,
    taken: tuple[Coroutine[Any, Any, Any], contextvars.Context],
    loop: asyncio.AbstractEventLoop,
) -> asyncio.Task[Any]:
    """Start the body as a task on `loop`, publishing its outcome when it
    finishes. A failure before the task exists is published through
    `_shared`, so other observers raise it too."""
    coro, ctx = taken
    body.loop = loop
    try:
        # `ctx.run`, because not every supported Python has
        # `create_task(context=)` (RF-1).
        task = ctx.run(loop.create_task, coro)
    except BaseException as exc:
        coro.close()
        body.settler.publish(_shared(exc))
        raise
    task.add_done_callback(body.settler.publish_task)
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


class _ThreadLoop:
    """A thread's own loop, reused by each ``.get()`` on that thread (RF-4)."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self.pid: int = os.getpid()

    def __del__(self) -> None:
        # Runs on the thread as it exits. A forked child keeps the loop it
        # inherited (RF-8). At interpreter shutdown, Python has already joined
        # the executor's threads and cannot start the one shutting the executor
        # down needs, so the loop is only closed.
        if getattr(self, "pid", None) != os.getpid():
            _INHERITED.append(self.loop)
            return
        try:
            if not sys.is_finalizing():
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
        finally:
            self.loop.close()


# Each thread's `_ThreadLoop`, and whether it is driving a body under `.get()`
# (RF-7).
_THIS_THREAD = threading.local()


def _loop() -> asyncio.AbstractEventLoop:
    """This thread's loop, made on first use and again in a forked child."""
    current = getattr(_THIS_THREAD, "loop", None)
    if current is None or current.pid != os.getpid() or current.loop.is_closed():
        current = _ThreadLoop()
        _THIS_THREAD.loop = current
    return current.loop


def _owning_loop(
    body: _BodyCell,
    taken: tuple[Coroutine[Any, Any, Any], contextvars.Context],
    make: Callable[[], asyncio.AbstractEventLoop],
) -> asyncio.AbstractEventLoop:
    """The loop for the body's owning `.get()`. If it cannot be had, close the
    unstarted body and publish the failure through `_shared`."""
    try:
        return make()
    except BaseException as exc:
        taken[0].close()
        body.settler.publish(_shared(exc))
        raise


def _cancel_leftovers(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel the tasks the body left running and drive them to completion, as
    ``asyncio.run`` does (RF-4)."""
    leftovers = asyncio.all_tasks(loop)
    for task in leftovers:
        task.cancel()
    if leftovers:
        loop.run_until_complete(asyncio.wait(leftovers))
    for task in leftovers:
        if not task.cancelled() and task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during returns_future shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _close(loop: asyncio.AbstractEventLoop) -> None:
    """Shut a helper's loop down as ``asyncio.run`` does (RF-7)."""
    try:
        _cancel_leftovers(loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()


def _cancel(
    loop: asyncio.AbstractEventLoop,
    task: asyncio.Task[Any],
    settler: _Settler,
    outcome: BaseException,
) -> None:
    """Publish `outcome`, then cancel the body and drive its cleanup (RF-4,
    RF-5)."""
    settler.publish(outcome)
    task.cancel()
    loop.run_until_complete(asyncio.wait({task}))


def returns_future(
    function: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Future[T]]:
    """Make an async function return a ``monarch.actor.Future``.

    Calling the decorated function is an ordinary synchronous call. It returns
    a Monarch Future rather than a Python coroutine. The body is still an
    ordinary asyncio coroutine: it can use normal Python code and await asyncio
    awaitables, Monarch Futures, and Handles.

    Calling the function does not start the body: it only runs Monarch's
    ``context()`` and copies the caller's context variables (RF-1). The first
    ``await`` runs the body on the caller's asyncio loop; cancelling an
    ``await`` cancels only that observer. The first ``.get()`` runs the body on
    the calling thread's loop, as ``asyncio.run`` does; if that ``.get()`` times
    out or is interrupted, the body is cancelled and the Future fails with
    ``TimeoutError`` or ``CancelledError``. Later observers share the outcome.
    ``.get()`` inside a running event loop runs the body the same way on a
    short-lived helper thread; inside a body that a ``.get()`` is driving it
    raises ``WouldBlockRuntime``, so await the Future there.

    If an API must perform a side effect when called, do that in a synchronous
    wrapper before calling the decorated function. For class or static
    methods, place ``@classmethod`` or ``@staticmethod`` above
    ``@returns_future``.
    """
    if not inspect.iscoroutinefunction(function):
        raise TypeError("returns_future requires an async function")

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        coro = function(*args, **kwargs)
        try:
            # Imported here, not at module level, so that actor_mesh can import
            # this module.
            from monarch._src.actor.actor_mesh import context

            context()
            ctx = contextvars.copy_context()
            body = _BodyCell(coro, ctx)
        except BaseException:
            coro.close()
            raise
        return Future._from_body(body)

    return wrapped
