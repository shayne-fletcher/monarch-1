# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-process witnesses for ``monarch._src.actor.returns_future`` (RF-*)."""

from __future__ import annotations

import asyncio
import contextvars
import faulthandler
import gc
import threading
import time
import unittest
import unittest.mock
import warnings
import weakref
from typing import Any

from monarch._rust_bindings.monarch_hyperactor.handle import (
    _new_handle_pair,
    WouldBlockRuntime,
)
from monarch._src.actor import (
    actor_mesh as actor_mesh_mod,
    future as future_mod,
    returns_future as returns_future_mod,
)
from monarch._src.actor.returns_future import returns_future
from monarch._src.actor.sync_state import fake_sync_state, in_fake_sync_state

_DEADLINE_S = 10.0
# Longer than any test takes; a test still running then is hung.
_TEST_TIMEOUT_S = 60.0

# Invariant coverage:
# RF-1: test_body_runs_in_the_call_time_context,
#       test_failed_call_preparation_closes_the_body
# RF-2: test_first_observer_owns_the_body
# RF-3: test_await_runs_on_the_awaiting_loop, test_started_task_is_held
# RF-4: test_get_behaves_like_asyncio_run, test_executor_outlives_a_call,
#       test_thread_exit_shuts_its_loop_down, test_get_timeout_cancels_the_body,
#       test_get_interrupt_cancels_the_body
# RF-5: test_outcome_is_claimed_before_cleanup
# RF-6: test_other_observers_wait_on_the_handle
# RF-8: test_inherited_loop_is_replaced_not_closed
# RF-7: test_get_inside_a_running_loop_raises,
#       test_fake_sync_state_nesting_is_tracked


def _bound(test: unittest.TestCase) -> None:
    """End the process, dumping every thread's stack, if `test` hangs."""
    faulthandler.dump_traceback_later(_TEST_TIMEOUT_S, exit=True)
    test.addCleanup(faulthandler.cancel_dump_traceback_later)


def setUpModule() -> None:
    # Bootstrap the client once, outside any event loop, before any test's
    # first call to a decorated function.
    from monarch._src.actor.actor_mesh import context

    context()


@returns_future
async def _value(value: object) -> object:
    return value


@returns_future
async def _loop_of_body() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def _await(future: Any) -> Any:
    return await future


def _run(coro: Any) -> Any:
    """Run `coro` on a new loop without asyncio.run's SIGINT handling."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class ReturnsFutureTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        _bound(self)

    # RF-1: the body sees call-time context; its own writes do not leak back.
    def test_body_runs_in_the_call_time_context(self) -> None:
        var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "var", default="unset"
        )

        @returns_future
        async def read_and_write() -> str:
            seen = var.get()
            var.set("body")
            return seen

        token = var.set("caller")
        try:
            future = read_and_write()
            var.set("after the call")
            self.assertEqual(future.get(), "caller")
            self.assertEqual(var.get(), "after the call")
        finally:
            var.reset(token)

    # RF-1: a failure preparing the call closes the body unstarted.
    def test_failed_call_preparation_closes_the_body(self) -> None:
        ran: list[int] = []

        @returns_future
        async def body() -> None:
            ran.append(1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with unittest.mock.patch.object(
                actor_mesh_mod, "context", side_effect=RuntimeError("no context")
            ):
                with self.assertRaisesRegex(RuntimeError, "no context"):
                    body()
            gc.collect()
        self.assertEqual(ran, [])
        self.assertFalse([w for w in caught if "never awaited" in str(w.message)])

    # RF-2: the body runs once, for its first observer; a Future dropped
    # unobserved runs none of it.
    def test_first_observer_owns_the_body(self) -> None:
        runs: list[int] = []

        @returns_future
        async def count() -> int:
            runs.append(1)
            return len(runs)

        future = count()
        self.assertEqual(future.get(), 1)
        self.assertEqual(future.get(), 1)
        self.assertEqual(_run(_await(future)), 1)
        self.assertEqual(runs, [1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            count()
            gc.collect()
        self.assertEqual(runs, [1])
        self.assertFalse([w for w in caught if "never awaited" in str(w.message)])

    # RF-4: `.get()` runs the body on the calling thread's loop and cancels its
    # leftover tasks before returning; the loop is reused, and closed when the
    # thread exits.
    def test_get_behaves_like_asyncio_run(self) -> None:
        leftover: list[str] = []

        @returns_future
        async def leaves_a_task() -> tuple[int, asyncio.AbstractEventLoop]:
            async def forever() -> None:
                try:
                    await asyncio.sleep(_DEADLINE_S)
                except asyncio.CancelledError:
                    leftover.append("cancelled")
                    raise

            task = asyncio.get_running_loop().create_task(forever())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            return threading.get_ident(), asyncio.get_running_loop()

        ident, loop = leaves_a_task().get()
        self.assertEqual(ident, threading.get_ident())
        self.assertEqual(leftover, ["cancelled"])
        self.assertFalse(loop.is_closed())
        self.assertEqual(asyncio.all_tasks(loop), set())
        self.assertIs(_loop_of_body().get(), loop)

        # A loop that cannot be made fails every observer with the same error
        # and closes the unstarted body.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failing = _loop_of_body()
            with unittest.mock.patch.object(
                returns_future_mod, "_loop", side_effect=OSError("no loop")
            ):
                with self.assertRaises(OSError) as first:
                    failing.get()
            with self.assertRaises(OSError) as later:
                failing.get()
            self.assertIs(later.exception, first.exception)
            del failing
            gc.collect()
        self.assertFalse([w for w in caught if "never awaited" in str(w.message)])

        box: dict[str, Any] = {}
        worker = threading.Thread(
            target=lambda: box.update(value=leaves_a_task().get()), daemon=True
        )
        worker.start()
        worker.join(_DEADLINE_S)
        worker_ident, worker_loop = box.pop("value")
        self.assertEqual(worker_ident, worker.ident)
        self.assertIsNot(worker_loop, loop)
        deadline = time.monotonic() + _DEADLINE_S
        while not worker_loop.is_closed() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(worker_loop.is_closed())

    # RF-4: the loop's default executor stays usable on a later call.
    def test_executor_outlives_a_call(self) -> None:
        @returns_future
        async def offload() -> int:
            return await asyncio.to_thread(threading.get_ident)

        self.assertNotEqual(offload().get(), threading.get_ident())
        self.assertNotEqual(offload().get(), threading.get_ident())

    # RF-4: when a thread exits, its loop closes the async generators a body
    # left open, shuts its executor down, and closes.
    def test_thread_exit_shuts_its_loop_down(self) -> None:
        closed: list[str] = []
        kept: list[Any] = []

        async def numbers() -> Any:
            try:
                yield 1
                yield 2
            finally:
                closed.append("generator")

        @returns_future
        async def leaves_a_generator() -> asyncio.AbstractEventLoop:
            generator = numbers()
            kept.append(generator)
            await generator.__anext__()
            await asyncio.to_thread(int)
            return asyncio.get_running_loop()

        box: dict[str, Any] = {}
        worker = threading.Thread(
            target=lambda: box.update(loop=leaves_a_generator().get()), daemon=True
        )
        worker.start()
        worker.join(_DEADLINE_S)
        loop = box["loop"]
        deadline = time.monotonic() + _DEADLINE_S
        while not loop.is_closed() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(loop.is_closed())
        self.assertEqual(closed, ["generator"])

    # RF-8: a loop cached by another process is replaced and kept open.
    def test_inherited_loop_is_replaced_not_closed(self) -> None:
        first = weakref.ref(_loop_of_body().get())
        returns_future_mod._THIS_THREAD.loop.pid = -1
        second = _loop_of_body().get()
        gc.collect()
        inherited = first()
        assert inherited is not None, "the inherited loop should be kept"
        self.assertIsNot(second, inherited)
        self.assertFalse(inherited.is_closed())
        returns_future_mod._INHERITED.remove(inherited)
        inherited.close()

    # RF-4, RF-5: a timeout cancels the body and runs its cleanup before `.get()`
    # raises; the Future then fails with TimeoutError for every observer.
    def test_get_timeout_cancels_the_body(self) -> None:
        cleaned: list[int] = []

        @returns_future
        async def blocks() -> None:
            try:
                await asyncio.sleep(_DEADLINE_S)
            finally:
                cleaned.append(1)

        future = blocks()
        with self.assertRaises(TimeoutError):
            future.get(timeout=0.05)
        self.assertEqual(cleaned, [1])
        with self.assertRaises(TimeoutError):
            future.get()

    # RF-4, RF-5: an interrupt cancels the body and runs its cleanup before
    # `.get()` re-raises it; the Future then fails with CancelledError.
    def test_get_interrupt_cancels_the_body(self) -> None:
        cleaned: list[int] = []

        def interrupt() -> None:
            raise KeyboardInterrupt

        @returns_future
        async def interrupted() -> None:
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.sleep(_DEADLINE_S)
            finally:
                cleaned.append(1)

        future = interrupted()
        with self.assertRaises(KeyboardInterrupt):
            future.get()
        self.assertEqual(cleaned, [1])
        with self.assertRaises(asyncio.CancelledError):
            future.get()

        # A body that returns while handling its cancellation does not replace
        # the interrupt's outcome.
        @returns_future
        async def recovers() -> str:
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.sleep(_DEADLINE_S)
            except asyncio.CancelledError:
                return "recovered"
            return "finished"

        recovering = recovers()
        with self.assertRaises(KeyboardInterrupt):
            recovering.get()
        with self.assertRaises(asyncio.CancelledError):
            recovering.get()

        # So does an interrupt while the loop is being obtained.
        setup = _value(1)
        with unittest.mock.patch.object(
            returns_future_mod, "_loop", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                setup.get()
        with self.assertRaises(asyncio.CancelledError):
            setup.get()

    # RF-5: a value the body returns while handling its cancellation, and
    # native work that finishes afterwards, do not replace the timeout.
    def test_outcome_is_claimed_before_cleanup(self) -> None:
        handle, completer = _new_handle_pair()

        @returns_future
        async def recovers() -> str:
            try:
                await handle
            except asyncio.CancelledError:
                return "recovered"
            return "finished"

        future = recovers()
        with self.assertRaises(TimeoutError):
            future.get(timeout=0.05)
        completer.set_result(None)
        with self.assertRaises(TimeoutError):
            future.get()

    # RF-6: observers after the owner wait on the Handle; their timeouts and
    # cancellations leave the body alone.
    def test_other_observers_wait_on_the_handle(self) -> None:
        started, gate = threading.Event(), threading.Event()

        @returns_future
        async def waits() -> int:
            started.set()
            while not gate.is_set():
                await asyncio.sleep(0.005)
            return 9

        future = waits()
        box: dict[str, Any] = {}
        owner = threading.Thread(
            target=lambda: box.update(value=future.get()), daemon=True
        )
        owner.start()
        self.assertTrue(started.wait(_DEADLINE_S))
        with self.assertRaises(TimeoutError):
            future.get(timeout=0.05)
        with self.assertRaises(asyncio.TimeoutError):
            _run(asyncio.wait_for(_await(future), 0.05))
        gate.set()
        owner.join(_DEADLINE_S)
        self.assertEqual(box["value"], 9)
        self.assertEqual(future.get(), 9)
        self.assertEqual(_run(_await(future)), 9)
        with self.assertRaises(ValueError):
            future._as_handle()

    # RF-7: `.get()` inside a running loop, visible or hidden, raises before
    # observing the body.
    def test_get_inside_a_running_loop_raises(self) -> None:
        runs: list[int] = []

        @returns_future
        async def body() -> int:
            runs.append(1)
            return 5

        future = body()

        async def main() -> None:
            with self.assertRaises(WouldBlockRuntime):
                future.get()
            with fake_sync_state():
                with self.assertRaises(WouldBlockRuntime):
                    future.get()

        _run(main())
        self.assertEqual(runs, [])
        self.assertEqual(future.get(), 5)

    # RF-7: `in_fake_sync_state()` tracks nested `fake_sync_state()` contexts.
    def test_fake_sync_state_nesting_is_tracked(self) -> None:
        async def main() -> None:
            loop = asyncio.get_running_loop()
            self.assertFalse(in_fake_sync_state())
            with fake_sync_state():
                self.assertIsNone(asyncio.events._get_running_loop())
                self.assertTrue(in_fake_sync_state())
                with fake_sync_state():
                    self.assertTrue(in_fake_sync_state())
                self.assertTrue(in_fake_sync_state())
            self.assertFalse(in_fake_sync_state())
            self.assertIs(asyncio.events._get_running_loop(), loop)

        _run(main())

    # A Tokio caller and an invalid timeout are refused before observing.
    def test_refusals_leave_the_body_unstarted(self) -> None:
        future = _value(4)
        with unittest.mock.patch.object(
            future_mod, "_is_in_tokio_runtime", return_value=True
        ):
            with self.assertRaises(WouldBlockRuntime):
                future.get()
            with self.assertRaises(RuntimeError):
                future.__await__()
        for bad in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                future.get(bad)
        self.assertEqual(future.get(), 4)


class ReturnsFutureAsyncTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        _bound(self)

    # RF-3: the first `await` runs the body as a task on its loop and keeps it
    # alive; cancelling an observer cancels only that observer.
    async def test_await_runs_on_the_awaiting_loop(self) -> None:
        self.assertIs(await _loop_of_body(), asyncio.get_running_loop())

        handle, completer = _new_handle_pair()
        finished: list[int] = []

        @returns_future
        async def waits() -> int:
            await handle
            finished.append(1)
            return 7

        future = waits()
        first = future.as_asyncio()
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        del first
        gc.collect()
        completer.set_result(None)
        self.assertEqual(await future, 7)
        self.assertEqual(finished, [1])

    # RF-3: a started body task is held while nothing else refers to it.
    async def test_started_task_is_held(self) -> None:
        tasks: list[weakref.ref[asyncio.Task[Any]]] = []

        @returns_future
        async def parks() -> None:
            task = asyncio.current_task()
            assert task is not None
            tasks.append(weakref.ref(task))
            await asyncio.get_running_loop().create_future()

        observer = parks().as_asyncio()
        await asyncio.sleep(0)
        observer.cancel()
        del observer
        gc.collect()
        self.assertIsNotNone(tasks[0]())
