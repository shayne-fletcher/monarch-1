# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator, Awaitable, Coroutine, Optional, Sequence, TypeVar

from monarch._src.job.job import DEFAULT_JOB_PATH, JobState, JobTrait

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def _silence_destroy_pending(task: "asyncio.Task[Any]") -> None:
    with contextlib.suppress(Exception):
        # pyre-ignore[16]: `_log_destroy_pending` is a private asyncio.Task attribute
        task._log_destroy_pending = False


def _drain_task_results(tasks: Sequence["asyncio.Task[Any]"]) -> None:
    for task in tasks:
        if task.done():
            # Exit-path drain: consume each done task's result/exception so it is
            # "retrieved" (silences asyncio's "exception was never retrieved").
            # Suppress BaseException deliberately -- this runs during bounded
            # Ctrl-C teardown, where a stray interrupt arriving mid-drain must not
            # turn a clean exit into a traceback. Narrowing it to Exception makes
            # a second Ctrl-C escape here and dump a stack.
            with contextlib.suppress(BaseException):
                task.result()


def _kill_quietly(job: JobTrait) -> None:
    """Best-effort ``job.kill()`` for teardown paths that must not raise.

    Suppresses BaseException on purpose: a stray Ctrl-C landing inside the
    synchronous kill (e.g. during its grace poll) must be absorbed, not allowed
    to escape the teardown context manager -- an interrupt escaping here breaks
    the manager's athrow unwinding ("generator didn't stop after athrow()") and
    leaves the process up. Callers that need the original interrupt to propagate
    re-raise it themselves after this returns.
    """
    with contextlib.suppress(BaseException):
        job.kill()


async def _await_bounded(awaitable: Awaitable[_T], timeout: float) -> _T:
    task = asyncio.ensure_future(awaitable)
    try:
        done, pending = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        if not task.done():
            task.cancel()
            _silence_destroy_pending(task)
        raise
    if pending:
        task.cancel()
        _silence_destroy_pending(task)
        raise asyncio.TimeoutError()
    _drain_task_results(list(done))
    return task.result()


async def cancel_async_tasks(
    tasks: Sequence["asyncio.Task[Any]"],
    timeout: float = 2.0,
) -> None:
    """Cancel tasks, but do not let uncooperative cancellation block exit."""
    task_set = set(tasks)
    if not task_set:
        return
    for task in task_set:
        task.cancel()
    done, pending = await asyncio.wait(task_set, timeout=timeout)
    _drain_task_results(list(done))
    for task in pending:
        _silence_destroy_pending(task)


def _cancel_loop_tasks(loop: asyncio.AbstractEventLoop, timeout: float) -> None:
    tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    done, pending = loop.run_until_complete(asyncio.wait(tasks, timeout=timeout))
    _drain_task_results(list(done))
    for task in pending:
        _silence_destroy_pending(task)


def run_async_main(
    coro: Coroutine[Any, Any, _T],
    *,
    interrupt_timeout: float = 15.0,
    cancel_timeout: float = 2.0,
) -> Optional[_T]:
    """Run an async demo main with bounded Ctrl-C and pending-task cleanup.

    ``asyncio.run`` waits for every pending task to acknowledge cancellation.
    That is reasonable for libraries, but wrong for local ProcessJob demos that
    intentionally create wedged calls: foreground exit must be bounded so the
    owning job cleanup can run and detached workers are not orphaned.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(coro)
        try:
            return loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            if main_task.done():
                raise
            main_task.cancel()
            done, pending = loop.run_until_complete(
                asyncio.wait({main_task}, timeout=interrupt_timeout)
            )
            if pending:
                _silence_destroy_pending(main_task)
                raise
            if main_task.cancelled():
                raise KeyboardInterrupt()
            return main_task.result()
        finally:
            _cancel_loop_tasks(loop, cancel_timeout)
            with contextlib.suppress(Exception):
                loop.run_until_complete(
                    asyncio.wait_for(
                        loop.shutdown_asyncgens(),
                        timeout=cancel_timeout,
                    )
                )
            with contextlib.suppress(Exception):
                loop.run_until_complete(
                    asyncio.wait_for(
                        loop.shutdown_default_executor(),
                        timeout=cancel_timeout,
                    )
                )
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@contextlib.asynccontextmanager
async def scoped_async_state(
    job: JobTrait,
    cached_path: Optional[str] = DEFAULT_JOB_PATH,
    shutdown_timeout: float = 10.0,
) -> AsyncIterator[JobState]:
    """Yield job state and always release job-owned hosts on exit.

    This is the async counterpart of the test ``scoped_state`` helper: callers
    get a bounded graceful host shutdown first, and if that cannot be proven the
    owning job is killed. This matters for local ``ProcessJob`` workers because
    they run in detached process groups and terminal Ctrl-C will not reap them.
    """
    try:
        state = job.state(cached_path=cached_path)
    except BaseException:
        _kill_quietly(job)
        raise

    try:
        yield state
    finally:
        try:
            for host_mesh in state.host_meshes():
                await _await_bounded(
                    host_mesh.shutdown(),
                    shutdown_timeout,
                )
        except asyncio.CancelledError:
            _kill_quietly(job)
            raise
        except (KeyboardInterrupt, SystemExit):
            _kill_quietly(job)
            raise
        except Exception:
            logger.warning(
                "job host shutdown did not complete; killing job",
                exc_info=True,
            )
            _kill_quietly(job)
        else:
            # Graceful shutdown released the meshes; also release the job's local
            # scratch (tmpdir / ipc sockets), which otherwise only the kill path
            # frees.
            with contextlib.suppress(Exception):
                job.cleanup()
