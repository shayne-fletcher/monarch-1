# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Witnesses for the worker-process fork contract (WF-* in bootstrap.py).

An actor endpoint forks from its live actor-loop thread, so the child is copied
mid-``run_forever`` and CPython's child prologue runs against real Monarch
state. The child then leaves at the fork site in one of three ways, each
permitted by WF-3: ``os._exit()`` directly ("exit"), ``os._exit()`` after
running a fresh loop with ``asyncio.run()`` ("asyncio"), or ``exec`` of
``/usr/bin/true`` ("exec"). After every child the parent's actor loop, default
executor and message path must still work (WF-4).

Both actor locations WF-1 names are covered: a proc started by native
``bootstrap_main()`` and an actor on the client's own proc, where CF-* in
``actor_mesh.py`` also applies. The ``_py314`` target runs the same matrix
under the other supported CPython, whose child prologue differs.
"""

import asyncio
import os
import select
import signal
import sys
import threading
import time
import unittest
from dataclasses import dataclass

from monarch.actor import Actor, endpoint, this_host, this_proc

# Written by a child only once it reaches its intended exit, so a child that
# fails earlier cannot pass.
_MARKER = b"wf-child-reached-exit"
# Distinct statuses for a child that failed before its intended exit, so a
# failure names the mode that broke.
_EXIT_BRANCH_FAILED = 101
_EXEC_BRANCH_FAILED = 102
_ASYNCIO_BRANCH_FAILED = 103
_BRANCH_FAILED = {
    "exit": _EXIT_BRANCH_FAILED,
    "asyncio": _ASYNCIO_BRANCH_FAILED,
    "exec": _EXEC_BRANCH_FAILED,
}
_CHILD_DEADLINE_SECS = 30.0
_CALL_TIMEOUT_SECS = 60.0
_ITERATIONS = 5


@dataclass(frozen=True)
class ForkResult:
    """What the forking (parent) process observed for one child."""

    # Distinguishes the native-bootstrap proc from the client process.
    parent_pid: int
    # The forking process's interpreter; the test asserts it matches its own so
    # the `_py314` target cannot silently fork from a default-interpreter proc.
    version: tuple[int, int]
    exit_code: int
    marker: bytes
    # Descriptors still open on the witness pipe after cleanup; must be zero.
    leaked_pipe_fds: int


def _fds_referencing(pipe: tuple[int, int]) -> int:
    """Count this process's descriptors open on ``pipe``, a (device, inode) pair.

    Matching by identity rather than descriptor number stays correct when another
    thread reuses a closed number. ``/dev/fd`` exists on both Linux and macOS.
    """
    count = 0
    for fd in os.listdir("/dev/fd"):
        try:
            st = os.fstat(int(fd))
        except OSError:
            # Includes the descriptor listdir used for /dev/fd, now closed.
            continue
        if (st.st_dev, st.st_ino) == pipe:
            count += 1
    return count


def _read_marker(fd: int, deadline: float) -> bytes:
    """Read the child's marker, giving up at ``deadline`` or on EOF."""
    data = b""
    while len(data) < len(_MARKER):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            break
        chunk = os.read(fd, len(_MARKER) - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _reap(pid: int, deadline: float) -> int | None:
    """Reap exactly ``pid`` by ``deadline``; return its exit code or ``None``.

    Waiting on the one PID, never ``-1``, leaves Monarch's own children alone.
    """
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    return None


def _kill_and_reap(pid: int) -> None:
    """Failure cleanup: kill and reap the child, bounded, so none is left."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if _reap(pid, time.monotonic() + _CHILD_DEADLINE_SECS) is None:
        raise RuntimeError(f"forked child {pid} survived SIGKILL")


async def _write_marker_on_fresh_loop(
    fd: int, copied_loop: asyncio.AbstractEventLoop
) -> None:
    """Child-side body for the "asyncio" mode (WF-3 permits a fresh loop)."""
    # Sleeping waits on the fresh loop's own selector. A child still on the
    # copied loop writes nothing, so the parent's marker check fails.
    await asyncio.sleep(0.01)
    if asyncio.get_running_loop() is not copied_loop:
        os.write(fd, _MARKER)


def _fork_and_reap(boundary: str) -> ForkResult:
    """Fork from the calling actor-loop thread and observe one child.

    The child takes the ``boundary`` exit (WF-3). The parent bounds every wait
    and, on any failure, closes the pipe and kills and reaps the exact child.
    """
    copied_loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    pipe_stat = os.fstat(read_fd)
    pipe = (pipe_stat.st_dev, pipe_stat.st_ino)
    try:
        pid = os.fork()
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    if pid == 0:
        # WF-3: however the branch below ends, the child leaves through
        # os._exit() and never awaits or returns into the copied loop.
        status = _BRANCH_FAILED[boundary]
        try:
            if boundary == "exit":
                os.write(write_fd, _MARKER)
                status = 0
            elif boundary == "asyncio":
                asyncio.run(_write_marker_on_fresh_loop(write_fd, copied_loop))
                status = 0
            else:
                # The one path to `true` on both Linux and macOS.
                os.execv("/usr/bin/true", ["true"])
        finally:
            os._exit(status)

    exit_code = None
    marker = b""
    try:
        os.close(write_fd)
        write_fd = -1
        deadline = time.monotonic() + _CHILD_DEADLINE_SECS
        if boundary != "exec":
            marker = _read_marker(read_fd, deadline)
        exit_code = _reap(pid, deadline)
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)
        if exit_code is None:
            _kill_and_reap(pid)
    if exit_code is None:
        raise TimeoutError(f"forked child {pid} did not exit in time")
    return ForkResult(
        parent_pid=os.getpid(),
        version=(sys.version_info.major, sys.version_info.minor),
        exit_code=exit_code,
        marker=marker,
        leaked_pipe_fds=_fds_referencing(pipe),
    )


class ForkingActor(Actor):
    """Forks from its endpoints and checks the parent afterwards (WF-4)."""

    def __init__(self) -> None:
        self._executor_prewarmed = False

    @endpoint
    async def prewarm_executor(self) -> bool:
        # asyncio creates the default executor lazily. Creating it before the
        # fork means the later round trip proves the existing executor
        # survived, not that a new one could be made.
        worker = await asyncio.get_running_loop().run_in_executor(
            None, threading.get_ident
        )
        self._executor_prewarmed = worker != threading.get_ident()
        return self._executor_prewarmed

    @endpoint
    async def fork_child(self, boundary: str) -> ForkResult:
        # Refusing without the prewarm makes its removal fail the witness
        # rather than silently weaken it.
        if not self._executor_prewarmed:
            raise RuntimeError("the default executor must exist before fork")
        return _fork_and_reap(boundary)

    @endpoint
    async def executor_round_trip(self) -> bool:
        # WF-4: the executor thread's completion wakes this loop through its
        # self-pipe, which the child shares with the parent.
        worker = await asyncio.get_running_loop().run_in_executor(
            None, threading.get_ident
        )
        return worker != threading.get_ident()

    @endpoint
    async def ping(self) -> str:
        # WF-4: the actor's message path still delivers and replies.
        return "pong"


class WorkerForkContractTest(unittest.TestCase):
    def _check_fork_matrix(self, actor: ForkingActor, in_client: bool) -> None:
        """Run every exit mode ``_ITERATIONS`` times, checking WF-4 after each."""
        for boundary in ("exit", "asyncio", "exec"):
            self.assertTrue(
                actor.prewarm_executor.call_one().get(timeout=_CALL_TIMEOUT_SECS)
            )
            for _ in range(_ITERATIONS):
                result = actor.fork_child.call_one(boundary).get(
                    timeout=_CALL_TIMEOUT_SECS
                )
                self.assertEqual(result.parent_pid == os.getpid(), in_client)
                self.assertEqual(result.version, sys.version_info[:2])
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(result.marker, b"" if boundary == "exec" else _MARKER)
                self.assertEqual(result.leaked_pipe_fds, 0)
                self.assertTrue(
                    actor.executor_round_trip.call_one().get(timeout=_CALL_TIMEOUT_SECS)
                )
                self.assertEqual(
                    actor.ping.call_one().get(timeout=_CALL_TIMEOUT_SECS), "pong"
                )

    def test_fork_from_native_bootstrap_proc(self) -> None:
        """WF-1's native ``bootstrap_main()`` proc: forks from a worker proc."""
        procs = this_host().spawn_procs(per_host={"procs": 1})
        try:
            actor = procs.spawn("forker", ForkingActor)
            self._check_fork_matrix(actor, in_client=False)
        finally:
            procs.stop().get(timeout=_CALL_TIMEOUT_SECS)

    def test_fork_from_client_process_actor(self) -> None:
        """An actor on the client's proc: WF-2/WF-3 and CF-* both apply."""
        actor = this_proc().spawn("client_forker", ForkingActor)
        self._check_fork_matrix(actor, in_client=True)
