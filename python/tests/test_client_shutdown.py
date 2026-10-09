# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import threading
import time
from typing import Any, Callable
from unittest.mock import patch

import pytest
from monarch._rust_bindings.monarch_hyperactor import host_mesh as host_mesh_binding
from monarch._src.actor import actor_mesh as actor_mesh_module
from monarch._src.actor.future import Future
from monarch.actor import Actor, endpoint, shutdown_context, this_host


class Simple(Actor):
    @endpoint
    def get_pid(self) -> int:
        return os.getpid()


def pid_exists(pid: int) -> bool:
    """True if pid exists, else false"""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


class _ShutdownCallThrough:
    """Record shutdown state and pause before calling the real native binding."""

    def __init__(self, real_shutdown: Callable[[float | None], Any]) -> None:
        self._real_shutdown = real_shutdown
        self.calls = 0
        self.shutdown_claimed_at_call: list[bool] = []
        self.timeouts: list[float | None] = []
        self.threads: list[int] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.departed = threading.Event()
        self.deadline = 0.0

    def __call__(self, timeout_secs: float | None = None) -> Any:
        self.shutdown_claimed_at_call.append(actor_mesh_module._shutdown_claimed)
        self.timeouts.append(timeout_secs)
        self.threads.append(threading.get_ident())
        self.calls += 1
        self.deadline = time.monotonic() + 30
        self.entered.set()
        try:
            assert self.release.wait(
                timeout=max(0, self.deadline - time.monotonic())
            ), "the test must release the committing call"
        finally:
            # A failed or timed-out wait must not look like a still-paused call.
            self.departed.set()
        return self._real_shutdown(timeout_secs)


class _ArrivalGate:
    """Stand in for `_client_context._lock`, recording which callers reached it."""

    def __init__(
        self, real_lock: threading.Lock, callers: list[threading.Thread]
    ) -> None:
        self._real_lock = real_lock
        self._callers = callers
        self._arrived: set[threading.Thread] = set()
        self._condition = threading.Condition()

    def __enter__(self) -> None:
        caller = threading.current_thread()
        if caller in self._callers:
            with self._condition:
                self._arrived.add(caller)
                self._condition.notify_all()
        self._real_lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._real_lock.release()

    def wait_for_all(self, timeout: float) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self._arrived) == len(self._callers), timeout
            )


# This test has to be in its own file so it does not share any process state
# with other tests. The client cannot be restarted after it has been shutdown.
@pytest.mark.timeout(240)
def test_client_shutdown() -> None:
    procs = this_host().spawn_procs(per_host={"gpus": 2})
    actors = procs.spawn("simple", Simple)
    pid_items = list(actors.get_pid.call().get(timeout=30).items())
    pids = [pid for _, pid in pid_items]

    # A rejected timeout must leave the client available for the valid shutdown
    # below. Include finite overflow, which a finiteness check alone misses.
    for invalid_timeout in (
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        float(2**64),
        1e300,
    ):
        with pytest.raises(ValueError):
            shutdown_context(host_timeout=invalid_timeout)
        assert not actor_mesh_module._shutdown_claimed

    call_through = _ShutdownCallThrough(host_mesh_binding.shutdown_local_host_mesh)
    futures: dict[int, Future[None]] = {}

    def call_shutdown() -> None:
        futures[threading.get_ident()] = shutdown_context()

    callers = [threading.Thread(target=call_shutdown) for _ in range(2)]
    client_context = actor_mesh_module._client_context
    real_lock = client_context._lock
    gate = _ArrivalGate(real_lock, callers)
    with patch.object(
        host_mesh_binding,
        "shutdown_local_host_mesh",
        call_through,
    ):
        # Both callers pass the outer _shutdown_claimed check before either takes
        # the lock, so only the recheck under the lock stops the second one
        # from committing.
        with patch.object(client_context, "_lock", gate):
            try:
                with real_lock:
                    for caller in callers:
                        caller.start()
                    assert gate.wait_for_all(timeout=30), (
                        "both callers must reach the lock"
                    )
                assert call_through.entered.wait(timeout=30), (
                    "the committing call must reach the native-call gate"
                )
                (committer,) = call_through.threads
                (loser,) = (caller for caller in callers if caller.ident != committer)

                # The loser must finish while the committer is still inside the
                # call-through, before native shutdown starts. The longer gate
                # deadline keeps its expiry from releasing a wrongly held lock
                # before the loser's bounded join can detect that regression.
                assert not call_through.departed.is_set()
                assert call_through.deadline - time.monotonic() > 5, (
                    "the gate must remain closed for the whole loser probe"
                )
                loser.join(timeout=5)
                assert not loser.is_alive(), (
                    "a repeat call must return while the committing call is paused"
                )
                other = loser.ident
                assert other is not None
                assert futures[other].get(timeout=0) is None
                assert not call_through.departed.is_set(), (
                    "the committing call must stay paused through the ready check"
                )
            finally:
                # Release before joining, including when the probe fails.
                call_through.release.set()
                for caller in callers:
                    if caller.ident is not None:
                        caller.join(timeout=30)
                        assert not caller.is_alive()

        # The committing call started shutdown before any Future was observed.
        assert call_through.calls == 1
        assert call_through.shutdown_claimed_at_call == [True]
        assert call_through.timeouts == [None]
        (committer,) = call_through.threads
        assert set(futures) == {caller.ident for caller in callers}
        (other,) = set(futures) - {committer}

        del futures[other]
        assert call_through.calls == 1

        # The client cannot be restarted after this one full shutdown: five
        # Rust registrations are OnceLock, and _client_context keeps the
        # stopped Context. Never reset _shutdown_claimed to rehearse it.
        del futures[committer]
        del procs
        del actors

        assert shutdown_context().get(timeout=30) is None
        assert call_through.calls == 1

    # After this, all the resources created by the client should be released,
    # including this_host and the procs, although the committing call's only
    # Future was dropped unobserved. This observes worker-process exit; it does
    # not exercise client-process exit or atexit.
    still_alive = []
    for _ in range(4):
        time.sleep(5)
        still_alive = [pid_exists(pid) for pid in pids]
        if not any(still_alive):
            # successfully shut off all pids.
            return
    raise ValueError(
        "Some pids are still alive at the end of the waiting period: {}", still_alive
    )
