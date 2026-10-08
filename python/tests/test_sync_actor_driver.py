# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
A sync actor (every endpoint a plain ``def``) runs on a driver thread with no
event loop. Witnesses for SA-1 to SA-4 in ``monarch._src.actor.actor_mesh``.

Actors record what they observe to a file, because ``__supervise__`` and
``__cleanup__`` have no caller to return to.
"""

import asyncio
import gc
import json
import operator
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import monarch.actor
import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._rust_bindings.monarch_hyperactor.handle import _new_handle_pair
from monarch._rust_bindings.monarch_hyperactor.pickle import (
    _get_pending_reserve_count,
    _reset_pending_reserve_count,
)
from monarch._rust_bindings.monarch_hyperactor.supervision import MeshFailure
from monarch._rust_bindings.monarch_hyperactor.testing import _sync_inbox_for_test
from monarch._src.actor import actor_mesh
from monarch._src.actor.actor_mesh import ActorMesh
from monarch._src.actor.host_mesh import this_host, this_proc
from monarch.actor import Accumulator, Actor, ActorError, as_endpoint, endpoint


def _record(path: str, label: str) -> None:
    try:
        asyncio.get_running_loop()
        loop = True
    except RuntimeError:
        loop = False
    entry = {
        "label": label,
        "ident": threading.get_ident(),
        "thread": threading.current_thread().name,
        "loop": loop,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _entries(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _labels(path: str) -> list[str]:
    return [entry["label"] for entry in _entries(path)]


async def _wait_for_label(path: str, label: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while label not in _labels(path):
        if time.monotonic() > deadline:
            raise AssertionError(f"{label!r} never recorded; saw {_labels(path)}")
        await asyncio.sleep(0.1)


def _log_path() -> str:
    return os.path.join(tempfile.mkdtemp(), "log.jsonl")


class _FailingChild(Actor):
    @endpoint
    async def fail(self) -> None:
        # No reply port (broadcast), so this kills the child and its owner
        # receives a supervision event.
        raise RuntimeError("child failure")


class _ThreadProbe(Actor):
    @endpoint
    async def alive(self, ident: int) -> bool:
        # Match the name as well: a dead driver's ident can be reused by a
        # foreign thread (such as the one that joined it), which Python then
        # lists as a `_DummyThread`.
        return any(
            t.ident == ident and t.name == "monarch-actor-driver"
            for t in threading.enumerate()
        )


class _Observer(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        _record(path, "__init__")
        self.child = this_proc().spawn("child", _FailingChild)

    @endpoint
    def observe(self) -> None:
        _record(self.path, "endpoint")

    @endpoint
    def fail_child(self) -> None:
        self.child.fail.broadcast()

    def __supervise__(self, failure: MeshFailure) -> bool:
        _record(self.path, "__supervise__")
        return True

    def __cleanup__(self, exc: Exception | None) -> None:
        _record(self.path, "__cleanup__")


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_sync_actor_code_runs_on_one_thread_with_no_loop() -> None:
    """SA-1, SA-2: `__init__`, an endpoint, `__supervise__` and `__cleanup__`
    all run on the actor's driver thread, with no running event loop."""
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    observer = proc.spawn("observer", _Observer, path)
    await observer.observe.call_one()
    await observer.fail_child.call_one()
    await _wait_for_label(path, "__supervise__")
    await cast(ActorMesh, observer).stop()
    await _wait_for_label(path, "__cleanup__")

    entries = _entries(path)
    assert [e["label"] for e in entries] == [
        "__init__",
        "endpoint",
        "__supervise__",
        "__cleanup__",
    ]
    assert not any(e["loop"] for e in entries), entries
    assert len({e["ident"] for e in entries}) == 1, entries
    assert {e["thread"] for e in entries} == {"monarch-actor-driver"}, entries
    await proc.stop()


class _Counter(Actor):
    @endpoint
    async def value(self) -> int:
        return 3


class _GetProbe(Actor):
    @endpoint
    def accumulate(self, counter: _Counter) -> tuple[int, list[str]]:
        started: list[str] = []
        original = threading.Thread.start

        def recording_start(thread: threading.Thread) -> None:
            started.append(thread.name)
            original(thread)

        with patch.object(threading.Thread, "start", recording_start):
            value = Accumulator(counter.value, 0, operator.add).accumulate().get()
        return value, started


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_get_in_sync_endpoint_drives_on_the_driver_thread() -> None:
    """SA-1, RF-4: `.get()` on a `@returns_future` body in a sync endpoint runs
    the body on the driver thread's own loop and starts no helper thread."""
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    probe = proc.spawn("get_probe", _GetProbe)
    counter = proc.spawn("counter", _Counter)
    value, started = await probe.accumulate.call_one(counter)
    assert value == 3
    assert not [name for name in started if "returns_future" in name], started
    await proc.stop()


class _Backlog(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.child = this_proc().spawn("child", _FailingChild)

    @endpoint
    def block(self, seconds: float) -> None:
        _record(self.path, "block-start")
        self.child.fail.broadcast()
        time.sleep(seconds)
        _record(self.path, "block-end")

    @endpoint
    def record(self, i: int) -> None:
        _record(self.path, f"record-{i}")

    @endpoint
    def ping(self) -> None:
        pass

    def __supervise__(self, failure: MeshFailure) -> bool:
        _record(self.path, "__supervise__")
        return True

    def __cleanup__(self, exc: Exception | None) -> None:
        _record(self.path, "__cleanup__")


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_supervision_runs_before_a_message_backlog() -> None:
    """SA-3, SI-1: a supervision event that arrives while a body runs is
    handled before the messages queued behind it, and its handled verdict
    leaves the actor running."""
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc.spawn("backlog", _Backlog, path)
    await actor.ping.call_one()
    # The child's failure reaches the actor well inside the block.
    actor.block.broadcast(3.0)
    for i in range(3):
        actor.record.broadcast(i)
    await _wait_for_label(path, "record-2")

    assert _labels(path) == [
        "block-start",
        "block-end",
        "__supervise__",
        "record-0",
        "record-1",
        "record-2",
    ]
    await actor.ping.call_one()
    await proc.stop()


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_supervision_queued_when_stop_begins_is_dropped() -> None:
    """SA-3, SI-6, end to end: a stop during a body is followed by
    `__cleanup__`, and the child failure raised during that body is not
    supervised. The sleep does not prove the failure was queued before stop
    began; `test_a_callback_runs_only_if_claimed_before_stop` pins the drop."""
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc.spawn("stop_drop", _Backlog, path)
    await actor.ping.call_one()
    actor.block.broadcast(4.0)
    # Long enough for the child's failure to be queued behind the block.
    await asyncio.sleep(2.0)
    await cast(ActorMesh, actor).stop()
    await _wait_for_label(path, "__cleanup__")

    assert _labels(path) == ["block-start", "block-end", "__cleanup__"]
    await proc.stop()


class _StandInSupervision:
    """Stands in for `QueuedSupervision`, which Python cannot construct."""


class _OneCallbackInbox:
    """Returns one supervision item, then reports the inbox closed."""

    def __init__(self, stopping: bool) -> None:
        self._items: list[object] = [_StandInSupervision()]
        self._stopping = stopping

    def next(self) -> object | None:
        return self._items.pop() if self._items else None

    def stopping(self) -> bool:
        return self._stopping


class _CountingSuperviser:
    def __init__(self) -> None:
        self.supervised = 0

    def _run_supervise(self, item: object) -> None:
        self.supervised += 1


@pytest.mark.parametrize("stopping, runs", [(True, 0), (False, 1)])
def test_a_callback_runs_only_if_claimed_before_stop(stopping: bool, runs: int) -> None:
    """SA-3, SI-6: the driver runs a dequeued supervision item only if it then
    reads `stopping()` as false; otherwise it drops the item unrun."""
    actor = _CountingSuperviser()
    instance = SimpleNamespace(kill=lambda reason: pytest.fail(reason))
    with patch.object(actor_mesh, "QueuedSupervision", _StandInSupervision):
        actor_mesh._sync_dispatch_loop(
            cast(Any, actor),
            cast(Any, _OneCallbackInbox(stopping)),
            cast(Any, instance),
        )
    assert actor.supervised == runs


class _Verdict(Actor):
    def __init__(self, path: str, raises: bool) -> None:
        self.path = path
        self.raises = raises
        self.child = this_proc().spawn("child", _FailingChild)

    @endpoint
    def fail_child(self) -> None:
        self.child.fail.broadcast()

    def __supervise__(self, failure: MeshFailure) -> bool:
        _record(self.path, "__supervise__")
        if self.raises:
            raise RuntimeError("supervise failure")
        return False


@pytest.mark.timeout(120)
@pytest.mark.parametrize("raises", [False, True], ids=["unhandled", "raised"])
@isolate_in_subprocess
async def test_a_sync_supervise_verdict_fails_the_actor(raises: bool) -> None:
    """A sync actor's `__supervise__` that returns a falsey value, or raises,
    fails the actor through `SupervisionOutcome`, so the failure reaches its
    owner; the hook runs on the driver thread."""
    faults: list[str] = []
    monarch.actor.unhandled_fault_hook = lambda failure: faults.append(str(failure))
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    owner = proc.spawn("verdict_owner", _Verdict, path, raises)
    await owner.fail_child.call_one()
    await _wait_for_label(path, "__supervise__")
    deadline = time.monotonic() + 30.0
    while not any("verdict_owner" in fault for fault in faults):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"the owner's failure never reached the client: {faults}"
            )
        await asyncio.sleep(0.1)

    entry = _entries(path)[0]
    assert entry["thread"] == "monarch-actor-driver" and not entry["loop"], entry
    if raises:
        assert any("supervise failure" in fault for fault in faults), faults
    await proc.stop()


class _AsyncActorWithPlainMethod(Actor):
    @endpoint
    async def ping(self) -> int:
        return 1

    def plain(self) -> str:
        return "plain ran"


class _SyncActorWithAsyncMethod(Actor):
    @endpoint
    def ping(self) -> int:
        return 1

    async def coroutine(self) -> str:
        return "coroutine ran"


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_as_endpoint_runs_a_plain_method_on_an_async_actor() -> None:
    """An async actor still calls a plain method reached through `as_endpoint`
    rather than awaiting its result."""
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc.spawn("plain_method", _AsyncActorWithPlainMethod)
    assert await as_endpoint(actor.plain).call_one() == "plain ran"
    await proc.stop()


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_as_endpoint_rejects_an_async_method_on_a_sync_actor() -> None:
    """A sync actor has no event loop, so an `async def` method reached through
    `as_endpoint` fails with `TypeError` instead of returning a coroutine."""
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc.spawn("async_method", _SyncActorWithAsyncMethod)
    # `Any`: the endpoint's declared result is the coroutine the actor rejects.
    rejected = cast(Any, as_endpoint(actor.coroutine))
    with pytest.raises(ActorError, match="is an async def method"):
        await rejected.call_one()
    assert await actor.ping.call_one() == 1
    await proc.stop()


class _Ordered(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        _record(path, "__init__")

    @endpoint
    def slow(self) -> None:
        _record(self.path, "slow-start")
        time.sleep(2.0)
        _record(self.path, "slow-end")

    @endpoint
    def ping(self) -> None:
        pass

    def __cleanup__(self, exc: Exception | None) -> None:
        _record(self.path, "__cleanup__")


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_stop_runs_cleanup_after_the_active_body_then_joins() -> None:
    """SA-4: stop lets the active body finish, then runs `__cleanup__` on the
    driver, and the driver thread is gone when stop returns."""
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc.spawn("ordered", _Ordered, path)
    probe = proc.spawn("probe", _ThreadProbe)
    await actor.ping.call_one()
    actor.slow.broadcast()
    await _wait_for_label(path, "slow-start")
    await cast(ActorMesh, actor).stop()

    assert _labels(path) == ["__init__", "slow-start", "slow-end", "__cleanup__"]
    driver = _entries(path)[0]["ident"]
    assert not await probe.alive.call_one(driver)
    await proc.stop()


class _Fatal(BaseException):
    pass


class _Draining(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        _record(path, "__init__")

    @endpoint
    def die(self) -> None:
        _record(self.path, "die")
        raise _Fatal()

    @endpoint
    def after(self) -> None:
        _record(self.path, "after")

    def __cleanup__(self, exc: Exception | None) -> None:
        _record(self.path, "__cleanup__")


class _FailingInit(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        _record(path, "__init__")
        raise RuntimeError("init failure")

    @endpoint
    def ping(self) -> None:
        pass

    def __cleanup__(self, exc: Exception | None) -> None:
        # Must not run: an actor whose __init__ failed has no instance.
        _record(self.path, "__cleanup__")


class _FailingCleanup(Actor):
    def __init__(self, path: str) -> None:
        self.path = path
        _record(path, "__init__")

    @endpoint
    def ping(self) -> None:
        pass

    def __cleanup__(self, exc: Exception | None) -> None:
        _record(self.path, "__cleanup__")
        raise RuntimeError("cleanup failure")


async def _wait_until_gone(probe: _ThreadProbe, ident: int) -> None:
    deadline = time.monotonic() + 30.0
    while await probe.alive.call_one(ident):
        if time.monotonic() > deadline:
            raise AssertionError("the driver thread never exited")
        await asyncio.sleep(0.1)


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_a_base_exception_drains_then_cleans_up() -> None:
    """SA-4: after a body raises a `BaseException`, the actor dies, the driver
    runs no further messages, runs `__cleanup__`, and exits."""
    monarch.actor.unhandled_fault_hook = lambda failure: None
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    probe = proc.spawn("probe", _ThreadProbe)
    actor = proc.spawn("draining", _Draining, path)
    actor.die.broadcast()
    actor.after.broadcast()
    await _wait_for_label(path, "__cleanup__")

    assert _labels(path) == ["__init__", "die", "__cleanup__"]
    await _wait_until_gone(probe, _entries(path)[0]["ident"])
    await proc.stop()


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_an_init_failure_ends_the_driver() -> None:
    """SA-4: an `__init__` failure kills the actor; with no instance there is no
    `__cleanup__` to run, and the driver exits."""
    monarch.actor.unhandled_fault_hook = lambda failure: None
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    probe = proc.spawn("probe", _ThreadProbe)
    proc.spawn("failing_init", _FailingInit, path)
    await _wait_for_label(path, "__init__")
    await _wait_until_gone(probe, _entries(path)[0]["ident"])
    assert _labels(path) == ["__init__"]
    await proc.stop()


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_a_failing_cleanup_still_ends_the_driver() -> None:
    """SA-4: a `__cleanup__` that raises still ends the driver, which stop
    joins."""
    monarch.actor.unhandled_fault_hook = lambda failure: None
    path = _log_path()
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    probe = proc.spawn("probe", _ThreadProbe)
    actor = proc.spawn("failing_cleanup", _FailingCleanup, path)
    await actor.ping.call_one()
    await cast(ActorMesh, actor).stop()
    await _wait_for_label(path, "__cleanup__")
    await _wait_until_gone(probe, _entries(path)[0]["ident"])
    await proc.stop()


class _Target(Actor):
    @endpoint
    async def ping(self) -> str:
        return "pong"


class _SyncSpawner(Actor):
    @endpoint
    def spawn_and_return_pending(self) -> _Target:
        # The counter is per process and cumulative; count only this reply.
        _reset_pending_reserve_count()
        # Returned before its init completes, so the reply carries a pending
        # mesh reference that the driver must resolve before sending.
        return this_host().spawn_procs(name="inner_proc").spawn("inner", _Target)

    @endpoint
    def pending_reserve_count(self) -> int:
        return _get_pending_reserve_count()


@pytest.mark.timeout(120)
@isolate_in_subprocess
async def test_a_sync_reply_resolves_a_pending_mesh_reference() -> None:
    """A sync endpoint's reply that holds a still-pending mesh reference is
    resolved on the driver thread and arrives as a working mesh."""
    proc = this_host().spawn_procs(per_host={"gpus": 1})
    spawner = proc.spawn("spawner", _SyncSpawner)
    returned = await spawner.spawn_and_return_pending.call_one()
    assert await returned.ping.call_one() == "pong"
    assert await spawner.pending_reserve_count.call_one() > 0
    await proc.stop()


def test_a_receive_failure_kills_the_actor_and_ends_the_driver() -> None:
    """SA-4, SI-4: when a real `SyncInbox` fails to convert a queued item, the
    driver kills the actor and returns, without running `__cleanup__`; with the
    driver gone, cleanup's control send fails, so cleanup cannot wait on it."""
    messages, control, inbox = _sync_inbox_for_test()
    messages.send_unconvertible()

    class _NoCleanup:
        def _run_cleanup(self, item: object) -> None:
            raise AssertionError("cleanup must not run")

    class _Instance:
        reason: str | None = None

        def kill(self, reason: str) -> None:
            self.reason = reason

    instance = _Instance()
    # Messages only: a kept exception's traceback would keep the inbox alive.
    raised: list[str] = []

    def drive(inbox: Any) -> None:
        try:
            actor_mesh._sync_dispatch_loop(
                cast(Any, _NoCleanup()), inbox, cast(Any, instance)
            )
        except BaseException as e:  # noqa: B036 - recorded for the assertions
            raised.append(str(e))

    driver = threading.Thread(target=drive, args=(inbox,))
    del inbox
    driver.start()
    driver.join(timeout=10)

    assert driver.ident not in {t.ident for t in threading.enumerate()}
    assert instance.reason is not None and "unconvertible test item" in instance.reason
    assert raised == []
    del driver
    gc.collect()
    with pytest.raises(ValueError, match="SendError"):
        control.send(None)


def test_a_failing_cleanup_fails_the_cleanup_handle() -> None:
    """SA-4: the driver reports a `__cleanup__` failure on the cleanup item's
    Handle, which Rust awaits, rather than reporting success."""

    class _Raising:
        def __cleanup__(self, exc: Exception | None) -> None:
            raise RuntimeError("cleanup failure")

    handle, completer = _new_handle_pair()
    item = SimpleNamespace(context=None, error=None, completer=completer)
    driver_actor = actor_mesh._Actor()
    driver_actor.instance = _Raising()
    driver_actor._run_cleanup(cast(Any, item))
    with pytest.raises(RuntimeError, match="cleanup failure"):
        handle.get(5.0)
