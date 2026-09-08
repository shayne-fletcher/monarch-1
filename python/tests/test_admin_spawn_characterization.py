# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Characterization tests for the task boundaries around admin creation.

The public ``_spawn_admin`` function returns a lazy Monarch ``Future``. Once an
asyncio caller observes that Future, a retained Handle owns the producer while
the Python asyncio Future is only an observer. These tests pin both boundaries:
which validation happens during the public call, and what happens when the
caller cancels an observer after the raw native task has been constructed.
"""

import asyncio
import os
import threading
from typing import Callable
from unittest.mock import patch

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._rust_bindings.monarch_hyperactor.host_mesh import PyMeshAdminRef
from monarch._rust_bindings.monarch_hyperactor.pytokio import Handle, PythonTask
from monarch._src.actor import future as future_module, host_mesh as host_mesh_module
from monarch._src.actor.actor_mesh import shutdown_context
from monarch._src.actor.future import Future
from monarch._src.actor.host_mesh import HostMesh, this_host


async def _wait_for_threading_event(event: threading.Event, message: str) -> None:
    reached = await asyncio.wait_for(
        asyncio.to_thread(event.wait, 30),
        timeout=35,
    )
    assert reached, message


async def _wait_for_handle_result(
    handle: Handle,
    message: str,
) -> tuple[str, PyMeshAdminRef]:
    deadline = asyncio.get_running_loop().time() + 30
    while (result := handle.poll()) is None:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(message)
        await asyncio.sleep(0)
    return result


class _AdminSpawnGate:
    """Hold a real native admin task after construction but before its first poll.

    Calling the real binding here preserves its synchronous validation. A valid
    call returns the production task, which this wrapper holds behind a bounded
    gate and then awaits normally. The construction event therefore says that
    the outer producer owns a native task, not that the native task has run.
    """

    def __init__(self, real_spawn: Callable[..., PythonTask]) -> None:
        self._real_spawn = real_spawn
        self.release = threading.Event()
        self.native_constructed = threading.Event()
        self.attempts: list[str | None] = []
        self.tasks_constructed = 0
        self.results: list[tuple[str, PyMeshAdminRef]] = []

    def __call__(
        self,
        host_meshes: object,
        instance: object,
        admin_addr: str | None,
        telemetry_url: str | None,
    ) -> PythonTask:
        self.attempts.append(admin_addr)
        native_task = self._real_spawn(
            host_meshes,
            instance,
            admin_addr,
            telemetry_url,
        )
        self.tasks_constructed += 1
        self.native_constructed.set()

        async def drive_native() -> tuple[str, PyMeshAdminRef]:
            released = await PythonTask.spawn_blocking(
                lambda: self.release.wait(timeout=30)
            )
            if not released:
                raise TimeoutError("admin spawn release was not published")
            result = await native_task
            self.results.append(result)
            return result

        return PythonTask.from_coroutine(drive_native())


async def _assert_input_errors(
    host: HostMesh,
    gate: _AdminSpawnGate,
) -> None:
    with pytest.raises(ValueError) as empty_error:
        host_mesh_module._spawn_admin([])
    assert type(empty_error.value) is ValueError
    assert str(empty_error.value) == "_spawn_admin requires at least one HostMesh"
    assert gate.attempts == []
    assert "MONARCH_ADMIN_URL" not in os.environ

    invalid = host_mesh_module._spawn_admin([host], admin_addr="not-an-address")
    assert gate.attempts == []
    with pytest.raises(Exception) as invalid_error:
        await asyncio.wait_for(invalid.as_asyncio(), timeout=30)
    assert type(invalid_error.value) is Exception
    assert str(invalid_error.value) == (
        "invalid admin_addr 'not-an-address': invalid socket address syntax"
    )
    assert gate.attempts == ["not-an-address"]
    assert gate.tasks_constructed == 0
    assert not gate.native_constructed.is_set()
    assert gate.results == []
    assert "MONARCH_ADMIN_URL" not in os.environ


@pytest.mark.timeout(240)
@isolate_in_subprocess
async def test_spawn_admin_is_lazy_and_survives_cancelled_observer() -> None:
    previous_admin_url = os.environ.pop("MONARCH_ADMIN_URL", None)
    gate = _AdminSpawnGate(host_mesh_module._hy_spawn_admin)
    success: Future[tuple[str, PyMeshAdminRef]] | None = None
    observer: asyncio.Future[tuple[str, PyMeshAdminRef]] | None = None
    success_started = False
    admin_ref: PyMeshAdminRef | None = None

    try:
        host = this_host()
        assert await asyncio.wait_for(host.initialized.as_asyncio(), timeout=30) is True

        with patch.object(host_mesh_module, "_hy_spawn_admin", gate):
            await _assert_input_errors(host, gate)

            successful_spawn = host_mesh_module._spawn_admin(
                [host], admin_addr="[::]:0"
            )
            success = successful_spawn
            # Future construction captures the current Monarch context, but it
            # must not resolve the host or invoke the native admin binding.
            assert gate.attempts == ["not-an-address"]
            assert "MONARCH_ADMIN_URL" not in os.environ

            observer = successful_spawn.as_asyncio()
            success_started = True
            status = successful_spawn._status
            assert isinstance(status, future_module._Handle)
            producer_handle = status.handle
            await _wait_for_threading_event(
                gate.native_constructed,
                "the outer producer did not construct the native admin task",
            )
            assert gate.attempts == ["not-an-address", "[::]:0"]
            assert gate.tasks_constructed == 1
            assert gate.results == []
            assert not observer.done()
            assert "MONARCH_ADMIN_URL" not in os.environ

            assert observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer

            gate.release.set()
            # Handle.poll() neither drives the producer nor retains another
            # waiter. The cancelled observer's internal completion waiter stays
            # alive, so require completion before attaching a replacement
            # Python observer rather than claiming that no waiter exists.
            cached_url, cached_ref = await _wait_for_handle_result(
                producer_handle,
                "admin spawn did not finish after its first observer was cancelled",
            )
            assert os.environ["MONARCH_ADMIN_URL"] == cached_url
            admin_url, admin_ref = await asyncio.wait_for(
                successful_spawn.as_asyncio(),
                timeout=30,
            )
            assert len(gate.results) == 1
            native_url, native_ref = gate.results[0]
            assert admin_url == native_url
            assert admin_url == cached_url
            assert admin_ref is native_ref
            assert admin_ref is cached_ref
            assert os.environ["MONARCH_ADMIN_URL"] == admin_url
    finally:
        if observer is not None and not observer.done():
            observer.cancel()
        gate.release.set()
        try:
            if success_started and success is not None:
                await asyncio.wait_for(success.as_asyncio(), timeout=30)
        finally:
            try:
                await asyncio.wait_for(shutdown_context().as_asyncio(), timeout=30)
            finally:
                if previous_admin_url is None:
                    os.environ.pop("MONARCH_ADMIN_URL", None)
                else:
                    os.environ["MONARCH_ADMIN_URL"] = previous_admin_url

    # Keep the opaque capability live through local admin shutdown.
    assert admin_ref is not None
