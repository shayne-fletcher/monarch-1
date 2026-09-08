# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Characterize host draining while proc-mesh logging is still initializing."""

import threading
from typing import Any
from unittest.mock import patch

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask
from monarch._src.actor.logging import LoggingManager
from monarch._src.job.process import ProcessJob
from scoped_state import scoped_state


@pytest.mark.timeout(90)
@isolate_in_subprocess
def test_flush_pending_spawns_does_not_await_logging_init() -> None:
    init_entered = threading.Event()
    release_init = threading.Event()
    real_init = LoggingManager.init

    async def gated_init(
        manager: LoggingManager,
        proc_mesh: Any,
        stream_to_client: bool,
    ) -> None:
        # ProcMesh calls init only after the native proc spawn resolves. Holding
        # this method therefore exposes the real interval in which the proc
        # exists but its logging client has not been installed.
        init_entered.set()
        released = await PythonTask.spawn_blocking(
            lambda: release_init.wait(timeout=30)
        )
        if not released:
            raise TimeoutError("logging initialization release was not published")
        await real_init(manager, proc_mesh, stream_to_client)

    proc_mesh = None
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        try:
            with patch.object(LoggingManager, "init", gated_init):
                proc_mesh = host.spawn_procs(name="logging_init_gate")
                assert init_entered.wait(timeout=30), (
                    "logging initialization did not reach the gate"
                )

                native_spawn = host._pending_spawns[-1]
                assert native_spawn.poll() is not None
                assert proc_mesh._logging_manager._logging_mesh_client is None
                assert proc_mesh._proc_mesh.poll() is None

                flush_without_client: list[bool] = []
                native_flush_constructions = 0
                flush_from_tokio = proc_mesh._logging_manager._flush_from_tokio
                new_flush_task = proc_mesh._logging_manager._new_flush_task

                async def record_flush_from_tokio() -> None:
                    flush_without_client.append(
                        proc_mesh._logging_manager._logging_mesh_client is None
                    )
                    await flush_from_tokio()

                def record_new_flush_task() -> PythonTask[None]:
                    nonlocal native_flush_constructions
                    native_flush_constructions += 1
                    return new_flush_task()

                with (
                    patch.object(
                        proc_mesh._logging_manager,
                        "_flush_from_tokio",
                        record_flush_from_tokio,
                    ),
                    patch.object(
                        proc_mesh._logging_manager,
                        "_new_flush_task",
                        record_new_flush_task,
                    ),
                ):
                    PythonTask.from_coroutine(
                        host._flush_pending_spawns()
                    ).with_timeout(10).block_on()

                assert flush_without_client == [True]
                assert native_flush_constructions == 0
                assert host._pending_spawns == []
                assert proc_mesh._proc_mesh.poll() is None
        finally:
            release_init.set()
            if proc_mesh is not None:
                assert (
                    proc_mesh._proc_mesh.task().with_timeout(30).block_on() is not None
                )
                assert proc_mesh._logging_manager._logging_mesh_client is not None
