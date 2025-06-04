# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import subprocess

import pytest
from monarch.actor_mesh import Actor, ActorMeshRefCallFailedException, endpoint

from monarch.proc_mesh import proc_mesh


class ExceptionActor(Actor):
    """An actor that has endpoints which raise exceptions."""

    @endpoint
    async def raise_exception(self) -> None:
        """Endpoint that raises an exception."""
        raise Exception("This is a test exception")


class ExceptionActorSync(Actor):
    """An actor that has endpoints which raise exceptions."""

    @endpoint  # pyre-ignore
    def raise_exception(self) -> None:
        """Endpoint that raises an exception."""
        raise Exception("This is a test exception")


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "exception_actor_async_call"),
        (ExceptionActorSync, "exception_actor_sync_call"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_actor_exception(actor_class, actor_name, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = await proc_mesh(gpus=num_procs)
    exception_actor = await proc.spawn(actor_name, actor_class)

    with pytest.raises(
        ActorMeshRefCallFailedException, match="This is a test exception"
    ):
        if num_procs == 1:
            await exception_actor.raise_exception.call_one()
        else:
            await exception_actor.raise_exception.call()


@pytest.mark.parametrize(
    "actor_class,actor_name",
    [
        (ExceptionActor, "exception_actor_async_call"),
        (ExceptionActorSync, "exception_actor_sync_call"),
    ],
)
@pytest.mark.parametrize("num_procs", [1, 2])
def test_actor_exception_sync(actor_class, actor_name, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = proc_mesh(gpus=num_procs).get()
    exception_actor = proc.spawn(actor_name, actor_class).get()

    with pytest.raises(
        ActorMeshRefCallFailedException, match="This is a test exception"
    ):
        if num_procs == 1:
            exception_actor.raise_exception.call_one().get()
        else:
            exception_actor.raise_exception.call().get()


# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
@pytest.mark.parametrize("num_procs", [1, 2])
@pytest.mark.parametrize("sync_endpoint", [False, True])
@pytest.mark.parametrize("sync_test_impl", [False, True])
@pytest.mark.parametrize("endpoint_name", ["cause_segfault", "cause_panic"])
def test_actor_segfault(num_procs, sync_endpoint, sync_test_impl, endpoint_name):
    """
    Test that segfaults in actor endpoints result in a non-zero exit code.
    This test spawns a subprocess that will segfault and checks its exit code.

    Tests both ExceptionActor and ExceptionActorSync using async API.
    """
    # Run the segfault test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        f"--num-procs={num_procs}",
        f"--sync-endpoint={sync_endpoint}",
        f"--sync-test-impl={sync_test_impl}",
        f"--endpoint-name={endpoint_name}",
    ]
    process = subprocess.run(cmd, capture_output=True, timeout=60)
    print(process.stdout.decode())
    print(process.stderr.decode())

    # Assert that the subprocess exited with a non-zero code
    assert "I actually ran" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"
