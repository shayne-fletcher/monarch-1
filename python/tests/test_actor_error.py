# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources
import subprocess

import pytest
from monarch.actor_mesh import Actor, ActorError, endpoint

from monarch.proc_mesh import proc_mesh


class ExceptionActor(Actor):
    @endpoint
    async def raise_exception(self) -> None:
        raise Exception("This is a test exception")


class ExceptionActorSync(Actor):
    @endpoint  # pyre-ignore
    def raise_exception(self) -> None:
        raise Exception("This is a test exception")


@pytest.mark.parametrize(
    "actor_class",
    [ExceptionActor, ExceptionActorSync],
)
@pytest.mark.parametrize("num_procs", [1, 2])
async def test_actor_exception(actor_class, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = await proc_mesh(gpus=num_procs)
    exception_actor = await proc.spawn("exception_actor", actor_class)

    with pytest.raises(ActorError, match="This is a test exception"):
        if num_procs == 1:
            await exception_actor.raise_exception.call_one()
        else:
            await exception_actor.raise_exception.call()


@pytest.mark.parametrize(
    "actor_class",
    [ExceptionActor, ExceptionActorSync],
)
@pytest.mark.parametrize("num_procs", [1, 2])
def test_actor_exception_sync(actor_class, num_procs):
    """
    Test that exceptions raised in actor endpoints are propagated to the client.
    """
    proc = proc_mesh(gpus=num_procs).get()
    exception_actor = proc.spawn("exception_actor", actor_class).get()

    with pytest.raises(ActorError, match="This is a test exception"):
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
def test_actor_supervision(num_procs, sync_endpoint, sync_test_impl, endpoint_name):
    """
    Test that an endpoint causing spontaenous process exit is handled by the supervisor.

    Today, these events are delivered to the client and cause the client process
    to exit with a non-zero code, so the only way we can test it is via a
    subprocess harness.
    """
    # Run the segfault test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-endpoint",
        f"--num-procs={num_procs}",
        f"--sync-endpoint={sync_endpoint}",
        f"--sync-test-impl={sync_test_impl}",
        f"--endpoint-name={endpoint_name}",
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Assert that the subprocess exited with a non-zero code
    assert "I actually ran" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"


# oss_skip: importlib not pulling resource correctly in git CI, needs to be revisited
@pytest.mark.oss_skip
def test_proc_mesh_bootstrap_error():
    """
    Test that attempts to spawn a ProcMesh with a failure during bootstrap.
    """
    # Run the segfault test in a subprocess
    test_bin = importlib.resources.files("monarch.python.tests").joinpath("test_bin")
    cmd = [
        str(test_bin),
        "error-bootstrap",
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as e:
        print("timeout expired")
        if e.stdout is not None:
            print(e.stdout.decode())
        if e.stderr is not None:
            print(e.stderr.decode())
        raise

    # Assert that the subprocess exited with a non-zero code
    assert "I actually ran" in process.stdout.decode()
    assert (
        process.returncode != 0
    ), f"Expected non-zero exit code, got {process.returncode}"
