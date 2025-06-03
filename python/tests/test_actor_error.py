# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from monarch.actor_mesh import Actor, ActorMeshRefCallFailedException, endpoint

from monarch.proc_mesh import proc_mesh


class ExceptionActor(Actor):
    """An actor that has an endpoint which raises an exception."""

    @endpoint
    async def raise_exception(self) -> None:
        """Endpoint that raises an exception."""
        raise Exception("This is a test exception")


class ExceptionActorSync(Actor):
    """An actor that has an endpoint which raises an exception."""

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
