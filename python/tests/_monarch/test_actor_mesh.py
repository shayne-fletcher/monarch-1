# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import pickle
from typing import Any, List

import monarch
import pytest

from monarch._rust_bindings.monarch_hyperactor.actor import (
    PanicFlag,
    PythonMessage,
    PythonMessageKind,
)
from monarch._rust_bindings.monarch_hyperactor.actor_mesh import (
    PythonActorMesh,
    PythonActorMeshRef,
)

from monarch._rust_bindings.monarch_hyperactor.alloc import (  # @manual=//monarch/monarch_extension:monarch_extension
    AllocConstraints,
    AllocSpec,
)

from monarch._rust_bindings.monarch_hyperactor.mailbox import Mailbox, PortReceiver
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh
from monarch._rust_bindings.monarch_hyperactor.selection import Selection
from monarch._rust_bindings.monarch_hyperactor.shape import Shape


async def allocate() -> ProcMesh:
    spec = AllocSpec(AllocConstraints(), replica=2, gpus=3, hosts=8)
    allocator = monarch.LocalAllocator()
    alloc = await allocator.allocate(spec)
    proc_mesh = await ProcMesh.allocate_nonblocking(alloc)
    return proc_mesh


class MyActor:
    async def handle(
        self,
        mailbox: Mailbox,
        rank: int,
        shape: Shape,
        message: PythonMessage,
        panic_flag: PanicFlag,
        local_state: List[Any] | None = None,
    ) -> None:
        assert rank is not None

        # Extract response_port from the message kind
        call_method = message.kind
        assert isinstance(call_method, PythonMessageKind.CallMethod)
        assert call_method.response_port is not None

        reply_port = call_method.response_port
        reply_port.send(
            mailbox,
            PythonMessage(
                PythonMessageKind.Result(rank), pickle.dumps(f"rank: {rank}")
            ),
        )


async def test_bind_and_pickling() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    with pytest.raises(NotImplementedError, match="use bind()"):
        pickle.dumps(actor_mesh)

    actor_mesh_ref = actor_mesh.bind()
    assert actor_mesh_ref.shape == actor_mesh.shape
    obj = pickle.dumps(actor_mesh_ref)
    unpickled = pickle.loads(obj)
    assert repr(actor_mesh_ref) == repr(unpickled)
    assert actor_mesh_ref.shape == unpickled.shape


async def verify_cast(
    actor_mesh: PythonActorMesh | PythonActorMeshRef,
    mailbox: Mailbox,
    cast_ranks: List[int],
) -> None:
    receiver: PortReceiver
    handle, receiver = mailbox.open_port()
    port_ref = handle.bind()

    message = PythonMessage(
        PythonMessageKind.CallMethod("echo", port_ref), pickle.dumps("ping")
    )
    sel = Selection.from_string("*")
    if isinstance(actor_mesh, PythonActorMesh):
        actor_mesh.cast(sel, message)
    elif isinstance(actor_mesh, PythonActorMeshRef):
        actor_mesh.cast(mailbox, sel, message)

    rcv_ranks = []
    for _ in range(len(cast_ranks)):
        message = await receiver.recv()
        result_kind = message.kind
        assert isinstance(result_kind, PythonMessageKind.Result)
        rank = result_kind.rank
        assert rank is not None
        rcv_ranks.append(rank)
    rcv_ranks.sort()
    for i in cast_ranks:
        assert rcv_ranks[i] == i


@pytest.mark.timeout(30)
async def test_cast_handle() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    await verify_cast(actor_mesh, proc_mesh.client, list(range(2 * 3 * 8)))


@pytest.mark.timeout(30)
async def test_cast_ref() -> None:
    proc_mesh = await allocate()
    actor_mesh = await proc_mesh.spawn_nonblocking("test", MyActor)
    actor_mesh_ref = actor_mesh.bind()
    await verify_cast(actor_mesh_ref, proc_mesh.client, list(range(2 * 3 * 8)))
