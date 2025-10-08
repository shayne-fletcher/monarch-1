# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
from typing import TYPE_CHECKING

from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints
from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice

from monarch._src.actor.allocator import AllocateMixin

from monarch._src.actor.endpoint import Extent
from monarch._src.actor.host_mesh import HostMesh as HostMeshV0
from monarch._src.actor.v1.host_mesh import HostMesh as HostMeshV1

enabled = os.environ.get("MONARCH_HOST_MESH_V1_REMOVE_ME_BEFORE_RELEASE", "0") != "0"

if TYPE_CHECKING or not enabled:
    from monarch._src.actor.host_mesh import HostMesh, this_host, this_proc
    from monarch._src.actor.proc_mesh import get_or_spawn_controller, ProcMesh
else:
    from monarch._src.actor.v1.host_mesh import HostMesh, this_host, this_proc
    from monarch._src.actor.v1.proc_mesh import get_or_spawn_controller, ProcMesh


def host_mesh_from_alloc(
    name: str, extent: Extent, allocator: AllocateMixin, constraints: AllocConstraints
) -> "HostMeshV0 | HostMeshV1":
    if enabled:
        return HostMeshV1.allocate_nonblocking(name, extent, allocator, constraints)
    else:
        return HostMeshV0(
            Shape(extent.labels, Slice.new_row_major(extent.sizes)),
            allocator,
            constraints,
        )


__all__ = [
    "HostMesh",
    "this_host",
    "this_proc",
    "get_or_spawn_controller",
    "ProcMesh",
    "host_mesh_from_alloc",
]
