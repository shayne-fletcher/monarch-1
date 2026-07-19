# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Tests for RDMA manager initialization failures.

These tests cover unsupported hosts and configuration errors that must surface
before backend fallback.
"""

import sys

import pytest

if sys.platform != "linux":
    pytest.skip("linux-only", allow_module_level=True)

from isolate_in_subprocess import isolate_in_subprocess
from monarch.config import configured
from monarch.rdma import is_ibverbs_available


needs_no_rdma = pytest.mark.skipif(
    is_ibverbs_available(),
    reason="RDMA is available, test only runs on systems without RDMA support",
)


async def _rdma_manager_init_fault(**config: object) -> str:
    import asyncio

    import monarch.actor
    from monarch._rust_bindings.rdma import _RdmaManager
    from monarch._src.actor.actor_mesh import context
    from monarch._src.actor.future import Future
    from monarch.actor import this_host

    faults = []
    faulted = asyncio.Event()

    def fault_hook(failure):
        faults.append(failure)
        faulted.set()

    monarch.actor.unhandled_fault_hook = fault_hook

    with configured(**config):
        proc_mesh = this_host().spawn_procs(per_host={"cpus": 1})
        await Future(
            coro=_RdmaManager.create_rdma_manager_nonblocking(
                await Future(coro=proc_mesh._proc_mesh.task()),
                context().actor_instance,
            )
        )
        await asyncio.wait_for(faulted.wait(), timeout=15.0)

    assert faults, "Expected a supervision fault, got none"
    return "\n".join(str(failure) for failure in faults)


@needs_no_rdma
@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_rdma_manager_creation_fails_when_unsupported() -> None:
    """RdmaManagerActor creation surfaces a supervision fault when RDMA is unsupported.

    This test only runs on systems where RDMA is not available. If RDMA is
    available, the test is skipped, since we cannot exercise the unsupported
    path.

    With TCP fallback disabled and no NIC backend present, the actor spawns
    successfully but its init fails. The failure is delivered to
    ``unhandled_fault_hook`` as a supervision fault, not raised as an exception
    from the creation call.

    We don't mock here because the failure originates from a cross-language call
    chain (Python -> Rust -> C ibverbs library); a Python mock cannot intercept
    the native device probe.
    """
    failure_message = await _rdma_manager_init_fault(rdma_allow_tcp_fallback=False)
    assert "no RDMA backend available" in failure_message, (
        f"Expected specific failure message not found. Actual fault: {failure_message}"
    )


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_malformed_rdma_ibverbs_target_fails_manager_creation() -> None:
    failure_message = await _rdma_manager_init_fault(
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="mlx5_0",  # Missing the required `nic:` target kind.
    )
    assert "RDMA_IBVERBS_TARGET" in failure_message, (
        f"Expected target configuration error not found. Actual fault: {failure_message}"
    )
