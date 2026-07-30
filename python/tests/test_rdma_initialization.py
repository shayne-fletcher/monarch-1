# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Tests for RDMA manager initialization and its failure surfaces.

These tests cover the owner-backed binding from local and remote actors,
production readiness caching and validation, public buffer operations, and
backend configuration errors.
"""

import gc
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

if sys.platform != "linux":
    pytest.skip("linux-only", allow_module_level=True)

from isolate_in_subprocess import isolate_in_subprocess
from monarch._src.actor.proc_mesh import ProcMesh
from monarch.actor import Actor, endpoint
from monarch.config import configured
from monarch.rdma import RDMABuffer


_PUBLIC_PATH_INITIAL = b"rdma-public-path-a"
_PUBLIC_PATH_UPDATED = b"rdma-public-path-b"


class _RdmaInitProbe(Actor):
    @endpoint
    async def ensure_here(self) -> None:
        from monarch._rust_bindings.rdma import _RdmaManager
        from monarch._src.actor.actor_mesh import context

        instance = context().actor_instance
        proc_mesh = instance.proc_mesh
        assert proc_mesh is not None
        handle = _RdmaManager.ensure_init_rdma_manager_nonblocking(
            proc_mesh._proc_mesh,
            instance,
        )
        assert await handle is None

    @endpoint
    async def ensure_from_nested_child(self) -> None:
        from monarch._src.actor.actor_mesh import context

        proc_mesh = context().actor_instance.proc_mesh
        assert proc_mesh is not None
        child = proc_mesh.spawn("nested_rdma_init_probe", _RdmaInitProbe)
        assert await child.ensure_here.call_one() is None

    @endpoint
    async def exercise_public_paths_with_malformed_target(self) -> tuple[str, str]:
        from monarch._rust_bindings.rdma import RdmaInitError
        from monarch.rdma import RDMAAction

        try:
            RDMABuffer(memoryview(bytearray(b"rdma")))
        except RdmaInitError as error:
            buffer_error = str(error)
        else:
            raise AssertionError(
                "public buffer creation unexpectedly ignored failed readiness"
            )

        try:
            await RDMAAction().submit()
        except RdmaInitError as error:
            submit_error = str(error)
        else:
            raise AssertionError("public submit unexpectedly ignored failed readiness")

        return buffer_error, submit_error


class _RdmaPublicPathProbe(Actor):
    def __init__(self) -> None:
        self.data = bytearray(_PUBLIC_PATH_INITIAL)
        self.buffer: RDMABuffer | None = None

    @endpoint
    async def create_buffer(self) -> RDMABuffer:
        self.buffer = RDMABuffer(memoryview(self.data))
        return self.buffer

    @endpoint
    async def read_and_write(
        self,
        buffer: RDMABuffer,
    ) -> bytes:
        readback = bytearray(len(_PUBLIC_PATH_INITIAL))
        assert await buffer.read_into(memoryview(readback)) is None
        assert await buffer.write_from(memoryview(_PUBLIC_PATH_UPDATED)) is None
        return bytes(readback)

    @endpoint
    async def verify_and_drop(self) -> None:
        assert self.buffer is not None
        assert bytes(self.data) == _PUBLIC_PATH_UPDATED
        assert await self.buffer.drop() is None


def test_manager_init_cache_reuses_handle_without_retaining_mesh(monkeypatch) -> None:
    from monarch._src.rdma import rdma as rdma_module

    class FakeMesh:
        _proc_mesh = object()

    class FakeManager:
        calls = 0

        @classmethod
        def ensure_init_rdma_manager_nonblocking(cls, shared, instance):
            cls.calls += 1
            return object()

    class FakeContext:
        actor_instance = object()

    monkeypatch.setattr(rdma_module, "_RdmaManager", FakeManager)
    monkeypatch.setattr(rdma_module, "context", FakeContext)

    with rdma_module._rdma_manager_init_cache_lock:
        initial_size = len(rdma_module._rdma_manager_init_cache)

    mesh = cast(ProcMesh, FakeMesh())
    other_mesh = cast(ProcMesh, FakeMesh())
    first = rdma_module._ensure_init_rdma_manager_on_mesh(mesh)
    second = rdma_module._ensure_init_rdma_manager_on_mesh(mesh)
    other = rdma_module._ensure_init_rdma_manager_on_mesh(other_mesh)
    assert second is first
    assert other is not first
    assert FakeManager.calls == 2

    mesh_ref = weakref.ref(mesh)
    other_mesh_ref = weakref.ref(other_mesh)
    del mesh
    del other_mesh
    gc.collect()
    assert mesh_ref() is None
    assert other_mesh_ref() is None
    with rdma_module._rdma_manager_init_cache_lock:
        assert len(rdma_module._rdma_manager_init_cache) == initial_size


def test_manager_init_cache_concurrent_first_miss_returns_one_handle(
    monkeypatch,
) -> None:
    from monarch._src.rdma import rdma as rdma_module

    class FakeMesh:
        _proc_mesh = object()

    first_creation_started = threading.Event()
    release_first_creation = threading.Event()

    class ContentionTrackingLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.waiter_started = threading.Event()

        def __enter__(self):
            if self._lock.locked():
                self.waiter_started.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._lock.release()

    cache_lock = ContentionTrackingLock()

    class FakeManager:
        calls = 0

        @classmethod
        def ensure_init_rdma_manager_nonblocking(cls, shared, instance):
            cls.calls += 1
            first_creation_started.set()
            assert release_first_creation.wait(timeout=5)
            return object()

    class FakeContext:
        actor_instance = object()

    monkeypatch.setattr(rdma_module, "_RdmaManager", FakeManager)
    monkeypatch.setattr(rdma_module, "context", FakeContext)
    monkeypatch.setattr(rdma_module, "_rdma_manager_init_cache_lock", cache_lock)

    mesh = cast(ProcMesh, FakeMesh())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rdma_module._ensure_init_rdma_manager_on_mesh, mesh)
        assert first_creation_started.wait(timeout=5)
        second = pool.submit(rdma_module._ensure_init_rdma_manager_on_mesh, mesh)
        assert cache_lock.waiter_started.wait(timeout=5)
        release_first_creation.set()
        first_handle = first.result(timeout=10)
        second_handle = second.result(timeout=10)

    assert second_handle is first_handle
    assert FakeManager.calls == 1


def test_invalid_submit_timeout_does_not_start_init(monkeypatch) -> None:
    from monarch._src.rdma import rdma as rdma_module

    init_called = False

    def unexpected_init():
        nonlocal init_called
        init_called = True
        raise AssertionError("invalid timeout must be rejected before initialization")

    monkeypatch.setattr(rdma_module, "_ensure_init_rdma_manager", unexpected_init)
    with pytest.raises(OverflowError):
        rdma_module.RDMAAction().submit(timeout=-1)
    assert not init_called


def test_unavailable_backend_does_not_start_init(monkeypatch) -> None:
    from monarch._src.rdma import rdma as rdma_module

    init_called = False

    def unexpected_init():
        nonlocal init_called
        init_called = True
        raise AssertionError("an unavailable backend must be rejected before init")

    monkeypatch.setattr(rdma_module, "get_rdma_backend", lambda: "none")
    monkeypatch.setattr(rdma_module, "_ensure_init_rdma_manager", unexpected_init)

    with pytest.raises(RuntimeError, match="RDMA is not available"):
        rdma_module.RDMABuffer(memoryview(bytearray(b"rdma")))
    assert not init_called


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_public_submit_and_drop_resolve_to_none_after_tcp_init() -> None:
    from monarch.actor import Actor, endpoint, this_host
    from monarch.rdma import RDMAAction, RDMABuffer

    class CpuActor(Actor):
        @endpoint
        async def exercise(self) -> None:
            data = bytearray(b"rdma")
            buffer = RDMABuffer(memoryview(data))
            assert await RDMAAction().submit() is None
            assert await buffer.drop() is None

    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="",
    ):
        proc = this_host().spawn_procs(per_host={"cpus": 1})
        actor = proc.spawn("cpu_actor", CpuActor)
        assert await actor.exercise.call_one() is None
        await proc.stop()


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_public_paths_propagate_readiness_failure() -> None:
    from monarch.actor import this_host

    with configured(
        rdma_disable_ibverbs=False,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="mlx5_0",
    ):
        proc = this_host().spawn_procs(per_host={"cpus": 1})
        try:
            probe = proc.spawn("rdma_readiness_failure_probe", _RdmaInitProbe)
            (
                buffer_failure,
                submit_failure,
            ) = await probe.exercise_public_paths_with_malformed_target.call_one()
            assert "RDMA_IBVERBS_TARGET" in buffer_failure
            assert "RDMA_IBVERBS_TARGET" in submit_failure
        finally:
            await proc.stop()


@pytest.mark.timeout(90)
@isolate_in_subprocess
async def test_public_rdma_paths() -> None:
    from monarch.actor import this_host

    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="",
    ):
        producer_proc = this_host().spawn_procs(per_host={"cpus": 1})
        try:
            consumer_proc = this_host().spawn_procs(per_host={"cpus": 1})
            try:
                producer = producer_proc.spawn(
                    "rdma_public_path_producer", _RdmaPublicPathProbe
                )
                consumer = consumer_proc.spawn(
                    "rdma_public_path_consumer", _RdmaPublicPathProbe
                )

                buffer = await producer.create_buffer.call_one()
                readback = await consumer.read_and_write.call_one(buffer)
                assert await producer.verify_and_drop.call_one() is None

                assert readback == _PUBLIC_PATH_INITIAL
            finally:
                await consumer_proc.stop()
        finally:
            await producer_proc.stop()


# The tests below cover the owner-driven binding
# `_RdmaManager.ensure_init_rdma_manager_nonblocking`, which returns a `Handle`
# and surfaces a typed owner failure as a catchable `RdmaInitError`.


def _ensure_init_handle():
    """Spawn a one-proc mesh and call the Handle-returning binding, returning the
    `Handle` for the caller to await. Must be called inside a live event loop and
    the caller's `configured(...)` context (the owner reads config during init)."""
    from monarch._rust_bindings.rdma import _RdmaManager
    from monarch._src.actor.actor_mesh import context
    from monarch.actor import this_host

    proc_mesh = this_host().spawn_procs(per_host={"cpus": 1})
    return _RdmaManager.ensure_init_rdma_manager_nonblocking(
        proc_mesh._proc_mesh,
        context().actor_instance,
    )


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_returns_handle_that_resolves_over_tcp() -> None:
    """The binding resolves to None once the owner's full post-init() barrier
    completes, driven from the raw Shared[ProcMesh] with no Python-side mesh
    resolution (RMB-1, RMB-4). TCP fallback lets init() succeed without a NIC.
    Observation is non-consuming, so a second await also yields None (HDL-3)."""
    from monarch._rust_bindings.monarch_hyperactor.pytokio import Handle

    # Pin rdma_ibverbs_target so inherited configuration cannot preempt the
    # intended TCP-fallback branch.
    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="",
    ):
        handle = _ensure_init_handle()
        assert isinstance(handle, Handle)
        assert await handle is None
        assert await handle is None


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_succeeds_from_remote_actor() -> None:
    """A remote actor initializes through its inherited native client-root
    capability, without any driver-side RDMA initialization (RMB-3)."""
    from monarch.actor import this_host

    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="",
    ):
        proc_mesh = this_host().spawn_procs(per_host={"processes": 1})
        try:
            probe = proc_mesh.spawn("remote_rdma_init_probe", _RdmaInitProbe)
            assert await probe.ensure_here.call_one() is None
        finally:
            await proc_mesh.stop()


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_succeeds_from_nested_actor_on_remote_worker() -> None:
    """A child spawned inside a remote worker retains the capability across a
    second gspawn/ActorSpec environment hop on that worker (RMB-3)."""
    from monarch.actor import this_host

    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="",
    ):
        proc_mesh = this_host().spawn_procs(per_host={"processes": 1})
        try:
            parent = proc_mesh.spawn("parent_rdma_init_probe", _RdmaInitProbe)
            assert await parent.ensure_from_nested_child.call_one() is None
        finally:
            await proc_mesh.stop()


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_observes_its_supplied_shared() -> None:
    """The binding observes the exact Shared it is handed rather than an ambient
    proc mesh: a Shared resolving to a non-ProcMesh value fails the native
    downcast with a TypeError naming the expected ProcMesh type, distinct from
    RdmaInitError (RMB-2, RMB-5)."""
    from monarch._rust_bindings.monarch_hyperactor.pytokio import Shared
    from monarch._rust_bindings.rdma import _RdmaManager
    from monarch._src.actor.actor_mesh import context

    # No configured(...) context: the Shared resolves to a non-ProcMesh value, so the
    # downcast fails before any RDMA config would be read.
    handle = _RdmaManager.ensure_init_rdma_manager_nonblocking(
        Shared.from_value(object()),
        context().actor_instance,
    )
    with pytest.raises(TypeError) as excinfo:
        await handle
    assert "ProcMesh" in str(excinfo.value), (
        f"the downcast failure should name the expected ProcMesh type: {excinfo.value}"
    )


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_raises_rdma_init_error_when_no_backend() -> None:
    """A forced no-backend failure (ibverbs disabled, TCP fallback off) surfaces
    as a catchable native RdmaInitError carrying the cause. Deterministic on
    hosts with or without ibverbs (RMB-4, RMB-5)."""
    from monarch._rust_bindings.rdma import RdmaInitError

    # Pin rdma_ibverbs_target so inherited configuration cannot preempt the
    # forced no-backend branch.
    with configured(
        rdma_disable_ibverbs=True,
        rdma_allow_tcp_fallback=False,
        rdma_ibverbs_target="",
    ):
        handle = _ensure_init_handle()
        with pytest.raises(RdmaInitError) as excinfo:
            await handle
        assert "no RDMA backend available" in str(excinfo.value), (
            f"Actual error: {excinfo.value}"
        )


@pytest.mark.timeout(60)
@isolate_in_subprocess
async def test_ensure_init_raises_rdma_init_error_on_malformed_target() -> None:
    """A malformed RDMA_IBVERBS_TARGET surfaces as a catchable native
    RdmaInitError, independent of hardware availability (RMB-5)."""
    from monarch._rust_bindings.rdma import RdmaInitError

    with configured(
        rdma_allow_tcp_fallback=True,
        rdma_ibverbs_target="mlx5_0",  # missing the required `nic:` target kind
    ):
        handle = _ensure_init_handle()
        with pytest.raises(RdmaInitError) as excinfo:
            await handle
        assert "RDMA_IBVERBS_TARGET" in str(excinfo.value), (
            f"Actual error: {excinfo.value}"
        )
