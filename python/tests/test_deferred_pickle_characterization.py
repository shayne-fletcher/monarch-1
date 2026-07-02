# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Characterization oracle for deferred pickling.

Pins today's observable behavior: a mesh reference that is still *pending*
(spawned and sent in the same breath, before its background init finishes)
arrives at the other side as a usable reference, on both send paths: as an
endpoint argument (request path) and as an endpoint return value (reply path).
Today this rides the deferred-pickle subsystem (resolve-before-send plus a
re-pickle); a change to how pending references serialize must preserve these
outcomes. This is the safety net such a change checks against.

There is no hook to force the pending path; it relies on the
spawn-then-immediately-send race, which the synchronous pickle wins in practice
(the async mesh init cannot complete in the gap). Every test asserts the
end-to-end outcome. The argument tests also assert the reserve counter: a slot
is reserved only for a mesh still pending at pickle, so a nonzero count proves
the risky send-side pending path ran instead of resolving inline. That leans on
the race timescale (init is milliseconds, the synchronous gap microseconds); if
it ever flips the assertion goes red rather than passing hollow, and the
hardening then is a barrier holding init until after the pickle. The return
tests assert the pop counter, which proves the reply's reference reunited from
the out-of-band table (received and reconstructed), not that it was pending: a
resolved mesh also travels out-of-band, so pop cannot distinguish. Return-side
pending-ness happens in the spawner subprocess, invisible here, so it stays
outcome-only.
"""

from __future__ import annotations

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._rust_bindings.monarch_hyperactor.pickle import (
    _get_mesh_pop_count,
    _get_pending_reserve_count,
    _reset_mesh_pop_count,
    _reset_pending_reserve_count,
)
from monarch._src.actor.actor_mesh import Actor
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.host_mesh import this_host
from monarch._src.job.process import ProcessJob
from scoped_state import scoped_state


class _Target(Actor):
    @endpoint
    async def ping(self) -> str:
        return "pong"


class _Receiver(Actor):
    @endpoint
    async def use_mesh(self, target: _Target) -> list[str]:
        # Use the mesh we were handed (it was pending when it was sent).
        results = await target.ping.call()
        return [value for _, value in results]


class _Spawner(Actor):
    @endpoint
    async def spawn_and_return_pending(self) -> _Target:
        # Spawn a fresh mesh and return it without awaiting init, so it is
        # pending when this reply is pickled.
        return this_host().spawn_procs(name="inner_proc").spawn("inner", _Target)


@pytest.mark.timeout(60)
@isolate_in_subprocess
def test_pending_mesh_sent_as_endpoint_argument_resolves_on_receiver() -> None:
    """Request path: a just-spawned (pending) mesh passed as an endpoint argument
    reaches the receiver as a working reference."""
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        receiver = host.spawn_procs(name="recv_proc").spawn("receiver", _Receiver)
        # Spawned and sent in the same breath: pending at pickle time.
        target = host.spawn_procs(name="target_proc").spawn("target", _Target)
        _reset_pending_reserve_count()
        result = receiver.use_mesh.call_one(target).get()
        assert result == ["pong"]
        # A slot is reserved only for a still-pending mesh (a resolved one fills
        # directly), so a nonzero count proves the send-side pending path ran.
        assert _get_pending_reserve_count() > 0


@pytest.mark.timeout(60)
@isolate_in_subprocess
def test_pending_mesh_returned_from_endpoint_resolves_on_caller() -> None:
    """Reply path: a just-spawned (pending) mesh returned from an endpoint reaches
    the caller as a working reference."""
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        spawner = host.spawn_procs(name="spawner_proc").spawn("spawner", _Spawner)
        _reset_mesh_pop_count()
        returned = spawner.spawn_and_return_pending.call_one().get()
        # Pop proves the reply's ref reunited from the out-of-band table
        # (received and reconstructed); a resolved mesh rides out-of-band too,
        # so this does not prove pending-ness.
        assert _get_mesh_pop_count() > 0
        result = returned.ping.call_one().get()
        assert result == "pong"


class _ProcReceiver(Actor):
    @endpoint
    async def use_proc_mesh(self, proc_mesh) -> list[str]:
        # Use the proc mesh we were handed (it was pending when it was sent).
        actors = proc_mesh.spawn("inner", _Target)
        results = await actors.ping.call()
        return [value for _, value in results]


class _ProcSpawner(Actor):
    @endpoint
    async def spawn_and_return_pending_proc(self):
        # Return a fresh proc mesh without awaiting init, so it is pending when
        # this reply is pickled.
        return this_host().spawn_procs(name="inner_proc")


@pytest.mark.timeout(60)
@isolate_in_subprocess
def test_pending_proc_mesh_sent_as_endpoint_argument_resolves_on_receiver() -> None:
    """Request path: a just-spawned (pending) proc mesh passed as an endpoint
    argument reaches the receiver as a working reference."""
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        receiver = host.spawn_procs(name="recv_proc").spawn("receiver", _ProcReceiver)
        # Spawned and sent in the same breath: pending at pickle time.
        proc_mesh = host.spawn_procs(name="target_proc")
        _reset_pending_reserve_count()
        result = receiver.use_proc_mesh.call_one(proc_mesh).get()
        assert result == ["pong"]
        # A slot is reserved only for a still-pending proc mesh (a resolved one
        # fills directly), so this proves the send-side pending path ran.
        assert _get_pending_reserve_count() > 0


@pytest.mark.timeout(60)
@isolate_in_subprocess
def test_pending_proc_mesh_returned_from_endpoint_resolves_on_caller() -> None:
    """Reply path: a just-spawned (pending) proc mesh returned from an endpoint
    reaches the caller as a working reference."""
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        spawner = host.spawn_procs(name="spawner_proc").spawn("spawner", _ProcSpawner)
        _reset_mesh_pop_count()
        returned = spawner.spawn_and_return_pending_proc.call_one().get()
        # Pop proves the reply's ref reunited from the out-of-band table
        # (received and reconstructed), not that it was pending.
        assert _get_mesh_pop_count() > 0
        actors = returned.spawn("t", _Target)
        result = actors.ping.call_one().get()
        assert result == "pong"


@pytest.mark.timeout(60)
@isolate_in_subprocess
def test_pending_mesh_returned_through_call_valuemesh_resolves_on_caller() -> None:
    """Reply path via .call(): a just-spawned (pending) mesh returned from an
    endpoint and collected into a valuemesh reaches the caller as a working
    reference. Exercises the accumulated-overlay decode, where per-rank refs
    must survive the RLE boundary."""
    with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
        host = state.hosts
        spawner = host.spawn_procs(name="spawner_proc").spawn("spawner", _Spawner)
        _reset_mesh_pop_count()
        returned = spawner.spawn_and_return_pending.call().get()
        # Pop proves each per-rank reply's ref reunited through the valuemesh
        # decode (received and reconstructed), not that it was pending.
        assert _get_mesh_pop_count() > 0
        for _, mesh in returned:
            assert mesh.ping.call_one().get() == "pong"
