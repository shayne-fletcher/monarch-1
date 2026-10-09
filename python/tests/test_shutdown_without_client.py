# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import concurrent.futures
from typing import cast

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._src.actor import actor_mesh as actor_mesh_module
from monarch.actor import shutdown_context


# OSS CI reuses a worker across files; these tests require a fresh client state.
@isolate_in_subprocess
def test_shutdown_without_client_creates_no_client() -> None:
    assert actor_mesh_module._client_context.try_get() is None

    assert shutdown_context().get(timeout=30) is None

    assert actor_mesh_module._client_context.try_get() is None
    assert not actor_mesh_module._shutdown_claimed


@isolate_in_subprocess
def test_shutdown_waits_for_client_bootstrap_in_progress() -> None:
    client_context = actor_mesh_module._client_context
    assert client_context.try_get() is None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Holding the lock stands in for another thread bootstrapping the client.
        with client_context._lock:
            result = executor.submit(lambda: shutdown_context().get(timeout=30))
            with pytest.raises(concurrent.futures.TimeoutError):
                result.result(timeout=0.5)
        assert result.result(timeout=30) is None

    assert client_context.try_get() is None
    assert not actor_mesh_module._shutdown_claimed


@isolate_in_subprocess
def test_shutdown_on_actor_thread_without_client_is_inert() -> None:
    # An actor endpoint in a worker process has a Monarch context but no client.
    stand_in = cast(actor_mesh_module.Context, object())
    token = actor_mesh_module._context.set(stand_in)
    try:
        assert actor_mesh_module._context.get() is stand_in
        assert actor_mesh_module._client_context.try_get() is None
        result = shutdown_context()
    finally:
        actor_mesh_module._context.reset(token)

    assert result.get(timeout=30) is None
    assert actor_mesh_module._client_context.try_get() is None
    assert not actor_mesh_module._shutdown_claimed
