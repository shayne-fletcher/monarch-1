# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, final

from monarch._rust_bindings.monarch_hyperactor.handle import Handle

@final
class TestStruct:
    """Minimal Rust struct for testing @rust_struct mixin patching."""

    def __init__(self, value: int) -> None: ...
    def rust_method(self) -> int: ...
    def shared_method(self) -> str: ...

def _make_test_struct(value: int) -> Any: ...

# The value a successful probe publishes.
_PROBE_SUCCESS_VALUE: int

@final
class _HandleProbe:
    """Closed control object over one Rust-produced ``Handle``.

    Private contract-test support. The handle comes from the real
    ``PyHandle::spawn`` path, not from ``PythonTask.spawn_handle()``. The probe
    accepts no coroutine, awaitable, callable, future or producer function: it
    only puts a genuine ``Handle`` into one of three reviewed terminal states.

    The probe exposes the permanent Handle type used by production bindings.
    """

    @property
    def _handle(self) -> Handle[Any]: ...
    @property
    def _started(self) -> bool: ...
    @property
    def _completed(self) -> bool: ...
    def _release(self) -> None: ...
    def _wait_completed(self) -> None: ...

def _make_handle_probe(outcome: str) -> _HandleProbe:
    """Build a probe whose handle reaches ``outcome`` once released.

    ``outcome`` is one of ``"success"``, ``"exception"`` (an ordinary
    ``Exception``) or ``"base_exception"``.
    """
    ...

def _make_delayed_handle(delay: float) -> Handle[None]:
    """Build a ``Handle`` that publishes after registering its first waiter.

    The producer waits at least ``delay`` seconds after detecting ``get()``,
    ``as_asyncio()``, or ``await``; ``poll()`` does not register a waiter.
    Private benchmark support. Raises ``ValueError`` for a negative or non-finite
    ``delay``.
    """
    ...

def _delayed_handle_gate_stats() -> tuple[int, int]:
    """Return the number of delayed Handles whose gate opened, and their total
    gate time in nanoseconds, since the last call; then reset both.

    Gate time runs from a producer's first poll until it detects its first
    waiter. It includes any time other tasks ran between its checks, so it is
    an upper bound on the gate's CPU.
    """
    ...
