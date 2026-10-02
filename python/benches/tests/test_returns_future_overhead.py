# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import statistics
import threading
import unittest
import unittest.mock
from typing import Any

from monarch._src.actor.future import Future
from monarch.python.benches import returns_future_overhead as overhead

# Longer than any healthy call here takes.
_DEADLINE_S = 10.0


def setUpModule() -> None:
    from monarch._src.actor.actor_mesh import context

    context()


def _stats(cpu: list[float], gate: list[float]) -> overhead._Stats:
    return overhead._Stats(
        0.0,
        0.0,
        statistics.median(cpu),
        statistics.median(gate),
        overhead._cpu_without_gate(cpu, gate),
    )


class _Raising(overhead._Ready):
    def results(self) -> Future[Any]:
        raise SystemExit("from results")


class _Blocking(overhead._Ready):
    def __init__(self, release: threading.Event) -> None:
        self.release = release

    def results(self) -> Future[Any]:
        self.release.wait(_DEADLINE_S)
        return super().results()


class ReturnsFutureOverheadTest(unittest.TestCase):
    def test_delta_without_gate_bounds_paired_batches(self) -> None:
        # B's middle batch spent up to 9 of its 20 on the gate, so B's median
        # excluding the gate can be as low as 11, against A's 10.
        a = _stats([10.0, 10.0, 10.0], [0.0, 0.0, 0.0])
        b = _stats([10.0, 20.0, 30.0], [0.0, 9.0, 0.0])
        self.assertEqual(overhead._delta_without_gate(a, b), (1.0, 10.0))

    def test_cold_sample_reraises_a_base_exception(self) -> None:
        with self.assertRaises(SystemExit):
            overhead._cold_samples(_Raising(), 1)

    def test_cold_sample_gives_up_on_a_call_that_does_not_return(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)
        outcome: list[BaseException] = []

        def sample() -> None:
            try:
                overhead._cold_samples(_Blocking(release), 1)
            except BaseException as e:  # noqa: B036 - reported to the test
                outcome.append(e)

        with unittest.mock.patch.object(overhead, "_COLD_JOIN_SECONDS", 0.2):
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            sampler.join(_DEADLINE_S)
        self.assertFalse(sampler.is_alive(), "_cold_samples did not give up")
        self.assertIsInstance(outcome[0], TimeoutError)
