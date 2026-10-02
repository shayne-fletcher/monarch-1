#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-call cost of `@returns_future` against the same job as plain synchronous
code.

Both versions wait for a result list, take the largest exit code, and wait for a
cleanup Future in a `finally`: the shape of the end of `job.exec_command`. The
difference is the whole cost of `@returns_future`: the call, starting the body
on the thread's cached event loop, and, for each `await` of a pending Future,
an asyncio wake in place of a blocking wait.

The comparison is equivalent to::

    def f(g):
        value = g.get()
        return sync_python(value)

    @returns_future
    async def f_prime(g):
        value = await g
        return sync_python(value)

    f(g)
    f_prime(g).get()

Within each regime, both versions receive equivalent fresh Futures and perform
identical synchronous Python work. The pending and ready regimes use
Handle-backed Futures; `actors` deliberately exercises the initially
task-backed Future returned by an endpoint call. The measured difference is
therefore the cost of expressing and synchronously observing that work through
``@returns_future``.

`--regimes` chooses where the Futures come from:

  - `pending`: Rust-produced Handles that publish only after a waiter is
    registered. Positive delays make both paths wait; 0 ms exercises the
    pending observation and wake path without a timer. This is the primary
    measurement.
  - `actors`: endpoint calls on a small proc mesh on this host.
  - `ready`: already-completed Handles. A reads each value through `.get()`'s
    ready fast path; B awaits each through `as_asyncio()`'s ready shortcut,
    which starts no observer and wakes no loop.
"""

from __future__ import annotations

import argparse
import functools
import statistics
import sys
import threading
import time
from typing import Any, Callable, NamedTuple

from monarch._rust_bindings.monarch_hyperactor.handle import _new_handle_pair
from monarch._rust_bindings.monarch_hyperactor.testing import (
    _delayed_handle_gate_stats,
    _make_delayed_handle,
)
from monarch._src.actor import returns_future as returns_future_mod
from monarch._src.actor.actor_mesh import context
from monarch._src.actor.future import Future
from monarch._src.actor.returns_future import returns_future
from monarch.actor import Actor, current_rank, endpoint, this_host

# Four stand-in rank results for the ready and pending versions of the
# `exec_command`-shaped reduction. Both paths must return the maximum, `1`.
_RESULTS: list[dict[str, int]] = [{"returncode": rc} for rc in (0, 1, 0, 0)]
# How many times `cleanup_future` has been called. `_check` resets it before
# each version and confirms one call from each path.
_cleanups = 0
# How long `_cold_samples` waits for one first-use call.
_COLD_JOIN_SECONDS = 60.0


def _ready(value: Any) -> Future[Any]:
    handle, completer = _new_handle_pair()
    completer.set_result(value)
    return Future._from_handle(handle)


class _Ready:
    """Already-completed Handle-backed Futures."""

    name = "ready"
    expected = 1

    def results(self) -> Future[Any]:
        return _ready(_RESULTS)

    def rows(self, value: list[dict[str, int]]) -> list[dict[str, int]]:
        return value

    def cleanup(self) -> Future[Any]:
        return _ready(None)


class _Delayed:
    """Rust-produced Handles that publish after registering their first waiter.

    A positive delay starts once the producer detects that waiter; a zero delay
    publishes without starting a timer. Both versions then use ``_RESULTS``.
    """

    expected = 1

    def __init__(self, delay_ms: float) -> None:
        self.name = f"pending {delay_ms:g} ms"
        self.delay_ms = delay_ms

    def results(self) -> Future[Any]:
        return Future._from_handle(_make_delayed_handle(self.delay_ms / 1000))

    def rows(self, value: None) -> list[dict[str, int]]:
        return _RESULTS

    def cleanup(self) -> Future[Any]:
        return Future._from_handle(_make_delayed_handle(self.delay_ms / 1000))


class ExitCodeActor(Actor):
    """Stands in for `exec_command`'s bash actor. `cleanup` stands in for its
    wait for `procs.stop()`; it is a second wait, not a process stop."""

    @endpoint
    async def run(self) -> dict[str, int]:
        return {"returncode": 1 if current_rank().rank == 1 else 0}

    @endpoint
    async def cleanup(self) -> None:
        return None


class _Actors:
    """Endpoint calls on a proc mesh on this host, spawned once before timing.

    Each call initially returns a task-backed Future. The synchronous path
    drives that task with ``.get()``; the decorated path converts it to a Handle
    when awaited. Spawning processes per call would hide this observation cost.
    """

    name = "actors"

    def __init__(self, ranks: int) -> None:
        self.procs = this_host().spawn_procs(per_host={"procs": ranks})
        self.actor: ExitCodeActor = self.procs.spawn("exit_code", ExitCodeActor)
        self.expected: int = 1 if ranks > 1 else 0

    def results(self) -> Future[Any]:
        return self.actor.run.call()

    def rows(self, value: Any) -> list[dict[str, int]]:
        # A value mesh iterates as `(rank, result)` pairs, as in `exec_command`.
        return [result for _rank, result in value]

    def cleanup(self) -> Future[Any]:
        return self.actor.cleanup.call()

    def stop(self) -> None:
        self.procs.stop().get()


_Source = _Ready | _Delayed | _Actors


def results_future(source: _Source) -> Future[Any]:
    return source.results()


def cleanup_future(source: _Source) -> Future[Any]:
    """Count one cleanup invocation and return a fresh cleanup Future from
    ``source``."""
    global _cleanups
    _cleanups += 1
    return source.cleanup()


def worst_exit_code(results: list[dict[str, int]]) -> int:
    return max(result["returncode"] for result in results)


def sync_job(source: _Source) -> int:
    """Run the representative job using ordinary synchronous control flow.

    Wait for the command results with ``.get()``, reduce them to the largest exit
    code, and wait for cleanup in ``finally``. This is the baseline for comparison
    with the same orchestration expressed through ``@returns_future``.
    """
    try:
        return worst_exit_code(source.rows(results_future(source).get()))
    finally:
        cleanup_future(source).get()


@returns_future
async def async_job(source: _Source) -> int:
    """Run the representative job through ``@returns_future``.

    The body performs the same reduction and guaranteed cleanup as ``sync_job()``,
    using ``await`` instead of ``.get()``. Calling this function returns a
    ``monarch.actor.Future[int]``; the benchmark observes it with ``.get()`` to
    exercise the caller-thread event-loop path.
    """
    try:
        return worst_exit_code(source.rows(await results_future(source)))
    finally:
        await cleanup_future(source)


def _get_async_job(source: _Source) -> int:
    return async_job(source).get()


def _check(source: _Source) -> None:
    """Check both results and one cleanup call before timing either path.

    For a delayed source, also check that ``poll()`` registers no waiter: a
    fresh Handle stays pending past its delay, and its gate never opens.
    """
    global _cleanups
    if isinstance(source, _Delayed):
        _delayed_handle_gate_stats()
        handle = _make_delayed_handle(source.delay_ms / 1000)
        assert handle.poll() is None, "a new delayed Handle should be pending"
        time.sleep(source.delay_ms / 1000 + 0.05)
        gated, _ = _delayed_handle_gate_stats()
        assert gated == 0 and handle.poll() is None, "poll() registered a waiter"
    for job in (sync_job, _get_async_job):
        _cleanups = 0
        result = job(source)
        assert result == source.expected, f"{job.__name__} returned {result}"
        assert _cleanups == 1, f"{job.__name__} made {_cleanups} cleanups"


def _time_calls(
    call: Callable[[], object], count: int
) -> tuple[list[int], float, float]:
    """Measure ``count`` calls and return their wall and CPU cost in nanoseconds.

    The list contains one wall-clock duration per call. The second value is the
    process CPU time for the whole batch divided by ``count``. The third is the
    delayed Handles' gate time for the batch divided by ``count``: an upper
    bound on the gate's share of that CPU, zero for other sources.
    """
    samples = []
    _delayed_handle_gate_stats()
    cpu_start = time.process_time_ns()
    for _ in range(count):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    cpu = (time.process_time_ns() - cpu_start) / count
    _gated, gate_ns = _delayed_handle_gate_stats()
    return samples, cpu, gate_ns / count


def _cold_samples(source: _Source, count: int) -> list[int]:
    """Measure first-use ``async_job().get()`` overhead on fresh threads.

    Each sample starts timing inside a new thread and stops when ``.get()`` returns.
    It includes creating the thread's event loop, plus first use of the Handle
    wake path. It excludes starting, joining, and tearing down the thread.
    """
    samples = []
    for _ in range(count):
        box: list[int | BaseException] = []

        def first_call(box: list[int | BaseException] = box) -> None:
            start = time.perf_counter_ns()
            try:
                async_job(source).get()
            except BaseException as error:  # noqa: B036 - re-raised below
                # Re-raised on the main thread, where the benchmark can report it.
                box.append(error)
                return
            box.append(time.perf_counter_ns() - start)

        # A daemon, so a call that never returns cannot block interpreter exit.
        thread = threading.Thread(target=first_call, daemon=True)
        thread.start()
        thread.join(_COLD_JOIN_SECONDS)
        if thread.is_alive():
            raise TimeoutError(
                f"a first-use call did not return within {_COLD_JOIN_SECONDS} s"
            )
        (outcome,) = box
        if isinstance(outcome, BaseException):
            raise outcome
        samples.append(outcome)
    return samples


def _p50_p99_us(samples: list[int]) -> tuple[float, float]:
    """Summarize call durations as p50 and p99, in microseconds.

    Half the sampled calls completed within p50; 99% completed within p99. Input
    samples are nanoseconds, and percentiles use inclusive interpolation.
    """
    percentiles = statistics.quantiles(samples, n=100, method="inclusive")
    return percentiles[49] / 1000, percentiles[98] / 1000


class _Stats(NamedTuple):
    """One version's results for one source, in microseconds per call."""

    p50: float
    p99: float
    # The median batch CPU.
    cpu: float
    # The median batch gate time, an upper bound on each batch's gate CPU.
    gate: float
    # A lower bound on the median batch CPU excluding the gate; `cpu` is the
    # upper bound.
    cpu_without_gate: float


def _cpu_without_gate(cpu: list[float], gate: list[float]) -> float:
    """A lower bound on the median batch CPU excluding the gate.

    Each batch's gate CPU lies between zero and its gate time, so its CPU
    excluding the gate lies in ``[cpu - gate, cpu]``. The median is monotone in
    each input, so the median excluding the gate lies between the medians of
    those bounds: the median of ``cpu - gate`` here, and of ``cpu``.
    """
    return statistics.median(c - g for c, g in zip(cpu, gate, strict=True))


def _delta_without_gate(a: _Stats, b: _Stats) -> tuple[float, float]:
    """Bounds on B's CPU minus A's, both excluding the gate."""
    return b.cpu_without_gate - a.cpu, b.cpu - a.cpu_without_gate


def _measure(
    source: _Source, iterations: int, rounds: int, warmup: int
) -> tuple[_Stats, _Stats]:
    """Time A and B warm for one source, and return A's results, then B's."""
    # Also prove that both paths return the same value and call cleanup once.
    _check(source)

    # Exercise both paths enough to stabilize their reusable runtime state.
    for _ in range(warmup):
        sync_job(source)
        _get_async_job(source)
    cached = getattr(returns_future_mod._THIS_THREAD, "loop", None)

    # Alternate baseline and decorated batches, and which goes first, so a
    # steady drift in host load lands on both alike.
    jobs = (sync_job, _get_async_job)
    samples: tuple[list[int], list[int]] = ([], [])
    cpu: tuple[list[float], list[float]] = ([], [])
    gate: tuple[list[float], list[float]] = ([], [])
    for round_ in range(rounds):
        for version in (0, 1) if round_ % 2 == 0 else (1, 0):
            batch, batch_cpu, batch_gate = _time_calls(
                functools.partial(jobs[version], source), iterations
            )
            samples[version].extend(batch)
            cpu[version].append(batch_cpu)
            gate[version].append(batch_gate)

    # Warm calls must keep reusing this thread's one private loop; otherwise
    # their measurements would include loop construction.
    assert cached is not None
    assert returns_future_mod._THIS_THREAD.loop is cached, (
        "warm calls should reuse one loop"
    )

    # Wall percentiles use every call. CPU is measured per batch, so report the
    # median batch average to reduce the influence of an anomalous round.
    a, b = (
        _Stats(
            *_p50_p99_us(samples[v]),
            statistics.median(cpu[v]) / 1000,
            statistics.median(gate[v]) / 1000,
            _cpu_without_gate(cpu[v], gate[v]) / 1000,
        )
        for v in (0, 1)
    )
    return a, b


def main() -> None:
    """Measure and report synchronous and ``@returns_future`` orchestration costs.

    For each requested source, validate both jobs, warm their reusable state,
    and collect steady-state samples in rounds that alternate which version runs
    first. When the run includes the 0 ms pending source, also measure
    first-use overhead on fresh threads with 0 ms delayed Handles. Report
    wall-clock percentiles, process CPU per call, the delayed Handles' gate
    bounds, and the corresponding deltas.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regimes", default="pending,actors")
    parser.add_argument("--delays-ms", default="0,1,10")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--cold-samples", type=int, default=200)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    if args.rounds <= 0 or args.rounds % 2:
        parser.error("--rounds must be an even positive number")
    if args.cold_samples < 2:
        parser.error("--cold-samples must be at least 2")
    regimes = args.regimes.split(",")
    delays = [float(d) for d in args.delays_ms.split(",")]

    # Initialize the Monarch client and Tokio runtime before any timed samples.
    context()

    results: list[tuple[str, _Stats, _Stats]] = []
    warm_p50_at_zero = None
    for regime in regimes:
        sources: list[_Source]
        if regime == "pending":
            sources = [_Delayed(delay) for delay in delays]
        elif regime == "actors":
            sources = [_Actors(args.ranks)]
        elif regime == "ready":
            sources = [_Ready()]
        else:
            raise SystemExit(f"unknown regime {regime!r}")
        for source in sources:
            # A delayed call waits twice, so fewer calls keep each run short.
            scale = 1 + source.delay_ms if isinstance(source, _Delayed) else 1
            try:
                a, b = _measure(
                    source,
                    max(1, int(args.iterations / scale)),
                    args.rounds,
                    max(1, int(args.warmup / scale)),
                )
            finally:
                if isinstance(source, _Actors):
                    source.stop()
            results.append((source.name, a, b))
            if isinstance(source, _Delayed) and source.delay_ms == 0:
                warm_p50_at_zero = b.p50

    print(f"python {sys.version.split()[0]}")
    print(
        f"{args.rounds} rounds per version, alternating which runs first; per "
        f"round, {args.iterations} calls, divided by 1 + the delay in ms for "
        f"pending sources"
    )

    # Gate time is shown beside wall time as a diagnostic only; it is not
    # separable wall overhead.
    print("\nwall time per call, us")
    print(
        f"{'':26}{'A p50':>9}{'A p99':>9}{'B p50':>9}{'B p99':>9}"
        f"{'B-A p50':>10}{'B-A %':>8}{'gate A':>9}{'gate B':>9}"
    )
    for name, a, b in results:
        print(
            f"{name:26}{a.p50:9.1f}{a.p99:9.1f}{b.p50:9.1f}{b.p99:9.1f}"
            f"{b.p50 - a.p50:+10.1f}{(b.p50 - a.p50) / a.p50:+8.1%}"
            f"{a.gate:9.1f}{b.gate:9.1f}"
        )

    # Bounded per batch, then by median: see `_cpu_without_gate`.
    print("\nprocess CPU per call, us")
    print(f"{'':26}{'A':>9}{'B':>9}{'B-A':>9}{'B-A excluding gate':>22}")
    for name, a, b in results:
        delta = b.cpu - a.cpu
        low, high = _delta_without_gate(a, b)
        interval = f"{low:+.1f} .. {high:+.1f}"
        print(f"{name:26}{a.cpu:9.1f}{b.cpu:9.1f}{delta:+9.1f}{interval:>22}")

    # Measure the decorated path's complete first-use cost on fresh threads.
    if warm_p50_at_zero is not None:
        cold_p50, cold_p99 = _p50_p99_us(_cold_samples(_Delayed(0), args.cold_samples))
        print(
            f"\nB first use on a new thread, pending 0 ms: p50 {cold_p50:.1f} us, "
            f"p99 {cold_p99:.1f} us"
        )
        print(
            f"first-use overhead (B cold - B warm): "
            f"{cold_p50 - warm_p50_at_zero:+.1f} us p50"
        )


if __name__ == "__main__":
    main()
