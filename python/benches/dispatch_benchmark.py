# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure noop calls and stream observations on the main thread and inside sync
actors, and burst admission into an actor whose event loop is busy."""

from __future__ import annotations

import argparse
import asyncio
import collections
import statistics
import time
from collections.abc import Callable

from monarch.actor import Accumulator, Actor, concurrent_endpoint, endpoint, this_host
from monarch.benches.noop_rpc_benchmark import (
    benchmark_noop_rpc,
    NoopActor,
    SyncNoopActor,
)


class BusyLoopActor(Actor):
    """Blocks its event loop for 11 ms per step, then yields once, like an
    inference engine's step loop."""

    def __init__(self) -> None:
        self.running = False
        self.blocks = 0
        self.started: list[int] = []

    @endpoint
    async def start(self) -> None:
        self.running = True
        self.task: asyncio.Task[None] = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while self.running:
            time.sleep(0.011)
            self.blocks += 1
            await asyncio.sleep(0)

    @concurrent_endpoint
    async def req(self, i: int) -> int:
        self.started.append(self.blocks)
        return i

    @endpoint
    async def finish(self) -> list[int]:
        self.running = False
        await self.task
        return self.started


class SyncCaller(Actor):
    """Measure noop replies waited for on a sync actor's driver thread."""

    @endpoint
    def measure(self, peer: SyncNoopActor, iterations: int) -> None:
        benchmark_noop_rpc(peer, iterations)

    @endpoint
    def measure_stream(self, peer: SyncNoopActor, iterations: int) -> None:
        benchmark_stream(peer, iterations)


def benchmark_observation(operation: Callable[[], None], iterations: int) -> None:
    """Report p50 and p99 for completing a noop stream operation, after warmup."""
    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    warmup_iterations = 20
    for _ in range(warmup_iterations):
        operation()
    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        latencies_ms.append((time.perf_counter() - start) * 1000)
    percentiles = statistics.quantiles(latencies_ms, n=100, method="inclusive")
    print(f"warmup iterations: {warmup_iterations}")
    print(f"measured iterations: {iterations}")
    print(f"p50 stream RPC latency: {percentiles[49]:.3f} ms")
    print(f"p99 stream RPC latency: {percentiles[98]:.3f} ms")


def benchmark_stream(peer: SyncNoopActor, iterations: int) -> None:
    """Measure untimed get() on each item, including the off-main inline path
    when called from SyncCaller."""

    def observe() -> None:
        for reply in peer.noop.stream():
            reply.get()

    benchmark_observation(observe, iterations)


def benchmark_accumulator(peer: SyncNoopActor, iterations: int) -> None:
    """Measure the stream's await path inside an Accumulator's private loop."""
    accumulator = Accumulator(peer.noop, None, lambda accumulated, _: accumulated)

    def observe() -> None:
        accumulator.accumulate().get()

    benchmark_observation(observe, iterations)


async def benchmark_burst(busy: BusyLoopActor, count: int) -> None:
    """Send `count` concurrent calls to `busy` at once and report how they got
    in."""
    await busy.start.call_one()
    begin = time.perf_counter()
    await asyncio.gather(*(busy.req.call_one(i) for i in range(count)))
    admission_ms = (time.perf_counter() - begin) * 1000
    started = await busy.finish.call_one()
    per_block = collections.Counter(started)
    print(f"burst size: {count}")
    print(f"burst admission: {admission_ms:.1f} ms")
    print(f"burst span: {max(per_block) - min(per_block) + 1} blocks")
    print(f"burst largest block: {max(per_block.values())} bodies")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--burst", type=int, default=256)
    args = parser.parse_args()
    procs = this_host().spawn_procs(per_host={"procs": 1})
    try:
        print("async endpoint:")
        benchmark_noop_rpc(procs.spawn("noop", NoopActor), args.iterations)
        print("sync endpoint:")
        sync_noop = procs.spawn("sync_noop", SyncNoopActor)
        benchmark_noop_rpc(sync_noop, args.iterations)
        print("nested sync-actor endpoint:")
        caller = procs.spawn("sync_caller", SyncCaller)
        caller.measure.call_one(sync_noop, args.iterations).get()
        print("main-thread stream get:")
        benchmark_stream(sync_noop, args.iterations)
        print("nested sync-actor stream get:")
        caller.measure_stream.call_one(sync_noop, args.iterations).get()
        print("main-thread Accumulator stream await:")
        benchmark_accumulator(sync_noop, args.iterations)
        asyncio.run(benchmark_burst(procs.spawn("busy", BusyLoopActor), args.burst))
    finally:
        procs.stop().get()


if __name__ == "__main__":
    main()
