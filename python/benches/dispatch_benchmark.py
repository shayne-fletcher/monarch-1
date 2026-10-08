# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure queue-dispatch cost: the latency of one outstanding noop call, and
how a burst of concurrent calls gets into an actor whose event loop is busy."""

from __future__ import annotations

import argparse
import asyncio
import collections
import time

from monarch.actor import Actor, concurrent_endpoint, endpoint, this_host
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
        benchmark_noop_rpc(procs.spawn("sync_noop", SyncNoopActor), args.iterations)
        asyncio.run(benchmark_burst(procs.spawn("busy", BusyLoopActor), args.burst))
    finally:
        procs.stop().get()


if __name__ == "__main__":
    main()
