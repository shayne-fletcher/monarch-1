# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import sys

from monarch.actor_mesh import Actor, endpoint
from monarch.proc_mesh import proc_mesh


class SegfaultActor(Actor):
    """An actor that has endpoints cause segfaults."""

    @endpoint
    async def cause_segfault(self) -> None:
        """Endpoint that causes a segmentation fault."""
        # Create a C function pointer to an invalid memory address
        # This will reliably cause a segmentation fault when called
        function_type = ctypes.CFUNCTYPE(None)
        # Use a non-zero but invalid address to avoid ctypes null pointer checks
        invalid_address = 0xDEADBEEF
        invalid_function = function_type(invalid_address)
        # Calling this function will cause a segfault
        invalid_function()


class SegfaultActorSync(Actor):
    """An actor that has endpoints cause segfaults."""

    @endpoint  # pyre-ignore
    def cause_segfault(self) -> None:
        """Endpoint that causes a segmentation fault."""
        # Create a C function pointer to an invalid memory address
        # This will reliably cause a segmentation fault when called
        function_type = ctypes.CFUNCTYPE(None)
        # Use a non-zero but invalid address to avoid ctypes null pointer checks
        invalid_address = 0xDEADBEEF
        invalid_function = function_type(invalid_address)
        # Calling this function will cause a segfault
        invalid_function()


def _run_segfault_test_sync(num_procs, sync_endpoint):
    proc = proc_mesh(gpus=num_procs).get()
    if sync_endpoint:
        actor_class = SegfaultActorSync
    else:
        actor_class = SegfaultActor
    segfault_actor = proc.spawn("segfault_actor", actor_class).get()

    # This output is checked in the test to make sure that the process actually got here
    print("I actually ran")
    sys.stdout.flush()

    # Exercise both call() and call_one() in our tests, to check that error
    # aggregation behavior is consistent.
    if num_procs == 1:
        segfault_actor.cause_segfault.call_one().get()
    else:
        segfault_actor.cause_segfault.call().get()


def _run_segfault_test(num_procs, sync_endpoint):
    import asyncio

    if sync_endpoint:
        actor_class = SegfaultActorSync
    else:
        actor_class = SegfaultActor

    async def run_test():
        proc = await proc_mesh(gpus=num_procs)
        segfault_actor = await proc.spawn("segfault_actor", actor_class)

        # This output is checked in the test to make sure that the process actually got here
        print("I actually ran")
        sys.stdout.flush()

        # Exercise both call() and call_one() in our tests, to check that error
        # aggregation behavior is consistent.
        if num_procs == 1:
            await segfault_actor.cause_segfault.call_one()
        else:
            await segfault_actor.cause_segfault.call()

    asyncio.run(run_test())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-procs", type=int)
    parser.add_argument("--sync-test-impl", type=bool)
    parser.add_argument("--sync-endpoint", type=bool)
    args = parser.parse_args()

    print(
        f"Running segfault test: {args.num_procs=} {args.sync_test_impl=} {args.sync_endpoint=}"
    )

    if args.sync_test_impl:
        _run_segfault_test_sync(args.num_procs, args.sync_endpoint)
    else:
        _run_segfault_test(args.num_procs, args.sync_endpoint)


if __name__ == "__main__":
    main()
