# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pytest
from monarch.actor import Channel


@pytest.mark.timeout(10)
def test_recv_futures_observed_in_reverse_receive_in_observation_order() -> None:
    """A public receive claims its message when observed, not when created."""
    sender, receiver = Channel[int].open()
    sender.send(11)
    sender.send(22)

    first = receiver.recv()
    second = receiver.recv()

    assert second.get(timeout=2) == 11
    assert first.get(timeout=2) == 22


@pytest.mark.timeout(10)
def test_discarded_regular_recv_claims_nothing() -> None:
    """Dropping an unobserved regular receive leaves its message pending."""
    sender, receiver = Channel[int].open()
    sender.send(31)

    discarded = receiver.recv()
    del discarded

    assert receiver.recv().get(timeout=2) == 31


@pytest.mark.timeout(10)
def test_discarded_once_recv_at_public_boundary_claims_nothing() -> None:
    """The public wrapper defers even though native once-receive does not."""
    sender, receiver = Channel[int].open(once=True)
    sender.send(41)

    discarded = receiver.recv()
    del discarded

    assert receiver.recv().get(timeout=2) == 41
