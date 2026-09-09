# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import operator
import sys
import unittest.mock

import pytest
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask
from monarch._src.actor import future as future_mod
from monarch._src.actor.future import Future
from monarch.actor import Accumulator


class _CombineError(Exception):
    pass


class _StreamError(Exception):
    pass


def _future_of(value):
    """A real, single-use monarch Future resolving to value."""

    async def _v():
        return value

    return Future._from_coro(_v())


def _stream_endpoint(values):
    """A fake Endpoint whose .stream() yields one real Future per value."""
    ep = unittest.mock.MagicMock()
    ep.stream.return_value = iter([_future_of(v) for v in values])
    return ep


def _recording_stream_endpoint(values, events, *, make_future=_future_of):
    """A fake Endpoint that records stream calls and iterator advancement."""
    ep = unittest.mock.MagicMock()

    def stream(*args, **kwargs):
        events.append(("stream", args, kwargs))

        def iterator():
            for value in values:
                events.append(("next", value))
                yield make_future(value)

        return iterator()

    ep.stream.side_effect = stream
    return ep


async def _await_result(future):
    return await future


async def _capture_error(future, error_type):
    with pytest.raises(error_type) as caught:
        await future
    return caught.value


def _failing_combine():
    def combine(total, value):
        if value == 2:
            raise _CombineError("combine failed on 2")
        return total + value

    return unittest.mock.MagicMock(side_effect=combine)


def test_accumulate_folds_with_combine():
    """accumulate reduces the per-rank stream through combine from identity."""
    acc = Accumulator(_stream_endpoint([1, 2, 3]), 0, operator.add)
    assert acc.accumulate().get() == 6


def test_accumulate_empty_stream_returns_identity():
    """With no streamed values, accumulate returns the identity seed."""
    acc = Accumulator(_stream_endpoint([]), 42, operator.add)
    assert acc.accumulate().get() == 42


def test_accumulate_folds_left_to_right():
    """combine is applied left-to-right over the stream, seeded by identity."""
    acc = Accumulator(_stream_endpoint([1, 2, 3]), [], lambda a, r: a + [r])
    assert acc.accumulate().get() == [1, 2, 3]


def test_accumulate_forwards_args_to_stream():
    """accumulate forwards its args/kwargs to endpoint.stream()."""
    ep = _stream_endpoint([1])
    Accumulator(ep, 0, operator.add).accumulate("a", k=1).get()
    ep.stream.assert_called_once_with("a", k=1)


@pytest.mark.timeout(10)
def test_accumulate_invokes_stream_before_any_observation():
    events = []
    endpoint = _recording_stream_endpoint([2], events)
    combine = unittest.mock.MagicMock(side_effect=operator.add)
    accumulator = Accumulator(endpoint, 1, combine)

    unobserved = accumulator.accumulate()
    # The private state check is intentional: the event assertions alone could
    # race a producer that started eagerly but has not advanced the iterator yet.
    assert isinstance(unobserved._status, future_mod._Unawaited)
    assert events == [("stream", (), {})]
    combine.assert_not_called()
    del unobserved
    assert events == [("stream", (), {})]
    combine.assert_not_called()

    assert accumulator.accumulate().get() == 3
    assert events == [
        ("stream", (), {}),
        ("stream", (), {}),
        ("next", 2),
    ]
    combine.assert_called_once_with(1, 2)


@pytest.mark.timeout(10)
def test_accumulate_short_circuits_and_propagates_combine_error():
    events = []
    combine = _failing_combine()
    endpoint = _recording_stream_endpoint([1, 2, 3], events)
    result = Accumulator(endpoint, 0, combine).accumulate()

    with pytest.raises(_CombineError) as caught:
        result.get()

    assert type(caught.value) is _CombineError
    assert str(caught.value) == "combine failed on 2"
    assert combine.call_args_list == [
        unittest.mock.call(0, 1),
        unittest.mock.call(1, 2),
    ]
    assert events == [
        ("stream", (), {}),
        ("next", 1),
        ("next", 2),
    ]


@pytest.mark.timeout(10)
def test_accumulate_short_circuits_and_propagates_stream_error():
    events = []

    def make_future(value):
        if value != 2:
            return _future_of(value)

        async def fail():
            raise _StreamError("stream failed on 2")

        return Future._from_coro(fail())

    combine = unittest.mock.MagicMock(side_effect=operator.add)
    endpoint = _recording_stream_endpoint([1, 2, 3], events, make_future=make_future)
    result = Accumulator(endpoint, 0, combine).accumulate()

    with pytest.raises(_StreamError) as first:
        result.get()
    second = asyncio.run(_capture_error(result, _StreamError))

    assert type(first.value) is _StreamError
    assert type(second) is _StreamError
    assert str(first.value) == "stream failed on 2"
    assert str(second) == "stream failed on 2"
    assert second is first.value
    assert combine.call_args_list == [unittest.mock.call(0, 1)]
    assert events == [
        ("stream", (), {}),
        ("next", 1),
        ("next", 2),
    ]


@pytest.mark.timeout(10)
def test_accumulate_get_then_await_does_not_rerun_combine():
    combine = unittest.mock.MagicMock(side_effect=operator.add)
    result = Accumulator(_stream_endpoint([1, 2, 3]), 0, combine).accumulate()

    assert result.get() == 6
    assert asyncio.run(_await_result(result)) == 6
    assert combine.call_args_list == [
        unittest.mock.call(0, 1),
        unittest.mock.call(1, 2),
        unittest.mock.call(3, 3),
    ]


@pytest.mark.timeout(10)
def test_accumulate_await_then_get_does_not_rerun_combine():
    combine = unittest.mock.MagicMock(side_effect=operator.add)
    result = Accumulator(_stream_endpoint([1, 2, 3]), 0, combine).accumulate()

    assert asyncio.run(_await_result(result)) == 6
    assert result.get() == 6
    assert combine.call_args_list == [
        unittest.mock.call(0, 1),
        unittest.mock.call(1, 2),
        unittest.mock.call(3, 3),
    ]


@pytest.mark.timeout(10)
def test_accumulate_replays_combine_error_get_then_await():
    combine = _failing_combine()
    result = Accumulator(_stream_endpoint([1, 2, 3]), 0, combine).accumulate()

    with pytest.raises(_CombineError) as first:
        result.get()
    second = asyncio.run(_capture_error(result, _CombineError))

    assert type(first.value) is _CombineError
    assert type(second) is _CombineError
    assert str(first.value) == "combine failed on 2"
    assert str(second) == "combine failed on 2"
    assert second is first.value
    assert combine.call_args_list == [
        unittest.mock.call(0, 1),
        unittest.mock.call(1, 2),
    ]


@pytest.mark.timeout(10)
def test_accumulate_replays_combine_error_await_then_get():
    combine = _failing_combine()
    result = Accumulator(_stream_endpoint([1, 2, 3]), 0, combine).accumulate()

    first = asyncio.run(_capture_error(result, _CombineError))
    with pytest.raises(_CombineError) as second:
        result.get()

    assert type(first) is _CombineError
    assert type(second.value) is _CombineError
    assert str(first) == "combine failed on 2"
    assert str(second.value) == "combine failed on 2"
    assert combine.call_args_list == [
        unittest.mock.call(0, 1),
        unittest.mock.call(1, 2),
    ]


@pytest.mark.timeout(10)
def test_accumulate_post_start_abandonment_completes_fold_once():
    gate = {"entered": False, "released": False, "completed": False}

    async def gated_value():
        gate["entered"] = True
        while not gate["released"]:
            await PythonTask.sleep(0.005)
        gate["completed"] = True
        return 2

    endpoint = unittest.mock.MagicMock()
    endpoint.stream.return_value = iter([Future._from_coro(gated_value())])
    combine = unittest.mock.MagicMock(side_effect=operator.add)
    result = Accumulator(endpoint, 1, combine).accumulate()

    async def cancel_after_start():
        observer = result.as_asyncio()

        async def wait_for_entry():
            while not gate["entered"]:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_entry(), timeout=5)
        assert not gate["completed"]
        assert observer.cancel()
        # asyncio.Future.cancel() changes the state synchronously; callbacks run
        # on a later loop turn, but cancelled() is true when cancel() returns true.
        assert observer.cancelled()
        gate["released"] = True
        return await asyncio.wait_for(result.as_asyncio(), timeout=5)

    try:
        observed = asyncio.run(cancel_after_start())
    finally:
        original_error = sys.exc_info()[1]
        gate["released"] = True
        try:
            drained = result.get(timeout=5)
        except Exception as cleanup_error:
            if original_error is None:
                raise
            # BaseException.add_note() is unavailable on the Python 3.10 OSS
            # test lane. The original failure still takes precedence there.
            if hasattr(original_error, "add_note"):
                original_error.add_note(f"cleanup also failed: {cleanup_error!r}")

    assert observed == 3
    assert drained == 3
    assert gate["completed"]
    combine.assert_called_once_with(1, 2)


@pytest.mark.timeout(10)
def test_accumulate_empty_stream_makes_no_combine_call():
    combine = unittest.mock.MagicMock(side_effect=operator.add)
    result = Accumulator(_stream_endpoint([]), 42, combine).accumulate()

    assert result.get() == 42
    combine.assert_not_called()
