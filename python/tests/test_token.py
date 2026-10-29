# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test connecting to existing actor meshes through a shared token string.

Most cases start `token_owner.py` in a separate Python process. That helper
creates a `Probe` actor and publishes its mesh. This test process reads the
token from a temporary file, connects with `connect(token, Probe)`, and calls
the probe's endpoints. A fresh interpreter avoids inheriting an actor runtime
from the test process through a fork.

The probe exposes a value stored in the helper process and can run trusted
test functions there. We check its PID, change the value through one
connection, and read it through another. Both connections originate from
this test process, not two independent client processes. The helper also
records each connection notification.

Two helper modes distinguish the token owner's lifetime from its process:

- `process`: publication happens outside an actor. Monarch's built-in client
  actor owns the token; exiting the helper ends it.
- `actor`: a `Publisher` actor owns the token. The test stops that actor while
  leaving the helper process running, then checks that connections time out.

Files carry the token, connection notifications, and stop acknowledgements;
stdin carries commands to the helper. Startup and cleanup waits are bounded.
Cleanup requests exit; on timeout it kills the child and makes a bounded wait
to reap it. It closes stdin even if cleanup fails.

Other cases check malformed tokens and invalid timeouts. A separate two-rank
test publishes a mesh directly and checks that connecting preserves both
actor ranks and selection by a dimension label.
"""

import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, NamedTuple

import pytest
import token_owner
from monarch._src.actor.token import connect, publish
from monarch.actor import this_host
from token_owner import Probe

# Seconds allowed for helper startup, ordinary endpoint calls, and each
# cleanup wait.
WAIT: float = 30.0


class Owner(NamedTuple):
    """The helper process, its control files, and the token it published."""

    # Child interpreter hosting `Probe` and, in actor mode, `Publisher`.
    process: subprocess.Popen[str]
    # Temporary directory shared by this test and the helper through file paths.
    directory: str
    # Text read from `directory/token`, passed to `connect()`.
    token: str


def _launch(mode: Literal["process", "actor"], directory: str) -> subprocess.Popen[str]:
    """Start the publishing helper in an independent Python process.

    `mode` chooses who publishes: the helper's built-in client actor or its
    `Publisher` actor. `directory` holds the files used to talk to the test.
    """
    env = dict(os.environ)
    if "FB_XAR_INVOKED_NAME" in os.environ:
        # Meta's packaged test executable (PAR) carries Python and its
        # dependencies. Re-enter it with the helper as its main module.
        command = [sys.argv[0], mode, directory]
        env["PAR_MAIN_OVERRIDE"] = "token_owner"
        # Keep the helper and its dependencies importable in the child.
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
    else:
        # Outside a packaged test, use this interpreter to run the helper file.
        command = [
            sys.executable,
            os.path.abspath(token_owner.__file__),
            mode,
            directory,
        ]
    # stdin is the control channel. Leave stdout and stderr inherited so
    # helper errors appear in the test's output rather than a hidden pipe.
    return subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.PIPE,
        text=True,
    )


def _wait_for_file(path: str, process: subprocess.Popen[str]) -> str:
    """Wait for a nonempty helper file and return its text.

    Used for the published token and the acknowledgement that `Publisher`
    stopped. While waiting, fail early if the helper exits; otherwise fail
    after `WAIT` seconds.
    """
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path) as f:
                return f.read().strip()
        assert process.poll() is None, f"the owner exited with {process.returncode}"
        time.sleep(0.1)
    raise AssertionError(f"{path} did not appear within {WAIT}s")


def _send(owner: Owner, command: str) -> None:
    """Send one command to the helper through its stdin, flushing it immediately."""
    assert owner.process.stdin is not None
    owner.process.stdin.write(command + "\n")
    owner.process.stdin.flush()


@contextmanager
def _owner(mode: Literal["process", "actor"]) -> Iterator[Owner]:
    """Yield the helper's published token, then stop and reap the helper.

    The token file acknowledges publication. Cleanup runs even if startup
    fails before yielding. Each graceful-exit and forced-reap wait is bounded
    by `WAIT`; stdin is closed even if either wait fails.
    """
    with tempfile.TemporaryDirectory() as directory:
        process = _launch(mode, directory)
        try:
            token = _wait_for_file(os.path.join(directory, "token"), process)
            yield Owner(process, directory, token)
        finally:
            try:
                # Ask the helper to exit and wait for it, rather than leaving
                # the publishing process alive after this test finishes.
                process.communicate("exit\n", timeout=WAIT)
            except subprocess.TimeoutExpired:
                # Force-stop our unresponsive child, then reap it within
                # another deadline. Do not signal a process group after reaping.
                process.kill()
                process.wait(timeout=WAIT)
            finally:
                # `communicate()` can time out before closing its stdin pipe.
                assert process.stdin is not None
                process.stdin.close()


def _peers(owner: Owner) -> list[str]:
    """Read connecting actors' addresses recorded by the helper in process mode.

    Repeated addresses count as separate notifications, not distinct clients.
    A missing file means nothing has been recorded yet; it does not prove that
    no connection was attempted.
    """
    path = os.path.join(owner.directory, "peers")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return f.read().split()


def _wait_for_peers(owner: Owner, count: int) -> None:
    """Wait for exactly `count` recorded notifications, failing on extras.

    Reaching the count also shows that the helper's notification-reporting
    thread is making progress, which an empty-file check cannot establish.
    """
    deadline = time.monotonic() + WAIT
    while len(_peers(owner)) < count and time.monotonic() < deadline:
        time.sleep(0.1)
    assert len(_peers(owner)) == count, _peers(owner)


@pytest.mark.timeout(120)
def test_connections_share_the_owner_state_and_the_owner_sees_each_peer() -> None:
    """Two connections share one remote state and produce two notifications."""
    with _owner("process") as owner:
        first = connect(owner.token, Probe)
        assert first.read.call_one().get(timeout=WAIT) == 0
        # The function executes in the helper, not in this test process.
        assert first.run.call_one(lambda state: os.getpid()).get(timeout=WAIT) == (
            owner.process.pid
        )
        assert first.run.call_one(lambda state: state.set(7)).get(timeout=WAIT) == 0

        # A new connection must see the first connection's change to the
        # helper's `State`, rather than an independently copied value.
        second = connect(owner.token, Probe)
        assert second.read.call_one().get(timeout=WAIT) == 7

        # Both requests come from the same client actor; each is reported.
        _wait_for_peers(owner, 2)


def test_a_malformed_token_is_rejected_before_connecting() -> None:
    """Malformed token text must raise `ValueError`."""
    with pytest.raises(ValueError, match="invalid token"):
        connect("not a token", Probe)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("timeout", [-1.0, math.nan, math.inf, 1e300])
def test_an_invalid_timeout_is_rejected_before_connecting(timeout: float) -> None:
    """Reject invalid deadlines without sending a connection notification.

    Cover negative, non-finite, and unrepresentable durations. A subsequent
    valid connection proves that the helper's notification reader works.
    """
    with _owner("process") as owner:
        with pytest.raises(ValueError, match="invalid timeout"):
            connect(owner.token, Probe, timeout=timeout)
        # An empty file alone could mean a dead reporting thread. Establish
        # its progress with a valid connection before checking for extras.
        connect(owner.token, Probe)
        _wait_for_peers(owner, 1)
        # Allow an extra notification from the rejected call to show up.
        time.sleep(1.0)
        assert len(_peers(owner)) == 1, _peers(owner)


@pytest.mark.timeout(120)
def test_stopping_the_owning_actor_ends_the_token() -> None:
    """Stopping `Publisher` ends its token without ending the helper process."""
    with _owner("actor") as owner:
        probe = connect(owner.token, Probe)
        # Establish that the token works before stopping its publishing actor.
        assert probe.read.call_one().get(timeout=WAIT) == 0
        _send(owner, "stop")
        # The helper writes this only after `Publisher` has finished stopping.
        _wait_for_file(os.path.join(owner.directory, "stopped"), owner.process)
        assert owner.process.poll() is None, "the owner's process keeps running"
        with pytest.raises(TimeoutError):
            connect(owner.token, Probe, timeout=3.0)


@pytest.mark.timeout(120)
def test_a_token_stops_working_when_its_owner_exits() -> None:
    """An exited publishing process leaves a token that cannot accept connections."""
    with _owner("process") as owner:
        _send(owner, "exit")
        # Reap the helper before retrying: this tests an exited process,
        # not a connection racing with its shutdown.
        owner.process.wait(timeout=WAIT)
        with pytest.raises(TimeoutError):
            connect(owner.token, Probe, timeout=3.0)


@pytest.mark.timeout(120)
def test_a_connection_reaches_every_rank_of_the_published_mesh() -> None:
    """Connecting preserves the published ranks and selection by dimension label."""
    # One `Probe` per proc gives each actor rank a distinct process ID.
    probes = this_host().spawn_procs(per_host={"gpus": 2}).spawn("probes", Probe)
    probes.initialized.get()
    published = publish(probes)
    connected = connect(published.token, Probe)
    pids = list(probes.run.call(lambda state: os.getpid()).get(timeout=WAIT).values())
    assert len(set(pids)) == 2
    # Compare against the original mesh in rank order, not just as a set.
    assert (
        list(connected.run.call(lambda state: os.getpid()).get(timeout=WAIT).values())
        == pids
    )
    # Selecting gpus=1 must reach the original mesh's second actor rank.
    assert (
        connected.slice(gpus=1)
        .run.call_one(lambda state: os.getpid())
        .get(timeout=WAIT)
        == pids[1]
    )
