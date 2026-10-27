# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Procs on a host can be launched and torn down by a Python launcher installed
with ``HostMesh.set_proc_launcher``.

Monarch's contract with a launcher is the ``ProcLauncher`` ABC: Monarch asks it
to ``launch``, ``terminate`` and ``kill`` procs, and the launcher reports each
proc's exit. How a launcher starts and stops what it launches is its own
business; these tests observe only the launcher's own record of what Monarch
asked it to do, and an environment token that only the installed launcher
sets.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._src.actor.actor_mesh import Actor, Port
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.proc_launcher import LaunchOptions, ProcExitResult, ProcLauncher
from monarch._src.job.process import ProcessJob
from scoped_state import scoped_state

TOKEN_VAR = "MONARCH_TEST_LAUNCH_TOKEN"


class _SubprocessLauncher(ProcLauncher):
    """Launch each proc as a child process of the host's process.

    ``params``:
      - ``event_log``: file to append one JSON line per request and exit to.
      - ``token``: value of ``TOKEN_VAR`` in every child's environment.
      - ``plain_child``: run a process that never becomes a proc instead of
        the proc program.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._config: dict[str, Any] = params if params is not None else {}
        self._children: dict[str, subprocess.Popen[bytes]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def _record(self, event: str, proc_id: str, **fields: Any) -> None:
        with open(self._config["event_log"], "a") as log:
            log.write(json.dumps({"event": event, "proc_id": proc_id, **fields}) + "\n")

    def _child_command(
        self, opts: LaunchOptions
    ) -> tuple[list[str], str, dict[str, str]]:
        if self._config.get("plain_child"):
            argv = [sys.executable, "-c", "import time; time.sleep(3600)"]
            return argv, sys.executable, dict(os.environ)
        arg0 = opts.arg0 if opts.arg0 is not None else os.path.basename(opts.program)
        env: dict[str, str] = dict(os.environ) if opts.inherit_env else {}
        for name in opts.env_remove:
            env.pop(name, None)
        env.update(opts.env)
        env["HYPERACTOR_MESH_BOOTSTRAP_MODE"] = opts.bootstrap_payload
        env["HYPERACTOR_PROCESS_NAME"] = opts.process_name
        if opts.log_channel is not None:
            env["BOOTSTRAP_LOG_CHANNEL"] = opts.log_channel
        return [arg0, *opts.args], opts.program, env

    def _start(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @endpoint(explicit_response_port=True)
    async def launch(
        self,
        exit_port: Port[ProcExitResult],
        proc_id: str,
        opts: LaunchOptions,
    ) -> None:
        try:
            self._record("launch", proc_id)
            argv, executable, env = self._child_command(opts)
            env[TOKEN_VAR] = self._config["token"]
            child = subprocess.Popen(argv, executable=executable, env=env)
        except Exception as e:
            exit_port.exception(e)
            return
        self._children[proc_id] = child
        self._start(self._report_exit(exit_port, proc_id, child))

    async def _report_exit(
        self,
        exit_port: Port[ProcExitResult],
        proc_id: str,
        child: subprocess.Popen[bytes],
    ) -> None:
        returncode = await asyncio.get_running_loop().run_in_executor(None, child.wait)
        self._children.pop(proc_id, None)
        exit_code, signal = (
            (returncode, None) if returncode >= 0 else (None, -returncode)
        )
        self._record("exit", proc_id, exit_code=exit_code, signal=signal)
        exit_port.send(
            ProcExitResult(
                exit_code=exit_code,
                signal=signal,
                core_dumped=False,
                failed_reason=None,
                stderr_tail=[],
            )
        )

    async def _escalate(
        self, child: subprocess.Popen[bytes], timeout_secs: float
    ) -> None:
        deadline = time.monotonic() + timeout_secs
        while child.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if child.poll() is None:
            child.kill()

    @endpoint
    async def terminate(self, proc_id: str, timeout_secs: float) -> None:
        self._record("terminate", proc_id)
        child = self._children.get(proc_id)
        if child is not None and child.poll() is None:
            child.terminate()
            self._start(self._escalate(child, timeout_secs))

    @endpoint
    async def kill(self, proc_id: str) -> None:
        self._record("kill", proc_id)
        child = self._children.get(proc_id)
        if child is not None and child.poll() is None:
            child.kill()


class _TokenReporter(Actor):
    @endpoint
    async def token(self) -> str | None:
        return os.environ.get(TOKEN_VAR)


def _events(event_log: str) -> list[dict[str, Any]]:
    path = Path(event_log)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for_events(
    event_log: str,
    done: Callable[[list[dict[str, Any]]], bool],
    what: str,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        events = _events(event_log)
        if done(events):
            return events
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for {what}; launcher events: {events}")
        time.sleep(0.1)


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_launch_and_exit_are_delegated() -> None:
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        event_log = os.path.join(tmp, "events.jsonl")
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            hosts.set_proc_launcher(
                _SubprocessLauncher, {"event_log": event_log, "token": token}
            ).get(timeout=60)

            procs = hosts.spawn_procs(per_host={"procs": 2})
            reporters = procs.spawn("reporters", _TokenReporter)
            tokens = list(reporters.token.call().get(timeout=60).values())
            assert tokens == [token, token], (
                f"every proc should carry the launcher's token, got {tokens}"
            )
            launched = {
                e["proc_id"] for e in _events(event_log) if e["event"] == "launch"
            }
            assert len(launched) == 2, f"expected two launch requests, got {launched}"

            procs.stop().get(timeout=60)
            events = _wait_for_events(
                event_log,
                lambda ev: launched
                <= {e["proc_id"] for e in ev if e["event"] == "exit"},
                "an exit report for every launched proc",
            )
            exit_codes = {
                e["proc_id"]: e["exit_code"] for e in events if e["event"] == "exit"
            }
            assert all(exit_codes[p] == 0 for p in launched), (
                f"every proc should exit cleanly after stop, got {exit_codes}"
            )


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_teardown_is_delegated() -> None:
    os.environ["HYPERACTOR_HOST_SPAWN_READY_TIMEOUT"] = "3s"
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        event_log = os.path.join(tmp, "events.jsonl")
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            hosts.set_proc_launcher(
                _SubprocessLauncher,
                {"event_log": event_log, "token": token, "plain_child": True},
            ).get(timeout=60)

            procs = hosts.spawn_procs(per_host={"procs": 1})
            with pytest.raises(Exception) as spawn_failure:
                procs.initialized.get(timeout=60)
            assert not isinstance(spawn_failure.value, TimeoutError), (
                "the spawn should fail on its readiness timeout, not outlast the wait"
            )

            events = _wait_for_events(
                event_log,
                lambda ev: [e["event"] for e in ev] == ["launch", "kill", "exit"],
                "launch, kill and exit for the child that never became a proc",
            )
            assert len({e["proc_id"] for e in events}) == 1, (
                f"all three events should be for the same proc, got {events}"
            )
