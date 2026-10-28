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
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pytest
from isolate_in_subprocess import isolate_in_subprocess
from monarch._src.actor.actor_mesh import Actor, Port, shutdown_context
from monarch._src.actor.endpoint import endpoint
from monarch._src.actor.host_mesh import HostMesh
from monarch._src.actor.proc_launcher import LaunchOptions, ProcExitResult, ProcLauncher
from monarch._src.job.process import ProcessJob
from monarch.actor import this_host
from scoped_state import scoped_state

TOKEN_VAR = "MONARCH_TEST_LAUNCH_TOKEN"
PROC_ID_VAR = "MONARCH_TEST_LAUNCH_PROC_ID"


class _LauncherCrash(BaseException):
    """Raised by the test launcher to fail its own actor."""


class _SubprocessLauncher(ProcLauncher):
    """Launch each proc as a child process of the host's process.

    ``params``:
      - ``event_log``: file to append one JSON line per request and exit to.
      - ``token``: value of ``TOKEN_VAR`` in every child's environment.
      - ``plain_child``: run a process that never becomes a proc instead of
        the proc program.
      - ``fail_init``: make the constructor raise.
      - ``fail_on_launch``: on the ``launch`` call with this number, start
        nothing and fail the launcher actor.
      - ``cleanup_request``: file the fixture creates to request child cleanup.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._config: dict[str, Any] = params if params is not None else {}
        self._launches = 0
        self._children: dict[str, subprocess.Popen[bytes]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        if self._config.get("fail_init"):
            raise ValueError("launcher refused to start: fail_init is set")

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
        self._launches += 1
        if self._launches == self._config.get("fail_on_launch"):
            self._record("launch_failed", proc_id)
            raise _LauncherCrash("injected launcher failure")
        try:
            self._record("launch", proc_id)
            argv, executable, env = self._child_command(opts)
            env[TOKEN_VAR] = self._config["token"]
            env[PROC_ID_VAR] = proc_id
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
        cleanup_request = self._config.get("cleanup_request")
        recovering = False
        while (returncode := child.poll()) is None:
            if (
                cleanup_request is not None
                and not recovering
                and Path(cleanup_request).exists()
            ):
                self._record("fixture_cleanup", proc_id)
                child.kill()
                recovering = True
            await asyncio.sleep(0.05)
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
        if recovering:
            self._record("fixture_cleanup_finished", proc_id)

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

    async def __cleanup__(self, exc: Exception | None) -> None:
        children = list(self._children.items())
        self._record("cleanup_started", "", proc_ids=[p for p, _ in children])
        for _, child in children:
            if child.poll() is None:
                child.kill()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=2)
            if pending:
                raise TimeoutError("launcher tasks did not stop during cleanup")
        deadline = time.monotonic() + 2
        while any(child.poll() is None for _, child in children):
            if time.monotonic() >= deadline:
                raise TimeoutError("launcher children did not exit during cleanup")
            await asyncio.sleep(0.05)
        self._children.clear()
        self._record(
            "cleanup_finished", "", exit_codes={p: c.returncode for p, c in children}
        )


class _TokenReporter(Actor):
    @endpoint
    async def token(self) -> str | None:
        return os.environ.get(TOKEN_VAR)

    @endpoint
    async def proc_id(self) -> str | None:
        return os.environ.get(PROC_ID_VAR)

    @endpoint
    async def stop_own_process_soon(self) -> int:
        """Stop this proc's process shortly after replying; return its pid so
        the caller can confirm the stop took effect."""
        asyncio.get_running_loop().call_later(0.2, os.kill, os.getpid(), signal.SIGSTOP)
        return os.getpid()


def _events(event_log: str) -> list[dict[str, Any]]:
    path = Path(event_log)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _launcher_params(tmp: str, **extra: Any) -> dict[str, Any]:
    return {
        "event_log": os.path.join(tmp, f"events-{uuid.uuid4().hex}.jsonl"),
        "cleanup_request": os.path.join(tmp, f"cleanup-{uuid.uuid4().hex}"),
        "token": uuid.uuid4().hex,
        **extra,
    }


@contextmanager
def _client_launcher(params: dict[str, Any]) -> Iterator[HostMesh]:
    try:
        host = this_host()
        host.set_proc_launcher(_SubprocessLauncher, params).get(timeout=60)
        yield host
    finally:
        Path(params["cleanup_request"]).touch()
        shutdown_context(host_timeout=2).get(timeout=10)


def _tokens(procs: Any) -> list[str | None]:
    reporters = procs.spawn("reporters", _TokenReporter)
    return list(reporters.token.call().get(timeout=60).values())


def _exited(params: dict[str, Any], proc_ids: set[str]) -> list[dict[str, Any]]:
    return _wait_for_events(
        params["event_log"],
        lambda ev: proc_ids <= {e["proc_id"] for e in ev if e["event"] == "exit"},
        f"an exit report for each of {sorted(proc_ids)}",
    )


def _wait_until_stopped(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state == "T":
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} did not stop within {timeout}s")


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


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_install_after_first_spawn_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            hosts.spawn_procs(per_host={"procs": 1}).initialized.get(timeout=60)
            with pytest.raises(
                Exception, match="launcher already initialized"
            ) as failure:
                hosts.set_proc_launcher(_SubprocessLauncher, _launcher_params(tmp)).get(
                    timeout=60
                )
            assert "host rank 0" in str(failure.value), str(failure.value)


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_second_install_fails_and_first_stays() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first, second = _launcher_params(tmp), _launcher_params(tmp)
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            hosts.set_proc_launcher(_SubprocessLauncher, first).get(timeout=60)
            with pytest.raises(Exception, match="launcher already initialized"):
                hosts.set_proc_launcher(_SubprocessLauncher, second).get(timeout=60)

            procs = hosts.spawn_procs(per_host={"procs": 1})
            assert _tokens(procs) == [first["token"]], (
                "procs should still be launched by the first launcher"
            )


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_failed_constructor_leaves_host_installable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            with pytest.raises(Exception, match="launcher refused to start") as failure:
                hosts.set_proc_launcher(
                    _SubprocessLauncher, _launcher_params(tmp, fail_init=True)
                ).get(timeout=60)
            assert "host rank 0" in str(failure.value), str(failure.value)

            good = _launcher_params(tmp)
            hosts.set_proc_launcher(_SubprocessLauncher, good).get(timeout=60)
            procs = hosts.spawn_procs(per_host={"procs": 1})
            assert _tokens(procs) == [good["token"]], (
                "the failed install should leave the host installable"
            )


@pytest.mark.timeout(180)
@isolate_in_subprocess
def test_failed_launcher_does_not_hang_host() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        params = _launcher_params(tmp, fail_on_launch=2)
        job = ProcessJob({"hosts": 1})
        try:
            hosts = job.state(cached_path=None).hosts
            hosts.set_proc_launcher(_SubprocessLauncher, params).get(timeout=60)
            hosts.spawn_procs(per_host={"procs": 1}, name="first").initialized.get(
                timeout=60
            )

            started = time.monotonic()
            with pytest.raises(Exception, match="launcher actor stopped"):
                hosts.spawn_procs(per_host={"procs": 1}, name="second").initialized.get(
                    timeout=60
                )
            assert time.monotonic() - started < 20, (
                "the spawn should fail on the stopped launcher, not on the "
                "default 30s readiness timeout"
            )
            with pytest.raises(Exception, match="launcher actor stopped"):
                hosts.spawn_procs(per_host={"procs": 1}, name="third").initialized.get(
                    timeout=20
                )

            events = [e["event"] for e in _events(params["event_log"])]
            assert events.count("launch") == 1, events
            assert events.count("launch_failed") == 1, events
            hosts.shutdown().get(timeout=30)
        finally:
            # Nothing supervises the first proc once its launcher has failed;
            # it is still in the host's session, which this reaps.
            job._kill()


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_terminate_routes_to_launcher() -> None:
    os.environ["HYPERACTOR_PROCESS_EXIT_TIMEOUT"] = "2s"
    with tempfile.TemporaryDirectory() as tmp:
        params = _launcher_params(tmp)
        with scoped_state(ProcessJob({"hosts": 1}), cached_path=None) as state:
            hosts = state.hosts
            hosts.set_proc_launcher(_SubprocessLauncher, params).get(timeout=60)
            procs = hosts.spawn_procs(per_host={"procs": 2})
            reporters = procs.spawn("reporters", _TokenReporter)
            proc_ids = [
                proc_id
                for proc_id in reporters.proc_id.call().get(timeout=60).values()
                if proc_id is not None
            ]
            assert len(proc_ids) == 2, f"both procs should report their id: {proc_ids}"
            stopped = proc_ids[0]
            stopped_pid = (
                reporters.slice(procs=0)
                .stop_own_process_soon.call_one()
                .get(timeout=30)
            )
            _wait_until_stopped(stopped_pid)

            procs.stop().get(timeout=60)
            events = _exited(params, set(proc_ids))
            terminated = [e["proc_id"] for e in events if e["event"] == "terminate"]
            assert terminated == [stopped], (
                f"only the stopped proc should be terminated, got {terminated}"
            )
            order = [e["event"] for e in events if e["proc_id"] == stopped]
            assert order.index("terminate") < order.index("exit"), order


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_launches_on_client_host() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        params = _launcher_params(tmp)
        with _client_launcher(params) as host:
            procs = host.spawn_procs(per_host={"procs": 2})
            assert _tokens(procs) == [params["token"]] * 2
            launched = {
                e["proc_id"]
                for e in _events(params["event_log"])
                if e["event"] == "launch"
            }
            assert len(launched) == 2, launched

            procs.stop().get(timeout=60)
            _exited(params, launched)


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_install_error_names_only_the_failed_host() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with scoped_state(ProcessJob({"hosts": 2}), cached_path=None) as state:
            hosts = state.hosts
            hosts.slice(hosts=0).spawn_procs(per_host={"procs": 1}).initialized.get(
                timeout=60
            )
            with pytest.raises(Exception, match="1 of 2 hosts") as failure:
                hosts.set_proc_launcher(_SubprocessLauncher, _launcher_params(tmp)).get(
                    timeout=60
                )
            message = str(failure.value)
            assert "host rank 0" in message and "host rank 1" not in message, message


@pytest.mark.timeout(120)
@isolate_in_subprocess
def test_recovery_after_bounded_wait_expires() -> None:
    os.environ["HYPERACTOR_HOST_SPAWN_READY_TIMEOUT"] = "60s"
    with tempfile.TemporaryDirectory() as tmp:
        params = _launcher_params(tmp, plain_child=True)
        with pytest.raises(TimeoutError) as recovery:
            with _client_launcher(params) as host:
                procs = host.spawn_procs(per_host={"procs": 1})
                _wait_for_events(
                    params["event_log"],
                    lambda ev: any(e["event"] == "launch" for e in ev),
                    "a launch before forcing the observer timeout",
                )
                started = time.monotonic()
                with pytest.raises(TimeoutError) as observer:
                    procs.initialized.get(timeout=1)
                raise observer.value
        assert recovery.value is observer.value, (
            "fixture cleanup should preserve the observer's timeout"
        )
        assert time.monotonic() - started < 15, (
            "fixture recovery should finish before the 60s startup timeout"
        )
        events = _events(params["event_log"])
        launched = {e["proc_id"] for e in events if e["event"] == "launch"}
        recovered = {e["proc_id"] for e in events if e["event"] == "fixture_cleanup"}
        finished = {
            e["proc_id"] for e in events if e["event"] == "fixture_cleanup_finished"
        }
        assert len(launched) == 1, events
        assert recovered == finished == launched, events
        exited = {e["proc_id"] for e in events if e["event"] == "exit"}
        assert launched <= exited, events
