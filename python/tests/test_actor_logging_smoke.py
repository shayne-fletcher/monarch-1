# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Actor-based logging smoke test.


Defines a `Logger` actor that routes INFO/WARNING to stdout and ERROR+
to stderr using two `logging.StreamHandler`s. The test captures
**process-level** stdout/stderr by temporarily redirecting file
descriptors (FD 1/2), so both Python and any Rust / native output
would be captured. It then spins up a small mesh, invokes the actor's
endpoints, and asserts the messages landed on the expected streams
(and include the expected actor prefix).

"""

import asyncio
import logging
import os
import re
import sys
import tempfile

import pytest
from monarch._src.actor.host_mesh import this_host
from monarch.actor import Actor, endpoint


class Logger(Actor):
    """Actor that emits log lines at different severities and routes them
    to separate streams.

    Setup:

      - Adds a stdout handler (INFO/WARNING only) and a stderr handler
        (ERROR+).

      - Flushes handlers after each endpoint call to minimize
        buffering effects.

    Notes:
      - We attach handlers to the *root* logger returned by
        `logging.getLogger()`.

      - The INFO/WARNING routing is enforced via a simple level
        filter: records with `levelno < logging.ERROR` go to stdout;
        others go to stderr.

    """

    def __init__(self) -> None:
        self._logger: logging.Logger = logging.getLogger()

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)

        self._logger.addHandler(stdout_handler)
        self._logger.addHandler(stderr_handler)

    @endpoint
    async def log_warn(self, content: str) -> None:
        """Emit a WARNING-level message and flush all handlers.

        Args:
            content: The message body to log.

        """
        self._logger.warning(f"{content}")
        for handler in self._logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()

    @endpoint
    async def log_info(self, content: str) -> None:
        """
        Emit an INFO-level message and flush all handlers.

        Args:
            content: The message body to log.
        """
        self._logger.info(f"{content}")
        for handler in self._logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()

    @endpoint
    async def log_error(self, content: str) -> None:
        """
        Emit an ERROR-level message and flush all handlers.

        Args:
            content: The message body to log.
        """
        self._logger.error(f"{content}")
        for handler in self._logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()


# oss_skip: (SF) broken in GitHub by D86994420. Passes internally.
@pytest.mark.oss_skip
async def test_actor_logging_smoke() -> None:
    """End-to-end smoke test of stdio routing for the Logger actor.

    Flow:

      1. Duplicate and redirect the process's stdout/stderr file
         descriptors to temporary files (captures both Python and
         native output).

      2. Start a small per-host mesh, enable logging, and spawn the
         `Logger` actor.

      3. Invoke `log_warn`, `log_info`, and `log_error`.

      4. Restore FDs, read back captured output, and assert:
         - WARNING/INFO appear on stdout,
         - ERROR appears on stderr,
         - an actor prefix like `[actor=...Logger...]` is present on
           stdout.

    This test intentionally uses FD-level redirection (not just
    `sys.stdout`) to validate the real streams that the parent process
    would see.

    """
    original_stdout_fd = None
    original_stderr_fd = None

    try:
        # Save original file descriptors.
        original_stdout_fd = os.dup(1)  # stdout
        original_stderr_fd = os.dup(2)  # stderr

        # Create temporary files to capture output.
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as stdout_file, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as stderr_file:
            stdout_path = stdout_file.name
            stderr_path = stderr_file.name

            # Redirect file descriptors to our temp files. This will
            # capture both Python and Rust output.
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            # Also redirect Python's sys.stdout/stderr for
            # completeness.
            original_sys_stdout = sys.stdout
            original_sys_stderr = sys.stderr
            sys.stdout = stdout_file
            sys.stderr = stderr_file

            try:
                # Make a logger mesh.
                pm = this_host().spawn_procs(per_host={"gpus": 2})
                await pm.logging_option(level=logging.INFO)
                am = pm.spawn("logger", Logger)

                # Do some logging actions.
                await am.log_warn.call("hello 1")
                await am.log_info.call("hello 2")
                await am.log_error.call("hello 3")

                # Wait a bit for output to be written.
                await asyncio.sleep(1)

                # Cleanup.
                stdout_file.flush()
                stderr_file.flush()
                os.fsync(stdout_file.fileno())
                os.fsync(stderr_file.fileno())

                await pm.stop()

            finally:
                # Restore Python's sys.stdout/stderr
                sys.stdout = original_sys_stdout
                sys.stderr = original_sys_stderr

        # Restore original file descriptors.
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)

        # Read the captured output.
        with open(stdout_path, "r") as f:
            stdout_content = f.read()
        with open(stderr_path, "r") as f:
            stderr_content = f.read()

        # Print the captured output.
        print("")
        print("=== Captured stdout ===")
        print(stdout_content)
        print("=== Captured stderr ===")
        print(stderr_content)

        # Clean up temp files.
        os.unlink(stdout_path)
        os.unlink(stderr_path)

        # Assertions on the captured output.
        assert re.search(
            r"hello 1", stdout_content
        ), f"Expected 'hello 1' in stdout: {stdout_content}"
        assert re.search(
            r"hello 2", stdout_content
        ), f"Expected 'hello 2' in stdout: {stdout_content}"
        assert re.search(
            r"hello 3", stderr_content
        ), f"Expected 'hello 3' in stderr: {stderr_content}"
        assert re.search(
            r"\[actor=.*Logger.*\]", stdout_content
        ), f"Expected actor prefix in stdout: {stdout_content}"

    finally:
        # Ensure file descriptors are restored even if something goes
        # wrong.
        try:
            if original_stdout_fd is not None:
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)
            if original_stderr_fd is not None:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
        except OSError:
            pass
