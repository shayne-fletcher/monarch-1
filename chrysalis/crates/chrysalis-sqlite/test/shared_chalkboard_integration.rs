/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Buck-only process-boundary test for the shared-chalkboard lesson.
//!
//! The test launches packaged chalkboard processes, isolates each child tree
//! in a process group, and bounds every readiness, completion, output-drain,
//! and cleanup wait.

use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::ChildStdout;
use std::process::Command;
use std::process::ExitStatus;
use std::process::Stdio;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use nix::errno::Errno;
use nix::sys::signal;
use nix::sys::signal::Signal;
use nix::sys::wait::Id;
use nix::sys::wait::WaitPidFlag;
use nix::sys::wait::WaitStatus;
use nix::unistd::Pid;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const JOIN_TIMEOUT: Duration = Duration::from_secs(10);
const REPLICATION_SETTLE_TIME: Duration = Duration::from_secs(3);
const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(5);
const FORCED_STOP_TIMEOUT: Duration = Duration::from_secs(5);
const OUTPUT_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

struct Binaries {
    chalkboard: PathBuf,
    process_fixture: PathBuf,
}

impl Binaries {
    fn locate() -> Result<Self> {
        Ok(Self {
            chalkboard: buck_resources::get(
                "monarch/chrysalis/crates/chrysalis-sqlite/chalkboard",
            )?
            .to_path_buf(),
            process_fixture: buck_resources::get(
                "monarch/chrysalis/crates/chrysalis-sqlite/process_fixture",
            )?
            .to_path_buf(),
        })
    }

    fn chalkboard_command(&self) -> Command {
        Command::new(&self.chalkboard)
    }
}

struct ProcessOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    timed_out: bool,
    forced: bool,
    post_exit_group_killed: bool,
}

impl ProcessOutput {
    fn diagnostic(&self) -> String {
        format!(
            "status: {}\ntimed out: {}\nforced: {}\npost-exit group killed: {}\nstdout:\n{}\nstderr:\n{}",
            self.status,
            self.timed_out,
            self.forced,
            self.post_exit_group_killed,
            String::from_utf8_lossy(&self.stdout),
            String::from_utf8_lossy(&self.stderr),
        )
    }
}

enum FinishMode {
    Wait(Duration),
    Interrupt(Duration),
}

struct CapturedProcess {
    name: String,
    child: Option<Child>,
    first_stdout_line: Receiver<std::io::Result<String>>,
    stdout: Receiver<std::io::Result<Vec<u8>>>,
    stderr: Receiver<std::io::Result<Vec<u8>>>,
}

impl CapturedProcess {
    fn spawn(name: impl Into<String>, mut command: Command) -> Result<Self> {
        let name = name.into();
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // A separate process group lets timeout cleanup signal the child and
        // any descendants without signaling the test process.
        command.process_group(0);

        let mut child = command.spawn().with_context(|| format!("spawn {name}"))?;
        let stdout = child.stdout.take().context("capture child stdout")?;
        let stderr = child.stderr.take().context("capture child stderr")?;
        let (first_stdout_line, stdout) = capture_stdout(stdout);
        let stderr = capture_stream(stderr);

        Ok(Self {
            name,
            child: Some(child),
            first_stdout_line,
            stdout,
            stderr,
        })
    }

    fn first_stdout_line(&self, timeout: Duration) -> Result<String> {
        self.first_stdout_line
            .recv_timeout(timeout)
            .with_context(|| format!("wait for {} to print its first stdout line", self.name))?
            .with_context(|| format!("read {} stdout", self.name))
    }

    fn ensure_running(&mut self) -> Result<()> {
        let child = self.child.as_ref().context("child already consumed")?;
        if child_exited_unreaped(child)? {
            anyhow::bail!("{} exited unexpectedly", self.name);
        }
        Ok(())
    }

    fn finish(mut self, mode: FinishMode) -> Result<ProcessOutput> {
        let child = self.child.as_mut().context("child already consumed")?;
        let (timeout, interrupt) = match mode {
            FinishMode::Wait(timeout) => (timeout, false),
            FinishMode::Interrupt(timeout) => (timeout, true),
        };
        let mut timed_out = false;
        let mut forced = false;
        if !child_exited_unreaped(child)? {
            if interrupt {
                signal_process_group(child.id(), Signal::SIGINT)
                    .with_context(|| format!("interrupt {}", self.name))?;
            }
            if !wait_for_exit(child, timeout)? {
                timed_out = !interrupt;
                forced = true;
                signal_process_group(child.id(), Signal::SIGKILL)
                    .with_context(|| format!("kill timed-out {}", self.name))?;
                anyhow::ensure!(
                    wait_for_exit(child, FORCED_STOP_TIMEOUT)?,
                    "{} did not exit within {:?} after SIGKILL",
                    self.name,
                    FORCED_STOP_TIMEOUT
                );
            }
        }
        // The leader may exit while a descendant in the same process group
        // keeps inherited stdout or stderr descriptors open. Kill any group
        // that remains before reaping the leader and waiting for those capture
        // streams to close. The unreaped leader pins the process-group ID.
        let post_exit_group_killed = signal_process_group(child.id(), Signal::SIGKILL)
            .with_context(|| format!("remove descendants of {}", self.name))?;
        let status = child
            .wait()
            .with_context(|| format!("reap {}", self.name))?;
        let stdout = receive_capture(&self.stdout, &self.name, "stdout")?;
        let stderr = receive_capture(&self.stderr, &self.name, "stderr")?;
        self.child.take();

        Ok(ProcessOutput {
            status,
            stdout,
            stderr,
            timed_out,
            forced,
            post_exit_group_killed,
        })
    }
}

impl Drop for CapturedProcess {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let pid = child.id();
        // The child has not been reaped, so its PID still names our process
        // group while cleanup signals it.
        let _ = signal_process_group(child.id(), Signal::SIGKILL);
        if matches!(wait_for_exit(&child, FORCED_STOP_TIMEOUT), Ok(true)) {
            let _ = signal_process_group(pid, Signal::SIGKILL);
            let _ = child.wait();
        } else {
            // Hand off reaping so Drop stays bounded.
            let _ = thread::Builder::new()
                .name("shared-chalkboard-child-reaper".to_owned())
                .spawn(move || {
                    let _ = child.wait();
                });
        }
    }
}

fn capture_stdout(
    stdout: ChildStdout,
) -> (
    Receiver<std::io::Result<String>>,
    Receiver<std::io::Result<Vec<u8>>>,
) {
    let (first_line_sender, first_line_receiver) = mpsc::channel();
    let (output_sender, output_receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut first_line = String::new();
        let first_line_result = match reader.read_line(&mut first_line) {
            Ok(0) => Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "stdout closed before the first line",
            )),
            Ok(_) => Ok(first_line.trim_end_matches(['\r', '\n']).to_owned()),
            Err(error) => Err(error),
        };
        let _ = first_line_sender.send(first_line_result);

        let mut output = first_line.into_bytes();
        let result = reader.read_to_end(&mut output).map(|_| output);
        let _ = output_sender.send(result);
    });
    (first_line_receiver, output_receiver)
}

fn capture_stream(mut stream: impl Read + Send + 'static) -> Receiver<std::io::Result<Vec<u8>>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut output = Vec::new();
        let result = stream.read_to_end(&mut output).map(|_| output);
        let _ = sender.send(result);
    });
    receiver
}

fn receive_capture(
    receiver: &Receiver<std::io::Result<Vec<u8>>>,
    process: &str,
    stream: &str,
) -> Result<Vec<u8>> {
    receiver
        .recv_timeout(OUTPUT_DRAIN_TIMEOUT)
        .with_context(|| format!("drain {process} {stream}"))?
        .with_context(|| format!("read {process} {stream}"))
}

fn signal_process_group(pid: u32, signal: Signal) -> Result<bool> {
    let pid = i32::try_from(pid).context("child PID does not fit in i32")?;
    anyhow::ensure!(pid > 0, "child PID must be positive");
    match signal::kill(Pid::from_raw(-pid), signal) {
        Ok(()) => Ok(true),
        Err(Errno::ESRCH) => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("send signal {signal:?} to process group {pid}"))
        }
    }
}

fn child_exited_unreaped(child: &Child) -> Result<bool> {
    let pid = i32::try_from(child.id()).context("child PID does not fit in i32")?;
    match nix::sys::wait::waitid(
        Id::Pid(Pid::from_raw(pid)),
        WaitPidFlag::WEXITED | WaitPidFlag::WNOWAIT | WaitPidFlag::WNOHANG,
    ) {
        Ok(WaitStatus::StillAlive) | Err(Errno::EINTR) => Ok(false),
        Ok(WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _)) => Ok(true),
        Ok(status) => anyhow::bail!("unexpected child wait status: {status:?}"),
        Err(error) => Err(error).context("observe child exit without reaping it"),
    }
}

fn wait_for_exit(child: &Child, timeout: Duration) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    loop {
        if child_exited_unreaped(child)? {
            return Ok(true);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        thread::sleep(POLL_INTERVAL.min(deadline - now));
    }
}

fn run_bounded(name: &str, command: Command) -> Result<ProcessOutput> {
    let output =
        CapturedProcess::spawn(name, command)?.finish(FinishMode::Wait(COMMAND_TIMEOUT))?;
    anyhow::ensure!(
        !output.timed_out,
        "{name} exceeded {COMMAND_TIMEOUT:?}:\n{}",
        output.diagnostic()
    );
    Ok(output)
}

fn require_success(name: &str, output: &ProcessOutput) -> Result<()> {
    anyhow::ensure!(
        output.status.success(),
        "{name} failed:\n{}",
        output.diagnostic()
    );
    Ok(())
}

fn init_chalkboard(binaries: &Binaries, database: &Path) -> Result<()> {
    let mut command = binaries.chalkboard_command();
    command.arg("init").arg(database);
    let output = run_bounded("initialize Alice's chalkboard", command)?;
    require_success("initialize Alice's chalkboard", &output)
}

struct Note<'a> {
    id: &'a str,
    author: &'a str,
    message: &'a str,
}

fn write_note(binaries: &Binaries, database: &Path, note: Note<'_>) -> Result<()> {
    let mut command = binaries.chalkboard_command();
    command
        .arg("write")
        .arg(database)
        .arg(note.id)
        .arg(note.author)
        .arg(note.message);
    let output = run_bounded(&format!("write note {}", note.id), command)?;
    require_success(&format!("write note {}", note.id), &output)
}

fn show_notes(binaries: &Binaries, database: &Path) -> Result<Vec<String>> {
    let mut command = binaries.chalkboard_command();
    command.arg("show").arg(database);
    let output = run_bounded(&format!("show {}", database.display()), command)?;
    require_success(&format!("show {}", database.display()), &output)?;
    let stdout = String::from_utf8(output.stdout).context("chalkboard output is not UTF-8")?;
    Ok(stdout
        .lines()
        .filter(|line| line.contains(" | "))
        .map(str::to_owned)
        .collect())
}

struct SynchronizerSpec<'a> {
    name: &'a str,
    database: &'a Path,
    join: Option<&'a str>,
    home: &'a Path,
}

fn spawn_synchronizer(binaries: &Binaries, spec: SynchronizerSpec<'_>) -> Result<CapturedProcess> {
    let mut command = binaries.chalkboard_command();
    command.env("HOME", spec.home).arg("sync");
    if let Some(join) = spec.join {
        command.arg("--join").arg(join);
    }
    command.arg(spec.database);
    CapturedProcess::spawn(spec.name, command)
}

fn stop_synchronizer(process: CapturedProcess) -> Result<ProcessOutput> {
    process.finish(FinishMode::Interrupt(GRACEFUL_STOP_TIMEOUT))
}

fn synchronizer_stopped(name: &str, output: &ProcessOutput) -> Result<()> {
    anyhow::ensure!(
        !output.forced,
        "{name} did not stop within {GRACEFUL_STOP_TIMEOUT:?} after SIGINT:\n{}",
        output.diagnostic()
    );
    require_success(name, output)
}

fn wait_for_replication_window(
    alice: &mut CapturedProcess,
    bob: &mut CapturedProcess,
) -> Result<()> {
    // The synchronizer has no external quiescence notification. Keep both
    // databases idle for several 250 ms replication polls, while failing early
    // if either synchronizer exits, before sampling state through the
    // short-lived chalkboard command.
    let deadline = Instant::now() + REPLICATION_SETTLE_TIME;
    loop {
        alice.ensure_running()?;
        bob.ensure_running()?;
        let now = Instant::now();
        if now >= deadline {
            return Ok(());
        }
        thread::sleep(POLL_INTERVAL.min(deadline - now));
    }
}

fn run_shared_chalkboard() -> Result<()> {
    let binaries = Binaries::locate()?;
    let directory = tempfile::tempdir().context("create isolated chalkboard directory")?;
    let alice_database = directory.path().join("alice.db");
    let bob_database = directory.path().join("bob.db");

    init_chalkboard(&binaries, &alice_database)?;
    write_note(
        &binaries,
        &alice_database,
        Note {
            id: "alice-1",
            author: "Alice",
            message: "tea at four",
        },
    )?;
    write_note(
        &binaries,
        &alice_database,
        Note {
            id: "alice-1",
            author: "Alice",
            message: "tea at five",
        },
    )?;
    anyhow::ensure!(
        !bob_database.exists(),
        "Bob's database should not exist before synchronization"
    );
    anyhow::ensure!(
        show_notes(&binaries, &alice_database)? == ["alice-1 | Alice | tea at five"],
        "Alice's chalkboard should contain the updated note before synchronization"
    );

    let mut alice = spawn_synchronizer(
        &binaries,
        SynchronizerSpec {
            name: "Alice synchronizer",
            database: &alice_database,
            join: None,
            home: directory.path(),
        },
    )?;
    let alice_token = match alice.first_stdout_line(JOIN_TIMEOUT) {
        Ok(token) => token,
        Err(error) => {
            let cleanup = stop_synchronizer(alice)
                .map(|output| output.diagnostic())
                .unwrap_or_else(|cleanup_error| format!("cleanup error: {cleanup_error:#}"));
            anyhow::bail!("Alice did not publish a join token: {error:#}\nAlice:\n{cleanup}");
        }
    };
    let mut bob = match spawn_synchronizer(
        &binaries,
        SynchronizerSpec {
            name: "Bob synchronizer",
            database: &bob_database,
            join: Some(&alice_token),
            home: directory.path(),
        },
    ) {
        Ok(bob) => bob,
        Err(error) => {
            let cleanup = stop_synchronizer(alice)
                .map(|output| output.diagnostic())
                .unwrap_or_else(|cleanup_error| format!("cleanup error: {cleanup_error:#}"));
            anyhow::bail!("spawn Bob synchronizer: {error:#}\nAlice:\n{cleanup}");
        }
    };

    let scenario = (|| -> Result<()> {
        bob.first_stdout_line(JOIN_TIMEOUT)
            .context("Bob should connect and publish his join token")?;
        wait_for_replication_window(&mut alice, &mut bob)?;
        anyhow::ensure!(
            show_notes(&binaries, &bob_database)? == ["alice-1 | Alice | tea at five"],
            "Bob's chalkboard command should read Alice's replicated note"
        );

        write_note(
            &binaries,
            &bob_database,
            Note {
                id: "bob-1",
                author: "Bob",
                message: "bring biscuits",
            },
        )?;
        wait_for_replication_window(&mut alice, &mut bob)?;
        anyhow::ensure!(
            show_notes(&binaries, &alice_database)?
                == [
                    "alice-1 | Alice | tea at five",
                    "bob-1 | Bob | bring biscuits",
                ],
            "Alice's chalkboard command should read Bob's replicated note"
        );

        write_note(
            &binaries,
            &alice_database,
            Note {
                id: "alice-1",
                author: "Alice",
                message: "tea at six",
            },
        )?;
        wait_for_replication_window(&mut alice, &mut bob)?;
        anyhow::ensure!(
            show_notes(&binaries, &bob_database)?
                == [
                    "alice-1 | Alice | tea at six",
                    "bob-1 | Bob | bring biscuits",
                ],
            "Bob's chalkboard command should read Alice's replicated update"
        );
        Ok(())
    })();

    let bob_output = stop_synchronizer(bob);
    let alice_output = stop_synchronizer(alice);
    let bob_diagnostic = bob_output
        .as_ref()
        .map(ProcessOutput::diagnostic)
        .unwrap_or_else(|error| format!("cleanup error: {error:#}"));
    let alice_diagnostic = alice_output
        .as_ref()
        .map(ProcessOutput::diagnostic)
        .unwrap_or_else(|error| format!("cleanup error: {error:#}"));
    if let Err(error) = scenario {
        anyhow::bail!(
            "shared-chalkboard scenario failed: {error:#}\nBob:\n{bob_diagnostic}\nAlice:\n{alice_diagnostic}"
        );
    }
    let bob_output = bob_output.context("stop Bob synchronizer")?;
    let alice_output = alice_output.context("stop Alice synchronizer")?;
    synchronizer_stopped("Bob synchronizer", &bob_output)?;
    synchronizer_stopped("Alice synchronizer", &alice_output)?;

    let expected = [
        "alice-1 | Alice | tea at six",
        "bob-1 | Bob | bring biscuits",
    ];
    anyhow::ensure!(
        show_notes(&binaries, &alice_database)?
            .iter()
            .map(String::as_str)
            .eq(expected),
        "Alice's database should remain locally readable after shutdown"
    );
    anyhow::ensure!(
        show_notes(&binaries, &bob_database)?
            .iter()
            .map(String::as_str)
            .eq(expected),
        "Bob's database should remain locally readable after shutdown"
    );
    Ok(())
}

#[test]
fn shared_chalkboard_converges_across_processes() {
    run_shared_chalkboard().expect("shared-chalkboard integration scenario should succeed");
}

#[test]
fn cleanup_timeout_kills_an_unresponsive_process_group() {
    let binaries = Binaries::locate().expect("locate integration-test resources");
    let command = Command::new(binaries.process_fixture);
    let output = CapturedProcess::spawn("cleanup timeout fixture", command)
        .expect("spawn cleanup timeout fixture")
        .finish(FinishMode::Wait(Duration::from_millis(200)))
        .expect("force-stop cleanup timeout fixture");
    assert!(output.timed_out, "fixture should reach the timeout path");
    assert!(output.forced, "timeout path should force-stop the process");
}

#[test]
fn cleanup_kills_descendant_after_group_leader_exits() {
    let binaries = Binaries::locate().expect("locate integration-test resources");
    let mut command = Command::new(binaries.process_fixture);
    command.arg("spawn-descendant-and-exit");
    let output = CapturedProcess::spawn("descendant-holding cleanup fixture", command)
        .expect("spawn descendant-holding cleanup fixture")
        .finish(FinishMode::Wait(Duration::from_secs(2)))
        .expect("clean up descendant after group leader exits");
    assert!(
        !output.timed_out,
        "fixture group leader should exit before the deadline"
    );
    assert!(
        !output.forced,
        "fixture group leader should not require a forced stop"
    );
    assert!(
        output.post_exit_group_killed,
        "cleanup should kill the descendant remaining in the process group"
    );
}
