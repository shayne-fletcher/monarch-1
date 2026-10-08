/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Pywaker provides a GIL-free way to wake up Python event loops from Rust.
//! This is accomplished by writing a single byte to a "self pipe" for every
//! wake up. The corresponding read-side FD is added to the Python-side event
//! loop.
//!
//! The approach does not add overhead in uncontended scenarios:
//!
//! ```ignore
//!   | Approach             | Mean   | Notes                           |
//!   |----------------------|--------|---------------------------------|
//!   | FD-based             | ~48 µs | No GIL acquisition on Rust side |
//!   | call_soon_threadsafe | ~56 µs | Acquires GIL on Rust side       |
//!   | Pure Python          | ~44 µs | Baseline for comparison         |
//! ```
//!
//! ([Full benchmarks](https://github.com/mariusae/tmp/tree/main/2025-12-16-wakerbench)).
//!
//! ## Implementation notes
//!
//! [`event`] creates a pipe(2), and wakeup events are implemented by writing a single
//! byte into the pipe. The Python (read side) event registers the read fd into its
//! asyncio event loop. Thus the pipe acts as a notification queue. When the fd becomes
//! readable, the python event drains the pipe and then sets an `asyncio.Event` to notify
//! any waiters. [`pipe`] supplies the same pipe to a reader that is not an event loop,
//! such as a sync actor's inbox, which waits on the read fd with `poll`.
//!
//! Each read side has a single reader thread, and is thus free of race conditions: an
//! event's reads all occur within its asyncio event loop, which processes events
//! sequentially on one thread, and a sync actor's inbox is read only by its driver.

use std::os::fd::FromRawFd;
use std::os::fd::IntoRawFd;
use std::os::fd::OwnedFd;
use std::os::fd::RawFd;

use monarch_types::MapPyErr;
use nix::errno::Errno;
use nix::fcntl::FcntlArg;
use nix::fcntl::OFlag;
use nix::fcntl::fcntl;
use nix::unistd;
use nix::unistd::write;
use pyo3::Bound;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;

/// Write side of a non-blocking self-pipe used to notify its receiver.
#[derive(Debug)]
pub struct Waker {
    /// Pipe descriptor written by [`Waker::wake`]. The corresponding read end
    /// is owned by a [`PyEvent`] or a blocking Python receiver.
    write_fd: OwnedFd,
}

impl Waker {
    /// Notify the receiver by writing one byte to the pipe.
    ///
    /// A full pipe already represents a pending notification and therefore
    /// counts as success. A closed read end returns `false`.
    pub fn wake(&self) -> Result<bool, nix::Error> {
        static DATA: [u8; 1] = *b"w";

        match retry_eintr(|| write(&self.write_fd, &DATA)) {
            Ok(_) => Ok(true),
            // Pipe is full. This is ok.
            Err(Errno::EAGAIN) => Ok(true),
            // The Python side closed the pipe. It is no longer listening.
            Err(Errno::EPIPE) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Wakers are not intended to be used from Python; TestWaker
/// is provided to faciliate pure-Python unit testing.
// NOTE: We can't use a Python calss name that starts with "Test" since
// during Python testing, Pytest will inspect anything that starts with
// "Test" and check if its callable which in pyo3 >= 0.26 will raise
// a TypeError.
#[pyclass(name = "PyTestWaker", module = "monarch._src.actor.waker")]
struct TestPyWaker(Waker);

#[pymethods]
impl TestPyWaker {
    fn wake(&self) -> PyResult<bool> {
        self.0.wake().map_pyerr()
    }

    #[staticmethod]
    fn create() -> PyResult<(TestPyWaker, PyEvent)> {
        let (waker, event) = event().map_pyerr()?;
        Ok((TestPyWaker(waker), event))
    }
}

/// An event that is awoken by a [`Waker`].
#[pyclass(name = "Event", module = "monarch._src.actor.waker")]
pub struct PyEvent {
    #[pyo3(get, name = "_read_fd")]
    read_fd: RawFd,

    #[pyo3(get, set, name = "_event_loop")]
    event_loop: Option<Py<PyAny>>,

    #[pyo3(get, set, name = "_event")]
    event: Option<Py<PyAny>>,
}

impl Drop for PyEvent {
    fn drop(&mut self) {
        // SAFETY: fd was obtained via into_raw_fd() in the event() function
        let _ = unsafe { OwnedFd::from_raw_fd(self.read_fd) };
    }
}

/// Create a new event, returning the (Rust only) [`Waker`], and
/// a Python [`PyEvent`], intended for passing to Python code.
pub fn event() -> Result<(Waker, PyEvent), nix::Error> {
    let (waker, read_fd) = pipe()?;

    Ok((
        waker,
        PyEvent {
            read_fd: read_fd.into_raw_fd(),
            event_loop: None,
            event: None,
        },
    ))
}

/// Create a non-blocking pipe, returning its write end as a [`Waker`] and
/// its read end for the caller to wait on.
pub(crate) fn pipe() -> Result<(Waker, OwnedFd), nix::Error> {
    let (read_fd, write_fd) = unistd::pipe()?;

    set_nonblocking(&read_fd)?;
    set_nonblocking(&write_fd)?;

    Ok((Waker { write_fd }, read_fd))
}

/// Call `f` again for as long as it fails with `EINTR`.
pub(crate) fn retry_eintr<T>(mut f: impl FnMut() -> nix::Result<T>) -> nix::Result<T> {
    loop {
        match f() {
            Err(Errno::EINTR) => continue,
            result => return result,
        }
    }
}

fn set_nonblocking(fd: &OwnedFd) -> Result<(), nix::Error> {
    let flags = OFlag::from_bits_truncate(fcntl(fd, FcntlArg::F_GETFL)?) | OFlag::O_NONBLOCK;
    fcntl(fd, FcntlArg::F_SETFL(flags))?;
    Ok(())
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    hyperactor_mod.add_class::<PyEvent>()?;
    hyperactor_mod.add_class::<TestPyWaker>()?;
    Ok(())
}
