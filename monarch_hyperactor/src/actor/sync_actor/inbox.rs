/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Blocking handoff from a Rust actor running on Tokio to a Python driver with
//! no persistent actor event loop.
//!
//! `PythonActor` owns the message and control senders.
//! `Handler<PythonMessage>` enqueues `PendingMessage`, `Handler<MeshFailure>`
//! enqueues `PendingSupervision`, undeliverable-message handling enqueues
//! `PendingUndeliverable`, and sync cleanup enqueues `PendingSyncCleanup`.
//! `Actor::init` passes the [`Receiver`] to `_sync_dispatch_loop` on
//! the Python driver thread, which receives and runs one item at a time.
//!
//! # Sync-inbox invariants
//!
//! - **SI-1 (two planes, one consumer):** message and control queues are each
//!   FIFO and have one consumer. When selecting the next item, control has
//!   priority over messages; it never preempts an item already running.
//! - **SI-2 (enqueue before wake):** a sender enqueues its value before writing
//!   the common wake pipe. A wake byte carries no queue or item identity and
//!   may be stale.
//! - **SI-3 (no lost wake):** after finding both queues empty, `next()` releases
//!   the GIL and polls the pipe. Any later send has already written the pipe;
//!   after waking, `next()` drains it and checks both queues again.
//! - **SI-4 (GIL boundary):** Tokio enqueues `PendingMessage`,
//!   `PendingSupervision`, `PendingUndeliverable`, and `PendingSyncCleanup`
//!   without the GIL. The driver holds the GIL while converting them to Python
//!   objects and releases it only while polling.
//! - **SI-5 (disconnect):** a sender's queue disconnects before its pipe
//!   writer closes (field order in `pympsc::Sender`), and closing the final
//!   writer wakes a blocked receiver. A disconnected control plane means the
//!   owning actor is gone: `next()` returns `None` without taking another
//!   message, after any control item queued before the close.
//! - **SI-6 (stop claim):** cleanup sets `stopping` before enqueueing
//!   `PendingSyncCleanup`. A dequeued `QueuedSupervision` or
//!   `QueuedUndeliverable` runs only if the driver observes `stopping` as
//!   false. The resulting `SyncCleanup` runs if the driver is still running
//!   and can convert it.

use std::os::fd::AsFd;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc;

use monarch_types::MapPyErr;
use nix::errno::Errno;
use nix::poll::PollFd;
use nix::poll::PollFlags;
use nix::poll::PollTimeout;
use nix::poll::poll;
use nix::unistd::read;
use pyo3::Bound;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;

use crate::pympsc::IntoPyObjectBox;
use crate::pympsc::Sender;
use crate::pywaker;
use crate::pywaker::retry_eintr;

/// The complete inbox created for one sync actor.
pub(in super::super) struct Inbox {
    /// Enqueues messages for the driver.
    pub(in super::super) message_sender: Sender,
    /// Enqueues `PendingSupervision`, `PendingUndeliverable`, and
    /// `PendingSyncCleanup` for the driver.
    pub(in super::super) control_sender: Sender,
    /// Receives both queues, with control taking priority over messages.
    pub(in super::super) receiver: Receiver,
}

/// Create a sync actor inbox. Both senders write the same wake pipe, so one
/// blocking wait observes both queues.
pub(in super::super) fn new() -> Result<Inbox, nix::Error> {
    // SI-1 message plane: actor messages in FIFO order.
    let (messages_tx, messages_rx) = mpsc::channel();

    // SI-1 control plane: PendingSupervision, PendingUndeliverable, and
    // PendingSyncCleanup values in FIFO order. The receiver checks this queue
    // before the message plane.
    let (control_tx, control_rx) = mpsc::channel();

    // SI-2 common waker: the pipe carries no work or queue identity. Both
    // pympsc::Senders enqueue first, then write through this shared Waker.
    let (waker, read_fd) = pywaker::pipe()?;
    let waker = Arc::new(waker);

    // Inbox: both producer sides plus the single driver-owned receiver.
    Ok(Inbox {
        message_sender: Sender::new(messages_tx, waker.clone()),
        control_sender: Sender::new(control_tx, waker),
        receiver: Receiver {
            control: Mutex::new(control_rx),
            messages: Mutex::new(messages_rx),
            read_fd,
            stopping: Arc::new(AtomicBool::new(false)),
        },
    })
}

/// Blocking receiver used by one sync actor's Python driver thread.
///
/// Messages and control items use separate FIFO queues. Their senders share a
/// non-blocking pipe and write one byte after enqueueing; the byte carries no
/// work or identity and may be stale. It only tells the driver to inspect the
/// queues again.
///
/// On each `_sync_dispatch_loop` iteration, the Python driver calls
/// [`Receiver::next`] with the GIL held. `next` checks control first
/// and then messages, converts one available item, and returns it to the driver.
/// If both queues are empty, that same call releases the GIL and blocks in
/// `poll()`. A pipe wake makes it reacquire the GIL and check both queues again;
/// a stale wake may send it back to `poll()`. After `next` returns, the driver
/// processes the item and begins its next iteration.
///
/// A control item cannot interrupt an item already running. After that item
/// finishes, the next `next()` call selects queued control before queued
/// messages. Once the control queue is disconnected, the owning actor is gone,
/// so `next` returns `None` without taking another message. Closing the final
/// pipe writer wakes a blocked `next`, so disconnect cannot be missed.
///
/// `pympsc::Sender::send` boxes each value as `Box<dyn IntoPyObjectBox>` before
/// enqueueing it. When `next` dequeues a value, it invokes that erased
/// conversion: `PendingMessage` becomes `QueuedMessage`, `PendingSupervision`
/// becomes `QueuedSupervision`, `PendingUndeliverable` becomes
/// `QueuedUndeliverable`, and `PendingSyncCleanup` becomes `SyncCleanup`. Both
/// senders therefore stay GIL-free, and conversion happens on the driver
/// thread.
///
/// Actor cleanup sets `stopping` before enqueueing `PendingSyncCleanup`. If
/// `next` returns an older `QueuedSupervision` or `QueuedUndeliverable` first,
/// the driver reads this flag and drops that callback instead of invoking user
/// code. The resulting `SyncCleanup` runs if the driver is still running. The
/// Python driver is the only
/// consumer of the control and message queues.
///
/// `mpsc::Receiver` is `Send` but not `Sync`. `next` holds a shared reference
/// to this object across [`Python::detach`], which releases the GIL while the
/// driver waits. Wrapping each receiver in a `Mutex` makes the enclosing
/// `Receiver` safe across that boundary. The mutexes do not represent
/// multiple consumers and are uncontended in production.
///
/// `Actor::init` starts a Python `threading.Thread` whose target is
/// `_sync_dispatch_loop`, passing this receiver as the function's `inbox`
/// argument. The driver calls `next()` and `stopping()` on it directly; the
/// class is an internal Rust/Python boundary, not a user API.
#[pyclass(
    name = "SyncInbox",
    module = "monarch._rust_bindings.monarch_hyperactor.actor"
)]
pub(in super::super) struct Receiver {
    /// Receives `PendingSupervision`, `PendingUndeliverable`, and
    /// `PendingSyncCleanup` values sent through `pympsc::Sender::send`, which
    /// erases each value to `Box<dyn IntoPyObjectBox>` before enqueueing it.
    /// Checked before messages.
    control: Mutex<mpsc::Receiver<Box<dyn IntoPyObjectBox>>>,
    /// Receives `PendingMessage` values sent through `pympsc::Sender::send`,
    /// which erases each value to `Box<dyn IntoPyObjectBox>` before enqueueing
    /// it, even though this queue has one logical payload type.
    messages: Mutex<mpsc::Receiver<Box<dyn IntoPyObjectBox>>>,
    /// Read end of the shared wake pipe. `Inbox::message_sender` and
    /// `Inbox::control_sender` are `pympsc::Sender`s: each owns a different
    /// `mpsc::Sender<Box<dyn IntoPyObjectBox>>`, and both hold a clone of the
    /// same `Arc<pywaker::Waker>`. That shared `Waker` owns the write end.
    read_fd: OwnedFd,
    /// Set when cleanup begins, before its control item is enqueued.
    stopping: Arc<AtomicBool>,
}

// `next` borrows the receiver across `Python::detach`; keep that boundary
// compiler-checked as the representation changes.
const _: fn() = {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Receiver>
};

impl std::fmt::Debug for Receiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncInboxReceiver").finish_non_exhaustive()
    }
}

/// Result of checking one inbox plane without blocking.
enum Take {
    /// One queued value, removed from the plane.
    Item(Box<dyn IntoPyObjectBox>),
    /// No value is available, but a sender may still enqueue one.
    Empty,
    /// No value remains and every sender for this plane is disconnected.
    Closed,
}

/// Check one inbox plane without blocking, consuming at most one value.
/// The receiver lock is released before that value is converted into Python.
fn try_take(rx: &Mutex<mpsc::Receiver<Box<dyn IntoPyObjectBox>>>) -> Take {
    match rx
        .lock()
        .expect("sync inbox receiver lock should not be poisoned: only the driver takes it")
        .try_recv()
    {
        Ok(item) => Take::Item(item),
        Err(mpsc::TryRecvError::Empty) => Take::Empty,
        Err(mpsc::TryRecvError::Disconnected) => Take::Closed,
    }
}

impl Receiver {
    /// The SI-6 flag cleanup sets before enqueueing `PendingSyncCleanup`.
    pub(crate) fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stopping.clone()
    }

    /// Block until the pipe is readable, then drain it (SI-3, SI-5). A send may
    /// occur after `next` checks both queues but before this method enters
    /// `poll()`. That send enqueues its value before writing the pipe, so the
    /// byte is already waiting and `poll()` returns immediately instead of
    /// missing the wake. Closing every write end also makes the pipe readable.
    fn wait(&self) -> nix::Result<()> {
        // `poll` accepts a slice, so use a one-entry array and wait indefinitely
        // for data or pipe closure.
        let mut wake_fd = [PollFd::new(self.read_fd.as_fd(), PollFlags::POLLIN)];
        // EINTR means a signal interrupted the wait, not that the pipe failed, so
        // retry it.
        retry_eintr(|| poll(&mut wake_fd, PollTimeout::NONE))?;

        // Drain all wake bytes currently in the pipe. After the last byte, the
        // non-blocking read returns EAGAIN instead of waiting for a future wake.
        // The receiver ignores the byte values and count; pipe readability
        // alone tells it to inspect the queues. Each read removes up to 64
        // accumulated wake bytes.
        let mut buf = [0u8; 64];
        loop {
            // Retry if a signal interrupts the read.
            match retry_eintr(|| read(&self.read_fd, &mut buf)) {
                // EOF: every Sender has released the pipe's write end.
                Ok(0) => return Ok(()),
                // One batch drained; keep reading until the pipe is empty.
                Ok(_) => {}
                // The non-blocking pipe is now empty.
                Err(Errno::EAGAIN) => return Ok(()),
                // A real pipe failure propagates through `next` to the driver.
                Err(err) => return Err(err),
            }
        }
    }
}

#[pymethods]
impl Receiver {
    /// Return one queued item, preferring control over messages (SI-1). Return
    /// `None` once the control queue is disconnected and drained, without
    /// taking another message (SI-5). While no item is ready and control is
    /// still connected, call `self.wait()` with the GIL released to block on
    /// and drain the shared wake pipe, then reacquire the GIL and recheck both
    /// queues (SI-3, SI-4).
    fn next(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        loop {
            // SI-1: a ready control item wins without inspecting the message
            // plane. Conversion happens after `try_take` releases its lock.
            match try_take(&self.control) {
                Take::Item(item) => return item.into_py_object(py).map(Some),
                // SI-5: the owning actor is gone, so queued messages have no
                // one to answer to.
                Take::Closed => return Ok(None),
                Take::Empty => {}
            }
            match try_take(&self.messages) {
                // Return exactly one message, converted under the GIL.
                Take::Item(item) => return item.into_py_object(py).map(Some),
                // Nothing is ready and control is still connected. Release the
                // GIL while waiting, then retry the priority checks in this
                // same call (SI-3, SI-4).
                Take::Empty | Take::Closed => py.detach(|| self.wait()).map_pyerr()?,
            }
        }
    }

    /// Whether the actor's cleanup has begun. The driver claims a callback it
    /// has dequeued only if this is still false.
    fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }
}

// Compiled into the normal extension because Python tests load that extension,
// not Rust's `#[cfg(test)]` build.
mod testing {
    use pyo3::pyfunction;
    use pyo3::types::PyAnyMethods;
    use pyo3::wrap_pyfunction;

    use super::*;
    use crate::pympsc::testing::PyTestSender;

    /// A sync inbox with its message and control senders.
    #[pyfunction(name = "_sync_inbox_for_test")]
    fn sync_inbox_for_test(_py: Python<'_>) -> PyResult<(PyTestSender, PyTestSender, Receiver)> {
        let Inbox {
            message_sender,
            control_sender,
            receiver,
        } = new().map_pyerr()?;
        // Wrap only the Rust senders; Receiver is already a pyclass.
        Ok((
            PyTestSender::new(message_sender),
            PyTestSender::new(control_sender),
            receiver,
        ))
    }

    pub(in super::super) fn register_python_bindings(
        testing_mod: &Bound<'_, PyModule>,
    ) -> PyResult<()> {
        let sync_inbox_for_test = wrap_pyfunction!(sync_inbox_for_test, testing_mod)?;
        sync_inbox_for_test.setattr(
            "__module__",
            "monarch._rust_bindings.monarch_hyperactor.testing",
        )?;
        testing_mod.add_function(sync_inbox_for_test)?;
        Ok(())
    }
}

pub(in super::super) fn register_python_bindings(
    hyperactor_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    hyperactor_mod.add_class::<Receiver>()?;
    Ok(())
}

pub(in super::super) fn register_testing_python_bindings(
    testing_mod: &Bound<'_, PyModule>,
) -> PyResult<()> {
    testing::register_python_bindings(testing_mod)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::pympsc::testing::Unconvertible;
    use crate::runtime::GilSite;
    use crate::runtime::ensure_python;
    use crate::runtime::monarch_with_gil_blocking;

    /// Receive and extract the integer payload used by these tests. The caller
    /// supplies the GIL token so tests can coordinate another thread against
    /// the GIL release inside `next`.
    fn next_value_with_gil(receiver: &Receiver, py: Python<'_>) -> Option<i64> {
        receiver
            .next(py)
            .expect("next should not fail")
            .map(|item| item.extract::<i64>(py).expect("items are ints"))
    }

    /// Receive through the production GIL and Python-conversion path.
    fn next_value(receiver: &Receiver) -> Option<i64> {
        monarch_with_gil_blocking(GilSite::Test, |py| next_value_with_gil(receiver, py))
    }

    // SI-2, SI-3: work published before a wait is visible immediately.
    #[test]
    fn a_send_before_the_wait_is_received() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender: _control_sender_guard,
            receiver,
        } = new().unwrap();
        message_sender.send(1i64).unwrap();
        assert_eq!(next_value(&receiver), Some(1));
    }

    // SI-3, SI-4: a send wakes a blocked `next`, which releases the GIL while
    // it waits.
    #[test]
    fn a_send_during_the_wait_wakes_it() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender: _control_sender_guard,
            receiver,
        } = new().unwrap();
        let (received, sender) = monarch_with_gil_blocking(GilSite::Test, |py| {
            let sender = thread::spawn(move || {
                // The test thread holds the GIL until `next` detaches. If
                // `next` kept it while blocked, this thread could never reach
                // `send` and the test would hang.
                monarch_with_gil_blocking(GilSite::Test, |_py| {});
                message_sender.send(7i64).unwrap();
                // Keep the sender alive until after the assertion, so its drop
                // cannot add a disconnect wake to the behavior under test.
                message_sender
            });
            (next_value_with_gil(&receiver, py), sender)
        });
        assert_eq!(received, Some(7));
        drop(sender.join().unwrap());
    }

    // SI-1: the message plane is FIFO.
    #[test]
    fn a_backlog_is_delivered_in_order() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender: _control_sender_guard,
            receiver,
        } = new().unwrap();
        for value in 1..=3i64 {
            message_sender.send(value).unwrap();
        }
        let received: Vec<_> = (0..3).map(|_| next_value(&receiver)).collect();
        assert_eq!(received, vec![Some(1), Some(2), Some(3)]);
    }

    // SI-1: control is selected before an existing message backlog.
    #[test]
    fn a_control_item_jumps_the_message_backlog() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender,
            receiver,
        } = new().unwrap();
        message_sender.send(1i64).unwrap();
        message_sender.send(2i64).unwrap();
        control_sender.send(99i64).unwrap();
        let received: Vec<_> = (0..3).map(|_| next_value(&receiver)).collect();
        assert_eq!(
            received,
            vec![Some(99), Some(1), Some(2)],
            "the control item should come first, then the messages in order"
        );
    }

    // SI-5: once control disconnects, a control item queued before the close
    // is still returned, but the queued message is not.
    #[test]
    fn a_closed_control_plane_returns_none_without_draining_messages() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender,
            receiver,
        } = new().unwrap();
        message_sender.send(5i64).unwrap();
        control_sender.send(7i64).unwrap();
        drop(message_sender);
        drop(control_sender);
        assert_eq!(next_value(&receiver), Some(7));
        assert_eq!(next_value(&receiver), None);
    }

    // SI-4: conversion occurs after dequeue on the GIL-holding receiver.
    #[test]
    fn a_conversion_failure_raises_and_consumes_the_item() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender: _control_sender_guard,
            receiver,
        } = new().unwrap();
        message_sender.send(Unconvertible).unwrap();
        // A valid item behind the failing one proves that conversion failure
        // consumes only the item it failed to convert.
        message_sender.send(9i64).unwrap();
        let error = monarch_with_gil_blocking(GilSite::Test, |py| {
            receiver
                .next(py)
                .expect_err("the conversion should fail")
                .to_string()
        });
        assert!(error.contains("unconvertible test item"), "{error}");
        assert_eq!(next_value(&receiver), Some(9));
    }

    // SI-5: the final sender's drop closes the pipe's write end, and that EOF
    // wakes a waiting `next`; without it this test hangs.
    #[test]
    fn a_disconnect_during_the_wait_wakes_it() {
        ensure_python();
        let Inbox {
            message_sender,
            control_sender,
            receiver,
        } = new().unwrap();
        let (received, dropper) = monarch_with_gil_blocking(GilSite::Test, |py| {
            let dropper = thread::spawn(move || {
                // The test thread holds the GIL until `next` detaches, so the
                // senders cannot be dropped before `next` begins waiting.
                monarch_with_gil_blocking(GilSite::Test, |_py| {});
                drop(control_sender);
                drop(message_sender);
            });
            (next_value_with_gil(&receiver, py), dropper)
        });
        assert_eq!(received, None);
        dropper.join().unwrap();
    }
}
