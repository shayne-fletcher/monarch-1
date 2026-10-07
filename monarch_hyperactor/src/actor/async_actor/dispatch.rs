/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Event-loop-backed execution state for an async [`PythonActor`].

use super::super::*;
use super::inbox::Receiver;

/// Async actor state retained between construction, initialization, and cleanup.
#[derive(Debug)]
pub(in super::super) struct State {
    /// Taken once by `init` and split between the two loop tasks.
    inbox_receiver: Option<Receiver>,
    /// The task-local state for the actor's dedicated event loop thread.
    task_locals: pyo3_async_runtimes::TaskLocals,
}

impl State {
    pub(in super::super) fn new(py: Python<'_>, inbox_receiver: Receiver) -> Self {
        Self {
            inbox_receiver: Some(inbox_receiver),
            task_locals: Python::detach(py, create_task_locals),
        }
    }

    /// Start the message and callback tasks on the actor's event loop.
    pub(in super::super) fn init(
        &mut self,
        py: Python<'_>,
        actor_mesh_mod: &Bound<'_, PyModule>,
        actor: &Py<PyAny>,
        self_instance: &Py<PyInstance>,
        callbacks_pending: Arc<AtomicUsize>,
    ) -> anyhow::Result<()> {
        let Receiver { messages, control } = self
            .inbox_receiver
            .take()
            .expect("async inbox receiver already taken");
        let loops = [
            (
                "message loop",
                actor_mesh_mod.call_method(
                    "_dispatch_loop",
                    (
                        actor.clone_ref(py),
                        messages,
                        self_instance.clone_ref(py),
                        CallbacksPending(callbacks_pending),
                    ),
                    None,
                )?,
            ),
            (
                "callback loop",
                actor_mesh_mod.call_method(
                    "_callback_loop",
                    (actor.clone_ref(py), control, self_instance.clone_ref(py)),
                    None,
                )?,
            ),
        ];
        for (name, awaitable) in loops {
            let future =
                pyo3_async_runtimes::into_future_with_locals(&self.task_locals, awaitable)?;
            tokio::spawn(async move {
                if let Err(error) = future.await {
                    tracing::error!("{} error: {}", name, error);
                }
            });
        }
        Ok(())
    }

    /// Run user cleanup on the actor's event loop, then cancel its remaining
    /// tasks and stop the loop. Loop shutdown still runs when user cleanup
    /// fails; the cleanup failure takes precedence when both fail.
    pub(in super::super) async fn cleanup(
        &self,
        actor: &Py<PyAny>,
        instance: Option<&Py<PyInstance>>,
        cx: &Context<'_, PythonActor>,
        this: &Instance<PythonActor>,
        error: Option<String>,
    ) -> anyhow::Result<()> {
        let future = monarch_with_gil(GilSite::EndpointCleanup, |py| {
            let py_cx = cleanup_context(py, cx, this, instance)?;
            let actor = actor.bind(py);
            // Some tests don't use the Actor base class, so add this check
            // to be defensive.
            match actor.hasattr("__cleanup__") {
                Ok(false) | Err(_) => return Ok(None),
                _ => {}
            }
            let awaitable = actor
                .call_method("__cleanup__", (py_cx.bind(py), error), None)
                .map_err(|error| anyhow::Error::from(SerializablePyErr::from(py, &error)))?;
            if awaitable.is_none() {
                Ok(None)
            } else {
                pyo3_async_runtimes::into_future_with_locals(&self.task_locals, awaitable)
                    .map(Some)
                    .map_err(anyhow::Error::from)
            }
        })
        .await;
        let cleanup_result = match future {
            Ok(Some(future)) => future.await.map(|_| ()).map_err(anyhow::Error::from),
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        let loop_shutdown_result = shutdown_loop(&self.task_locals);
        cleanup_result?;
        loop_shutdown_result
    }
}

/// Construct the context for async cleanup, preserving the inherited fallback
/// for cleanup after partial initialization.
fn cleanup_context(
    py: Python<'_>,
    cx: &Context<'_, PythonActor>,
    this: &Instance<PythonActor>,
    instance: Option<&Py<PyInstance>>,
) -> anyhow::Result<Py<PyContext>> {
    let instance = match instance {
        Some(instance) => instance.clone_ref(py),
        None => {
            // Async cleanup may run after initialization failed before a
            // cached PyInstance was installed. Derive one from the Rust actor
            // instance so `__cleanup__` still receives a Context.
            let instance: PyInstance = this.into();
            Py::new(py, instance)?
        }
    };
    Ok(Py::new(py, PyContext::new(cx, instance))?)
}

/// Cancel every task on the actor's event loop, wait for the loop to observe
/// those cancellations, then ask it to stop.
fn shutdown_loop_with_gil(
    py: Python<'_>,
    task_locals: &pyo3_async_runtimes::TaskLocals,
) -> PyResult<()> {
    let asyncio = py.import("asyncio")?;
    let event_loop = task_locals.event_loop(py);
    let tasks = asyncio.call_method1("all_tasks", (&event_loop,))?;
    let mut has_tasks = false;
    for task in tasks.try_iter()? {
        let task = task?;
        let cancel = task.getattr("cancel")?;
        // The loop thread, not this Rust caller, must invoke `Task.cancel`.
        event_loop.call_method1("call_soon_threadsafe", (cancel,))?;
        has_tasks = true;
    }
    if has_tasks {
        // This zero-delay coroutine is queued behind the cancellation
        // callbacks. Waiting for its result gives the loop a turn to deliver
        // cancellation before it is stopped.
        asyncio
            .call_method1(
                "run_coroutine_threadsafe",
                (asyncio.call_method1("sleep", (0,))?, &event_loop),
            )?
            .call_method0("result")?;
    }
    // `stop` must also run on the loop's owning thread.
    let stop = event_loop.getattr("stop")?;
    event_loop.call_method1("call_soon_threadsafe", (stop,))?;
    Ok(())
}

/// Acquire the GIL for loop shutdown and translate a Python failure into the
/// error returned by Rust actor cleanup.
fn shutdown_loop(task_locals: &pyo3_async_runtimes::TaskLocals) -> anyhow::Result<()> {
    monarch_with_gil_blocking(GilSite::Stop, |py| -> anyhow::Result<()> {
        shutdown_loop_with_gil(py, task_locals)
            .map_err(|error| anyhow::Error::from(SerializablePyErr::from(py, &error)))?;
        Ok(())
    })
}

/// Create TaskLocals backed by a new event loop on a dedicated Python thread.
fn create_task_locals() -> pyo3_async_runtimes::TaskLocals {
    monarch_with_gil_blocking(GilSite::TaskLocals, |py| {
        let asyncio = Python::import(py, "asyncio").unwrap();
        let event_loop = asyncio.call_method0("new_event_loop").unwrap();
        let task_locals = pyo3_async_runtimes::TaskLocals::new(event_loop.clone())
            .copy_context(py)
            .unwrap();

        let kwargs = PyDict::new(py);
        let target = event_loop.getattr("run_forever").unwrap();
        kwargs.set_item("target", target).unwrap();
        // Need to make this a daemon thread, otherwise shutdown will hang.
        kwargs.set_item("daemon", true).unwrap();
        kwargs
            .set_item("name", "monarch-actor-event-loop")
            .expect("thread name should be accepted");
        let thread = py
            .import("threading")
            .unwrap()
            .call_method("Thread", (), Some(&kwargs))
            .unwrap();
        mark_actor_event_loop_thread(&thread, &event_loop)
            .expect("actor event loop should attach to its owning thread");
        thread.call_method0("start").unwrap();
        task_locals
    })
}
