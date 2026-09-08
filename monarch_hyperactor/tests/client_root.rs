/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! End-to-end proof that Python bootstrap gives the application actor hierarchy
//! access to services owned by the program root.
//!
//! Python's `RootClientActor` and the root ProcAgent are different actors on the
//! same client proc. `RootClientActor` is the program's singleton root
//! orchestration actor; the root ProcAgent owns program-level services.
//! `bootstrap_host` binds the restricted client-root API on that ProcAgent and
//! stores the resulting `ClientRootRef` in the root actor's environment, where
//! descendants can inherit it.
//!
//! This test drives that production bootstrap, reads the capability from the
//! root actor's environment, and uses it from an unrelated `Proc::client` to
//! ensure a service. The `Proc::client` is only a temporary request/reply
//! context; the service must be created on the root proc, not the caller's proc.
//!
//! Rust's `GlobalClientActor` provides the same root role. Its bootstrap path is
//! covered by `hyperactor_mesh::global_context::tests::test_bootstrap_seeds_client_root`;
//! capability inheritance and the missing-capability case are covered by
//! `hyperactor_mesh::proc_agent::tests::client_root_service_observes_inherited_capability`
//! and `hyperactor_mesh::proc_agent::tests::client_root_from_env_absent_fails_closed`.
//!
//! `bootstrap_host` initializes process-global state that shutdown does not
//! reset. This must remain the only test in this binary that calls it.
//!
//! Requires the real `monarch` Python package via `py_deps` on
//! `test_monarch_hyperactor`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use hyperactor::Proc;
use hyperactor::actor::ActorStatus;
use hyperactor::actor::Signal;
use hyperactor::channel::ChannelTransport;
use hyperactor::id::ProcId;
use hyperactor::proc::Instance;
use hyperactor_mesh::client_root::ClientRootRef;
use hyperactor_mesh::client_root::ClientRootService;
use hyperactor_mesh::testactor::TestActor;
use monarch_hyperactor::actor::PythonActor;
use monarch_hyperactor::context::PyInstance;
use monarch_hyperactor::host_mesh::PyBootstrapCommand;
use monarch_hyperactor::host_mesh::python_client_root;
use monarch_hyperactor::pytokio::PyPythonTask;
use monarch_hyperactor::runtime::GilSite;
use monarch_hyperactor::runtime::get_tokio_runtime;
use monarch_hyperactor::runtime::monarch_with_gil_blocking;
use monarch_hyperactor::shape::PyExtent;
use pyo3::PyTypeInfo;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyTuple;

const WAIT: Duration = Duration::from_secs(30);
const NO_SIGNAL_WAIT: Duration = Duration::from_secs(1);

// Both tests use the embedded Python interpreter and Monarch runtime, while
// the bootstrap test also owns process-global one-shot lifecycle state.
// Do not recover a poisoned guard: a panic can leave that external state
// inconsistent even though the mutex payload itself is only `()`.
static CLIENT_ROOT_TEST_LOCK: Mutex<()> = Mutex::new(());

type PythonTaskFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send + 'static>>;

#[test]
#[cfg_attr(not(fbcode_build), ignore)]
fn python_bootstrap_seeds_client_root_on_local_proc_agent() -> Result<()> {
    let _test_guard = CLIENT_ROOT_TEST_LOCK
        .lock()
        .expect("client-root test lock must not be poisoned");
    pyo3::Python::initialize();

    let test_result = assert_bootstrap_is_lazy_and_rejects_invalid_via()
        .and_then(|()| exercise_python_bootstrap_contract());
    let shutdown_result = shutdown_python_host_if_present();

    match (test_result, shutdown_result) {
        (Ok(()), Ok(true)) => Ok(()),
        (Ok(()), Ok(false)) => anyhow::bail!("client-root contract completed without a host"),
        (Err(test_error), Ok(_)) => Err(test_error),
        (Ok(()), Err(shutdown_error)) => Err(shutdown_error),
        (Err(test_error), Err(shutdown_error)) => anyhow::bail!(
            "client-root contract failed: {test_error:#}; host shutdown also failed: {shutdown_error:#}"
        ),
    }
}

fn assert_bootstrap_is_lazy_and_rejects_invalid_via() -> Result<()> {
    anyhow::ensure!(
        python_client_root()?.is_none(),
        "the Python client root must be absent before bootstrap",
    );
    assert_shutdown_without_host_is_runtime_error()?;

    monarch_with_gil_blocking(GilSite::Test, |py| -> Result<()> {
        py.run(c_str!("import monarch._rust_bindings"), None, None)?;
        let host_mesh_mod = py.import("monarch._rust_bindings.monarch_hyperactor.host_mesh")?;
        let host_mesh_src = py.import("monarch._src.actor.host_mesh")?;
        let cmd = host_mesh_src.getattr("default_bootstrap_cmd")?.call0()?;

        // Constructing and dropping the raw task must not publish the
        // observable client-root or host-agent registrations.
        let discarded = host_mesh_mod.getattr("bootstrap_host")?.call1((&cmd,))?;
        drop(discarded);

        let error = match host_mesh_mod
            .getattr("bootstrap_host")?
            .call1((&cmd, "invalid://scheme"))
        {
            Ok(_) => anyhow::bail!("an invalid via address returned a task"),
            Err(error) => error,
        };
        anyhow::ensure!(
            error.get_type(py).is(PyValueError::type_object(py)),
            "invalid via must raise exactly ValueError, got {error}",
        );
        let message = error.value(py).to_string();
        anyhow::ensure!(
            message.contains("via address")
                && message.contains("unsupported ZMQ scheme")
                && message.contains("invalid"),
            "invalid via must identify the address, unsupported scheme, and supplied scheme: {message}",
        );
        Ok(())
    })?;

    anyhow::ensure!(
        python_client_root()?.is_none(),
        "discarded and rejected bootstrap calls must not register a client root",
    );
    assert_shutdown_without_host_is_runtime_error()
}

fn assert_shutdown_without_host_is_runtime_error() -> Result<()> {
    monarch_with_gil_blocking(GilSite::Test, |py| -> Result<()> {
        py.run(c_str!("import monarch._rust_bindings"), None, None)?;
        let host_mesh_mod = py.import("monarch._rust_bindings.monarch_hyperactor.host_mesh")?;
        let error = match host_mesh_mod.getattr("shutdown_local_host_mesh")?.call0() {
            Ok(_) => anyhow::bail!("shutdown without a host returned a task"),
            Err(error) => error,
        };
        anyhow::ensure!(
            is_no_local_host_error(py, &error),
            "shutdown without a host must raise exactly RuntimeError and identify the missing local host mesh: {error}",
        );
        Ok(())
    })
}

fn is_no_local_host_error(py: Python<'_>, error: &PyErr) -> bool {
    if !error.get_type(py).is(PyRuntimeError::type_object(py)) {
        return false;
    }
    let normalized = error.value(py).to_string().to_ascii_lowercase();
    normalized.contains("no local host mesh") && normalized.contains("shutdown")
}

fn exercise_python_bootstrap_contract() -> Result<()> {
    // Run the production Python host bootstrap and recover the native instance
    // of Python's RootClientActor (the program's singleton root orchestration actor).
    // Read the restricted root ProcAgent reference from that actor's environment
    // and record which proc hosts the actor.
    let (root_proc_agent_capability, root_actor_proc_id, host_mesh, root_actor_instance) =
        monarch_with_gil_blocking(GilSite::Test, |py| {
            py.run(c_str!("import monarch._rust_bindings"), None, None)?;

            let host_mesh_mod = py.import("monarch._rust_bindings.monarch_hyperactor.host_mesh")?;
            let host_mesh_src = py.import("monarch._src.actor.host_mesh")?;
            let cmd = host_mesh_src.getattr("default_bootstrap_cmd")?.call0()?;

            // The native binding returns a PyPythonTask holding the Rust
            // bootstrap future. block_on consumes it and drives it on Monarch's
            // embedded Tokio runtime, producing
            // (host_mesh, proc_mesh, root_actor_instance).
            let bootstrap_result = host_mesh_mod
                .getattr("bootstrap_host")?
                .call1((cmd,))?
                .call_method0("block_on")?;
            let bootstrap_result = bootstrap_result.cast::<PyTuple>()?;
            let host_mesh = bootstrap_result.get_item(0)?.unbind();
            let root_actor_obj = bootstrap_result.get_item(2)?;
            let root_actor = root_actor_obj.cast::<PyInstance>()?.borrow();
            let root_actor_instance: &Instance<PythonActor> = &root_actor;

            // This is the contract under test: the root actor is not the root
            // ProcAgent. Its persistent environment must contain the restricted
            // capability that bootstrap_host bound to that separate ProcAgent.
            let root_actor_environment = root_actor_instance.actor_environment();
            let root_proc_agent_capability = ClientRootRef::from_env(root_actor_environment)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let root_actor_proc_id = root_actor_instance.self_addr().id().proc_id().clone();
            Ok::<_, PyErr>((
                root_proc_agent_capability,
                root_actor_proc_id,
                host_mesh,
                root_actor_instance.clone_for_py(),
            ))
        })?;

    // Exercise the capability from an unrelated caller context. This caller is
    // only a mailbox and identity; it neither owns nor places the service.
    let (service, caller_proc_id) =
        ensure_root_service_with_timeout(&root_proc_agent_capability, "client_root_test")?;

    // The root ProcAgent and RootClientActor live on the same canonical local
    // proc. A service created through this capability must therefore land on
    // that proc, rather than on the unrelated caller's proc.
    let service_proc_id = service.actor_addr().id().proc_id();
    anyhow::ensure!(
        service_proc_id == &root_actor_proc_id,
        "root-owned service landed on {service_proc_id}, not root proc {root_actor_proc_id}",
    );
    anyhow::ensure!(
        service_proc_id != &caller_proc_id,
        "root-owned service was incorrectly created on requester proc {caller_proc_id}",
    );

    // The raw shutdown task is lazy. Dropping it must leave the registered
    // host agent and root service reachable.
    discard_python_host_shutdown_task()?;
    assert_host_agent_accepts_probe_before_shutdown(&host_mesh, &root_actor_instance)?;
    let (service_after_discard, _) =
        ensure_root_service_with_timeout(&root_proc_agent_capability, "client_root_after_discard")?;
    anyhow::ensure!(
        service_after_discard.actor_addr() == service.actor_addr(),
        "discarding the raw shutdown task replaced or stopped the root-owned service",
    );

    Ok(())
}

async fn ensure_root_service(
    root_proc_agent_capability: &ClientRootRef,
    caller_name: &str,
) -> Result<(hyperactor::ActorRef<TestActor>, ProcId)> {
    // Proc::direct registers network I/O, so construct it inside the runtime
    // that drives the request.
    let test_proc = Proc::direct(ChannelTransport::Unix.any(), format!("{caller_name}_proc"))?;
    let caller = test_proc.client(caller_name);
    let caller_proc_id = caller.self_addr().id().proc_id().clone();

    // This is only a local descriptor. `ensure` performs the request, creating
    // or reusing the named root-owned service.
    let service_descriptor =
        ClientRootService::<TestActor>::declare("client-root-integration-test");
    let service_ref = service_descriptor
        .ensure(&caller, root_proc_agent_capability, ())
        .await?;
    Ok((service_ref, caller_proc_id))
}

fn ensure_root_service_with_timeout(
    root_proc_agent_capability: &ClientRootRef,
    caller_name: &str,
) -> Result<(hyperactor::ActorRef<TestActor>, ProcId)> {
    get_tokio_runtime().block_on(async {
        tokio::time::timeout(
            WAIT,
            ensure_root_service(root_proc_agent_capability, caller_name),
        )
        .await
        .map_err(|_| anyhow::anyhow!("timed out ensuring root service for {caller_name}"))?
    })
}

fn discard_python_host_shutdown_task() -> Result<()> {
    monarch_with_gil_blocking(GilSite::Test, |py| -> PyResult<()> {
        let host_mesh_mod = py.import("monarch._rust_bindings.monarch_hyperactor.host_mesh")?;
        let discarded = host_mesh_mod.getattr("shutdown_local_host_mesh")?.call0()?;
        drop(discarded);
        Ok(())
    })?;
    Ok(())
}

fn assert_host_agent_accepts_probe_before_shutdown(
    host: &Py<PyAny>,
    instance: &Instance<PythonActor>,
) -> Result<()> {
    // Use the exact HostMesh returned by this test's Python bootstrap. A
    // process-global lookup can prefer an unrelated Rust host initialized by
    // another test in this binary, which would make the shutdown witness pass
    // against the wrong HostAgent.
    let task = monarch_with_gil_blocking(GilSite::Test, |py| -> PyResult<PythonTaskFuture> {
        let command = Py::new(
            py,
            PyBootstrapCommand {
                program: "/bin/false".to_string(),
                arg0: None,
                args: Vec::new(),
                env: Default::default(),
            },
        )?;
        let extent = Py::new(py, PyExtent::new(vec!["probes".to_string()], vec![1])?)?;
        let instance = Py::new(py, PyInstance::from(instance))?;
        let host = host.bind(py).call_method1("with_bootstrap", (command,))?;
        let task = host.call_method1(
            "spawn_nonblocking",
            (instance, "shutdown-laziness-barrier", extent),
        )?;
        take_python_task(&task)
    })?;

    // The probe deliberately launches /bin/false. A live HostAgent accepts
    // the request and reports that exit; an agent that already handled an
    // eager ShutdownHost rejects creation before launching the process.
    // The current PyPythonTask constructor only stores the future; this probe
    // does not claim to cover an arbitrary future implementation that detaches
    // work before it posts to the agent.
    let outcome = get_tokio_runtime()
        .block_on(async { tokio::time::timeout(WAIT, task).await })
        .map_err(|_| anyhow::anyhow!("timed out probing the host agent after task drop"))?;
    let error = match outcome {
        Ok(_) => anyhow::bail!("the /bin/false probe process unexpectedly succeeded"),
        Err(error) => error,
    };
    monarch_with_gil_blocking(GilSite::Test, |py| -> Result<()> {
        anyhow::ensure!(
            error.get_type(py).is(PyValueError::type_object(py)),
            "the failed probe must raise exactly ValueError, got {error}",
        );
        let message = error.value(py).to_string();
        // This structured status fragment is the available discriminator that
        // the HostAgent accepted and launched /bin/false rather than rejecting
        // the request before process creation.
        anyhow::ensure!(
            message.contains("exit_code: 1"),
            "the host agent did not accept and launch the probe process: {message}",
        );
        Ok(())
    })?;
    Ok(())
}

fn shutdown_python_host_if_present() -> Result<bool> {
    // Always exercise the public shutdown path after the contract body returns,
    // including when a lookup, ensure, or ownership check fails.
    let task =
        monarch_with_gil_blocking(GilSite::Test, |py| -> Result<Option<PythonTaskFuture>> {
            let host_mesh_mod = py.import("monarch._rust_bindings.monarch_hyperactor.host_mesh")?;
            match host_mesh_mod.getattr("shutdown_local_host_mesh")?.call0() {
                Ok(task) => Ok(Some(take_python_task(&task)?)),
                Err(error) => {
                    if is_no_local_host_error(py, &error) {
                        Ok(None)
                    } else {
                        Err(anyhow::Error::new(error))
                    }
                }
            }
        })?;
    let Some(task) = task else {
        return Ok(false);
    };
    let result = get_tokio_runtime()
        .block_on(async { tokio::time::timeout(WAIT, task).await })
        .map_err(|_| anyhow::anyhow!("timed out shutting down the Python host"))??;
    assert_python_unit(&result, "host shutdown")?;

    Ok(true)
}

fn take_python_task(task: &Bound<'_, PyAny>) -> PyResult<PythonTaskFuture> {
    let task = task.cast::<PyPythonTask>()?;

    task.borrow_mut().take_task()
}

fn assert_python_unit(result: &Py<PyAny>, operation: &str) -> Result<()> {
    monarch_with_gil_blocking(GilSite::Test, |py| -> Result<()> {
        let result = result.bind(py).cast::<PyTuple>().map_err(|error| {
            anyhow::anyhow!("Rust unit did not convert to a Python tuple: {error}")
        })?;
        anyhow::ensure!(
            result.is_empty(),
            "{operation} must resolve to the empty-tuple conversion of Rust unit",
        );
        Ok(())
    })
}

fn stop_and_wait_task(instance: &Py<PyInstance>, reason: &str) -> PyResult<PythonTaskFuture> {
    monarch_with_gil_blocking(GilSite::Test, |py| {
        let task = instance.bind(py).call_method1("stop_and_wait", (reason,))?;
        take_python_task(&task)
    })
}

#[test]
#[cfg_attr(not(fbcode_build), ignore)]
fn py_instance_stop_and_wait_is_lazy_and_repeatable() -> Result<()> {
    let _test_guard = CLIENT_ROOT_TEST_LOCK
        .lock()
        .expect("client-root test lock must not be poisoned");
    pyo3::Python::initialize();

    let runtime = tokio::runtime::Runtime::new()?;
    let actor_instance = {
        let _runtime_guard = runtime.enter();
        Proc::isolated().actor_instance::<PythonActor>("stop_and_wait_test")?
    };
    let instance = actor_instance.instance.clone_for_py();
    let mut signal = actor_instance.signal;
    let py_instance =
        monarch_with_gil_blocking(GilSite::Test, |py| Py::new(py, PyInstance::from(&instance)))?;

    {
        let _runtime_guard = runtime.enter();
        monarch_with_gil_blocking(GilSite::Test, |py| -> PyResult<()> {
            let discarded = py_instance
                .bind(py)
                .call_method1("stop_and_wait", ("discarded stop",))?;
            drop(discarded);
            Ok(())
        })?;
    }
    // PyPythonTask::new currently stores rather than spawns the future, so no
    // producer remains after this drop. The bounded wait observes that state;
    // the driven and repeated controls then consume this same channel in order
    // and require their exact reasons, catching any queued stray signal. This
    // does not claim to cover a future implementation that detaches work.
    let absent =
        runtime.block_on(async { tokio::time::timeout(NO_SIGNAL_WAIT, signal.recv()).await });
    anyhow::ensure!(
        absent.is_err(),
        "dropping an undriven stop_and_wait task must not signal the actor; got {absent:?}",
    );

    let reason = "driven stop".to_string();
    let task = {
        let _runtime_guard = runtime.enter();
        stop_and_wait_task(&py_instance, &reason)?
    };
    let status_instance = instance.clone_for_py();
    let driver_reason = reason.clone();
    let signal_driver = runtime.spawn(async move {
        let observed = tokio::time::timeout(WAIT, signal.recv())
            .await
            .map_err(|_| anyhow::anyhow!("stop signal did not arrive before the deadline"))?
            .ok_or_else(|| anyhow::anyhow!("stop signal channel closed"))?;
        match observed {
            Signal::Stop(actual) => anyhow::ensure!(
                actual == driver_reason,
                "expected stop reason {driver_reason:?}, got {actual:?}",
            ),
            other => anyhow::bail!("expected Stop signal, got {other:?}"),
        }
        status_instance.change_status(ActorStatus::Stopped(driver_reason));
        Ok::<_, anyhow::Error>(signal)
    });

    let (result, mut signal) = runtime.block_on(async {
        let result = tokio::time::timeout(WAIT, task)
            .await
            .map_err(|_| anyhow::anyhow!("stop_and_wait did not complete before the deadline"))??;
        let signal = tokio::time::timeout(WAIT, signal_driver)
            .await
            .map_err(|_| anyhow::anyhow!("signal driver did not complete before the deadline"))?
            .map_err(|error| anyhow::anyhow!("signal driver panicked: {error}"))??;
        Ok::<_, anyhow::Error>((result, signal))
    })?;
    assert_python_unit(&result, "stop_and_wait")?;

    let repeated_reason = "repeated stop";
    let repeated = {
        let _runtime_guard = runtime.enter();
        stop_and_wait_task(&py_instance, repeated_reason)?
    };
    let (repeated_result, repeated_signal) = runtime.block_on(async {
        let result = tokio::time::timeout(WAIT, repeated).await.map_err(|_| {
            anyhow::anyhow!("repeated stop_and_wait did not complete before the deadline")
        })??;
        let signal = tokio::time::timeout(WAIT, signal.recv())
            .await
            .map_err(|_| {
                anyhow::anyhow!("repeated stop signal did not arrive before the deadline")
            })?
            .ok_or_else(|| anyhow::anyhow!("stop signal channel closed"))?;
        Ok::<_, anyhow::Error>((result, signal))
    })?;
    assert_python_unit(&repeated_result, "repeated stop_and_wait")?;
    anyhow::ensure!(
        matches!(repeated_signal, Signal::Stop(reason) if reason == repeated_reason),
        "repeated stop_and_wait must signal the actor again",
    );

    Ok(())
}
