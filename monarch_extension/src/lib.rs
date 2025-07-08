/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "tensor_engine")]
mod client;
#[cfg(feature = "tensor_engine")]
mod controller;
#[cfg(feature = "tensor_engine")]
pub mod convert;
#[cfg(feature = "tensor_engine")]
mod debugger;
#[cfg(feature = "tensor_engine")]
mod mesh_controller;
mod simulation_tools;
mod simulator_client;
#[cfg(feature = "tensor_engine")]
mod tensor_worker;

use pyo3::prelude::*;

#[pyfunction]
fn has_tensor_engine() -> bool {
    cfg!(feature = "tensor_engine")
}

fn get_or_add_new_module<'py>(
    module: &Bound<'py, PyModule>,
    module_name: &str,
) -> PyResult<Bound<'py, pyo3::types::PyModule>> {
    let mut current_module = module.clone();
    let mut parts = Vec::new();
    for part in module_name.split(".") {
        parts.push(part);
        let submodule = current_module.getattr(part).ok();
        if let Some(submodule) = submodule {
            current_module = submodule.extract()?;
        } else {
            let new_module = PyModule::new(current_module.py(), part)?;
            current_module.add_submodule(&new_module)?;
            current_module
                .py()
                .import("sys")?
                .getattr("modules")?
                .set_item(
                    format!("monarch._rust_bindings.{}", parts.join(".")),
                    new_module.clone(),
                )?;
            current_module = new_module;
        }
    }
    Ok(current_module)
}

#[pymodule]
#[pyo3(name = "_rust_bindings")]
pub fn mod_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "tensor_engine")]
    {
        client::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.client",
        )?)?;
        tensor_worker::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.tensor_worker",
        )?)?;
        controller::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.controller",
        )?)?;
        debugger::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.debugger",
        )?)?;
        monarch_messages::debugger::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_messages.debugger",
        )?)?;
        simulator_client::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.simulator_client",
        )?)?;
        ::controller::bootstrap::register_python_bindings(&get_or_add_new_module(
            module,
            "controller.bootstrap",
        )?)?;
        ::monarch_tensor_worker::bootstrap::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_tensor_worker.bootstrap",
        )?)?;
        crate::convert::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.convert",
        )?)?;
        crate::mesh_controller::register_python_bindings(&get_or_add_new_module(
            module,
            "monarch_extension.mesh_controller",
        )?)?;
    }
    simulation_tools::register_python_bindings(&get_or_add_new_module(
        module,
        "monarch_extension.simulation_tools",
    )?)?;

    // Add feature detection function
    module.add_function(wrap_pyfunction!(has_tensor_engine, module)?)?;

    Ok(())
}
