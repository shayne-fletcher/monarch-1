/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_mesh::bootstrap_or_die;
use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::pyfunction;
use pyo3::types::PyAnyMethods;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[pyo3(signature = ())]
pub fn bootstrap_main(py: Python) -> PyResult<Bound<PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py::<_, ()>(py, async move {
        bootstrap_or_die().await;
    })
}

pub fn register_python_bindings(hyperactor_mod: &Bound<'_, PyModule>) -> PyResult<()> {
    let f = wrap_pyfunction!(bootstrap_main, hyperactor_mod)?;
    f.setattr(
        "__module__",
        "monarch._rust_bindings.monarch_hyperactor.bootstrap",
    )?;
    hyperactor_mod.add_function(f)?;

    Ok(())
}
