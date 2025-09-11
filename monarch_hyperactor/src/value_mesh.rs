/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_mesh::v1::ValueMesh;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass(
    name = "ValueMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.value_mesh"
)]
pub struct PyValueMesh {
    inner: ValueMesh<Py<PyAny>>,
}

#[pymethods]
impl PyValueMesh {
    // TODO(SF, 2025-09-10): Implement bindings.
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyValueMesh>()?;
    Ok(())
}
