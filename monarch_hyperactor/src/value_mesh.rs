/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use hyperactor_mesh::v1::ValueMesh;
use ndslice::Extent;
use ndslice::Region;
use ndslice::view::BuildFromRegion;
use ndslice::view::Ranked;
use ndslice::view::ViewExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyList;

use crate::shape::PyShape;

#[pyclass(
    name = "ValueMesh",
    module = "monarch._rust_bindings.monarch_hyperactor.value_mesh"
)]
pub struct PyValueMesh {
    inner: ValueMesh<Py<PyAny>>,
}

#[pymethods]
impl PyValueMesh {
    /// __init__(self, shape: Shape, values: list)
    #[new]
    fn new(_py: Python<'_>, shape: &PyShape, values: Bound<'_, PyList>) -> PyResult<Self> {
        // Convert shape to region.
        let extent: Extent = shape.get_inner().clone().into();
        let region: Region = extent.into();
        let vals: Vec<Py<PyAny>> = values.extract()?;

        // Build & validate cardinality against region.
        let inner = <ValueMesh<Py<PyAny>> as BuildFromRegion<Py<PyAny>>>::build_dense(region, vals)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Return number of ranks (Python: len(vm))
    fn __len__(&self) -> usize {
        self.inner.region().num_ranks()
    }

    /// Return the values in region/iteration order as a Python list.
    fn values(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Clone the inner Py objects into a Python list (just bumps
        // refcounts).
        let vec: Vec<Py<PyAny>> = self.inner.values().collect();
        Ok(PyList::new(py, vec)?.into())
    }

    // TODO(SF, 2025-09-10): Implement more bindings.
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyValueMesh>()?;
    Ok(())
}
