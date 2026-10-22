/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Optional creation tracebacks for unobserved errors from Monarch operations.

use hyperactor_config::CONFIG;
use hyperactor_config::ConfigAttr;
use hyperactor_config::attrs::declare_attrs;
use pyo3::prelude::*;

use crate::runtime::GilSite;
use crate::runtime::monarch_with_gil_blocking;

declare_attrs! {
    /// Capture a Python stack trace when an endpoint call, stream item or port
    /// receive is created, and log it if the operation fails unobserved.
    @meta(CONFIG = ConfigAttr::new(
        Some("MONARCH_HYPERACTOR_ENABLE_UNAWAITED_PYTHON_TASK_TRACEBACK".to_string()),
        Some("enable_unawaited_python_task_traceback".to_string()),
    ))
    pub attr ENABLE_UNAWAITED_PYTHON_TASK_TRACEBACK: bool = false;
}

/// Capture the creation call site when the traceback setting is enabled.
/// Return `None` when disabled, without calling `traceback.extract_stack()`.
pub(crate) fn current_traceback() -> PyResult<Option<Py<PyAny>>> {
    if hyperactor_config::global::get(ENABLE_UNAWAITED_PYTHON_TASK_TRACEBACK) {
        monarch_with_gil_blocking(GilSite::Traceback, |py| {
            Ok(Some(
                py.import("traceback")?
                    .call_method0("extract_stack")?
                    .unbind(),
            ))
        })
    } else {
        Ok(None)
    }
}
