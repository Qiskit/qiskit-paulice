// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::check_picker::CheckPicker;
use super::metric::PyMetric as Metric;
use super::noise_model::NoiseModel;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::sync::OnceLock;

/// Matches `PAULICE_ENGINE_CAPSULE` in `include/paulice_engine.h`.
const ENGINE_CAPSULE: &std::ffi::CStr = c"qiskit_paulice.engine";

/// Holds the registered engine's capsule, and so its module, for the rest of
/// the process: the engine table lives in that module's library.
static ENGINE_OWNER: OnceLock<Py<PyCapsule>> = OnceLock::new();

/// Installs the scoring engine a `qiskit_paulice.engine` capsule carries, for
/// the rest of the process. Raises `ValueError` if the capsule is not one, the
/// engine speaks another ABI version, or an engine is already registered.
#[pyfunction]
fn register_scoring_engine(capsule: &Bound<'_, PyCapsule>) -> PyResult<()> {
    let table = capsule.pointer_checked(Some(ENGINE_CAPSULE))?;
    // SAFETY: a `qiskit_paulice.engine` capsule carries a `PauliceEngine`
    // table in its engine's library, which `ENGINE_OWNER` keeps loaded.
    unsafe { crate::engine::register_engine(table.as_ptr()) }.map_err(PyValueError::new_err)?;
    let _ = ENGINE_OWNER.set(capsule.clone().unbind());
    Ok(())
}

/// The registered scoring engine's name, or `None`.
#[pyfunction]
fn scoring_engine_name() -> Option<String> {
    crate::engine::engine_name()
}

/// Runs the engine conformance suite against the registered engine and
/// returns `(check, outcome, detail)` for each check, where `outcome` is
/// `"passed"`, `"failed"` or `"skipped"`.
#[pyfunction]
fn run_engine_conformance(py: Python<'_>) -> Vec<(String, String, String)> {
    py.detach(crate::engine::run_conformance)
        .into_iter()
        .map(|(name, outcome)| {
            let (status, detail) = match outcome {
                crate::engine::CheckOutcome::Passed => ("passed", String::new()),
                crate::engine::CheckOutcome::Failed(why) => ("failed", why),
                crate::engine::CheckOutcome::Skipped(why) => ("skipped", why),
            };
            (name.to_string(), status.to_string(), detail)
        })
        .collect()
}

#[pymodule]
fn _internal_r(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheckPicker>()?;
    m.add_class::<Metric>()?;
    m.add_class::<NoiseModel>()?;
    m.add_function(wrap_pyfunction!(register_scoring_engine, m)?)?;
    m.add_function(wrap_pyfunction!(scoring_engine_name, m)?)?;
    m.add_function(wrap_pyfunction!(run_engine_conformance, m)?)?;
    Ok(())
}
