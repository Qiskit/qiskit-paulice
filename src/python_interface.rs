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
use pyo3::prelude::*;

#[pymodule]
fn _internal_r(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheckPicker>()?;
    m.add_class::<Metric>()?;
    m.add_class::<NoiseModel>()?;
    Ok(())
}
