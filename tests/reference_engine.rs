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

//! Runs every conformance check through the dispatch code against the CPU
//! reference engine, so the code that talks to an engine is exercised
//! without a GPU.

mod common;

#[test]
fn the_reference_engine_passes_every_conformance_check() {
    common::register();
    let results = common::run();
    assert_eq!(
        _internal_r::engine::engine_name().as_deref(),
        Some("reference")
    );
    assert_eq!(results.len(), 14);
    let not_passed: Vec<_> = results
        .iter()
        .filter(|(_, outcome, _)| *outcome != "passed")
        .collect();
    assert!(not_passed.is_empty(), "{not_passed:#?}");
}
