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

//! A broken engine must fail conformance: the checks have to tell a wrong
//! engine from a right one. One registration serves every fault, since the
//! fault is switched at run time.

mod common;

use common::Fault;

/// The checks that failed with `fault` injected.
fn failures(fault: Fault) -> Vec<&'static str> {
    common::inject(fault);
    let results = common::run();
    assert!(
        results.iter().all(|(_, outcome, _)| *outcome != "skipped"),
        "{results:#?}"
    );
    results
        .into_iter()
        .filter(|(_, outcome, _)| *outcome == "failed")
        .map(|(name, _, _)| name)
        .collect()
}

#[test]
fn every_injected_fault_fails_conformance() {
    common::register();
    // The failing checks panic by design; keep their messages out of the log.
    std::panic::set_hook(Box::new(|_| {}));

    let gamma = failures(Fault::GammaCount);
    let logical = failures(Fault::LogicalCount);
    let draw = failures(Fault::DrawDuplicates);
    let clean = failures(Fault::None);
    let _ = std::panic::take_hook();

    assert!(
        gamma.contains(&"device_matches_cpu_bit_for_bit"),
        "{gamma:?}"
    );
    assert!(
        gamma.contains(&"device_hook_matches_evaluate_with_a_committed_check"),
        "{gamma:?}"
    );
    assert!(
        logical.contains(&"device_scores_a_host_sample_like_the_host"),
        "{logical:?}"
    );
    assert!(
        logical.contains(&"chunked_ler_matches_evaluate_with_a_committed_check"),
        "{logical:?}"
    );
    assert!(
        draw.contains(&"device_draw_matches_the_host_draw"),
        "{draw:?}"
    );
    assert!(
        draw.contains(&"device_leaves_the_stream_where_the_host_does"),
        "{draw:?}"
    );
    assert!(clean.is_empty(), "{clean:?}");
}
