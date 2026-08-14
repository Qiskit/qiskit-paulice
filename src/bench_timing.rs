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

//! Lightweight wall-clock stage timers used by the benchmarking scripts.
//!
//! Timers accumulate into a process-global table keyed by stage name; the
//! Python side drains the table via `bench_timings_take`. Records only happen
//! at coarse stage boundaries on the calling thread, so lock contention and
//! overhead are negligible relative to the stages being timed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

static TIMINGS: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

/// Accumulate the elapsed time since `start` under `name`.
pub fn record(name: &str, start: Instant) {
    let nanos = start.elapsed().as_nanos() as u64;
    let mut guard = TIMINGS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = map.entry(name.to_string()).or_insert((0, 0));
    entry.0 += nanos;
    entry.1 += 1;
}

/// Drain and return all accumulated timings as (name, seconds, call_count).
pub fn take() -> Vec<(String, f64, u64)> {
    let mut guard = TIMINGS.lock().unwrap();
    match guard.take() {
        Some(map) => {
            let mut rows: Vec<_> = map
                .into_iter()
                .map(|(name, (nanos, count))| (name, nanos as f64 * 1e-9, count))
                .collect();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            rows
        }
        None => Vec::new(),
    }
}
