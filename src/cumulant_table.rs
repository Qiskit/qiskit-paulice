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

//! Bit-packed cumulant storage for `Coverage`.
//!
//! A cumulant records, for every wire of interest, the value of each
//! propagated Pauli row at that wire. The legacy representation is one
//! `SparsePauli` (a `HashMap<Wire, u8>`) per row; this table stores the same
//! values column-wise: for each recorded wire, the X and Z bits of every row
//! packed into `u64` words. Coverage queries (`is_covered` / `get_syndrome`)
//! become a couple of word-level symplectic products per error wire instead of
//! per-row hash lookups, and construction becomes bit writes instead of
//! per-row hash inserts.
//!
//! Semantics are identical to the `SparsePauli` version: a missing wire means
//! "identity on every row" (commutes), and a row anticommutes with an error
//! Pauli `p` at a wire exactly when its recorded value is non-identity and
//! different from `p` — which is the single-qubit symplectic product
//! `x_row * z_err XOR z_row * x_err`.

use super::sparse_pauli::SparsePauli;
use super::wire::Wire;
use rustiq_core::structures::PauliSet;
use std::collections::HashMap;

const WIDTH: usize = 64;

fn pauli_xz_bits(p: u8) -> (bool, bool) {
    match p {
        1 => (true, false),  // X
        2 => (true, true),   // Y
        3 => (false, true),  // Z
        _ => (false, false), // I
    }
}

#[derive(Clone, Debug, Default)]
pub struct CumulantTable {
    nrows: usize,
    nwords: usize,
    wire_index: HashMap<Wire, usize>,
    x_cols: Vec<Vec<u64>>,
    z_cols: Vec<Vec<u64>>,
}

impl CumulantTable {
    pub fn new(nrows: usize) -> Self {
        Self {
            nrows,
            nwords: nrows.div_ceil(WIDTH),
            wire_index: HashMap::new(),
            x_cols: Vec::new(),
            z_cols: Vec::new(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Record the current value of every row at `wire`, reading qubit `qbit`
    /// of `pset` (X part at row `qbit`, Z part at row `qbit + nqbits` —
    /// the same accesses the legacy per-row loop performs).
    pub fn record(&mut self, wire: Wire, pset: &PauliSet, qbit: usize, nqbits: usize) {
        let mut xw = vec![0u64; self.nwords];
        let mut zw = vec![0u64; self.nwords];
        for r in 0..self.nrows {
            if pset.get_entry(qbit, r) {
                xw[r / WIDTH] |= 1 << (r % WIDTH);
            }
            if pset.get_entry(qbit + nqbits, r) {
                zw[r / WIDTH] |= 1 << (r % WIDTH);
            }
        }
        let idx = self.x_cols.len();
        self.wire_index.insert(wire, idx);
        self.x_cols.push(xw);
        self.z_cols.push(zw);
    }

    /// Append the rows of `other` after the rows of `self` (wire union; a wire
    /// missing from one table contributes identity rows for that table's side).
    /// Mirrors `Vec::extend` on the legacy `Vec<SparsePauli>` representation.
    pub fn extend(&mut self, other: CumulantTable) {
        if other.nrows == 0 {
            return;
        }
        if self.nrows == 0 {
            *self = other;
            return;
        }
        let nrows = self.nrows + other.nrows;
        let nwords = nrows.div_ceil(WIDTH);
        let mut merged = CumulantTable::new(nrows);
        let mut wires: Vec<&Wire> = self
            .wire_index
            .keys()
            .chain(other.wire_index.keys())
            .collect();
        wires.sort();
        wires.dedup();
        for wire in wires {
            let mut xw = vec![0u64; nwords];
            let mut zw = vec![0u64; nwords];
            if let Some(&i) = self.wire_index.get(wire) {
                for r in 0..self.nrows {
                    if (self.x_cols[i][r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        xw[r / WIDTH] |= 1 << (r % WIDTH);
                    }
                    if (self.z_cols[i][r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        zw[r / WIDTH] |= 1 << (r % WIDTH);
                    }
                }
            }
            if let Some(&i) = other.wire_index.get(wire) {
                for r in 0..other.nrows {
                    let rr = self.nrows + r;
                    if (other.x_cols[i][r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        xw[rr / WIDTH] |= 1 << (rr % WIDTH);
                    }
                    if (other.z_cols[i][r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        zw[rr / WIDTH] |= 1 << (rr % WIDTH);
                    }
                }
            }
            let idx = merged.x_cols.len();
            merged.wire_index.insert(wire.clone(), idx);
            merged.x_cols.push(xw);
            merged.z_cols.push(zw);
        }
        *self = merged;
    }

    /// XOR-combine the per-row anticommutation bits of `error` across its
    /// wires and report whether any row ends up odd. Exactly matches the
    /// legacy `is_covered` (parity) semantics.
    pub fn covered_parity_any(&self, error: &SparsePauli) -> bool {
        if self.nrows == 0 {
            return false;
        }
        if self.nwords == 1 {
            let mut acc = 0u64;
            for (w, p) in error.paulis.iter() {
                if let Some(&i) = self.wire_index.get(w) {
                    let (xe, ze) = pauli_xz_bits(*p);
                    let mut a = 0u64;
                    if ze {
                        a ^= self.x_cols[i][0];
                    }
                    if xe {
                        a ^= self.z_cols[i][0];
                    }
                    acc ^= a;
                }
            }
            return acc != 0;
        }
        let mut acc = vec![0u64; self.nwords];
        for (w, p) in error.paulis.iter() {
            if let Some(&i) = self.wire_index.get(w) {
                let (xe, ze) = pauli_xz_bits(*p);
                for k in 0..self.nwords {
                    let mut a = 0u64;
                    if ze {
                        a ^= self.x_cols[i][k];
                    }
                    if xe {
                        a ^= self.z_cols[i][k];
                    }
                    acc[k] ^= a;
                }
            }
        }
        acc.iter().any(|w| *w != 0)
    }

    /// OR-combine the per-row anticommutation bits of `error` across its
    /// wires. Exactly matches the legacy `get_syndrome` semantics (one bool
    /// per row: does any wire of the error anticommute with that row).
    pub fn syndrome(&self, error: &SparsePauli) -> Vec<bool> {
        let mut acc = vec![0u64; self.nwords];
        for (w, p) in error.paulis.iter() {
            if let Some(&i) = self.wire_index.get(w) {
                let (xe, ze) = pauli_xz_bits(*p);
                for k in 0..self.nwords {
                    let mut a = 0u64;
                    if ze {
                        a ^= self.x_cols[i][k];
                    }
                    if xe {
                        a ^= self.z_cols[i][k];
                    }
                    acc[k] |= a;
                }
            }
        }
        (0..self.nrows)
            .map(|r| (acc[r / WIDTH] >> (r % WIDTH)) & 1 != 0)
            .collect()
    }
}

#[cfg(test)]
mod cumulant_table_tests {
    use super::super::coverage::{is_covered_legacy, get_syndrome_legacy};
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Build (legacy rows, packed table) with identical random contents.
    fn random_pair(
        rng: &mut StdRng,
        nrows: usize,
        wires: &[Wire],
    ) -> (Vec<SparsePauli>, CumulantTable) {
        // Values per (wire, row) drawn once, then written to both structures.
        let mut values: HashMap<(usize, usize), u8> = HashMap::new();
        for (wi, _) in wires.iter().enumerate() {
            for r in 0..nrows {
                values.insert((wi, r), rng.random_range(0..4) as u8);
            }
        }
        let mut rows = vec![SparsePauli::new(); nrows];
        for (wi, wire) in wires.iter().enumerate() {
            for (r, row) in rows.iter_mut().enumerate() {
                row.update(wire.clone(), values[&(wi, r)]);
            }
        }
        // Pack the same values by building a PauliSet whose operator r holds,
        // at qubit wi, the value for (wi, r); record each wire off qubit wi.
        let nqbits = wires.len();
        let mut pset = PauliSet::new_empty(nqbits, nrows);
        for (wi, _) in wires.iter().enumerate() {
            for r in 0..nrows {
                let (x, z) = pauli_xz_bits(values[&(wi, r)]);
                pset.set_entry(r, wi, x, z);
            }
        }
        let mut table = CumulantTable::new(nrows);
        for (wi, wire) in wires.iter().enumerate() {
            table.record(wire.clone(), &pset, wi, nqbits);
        }
        (rows, table)
    }

    #[test]
    fn matches_legacy_on_random_inputs() {
        let mut rng = StdRng::seed_from_u64(12345);
        for trial in 0..200 {
            let nrows = 1 + (trial % 70);
            let wires: Vec<Wire> = (0..6)
                .map(|g| Wire::GateWire(g, g % 2))
                .chain([Wire::Input(0), Wire::Input(3)])
                .collect();
            let (rows, table) = random_pair(&mut rng, nrows, &wires);
            for _ in 0..40 {
                // Random error touching a random subset of wires (some of
                // which may be absent from the table in other tests).
                let mut error = SparsePauli::new();
                for wire in wires.iter() {
                    if rng.random_bool(0.4) {
                        error.update(wire.clone(), rng.random_range(1..4) as u8);
                    }
                }
                // Also probe a wire the table never recorded.
                if rng.random_bool(0.3) {
                    error.update(Wire::GateWire(999, 0), rng.random_range(1..4) as u8);
                }
                assert_eq!(
                    is_covered_legacy(&rows, &error),
                    table.covered_parity_any(&error),
                    "is_covered mismatch (trial {trial})"
                );
                assert_eq!(
                    get_syndrome_legacy(&rows, &error),
                    table.syndrome(&error),
                    "syndrome mismatch (trial {trial})"
                );
            }
        }
    }

    #[test]
    fn extend_matches_legacy_extend() {
        let mut rng = StdRng::seed_from_u64(999);
        let wires_a: Vec<Wire> = (0..4).map(|g| Wire::GateWire(g, 0)).collect();
        let wires_b: Vec<Wire> = (2..7).map(|g| Wire::GateWire(g, 0)).collect();
        let (rows_a, mut table) = random_pair(&mut rng, 70, &wires_a);
        let (rows_b, table_b) = random_pair(&mut rng, 3, &wires_b);
        let mut rows = rows_a;
        rows.extend(rows_b);
        table.extend(table_b);
        for _ in 0..100 {
            let mut error = SparsePauli::new();
            for wire in wires_a.iter().chain(wires_b.iter()) {
                if rng.random_bool(0.4) {
                    error.update(wire.clone(), rng.random_range(1..4) as u8);
                }
            }
            assert_eq!(is_covered_legacy(&rows, &error), table.covered_parity_any(&error));
            assert_eq!(get_syndrome_legacy(&rows, &error), table.syndrome(&error));
        }
    }
}
