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
//! Wires are addressed by their dense id (`Wire::dense_id`), so finding a
//! wire's column is an array index rather than a hash lookup, and the whole
//! table is three flat allocations.
//!
//! Semantics are identical to the `SparsePauli` version: a missing wire means
//! "identity on every row" (commutes), and a row anticommutes with an error
//! Pauli `p` at a wire exactly when its recorded value is non-identity and
//! different from `p` — which is the single-qubit symplectic product
//! `x_row * z_err XOR z_row * x_err`.

use super::sparse_pauli::SparsePauli;
use super::wire::Wire;
use rustiq_core::structures::PauliSet;

const WIDTH: usize = 64;

/// Marks a dense wire id that this table never recorded.
const NO_COLUMN: i32 = -1;

pub fn pauli_xz_bits(p: u8) -> (bool, bool) {
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
    nqbits: usize,
    /// Column holding each dense wire id, or `NO_COLUMN`.
    wire_columns: Vec<i32>,
    /// Column-major words: column `c`, word `k` lives at `c * nwords + k`.
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    ncolumns: usize,
}

impl CumulantTable {
    /// `nwire_slots` is the dense wire-id range of the circuit being recorded
    /// (see `wire::wire_id_space`).
    pub fn new(nrows: usize, nqbits: usize, nwire_slots: usize) -> Self {
        Self {
            nrows,
            nwords: nrows.div_ceil(WIDTH),
            nqbits,
            wire_columns: vec![NO_COLUMN; nwire_slots],
            x_words: Vec::new(),
            z_words: Vec::new(),
            ncolumns: 0,
        }
    }

    /// Pre-size the column buffers for `columns` recorded wires, so a
    /// propagation walk does not reallocate them as it records.
    pub fn reserve_columns(&mut self, columns: usize) {
        self.x_words.reserve(columns * self.nwords);
        self.z_words.reserve(columns * self.nwords);
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn nwords(&self) -> usize {
        self.nwords
    }

    pub fn nqbits(&self) -> usize {
        self.nqbits
    }

    pub fn ncolumns(&self) -> usize {
        self.ncolumns
    }

    /// Column of a dense wire id, or `NO_COLUMN` if it was never recorded.
    /// Ids past the end of the recorded range are absent by definition.
    #[inline]
    pub fn column_of_id(&self, id: usize) -> i32 {
        match self.wire_columns.get(id) {
            Some(c) => *c,
            None => NO_COLUMN,
        }
    }

    #[inline]
    pub fn column_of(&self, wire: &Wire) -> i32 {
        self.column_of_id(wire.dense_id(self.nqbits))
    }

    /// Record the current value of every row at `wire`, reading qubit `qbit`
    /// of `pset` (X part at row `qbit`, Z part at row `qbit + nqbits` —
    /// the same accesses the legacy per-row loop performs).
    pub fn record(&mut self, wire: Wire, pset: &PauliSet, qbit: usize, nqbits: usize) {
        let base = self.x_words.len();
        self.x_words.resize(base + self.nwords, 0);
        self.z_words.resize(base + self.nwords, 0);
        for r in 0..self.nrows {
            if pset.get_entry(qbit, r) {
                self.x_words[base + r / WIDTH] |= 1 << (r % WIDTH);
            }
            if pset.get_entry(qbit + nqbits, r) {
                self.z_words[base + r / WIDTH] |= 1 << (r % WIDTH);
            }
        }
        let id = wire.dense_id(self.nqbits);
        if id >= self.wire_columns.len() {
            self.wire_columns.resize(id + 1, NO_COLUMN);
        }
        self.wire_columns[id] = self.ncolumns as i32;
        self.ncolumns += 1;
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
        let slots = self.wire_columns.len().max(other.wire_columns.len());
        let mut merged = CumulantTable::new(nrows, self.nqbits, slots);
        for id in 0..slots {
            let mine = self.column_of_id(id);
            let theirs = other.column_of_id(id);
            if mine == NO_COLUMN && theirs == NO_COLUMN {
                continue;
            }
            let base = merged.x_words.len();
            merged.x_words.resize(base + merged.nwords, 0);
            merged.z_words.resize(base + merged.nwords, 0);
            if mine != NO_COLUMN {
                let src = mine as usize * self.nwords;
                for r in 0..self.nrows {
                    if (self.x_words[src + r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        merged.x_words[base + r / WIDTH] |= 1 << (r % WIDTH);
                    }
                    if (self.z_words[src + r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        merged.z_words[base + r / WIDTH] |= 1 << (r % WIDTH);
                    }
                }
            }
            if theirs != NO_COLUMN {
                let src = theirs as usize * other.nwords;
                for r in 0..other.nrows {
                    let rr = self.nrows + r;
                    if (other.x_words[src + r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        merged.x_words[base + rr / WIDTH] |= 1 << (rr % WIDTH);
                    }
                    if (other.z_words[src + r / WIDTH] >> (r % WIDTH)) & 1 != 0 {
                        merged.z_words[base + rr / WIDTH] |= 1 << (rr % WIDTH);
                    }
                }
            }
            merged.wire_columns[id] = merged.ncolumns as i32;
            merged.ncolumns += 1;
        }
        *self = merged;
    }

    /// XOR contribution of one column to word `word` of a parity accumulator.
    #[inline]
    pub fn parity_word_at(&self, column: i32, xe: bool, ze: bool, word: usize) -> u64 {
        if column == NO_COLUMN {
            return 0;
        }
        let base = column as usize * self.nwords + word;
        let mut a = 0u64;
        if ze {
            a ^= self.x_words[base];
        }
        if xe {
            a ^= self.z_words[base];
        }
        a
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
                let (xe, ze) = pauli_xz_bits(*p);
                acc ^= self.parity_word_at(self.column_of(w), xe, ze, 0);
            }
            return acc != 0;
        }
        let mut acc = vec![0u64; self.nwords];
        for (w, p) in error.paulis.iter() {
            let (xe, ze) = pauli_xz_bits(*p);
            let column = self.column_of(w);
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot ^= self.parity_word_at(column, xe, ze, k);
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
            let (xe, ze) = pauli_xz_bits(*p);
            let column = self.column_of(w);
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot |= self.parity_word_at(column, xe, ze, k);
            }
        }
        (0..self.nrows)
            .map(|r| (acc[r / WIDTH] >> (r % WIDTH)) & 1 != 0)
            .collect()
    }
}

/// The `Vec<SparsePauli>` coverage queries the table replaces, kept as the
/// reference the equivalence tests compare against.
#[cfg(test)]
pub(crate) mod legacy {
    use super::super::sparse_pauli::SparsePauli;

    fn is_covered_single_cumulant(cumulant: &SparsePauli, error: &SparsePauli) -> bool {
        for (w, p) in error.paulis.iter() {
            if *cumulant.paulis.get(w).unwrap_or(p) != *p {
                return true;
            }
        }
        false
    }

    pub(crate) fn get_syndrome_legacy(cumulants: &[SparsePauli], error: &SparsePauli) -> Vec<bool> {
        cumulants
            .iter()
            .map(|c| is_covered_single_cumulant(c, error))
            .collect()
    }

    pub(crate) fn is_covered_legacy(cumulants: &[SparsePauli], error: &SparsePauli) -> bool {
        for cumulant in cumulants.iter() {
            if error
                .paulis
                .iter()
                .filter(|(w, p)| *cumulant.paulis.get(w).unwrap_or(*p) != **p)
                .count()
                % 2
                == 1
            {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod cumulant_table_tests {
    use super::legacy::{get_syndrome_legacy, is_covered_legacy};
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    /// Wire space wide enough for every wire the tests build by hand.
    const SLOTS: usize = 4096;
    const NQBITS: usize = 8;

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
        let mut table = CumulantTable::new(nrows, NQBITS, SLOTS);
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

    /// A wire id past the recorded range must read as absent, not panic --
    /// generators can name wires the cumulants never snapshot.
    #[test]
    fn wire_beyond_recorded_range_is_absent() {
        let mut rng = StdRng::seed_from_u64(4);
        let wires: Vec<Wire> = (0..3).map(|g| Wire::GateWire(g, 0)).collect();
        let (_, table) = random_pair(&mut rng, 5, &wires);
        assert_eq!(table.column_of_id(SLOTS + 10), NO_COLUMN);
        assert_eq!(table.column_of(&Wire::GateWire(100_000, 1)), NO_COLUMN);
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
            assert_eq!(
                is_covered_legacy(&rows, &error),
                table.covered_parity_any(&error)
            );
            assert_eq!(get_syndrome_legacy(&rows, &error), table.syndrome(&error));
        }
    }
}
