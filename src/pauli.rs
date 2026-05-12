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

pub type Pauli = Vec<bool>;

pub fn string_to_pauli(s: &str, n: usize) -> Pauli {
    let mut pauli = vec![false; n * 2];
    for i in 0..s.len() {
        match s.chars().nth(i).unwrap() {
            'X' => pauli[i] = true,
            'Y' => {
                pauli[i] = true;
                pauli[i + s.len()] = true;
            }
            'Z' => pauli[i + s.len()] = true,
            _ => (),
        }
    }
    pauli
}
