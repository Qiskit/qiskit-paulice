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

/// Parses an internal Pauli label, in which character `i` acts on qubit `i` (the reverse of a
/// Qiskit label, whose rightmost character is qubit 0), into the symplectic vector
/// `[x_0..x_{n-1}, z_0..z_{n-1}]` on `n` qubits.
///
/// A label shorter than `n` is padded with identities; characters beyond `n` are ignored.
pub fn string_to_pauli(s: &str, n: usize) -> Pauli {
    let mut pauli = vec![false; n * 2];
    for (i, c) in s.chars().take(n).enumerate() {
        match c {
            'X' => pauli[i] = true,
            'Y' => {
                pauli[i] = true;
                pauli[i + n] = true;
            }
            'Z' => pauli[i + n] = true,
            _ => (),
        }
    }
    pauli
}

#[cfg(test)]
mod pauli_tests {
    use super::*;

    #[test]
    fn full_width_label() {
        assert_eq!(
            string_to_pauli("XYZI", 4),
            vec![true, true, false, false, false, true, true, false]
        );
    }

    #[test]
    fn short_label_is_padded_with_identities() {
        // Z on qubit 1 of a 3-qubit register, given as a 2-character label.
        assert_eq!(
            string_to_pauli("IZ", 3),
            vec![false, false, false, false, true, false]
        );
    }
}
