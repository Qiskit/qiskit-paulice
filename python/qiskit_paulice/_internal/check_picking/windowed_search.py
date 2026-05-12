# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A windowed search for check picking.
"""

import itertools

import numpy as np

from ..station import CheckPickerStation


def windowed_iterator(support, min_size, max_size, number_of_windows, ntries=30):
    """A randomized iterator over segments of a some list
    """
    min_size = max(min_size, 3)
    # Fitting `number_of_windows` non-overlapping windows of size `window_size` requires
    # window_size * number_of_windows <= len(support); equivalently the start_index range
    # used below, len(support) - (number_of_windows - i - 1) * window_size, has to be > 0
    # for every i. Cap max_size accordingly. If even one min-sized window per slot won't
    # fit, fall back to yielding the whole support once so the caller still has a candidate.
    n = max(number_of_windows, 1)
    max_size = min(max_size, len(support) // n)
    if max_size < min_size:
        if len(support) >= 3:
            yield sorted(support)
        return
    trials = 0
    while trials < ntries:
        np.random.shuffle(support)
        actual_support = []
        window_size = np.random.randint(min_size, max_size + 1)
        previous_end = 0
        for i in range(number_of_windows):
            start_index = np.random.randint(
                previous_end, len(support) - (number_of_windows - i - 1) * window_size
            )
            end_index = start_index + window_size
            previous_end = end_index
            actual_support.extend(support[start_index:end_index])
        if len(actual_support) >= 3:
            yield sorted(actual_support)
            trials += 1


def _get_good_checks_randomized(support, max_width, check_picker, ntries, paulis=None):

    it1 = windowed_iterator(
        support, int(0.15 * len(support)), int(max_width * len(support)), 2, ntries // 2
    )
    it2 = windowed_iterator(
        support, int(0.15 * len(support)), int(max_width * len(support)), 1, ntries // 2
    )
    all_candidates = []
    it = itertools.chain(it1, it2)
    for actual_support in it:
        picker_copy = check_picker.copy()
        # Derive a Rust-side seed from `np.random` so the decoder's middle-wire
        # choice is reproducible whenever the caller has seeded `np.random`. The
        # int(...) cast avoids passing a numpy scalar through PyO3.
        rust_seed = int(np.random.randint(0, 2**32 - 1))
        picker_copy.set_support(actual_support, paulis or [1, 2, 3], seed=rust_seed)
        score = picker_copy.find_good_check()
        if score is not None:
            candidate = (picker_copy, score)
            all_candidates.append(candidate)
    return all_candidates


def windowed_check_picker(
    check_picker: CheckPickerStation,
    targets: list[int],
    ntries: int = 30,
    max_width: float = 0.3,
    verbose: bool = False,
    paulis=None,
    seed=None,
):
    """A check picking strategy that explores windows of the support
    
    Args:
        check_picker: The check picking station
        targets: Target qubits to protect
        ntries: Number of random trials per target
        max_width: Maximum width of support window
        verbose: Print progress information
        paulis: Pauli types to consider
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    circuits = [check_picker.get_circuit()]
    costs = [check_picker.get_current_energy()]
    for idx, target in enumerate(targets):

        if verbose:
            print(
                f"[WIN] Optimizing check {idx + 1}/{len(targets)} for target qubit {target}",
                flush=True,
            )
        support = check_picker.get_wires(target)

        candidatechecks = _get_good_checks_randomized(
            support, max_width, check_picker, ntries, paulis
        )
        best_check = min(candidatechecks, key=lambda x: x[-1])
        if verbose:
            print(f"[WIN] Best check has score: {best_check[-1]}.")
        check_picker = best_check[0]
        circuits.append(check_picker.get_circuit())
        costs.append(check_picker.get_current_energy())
    return (circuits, *check_picker.get_check_data(), costs)
