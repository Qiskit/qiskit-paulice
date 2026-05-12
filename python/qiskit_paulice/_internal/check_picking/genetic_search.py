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

"""Docstring for qiskit_paulice_r.check_picking.genetic_search
"""

import itertools

import numpy as np

from ..station import CheckPickerStation
from .windowed_search import windowed_iterator


def tournament(population, energies):
    a, b = np.random.randint(0, len(population), 2)
    return population[a if energies[a] < energies[b] else b]


def crossover(parent1, parent2):
    cut = np.random.randint(1, len(parent1))
    return parent1[:cut] + parent2[cut:]


def mutate(individual, mutation_rate):
    return [(not gene) if (np.random.rand() < mutation_rate) else gene for gene in individual]


def genetic_algorithm(
    check_picker: CheckPickerStation,
    targets: list[int],
    verbose: bool = True,
    p_mutation: float = 0.1,
    n_stagnant: int = 20,
    pop_size: int = 100,
    seed=None,
):
    """A genetic algorithm for check picking
    
    Args:
        check_picker: The check picking station
        targets: Target qubits to protect
        verbose: Print progress information
        p_mutation: Mutation probability
        n_stagnant: Number of generations without improvement before stopping
        pop_size: Population size
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    circuits = [check_picker.get_circuit()]
    costs = [check_picker.get_current_energy()]
    for idx, qubit in enumerate(targets):
        if verbose:
            print(
                f"[GA] Optimizing check {idx + 1}/{len(targets)} for target qubit {qubit}",
                flush=True,
            )
        support = check_picker.get_wires(qubit)

        check_picker = _single_round_genetic(
            check_picker, support, verbose, p_mutation, n_stagnant, pop_size
        )

        circuits.append(check_picker.get_circuit())
        costs.append(check_picker.get_current_energy())
    return (circuits, *check_picker.get_check_data(), costs)


def _single_round_genetic(check_picker, support, verbose, p_mutation, n_stagnant, pop_size):
    check_picker.set_support(support, [1, 2, 3])
    dim = check_picker.get_dimension()
    if verbose:
        print("[GA] * Check space dimension:", dim)
    n_children = max(1, pop_size // 2)
    population = [(np.random.random(dim) < 0.5).tolist() for _ in range(pop_size)] + [[False] * dim]
    energies = [check_picker.evaluate(ind) for ind in population]
    if verbose:
        print("[GA] * Starting with an energy of", check_picker.get_current_energy())

    best_idx = int(np.argmin(energies))
    best_ind = population[best_idx].copy()
    best_energy = energies[best_idx]

    generations_without_improvement = 0
    gen = 0

    while generations_without_improvement < n_stagnant:
        gen += 1
        children = []
        for _ in range(n_children):
            p1 = tournament(population, energies)
            p2 = tournament(population, energies)
            child = mutate(crossover(p1, p2), p_mutation)
            children.append(child)
        population.extend(children)
        new_energies = [check_picker.evaluate(ind) for ind in children]
        energies.extend(new_energies)
        idx_sorted = np.argsort(energies)
        population = [population[i] for i in idx_sorted[:pop_size]]
        energies = [energies[i] for i in idx_sorted[:pop_size]]

        if energies[0] < best_energy:
            best_energy = energies[0]
            best_ind = population[0].copy()
            generations_without_improvement = 0
            if verbose:
                print(
                    f"[GA] * Generation {gen}: new best = {best_energy:.6f}",
                    flush=True,
                )
        else:
            generations_without_improvement += 1
    if verbose:
        print(
            f"[GA] * Stopped after {gen} generations (no improvement for {n_stagnant}). Final energy = {best_energy:.6f}",
            flush=True,
        )
    return check_picker.commit_check(best_ind)


def exhaustive_search(check_picker, verbose):
    dim = check_picker.get_dimension()
    print("[EXH] Dimension:", dim)
    candidates = []
    for bv in itertools.product([False, True], repeat=dim):
        if any(bv):
            bv = list(bv)
            candidates.append((bv, check_picker.evaluate(bv)))
    best = min(candidates, key=lambda x: x[1])
    if verbose:
        print("[EXH] Best check has score:", best[1])
    return check_picker.commit_check(best[0])


def windowed_genetic_algorithm(
    check_picker: CheckPickerStation,
    targets: list[int],
    verbose: bool = True,
    max_width: float = 0.6,
    nwindows=5,
    p_mutation: float = 0.1,
    n_stagnant: int = 20,
    pop_size: int = 20,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    circuits = [check_picker.get_circuit()]
    costs = [check_picker.get_current_energy()]
    for idx, qubit in enumerate(targets):
        if verbose:
            print(
                f"[GA] Optimizing check {idx + 1}/{len(targets)} for target qubit {qubit}",
                flush=True,
            )
        support = check_picker.get_wires(qubit)
        picked = []

        for window in windowed_iterator(support, 3, int(max_width * len(support)), 2, nwindows):
            # np.random.shuffle(support)
            # loc_support = support[: int(max_width * len(support))]
            check_picker.set_support(window, [1, 2, 3])
            dim = check_picker.get_dimension()
            if dim <= 10:
                new_check_picker = exhaustive_search(check_picker, verbose)
            else:
                new_check_picker = _single_round_genetic(
                    check_picker, window, verbose, p_mutation, n_stagnant, pop_size
                )
            picked.append(new_check_picker)

        best_check = min(picked, key=lambda x: x.get_current_energy())
        if verbose:
            print(f"[WGA] Best window gave score: {best_check.get_current_energy()}.")
        check_picker = best_check

        circuits.append(check_picker.get_circuit())
        costs.append(check_picker.get_current_energy())
    return (circuits, *check_picker.get_check_data(), costs)
