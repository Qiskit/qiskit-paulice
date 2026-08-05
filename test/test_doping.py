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

"""Tests for the ``doping`` module."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import (
    Clifford,
    Pauli,
    SparsePauliOp,
    Statevector,
    random_clifford,
)
from qiskit_paulice import CheckedCircuit, DopingSite, dope_clifford_circuit
from qiskit_paulice.checks import add_pauli_checks
from qiskit_paulice.noise_models import NoiseModel


def _stabilizer_renyi_2(circuit: QuantumCircuit) -> float:
    """The 2-stabilizer Renyi entropy M2 = -log2( sum_P <P>^4 / 2^n ) of the output state.

    Zero exactly on stabilizer states, positive on magic states.
    """
    state = Statevector(circuit)
    rho = np.outer(state.data, state.data.conj())
    expvals = SparsePauliOp.from_operator(rho).coeffs.real * 2**state.num_qubits
    return float(-np.log2(np.sum(expvals**4) / 2**state.num_qubits))


def _paper_ansatz(num_qubits: int, seed: int) -> QuantumCircuit:
    """The paper's graph-state ansatz: H layer, then brickwork CZ + random S/sqrt(X) layers."""
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    for layer in range(num_qubits):
        for a in range(layer % 2, num_qubits - 1, 2):
            circuit.cz(a, a + 1)
        for q in range(num_qubits):
            if rng.random() < 0.5:
                circuit.s(q)
            circuit.sx(q)
    return circuit


def _position(site: DopingSite) -> int:
    """The number of instructions preceding a site's wire boundary."""
    return 0 if site.after_instruction is None else site.after_instruction + 1


def _split(circuit: QuantumCircuit, position: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Split a circuit into prefix/suffix around a wire boundary, dropping non-unitaries."""
    prefix = QuantumCircuit(circuit.num_qubits)
    suffix = QuantumCircuit(circuit.num_qubits)
    for index, inst in enumerate(circuit.data):
        if inst.operation.name not in ("measure", "barrier"):
            target = prefix if index < position else suffix
            target.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits])
    return prefix, suffix


def _propagated(circuit: QuantumCircuit, site: DopingSite) -> tuple[Pauli, Pauli]:
    """A site generator's forward (output) and backward (input) propagation via ``Pauli.evolve``.

    Deliberately a different implementation from the module's single tableau sweep.
    """
    z = Pauli("I" * circuit.num_qubits)
    z.z[site.qubit] = True
    prefix, suffix = _split(circuit, _position(site))
    return z.evolve(Clifford(suffix), frame="s"), z.evolve(Clifford(prefix), frame="h")


def _syndrome_values(checked: CheckedCircuit, circuit: QuantumCircuit) -> list[float]:
    """The expectation value of each check's syndrome operator on the pre-measurement state."""
    state = Statevector(circuit.remove_final_measurements(inplace=False))
    values = []
    for support in checked.check_support:
        label = "".join("Z" if q in support else "I" for q in reversed(range(circuit.num_qubits)))
        values.append(complex(state.expectation_value(Pauli(label))).real)
    return values


def _checked_circuit() -> CheckedCircuit:
    """A small checked Clifford circuit with one spacetime Pauli check."""
    circuit = QuantumCircuit(3)
    for _ in range(2):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.s(0)
        circuit.s(2)
    circuit.measure_all()
    noise = NoiseModel(gate_noise=1e-3, readout_noise=1e-2)
    return add_pauli_checks(circuit, [1], noise, seed=0)[-1]


class TestDopeCliffordCircuit(unittest.TestCase):
    """Tests for :func:`dope_clifford_circuit` in distribution-preserving mode."""

    def test_hadamard_yields_t_magic_state(self):
        """Doping a bare H produces H;T -- the canonical |T> magic state -- at the only site."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertEqual(sites, [DopingSite(qubit=0, after_instruction=0)])
        self.assertEqual([inst.name for inst in doped], ["h", "t"])
        np.testing.assert_allclose(
            Statevector(doped).probabilities(), Statevector(circuit).probabilities(), atol=1e-12
        )
        self.assertAlmostEqual(_stabilizer_renyi_2(circuit), 0.0, places=10)
        # M2 of |T> = -log2((1 + 2*(1/sqrt2)^4)/2) = log2(4/3)
        self.assertAlmostEqual(_stabilizer_renyi_2(doped), np.log2(4 / 3), places=10)

    def test_distribution_preserved_and_magic_injected(self):
        """Full doping never disturbs the sampled distribution but always injects magic."""
        for seed in range(5):
            circuit = random_clifford(4, seed=seed).to_circuit()
            doped, sites = dope_clifford_circuit(circuit)
            self.assertGreater(len(sites), 0, msg=f"seed {seed}")
            np.testing.assert_allclose(
                Statevector(doped).probabilities(),
                Statevector(circuit).probabilities(),
                atol=1e-12,
                err_msg=f"seed {seed}",
            )
            self.assertAlmostEqual(_stabilizer_renyi_2(circuit), 0.0, places=10, msg=f"seed {seed}")
            self.assertGreater(_stabilizer_renyi_2(doped), 0.2, msg=f"seed {seed}")

    def test_paper_ansatz(self):
        """Distribution preservation and magic injection on the paper's brickwork ansatz."""
        circuit = _paper_ansatz(5, seed=1)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreaterEqual(len(sites), 5)
        np.testing.assert_allclose(
            Statevector(doped).probabilities(), Statevector(circuit).probabilities(), atol=1e-12
        )
        self.assertGreater(_stabilizer_renyi_2(doped), 1.0)

    def test_sites_satisfy_paper_criteria(self):
        """Independent check of every pruning criterion at every returned site."""
        for seed in range(5):
            circuit = random_clifford(4, seed=seed).to_circuit()
            _, sites = dope_clifford_circuit(circuit)
            propagated = []
            for site in sites:
                forward, backward = _propagated(circuit, site)
                self.assertFalse(forward.x.any(), msg="forward propagation must be diagonal")
                self.assertTrue(backward.x.any(), msg="site must not be an input stabilizer")
                propagated.append(forward.z.tobytes())
            self.assertEqual(
                len(propagated), len(set(propagated)), msg="propagated generators must be distinct"
            )

    def test_diagonal_circuit_has_no_sites(self):
        """A diagonal Clifford circuit admits no site: every Z is a stabilizer of the state."""
        circuit = QuantumCircuit(3)
        circuit.s(0)
        circuit.cz(0, 1)
        circuit.z(2)
        circuit.cz(1, 2)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertEqual(sites, [])
        self.assertEqual(doped.count_ops().get("t", 0), 0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit, num_t_gates=1)

    def test_num_t_gates_and_seed(self):
        """num_t_gates is honored exactly, selection is seed-reproducible, excess raises."""
        circuit = random_clifford(4, seed=0).to_circuit()
        _, all_sites = dope_clifford_circuit(circuit)
        doped, sites = dope_clifford_circuit(circuit, num_t_gates=2, seed=123)
        self.assertEqual(len(sites), 2)
        self.assertEqual(doped.count_ops()["t"], 2)
        self.assertTrue(set(sites) <= set(all_sites))
        _, again = dope_clifford_circuit(circuit, num_t_gates=2, seed=123)
        self.assertEqual(sites, again)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit, num_t_gates=len(all_sites) + 1)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit, num_t_gates=-1)

    def test_only_t_gates_inserted(self):
        """The doped circuit is the original instruction sequence with only T gates added."""
        circuit = _paper_ansatz(4, seed=3)
        doped, sites = dope_clifford_circuit(circuit, num_t_gates=3, seed=0)
        stripped = [
            (inst.name, tuple(doped.find_bit(q).index for q in inst.qubits))
            for inst in doped
            if inst.name != "t"
        ]
        original = [
            (inst.name, tuple(circuit.find_bit(q).index for q in inst.qubits)) for inst in circuit
        ]
        self.assertEqual(stripped, original)
        self.assertEqual(len(doped), len(circuit) + len(sites))

    def test_non_clifford_raises(self):
        """A non-Clifford instruction is rejected."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.3, 0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit)

    def test_barriers_ignored(self):
        """Barriers are transparent to the propagation and preserved in the output."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.barrier()
        circuit.cx(0, 1)
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(sites), 0)
        self.assertIn("barrier", doped.count_ops())
        np.testing.assert_allclose(
            Statevector(doped).probabilities(), Statevector(circuit).probabilities(), atol=1e-12
        )

    def test_measured_circuit(self):
        """Terminal measurements are preserved and no site lies past a qubit's measurement."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure(0, 0)
        circuit.h(1)
        circuit.measure(1, 1)
        measure_pos = {0: 2, 1: 4}
        doped, sites = dope_clifford_circuit(circuit)
        self.assertGreater(len(sites), 0)
        self.assertEqual(doped.count_ops()["measure"], 2)
        self.assertEqual(doped.count_ops()["t"], len(sites))
        for site in sites:
            self.assertLessEqual(_position(site), measure_pos[site.qubit])
        circuit.x(0)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(circuit)


class TestXebMode(unittest.TestCase):
    """Tests for :func:`dope_clifford_circuit` with ``preserve_distribution=False``."""

    def _assert_irreducible(self, circuit: QuantumCircuit, sites: list[DopingSite]):
        """Independently verify that no pruning rewrite applies to the returned rotations."""
        forward = []
        backward = []
        for site in sites:
            gens = _propagated(circuit, site)
            forward.append(gens[0])
            backward.append(gens[1])
        for i, gen in enumerate(forward):
            if not backward[i].x.any():
                self.assertFalse(
                    all(gen.commutes(other) for other in forward[:i]),
                    msg="trivial rotation commutes into the input",
                )
            if not gen.x.any():
                self.assertFalse(
                    all(gen.commutes(other) for other in forward[i + 1 :]),
                    msg="diagonal rotation commutes into the measurement",
                )
            for j in range(i + 1, len(forward)):
                if np.array_equal(gen.x, forward[j].x) and np.array_equal(gen.z, forward[j].z):
                    self.assertTrue(
                        any(not gen.commutes(k) for k in forward[i + 1 : j]),
                        msg="equivalent rotations merge into a Clifford",
                    )

    def test_changes_distribution_and_injects_magic(self):
        """XEB doping alters the sampled distribution and injects magic."""
        circuit = _paper_ansatz(5, seed=1)
        doped, sites = dope_clifford_circuit(circuit, preserve_distribution=False)
        self.assertGreater(len(sites), 0)
        self.assertFalse(
            np.allclose(
                Statevector(doped).probabilities(), Statevector(circuit).probabilities(), atol=1e-6
            )
        )
        self.assertGreater(_stabilizer_renyi_2(doped), 0.5)

    def test_trivial_circuits_have_no_sites(self):
        """Circuits whose rotations all prune away (bare H, diagonal circuit) yield no site."""
        h_only = QuantumCircuit(1)
        h_only.h(0)
        diagonal = QuantumCircuit(2)
        diagonal.s(0)
        diagonal.cz(0, 1)
        for circuit in (h_only, diagonal):
            doped, sites = dope_clifford_circuit(circuit, preserve_distribution=False)
            self.assertEqual(sites, [])
            self.assertEqual(doped.count_ops().get("t", 0), 0)

    def test_sites_are_irreducible(self):
        """No pruning rewrite of the reference applies to the returned site set."""
        for seed in range(3):
            circuit = random_clifford(4, seed=seed).to_circuit()
            _, sites = dope_clifford_circuit(circuit, preserve_distribution=False)
            self.assertGreater(len(sites), 0, msg=f"seed {seed}")
            self._assert_irreducible(circuit, sites)
        circuit = _paper_ansatz(4, seed=2)
        _, sites = dope_clifford_circuit(circuit, preserve_distribution=False)
        self._assert_irreducible(circuit, sites)

    def test_num_t_gates_and_seed(self):
        """Random subsets are exact in size, seed-reproducible, and themselves irreducible."""
        circuit = _paper_ansatz(4, seed=0)
        _, all_sites = dope_clifford_circuit(circuit, preserve_distribution=False)
        self.assertGreater(len(all_sites), 3)
        doped, sites = dope_clifford_circuit(
            circuit, num_t_gates=3, preserve_distribution=False, seed=42
        )
        self.assertEqual(len(sites), 3)
        self.assertEqual(doped.count_ops()["t"], 3)
        self.assertTrue(set(sites) <= set(all_sites))
        self._assert_irreducible(circuit, sites)
        _, again = dope_clifford_circuit(
            circuit, num_t_gates=3, preserve_distribution=False, seed=42
        )
        self.assertEqual(sites, again)
        with self.assertRaises(ValueError):
            dope_clifford_circuit(
                circuit, num_t_gates=len(all_sites) + 1, preserve_distribution=False
            )


class TestCheckedCircuitDoping(unittest.TestCase):
    """Tests for doping a :class:`.CheckedCircuit` without breaking its spacetime code."""

    def _assert_code_preserved(self, checked: CheckedCircuit, doped: CheckedCircuit, sites):
        """The doped circuit keeps the checks' metadata, wires, syndromes, and cumulants."""
        self.assertIsInstance(doped, CheckedCircuit)
        for field in ("target_qubits", "check_qubits", "check_support", "cost"):
            self.assertEqual(getattr(doped, field), getattr(checked, field))
        num_qubits = checked.circuit.num_qubits
        # Sites avoid ancilla wires and post-measurement wires.
        measure_pos = {
            checked.circuit.find_bit(inst.qubits[0]).index: index
            for index, inst in enumerate(checked.circuit.data)
            if inst.operation.name == "measure"
        }
        for site in sites:
            self.assertNotIn(site.qubit, checked.check_qubits)
            self.assertLessEqual(_position(site), measure_pos[site.qubit])
        # Every syndrome stays deterministic with its original sign.
        original = _syndrome_values(checked, checked.circuit)
        for value in original:
            self.assertAlmostEqual(abs(value), 1.0, places=10)
        np.testing.assert_allclose(_syndrome_values(doped, doped.circuit), original, atol=1e-10)
        # Z on every doped wire commutes with each check's back-cumulant there.
        for site in sites:
            site_z = Pauli("I" * num_qubits)
            site_z.z[site.qubit] = True
            suffix = Clifford(_split(checked.circuit, _position(site))[1])
            for support in checked.check_support:
                label = "".join("Z" if q in support else "I" for q in reversed(range(num_qubits)))
                cumulant = Pauli(label).evolve(suffix, frame="h")
                self.assertTrue(cumulant.commutes(site_z))

    def test_preserving_mode(self):
        """Distribution-preserving doping keeps the full output distribution and the code."""
        checked = _checked_circuit()
        self.assertEqual(len(checked.check_qubits), 1)
        doped, sites = dope_clifford_circuit(checked)
        self.assertGreater(len(sites), 0)
        self._assert_code_preserved(checked, doped, sites)
        np.testing.assert_allclose(
            Statevector(doped.circuit.remove_final_measurements(inplace=False)).probabilities(),
            Statevector(checked.circuit.remove_final_measurements(inplace=False)).probabilities(),
            atol=1e-12,
        )

    def test_xeb_mode(self):
        """XEB doping changes the payload distribution but never breaks a check."""
        checked = _checked_circuit()
        doped, sites = dope_clifford_circuit(checked, preserve_distribution=False)
        self.assertGreater(len(sites), 0)
        self._assert_code_preserved(checked, doped, sites)

    def test_hardware_style_template(self):
        """A parametric after-entangling template keeps every syndrome at any angles."""
        checked = _checked_circuit()
        doped, sites = dope_clifford_circuit(
            checked, preserve_distribution=False, after_entangling_only=True, parametric=True
        )
        self.assertGreater(len(sites), 0)
        for site in sites:
            inst = checked.circuit.data[site.after_instruction]
            self.assertGreater(inst.operation.num_qubits, 1)
        angles = np.random.default_rng(0).uniform(0, 2 * np.pi, len(sites))
        bound = doped.circuit.assign_parameters(angles)
        original = _syndrome_values(checked, checked.circuit)
        np.testing.assert_allclose(_syndrome_values(checked, bound), original, atol=1e-10)


class TestHardwareStyle(unittest.TestCase):
    """Tests for ``after_entangling_only`` site restriction and ``parametric`` templates."""

    def test_after_entangling_only(self):
        """Sites are restricted to wires directly following an entangling gate."""
        circuit = _paper_ansatz(5, seed=1)
        for preserve in (True, False):
            _, all_sites = dope_clifford_circuit(circuit, preserve_distribution=preserve)
            doped, sites = dope_clifford_circuit(
                circuit, preserve_distribution=preserve, after_entangling_only=True
            )
            self.assertGreater(len(sites), 0, msg=f"preserve {preserve}")
            self.assertLessEqual(len(sites), len(all_sites))
            for site in sites:
                inst = circuit.data[site.after_instruction]
                self.assertGreater(inst.operation.num_qubits, 1)
                self.assertIn(site.qubit, [circuit.find_bit(q).index for q in inst.qubits])
            if preserve:
                np.testing.assert_allclose(
                    Statevector(doped).probabilities(),
                    Statevector(circuit).probabilities(),
                    atol=1e-12,
                )

    def test_parametric_template(self):
        """One template reproduces T doping at pi/4, the base circuit at 0, and always
        preserves the distribution in distribution-preserving mode."""
        circuit = _paper_ansatz(4, seed=1)
        template, sites = dope_clifford_circuit(circuit, parametric=True)
        self.assertEqual(len(template.parameters), len(sites))
        t_doped, t_sites = dope_clifford_circuit(circuit)
        self.assertEqual(sites, t_sites)
        bound = template.assign_parameters([np.pi / 4] * len(sites))
        self.assertTrue(Statevector(bound).equiv(Statevector(t_doped)))
        zero = template.assign_parameters([0.0] * len(sites))
        self.assertTrue(Statevector(zero).equiv(Statevector(circuit)))
        angles = np.random.default_rng(5).uniform(0, 2 * np.pi, len(sites))
        np.testing.assert_allclose(
            Statevector(template.assign_parameters(angles)).probabilities(),
            Statevector(circuit).probabilities(),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
