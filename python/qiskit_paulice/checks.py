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

"""Functionality for finding effective spacetime Pauli checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeGuard

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli

from ._internal import Metric as _Metric
from ._internal import NoiseModel as _NoiseModel
from ._internal import pick_checks as _pick_checks
from .checked_circuit import CheckedCircuit
from .noise_models import GateNoise, GateWiseNoise, LayeredGateNoise, NoiseModel


def add_pauli_checks(
    circuit: QuantumCircuit,
    target_qubits: Sequence[int],
    noise_model: NoiseModel,
    cost: Literal["gamma", "LER"] = "gamma",
    cost_nshots: int = 10_000,
    method: Literal["windowed", "genetic", "windowed_genetic"] = "windowed",
    ancilla_qubits: Sequence[int] | None = None,
    check_creg_name: str = "checks_c",
    check_qreg_name: str = "checks_q",
    seed: int | None = None,
):
    r"""Add spacetime Pauli checks to a Clifford circuit.

    The check picking algorithm finds valid, low weight checks on each target qubit in the order
    they are specified in ``target_qubits`` and chooses the check which provides the most error
    detection coverage (i.e. minimizes the ``cost`` value). Once a check is committed on
    ``target_qubits[0]``, it will be set for the remainder of the algorithm and a search for a
    good check on ``target_qubits[1]`` will begin. For this reason, the ordering of
    ``target_qubits`` can have some impact on the set of checks produced by the function.

    This function produces :class:`qiskit_paulice.CheckedCircuit` instances containing numbers of
    checks ranging from ``0`` to one check per target qubit. It can be instructive to view the
    convergence of the cost function as more checks are added, as one may see convergence of the
    cost using fewer checks.

    For details on finding effective spacetime Pauli checks, see `Supplemental Sec. II-VI of Martiel, Javadi <https://arxiv.org/abs/2504.15725>`_.

    Args:
        circuit: The Clifford circuit to dress with spacetime Pauli checks. The circuit must be
            terminated with a measurement on at least one qubit. The circuit may be defined on
            virtual or physical qubits. If the circuit has a layout, the user must provide
            ``ancilla_qubits``.
        target_qubits: Qubit indices of ``circuit`` which will be used to entangle the check
            qubits to the payload. When ``circuit`` has a layout (ISA mode), these are physical
            qubit indices, in the same index space as ``ancilla_qubits``.
        noise_model: A noise model describing the effect of noise on the target device. This
            model will be used to estimate the effect of a given check during the check picking
            process. While one can generate a noise model from learned Pauli-Lindblad noise, a rougher
            approximation of the noise generated from backend benchmark data is often sufficient.
        cost: Metric to optimize. Can be ``"gamma"`` or ``"LER"`` (logical error rate).

            - ``"gamma"``: The gamma value associated with the inverse logical noise channel
              (i.e. the noise channel consisting of errors within the measurement lightcone that are
              undetectable by the checks).
            - ``"LER"``: The empirical logical error rate after postselection. Performs Monte Carlo
              simulations to calculate the fraction of postselected shots affected by uncovered logical noise.
        cost_nshots: Number of Monte Carlo shots used by Monte Carlo-based cost metrics (currently only ``"LER"``).
        method: Check picking method (``"windowed"``, ``"genetic"``, or ``"windowed_genetic"``). Each method
            will add checks to ``target_qubits`` sequentially. Once a check is committed to a given target
            qubit, it will not be undone as more checks are added. Each method picks checks which provide
            maximum error detection capability (i.e. lowest ``cost``).

            - ``"windowed"``: Sample small subsets of the wires on each target qubit to find good checks
            - ``"genetic"``: Evolve a set of candidate checks from the full wire-space of each target qubit
            - ``"windowed_genetic"``: Run a genetic search within random windows of wires on each target qubit
        ancilla_qubits: List of physical qubit indices (one per index in ``target_qubits``) specifying where to
            place the check ancillas in the output circuit. Required when ``circuit.layout`` is not ``None``.
            ``ancilla_qubits[i]`` will share entangling gates with ``target_qubits[i]`` when implementing the
            ``i``\\ th check.
        check_creg_name: Name of the classical register for check measurements (default: "checks_c")
        check_qreg_name: Name of the quantum register holding the check ancillas in the output circuits
            (default: "checks_q"). Ignored in ISA mode.
        seed: Random seed for controlling randomness during the search for good checks. While this seed controls
            some randomness in the algorithm, some non-determinism still exists when using ``LER`` cost function,
            or either variety of genetic check picking method. The combination of ``cost="gamma"`` and
            ``method="windowed"`` is fully deterministic if ``seed`` is not ``None``.

    Returns:
        A list of :class:`qiskit_paulice.CheckedCircuit` instances -- instances containing the bare circuit with
        no checks and one for each added check. The final element in the output contains the :class:`qiskit_paulice.CheckedCircuit`
        with checks on every target qubit, assuming a valid set of checks could be found.
    """
    # Set global random seed if provided for full reproducibility
    if seed is not None:
        np.random.seed(seed)

    if cost.lower() == "gamma":
        metric = _Metric.gamma()
    elif cost.lower() == "balanced_gamma":
        metric = _Metric.balanced_gamma()
    elif cost.lower() == "ler":
        metric = _Metric.logical_error_rate(cost_nshots)
    else:
        raise ValueError(f"Invalid cost value: {cost}")

    circuit = circuit.copy()
    virtual_circuit, measurement_info, cregs, qregs = _strip_measurements_cregs_barriers(circuit)

    # ISA mode: input has been transpiled with an `initial_layout`, so qubit
    # indices in `circuit` are physical and `circuit.layout` records the
    # original-virtual -> physical mapping. We run the picker on a small
    # virtual payload and lift the result back onto the input's full physical
    # qreg layout, routing each new ancilla to `ancilla_qubits[i]`.
    is_isa = circuit.layout is not None
    payload_phys: list[int] = []

    if is_isa:
        if ancilla_qubits is None:
            raise ValueError(
                "Input circuit was transpiled onto a layout (has a `circuit.layout`); "
                "`add_pauli_checks` cannot infer where to place the check ancillas. "
                "Pass `ancilla_qubits` (one physical qubit index per target, in the "
                "same order as `target_qubits`)."
            )
        ancilla_qubits = list(ancilla_qubits)
        if len(ancilla_qubits) != len(target_qubits):
            raise ValueError(
                f"`ancilla_qubits` must have one entry per target "
                f"(got {len(ancilla_qubits)} for {len(target_qubits)} targets)."
            )
        if any(not 0 <= a < circuit.num_qubits for a in ancilla_qubits):
            raise ValueError(
                f"`ancilla_qubits` must be physical qubit indices in "
                f"[0, {circuit.num_qubits}); got {ancilla_qubits}."
            )
        if len(set(ancilla_qubits)) != len(ancilla_qubits):
            raise ValueError(f"`ancilla_qubits` contains duplicates: {ancilla_qubits}")

        payload_phys = list(circuit.layout.initial_index_layout(filter_ancillas=True))
        overlap = sorted(set(ancilla_qubits) & set(payload_phys))
        if overlap:
            raise ValueError(
                f"`ancilla_qubits` overlap with payload qubits {overlap}; pick "
                f"physical qubits not used by the payload."
            )

        # Rebuild the virtual circuit on the smaller `len(payload_phys)` width
        # by remapping physical qubit indices back to original virtual ones.
        phys_to_virt = {p: v for v, p in enumerate(payload_phys)}
        virtual_circuit = QuantumCircuit(len(payload_phys))
        for inst in circuit.data:
            if inst.operation.name in ("measure", "barrier"):
                continue
            old = [circuit.find_bit(q).index for q in inst.qubits]
            if all(q in phys_to_virt for q in old):
                virtual_circuit.append(inst.operation, [phys_to_virt[q] for q in old])

        measured_qubits = list(
            {phys_to_virt[m[0]] for m in measurement_info if m[0] in phys_to_virt}
        )

        # target_qubits are physical indices the payload occupies (same index
        # space as ancilla_qubits); map them to payload-virtual for the picker.
        target_phys = [int(q) for q in target_qubits]
        not_payload = sorted(set(target_phys) - set(payload_phys))
        if not_payload:
            raise ValueError(
                f"target_qubits {not_payload} are not payload qubits of the "
                f"transpiled circuit. In ISA mode (the circuit has a layout), "
                f"`target_qubits` must be physical qubit indices the payload "
                f"occupies -- the same index space as `ancilla_qubits`."
            )
        picker_targets = [phys_to_virt[q] for q in target_phys]

        # In ISA mode no new qreg is added (we reuse existing physical qubits);
        # only the check_creg_name needs to be collision-free.
        if any(cr.name == check_creg_name for cr in cregs):
            raise ValueError(
                f"Input circuit already has a classical register named "
                f"{check_creg_name!r}; pass a different `check_creg_name`."
            )
    else:
        # `ancilla_qubits` has no effect for non-ISA circuits; silently ignore
        # so callers can pass the same args dict in both modes.
        ancilla_qubits = None

        # Reject register-name collisions up front so the output circuits are well-formed.
        if check_qreg_name == check_creg_name:
            raise ValueError(
                f"check_qreg_name and check_creg_name must differ ({check_qreg_name!r} given for both); "
                f"Qiskit forbids two registers in the same circuit sharing a name."
            )
        if any(qr.name == check_qreg_name for qr in qregs):
            raise ValueError(
                f"Input circuit already has a quantum register named {check_qreg_name!r}; "
                f"pass a different `check_qreg_name`."
            )
        if any(cr.name == check_creg_name for cr in cregs):
            raise ValueError(
                f"Input circuit already has a classical register named {check_creg_name!r}; "
                f"pass a different `check_creg_name`."
            )

        measured_qubits = list(set(m[0] for m in measurement_info))

        # Non-ISA: target_qubits index the circuit you passed.
        n = virtual_circuit.num_qubits
        out_of_range = sorted({int(q) for q in target_qubits if not 0 <= int(q) < n})
        if out_of_range:
            raise ValueError(
                f"target_qubits {out_of_range} are out of range for a {n}-qubit "
                f"circuit. If these are physical qubit indices, transpile/lay out "
                f"the circuit onto hardware first (then pass physical targets in "
                f"ISA mode) before calling add_pauli_checks."
            )
        picker_targets = [int(q) for q in target_qubits]

    gate_noise = noise_model.gate_noise
    _gate_noise = None
    if gate_noise is not None:
        if _is_uniform_gate_noise(gate_noise):
            _gate_noise = _NoiseModel.uniform_depolarizing(gate_noise)
        elif _is_layered_gate_noise(gate_noise):
            _gate_noise = _NoiseModel.layered(_convert_layered_noise(gate_noise))
        elif _is_gate_wise_noise(gate_noise):
            _gate_noise = _NoiseModel.gate_wise(_convert_gate_wise_noise(gate_noise))
    _readout_noise = (
        _NoiseModel.readout(noise_model.readout_noise)
        if noise_model.readout_noise is not None
        else None
    )

    _noise_model = [x for x in (_gate_noise, _readout_noise) if x is not None]
    if len(_noise_model) == 0:
        raise ValueError("The noise model may not be empty.")
    result = _pick_checks(
        virtual_circuit,
        picker_targets,
        _noise_model,
        measured_qubits=measured_qubits,
        metric=metric,
        method=method,
        seed=seed,
        verbose=False,
    )

    if is_isa:
        # Lift each picker variant onto the input ISA's physical-qubit layout.
        anc_list = list(ancilla_qubits)  # type: ignore[arg-type]
        result.circuits = [
            _lift_to_isa_circuit(
                variant=v,
                qregs=qregs,
                cregs=cregs,
                measurement_info=measurement_info,
                num_payload_virtual=len(payload_phys),
                num_active_checks=k,
                payload_phys=payload_phys,
                ancilla_qubits=anc_list,
                check_creg_name=check_creg_name,
            )
            for k, v in enumerate(result.circuits)
        ]
        # `virtual_zs` and `check_qubits` from the picker reference virtual
        # indices into the small payload-only circuit. Remap them to the
        # physical indices used by the lifted output so post-selection reads
        # the right measurement bits.
        n_payload_v = len(payload_phys)
        result.virtual_zs = [
            [payload_phys[q] if q < n_payload_v else anc_list[q - n_payload_v] for q in vzs]
            for vzs in result.virtual_zs
        ]
        result.check_qubits = [
            anc_list[q - n_payload_v] if q >= n_payload_v else payload_phys[q]
            for q in result.check_qubits
        ]
        target_qubits_out = [payload_phys[t] for t in picker_targets]
    else:
        # Restore original measurements and add check measurements (non-ISA path).
        _restore_measurements_and_cregs(
            result.circuits,
            measurement_info,
            cregs,
            num_original_qubits=virtual_circuit.num_qubits,
            qregs=qregs,
            check_qubits=result.check_qubits,
            check_creg_name=check_creg_name,
            check_qreg_name=check_qreg_name,
        )
        target_qubits_out = list(target_qubits)

    # Build the public per-variant list. `result.costs[k]` is the picker metric
    # after committing the first `k` checks.
    costs = result.costs if result.costs is not None else [None] * len(result.circuits)
    target_qubits_tuple = tuple(target_qubits_out)
    check_qubits_tuple = tuple(result.check_qubits)
    check_support_tuple = tuple(tuple(s) for s in result.virtual_zs)
    return [
        CheckedCircuit(
            circuit=circ,
            target_qubits=target_qubits_tuple[:k],
            check_qubits=check_qubits_tuple[:k],
            check_support=check_support_tuple[:k],
            cost=costs[k],
            cost_metric=cost,
        )
        for k, circ in enumerate(result.circuits)
    ]


def _strip_measurements_cregs_barriers(circuit: QuantumCircuit):
    """Strip measurements, classical registers, and barriers from a circuit.

    Returns:
        tuple: (virtual_circuit, measurement_info, cregs, qregs) where:
            - virtual_circuit: Circuit without measurements, barriers, or cregs
            - measurement_info: List of (qubit_idx, clbit_idx, creg_name) tuples
            - cregs: List of ClassicalRegister objects from the input
            - qregs: List of QuantumRegister objects from the input (preserved so
              the output circuit can be rebuilt with the same payload-register
              layout, with any check ancillas appended in their own register).
    """
    measurement_info = []
    for inst in circuit.data:
        if inst.operation.name == "measure":
            qubit_idx = circuit.find_bit(inst.qubits[0]).index
            clbit = inst.clbits[0]
            for creg in circuit.cregs:
                if clbit in creg:
                    measurement_info.append((qubit_idx, creg.index(clbit), creg.name))
                    break

    cregs = list(circuit.cregs)
    qregs = list(circuit.qregs)

    # Create virtual circuit without measurements, barriers, or classical bits
    virtual_circuit = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        if inst.operation.name not in ("measure", "barrier"):
            qubit_indices = [circuit.find_bit(q).index for q in inst.qubits]
            virtual_circuit.append(inst.operation, qubit_indices)

    return virtual_circuit, measurement_info, cregs, qregs


def _remove_inactive_qubits(
    circuit,
    num_original_qubits,
    num_active_checks,
    qregs=None,
    check_qreg_name="checks_q",
):
    """Drop inactive ancillas and rebuild with a payload + dedicated check register.

    The output circuit's quantum-register layout is ``[*qregs, pchecks_qreg]``
    when ``num_active_checks > 0``, or just ``[*qregs]`` for the bare circuit
    (``circ_idx == 0``). Old qubit indices ``[0, num_original_qubits)`` map onto
    the payload qregs (in the order given), and ``[num_original_qubits,
    num_original_qubits + num_active_checks)`` map onto the new check qreg.
    Gates touching inactive ancillas (indices beyond that range) are dropped.

    Args:
        circuit: Circuit with all ancilla qubits allocated by the picker.
        num_original_qubits: Number of qubits in the input payload.
        num_active_checks: Number of check ancillas to keep for this variant.
        qregs: Quantum registers from the original input circuit. If None or the
            sizes don't sum to ``num_original_qubits``, falls back to a single
            anonymous payload register.
        check_qreg_name: Name for the new register holding active check ancillas.

    Returns:
        New circuit with the chosen qreg layout.
    """
    if qregs is None or sum(qr.size for qr in qregs) != num_original_qubits:
        qregs = [QuantumRegister(num_original_qubits, "q")]

    payload_qubits = [q for qr in qregs for q in qr]
    if num_active_checks > 0:
        check_qreg = QuantumRegister(num_active_checks, check_qreg_name)
        new_circuit = QuantumCircuit(*qregs, check_qreg)
        qubit_objects = payload_qubits + list(check_qreg)
    else:
        new_circuit = QuantumCircuit(*qregs)
        qubit_objects = payload_qubits

    n_keep = len(qubit_objects)
    for inst in circuit.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        old_qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if all(q < n_keep for q in old_qubits):
            new_circuit.append(inst.operation, [qubit_objects[q] for q in old_qubits])

    return new_circuit


def _lift_to_isa_circuit(
    variant,
    qregs,
    cregs,
    measurement_info,
    num_payload_virtual,
    num_active_checks,
    payload_phys,
    ancilla_qubits,
    check_creg_name="checks_c",
):
    """Lift a picker-output variant onto an ISA circuit's qreg layout.

    The picker runs on a small virtual payload (``num_payload_virtual`` qubits)
    plus appended ancillas at virtual indices ``[num_payload_virtual, ...)``.
    This helper rebuilds the variant on the *input* ISA circuit's full-width
    quantum registers, mapping virtual qubit indices to physical ones:

    * virtual ``v < num_payload_virtual`` -> physical ``payload_phys[v]``
    * virtual ``num_payload_virtual + j`` (``j < num_active_checks``) ->
      physical ``ancilla_qubits[j]``
    * gates touching higher (inactive) ancilla indices are dropped

    Original measurements are restored at their physical qubit positions, and
    a fresh ``check_creg_name`` ClassicalRegister of width
    ``num_active_checks`` records the active check ancillas' Z outcomes.
    """
    new_circuit = QuantumCircuit(*qregs)
    for cr in cregs:
        new_circuit.add_register(cr)

    def _remap(q):
        if q < num_payload_virtual:
            return payload_phys[q]
        if q < num_payload_virtual + num_active_checks:
            return ancilla_qubits[q - num_payload_virtual]
        return None

    for inst in variant.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        old = [variant.find_bit(q).index for q in inst.qubits]
        new = [_remap(q) for q in old]
        if any(nq is None for nq in new):
            continue
        new_circuit.append(inst.operation, new)

    for qubit_idx, clbit_idx, creg_name in measurement_info:
        creg = next(reg for reg in new_circuit.cregs if reg.name == creg_name)
        new_circuit.measure(qubit_idx, creg[clbit_idx])

    if num_active_checks > 0:
        check_creg = ClassicalRegister(num_active_checks, name=check_creg_name)
        new_circuit.add_register(check_creg)
        for j in range(num_active_checks):
            new_circuit.measure(ancilla_qubits[j], check_creg[j])

    return new_circuit


def _restore_measurements_and_cregs(
    circuits,
    measurement_info,
    cregs,
    num_original_qubits,
    qregs=None,
    check_qubits=None,
    check_creg_name="checks_c",
    check_qreg_name="checks_q",
):
    """Restore measurements and classical registers to circuits.

    Args:
        circuits: List of circuits to restore metadata to. circuits[i] has i active checks.
        measurement_info: List of (qubit_idx, clbit_idx, creg_name) tuples
        cregs: List of ClassicalRegister objects to restore
        num_original_qubits: Number of qubits in the original circuit (before adding checks)
        qregs: List of QuantumRegister objects from the input circuit (preserved
            in the output's payload-qreg layout). Defaults to a single anonymous
            register sized ``num_original_qubits`` if not provided.
        check_qubits: List of all ancilla qubit indices for Pauli checks (optional)
        check_creg_name: Name of the classical register for check measurements
        check_qreg_name: Name of the quantum register holding active check ancillas
    """
    for circ_idx, circ in enumerate(circuits):
        num_active_checks = circ_idx
        circ = _remove_inactive_qubits(
            circ,
            num_original_qubits,
            num_active_checks,
            qregs=qregs,
            check_qreg_name=check_qreg_name,
        )
        circuits[circ_idx] = circ

        for creg in cregs:
            circ.add_register(creg)

        for qubit_idx, clbit_idx, creg_name in measurement_info:
            creg = next(reg for reg in circ.cregs if reg.name == creg_name)
            circ.measure(qubit_idx, creg[clbit_idx])

        # Active check ancillas occupy the dedicated `check_qreg`, which lives
        # at the tail of the qubit list (indices [num_original_qubits, ...)).
        if check_qubits and num_active_checks > 0:
            check_creg = ClassicalRegister(num_active_checks, name=check_creg_name)
            circ.add_register(check_creg)
            for idx in range(num_active_checks):
                qubit_idx = num_original_qubits + idx
                circ.measure(qubit_idx, check_creg[idx])


def _is_uniform_gate_noise(noise: GateNoise) -> TypeGuard[float]:
    return isinstance(noise, float)


def _is_layered_gate_noise(noise: GateNoise) -> TypeGuard[dict]:
    if not isinstance(noise, dict) or not noise:
        return False
    first_key = next(iter(noise.keys()))
    return isinstance(first_key, tuple) and len(first_key) > 0 and isinstance(first_key[0], tuple)


def _is_gate_wise_noise(noise: GateNoise) -> TypeGuard[dict]:
    if not isinstance(noise, dict) or not noise:
        return False
    first_key = next(iter(noise.keys()))
    return isinstance(first_key, tuple) and len(first_key) == 2 and isinstance(first_key[0], int)


def _convert_layered_noise(noise: LayeredGateNoise):
    new_noise = {}
    for layer in noise:
        converted_noise = []
        for p, r in noise[layer]:
            p_str = p.to_label() if isinstance(p, Pauli) else p
            converted_noise.append((p_str, r))
        new_noise[layer] = converted_noise
    return new_noise


def _convert_gate_wise_noise(noise: GateWiseNoise):
    pauli_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    new_noise = {}
    for edge in noise:
        converted_noise = []
        for p, r in noise[edge]:
            p_str = p.to_label() if isinstance(p, Pauli) else p
            if len(p_str) != 2:
                raise ValueError("Pauli generators must be defined on 2 qubits.")
            p_tuple = (pauli_map[p_str[0]], pauli_map[p_str[1]])
            converted_noise.append((p_tuple, r))
        new_noise[edge] = converted_noise
    return new_noise
