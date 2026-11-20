# Handoff: error detection + PEC guide on branch `qed-pec`

**Context.** In the scratch repo `qiskit-paulice-direct` (old machine, branch `profile`, dirty)
we ran a July–August 2026 campaign combining spacetime Pauli checks (QED) with probabilistic error
cancellation (PEC): learn a Pauli-Lindblad model per layer, mask out the generators the checks
detect, let PEC invert only the rest. Two concessions came out of it: one learned noise layer per
check, and a noise-model truncation that is only first order, whose residual bias grows faster
than linearly with depth. The port of that work into this project is **done and pruned**: after
discussion, nothing was added to the package API. The deliverable is one executed guide notebook
plus a folder of reference figures and scripts. This document is everything a fresh session on a
new machine needs; the old machine's Claude memory notes do not travel.

## State of the branch (all uncommitted on top of `origin/main` = `2b8ed2a`)

Tracked changes, to commit:

- `docs/guides/error_detection_with_error_cancellation.ipynb` — new, executed (outputs embedded).
- `docs/guides/index.rst` — one toctree entry for it.
- `releasenotes/notes/postselected-pec-6a2d9f3b7c1e4850.yaml` — short `other:` note for the guide.

Untracked, **not to commit** (user's call), but needed on the new machine:

- `qed_pec_artifacts/` (4.6 MB): campaign figures, data, driver scripts, and the notebook tooling.
  Its `README.md` maps every figure to its result and the old scratch-repo API to the guide's
  steps. Copy it out of band (zip / scp); it is not in git.
- This file.

The package (`python/`, `src/`, `pyproject.toml`, tests) is byte-identical to `origin/main`.
`pytest` → 184 passed; `ruff format --check`, `ruff check` (incl. the notebook) clean.

**To move:** on the old machine `git add docs releasenotes && git commit && git push -u origin
qed-pec`; copy `qed_pec_artifacts/` and this file separately. Nothing else is needed.

## Decisions already made (do not relitigate without the user)

1. **No PEC code in the package.** This is an error-detection project. Three API additions were
   written, tested, and then removed on the user's call: `CheckedCircuit.compute_local_scales` /
   `postselected_noise_mask`, `get_syndrome_flips` + `get_acceptance_mask`, and a `qiskit_paulice.pec`
   ratio estimator. Everything they did is done in the guide with existing API plus
   `qiskit-mitigation`. The user wants the API pruned to necessary/interesting endpoints.
2. **The mask comes from `qiskit_mitigation.noise.create_postselected_noise_mask`** (released in
   qiskit-mitigation 0.1.0/0.1.1 on PyPI; `(scales, gamma'^2)`), never from a local
   back-propagation.
3. **Guide style follows the well-received Ising tutorial** at
   `~/projects/repos_pvt/spacetime_CPC/docs/tutorials/ising_ed_pec_ieee/ising_ed_pec_ieee.ipynb`
   (old machine): detectors built by hand, `measurement_flips` XORed into the bits,
   `executor_expectation_values` for the estimate.
4. **No batched `get_postselection_method`.** Considered and declined: today an array argument
   means one qubit-indexed shot; changing that breaks the array contract for no real need.
5. **Artifacts stay untracked.** Figures/JSON/scripts are reference material, not source.

## What the guide does (and the facts it rests on)

Sections: (1) 10-qubit mirror brickwork Clifford (U then U†, so ⟨Z_q⟩ = 1 exactly);
(2) `add_pauli_checks` → all 10 checks, `checked.box(payload_layers=[even_edges, odd_edges])` →
43 noise boxes, 12 unique layers (2 payload + 1 per check); (3) synthetic per-layer
`PauliLindbladMap`s stand in for `NoiseLearnerV3`, mask → 84.7 % of generators detected,
γ = 1.74 → γ' = 1.08; (4) `samplomatic.build`, noise applied in Aer as a `PauliLindbladError`
appended after each box's closing barrier, stabilizer simulator; (5) estimate raw / QED / PEC /
QED+PEC; (6) exact Pauli-level Monte Carlo of the first-order truncation bias vs depth.
Executes in ~3–4 min. Final numbers: raw 0.9123, QED 0.9956, PEC 0.9842 ± 0.006,
QED+PEC 0.9999 ± 0.0012; bias 0.007 % @ L=6 … 0.243 % @ L=96, fit ∝ L^1.31.

Verified conventions (samplomatic 0.21.0) — these cost real debugging time, keep them:

- A box's `PauliLindbladMap` must have `num_qubits == box width` and index the box's qubits in
  **sorted physical order** (`InjectNoise` docstring; full-width maps are rejected at bind time).
- `basis_changes` values are symplectic `I=0, Z=1, X=2, Y=3`; measuring in Z = all ones.
- **XOR `measurement_flips.<creg>` into every measured bit** (payload and ancilla) before doing
  anything else. Forgetting the payload reads ⟨Z⟩ ≈ 0 on a noiseless mirror circuit; forgetting
  the ancillas post-selects exactly the wrong shots and nothing looks wrong.
- `add_pauli_checks` measures qubit q into clbit q, so bit arrays pass to
  `get_postselection_method` directly (its array path is qubit-indexed, one shot at a time).
- A check's detector is the Z string on its `check_support` (a parity over several bits), not a
  single ancilla bit as in the Ising tutorial.
- The template closes box k with a barrier labelled `R<k>…`; the guide inserts the Aer error
  after it. The twirl's `rz(kπ/2)` gates are rewritten to `id/s/z/sdg` for the stabilizer method.
- The bias Monte Carlo packs detector flips into a `uint64`, so `num_checks + NUM_QUBITS ≤ 64`.
- The mask must be computed on the boxed **checked** circuit, never on a boxing of the bare
  circuit (different `ref`s, colliding `modifier_ref`s).

**Error bars.** `executor_expectation_values`' reported variance treats shots as independent, but
shots within a randomization share a sign and an inserted Pauli. On the Aer demo it gave plain
PEC ± 0.0024 where a jackknife/bootstrap over randomizations gives ± 0.006 (would have made a
1.6 % fluctuation look like a 6σ bias); on synthetic i.i.d. data it errs the other way (reported
0.0082 vs true 0.0045). The guide therefore takes point estimates from
`executor_expectation_values` and errors from a 200-sample bootstrap over randomizations (19 ms
per call, ~15 s total). Worth raising upstream in qiskit-mitigation.

**Bias exponent.** The campaign's 20-qubit hardware model with 9 checks gave a clean quadratic,
3.5e-4·L² % at 2.9e-3 EPLG (`qed_pec_artifacts/figures/truncation_bias_scaling.png`). The 10-qubit
toy gives L^1.3 at both noise strengths tried (so not a higher-order effect; it is the check
geometry). The guide fits a power law and says so honestly; don't force a quadratic.

## Environment (old machine; reproduce on the new one)

Python 3.12.9, rustc 1.97. `~/venvs/dev` had: qiskit 2.3.0, qiskit-aer 0.17.2, samplomatic
0.21.0, numpy 2.4.2, matplotlib 3.10.7, maturin 1.12.4, pytest 9.0.1, ruff 0.14.14, pylint 4.0.8,
mypy 1.19.0, nbclient 0.10.2, ipykernel 7.1.0, qiskit-mitigation (editable from
`~/projects/repos_pub/qiskit-mitigation`; released 0.1.1 is fine).

```
python -m venv ~/venvs/dev && source ~/venvs/dev/bin/activate
pip install maturin "qiskit-mitigation>=0.1" matplotlib nbclient ipykernel
pip install -e ".[dev]"          # builds the Rust extension via maturin
pytest test -q                   # 184 passed on origin/main
ruff format --check python/ docs/ test/ && ruff check python/ docs/ test/
jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=1800 docs/guides/error_detection_with_error_cancellation.ipynb
```

The notebook was generated from `qed_pec_artifacts/notebook_tooling/build_nb.py` (nbformat
cells) and executed by `exec_nb.py` (starts the kernel on the venv's python; only needed when the
package is *not* installed in the kernel's environment). From here on edit the `.ipynb` directly;
the builder is kept only as a record. `demo.py` is the pre-notebook prototype, including a
`save fig` version of both charts.

## Not yet done / next steps

- `tox -e docs` was never run (no sphinx in the venv): check the guide renders and the toctree
  entry is right. `tox -e lint` runs `nbqa pylint` on docs — not run either; `ruff` on the
  notebook is clean.
- `tox -e py-notebook` (nbmake) not run; the guide takes ~4 min, within the 3000 s timeout.
- Release-note wording is a placeholder `other:` entry; adjust to taste or drop.
- Possible follow-up guide section: hardware run with `NoiseLearnerV3` (maps come back in backend
  qubit order → remap to box-local sorted order, as the Ising tutorial's cell 12 does) and TREX via
  `rescale_factors`. The campaign scripts in `qed_pec_artifacts/scripts/` show the full hardware
  driver (`run_best_20x20.py`) but target the old scratch-repo API (see the README's mapping).
- The campaign's job-id caches (`best_20x*_jobs.json`) and npz slices were not found on disk; the
  learned models would have to be re-fetched from the IBM jobs named in the logs under
  `qed_pec_artifacts/data/`.
- Candidate upstream issue: the `executor_expectation_values` variance (see Error bars).
