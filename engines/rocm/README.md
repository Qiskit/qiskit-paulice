# qiskit-paulice-rocm

A scoring engine for [qiskit-paulice](https://github.com/Qiskit/qiskit-paulice) that
scores candidate checks for `cost="gamma"` and `cost="LER"` on an AMD GPU. It is enabled
through [Coheriq](https://github.com/Qiskit/coheriq) under the name `rocm`.

## Supported devices

The kernels need a GPU with 64-wide wavefronts, such as the AMD Instinct accelerators.

- Validated on hardware: AMD Instinct MI355X (`gfx950`).
- Built and loaded without a device in CI: `gfx950` and `gfx942`.
- Other 64-wide targets can be compiled with `PAULICE_ROCM_ARCH`, but they are untested.

On a GPU with 32-wide wavefronts (for example an RDNA-based Radeon), or with no GPU at
all, the engine opens no device session and qiskit-paulice scores everything on the CPU.

## Build and install

The engine is not published as a wheel. It builds from a qiskit-paulice checkout,
against `include/paulice_engine.h` from the same checkout, and is installed next to the
qiskit-paulice built from that checkout. It needs CMake 3.21 or later and a ROCm install
with the HIP runtime and rocPRIM. From the root of the checkout:

```sh
pip install .
pip install ./engines/rocm
```

| Variable | Default | Effect |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` | ROCm install to build against |
| `PAULICE_ROCM_ARCH` | `gfx950` | Comma-separated GPU targets to compile for |

qiskit-paulice refuses an engine built for another version of its engine ABI, so
after updating the checkout, rebuild both.

## Use

Enable the engine before the first call to `add_pauli_checks`, either with
`PAULICE_ENGINE=rocm` or in code:

```python
import coheriq
import qiskit_paulice

coheriq.enable_engine(qiskit_paulice, "rocm")
```

| Variable | Default | Effect |
|---|---|---|
| `PAULICE_ENGINE_DEVICE` | `0` | Index of the GPU to open |
| `PAULICE_ROCM_PROFILE` | unset | `1` prints the time of each device stage to standard error |

## What runs on the CPU instead

The engine returns the same costs as the CPU, bit for bit, and anything it cannot score
that way runs on the CPU as before. qiskit-paulice keeps `cost="gamma"` searches with
layered or gate-wise noise models, or with stabilizers, on the CPU for every engine.
This engine also declines work past limits of its own kernels, defined in
`src/paulice_rocm.h`:

| Limit | Value | Why |
|---|---|---|
| `cost="gamma"` circuit width | 256 qubits | Propagation keeps a row's X and Z words in registers, specialised on 1 to 4 words |
| `cost="LER"` packed scoring row | 1024 words | One wavefront holds a shot's accumulator in registers |
| `cost="LER"` error generators | 65535 | The transpose uses the generator as its grid's y index |

It also declines any work whose device call fails.

For `cost="gamma"` the engine returns summed fault rates and the CPU applies `exp`, so
the device's maths library never enters a score. For `cost="LER"` it draws from the same
random streams and Poisson tables as the CPU and counts shots as integers. The
qiskit-paulice installation guide describes what is reproducible across platforms.

## Check the engine on a GPU

Conformance compares the engine with the CPU, bit for bit, in 14 checks:

```sh
PAULICE_ENGINE=rocm python -c "
from qiskit_paulice import acceleration
for name, outcome, detail in acceleration.run_engine_conformance():
    print(outcome, name, detail)
"
```

Every check should report `passed`. Without a usable device every check reports
`skipped`, which is also what the CI workflow `engine_rocm.yml` checks. To run the full
qiskit-paulice test suite through the engine:

```sh
PAULICE_ENGINE=rocm pytest test
```
