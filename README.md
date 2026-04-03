# Qamomile

[![PyPI version](https://badge.fury.io/py/qamomile.svg)](https://badge.fury.io/py/qamomile)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

> [!WARNING]
> This repository tracks an actively developed version of Qamomile.
> The version currently available on PyPI is not the same as this branch.
> APIs in this branch may still change, including breaking changes, while active development continues.

Qamomile is a typed quantum programming SDK for writing quantum kernels in Python, inspecting them as Qamomile IR, estimating resources symbolically, and transpiling them to concrete execution quantum SDK such as Qiskit, QURI Parts, CUDA-Q. Furthremore, as a backend for Qiskit, we support qBraid.

The current workflow is:

```text
@qkernel define -> draw() / estimate_resources() -> transpile() -> sample() / run() -> .result()
```

## Why Qamomile?

- Write quantum programs as typed Python functions with `@qkernel`.
- Use typed handles such as `Qubit`, `Bit`, `Float`, `UInt`, and `Observable`.
- Inspect kernels before execution with `draw()` and `estimate_resources()`.
- Build parameterized circuits and reuse a transpiled executable with different runtime bindings.
- Run measured programs with `sample()` and expectation-value programs with `run()`.
- Express circuit structure with classical control flow such as `qmc.range()`, `qmc.items()`, `if`, and `while`.
- Reuse circuit logic with helper kernels and `@composite_gate`.

## Installation

This README describes the current source tree, not the older PyPI release.
If you want this version, install from source.

Requirements:

- Python 3.12+
- `uv`

Clone the repository:

```bash
git clone https://github.com/Jij-Inc/Qamomile.git
cd Qamomile
```

Choose the installation style that matches your use case.

Full development environment:

```bash
uv sync
```

This installs the default development dependency group.
In the current `pyproject.toml`, that gives you the core Qiskit-based environment together with documentation and test tooling.
Optional backend integrations such as QURI Parts, qBraid, and CUDA-Q still need their corresponding extras.

Runtime-only environment from source:

```bash
uv sync --no-dev
```

Runtime-only environment from source with QURI Parts support:

```bash
uv sync --no-dev --extra quri_parts
```

Runtime-only environment from source with qBraid support:

```bash
uv sync --no-dev --extra qbraid
```

Runtime-only environment from source with CUDA-Q v0.14.0 support:

```bash
uv sync --no-dev --extra cudaq-cu12   # for CUDA 12.x
uv sync --no-dev --extra cudaq-cu13   # for CUDA 13.x (or MacOS)
```

CUDA-Q v0.14.0 currently supports Linux, macOS ARM64 (Apple Silicon), and Windows via WSL2. For MacOS, please use `cudaq-cu13`.

> [!NOTE]
> **Why `cudaq-cu12` / `cudaq-cu13` instead of `cudaq`?**
>
> The upstream `cudaq` meta-package provides only an sdist whose `setup.py` dynamically computes `install_requires`.
> This causes `uv pip install cudaq` to silently install the package without its dependencies on the first attempt
> ([astral-sh/uv#12759](https://github.com/astral-sh/uv/issues/12759),
> [NVIDIA/cuda-quantum#3616](https://github.com/NVIDIA/cuda-quantum/issues/3616)).
> To avoid this issue, Qamomile specifies the concrete wheel packages `cuda-quantum-cu12` / `cuda-quantum-cu13` directly
> as optional dependencies, split by CUDA version.

If you prefer an explicit editable install inside your environment, this also works from the cloned repository:

```bash
pip install -e .
pip install -e ".[quri_parts]"   # optional
pip install -e ".[qbraid]"       # optional
pip install -e ".[cudaq-cu12]"   # optional, CUDA 12.x
pip install -e ".[cudaq-cu13]"   # optional, CUDA 13.x
```

If you intentionally want the latest published release instead, `pip install qamomile` installs the PyPI package, not this work-in-progress branch.

## Quick Start

```python
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def biased_coin(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


# Inspect the kernel before execution
biased_coin.draw(theta=0.6)
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# Transpile once, keep theta as a runtime parameter
transpiler = QiskitTranspiler()
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# Execute with a concrete binding
result = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
).result()

print(result.results)
```

If a kernel returns measured bits, use `sample()`.
If it returns a `qmc.Float` from `qmc.expval(...)`, use `run()` instead.

## Main Packages

- `qamomile.circuit`: the main entry point for typed quantum kernels, gates, control flow, drawing, and resource estimation
- `qamomile.observable`: Hamiltonians and Pauli observables used with expectation-value workflows
- `qamomile.qiskit`: Qiskit transpiler and executor support
- `qamomile.cudaq`: optional CUDA-Q transpiler, executor, and observable conversion (supports both static sampling and runtime control-flow modes)
- `qamomile.qbraid`: optional qBraid executor support for running Qiskit circuits on qBraid-supported devices
- `qamomile.quri_parts`: optional QURI Parts transpiler and executor support
- `qamomile.optimization`: optimization-oriented functionality retained for continuity with older Qamomile workflows

## Optimization Support

Qamomile still supports the optimization-oriented workflow that older versions focused on.
That functionality lives under `qamomile.optimization`, including QAOA, FQAOA, and QRAO-related modules.
This README focuses on the current circuit-first API, but optimization support remains part of the project.

## Learn More

- Documentation: [English](https://jij-inc-qamomile.readthedocs-hosted.com/latest/en/) and [Japanese](https://jij-inc-qamomile.readthedocs-hosted.com/latest/ja/)
- Tutorials: [English](https://jij-inc-qamomile.readthedocs-hosted.com/latest/en/tutorial/) and [Japanese](https://jij-inc-qamomile.readthedocs-hosted.com/latest/ja/tutorial/)
- API reference: https://jij-inc-qamomile.readthedocs-hosted.com/latest/en/api/
- Repository: [https://github.com/Jij-Inc/Qamomile](https://github.com/Jij-Inc/Qamomile)

## Contributing

Contributions, bug reports, and feedback are welcome via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues) and pull requests.

## License

Qamomile is released under the [Apache 2.0 License](LICENSE.txt).

## Citation

If you use Qamomile in your research, please cite:

```bibtex
@INPROCEEDINGS{11249901,
  author={Huang, Wei-Hao and Matsuyama, Hiromichi and Tam, Wai-Hong and Sato, Keisuke and Yamashiro, Yu},
  booktitle={2025 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  title={Qamomile: A Cross-SDK Bridge for Quantum Optimization},
  year={2025},
  volume={02},
  pages={516-517},
  doi={10.1109/QCE65121.2025.10423}
}
```
