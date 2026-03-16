# Qamomile

> [!WARNING]
> This repository tracks an actively developed version of Qamomile.
> The version currently available on PyPI is not the same as this branch.
> APIs in this branch may still change, including breaking changes, while active development continues.

Qamomile is a typed quantum programming SDK for writing quantum kernels in Python, inspecting them as Qamomile IR, estimating resources symbolically, and transpiling them to concrete execution backends such as Qiskit and QURI Parts.

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
In the current `pyproject.toml`, that means you get the Qiskit stack, documentation and test tooling, and QURI Parts-related packages as well.

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

If you prefer an explicit editable install inside your environment, this also works from the cloned repository:

```bash
pip install -e .
pip install -e ".[quri_parts]"  # optional
pip install -e ".[qbraid]"      # optional
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
- `qamomile.qbraid`: optional qBraid executor support for running Qiskit circuits on qBraid-supported devices
- `qamomile.quri_parts`: optional QURI Parts transpiler and executor support
- `qamomile.optimization`: optimization-oriented functionality retained for continuity with older Qamomile workflows

## Optimization Support

Qamomile still supports the optimization-oriented workflow that older versions focused on.
That functionality lives under `qamomile.optimization`, including QAOA, FQAOA, and QRAO-related modules.
This README focuses on the current circuit-first API, but optimization support remains part of the project.

## Learn More

- Documentation: [https://jij-inc.github.io/Qamomile/](https://jij-inc.github.io/Qamomile/)
- Tutorials: [docs/en/index.md](docs/en/index.md) and [docs/ja/index.md](docs/ja/index.md)
- API reference: [https://jij-inc.github.io/Qamomile/api/](https://jij-inc.github.io/Qamomile/api/)
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
