# Welcome to Qamomile Documentation

**Qamomile** (pronounced /ˈkæməˌmiːl/, like "chamomile") is named after the chamomile flower — a symbol of calm and clarity.

:::{note}
Qamomile is under active development, and breaking changes may be introduced between releases.
If you find a bug, we'd really appreciate it if you could let us know via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new).
:::

## What is Qamomile?

Qamomile is a quantum programming SDK for writing quantum circuits as typed Python functions. You can run the circuits you build on quantum SDKs such as Qiskit, CUDA-Q, QURI Parts, and qBraid. Furthermore, Qamomile supports symbolic algebraic resource estimation and can estimate resources for circuits containing black-box oracles, even when those circuits cannot be executed directly.

::::{grid} 1 1 3 3

:::{card}
**Quantum circuits in Python**

Write circuits as typed Python functions, then draw, check, and reuse them like regular code.
:::

:::{card}
**One program, many quantum SDKs**

Move between Qiskit, CUDA-Q, QURI Parts, and qBraid without rewriting your circuit.
:::

:::{card}
**Made for optimization**

Connect QUBO and Ising models to quantum algorithms, then implement and run optimization workflows end to end.
:::

::::

---

## Quick start

Qamomile can be installed from pip.

```bash
pip install "qamomile[qiskit,visualization]"
```

The following example runs a quantum algorithm with Qamomile. For details, see [Your First Quantum Kernel](tutorial/01_your_first_quantum_kernel.ipynb).

```python
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def biased_coin(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


# Visualize the kernel and estimate resources before execution
biased_coin.draw(theta=0.6)
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# Transpile while keeping theta available as a runtime parameter
transpiler = QiskitTranspiler()
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# Execute with a concrete value for theta
result = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
).result()

print(result.results)
```

---

## Get started

::::{grid} 1 2 2 2

:::{card}
:header: **Installation**
:link: installation_guide.md
Install Qamomile itself, then add quantum SDK integrations such as CUDA-Q, QURI Parts, and qBraid as needed.
:::

:::{card}
:header: **Tutorials**
:link: tutorial/index.md
Learn the basics of Qamomile step by step, including kernels, parameters, execution, and transpilation.
:::

::::

---

## Use Qamomile

::::{grid} 1 2 2 2

:::{card}
:header: **Algorithms**
:link: algorithm/index.md
Practical guides for implementing and running quantum algorithms such as QAOA and VQE with Qamomile.
:::

:::{card}
:header: **Usage**
:link: usage/index.md
Task-focused guides for individual Qamomile modules, such as `BinaryModel`.
:::

:::{card}
:header: **Integration**
:link: integration/index.md
Guides for using Qamomile with quantum platforms and external libraries, such as qBraid.
:::

::::

---

## Reference

::::{grid} 1 2 2 2

:::{card}
:header: **Release Notes**
:link: release_notes/index.md
Per-version changelog with feature highlights and breaking changes.
:::

:::{card}
:header: **API Reference**
:link: api/index.md
How to use the APIs provided by Qamomile, including their arguments and return values.
:::

::::

---

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

---

## Links

- [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
