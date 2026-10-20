# Welcome to Qamomile Documentation

**Qamomile** (pronounced /ˈkæməˌmiːl/, like "chamomile") is named after the chamomile flower — a symbol of calm and clarity.

:::{note}
Qamomile is under active development, and breaking changes may be introduced between releases.
If you find a bug, we'd really appreciate it if you could let us know via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new).
:::

## What is Qamomile?

Qamomile is a quantum programming language embedded in Python, an embedded domain-specific language (eDSL). The same source supports resource estimation and execution across supported quantum SDKs.

## Motivations

As quantum hardware improves, we need to assess which problem sizes it can handle and when algorithms might become practical for a given problem size. Estimating the required qubits and gates is therefore as important as running algorithms on hardware.

We also aim for "write once, run anywhere": you can choose a quantum SDK for its strengths without rewriting your program. Higher-level language constructs and reusable algorithms help you express quantum programs beyond low-level circuits and reduce the expertise needed to build applications.

Existing SDKs address parts of these needs. Qamomile brings resource estimation, portability, and higher-level programming together in one quantum programming language.

## Key features

- **Type safety.** Qamomile's affine type system allows each quantum value to be consumed at most once. It catches invalid reuse before execution.
- **Built-in modules.** Reuse common operation patterns and well-known algorithms without writing complex logic and boilerplate from scratch.
- **Algorithm-level resource estimation.** Estimate qubits and gates symbolically to assess implementations and identify bottlenecks. Oracles, black-box operations with specified resource requirements, let you estimate a program before every operation is implemented. Execution requires implementations for the oracles.
- **Portable quantum programs.** Transpilers convert your programs to supported targets such as Qiskit, CUDA-Q, and HUGR. The corresponding executors run the converted programs through Qamomile.
- **One source for estimation and execution.** Use the same Qamomile program for both tasks without maintaining separate implementations.
- **Optimization modules.** Run quantum optimization algorithms with OMMX, a solver-independent data format for optimization problems. OMMX provides adapters for classical and quantum solvers, so you can compare their results with Qamomile's algorithms.

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
