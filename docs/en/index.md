# Welcome to Qamomile Documentation

**Qamomile** (pronounced /ˈkæməˌmiːl/, like "chamomile") is named after the chamomile flower — a symbol of calm and clarity.

:::{note}
Qamomile is under active development, and breaking changes may be introduced between releases.
If you find a bug, we'd really appreciate it if you could let us know via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new).
:::

## What is Qamomile?

Qamomile is a source-compiled, type-safe functional quantum programming language embedded in Python, an embedded domain-specific language (eDSL). The same source supports resource estimation and execution across supported quantum SDKs.

## Motivations

::::{grid} 1 1 3 3

:::{card}
:header: Resource estimation

As quantum hardware improves, we need to assess which problem sizes it can handle and when algorithms might become practical for a given problem size. Estimating the required qubits and gates is therefore as important as running algorithms on hardware.
:::

:::{card}
:header: Portability

We also aim for "write once, run anywhere": you can choose a quantum SDK for its strengths without rewriting your program.
:::

:::{card}
:header: Higher-level programming

Higher-level language constructs and reusable algorithms help you express quantum programs beyond low-level circuits and reduce the expertise needed to build applications.
:::

::::

## Key features

::::{grid} 1 1 3 3
:class: qamomile-feature-cards

:::{card}
:header: Type safety

Qamomile's linear type system requires each quantum state value to be consumed exactly once. It catches invalid reuse and unconsumed states before execution.
:::

:::{card}
:header: Built-in modules

Reuse common operation patterns and well-known algorithms without writing complex logic and boilerplate from scratch.
:::

:::{card}
:header: Algorithm-level resource estimation

Estimate qubits and gates symbolically to assess implementations and identify bottlenecks. Oracles, black-box operations with specified resource requirements, let you estimate a program before every operation is implemented. Execution requires implementations for the oracles.
:::

:::{card}
:header: Portable quantum programs

`qm.compile` converts your programs to supported targets such as Qiskit, CUDA-Q, and HUGR. Use `qm.run` to execute the source or compiled program on a compatible execution endpoint.
:::

:::{card}
:header: One source for estimation and execution

Use the same Qamomile program for both tasks without maintaining separate implementations.
:::

:::{card}
:header: Optimization modules

Run quantum optimization algorithms with OMMX, a solver-independent data format for optimization problems. OMMX provides adapters for classical and quantum solvers, so you can compare their results with Qamomile's algorithms.
:::

::::

---

## Quick start

Install Qamomile from pip. The distribution is named `qamomile`; the new language module is `qamomile`.

```bash
pip install qamomile
```

The following example estimates and runs the same Bell program. For details, see [Your First Qamomile Function](tutorial/01_your_first_quantum_kernel.ipynb).

```python
import qamomile as qm


@qm.qkernel
def bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


study = qm.resources.logical(bell)
print(study.metric(qm.resources.LOGICAL_GATES))

record = qm.run(bell, on=qm.Reference(), shots=16, seed=7)
print(record.result.counts)
```

---

## Get started

::::{grid} 1 2 2 2

:::{card}
:header: **Installation**
:link: installation_guide.md
Install Qamomile and add Qiskit, Cirq, QURI Parts, CUDA-Q, Braket, or HUGR only when needed.
:::

:::{card}
:header: **Tutorials**
:link: tutorial/index.md
Learn typed functions, functional quantum values, parameters, resources, execution, and target compilation.
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
Target-specific guides for Qiskit, Cirq, QURI Parts, CUDA-Q, Amazon Braket, and HUGR use the same Qamomile lifecycle.
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
Reference for the public `qamomile` package.
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
