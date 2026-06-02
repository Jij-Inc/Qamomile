# Welcome to Qamomile Documentation

**Qamomile** (pronounced /ˈkæməˌmiːl/, like "chamomile") is named after the chamomile flower — a symbol of calm and clarity.

Qamomile is a quantum programming SDK. Write quantum circuits as typed Python functions and run them on quantum SDKs like Qiskit, CUDA-Q, QURI Parts, and qBraid. Furthermore, Qamomile supports symbolic algebraic resource estimation and can estimate resources for circuits containing black-box oracles — even when the circuit itself cannot be executed.

> **Note** Qamomile is under active development, and breaking changes may be introduced between releases. If you find a bug, we'd really appreciate it if you could let us know via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new).

## Installation

```bash
pip install qamomile
```

## Supported Quantum SDKs

Qamomile supports multiple quantum SDKs as execution backends. Qiskit is included by default; the others are optional extras. The first three (Qiskit, CUDA-Q, QURI Parts) each ship their own `Transpiler` + `Executor` pair, so you can swap between them by switching the import below. qBraid is structured differently — it's an execution-only adapter that runs a Qiskit-transpiled circuit on qBraid-supported devices, so it has its own section.

::::{tab-set}
:::{tab-item} Qiskit (default)
:sync: qiskit

Included with `pip install qamomile`. No extra flags needed.

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```
:::

:::{tab-item} CUDA-Q
:sync: cudaq

CUDA-Q supports Linux and macOS ARM64 (Apple Silicon). Choose the extra that matches your CUDA version:

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x, Linux
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x, Linux or macOS ARM64
```

```python
from qamomile.cudaq import CudaqTranspiler, CudaqExecutor
```
:::

:::{tab-item} QURI Parts
:sync: quri_parts

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import QuriPartsTranspiler, QuriPartsExecutor
```
:::
::::

### qBraid (optional, execution-only)

qBraid is an execution-only adapter — there is no `QBraidTranspiler`. The
workflow is to transpile through `QiskitTranspiler` first, then dispatch
the resulting Qiskit circuit to a qBraid-supported device or simulator
via `QBraidExecutor`:

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qiskit import QiskitTranspiler
from qamomile.qbraid import QBraidExecutor
```

See [integration/qbraid_executor](integration/qbraid_executor.ipynb)
for the full walk-through, including how to wire your qBraid API key.

## Sections

- [Algorithms](algorithm/index.md) — Concrete walkthroughs of variational and quantum algorithms (QAOA, VQE, …) end-to-end with Qamomile.
- [Usage](usage/index.md) — How-to guides for individual Qamomile modules (`BinaryModel`, future module-walkthroughs).
- [Integration](integration/index.md) — Notes on using Qamomile with external libraries and quantum platforms (qBraid, …).
- [Release Notes](release_notes/index.md) — Per-version changelog with feature highlights and breaking changes.

For SDK-level fundamentals (kernels, parameters, execution, transpilation, QEC primitives), see the [Tutorials](tutorial/index.md) section.

## Links

- [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
- [API Reference](api/index.md)
