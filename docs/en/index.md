# Welcome to Qamomile Documentation

**Qamomile** (pronounced /ˈkæməˌmiːl/, like "chamomile") is named after the chamomile flower — a symbol of calm and clarity.

Qamomile is a quantum programming SDK. Write quantum circuits as typed Python functions and run them on quantum SDKs like Qiskit, CUDA-Q, QURI Parts, and qBraid. Furthermore, Qamomile supports symbolic algebraic resource estimation and can estimate resources for circuits containing black-box oracles — even when the circuit itself cannot be executed.

> **Note** Qamomile is under active development, and breaking changes may be introduced between releases. If you find a bug, we'd really appreciate it if you could let us know via [GitHub Issues](https://github.com/Jij-Inc/Qamomile/issues/new).

## Tutorials

1. [Your First Quantum Kernel](tutorial/01_your_first_quantum_kernel) — Define, visualize, and execute a kernel; the affine rule
2. [Parameterized Kernels](tutorial/02_parameterized_kernels) — Structure vs runtime parameters, bind/sweep pattern
3. [Resource Estimation](tutorial/03_resource_estimation) — Symbolic cost analysis, gate breakdown, scaling
4. [Execution Models](tutorial/04_execution_models) — `sample()` vs `run()`, observables, bit ordering
5. [Classical Flow Patterns](tutorial/05_classical_flow_patterns) — Loops, sparse data, conditional branching
6. [Reuse Patterns](tutorial/06_reuse_patterns) — Helper kernels, composite gates, stubs

## Algorithms

Concrete algorithm examples — [browse by tag](algorithm/index.md):

- [QAOA for MaxCut](algorithm/qaoa_maxcut) — Build QAOA from scratch to solve MaxCut
- [QAOA for Graph Partitioning](algorithm/qaoa_graph_partition) — End-to-end with JijModeling and `QAOAConverter`
- [Hamiltonian Simulation with Suzuki–Trotter](algorithm/hamiltonian_simulation) — Rabi oscillation and convergence orders
- [From a Hermitian Matrix to a Quantum Circuit](algorithm/hermitian_decomposition) — Dense Hermitian matrix to a Pauli sum and time-evolution circuit
- [VQE for the Hydrogen Molecule](algorithm/vqe_for_hydrogen) — Build a molecular Hamiltonian with OpenFermion and find the ground state energy using VQE

## Optimization

- [How to use `BinaryModel`](optimization/binary_model) — Build unconstrained binary/spin models from BinaryExpr, QUBO/HUBO/Ising, or OMMX

## Supported Quantum SDKs

Qamomile supports multiple quantum SDKs as execution backends. Qiskit is included by default; the others are optional extras.

### Qiskit (default)

Included with `pip install qamomile`. No extra flags needed.

```python
from qamomile.qiskit import QiskitTranspiler, QiskitExecutor
```

### CUDA-Q (optional)

CUDA-Q supports Linux and macOS ARM64 (Apple Silicon). Choose the extra that matches your CUDA version:

```bash
pip install "qamomile[cudaq-cu12]"   # CUDA 12.x, Linux
pip install "qamomile[cudaq-cu13]"   # CUDA 13.x, Linux or macOS ARM64
```

```python
from qamomile.cudaq import CudaqTranspiler, CudaqExecutor
```

### QURI Parts (optional)

```bash
pip install "qamomile[quri_parts]"
```

```python
from qamomile.quri_parts import QuriPartsTranspiler, QuriPartsExecutor
```

### qBraid (optional)

Runs Qiskit circuits on qBraid-supported devices and simulators.

```bash
pip install "qamomile[qbraid]"
```

```python
from qamomile.qbraid import QBraidExecutor
```

## Installation

```bash
pip install qamomile
```

## Quick Example

```python
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

@qmc.qkernel
def bell_state() -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)

transpiler = QiskitTranspiler()
exe = transpiler.transpile(bell_state)
result = exe.sample(transpiler.executor(), shots=1000).result()

for outcome, count in result.results:
    print(f"  {outcome}: {count}")
```

## Links

- [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
- [API Reference](api/index.md)
