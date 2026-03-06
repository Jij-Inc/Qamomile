# Welcome to Qamomile Documentation

Qamomile is a powerful SDK for quantum optimization algorithms, specializing in the conversion of mathematical models into quantum circuits.

## Tutorials

### Foundations

- [Introduction to Qamomile](tutorial/01_introduction) - First circuit, linear types, execution with QiskitTranspiler
- [Type System](tutorial/02_type_system) - Full type catalog: Qubit, Float, UInt, Bit, Vector, Dict
- [Quantum Gates](tutorial/03_gates) - Complete gate reference (all 11 gates)
- [Superposition & Entanglement](tutorial/04_superposition_entanglement) - Superposition, interference, Bell/GHZ states

### Standard Library & Algorithms

- [Standard Library](tutorial/05_stdlib) - QFT, IQFT, QPE, and the algorithm module
- [Composite Gates](tutorial/06_composite_gate) - CompositeGate, `@composite_gate`, stub gates
- [First Algorithm: Deutsch-Jozsa](tutorial/07_first_algorithm) - Oracle pattern, quantum parallelism, and interference
- [Parametric Circuits & VQA](tutorial/08_parametric_circuits) - bindings vs parameters, Observable, expval, variational classifier

### Advanced Topics

- [Resource Estimation](tutorial/09_resource_estimation) - Algebraic gate counts and circuit depth as SymPy expressions
- [Transpiler Internals](tutorial/10_transpile) - The full transpiler pipeline from @qkernel to executable
- [Custom Executor](tutorial/11_custom_executor) - Connecting to cloud quantum backends (IBM Quantum, etc.)

## Optimization

- [QAOA](optimization/qaoa) - Solving Max-Cut with Quantum Approximate Optimization Algorithm
- [FQAOA](optimization/fqaoa) - Constrained optimization with Fermionic QAOA
- [QRAO](optimization/qrao31) - Qubit-efficient encoding with Quantum Random Access Optimization
- [Custom Converter](optimization/custom_converter) - Building your own optimization converter

## Installation

```bash
pip install qamomile
```

## Quick Example

```python
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

@qmc.qkernel
def bell_state(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Bit, qmc.Bit]:
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return qmc.measure(q0), qmc.measure(q1)

transpiler = QiskitTranspiler()
executor = transpiler.executor()
executable = transpiler.compile(bell_state)
job = executable.sample(executor, shots=1000)
result = job.result()
print(f"Counts: {result.counts}")
```

## Error Handling

`@qkernel` errors are raised in two stages:

- **AST transform stage**: Unsupported control-flow patterns (e.g. direct sequence iteration, quantum operations in `while` conditions) are rejected with `SyntaxError`.
- **Transpiler / backend stage**: Type violations, linear-type errors, and backend-specific issues use the `QamomileCompileError` family.

When catching errors from `@qkernel`, include both `SyntaxError` and `QamomileCompileError` in your exception handling.

## Links

- [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
- [API Reference](api/index.md)
