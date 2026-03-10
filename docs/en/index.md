# Welcome to Qamomile Documentation

Qamomile is a quantum programming SDK. Write quantum circuits as typed Python functions and run them on quantum SDKs like Qiskit and QuriParts.

## Tutorials

1. [Your First Quantum Kernel](tutorial/01_your_first_quantum_kernel) — Define, visualize, and execute a kernel; the affine rule
2. [Parameterized Kernels](tutorial/02_parameterized_kernels) — Structure vs runtime parameters, bind/sweep pattern
3. [Resource Estimation](tutorial/03_resource_estimation) — Symbolic cost analysis, gate breakdown, scaling
4. [Execution Models](tutorial/04_execution_models) — `sample()` vs `run()`, observables, bit ordering
5. [Classical Flow Patterns](tutorial/05_classical_flow_patterns) — Loops, sparse data, conditional branching
6. [Reuse Patterns](tutorial/06_reuse_patterns) — Helper kernels, composite gates, stubs

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
