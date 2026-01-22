# Welcome to Qamomile Documentation

Qamomile is a powerful SDK for quantum optimization algorithms, specializing in the conversion of mathematical models into quantum circuits.

## Getting Started

Explore our tutorials to learn how to use Qamomile:

- [Transpile & Execute](transpile/transpile_flow) - Learn how to write quantum kernels and execute them on quantum simulators/hardware

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

## Links

- [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
- [API Reference](https://jij-inc.github.io/Qamomile/en/)
