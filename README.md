# Qamomile

[![PyPI version](https://badge.fury.io/py/qamomile.svg)](https://badge.fury.io/py/qamomile)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

Qamomile is a powerful SDK designed for quantum optimization algorithms, specializing in the conversion of mathematical models into quantum circuits. It serves as a bridge between classical optimization problems and quantum computing solutions.

Documentation: [https://jij-inc.github.io/Qamomile/](https://jij-inc.github.io/Qamomile/)

LP: [https://jij-inc.github.io/Qamomile/landing.html](https://jij-inc.github.io/Qamomile/landing.html)

## Features

- **Versatile Compatibility**: Supports leading quantum circuit SDKs including Qiskit and QuriParts.
- **Advanced Algorithm Support**: Implements sophisticated encoding and algorithms like QAOA and QRAO.
- **Flexible Model Conversion**: Utilizes JijModeling for describing mathematical models and converting them to various quantum circuit SDKs.
- **Intermediate Representation**: Capable of representing both Hamiltonians and quantum circuits as intermediate forms.
- **Standalone Functionality**: Can implement quantum circuits independently, similar to other quantum circuit SDKs.

## Installation

To install Qamomile, use pip:

```bash
pip install qamomile
```

For optional dependencies:

```bash
pip install "qamomile[qiskit]"  # For Qiskit integration
pip install "qamomile[quri-parts]"  # For QuriParts integration
pip install "qamomile[pennylane]"  # For QuriParts integration
pip install "qamomile[qutip]"  # For QuTiP integration
pip install "qamomile[cudaq]"  # For CUDA-Q integration
pip install "qamomile[udm]"  # For bloqade-analog integration
```

Note that, CUDA-Q is currently supported only on Linux (see https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility).

## Quick Start

Here's a simple example of how to use Qamomile with QAOA:

```python
import jijmodeling as jm
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.qiskit.transpiler import QiskitTranspiler

# Define QUBO problem
Q = jm.Placeholder("Q", ndim=2)
n = Q.len_at(0, latex="n")
x = jm.BinaryVar("x", shape=(n,))
problem = jm.Problem("qubo")
i, j = jm.Element("i", n), jm.Element("j", n)
problem += jm.sum([i, j], Q[i, j] * x[i] * x[j])

# Prepare instance data
instance_data = {"Q": [[0.1, 0.2, -0.1], [0.2, 0.3, 0.4], [-0.1, 0.4, 0.0]]}

# Have an intermediate representation of the problem with the instance data substituted
interpreter = jm.Interpreter(instance_data)
compiled_instance = interpreter.eval_problem(problem)

# Create QAOA converter
qaoa_converter = QAOAConverter(compiled_instance)

# Create Qiskit transpiler
qiskit_transpiler = QiskitTranspiler()

# Get QAOA circuit
p = 2  # Number of QAOA layers
qaoa_circuit = qaoa_converter.get_qaoa_ansatz(p)

# Convert to Qiskit circuit
qiskit_circuit = qiskit_transpiler.transpile_circuit(qaoa_circuit)

# ... (continue with quantum execution and result processing)
```

## Documentation

For more detailed information, please refer to our [documentation](https://jij-inc.github.io/Qamomile/).

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contribute.md) for more details.

## License

Qamomile is released under the [Apache 2.0 License](LICENSE).
