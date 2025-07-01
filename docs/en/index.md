# Welcome to Qamomile

Qamomile is a powerful SDK designed for quantum optimization algorithms, specializing in the conversion of mathematical models into quantum circuits. It serves as a bridge between classical optimization problems and quantum computing solutions.

## Key Features

- **Versatile Compatibility**: Supports leading quantum circuit SDKs including Qiskit and Quri-parts.
- **Advanced Algorithm Support**: Goes beyond QAOA to include sophisticated encoding and algorithms like QRAO.
- **Flexible Model Conversion**: Utilizes JijModeling for describing mathematical models and converting them to various quantum circuit SDKs.
- **Intermediate Representation**: Capable of representing both Hamiltonians and quantum circuits as intermediate forms.
- **Standalone Functionality**: Can implement quantum circuits independently, similar to other quantum circuit SDKs.

## Quick Start

To get started with Qamomile, please see the [Quick Start Guide](quickstart.ipynb) for installation instructions and a simple example.

## Learn More

Before diving into each documentation, we'd like to clarify the bit representation convention. Qamomile considers the first qubit as the least significant qubit, which is specified as `0`. To align with this convention, we regard the first element of a list representing classical bits as the least significant bit: `bits[0]`. These conventions are quite simple, yet can cause confusion if we don't keep this rule in mind.

Explore our documentation to dive deeper into Qamomile's capabilities:

- [Quick Start Guide](quickstart.ipynb): Installation instructions and a simple example to get you started.
- [API Reference](api_index.md): Complete documentation of Qamomile's API.
- Tutorials: Step-by-step guides and examples to get you started.
    - [Basic Usage of the Library](tutorial/usage/index_usage.md)
    - [Solving Problems with QAOA](tutorial/qaoa/index_qaoa.md)
    - [Advanced techniques for Quantum Optimization](tutorial/opt_advance/index_advance.md)
    - [Quantum Chemistry](tutorial/chemistry/index_chemistry.md)

## Contributing

We welcome contributions from the community! If you're interested in improving Qamomile, please check out our [Contribution Guidelines](contribute.md).

## Support

If you encounter any issues or have questions, please file an issue on our [GitHub repository](https://github.com/Jij-Inc/Qamomile) or join our community discussion forum.

Welcome aboard, and happy quantum optimizing with Qamomile!