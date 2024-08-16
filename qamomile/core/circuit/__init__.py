"""
Quantum Circuit Package

This package provides tools for creating and manipulating quantum circuits,
including parameter expressions and various quantum gates.

It includes:
- Parameter expressions for defining parameterized quantum operations
- Quantum gate definitions (single-qubit, two-qubit, three-qubit, and parametric gates)
- Quantum circuit class for building and manipulating quantum circuits

Usage:
    from qamomile.core.circuit import QuantumCircuit, Parameter
    qc = QuantumCircuit(2)
    theta = Parameter('theta')
    qc.rx(theta, 0)
    qc.cnot(0, 1)
"""

# Import main classes and functions from parameter.py
from .parameter import (
    Parameter,
    ParameterExpression,
    BinaryOperator,
    Value,
    BinaryOpeKind,
)

# Import main classes and functions from circuit.py
from .circuit import (
    QuantumCircuit,
    Gate,
    SingleQubitGate,
    ParametricSingleQubitGate,
    TwoQubitGate,
    ThreeQubitGate,
    Operator,
    SingleQubitGateType,
    ParametricSingleQubitGateType,
    TwoQubitGateType,
    ThreeQubitGateType,
    ParametricTwoQubitGate,
    ParametricTwoQubitGateType,
    MeasurementGate,
)


# Define what should be imported when using "from circuit import *"
__all__ = [
    "Parameter",
    "ParameterExpression",
    "QuantumCircuit",
    "Gate",
    "SingleQubitGate",
    "ParametricSingleQubitGate",
    "TwoQubitGate",
    "ThreeQubitGate",
    "Operator",
    "SingleQubitGateType",
    "ParametricSingleQubitGateType",
    "TwoQubitGateType",
    "ThreeQubitGateType",
    "MeasurementGate",
    "BinaryOperator",
    "Value",
    "BinaryOpeKind",
    "ParametricTwoQubitGate",
    "ParametricTwoQubitGateType",
]
