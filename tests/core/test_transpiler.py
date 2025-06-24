import pytest

from tests.mock import (
    InvalidTranspilerNoConvertResult,
    InvalidTranspilerNoTranspileCircuit,
    InvalidTranspilerNoTranspileHamiltonian,
    ValidTranspilerWithAllImplementations,
)


def test_creation_without_convert_result_implementation():
    """Create an instance of the child class of QuantumSDKTranspiler without implmenting convert_result.

    Check if
    1. NotImplementedError is raised.
    """
    with pytest.raises(TypeError):
        InvalidTranspilerNoConvertResult()


def test_creation_without_transpile_circuit_implementation():
    """Create an instance of the child class of QuantumSDKTranspiler without implmenting transpile_circuit.

    Check if
    1. NotImplementedError is raised.
    """
    with pytest.raises(TypeError):
        InvalidTranspilerNoTranspileCircuit()


def test_creation_without_transpile_hamiltonian_implementation():
    """Create an instance of the child class of QuantumSDKTranspiler without implmenting transpile_hamiltonian.

    Check if
    1. NotImplementedError is raised.
    """
    with pytest.raises(TypeError):
        InvalidTranspilerNoTranspileHamiltonian()


def test_creation_with_all_implementations():
    """Create an instance of the child class of QuantumSDKTranspiler with all methods implemented.

    Check if
    1. No exception is raised.
    2. convert_result can be called without error.
    3. transpile_circuit can be called without error.
    4. transpile_hamiltonian can be called without error.
    5. transpile_operators can be called without error.
    """
    # 1. No exception is raised.
    transpiler = ValidTranspilerWithAllImplementations()

    # 2. convert_result can be called without error.
    transpiler.convert_result(1)
    # 3. transpile_circuit can be called without error.
    transpiler.transpile_circuit(1)
    # 4. transpile_hamiltonian can be called without error.
    transpiler.transpile_hamiltonian(1)
    # 5. transpile_operators can be called without error.
    transpiler.transpile_operators([1, 2, 3])
