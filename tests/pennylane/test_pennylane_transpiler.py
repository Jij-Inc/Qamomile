import pytest
import pennylane as qml
import numpy as np
import qamomile.core.operator as qm_o
from qamomile.pennylane.transpiler import PennylaneTranspiler

@pytest.fixture
def transpiler():
    """Fixture to initialize the PennylaneTranspiler."""
    return PennylaneTranspiler()

def test_transpile_hamiltonian(transpiler):
    """Test the transpilation of Qamomile Hamiltonian to Pennylane Hamiltonian."""
    # Define a Qamomile Hamiltonian
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)

    # Transpile the Hamiltonian
    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Assert the result is a Pennylane Hamiltonian
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)

    # Validate number of qubits and terms
    assert len(pennylane_hamiltonian.operands) == 1 # Only one term
    assert np.all((pennylane_hamiltonian.coeffs , [1.0, ])) # Default coefficient is 1.0

    # Validate term content
    term_ops = pennylane_hamiltonian.terms()[1]
    # assert isinstance(term_ops, qml.operation)
    assert len(term_ops[0]) == 2  # Two operators in the term
    assert isinstance(term_ops[0][0], qml.PauliX)  # X on qubit 0
    assert isinstance(term_ops[0][1], qml.PauliZ)  # Z on qubit 1




def test_transpile_complex_hamiltonian(transpiler):
    """Test the transpilation of Qamomile Hamiltonian to Pennylane Hamiltonian."""
    # Define a Qamomile Hamiltonian
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    hamiltonian += qm_o.Y(0) * qm_o.Y(1)

    # Transpile the Hamiltonian
    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Assert the result is a Pennylane Hamiltonian
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)

    # Validate number of qubits and terms
    assert len(pennylane_hamiltonian.operands) == 2 # Only one term
    assert np.all((pennylane_hamiltonian.coeffs , [1.0, 1.0])) # Default coefficient is 1.0

    # Validate term content
    term_ops = pennylane_hamiltonian.terms()[1]
    # assert isinstance(term_ops, qml.operation)
    assert len(term_ops[0]) == 2  # Two operators in the term
    assert len(term_ops[1]) == 2 
    assert isinstance(term_ops[0][0], qml.PauliX)  # X on qubit 0
    assert isinstance(term_ops[0][1], qml.PauliZ)  # Z on qubit 1
    assert isinstance(term_ops[1][0], qml.PauliY)  # Y on qubit 0
    assert isinstance(term_ops[1][1], qml.PauliY)  # Y on qubit 1

