import pytest
import pennylane as qml
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
    assert len(pennylane_hamiltonian.ops) == 1  # Only one term
    assert pennylane_hamiltonian.coeffs[0] == 1.0  # Default coefficient is 1.0

    # Validate term content
    term_ops = pennylane_hamiltonian.ops[0]
    assert isinstance(term_ops, qml.operation.Tensor)
    assert len(term_ops.obs) == 2  # Two operators in the term
    assert isinstance(term_ops.obs[0], qml.PauliX)  # X on qubit 0
    assert isinstance(term_ops.obs[1], qml.PauliZ)  # Z on qubit 1

def test_transpile_complex_hamiltonian(transpiler):
    """Test transpilation of a more complex Qamomile Hamiltonian."""
    # Define a more complex Qamomile Hamiltonian
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Y(1)
    hamiltonian += qm_o.Z(0) * qm_o.X(1)

    # Transpile the Hamiltonian
    pennylane_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Assert the result is a Pennylane Hamiltonian
    assert isinstance(pennylane_hamiltonian, qml.Hamiltonian)

    # Validate number of terms and coefficients
    assert len(pennylane_hamiltonian.ops) == 2  # Two terms
    assert pennylane_hamiltonian.coeffs[0] == 1.0  # First term coefficient
    assert pennylane_hamiltonian.coeffs[1] == 1.0  # Second term coefficient

    # Validate term contents
    term1_ops = pennylane_hamiltonian.ops[0]
    term2_ops = pennylane_hamiltonian.ops[1]

    # First term: X on qubit 0 and Y on qubit 1
    assert isinstance(term1_ops, qml.operation.Tensor)
    assert len(term1_ops.obs) == 2
    assert isinstance(term1_ops.obs[0], qml.PauliX)
    assert isinstance(term1_ops.obs[1], qml.PauliY)

    # Second term: Z on qubit 0 and X on qubit 1
    assert isinstance(term2_ops, qml.operation.Tensor)
    assert len(term2_ops.obs) == 2
    assert isinstance(term2_ops.obs[0], qml.PauliZ)
    assert isinstance(term2_ops.obs[1], qml.PauliX)

