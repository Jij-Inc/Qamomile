# File: tests/qiskit/test_transpiler.py

import pytest
import numpy as np
import qiskit
import qiskit.quantum_info as qk_ope
from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli
import qamomile.core.bitssample as qm_bs
from qamomile.qiskit.transpiler import QiskitTranspiler
from qamomile.qiskit.exceptions import QamomileQiskitTranspileError


@pytest.fixture
def transpiler():
    return QiskitTranspiler()


def test_transpile_simple_circuit(transpiler):
    qc = QamomileCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.num_qubits == 2
    assert len(qiskit_circuit.data) == 2
    assert qiskit_circuit.data[0].operation.name == "h"
    assert qiskit_circuit.data[1].operation.name == "cx"


def test_transpile_parametric_circuit(transpiler):
    qc = QamomileCircuit(1)
    theta = Parameter("theta")
    qc.rx(theta, 0)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert len(qiskit_circuit.parameters) == 1
    assert qiskit_circuit.parameters[0].name == "theta"


def test_transpile_complex_circuit(transpiler):
    qc = QamomileCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0, 0)

    qiskit_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(qiskit_circuit, qiskit.QuantumCircuit)
    assert qiskit_circuit.num_qubits == 3
    assert qiskit_circuit.num_clbits == 3
    assert len(qiskit_circuit.data) == 4


def test_transpile_unsupported_gate(transpiler):
    class UnsupportedGate:
        pass

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(QamomileQiskitTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler):
    # Simulate Qiskit BitArray result
    class MockBitArray:
        def __init__(self, counts, num_bits):
            self.counts = counts
            self.num_bits = num_bits

        def get_int_counts(self):
            return self.counts

    mock_result = MockBitArray({0: 500, 3: 500}, 2)

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler):
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.Z, 1),), 2.0)

    qiskit_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(qiskit_hamiltonian, qk_ope.SparsePauliOp)
    assert len(qiskit_hamiltonian) == 2
    assert np.allclose(qiskit_hamiltonian.coeffs, [1.0, 2.0])
