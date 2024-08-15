import pytest
import numpy as np
import collections
import quri_parts.circuit as qp_c
import quri_parts.core.operator as qp_o
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
import qamomile.core.bitssample as qm_bs
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.quri_parts.transpiler import QuriPartsTranspiler



@pytest.fixture
def transpiler():
    return QuriPartsTranspiler()


def test_transpile_simple_circuit(transpiler):
    qc = qm_c.QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 2
    assert len(quri_circuit.gates) == 2
    assert isinstance(quri_circuit.gates[0], qp_c.gate.QuantumGate)
    assert isinstance(quri_circuit.gates[1], qp_c.gate.QuantumGate)


def test_transpile_parametric_circuit(transpiler):
    qc = qm_c.QuantumCircuit(1)
    theta = qm_c.Parameter("theta")
    qc.rx(theta, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 1
    assert len(quri_circuit.gates) == 1
    assert isinstance(quri_circuit.gates[0], qp_c.gate.ParametricQuantumGate)


def test_transpile_complex_circuit(transpiler):
    qc = qm_c.QuantumCircuit(3, 3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert quri_circuit.qubit_count == 3
    assert quri_circuit.cbit_count == 3
    assert len(quri_circuit.gates) == 4


def test_transpile_unsupported_gate(transpiler):
    class UnsupportedGate(qm_c.Gate):
        pass

    qc = qm_c.QuantumCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(Exception):  # Replace with specific exception if known
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler):
    mock_result = (collections.Counter({0: 500, 3: 500}), 2)

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler):
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 1.0)
    hamiltonian.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 1),), 2.0)

    quri_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(quri_hamiltonian, qp_o.Operator)
    assert len(quri_hamiltonian) == 2
    assert np.isclose(
        quri_hamiltonian[qp_o.pauli_label([(0, qp_o.SinglePauli.X)])], 1.0
    )
    assert np.isclose(
        quri_hamiltonian[qp_o.pauli_label([(1, qp_o.SinglePauli.Z)])], 2.0
    )


def test_parametric_two_qubit_gate(transpiler):
    qc = qm_c.QuantumCircuit(2)
    theta = qm_c.Parameter("theta")
    qc.gates.append(
        qm_c.ParametricTwoQubitGate(qm_c.ParametricTwoQubitGateType.RXX, 0, 1, theta)
    )
    qc.ry(theta, 0)

    quri_circuit = transpiler.transpile_circuit(qc)

    assert isinstance(quri_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert len(quri_circuit.gates) == 2
    assert isinstance(quri_circuit.gates[0], qp_c.ParametricQuantumGate)
    assert quri_circuit.gates[0].target_indices == (0, 1)
    assert quri_circuit.gates[0].pauli_ids == (0, 0)  # XX

    assert quri_circuit.parameter_count == 1


def test_qaoa_circuit():
    import jijmodeling as jm
    import jijmodeling_transpiler.core as jmt

    x = jm.BinaryVar("x", shape=(3, ))
    problem = jm.Problem("qubo")
    problem += -x[0]*x[1] + x[1]*x[2] + x[2]*x[0]

    compiled_instance = jmt.compile_model(problem, {})
    qaoa_converter = QAOAConverter(compiled_instance)

    qaoa_circuit = qaoa_converter.get_qaoa_ansatz(2)

    qp_circuit = QuriPartsTranspiler().transpile_circuit(qaoa_circuit)

    assert isinstance(qp_circuit, qp_c.LinearMappedUnboundParametricQuantumCircuit)
    assert qp_circuit.qubit_count == 3
    assert qp_circuit.parameter_count == 4
