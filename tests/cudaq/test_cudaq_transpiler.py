# File: tests/cudaq/test_transpiler.py
import cudaq
import numpy as np
import pytest

import qamomile.core as qm
from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli, X, Y, Z
import qamomile.core.bitssample as qm_bs
from qamomile.cudaq.transpiler import CudaqTranspiler
from qamomile.cudaq.exceptions import QamomileCudaqTranspileError
from tests.utils import *
from tests.cudaq.utils import *


@pytest.fixture
def transpiler():
    return CudaqTranspiler()


def test_transpile_simple_circuit(transpiler: CudaqTranspiler):
    # Create a simple circuit and the expected statevector.
    num_qubits = 2
    qc = QamomileCircuit(num_qubits)
    state_0 = np.kron(KET_0, KET_0)  # |00>
    qc.x(0)
    x_applied_state = np.kron(X_MATRIX, I_MATRIX) @ state_0  # |10>
    qc.cx(0, 1)
    expected_statevector = (CX_MATRIX @ x_applied_state).flatten()  # |11>

    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Get the statevector of the kernel.
    cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, []))
    # Check if the cudaq statevector matches the expected statevector.
    assert np.allclose(cudaq_statevector, expected_statevector)


def test_transpile_parametric_circuit(transpiler: CudaqTranspiler):
    # Create a simple parametric circuit and the expected statevector.
    num_qubits = 1
    qc = QamomileCircuit(num_qubits)
    state_0 = KET_0  # |0>
    theta = Parameter("theta")
    qc.rx(theta, 0)
    expected_statevector = (
        lambda theta: RX_MATRIX(theta) @ state_0
    )  # |0> rotated by theta
    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)

    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Check if the kernel has the expected number of qubits.
    qir_str = cudaq.translate(cudaq_kernel, [0], format="qir")
    assert count_qir_parameters(qir_str) == 1  # One parameter for theta
    # Check if the kernel has only one operation, which is the RX.
    operations = count_qir_operations(qir_str)
    assert len(operations) == 1
    assert operations["__quantum__qis__rx"] == 1

    # Check if the statevector matches the expected statevector for several thetas.
    np.random.seed(901)
    num_trials = 100
    for _ in range(num_trials):
        theta_value = np.random.uniform(0, 2 * np.pi)
        cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, [theta_value]))
        assert np.allclose(
            cudaq_statevector, expected_statevector(theta_value).flatten()
        )


def test_transpile_complex_circuit(transpiler: CudaqTranspiler):
    # Create a more complex circuit with multiple gates.
    num_qubits = 3
    num_cbits = 3
    num_measured_cbits = 0
    qc = QamomileCircuit(num_qubits, num_cbits)
    state_0 = take_tensor_product(KET_0, KET_0, KET_0)  # |000>
    qc.h(0)
    state_1 = take_tensor_product(H_MATRIX, I_MATRIX, I_MATRIX) @ state_0
    qc.cx(0, 1)
    state_2 = take_tensor_product(CX_MATRIX, I_MATRIX) @ state_1
    qc.ccx(0, 1, 2)
    expected_statevector = (CCX_MATRIX @ state_2).flatten()

    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)

    # Check if the statevector matches the expected statevector.
    cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, []))
    assert np.allclose(cudaq_statevector, expected_statevector)

    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Check if the kernel has the expected number of measured classical bits.
    # CUDA-Q returns a statevector after measured a classical bit if the kernel has measurement operations.
    # Thus, the measurement operation is added here not before checking the statevector.
    qc.measure(0, 0)
    num_measured_cbits += 1
    cudaq_kernel = transpiler.transpile_circuit(qc)
    sample = cudaq.sample(cudaq_kernel, [])
    for key, _ in sample.items():
        break
    assert len(key) == num_measured_cbits


def test_transpile_unsupported_gate(transpiler: CudaqTranspiler):
    class UnsupportedGate:
        pass

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    with pytest.raises(QamomileCudaqTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler: CudaqTranspiler):
    mock_result = {"000": 500, "010": 500}

    result = transpiler.convert_result(mock_result)

    assert isinstance(result, qm_bs.BitsSampleSet)
    assert len(result.bitarrays) == 2
    assert result.total_samples() == 1000


def test_transpile_hamiltonian(transpiler: CudaqTranspiler):
    # Create a Hamiltonian in Qamomile format.
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.Z, 1),), 2.0)

    # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
    cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Check if the transpiled Hamiltonian is a CUDA-Q SpinOperator.
    assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

    # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
    num_terms = 0
    cudaq_coeffs = []
    for term in cudaq_hamiltonian:
        coeff = term.coefficient.evaluate()
        if coeff != 0:
            num_terms += 1
            cudaq_coeffs.append(coeff)
    # Check the number of terms and coefficients.
    assert num_terms == 2
    # Check the coefficients of the terms.
    assert np.allclose(cudaq_coeffs, [1.0, 2.0])

    # Create another Hamiltonian with duplicate terms.
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)
    hamiltonian.add_term((PauliOperator(Pauli.X, 0),), 1.0)

    # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
    cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    # Check if the transpiled Hamiltonian is a CUDA-Q SpinOperator.
    assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

    # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
    num_terms = 0
    cudaq_coeffs = []
    for term in cudaq_hamiltonian:
        coeff = term.coefficient.evaluate()
        if coeff != 0:
            num_terms += 1
            cudaq_coeffs.append(coeff)
    # Check the number of terms and coefficients.
    assert num_terms == 1
    # Check the coefficients of the terms.
    assert np.allclose(cudaq_coeffs, [2.0])


def test_parametric_exp_gate(transpiler: CudaqTranspiler):
    hamiltonian = Hamiltonian()
    hamiltonian += X(0) * Z(1)
    qc = QamomileCircuit(2)
    theta = Parameter("theta")
    qc.exp_evolution(theta, hamiltonian)

    cudaq_kernel = transpiler.transpile_circuit(qc)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    qir_str = cudaq.translate(cudaq_kernel, [1], format="qir")
    assert count_qir_parameters(qir_str) == 1  # One parameter for theta
    # Check if the kernel has only one operation, which is the Pauli evolution.
    operations = count_qir_operations(qir_str)
    assert len(operations) == 1
    assert operations["__quantum__qis__exp_pauli"] == 1
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == 2

    hamiltonian2 = Hamiltonian()
    hamiltonian2 += X(0) * Y(1) + Z(0) * X(1)
    qc2 = QamomileCircuit(2)
    qc2.exp_evolution(theta, hamiltonian2)
    cudaq_kernel2 = transpiler.transpile_circuit(qc2)

    # Check if the transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel2, cudaq.Kernel)
    # Check if the kernel has the expected number of qubits.
    qir_str2 = cudaq.translate(cudaq_kernel2, [1], format="qir")
    assert (
        count_qir_parameters(qir_str2) == 2
    )  # Two parameter for X(0) * Y(1) + Z(0) * X(1)
    # Check if the kernel has only one operation, which is the Pauli evolution.
    operations2 = count_qir_operations(qir_str2)
    assert len(operations2) == 1
    assert (
        operations2["__quantum__qis__exp_pauli"] == 2
    )  # Two operations for X(0) * Y(1) + Z(0) * X(1)
    # Check if the kernel has the expected number of qubits.
    cudaq_num_qubits2 = cudaq.get_state(cudaq_kernel2, []).num_qubits()
    assert cudaq_num_qubits2 == 2
