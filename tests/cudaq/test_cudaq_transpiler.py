# File: tests/cudaq/test_transpiler.py
import sys
import pytest

# Skip all tests if the platform is not Linux.
if sys.platform != "linux":
    pytest.skip("CUDA Quantum requires Linux", allow_module_level=True)


import itertools

import cudaq
import numpy as np

from qamomile.core.circuit import QuantumCircuit as QamomileCircuit
from qamomile.core.circuit import Parameter
from qamomile.core.operator import Hamiltonian, PauliOperator, Pauli, X, Y, Z
import qamomile.core.bitssample as qm_bs
from qamomile.cudaq.transpiler import CudaqTranspiler
from qamomile.cudaq.exceptions import QamomileCudaqTranspileError
from tests.cudaq.utils import CudaqUtils
from tests.mock import UnsupportedGate
from tests.utils import Utils


@pytest.fixture
def transpiler():
    return CudaqTranspiler()


def test_transpile_hamiltonian(transpiler: CudaqTranspiler):
    """Transpile a Qamomile Hamiltonian to a CUDA-Q Hamiltonian.

    Check if
    1. The transpiled Hamiltonian is a CUDA-Q SpinOperator.
    2. The number of terms in the CUDA-Q Hamiltonian is correct.
    3. The coefficients of the terms in the CUDA-Q Hamiltonian are correct.
    """
    np.random.seed(901)  # For reproducibility
    max_num_qubits = (
        2  # Until this number, all combinations of bitstrings will be tested.
    )

    for num_qubits in range(1, max_num_qubits + 1):
        # Create a Hamiltonian with all combinations of Pauli operators.
        pauli_products = list(
            itertools.product((Pauli.X, Pauli.Y, Pauli.Z), repeat=num_qubits)
        )
        for r in range(1, len(pauli_products) + 1):
            # Get all combinations of Pauli operators of size r.
            pauli_combinations = list(itertools.combinations(pauli_products, r))
            for pauli_combination in pauli_combinations:
                # Create a Hamiltonian with the combination of Pauli operators.
                hamiltonian = Hamiltonian()
                num_terms = len(pauli_combination)
                coeffs = []

                for paulis in pauli_combination:
                    pauli_operator = tuple(
                        PauliOperator(pauli, idx) for idx, pauli in enumerate(paulis)
                    )
                    coeff = np.random.uniform(0.1, 10.0)
                    coeffs.append(coeff)
                    hamiltonian.add_term(
                        pauli_operator,
                        coeff,
                    )

                # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
                cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

                # 1. The transpiled Hamiltonian is a CUDA-Q SpinOperator.
                assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

                # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
                cudaq_num_terms = 0
                cudaq_coeffs = []
                for term in cudaq_hamiltonian:
                    coeff = term.coefficient.evaluate()
                    if coeff != 0:
                        cudaq_num_terms += 1
                        cudaq_coeffs.append(coeff)
                # 2. The number of terms in the CUDA-Q Hamiltonian is correct.
                print("======")
                print(hamiltonian)
                print(cudaq_hamiltonian)
                assert cudaq_num_terms == num_terms
                # 3. The coefficients of the terms in the CUDA-Q Hamiltonian are correct.
                assert np.allclose(cudaq_coeffs, coeffs)


def test_transpile_duplicated_hamiltonian(transpiler: CudaqTranspiler):
    """Transpile a Qamomile Hamiltonian whose terms are duplicated to a CUDA-Q Hamiltonian.

    Check if
    1. The transpiled Hamiltonian is a CUDA-Q SpinOperator.
    2. The number of terms in the CUDA-Q Hamiltonian is correct.
    3. The coefficients of the terms in the CUDA-Q Hamiltonian are correct.
    """
    np.random.seed(901)  # For reproducibility
    max_duplication = 10

    for pauli in (Pauli.X, Pauli.Y, Pauli.Z):
        for duplication in range(1, max_duplication + 1):
            # Create a Hamiltonian with duplicate terms.
            hamiltonian = Hamiltonian()
            coeff = 0
            for _ in range(duplication):
                _coeff = np.random.uniform(0.1, 10.0)
                coeff += _coeff
                hamiltonian.add_term((PauliOperator(pauli, 0),), _coeff)

            # Transpile the Hamiltonian to a CUDA-Q Hamiltonian.
            cudaq_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

            # 1. The transpiled Hamiltonian is a CUDA-Q SpinOperator.
            assert isinstance(cudaq_hamiltonian, cudaq.SpinOperator)

            # Get the number of terms and coefficients in the CUDA-Q Hamiltonian.
            num_terms = 0
            cudaq_coeffs = []
            for term in cudaq_hamiltonian:
                coeff = term.coefficient.evaluate()
                if coeff != 0:
                    num_terms += 1
                    cudaq_coeffs.append(coeff)
            # 2. The number of terms in the CUDA-Q Hamiltonian is correct.
            assert num_terms == 1
            # 3. The coefficients of the terms in the CUDA-Q Hamiltonian are correct.
            assert np.allclose(cudaq_coeffs, [coeff])


def test_transpile_all_gates(transpiler: CudaqTranspiler):
    """Transpile a Qamomile circuit with all gates to a CUDA-Q kernel.

    Check if
    1. The transpiled circuit is a CUDA-Q kernel.
    2. The kernel has the expected number of qubits.
    3. The statevector of the kernel matches the expected statevector.
    4. Check if the kernel has the expected number of measured classical bits.
    """
    # >>> Circuit and the expected statevector criation >>>
    num_qubits = 3
    num_cbits = 3
    num_patameters = 0
    num_operations = 0
    qc = QamomileCircuit(num_qubits, num_cbits)
    state_0 = Utils.take_tensor_product([Utils.KET_0, Utils.KET_0, Utils.KET_0])
    # Apply the X gate.
    qc.x(0)
    num_operations += 1
    state_1 = (
        Utils.take_tensor_product([Utils.I_MATRIX, Utils.I_MATRIX, Utils.X_MATRIX])
        @ state_0
    )
    # Apply the Y gate.
    qc.y(1)
    num_operations += 1
    state_2 = (
        Utils.take_tensor_product([Utils.I_MATRIX, Utils.Y_MATRIX, Utils.I_MATRIX])
        @ state_1
    )
    # Apply the Z gate.
    qc.z(2)
    num_operations += 1
    state_3 = (
        Utils.take_tensor_product([Utils.Z_MATRIX, Utils.I_MATRIX, Utils.I_MATRIX])
        @ state_2
    )
    # Apply the H gate.
    qc.h(0)
    num_operations += 1
    state_4 = (
        Utils.take_tensor_product([Utils.I_MATRIX, Utils.I_MATRIX, Utils.H_MATRIX])
        @ state_3
    )
    # Apply the S gate.
    qc.s(1)
    num_operations += 1
    state_5 = (
        Utils.take_tensor_product([Utils.I_MATRIX, Utils.S_MATRIX, Utils.I_MATRIX])
        @ state_4
    )
    # Apply the T gate.
    qc.t(2)
    num_operations += 1
    state_6 = (
        Utils.take_tensor_product([Utils.T_MATRIX, Utils.I_MATRIX, Utils.I_MATRIX])
        @ state_5
    )
    # Apply the RX gate.
    rx_theta = Parameter("rx")
    qc.rx(rx_theta, 0)
    num_patameters += 1
    num_operations += 1
    state_7 = (
        lambda rx_theta_value: Utils.take_tensor_product(
            [Utils.I_MATRIX, Utils.I_MATRIX, Utils.RX_MATRIX(rx_theta_value)]
        )
        @ state_6
    )
    # Apply the RY gate.
    ry_theta = Parameter("ry")
    qc.ry(ry_theta, 1)
    num_patameters += 1
    num_operations += 1
    state_8 = lambda rx_theta_value, ry_theta_value: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.RY_MATRIX(ry_theta_value), Utils.I_MATRIX]
    ) @ state_7(rx_theta_value)
    # Apply the RZ gate.
    rz_theta = Parameter("rz")
    qc.rz(rz_theta, 2)
    num_patameters += 1
    num_operations += 1
    state_9 = lambda rx_theta_value, ry_theta_value, rz_theta_value: Utils.take_tensor_product(
        [Utils.RZ_MATRIX(rz_theta_value), Utils.I_MATRIX, Utils.I_MATRIX]
    ) @ state_8(
        rx_theta_value, ry_theta_value
    )
    # Apply the CX gate.
    qc.cx(0, 1)
    num_operations += 1
    state_10 = lambda theta_values: Utils.take_tensor_product(
        [Utils.I_MATRIX, CudaqUtils.CX_MATRIX]
    ) @ state_9(*theta_values)
    # Apply the CZ gate.
    qc.cz(1, 2)
    num_operations += 1
    state_11 = lambda theta_values: Utils.take_tensor_product(
        [CudaqUtils.CZ_MATRIX, Utils.I_MATRIX]
    ) @ state_10(theta_values)
    # Apply the CRX gate.
    crx_theta = Parameter("crx")
    qc.crx(crx_theta, 0, 1)
    num_patameters += 1
    num_operations += 1
    state_12 = lambda theta_values: Utils.take_tensor_product(
        [Utils.I_MATRIX, CudaqUtils.CRX_MATRIX(theta_values[-1])]
    ) @ state_11(theta_values[:-1])
    # Apply the CRY gate.
    cry_theta = Parameter("cry")
    qc.cry(cry_theta, 1, 2)
    num_patameters += 1
    num_operations += 1
    state_13 = lambda theta_values: Utils.take_tensor_product(
        [CudaqUtils.CRY_MATRIX(theta_values[-1]), Utils.I_MATRIX]
    ) @ state_12(theta_values[:-1])
    # Apply the CRZ gate.
    crz_theta = Parameter("crz")
    qc.crz(crz_theta, 0, 1)
    num_patameters += 1
    num_operations += 1
    state_14 = lambda theta_values: Utils.take_tensor_product(
        [Utils.I_MATRIX, CudaqUtils.CRZ_MATRIX(theta_values[-1])]
    ) @ state_13(theta_values[:-1])
    # Apply the RXX gate.
    rxx_theta = Parameter("rxx")
    qc.rxx(rxx_theta, 0, 1)
    num_patameters += 1
    num_operations += 7  # 7 operations for RXX gate (See _apply_parametric_two_qubit_gate function qamomile/cudaq/transpiler.py)
    state_15 = lambda theta_values: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.RXX_MATRIX(theta_values[-1])]
    ) @ state_14(theta_values[:-1])
    # Apply the RYY gate.
    ryy_theta = Parameter("ryy")
    qc.ryy(ryy_theta, 1, 2)
    num_patameters += 1
    num_operations += 7  # 7 operations for RYY gate (See _apply_parametric_two_qubit_gate function qamomile/cudaq/transpiler.py)
    state_16 = lambda theta_values: Utils.take_tensor_product(
        [Utils.RYY_MATRIX(theta_values[-1]), Utils.I_MATRIX]
    ) @ state_15(theta_values[:-1])
    # Apply the RZZ gate.
    rzz_theta = Parameter("rzz")
    qc.rzz(rzz_theta, 0, 1)
    num_patameters += 1
    num_operations += 3  # 3 operations for RZZ gate (See _apply_parametric_two_qubit_gate function qamomile/cudaq/transpiler.py)
    state_17 = lambda theta_values: Utils.take_tensor_product(
        [Utils.I_MATRIX, Utils.RZZ_MATRIX(theta_values[-1])]
    ) @ state_16(theta_values[:-1])
    # Apply the CCX gate.
    qc.ccx(0, 1, 2)
    num_operations += 1
    state_18 = lambda theta_values: CudaqUtils.CCX_MATRIX @ state_17(theta_values)
    # Apply the parameteric exp gate.
    exp_theta = Parameter("exp")
    hamiltonian = Hamiltonian()
    hamiltonian += X(0) * Y(1) * Z(2)
    qc.exp_evolution(exp_theta, hamiltonian)
    num_patameters += 1
    num_operations += 1
    hamiltonian_matrix = Utils.take_tensor_product(
        [Utils.Z_MATRIX, Utils.Y_MATRIX, Utils.X_MATRIX]
    )
    state_19 = lambda theta_values: CudaqUtils.EXP_PAULI_MATRIX(
        theta_values[-1], hamiltonian_matrix
    ) @ state_18(theta_values[:-1])
    # <<< Circuit and the expected statevector criation <<<

    # Transpile the circuit to a CUDA-Q kernel.
    cudaq_kernel = transpiler.transpile_circuit(qc)

    # 1. The transpiled circuit is a CUDA-Q kernel.
    assert isinstance(cudaq_kernel, cudaq.Kernel)

    # 2. The kernel has the expected number of qubits.
    cudaq_num_qubits = cudaq.get_state(cudaq_kernel, []).num_qubits()
    assert cudaq_num_qubits == num_qubits

    # Get the statevector of the kernel.
    cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, []))

    # 3. The statevector of the kernel matches the expected statevector.
    np.random.seed(901)  # For reproducibility
    num_trials = 50
    for _ in range(num_trials):
        theta_values = np.random.uniform(0, 2 * np.pi, num_patameters).tolist()
        cudaq_statevector = np.array(cudaq.get_state(cudaq_kernel, theta_values))
        expected_statevector = state_19(theta_values).flatten()
        assert np.allclose(cudaq_statevector, expected_statevector)

    # 4. Check if the kernel has the expected number of measured classical bits.
    # CUDA-Q returns a statevector after measured a classical bit if the kernel has measurement operations.
    # Thus, the measurement operation is added here not before checking the statevector.
    num_measured_cbits = 0
    qc.measure(0, 0)
    num_measured_cbits += 1
    cudaq_kernel = transpiler.transpile_circuit(qc)
    sample = cudaq.sample(cudaq_kernel, theta_values)
    for key, _ in sample.items():
        break
    assert len(key) == num_measured_cbits


def test_transpile_unsupported_gate(transpiler: CudaqTranspiler):
    """Transpile a Qamomile circuit with an unsupported gate to a CUDA-Q kernel.

    Check if
    1. The transpiler raises a QamomileCudaqTranspileError.
    """

    qc = QamomileCircuit(1)
    qc.gates.append(UnsupportedGate())

    # 1. The transpiler raises a QamomileCudaqTranspileError.
    with pytest.raises(QamomileCudaqTranspileError):
        transpiler.transpile_circuit(qc)


def test_convert_result(transpiler: CudaqTranspiler):
    """Convert a result from CUDA-Q to Qamomile BitsSampleSet.

    Check if
    1. The converted result is a Qamomile BitsSampleSet.
    2. The BitsSampleSet has the expected number of bitarrays.
    3. The total number of samples in the BitsSampleSet matches the sum of the counts in the result.
    """
    np.random.seed(901)  # For reproducibility
    max_num_cbits = (
        3  # Until this number, all combinations of bitstrings will be tested.
    )
    for num_cbits in range(1, max_num_cbits + 1):
        # Get all combinations of bitstrings for the given number of classical bits.
        bitstrings = [
            "".join(bits) for bits in itertools.product("01", repeat=num_cbits)
        ]
        # Iterate over all combinations of bitstrings to create a mock result.
        for r in range(1, len(bitstrings) + 1):
            # Get all combinations of bitstrings of size r.
            combinations = list(itertools.combinations(bitstrings, r))
            for combination in combinations:
                # Create a mock resutl with random counts for each bitstring in the combination.
                mock_result = {
                    bitstring: np.random.randint(100, 1000) for bitstring in combination
                }
                # Compute the total number of shots.
                num_shots = sum(mock_result.values())
                # Convert the mock result to a Qamomile BitsSampleSet.
                result = transpiler.convert_result(mock_result)

                # 1. The converted result is a Qamomile BitsSampleSet.
                assert isinstance(result, qm_bs.BitsSampleSet)
                # 2. The BitsSampleSet has the expected number of bitarrays.
                assert len(result.bitarrays) == len(combination)
                # 3. The total number of samples in the BitsSampleSet matches the sum of the counts in the result.
                assert result.total_samples() == num_shots
