"""SciPy sparse-array block construction and backend-neutral consumption."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import qamomile as qm
from qamomile.stdlib import (
    BlockEncoding,
    qsvt_application,
    state_preparation_block_encoding,
)

scipy_sparse = pytest.importorskip("scipy.sparse")
sparse_qsvt_step = qsvt_application(signal_width=3, system_width=1, degree=1)
sparse_qsvt_roundtrip = qsvt_application(signal_width=3, system_width=1, degree=2)


@qm.qkernel
def sparse_swap_state(block: BlockEncoding[3, 1]) -> qm.Q[qm.Bits[4]]:
    """Apply a later-selected sparse block to the zero state."""

    return block(qm.qubits(4))


@qm.qkernel
def sparse_swap_sample(block: BlockEncoding[3, 1]) -> qm.Shot[qm.Bits[4]]:
    """Measure a later-selected sparse block on the zero state."""

    return qm.measure(sparse_swap_state(block))


@qm.qkernel
def sparse_basis_one(block: BlockEncoding[3, 1]) -> qm.Q[qm.Bits[4]]:
    """Apply a later-selected sparse block to system basis one."""

    return block(qm.x(qm.qubits(4), target=3))


@qm.qkernel
def sparse_qsvt_sample(
    block: BlockEncoding[3, 1],
    phases: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[4]]:
    """Transform a later-selected sparse block with runtime QSVT phases."""

    state = sparse_qsvt_step(block, qm.qubits(4), phases)
    return qm.measure(state)


@qm.qkernel
def sparse_four_state(block: BlockEncoding[4, 2]) -> qm.Q[qm.Bits[6]]:
    """Apply a four-row sparse block to system basis three."""

    state = qm.qubits(6)
    state = qm.x(state, target=4)
    state = qm.x(state, target=5)
    return block(state)


@qm.qkernel
def sparse_qsvt_roundtrip_sample(
    block: BlockEncoding[3, 1],
    phases: qm.Param[qm.Vector[qm.Real, 3]],
) -> qm.Shot[qm.Bits[4]]:
    """Use a sparse block and its adjoint in a deferred QSVT application."""

    state = sparse_qsvt_roundtrip(block, qm.qubits(4), phases)
    return qm.measure(state)


def test_sparse_matrix_projected_block_matches_complex_input() -> None:
    """Independent statevector projections recover every complex matrix entry."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    values = np.array([[0, 1j], [0.5, -0.25]], dtype=complex)
    matrix = scipy_sparse.csr_array(values)
    block = state_preparation_block_encoding(matrix, max_error=1e-8)
    assert block.normalization == 2.0
    assert 0 < block.error_bound <= 1e-8
    assert (block.signal_width, block.system_width) == (3, 1)
    with pytest.raises(ValueError, match="certified error bound"):
        state_preparation_block_encoding(matrix, max_error=0)
    for column, source in enumerate((sparse_swap_state, sparse_basis_one)):
        circuit = to_circuit(source, target=Qiskit(), block=block)
        amplitudes = Statevector.from_instruction(circuit).data
        np.testing.assert_allclose(
            amplitudes[[0, 8]],
            values[:, column] / block.normalization,
            atol=1e-12,
        )
    np.testing.assert_array_equal(matrix.toarray(), values)


def test_sparse_matrix_multibit_register_order() -> None:
    """Four-row input verifies the index/system swap and signal projection."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    matrix = scipy_sparse.csr_array(
        np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, -0.5, 0, 0], [0, 0, 0, 0]])
    )
    block = state_preparation_block_encoding(matrix)
    assert (block.normalization, block.signal_width, block.system_width) == (1, 4, 2)
    circuit = to_circuit(sparse_four_state, target=Qiskit(), block=block)
    amplitudes = Statevector.from_instruction(circuit).data
    np.testing.assert_allclose(amplitudes[[0, 16, 32, 48]], (1, 0, 0, 0), atol=1e-12)


def test_sparse_matrix_canonicalization_zero_and_validation() -> None:
    """Duplicates, zero rows, and invalid array contracts are handled early."""

    matrix = scipy_sparse.coo_array(
        (np.array([0.25, 0.25]), (np.array([0, 0]), np.array([0, 0]))),
        shape=(2, 2),
    )
    block = state_preparation_block_encoding(matrix)
    assert block.normalization == 0.5
    zero = state_preparation_block_encoding(scipy_sparse.csr_array((2, 2)))
    assert zero.normalization == 1.0
    assert (zero.signal_width, zero.system_width) == (3, 1)
    assert qm.evaluate(sparse_swap_sample.specialize(block=zero)) != (0, 0, 0, 0)
    with pytest.raises(TypeError, match="sparse array"):
        state_preparation_block_encoding(np.eye(2))
    with pytest.raises(TypeError, match="sparse array"):
        state_preparation_block_encoding(scipy_sparse.csr_matrix(np.eye(2)))
    with pytest.raises(ValueError, match="square"):
        state_preparation_block_encoding(scipy_sparse.csr_array((2, 4)))
    with pytest.raises(ValueError, match="power of two"):
        state_preparation_block_encoding(scipy_sparse.csr_array((3, 3)))
    with pytest.raises(ValueError, match="budget"):
        state_preparation_block_encoding(scipy_sparse.csr_array((32, 32)))
    with pytest.raises(ValueError, match="finite"):
        state_preparation_block_encoding(scipy_sparse.csr_array([[np.nan, 0], [0, 1]]))
    with pytest.raises(ValueError, match="exact binary64"):
        state_preparation_block_encoding(
            scipy_sparse.csr_array([[2**53 + 1, 0], [0, 0]])
        )
    with pytest.raises(ValueError, match="max_error"):
        state_preparation_block_encoding(scipy_sparse.csr_array((2, 2)), max_error=-1)


def test_sparse_blocks_bind_after_algorithm_declaration() -> None:
    """Two sparse matrices specialize one deferred algorithm and QSVT source."""

    swap = state_preparation_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    identity = state_preparation_block_encoding(scipy_sparse.eye_array(2))
    assert qm.evaluate(sparse_swap_sample.specialize(block=swap)) == (0, 0, 0, 1)
    assert qm.evaluate(sparse_swap_sample.specialize(block=identity)) == (0, 0, 0, 0)
    assert qm.evaluate(
        sparse_qsvt_sample.specialize(block=swap), phases=(0.0, 0.0)
    ) == (0, 0, 0, 1)
    assert qm.evaluate(
        sparse_qsvt_roundtrip_sample.specialize(block=swap),
        phases=(0.0, 0.0, 0.0),
    ) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("module_name", "target_name", "endpoint_name", "dependency"),
    [
        ("qamomile.qiskit", "Qiskit", "Aer", "qiskit_aer"),
        pytest.param(
            "qamomile.cirq", "Cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq
        ),
        pytest.param(
            "qamomile.quri_parts",
            "QuriParts",
            "Qulacs",
            "qulacs",
            marks=pytest.mark.quri_parts,
        ),
        pytest.param(
            "qamomile.cudaq", "Cudaq", "CudaqLocal", "cudaq", marks=pytest.mark.cudaq
        ),
        pytest.param(
            "qamomile.braket",
            "Braket",
            "BraketLocal",
            "braket.circuits",
            marks=pytest.mark.braket,
        ),
    ],
)
def test_sparse_block_executes_on_supported_backends(
    module_name: str, target_name: str, endpoint_name: str, dependency: str
) -> None:
    """The same deferred source compiles, samples, and evaluates on each SDK."""

    pytest.importorskip(dependency)
    module = importlib.import_module(module_name)
    endpoint = getattr(module, endpoint_name)()
    target = getattr(module, target_name)()
    seed = 0 if endpoint_name == "BraketLocal" else 2
    block = state_preparation_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    artifact = qm.compile(sparse_swap_sample, to=target, block=block)
    assert qm.run(artifact, on=endpoint, shots=4, seed=seed).result.values == {
        (0, 0, 0, 1): 4
    }
    state = sparse_swap_state.bind(block=block)
    assert qm.expectation(state, qm.Z(3), on=endpoint, shots=4, seed=seed).value == -1.0


@pytest.mark.hugr
def test_sparse_block_compiles_to_hugr_and_persists() -> None:
    """Bound sparse data becomes a portable program without a SciPy object."""

    pytest.importorskip("hugr")
    from qamomile.hugr import Hugr

    block = state_preparation_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    program = sparse_swap_sample.specialize(block=block)
    restored = qm.Program.from_bytes(program.to_bytes())
    assert qm.evaluate(restored) == (0, 0, 0, 1)
    assert qm.compile(restored, to=Hugr())
    roundtrip = sparse_qsvt_roundtrip_sample.specialize(block=block)
    assert qm.compile(roundtrip, to=Hugr())
