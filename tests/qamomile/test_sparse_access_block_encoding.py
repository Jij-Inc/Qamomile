"""Sparse-access block encoding projection, binding, and portability."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import qamomile as qm
from qamomile.stdlib import (
    BlockEncoding,
    pad_block_encoding,
    qsvt_application,
    reciprocal_svt,
    sparse_access_block_encoding,
    state_preparation_block_encoding,
)

scipy_sparse = pytest.importorskip("scipy.sparse")
sparse_access_qsvt_step = qsvt_application(signal_width=5, system_width=1, degree=1)


@qm.qkernel
def sparse_access_two(block: BlockEncoding[5, 1]) -> qm.Q[qm.Bits[6]]:
    """Apply a later-bound sparse-access block to system basis zero."""

    return block(qm.qubits(6))


@qm.qkernel
def sparse_access_two_one(block: BlockEncoding[5, 1]) -> qm.Q[qm.Bits[6]]:
    """Apply a later-bound sparse-access block to system basis one."""

    return block(qm.x(qm.qubits(6), target=5))


@qm.qkernel
def sparse_access_sample(block: BlockEncoding[5, 1]) -> qm.Shot[qm.Bits[6]]:
    """Measure a later-bound sparse-access block."""

    return qm.measure(sparse_access_two(block))


@qm.qkernel
def sparse_access_complex_zero(block: BlockEncoding[7, 1]) -> qm.Q[qm.Bits[8]]:
    """Apply a complex two-row block to system basis zero."""

    return block(qm.qubits(8))


@qm.qkernel
def sparse_access_complex_sample(block: BlockEncoding[7, 1]) -> qm.Shot[qm.Bits[8]]:
    """Measure the complex sparse component for executable capability checks."""
    return qm.measure(sparse_access_complex_zero(block))


@qm.qkernel
def sparse_access_complex_one(block: BlockEncoding[7, 1]) -> qm.Q[qm.Bits[8]]:
    """Apply a complex two-row block to system basis one."""

    return block(qm.x(qm.qubits(8), target=7))


@qm.qkernel
def sparse_access_qsvt(
    block: BlockEncoding[5, 1],
    phases: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Q[qm.Bits[6]]:
    """Apply a later-selected sparse block through a QSVT step."""

    return sparse_access_qsvt_step(block, qm.qubits(6), phases)


@qm.qkernel
def sparse_access_four_basis_three(
    block: BlockEncoding[7, 2],
) -> qm.Q[qm.Bits[9]]:
    """Apply a later-bound four-row block to system basis three."""

    state = qm.qubits(9)
    state = qm.x(state, target=7)
    state = qm.x(state, target=8)
    return block(state)


@qm.qkernel
def sparse_access_phase_interference(
    block: BlockEncoding[6, 1],
) -> qm.Shot[qm.Bits[7]]:
    """Interfere selected signed values before measuring the signal."""

    state = qm.h(qm.qubits(7), target=6)
    state = block(state)
    return qm.measure(qm.h(state, target=6))


def test_sparse_access_projected_block_and_error_certificate() -> None:
    """The projected block recovers every asymmetric complex matrix entry."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    values = np.array([[0, 1j], [0.5, -0.25]], dtype=complex)
    block = sparse_access_block_encoding(scipy_sparse.csr_array(values))
    assert (block.normalization, block.signal_width, block.system_width) == (4, 7, 1)
    assert 0 < block.error_bound <= 1e-8
    for column, source in enumerate(
        (sparse_access_complex_zero, sparse_access_complex_one)
    ):
        circuit = to_circuit(source, target=Qiskit(), block=block)
        actual = Statevector.from_instruction(circuit).data[[0, 128]]
        np.testing.assert_allclose(
            actual, values[:, column] / block.normalization, atol=1e-14
        )


def test_sparse_access_deferred_binding_and_exact_error() -> None:
    """One source accepts different prepared operators after its definition."""

    swap = sparse_access_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    identity = sparse_access_block_encoding(scipy_sparse.eye_array(2))
    assert swap.error_bound == identity.error_bound == 0.0
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    swap_state = Statevector.from_instruction(
        to_circuit(sparse_access_two, target=Qiskit(), block=swap)
    ).data
    identity_state = Statevector.from_instruction(
        to_circuit(sparse_access_two, target=Qiskit(), block=identity)
    ).data
    np.testing.assert_allclose(swap_state[[0, 32]], (0, 0.5), atol=1e-14)
    np.testing.assert_allclose(identity_state[[0, 32]], (0.5, 0), atol=1e-14)
    assert to_circuit(sparse_access_qsvt.bind(block=swap))
    assert to_circuit(sparse_access_qsvt.bind(block=identity))


def test_state_preparation_padding_shares_one_later_bound_signature() -> None:
    """A smaller state-preparation block pads into a sparse-access signature."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    matrix = scipy_sparse.csr_array([[0, 1], [1, 0]])
    smaller = state_preparation_block_encoding(matrix)
    padded = pad_block_encoding(smaller, signal_width=5)
    assert padded.normalization == smaller.normalization
    assert padded.error_bound == smaller.error_bound
    assert (padded.signal_width, padded.system_width) == (5, 1)
    circuit = to_circuit(sparse_access_two, target=Qiskit(), block=padded)
    amplitudes = Statevector.from_instruction(circuit).data
    np.testing.assert_allclose(amplitudes[[0, 32]], (0, 1), atol=1e-14)
    assert to_circuit(sparse_access_qsvt.bind(block=padded))
    with pytest.raises(ValueError, match="smaller"):
        pad_block_encoding(smaller, signal_width=2)


def test_sparse_access_validation_and_zero() -> None:
    """Invalid inputs fail early and the zero block is still a unitary."""

    zero = sparse_access_block_encoding(scipy_sparse.csr_array((2, 2)), max_error=0)
    assert zero.normalization == 1
    assert zero.error_bound == 0
    with pytest.raises(ValueError, match="max_error"):
        sparse_access_block_encoding(
            scipy_sparse.csr_array([[0, 1j], [0.5, -0.25]]), max_error=0
        )
    approximate = sparse_access_block_encoding(
        scipy_sparse.csr_array([[0, 1j], [0.5, -0.25]])
    )
    with pytest.raises(ValueError, match="exact block encoding"):
        reciprocal_svt(approximate, minimum_singular_value=0.25)
    with pytest.raises(TypeError, match="sparse array"):
        sparse_access_block_encoding(np.eye(2))
    with pytest.raises(TypeError, match="sparse array"):
        sparse_access_block_encoding(scipy_sparse.csr_matrix(np.eye(2)))
    with pytest.raises(ValueError, match="square"):
        sparse_access_block_encoding(scipy_sparse.csr_array((2, 4)))
    with pytest.raises(ValueError, match="power of two"):
        sparse_access_block_encoding(scipy_sparse.csr_array((3, 3)))
    with pytest.raises(ValueError, match="work budget"):
        sparse_access_block_encoding(scipy_sparse.eye_array(8192))
    for invalid_budget in (-1, float("nan"), float("inf"), 10**1000):
        with pytest.raises(ValueError, match="max_error"):
            sparse_access_block_encoding(
                scipy_sparse.csr_array((2, 2)), max_error=invalid_budget
            )


def test_sparse_access_exceeds_state_preparation_dimension_limit() -> None:
    """A sparse 32-row operator is built without quadratic state preparation."""

    matrix = scipy_sparse.eye_array(32, format="csr")
    block = sparse_access_block_encoding(matrix, max_error=0)
    assert (block.normalization, block.signal_width, block.system_width) == (
        2,
        13,
        5,
    )
    assert block.error_bound == 0
    with pytest.raises(ValueError, match="budget"):
        state_preparation_block_encoding(matrix)


def test_sparse_access_canonicalizes_duplicate_coordinates() -> None:
    """Duplicate COO entries are summed before color and value assignment."""

    matrix = scipy_sparse.coo_array(
        (np.array([0.25, 0.25]), (np.array([1, 1]), np.array([0, 0]))),
        shape=(2, 2),
    )
    block = sparse_access_block_encoding(matrix)
    assert block.normalization == 1.0
    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    state = Statevector.from_instruction(
        to_circuit(sparse_access_two, target=Qiskit(), block=block)
    ).data
    np.testing.assert_allclose(state[[0, 32]], (0, 0.5), atol=1e-14)


def test_sparse_access_multibit_row_and_column_indices() -> None:
    """Two-bit row and column lookups preserve the matrix index order."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    matrix = scipy_sparse.csr_array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    block = sparse_access_block_encoding(matrix, max_error=0)
    assert (block.normalization, block.signal_width, block.system_width) == (2, 7, 2)
    state = Statevector.from_instruction(
        to_circuit(sparse_access_four_basis_three, target=Qiskit(), block=block)
    ).data
    np.testing.assert_allclose(state[[0, 128, 256, 384]], (0.5, 0, 0, 0), atol=1e-14)


@pytest.mark.backend_slow
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
def test_sparse_access_executes_on_supported_backends(
    module_name: str, target_name: str, endpoint_name: str, dependency: str
) -> None:
    """The same later-bound source runs on all supported execution targets."""

    pytest.importorskip(dependency)
    module = importlib.import_module(module_name)
    endpoint = getattr(module, endpoint_name)()
    target = getattr(module, target_name)()
    seed = 0 if endpoint_name == "BraketLocal" else 2
    block = sparse_access_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    artifact = qm.compile(sparse_access_sample, to=target, block=block)
    counts = qm.run(artifact, on=endpoint, shots=256, seed=seed).result.values
    assert 35 <= counts.get((0, 0, 0, 0, 0, 1), 0) <= 95
    state = sparse_access_two.bind(block=block)
    assert (
        abs(qm.expectation(state, qm.Z(5), on=endpoint, shots=256, seed=seed).value)
        <= 0.2
    )
    signed = sparse_access_block_encoding(
        scipy_sparse.diags_array([1, -1], dtype=float)
    )
    phase_artifact = qm.compile(
        sparse_access_phase_interference, to=target, block=signed
    )
    phase_counts = qm.run(
        phase_artifact, on=endpoint, shots=256, seed=seed
    ).result.values
    assert 35 <= phase_counts.get((0, 0, 0, 0, 0, 0, 1), 0) <= 95


@pytest.mark.hugr
def test_sparse_access_persists_and_compiles_to_hugr() -> None:
    """HUGR compiles a simple block and rejects unsupported value controls."""

    from qamomile.errors import CapabilityError

    pytest.importorskip("hugr")
    from qamomile.hugr import Hugr

    block = sparse_access_block_encoding(scipy_sparse.csr_array([[0, 1], [1, 0]]))
    program = sparse_access_sample.specialize(block=block)
    restored = qm.Program.from_bytes(program.to_bytes())
    assert restored.to_bytes() == program.to_bytes()
    assert qm.compile(restored, to=Hugr())
    complex_block = sparse_access_block_encoding(
        scipy_sparse.csr_array([[0, 1j], [0.5, -0.25]])
    )
    with pytest.raises(CapabilityError, match="Q8660"):
        qm.compile(sparse_access_complex_sample, to=Hugr(), block=complex_block)
