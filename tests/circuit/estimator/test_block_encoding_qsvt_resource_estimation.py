"""Resource regressions for catalogued block encodings and QSVT."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import ValidationError
from tests.circuit.qkernel_catalog import QKERNEL_BY_ID, entries_with_tag

_ENCODING_EXPECTATIONS = {
    "identity": (1, 2, 0, 0),
    "pauli": (1, 1, 5, 5),
    "ising_z": (2, 2, 13, 12),
    "periodic_shift": (2, 2, 15, 14),
    "recursive_lcu": (3, 2, 16, 15),
}


def _shift_matrix(system_width: int, offset: int) -> np.ndarray:
    """Return a periodic displacement in computational-basis order.

    Args:
        system_width (int): System-register width in qubits.
        offset (int): Signed modular basis displacement.

    Returns:
        np.ndarray: Dense periodic-shift matrix.
    """
    dimension = 1 << system_width
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis in range(dimension):
        matrix[(basis + offset) % dimension, basis] = 1.0
    return matrix


_ISING_Z_MATRIX = np.diag(
    [
        0.5
        + (1.0 if basis & 1 == 0 else -1.0)
        - 0.25j * (1.0 if basis & 2 == 0 else -1.0)
        for basis in range(4)
    ]
)
_ENCODED_MATRICES = {
    "identity": np.eye(4, dtype=np.complex128),
    "pauli": np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
    "ising_z": _ISING_Z_MATRIX,
    "periodic_shift": (
        _shift_matrix(2, -1)
        - 2.0 * np.eye(4, dtype=np.complex128)
        + _shift_matrix(2, 1)
    ),
    "recursive_lcu": (0.75 * np.eye(4, dtype=np.complex128) - 0.5j * _ISING_Z_MATRIX),
}


@qmc.qkernel
def _measured_qsvt_consumer(
    encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
    phase_count: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply QSVT and measure its public registers for backend emission.

    Args:
        encoding (qmc.LCUBlockEncoding): Static block encoding to transform.
        phases (qmc.Vector[qmc.Float]): Projector-rotation phase vector.
        phase_count (qmc.UInt): Number of phase entries to consume.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.qsvt(
        signal,
        system,
        phases,
        encoding,
        phase_count=phase_count,
    )
    return qmc.measure(signal), qmc.measure(system)


@pytest.mark.parametrize(
    "entry",
    entries_with_tag("block_encoding"),
    ids=lambda entry: entry.id,
)
def test_catalogued_block_encoding_logical_resources(entry: Any) -> None:
    """Every public encoding family has a stable logical resource canary."""
    name = entry.id.removeprefix("block_encoding_")
    signal_width, system_width, gates, depth = _ENCODING_EXPECTATIONS[name]

    estimate = entry.qkernel.estimate_resources(inputs=entry.minimum_inputs())

    assert estimate.qubits == signal_width + system_width
    assert estimate.width.allocated_qubits == signal_width + system_width
    assert estimate.width.peak_qubits == signal_width + system_width
    assert estimate.gates.total == gates
    assert estimate.depth.depth == depth
    assert estimate.assumptions == ()
    assert estimate.parameters == {}


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 8])
@pytest.mark.parametrize(
    "entry",
    entries_with_tag("qsvt"),
    ids=lambda entry: entry.id,
)
def test_qsvt_resources_follow_query_and_projector_formula(
    entry: Any,
    degree: int,
) -> None:
    """Degree d uses d encodings and d + 1 projector rotations.

    Args:
        entry (Any): Catalogued QSVT consumer and fixed block encoding.
        degree (int): Singular-value polynomial degree, equal to the number of
            block-encoding or inverse-block-encoding queries.
    """
    name = entry.id.removeprefix("qsvt_")
    encoding_entry = QKERNEL_BY_ID[f"block_encoding_{name}"]
    encoding = entry.fixed_inputs["encoding"]
    phase_count = degree + 1

    encoding_estimate = encoding_entry.qkernel.estimate_resources(
        inputs=encoding_entry.minimum_inputs()
    )
    projector_estimate = entry.qkernel.estimate_resources(
        inputs={**entry.fixed_inputs, "phase_count": 1}
    )
    estimate = entry.qkernel.estimate_resources(
        inputs={**entry.fixed_inputs, "phase_count": phase_count}
    )

    assert estimate.qubits == (
        encoding.num_signal_qubits + encoding.num_system_qubits + 1
    )
    assert estimate.gates.total == (
        degree * encoding_estimate.gates.total
        + phase_count * (2 * encoding.num_signal_qubits + 3)
    )
    for field in dataclasses.fields(estimate.gates):
        actual = getattr(estimate.gates, field.name)
        expected = degree * getattr(
            encoding_estimate.gates, field.name
        ) + phase_count * getattr(projector_estimate.gates, field.name)
        assert actual == expected, field.name
    for field in dataclasses.fields(estimate.depth):
        actual = getattr(estimate.depth, field.name)
        expected = degree * getattr(
            encoding_estimate.depth, field.name
        ) + phase_count * getattr(projector_estimate.depth, field.name)
        assert actual == expected, field.name
    assert estimate.assumptions == ()
    assert estimate.parameters == {}


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        pytest.param({"phase_count": 0}, "Index -1", id="zero-count"),
        pytest.param(
            {"phase_count": 1, "phases": []},
            "Index 0",
            id="empty-phases",
        ),
        pytest.param(
            {"phase_count": 3, "phases": [0.1, 0.2]},
            "Index 2",
            id="short-phases",
        ),
    ],
)
def test_qsvt_resource_estimation_rejects_invalid_phase_extent(
    inputs: dict[str, Any],
    message: str,
) -> None:
    """Resource specialization enforces the compiler's phase bounds.

    Args:
        inputs (dict[str, Any]): Invalid phase-count or vector specialization.
        message (str): Concrete invalid index expected in the diagnostic.
    """
    entry = QKERNEL_BY_ID["qsvt_identity"]

    with pytest.raises(ValidationError, match=message):
        entry.qkernel.estimate_resources(inputs={**entry.fixed_inputs, **inputs})


@pytest.mark.parametrize(
    "entry",
    entries_with_tag("qsvt"),
    ids=lambda entry: entry.id,
)
def test_qsvt_estimated_width_matches_qiskit_emission(
    qiskit_transpiler: Any,
    entry: Any,
) -> None:
    """The logical width agrees with one emitted degree-two QSVT circuit.

    Args:
        qiskit_transpiler (Any): Qiskit transpiler fixture.
        entry (Any): Catalogued QSVT consumer and fixed block encoding.
    """
    degree = 2
    phases = [0.0, -math.pi / 2.0, math.pi / 2.0]
    estimate = entry.qkernel.estimate_resources(
        inputs={**entry.fixed_inputs, "phase_count": degree + 1}
    )
    executable = qiskit_transpiler.transpile(
        _measured_qsvt_consumer,
        bindings={**entry.fixed_inputs, "phase_count": degree + 1},
        parameters=["phases"],
    )

    assert executable.quantum_circuit.num_qubits == estimate.qubits
    assert executable.compiled_quantum[0].parameter_metadata.arrays[
        "phases"
    ].expected_shape == (len(phases),)


@pytest.mark.parametrize("name", _ENCODED_MATRICES)
def test_degree_two_qsvt_matches_even_singular_value_transform(
    qiskit_transpiler: Any,
    name: str,
) -> None:
    """The T2 phase canary transforms every catalogued encoding family.

    For ``B = A / alpha``, the even singular-value transform of
    ``T2(x) = 2 x**2 - 1`` is ``2 B dagger B - I``. The non-Hermitian Pauli
    case is therefore a canary against accidentally computing ``B**2``.

    Args:
        qiskit_transpiler (Any): Qiskit transpiler fixture.
        name (str): Catalog suffix selecting an encoding and target matrix.
    """
    encoding = QKERNEL_BY_ID[f"qsvt_{name}"].fixed_inputs["encoding"]
    phases = [0.0, -math.pi / 2.0, math.pi / 2.0]
    executable = qiskit_transpiler.transpile(
        _measured_qsvt_consumer,
        bindings={
            "encoding": encoding,
            "phases": phases,
            "phase_count": len(phases),
        },
    )

    from qiskit.quantum_info import Operator

    circuit = executable.quantum_circuit.remove_final_measurements(inplace=False)
    unitary = np.asarray(Operator(circuit).data)
    signal_width = encoding.num_signal_qubits
    system_width = encoding.num_system_qubits
    system_dimension = 1 << system_width
    projected_indices = np.arange(system_dimension) << signal_width
    projected_block = unitary[np.ix_(projected_indices, projected_indices)]

    normalized = _ENCODED_MATRICES[name] / encoding.normalization
    expected = 2.0 * normalized.conj().T @ normalized - np.eye(
        system_dimension, dtype=np.complex128
    )
    np.testing.assert_allclose(projected_block, expected, rtol=0.0, atol=1e-8)

    auxiliary_one = np.arange(1 << (signal_width + system_width)) + (
        1 << (signal_width + system_width)
    )
    np.testing.assert_allclose(
        unitary[np.ix_(auxiliary_one, projected_indices)],
        0.0,
        rtol=0.0,
        atol=1e-8,
    )
