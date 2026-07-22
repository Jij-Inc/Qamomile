"""Execution and serialization tests for the QSVT circuit primitive."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.types import UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import (
    EmitError,
    QamomileCompileError,
    ValidationError,
)
from qamomile.linalg import PauliLCU, PeriodicShiftLCU


@qmc.qkernel
def _compile_time_template(
    block_encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply QSVT with a phase count inferred from a bound phase vector.

    Args:
        block_encoding (qmc.LCUBlockEncoding): Static block encoding to apply.
        phases (qmc.Vector[qmc.Float]): Compile-time QSVT phase vector.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
    signal, system = qmc.qsvt(signal, system, phases, block_encoding)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _runtime_sample_template(
    block_encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
    phase_count: qmc.UInt,
    initial_bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Measure the QSVT signal with phases retained as runtime parameters.

    Args:
        block_encoding (qmc.LCUBlockEncoding): Static block encoding to apply.
        phases (qmc.Vector[qmc.Float]): Runtime QSVT phase vector.
        phase_count (qmc.UInt): Compile-time phase-vector length.
        initial_bits (qmc.Vector[qmc.UInt]): Compile-time system basis bits.

    Returns:
        qmc.Vector[qmc.Bit]: Measured signal register.
    """
    signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
    system = qmc.computational_basis_state(system, initial_bits)
    signal, _ = qmc.qsvt(
        signal,
        system,
        phases,
        block_encoding,
        phase_count=phase_count,
    )
    return qmc.measure(signal)


@qmc.qkernel
def _runtime_expval_template(
    block_encoding: qmc.LCUBlockEncoding,
    phases: qmc.Vector[qmc.Float],
    phase_count: qmc.UInt,
    initial_bits: qmc.Vector[qmc.UInt],
    observable: qmc.Observable,
) -> qmc.Float:
    """Estimate a signal observable with runtime QSVT phases.

    Args:
        block_encoding (qmc.LCUBlockEncoding): Static block encoding to apply.
        phases (qmc.Vector[qmc.Float]): Runtime QSVT phase vector.
        phase_count (qmc.UInt): Compile-time phase-vector length.
        initial_bits (qmc.Vector[qmc.UInt]): Compile-time system basis bits.
        observable (qmc.Observable): Signal-register observable to estimate.

    Returns:
        qmc.Float: Expected value of ``observable`` after QSVT.
    """
    signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
    system = qmc.computational_basis_state(system, initial_bits)
    signal, _ = qmc.qsvt(
        signal,
        system,
        phases,
        block_encoding,
        phase_count=phase_count,
    )
    return qmc.expval(signal, observable)


@qmc.qkernel
def _two_encoding_template(
    first_encoding: qmc.LCUBlockEncoding,
    second_encoding: qmc.LCUBlockEncoding,
    first_phases: qmc.Vector[qmc.Float],
    second_phases: qmc.Vector[qmc.Float],
) -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
]:
    """Apply independently bound QSVT instances in one serialized template.

    Args:
        first_encoding (qmc.LCUBlockEncoding): First static block encoding.
        second_encoding (qmc.LCUBlockEncoding): Second static block encoding.
        first_phases (qmc.Vector[qmc.Float]): First QSVT phase vector.
        second_phases (qmc.Vector[qmc.Float]): Second QSVT phase vector.

    Returns:
        tuple[qmc.Vector[qmc.Bit], ...]: Measured signal and system registers
            for both transformations.
    """
    first_signal = qmc.qubit_array(
        first_encoding.num_signal_qubits,
        "first_signal",
    )
    first_system = qmc.qubit_array(
        first_encoding.num_system_qubits,
        "first_system",
    )
    second_signal = qmc.qubit_array(
        second_encoding.num_signal_qubits,
        "second_signal",
    )
    second_system = qmc.qubit_array(
        second_encoding.num_system_qubits,
        "second_system",
    )
    first_signal, first_system = qmc.qsvt(
        first_signal,
        first_system,
        first_phases,
        first_encoding,
    )
    second_signal, second_system = qmc.qsvt(
        second_signal,
        second_system,
        second_phases,
        second_encoding,
    )
    return (
        qmc.measure(first_signal),
        qmc.measure(first_system),
        qmc.measure(second_signal),
        qmc.measure(second_system),
    )


def _periodic_encoding(
    coefficients: dict[int, complex],
    system_width: int,
) -> qmc.PeriodicShiftLCUBlockEncoding:
    """Build a one-axis periodic-shift block encoding.

    Args:
        coefficients (dict[int, complex]): Offset coefficients.
        system_width (int): System-register width in qubits.

    Returns:
        qmc.PeriodicShiftLCUBlockEncoding: Exact periodic LCU descriptor.
    """
    lcu = PeriodicShiftLCU.from_coefficients(
        coefficients,
        register_sizes=(system_width,),
    )
    return qmc.periodic_shift_lcu_block_encoding(lcu)


def _shift_matrix(system_width: int, offset: int = 1) -> np.ndarray:
    """Return a periodic increment matrix in computational-basis order.

    Args:
        system_width (int): System-register width in qubits.
        offset (int): Modular basis displacement.

    Returns:
        np.ndarray: Dense periodic-shift matrix.
    """
    dimension = 1 << system_width
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis in range(dimension):
        matrix[(basis + offset) % dimension, basis] = 1.0
    return matrix


def _two_term_encoding_unitary(
    identity_weight: complex,
    shift_weight: complex,
    system_width: int,
) -> np.ndarray:
    """Return the dense unitary of a two-term periodic LCU encoding.

    Qiskit orders the first allocated signal qubit as the least-significant
    tensor factor, so system operators appear to the left of signal operators.

    Args:
        identity_weight (complex): Coefficient of the identity shift.
        shift_weight (complex): Coefficient of the one-step shift.
        system_width (int): System-register width in qubits.

    Returns:
        np.ndarray: Dense PREPARE-SELECT-PREPARE-dagger unitary.
    """
    normalization = abs(identity_weight) + abs(shift_weight)
    cosine = math.sqrt(abs(identity_weight) / normalization)
    sine = math.sqrt(abs(shift_weight) / normalization)
    prepare = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.complex128,
    )
    signal_zero = np.diag([1.0, 0.0]).astype(np.complex128)
    signal_one = np.diag([0.0, 1.0]).astype(np.complex128)
    system_identity = np.eye(1 << system_width, dtype=np.complex128)
    selector = np.kron(
        identity_weight / abs(identity_weight) * system_identity,
        signal_zero,
    ) + np.kron(
        shift_weight / abs(shift_weight) * _shift_matrix(system_width),
        signal_one,
    )
    preparation = np.kron(system_identity, prepare)
    return preparation.conj().T @ selector @ preparation


def _four_term_pauli_encoding() -> tuple[qmc.PauliLCUBlockEncoding, np.ndarray]:
    """Build a signal-width-two Pauli encoding and its dense unitary.

    Four equal positive coefficients make the Möttönen PREPARE apply
    ``RY(pi / 2)`` to each signal qubit. This is ``H Z`` rather than ``H`` on
    each qubit, so the complete block-encoding unitary includes those signs.

    Returns:
        tuple[qmc.PauliLCUBlockEncoding, np.ndarray]: Static descriptor and
            dense PREPARE-SELECT-PREPARE-dagger unitary in Qiskit order.
    """
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    matrix = (
        np.eye(4, dtype=np.complex128)
        + np.kron(identity, pauli_x)
        + np.kron(identity, pauli_z)
        + np.kron(pauli_x, identity)
    )
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    pauli_matrices = {
        qm_o.Pauli.X: pauli_x,
        qm_o.Pauli.Y: np.asarray(
            [[0.0, -1j], [1j, 0.0]],
            dtype=np.complex128,
        ),
        qm_o.Pauli.Z: pauli_z,
    }
    signal_dimension = 4
    selector = np.zeros((16, 16), dtype=np.complex128)
    for index, term in enumerate(lcu.terms):
        factors = [identity.copy() for _ in range(lcu.num_qubits)]
        for operator in term.operators:
            factors[operator.index] = pauli_matrices[operator.pauli]
        word = np.asarray([[1.0]], dtype=np.complex128)
        for factor in reversed(factors):
            word = np.kron(word, factor)
        signal_projector = np.zeros(
            (signal_dimension, signal_dimension),
            dtype=np.complex128,
        )
        signal_projector[index, index] = 1.0
        selector += np.kron(
            term.coefficient / abs(term.coefficient) * word,
            signal_projector,
        )

    ry_pi_over_two = np.asarray(
        [[1.0, -1.0], [1.0, 1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    prepare = np.kron(ry_pi_over_two, ry_pi_over_two)
    preparation = np.kron(np.eye(4, dtype=np.complex128), prepare)
    return encoding, preparation.conj().T @ selector @ preparation


def _reference_qsvt(
    encoding_unitary: np.ndarray,
    phases: list[float],
    signal_dimension: int = 2,
) -> np.ndarray:
    """Compose the documented QSVT sequence as a dense unitary.

    Args:
        encoding_unitary (np.ndarray): Dense block-encoding unitary.
        phases (list[float]): Projector phases in sequence order.
        signal_dimension (int): Dimension of the signal register. Defaults to
            ``2`` for one signal qubit.

    Returns:
        np.ndarray: Dense QSVT unitary in Qiskit qubit order.
    """
    dimension = encoding_unitary.shape[0]
    system_dimension = dimension // signal_dimension

    def projector_rotation(phase: float) -> np.ndarray:
        """Return one dense all-zero projector rotation.

        Args:
            phase (float): Projector phase angle in radians.

        Returns:
            np.ndarray: Dense rotation in Qiskit qubit order.
        """
        signal = np.eye(signal_dimension, dtype=np.complex128) * np.exp(-1j * phase)
        signal[0, 0] = np.exp(1j * phase)
        return np.kron(np.eye(system_dimension), signal)

    transformed = projector_rotation(phases[0])
    for index, phase in enumerate(phases[1:]):
        step = encoding_unitary if index % 2 == 0 else encoding_unitary.conj().T
        transformed = projector_rotation(phase) @ step @ transformed
    return transformed


def _all_zero(value: Any) -> bool:
    """Return whether a backend-independent sampled value is all zero.

    Args:
        value (Any): Scalar or nested tuple returned by a backend sampler.

    Returns:
        bool: Whether every sampled bit is zero.
    """
    if isinstance(value, tuple):
        return all(_all_zero(item) for item in value)
    return int(value) == 0


def _zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the empirical all-zero probability from sample counts.

    Args:
        results (list[tuple[Any, int]]): Sampled values paired with counts.

    Returns:
        float: Fraction of shots whose sampled value is all zero.
    """
    shots = sum(count for _, count in results)
    return sum(count for value, count in results if _all_zero(value)) / shots


def _zero_projector_observable(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the projector onto an all-zero register as a Hamiltonian.

    Args:
        num_qubits (int): Positive register width.

    Returns:
        qm_o.Hamiltonian: Product of ``(I + Z_i) / 2`` over every qubit.
    """
    projector = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    identity = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    for index in range(num_qubits):
        projector = projector * (0.5 * (identity + qm_o.Z(index)))
    return projector


def _assert_serialized_runtime_qsvt_results(
    sdk_transpiler: Any,
    encoding: qmc.LCUBlockEncoding,
    phases: list[float],
    initial_bits: list[int],
    observable: qm_o.Hamiltonian,
    expected_zero_probability: float,
    expected_observable: float,
) -> None:
    """Execute serialized runtime QSVT sampling and estimation on one SDK.

    Args:
        sdk_transpiler (Any): Cross-backend transpiler fixture case.
        encoding (qmc.LCUBlockEncoding): Static block encoding to bind.
        phases (list[float]): Runtime QSVT phase values.
        initial_bits (list[int]): Compile-time system basis bits.
        observable (qm_o.Hamiltonian): Signal observable to estimate.
        expected_zero_probability (float): Expected all-zero signal probability.
        expected_observable (float): Expected signal-observable value.

    Returns:
        None: Results and serialization round trips are asserted in place.
    """
    sample_payload = serialize(_runtime_sample_template)
    expval_payload = serialize(_runtime_expval_template)
    sample_template = deserialize(sample_payload)
    expval_template = deserialize(expval_payload)
    compile_bindings = {
        "block_encoding": encoding,
        "phase_count": len(phases),
        "initial_bits": initial_bits,
    }
    sample_executable = sdk_transpiler.transpiler.transpile(
        sample_template,
        bindings=compile_bindings,
        parameters=["phases"],
    )
    expval_executable = sdk_transpiler.transpiler.transpile(
        expval_template,
        bindings={**compile_bindings, "observable": observable},
        parameters=["phases"],
    )

    shots = 2048
    executor = sdk_transpiler.transpiler.executor()
    sample_result = sample_executable.sample(
        executor,
        bindings={"phases": phases},
        shots=shots,
    ).result()
    observed = float(
        expval_executable.run(executor, bindings={"phases": phases}).result()
    )

    sampling_tolerance = (
        6.0
        * math.sqrt(
            expected_zero_probability * (1.0 - expected_zero_probability) / shots
        )
        + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero_probability,
        abs=sampling_tolerance,
    )
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(expected_observable, abs=expval_tolerance)
    assert serialize(sample_template) == sample_payload
    assert serialize(expval_template) == expval_payload


@pytest.mark.parametrize(
    "phases",
    [
        pytest.param([0.37], id="one-phase"),
        pytest.param([0.0, 0.61], id="two-phases"),
        pytest.param([math.pi, -0.43, 2.0 * math.pi], id="three-phases"),
        pytest.param([0.0, math.pi, 2.0 * math.pi, -0.29], id="four-phases"),
    ],
)
def test_qsvt_exact_sequence_for_one_through_four_phases(
    qiskit_transpiler: Any,
    phases: list[float],
) -> None:
    """Qiskit's exact unitary matches the alternating dense reference."""
    from qiskit.quantum_info import Operator

    coefficient_phase = 0.41
    coefficient = np.exp(1j * coefficient_phase)
    encoding = _periodic_encoding({1: coefficient}, system_width=2)
    executable = qiskit_transpiler.transpile(
        _compile_time_template,
        bindings={"block_encoding": encoding, "phases": phases},
    )
    circuit = executable.compiled_quantum[0].circuit.remove_final_measurements(
        inplace=False
    )
    actual = np.asarray(Operator(circuit).data)

    signal_phase = np.diag(
        [
            np.exp(1j * sum(phases)),
            np.exp(-1j * sum(phases)),
        ]
    )
    if len(phases) % 2 == 0:
        system = coefficient * _shift_matrix(system_width=2)
    else:
        system = np.eye(4, dtype=np.complex128)
    expected = np.kron(system, signal_phase)
    logical_dimension = expected.shape[0]

    np.testing.assert_allclose(
        actual[:logical_dimension, :logical_dimension],
        expected,
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        actual[logical_dimension:, :logical_dimension],
        0.0,
        rtol=0.0,
        atol=1e-8,
    )


def test_qiskit_preserves_named_projector_rotation_with_native_mcx(
    qiskit_transpiler: Any,
) -> None:
    """A width-three projector rotation stays named and uses native MCX gates."""
    from qiskit.circuit.library import MCXGate

    identity = qmc.identity_block_encoding(1)
    diagonal = qmc.ising_z_block_encoding({(): 0.3, (0,): -0.8}, 1)
    inner = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(0.6 + 0.2j, identity),
            qmc.LCUBlockEncodingTerm(-0.4, diagonal),
        )
    )
    encoding = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(-0.7j, inner),
            qmc.LCUBlockEncodingTerm(0.25, identity),
        )
    )
    assert encoding.num_signal_qubits == 3

    executable = qiskit_transpiler.transpile(
        _compile_time_template,
        bindings={"block_encoding": encoding, "phases": [0.37]},
    )
    top_level_operations = [
        instruction.operation
        for instruction in executable.quantum_circuit.data
        if instruction.operation.name != "measure"
    ]
    assert [operation.name for operation in top_level_operations] == [
        "qsvt_projector_phase_rotation"
    ]

    decomposed = executable.quantum_circuit.decompose(reps=2)
    operation_counts = decomposed.count_ops()
    assert operation_counts["x"] == 6
    assert operation_counts["mcx"] == 2
    assert operation_counts["rz"] == 1
    mcx_operations = [
        instruction.operation
        for instruction in decomposed.data
        if instruction.operation.name == "mcx"
    ]
    assert len(mcx_operations) == 2
    assert all(isinstance(operation, MCXGate) for operation in mcx_operations)


def test_serialized_template_binds_periodic_encoding_to_generic_slot(
    qiskit_transpiler: Any,
) -> None:
    """A received template resolves ``block_encoding`` and runtime phases."""
    payload = serialize(_runtime_sample_template)
    restored = deserialize(payload)
    encoding = _periodic_encoding({0: -2.0, -1: 1.0, 1: 1.0}, system_width=2)

    assert restored.block.static_bindings[0].name == "block_encoding"
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.lcu_block_encoding"
    )
    executable = qiskit_transpiler.transpile(
        restored,
        bindings={
            "block_encoding": encoding,
            "phase_count": 3,
            "initial_bits": [1, 0],
        },
        parameters=["phases"],
    )
    phase_metadata = executable.compiled_quantum[0].parameter_metadata.arrays["phases"]
    result = executable.sample(
        qiskit_transpiler.executor(),
        bindings={"phases": [0.0, 0.0, 0.0]},
        shots=16,
    ).result()

    assert phase_metadata.expected_shape == (3,)
    assert result.results == [((0, 0), 16)]
    assert serialize(restored) == payload


def test_serialized_template_binds_two_independent_encoding_slots(
    qiskit_transpiler: Any,
) -> None:
    """Two differently sized static descriptors coexist after round-trip."""
    payload = serialize(_two_encoding_template)
    restored = deserialize(payload)
    first_encoding = _periodic_encoding({0: 1.0, 1: 1.0}, system_width=1)
    second_encoding = _periodic_encoding(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        system_width=2,
    )

    assert [slot.name for slot in restored.block.static_bindings] == [
        "first_encoding",
        "second_encoding",
    ]
    executable = qiskit_transpiler.transpile(
        restored,
        bindings={
            "first_encoding": first_encoding,
            "second_encoding": second_encoding,
            "first_phases": [0.2, -0.1],
            "second_phases": [0.3, -0.4, 0.5],
        },
    )

    assert executable.quantum_circuit.num_qubits == 8
    assert serialize(restored) == payload


@pytest.mark.parametrize("system_width", [1, 2])
@pytest.mark.parametrize(
    ("seed", "phase_count"),
    [
        pytest.param(0, 1, id="seed-0-one-phase"),
        pytest.param(1, 2, id="seed-1-two-phases"),
        pytest.param(2, 3, id="seed-2-three-phases"),
        pytest.param(42, 4, id="seed-42-four-phases"),
    ],
)
def test_serialized_qsvt_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    system_width: int,
    seed: int,
    phase_count: int,
) -> None:
    """Random phases, weights, states, and widths execute on every backend."""
    rng = np.random.default_rng(seed + 101 * system_width)
    identity_weight = rng.uniform(0.4, 1.2) * np.exp(
        1j * rng.uniform(-math.pi, math.pi)
    )
    shift_weight = rng.uniform(0.4, 1.2) * np.exp(1j * rng.uniform(-math.pi, math.pi))
    phases = rng.uniform(-math.pi, math.pi, size=phase_count).tolist()
    initial_bits = rng.integers(0, 2, size=system_width).astype(int).tolist()
    initial_basis = sum(bit << index for index, bit in enumerate(initial_bits))

    encoding = _periodic_encoding(
        {0: identity_weight, 1: shift_weight},
        system_width,
    )
    encoding_unitary = _two_term_encoding_unitary(
        identity_weight,
        shift_weight,
        system_width,
    )
    qsvt_unitary = _reference_qsvt(encoding_unitary, phases)
    initial_state = np.zeros(1 << (system_width + 1), dtype=np.complex128)
    initial_state[2 * initial_basis] = 1.0
    expected_state = qsvt_unitary @ initial_state
    expected_zero = float(np.sum(np.abs(expected_state[0::2]) ** 2))
    pauli_y = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    signal_y = np.kron(np.eye(1 << system_width), pauli_y)
    expected_y = float(np.real(np.vdot(expected_state, signal_y @ expected_state)))

    _assert_serialized_runtime_qsvt_results(
        sdk_transpiler,
        encoding,
        phases,
        initial_bits,
        qm_o.Y(0),
        expected_zero,
        expected_y,
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_multisignal_repeated_qsvt_executes_on_every_sdk(
    sdk_transpiler: Any,
    seed: int,
) -> None:
    """Two signal qubits and repeated QSVT pairs match a dense reference."""
    rng = np.random.default_rng(seed + 709)
    phases = rng.uniform(-math.pi, math.pi, size=5).tolist()
    initial_bits = rng.integers(0, 2, size=2).astype(int).tolist()
    initial_basis = sum(bit << index for index, bit in enumerate(initial_bits))
    encoding, encoding_unitary = _four_term_pauli_encoding()
    qsvt_unitary = _reference_qsvt(
        encoding_unitary,
        phases,
        signal_dimension=4,
    )
    initial_state = np.zeros(16, dtype=np.complex128)
    initial_state[4 * initial_basis] = 1.0
    expected_state = qsvt_unitary @ initial_state
    expected_zero = float(np.sum(np.abs(expected_state[0::4]) ** 2))
    pauli_y = np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    signal_y = np.kron(np.eye(2, dtype=np.complex128), pauli_y)
    full_signal_y = np.kron(np.eye(4, dtype=np.complex128), signal_y)
    expected_y = float(np.real(np.vdot(expected_state, full_signal_y @ expected_state)))

    _assert_serialized_runtime_qsvt_results(
        sdk_transpiler,
        encoding,
        phases,
        initial_bits,
        qm_o.Y(0),
        expected_zero,
        expected_y,
    )


def test_serialized_qsvt_with_recursive_lcu_executes_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A nested descriptor matches its analytic projector probability.

    On the initial system state ``|0>``, every child is diagonal and the
    recursive leading block has the scalar eigenvalue assembled below. For
    phases ``(0, phi, 0)``, the all-zero signal probability is
    ``1 - 4 sin(phi)^2 t (1 - t)``, where ``t`` is the squared magnitude of
    that normalized eigenvalue.
    """
    diagonal_identity_coefficient = 0.3
    diagonal_z_coefficient = -0.8
    inner_identity_coefficient = 0.6 + 0.2j
    inner_diagonal_coefficient = -0.4
    outer_inner_coefficient = -0.7j
    outer_identity_coefficient = 0.25

    identity = qmc.identity_block_encoding(1)
    diagonal = qmc.ising_z_block_encoding(
        {(): diagonal_identity_coefficient, (0,): diagonal_z_coefficient},
        1,
    )
    inner = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(inner_identity_coefficient, identity),
            qmc.LCUBlockEncodingTerm(inner_diagonal_coefficient, diagonal),
        )
    )
    encoding = qmc.lcu_block_encoding(
        (
            qmc.LCUBlockEncodingTerm(outer_inner_coefficient, inner),
            qmc.LCUBlockEncodingTerm(outer_identity_coefficient, identity),
        )
    )
    middle_phase = 0.63
    phases = [0.0, middle_phase, 0.0]
    diagonal_eigenvalue = diagonal_identity_coefficient + diagonal_z_coefficient
    inner_eigenvalue = (
        inner_identity_coefficient + inner_diagonal_coefficient * diagonal_eigenvalue
    )
    outer_eigenvalue = (
        outer_inner_coefficient * inner_eigenvalue + outer_identity_coefficient
    )
    normalized_squared = abs(outer_eigenvalue / encoding.normalization) ** 2
    expected_zero = 1.0 - (
        4.0
        * math.sin(middle_phase) ** 2
        * normalized_squared
        * (1.0 - normalized_squared)
    )
    assert encoding.num_signal_qubits == 3
    _assert_serialized_runtime_qsvt_results(
        sdk_transpiler,
        encoding,
        phases,
        [0],
        _zero_projector_observable(encoding.num_signal_qubits),
        expected_zero,
        expected_zero,
    )


@pytest.mark.parametrize(
    ("phase_count", "error", "message"),
    [
        pytest.param(True, TypeError, "not bool", id="bool"),
        pytest.param(np.bool_(True), TypeError, "not bool", id="numpy-bool"),
        pytest.param(1.5, TypeError, "integer or qmc.UInt", id="float"),
        pytest.param(0, ValueError, "positive", id="zero"),
        pytest.param(-1, ValueError, "positive", id="negative"),
    ],
)
def test_phase_count_rejects_invalid_host_values(
    phase_count: Any,
    error: type[Exception],
    message: str,
) -> None:
    """Invalid explicit host phase counts fail while the wrapper is traced."""
    with pytest.raises(error, match=message):

        @qmc.qkernel
        def invalid(
            block_encoding: qmc.LCUBlockEncoding,
            phases: qmc.Vector[qmc.Float],
        ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            """Attempt to invoke QSVT with one invalid phase count.

            Args:
                block_encoding (qmc.LCUBlockEncoding): Static block encoding.
                phases (qmc.Vector[qmc.Float]): QSVT phase vector.

            Returns:
                tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured
                    signal and system registers.
            """
            signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
            system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
            signal, system = qmc.qsvt(
                signal,
                system,
                phases,
                block_encoding,
                phase_count=phase_count,
            )
            return qmc.measure(signal), qmc.measure(system)

        _ = invalid.block


@pytest.mark.parametrize(
    "constant",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ],
)
def test_phase_count_rejects_boolean_uint_constant(constant: Any) -> None:
    """A boolean constant wrapped by a UInt handle remains invalid."""
    wrapped = qmc.UInt(
        value=Value(type=UIntType(), name="wrapped_bool").with_const(constant)
    )

    with pytest.raises(TypeError, match="phase_count.*not bool"):

        @qmc.qkernel
        def invalid(
            block_encoding: qmc.LCUBlockEncoding,
            phases: qmc.Vector[qmc.Float],
        ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            """Attempt to invoke QSVT with a boolean UInt phase count.

            Args:
                block_encoding (qmc.LCUBlockEncoding): Static block encoding.
                phases (qmc.Vector[qmc.Float]): QSVT phase vector.

            Returns:
                tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured
                    signal and system registers.
            """
            signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
            system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
            signal, system = qmc.qsvt(
                signal,
                system,
                phases,
                block_encoding,
                phase_count=wrapped,
            )
            return qmc.measure(signal), qmc.measure(system)

        _ = invalid.block


@pytest.mark.parametrize(
    "invalid_phases",
    [
        pytest.param([0.1], id="list"),
        pytest.param(np.asarray([0.1], dtype=np.float64), id="numpy-array"),
    ],
)
def test_phases_reject_non_vector_host_values(invalid_phases: Any) -> None:
    """Host sequences fail with the public phase argument and expected type."""
    with pytest.raises(TypeError) as error:

        @qmc.qkernel
        def invalid(
            block_encoding: qmc.LCUBlockEncoding,
        ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            """Attempt to pass a host sequence directly to QSVT.

            Args:
                block_encoding (qmc.LCUBlockEncoding): Static block encoding.

            Returns:
                tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured
                    signal and system registers.
            """
            signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
            system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
            signal, system = qmc.qsvt(
                signal,
                system,
                invalid_phases,
                block_encoding,
                phase_count=1,
            )
            return qmc.measure(signal), qmc.measure(system)

        _ = invalid.block

    message = str(error.value)
    assert "phases" in message
    assert "qmc.Vector[qmc.Float]" in message
    assert type(invalid_phases).__name__ in message


def test_phases_reject_uint_vector_handle() -> None:
    """A UInt vector fails with the public phase argument and expected type."""
    with pytest.raises(TypeError) as error:

        @qmc.qkernel
        def invalid(
            block_encoding: qmc.LCUBlockEncoding,
            phases: qmc.Vector[qmc.UInt],
        ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
            """Attempt to pass an integer vector to QSVT.

            Args:
                block_encoding (qmc.LCUBlockEncoding): Static block encoding.
                phases (qmc.Vector[qmc.UInt]): Invalid integer phase vector.

            Returns:
                tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured
                    signal and system registers.
            """
            signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
            system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
            signal, system = qmc.qsvt(
                signal,
                system,
                phases,
                block_encoding,
                phase_count=1,
            )
            return qmc.measure(signal), qmc.measure(system)

        _ = invalid.block

    message = str(error.value)
    assert "phases" in message
    assert "qmc.Vector[qmc.Float]" in message


def test_phase_count_accepts_numpy_integer() -> None:
    """A positive NumPy integer phase count traces as a host constant."""

    @qmc.qkernel
    def valid(
        block_encoding: qmc.LCUBlockEncoding,
        phases: qmc.Vector[qmc.Float],
    ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply QSVT with a positive NumPy integer phase count.

        Args:
            block_encoding (qmc.LCUBlockEncoding): Static block encoding.
            phases (qmc.Vector[qmc.Float]): QSVT phase vector.

        Returns:
            tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
                system registers.
        """
        signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
        signal, system = qmc.qsvt(
            signal,
            system,
            phases,
            block_encoding,
            phase_count=np.int64(1),
        )
        return qmc.measure(signal), qmc.measure(system)

    _ = valid.block


def test_bound_phase_count_must_be_positive() -> None:
    """A symbolic count bound to zero is rejected during specialization."""
    encoding = _periodic_encoding({1: 1.0}, system_width=1)

    with pytest.raises(ValueError, match="phase_count must be positive"):
        _runtime_sample_template.build(
            block_encoding=encoding,
            phases=[0.1],
            phase_count=0,
            initial_bits=[0],
        )


def test_serialized_runtime_phases_require_compile_time_phase_count(
    qiskit_transpiler: Any,
) -> None:
    """Runtime phases without a count fail early with both argument names."""
    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    restored = deserialize(serialize(_compile_time_template))

    with pytest.raises(QamomileCompileError) as error:
        qiskit_transpiler.transpile(
            restored,
            bindings={"block_encoding": encoding},
            parameters=["phases"],
        )

    assert not isinstance(error.value, EmitError)
    message = str(error.value)
    assert "phases" in message
    assert "phase_count" in message
    assert "compile time" in message


def test_serialized_zero_phase_count_has_clear_bounds_diagnostic(
    qiskit_transpiler: Any,
) -> None:
    """A received zero count names phases and phase_count in its bounds error."""
    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    restored = deserialize(serialize(_runtime_sample_template))

    with pytest.raises(ValidationError) as error:
        qiskit_transpiler.transpile(
            restored,
            bindings={
                "block_encoding": encoding,
                "phases": [0.1],
                "phase_count": 0,
                "initial_bits": [0],
            },
        )

    message = str(error.value)
    assert "out of range" in message
    assert "phases" in message
    assert "phase_count" in message


def test_serialized_compile_time_phases_select_explicit_prefix(
    qiskit_transpiler: Any,
) -> None:
    """A received template slices a compile-time phase-vector binding."""
    from qiskit.quantum_info import Operator

    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    common_bindings = {
        "block_encoding": encoding,
        "phase_count": 2,
        "initial_bits": [0],
    }
    restored = deserialize(serialize(_runtime_sample_template))
    prefix_executable = qiskit_transpiler.transpile(
        restored,
        bindings={**common_bindings, "phases": [0.1, 0.2, 0.3]},
    )
    exact_executable = qiskit_transpiler.transpile(
        _runtime_sample_template,
        bindings={**common_bindings, "phases": [0.1, 0.2]},
    )
    prefix_circuit = prefix_executable.compiled_quantum[
        0
    ].circuit.remove_final_measurements(inplace=False)
    exact_circuit = exact_executable.compiled_quantum[
        0
    ].circuit.remove_final_measurements(inplace=False)

    np.testing.assert_allclose(
        Operator(prefix_circuit).data,
        Operator(exact_circuit).data,
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("phase_count", [2, 5])
def test_serialized_compile_time_phases_reject_short_vector(
    qiskit_transpiler: Any,
    phase_count: int,
) -> None:
    """A compile-time phase prefix must cover its explicit phase count.

    Args:
        qiskit_transpiler (Any): Qiskit transpiler fixture.
        phase_count (int): Explicit prefix length exceeding the bound vector.
    """
    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    restored = deserialize(serialize(_runtime_sample_template))

    with pytest.raises(ValidationError) as error:
        qiskit_transpiler.transpile(
            restored,
            bindings={
                "block_encoding": encoding,
                "phases": [0.1],
                "phase_count": phase_count,
                "initial_bits": [0],
            },
        )

    message = str(error.value)
    assert f"Index {phase_count - 1} is out of range" in message
    assert "phases" in message
    assert f"at least {phase_count}" in message


@pytest.mark.parametrize(
    ("phases", "message"),
    [
        pytest.param([0.1], "Missing parameter bindings", id="too-short"),
        pytest.param(
            [0.1, 0.2, 0.3],
            "beyond the emitted shape",
            id="too-long",
        ),
    ],
)
def test_runtime_phase_length_must_equal_phase_count(
    qiskit_transpiler: Any,
    phases: list[float],
    message: str,
) -> None:
    """Runtime phase vectors must match the specialized executable ABI.

    Args:
        qiskit_transpiler (Any): Qiskit transpiler fixture.
        phases (list[float]): Runtime phase values with an invalid length.
        message (str): Expected binding diagnostic fragment.
    """
    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    restored = deserialize(serialize(_runtime_sample_template))
    executable = qiskit_transpiler.transpile(
        restored,
        bindings={
            "block_encoding": encoding,
            "phase_count": 2,
            "initial_bits": [0],
        },
        parameters=["phases"],
    )

    with pytest.raises(ValueError, match=message):
        executable.sample(
            qiskit_transpiler.executor(),
            bindings={"phases": phases},
            shots=1,
        )


def test_serialized_empty_phase_vector_has_compile_diagnostic(
    qiskit_transpiler: Any,
) -> None:
    """A received empty phase sequence fails with an array-bound diagnostic."""
    encoding = _periodic_encoding({1: 1.0}, system_width=1)
    restored = deserialize(serialize(_compile_time_template))

    with pytest.raises(ValidationError) as error:
        qiskit_transpiler.transpile(
            restored,
            bindings={"block_encoding": encoding, "phases": []},
        )

    message = str(error.value)
    assert "Index 0 is out of range" in message
    assert "phases" in message
    assert "at least 1" in message
