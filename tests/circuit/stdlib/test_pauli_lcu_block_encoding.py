"""Execution tests for complex Pauli LCU block encodings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.circuit_ir import (
    SELECT_SEMANTIC_KEY,
    CallInstruction,
    lower_circuit_plan,
)
from qamomile.linalg import PauliLCU, PauliLCUTerm

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_DENSE_RNG = np.random.default_rng(19)
DENSE_TWO_QUBIT_MATRIX = (
    _DENSE_RNG.normal(size=(4, 4)) + 1j * _DENSE_RNG.normal(size=(4, 4))
) / 4.0
DENSE_TWO_QUBIT_MATRIX += (0.5 + 0.25j) * np.eye(4, dtype=np.complex128)


def _build_unitary_kernel(lcu: PauliLCU, *, invert: bool = False) -> qmc.QKernel:
    """Build an allocation-only kernel exposing the block-encoding unitary."""
    gate = qmc.pauli_lcu_block_encoding(lcu)
    applied_gate = qmc.inverse(gate) if invert else gate

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Allocate registers and apply the selected block-encoding direction."""
        selection = qmc.qubit_array(
            qmc.pauli_lcu_num_selection_qubits(lcu), "selection"
        )
        system = qmc.qubit_array(lcu.num_qubits, "system")
        selection, _ = applied_gate(selection, system)
        return qmc.measure(selection[0])

    return kernel


def _qiskit_unitary(lcu: PauliLCU, *, invert: bool = False) -> np.ndarray:
    """Transpile a block encoding and return its exact Qiskit unitary."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(_build_unitary_kernel(lcu, invert=invert))
    quantum_circuit = executable.quantum_circuit.remove_final_measurements(
        inplace=False
    )
    return np.asarray(Operator(quantum_circuit).data)


def _top_left_block(unitary: np.ndarray, lcu: PauliLCU) -> np.ndarray:
    """Extract the all-zero-selection block in Qamomile LSB order."""
    selection_width = qmc.pauli_lcu_num_selection_qubits(lcu)
    system_indices = [
        basis_index << selection_width for basis_index in range(1 << lcu.num_qubits)
    ]
    return unitary[np.ix_(system_indices, system_indices)]


def _select_case_fingerprints(
    lcu: PauliLCU,
    *,
    invert: bool = False,
) -> tuple[str, ...]:
    """Lower an LCU composite and return its semantic SELECT fingerprints.

    Args:
        lcu (PauliLCU): Decomposition whose composite should be lowered.
        invert (bool): Whether to lower the explicit inverse implementation.
            Defaults to ``False``.

    Returns:
        tuple[str, ...]: Ordered SHA-256 case fingerprints.
    """
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_build_unitary_kernel(lcu, invert=invert))
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    pending = [lowered.quantum_circuit]
    while pending:
        program = pending.pop()
        for operation in program.operations:
            if not isinstance(operation, CallInstruction):
                continue
            identity = operation.callee.identity
            if identity is not None and identity.key == SELECT_SEMANTIC_KEY:
                fingerprints = identity.arguments.get("case_fingerprints")
                assert isinstance(fingerprints, tuple)
                assert all(isinstance(value, str) for value in fingerprints)
                return fingerprints
            pending.append(operation.callee.body)
    raise AssertionError("Lowered Pauli LCU did not contain a semantic SELECT call.")


def _prepare_basis(
    system: qmc.Vector[qmc.Qubit],
    basis_index: int,
    num_qubits: int,
) -> None:
    """Prepare a little-endian computational-basis system state in place."""
    for qubit in range(num_qubits):
        if (basis_index >> qubit) & 1:
            system[qubit] = qmc.x(system[qubit])


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case."""
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the measured probability of an all-zero output value."""
    total = sum(count for _, count in results)
    zero_count = sum(
        count
        for outcome, count in results
        if all(bit == 0 for bit in _flatten(outcome))
    )
    return zero_count / total


def _flatten(value: Any) -> tuple[int, ...]:
    """Flatten nested scalar/vector measurement values into integer bits."""
    if isinstance(value, tuple):
        return tuple(bit for item in value for bit in _flatten(item))
    return (int(value),)


@pytest.mark.parametrize(
    ("matrix", "case_id"),
    [
        (np.array([[0, 1], [0, 0]], dtype=np.complex128), "non_hermitian"),
        ((-0.25 + 0.75j) * I2, "single_complex_identity"),
        (np.zeros((2, 2), dtype=np.complex128), "zero"),
        (1j * I2 + 0.5 * X + (0.2 - 0.3j) * np.array([[1, 0], [0, -1]]), "three_terms"),
        (DENSE_TWO_QUBIT_MATRIX, "dense_sixteen_terms"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_qiskit_unitary_has_exact_normalized_top_left_block(
    matrix: np.ndarray,
    case_id: str,
) -> None:
    """Every term-count path embeds the complex matrix without phase alignment."""
    del case_id
    lcu = PauliLCU.from_matrix(matrix)
    unitary = _qiskit_unitary(lcu)
    normalization = lcu.alpha if lcu.alpha != 0.0 else 1.0

    np.testing.assert_allclose(
        _top_left_block(unitary, lcu),
        matrix / normalization,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        unitary.conj().T @ unitary,
        np.eye(unitary.shape[0]),
        atol=1e-10,
        rtol=0.0,
    )


def test_inverse_top_left_block_is_adjoint_matrix() -> None:
    """Inverting the composite exposes ``A dagger / alpha`` in its top block."""
    matrix = np.array(
        [[0.2 + 0.1j, 0.7 - 0.4j], [-0.3j, -0.5 + 0.2j]],
        dtype=np.complex128,
    )
    lcu = PauliLCU.from_matrix(matrix)

    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(lcu, invert=True), lcu),
        matrix.conj().T / lcu.alpha,
        atol=1e-10,
        rtol=0.0,
    )


def test_explicit_inverse_select_has_stable_conjugated_case_fingerprints() -> None:
    """Lowered identity-case phases distinguish forward and inverse bodies."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)

    forward = _select_case_fingerprints(lcu)
    repeated = _select_case_fingerprints(lcu)
    adjoint = _select_case_fingerprints(lcu, invert=True)

    assert forward == repeated
    assert len(forward) == len(adjoint) == 2
    assert forward[0] != adjoint[0]
    assert forward[1] == adjoint[1]


def test_selection_width_covers_zero_single_and_non_power_of_two_terms() -> None:
    """The stable two-register ABI retains one signal qubit at small term counts."""
    zero = PauliLCU.from_matrix(np.zeros((2, 2), dtype=np.complex128))
    single = PauliLCU.from_matrix(1j * I2)
    three = PauliLCU.from_matrix(1j * I2 + 0.5 * X + 0.25 * np.diag([1, -1]))

    assert qmc.pauli_lcu_num_selection_qubits(zero) == 1
    assert qmc.pauli_lcu_num_selection_qubits(single) == 1
    assert qmc.pauli_lcu_num_selection_qubits(three) == 2


def test_scalar_lcu_is_rejected_by_circuit_factory() -> None:
    """The linalg scalar domain is explicit rather than silently padded."""
    lcu = PauliLCU.from_matrix(np.array([[1j]]))

    with pytest.raises(ValueError, match="system qubit"):
        qmc.pauli_lcu_block_encoding(lcu)


def test_lcu_helpers_reject_non_lcu_values() -> None:
    """Public circuit helpers reject values outside their method-specific API."""
    with pytest.raises(TypeError, match="PauliLCU"):
        qmc.pauli_lcu_num_selection_qubits(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PauliLCU"):
        qmc.pauli_lcu_block_encoding(object())  # type: ignore[arg-type]


def test_signed_zero_does_not_change_block_compiler_identity() -> None:
    """Equal negative-real terms produce one canonical callable identity."""
    from qamomile.circuit.frontend.qkernel_callable import qkernel_callable_ref

    positive_zero = PauliLCU(1, (PauliLCUTerm(complex(-1.0, 0.0), ()),))
    negative_zero = PauliLCU(1, (PauliLCUTerm(complex(-1.0, -0.0), ()),))

    positive_ref = qkernel_callable_ref(qmc.pauli_lcu_block_encoding(positive_zero))
    negative_ref = qkernel_callable_ref(qmc.pauli_lcu_block_encoding(negative_zero))
    assert positive_ref == negative_ref


def test_composite_rejects_wrong_concrete_register_widths() -> None:
    """Selection and system widths are checked at each specialized call site."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X + 0.25 * np.diag([1, -1]))
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def wrong_selection() -> qmc.Bit:
        """Call the two-index-qubit composite with one selection qubit."""
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        selection, _ = gate(selection, system)
        return qmc.measure(selection[0])

    @qmc.qkernel
    def wrong_system() -> qmc.Bit:
        """Call the one-system-qubit composite with two system qubits."""
        selection = qmc.qubit_array(2, "selection")
        system = qmc.qubit_array(2, "system")
        selection, _ = gate(selection, system)
        return qmc.measure(selection[0])

    with pytest.raises(ValueError, match="requires 2 selection qubits"):
        wrong_selection.build()
    with pytest.raises(ValueError, match="requires 1 system qubits"):
        wrong_system.build()


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_random_complex_two_term_lcu_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    num_qubits: int,
    seed: int,
) -> None:
    """Random phases and widths execute through sampler and estimator paths."""
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(-math.pi, math.pi))
    x_weight = float(rng.uniform(0.2, 1.2))
    identity_weight = np.exp(1j * phase)
    dim = 1 << num_qubits
    x_on_q0 = np.kron(np.eye(dim // 2, dtype=np.complex128), X)
    matrix = identity_weight * np.eye(dim) + x_weight * x_on_q0
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    gate = qmc.pauli_lcu_block_encoding(lcu)
    initial_basis = (seed % (1 << max(0, num_qubits - 1))) << 1

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the random LCU and measure its selection register."""
        selection = qmc.qubit_array(
            qmc.pauli_lcu_num_selection_qubits(lcu), "selection"
        )
        system = qmc.qubit_array(num_qubits, "system")
        _prepare_basis(system, initial_basis, num_qubits)
        selection, _ = gate(selection, system)
        return qmc.measure(selection)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the success-projected phase-sensitive system observable."""
        selection = qmc.qubit_array(
            qmc.pauli_lcu_num_selection_qubits(lcu), "selection"
        )
        system = qmc.qubit_array(num_qubits, "system")
        _prepare_basis(system, initial_basis, num_qubits)
        selection, system = gate(selection, system)
        return qmc.expval((selection[0], system[0]), observable)

    shots = 2048
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    results = sample_executable.sample(_executor(sdk_transpiler), shots=shots).result()
    observed_success = _zero_probability(results.results)

    alpha = 1.0 + x_weight
    expected_success = (1.0 + x_weight**2) / alpha**2
    sampling_tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert observed_success == pytest.approx(
        expected_success,
        abs=sampling_tolerance,
    )

    observable = qm_o.Hamiltonian(num_qubits=2)
    observable.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 0.5)
    observable.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        ),
        0.5,
    )
    expval_executable = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": observable},
    )
    observed_expval = float(expval_executable.run(_executor(sdk_transpiler)).result())
    expected_expval = 2.0 * np.imag(np.conj(identity_weight) * x_weight) / alpha**2
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_expval == pytest.approx(expected_expval, abs=expval_tolerance)


def test_three_term_lcu_executes_two_bit_select_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A padded non-power-of-two SELECT executes with two index qubits."""
    matrix = 1j * I2 + 0.5 * X + 0.25 * np.diag([1.0, -1.0])
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        """Apply the three-term encoding and measure its selection register."""
        selection = qmc.qubit_array(2, "selection")
        system = qmc.qubit_array(1, "system")
        selection, _ = gate(selection, system)
        return qmc.measure(selection)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_success = 3.0 / 7.0
    tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert _zero_probability(result.results) == pytest.approx(
        expected_success,
        abs=tolerance,
    )


def test_dense_two_qubit_lcu_executes_wide_select_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A seeded dense matrix executes all sixteen four-bit SELECT cases."""
    matrix = DENSE_TWO_QUBIT_MATRIX
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    gate = qmc.pauli_lcu_block_encoding(lcu)
    basis_index = 3

    assert lcu.num_terms == 16
    assert qmc.pauli_lcu_num_selection_qubits(lcu) == 4

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        """Apply the dense encoding and measure its four selection qubits."""
        selection = qmc.qubit_array(4, "selection")
        system = qmc.qubit_array(2, "system")
        _prepare_basis(system, basis_index, 2)
        selection, _ = gate(selection, system)
        return qmc.measure(selection)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_success = float(np.linalg.norm(matrix[:, basis_index]) ** 2 / lcu.alpha**2)
    tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert _zero_probability(result.results) == pytest.approx(
        expected_success,
        abs=tolerance,
    )


def test_inverse_round_trip_executes_on_every_sdk(sdk_transpiler: Any) -> None:
    """The generated composite followed by its inverse restores both registers."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply the block encoding and its inverse to a basis state."""
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        system[0] = qmc.x(system[0])
        selection, system = gate(selection, system)
        selection, system = qmc.inverse(gate)(selection, system)
        return qmc.measure(selection), qmc.measure(system)

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=128).result()
    assert result.results == [(((0,), (1,)), 128)]


def test_inverse_alone_conjugates_complex_phase_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The explicit inverse body independently exposes the adjoint block."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    inverse_gate = qmc.inverse(qmc.pauli_lcu_block_encoding(lcu))

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Float:
        """Estimate projected system Y after only the inverse encoding."""
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        selection, system = inverse_gate(selection, system)
        return qmc.expval((selection[0], system[0]), observable)

    projected_y = qm_o.Hamiltonian(num_qubits=2)
    projected_y.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 0.5)
    projected_y.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        ),
        0.5,
    )
    executable = sdk_transpiler.transpiler.transpile(
        circuit,
        bindings={"observable": projected_y},
    )

    observed = float(executable.run(_executor(sdk_transpiler)).result())
    assert observed == pytest.approx(4.0 / 9.0, abs=1e-6)


def test_outer_control_observes_single_identity_term_phase(
    sdk_transpiler: Any,
) -> None:
    """A negative identity coefficient kicks its phase onto an outer control."""
    lcu = PauliLCU.from_matrix(-I2)
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Run a Hadamard test around the single-term block encoding."""
        outer = qmc.h(qmc.qubit("outer"))
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        outer, selection, system = qmc.control(gate)(outer, selection, system)
        outer = qmc.h(outer)
        return qmc.measure(outer)

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=128).result()
    assert result.results == [(1, 128)]


def test_outer_control_executes_multi_term_prepare_select_path(
    sdk_transpiler: Any,
) -> None:
    """A Hadamard test observes a controlled multi-term block encoding."""
    identity_weight = np.exp(1j * math.pi / 3.0)
    lcu = PauliLCU.from_matrix(identity_weight * I2 + 0.5 * X, atol=1e-12)
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Control the complete PREPARE-SELECT-PREPARE-dagger composite."""
        outer = qmc.h(qmc.qubit("outer"))
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        outer, selection, system = qmc.control(gate)(outer, selection, system)
        outer = qmc.h(outer)
        return qmc.measure(outer)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    zero_probability = (
        sum(count for outcome, count in result.results if int(outcome) == 0) / shots
    )
    expected = 2.0 / 3.0
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / shots) + 0.02
    assert zero_probability == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize(
    ("invert", "expected_y"),
    [(False, -2.0 / 3.0), (True, 2.0 / 3.0)],
    ids=["forward", "inverse"],
)
def test_patterned_outer_control_composes_with_lcu_select_on_every_sdk(
    sdk_transpiler: Any,
    invert: bool,
    expected_y: float,
) -> None:
    """An LSB-first zero control composes with the phase-bearing inner SELECT.

    With ``alpha = 3 / 2``, the active branch has selection-zero amplitude
    ``2j / 3`` and success probability ``5 / 9``. The inactive identity branch
    succeeds with probability one, so their equal superposition succeeds with
    probability ``7 / 9`` and has control-Y expectation ``-2 / 3`` (conjugated
    by inverse).
    """
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def forward_lcu(
        selection: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the forward composite through a fixed qkernel signature."""
        return gate(selection, system)

    @qmc.qkernel
    def inverse_lcu(
        selection: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the inverse composite through a fixed qkernel signature."""
        return qmc.inverse(gate)(selection, system)

    applied_gate = inverse_lcu if invert else forward_lcu
    controlled_gate = qmc.control(
        applied_gate,
        num_controls=2,
        control_value=2,
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Measure LCU success across active and inactive control branches."""
        controls = qmc.qubit_array(2, "controls")
        controls[0] = qmc.h(controls[0])
        controls[1] = qmc.x(controls[1])
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        controls, selection, system = controlled_gate(
            controls,
            selection,
            system,
        )
        return qmc.measure(selection)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the phase kickback on the zero-activated control qubit."""
        controls = qmc.qubit_array(2, "controls")
        controls[0] = qmc.h(controls[0])
        controls[1] = qmc.x(controls[1])
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        controls, selection, system = controlled_gate(
            controls,
            selection,
            system,
        )
        return qmc.expval(controls[0], observable)

    shots = 4096
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        _executor(sdk_transpiler),
        shots=shots,
    ).result()
    expected_success = 7.0 / 9.0
    sampling_tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_success,
        abs=sampling_tolerance,
    )

    expval_executable = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Y(0)},
    )
    observed_y = float(expval_executable.run(_executor(sdk_transpiler)).result())
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_y == pytest.approx(expected_y, abs=expval_tolerance)


def test_zero_operator_path_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The exact-zero path flips its signal and remains executable everywhere."""
    lcu = PauliLCU.from_matrix(np.zeros((2, 2), dtype=np.complex128))
    gate = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Apply the zero encoding and measure its signal qubit."""
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        selection, _ = gate(selection, system)
        return qmc.measure(selection[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Apply the zero encoding and estimate signal Z."""
        selection = qmc.qubit_array(1, "selection")
        system = qmc.qubit_array(1, "system")
        selection, _ = gate(selection, system)
        return qmc.expval(selection[0], observable)

    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        _executor(sdk_transpiler), shots=128
    ).result()
    assert sample_result.results == [(1, 128)]

    observable = qm_o.Z(0)
    expval_executable = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": observable},
    )
    assert float(
        expval_executable.run(_executor(sdk_transpiler)).result()
    ) == pytest.approx(-1.0, abs=1e-8)
