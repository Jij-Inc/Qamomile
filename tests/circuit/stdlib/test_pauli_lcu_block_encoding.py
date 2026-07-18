"""Execution tests for complex Pauli LCU block encodings."""

from __future__ import annotations

import dataclasses
import importlib
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


@qmc.qkernel
def _identity_block_case(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return a block-encoding target pair unchanged.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.
        system (qmc.Vector[qmc.Qubit]): System register to preserve.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged signal
            and system registers in their original order.
    """
    return signal, system


@qmc.qkernel
def _invalid_scalar_block_unitary(
    signal: qmc.Qubit,
    system: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Return scalar qubits through an intentionally invalid block ABI.

    Args:
        signal (qmc.Qubit): Scalar signal qubit.
        system (qmc.Qubit): Scalar system qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Unchanged scalar qubits.
    """
    return signal, system


@qmc.qkernel
def _invalid_keyword_block_unitary(
    *,
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return vectors through an intentionally keyword-only block ABI.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.
        system (qmc.Vector[qmc.Qubit]): System register to preserve.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged input
            registers.
    """
    return signal, system


@qmc.qkernel
def _invalid_reversed_block_unitary(
    system: qmc.Vector[qmc.Qubit],
    signal: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return vectors through an intentionally reversed block ABI.

    Args:
        system (qmc.Vector[qmc.Qubit]): System register to preserve.
        signal (qmc.Vector[qmc.Qubit]): Signal register to preserve.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Registers in the
            required output order despite their invalid input order.
    """
    return signal, system


def _build_unitary_kernel(
    encoding: qmc.LCUBlockEncoding,
    *,
    invert: bool = False,
) -> qmc.QKernel:
    """Build an allocation-only kernel exposing the block-encoding unitary.

    Args:
        encoding (qmc.LCUBlockEncoding): Descriptor whose unitary should
            be exposed.
        invert (bool): Whether to apply the inverse transform.
            Defaults to ``False``.

    Returns:
        qmc.QKernel: Allocation-only kernel for exact unitary inspection.
    """
    applied_unitary = qmc.inverse(encoding.unitary) if invert else encoding.unitary

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Allocate registers and apply the selected block-encoding direction."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = applied_unitary(signal, system)
        return qmc.measure(signal[0])

    return kernel


def _qiskit_unitary(
    encoding: qmc.LCUBlockEncoding,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Transpile a block encoding and return its exact Qiskit unitary.

    Args:
        encoding (qmc.LCUBlockEncoding): Descriptor whose unitary should
            be materialized.
        invert (bool): Whether to materialize the inverse transform.
            Defaults to ``False``.

    Returns:
        np.ndarray: Dense unitary matrix emitted through Qiskit.
    """
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import QiskitTranspiler

    executable = QiskitTranspiler().transpile(
        _build_unitary_kernel(encoding, invert=invert)
    )
    quantum_circuit = executable.quantum_circuit.remove_final_measurements(
        inplace=False
    )
    return np.asarray(Operator(quantum_circuit).data)


def _top_left_block(
    unitary: np.ndarray,
    encoding: qmc.LCUBlockEncoding,
) -> np.ndarray:
    """Extract the all-zero-signal block in Qamomile LSB order.

    Args:
        unitary (np.ndarray): Dense block-encoding unitary.
        encoding (qmc.LCUBlockEncoding): Descriptor defining the register
            widths.

    Returns:
        np.ndarray: Projected all-zero-signal system block.
    """
    system_indices = [
        basis_index << encoding.num_signal_qubits
        for basis_index in range(1 << encoding.num_system_qubits)
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
        invert (bool): Whether to lower the inverse transform.
            Defaults to ``False``.

    Returns:
        tuple[str, ...]: Ordered SHA-256 case fingerprints.
    """
    from qamomile.qiskit import QiskitTranspiler

    encoding = qmc.pauli_lcu_block_encoding(lcu)
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_build_unitary_kernel(encoding, invert=invert))
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
    """Prepare a little-endian computational-basis system state in place.

    Args:
        system (qmc.Vector[qmc.Qubit]): System register to prepare.
        basis_index (int): Computational-basis value to encode.
        num_qubits (int): Number of little-endian basis bits to inspect.

    Returns:
        None: The register is updated in place.
    """
    for qubit in range(num_qubits):
        if (basis_index >> qubit) & 1:
            system[qubit] = qmc.x(system[qubit])


def _executor(case: Any) -> Any:
    """Return a local simulator executor for one SDK fixture case.

    Args:
        case (Any): Cross-backend fixture carrying a transpiler and name.

    Returns:
        Any: Backend-specific local executor.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _zero_probability(results: list[tuple[Any, int]]) -> float:
    """Return the measured probability of an all-zero output value.

    Args:
        results (list[tuple[Any, int]]): Outcome and count pairs to aggregate.

    Returns:
        float: Fraction of shots whose flattened outcome is all zero.
    """
    total = sum(count for _, count in results)
    zero_count = sum(
        count
        for outcome, count in results
        if all(bit == 0 for bit in _flatten(outcome))
    )
    return zero_count / total


def _flatten(value: Any) -> tuple[int, ...]:
    """Flatten nested scalar/vector measurement values into integer bits.

    Args:
        value (Any): Scalar or nested tuple measurement value.

    Returns:
        tuple[int, ...]: Depth-first tuple of integer measurement bits.
    """
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
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    unitary = _qiskit_unitary(encoding)

    np.testing.assert_allclose(
        _top_left_block(unitary, encoding),
        matrix / encoding.normalization,
        atol=1e-10,
        rtol=0.0,
    )
    expected_dimension = 1 << (encoding.num_signal_qubits + encoding.num_system_qubits)
    assert unitary.shape == (expected_dimension, expected_dimension)
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
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    np.testing.assert_allclose(
        _top_left_block(_qiskit_unitary(encoding, invert=True), encoding),
        matrix.conj().T / encoding.normalization,
        atol=1e-10,
        rtol=0.0,
    )


def test_qiskit_inverse_is_full_adjoint_for_wide_prepare() -> None:
    """PREPARE and UNPREPARE remain adjoints on the full signal space."""
    encoding = qmc.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(DENSE_TWO_QUBIT_MATRIX)
    )
    forward = _qiskit_unitary(encoding)
    adjoint = _qiskit_unitary(encoding, invert=True)

    np.testing.assert_allclose(
        adjoint,
        forward.conj().T,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        adjoint @ forward,
        np.eye(forward.shape[0]),
        atol=1e-10,
        rtol=0.0,
    )


def test_lazy_inverse_select_has_stable_conjugated_case_fingerprints() -> None:
    """Lowered identity-case phases distinguish forward and inverse bodies."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)

    forward = _select_case_fingerprints(lcu)
    repeated = _select_case_fingerprints(lcu)
    adjoint = _select_case_fingerprints(lcu, invert=True)

    assert forward == repeated
    assert len(forward) == len(adjoint) == 2
    assert forward[0] != adjoint[0]
    assert forward[1] == adjoint[1]


def test_multi_term_root_composite_round_trips_qkernel_serialization() -> None:
    """A symbolic root signature retains its constant SELECT width on the wire."""
    from qamomile.circuit.serialization import deserialize, serialize

    encoding = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(1j * I2 + 0.5 * X))

    payload = serialize(encoding.unitary)
    assert serialize(deserialize(payload)) == payload


def test_factory_builds_only_the_forward_multi_term_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction leaves inverse materialization to the lazy transform path."""
    module = importlib.import_module("qamomile.circuit.stdlib.pauli_lcu_block_encoding")
    original_builder = module._build_multi_term_encoding
    calls: list[PauliLCU] = []

    def counted_builder(lcu: PauliLCU) -> qmc.QKernel:
        """Record and delegate one multi-term kernel construction.

        Args:
            lcu (PauliLCU): Decomposition passed to the patched builder.

        Returns:
            qmc.QKernel: Kernel produced by the original builder.
        """
        calls.append(lcu)
        return original_builder(lcu)

    monkeypatch.setattr(module, "_build_multi_term_encoding", counted_builder)
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    encoding = module.pauli_lcu_block_encoding(lcu)

    assert calls == [lcu]
    assert encoding.unitary._block is None
    assert qmc.inverse(qmc.inverse(encoding.unitary)) is encoding.unitary


def test_descriptor_contract_is_frozen_noncallable_and_identity_based() -> None:
    """The public descriptor exposes exactly the agreed static four fields."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    repeated = qmc.pauli_lcu_block_encoding(lcu)
    generic = qmc.LCUBlockEncoding(
        unitary=encoding.unitary,
        normalization=encoding.normalization,
        num_signal_qubits=encoding.num_signal_qubits,
        num_system_qubits=encoding.num_system_qubits,
    )

    assert issubclass(qmc.PauliLCUBlockEncoding, qmc.LCUBlockEncoding)
    assert isinstance(encoding, qmc.LCUBlockEncoding)
    assert isinstance(encoding, qmc.PauliLCUBlockEncoding)
    assert type(generic) is qmc.LCUBlockEncoding
    assert tuple(field.name for field in dataclasses.fields(encoding)) == (
        "unitary",
        "normalization",
        "num_signal_qubits",
        "num_system_qubits",
    )
    assert isinstance(encoding.unitary, qmc.QKernel)
    assert tuple(encoding.unitary.signature.parameters) == ("signal", "system")
    assert encoding.unitary.input_types == {
        "signal": qmc.Vector[qmc.Qubit],
        "system": qmc.Vector[qmc.Qubit],
    }
    assert encoding.unitary.output_types == [
        qmc.Vector[qmc.Qubit],
        qmc.Vector[qmc.Qubit],
    ]
    assert encoding.normalization == pytest.approx(lcu.alpha)
    assert type(encoding.num_signal_qubits) is int
    assert type(encoding.num_system_qubits) is int
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 1
    assert not callable(encoding)
    assert not hasattr(encoding, "__dict__")
    assert not hasattr(encoding, "kernel")
    assert not hasattr(encoding, "error_bound")
    assert encoding is not repeated
    assert encoding != repeated
    assert not callable(generic)
    assert not hasattr(generic, "__dict__")
    assert generic != encoding

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.normalization = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        generic.normalization = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        encoding()  # type: ignore[operator]


@pytest.mark.parametrize(
    "descriptor_type",
    [qmc.LCUBlockEncoding, qmc.PauliLCUBlockEncoding],
)
@pytest.mark.parametrize("normalization", [0.0, -1.0, math.inf, math.nan, 10**1000])
def test_descriptor_rejects_invalid_normalization(
    descriptor_type: type[qmc.LCUBlockEncoding],
    normalization: float,
) -> None:
    """Descriptor construction rejects non-positive or non-finite scaling."""
    unitary = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2)).unitary

    with pytest.raises(ValueError, match="normalization"):
        descriptor_type(unitary, normalization, 1, 1)


@pytest.mark.parametrize(
    "descriptor_type",
    [qmc.LCUBlockEncoding, qmc.PauliLCUBlockEncoding],
)
def test_descriptor_rejects_invalid_field_types_and_widths(
    descriptor_type: type[qmc.LCUBlockEncoding],
) -> None:
    """Descriptor construction enforces its QKernel ABI and positive widths."""
    unitary = qmc.pauli_lcu_block_encoding(PauliLCU.from_matrix(I2)).unitary

    with pytest.raises(TypeError, match="unitary must be a QKernel"):
        descriptor_type(object(), 1.0, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unitary must have signature"):
        descriptor_type(_invalid_scalar_block_unitary, 1.0, 1, 1)
    with pytest.raises(TypeError, match="unitary must have signature"):
        descriptor_type(_invalid_keyword_block_unitary, 1.0, 1, 1)
    with pytest.raises(TypeError, match="unitary must have signature"):
        descriptor_type(_invalid_reversed_block_unitary, 1.0, 1, 1)
    with pytest.raises(TypeError, match="normalization"):
        descriptor_type(unitary, "1", 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_signal_qubits"):
        descriptor_type(unitary, 1.0, True, 1)
    with pytest.raises(TypeError, match="num_system_qubits"):
        descriptor_type(unitary, 1.0, 1, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_signal_qubits"):
        descriptor_type(unitary, 1.0, 0, 1)
    with pytest.raises(ValueError, match="num_system_qubits"):
        descriptor_type(unitary, 1.0, 1, -1)


def test_signal_width_covers_zero_single_and_non_power_of_two_terms() -> None:
    """The static ABI retains one signal qubit at small term counts."""
    zero = PauliLCU.from_matrix(np.zeros((2, 2), dtype=np.complex128))
    single = PauliLCU.from_matrix(1j * I2)
    three = PauliLCU.from_matrix(1j * I2 + 0.5 * X + 0.25 * np.diag([1, -1]))

    zero_encoding = qmc.pauli_lcu_block_encoding(zero)
    single_encoding = qmc.pauli_lcu_block_encoding(single)
    three_encoding = qmc.pauli_lcu_block_encoding(three)

    assert zero_encoding.normalization == 1.0
    assert zero.alpha == 0.0
    assert zero_encoding.num_signal_qubits == 1
    assert single_encoding.num_signal_qubits == 1
    assert three_encoding.num_signal_qubits == 2


def test_scalar_lcu_is_rejected_by_circuit_factory() -> None:
    """The linalg scalar domain is explicit rather than silently padded."""
    lcu = PauliLCU.from_matrix(np.array([[1j]]))

    with pytest.raises(ValueError, match="system qubit"):
        qmc.pauli_lcu_block_encoding(lcu)


def test_lcu_factory_rejects_non_lcu_values() -> None:
    """The public factory rejects values outside its Pauli-specific API."""
    with pytest.raises(TypeError, match="PauliLCU"):
        qmc.pauli_lcu_block_encoding(object())  # type: ignore[arg-type]


def test_numpy_integer_pauli_index_builds_block_encoding() -> None:
    """Accepted NumPy operator indices remain valid qkernel subscripts."""
    lcu = PauliLCU(
        1,
        (
            PauliLCUTerm(
                1.0,
                (qm_o.PauliOperator(qm_o.Pauli.X, np.int64(0)),),
            ),
        ),
    )

    assert qmc.pauli_lcu_block_encoding(lcu).unitary.block is not None


def test_signed_zero_does_not_change_block_compiler_identity() -> None:
    """Equal negative-real terms produce one canonical callable identity."""
    from qamomile.circuit.frontend.qkernel_callable import qkernel_callable_ref

    positive_zero = PauliLCU(1, (PauliLCUTerm(complex(-1.0, 0.0), ()),))
    negative_zero = PauliLCU(1, (PauliLCUTerm(complex(-1.0, -0.0), ()),))

    positive_ref = qkernel_callable_ref(
        qmc.pauli_lcu_block_encoding(positive_zero).unitary
    )
    negative_ref = qkernel_callable_ref(
        qmc.pauli_lcu_block_encoding(negative_zero).unitary
    )
    assert positive_ref == negative_ref


def test_composite_rejects_wrong_concrete_register_widths() -> None:
    """Plain, inverse, control, and SELECT share exact width diagnostics."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X + 0.25 * np.diag([1, -1]))
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    inverse_unitary = qmc.inverse(encoding.unitary)
    controlled_unitary = qmc.control(encoding.unitary)
    selected_unitary = qmc.select(
        (_identity_block_case, encoding.unitary),
        num_index_qubits=1,
    )

    @qmc.qkernel
    def plain_wrong_signal() -> qmc.Bit:
        """Call the plain unitary with one signal qubit."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverse_wrong_signal() -> qmc.Bit:
        """Call the inverse unitary with one signal qubit."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        signal, _ = inverse_unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def control_wrong_signal() -> qmc.Bit:
        """Call the controlled unitary with one signal qubit."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        control, signal, _ = controlled_unitary(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def select_wrong_signal() -> qmc.Bit:
        """Call the nested-SELECT unitary with one signal qubit."""
        outer = qmc.qubit_array(1, "outer")
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        outer, signal, _ = selected_unitary(outer, signal, system)
        return qmc.measure(outer[0])

    @qmc.qkernel
    def plain_wrong_system() -> qmc.Bit:
        """Call the plain unitary with two system qubits."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(2, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverse_wrong_system() -> qmc.Bit:
        """Call the inverse unitary with two system qubits."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(2, "system")
        signal, _ = inverse_unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def control_wrong_system() -> qmc.Bit:
        """Call the controlled unitary with two system qubits."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(2, "system")
        control, signal, _ = controlled_unitary(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def select_wrong_system() -> qmc.Bit:
        """Call the nested-SELECT unitary with two system qubits."""
        outer = qmc.qubit_array(1, "outer")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(2, "system")
        outer, signal, _ = selected_unitary(outer, signal, system)
        return qmc.measure(outer[0])

    wrong_signal_kernels = (
        plain_wrong_signal,
        inverse_wrong_signal,
        control_wrong_signal,
        select_wrong_signal,
    )
    for kernel in wrong_signal_kernels:
        with pytest.raises(ValueError) as error:
            kernel.build()
        assert str(error.value) == (
            "Pauli LCU block encoding requires 2 signal qubits, got 1."
        )

    wrong_system_kernels = (
        plain_wrong_system,
        inverse_wrong_system,
        control_wrong_system,
        select_wrong_system,
    )
    for kernel in wrong_system_kernels:
        with pytest.raises(ValueError) as error:
            kernel.build()
        assert str(error.value) == (
            "Pauli LCU block encoding requires 1 system qubit, got 2."
        )


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
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    initial_basis = (seed % (1 << max(0, num_qubits - 1))) << 1

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the random LCU and measure its signal register."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(num_qubits, "system")
        _prepare_basis(system, initial_basis, num_qubits)
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate a phase-sensitive observable on the all-zero signal block."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(num_qubits, "system")
        _prepare_basis(system, initial_basis, num_qubits)
        signal, system = encoding.unitary(signal, system)
        return qmc.expval((signal[0], system[0]), observable)

    shots = 2048
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    results = sample_executable.sample(_executor(sdk_transpiler), shots=shots).result()
    observed_zero_signal_probability = _zero_probability(results.results)

    expected_zero_signal_probability = (1.0 + x_weight**2) / encoding.normalization**2
    sampling_tolerance = (
        6.0
        * math.sqrt(
            expected_zero_signal_probability
            * (1.0 - expected_zero_signal_probability)
            / shots
        )
        + 0.02
    )
    assert observed_zero_signal_probability == pytest.approx(
        expected_zero_signal_probability,
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
    expected_expval = (
        2.0 * np.imag(np.conj(identity_weight) * x_weight) / encoding.normalization**2
    )
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_expval == pytest.approx(expected_expval, abs=expval_tolerance)


def test_three_term_lcu_executes_two_bit_select_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A padded non-power-of-two SELECT executes with two index qubits."""
    matrix = 1j * I2 + 0.5 * X + 0.25 * np.diag([1.0, -1.0])
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        """Apply the three-term encoding and measure its signal register."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_zero_signal_probability = float(
        np.linalg.norm(matrix[:, 0]) ** 2 / encoding.normalization**2
    )
    tolerance = (
        6.0
        * math.sqrt(
            expected_zero_signal_probability
            * (1.0 - expected_zero_signal_probability)
            / shots
        )
        + 0.02
    )
    assert _zero_probability(result.results) == pytest.approx(
        expected_zero_signal_probability,
        abs=tolerance,
    )


def test_dense_two_qubit_lcu_executes_wide_select_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A seeded dense matrix executes all sixteen four-bit SELECT cases."""
    matrix = DENSE_TWO_QUBIT_MATRIX
    lcu = PauliLCU.from_matrix(matrix, atol=1e-12)
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    basis_index = 3

    assert lcu.num_terms == 16
    assert encoding.num_signal_qubits == 4

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        """Apply the dense encoding and measure its four signal qubits."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        _prepare_basis(system, basis_index, 2)
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    expected_zero_signal_probability = float(
        np.linalg.norm(matrix[:, basis_index]) ** 2 / encoding.normalization**2
    )
    tolerance = (
        6.0
        * math.sqrt(
            expected_zero_signal_probability
            * (1.0 - expected_zero_signal_probability)
            / shots
        )
        + 0.02
    )
    assert _zero_probability(result.results) == pytest.approx(
        expected_zero_signal_probability,
        abs=tolerance,
    )


def test_inverse_round_trip_executes_on_every_sdk(sdk_transpiler: Any) -> None:
    """The generated composite followed by its inverse restores both registers."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    double_inverse = qmc.inverse(qmc.inverse(encoding.unitary))
    assert double_inverse is encoding.unitary

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply the block encoding and its inverse to a basis state."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        system[0] = qmc.x(system[0])
        signal, system = double_inverse(signal, system)
        signal, system = qmc.inverse(encoding.unitary)(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=128).result()
    assert result.results == [(((0,), (1,)), 128)]


def test_inverse_alone_conjugates_complex_phase_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The lazy inverse transform independently exposes the adjoint block."""
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    encoding = qmc.pauli_lcu_block_encoding(lcu)
    inverse_unitary = qmc.inverse(encoding.unitary)

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Float:
        """Estimate projected system Y after only the inverse encoding."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, system = inverse_unitary(signal, system)
        return qmc.expval((signal[0], system[0]), observable)

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
    inverse_identity_weight = -1j
    x_weight = 0.5
    expected_y = (
        2.0
        * np.imag(np.conj(inverse_identity_weight) * x_weight)
        / encoding.normalization**2
    )
    assert observed == pytest.approx(expected_y, abs=1e-6)


def test_outer_control_observes_single_identity_term_phase(
    sdk_transpiler: Any,
) -> None:
    """A non-Clifford identity phase interferes through an outer control."""
    phase = math.pi / 3.0
    lcu = PauliLCU.from_matrix(np.exp(1j * phase) * I2)
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Run a Hadamard test around the single-term block encoding."""
        outer = qmc.h(qmc.qubit("outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        outer, signal, system = qmc.control(encoding.unitary)(
            outer,
            signal,
            system,
        )
        outer = qmc.h(outer)
        return qmc.measure(outer)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    expected = (1.0 + math.cos(phase)) / 2.0
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / shots) + 0.02
    assert _zero_probability(result.results) == pytest.approx(expected, abs=tolerance)


def test_outer_control_executes_multi_term_prepare_select_path(
    sdk_transpiler: Any,
) -> None:
    """A Hadamard test observes a controlled multi-term block encoding."""
    identity_weight = np.exp(1j * math.pi / 3.0)
    lcu = PauliLCU.from_matrix(identity_weight * I2 + 0.5 * X, atol=1e-12)
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Control the complete PREPARE-SELECT-PREPARE-dagger composite."""
        outer = qmc.h(qmc.qubit("outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        outer, signal, system = qmc.control(encoding.unitary)(
            outer,
            signal,
            system,
        )
        outer = qmc.h(outer)
        return qmc.measure(outer)

    shots = 4096
    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(_executor(sdk_transpiler), shots=shots).result()
    zero_probability = (
        sum(count for outcome, count in result.results if int(outcome) == 0) / shots
    )
    block_overlap = identity_weight / encoding.normalization
    expected = (1.0 + float(np.real(block_overlap))) / 2.0
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / shots) + 0.02
    assert zero_probability == pytest.approx(expected, abs=tolerance)


def test_outer_select_observes_single_identity_term_phase_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """A nested SELECT preserves a nontrivial child-unitary global phase."""
    phase = math.pi / 3.0
    encoding = qmc.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(np.exp(1j * phase) * I2)
    )
    selected_unitary = qmc.select(
        (_identity_block_case, encoding.unitary),
        num_index_qubits=1,
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Interfere the identity and phased-unitary SELECT branches."""
        outer = qmc.qubit_array(1, "outer")
        outer[0] = qmc.h(outer[0])
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        outer, signal, system = selected_unitary(outer, signal, system)
        outer[0] = qmc.h(outer[0])
        return qmc.measure(outer[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate phase-sensitive outer-selector Y interference."""
        outer = qmc.qubit_array(1, "outer")
        outer[0] = qmc.h(outer[0])
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        outer, signal, system = selected_unitary(outer, signal, system)
        return qmc.expval(outer[0], observable)

    shots = 4096
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        _executor(sdk_transpiler),
        shots=shots,
    ).result()
    expected_zero = (1.0 + math.cos(phase)) / 2.0
    sample_tolerance = (
        6.0 * math.sqrt(expected_zero * (1.0 - expected_zero) / shots) + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero,
        abs=sample_tolerance,
    )

    expval_executable = sdk_transpiler.transpiler.transpile(
        expval_kernel,
        bindings={"observable": qm_o.Y(0)},
    )
    observed_y = float(expval_executable.run(_executor(sdk_transpiler)).result())
    expval_tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_y == pytest.approx(math.sin(phase), abs=expval_tolerance)


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

    With ``alpha = 3 / 2``, the active branch has signal-zero amplitude
    ``2j / 3`` and all-zero signal projection probability ``5 / 9``. The
    inactive identity branch has projection probability one, so their equal
    superposition has projection probability ``7 / 9`` and control-Y
    expectation ``-2 / 3`` (conjugated by inverse).
    """
    lcu = PauliLCU.from_matrix(1j * I2 + 0.5 * X)
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def forward_lcu(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the forward composite through a fixed qkernel signature."""
        return encoding.unitary(signal, system)

    @qmc.qkernel
    def inverse_lcu(
        signal: qmc.Vector[qmc.Qubit],
        system: qmc.Vector[qmc.Qubit],
    ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
        """Expose the inverse composite through a fixed qkernel signature."""
        return qmc.inverse(encoding.unitary)(signal, system)

    applied_gate = inverse_lcu if invert else forward_lcu
    controlled_gate = qmc.control(
        applied_gate,
        num_controls=2,
        control_value=2,
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Measure the all-zero signal component across both control branches."""
        controls = qmc.qubit_array(2, "controls")
        controls[0] = qmc.h(controls[0])
        controls[1] = qmc.x(controls[1])
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        controls, signal, system = controlled_gate(
            controls,
            signal,
            system,
        )
        return qmc.measure(signal)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Estimate the phase kickback on the zero-activated control qubit."""
        controls = qmc.qubit_array(2, "controls")
        controls[0] = qmc.h(controls[0])
        controls[1] = qmc.x(controls[1])
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(1, "system")
        controls, signal, system = controlled_gate(
            controls,
            signal,
            system,
        )
        return qmc.expval(controls[0], observable)

    shots = 4096
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        _executor(sdk_transpiler),
        shots=shots,
    ).result()
    expected_zero_signal_probability = 7.0 / 9.0
    sampling_tolerance = (
        6.0
        * math.sqrt(
            expected_zero_signal_probability
            * (1.0 - expected_zero_signal_probability)
            / shots
        )
        + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero_signal_probability,
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
    encoding = qmc.pauli_lcu_block_encoding(lcu)

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Apply the zero encoding and measure its signal qubit."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Apply the zero encoding and estimate signal Z."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.expval(signal[0], observable)

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
