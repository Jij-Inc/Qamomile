"""Tests for periodic-shift LCU block encodings."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.circuit_ir import (
    SELECT_SEMANTIC_KEY,
    CallInstruction,
    CircuitProgram,
    GateInstruction,
    lower_circuit_plan,
)
from qamomile.linalg import PauliLCU, PeriodicShiftLCU


def _periodic_encoding(
    coefficients: Mapping[Any, complex],
    register_sizes: Sequence[int],
) -> qmc.PeriodicShiftLCUBlockEncoding:
    """Build a circuit descriptor through the public linalg decomposition.

    Args:
        coefficients (Mapping[Any, complex]): Periodic-shift coefficients.
        register_sizes (Sequence[int]): Per-axis qubit widths.

    Returns:
        qmc.PeriodicShiftLCUBlockEncoding: Generated circuit descriptor.
    """
    lcu = PeriodicShiftLCU.from_coefficients(
        coefficients,
        register_sizes=register_sizes,
    )
    return qmc.periodic_shift_lcu_block_encoding(lcu)


@qmc.qkernel
def _prepare_basis(
    system: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a flattened system register in an LSB-first basis state.

    Args:
        system (qmc.Vector[qmc.Qubit]): All-zero system register.
        bits (qmc.Vector[qmc.UInt]): Compile-time basis bits.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared system register.
    """
    for index in qmc.range(system.shape[0]):
        system[index] = qmc.rx(system[index], math.pi * bits[index])
    return system


@qmc.qkernel
def _identity_registers(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return signal and system registers unchanged.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register.
        system (qmc.Vector[qmc.Qubit]): System register.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: The unchanged
        registers.
    """
    return signal, system


@qmc.qkernel
def _apply_static_encoding(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply a block encoding supplied through a generic static slot.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Complete signal register.
        system (qmc.Vector[qmc.Qubit]): Ordered system register.
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated signal and
            system registers.
    """
    return encoding.unitary(signal, system)


@qmc.qkernel
def _static_direct_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Apply a generic statically bound encoding directly.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(system)


@qmc.qkernel
def _periodic_static_direct_template(
    encoding: qmc.PeriodicShiftLCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Apply a Periodic-specific statically bound encoding directly.

    Args:
        encoding (qmc.PeriodicShiftLCUBlockEncoding): Compile-time periodic
            shift LCU descriptor.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    return qmc.measure(system)


@qmc.qkernel
def _static_inverse_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Invert a helper carrying a generic static encoding argument.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = qmc.inverse(_apply_static_encoding)(
        signal,
        system,
        encoding,
    )
    return qmc.measure(system)


@qmc.qkernel
def _static_control_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Control a unitary reached through a generic static encoding slot.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """
    outer = qmc.x(qmc.qubit("outer"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    outer, signal, system = qmc.control(encoding.unitary)(outer, signal, system)
    return qmc.measure(system)


@qmc.qkernel
def _static_select_template(
    encoding: qmc.LCUBlockEncoding,
) -> qmc.Vector[qmc.Bit]:
    """Select a unitary reached through a generic static encoding slot.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        qmc.Vector[qmc.Bit]: Measured system register.
    """
    index = qmc.x(qmc.qubit("index"))
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    index, signal, system = qmc.select(
        (_identity_registers, encoding.unitary),
        num_index_qubits=1,
    )(index, signal, system)
    return qmc.measure(system)


@qmc.qkernel
def _static_round_trip_sample_template(
    encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Apply and unapply a statically bound generic LCU encoding.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _static_round_trip_expval_template(
    encoding: qmc.LCUBlockEncoding,
    observable: qmc.Observable,
) -> qmc.Float:
    """Evaluate an observable after a statically bound LCU round trip.

    Args:
        encoding (qmc.LCUBlockEncoding): Compile-time exact LCU descriptor.
        observable (qmc.Observable): System observable to evaluate.

    Returns:
        qmc.Float: Expectation value after applying the unitary and inverse.
    """
    signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(encoding.num_system_qubits, "system")
    signal, system = encoding.unitary(signal, system)
    signal, system = qmc.inverse(encoding.unitary)(signal, system)
    return qmc.expval(system, observable)


def _stencil_matrix(
    coefficients: dict[tuple[int, ...], complex],
    register_sizes: tuple[int, ...],
) -> np.ndarray:
    """Return the dense periodic-stencil matrix in Qamomile bit order.

    Args:
        coefficients (dict[tuple[int, ...], complex]): Stencil coefficients.
        register_sizes (tuple[int, ...]): Qubit widths of the periodic axes.

    Returns:
        np.ndarray: Dense matrix with axis zero occupying the least-significant
        system bits.
    """
    dimensions = tuple(1 << width for width in register_sizes)
    total_dimension = int(np.prod(dimensions))
    matrix = np.zeros((total_dimension, total_dimension), dtype=np.complex128)
    for column in range(total_dimension):
        coordinates: list[int] = []
        remaining = column
        for dimension in dimensions:
            coordinates.append(remaining % dimension)
            remaining //= dimension
        for offset, coefficient in coefficients.items():
            shifted = tuple(
                (coordinate + delta) % dimension
                for coordinate, delta, dimension in zip(
                    coordinates,
                    offset,
                    dimensions,
                    strict=True,
                )
            )
            row = 0
            stride = 1
            for coordinate, dimension in zip(shifted, dimensions, strict=True):
                row += coordinate * stride
                stride *= dimension
            matrix[row, column] += coefficient
    return matrix


def _z_hamiltonian(coefficients: Sequence[float]) -> qm_o.Hamiltonian:
    """Build a weighted sum of local Pauli-Z observables.

    Args:
        coefficients (Sequence[float]): Per-qubit real weights.

    Returns:
        qm_o.Hamiltonian: Weighted Pauli-Z sum.
    """
    hamiltonian = qm_o.Hamiltonian()
    for index, coefficient in enumerate(coefficients):
        hamiltonian += float(coefficient) * qm_o.Z(index)
    return hamiltonian


def _expected_z(coefficients: Sequence[float], bits: Sequence[int]) -> float:
    """Return the exact weighted-Z expectation of one basis state.

    Args:
        coefficients (Sequence[float]): Per-qubit Pauli-Z weights.
        bits (Sequence[int]): LSB-first computational-basis bits.

    Returns:
        float: Sum of ``coefficient * (1 - 2 * bit)``.
    """
    return float(
        sum(
            coefficient * (1.0 - 2.0 * bit)
            for coefficient, bit in zip(coefficients, bits, strict=True)
        )
    )


def _shifted_basis_bits(
    bits: Sequence[int],
    register_sizes: Sequence[int],
    offset: Sequence[int],
) -> tuple[int, ...]:
    """Return basis bits after one axis-wise periodic shift.

    Args:
        bits (Sequence[int]): Flattened LSB-first input bits.
        register_sizes (Sequence[int]): Qubit widths of the consecutive axes.
        offset (Sequence[int]): Signed modular displacement of every axis.

    Returns:
        tuple[int, ...]: Flattened LSB-first shifted bits.
    """
    shifted_bits: list[int] = []
    start = 0
    for width, displacement in zip(register_sizes, offset, strict=True):
        value = sum(int(bits[start + index]) << index for index in range(width))
        shifted = (value + displacement) % (1 << width)
        shifted_bits.extend((shifted >> index) & 1 for index in range(width))
        start += width
    return tuple(shifted_bits)


def _sample_only_outcome(
    sdk_case: Any,
    kernel: Any,
    bindings: dict[str, Any],
) -> Any:
    """Transpile and sample a deterministic kernel on one SDK backend.

    Args:
        sdk_case (Any): Backend fixture with transpiler and backend name.
        kernel (Any): Classical-output qkernel to execute.
        bindings (dict[str, Any]): Compile-time kernel bindings.

    Returns:
        Any: The only sampled classical outcome.
    """
    executable = sdk_case.transpiler.transpile(kernel, bindings=bindings)
    result = executable.sample(
        sdk_case.transpiler.executor(),
        bindings={},
        shots=16,
    ).result()
    counts = {outcome: count for outcome, count in result.results}
    assert sum(counts.values()) == 16
    assert len(counts) == 1, f"{sdk_case.backend_name}: got {counts}"
    return next(iter(counts))


def _run_expval(
    sdk_case: Any,
    kernel: Any,
    bindings: dict[str, Any],
) -> float:
    """Transpile and execute an expectation-value kernel on one backend.

    Args:
        sdk_case (Any): Backend fixture with transpiler and executor.
        kernel (Any): Expectation-value qkernel to execute.
        bindings (dict[str, Any]): Compile-time kernel bindings.

    Returns:
        float: Backend expectation-value result.
    """
    executable = sdk_case.transpiler.transpile(kernel, bindings=bindings)
    return float(executable.run(sdk_case.transpiler.executor(), bindings={}).result())


def _is_all_zero(value: Any) -> bool:
    """Return whether a sampled scalar or nested register value is all zero.

    Args:
        value (Any): Backend-independent sampled outcome value.

    Returns:
        bool: Whether every contained classical bit is zero.
    """
    if isinstance(value, tuple):
        return all(_is_all_zero(item) for item in value)
    return int(value) == 0


def _zero_probability(results: Sequence[tuple[Any, int]]) -> float:
    """Return the empirical probability of an all-zero sampled outcome.

    Args:
        results (Sequence[tuple[Any, int]]): Backend result-count pairs.

    Returns:
        float: Fraction of shots assigned to the all-zero outcome.
    """
    shots = sum(count for _, count in results)
    return sum(count for outcome, count in results if _is_all_zero(outcome)) / shots


def _build_unitary_kernel(
    encoding: qmc.PeriodicShiftLCUBlockEncoding,
    *,
    invert: bool = False,
) -> qmc.QKernel:
    """Build an allocation-only kernel for exact unitary inspection.

    Args:
        encoding (qmc.PeriodicShiftLCUBlockEncoding): Descriptor to inspect.
        invert (bool): Whether to apply the inverse unitary. Defaults to
            ``False``.

    Returns:
        qmc.QKernel: Allocation-only kernel ending in removable measurements.
    """
    applied_unitary = qmc.inverse(encoding.unitary) if invert else encoding.unitary

    @qmc.qkernel
    def kernel() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Allocate and apply the selected periodic encoding direction."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        signal, system = applied_unitary(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    return kernel


def _qiskit_unitary(
    qiskit_transpiler: Any,
    encoding: qmc.PeriodicShiftLCUBlockEncoding,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Materialize one periodic encoding as an exact dense unitary.

    Args:
        qiskit_transpiler (Any): Qiskit backend fixture.
        encoding (qmc.PeriodicShiftLCUBlockEncoding): Descriptor to inspect.
        invert (bool): Whether to materialize the inverse. Defaults to
            ``False``.

    Returns:
        np.ndarray: Dense unitary in Qamomile's emitted qubit order.
    """
    from qiskit.quantum_info import Operator

    executable = qiskit_transpiler.transpile(
        _build_unitary_kernel(encoding, invert=invert)
    )
    circuit = executable.compiled_quantum[0].circuit.remove_final_measurements(
        inplace=False
    )
    unitary = np.asarray(Operator(circuit).data, dtype=np.complex128)
    expected_dimension = 1 << (encoding.num_signal_qubits + encoding.num_system_qubits)
    assert unitary.shape == (expected_dimension, expected_dimension)
    return unitary


def _top_left_block(
    unitary: np.ndarray,
    encoding: qmc.PeriodicShiftLCUBlockEncoding,
) -> np.ndarray:
    """Extract the all-zero-signal block from one dense unitary.

    Args:
        unitary (np.ndarray): Full block-encoding unitary.
        encoding (qmc.PeriodicShiftLCUBlockEncoding): Register-width metadata.

    Returns:
        np.ndarray: Projected system block in LSB-first order.
    """
    system_dimension = 1 << encoding.num_system_qubits
    signal_dimension = 1 << encoding.num_signal_qubits
    indices = np.arange(system_dimension) * signal_dimension
    return unitary[np.ix_(indices, indices)]


def _lower_first_circuit(kernel: qmc.QKernel, transpiler: Any) -> CircuitProgram:
    """Lower one qkernel and return its first backend-neutral circuit.

    Args:
        kernel (qmc.QKernel): Entrypoint to lower.
        transpiler (Any): Transpiler providing prepare and plan stages.

    Returns:
        CircuitProgram: First lowered quantum segment.
    """
    prepared = transpiler.prepare(kernel)
    program = lower_circuit_plan(transpiler.plan_circuit(prepared)).get_first_circuit()
    assert isinstance(program, CircuitProgram)
    return program


def _nested_calls(program: CircuitProgram) -> tuple[CallInstruction, ...]:
    """Collect reusable calls recursively from a lowered circuit.

    Args:
        program (CircuitProgram): Root circuit to inspect.

    Returns:
        tuple[CallInstruction, ...]: Calls in depth-first traversal order.
    """
    calls: list[CallInstruction] = []
    pending = [program]
    while pending:
        current = pending.pop()
        for operation in current.operations:
            if isinstance(operation, CallInstruction):
                calls.append(operation)
                pending.append(operation.callee.body)
    return tuple(calls)


@pytest.mark.parametrize(
    ("register_sizes", "coefficients"),
    [
        ((2,), {(0,): complex(0.5, math.sqrt(3.0) / 2.0)}),
        ((2,), {(-1,): 0.7 - 0.2j, (0,): -1.3, (1,): 0.4 + 0.8j}),
        ((3,), {(-2,): -0.6j, (0,): 0.25, (3,): -0.4 + 0.1j}),
        (
            (1, 2),
            {
                (0, 0): -2.0,
                (1, 0): 0.25 + 0.5j,
                (-1, 0): 0.75 - 0.1j,
                (0, 1): 1.2,
                (0, -1): -0.3j,
            },
        ),
    ],
)
def test_periodic_stencil_top_left_block_matches_dense_matrix(
    qiskit_transpiler: Any,
    register_sizes: tuple[int, ...],
    coefficients: dict[tuple[int, ...], complex],
) -> None:
    """The all-zero signal block equals the requested complex stencil."""
    encoding = _periodic_encoding(coefficients, register_sizes)
    unitary = _qiskit_unitary(qiskit_transpiler, encoding)
    encoded_block = _top_left_block(unitary, encoding)
    expected = _stencil_matrix(coefficients, register_sizes)
    assert np.allclose(
        encoding.normalization * encoded_block,
        expected,
        atol=1e-8,
        rtol=1e-8,
    )


def test_periodic_shift_uses_binary_fixed_window_ladders(
    qiskit_transpiler: Any,
) -> None:
    """All dimensions share binary ladders without displacement repetition."""

    def controlled_x_count(
        register_sizes: tuple[int, ...],
        offset: tuple[int, ...],
    ) -> int:
        """Count controlled-X ladder calls for one single-term shift.

        Args:
            register_sizes (tuple[int, ...]): Qubit widths of the periodic axes.
            offset (tuple[int, ...]): Shift applied on each axis.

        Returns:
            int: Number of controlled-X calls in the lowered shift.
        """
        encoding = _periodic_encoding({offset: 1.0}, register_sizes)
        program = _lower_first_circuit(
            _build_unitary_kernel(encoding),
            qiskit_transpiler,
        )
        return sum(
            len(call.callee.body.operations) == 1
            and isinstance(call.callee.body.operations[0], GateInstruction)
            and call.callee.body.operations[0].kind.name == "X"
            for call in _nested_calls(program)
        )

    assert controlled_x_count((3,), (3,)) == controlled_x_count((3, 1), (3, 0)) == 3
    assert controlled_x_count((4,), (7,)) == 6
    assert controlled_x_count((4,), (8,)) == 0


def test_periodic_stencil_inverse_is_exact_adjoint_of_full_unitary(
    qiskit_transpiler: Any,
) -> None:
    """The inverse is the full adjoint and exposes the adjoint encoded block."""
    coefficients = {
        (0,): 0.7 + 0.2j,
        (1,): -0.4 + 0.8j,
        (-1,): 0.3 - 0.6j,
    }
    register_sizes = (2,)
    encoding = _periodic_encoding(coefficients, register_sizes)
    forward = _qiskit_unitary(qiskit_transpiler, encoding)
    inverse = _qiskit_unitary(qiskit_transpiler, encoding, invert=True)
    identity = np.eye(forward.shape[0], dtype=np.complex128)
    expected_block = _stencil_matrix(coefficients, register_sizes)

    assert np.allclose(forward.conj().T @ forward, identity, atol=1e-8, rtol=1e-8)
    assert np.allclose(inverse, forward.conj().T, atol=1e-8, rtol=1e-8)
    assert np.allclose(inverse @ forward, identity, atol=1e-8, rtol=1e-8)
    assert np.allclose(
        encoding.normalization * _top_left_block(inverse, encoding),
        expected_block.conj().T,
        atol=1e-8,
        rtol=1e-8,
    )


def test_zero_periodic_shift_lcu_has_exact_zero_block(
    qiskit_transpiler: Any,
) -> None:
    """The empty retained decomposition uses normalization one and a zero block."""
    lcu = PeriodicShiftLCU.from_coefficients({}, register_sizes=(2,))
    encoding = qmc.periodic_shift_lcu_block_encoding(lcu)
    unitary = _qiskit_unitary(qiskit_transpiler, encoding)

    assert lcu.alpha == 0.0
    assert encoding.normalization == 1.0
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 2
    assert encoding.offsets == ()
    assert encoding.coefficients == ()
    np.testing.assert_allclose(
        _top_left_block(unitary, encoding),
        np.zeros((4, 4), dtype=np.complex128),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        unitary.conj().T @ unitary,
        np.eye(unitary.shape[0]),
        atol=1e-12,
        rtol=0.0,
    )


def test_factory_encodes_lcu_recovered_from_dense_matrix(
    qiskit_transpiler: Any,
) -> None:
    """The Pauli-like matrix entry point composes with the circuit factory."""
    matrix = _stencil_matrix(
        {(0, 0): -2.0, (1, 0): 0.5j, (0, 1): 1.25},
        (1, 1),
    )
    lcu = PeriodicShiftLCU.from_matrix(matrix, register_sizes=(1, 1))
    encoding = qmc.periodic_shift_lcu_block_encoding(lcu)
    unitary = _qiskit_unitary(qiskit_transpiler, encoding)

    np.testing.assert_allclose(
        encoding.normalization * _top_left_block(unitary, encoding),
        matrix,
        atol=1e-8,
        rtol=1e-8,
    )


def test_periodic_stencil_combines_equivalent_offsets() -> None:
    """Offsets equal modulo an axis size combine before normalization."""
    encoding = _periodic_encoding(
        {1: 2.0, -3: -0.5, 0: -1.0},
        register_sizes=(2,),
    )

    assert encoding.offsets == ((0,), (1,))
    assert np.allclose(
        encoding.coefficients,
        (-1.0 + 0.0j, 1.5 + 0.0j),
        atol=1e-12,
        rtol=1e-12,
    )
    assert encoding.normalization == pytest.approx(2.5)
    assert encoding.num_signal_qubits == 1
    assert encoding.num_system_qubits == 2


def test_periodic_stencil_canonical_sum_is_mapping_order_independent() -> None:
    """Equivalent residues use canonical ``fsum`` order before normalization."""
    items = [
        (1, 1e16),
        (-3, -1e16),
        (5, 1.0),
        (0, -2.0),
    ]
    forward = _periodic_encoding(dict(items), (2,))
    reversed_order = _periodic_encoding(
        dict(reversed(items)),
        (2,),
    )

    assert forward.offsets == reversed_order.offsets == ((0,), (1,))
    np.testing.assert_allclose(
        forward.coefficients,
        reversed_order.coefficients,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        forward.coefficients,
        (-2.0, 1.0),
        atol=0.0,
        rtol=0.0,
    )
    assert forward.normalization == pytest.approx(3.0, rel=1e-15, abs=0.0)
    assert reversed_order.normalization == pytest.approx(
        3.0,
        rel=1e-15,
        abs=0.0,
    )


def test_periodic_stencil_canonicalizes_signed_zero_components() -> None:
    """Signed-zero coefficient components have one canonical representation."""
    positive_zero = _periodic_encoding(
        {0: complex(-1.0, 0.0)},
        (1,),
    )
    negative_zero = _periodic_encoding(
        {0: complex(-1.0, -0.0)},
        (1,),
    )

    np.testing.assert_allclose(
        positive_zero.coefficients,
        negative_zero.coefficients,
        atol=0.0,
        rtol=0.0,
    )
    assert math.copysign(1.0, negative_zero.coefficients[0].imag) == 1.0


def test_periodic_stencil_drops_only_exactly_cancelled_residues() -> None:
    """Partial cancellation removes one residue without losing the operator."""
    encoding = _periodic_encoding(
        {1: 1.0, -3: -1.0, 0: 2.0},
        (2,),
    )

    assert encoding.offsets == ((0,),)
    np.testing.assert_allclose(
        encoding.coefficients,
        (2.0 + 0.0j,),
        atol=0.0,
        rtol=0.0,
    )
    assert encoding.normalization == pytest.approx(2.0, rel=1e-15, abs=0.0)
    assert encoding.num_signal_qubits == 1


def test_periodic_laplacian_normalizations() -> None:
    """Canonical one- and two-dimensional Laplacians use factors four and eight."""
    one_dimensional = _periodic_encoding(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        register_sizes=(3,),
    )
    two_dimensional = _periodic_encoding(
        {
            (0, 0): -4.0,
            (-1, 0): 1.0,
            (1, 0): 1.0,
            (0, -1): 1.0,
            (0, 1): 1.0,
        },
        register_sizes=(2, 2),
    )

    assert one_dimensional.normalization == pytest.approx(4.0)
    assert one_dimensional.num_signal_qubits == 2
    assert two_dimensional.normalization == pytest.approx(8.0)
    assert two_dimensional.num_signal_qubits == 3


def test_periodic_shift_factory_rejects_non_lcu_input() -> None:
    """The circuit factory accepts only the validated linalg value object."""
    with pytest.raises(TypeError, match="PeriodicShiftLCU"):
        qmc.periodic_shift_lcu_block_encoding({0: 1.0})  # type: ignore[arg-type]


def test_periodic_shift_factory_rejects_scalar_lcu() -> None:
    """The linalg scalar domain has no circuit-level system register."""
    lcu = PeriodicShiftLCU.from_coefficients(
        {(): 1.0},
        register_sizes=(),
    )

    with pytest.raises(ValueError, match="at least one system qubit"):
        qmc.periodic_shift_lcu_block_encoding(lcu)


def test_periodic_stencil_normalizes_numpy_integer_widths() -> None:
    """NumPy integer axis widths become plain immutable Python integers."""
    encoding = _periodic_encoding(
        {(0, 0): 1.0, (1, 0): -0.5j},
        register_sizes=(np.int64(2), np.int32(1)),
    )
    replaced = dataclasses.replace(
        encoding,
        register_sizes=(np.int32(2), np.int64(1)),
    )

    assert encoding.register_sizes == (2, 1)
    assert all(type(width) is int for width in encoding.register_sizes)
    assert type(encoding.num_system_qubits) is int
    assert replaced.register_sizes == (2, 1)
    assert all(type(width) is int for width in replaced.register_sizes)


@pytest.mark.parametrize("width", [True, np.bool_(False)])
def test_periodic_stencil_rejects_boolean_widths(width: Any) -> None:
    """Boolean scalar types never act as periodic axis widths."""
    with pytest.raises(TypeError, match="must be an int"):
        _periodic_encoding({0: 1.0}, (width,))


def test_periodic_shift_lcu_factory_is_publicly_exported() -> None:
    """The factory is public only through the circuit and stdlib surfaces."""
    from qamomile.circuit.stdlib import (
        PeriodicShiftLCUBlockEncoding,
        periodic_shift_lcu_block_encoding,
    )

    assert qmc.periodic_shift_lcu_block_encoding is periodic_shift_lcu_block_encoding
    assert qmc.PeriodicShiftLCUBlockEncoding is PeriodicShiftLCUBlockEncoding


def test_periodic_stencil_descriptor_is_frozen_noncallable_and_identity_based() -> None:
    """The static descriptor has the agreed ABI without value equality."""
    encoding = _periodic_encoding({0: 1j, 1: 0.5}, (2,))
    repeated = _periodic_encoding({0: 1j, 1: 0.5}, (2,))

    assert isinstance(encoding, qmc.PeriodicShiftLCUBlockEncoding)
    assert isinstance(encoding, qmc.LCUBlockEncoding)
    assert tuple(field.name for field in dataclasses.fields(encoding)) == (
        "unitary",
        "normalization",
        "num_signal_qubits",
        "num_system_qubits",
        "register_sizes",
        "offsets",
        "coefficients",
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
    assert type(encoding.normalization) is float
    assert type(encoding.num_signal_qubits) is int
    assert type(encoding.num_system_qubits) is int
    assert isinstance(encoding.register_sizes, tuple)
    assert isinstance(encoding.offsets, tuple)
    assert all(isinstance(offset, tuple) for offset in encoding.offsets)
    assert isinstance(encoding.coefficients, tuple)
    assert not callable(encoding)
    assert not hasattr(encoding, "__dict__")
    assert not hasattr(encoding, "kernel")
    assert not hasattr(encoding, "error_bound")
    assert encoding is not repeated
    assert encoding != repeated

    with pytest.raises(dataclasses.FrozenInstanceError):
        encoding.normalization = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        encoding()  # type: ignore[operator]


def test_periodic_descriptor_metadata_is_deeply_immutable() -> None:
    """Mutable factory inputs cannot mutate stored canonical metadata."""
    register_sizes = [2, 1]
    coefficients = {(0, 0): 1j, (1, -1): 0.5}
    encoding = _periodic_encoding(coefficients, register_sizes)
    expected = (
        encoding.register_sizes,
        encoding.offsets,
        encoding.coefficients,
    )

    register_sizes[0] = 7
    coefficients.clear()

    assert (
        encoding.register_sizes,
        encoding.offsets,
        encoding.coefficients,
    ) == expected
    with pytest.raises(TypeError):
        encoding.offsets[0][0] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        encoding.coefficients[0] = 0j  # type: ignore[index]
    with pytest.raises(TypeError, match="register_sizes must be a tuple"):
        dataclasses.replace(
            encoding,
            register_sizes=list(encoding.register_sizes),
        )


def test_generic_lcu_template_rebinds_periodic_encodings_after_round_trip() -> None:
    """One serialized generic slot accepts Periodic descriptors of new widths."""
    payload = serialize(_static_direct_template)
    restored = deserialize(payload)
    encodings = (
        _periodic_encoding({1: 1.0}, (1,)),
        _periodic_encoding({1: 1.0}, (2,)),
        _periodic_encoding({0: 1.0, 1: 0.5, 2: -0.25j}, (2,)),
    )

    assert restored.input_types == {"encoding": qmc.LCUBlockEncoding}
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.lcu_block_encoding"
    )
    for encoding in encodings:
        specialized = restored.build(encoding=encoding)
        assert specialized.static_bindings == ()
        assert serialize(restored) == payload


def test_periodic_descriptor_canonicalizes_equivalent_normalization() -> None:
    """Direct construction tolerates host rounding and stores the canonical sum."""
    encoding = _periodic_encoding({0: 0.1, 1: 0.2}, (2,))

    replaced = dataclasses.replace(encoding, normalization=0.3)

    assert replaced.normalization == pytest.approx(
        math.fsum((0.1, 0.2)),
        rel=0.0,
        abs=0.0,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"normalization": 0.4}, "normalization"),
        ({"num_signal_qubits": 2}, "num_signal_qubits"),
        ({"num_system_qubits": 3}, "num_system_qubits"),
        ({"offsets": ((1,), (0,))}, "canonically sorted"),
        ({"offsets": ((0,), (0,))}, "unique"),
        ({"offsets": ((0,), (4,))}, "canonical modular residue"),
        ({"coefficients": (0.1 + 0.0j,)}, "same length"),
        ({"coefficients": (0.0j, 0.2 + 0.0j)}, "must be nonzero"),
    ],
)
def test_periodic_descriptor_rejects_inconsistent_metadata(
    changes: dict[str, Any],
    message: str,
) -> None:
    """Direct construction rejects inconsistent method-specific metadata."""
    encoding = _periodic_encoding({0: 0.1, 1: 0.2}, (2,))

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(encoding, **changes)


def test_periodic_stencil_signal_width_tracks_canonical_term_count() -> None:
    """One and two terms use one signal qubit while three terms use two."""
    single = _periodic_encoding({0: 1.0}, (2,))
    two = _periodic_encoding({0: 1.0, 1: 0.5}, (2,))
    three = _periodic_encoding(
        {0: 1.0, 1: 0.5, 2: -0.25j},
        (2,),
    )

    assert single.num_signal_qubits == 1
    assert two.num_signal_qubits == 1
    assert three.num_signal_qubits == 2


def test_periodic_stencil_single_term_omits_prepare_and_select(
    qiskit_transpiler: Any,
) -> None:
    """The pass-through signal ABI does not force dummy LCU operations."""
    encoding = _periodic_encoding({1: 1j}, (2,))
    calls = _nested_calls(
        _lower_first_circuit(
            _build_unitary_kernel(encoding),
            qiskit_transpiler,
        )
    )
    identities = tuple(
        call.callee.identity for call in calls if call.callee.identity is not None
    )

    assert all(identity.key != SELECT_SEMANTIC_KEY for identity in identities)
    assert all(
        identity.key.name != "periodic_shift_lcu_prepare" for identity in identities
    )


def test_periodic_stencil_preserves_tiny_nonzero_terms() -> None:
    """Every representable nonzero term remains part of the encoded operator."""
    encoding = _periodic_encoding(
        {0: 1e-16, 1: 2.0},
        register_sizes=(2,),
    )

    assert encoding.offsets == ((0,), (1,))
    np.testing.assert_allclose(
        encoding.coefficients,
        (1e-16 + 0.0j, 2.0 + 0.0j),
        atol=0.0,
        rtol=0.0,
    )
    assert encoding.normalization == pytest.approx(2.0)


@pytest.mark.parametrize(
    "coefficients",
    [
        {0: 10**1000},
        {0: 1e308, 1: 1e308},
        {1: 1e308, -3: 1e308},
    ],
)
def test_periodic_stencil_rejects_unrepresentable_normalization(
    coefficients: dict[int, complex],
) -> None:
    """Unrepresentable coefficients and overflowing normalizations fail early."""
    with pytest.raises(ValueError, match="finite"):
        _periodic_encoding(coefficients, register_sizes=(2,))


@pytest.mark.parametrize(
    ("signal_delta", "system_delta", "role"),
    [(1, 0, "signal"), (0, 1, "system")],
)
def test_periodic_stencil_rejects_wrong_register_widths(
    qiskit_transpiler: Any,
    signal_delta: int,
    system_delta: int,
    role: str,
) -> None:
    """Concrete caller registers must match the factory's declared ABI."""
    encoding = _periodic_encoding(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        register_sizes=(2,),
    )
    signal_width = encoding.num_signal_qubits + signal_delta
    system_width = encoding.num_system_qubits + system_delta

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        signal = qmc.qubit_array(signal_width, name="signal")
        system = qmc.qubit_array(system_width, name="system")
        signal, system = encoding.unitary(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    with pytest.raises(ValueError, match=rf"requires .* {role} qubits"):
        qiskit_transpiler.transpile(circuit)


def test_periodic_stencil_width_validation_survives_all_composition_paths() -> None:
    """Plain, inverse, control, and nested SELECT report the same width ABI."""
    encoding = _periodic_encoding(
        {0: 1.0, 1: 0.5j, 2: -0.25},
        (2,),
    )
    inverse_unitary = qmc.inverse(encoding.unitary)
    controlled_unitary = qmc.control(encoding.unitary)
    selected_unitary = qmc.select(
        (_identity_registers, encoding.unitary),
        num_index_qubits=1,
    )

    @qmc.qkernel
    def plain_wrong_signal() -> qmc.Bit:
        """Invoke the plain unitary with a narrow signal register."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(2, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverse_wrong_signal() -> qmc.Bit:
        """Invoke the inverse unitary with a narrow signal register."""
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(2, "system")
        signal, _ = inverse_unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def control_wrong_signal() -> qmc.Bit:
        """Invoke the controlled unitary with a narrow signal register."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(2, "system")
        control, signal, _ = controlled_unitary(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def select_wrong_signal() -> qmc.Bit:
        """Invoke the nested SELECT with a narrow signal register."""
        outer = qmc.qubit("outer")
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(2, "system")
        outer, signal, _ = selected_unitary(outer, signal, system)
        return qmc.measure(outer)

    @qmc.qkernel
    def plain_wrong_system() -> qmc.Bit:
        """Invoke the plain unitary with a wide system register."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(3, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def inverse_wrong_system() -> qmc.Bit:
        """Invoke the inverse unitary with a wide system register."""
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(3, "system")
        signal, _ = inverse_unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def control_wrong_system() -> qmc.Bit:
        """Invoke the controlled unitary with a wide system register."""
        control = qmc.qubit("control")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(3, "system")
        control, signal, _ = controlled_unitary(control, signal, system)
        return qmc.measure(control)

    @qmc.qkernel
    def select_wrong_system() -> qmc.Bit:
        """Invoke the nested SELECT with a wide system register."""
        outer = qmc.qubit("outer")
        signal = qmc.qubit_array(2, "signal")
        system = qmc.qubit_array(3, "system")
        outer, signal, _ = selected_unitary(outer, signal, system)
        return qmc.measure(outer)

    for kernel in (
        plain_wrong_signal,
        inverse_wrong_signal,
        control_wrong_signal,
        select_wrong_signal,
    ):
        with pytest.raises(ValueError) as error:
            kernel.build()
        assert str(error.value) == (
            "periodic shift LCU block encoding requires 2 signal qubits, got 1."
        )

    for kernel in (
        plain_wrong_system,
        inverse_wrong_system,
        control_wrong_system,
        select_wrong_system,
    ):
        with pytest.raises(ValueError) as error:
            kernel.build()
        assert str(error.value) == (
            "periodic shift LCU block encoding requires 2 system qubits, got 3."
        )


def test_distinct_periodic_instances_compose_in_one_caller(
    qiskit_transpiler: Any,
) -> None:
    """Equal-signature ordinary qkernels retain distinct shift semantics."""
    plus_one = _periodic_encoding({1: 1.0}, (2,))
    plus_two = _periodic_encoding({2: 1.0}, (2,))
    controlled_one = qmc.control(plus_one.unitary)
    controlled_two = qmc.control(plus_two.unitary)

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Bit]:
        """Apply shifts plus one and plus two under one control."""
        control = qmc.x(qmc.qubit("control"))
        signal = qmc.qubit_array(1, "signal")
        system = qmc.qubit_array(2, "system")
        control, signal, system = controlled_one(control, signal, system)
        control, signal, system = controlled_two(control, signal, system)
        return qmc.measure(system)

    executable = qiskit_transpiler.transpile(circuit)
    result = executable.sample(qiskit_transpiler.executor(), shots=16).result()
    assert result.results == [((1, 1), 16)]


def test_zero_periodic_encoding_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The exact zero path and its transforms execute on every backend."""
    lcu = PeriodicShiftLCU.from_coefficients({}, register_sizes=(2,))
    encoding = qmc.periodic_shift_lcu_block_encoding(lcu)
    inverse_unitary = qmc.inverse(encoding.unitary)
    controlled_unitary = qmc.control(encoding.unitary)
    selected_unitary = qmc.select(
        (_identity_registers, encoding.unitary),
        num_index_qubits=1,
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        """Apply the zero encoding and measure its flipped signal."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Apply the zero encoding and observe its flipped signal."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.expval(signal[0], observable)

    @qmc.qkernel
    def inverse_kernel() -> qmc.Bit:
        """Apply the inverse zero encoding and measure its signal."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = inverse_unitary(signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def control_kernel() -> qmc.Bit:
        """Apply the controlled zero encoding and measure its signal."""
        outer = qmc.x(qmc.qubit("outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        _, signal, _ = controlled_unitary(outer, signal, system)
        return qmc.measure(signal[0])

    @qmc.qkernel
    def select_kernel() -> qmc.Bit:
        """Apply the selected zero encoding and measure its signal."""
        index = qmc.x(qmc.qubit("index"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        _, signal, _ = selected_unitary(index, signal, system)
        return qmc.measure(signal[0])

    for kernel in (sample_kernel, inverse_kernel, control_kernel, select_kernel):
        sampled = _sample_only_outcome(sdk_transpiler, kernel, {})
        assert int(sampled) == 1
    observed = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {"observable": qm_o.Z(0)},
    )

    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(-1.0, abs=tolerance)


@pytest.mark.parametrize(
    ("template", "expected_bits"),
    [
        (_static_direct_template, (1, 0)),
        (_static_inverse_template, (1, 1)),
        (_static_control_template, (1, 0)),
        (_static_select_template, (1, 0)),
    ],
    ids=["direct", "inverse-helper", "outer-control", "nested-select"],
)
def test_generic_static_binding_composes_and_executes_on_every_sdk(
    sdk_transpiler: Any,
    template: qmc.QKernel,
    expected_bits: tuple[int, ...],
) -> None:
    """Serialized generic LCU slots materialize Periodic encodings everywhere."""
    restored = deserialize(serialize(template))
    encoding = _periodic_encoding({1: 1.0}, (2,))

    sampled = _sample_only_outcome(
        sdk_transpiler,
        restored,
        {"encoding": encoding},
    )

    assert tuple(sampled) == expected_bits


def test_periodic_specific_static_binding_executes_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """The public Periodic subtype annotation round-trips and binds directly."""
    restored = deserialize(serialize(_periodic_static_direct_template))
    encoding = _periodic_encoding({1: 1.0}, (2,))

    assert len(restored.block.static_bindings) == 1
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.periodic_shift_lcu_block_encoding"
    )
    sampled = _sample_only_outcome(
        sdk_transpiler,
        restored,
        {"encoding": encoding},
    )

    assert tuple(sampled) == (1, 0)


def test_periodic_specific_static_binding_rejects_other_lcu_producers(
    qiskit_transpiler: Any,
) -> None:
    """A Periodic-specific static slot rejects a Pauli descriptor."""
    restored = deserialize(serialize(_periodic_static_direct_template))
    pauli = qmc.pauli_lcu_block_encoding(
        PauliLCU.from_matrix(np.eye(2, dtype=np.complex128))
    )

    with pytest.raises(TypeError, match="PeriodicShiftLCUBlockEncoding"):
        qiskit_transpiler.transpile(restored, bindings={"encoding": pauli})


def test_serialized_generic_lcu_template_rebinds_periodic_on_every_sdk(
    sdk_transpiler: Any,
) -> None:
    """One generic payload executes Periodic encodings through both SDK APIs."""
    sample_payload = serialize(_static_round_trip_sample_template)
    expval_payload = serialize(_static_round_trip_expval_template)
    sample_template = deserialize(sample_payload)
    expval_template = deserialize(expval_payload)
    encodings = (
        _periodic_encoding({1: np.exp(0.37j)}, (1,)),
        _periodic_encoding(
            {0: 1j, 1: 0.5, 2: -0.25j},
            (2,),
        ),
    )

    assert {
        (encoding.num_signal_qubits, encoding.num_system_qubits)
        for encoding in encodings
    } == {(1, 1), (2, 2)}
    for encoding in encodings:
        sampled = _sample_only_outcome(
            sdk_transpiler,
            sample_template,
            {"encoding": encoding},
        )
        assert _is_all_zero(sampled)

        weights = tuple(
            1.0 / (index + 1) for index in range(encoding.num_system_qubits)
        )
        observed = _run_expval(
            sdk_transpiler,
            expval_template,
            {
                "encoding": encoding,
                "observable": _z_hamiltonian(weights),
            },
        )
        tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert observed == pytest.approx(sum(weights), abs=tolerance)
        assert serialize(sample_template) == sample_payload
        assert serialize(expval_template) == expval_payload


@pytest.mark.parametrize("num_system_qubits", [1, 2])
@pytest.mark.parametrize("seed", [0, 42])
def test_two_term_periodic_encoding_samples_and_estimates_on_every_sdk(
    sdk_transpiler: Any,
    num_system_qubits: int,
    seed: int,
) -> None:
    """Random two-term stencils execute sampler and estimator backend paths.

    Acting on ``|0>`` makes the identity and one-step-shift outputs orthogonal,
    so success probability is ``(|a|² + |b|²) / normalization²``. Projecting
    signal zero and measuring system Y gives
    ``2 Im(conj(a) b) / normalization²``.
    """
    rng = np.random.default_rng(seed)
    identity_weight = rng.uniform(0.4, 1.2) * np.exp(
        1j * rng.uniform(-math.pi, math.pi)
    )
    shift_weight = rng.uniform(0.4, 1.2) * np.exp(1j * rng.uniform(-math.pi, math.pi))
    encoding = _periodic_encoding(
        {0: identity_weight, 1: shift_weight},
        (num_system_qubits,),
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Vector[qmc.Bit]:
        """Apply the encoding and measure its success signal."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        """Measure the success-projected system-Y interference."""
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, system = encoding.unitary(signal, system)
        return qmc.expval((signal[0], system[0]), observable)

    shots = 2048
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        sdk_transpiler.transpiler.executor(),
        shots=shots,
    ).result()
    expected_success = (
        abs(identity_weight) ** 2 + abs(shift_weight) ** 2
    ) / encoding.normalization**2
    sampling_tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_success,
        abs=sampling_tolerance,
    )

    projected_y = qm_o.Hamiltonian(num_qubits=2)
    projected_y.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 1),), 0.5)
    projected_y.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        ),
        0.5,
    )
    observed_y = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {"observable": projected_y},
    )
    expected_y = (
        2.0
        * np.imag(np.conj(identity_weight) * shift_weight)
        / encoding.normalization**2
    )
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed_y == pytest.approx(expected_y, abs=tolerance)


@pytest.mark.parametrize(
    ("register_sizes", "offset"),
    [
        ((4,), (7,)),
        ((3, 3), (4, -3)),
    ],
    ids=["large-one-dimensional", "half-turn-and-negative-multiaxis"],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_binary_periodic_shifts_execute_on_every_sdk(
    sdk_transpiler: Any,
    register_sizes: tuple[int, ...],
    offset: tuple[int, ...],
    seed: int,
) -> None:
    """Binary, half-turn, and multi-axis shifts sample and estimate exactly."""
    rng = np.random.default_rng(seed)
    encoding = _periodic_encoding({offset: 1.0}, register_sizes)
    initial_bits = (
        rng.integers(
            0,
            2,
            size=encoding.num_system_qubits,
        )
        .astype(int)
        .tolist()
    )
    expected_bits = _shifted_basis_bits(initial_bits, register_sizes, offset)
    z_coefficients = rng.uniform(
        -1.0,
        1.0,
        size=encoding.num_system_qubits,
    ).tolist()

    @qmc.qkernel
    def sample_kernel(
        bits: qmc.Vector[qmc.UInt],
    ) -> qmc.Vector[qmc.Bit]:
        """Apply one captured shift and measure the resulting basis state.

        Args:
            bits (qmc.Vector[qmc.UInt]): Initial computational-basis bits.

        Returns:
            qmc.Vector[qmc.Bit]: Measured system bits after the shift.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        system = _prepare_basis(system, bits)
        signal, system = encoding.unitary(signal, system)
        return qmc.measure(system)

    @qmc.qkernel
    def expval_kernel(
        bits: qmc.Vector[qmc.UInt],
        observable: qmc.Observable,
    ) -> qmc.Float:
        """Apply one captured shift and estimate a diagonal observable.

        Args:
            bits (qmc.Vector[qmc.UInt]): Initial computational-basis bits.
            observable (qmc.Observable): Observable evaluated after the shift.

        Returns:
            qmc.Float: Expectation value of ``observable``.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        system = _prepare_basis(system, bits)
        signal, system = encoding.unitary(signal, system)
        return qmc.expval(system, observable)

    sampled = _sample_only_outcome(
        sdk_transpiler,
        sample_kernel,
        {"bits": initial_bits},
    )
    observed = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {
            "bits": initial_bits,
            "observable": _z_hamiltonian(z_coefficients),
        },
    )

    assert tuple(sampled) == expected_bits
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(
        _expected_z(z_coefficients, expected_bits),
        abs=tolerance,
    )


@pytest.mark.parametrize("register_sizes", [(1,), (2,), (1, 2)])
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_periodic_stencil_inverse_cross_backend_sample_and_expval(
    sdk_transpiler: Any,
    register_sizes: tuple[int, ...],
    seed: int,
) -> None:
    """Random complex stencils cancel with inverse on every backend path."""
    rng = np.random.default_rng(seed)
    dimensions = len(register_sizes)
    offsets = [tuple(0 for _ in register_sizes)]
    for axis in range(dimensions):
        plus = [0] * dimensions
        plus[axis] = 1
        offsets.append(tuple(plus))
        minus = [0] * dimensions
        minus[axis] = -1
        offsets.append(tuple(minus))
    coefficients = {
        offset: complex(*rng.uniform(-1.0, 1.0, size=2)) for offset in offsets
    }
    encoding = _periodic_encoding(coefficients, register_sizes)
    width = encoding.num_system_qubits
    bits = rng.integers(0, 2, size=width).astype(int).tolist()
    z_coefficients = rng.uniform(-1.0, 1.0, size=width).tolist()
    hamiltonian = _z_hamiltonian(z_coefficients)

    @qmc.qkernel
    def sample_kernel(
        initial_bits: qmc.Vector[qmc.UInt],
    ) -> qmc.Vector[qmc.Bit]:
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        system = _prepare_basis(system, initial_bits)
        signal, system = encoding.unitary(signal, system)
        signal, system = qmc.inverse(encoding.unitary)(signal, system)
        _ = qmc.measure(signal)
        return qmc.measure(system)

    @qmc.qkernel
    def expval_kernel(
        initial_bits: qmc.Vector[qmc.UInt],
        observable: qmc.Observable,
    ) -> qmc.Float:
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        system = _prepare_basis(system, initial_bits)
        signal, system = encoding.unitary(signal, system)
        signal, system = qmc.inverse(encoding.unitary)(signal, system)
        return qmc.expval(system, observable)

    sampled = _sample_only_outcome(
        sdk_transpiler,
        sample_kernel,
        {"initial_bits": bits},
    )
    observed = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {"initial_bits": bits, "observable": hamiltonian},
    )
    assert tuple(sampled) == tuple(bits)
    assert observed == pytest.approx(_expected_z(z_coefficients, bits), abs=1e-7)


@pytest.mark.parametrize("composition", ["control", "select"])
@pytest.mark.parametrize(
    "shift_weight",
    [None, 0.5],
    ids=["single-term", "multi-term"],
)
def test_periodic_phase_composes_cross_backend(
    sdk_transpiler: Any,
    composition: str,
    shift_weight: float | None,
) -> None:
    """Outer control and nested SELECT preserve a non-Clifford block phase.

    On the all-zero input only the identity term overlaps the initial state,
    so outer-branch interference directly measures ``a / normalization``.
    The single-term case exercises the optimized path without PREPARE or the
    inner SELECT, while the multi-term case exercises the complete LCU path.
    """
    phase = math.pi / 3.0
    identity_weight = np.exp(1j * phase)
    coefficients = {0: identity_weight}
    if shift_weight is not None:
        coefficients[1] = shift_weight
    encoding = _periodic_encoding(
        coefficients,
        (1,),
    )

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        outer = qmc.h(qmc.qubit(name="outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        if composition == "control":
            outer, signal, system = qmc.control(encoding.unitary)(
                outer,
                signal,
                system,
            )
        else:
            outer, signal, system = qmc.select([_identity_registers, encoding.unitary])(
                outer, signal, system
            )
        outer = qmc.h(outer)
        return qmc.measure(outer)

    @qmc.qkernel
    def expval_kernel(observable: qmc.Observable) -> qmc.Float:
        outer = qmc.h(qmc.qubit(name="outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        if composition == "control":
            outer, signal, system = qmc.control(encoding.unitary)(
                outer,
                signal,
                system,
            )
        else:
            outer, signal, system = qmc.select([_identity_registers, encoding.unitary])(
                outer, signal, system
            )
        return qmc.expval(outer, observable)

    shots = 4096
    sample_executable = sdk_transpiler.transpiler.transpile(sample_kernel)
    sample_result = sample_executable.sample(
        sdk_transpiler.transpiler.executor(),
        shots=shots,
    ).result()
    observed = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {"observable": qm_o.Y(0)},
    )
    overlap = identity_weight / encoding.normalization
    expected_zero = (1.0 + overlap.real) / 2.0
    sampling_tolerance = (
        6.0 * math.sqrt(expected_zero * (1.0 - expected_zero) / shots) + 0.02
    )
    assert _zero_probability(sample_result.results) == pytest.approx(
        expected_zero,
        abs=sampling_tolerance,
    )
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert observed == pytest.approx(overlap.imag, abs=tolerance)
