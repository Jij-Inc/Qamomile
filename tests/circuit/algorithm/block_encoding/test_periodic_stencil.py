"""Tests for periodic-stencil LCU block encodings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o


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


@pytest.mark.parametrize(
    ("register_sizes", "coefficients"),
    [
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
    from qiskit.quantum_info import Operator

    encoding = qmc.periodic_stencil_block_encoding(coefficients, register_sizes)

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        signal, system = encoding(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    executable = qiskit_transpiler.transpile(circuit)
    unitary = np.asarray(
        Operator(
            executable.compiled_quantum[0].circuit.remove_final_measurements(
                inplace=False
            )
        ).data,
        dtype=np.complex128,
    )
    system_dimension = 1 << encoding.num_system_qubits
    signal_dimension = 1 << encoding.num_signal_qubits
    zero_signal_indices = np.arange(system_dimension) * signal_dimension
    encoded_block = unitary[np.ix_(zero_signal_indices, zero_signal_indices)]
    expected = _stencil_matrix(coefficients, register_sizes)
    assert np.allclose(
        encoding.normalization * encoded_block,
        expected,
        atol=1e-8,
        rtol=1e-8,
    )


def test_periodic_stencil_combines_equivalent_offsets() -> None:
    """Offsets equal modulo an axis size combine before normalization."""
    encoding = qmc.periodic_stencil_block_encoding(
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


def test_periodic_laplacian_normalizations() -> None:
    """Canonical one- and two-dimensional Laplacians use factors four and eight."""
    one_dimensional = qmc.periodic_stencil_block_encoding(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        register_sizes=(3,),
    )
    two_dimensional = qmc.periodic_stencil_block_encoding(
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


@pytest.mark.parametrize(
    ("coefficients", "register_sizes", "exception", "message"),
    [
        ({}, (1,), ValueError, "at least one stencil term"),
        ({0: 1.0}, (), ValueError, "at least one axis"),
        ({0: 1.0}, (0,), ValueError, "must be positive"),
        ({(0, 1): 1.0}, (2,), ValueError, "dimensions"),
        ({0: "1.0"}, (1,), TypeError, "must be numeric"),
        ({0: complex(np.inf, 0.0)}, (1,), ValueError, "must be finite"),
        ({0: 1.0, 2: -1.0}, (1,), ValueError, "zero operator"),
    ],
)
def test_periodic_stencil_rejects_invalid_descriptions(
    coefficients: dict[Any, complex],
    register_sizes: tuple[int, ...],
    exception: type[Exception],
    message: str,
) -> None:
    """Invalid dimensions, coefficients, and zero operators fail clearly."""
    with pytest.raises(exception, match=message):
        qmc.periodic_stencil_block_encoding(coefficients, register_sizes)


def test_periodic_stencil_factory_is_publicly_exported() -> None:
    """The method-specific factory is available from both circuit API levels."""
    from qamomile.circuit.algorithm import periodic_stencil_block_encoding

    assert qmc.periodic_stencil_block_encoding is periodic_stencil_block_encoding


def test_periodic_stencil_preserves_tiny_nonzero_terms() -> None:
    """Every representable nonzero term remains part of the encoded operator."""
    encoding = qmc.periodic_stencil_block_encoding(
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
    ],
)
def test_periodic_stencil_rejects_unrepresentable_normalization(
    coefficients: dict[int, complex],
) -> None:
    """Unrepresentable coefficients and overflowing normalizations fail early."""
    with pytest.raises(ValueError, match="finite"):
        qmc.periodic_stencil_block_encoding(coefficients, register_sizes=(2,))


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
    encoding = qmc.periodic_stencil_block_encoding(
        {-1: 1.0, 0: -2.0, 1: 1.0},
        register_sizes=(2,),
    )
    signal_width = encoding.num_signal_qubits + signal_delta
    system_width = encoding.num_system_qubits + system_delta

    @qmc.qkernel
    def circuit() -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        signal = qmc.qubit_array(signal_width, name="signal")
        system = qmc.qubit_array(system_width, name="system")
        signal, system = encoding(signal, system)
        return qmc.measure(signal), qmc.measure(system)

    with pytest.raises(ValueError, match=rf"requires .* {role} qubits"):
        qiskit_transpiler.transpile(circuit)


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
    encoding = qmc.periodic_stencil_block_encoding(coefficients, register_sizes)
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
        signal, system = encoding(signal, system)
        signal, system = qmc.inverse(encoding.kernel)(signal, system)
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
        signal, system = encoding(signal, system)
        signal, system = qmc.inverse(encoding.kernel)(signal, system)
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
def test_negative_identity_phase_composes_cross_backend(
    sdk_transpiler: Any,
    composition: str,
) -> None:
    """Outer control and nested SELECT observe a negative identity term."""
    encoding = qmc.periodic_stencil_block_encoding({0: -2.0}, (1,))

    @qmc.qkernel
    def sample_kernel() -> qmc.Bit:
        outer = qmc.h(qmc.qubit(name="outer"))
        signal = qmc.qubit_array(encoding.num_signal_qubits, name="signal")
        system = qmc.qubit_array(encoding.num_system_qubits, name="system")
        if composition == "control":
            outer, signal, system = qmc.control(encoding.kernel)(
                outer,
                signal,
                system,
            )
        else:
            outer, signal, system = qmc.select([_identity_registers, encoding.kernel])(
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
            outer, signal, system = qmc.control(encoding.kernel)(
                outer,
                signal,
                system,
            )
        else:
            outer, signal, system = qmc.select([_identity_registers, encoding.kernel])(
                outer, signal, system
            )
        return qmc.expval(outer, observable)

    sampled = _sample_only_outcome(sdk_transpiler, sample_kernel, {})
    observed = _run_expval(
        sdk_transpiler,
        expval_kernel,
        {"observable": qm_o.X(0)},
    )
    assert sampled == 1
    assert observed == pytest.approx(-1.0, abs=1e-7)
