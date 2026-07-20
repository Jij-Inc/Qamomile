"""Tests for executable, body-derived Shor order finding."""

from __future__ import annotations

from functools import lru_cache

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel


def _basis_value(bits: tuple[int, ...]) -> int:
    """Decode a little-endian measured bit tuple.

    Args:
        bits (tuple[int, ...]): Measured bits in register order.

    Returns:
        int: Unsigned basis-state value.
    """
    return sum(bit << index for index, bit in enumerate(bits))


@lru_cache(maxsize=None)
def _shor_estimate(base: int, modulus: int) -> qmc.ResourceEstimate:
    """Cache the expensive expansion of one specialized Shor body."""
    return qmc.shor_order_finding(base, modulus).estimate_resources()


def test_shor_factory_returns_one_executable_qkernel() -> None:
    """The same returned QKernel supports estimation and transpilation."""
    kernel = qmc.shor_order_finding(base=2, modulus=15)

    assert isinstance(kernel, QKernel)
    estimate = _shor_estimate(2, 15)
    assert estimate.parameters == {}
    assert estimate.qubits == 21
    assert estimate.width.dirty_ancilla_qubits == 0
    assert estimate.gates.total > 0
    assert "modmul_const" not in estimate.calls.calls_by_name
    assert estimate.trace is None

    assert kernel.build().operations


@pytest.mark.parametrize(
    ("base", "modulus"),
    [(2, 3), (2, 5), (2, 15)],
)
def test_shor_width_is_body_derived_three_n_plus_constant(
    base: int,
    modulus: int,
) -> None:
    """Specialized bodies expose the expected ``3n + w + 7`` peak width."""
    n = modulus.bit_length()
    estimate = _shor_estimate(base, modulus)

    assert estimate.qubits == 3 * n + 2 + 7


def test_shor_skips_identity_modular_multiplication_rounds() -> None:
    """Identity powers retain phase work without expanding arithmetic."""
    arithmetic_rounds = qmc.shor_order_finding(
        base=2,
        modulus=15,
        precision=2,
    ).estimate_resources()
    full_schedule = _shor_estimate(2, 15)

    assert arithmetic_rounds.gates.total == 3420
    assert full_schedule.gates.total == 3465
    assert full_schedule.gates.two_qubit == arithmetic_rounds.gates.two_qubit
    assert full_schedule.gates.multi_qubit == arithmetic_rounds.gates.multi_qubit


def test_shor_rejects_invalid_window_size() -> None:
    """The order-finding factory rejects an empty lookup window."""
    with pytest.raises(ValueError, match="window_size must be positive"):
        qmc.shor_order_finding(base=2, modulus=15, window_size=0)


def test_shor_rejects_invalid_precision() -> None:
    """The order-finding factory rejects an empty phase estimate."""
    with pytest.raises(ValueError, match="precision must be positive"):
        qmc.shor_order_finding(base=2, modulus=15, precision=0)


@pytest.mark.parametrize(
    ("base", "modulus", "message"),
    [
        (2, 2, "greater than two"),
        (1, 15, "1 < base"),
        (3, 15, "coprime"),
    ],
)
def test_shor_rejects_invalid_problem_instances(
    base: int,
    modulus: int,
    message: str,
) -> None:
    """Invalid order-finding instances fail before tracing.

    Args:
        base (int): Candidate base.
        modulus (int): Candidate modulus.
        message (str): Expected diagnostic fragment.
    """
    with pytest.raises(ValueError, match=message):
        qmc.shor_order_finding(base=base, modulus=modulus)


def test_small_shor_order_finding_recovers_period_two(sdk_transpiler) -> None:
    """The simulatable two-bit instance recovers the period-two peaks."""
    if sdk_transpiler.backend_name == "quri_parts":
        pytest.skip("QURI Parts cannot represent Shor's mid-circuit reset")

    kernel = qmc.shor_order_finding(base=2, modulus=3)
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(kernel)
    shots = 16 if sdk_transpiler.backend_name == "cudaq" else 128
    result = executable.sample(transpiler.executor(), shots=shots).result()

    counts = {_basis_value(bits): count for bits, count in result.results}
    targets = {0, 8}
    on_peak = sum(counts.get(value, 0) for value in targets)

    assert on_peak / result.shots > 0.9
    assert all(counts.get(value, 0) > 0 for value in targets)
    if sdk_transpiler.backend_name == "qiskit":
        assert all(counts.get(value, 0) / result.shots > 0.3 for value in targets)


def test_four_bit_shor_transpiles_without_statevector_execution(
    sdk_transpiler,
) -> None:
    """Transpile the 21-qubit benchmark without allocating its statevector.

    This test covers the realistic four-bit circuit without coupling its
    runtime to statevector sampling.

    Args:
        sdk_transpiler: Parametrized SDK backend fixture.
    """
    if sdk_transpiler.backend_name == "quri_parts":
        pytest.skip("QURI Parts cannot represent Shor's mid-circuit reset")

    kernel = qmc.shor_order_finding(base=2, modulus=15)
    executable = sdk_transpiler.transpiler.transpile(kernel)

    assert _shor_estimate(2, 15).qubits == 21
    assert executable.compiled_quantum
    assert executable.plan.steps


def test_ekera_hastad_uses_two_short_exponent_registers() -> None:
    """The short-DLP schedules reuse the same low-width arithmetic body."""
    kernel = qmc.ekera_hastad_factoring(generator=2, modulus=5, window_size=2)
    estimate = kernel.estimate_resources()

    assert isinstance(kernel, QKernel)
    assert estimate.qubits == 18
    assert "modmul_const" not in estimate.calls.calls_by_name
    assert len(kernel.output_types) == 1
    assert kernel.output_types[0] == qmc.Vector[qmc.Bit]


def test_small_ekera_hastad_schedule_executes(qiskit_transpiler) -> None:
    """The 2 mod 3 short-DLP schedule executes with phase-qubit reuse."""
    kernel = qmc.ekera_hastad_factoring(generator=2, modulus=3, window_size=2)
    result = (
        qiskit_transpiler.transpile(kernel)
        .sample(qiskit_transpiler.executor(), shots=64)
        .result()
    )

    counts = {}
    for bits, count in result.results:
        long_bits = bits[:4]
        short_bits = bits[4:]
        counts[(_basis_value(long_bits), _basis_value(short_bits))] = count
    assert sum(counts.get((peak, 0), 0) for peak in (0, 8)) / result.shots > 0.9


def test_shor_algorithms_live_in_algorithm_namespace() -> None:
    """Algorithms are exported from algorithm, not the primitive stdlib."""
    import qamomile.circuit.algorithm as algorithm
    import qamomile.circuit.stdlib as stdlib

    assert algorithm.shor_order_finding is qmc.shor_order_finding
    assert algorithm.ekera_hastad_factoring is qmc.ekera_hastad_factoring
    assert not hasattr(stdlib, "shor_order_finding")
    assert not hasattr(stdlib, "windowed_modmul_const")
