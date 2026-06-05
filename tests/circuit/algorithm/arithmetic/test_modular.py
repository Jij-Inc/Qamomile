"""Tests for modular arithmetic algorithm primitives."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.arithmetic import modular_decrement, modular_increment


@qmc.qkernel
def _prepare_basis_prefix(
    q: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
    n: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Prepare ``q[0:n]`` in the computational basis state labeled by ``bits``."""
    for i in qmc.range(n):
        angle = cast(qmc.Float, math.pi * bits[i])
        q[i] = qmc.rx(q[i], angle)
    return q


@qmc.qkernel
def _shift_sample_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    direction: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Prepare a basis state, shift it, and measure the register."""
    q = qmc.qubit_array(n, name="q")
    q = _prepare_basis_prefix(q, bits, n)
    if direction == qmc.uint(0):
        q = modular_increment(q)
    else:
        q = modular_decrement(q)
    return qmc.measure(q)


@qmc.qkernel
def _shift_expval_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    direction: qmc.UInt,
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Prepare a shifted basis state and return a diagonal expectation value."""
    q = qmc.qubit_array(n, name="q")
    q = _prepare_basis_prefix(q, bits, n)
    if direction == qmc.uint(0):
        q = modular_increment(q)
    else:
        q = modular_decrement(q)
    return qmc.expval(q, hamiltonian)


@qmc.qkernel
def _controlled_shift_sample_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    direction: qmc.UInt,
    enabled: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Apply a qmc.control-created modular shift and measure the register."""
    q = qmc.qubit_array(n, name="q")
    q = _prepare_basis_prefix(q, bits, n)
    control = qmc.qubit(name="control")
    if enabled == qmc.uint(1):
        control = qmc.x(control)

    if direction == qmc.uint(0):
        controlled_shift = qmc.control(modular_increment, num_controls=1)
        control, q = controlled_shift(control, q)
    else:
        controlled_shift = qmc.control(modular_decrement, num_controls=1)
        control, q = controlled_shift(control, q)
    return qmc.measure(q)


@qmc.qkernel
def _controlled_shift_expval_kernel(
    n: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    direction: qmc.UInt,
    enabled: qmc.UInt,
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Apply a qmc.control-created modular shift and return an expval."""
    q = qmc.qubit_array(n, name="q")
    q = _prepare_basis_prefix(q, bits, n)
    control = qmc.qubit(name="control")
    if enabled == qmc.uint(1):
        control = qmc.x(control)

    if direction == qmc.uint(0):
        controlled_shift = qmc.control(modular_increment, num_controls=1)
        control, q = controlled_shift(control, q)
    else:
        controlled_shift = qmc.control(modular_decrement, num_controls=1)
        control, q = controlled_shift(control, q)
    return qmc.expval(q, hamiltonian)


_SIZES = [1, 2, 3, 5]
_SEEDS = [0, 1, 2, 42]
_CASES = [
    ("plus", "plain"),
    ("minus", "plain"),
    ("plus", "controlled"),
    ("minus", "controlled"),
]


def _backend_name(transpiler: Any) -> str:
    """Return a stable test backend label for ``transpiler``."""
    module = transpiler.__class__.__module__
    if ".quri_parts." in module:
        return "quri_parts"
    if ".cudaq." in module:
        return "cudaq"
    if ".qiskit." in module:
        return "qiskit"
    return module


def _random_bits(rng: np.random.Generator, n: int) -> list[int]:
    """Draw a seeded computational-basis bit pattern."""
    return [int(bit) for bit in rng.integers(0, 2, size=n)]


def _shift_bits(bits: list[int], direction: str) -> list[int]:
    """Return the exact little-endian cyclic shift of ``bits``."""
    n = len(bits)
    value = sum(bit << index for index, bit in enumerate(bits))
    delta = 1 if direction == "plus" else -1
    shifted = (value + delta) % (1 << n)
    return [(shifted >> index) & 1 for index in range(n)]


def _expected_bits(bits: list[int], direction: str, enabled: int | None) -> list[int]:
    """Return expected system measurement bits."""
    return _shift_bits(bits, direction) if enabled is None or enabled else bits


def _z_hamiltonian(coeffs: list[float]) -> qm_o.Hamiltonian:
    """Build ``sum_i coeffs[i] * Z(i)``."""
    hamiltonian = qm_o.Hamiltonian()
    for index, coeff in enumerate(coeffs):
        hamiltonian += float(coeff) * qm_o.Z(index)
    return hamiltonian


def _exact_z_expval(coeffs: list[float], bits: list[int]) -> float:
    """Evaluate ``sum_i coeffs[i] * (1 - 2 * bits[i])``."""
    return float(sum(coeff * (1.0 - 2.0 * bit) for coeff, bit in zip(coeffs, bits)))


def _assert_deterministic(
    results: Any,
    expected: list[int],
    shots: int,
    *,
    context: str,
) -> None:
    """Assert every sample equals ``expected``."""
    total = 0
    expected_tuple = tuple(expected)
    for sampled, count in results:
        sampled_tuple = tuple(int(bit) for bit in sampled)
        assert sampled_tuple == expected_tuple, (
            f"{context}: sampled {sampled_tuple}, expected {expected_tuple}"
        )
        total += int(count)
    assert total == shots, f"{context}: got {total} shots, expected {shots}"


def _sample(
    transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any],
    shots: int,
) -> Any:
    """Transpile and sample ``kernel`` with compile-time bindings."""
    executable = transpiler.transpile(kernel, bindings=bindings)
    job = executable.sample(transpiler.executor(), bindings={}, shots=shots)
    return job.result().results


def _expval(transpiler: Any, kernel: Any, bindings: dict[str, Any]) -> float:
    """Transpile and run ``kernel`` with compile-time bindings."""
    executable = transpiler.transpile(kernel, bindings=bindings)
    return float(executable.run(transpiler.executor(), bindings={}).result())


def _expval_if_supported(
    transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any],
    backend_name: str,
) -> float:
    """Return expval or mark a known backend limitation explicitly."""
    try:
        return _expval(transpiler, kernel, bindings)
    except TypeError as exc:
        if (
            backend_name == "cudaq"
            and "cudaq.observe() is not supported for runtime control flow" in str(exc)
        ):
            pytest.xfail(
                "CUDA-Q observe() does not support the runtime control flow "
                "used by controlled modular arithmetic yet."
            )
        raise


def _bindings_for_case(
    n: int,
    bits: list[int],
    direction: str,
    mode: str,
    enabled: int | None,
    hamiltonian: qm_o.Hamiltonian,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Build sample and expval inputs for one modular arithmetic case."""
    direction_id = 0 if direction == "plus" else 1
    if mode == "controlled":
        if enabled is None:
            raise ValueError("controlled modular arithmetic cases require enabled")
        sample_bindings = {
            "n": n,
            "bits": bits,
            "direction": direction_id,
            "enabled": enabled,
        }
        return (
            sample_bindings,
            sample_bindings | {"hamiltonian": hamiltonian},
            _controlled_shift_sample_kernel,
            _controlled_shift_expval_kernel,
        )

    sample_bindings = {"n": n, "bits": bits, "direction": direction_id}
    return (
        sample_bindings,
        sample_bindings | {"hamiltonian": hamiltonian},
        _shift_sample_kernel,
        _shift_expval_kernel,
    )


def _run_shift_case(
    transpiler: Any,
    backend_name: str,
    n: int,
    bits: list[int],
    direction: str,
    mode: str,
    enabled: int | None,
    coeffs: list[float],
    shots: int,
) -> None:
    """Execute sample and expval assertions for one concrete case."""
    hamiltonian = _z_hamiltonian(coeffs)
    expected = _expected_bits(bits, direction, enabled)
    sample_bindings, expval_bindings, sample_kernel, expval_kernel = _bindings_for_case(
        n, bits, direction, mode, enabled, hamiltonian
    )
    context = (
        f"{transpiler.__class__.__name__} n={n} "
        f"direction={direction} mode={mode} enabled={enabled}"
    )

    sample_results = _sample(transpiler, sample_kernel, sample_bindings, shots)
    observed = _expval_if_supported(
        transpiler,
        expval_kernel,
        expval_bindings,
        backend_name,
    )

    _assert_deterministic(sample_results, expected, shots, context=context)
    assert np.isclose(observed, _exact_z_expval(coeffs, expected), atol=1e-8), (
        f"{context}: expval={observed}, expected={_exact_z_expval(coeffs, expected)}"
    )


def _skip_unsupported_backend_case(backend_name: str, n: int, mode: str) -> None:
    """Skip modular arithmetic cases outside a backend's current support."""
    if backend_name == "quri_parts" and n > 2:
        pytest.skip(
            "QURI Parts cannot emit the multi-controlled X gates required "
            "for modular registers larger than two qubits yet."
        )
    if backend_name == "quri_parts" and mode == "controlled" and n > 1:
        pytest.skip(
            "QURI Parts cannot yet recursively emit the BinOp-bearing loop body "
            "generated by controlled modular shifts for multi-qubit registers."
        )


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize(("direction", "mode"), _CASES)
def test_modular_arithmetic_cross_backend_sample_and_expval(
    sdk_transpiler: Any,
    n: int,
    seed: int,
    direction: str,
    mode: str,
) -> None:
    """Execute seeded primitive shifts and qmc.control-created shifts."""
    transpiler = sdk_transpiler
    backend_name = _backend_name(transpiler)
    _skip_unsupported_backend_case(backend_name, n, mode)

    rng = np.random.default_rng(seed)
    bits = _random_bits(rng, n)
    enabled = int(rng.integers(0, 2)) if mode == "controlled" else None
    coeffs = rng.uniform(-1.0, 1.0, size=n).tolist()
    _run_shift_case(
        transpiler,
        backend_name,
        n,
        bits,
        direction,
        mode,
        enabled,
        coeffs,
        shots=32,
    )


_BOUNDARY_CASES = [
    ("plus", "plain", None, "ones"),
    ("minus", "plain", None, "zeros"),
    ("plus", "controlled", 0, "ones"),
    ("plus", "controlled", 1, "ones"),
    ("minus", "controlled", 0, "zeros"),
    ("minus", "controlled", 1, "zeros"),
]


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize(("direction", "mode", "enabled", "pattern"), _BOUNDARY_CASES)
def test_modular_arithmetic_wraparound_and_control_boundaries(
    sdk_transpiler: Any,
    n: int,
    direction: str,
    mode: str,
    enabled: int | None,
    pattern: str,
) -> None:
    """Exercise wrap-around and both controlled enabled states explicitly."""
    transpiler = sdk_transpiler
    backend_name = _backend_name(transpiler)
    _skip_unsupported_backend_case(backend_name, n, mode)

    bits = [1] * n if pattern == "ones" else [0] * n
    coeffs = [((-1.0) ** index) * (index + 1) / (n + 1) for index in range(n)]
    _run_shift_case(
        transpiler,
        backend_name,
        n,
        bits,
        direction,
        mode,
        enabled,
        coeffs,
        shots=32,
    )
