"""Tests for modular arithmetic algorithm primitives."""

from __future__ import annotations

import math
from collections.abc import Callable
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


def _qiskit_transpiler() -> Any:
    """Return a Qiskit transpiler or skip when Qiskit is unavailable."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


def _quri_parts_transpiler() -> Any:
    """Return a QURI Parts transpiler or skip when QURI Parts is unavailable."""
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler() -> Any:
    """Return a CUDA-Q transpiler or skip when CUDA-Q is unavailable."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


_BACKENDS: list[Any] = [
    pytest.param(_qiskit_transpiler, id="qiskit"),
    pytest.param(_quri_parts_transpiler, marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param(_cudaq_transpiler, marks=pytest.mark.cudaq, id="cudaq"),
]
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
) -> float | None:
    """Return expval or ``None`` for known backend limitations."""
    try:
        return _expval(transpiler, kernel, bindings)
    except TypeError as exc:
        if (
            backend_name == "cudaq"
            and "cudaq.observe() is not supported for runtime control flow" in str(exc)
        ):
            return None
        raise


@pytest.mark.parametrize("transpiler_factory", _BACKENDS)
@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("seed", _SEEDS)
def test_modular_arithmetic_cross_backend_sample_and_expval(
    transpiler_factory: Callable[[], Any],
    n: int,
    seed: int,
) -> None:
    """Execute primitive shifts and qmc.control-created shifts where supported."""
    transpiler = transpiler_factory()
    backend_name = _backend_name(transpiler)
    if backend_name == "quri_parts" and n > 2:
        pytest.skip(
            "QURI Parts cannot emit the multi-controlled X gates required "
            "for modular registers larger than two qubits yet."
        )

    rng = np.random.default_rng(seed)
    shots = 32

    for direction, mode in _CASES:
        # QURI Parts cannot yet recursively emit the BinOp-bearing loop body
        # generated by controlled modular shifts for multi-qubit registers.
        if backend_name == "quri_parts" and mode == "controlled" and n > 1:
            continue

        bits = _random_bits(rng, n)
        enabled = int(rng.integers(0, 2)) if mode == "controlled" else None
        expected = _expected_bits(bits, direction, enabled)
        coeffs = rng.uniform(-1.0, 1.0, size=len(expected)).tolist()
        hamiltonian = _z_hamiltonian(coeffs)
        direction_id = 0 if direction == "plus" else 1
        context = (
            f"{transpiler.__class__.__name__} n={n} seed={seed} "
            f"direction={direction} mode={mode}"
        )

        if mode == "controlled":
            sample_bindings = {
                "n": n,
                "bits": bits,
                "direction": direction_id,
                "enabled": enabled,
            }
            expval_bindings = sample_bindings | {"hamiltonian": hamiltonian}
            sample_results = _sample(
                transpiler, _controlled_shift_sample_kernel, sample_bindings, shots
            )
            observed = _expval_if_supported(
                transpiler,
                _controlled_shift_expval_kernel,
                expval_bindings,
                backend_name,
            )
        else:
            sample_bindings = {"n": n, "bits": bits, "direction": direction_id}
            expval_bindings = sample_bindings | {"hamiltonian": hamiltonian}
            sample_results = _sample(
                transpiler, _shift_sample_kernel, sample_bindings, shots
            )
            observed = _expval_if_supported(
                transpiler,
                _shift_expval_kernel,
                expval_bindings,
                backend_name,
            )

        _assert_deterministic(sample_results, expected, shots, context=context)
        if observed is None:
            continue

        assert np.isclose(observed, _exact_z_expval(coeffs, expected), atol=1e-8), (
            f"{context}: expval={observed}, expected="
            f"{_exact_z_expval(coeffs, expected)}"
        )
