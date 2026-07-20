"""Tests for executable body-derived standard-library arithmetic."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o


@qmc.qkernel
def _prepare_basis(
    register: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
    size: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a little-endian computational-basis register.

    Args:
        register (qmc.Vector[qmc.Qubit]): Register initialized to zero.
        bits (qmc.Vector[qmc.UInt]): Little-endian classical bit values.
        size (qmc.UInt): Number of elements to prepare.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared register.
    """
    for index in qmc.range(size):
        angle = cast(qmc.Float, math.pi * bits[index])
        register[index] = qmc.rx(register[index], angle)
    return register


@qmc.qkernel
def _ripple_sample(
    size: qmc.UInt,
    left_bits: qmc.Vector[qmc.UInt],
    right_bits: qmc.Vector[qmc.UInt],
) -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
]:
    """Prepare, add, and measure a ripple-carry instance."""
    left = _prepare_basis(qmc.qubit_array(size, "left"), left_bits, size)
    right = _prepare_basis(qmc.qubit_array(size, "right"), right_bits, size)
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    left, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return (
        qmc.measure(left),
        qmc.measure(right),
        qmc.measure(carry),
        qmc.measure(overflow),
    )


@qmc.qkernel
def _ripple_expval(
    size: qmc.UInt,
    left_bits: qmc.Vector[qmc.UInt],
    right_bits: qmc.Vector[qmc.UInt],
    observable: qmc.Observable,
) -> qmc.Float:
    """Prepare and add a ripple-carry instance for expectation evaluation."""
    left = _prepare_basis(qmc.qubit_array(size, "left"), left_bits, size)
    right = _prepare_basis(qmc.qubit_array(size, "right"), right_bits, size)
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    _, right, _, _ = qmc.ripple_carry_add(left, right, carry, overflow)
    return qmc.expval(right, observable)


@qmc.qkernel
def _modular_add_sample(
    size: qmc.UInt,
    addend_bits: qmc.Vector[qmc.UInt],
    modulus_bits: qmc.Vector[qmc.UInt],
    target_bits: qmc.Vector[qmc.UInt],
    enabled: qmc.UInt,
) -> tuple[
    qmc.Bit,
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
]:
    """Prepare, apply, and measure a controlled modular addition."""
    addend = _prepare_basis(qmc.qubit_array(size, "addend"), addend_bits, size)
    modulus = _prepare_basis(qmc.qubit_array(size, "modulus"), modulus_bits, size)
    target = _prepare_basis(qmc.qubit_array(size, "target"), target_bits, size)
    control = qmc.qubit("control")
    if enabled == qmc.uint(1):
        control = qmc.x(control)
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    flag = qmc.qubit("flag")
    control, addend, modulus, target, carry, overflow, flag = (
        qmc.controlled_modular_add(
            control, addend, modulus, target, carry, overflow, flag
        )
    )
    return (
        qmc.measure(control),
        qmc.measure(addend),
        qmc.measure(modulus),
        qmc.measure(target),
        qmc.measure(carry),
        qmc.measure(overflow),
        qmc.measure(flag),
    )


@qmc.qkernel
def _modular_add_expval(
    size: qmc.UInt,
    addend_bits: qmc.Vector[qmc.UInt],
    modulus_bits: qmc.Vector[qmc.UInt],
    target_bits: qmc.Vector[qmc.UInt],
    enabled: qmc.UInt,
    observable: qmc.Observable,
) -> qmc.Float:
    """Apply controlled modular addition for expectation evaluation."""
    addend = _prepare_basis(qmc.qubit_array(size, "addend"), addend_bits, size)
    modulus = _prepare_basis(qmc.qubit_array(size, "modulus"), modulus_bits, size)
    target = _prepare_basis(qmc.qubit_array(size, "target"), target_bits, size)
    control = qmc.qubit("control")
    if enabled == qmc.uint(1):
        control = qmc.x(control)
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    flag = qmc.qubit("flag")
    _, _, _, target, _, _, _ = qmc.controlled_modular_add(
        control, addend, modulus, target, carry, overflow, flag
    )
    return qmc.expval(target, observable)


@qmc.qkernel
def _symbolic_add(
    size: qmc.UInt,
) -> tuple[
    qmc.Vector[qmc.Qubit],
    qmc.Vector[qmc.Qubit],
    qmc.Qubit,
    qmc.Qubit,
]:
    """Apply ripple-carry addition to symbolic-width registers."""
    left = qmc.qubit_array(size, "left")
    right = qmc.qubit_array(size, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    return qmc.ripple_carry_add(left, right, carry, overflow)


@qmc.qkernel
def _symbolic_modular_add(
    size: qmc.UInt,
) -> tuple[
    qmc.Vector[qmc.Qubit],
    qmc.Vector[qmc.Qubit],
    qmc.Vector[qmc.Qubit],
    qmc.Qubit,
    qmc.Qubit,
    qmc.Qubit,
]:
    """Apply modular addition to symbolic-width registers."""
    addend = qmc.qubit_array(size, "addend")
    modulus = qmc.qubit_array(size, "modulus")
    target = qmc.qubit_array(size, "target")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    flag = qmc.qubit("flag")
    return qmc.modular_add(addend, modulus, target, carry, overflow, flag)


def _bits(value: int, size: int) -> list[int]:
    """Encode an integer as little-endian bits.

    Args:
        value (int): Non-negative integer to encode.
        size (int): Number of output bits.

    Returns:
        list[int]: Little-endian bit values.
    """
    return [(value >> index) & 1 for index in range(size)]


def _basis_value(bits: tuple[int, ...]) -> int:
    """Decode little-endian measured bits.

    Args:
        bits (tuple[int, ...]): Measured bit values.

    Returns:
        int: Unsigned integer value.
    """
    return sum(bit << index for index, bit in enumerate(bits))


def _execute_sample(
    sdk_transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any],
) -> tuple[Any, ...]:
    """Transpile and return the deterministic most-common sample.

    Args:
        sdk_transpiler (Any): Supported backend fixture case.
        kernel (Any): QKernel to execute.
        bindings (dict[str, Any]): Compile-time kernel inputs.

    Returns:
        tuple[Any, ...]: Most-common measured output tuple.
    """
    transpiler = sdk_transpiler.transpiler
    result = (
        transpiler.transpile(kernel, bindings=bindings)
        .sample(transpiler.executor(), shots=16)
        .result()
    )
    assert len(result.results) == 1
    measured, count = result.results[0]
    assert count == 16
    return measured


def _execute_expval(
    sdk_transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any],
) -> float:
    """Transpile and evaluate one expectation-value kernel.

    Args:
        sdk_transpiler (Any): Supported backend fixture case.
        kernel (Any): QKernel to execute.
        bindings (dict[str, Any]): Compile-time kernel inputs.

    Returns:
        float: Backend expectation value.
    """
    transpiler = sdk_transpiler.transpiler
    return float(
        transpiler.transpile(kernel, bindings=bindings)
        .run(transpiler.executor())
        .result()
    )


def test_ripple_carry_resources_are_derived_symbolically() -> None:
    """The executable adder body yields its exact symbolic gate polynomial."""
    estimate = _symbolic_add.estimate_resources()
    size = estimate.parameters["size"]
    positive_size = sp.Symbol("positive_size", integer=True, positive=True)
    total = estimate.gates.total.subs(size, positive_size)
    toffoli = estimate.gates.toffoli.subs(size, positive_size)
    two_qubit = estimate.gates.two_qubit.subs(size, positive_size)

    assert sp.simplify(total - (6 * positive_size + 1)) == 0
    assert sp.simplify(toffoli - 2 * positive_size) == 0
    assert sp.simplify(two_qubit - (4 * positive_size + 1)) == 0
    assert estimate.qubits == 2 * size + 2
    assert _symbolic_add.estimate_resources(inputs={"size": 2048}).gates.total == (
        6 * 2048 + 1
    )


def test_modular_add_resources_are_derived_symbolically() -> None:
    """Modular-add resources remain a body-derived linear expression."""
    estimate = _symbolic_modular_add.estimate_resources()
    size = estimate.parameters["size"]
    positive_size = sp.Symbol("positive_size", integer=True, positive=True)

    assert (
        sp.Poly(estimate.gates.total.subs(size, positive_size), positive_size).degree()
        == 1
    )
    assert (
        sp.Poly(
            estimate.gates.toffoli.subs(size, positive_size), positive_size
        ).degree()
        == 1
    )
    assert estimate.qubits == 3 * size + 3
    concrete = _symbolic_modular_add.estimate_resources(inputs={"size": 2048})
    assert concrete.gates.total == estimate.gates.total.subs(size, 2048)


@pytest.mark.parametrize("size,seed", [(1, 0), (2, 1), (3, 2), (5, 42)])
def test_ripple_carry_cross_backend(
    sdk_transpiler: Any,
    size: int,
    seed: int,
) -> None:
    """Representative random additions pass sampling and expval paths."""
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, 1 << size))
    right = int(rng.integers(0, 1 << size))
    output = left + right
    bindings = {
        "size": size,
        "left_bits": _bits(left, size),
        "right_bits": _bits(right, size),
    }

    left_bits, right_bits, carry, overflow = _execute_sample(
        sdk_transpiler, _ripple_sample, bindings
    )
    assert _basis_value(left_bits) == left
    assert _basis_value(right_bits) == output % (1 << size)
    assert carry == 0
    assert overflow == output >> size

    qubit = seed % size
    actual = _execute_expval(
        sdk_transpiler,
        _ripple_expval,
        {**bindings, "observable": qm_o.Z(qubit)},
    )
    expected = -1.0 if output >> qubit & 1 else 1.0
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(actual, expected, atol=tolerance)


@pytest.mark.parametrize("size,seed", [(2, 0), (3, 2), (5, 42)])
@pytest.mark.parametrize("enabled", [0, 1])
def test_controlled_modular_add_cross_backend(
    sdk_transpiler: Any,
    size: int,
    seed: int,
    enabled: int,
) -> None:
    """Representative controlled modular additions pass both backend paths."""
    rng = np.random.default_rng(seed)
    modulus = (1 << size) - 1
    addend = int(rng.integers(0, modulus))
    target = int(rng.integers(0, modulus))
    output = (target + addend) % modulus if enabled else target
    bindings = {
        "size": size,
        "addend_bits": _bits(addend, size),
        "modulus_bits": _bits(modulus, size),
        "target_bits": _bits(target, size),
        "enabled": enabled,
    }

    control, addend_bits, modulus_bits, target_bits, carry, overflow, flag = (
        _execute_sample(sdk_transpiler, _modular_add_sample, bindings)
    )
    assert control == enabled
    assert _basis_value(addend_bits) == addend
    assert _basis_value(modulus_bits) == modulus
    assert _basis_value(target_bits) == output
    assert (carry, overflow, flag) == (0, 0, 0)

    qubit = seed % size
    actual = _execute_expval(
        sdk_transpiler,
        _modular_add_expval,
        {**bindings, "observable": qm_o.Z(qubit)},
    )
    expected = -1.0 if output >> qubit & 1 else 1.0
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(actual, expected, atol=tolerance)
