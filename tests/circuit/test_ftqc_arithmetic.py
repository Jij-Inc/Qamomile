"""FTQC-only tests for measurement-assisted arithmetic primitives."""

from __future__ import annotations

import math

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import (
    _controlled_modular_add_const_modulus_dirty,
    _dirty_const_add_extended,
    _xor_constant,
)


@qmc.qkernel
def _dirty_add_sample(
    initial: qmc.UInt,
    value: qmc.UInt,
) -> tuple[
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
]:
    """Prepare and sample a three-bit carry-venting addition."""
    target = qmc.qubit_array(2, name="target")
    overflow = qmc.qubit("overflow")
    dirty = qmc.qubit_array(1, name="dirty")
    first = qmc.qubit("first")
    second = qmc.qubit("second")
    target = _xor_constant(target, initial)
    _, target, overflow, dirty, first, second = _dirty_const_add_extended(
        target,
        overflow,
        dirty,
        first,
        second,
        value,
        None,
    )
    return (
        qmc.measure(target),
        qmc.measure(overflow),
        qmc.measure(dirty),
        qmc.measure(first),
        qmc.measure(second),
    )


@qmc.qkernel
def _controlled_dirty_add_sample(
    initial: qmc.UInt,
    value: qmc.UInt,
    enabled: qmc.UInt,
) -> tuple[
    qmc.Bit,
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
]:
    """Prepare and sample a controlled carry-venting addition."""
    control = qmc.qubit("control")
    target = qmc.qubit_array(2, name="target")
    overflow = qmc.qubit("overflow")
    dirty = qmc.qubit_array(1, name="dirty")
    first = qmc.qubit("first")
    second = qmc.qubit("second")
    control = qmc.rx(control, math.pi * enabled)
    target = _xor_constant(target, initial)
    control_out, target, overflow, dirty, first, second = _dirty_const_add_extended(
        target,
        overflow,
        dirty,
        first,
        second,
        value,
        control,
    )
    assert control_out is not None
    return (
        qmc.measure(control_out),
        qmc.measure(target),
        qmc.measure(overflow),
        qmc.measure(dirty),
        qmc.measure(first),
        qmc.measure(second),
    )


@qmc.qkernel
def _modular_add_sample(
    addend_value: qmc.UInt,
    target_value: qmc.UInt,
    enabled: qmc.UInt,
) -> tuple[
    qmc.Bit,
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Vector[qmc.Bit],
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
]:
    """Prepare and sample the dirty-workspace modular-add primitive."""
    control = qmc.qubit("control")
    addend = qmc.qubit_array(2, name="addend")
    target = qmc.qubit_array(2, name="target")
    dirty = qmc.qubit_array(2, name="dirty")
    carry = qmc.qubit("carry")
    vent = qmc.qubit("vent")
    overflow = qmc.qubit("overflow")
    flag = qmc.qubit("flag")
    control = qmc.rx(control, math.pi * enabled)
    addend = _xor_constant(addend, addend_value)
    target = _xor_constant(target, target_value)
    control, addend, target, dirty, carry, vent, overflow, flag = (
        _controlled_modular_add_const_modulus_dirty(
            control,
            addend,
            target,
            dirty,
            carry,
            vent,
            overflow,
            flag,
            qmc.uint(3),
            2,
        )
    )
    return (
        qmc.measure(control),
        qmc.measure(addend),
        qmc.measure(target),
        qmc.measure(dirty),
        qmc.measure(carry),
        qmc.measure(vent),
        qmc.measure(overflow),
        qmc.measure(flag),
    )


@pytest.mark.parametrize("initial", range(4))
@pytest.mark.parametrize("value", range(8))
def test_dirty_const_add_extended_basis_states(
    qiskit_transpiler,
    initial: int,
    value: int,
) -> None:
    """Carry venting adds every three-bit constant and restores workspace."""
    result = (
        qiskit_transpiler.transpile(
            _dirty_add_sample,
            bindings={"initial": initial, "value": value},
        )
        .sample(qiskit_transpiler.executor(), shots=32)
        .result()
    )
    measured, count = result.most_common(1)[0]
    target_bits, overflow, dirty_bits, first, second = measured
    actual = sum(bit << index for index, bit in enumerate(target_bits))
    actual += overflow << 2
    assert count == 32
    assert actual == (initial + value) % 8
    assert dirty_bits == (0,)
    assert (first, second) == (0, 0)


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("initial", range(4))
@pytest.mark.parametrize("value", range(8))
def test_controlled_dirty_const_add_extended_basis_states(
    qiskit_transpiler,
    enabled: int,
    initial: int,
    value: int,
) -> None:
    """Controlled carry venting preserves both branches and its workspace."""
    result = (
        qiskit_transpiler.transpile(
            _controlled_dirty_add_sample,
            bindings={"initial": initial, "value": value, "enabled": enabled},
        )
        .sample(qiskit_transpiler.executor(), shots=32)
        .result()
    )
    measured, count = result.most_common(1)[0]
    control, target_bits, overflow, dirty_bits, first, second = measured
    actual = sum(bit << index for index, bit in enumerate(target_bits))
    actual += overflow << 2
    expected = (initial + enabled * value) % 8
    assert count == 32
    assert control == enabled
    assert actual == expected
    assert dirty_bits == (0,)
    assert (first, second) == (0, 0)


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("addend", range(3))
@pytest.mark.parametrize("target", range(3))
def test_dirty_modular_add_basis_states(
    qiskit_transpiler,
    enabled: int,
    addend: int,
    target: int,
) -> None:
    """Linear modular addition is correct and restores every workspace."""
    result = (
        qiskit_transpiler.transpile(
            _modular_add_sample,
            bindings={
                "addend_value": addend,
                "target_value": target,
                "enabled": enabled,
            },
        )
        .sample(qiskit_transpiler.executor(), shots=32)
        .result()
    )
    measured, count = result.most_common(1)[0]
    (
        control,
        addend_bits,
        target_bits,
        dirty_bits,
        carry,
        vent,
        overflow,
        flag,
    ) = measured
    actual_addend = sum(bit << index for index, bit in enumerate(addend_bits))
    actual_target = sum(bit << index for index, bit in enumerate(target_bits))
    assert count == 32
    assert control == enabled
    assert actual_addend == addend
    assert actual_target == (target + enabled * addend) % 3
    assert dirty_bits == (0, 0)
    assert (carry, vent, overflow, flag) == (0, 0, 0, 0)


def test_carry_venting_constant_adder_has_linear_body_growth() -> None:
    """Direct body estimates keep constant-add gate cost linear in width."""
    normalized = []
    for size in (2, 3, 4, 5):

        @qmc.qkernel
        def add() -> qmc.Vector[qmc.Qubit]:
            """Build one specialized carry-venting constant addition."""
            target = qmc.qubit_array(size, name="target")
            overflow = qmc.qubit("overflow")
            dirty = qmc.qubit_array(size - 1, name="dirty")
            first = qmc.qubit("first")
            second = qmc.qubit("second")
            _, target, _, _, _, _ = _dirty_const_add_extended(
                target,
                overflow,
                dirty,
                first,
                second,
                (1 << (size + 1)) - 1,
                None,
            )
            return target

        estimate = add.estimate_resources()
        assert estimate.qubits == 2 * size + 2
        normalized.append(float(estimate.gates.total) / size)

    assert max(normalized) / min(normalized) < 1.5
