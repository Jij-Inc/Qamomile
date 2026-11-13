"""Regression tests for several owners carried through structured control flow.

A call consuming more than one owner, such as ``a, b = helper(a, b)`` or
``controls, target = qm.control(f, controls, target)``, returns successors that
are projections of the call's ownership region. Loops and branches must yield
their carries with the topology they entered with, so lowering regroups such
carries into sibling projections of one owner before the structured statement.
"""

from __future__ import annotations

import pytest

import qamomile as qm
import qamomile.resources as qr

HELPERS = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def mark(
    register: qm.Q[qm.Bits[n]], flag: qm.Q[qm.Bits[1]], index: qm.Int
) -> tuple[qm.Q[qm.Bits[n]], qm.Q[qm.Bits[1]]]:
    register = qm.x(register, target=index)
    return register, qm.x(flag)
"""


def _module(body: str) -> qm.Module:
    return qm.build_source("from __future__ import annotations\n" + HELPERS + body)


def test_tuple_returning_call_is_a_loop_carry() -> None:
    """Both successors of a two-owner helper call are carried by a for loop."""

    module = _module(
        """
@qm.qkernel
def marked(width: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    register = qm.qubits(width)
    flag = qm.qubits(1)
    for index in range(width):
        register, flag = mark(register, flag, index)
    return qm.measure(register), qm.measure(flag)
"""
    )
    program = module.specialize("marked", width=3)
    result = qm.run(program, on=qm.Reference(), shots=4, seed=3).result
    assert result.values == {((1, 1, 1), (1,)): 4}


def test_coherent_control_with_selected_lane_inside_a_loop() -> None:
    """qm.control over separately allocated owners is legal in a loop body."""

    module = _module(
        """
@qm.qkernel
def parity(width: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    controls = qm.x(qm.qubits(width))
    target = qm.qubits(1)
    for lane in range(width):
        controls, target = qm.control(flip, controls, target, at=lane)
    return qm.measure(controls), qm.measure(target)
"""
    )
    for width, flag in ((2, 0), (3, 1)):
        program = module.specialize("parity", width=width)
        result = qm.run(program, on=qm.Reference(), shots=4, seed=5).result
        assert result.values == {((1,) * width, (flag,)): 4}
    circuit = qm.qiskit.to_circuit(module["parity"], width=3)
    assert circuit.num_qubits == 4


@pytest.mark.parametrize(("selector", "expected"), [(2, (1, 1)), (1, (1, 0))])
def test_coherent_control_inside_a_const_branch(
    selector: int, expected: tuple[int, int]
) -> None:
    """A Const branch may update two independent owners through qm.control."""

    module = _module(
        """
@qm.qkernel
def guarded(selector: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    control = qm.x(qm.qubits(1))
    target = qm.qubits(1)
    if selector != 1:
        control, target = qm.control(flip, control, target)
    return qm.measure(qm.concat(control, target))
"""
    )
    program = module.specialize("guarded", selector=selector)
    result = qm.run(program, on=qm.Reference(), shots=4, seed=7).result
    assert result.counts == {expected: 4}


def test_multi_owner_carries_in_nested_and_while_loops() -> None:
    """Regrouping composes with nested loops and while carries."""

    module = _module(
        """
@qm.qkernel
def nested(width: qm.Const[qm.Int], rounds: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    register = qm.qubits(width)
    flag = qm.qubits(1)
    for _round in range(rounds):
        for index in range(width):
            register, flag = mark(register, flag, index)
    return qm.measure(register), qm.measure(flag)

@qm.qkernel
def repeated(width: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    register = qm.qubits(width)
    flag = qm.qubits(1)
    index = 0
    while index < width:
        register, flag = mark(register, flag, index)
        index += 1
    return qm.measure(register), qm.measure(flag)
"""
    )
    nested = module.specialize("nested", width=2, rounds=3)
    assert qm.run(nested, on=qm.Reference(), shots=2, seed=1).result.values == {
        ((1, 1), (0,)): 2
    }
    repeated = module.specialize("repeated", width=3)
    assert qm.run(repeated, on=qm.Reference(), shots=2, seed=1).result.values == {
        ((1, 1, 1), (1,)): 2
    }


def test_resource_study_counts_regrouped_carries_symbolically() -> None:
    """Regrouping adds no gates: the family estimate stays the source count."""

    module = _module(
        """
@qm.qkernel
def marked(width: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    register = qm.qubits(width)
    flag = qm.qubits(1)
    for index in range(width):
        register, flag = mark(register, flag, index)
    return qm.measure(register), qm.measure(flag)
"""
    )
    gates = qr.logical(module["marked"]).metric(qr.LOGICAL_GATES)
    assert gates.evaluate({"width": 5}).numeric_enclosure() == (10, 10)


@pytest.mark.parametrize("schoolbook", [False, True])
def test_controlled_modular_multiplication_projects_to_a_circuit(
    schoolbook: bool,
) -> None:
    """A helper whose arithmetic is a checked deferred definition is controllable."""

    from qamomile.qiskit import Aer

    module = qm.build_source(
        """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def times_five(state: qm.Q[qm.Bits[5]]) -> qm.Q[qm.Bits[5]]:
    return qm.as_bits(qm.mod_mul(qm.as_uint(state), factor=5, modulus=21))

@qm.qkernel
def controlled_product(
    enabled: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[5]]]:
    control = qm.qubits(1)
    if enabled == 1:
        control = qm.x(control)
    target = qm.x(qm.qubits(5), target=1)
    control, target = qm.control(times_five, control, target)
    return qm.measure(control), qm.measure(target)
"""
    )
    options = (
        qm.RealizationOptions({qm.mod_mul: "mod_mul.schoolbook_ripple"})
        if schoolbook
        else qm.RealizationOptions()
    )
    endpoint = Aer(realizations=options)
    # Little-endian lanes: 2 = (0, 1, 0, 0, 0) and 2 * 5 mod 21 = 10 = (0, 1, 0, 1, 0).
    for enabled, product in ((0, (0, 1, 0, 0, 0)), (1, (0, 1, 0, 1, 0))):
        program = module.specialize("controlled_product", enabled=enabled)
        artifact = qm.compile(program, to=endpoint.target)
        result = qm.execute(artifact, on=endpoint, shots=8, seed=11)
        assert result.values == {((enabled,), product): 8}
