"""Structured regions keep owners consistent when one statement rebinds several.

A multi-owner operation such as ``qm.control`` or a call returning a tuple
consumes several carried quantum owners and returns their successors under
one product owner. Loops and branches must still yield those successors with
the ownership topology of their carried values. These programs once failed
frontend lowering with ``Q1010: yield/result mismatch``.
"""

from __future__ import annotations

import itertools

import pytest

import qamomile as qm


@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)


@qm.qkernel
def controlled_loop(
    precision: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[precision]]:  # noqa: F821
    counting = qm.qubits(precision)
    target = qm.qubits(1)
    for k in range(precision):
        counting = qm.h(counting, target=k)
        counting, target = qm.control(flip, counting, target, at=k)
    ignored = qm.measure(target)  # noqa: F841 -- discards the target sample.
    return qm.measure(counting)


PRELUDE = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def times_two(state: qm.Q[qm.UInt[3]]) -> qm.Q[qm.UInt[3]]:
    return qm.mod_mul(state, factor=2, modulus=5)

@qm.qkernel
def bump(
    a: qm.Q[qm.Bits[1]], b: qm.Q[qm.Bits[2]]
) -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[2]]]:
    b = qm.x(b, target=1)
    return qm.x(a), b
"""

KERNELS = {
    "control_for": """
@qm.qkernel
def control_for(
    precision: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[precision]], qm.Shot[qm.Bits[1]]]:
    counting = qm.qubits(precision)
    target = qm.qubits(1)
    for k in range(precision):
        counting = qm.x(counting, target=k)
        counting, target = qm.control(flip, counting, target, at=k)
    return qm.measure(counting), qm.measure(target)
""",
    "control_if": """
@qm.qkernel
def control_if(
    flag: qm.Const[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[1]]]:
    counting = qm.qubits(2)
    target = qm.qubits(1)
    counting = qm.x(counting, target=1)
    if flag:
        counting, target = qm.control(flip, counting, target, at=1)
    return qm.measure(counting), qm.measure(target)
""",
    "control_if_in_for": """
@qm.qkernel
def control_if_in_for(
    precision: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[precision]], qm.Shot[qm.Bits[1]]]:
    counting = qm.qubits(precision)
    target = qm.qubits(1)
    for k in range(precision):
        counting = qm.x(counting, target=k)
        if k % 2 == 0:
            counting, target = qm.control(flip, counting, target, at=k)
    return qm.measure(counting), qm.measure(target)
""",
    "control_while": """
@qm.qkernel
def control_while(
    limit: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[1]]]:
    counting = qm.qubits(2)
    target = qm.qubits(1)
    counting = qm.x(counting, target=0)
    step = 0
    while step < limit:
        counting, target = qm.control(flip, counting, target, at=0)
        step += 1
    return qm.measure(counting), qm.measure(target)
""",
    "controlled_mod_mul": """
@qm.qkernel
def controlled_mod_mul(
    precision: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[precision]], qm.Shot[qm.Bits[3]]]:
    counting = qm.qubits(precision)
    target = qm.as_uint(qm.prepare((1, 0, 0)))
    for k in range(precision):
        counting = qm.x(counting, target=k)
        counting, target = qm.control(times_two, counting, target, at=k)
    return qm.measure(counting), qm.measure(qm.as_bits(target))
""",
    "reordered_call": """
@qm.qkernel
def reordered_call(
    rounds: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[1]]]:
    a = qm.qubits(2)
    b = qm.qubits(1)
    for k in range(rounds):
        b, a = bump(b, a)
    return qm.measure(a), qm.measure(b)
""",
    "reallocated_owner": """
@qm.qkernel
def reallocated_owner(
    rounds: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[1]]]:
    a = qm.qubits(2)
    b = qm.qubits(1)
    for k in range(rounds):
        a = qm.x(a, target=0)
        dropped = qm.measure(b)
        b = qm.qubits(1)
        b = qm.x(b)
    return qm.measure(a), qm.measure(b)
""",
}


def _values(function: str, **constants: object) -> dict:
    """Build one kernel alone, round-trip its artifacts, and sample it."""

    module = qm.build_source(PRELUDE + KERNELS[function])
    module = qm.Module.from_bytes(module.to_bytes())
    program = qm.Program.from_bytes(module.specialize(function, **constants).to_bytes())
    return qm.run(program, on=qm.Reference(), shots=8, seed=3).result.values


@pytest.mark.parametrize("precision", [1, 2, 3])
def test_reported_control_loop_lowers_and_executes(precision: int) -> None:
    """The reported kernel lowers and samples only valid counting registers."""

    controlled_loop.module()
    program = qm.specialize(controlled_loop, precision=precision)
    values = qm.run(program, on=qm.Reference(), shots=64, seed=5).result.values

    assert sum(values.values()) == 64
    assert set(values) <= set(itertools.product((0, 1), repeat=precision))


@pytest.mark.parametrize("precision", [1, 2, 3])
def test_control_in_const_for_loop(precision: int) -> None:
    """Every iteration's controlled flip reaches the shared target."""

    assert _values("control_for", precision=precision) == {
        ((1,) * precision, (precision % 2,)): 8
    }


@pytest.mark.parametrize(("flag", "target"), [(True, 1), (False, 0)])
def test_control_in_const_if(flag: bool, target: int) -> None:
    """Both branches yield the counting and target owners consistently."""

    assert _values("control_if", flag=flag) == {((0, 1), (target,)): 8}


def test_control_in_const_if_inside_for() -> None:
    """A rebinding branch nested in a loop body agrees with the loop carries."""

    assert _values("control_if_in_for", precision=3) == {((1, 1, 1), (0,)): 8}


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_control_in_while_loop(limit: int) -> None:
    """While bodies rebind the same owners with the same topology."""

    assert _values("control_while", limit=limit) == {((1, 0), (limit % 2,)): 8}


@pytest.mark.parametrize(
    ("precision", "target"),
    [(1, (0, 1, 0)), (2, (0, 0, 1)), (3, (1, 1, 0))],
)
def test_controlled_mod_mul_in_const_for_loop(
    precision: int, target: tuple[int, ...]
) -> None:
    """Each active control multiplies the little-endian target by two mod five."""

    assert _values("controlled_mod_mul", precision=precision) == {
        ((1,) * precision, target): 8
    }


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [(1, ((0, 1), (1,))), (2, ((0, 0), (0,)))],
)
def test_multi_result_call_rebinding_in_reversed_order(
    rounds: int, expected: tuple
) -> None:
    """Owners passed out of carry order still return to their carried roles."""

    assert _values("reordered_call", rounds=rounds) == {expected: 8}


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [(1, ((1, 0), (1,))), (2, ((0, 0), (1,)))],
)
def test_single_reallocated_owner_is_grouped_with_another_carry(
    rounds: int, expected: tuple
) -> None:
    """A single fresh owner is re-rooted together with its sibling carry."""

    assert _values("reallocated_owner", rounds=rounds) == {expected: 8}
