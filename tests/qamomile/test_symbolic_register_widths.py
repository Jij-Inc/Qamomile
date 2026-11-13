"""Registers whose widths are polynomials over Const Int parameters.

``qm.split_at`` accepts a boundary written over Const Int parameters, and
``qm.concat`` and ``qm.qubits`` produce the canonical polynomial width. Widths
are compared as polynomials, so ``k + (n - k)`` and ``n`` are the same width,
and specialization rejects bindings that make any width nonpositive.
"""

from __future__ import annotations

import pytest

import qamomile as qm
import qamomile.resources as qr


def _module(body: str) -> qm.Module:
    return qm.build_source(
        "from __future__ import annotations\n\nimport qamomile as qm\n" + body
    )


def _values(program: qm.Program) -> dict[object, int]:
    return dict(qm.run(program, on=qm.Reference(), shots=4, seed=1).result.values)


def test_concat_of_symbolic_and_fixed_widths_matches_the_annotation() -> None:
    """``Bits[width]`` and ``Bits[1]`` concatenate to ``Bits[width + 1]``."""

    module = _module(
        """
@qm.qkernel
def flagged(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width + 1]]:
    register = qm.x(qm.qubits(width), target=0)
    flag = qm.x(qm.qubits(1))
    return qm.measure(qm.concat(register, flag))
"""
    )
    assert _values(module.specialize("flagged", width=3)) == {(1, 0, 0, 1): 4}


@pytest.mark.parametrize(("n", "boundary"), [(5, 2), (3, 1), (4, 3)])
def test_split_at_a_const_parameter_boundary(n: int, boundary: int) -> None:
    """The suffix ``Bits[n-k]`` is retained symbolically until specialization."""

    module = _module(
        """
@qm.qkernel
def suffix_flip(n: qm.Const[qm.Int], k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    prefix, rest = qm.split_at(qm.qubits(n), k)
    rest = qm.x(rest)
    return qm.measure(qm.concat(prefix, rest))
"""
    )
    program = module.specialize("suffix_flip", n=n, k=boundary)
    assert _values(program) == {(0,) * boundary + (1,) * (n - boundary): 4}
    assert "Bits[-k+n]" in module.inspect()


def test_split_at_accepts_affine_boundaries_and_helpers() -> None:
    """Boundaries may combine Const parameters, literals, and helper binders."""

    module = _module(
        """
@qm.qkernel
def flip_tail(state: qm.Q[qm.Bits[n]], size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:
    head, tail = qm.split_at(state, size - 1)
    return qm.concat(head, qm.x(tail))

@qm.qkernel
def last(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2 * width]]:
    state = qm.qubits(2 * width)
    return qm.measure(flip_tail(state, 2 * width))
"""
    )
    assert _values(module.specialize("last", width=2)) == {(0, 0, 0, 1): 4}


def test_qubits_of_a_polynomial_size_has_that_exact_width() -> None:
    """``qm.qubits(2 * n + 1)`` is ``Bits[2*n+1]`` instead of an unknown width."""

    module = _module(
        """
@qm.qkernel
def odd(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2 * n + 1]]:
    return qm.measure(qm.x(qm.qubits(2 * n + 1), target=0))
"""
    )
    assert "Bits[2*n+1]" in module.inspect()
    assert _values(module.specialize("odd", n=2)) == {(1, 0, 0, 0, 0): 4}
    peak = qr.logical(module["odd"]).metric(qr.LOGICAL_PEAK_QUBITS)
    assert peak.evaluate({"n": 10}).numeric_enclosure() == (21, 21)


def test_split_and_concat_round_trip_is_a_loop_carry() -> None:
    """``concat(split_at(x, k))`` keeps the carried type ``Bits[n]``."""

    module = _module(
        """
@qm.qkernel
def toggled(
    n: qm.Const[qm.Int], k: qm.Const[qm.Int], rounds: qm.Const[qm.Int]
) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    for _round in range(rounds):
        prefix, rest = qm.split_at(state, k)
        rest = qm.x(rest, target=0)
        state = qm.concat(prefix, rest)
    return qm.measure(state)
"""
    )
    program = module.specialize("toggled", n=4, k=1, rounds=3)
    assert _values(program) == {(0, 1, 0, 0): 4}
    restored = qm.Program.from_bytes(program.to_bytes())
    assert _values(restored) == {(0, 1, 0, 0): 4}


def test_symbolic_split_costs_are_family_expressions() -> None:
    """Resource analysis prices the suffix by its polynomial width."""

    module = _module(
        """
@qm.qkernel
def suffix_h(n: qm.Const[qm.Int], k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    prefix, rest = qm.split_at(qm.qubits(n), k)
    return qm.measure(qm.concat(prefix, qm.h(rest)))
"""
    )
    gates = qr.logical(module["suffix_h"]).metric(qr.LOGICAL_GATES)
    assert gates.evaluate({"n": 9, "k": 4}).numeric_enclosure() == (5, 5)
    circuit = qm.qiskit.to_circuit(module["suffix_h"], n=5, k=2)
    assert circuit.count_ops()["h"] == 3


def test_nonpositive_symbolic_width_is_rejected_at_specialization() -> None:
    """A boundary equal to the width leaves an empty suffix, which is invalid."""

    module = _module(
        """
@qm.qkernel
def whole(n: qm.Const[qm.Int], k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    prefix, rest = qm.split_at(qm.qubits(n), k)
    return qm.measure(qm.concat(prefix, rest))
"""
    )
    with pytest.raises(qm.SpecializationError, match="Q3007"):
        module.specialize("whole", n=3, k=3)


@pytest.mark.parametrize(
    "boundary",
    ["index", "offset", "0", "n"],
)
def test_boundaries_outside_the_dimension_language_fail_in_the_frontend(
    boundary: str,
) -> None:
    """Loop indices, rebound names, and empty parts are rejected with a reason."""

    source = f"""
@qm.qkernel
def invalid(n: qm.Const[qm.Int], offset: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    offset = offset + 1
    state = qm.qubits(n)
    for index in range(1, n):
        prefix, rest = qm.split_at(state, {boundary})
        state = qm.concat(prefix, rest)
    return qm.measure(state)
"""
    expected = "Q5097" if boundary in {"0", "n"} else "Q5096"
    with pytest.raises(qm.FrontendError, match=expected):
        _module(source)


def test_register_rotation_in_a_loop_goes_through_a_helper() -> None:
    """A reordered owner is a new owner: rotate in a helper, not in the loop body."""

    module = _module(
        """
@qm.qkernel
def rotate_once(state: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]:
    head, tail = qm.split_at(state, 1)
    return qm.concat(tail, head)

@qm.qkernel
def rotate(n: qm.Const[qm.Int], rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.x(qm.qubits(n), target=0)
    for _round in range(rounds):
        state = rotate_once(state)
    return qm.measure(state)
"""
    )
    assert _values(module.specialize("rotate", n=3, rounds=1)) == {(0, 0, 1): 4}
    assert _values(module.specialize("rotate", n=3, rounds=3)) == {(1, 0, 0): 4}

    with pytest.raises(qm.CapabilityError, match="move the reordering into a helper"):
        _module(
            """
@qm.qkernel
def inline_rotate(n: qm.Const[qm.Int], rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    for _round in range(rounds):
        head, tail = qm.split_at(state, 1)
        state = qm.concat(tail, head)
    return qm.measure(state)
"""
        )


@pytest.mark.parametrize(("n", "half"), [(5, 2), (8, 4), (7, 3)])
def test_floor_division_widths_split_and_allocate(n: int, half: int) -> None:
    """``n // 2`` is a dimension: it splits, allocates, and annotates widths."""

    module = _module(
        """
@qm.qkernel
def halves(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    head, tail = qm.split_at(qm.qubits(n), n // 2)
    return qm.measure(qm.concat(qm.x(head), tail))

@qm.qkernel
def half_register(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n // 2]]:
    return qm.measure(qm.x(qm.qubits(n // 2)))
"""
    )
    assert "Bits[-(n//2)+n]" in module.inspect()
    assert _values(module.specialize("halves", n=n)) == {
        (1,) * half + (0,) * (n - half): 4
    }
    assert _values(module.specialize("half_register", n=n)) == {(1,) * half: 4}
    peak = qr.logical(module["halves"]).metric(qr.LOGICAL_PEAK_QUBITS)
    assert peak.evaluate({"n": n}).numeric_enclosure() == (n, n)


def test_floor_division_rejects_an_empty_part_at_specialization() -> None:
    """``n // 2`` is zero for ``n = 1``, which leaves an empty prefix."""

    module = _module(
        """
@qm.qkernel
def halves(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    head, tail = qm.split_at(qm.qubits(n), n // 2)
    return qm.measure(qm.concat(head, tail))
"""
    )
    with pytest.raises(qm.SpecializationError, match="Q3007"):
        module.specialize("halves", n=1)


def test_peak_width_survives_loop_index_splits_in_helpers() -> None:
    """Widths ``k``, ``1``, and ``p-k-1`` cancel to ``p`` across loop iterations."""

    module = _module(
        """
@qm.qkernel
def lane(
    counting: qm.Q[qm.Bits[p]], target: qm.Q[qm.Bits[2]], k: qm.Const[qm.Int]
) -> tuple[qm.Q[qm.Bits[p]], qm.Q[qm.Bits[2]]]:
    left, rest = qm.split_at(counting, k)
    head, right = qm.split_at(rest, 1)
    return qm.concat(left, qm.x(head), right), target

@qm.qkernel
def lanes(p: qm.Const[qm.Int]) -> tuple[qm.Shot[qm.Bits[p]], qm.Shot[qm.Bits[2]]]:
    counting = qm.qubits(p)
    target = qm.qubits(2)
    for k in range(1, p - 1):
        counting, target = lane(counting, target, k)
    return qm.measure(counting), qm.measure(target)

@qm.qkernel
def chain(state: qm.Q[qm.Bits[w]], size: qm.Const[qm.Int]) -> qm.Q[qm.Bits[w]]:
    for i in range(size - 1):
        state = qm.cx(state, control=i, target=i + 1)
    return state

@qm.qkernel
def two_chains(state: qm.Q[qm.Bits[w]], size: qm.Const[qm.Int], half: qm.Const[qm.Int]) -> qm.Q[qm.Bits[w]]:
    state = qm.cx(state, control=0, target=half)
    a, b = qm.split_at(state, half)
    return qm.concat(chain(a, half), chain(b, size - half))

@qm.qkernel
def ghz(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    return qm.measure(two_chains(qm.h(qm.qubits(n), target=0), n, n // 2))
"""
    )
    lanes = qr.logical(module["lanes"]).metric(qr.LOGICAL_PEAK_QUBITS)
    assert lanes.evaluate({"p": 6}).numeric_enclosure() == (8, 8)
    ghz = qr.logical(module["ghz"])
    assert ghz.metric(qr.LOGICAL_PEAK_QUBITS).evaluate(
        {"n": 64}
    ).numeric_enclosure() == (
        64,
        64,
    )
    assert (
        ghz.metric(qr.LOGICAL_GATE_DEPTH).evaluate({"n": 64}).numeric_enclosure()[1]
        == 33
    )
    outcomes = set(_values(module.specialize("ghz", n=7)))
    assert outcomes <= {(0,) * 7, (1,) * 7}
