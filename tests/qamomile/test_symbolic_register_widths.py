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
