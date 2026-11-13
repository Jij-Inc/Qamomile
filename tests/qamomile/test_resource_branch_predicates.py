"""Symbolic resource estimates select the branch of a Const condition."""

from __future__ import annotations

import pytest

import qamomile as qm
import qamomile.resources as qr

SOURCE = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def guarded(n: qm.Const[qm.Int], steps: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    if steps > 0:
        state = qm.h(state)
        for index in range(n - 1):
            state = qm.cx(state, control=index, target=index + 1)
    else:
        state = qm.x(state, target=0)
    return qm.measure(state)
"""


@pytest.mark.parametrize(
    ("steps", "gates", "two_qubit"),
    [(0, 1, 0), (1, 9, 4), (7, 9, 4)],
)
def test_const_branch_costs_are_exact_piecewise_functions(
    steps: int, gates: int, two_qubit: int
) -> None:
    """A Const predicate yields one exact value instead of a [min, max] join."""

    study = qr.logical(qm.build_source(SOURCE)["guarded"])
    binding = {"n": 5, "steps": steps}
    assert study.metric(qr.LOGICAL_GATES).evaluate(binding).numeric_enclosure() == (
        gates,
        gates,
    )
    assert study.metric(qr.LOGICAL_TWO_QUBIT_GATES).evaluate(
        binding
    ).numeric_enclosure() == (two_qubit, two_qubit)


def test_piecewise_estimate_renders_its_branch_condition() -> None:
    """The printed estimate names the selecting predicate and every arm."""

    gates = qr.logical(qm.build_source(SOURCE)["guarded"]).metric(qr.LOGICAL_GATES)
    text = str(gates)
    assert "piecewise(" in text
    assert "steps > 0" in text
    assert "otherwise: 1" in text


def test_measured_branch_keeps_an_enclosure() -> None:
    """A Shot condition is not a symbolic predicate, so both arms stay possible."""

    source = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def measured() -> qm.Shot[qm.Bool]:
    first = qm.measure(qm.h(qm.prepare(False)))
    second = qm.prepare(False)
    if first:
        second = qm.x(second)
    return qm.measure(second)
"""
    gates = qr.logical(qm.build_source(source)["measured"]).metric(qr.LOGICAL_GATES)
    assert gates.numeric_enclosure() == (1, 2)
