"""Body-derived resource tests using a user-written Shor program."""

from __future__ import annotations

import pytest

import qamomile as qm
from tests.qamomile.shor_program import shor_15


def _operation_count(study: qm.resources.ResourceStudy, opcode: str) -> int:
    """Return one exact operation count inferred from a program body."""

    value = study.metric(qm.resources.logical_operation_count(opcode))
    assert value.relation is qm.resources.Relation.EQUALITY
    assert value.expression is not None
    count = value.expression.numeric()
    assert isinstance(count, int)
    return count


@pytest.fixture(scope="module")
def shor_study() -> qm.resources.ResourceStudy:
    """Analyze the unchanged Shor body once for the module."""

    return qm.resources.logical(shor_15)


def test_shor_resources_are_inferred_from_user_source(
    shor_study: qm.resources.ResourceStudy,
) -> None:
    """The generic QCore analyzer derives all counts without naming Shor."""

    study = shor_study

    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (
        12,
        12,
    )
    assert _operation_count(study, "quantum.h") == 34
    assert _operation_count(study, "quantum.cx") == 128
    assert _operation_count(study, "quantum.t") == 225
    assert _operation_count(study, "quantum.rz") == 84
    assert _operation_count(study, "quantum.swap") == 4
    assert _operation_count(study, "quantum.measure") == 1
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study


def test_resource_analysis_follows_body_not_algorithm_name(
    shor_study: qm.resources.ResourceStudy,
) -> None:
    """Names are irrelevant while an implementation edit changes its facts."""

    source = shor_15.source_text.replace("def shor_15()", "def my_experiment()")
    renamed = qm.resources.logical(qm.build_source(source)["my_experiment"])
    original = shor_study

    for opcode in ("quantum.h", "quantum.cx", "quantum.t", "quantum.rz"):
        assert _operation_count(renamed, opcode) == _operation_count(original, opcode)

    edited_source = source.replace(
        "state = inverse_qft(state, precision)",
        "state = inverse_qft(state, precision)\n"
        "    state = qm.x(state, target=precision)",
    )
    edited = qm.resources.logical(qm.build_source(edited_source)["my_experiment"])
    assert _operation_count(original, "quantum.x") == 1
    assert _operation_count(edited, "quantum.x") == 2


def test_nested_symbolic_loops_remain_analyzable() -> None:
    """Dependent loop bounds produce evaluable bounded sums, not opaque costs."""

    source = """
import qamomile as qm

@qm.qkernel
def any_user_algorithm(n: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for outer in range(n):
        for inner in range(outer):
            state = qm.h(state)
    return qm.measure(state)
"""
    study = qm.resources.logical(qm.build_source(source)["any_user_algorithm"])
    h_count = study.metric(qm.resources.logical_operation_count("quantum.h"))

    assert h_count.expression is not None
    assert "bounded_sum" in str(h_count.expression)
    assert h_count.evaluate({"n": 8}).numeric_enclosure() == (28, 28)


def test_symbolic_loop_bounds_follow_user_arithmetic() -> None:
    """Power expressions in ordinary source become resource expressions."""

    source = """
import qamomile as qm

@qm.qkernel
def exponential_work(n: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(2 ** n):
        state = qm.x(state)
    return qm.measure(state)
"""
    study = qm.resources.logical(qm.build_source(source)["exponential_work"])
    x_count = study.metric(qm.resources.logical_operation_count("quantum.x"))

    assert x_count.evaluate({"n": 4}).numeric_enclosure() == (16, 16)
