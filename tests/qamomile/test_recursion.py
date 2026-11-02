"""Recursive language calls preserve typing, ownership, and bounded execution."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile import resources as qr

SOURCE = """
import qamomile as qm
@qm.qkernel
def recurse(state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if rounds == 0:
        return state
    else:
        return recurse(qm.x(state), rounds - 1)
@qm.qkernel
def sample(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(recurse(qm.qubits(1), rounds))
@qm.qkernel
def inverse(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.adjoint(recurse, recurse(qm.qubits(1), rounds), rounds))
@qm.qkernel
def controlled(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    control = qm.prepare((1,))
    control, target = qm.control(recurse, control, qm.qubits(1), rounds)
    return qm.measure(qm.concat(control, target))
@qm.qkernel
def factorial(n: qm.Param[qm.Int]) -> qm.Int:
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

@qm.qkernel
def mutual_a(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n == 0:
        return state
    else:
        return mutual_b(qm.x(state), n - 1)
@qm.qkernel
def mutual_b(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n == 0:
        return state
    else:
        return mutual_a(state, n - 1)
@qm.qkernel
def mutual_sample(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(mutual_a(qm.qubits(1), n))
@qm.qkernel
def tree(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n <= 1:
        return qm.x(state)
    else:
        state = tree(state, n // 2)
        return tree(state, n - n // 2)
@qm.qkernel
def tree_sample(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(tree(qm.qubits(1), n))

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def recursive_apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
    n: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    if n == 0:
        return state
    else:
        return recursive_apply(operation, operation(state), n - 1)

@qm.qkernel
def higher_order_sample(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(recursive_apply(flip, qm.qubits(1), n))

@qm.qkernel
def state(rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return recurse(qm.qubits(1), rounds)
"""


def exercise_recursive_backend(endpoint, *, seed=7):
    """Execute the same recursive ownership transfers on each capable SDK."""
    module = qm.build_source(SOURCE)
    for rounds in (0, 3, 4):
        program = qm.Program.from_bytes(
            module.specialize("sample", rounds=rounds).to_bytes()
        )
        artifact = qm.compile(program, to=endpoint.target)
        assert qm.execute(artifact, on=endpoint, shots=8, seed=seed).counts == {
            (rounds % 2,): 8
        }
    for entry, bindings, expected in [
        ("inverse", {"rounds": 3}, (0,)),
        ("controlled", {"rounds": 3}, (1, 1)),
        ("mutual_sample", {"n": 1}, (1,)),
        ("mutual_sample", {"n": 3}, (0,)),
        ("tree_sample", {"n": 3}, (1,)),
        ("higher_order_sample", {"n": 3}, (1,)),
    ]:
        assert qm.run(
            module[entry], on=endpoint, shots=8, seed=seed, **bindings
        ).result.counts == {expected: 8}
    assert (
        qm.expectation(
            module["state"], qm.Z(0), on=endpoint, rounds=3, shots=8, seed=seed
        ).value
        == -1.0
    )


def test_recursive_reference_execution_and_persistence():
    """Self, mutual, and branching recursion execute from canonical programs."""
    exercise_recursive_backend(qm.Reference())


def test_recursive_classical_param_calls_execute_in_native_reference():
    """Pure-classical recursion accepts launch values and returns exact data."""
    program = qm.build_source(SOURCE).specialize("factorial")
    assert qm.evaluate(qm.Program.from_bytes(program.to_bytes()), n=5) == 120
    assert qm.evaluate(program, n=0) == 1


@pytest.mark.parametrize(
    "entry,parameter,arguments,expected",
    [
        ("sample", "rounds", (0, 3, 10_000), (0, 3, 10_000)),
        ("mutual_sample", "n", (0, 3, 10), (0, 2, 5)),
        ("tree_sample", "n", (0, 3, 10_000), (1, 3, 10_000)),
        ("higher_order_sample", "n", (0, 3, 10), (0, 3, 10)),
    ],
)
def test_recursive_additive_equations_evaluate_and_round_trip(
    entry, parameter, arguments, expected
):
    """Open recurrences evaluate in Rust without circuit expansion or SymPy."""
    module = qm.build_source(SOURCE)
    study = qr.logical(module[entry])
    restored = qr.ResourceStudy.from_bytes(study.to_bytes())
    assert restored.to_bytes() == study.to_bytes()
    value = restored.metric(qr.LOGICAL_GATES)
    assert value.relation is qr.Relation.EQUALITY
    assert value.expression is not None
    assert "recurrence_system" in str(value.expression)
    for argument, result in zip(arguments, expected):
        assert value.expression.evaluate({parameter: argument}).numeric() == result
        closed = qr.logical(module.specialize(entry, **{parameter: argument}))
        assert closed.metric(qr.LOGICAL_GATES).expression.numeric() == result
    assert (
        restored.metric(qr.LOGICAL_MEASUREMENTS)
        .expression.evaluate({parameter: 3})
        .numeric()
        == 1
    )
    for metric in (qr.LOGICAL_PEAK_QUBITS, qr.LOGICAL_SPAN):
        unsupported = restored.metric(metric)
        assert unsupported.upper is None
        assert any(
            "recurrence" in reason for reason in unsupported.unknown_contributors
        )
    with pytest.raises(ValueError, match="strict resource analysis"):
        qr.logical(module[entry], strict=True)


def test_recursive_resource_domains_and_budgets_preserve_unresolved_equations():
    """A divergent signed input cannot acquire a fictitious finite gate count."""
    study = qr.logical(qm.build_source(SOURCE)["sample"])
    expression = study.metric(qr.LOGICAL_GATES).expression
    assert expression.evaluate({"rounds": -1}).numeric() is None
    assert expression.evaluate({"rounds": 1 << 63}).numeric() is None
    assert expression.evaluate({"rounds": 1_000_000}).numeric() is None
    closed = qr.logical(qm.build_source(SOURCE).specialize("sample", rounds=-1))
    assert closed.metric(qr.LOGICAL_GATES).expression.numeric() is None


def test_recursive_early_return_resource_condition():
    """The fallthrough cost is guarded by the negated early return condition."""
    source = SOURCE.replace(
        "    else:\n        return recurse(qm.x(state), rounds - 1)",
        "    return recurse(qm.x(state), rounds - 1)",
        1,
    )
    expression = (
        qr.logical(qm.build_source(source)["sample"])
        .metric(qr.LOGICAL_GATES)
        .expression
    )
    assert expression is not None
    assert expression.evaluate({"rounds": 0}).numeric() == 0
    assert expression.evaluate({"rounds": 3}).numeric() == 3


def test_unused_recursion_does_not_poison_unrelated_resource_metrics():
    """An unused cycle has no effect on the selected entrypoint's evidence."""
    source = (
        SOURCE
        + """
@qm.qkernel
def independent() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.x(qm.qubits(1)))
"""
    )
    study = qr.logical(qm.build_source(source)["independent"], strict=True)
    assert study.metric(qr.LOGICAL_GATES).expression.numeric() == 1
    assert study.metric(qr.LOGICAL_PEAK_QUBITS).expression.numeric() == 1


def test_native_recurrence_definition_scope_and_signature_validation():
    """Equation binders survive substitution and malformed systems fail closed."""
    from qamomile.resources.expr import _root_free_symbols

    expression = (
        qr.logical(qm.build_source(SOURCE)["sample"])
        .metric(qr.LOGICAL_GATES)
        .expression
    )
    assert [
        (symbol.scope, symbol.name) for symbol in _root_free_symbols(expression)
    ] == [("module.sample", "rounds")]
    restored = qr.Expr.from_data(expression.to_data())
    assert restored.evaluate({"rounds": 4}).numeric() == 4
    system = next(
        item for item in expression.arguments if item.kind == "recurrence_system"
    )
    equation = system.arguments[1]
    # Qualified formal identities are bound and cannot be changed by host mappings.
    assert (
        restored.substitute({equation.arguments[0].symbol.identity: 999})
        .evaluate({"rounds": 4})
        .numeric()
        == 4
    )
    data = system.to_data()
    from copy import deepcopy

    data["arguments"].append(deepcopy(data["arguments"][1]))
    with pytest.raises(ValueError, match="duplicate recurrence equation"):
        qr.Expr.from_data(data)
    data = system.to_data()
    data["arguments"][1]["arguments"].pop(0)
    with pytest.raises(ValueError, match="equation signature"):
        qr.Expr.from_data(data)


def test_nonterminating_quantum_recursion_fails_with_bounded_expansion():
    """Recursive source is valid; circuit expansion rejects runaway execution."""
    source = """
import qamomile as qm
@qm.qkernel
def forever(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return forever(state)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(forever(qm.qubits(1)))
"""
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match="call expansion.*depth budget"):
        qm.run(program, on=qm.Reference(), shots=1)


def test_recursive_classical_reference_execution_is_bounded():
    """Nonterminating classical calls use the existing reference work limit."""
    source = """
import qamomile as qm
@qm.qkernel
def forever(n: qm.Param[qm.Int]) -> qm.Int:
    return forever(n)
"""
    program = qm.build_source(source).specialize("forever")
    with pytest.raises(qm.QamomileError, match="work budget"):
        qm.evaluate(program, n=0)


def test_recursive_quantum_ownership_cannot_duplicate_a_successor():
    """Admitting cycles does not weaken native consumption verification."""
    source = """
import qamomile as qm
@qm.qkernel
def duplicate(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    result = duplicate(state, n - 1)
    return qm.x(state)
"""
    with pytest.raises(qm.QamomileError, match="Q5020|Q2001"):
        qm.build_source(source)


def test_quantum_param_recursion_is_materialized_at_launch():
    """A launch-bound recursive selector reuses the ordinary Param artifact."""
    source = SOURCE.replace("rounds: qm.Const[qm.Int]", "rounds: qm.Param[qm.Int]")
    program = qm.build_source(source).specialize("sample")
    artifact = qm.compile(program, to=qm.Reference().target)
    for rounds in (0, 3, 4):
        assert qm.execute(
            artifact, on=qm.Reference(), parameters={"rounds": rounds}, shots=8, seed=7
        ).counts == {(rounds % 2,): 8}


@pytest.mark.parametrize("transform", ["adjoint", "control"])
def test_recursive_transforms_reject_transitive_measurement_effects(transform):
    """A measurement inside an SCC invalidates transformations of every member."""
    source = """
import qamomile as qm
@qm.qkernel
def a(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n == 0:
        bit = qm.measure(state)
        return qm.prepare(bit)
    else:
        return b(state, n - 1)
@qm.qkernel
def b(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return a(state, n)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    TRANSFORM
"""
    body = (
        "return qm.measure(qm.adjoint(b, qm.qubits(1), 3))"
        if transform == "adjoint"
        else "control, state = qm.control(b, qm.qubits(1), qm.qubits(1), 3)\n    return qm.measure(qm.concat(control, state))"
    )
    expected = "Q1029" if transform == "adjoint" else "Q1051"
    if transform == "control":
        source = source.replace(
            "def sample() -> qm.Shot[qm.Bits[1]]:",
            "def sample() -> qm.Shot[qm.Bits[2]]:",
        )
    with pytest.raises(qm.QamomileError, match=expected):
        qm.build_source(source.replace("TRANSFORM", body))


def test_shot_dependent_recursion_is_valid_language_but_requires_runtime_calls():
    """Typed measured recursion survives build while finite circuit emission fails."""
    source = """
import qamomile as qm
@qm.qkernel
def retry(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    bits, state = qm.measure_reset(state)
    if bool(bits[0]):
        return state
    else:
        return retry(qm.h(state))
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(retry(qm.qubits(1)))
"""
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.QamomileError, match="call expansion.*depth budget"):
        qm.run(program, on=qm.Reference(), shots=1)


def test_recursive_calls_preserve_dependent_quantum_widths():
    """The same generic owner signature remains valid across recursive frames."""
    source = """
import qamomile as qm
@qm.qkernel
def recursively_flip(state: qm.Q[qm.Bits[n]], rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[n]]:
    if rounds == 0:
        return state
    else:
        return recursively_flip(qm.x(state), rounds - 1)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    return qm.measure(recursively_flip(qm.prepare((0, 1, 0)), 3))
"""
    program = qm.build_source(source).specialize("sample")
    assert qm.run(program, on=qm.Reference(), shots=8, seed=7).result.counts == {
        (1, 0, 1): 8
    }


def test_recursive_symbolic_widths_and_arbitrary_precision_parameters():
    """Equation parameters preserve quantum width and arbitrary precision Int."""
    source = """
import qamomile as qm
@qm.qkernel
def wide_step(
    state: qm.Q[qm.Bits[n]], n: qm.Const[qm.Int], rounds: qm.Const[qm.Int]
) -> qm.Q[qm.Bits[n]]:
    if rounds == 0:
        return state
    else:
        for index in range(n):
            state = qm.x(state, target=index)
        return wide_step(state, n, rounds - 1)
@qm.qkernel
def wide_sample(n: qm.Const[qm.Int], rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    return qm.measure(wide_step(qm.qubits(n), n, rounds))
@qm.qkernel
def huge_base(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n > 0:
        return qm.x(state)
    else:
        return huge_base(state, n - 1)
"""
    module = qm.build_source(source)
    study = qr.logical(module["wide_sample"])
    assert (
        study.metric(qr.LOGICAL_GATES)
        .expression.evaluate({"n": 2, "rounds": 3})
        .numeric()
        == 6
    )
    assert (
        study.metric(qr.LOGICAL_MEASUREMENTS)
        .expression.evaluate({"n": 2, "rounds": 3})
        .numeric()
        == 2
    )
    huge = qr.logical(module["huge_base"]).metric(qr.LOGICAL_GATES).expression
    assert huge.evaluate({"n": 1 << 100}).numeric() == 1
    assert qm.run(
        module["wide_sample"], on=qm.Reference(), n=2, rounds=3, shots=8, seed=7
    ).result.counts == {(1, 1): 8}


def test_recursive_resource_rule_evidence_survives_equation_closure():
    """Trusted rule assumptions stay attached to their recursive contribution."""
    source = """
import qamomile as qm
@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...
@qm.qkernel
def recurse(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    if n == 0:
        return state
    else:
        return recurse(oracle(state), n - 1)
"""
    module = qm.build_source(source)
    rule = qr.ResourceRule(
        identity="test.recursive.oracle.v1",
        callable=module.function("oracle"),
        metric=qr.LOGICAL_GATES,
        relation=qr.Relation.EQUALITY,
        expression=qr.literal(2),
        assumptions=("The oracle has the declared gate cost.",),
        provenance="test.recursive.oracle.cost.v1",
    )
    duration = qr.MetricDefinition(
        "test.recursive.duration",
        qr.Coordinate.LOGICAL,
        qr.Domain.REAL,
        qr.SECONDS,
        qr.Composition.ADDITIVE_WORK,
    )
    duration_rule = qr.ResourceRule(
        identity="test.recursive.duration.v1",
        callable=module.function("oracle"),
        metric=duration,
        relation=qr.Relation.EQUALITY,
        expression=qr.literal(2.5),
        provenance="test.recursive.duration.cost.v1",
    )
    study = qr.logical(module["recurse"], rules=(rule, duration_rule))
    assert study.metric(duration).upper is None
    assert study.metric(duration).unknown_contributors
    value = study.metric(qr.LOGICAL_GATES)
    assert value.expression.evaluate({"n": 3}).numeric() == 6
    assert rule.assumptions[0] in value.assumptions
    assert any(
        evidence.authority is qr.EvidenceAuthority.TRUSTED_AXIOM
        for evidence in value.evidence
    )
    queries = study.metric(qr.algorithmic_query_count(module.function("oracle").symbol))
    assert queries.expression.evaluate({"n": 3}).numeric() == 3
    calls = study.metric(qr.algorithmic_call_count(module.function("oracle").symbol))
    assert calls.expression.evaluate({"n": 3}).numeric() == 3
    restored = qr.ResourceStudy.from_bytes(study.to_bytes()).metric(qr.LOGICAL_GATES)
    assert restored.evidence == value.evidence
    assert restored.assumptions == value.assumptions


@pytest.mark.parametrize("stage", ["Const", "Param"])
def test_recursive_portable_projection_never_reports_a_missing_body_as_zero(stage):
    """Deferred projection metrics retain uncertainty instead of an exact zero."""
    source = SOURCE.replace("qm.Const[qm.Int]", f"qm.{stage}[qm.Int]")
    module = qm.build_source(source)
    study = qr.logical(module["sample"])
    projected = qr.project(study, qr.GateCostModel())
    gates = projected.metric(qr.portable_metric("gates"))
    assert gates.upper is None
    assert gates.unknown_contributors
    assert any(
        "recursive observation" in reason for reason in gates.unknown_contributors
    )
    if stage == "Const":
        assert (
            study.metric(qr.LOGICAL_GATES).expression.evaluate({"rounds": 3}).numeric()
            == 3
        )
