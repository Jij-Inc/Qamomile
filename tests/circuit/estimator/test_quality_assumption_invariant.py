"""Regression tests for non-exact resource-quality explanations."""

from __future__ import annotations

import ast
import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator._wire import (
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)

_UNPRICED_ORACLE = qmc.opaque("quality_reason_unpriced", num_qubits=1)


@qmc.qkernel
def _unpriced_kernel() -> qmc.Qubit:
    """Invoke one callable without an implementation or declared cost."""
    (target,) = _UNPRICED_ORACLE(qmc.qubit("target"))
    return target


@qmc.qkernel
def _one_h_body(target: qmc.Qubit) -> qmc.Qubit:
    """Apply one Hadamard gate."""
    return qmc.h(target)


@qmc.qkernel
def _four_h_body(target: qmc.Qubit) -> qmc.Qubit:
    """Apply four Hadamard gates."""
    for _ in qmc.range(4):
        target = qmc.h(target)
    return target


@qmc.qkernel
def _controlled_single_gate_kernel() -> qmc.Qubit:
    """Apply one gate under three coherent controls."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    *_, target = qmc.control(_one_h_body, num_controls=3)(controls, target)
    return target


@qmc.qkernel
def _controlled_body_kernel() -> qmc.Qubit:
    """Apply a multi-gate body under three coherent controls."""
    controls = qmc.qubit_array(3, "controls")
    target = qmc.qubit("target")
    *_, target = qmc.control(_four_h_body, num_controls=3)(controls, target)
    return target


_FIXED_CONSERVATIVE_COST = qmc.ResourceEstimate(
    gates=qmc.GateResources(total=1, single_qubit=1),
    quality=qmc.EstimateQuality.CONSERVATIVE,
)
_FIXED_CONSERVATIVE_ORACLE = qmc.opaque(
    "quality_reason_fixed",
    num_qubits=1,
    cost=_FIXED_CONSERVATIVE_COST,
)


def _unknown_callback_cost(
    _context: qmc.OpaqueCostContext,
) -> qmc.ResourceEstimate:
    """Return a deliberately unknown callback-provided resource cost.

    Args:
        _context (qmc.OpaqueCostContext): Opaque-call context.

    Returns:
        qmc.ResourceEstimate: Unknown one-gate estimate.
    """
    return qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, single_qubit=1),
        quality=qmc.EstimateQuality.UNKNOWN,
    )


_CALLBACK_UNKNOWN_ORACLE = qmc.opaque(
    "quality_reason_callback",
    num_qubits=1,
    cost=_unknown_callback_cost,
)


@qmc.qkernel
def _fixed_cost_kernel() -> qmc.Qubit:
    """Invoke one callable with a fixed conservative cost."""
    (target,) = _FIXED_CONSERVATIVE_ORACLE(qmc.qubit("target"))
    return target


@qmc.qkernel
def _callback_cost_kernel() -> qmc.Qubit:
    """Invoke one callable with a callback-provided unknown cost."""
    (target,) = _CALLBACK_UNKNOWN_ORACLE(qmc.qubit("target"))
    return target


@qmc.qkernel
def _domain_quality_kernel(length: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Create a symbolic chain whose first access requires positive length."""
    register = qmc.qubit_array(length, "register")
    register[0] = qmc.h(register[0])
    for index in qmc.range(1, length):
        register[index - 1], register[index] = qmc.cx(
            register[index - 1],
            register[index],
        )
    return register


def _active_quality_facts(
    estimate: qmc.ResourceEstimate,
) -> tuple[Any, ...]:
    """Return active or potentially active non-exact quality facts.

    Args:
        estimate (qmc.ResourceEstimate): Estimate to inspect.

    Returns:
        tuple[Any, ...]: Non-exact guarded quality facts.
    """
    return tuple(
        fact
        for fact in (estimate._guarded_qualities or ())
        if fact.active_when is not sp.false
    )


def _assert_quality_reasons_are_exposed(
    estimate: qmc.ResourceEstimate,
) -> None:
    """Assert that every active non-exact quality exposes its reason.

    Args:
        estimate (qmc.ResourceEstimate): Estimate to validate.
    """
    assert estimate.quality is not qmc.EstimateQuality.EXACT
    facts = _active_quality_facts(estimate)
    assert facts
    assert estimate.assumptions
    assert all(fact.reason in estimate.assumptions for fact in facts)


@pytest.mark.parametrize(
    "quality",
    [qmc.EstimateQuality.CONSERVATIVE, qmc.EstimateQuality.UNKNOWN],
)
def test_public_non_exact_construction_synthesizes_a_quality_reason(
    quality: qmc.EstimateQuality,
) -> None:
    """Direct non-exact construction always synthesizes a visible reason.

    Args:
        quality (qmc.EstimateQuality): Non-exact quality to construct.
    """
    estimate = qmc.ResourceEstimate(quality=quality)

    assert estimate.quality is quality
    _assert_quality_reasons_are_exposed(estimate)


def test_public_construction_uses_a_supplied_assumption_as_the_reason() -> None:
    """The first supplied premise explains direct non-exact quality."""
    reason = qmc.ResourceAssumption("declared upper-bound model", source="test")
    estimate = qmc.ResourceEstimate(
        assumptions=(reason,),
        quality=qmc.EstimateQuality.CONSERVATIVE,
    )

    assert estimate.assumptions == (reason,)
    assert _active_quality_facts(estimate)[0].reason is reason


def test_exact_quality_may_still_have_an_assumption() -> None:
    """The quality-to-assumption guarantee remains intentionally one-way."""
    assumption = qmc.ResourceAssumption("valid only on a declared domain")
    estimate = qmc.ResourceEstimate(assumptions=(assumption,))

    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == (assumption,)
    assert _active_quality_facts(estimate) == ()


def test_explicit_quality_reason_composes_with_ordinary_assumptions() -> None:
    """Ordinary premises and a distinct quality reason remain visible once."""
    ordinary = qmc.ResourceAssumption("independent model premise", source="test")
    reason = qmc.ResourceAssumption("unknown directional relation", source="test")

    estimate = qmc.ResourceEstimate.zero()._with_metadata(
        assumptions=(ordinary, reason),
        quality=qmc.EstimateQuality.UNKNOWN,
        quality_reason=reason,
    )

    assert estimate.assumptions == (ordinary, reason)
    assert _active_quality_facts(estimate)[0].reason is reason


def test_choice_records_its_specific_conservative_reason() -> None:
    """A field-wise choice explains why its result is conservative."""
    left = qmc.ResourceEstimate(gates=qmc.GateResources(total=1, single_qubit=1))
    right = qmc.ResourceEstimate(gates=qmc.GateResources(total=2, two_qubit=2))

    estimate = left.choice(right)

    _assert_quality_reasons_are_exposed(estimate)
    assert any(
        "choice" in f"{assumption.source} {assumption.message}".lower()
        for assumption in estimate.assumptions
    )


@pytest.mark.parametrize(
    ("policy", "message_fragment"),
    [
        (qmc.UnknownResourcePolicy.OPAQUE_CALL, "opaque call"),
        (qmc.UnknownResourcePolicy.ZERO_WITH_WARNING, "counted as zero"),
    ],
)
def test_unknown_callable_policies_expose_one_precise_reason(
    policy: qmc.UnknownResourcePolicy,
    message_fragment: str,
) -> None:
    """Each non-error unknown policy reports one non-duplicated reason.

    Args:
        policy (qmc.UnknownResourcePolicy): Unknown-call handling policy.
        message_fragment (str): Text identifying the selected policy.
    """
    estimate = _unpriced_kernel.estimate_resources(unknown_policy=policy)

    assert estimate.quality is qmc.EstimateQuality.UNKNOWN
    assert len(estimate.assumptions) == 1
    assert message_fragment in estimate.assumptions[0].message
    assert estimate.assumptions[0].source == "quality_reason_unpriced"
    _assert_quality_reasons_are_exposed(estimate)


@pytest.mark.parametrize(
    "kernel",
    [_controlled_single_gate_kernel, _controlled_body_kernel],
)
def test_clean_ancilla_control_paths_expose_their_upper_bound_reason(
    kernel: qmc.QKernel,
) -> None:
    """Primitive and shared-ladder control paths both explain their bound.

    Args:
        kernel (qmc.QKernel): Controlled qkernel to estimate.
    """
    estimate = kernel.estimate_resources()

    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    _assert_quality_reasons_are_exposed(estimate)
    assert any(
        "control" in f"{assumption.source} {assumption.message}".lower()
        or "ladder" in assumption.message.lower()
        for assumption in estimate.assumptions
    )


def test_aggregate_control_complete_partial_and_zero_profiles_keep_reasons() -> None:
    """Aggregate control projection explains every active non-exact branch."""
    complete = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=2, single_qubit=1, two_qubit=1)
    )
    partial = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=3, single_qubit=1, two_qubit=1)
    )

    conservative = complete.controlled(2)
    unknown = partial.controlled(2)
    inactive = complete.controlled(0)

    assert conservative.quality is qmc.EstimateQuality.CONSERVATIVE
    assert unknown.quality is qmc.EstimateQuality.UNKNOWN
    _assert_quality_reasons_are_exposed(conservative)
    _assert_quality_reasons_are_exposed(unknown)
    assert inactive.quality is qmc.EstimateQuality.EXACT
    assert inactive.assumptions == ()


def test_algebra_preserves_or_prunes_quality_reasons_with_their_facts() -> None:
    """Composition, conditionals, repetition, and sums keep reasons aligned."""
    reason = qmc.ResourceAssumption("conditional upper bound", source="test")
    flag = sp.Symbol("quality_flag", integer=True, nonnegative=True)
    repetitions = sp.Symbol("quality_repetitions", integer=True, nonnegative=True)
    index = sp.Symbol("quality_index", integer=True, nonnegative=True)
    non_exact = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1, single_qubit=1)
    )._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=reason,
    )
    exact = qmc.ResourceEstimate.zero()

    for estimate in (non_exact.seq(exact), non_exact.parallel(exact)):
        _assert_quality_reasons_are_exposed(estimate)
        assert reason in estimate.assumptions

    conditional = non_exact.conditional(exact, sp.Gt(flag, 0))
    repeated = non_exact.repeat(repetitions)
    summed = non_exact.sum_over(index, 0, repetitions)

    for symbolic in (conditional, repeated, summed):
        _assert_quality_reasons_are_exposed(symbolic)
        inactive = symbolic.substitute(**{next(iter(symbolic.parameters)): 0})
        active = symbolic.substitute(**{next(iter(symbolic.parameters)): 2})
        assert inactive.quality is qmc.EstimateQuality.EXACT
        assert inactive.assumptions == ()
        _assert_quality_reasons_are_exposed(active)
        assert reason in active.assumptions


@pytest.mark.parametrize(
    ("conservative_active", "unknown_active", "expected_quality", "expected_count"),
    [
        (0, 0, qmc.EstimateQuality.EXACT, 0),
        (1, 0, qmc.EstimateQuality.CONSERVATIVE, 1),
        (0, 1, qmc.EstimateQuality.UNKNOWN, 1),
        (1, 1, qmc.EstimateQuality.UNKNOWN, 2),
    ],
)
def test_guarded_quality_specialization_keeps_each_fact_with_its_own_reason(
    conservative_active: int,
    unknown_active: int,
    expected_quality: qmc.EstimateQuality,
    expected_count: int,
) -> None:
    """Independent guards prune only their corresponding quality reasons.

    Args:
        conservative_active (int): Whether the conservative fact is active.
        unknown_active (int): Whether the unknown fact is active.
        expected_quality (qmc.EstimateQuality): Combined specialized quality.
        expected_count (int): Number of surviving quality reasons.
    """
    conservative_flag = sp.Symbol("conservative_active", integer=True, nonnegative=True)
    unknown_flag = sp.Symbol("unknown_active", integer=True, nonnegative=True)
    conservative_reason = qmc.ResourceAssumption(
        "conservative scheduling envelope", source="conservative fact"
    )
    unknown_reason = qmc.ResourceAssumption(
        "unknown opaque remainder", source="unknown fact"
    )
    symbolic = (
        qmc.ResourceEstimate.zero()
        ._with_metadata(
            quality=qmc.EstimateQuality.CONSERVATIVE,
            quality_reason=conservative_reason,
            active_when=sp.Gt(conservative_flag, 0),
        )
        ._with_metadata(
            quality=qmc.EstimateQuality.UNKNOWN,
            quality_reason=unknown_reason,
            active_when=sp.Gt(unknown_flag, 0),
        )
    )

    estimate = symbolic.substitute(
        conservative_active=conservative_active,
        unknown_active=unknown_active,
    )

    assert estimate.quality is expected_quality
    assert len(_active_quality_facts(estimate)) == expected_count
    assert len(estimate.assumptions) == expected_count
    if conservative_active:
        assert conservative_reason in estimate.assumptions
    if unknown_active:
        assert unknown_reason in estimate.assumptions


def test_appended_identity_duplicate_explains_a_later_unknown_quality() -> None:
    """An appended duplicate remains the reason after an older guard expires."""
    flag = sp.Symbol("identity_prefix_flag", integer=True, nonnegative=True)
    reason = qmc.ResourceAssumption("shared reason object", source="identity test")
    conservative = qmc.ResourceEstimate.zero()._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=reason,
        active_when=sp.Gt(flag, 0),
    )

    unknown = dataclasses.replace(
        conservative,
        assumptions=(*conservative.assumptions, reason),
        quality=qmc.EstimateQuality.UNKNOWN,
    )
    specialized = unknown.substitute(identity_prefix_flag=0)

    assert unknown.quality is qmc.EstimateQuality.UNKNOWN
    assert len(_active_quality_facts(unknown)) == 2
    assert all(fact.reason is reason for fact in _active_quality_facts(unknown))
    assert specialized.quality is qmc.EstimateQuality.UNKNOWN
    assert specialized.assumptions == (reason,)
    (unknown_fact,) = _active_quality_facts(specialized)
    assert unknown_fact.quality is qmc.EstimateQuality.UNKNOWN
    assert unknown_fact.reason is reason


def test_later_unknown_quality_does_not_reuse_an_old_conservative_reason() -> None:
    """Weakening quality without a new premise keeps two distinct reasons."""
    conservative_reason = qmc.ResourceAssumption(
        "original conservative envelope",
        source="conservative fact",
    )
    conservative = qmc.ResourceEstimate.zero()._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=conservative_reason,
    )

    unknown = dataclasses.replace(
        conservative,
        quality=qmc.EstimateQuality.UNKNOWN,
    )

    assert unknown.quality is qmc.EstimateQuality.UNKNOWN
    assert len(_active_quality_facts(unknown)) == 2
    assert conservative_reason in unknown.assumptions
    (unknown_fact,) = tuple(
        fact
        for fact in _active_quality_facts(unknown)
        if fact.quality is qmc.EstimateQuality.UNKNOWN
    )
    assert unknown_fact.reason != conservative_reason
    assert unknown_fact.reason in unknown.assumptions


def test_domain_restore_keeps_quality_reasons_and_discharges_only_domain() -> None:
    """Composition and specialization preserve reasons outside the domain."""
    conservative_reason = qmc.ResourceAssumption(
        "domain-independent conservative reason",
        source="conservative fact",
    )
    symbolic = _domain_quality_kernel.estimate_resources()._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=conservative_reason,
    )
    weakened = dataclasses.replace(
        symbolic,
        quality=qmc.EstimateQuality.UNKNOWN,
    ).seq(qmc.ResourceEstimate.zero())

    specialized = weakened.substitute(length=2)

    assert specialized.quality is qmc.EstimateQuality.UNKNOWN
    assert len(_active_quality_facts(specialized)) == 2
    assert conservative_reason in specialized.assumptions
    assert all(
        assumption.source != "qkernel input domain"
        for assumption in specialized.assumptions
    )
    assert all(
        fact.reason in specialized.assumptions
        for fact in _active_quality_facts(specialized)
    )


def test_seq_all_deferred_refresh_preserves_symbolic_quality_reasons() -> None:
    """Deferred composition keeps symbolic parameters and quality reasons."""
    symbol = sp.Symbol("deferred_quality_count", integer=True, nonnegative=True)
    reason = qmc.ResourceAssumption(
        "symbolic count uses a conservative declaration",
        source="deferred composition",
    )
    leaf = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=symbol, single_qubit=symbol),
    )._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=reason,
    )

    estimate = qmc.ResourceEstimate.seq_all([leaf, leaf, leaf])

    assert estimate.parameters == {"deferred_quality_count": symbol}
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.assumptions == (reason,)
    assert estimate.substitute(deferred_quality_count=2).assumptions == (reason,)


@pytest.mark.parametrize(
    ("kernel", "quality"),
    [
        (_fixed_cost_kernel, qmc.EstimateQuality.CONSERVATIVE),
        (_callback_cost_kernel, qmc.EstimateQuality.UNKNOWN),
    ],
)
def test_opaque_declared_costs_preserve_synthesized_quality_reasons(
    kernel: qmc.QKernel,
    quality: qmc.EstimateQuality,
) -> None:
    """Fixed and callback opaque costs retain their declared-quality reason.

    Args:
        kernel (qmc.QKernel): Kernel invoking the declared opaque cost.
        quality (qmc.EstimateQuality): Declared non-exact quality.
    """
    estimate = kernel.estimate_resources()

    assert estimate.quality is quality
    _assert_quality_reasons_are_exposed(estimate)


def _wire_estimate() -> qmc.ResourceEstimate:
    """Build a non-exact estimate for wire-format checks.

    Returns:
        qmc.ResourceEstimate: Estimate with one explicit quality reason.
    """
    return qmc.ResourceEstimate.zero()._with_metadata(
        quality=qmc.EstimateQuality.CONSERVATIVE,
        quality_reason=qmc.ResourceAssumption(
            "wire-format upper bound", source="wire test"
        ),
    )


def test_wire_round_trip_preserves_embedded_quality_reasons() -> None:
    """Wire serialization preserves each quality fact and its reason."""
    estimate = _wire_estimate()

    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))

    assert restored == estimate
    assert restored._guarded_qualities == estimate._guarded_qualities
    _assert_quality_reasons_are_exposed(restored)


@pytest.mark.parametrize(
    "tamper",
    ["missing", "extra", "wrong_type", "blank", "exact", "public_assumption"],
)
def test_wire_rejects_malformed_or_disconnected_quality_reasons(
    tamper: str,
) -> None:
    """Wire decoding rejects malformed reasons and public disagreement.

    Args:
        tamper (str): Mutation applied to an otherwise canonical payload.
    """
    wire = deepcopy(resource_estimate_to_wire(_wire_estimate()))
    quality = wire["provenance"]["qualities"][0]
    if tamper == "missing":
        del quality["reason"]
    elif tamper == "extra":
        quality["unexpected"] = "field"
    elif tamper == "wrong_type":
        quality["reason"] = "not an assumption record"
    elif tamper == "blank":
        quality["reason"] = {"message": "", "source": "wire test"}
    elif tamper == "exact":
        quality["quality"] = qmc.EstimateQuality.EXACT.value
        wire["quality"] = qmc.EstimateQuality.EXACT.value
    else:
        wire["assumptions"] = []

    with pytest.raises(ValueError):
        resource_estimate_from_wire(wire)


@pytest.mark.parametrize("boundary", ["report", "sequential", "wire"])
def test_public_quality_reason_mutation_is_rejected_at_boundaries(
    boundary: str,
) -> None:
    """Removing a visible quality reason invalidates canonical provenance.

    Args:
        boundary (str): Public validation boundary to exercise.
    """
    estimate = _wire_estimate()
    estimate.assumptions = ()

    with pytest.raises(RuntimeError, match="provenance|assumptions"):
        if boundary == "report":
            estimate.to_dict()
        elif boundary == "sequential":
            estimate.seq(qmc.ResourceEstimate.zero())
        else:
            resource_estimate_to_wire(estimate)


def _call_name(node: ast.Call) -> str:
    """Return the final identifier of an AST call target.

    Args:
        node (ast.Call): Call expression to inspect.

    Returns:
        str: Final callable identifier, or an empty string.
    """
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _is_non_exact_literal(node: ast.expr) -> bool:
    """Return whether an AST expression names a non-exact quality literal.

    Args:
        node (ast.expr): Expression assigned to a quality keyword.

    Returns:
        bool: Whether the expression is ``CONSERVATIVE`` or ``UNKNOWN``.
    """
    return isinstance(node, ast.Attribute) and node.attr in {
        "CONSERVATIVE",
        "UNKNOWN",
    }


def _call_supplies_nonempty_assumptions(
    keywords: dict[str, ast.expr],
) -> bool:
    """Return whether a call supplies assumptions that are not visibly empty.

    Args:
        keywords (dict[str, ast.expr]): Call keywords indexed by name.

    Returns:
        bool: Whether ``assumptions`` exists and is not an empty literal.
    """
    assumptions = keywords.get("assumptions")
    return assumptions is not None and not (
        isinstance(assumptions, (ast.Tuple, ast.List)) and not assumptions.elts
    )


def test_estimator_sources_cannot_create_an_unexplained_quality_fact() -> None:
    """Production source uses the central quality-with-reason construction."""
    repository = Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for path in sorted((repository / "qamomile/circuit/estimator").glob("*.py")):
        tree = ast.parse(path.read_text())
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            ancestor = parents.get(node)
            while ancestor is not None and not isinstance(ancestor, ast.keyword):
                ancestor = parents.get(ancestor)
            if (
                call_name == "replace"
                and isinstance(ancestor, ast.keyword)
                and ancestor.arg == "_guarded_qualities"
            ):
                violations.append(
                    f"{path.relative_to(repository)}:{node.lineno}: "
                    "guarded quality reconstructed without the central factory"
                )
            if call_name == "_GuardedQuality":
                parent = parents.get(node)
                while parent is not None and not isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    parent = parents.get(parent)
                if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) or (
                    parent.name != "_guarded_quality_with_reason"
                ):
                    violations.append(
                        f"{path.relative_to(repository)}:{node.lineno}: "
                        "direct _GuardedQuality construction"
                    )
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            quality = keywords.get("quality")
            if quality is None or not _is_non_exact_literal(quality):
                continue
            if call_name == "_guarded_quality_with_reason":
                continue
            if call_name in {"_with_metadata", "_with_estimate_metadata"}:
                if "quality_reason" in keywords or _call_supplies_nonempty_assumptions(
                    keywords
                ):
                    continue
            elif call_name == "ResourceEstimate" and (
                _call_supplies_nonempty_assumptions(keywords)
            ):
                continue
            violations.append(
                f"{path.relative_to(repository)}:{node.lineno}: "
                f"{call_name or '<call>'} sets non-exact quality without a reason"
            )

    assert violations == []
