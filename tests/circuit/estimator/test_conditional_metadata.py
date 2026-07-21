"""Tests for condition-aware resource provenance and call summaries."""

from __future__ import annotations

import sympy as sp

import qamomile.circuit as qmc


def test_conditional_algebra_prunes_inactive_metadata_and_calls() -> None:
    """Concrete substitution keeps metadata from only the selected branch."""
    flag = sp.Symbol("flag", integer=True, nonnegative=True)
    note = qmc.ResourceAssumption("fallback cost is modeled", source="untaken")
    taken = qmc.ResourceEstimate(
        calls=qmc.CallResources(calls_by_name={"taken": sp.Integer(1)}),
    )
    untaken = qmc.ResourceEstimate(
        calls=qmc.CallResources(calls_by_name={"untaken": sp.Integer(1)}),
        assumptions=(note,),
        quality=qmc.EstimateQuality.MODELED,
    )

    symbolic = taken.conditional(untaken, sp.Eq(flag, 1))
    selected_taken = symbolic.substitute(flag=1)
    selected_untaken = symbolic.substitute(flag=0)

    assert symbolic.quality is qmc.EstimateQuality.MODELED
    assert symbolic.assumptions == (note,)
    assert selected_taken.calls.calls_by_name == {"taken": 1}
    assert selected_taken.assumptions == ()
    assert selected_taken.quality is qmc.EstimateQuality.EXACT
    assert selected_untaken.calls.calls_by_name == {"untaken": 1}
    assert selected_untaken.assumptions == (note,)
    assert selected_untaken.quality is qmc.EstimateQuality.MODELED


def _conditional_opaque_kernel() -> qmc.QKernel:
    """Build a symbolic branch with an exact and a bodyless alternative.

    Returns:
        qmc.QKernel: Kernel selecting Hadamard or an opaque callable.
    """
    oracle = qmc.opaque("conditional_oracle", num_qubits=1)

    @qmc.qkernel
    def circuit(flag: qmc.UInt) -> qmc.Qubit:
        """Select an exact gate or a bodyless oracle."""
        target = qmc.qubit("target")
        if flag:
            target = qmc.h(target)
        else:
            (target,) = oracle(target)
        return target

    return circuit


def test_qkernel_branch_prunes_inactive_opaque_call_and_trace() -> None:
    """Selecting the exact branch removes opaque provenance completely."""
    circuit = _conditional_opaque_kernel()
    symbolic = circuit.estimate_resources(
        unknown_policy=qmc.UnknownResourcePolicy.OPAQUE_CALL,
        trace=True,
    )

    taken = symbolic.substitute(flag=1)
    untaken = symbolic.substitute(flag=0)

    assert taken.calls.calls_by_name == {}
    assert taken.quality is qmc.EstimateQuality.EXACT
    assert "conditional_oracle" not in taken.explain()
    assert untaken.calls.calls_by_name == {"conditional_oracle": 1}
    assert untaken.quality is qmc.EstimateQuality.MODELED
    assert "conditional_oracle" in untaken.explain()


def test_qkernel_branch_prunes_inactive_zero_policy_warning() -> None:
    """An untaken zero-cost fallback does not leave a warning or modeled tag."""
    circuit = _conditional_opaque_kernel()
    symbolic = circuit.estimate_resources(
        unknown_policy=qmc.UnknownResourcePolicy.ZERO_WITH_WARNING,
    )

    taken = symbolic.substitute(flag=1)
    untaken = symbolic.substitute(flag=0)

    assert taken.assumptions == ()
    assert taken.quality is qmc.EstimateQuality.EXACT
    assert len(untaken.assumptions) == 1
    assert untaken.quality is qmc.EstimateQuality.MODELED


def test_call_resources_simplify_prunes_exact_zero_entries() -> None:
    """Zero-valued call names do not remain in user-facing dictionaries."""
    calls = qmc.CallResources(
        calls_by_name={"zero": sp.Integer(0), "one": sp.Integer(1)},
        queries_by_name={"zero": sp.Integer(0)},
    ).simplify()

    assert calls.calls_by_name == {"one": 1}
    assert calls.queries_by_name == {}
