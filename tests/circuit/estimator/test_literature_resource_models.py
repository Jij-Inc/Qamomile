"""Tests for literature/asymptotic resource-policy model selection."""

from __future__ import annotations

import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator import EstimateKind, ResourcePolicy
from qamomile.circuit.ir.operation.callable import ResourceModelBinding


class _ConstModel:
    """Resource model returning a fixed total gate count.

    Args:
        total (int): Total gate count reported for every call.
    """

    def __init__(self, total: int) -> None:
        """Store the fixed total.

        Args:
            total (int): Total gate count reported for every call.
        """
        self._total = total

    def estimate(self, ctx: qm.ResourceContext) -> qm.ResourceEstimate:
        """Return the fixed gate estimate.

        Args:
            ctx (qm.ResourceContext): Call-site context (unused).

        Returns:
            qm.ResourceEstimate: Estimate with ``gates.total == total``.
        """
        return qm.ResourceEstimate(
            gates=qm.GateResources(total=sp.Integer(self._total))
        )


def _dual_model_composite() -> qm.CompositeGate:
    """Build a composite carrying asymptotic and literature models.

    Returns:
        qm.CompositeGate: Composite whose asymptotic model reports total 11 and
        whose literature model reports total 23.
    """
    from qamomile.circuit.frontend.composite_gate_wrapped import composite_gate

    @composite_gate(
        name="dual_modeled",
        resource_models=[
            ResourceModelBinding(model=_ConstModel(11), estimate_kind="asymptotic"),
            ResourceModelBinding(model=_ConstModel(23), estimate_kind="literature"),
        ],
    )
    @qm.qkernel
    def dual_modeled(q: qm.Qubit) -> qm.Qubit:
        """Apply one H gate as the fallback body."""
        return qm.h(q)

    return dual_modeled


def _call_dual(composite: qm.CompositeGate):
    """Return an estimator-ready qkernel calling the dual-model composite.

    Args:
        composite (qm.CompositeGate): Dual-model composite to call.

    Returns:
        qm.QKernel: Kernel invoking the composite once.
    """

    @qm.qkernel
    def circuit() -> qm.Qubit:
        """Call the dual-model composite once."""
        q = qm.qubit("q")
        q = composite(q)
        return q

    return circuit


def _dual_model_composite_with_default() -> qm.CompositeGate:
    """Build a dual-model composite pinning literature as the default.

    The asymptotic model is listed FIRST, but ``default_estimate_kind`` pins the
    literature model, so the default policy must not fall back to binding order.

    Returns:
        qm.CompositeGate: Composite with asymptotic (total 11) listed first,
        literature (total 23) pinned as the default.
    """
    from qamomile.circuit.frontend.composite_gate_wrapped import composite_gate

    @composite_gate(
        name="dual_pinned",
        resource_models=[
            ResourceModelBinding(model=_ConstModel(11), estimate_kind="asymptotic"),
            ResourceModelBinding(model=_ConstModel(23), estimate_kind="literature"),
        ],
        default_estimate_kind="literature",
    )
    @qm.qkernel
    def dual_pinned(q: qm.Qubit) -> qm.Qubit:
        """Apply one H gate as the fallback body."""
        return qm.h(q)

    return dual_pinned


def test_default_policy_selects_first_compatible_model() -> None:
    """MODEL_IF_AVAILABLE picks the first compatible binding (asymptotic)."""
    circuit = _call_dual(_dual_model_composite())
    est = qm.ResourceEstimator().estimate(circuit)
    assert est.gates.total == 11


def test_default_estimate_kind_overrides_binding_order() -> None:
    """An explicit default_estimate_kind pins the default regardless of order."""
    circuit = _call_dual(_dual_model_composite_with_default())
    est = qm.ResourceEstimator().estimate(circuit)
    # Literature is second in the binding list but pinned as the default.
    assert est.gates.total == 23


def test_default_estimate_kind_pin_without_matching_model_raises() -> None:
    """A pin that matches no attached model fails loudly instead of falling back.

    ``default_estimate_kind="literature"`` with only an asymptotic model attached
    is an authoring error and must raise rather than silently return the
    asymptotic estimate.
    """
    import pytest

    from qamomile.circuit.frontend.composite_gate_wrapped import composite_gate

    @composite_gate(
        name="mispinned",
        resource_models=[
            ResourceModelBinding(model=_ConstModel(11), estimate_kind="asymptotic"),
        ],
        default_estimate_kind="literature",
    )
    @qm.qkernel
    def mispinned(q: qm.Qubit) -> qm.Qubit:
        """Apply one H gate as the fallback body."""
        return qm.h(q)

    circuit = _call_dual(mispinned)
    with pytest.raises(ValueError, match="pins default_estimate_kind"):
        qm.ResourceEstimator().estimate(circuit)


def test_default_estimate_kind_rejects_unknown_tag() -> None:
    """A typo'd default_estimate_kind is rejected at definition time."""
    import pytest

    from qamomile.circuit.frontend.composite_gate_wrapped import composite_gate

    with pytest.raises(ValueError, match="not a recognized estimate kind"):

        @composite_gate(name="bad", default_estimate_kind="literture")
        @qm.qkernel
        def bad(q: qm.Qubit) -> qm.Qubit:
            """Apply one H gate."""
            return qm.h(q)


def test_literature_policy_prefers_literature_model() -> None:
    """ResourcePolicy.LITERATURE selects the literature-tagged binding."""
    circuit = _call_dual(_dual_model_composite())
    est = qm.ResourceEstimator(policy=ResourcePolicy.LITERATURE).estimate(circuit)
    assert est.gates.total == 23


def test_asymptotic_policy_prefers_asymptotic_model() -> None:
    """ResourcePolicy.ASYMPTOTIC selects the asymptotic-tagged binding."""
    circuit = _call_dual(_dual_model_composite())
    est = qm.ResourceEstimator(policy=ResourcePolicy.ASYMPTOTIC).estimate(circuit)
    assert est.gates.total == 11


def test_literature_estimate_kind_tags_trace() -> None:
    """The literature binding tags its trace node with EstimateKind.LITERATURE."""
    circuit = _call_dual(_dual_model_composite())
    est = qm.ResourceEstimator(policy=ResourcePolicy.LITERATURE).estimate(circuit)

    kinds = _trace_estimate_kinds(est.trace)
    assert EstimateKind.LITERATURE in kinds


def _trace_estimate_kinds(node) -> set:
    """Collect all estimate_kind values in a trace subtree.

    Args:
        node: Trace node root (or ``None``).

    Returns:
        set: Set of ``EstimateKind`` values found in the subtree.
    """
    if node is None:
        return set()
    found = set()
    kind = getattr(node, "estimate_kind", None)
    if kind is not None:
        found.add(kind)
    for child in getattr(node, "children", ()):
        found |= _trace_estimate_kinds(child)
    return found
