"""Input-domain-aware simplification of symbolic resource estimates."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable
from copy import deepcopy
from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit as qm
import qamomile.circuit.estimator._domain_affine as domain_affine
import qamomile.circuit.estimator._estimate_composition as composition_module
from qamomile.circuit.estimator import (
    ResourceTraceNode,
    _estimate_domain as estimate_domain,
    _parameter_domain as parameter_domain,
)
from qamomile.circuit.estimator._parameter_domain import (
    _ConsumedDomainRequirement,
    _rewrite_expression_over_domain,
)
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintOrigin,
    _ConstraintProvenance,
    _ConstraintRange,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._wire import (
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)


@qm.qkernel
def _ghz_like(length: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Prepare a serial GHZ state.

    Args:
        length (qm.UInt): Number of qubits to prepare.

    Returns:
        qm.Vector[qm.Qubit]: Prepared register.
    """
    register = qm.qubit_array(length, "register")
    register[0] = qm.h(register[0])
    for index in qm.range(length - 1):
        register[index], register[index + 1] = qm.cx(
            register[index],
            register[index + 1],
        )
    return register


@qm.qkernel
def _guarded_ghz_like(length: qm.UInt) -> qm.Vector[qm.Qubit]:
    """Prepare a GHZ state only for a nonempty register.

    Args:
        length (qm.UInt): Number of qubits to prepare.

    Returns:
        qm.Vector[qm.Qubit]: Prepared or empty register.
    """
    register = qm.qubit_array(length, "register")
    if length > 0:
        register[0] = qm.h(register[0])
        for index in qm.range(length - 1):
            register[index], register[index + 1] = qm.cx(
                register[index],
                register[index + 1],
            )
    return register


@qm.qkernel
def _domain_only(length: qm.UInt) -> qm.Qubit:
    """Expose an unguarded root-input access without a gate cost.

    Args:
        length (qm.UInt): Size of the allocated register.

    Returns:
        qm.Qubit: First register element.
    """
    register = qm.qubit_array(length, "register")
    return register[0]


@qm.qkernel
def _second_element_metric(length: qm.UInt) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Create a metric reducible only when the second element exists.

    Args:
        length (qm.UInt): Size of the allocated register.

    Returns:
        tuple[qm.Vector[qm.Qubit], qm.Qubit]: Register and loop workspace.
    """
    register = qm.qubit_array(length, "register")
    register[1] = qm.h(register[1])
    workspace = qm.qubit("workspace")
    for _ in qm.range(length - 2):
        workspace = qm.x(workspace)
    return register, workspace


@qm.qkernel
def _indexed_metric(
    length: qm.UInt,
    selected: qm.UInt,
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Create a metric reducible on a relational index domain.

    Args:
        length (qm.UInt): Size of the allocated register.
        selected (qm.UInt): Register element to access.

    Returns:
        tuple[qm.Vector[qm.Qubit], qm.Qubit]: Register and loop workspace.
    """
    register = qm.qubit_array(length, "register")
    register[selected] = qm.h(register[selected])
    workspace = qm.qubit("workspace")
    for _ in qm.range(length - selected - 1):
        workspace = qm.x(workspace)
    return register, workspace


@qm.composite_gate(name="domain_seed_first")
def _seed_first(register: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
    """Apply one gate through a callee-local first-element access.

    Args:
        register (qm.Vector[qm.Qubit]): Register whose first element is used.

    Returns:
        qm.Vector[qm.Qubit]: Updated register.
    """
    register[0] = qm.h(register[0])
    return register


@qm.qkernel
def _helper_mapped_metric(
    length: qm.UInt,
) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
    """Map a helper access requirement back to a root formal.

    Args:
        length (qm.UInt): Size of the allocated register.

    Returns:
        tuple[qm.Vector[qm.Qubit], qm.Qubit]: Register and loop workspace.
    """
    register = qm.qubit_array(length, "register")
    register = _seed_first(register)
    workspace = qm.qubit("workspace")
    for _ in qm.range(length - 1):
        workspace = qm.x(workspace)
    return register, workspace


@qm.qkernel
def _view_metric(
    length: qm.UInt,
) -> tuple[qm.Vector[qm.Bit], qm.Qubit]:
    """Create a reducible metric beside a domain-constraining view.

    Args:
        length (qm.UInt): Size of the allocated register.

    Returns:
        tuple[qm.Vector[qm.Bit], qm.Qubit]: View measurements and workspace.
    """
    register = qm.qubit_array(length, "register")
    measurements = qm.measure(register[1:length])
    workspace = qm.h(qm.qubit("workspace"))
    for _ in qm.range(length - 1):
        workspace = qm.x(workspace)
    return measurements, workspace


@qm.qkernel
def _matrix_domain_metric(
    values: qm.Matrix[qm.Float],
    row: qm.UInt,
    column: qm.UInt,
) -> qm.Qubit:
    """Use both matrix-axis bounds beside a row-dependent metric.

    Args:
        values (qm.Matrix[qm.Float]): Gate angles and symbolic matrix shape.
        row (qm.UInt): Selected matrix row.
        column (qm.UInt): Selected matrix column.

    Returns:
        qm.Qubit: Target carrying the selected rotation and loop gates.
    """
    target = qm.rx(qm.qubit("target"), values[row, column])
    for _ in qm.range(values.shape[0] - row - 1):
        target = qm.x(target)
    return target


def _domain_assumptions(
    estimate: qm.ResourceEstimate,
) -> tuple[qm.ResourceAssumption, ...]:
    """Return assumptions generated by the qkernel input-domain pass.

    Args:
        estimate (qm.ResourceEstimate): Estimate to inspect.

    Returns:
        tuple[qm.ResourceAssumption, ...]: Domain-specific assumptions.
    """
    return tuple(
        assumption
        for assumption in estimate.assumptions
        if assumption.source == "qkernel input domain"
    )


def _public_expressions(estimate: qm.ResourceEstimate) -> tuple[sp.Expr, ...]:
    """Collect every scalar public resource expression.

    Args:
        estimate (qm.ResourceEstimate): Estimate whose fields are collected.

    Returns:
        tuple[sp.Expr, ...]: Public scalar expressions in stable field order.
    """
    return (
        estimate.width.input_qubits,
        estimate.width.allocated_qubits,
        estimate.width.clean_ancilla_qubits,
        estimate.width.dirty_ancilla_qubits,
        estimate.width.peak_qubits,
        estimate.gates.total,
        estimate.gates.single_qubit,
        estimate.gates.two_qubit,
        estimate.gates.multi_qubit,
        estimate.gates.clifford,
        estimate.gates.rotation,
        estimate.gates.t,
        estimate.gates.toffoli,
        estimate.gates.non_clifford,
        estimate.measurements.total,
        estimate.resets.total,
        estimate.depth.depth,
        estimate.depth.clifford_depth,
        estimate.depth.rotation_depth,
        estimate.depth.t_depth,
        estimate.depth.toffoli_depth,
        estimate.depth.non_clifford_depth,
        estimate.depth.measurement_depth,
        estimate.depth.gate_depth,
        estimate.depth.reset_depth,
        *estimate.calls.calls_by_name.values(),
        *estimate.calls.queries_by_name.values(),
    )


def _expression_model(symbol: sp.Symbol, expression: sp.Expr) -> qm.ResourceEstimate:
    """Create a hand-built estimate carrying one expression in every field.

    Args:
        symbol (sp.Symbol): Root symbol used only to retain a stable alias.
        expression (sp.Expr): Expression copied into all public metric groups.

    Returns:
        qm.ResourceEstimate: Inherited-policy estimate with public expressions.
    """
    return qm.ResourceEstimate(
        width=qm.WidthResources(
            input_qubits=expression,
            allocated_qubits=expression,
            clean_ancilla_qubits=expression,
            dirty_ancilla_qubits=expression,
            peak_qubits=expression,
        ),
        gates=qm.GateResources(
            total=expression,
            single_qubit=expression,
            two_qubit=expression,
            multi_qubit=expression,
            clifford=expression,
            rotation=expression,
            t=expression,
            toffoli=expression,
            non_clifford=expression,
        ),
        measurements=qm.MeasurementResources(total=expression),
        resets=qm.ResetResources(total=expression),
        depth=qm.DepthResources(
            depth=expression,
            clifford_depth=expression,
            rotation_depth=expression,
            t_depth=expression,
            toffoli_depth=expression,
            non_clifford_depth=expression,
            measurement_depth=expression,
            gate_depth=expression,
            reset_depth=expression,
        ),
        calls=qm.CallResources(
            calls_by_name={"modeled": expression},
            queries_by_name={"modeled": expression},
        ),
        parameters={str(symbol): symbol},
    )


def _eligible_constraint(
    expression: sp.Expr,
    *,
    minimum: int | None = 0,
    expected: sp.Expr | None = None,
    label: str = "test input domain",
) -> _ResourceConstraint:
    """Build one explicitly trusted root-input constraint for prover tests.

    Args:
        expression (sp.Expr): Integer expression being constrained.
        minimum (int | None): Inclusive lower bound. Defaults to zero.
        expected (sp.Expr | None): Optional exact value. Defaults to ``None``.
        label (str): Diagnostic evidence label.

    Returns:
        _ResourceConstraint: Constraint with explicit array-access provenance.
    """
    return _ResourceConstraint(
        expression=expression,
        minimum=minimum,
        expected=expected,
        label=label,
        provenance=_ConstraintProvenance(
            origin=_ConstraintOrigin.ARRAY_ACCESS,
            source_expressions=(expression,),
            root_formal_names=("length",),
        ),
    )


def test_ghz_uses_its_unguarded_access_domain_for_exact_symbolic_counts() -> None:
    """GHZ Max expressions reduce to exact formulas on the valid domain."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]

    assert estimate.gates.total == length
    assert estimate.gates.two_qubit == length - 1
    assert estimate.depth.depth == length
    assert estimate.depth.gate_depth == length
    assert estimate.quality is qm.EstimateQuality.EXACT
    (domain,) = _domain_assumptions(estimate)
    assert "length" in domain.message
    assert "1" in domain.message
    assert estimate.to_dict()["assumptions"] == [
        {"message": domain.message, "source": domain.source}
    ]
    assert estimate.explain() == "Resource estimate"


@pytest.mark.parametrize("length", [1, 2, 3, 7])
def test_ghz_direct_and_late_valid_specialization_agree(length: int) -> None:
    """Valid direct and late GHZ specialization discharge the domain."""
    direct = _ghz_like.estimate_resources(inputs={"length": length})
    late = _ghz_like.estimate_resources().substitute(length=length)

    assert direct.gates == late.gates
    assert direct.depth == late.depth
    assert direct.gates.total == length
    assert direct.depth.depth == length
    assert _domain_assumptions(direct) == ()
    assert _domain_assumptions(late) == ()


def test_ghz_zero_is_invalid_through_direct_and_late_paths() -> None:
    """An unguarded first-element access rejects zero-length GHZ inputs."""
    with pytest.raises(ValueError, match="in-bounds"):
        _ghz_like.estimate_resources(inputs={"length": 0})
    with pytest.raises(ValueError, match="in-bounds"):
        _ghz_like.estimate_resources().substitute(length=0)


def test_guarded_zero_width_kernel_has_zero_resources() -> None:
    """A guarded first-element access makes the empty input well-defined."""
    symbolic = _guarded_ghz_like.estimate_resources()
    length = symbolic.parameters["length"]
    direct = _guarded_ghz_like.estimate_resources(inputs={"length": 0})

    assert symbolic.gates.total == sp.Piecewise(
        (1 + sp.Max(0, length - 1), length > 0),
        (0, True),
    )
    assert _domain_assumptions(symbolic) == ()
    assert direct.width.circuit_qubits == 0
    assert direct.gates.total == 0
    assert direct.depth.depth == 0
    assert _domain_assumptions(direct) == ()


def test_constant_metric_does_not_publish_an_unused_domain() -> None:
    """A structural requirement alone does not become an assumption."""
    estimate = _domain_only.estimate_resources()

    assert estimate.gates.total == 0
    assert _domain_assumptions(estimate) == ()
    with pytest.raises(ValueError, match="in-bounds"):
        estimate.substitute(length=0)


def test_second_element_domain_rewrites_metric_and_validates_both_paths() -> None:
    """A q[1] access proves the reducible metric only for length >= 2."""
    estimate = _second_element_metric.estimate_resources()
    length = estimate.parameters["length"]

    assert estimate.gates.total == length - 1
    assert len(_domain_assumptions(estimate)) == 1
    assert (
        _second_element_metric.estimate_resources(inputs={"length": 2}).gates.total == 1
    )
    assert estimate.substitute(length=4).gates.total == 3
    with pytest.raises(ValueError, match="in-bounds"):
        _second_element_metric.estimate_resources(inputs={"length": 1})
    with pytest.raises(ValueError, match="in-bounds"):
        estimate.substitute(length=1)


def test_indexed_domain_rewrites_relational_metric() -> None:
    """A q[k] access retains the relation length >= selected + 1."""
    estimate = _indexed_metric.estimate_resources()
    length = estimate.parameters["length"]
    selected = estimate.parameters["selected"]

    assert estimate.gates.total == length - selected
    assert len(_domain_assumptions(estimate)) == 1
    assert estimate._domain_rewrite_state is not None
    assert estimate._domain_rewrite_state.requirements[0].source_formals == (
        "length",
        "selected",
    )
    partial = estimate.substitute(selected=1)
    direct_partial = _indexed_metric.estimate_resources(inputs={"selected": 1})
    partial_length = partial.parameters["length"]
    assert partial.gates.total == partial_length - 1
    assert direct_partial.gates == partial.gates
    assert len(_domain_assumptions(partial)) == 1
    assert estimate.substitute(length=5, selected=2).gates.total == 3
    with pytest.raises(ValueError, match="in-bounds"):
        estimate.substitute(length=2, selected=2)
    assert estimate.substitute(length=5, selected=2).gates.total == 3


def test_helper_formal_access_maps_to_the_root_input_domain() -> None:
    """A callee-local q[0] requirement proves a caller metric rewrite."""
    estimate = _helper_mapped_metric.estimate_resources()
    length = estimate.parameters["length"]

    assert estimate.gates.total == length
    assert len(_domain_assumptions(estimate)) == 1
    assert estimate.substitute(length=3).gates.total == 3
    with pytest.raises(ValueError, match="in-bounds"):
        estimate.substitute(length=0)


def test_view_requirement_rewrites_a_neighboring_symbolic_metric() -> None:
    """A mapped view domain removes a neighboring valid-domain Max wrapper."""
    estimate = _view_metric.estimate_resources()
    length = estimate.parameters["length"]

    assert estimate.gates.total == length
    assert estimate.measurements.total == length - 1
    assert len(_domain_assumptions(estimate)) == 1
    assert estimate.substitute(length=1).gates.total == 1
    assert estimate.substitute(length=1).measurements.total == 0
    with pytest.raises(ValueError, match="view.*length|coverage|in-bounds"):
        estimate.substitute(length=0)


def test_multidimensional_access_projects_each_axis_without_overclaiming() -> None:
    """A matrix access uses row proof for rewriting and retains column validation."""
    estimate = _matrix_domain_metric.estimate_resources()
    row = estimate.parameters["row"]
    row_count = estimate.parameters["values_dim0"]

    assert estimate.gates.total == row_count - row
    assert len(_domain_assumptions(estimate)) == 1
    valid_inputs = {
        "values_dim0": 2,
        "values_dim1": 3,
        "row": 1,
        "column": 2,
    }
    assert estimate.substitute(**valid_inputs).gates.total == 1
    with pytest.raises(ValueError, match="axis 1 in-bounds"):
        estimate.substitute(**{**valid_inputs, "column": 3})


def test_symbolic_input_mapping_preserves_source_formal_provenance() -> None:
    """Mapping length to m + 1 retains and rerenders its domain condition."""
    mapped_symbol = sp.Symbol("m", integer=True)
    estimate = _ghz_like.estimate_resources(inputs={"length": mapped_symbol + 1})

    assert estimate.gates.total == mapped_symbol + 1
    assert set(estimate.parameters) == {"m"}
    (domain,) = _domain_assumptions(estimate)
    assert "m" in domain.message
    assert estimate.substitute(m=0).gates.total == 1
    with pytest.raises(ValueError, match="in-bounds"):
        estimate.substitute(m=-1)


def test_equal_ordinary_assumption_survives_domain_discharge() -> None:
    """Value-equal user metadata is not mistaken for a rendered domain view."""
    estimate = _ghz_like.estimate_resources()
    generated = _domain_assumptions(estimate)[0]
    ordinary = qm.ResourceAssumption(generated.message, generated.source)

    augmented = dataclasses.replace(
        estimate,
        assumptions=(*estimate.assumptions, ordinary),
    )
    specialized = augmented.substitute(length=2)

    assert specialized.assumptions == (ordinary,)
    assert specialized.assumptions[0] is ordinary


def test_reused_domain_assumption_object_appended_as_ordinary_survives() -> None:
    """A reused rendered object is ordinary when appended after its snapshot."""
    estimate = _ghz_like.estimate_resources()
    generated = estimate.assumptions[0]

    augmented = dataclasses.replace(
        estimate,
        assumptions=(*estimate.assumptions, generated),
    )
    specialized = augmented.substitute(length=2)

    assert specialized.assumptions == (generated,)
    assert specialized.assumptions[0] is generated


def test_nonprefix_assumption_edit_is_rejected_immediately() -> None:
    """Rendered domain assumptions cannot be inserted, removed, or reordered."""
    estimate = _ghz_like.estimate_resources()
    ordinary = qm.ResourceAssumption("ordinary", "test")

    with pytest.raises(RuntimeError, match="domain snapshot prefix"):
        dataclasses.replace(
            estimate,
            assumptions=(ordinary, *estimate.assumptions),
        )


def test_ordinary_assumptions_retain_normal_dataclass_replace_semantics() -> None:
    """Estimates without domain evidence may still replace assumption tuples."""
    first = qm.ResourceAssumption("first", "test")
    second = qm.ResourceAssumption("second", "test")
    estimate = qm.ResourceEstimate(assumptions=(first,))

    replaced = dataclasses.replace(estimate, assumptions=(second, first))
    cleared = dataclasses.replace(estimate, assumptions=())

    assert replaced.assumptions == (first, second)
    assert cleared.assumptions == (first,)


def test_domain_from_one_operand_rewrites_every_public_metric_group() -> None:
    """An enabled domain source rewrites all public metric fields after seq."""
    source = _domain_only.estimate_resources()
    length = source.parameters["length"]
    original = sp.Max(0, length - 1)

    combined = source.seq(_expression_model(length, original))

    assert all(
        not expression.has(sp.Max) for expression in _public_expressions(combined)
    )
    assert combined.gates.total == length - 1
    assert combined.measurements.total == length - 1
    assert combined.resets.total == length - 1
    assert combined.depth.reset_depth == length - 1
    assert combined.calls.calls_by_name == {"modeled": length - 1}
    assert combined.calls.queries_by_name == {"modeled": length - 1}
    assert len(_domain_assumptions(combined)) == 1


def test_ordered_piecewise_and_nary_extrema_preserve_semantics() -> None:
    """Overlapping branches and n-ary extrema rewrite without reordering."""
    source = _domain_only.estimate_resources()
    length = source.parameters["length"]
    piecewise = sp.Piecewise(
        (sp.Max(0, length - 1), length > 2),
        (sp.Max(0, length - 2), length > 1),
        (sp.Max(0, length - 3), True),
    )
    model = _expression_model(length, piecewise)
    model.gates.single_qubit = sp.Max(0, length - 1, length - 2)
    model.gates.two_qubit = sp.Min(length, length + 1, length + 2)

    rewritten = source.seq(model)

    assert not rewritten.gates.total.has(sp.Max)
    assert rewritten.gates.single_qubit == length - 1
    assert rewritten.gates.two_qubit == length
    for concrete in range(1, 7):
        expected = piecewise.subs(length, concrete)
        assert rewritten.substitute(length=concrete).gates.total == expected


@pytest.mark.parametrize(
    "node_kind",
    ["sum", "product", "integral", "derivative", "limit", "lambda", "custom"],
)
def test_leaf_rewriter_treats_protected_nodes_as_atomic(node_kind: str) -> None:
    """Binders, calculus objects, and custom functions fail closed as a whole."""
    length = sp.Symbol("length", integer=True)
    index = sp.Dummy("index", integer=True)
    original = sp.Max(0, length - 1)
    expressions = {
        "sum": sp.Sum(original, (index, 0, 1)),
        "product": sp.Product(original, (index, 0, 1)),
        "integral": sp.Integral(original, (index, 0, 1)),
        "derivative": sp.Derivative(original, length, evaluate=False),
        "limit": sp.Limit(original, index, 0),
        "lambda": sp.Lambda(index, original),
        "custom": sp.Function("custom_metric")(original),
    }
    protected = expressions[node_kind]

    rewritten, requirements = _rewrite_expression_over_domain(
        protected,
        (_eligible_constraint(length - 1),),
    )

    assert rewritten == protected
    assert requirements == ()


def test_leaf_rewriter_rejects_nonlinear_target_without_symbolic_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compact nonlinear target fails closed without expand or Poly calls."""
    symbols = sp.symbols("x0:18", integer=True)
    nonlinear = sp.Mul(*(symbol + 1 for symbol in symbols), evaluate=False)
    original = sp.Max(0, nonlinear, evaluate=False)
    constraint = _eligible_constraint(symbols[0])
    expand = Mock(side_effect=AssertionError("unexpected symbolic expansion"))
    polynomial = Mock(side_effect=AssertionError("unexpected polynomial conversion"))
    monkeypatch.setattr(sp, "expand", expand)
    monkeypatch.setattr(sp, "Poly", polynomial)

    rewritten, requirements = _rewrite_expression_over_domain(
        original,
        (constraint,),
    )

    assert requirements == ()
    expand.assert_not_called()
    polynomial.assert_not_called()
    values = {symbol: index % 2 for index, symbol in enumerate(symbols)}
    assert rewritten.subs(values) == original.subs(values)


def test_leaf_rewriter_rejects_nonlinear_source_without_polynomial_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonlinear source is validation-only without constructing a Poly."""
    symbols = sp.symbols("x0:18", integer=True)
    nonlinear = sp.Mul(*(symbol + 1 for symbol in symbols), evaluate=False)
    original = sp.Max(0, symbols[0] - 1)
    constraint = _eligible_constraint(nonlinear)
    polynomial = Mock(side_effect=AssertionError("unexpected polynomial conversion"))
    monkeypatch.setattr(sp, "Poly", polynomial)

    rewritten, requirements = _rewrite_expression_over_domain(
        original,
        (constraint,),
    )

    assert rewritten == original
    assert requirements == ()
    polynomial.assert_not_called()


def test_affine_extractor_uses_the_current_hard_node_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkeypatched affine budgets fail closed before any domain proof."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)
    monkeypatch.setattr(domain_affine, "_DOMAIN_AFFINE_NODE_LIMIT", 0)

    rewritten, requirements = _rewrite_expression_over_domain(
        original,
        (_eligible_constraint(length - 1),),
    )

    assert rewritten == original
    assert requirements == ()


def test_nonlinear_source_checks_affinity_before_global_type_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized nonlinear sources fail before expensive SymPy type queries."""
    symbols = sp.symbols("x0:600", integer=True)
    nonlinear = sp.Mul(*(symbol + 1 for symbol in symbols), evaluate=False)
    constraint = _eligible_constraint(nonlinear)
    finite = Mock(side_effect=AssertionError("unexpected global type query"))
    integer = Mock(side_effect=AssertionError("unexpected global type query"))
    monkeypatch.setattr(type(nonlinear), "is_finite", property(lambda _self: finite()))
    monkeypatch.setattr(
        type(nonlinear),
        "is_integer",
        property(lambda _self: integer()),
    )

    assert constraint.domain_eligible is False
    finite.assert_not_called()
    integer.assert_not_called()


def test_same_named_distinct_symbol_is_not_proved_by_root_domain() -> None:
    """Domain facts match SymPy identity rather than a display name."""
    source = _domain_only.estimate_resources()
    other_length = sp.Symbol("length", integer=True)
    model = _expression_model(other_length, sp.Max(0, other_length - 1))

    rewritten = source.seq(model)

    assert rewritten.gates.total.has(sp.Max)
    assert len(rewritten.parameters) == 2


def test_hand_built_constraint_remains_validation_only() -> None:
    """A constraint without explicit root provenance cannot seed a rewrite."""
    length = sp.Symbol("length", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=1 + sp.Max(0, length - 1)),
        _constraints=(
            _ResourceConstraint(
                expression=length,
                minimum=1,
                label="hand-built lower bound",
            ),
        ),
    ).simplify()

    assert estimate.gates.total == 1 + sp.Max(0, length - 1)
    assert _domain_assumptions(estimate) == ()
    with pytest.raises(ValueError, match="hand-built lower bound"):
        estimate.substitute(length=0)


def test_leaf_rewriter_returns_exact_evidence_for_a_supported_rewrite() -> None:
    """A successful leaf rewrite reports only the predicate it consumed."""
    length = sp.Symbol("length", integer=True)
    unused = _eligible_constraint(length, label="unsigned input")
    constraint = _eligible_constraint(length - 1)

    rewritten, requirements = _rewrite_expression_over_domain(
        1 + sp.Max(0, length - 1),
        (unused, constraint),
    )
    reproved, repeated_requirements = _rewrite_expression_over_domain(
        1 + sp.Max(0, length - 1),
        (constraint,),
    )

    assert rewritten == reproved == length
    assert len(requirements) == 1
    assert requirements[0].predicate == sp.Ge(length - 1, 0)
    assert requirements[0].source_formals == ("length",)
    assert requirements[0].labels == ("test input domain",)
    assert repeated_requirements == requirements


def test_leaf_rewriter_rejects_a_domain_too_weak_for_the_target() -> None:
    """A weaker lower bound leaves the original Max and consumes nothing."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)

    rewritten, requirements = _rewrite_expression_over_domain(
        original,
        (_eligible_constraint(length),),
    )

    assert rewritten == original
    assert requirements == ()


def test_leaf_rewriter_rejects_contradictory_sources() -> None:
    """An inconsistent projected domain cannot justify ex-falso rewriting."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)
    constraints = (
        _eligible_constraint(length - 1, label="length >= 1"),
        _eligible_constraint(-length, label="length <= 0"),
    )

    rewritten, requirements = _rewrite_expression_over_domain(original, constraints)

    assert rewritten == original
    assert requirements == ()


def test_leaf_rewriter_rejects_contradictory_equalities() -> None:
    """Distinct exact values for one affine input fail closed atomically."""
    length = sp.Symbol("length", integer=True)
    original = sp.Piecewise((3, sp.Eq(length, 1)), (5, True))
    constraints = (
        _eligible_constraint(length, minimum=None, expected=sp.Integer(1)),
        _eligible_constraint(length, minimum=None, expected=sp.Integer(2)),
    )

    rewritten, requirements = _rewrite_expression_over_domain(original, constraints)

    assert rewritten == original
    assert requirements == ()


def test_leaf_rewriter_accepts_consistent_scaled_equalities() -> None:
    """Equivalent affine equalities retain a valid exact branch rewrite."""
    length = sp.Symbol("length", integer=True)
    original = sp.Piecewise((3, sp.Eq(length, 1)), (5, True))
    constraints = (
        _eligible_constraint(length, minimum=None, expected=sp.Integer(1)),
        _eligible_constraint(2 * length, minimum=None, expected=sp.Integer(2)),
        _eligible_constraint(1 - length, minimum=None, expected=sp.Integer(0)),
    )

    rewritten, requirements = _rewrite_expression_over_domain(original, constraints)

    assert rewritten == 3
    assert tuple(requirement.predicate for requirement in requirements) == (
        sp.Eq(length, 1),
    )


@pytest.mark.parametrize(
    "unsupported_kind",
    [
        "guard",
        "runtime outcome",
        "range",
        "equality range",
        "nested range",
        "nonlinear range",
        "dummy",
        "model contract",
        "internal origin",
    ],
)
def test_leaf_rewriter_fails_closed_for_unsupported_constraint_state(
    unsupported_kind: str,
) -> None:
    """Guarded, quantified, opaque, and internal facts never seed the pass."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)
    constraint = _eligible_constraint(length - 1)
    if unsupported_kind in {"guard", "runtime outcome"}:
        flag = sp.Symbol("flag", integer=True)
        constraint = dataclasses.replace(
            constraint,
            active_when=sp.Eq(flag, 1),
        )
    elif unsupported_kind in {
        "range",
        "equality range",
        "nested range",
        "nonlinear range",
    }:
        index = sp.Dummy("index", integer=True)
        start = length**2 if unsupported_kind == "nonlinear range" else 0
        constraint = dataclasses.replace(
            constraint,
            expected=(sp.Integer(0) if unsupported_kind == "equality range" else None),
            minimum=(None if unsupported_kind == "equality range" else 0),
            ranges=(
                _ConstraintRange(index, start, 1, length),
                *(
                    (_ConstraintRange(sp.Dummy("nested"), 0, 1, length),)
                    if unsupported_kind == "nested range"
                    else ()
                ),
            ),
        )
    elif unsupported_kind == "dummy":
        internal = sp.Dummy("internal", integer=True)
        constraint = _eligible_constraint(internal - 1)
    else:
        origin = {
            "model contract": _ConstraintOrigin.MODEL_CONTRACT,
            "internal origin": _ConstraintOrigin.INTERNAL,
        }[unsupported_kind]
        constraint = dataclasses.replace(
            constraint,
            provenance=dataclasses.replace(constraint.provenance, origin=origin),
        )

    rewritten, requirements = _rewrite_expression_over_domain(
        original,
        (constraint,),
    )

    assert rewritten == original
    assert requirements == ()


def test_leaf_rewriter_source_budget_exhaustion_is_atomic() -> None:
    """Too many distinct sources return the original formula with no evidence."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)
    constraints = tuple(
        _eligible_constraint(length - offset, label=f"source {offset}")
        for offset in range(129)
    )

    rewritten, requirements = _rewrite_expression_over_domain(original, constraints)

    assert rewritten == original
    assert requirements == ()


@pytest.mark.parametrize(
    "limit_name",
    [
        "_DOMAIN_RAW_CONSTRAINT_LIMIT",
        "_DOMAIN_SYMBOL_LIMIT",
        "_DOMAIN_EXPRESSION_NODE_LIMIT",
        "_DOMAIN_PREDICATE_NODE_LIMIT",
        "_DOMAIN_COMPARISON_LIMIT",
        "_DOMAIN_PROOF_CACHE_LIMIT",
    ],
)
def test_leaf_rewriter_budget_exhaustion_never_returns_partial_evidence(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    """Every bounded prover limit falls back to the untouched expression."""
    length = sp.Symbol("length", integer=True)
    original = 1 + sp.Max(0, length - 1)
    constraints = (
        _eligible_constraint(length - 1, label="required"),
        _eligible_constraint(length, label="unused"),
    )
    monkeypatch.setattr(parameter_domain, limit_name, 0)

    rewritten, requirements = _rewrite_expression_over_domain(original, constraints)

    assert rewritten == original
    assert requirements == ()


def test_simplify_false_preserves_original_formula_and_policy() -> None:
    """Estimator-level simplify=False does not create domain rewrite state."""
    estimate = qm.ResourceEstimator(simplify=False).estimate(_ghz_like)
    length = estimate.parameters["length"]

    assert estimate.gates.total == 1 + sp.Max(0, length - 1)
    assert _domain_assumptions(estimate) == ()
    assert estimate.substitute(length=2).gates.total == 2
    assert _domain_assumptions(estimate.substitute(length=2)) == ()


def test_explicit_simplify_enables_a_previously_disabled_estimate() -> None:
    """Only explicit simplify enables domain rewriting after simplify=False."""
    disabled = qm.ResourceEstimator(simplify=False).estimate(_ghz_like)
    length = disabled.parameters["length"]

    enabled = disabled.simplify()
    simplified_again = enabled.simplify()

    assert enabled.gates.total == length
    assert len(_domain_assumptions(enabled)) == 1
    assert simplified_again.gates == enabled.gates
    assert simplified_again.depth == enabled.depth
    assert simplified_again.assumptions == enabled.assumptions


@pytest.mark.parametrize("zero_on_left", [False, True])
def test_inherited_zero_composes_with_enabled_policy(zero_on_left: bool) -> None:
    """An inherited zero does not disable an enabled domain pass."""
    enabled = _ghz_like.estimate_resources()
    zero = qm.ResourceEstimate.zero()

    combined = zero.seq(enabled) if zero_on_left else enabled.seq(zero)

    assert combined.gates.total == enabled.gates.total
    assert len(_domain_assumptions(combined)) == 1


@pytest.mark.parametrize("disabled_on_left", [False, True])
def test_disabled_operand_prevents_binary_domain_rewrite(
    disabled_on_left: bool,
) -> None:
    """Either operand's disabled policy makes binary composition fail closed."""
    enabled = _ghz_like.estimate_resources()
    disabled = qm.ResourceEstimator(simplify=False).estimate(_ghz_like)
    length = enabled.parameters["length"]

    combined = disabled.seq(enabled) if disabled_on_left else enabled.seq(disabled)

    assert combined.gates.total == 2 + 2 * sp.Max(0, length - 1)
    assert _domain_assumptions(combined) == ()


def test_inherited_operands_do_not_start_domain_rewriting_without_a_root() -> None:
    """Two inherited operands retain formulas until public simplify enables them."""
    length = sp.Symbol("length", integer=True)
    original = sp.Max(0, length - 1)
    inherited = dataclasses.replace(
        _expression_model(length, original),
        _constraints=(_eligible_constraint(length - 1),),
    )

    combined = inherited.seq(qm.ResourceEstimate.zero())
    enabled = combined.simplify()

    assert combined.gates.total == original
    assert _domain_assumptions(combined) == ()
    assert enabled.gates.total == length - 1
    assert len(_domain_assumptions(enabled)) == 1


def test_simplify_then_seq_parallel_choice_and_inverse_restore_snapshot() -> None:
    """Public algebra restores original formulas before proving a fresh result."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]

    sequential = estimate.seq(estimate)
    parallel = estimate.parallel(estimate)
    choice = estimate.choice(estimate)
    inverse = estimate.inverse()

    assert sequential.gates.total == 2 * length
    assert parallel.gates.total == 2 * length
    assert choice.gates.total == length
    assert inverse.gates.total == length
    assert all(
        _domain_assumptions(item) for item in (sequential, parallel, choice, inverse)
    )


def test_repeat_zero_one_and_symbolic_recompute_domain_requirements() -> None:
    """Repeat drops vacuous domains and fails closed for a symbolic guard."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]
    repeats = sp.Symbol("repeats", integer=True, nonnegative=True)

    never = estimate.repeat(0)
    once = estimate.repeat(1)
    symbolic = estimate.repeat(repeats)

    assert never.gates.total == 0
    assert _domain_assumptions(never) == ()
    assert once.gates.total == length
    assert len(_domain_assumptions(once)) == 1
    assert symbolic.gates.total == repeats * (1 + sp.Max(0, length - 1))
    assert _domain_assumptions(symbolic) == ()


def test_conditional_rewrites_only_after_its_guard_is_concrete() -> None:
    """A symbolic branch guards its constraint, while concrete paths re-prove."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]
    selected = sp.Symbol("selected", integer=True)

    symbolic = estimate.conditional(qm.ResourceEstimate.zero(), sp.Eq(selected, 1))
    active = estimate.conditional(qm.ResourceEstimate.zero(), sp.true)
    inactive = estimate.conditional(qm.ResourceEstimate.zero(), sp.false)

    assert symbolic.gates.total == sp.Piecewise(
        (1 + sp.Max(0, length - 1), sp.Eq(selected, 1)),
        (0, True),
    )
    assert _domain_assumptions(symbolic) == ()
    assert active.gates.total == length
    assert len(_domain_assumptions(active)) == 1
    assert inactive.gates.total == 0
    assert _domain_assumptions(inactive) == ()


def test_sum_empty_concrete_and_symbolic_recompute_domain_requirements() -> None:
    """Range sums drop vacuous facts and retain only unguarded active domains."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]
    index = sp.Dummy("iteration", integer=True, nonnegative=True)
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)

    empty = estimate.sum_over(index, 0, 0)
    nonempty = estimate.sum_over(index, 0, 2)
    symbolic = estimate.sum_over(index, 0, iterations)

    assert empty.gates.total == 0
    assert _domain_assumptions(empty) == ()
    assert nonempty.gates.total == 2 * length
    assert len(_domain_assumptions(nonempty)) == 1
    assert symbolic.gates.total == iterations * (1 + sp.Max(0, length - 1))
    assert _domain_assumptions(symbolic) == ()


def test_control_and_inverse_revalidate_domain_after_snapshot_restore() -> None:
    """Unary transforms keep a domain only when the transformed formula uses it."""
    estimate = _ghz_like.estimate_resources()

    inverse = estimate.inverse()
    controlled = estimate.controlled(1)

    assert not inverse.gates.total.has(sp.Max)
    assert len(_domain_assumptions(inverse)) == 1
    assert not controlled.gates.total.has(sp.Max)
    assert len(_domain_assumptions(controlled)) == 1


def test_dataclasses_replace_rejects_stale_domain_metrics_immediately() -> None:
    """Replacing rewritten metrics cannot retain the prior domain proof."""
    estimate = _ghz_like.estimate_resources()

    with pytest.raises(RuntimeError, match="domain.*rewrite.*visible metrics"):
        dataclasses.replace(estimate, gates=qm.GateResources(total=999))


@pytest.mark.parametrize(
    ("mutation_kind", "entrypoint"),
    [
        ("field", "report"),
        ("nested", "report"),
        ("assumptions", "report"),
        ("parameters", "report"),
        ("field", "simplify"),
        ("field", "substitute"),
        ("field", "seq"),
        ("field", "seq_all"),
        ("field", "parallel"),
        ("field", "choice"),
        ("field", "conditional"),
        ("field", "repeat"),
        ("field", "sum"),
        ("field", "control"),
        ("field", "inverse"),
        ("field", "wire"),
        ("field", "physical"),
        ("field", "explain"),
    ],
)
def test_mutated_rewritten_estimate_is_rejected(
    mutation_kind: str,
    entrypoint: str,
) -> None:
    """Every public boundary rejects visible state that no longer matches proof."""
    estimate = _ghz_like.estimate_resources()
    if mutation_kind == "field":
        estimate.gates = qm.GateResources(total=999)
    elif mutation_kind == "nested":
        estimate.gates.total = 999
    elif mutation_kind == "assumptions":
        estimate.assumptions = ()
    else:
        estimate.parameters["length"] = sp.Symbol("alien", integer=True)

    with pytest.raises(
        (RuntimeError, ValueError), match="domain|rewrite|state|snapshot"
    ):
        if entrypoint == "simplify":
            estimate.simplify()
        elif entrypoint == "substitute":
            estimate.substitute(length=2)
        elif entrypoint == "seq":
            estimate.seq(qm.ResourceEstimate.zero())
        elif entrypoint == "seq_all":
            qm.ResourceEstimate.seq_all([estimate])
        elif entrypoint == "parallel":
            estimate.parallel(qm.ResourceEstimate.zero())
        elif entrypoint == "choice":
            estimate.choice(qm.ResourceEstimate.zero())
        elif entrypoint == "conditional":
            estimate.conditional(qm.ResourceEstimate.zero(), sp.true)
        elif entrypoint == "repeat":
            estimate.repeat(1)
        elif entrypoint == "sum":
            estimate.sum_over(sp.Dummy("index", integer=True), 0, 1)
        elif entrypoint == "control":
            estimate.controlled(1)
        elif entrypoint == "inverse":
            estimate.inverse()
        elif entrypoint == "wire":
            resource_estimate_to_wire(estimate)
        elif entrypoint == "physical":
            from qamomile.circuit.estimator.physical import (
                estimate_physical_resources,
            )

            estimate_physical_resources(
                estimate,
                logical_qubits=1,
                non_clifford_gates=1,
            )
        elif entrypoint == "explain":
            estimate.explain()
        else:
            estimate.to_dict()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_seq_all_rejects_mutated_parameter_metadata_at_every_position(
    position: int,
) -> None:
    """Sequential streaming validates every externally supplied estimate."""
    corrupted = _ghz_like.estimate_resources()
    corrupted.parameters["length"] = sp.Symbol("alien", integer=True)
    estimates = [qm.ResourceEstimate.zero() for _ in range(3)]
    estimates[position] = corrupted

    with pytest.raises(RuntimeError, match="parameter metadata"):
        qm.ResourceEstimate.seq_all(estimate for estimate in estimates)


def test_seq_all_refreshes_a_nested_reduction_yielded_late() -> None:
    """A nested reduction remains canonical when a generator yields it late."""
    leaf = _ghz_like.estimate_resources()
    length = leaf.parameters["length"]

    def estimates() -> Iterable[qm.ResourceEstimate]:
        """Yield two empty estimates and one nested composition.

        Returns:
            Iterable[qm.ResourceEstimate]: Sequential estimates to compose.
        """
        yield qm.ResourceEstimate.zero()
        yield qm.ResourceEstimate.zero()
        yield qm.ResourceEstimate.seq_all([leaf, leaf])

    combined = qm.ResourceEstimate.seq_all(estimates())

    assert combined.gates.total == 2 * length
    assert combined.parameters == {"length": length}
    assert combined.substitute(length=3).gates.total == 6


@pytest.mark.parametrize("mutate_after_last_yield", [False, True])
def test_seq_all_rejects_generator_mutation_of_a_yielded_estimate(
    mutate_after_last_yield: bool,
) -> None:
    """Generator side effects cannot launder retained parameter metadata."""
    target = _ghz_like.estimate_resources()

    def estimates() -> Iterable[qm.ResourceEstimate]:
        """Mutate a retained estimate while advancing the generator.

        Returns:
            Iterable[qm.ResourceEstimate]: Deliberately corrupt estimate stream.
        """
        if mutate_after_last_yield:
            yield qm.ResourceEstimate.zero()
            yield target
            target.parameters.clear()
        else:
            yield target
            target.parameters.clear()
            yield qm.ResourceEstimate.zero()

    with pytest.raises(RuntimeError, match="parameter metadata"):
        qm.ResourceEstimate.seq_all(estimates())


def test_private_scheduler_and_trace_expressions_are_not_domain_rewritten() -> None:
    """Domain rewriting changes public metrics but not physical/private metadata."""
    source = _domain_only.estimate_resources()
    length = source.parameters["length"]
    original = sp.Max(0, length - 1)
    guarded = sp.Eq(length, 1)
    estimate = dataclasses.replace(
        source,
        gates=qm.GateResources(total=original),
        trace=ResourceTraceNode(
            "private",
            "test",
            active_when=guarded,
        ),
        _allocation_sites={"site": original},
        _output_sizes={"owner": original},
        _input_sizes={"owner": original},
        _dependency_completion={("owner", 0): original},
        _dependency_synchronized_entry_conditions={("owner", 0): guarded},
    ).simplify()

    assert estimate.gates.total == length - 1
    assert estimate._allocation_sites == {"site": original}
    assert estimate._output_sizes == {"owner": original}
    assert estimate._input_sizes == {"owner": original}
    assert estimate._dependency_completion == {("owner", 0): original}
    assert estimate._dependency_synchronized_entry_conditions == {("owner", 0): guarded}
    assert estimate.trace is not None
    assert estimate.trace.active_when == guarded


def test_physical_conversion_requires_domain_specialization_or_two_overrides() -> None:
    """Physical conversion cannot silently drop an unresolved logical domain."""
    from qamomile.circuit.estimator.physical import estimate_physical_resources

    symbolic = _ghz_like.estimate_resources()

    with pytest.raises(ValueError, match="qkernel input-domain|specialize"):
        estimate_physical_resources(symbolic)
    specialized = estimate_physical_resources(symbolic.substitute(length=2))
    overridden = estimate_physical_resources(
        symbolic,
        logical_qubits=3,
        non_clifford_gates=4,
    )

    assert specialized.logical_qubits == 2
    assert overridden.logical_qubits == 3
    assert overridden.non_clifford_gates == 4


@pytest.mark.parametrize("cost_kind", ["fixed", "callback"])
@pytest.mark.parametrize("transform_kind", ["plain", "repeat", "inverse", "control"])
def test_opaque_boundary_demotes_a_reused_qkernel_domain(
    cost_kind: str,
    transform_kind: str,
) -> None:
    """A former root domain cannot simplify a fixed or callback opaque cost."""
    former_root = _ghz_like.estimate_resources()
    reused = {
        "plain": lambda: former_root,
        "repeat": lambda: former_root.repeat(1),
        "inverse": former_root.inverse,
        "control": lambda: former_root.controlled(1),
    }[transform_kind]()

    def callback(_: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return a former root estimate as an opaque model.

        Args:
            _ (qm.OpaqueCostContext): Unused opaque definition context.

        Returns:
            qm.ResourceEstimate: Former root qkernel estimate.
        """
        return reused

    cost = reused if cost_kind == "fixed" else callback
    oracle = qm.opaque(
        f"reused_root_{cost_kind}_{transform_kind}",
        num_qubits=1,
        cost=cost,
    )

    @qm.qkernel
    def caller(length: qm.UInt) -> qm.Qubit:
        """Invoke a cost whose old formal merely shares this input name.

        Args:
            length (qm.UInt): Caller input sharing the old formal's name.

        Returns:
            qm.Qubit: Oracle target.
        """
        (target,) = oracle(qm.qubit("target"))
        return target

    estimate = caller.estimate_resources()

    assert estimate.gates.total.has(sp.Max)
    assert _domain_assumptions(estimate) == ()
    assert caller.estimate_resources(inputs={"length": 2}).gates.total.is_number
    with pytest.raises(ValueError, match="in-bounds"):
        caller.estimate_resources(inputs={"length": 0})


@pytest.mark.parametrize("cost_kind", ["fixed", "callback"])
def test_opaque_boundary_rejects_stale_parameter_metadata(cost_kind: str) -> None:
    """Fixed and callback opaque costs reject tampered domain parameters."""
    stale = _ghz_like.estimate_resources()
    stale.parameters["length"] = sp.Symbol("alien", integer=True)

    def callback(_: qm.OpaqueCostContext) -> qm.ResourceEstimate:
        """Return the stale estimate through an opaque callback.

        Args:
            _ (qm.OpaqueCostContext): Unused opaque definition context.

        Returns:
            qm.ResourceEstimate: Deliberately stale estimate.
        """
        return stale

    oracle = qm.opaque(
        f"stale_domain_{cost_kind}",
        num_qubits=1,
        cost=stale if cost_kind == "fixed" else callback,
    )

    @qm.qkernel
    def caller() -> qm.Qubit:
        """Invoke the stale opaque cost.

        Returns:
            qm.Qubit: Oracle target.
        """
        (target,) = oracle(qm.qubit("target"))
        return target

    with pytest.raises(RuntimeError, match="parameter metadata"):
        caller.estimate_resources()


def test_wire_round_trip_preserves_rewrite_lifecycle() -> None:
    """Semantic wire state remains safe for substitution and composition."""
    estimate = _ghz_like.estimate_resources()
    payload = resource_estimate_to_wire(estimate)

    restored = resource_estimate_from_wire(payload)

    assert restored.gates == estimate.gates
    assert restored.depth == estimate.depth
    assert restored.assumptions == estimate.assumptions
    assert restored.substitute(length=2).gates.total == 2
    assert restored.seq(restored).substitute(length=2).gates.total == 4
    with pytest.raises(ValueError, match="in-bounds"):
        restored.substitute(length=0)


def test_wire_rejects_unknown_fields_and_noncanonical_visible_state() -> None:
    """Strict decoding rejects schema drift and forged rewritten metrics."""
    payload = resource_estimate_to_wire(_ghz_like.estimate_resources())
    unknown = deepcopy(payload)
    unknown["unknown_domain_field"] = None
    forged = deepcopy(payload)
    forged["gates"]["total"] = "Integer(999)"

    with pytest.raises(ValueError, match="fields must be"):
        resource_estimate_from_wire(unknown)
    with pytest.raises(ValueError, match="domain|state|canonical|visible"):
        resource_estimate_from_wire(forged)


@pytest.mark.parametrize(
    "tamper_kind",
    ["policy", "original", "rewritten", "evidence", "provenance", "assumption"],
)
def test_wire_rederives_every_domain_lifecycle_component(
    tamper_kind: str,
) -> None:
    """Decode rejects every independently forged domain-state component."""
    payload = deepcopy(resource_estimate_to_wire(_ghz_like.estimate_resources()))
    state = payload["domain_rewrite_state"]
    assert state is not None
    if tamper_kind == "policy":
        payload["domain_rewrite_policy"] = "disabled"
    elif tamper_kind == "original":
        state["original"]["gates"]["total"] = "Integer(999)"
    elif tamper_kind == "rewritten":
        state["rewritten"]["gates"]["total"] = "Integer(999)"
    elif tamper_kind == "evidence":
        state["requirements"][0]["labels"] = ["forged evidence"]
    elif tamper_kind == "provenance":
        for requirement in payload["requirements"]:
            requirement["provenance"]["root_formal_names"] = []
    else:
        payload["assumptions"] = []

    with pytest.raises(ValueError, match="domain|canonical|visible|assumptions"):
        resource_estimate_from_wire(payload)


def test_wire_round_trip_keeps_equal_ordinary_assumption_distinct_from_domain() -> None:
    """Wire reconstruction removes only generated evidence on specialization."""
    estimate = _ghz_like.estimate_resources()
    generated = _domain_assumptions(estimate)[0]
    ordinary = qm.ResourceAssumption(generated.message, generated.source)
    augmented = dataclasses.replace(
        estimate,
        assumptions=(*estimate.assumptions, ordinary),
    )

    restored = resource_estimate_from_wire(resource_estimate_to_wire(augmented))
    specialized = restored.substitute(length=2)

    assert restored.assumptions == (ordinary, generated)
    assert specialized.assumptions == (ordinary,)


def test_many_composed_constraints_deduplicate_domain_assumptions() -> None:
    """At least 1,000 composed constraints publish one deduplicated condition."""
    estimate = _ghz_like.estimate_resources()
    length = estimate.parameters["length"]
    repetitions = (1_000 + len(estimate._constraints) - 1) // len(estimate._constraints)

    composed = qm.ResourceEstimate.seq_all(itertools.repeat(estimate, repetitions))

    assert len(composed._constraints) >= 1_000
    assert composed.gates.total == repetitions * length
    assert len(_domain_assumptions(composed)) == 1
    assert composed.substitute(length=2).gates.total == 2 * repetitions


def test_seq_all_applies_parameter_domain_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Balanced sequential reduction defers domain proof until completion."""
    estimate = _ghz_like.estimate_resources()
    original_apply = composition_module._apply_domain_rewrite
    apply_count = 0

    def counted_apply(
        candidate: qm.ResourceEstimate,
        *,
        policy: parameter_domain._DomainRewritePolicy | None = None,
    ) -> qm.ResourceEstimate:
        """Count batched domain passes before delegating.

        Args:
            candidate (qm.ResourceEstimate): Domain-independent composition.
            policy (_DomainRewritePolicy | None): Merged rewrite policy.

        Returns:
            qm.ResourceEstimate: Canonically rewritten composition.
        """
        nonlocal apply_count
        apply_count += 1
        return original_apply(candidate, policy=policy)

    monkeypatch.setattr(composition_module, "_apply_domain_rewrite", counted_apply)

    composed = qm.ResourceEstimate.seq_all([estimate] * 256)

    assert apply_count == 1
    assert composed.gates.total == 256 * estimate.parameters["length"]


def test_domain_rewrite_proves_each_distinct_public_expression_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated public metrics share one domain-proof result per expression."""
    source = _domain_only.estimate_resources()
    length = source.parameters["length"]
    repeated = sp.Max(0, length - 1)
    estimate = dataclasses.replace(
        source,
        gates=qm.GateResources(total=repeated, single_qubit=repeated),
        calls=qm.CallResources(
            calls_by_name={f"call_{index}": repeated for index in range(200)}
        ),
    )
    original_rewrite = estimate_domain._rewrite_expression_in_domain
    rewrite_counts: dict[sp.Expr, int] = {}

    def counted_rewrite(
        expression: ResourceExpr,
        environment: parameter_domain._ProofEnvironment,
    ) -> tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
        """Count domain prover invocations before delegating.

        Args:
            expression (ResourceExpr): Public metric expression to rewrite.
            environment (_ProofEnvironment): Shared bounded proof state.

        Returns:
            tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
                Rewritten expression and proof evidence returned by the
                production helper.
        """
        rewrite_counts[expression] = rewrite_counts.get(expression, 0) + 1
        return original_rewrite(expression, environment)

    monkeypatch.setattr(
        estimate_domain,
        "_rewrite_expression_in_domain",
        counted_rewrite,
    )

    rewritten = estimate.simplify()

    assert rewrite_counts[repeated] == 1
    assert rewritten.gates.total == length - 1
    assert set(rewritten.calls.calls_by_name.values()) == {length - 1}


def test_estimate_domain_projection_is_prepared_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct public formulas share one projected proof environment."""
    source = _domain_only.estimate_resources()
    length = source.parameters["length"]
    estimate = dataclasses.replace(
        source,
        calls=qm.CallResources(
            calls_by_name={
                f"call_{offset}": sp.Max(0, length - offset) for offset in range(1, 201)
            }
        ),
    )
    original_prepare = estimate_domain._prepare_domain_proof_environment
    prepare_count = 0

    def counted_prepare(
        constraints: tuple[_ResourceConstraint, ...],
    ) -> parameter_domain._ProofEnvironment | None:
        """Count source projection before delegating.

        Args:
            constraints (tuple[_ResourceConstraint, ...]): Candidate sources.

        Returns:
            _ProofEnvironment | None: Prepared production proof state.
        """
        nonlocal prepare_count
        prepare_count += 1
        return original_prepare(constraints)

    monkeypatch.setattr(
        estimate_domain,
        "_prepare_domain_proof_environment",
        counted_prepare,
    )

    estimate.simplify()

    assert prepare_count == 1


def test_estimate_domain_budget_exhaustion_rolls_back_every_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exhausted scalar proof discards every candidate metric rewrite."""
    estimate = qm.ResourceEstimator(simplify=False).estimate(_domain_only)
    original = deepcopy(estimate)
    rewrite_count = 0

    def exhausting_rewrite(
        expression: ResourceExpr,
        environment: parameter_domain._ProofEnvironment,
    ) -> tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
        """Return one candidate rewrite and then exhaust the shared budget.

        Args:
            expression (ResourceExpr): Current public scalar.
            environment (_ProofEnvironment): Shared bounded proof state.

        Returns:
            tuple[ResourceExpr, tuple[_ConsumedDomainRequirement, ...]]:
                Synthetic candidate or untouched expression.
        """
        nonlocal rewrite_count
        rewrite_count += 1
        if rewrite_count == 1:
            return sp.Integer(123), (environment.sources[0].requirement,)
        environment.exhausted = True
        return expression, ()

    monkeypatch.setattr(
        estimate_domain,
        "_rewrite_expression_in_domain",
        exhausting_rewrite,
    )

    rewritten = estimate.simplify()

    assert rewritten.width == original.width
    assert rewritten.gates == original.gates
    assert rewritten.depth == original.depth
    assert rewritten.calls == original.calls
    assert rewritten._domain_rewrite_state is None
