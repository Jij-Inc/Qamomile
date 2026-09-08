"""Regression tests for estimator analysis and composition maintenance."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Iterable
from typing import get_type_hints
from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.circuit.estimator._call_liveness as call_liveness_module
import qamomile.circuit.estimator._config as config_module
import qamomile.circuit.estimator._estimate as estimate_module
import qamomile.circuit.estimator._estimate_composition as composition_module
import qamomile.circuit.estimator._gate_catalog as gate_catalog_module
import qamomile.circuit.estimator._interpreter_calls as interpreter_calls_module
import qamomile.circuit.estimator._interpreter_for as interpreter_for_module
import qamomile.circuit.estimator._interpreter_for_items as interpreter_for_items_module
import qamomile.circuit.estimator._interpreter_loop_support as loop_support_module
import qamomile.circuit.estimator._interpreter_region_analysis as region_analysis_module
import qamomile.circuit.estimator._loop_scheduling as loop_scheduling_module
import qamomile.circuit.estimator._resource_bounds as resource_bounds_module
import qamomile.circuit.estimator._resource_expressions as resource_expressions_module
import qamomile.circuit.estimator._resource_types as resource_types_module
import qamomile.circuit.estimator._runtime_observation as runtime_observation_module
import qamomile.circuit.estimator._scheduling as scheduling_module
import qamomile.circuit.estimator._symbolic as symbolic_module
import qamomile.observable as qm_o
import qamomile.observable.hamiltonian as hamiltonian_module
from qamomile.circuit.estimator import resource_estimator as estimator_module
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintOrigin,
    _ConstraintProvenance,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.estimator._wire import resource_estimate_to_wire
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import GateOperationType, MeasureOperation
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType
from qamomile.circuit.ir.value import Value
from tests.circuit.qkernel_catalog import grover_network_decomposition


@qmc.qkernel
def _taint_cache_body(target: qmc.Qubit) -> qmc.Qubit:
    """Provide one stable operation list for dependency-analysis caching."""
    return qmc.h(target)


@qmc.qkernel
def _all_z_pauli_evolution(observable: qmc.Observable) -> qmc.Vector[qmc.Qubit]:
    """Evolve a fixed register under a supplied all-Z Hamiltonian."""
    register = qmc.qubit_array(3, "register")
    return qmc.pauli_evolve(register, observable, qmc.float_(0.25))


@qmc.qkernel
def _affine_disjoint_loop(width: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply independent gates through one affine loop index.

    Args:
        width (qmc.UInt): Number of independently addressed qubits.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated register.
    """
    register = qmc.qubit_array(width, "register")
    for index in qmc.range(width):
        register[index] = qmc.h(register[index])
    return register


@qmc.qkernel
def _nonlinear_disjoint_loop() -> qmc.Vector[qmc.Qubit]:
    """Apply independent gates through concrete nonlinear loop indices.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated fixed-width register.
    """
    register = qmc.qubit_array(9, "register")
    for index in qmc.range(3):
        register[index * index] = qmc.h(register[index * index])
    return register


def _trace_leaf_names(
    node: estimator_module.ResourceTraceNode | None,
) -> list[str]:
    """Return sequential trace leaves in execution order."""
    if node is None:
        return []
    if node.name == "seq" and node.source_kind == "algebra":
        return [name for child in node.children for name in _trace_leaf_names(child)]
    return [node.name]


def test_public_resource_types_retain_owner_introspection() -> None:
    """Moved public types remain introspectable through their owner modules."""
    for resource_type in (
        qmc.GateResources,
        resource_types_module.ResourceTraceNode,
        qmc.WidthResources,
        estimator_module.OpaqueCostContext,
        estimator_module.ResourceEstimate,
        estimator_module.ResourceInterpreter,
        estimator_module.UnknownResourcePolicy,
    ):
        assert inspect.getsource(resource_type)

    assert "gates" in get_type_hints(estimator_module.ResourceEstimate)
    assert "definition_control_qubits" in get_type_hints(
        estimator_module.OpaqueCostContext
    )
    assert "total" in get_type_hints(qmc.GateResources)
    resource = qmc.GateResources(total=1)
    restored = pickle.loads(pickle.dumps(resource))
    assert restored == resource
    assert type(restored) is qmc.GateResources


def test_ir_gate_arity_profile_is_exhaustive_and_disjoint() -> None:
    """Every IR gate has one explicit arity without synthetic-name fallback."""
    gate_catalog_module._validate_ir_gate_arity_profiles()
    assert set(gate_catalog_module._GATE_OPERATION_ARITY) == set(GateOperationType)
    assert set(gate_catalog_module._GATE_OPERATION_ARITY.values()) <= {1, 2, 3}
    assert gate_catalog_module._CLEAN_ANCILLA_EXPLICIT_MULTI_TARGET_GATE_TYPES == {
        gate_type
        for gate_type, arity in gate_catalog_module._GATE_OPERATION_ARITY.items()
        if arity > 1
    }

    ir_names = {gate_type.name.lower() for gate_type in GateOperationType}
    synthetic_names = (
        gate_catalog_module._SYNTHETIC_SINGLE_QUBIT_GATE_NAMES
        | gate_catalog_module._SYNTHETIC_MULTI_QUBIT_GATE_NAMES
    )
    assert ir_names.isdisjoint(synthetic_names)


def test_ir_gate_arity_validator_rejects_overlapping_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source profiles expose overlap hidden by the merged arity mapping."""
    monkeypatch.setattr(
        gate_catalog_module,
        "_IR_SINGLE_QUBIT_GATE_TYPES",
        gate_catalog_module._IR_SINGLE_QUBIT_GATE_TYPES | {GateOperationType.CX},
    )

    with pytest.raises(RuntimeError, match=r"duplicated=\['CX'\]"):
        gate_catalog_module._validate_ir_gate_arity_profiles()


def test_runtime_observation_analysis_is_cached_by_block_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated observation summaries reuse analysis for the same block."""
    block = _taint_cache_body.build()
    original = runtime_observation_module.build_dependency_graph
    analyzed: list[list[estimator_module.Operation]] = []

    def record_analysis(
        candidate: list[estimator_module.Operation],
    ) -> dict[str, set[str]]:
        """Record each dependency-graph construction."""
        analyzed.append(candidate)
        return original(candidate)

    monkeypatch.setattr(
        runtime_observation_module,
        "build_dependency_graph",
        record_analysis,
    )
    interpreter = estimator_module.ResourceInterpreter(
        config=config_module._ResourceEstimatorConfig(),
        bindings={},
    )

    runtime_observation_module._block_runtime_observation_summary(
        block,
        strategy_for=interpreter._strategy_for,
        cache=interpreter._runtime_observation_cache,
    )
    first_analysis_count = len(analyzed)
    assert first_analysis_count > 0

    runtime_observation_module._block_runtime_observation_summary(
        block,
        strategy_for=interpreter._strategy_for,
        cache=interpreter._runtime_observation_cache,
    )
    assert len(analyzed) == first_analysis_count
    assert interpreter._runtime_observation_cache[id(block)][0] is block

    distinct_block = _taint_cache_body.build()
    runtime_observation_module._block_runtime_observation_summary(
        distinct_block,
        strategy_for=interpreter._strategy_for,
        cache=interpreter._runtime_observation_cache,
    )
    assert len(analyzed) > first_analysis_count
    assert interpreter._runtime_observation_cache[id(distinct_block)][0] is (
        distinct_block
    )


def test_interpreter_reuse_restores_root_condition_values() -> None:
    """Each estimate starts with the interpreter's original condition values."""

    @qmc.qkernel
    def first_block(first_flag: qmc.UInt) -> qmc.Qubit:
        """Conditionally apply a gate using the first root parameter."""
        target = qmc.qubit("target")
        if first_flag:
            target = qmc.h(target)
        return target

    @qmc.qkernel
    def second_block(second_flag: qmc.UInt) -> qmc.Qubit:
        """Conditionally apply a gate using the second root parameter."""
        target = qmc.qubit("target")
        if second_flag:
            target = qmc.x(target)
        return target

    interpreter = estimator_module.ResourceInterpreter(
        config=config_module._ResourceEstimatorConfig(),
        bindings={},
        condition_values={
            "first_flag": sp.Integer(0),
            "second_flag": sp.Integer(1),
        },
    )

    first_estimate = interpreter.estimate(first_block.build())
    second_estimate = interpreter.estimate(second_block.build())

    assert first_estimate.gates.total == 0
    assert second_estimate.gates.total == 1
    assert interpreter.branch_condition_names == {"second_flag"}


def test_call_body_restores_interpreter_state_when_preparation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed call setup leaves the interpreter reusable."""
    block = Block(name="callee")
    interpreter = estimator_module.ResourceInterpreter(
        config=config_module._ResourceEstimatorConfig(),
        bindings={"sentinel": sp.Integer(7)},
    )
    original_bindings = interpreter.bindings
    original_taint = interpreter._run_state.measurement_taint_conditions

    with monkeypatch.context() as patch:
        patch.setattr(
            interpreter_calls_module,
            "_merge_measurement_taint_conditions",
            Mock(side_effect=RuntimeError("taint merge failed")),
        )
        with pytest.raises(RuntimeError, match="taint merge failed"):
            interpreter._eval_call_body(
                block,
                ExprResolver(block=block),
                [],
                controls=0,
            )

    assert interpreter.bindings is original_bindings
    assert interpreter._run_state.measurement_taint_conditions is original_taint
    assert interpreter._run_state.active_call_states == {}

    estimate = interpreter._eval_call_body(
        block,
        ExprResolver(block=block),
        [],
        controls=0,
    )
    assert estimate.gates.total == 0


def test_eval_operations_reuses_precomputed_wire_footprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency aggregation and scheduling share one resolved footprint."""
    block = _taint_cache_body.build()
    wire_keys = Mock(wraps=region_analysis_module._quantum_wire_keys)
    monkeypatch.setattr(region_analysis_module, "_quantum_wire_keys", wire_keys)
    interpreter = estimator_module.ResourceInterpreter(
        config=config_module._ResourceEstimatorConfig(),
        bindings={},
    )

    estimate = interpreter.eval_operations(
        block.operations,
        ExprResolver(block=block),
    )

    assert estimate.depth.depth == 1
    assert wire_keys.call_count == 1


def test_affine_loop_uses_symbolic_disjointness_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete affine bounds take the constant-time symbolic proof first."""
    concrete = Mock(
        side_effect=AssertionError(
            "an affine footprint must not use concrete enumeration"
        )
    )
    monkeypatch.setattr(
        loop_support_module,
        "_disjoint_concrete_loop_depth",
        concrete,
    )

    estimate = _affine_disjoint_loop.estimate_resources(
        inputs={"width": 64},
    )

    concrete.assert_not_called()
    assert estimate.gates.total == 64
    assert estimate.depth.depth == 1
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_observation_free_concrete_loop_skips_consumption_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unitary loop does not construct per-iteration liveness resolvers."""

    @qmc.qkernel
    def circuit() -> qmc.Vector[qmc.Qubit]:
        """Apply one gate to every slot without destructive observation."""
        register = qmc.qubit_array(256, "register")
        for index in qmc.range(256):
            register[index] = qmc.h(register[index])
        return register

    local_block = Mock(wraps=interpreter_for_module._LocalBlock)
    monkeypatch.setattr(interpreter_for_module, "_LocalBlock", local_block)

    estimate = circuit.estimate_resources()

    assert estimate.gates.total == 256
    local_block.assert_not_called()


def test_loop_observation_scan_keeps_invoke_definitions_opaque() -> None:
    """The fast-path scan matches nested-region and invoke boundaries."""
    source = Value(type=QubitType(), name="source")
    result = Value(type=BitType(), name="result")
    measurement = MeasureOperation(operands=[source], results=[result])
    observable = Value(type=ObservableType(), name="observable")
    expectation = Value(type=FloatType(), name="expectation")
    expval = ExpvalOp(
        operands=[source, observable],
        results=[expectation],
    )
    nested = IfOperation(true_operations=[measurement])
    nested_expval = IfOperation(false_operations=[expval])
    invocation = InvokeOperation(
        definition=CallableDef(
            ref=CallableRef(namespace="test", name="measured_helper"),
            body=Block(operations=[measurement]),
        )
    )

    assert call_liveness_module._loop_body_has_destructive_observation([measurement])
    assert call_liveness_module._loop_body_has_destructive_observation([expval])
    assert call_liveness_module._loop_body_has_destructive_observation([nested])
    assert call_liveness_module._loop_body_has_destructive_observation([nested_expval])
    assert not call_liveness_module._loop_body_has_destructive_observation([invocation])


def test_observation_free_concrete_items_skip_entry_consumption_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unitary items loop builds no entry-specific liveness resolvers."""

    @qmc.qkernel
    def circuit(
        data: qmc.Dict[qmc.UInt, qmc.Float],
    ) -> qmc.Vector[qmc.Qubit]:
        """Apply one gate to the register slot selected by each item key."""
        register = qmc.qubit_array(256, "register")
        for index, _value in qmc.items(data):
            register[index] = qmc.h(register[index])
        return register

    local_block = Mock(wraps=interpreter_for_items_module._LocalBlock)
    monkeypatch.setattr(interpreter_for_items_module, "_LocalBlock", local_block)

    data = {index: 0.0 for index in range(256)}
    estimate = circuit.estimate_resources(inputs={"data": data})

    assert estimate.gates.total == 256
    assert local_block.call_count == len(data) + 1


def test_nonlinear_loop_falls_back_to_concrete_disjointness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concrete nonlinear footprint retains exact parallel loop depth."""
    concrete = Mock(wraps=loop_support_module._disjoint_concrete_loop_depth)
    monkeypatch.setattr(
        loop_support_module,
        "_disjoint_concrete_loop_depth",
        concrete,
    )

    estimate = _nonlinear_disjoint_loop.estimate_resources()

    assert concrete.call_count == 1
    assert estimate.gates.total == 3
    assert estimate.depth.depth == 1
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_concrete_loop_simplifies_constant_depth_fields_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concrete enumeration hoists every loop-invariant depth expression."""
    block = _nonlinear_disjoint_loop.build()
    operation = next(
        candidate
        for candidate in block.operations
        if isinstance(candidate, ForOperation)
    )
    invariants = {
        field: sp.Symbol(f"invariant_{field}", positive=True)
        for field in qmc.DepthResources.__dataclass_fields__
    }
    simplify = Mock(wraps=loop_scheduling_module._safe_simplify)
    monkeypatch.setattr(loop_scheduling_module, "_safe_simplify", simplify)

    depth = loop_scheduling_module._disjoint_concrete_loop_depth(
        operation,
        ExprResolver(block=block),
        qmc.DepthResources(**invariants),
        start=sp.Integer(0),
        stop=sp.Integer(3),
        step=sp.Integer(1),
        loop_symbol=sp.Dummy("loop", integer=True),
        allocated_qubits=sp.Integer(0),
        clean_ancillas=sp.Integer(0),
        dirty_ancillas=sp.Integer(0),
    )

    assert depth is not None
    for field, invariant in invariants.items():
        assert getattr(depth, field) == invariant
        assert sum(call.args[0] == invariant for call in simplify.call_args_list) == 1


def test_public_simplification_skips_branching_resource_expression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deep width summaries bypass SymPy's global simplifier."""
    parameter = sp.Symbol("n", integer=True, nonnegative=True)
    expression = parameter
    for offset in range(symbolic_module._PUBLIC_RESOURCE_SIMPLIFY_NODE_LIMIT):
        expression = sp.Max(
            expression,
            sp.Piecewise(
                (parameter + offset + 1, parameter > offset),
                (offset, True),
            ),
            evaluate=False,
        )
    simplify = Mock(
        side_effect=AssertionError(
            "branching resource expressions must stay structurally normalized"
        )
    )
    monkeypatch.setattr(symbolic_module, "_safe_simplify", simplify)

    assert (
        symbolic_module._simplify_public_resource_expression(expression) == expression
    )
    simplify.assert_not_called()


def test_resource_max_bounds_conditionally_active_work() -> None:
    """A zero-or-one activity guard cannot exceed its nonnegative work."""
    parameter = sp.Symbol("n", integer=True, nonnegative=True)
    work = 5 * sp.Max(0, parameter - 3) + 5
    active = resource_expressions_module._ConditionIndicator(sp.Gt(parameter, 0))

    assert resource_bounds_module._resource_max(work, work * active) == work


def test_resource_max_recovers_activity_guarded_extension() -> None:
    """An activity-gated extension keeps its compact dominating completion."""
    parameter = sp.Symbol("n", integer=True, nonnegative=True)
    base = sp.Integer(5)
    extension = sp.Max(0, parameter - 3)
    active = resource_expressions_module._ConditionIndicator(
        resource_expressions_module._resource_activity_condition(extension)
    )

    assert (
        resource_bounds_module._resource_max(
            base,
            (base + extension) * active,
        )
        == base + extension
    )


def test_resource_max_fallback_avoids_sympy_relation_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomparable maxima remain symbolic without eager pairwise solving."""
    left, right = sp.symbols("left right", nonnegative=True)

    def reject_relation_proof(*args: object, **kwargs: object) -> None:
        """Fail if SymPy attempts its general Max relation proof."""
        raise AssertionError("unexpected eager SymPy Max relation proof")

    monkeypatch.setattr(
        sp.Max,
        "_is_connected",
        staticmethod(reject_relation_proof),
    )

    maximum = resource_bounds_module._resource_max(left, right)

    assert isinstance(maximum, sp.Max)
    assert set(maximum.args) == {left, right}


def test_resource_max_requires_a_nonnegative_shared_factor() -> None:
    """Coefficient ordering cannot cancel an unproven-sign common factor."""
    factor = sp.Symbol("factor")
    maximum = resource_bounds_module._resource_max(-2 * factor, -factor)

    assert isinstance(maximum, sp.Max)
    assert maximum.subs(factor, -1) == 2

    nonnegative = sp.Symbol("nonnegative", nonnegative=True)
    assert (
        resource_bounds_module._resource_max(nonnegative, 2 * nonnegative)
        == 2 * nonnegative
    )


def test_structural_nonnegativity_translates_single_nested_extremum() -> None:
    """Affine offsets around one nested extremum remain structurally provable."""
    parameter = sp.Symbol("n", integer=True, nonnegative=True)
    branch = sp.Piecewise(
        (sp.Integer(2), sp.Gt(sp.Max(0, parameter - 1), 0)),
        (sp.Integer(0), True),
    )
    expression = (
        sp.Min(
            2,
            sp.Max(
                0,
                -2 * parameter
                + sp.Max(2 * parameter + 4, 2 * parameter + branch + 2)
                - 2,
            ),
        )
        - 2
    )

    assert resource_bounds_module._is_structurally_nonnegative(expression)
    assert not resource_bounds_module._is_structurally_nonnegative(
        sp.Min(2, sp.Max(0, parameter)) - 2
    )


def test_symbolic_catalog_estimation_retains_grover_cost() -> None:
    """A nested symbolic catalog kernel remains practical and exact."""
    estimate = grover_network_decomposition.estimate_resources()
    n = estimate.parameters["n"]
    iterations = estimate.parameters["n_iters"]

    assert estimate.gates.total == (
        n + iterations * (4 * n + 2 * sp.Max(0, n - 3) + 5) + 2
    )


def test_seq_all_matches_left_fold_and_preserves_trace_order() -> None:
    """Balanced composition preserves every sequential resource contract."""
    estimates: list[qmc.ResourceEstimate] = []
    for index in range(1, 6):
        value = sp.Integer(index)
        estimates.append(
            qmc.ResourceEstimate(
                width=qmc.WidthResources(
                    allocated_qubits=value,
                    clean_ancilla_qubits=value,
                    peak_qubits=value,
                ),
                gates=qmc.GateResources(
                    total=value,
                    single_qubit=value,
                    clifford=value,
                ),
                depth=qmc.DepthResources(
                    depth=value,
                    clifford_depth=value,
                    gate_depth=value,
                ),
                calls=qmc.CallResources(
                    calls_by_name={"shared": value, f"call_{index}": value},
                    queries_by_name={f"query_{index}": value},
                ),
                measurements=qmc.MeasurementResources(total=value),
                resets=qmc.ResetResources(total=value),
                assumptions=(
                    qmc.ResourceAssumption(
                        f"assumption {index}",
                        source=f"source_{index}",
                    ),
                ),
                derivation=(
                    qmc.EstimateDerivation.MODELED
                    if index == 5
                    else qmc.EstimateDerivation.STRUCTURAL
                ),
                quality=(
                    qmc.EstimateQuality.CONSERVATIVE
                    if index != 5
                    else qmc.EstimateQuality.EXACT
                ),
                trace=estimator_module.ResourceTraceNode(
                    name=f"leaf_{index}",
                    source_kind="primitive",
                ),
                _allocation_sites={
                    "shared": value,
                    f"site_{index}": value,
                },
                _constraints=(
                    _ResourceConstraint(
                        expression=value,
                        minimum=0,
                        label=f"constraint_{index}",
                    ),
                ),
                _dependency_keys=frozenset({(f"owner_{index}", index)}),
            )
        )

    left_fold = estimates[0]
    for estimate in estimates[1:]:
        left_fold = left_fold.seq(estimate)
    balanced = qmc.ResourceEstimate.seq_all(estimates)

    assert balanced.to_dict() == left_fold.to_dict()
    assert balanced._allocation_sites == left_fold._allocation_sites
    assert balanced._constraints == left_fold._constraints
    assert balanced._dependency_keys == left_fold._dependency_keys
    assert balanced._guarded_assumptions == left_fold._guarded_assumptions
    assert balanced._guarded_derivations == left_fold._guarded_derivations
    assert balanced._guarded_qualities == left_fold._guarded_qualities
    assert _trace_leaf_names(balanced.trace) == [
        f"leaf_{index}" for index in range(1, 6)
    ]
    assert qmc.ResourceEstimate.seq_all([]) == qmc.ResourceEstimate.zero()


@pytest.mark.parametrize(
    ("mutation_kind", "boundary"),
    [
        ("assumptions", "report"),
        ("derivation", "report"),
        ("quality", "report"),
        ("approximation", "report"),
        ("assumptions", "sequential"),
        ("assumptions", "wire"),
    ],
)
def test_public_metadata_mutation_is_rejected_at_resource_boundaries(
    mutation_kind: str,
    boundary: str,
) -> None:
    """Public metadata cannot diverge from its guarded provenance."""
    original_assumption = qmc.ResourceAssumption("original", "test")
    replacement_assumption = qmc.ResourceAssumption("replacement", "test")
    estimate = qmc.ResourceEstimate(
        assumptions=(original_assumption,),
        derivation=qmc.EstimateDerivation.MODELED,
        quality=qmc.EstimateQuality.CONSERVATIVE,
        approximation=qmc.ApproximationStatus.APPROXIMATE,
    )
    if mutation_kind == "assumptions":
        estimate.assumptions = (replacement_assumption,)
    elif mutation_kind == "derivation":
        estimate.derivation = qmc.EstimateDerivation.STRUCTURAL
    elif mutation_kind == "quality":
        estimate.quality = qmc.EstimateQuality.EXACT
    else:
        estimate.approximation = qmc.ApproximationStatus.EXACT

    with pytest.raises(RuntimeError, match="guarded provenance"):
        if boundary == "report":
            estimate.to_dict()
        elif boundary == "sequential":
            estimate.seq(qmc.ResourceEstimate.zero())
        else:
            resource_estimate_to_wire(estimate)


def test_seq_all_keeps_large_metadata_reduction_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata work grows by balanced levels rather than left-fold history."""
    estimate_count = 1024
    leaf = qmc.ResourceEstimate(quality=qmc.EstimateQuality.CONSERVATIVE)
    original = composition_module._compose_sequential_without_domain_rewrite
    seq_calls = 0
    metadata_visits = 0

    def record_seq(
        left: qmc.ResourceEstimate,
        other: qmc.ResourceEstimate,
    ) -> qmc.ResourceEstimate:
        """Count guarded-quality records visited by each composition.

        Args:
            left (qmc.ResourceEstimate): Earlier partial composition.
            other (qmc.ResourceEstimate): Later partial composition.

        Returns:
            qmc.ResourceEstimate: Domain-independent sequential composition.
        """
        nonlocal metadata_visits, seq_calls
        seq_calls += 1
        metadata_visits += len(left._guarded_qualities or ())
        metadata_visits += len(other._guarded_qualities or ())
        return original(left, other)

    monkeypatch.setattr(
        composition_module,
        "_compose_sequential_without_domain_rewrite",
        record_seq,
    )
    combined = qmc.ResourceEstimate.seq_all(leaf for _ in range(estimate_count))

    assert seq_calls == estimate_count - 1
    assert metadata_visits <= estimate_count * estimate_count.bit_length()
    assert len(combined._guarded_qualities or ()) == estimate_count
    assert combined.quality is qmc.EstimateQuality.CONSERVATIVE


def test_seq_all_refreshes_public_symbols_only_after_the_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Balanced composition derives the public symbol registry only once."""
    symbol = sp.Symbol("symbol", integer=True, nonnegative=True)
    leaf = qmc.ResourceEstimate(gates=qmc.GateResources(total=symbol))
    original = estimate_module._refresh_estimate_symbol_metadata
    refresh_count = 0

    def counted_refresh(
        estimate: qmc.ResourceEstimate,
        registry: SymbolRegistry | None = None,
    ) -> SymbolRegistry:
        """Count final symbol-registry refreshes before delegating.

        Args:
            estimate (qmc.ResourceEstimate): Estimate being refreshed.
            registry (SymbolRegistry | None): Optional prebuilt registry.

        Returns:
            SymbolRegistry: Registry returned by the production helper.
        """
        nonlocal refresh_count
        refresh_count += 1
        return original(estimate, registry)

    monkeypatch.setattr(
        estimate_module,
        "_refresh_estimate_symbol_metadata",
        counted_refresh,
    )

    combined = qmc.ResourceEstimate.seq_all([leaf] * 1024)

    assert refresh_count == 1
    assert combined.parameters == {"symbol": symbol}
    assert qmc.ResourceEstimate.seq_all([leaf]) is leaf
    assert refresh_count == 1


def test_seq_all_validates_first_item_before_advancing_generator() -> None:
    """A stale first estimate fails before requesting a second item."""
    symbol = sp.Symbol("stale_symbol", integer=True, nonnegative=True)
    first = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1 + sp.Max(0, symbol - 1)),
        _constraints=(
            _ResourceConstraint(
                expression=symbol - 1,
                minimum=0,
                label="first accessed qubit",
                provenance=_ConstraintProvenance(
                    origin=_ConstraintOrigin.ARRAY_ACCESS,
                    source_expressions=(symbol,),
                    root_formal_names=("stale_symbol",),
                ),
            ),
        ),
    ).simplify()
    assert first._domain_rewrite_state is not None
    first.gates.total = 999
    advanced = False

    def estimates() -> Iterable[qmc.ResourceEstimate]:
        """Yield one stale estimate and record any later advancement.

        Returns:
            Iterable[qmc.ResourceEstimate]: Stale-first estimate stream.
        """
        nonlocal advanced
        yield first
        advanced = True
        yield qmc.ResourceEstimate.zero()

    with pytest.raises(RuntimeError, match="domain rewrite state"):
        qmc.ResourceEstimate.seq_all(estimates())

    assert advanced is False


def test_sequential_composer_is_transactional_after_failed_append() -> None:
    """A failed append leaves the reusable internal reduction unchanged."""
    abstract = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1),
        control_decomposition=qmc.ControlDecomposition.ABSTRACT,
    )
    clean = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1),
        control_decomposition=qmc.ControlDecomposition.CLEAN_ANCILLA_TOFFOLI,
    )
    composer = composition_module._SequentialEstimateComposer(
        qmc.ResourceEstimate.zero()
    )
    composer.append(abstract)

    with pytest.raises(ValueError, match="different control decompositions"):
        composer.append(clean)

    assert composer.finish() is abstract
    composer.append(abstract)
    assert composer.finish().gates.total == 2


def test_all_z_pauli_evolution_skips_pairwise_commutation_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A common local basis takes the linear Pauli-commutation fast path."""
    pairwise = Mock(
        side_effect=AssertionError("all-Z terms must not use the pairwise scan")
    )
    monkeypatch.setattr(
        hamiltonian_module,
        "_pauli_strings_anticommute",
        pairwise,
    )
    hamiltonian = (
        qm_o.Z(0)
        + qm_o.Z(1)
        + qm_o.Z(2)
        + qm_o.Z(0) * qm_o.Z(1)
        + qm_o.Z(1) * qm_o.Z(2)
    )

    estimate = _all_z_pauli_evolution.estimate_resources(
        inputs={"observable": hamiltonian}
    )

    pairwise.assert_not_called()
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert not any(
        "Lie-Trotter" in assumption.message for assumption in estimate.assumptions
    )


def test_completion_uniformity_avoids_general_symbolic_simplification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed structural proof stays conservative without calling simplify."""
    value = sp.Symbol("value", integer=True, nonnegative=True)
    expression = sp.Piecewise((1, sp.Eq(value, 0)), (value, True))
    simplify = Mock(
        side_effect=AssertionError(
            "completion uniformity must not run general simplification"
        )
    )
    monkeypatch.setattr(scheduling_module.sp, "simplify", simplify)

    assert scheduling_module._expressions_proven_equal_without_simplify(
        expression,
        expression,
    )
    assert not scheduling_module._expressions_proven_equal_without_simplify(
        expression,
        sp.Integer(1),
    )
    simplify.assert_not_called()
