"""Regression tests for estimator analysis and composition maintenance."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.circuit.estimator._scheduling as scheduling_module
import qamomile.observable as qm_o
import qamomile.observable.hamiltonian as hamiltonian_module
from qamomile.circuit.estimator import resource_estimator as estimator_module
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperationType


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


def test_ir_gate_arity_profile_is_exhaustive_and_disjoint() -> None:
    """Every IR gate has one explicit arity without synthetic-name fallback."""
    assert set(estimator_module._GATE_OPERATION_ARITY) == set(GateOperationType)
    assert set(estimator_module._GATE_OPERATION_ARITY.values()) <= {1, 2, 3}
    assert estimator_module._PORTABLE_EXPLICIT_MULTI_TARGET_GATE_TYPES == {
        gate_type
        for gate_type, arity in estimator_module._GATE_OPERATION_ARITY.items()
        if arity > 1
    }

    ir_names = {gate_type.name.lower() for gate_type in GateOperationType}
    synthetic_names = (
        estimator_module._SYNTHETIC_SINGLE_QUBIT_GATE_NAMES
        | estimator_module._SYNTHETIC_MULTI_QUBIT_GATE_NAMES
    )
    assert ir_names.isdisjoint(synthetic_names)


def test_operation_taint_analysis_is_cached_by_operation_list_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated evaluation reuses structural taint analysis for the same list."""
    block = _taint_cache_body.build()
    operations = block.operations
    original = estimator_module.build_dependency_graph
    analyzed: list[list[estimator_module.Operation]] = []

    def record_analysis(
        candidate: list[estimator_module.Operation],
    ) -> dict[str, set[str]]:
        """Record each dependency-graph construction."""
        analyzed.append(candidate)
        return original(candidate)

    monkeypatch.setattr(
        estimator_module,
        "build_dependency_graph",
        record_analysis,
    )
    interpreter = estimator_module.ResourceInterpreter(
        config=estimator_module.ResourceEstimatorConfig(),
        bindings={},
    )

    interpreter.eval_operations(operations, ExprResolver(block=block))
    interpreter.eval_operations(operations, ExprResolver(block=block))
    copied_operations = list(operations)
    interpreter.eval_operations(copied_operations, ExprResolver(block=block))

    assert len(analyzed) == 2
    assert analyzed[0] is operations
    assert analyzed[1] is copied_operations
    assert interpreter._operation_taint_cache[id(operations)][0] is operations


def test_eval_operations_reuses_precomputed_wire_footprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency aggregation and scheduling share one resolved footprint."""
    block = _taint_cache_body.build()
    wire_keys = Mock(wraps=estimator_module._quantum_wire_keys)
    monkeypatch.setattr(estimator_module, "_quantum_wire_keys", wire_keys)
    interpreter = estimator_module.ResourceInterpreter(
        config=estimator_module.ResourceEstimatorConfig(basis=qmc.GateBasis.LOGICAL),
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
        estimator_module,
        "_disjoint_concrete_loop_depth",
        concrete,
    )

    estimate = _affine_disjoint_loop.estimate_resources(
        inputs={"width": 64},
        basis=qmc.GateBasis.LOGICAL,
    )

    concrete.assert_not_called()
    assert estimate.gates.total == 64
    assert estimate.depth.depth == 1
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_nonlinear_loop_falls_back_to_concrete_disjointness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concrete nonlinear footprint retains exact parallel loop depth."""
    concrete = Mock(wraps=estimator_module._disjoint_concrete_loop_depth)
    monkeypatch.setattr(
        estimator_module,
        "_disjoint_concrete_loop_depth",
        concrete,
    )

    estimate = _nonlinear_disjoint_loop.estimate_resources(basis=qmc.GateBasis.LOGICAL)

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
    simplify = Mock(wraps=scheduling_module._safe_simplify)
    monkeypatch.setattr(scheduling_module, "_safe_simplify", simplify)

    depth = scheduling_module._disjoint_concrete_loop_depth(
        operation,
        ExprResolver(block=block),
        qmc.DepthResources(**invariants),
        start=sp.Integer(0),
        stop=sp.Integer(3),
        step=sp.Integer(1),
        loop_symbol=sp.Dummy("loop", integer=True),
        clean_ancillas=sp.Integer(0),
    )

    assert depth is not None
    for field, invariant in invariants.items():
        assert getattr(depth, field) == invariant
        assert sum(call.args[0] == invariant for call in simplify.call_args_list) == 1


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
                quality=(
                    qmc.EstimateQuality.MODELED
                    if index == 5
                    else qmc.EstimateQuality.UPPER_BOUND
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
                    estimator_module._ResourceConstraint(
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
    assert balanced._guarded_qualities == left_fold._guarded_qualities
    assert _trace_leaf_names(balanced.trace) == [
        f"leaf_{index}" for index in range(1, 6)
    ]
    assert qmc.ResourceEstimate.seq_all([]) == qmc.ResourceEstimate.zero()


def test_seq_all_keeps_large_metadata_reduction_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata work grows by balanced levels rather than left-fold history."""
    estimate_count = 1024
    leaf = qmc.ResourceEstimate(quality=qmc.EstimateQuality.UPPER_BOUND)
    original = qmc.ResourceEstimate.seq
    seq_calls = 0
    metadata_visits = 0

    def record_seq(
        self: qmc.ResourceEstimate,
        other: qmc.ResourceEstimate,
    ) -> qmc.ResourceEstimate:
        """Count guarded-quality records visited by each binary composition."""
        nonlocal metadata_visits, seq_calls
        seq_calls += 1
        metadata_visits += len(self._guarded_qualities or ())
        metadata_visits += len(other._guarded_qualities or ())
        return original(self, other)

    monkeypatch.setattr(qmc.ResourceEstimate, "seq", record_seq)
    combined = qmc.ResourceEstimate.seq_all(leaf for _ in range(estimate_count))

    assert seq_calls == estimate_count - 1
    assert metadata_visits <= estimate_count * estimate_count.bit_length()
    assert len(combined._guarded_qualities or ()) == estimate_count
    assert combined.quality is qmc.EstimateQuality.UPPER_BOUND


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
        assumption.source == "PauliEvolveOp" for assumption in estimate.assumptions
    )
