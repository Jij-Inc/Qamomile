"""Regression tests for estimator analysis and composition maintenance."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import sympy as sp

import qamomile.circuit as qm
import qamomile.observable as qm_o
import qamomile.observable.hamiltonian as hamiltonian_module
from qamomile.circuit.estimator import resource_estimator as estimator_module
from qamomile.circuit.estimator._resolver import ExprResolver


@qm.qkernel
def _taint_cache_body(target: qm.Qubit) -> qm.Qubit:
    """Provide one stable operation list for dependency-analysis caching."""
    return qm.h(target)


@qm.qkernel
def _all_z_pauli_evolution(observable: qm.Observable) -> qm.Vector[qm.Qubit]:
    """Evolve a fixed register under a supplied all-Z Hamiltonian."""
    register = qm.qubit_array(3, "register")
    return qm.pauli_evolve(register, observable, qm.float_(0.25))


def _trace_leaf_names(
    node: estimator_module.ResourceTraceNode | None,
) -> list[str]:
    """Return sequential trace leaves in execution order."""
    if node is None:
        return []
    if node.name == "seq" and node.source_kind == "algebra":
        return [name for child in node.children for name in _trace_leaf_names(child)]
    return [node.name]


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


def test_seq_all_matches_left_fold_and_preserves_trace_order() -> None:
    """Balanced composition preserves every sequential resource contract."""
    estimates: list[qm.ResourceEstimate] = []
    for index in range(1, 6):
        value = sp.Integer(index)
        estimates.append(
            qm.ResourceEstimate(
                width=qm.WidthResources(
                    allocated_qubits=value,
                    clean_ancilla_qubits=value,
                    peak_qubits=value,
                ),
                gates=qm.GateResources(
                    total=value,
                    single_qubit=value,
                    clifford=value,
                ),
                depth=qm.DepthResources(
                    depth=value,
                    clifford_depth=value,
                    gate_depth=value,
                ),
                calls=qm.CallResources(
                    calls_by_name={"shared": value, f"call_{index}": value},
                    queries_by_name={f"query_{index}": value},
                ),
                measurements=qm.MeasurementResources(total=value),
                resets=qm.ResetResources(total=value),
                assumptions=(
                    qm.ResourceAssumption(
                        f"assumption {index}",
                        source=f"source_{index}",
                    ),
                ),
                quality=(
                    qm.EstimateQuality.MODELED
                    if index == 5
                    else qm.EstimateQuality.UPPER_BOUND
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
    balanced = qm.ResourceEstimate.seq_all(estimates)

    assert balanced.to_dict() == left_fold.to_dict()
    assert balanced._allocation_sites == left_fold._allocation_sites
    assert balanced._constraints == left_fold._constraints
    assert balanced._dependency_keys == left_fold._dependency_keys
    assert balanced._guarded_assumptions == left_fold._guarded_assumptions
    assert balanced._guarded_qualities == left_fold._guarded_qualities
    assert _trace_leaf_names(balanced.trace) == [
        f"leaf_{index}" for index in range(1, 6)
    ]
    assert qm.ResourceEstimate.seq_all([]) == qm.ResourceEstimate.zero()


def test_seq_all_keeps_large_metadata_reduction_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata work grows by balanced levels rather than left-fold history."""
    estimate_count = 1024
    leaf = qm.ResourceEstimate(quality=qm.EstimateQuality.UPPER_BOUND)
    original = qm.ResourceEstimate.seq
    seq_calls = 0
    metadata_visits = 0

    def record_seq(
        self: qm.ResourceEstimate,
        other: qm.ResourceEstimate,
    ) -> qm.ResourceEstimate:
        """Count guarded-quality records visited by each binary composition."""
        nonlocal metadata_visits, seq_calls
        seq_calls += 1
        metadata_visits += len(self._guarded_qualities or ())
        metadata_visits += len(other._guarded_qualities or ())
        return original(self, other)

    monkeypatch.setattr(qm.ResourceEstimate, "seq", record_seq)
    combined = qm.ResourceEstimate.seq_all(leaf for _ in range(estimate_count))

    assert seq_calls == estimate_count - 1
    assert metadata_visits <= estimate_count * estimate_count.bit_length()
    assert len(combined._guarded_qualities or ()) == estimate_count
    assert combined.quality is qm.EstimateQuality.UPPER_BOUND


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
    assert estimate.quality is qm.EstimateQuality.EXACT
    assert not any(
        assumption.source == "PauliEvolveOp" for assumption in estimate.assumptions
    )
