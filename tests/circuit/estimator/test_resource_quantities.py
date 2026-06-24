"""Tests for canonical resource quantity metadata and comparisons."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.resource_estimation import (
    FTQCCostModel,
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    ResourceCategory,
    ResourceQuantity,
    SurfaceCodeCostModel,
    compare_resource_values,
    describe_resource_quantity,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    iter_resource_quantity_specs,
    resource_values_from_estimate,
    summarize_pauli_hamiltonian,
)


def test_quantity_specs_cover_core_resource_layers():
    """The quantity catalog covers problem, logical, physical, and architecture layers."""
    specs = iter_resource_quantity_specs()
    quantities = {spec.quantity for spec in specs}
    categories = {spec.category for spec in specs}

    assert ResourceQuantity.LAMBDA_NORM in quantities
    assert ResourceQuantity.NON_CLIFFORD_COUNT in quantities
    assert ResourceQuantity.LOGICAL_SPACETIME_VOLUME in quantities
    assert ResourceQuantity.PHYSICAL_QUBITS in quantities
    assert ResourceQuantity.PHYSICAL_QUBIT_SECONDS in quantities
    assert ResourceQuantity.CODE_DISTANCE in quantities
    assert ResourceQuantity.FACTORY_COUNT in quantities
    assert {
        ResourceCategory.PROBLEM,
        ResourceCategory.ALGORITHM,
        ResourceCategory.LOGICAL,
        ResourceCategory.PHYSICAL,
        ResourceCategory.ARCHITECTURE,
    }.issubset(categories)
    assert len(quantities) == len(specs)


def test_describe_resource_quantity_normalizes_strings():
    """String quantity keys resolve to metadata with stable units."""
    spec = describe_resource_quantity("lambda_norm")

    assert spec.quantity == ResourceQuantity.LAMBDA_NORM
    assert spec.category == ResourceCategory.PROBLEM
    assert spec.unit == "energy"


def test_describe_resource_quantity_rejects_unknown_key():
    """Unknown quantity keys fail with a finite-set validation error."""
    with pytest.raises(ValueError, match="Unknown resource quantity"):
        describe_resource_quantity("not-a-resource")


def test_surface_code_model_exposes_raw_and_derived_resource_values():
    """Surface-code models expose raw knobs and derived architecture values."""
    architecture = SurfaceCodeCostModel(
        code_distance=7,
        physical_cycle_time_seconds=sp.Float("2e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=3,
        factory_count=5,
        physical_qubits_per_factory=1000,
        factory_cycles_per_non_clifford=4,
    )
    values = architecture.resource_values()

    assert values["code_distance"] == 7
    assert values["factory_count"] == 5
    assert values["physical_qubits_per_logical"] == 98
    assert values["factory_qubits"] == 5000
    expected_throughput = sp.Float("5e6") / 168
    assert sp.Abs(
        values["non_clifford_throughput_per_second"] - expected_throughput
    ) < sp.Float("1e-12")


def test_compare_resource_values_reports_ratios_and_reductions():
    """Comparison rows quantify savings for a lower-cost candidate."""
    baseline = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    candidate = baseline.with_lambda_scale(sp.Rational(1, 2), source="compressed")

    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=(
            ResourceQuantity.LAMBDA_NORM,
            "n_pauli_terms",
        ),
    )

    assert rows[0].quantity == ResourceQuantity.LAMBDA_NORM
    assert sp.simplify(rows[0].ratio - sp.Rational(1, 2)) == 0
    assert sp.simplify(rows[0].reduction - sp.Rational(1, 2)) == 0
    assert rows[0].to_dict()["unit"] == "energy"
    assert rows[1].quantity == ResourceQuantity.N_PAULI_TERMS
    assert rows[1].ratio == 1


def test_compare_physical_estimates_uses_canonical_values():
    """Physical estimates can be compared through the same quantity API."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    baseline_workload = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
    )
    candidate_workload = HamiltonianQPEWorkload(
        hamiltonian=summary.with_lambda_scale(sp.Rational(1, 2)),
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
    )
    cost_model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        non_clifford_throughput_per_second=sp.Float("1e5"),
    )
    baseline = estimate_physical_resources(
        estimate_qubitized_qpe_resources_from_workload(
            baseline_workload,
            precision=1,
        ),
        cost_model,
    )
    candidate = estimate_physical_resources(
        estimate_qubitized_qpe_resources_from_workload(
            candidate_workload,
            precision=1,
        ),
        cost_model,
    )

    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=(
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
        ),
    )

    assert rows[0].quantity == ResourceQuantity.QPE_ITERATIONS
    assert sp.simplify(rows[0].ratio - sp.Rational(1, 2)) == 0
    assert rows[1].quantity == ResourceQuantity.NON_CLIFFORD_COUNT
    assert sp.simplify(rows[1].ratio - sp.Rational(1, 2)) == 0


def test_logical_estimates_expose_canonical_values_for_comparison():
    """Logical ResourceEstimate objects compare without physical lifting."""
    baseline = estimate_qubitized_qpe_resources(
        n_qubits=16,
        lambda_norm=100,
        precision=sp.Rational(1, 2),
        walk_cost_toffoli=20,
        representation=HamiltonianRepresentation.DOUBLE_FACTORIZATION,
        second_factor_rank=4,
    )
    candidate = estimate_qubitized_qpe_resources(
        n_qubits=16,
        lambda_norm=25,
        precision=sp.Rational(1, 2),
        walk_cost_toffoli=10,
        representation=HamiltonianRepresentation.DOUBLE_FACTORIZATION,
        second_factor_rank=4,
    )
    values = resource_values_from_estimate(candidate)
    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=(
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
            ResourceQuantity.LOGICAL_SPACETIME_VOLUME,
        ),
    )

    assert values["logical_qubits"] == candidate.qubits
    assert values["logical_depth"] == candidate.gates.total
    assert values["qpe_iterations"] == candidate.gates.oracle_calls["qpe_iterations"]
    assert rows[0].quantity == ResourceQuantity.QPE_ITERATIONS
    assert sp.simplify(rows[0].ratio - sp.Rational(1, 4)) == 0
    assert rows[1].quantity == ResourceQuantity.NON_CLIFFORD_COUNT
    assert sp.simplify(rows[1].ratio - sp.Rational(1, 8)) == 0
    assert rows[2].quantity == ResourceQuantity.LOGICAL_SPACETIME_VOLUME
    assert sp.simplify(rows[2].ratio - sp.Rational(1, 8)) == 0


def test_compare_resource_values_accepts_mappings_and_rejects_bad_estimate_overrides():
    """Raw mappings are valid providers, while negative logical overrides fail."""
    baseline = {
        ResourceQuantity.LOGICAL_QUBITS: sp.Integer(10),
        "logical_depth": sp.Integer(100),
    }
    candidate = {
        "logical_qubits": sp.Integer(5),
        ResourceQuantity.LOGICAL_DEPTH: sp.Integer(25),
    }
    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=(ResourceQuantity.LOGICAL_QUBITS, ResourceQuantity.LOGICAL_DEPTH),
    )
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=2,
    )

    assert [row.quantity for row in rows] == [
        ResourceQuantity.LOGICAL_QUBITS,
        ResourceQuantity.LOGICAL_DEPTH,
    ]
    assert rows[0].ratio == sp.Rational(1, 2)
    assert rows[1].ratio == sp.Rational(1, 4)
    with pytest.raises(ValueError, match="logical_depth"):
        resource_values_from_estimate(logical, logical_depth=-1)
    with pytest.raises(TypeError, match="resource value providers"):
        compare_resource_values(object(), candidate)


def test_physical_estimate_exposes_spacetime_values():
    """Physical estimates expose logical and physical space-time proxies."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    logical = estimate_qubitized_qpe_resources_from_workload(
        HamiltonianQPEWorkload(
            hamiltonian=summary,
            representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
            walk_cost_toffoli=10,
        ),
        precision=1,
    )
    physical = estimate_physical_resources(
        logical,
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=sp.Float("1e-6"),
            factory_qubits=10,
            non_clifford_throughput_per_second=sp.Float("1e5"),
        ),
    )
    values = physical.resource_values()

    assert values["logical_spacetime_volume"] == (
        physical.logical.qubits * physical.logical_depth
    )
    assert values["physical_qubit_seconds"] == (
        physical.physical_qubits * physical.runtime_seconds
    )


def test_compare_resource_values_defaults_to_common_quantities():
    """Default comparison uses comparable nonzero-baseline common values."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + qm_o.X(1))
    rows = compare_resource_values(
        summary,
        summary.with_lambda_scale(sp.Rational(1, 2)),
    )

    assert [row.quantity for row in rows] == [
        ResourceQuantity.N_QUBITS,
        ResourceQuantity.N_PAULI_TERMS,
        ResourceQuantity.LAMBDA_NORM,
        ResourceQuantity.MAX_LOCALITY,
    ]
    lambda_row = rows[2]
    assert sp.simplify(lambda_row.ratio - sp.Rational(1, 2)) == 0


def test_compare_resource_values_default_skips_zero_baseline_quantities():
    """Optional zero-baseline quantities do not break default comparisons."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + qm_o.X(1))
    baseline = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
    )
    candidate = HamiltonianQPEWorkload(
        hamiltonian=summary.with_lambda_scale(sp.Rational(1, 2)),
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=5,
        representation_error=sp.Rational(1, 10),
        qpe_register_qubits=2,
    )
    rows = compare_resource_values(baseline, candidate)

    quantities = [row.quantity for row in rows]
    assert ResourceQuantity.REPRESENTATION_ERROR not in quantities
    assert ResourceQuantity.QPE_REGISTER_QUBITS not in quantities
    assert ResourceQuantity.LAMBDA_NORM in quantities
    assert ResourceQuantity.WALK_COST_TOFFOLI in quantities


def test_compare_resource_values_rejects_missing_or_zero_baseline():
    """Invalid comparison requests fail before returning misleading ratios."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="missing"):
        compare_resource_values(
            summary,
            summary,
            quantities=(ResourceQuantity.NON_CLIFFORD_COUNT,),
        )

    with pytest.raises(ValueError, match="zero baseline"):
        compare_resource_values(
            summary.with_lambda_scale(0),
            summary,
            quantities=(ResourceQuantity.LAMBDA_NORM,),
        )
