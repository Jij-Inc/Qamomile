"""Tests for canonical FTQC resource quantity metadata."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCCostModel,
    FTQCResourceCategory,
    FTQCResourceComparisonRow,
    FTQCResourceComparisonSummary,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    compare_ftqc_resource_estimates,
    describe_ftqc_resource_quantity,
    estimate_qubitized_chemistry_qpe_from_model,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
    summarize_pauli_hamiltonian,
)


def test_ftqc_quantity_specs_cover_core_resource_layers():
    """The FTQC quantity catalog covers problem, logical, and physical layers."""
    specs = iter_ftqc_resource_quantity_specs()
    quantities = {spec.quantity for spec in specs}
    categories = {spec.category for spec in specs}

    assert FTQCResourceQuantity.LAMBDA_NORM in quantities
    assert FTQCResourceQuantity.TARGET_PRECISION in quantities
    assert FTQCResourceQuantity.TRUNCATION_ERROR in quantities
    assert FTQCResourceQuantity.TOFFOLI_GATES in quantities
    assert FTQCResourceQuantity.LOGICAL_ERROR_RATE in quantities
    assert FTQCResourceQuantity.PHYSICAL_ERROR_RATE in quantities
    assert FTQCResourceQuantity.THRESHOLD_ERROR_RATE in quantities
    assert FTQCResourceQuantity.TARGET_LOGICAL_FAILURE_PROBABILITY in quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBITS in quantities
    assert FTQCResourceQuantity.CODE_DISTANCE in quantities
    assert FTQCResourceQuantity.FACTORY_COUNT in quantities
    assert {
        FTQCResourceCategory.PROBLEM,
        FTQCResourceCategory.ALGORITHM,
        FTQCResourceCategory.LOGICAL,
        FTQCResourceCategory.PHYSICAL,
        FTQCResourceCategory.ARCHITECTURE,
    }.issubset(categories)
    assert len(quantities) == len(specs)


def test_describe_ftqc_resource_quantity_normalizes_strings():
    """String quantity keys resolve to metadata with stable units."""
    spec = describe_ftqc_resource_quantity("lambda_norm")

    assert spec.quantity == FTQCResourceQuantity.LAMBDA_NORM
    assert spec.category == FTQCResourceCategory.PROBLEM
    assert spec.unit == "energy"


def test_describe_ftqc_resource_quantity_rejects_unknown_key():
    """Unknown quantity keys fail with a finite-set validation error."""
    with pytest.raises(ValueError, match="Unknown FTQC resource quantity"):
        describe_ftqc_resource_quantity("not-a-resource")


def test_ftqc_models_expose_canonical_resource_values():
    """Hamiltonian, model, cost, and estimate values share canonical keys."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    model = ChemistryQPEModel(
        hamiltonian=summary,
        method=ChemistryQPEMethod.SPARSE,
        walk_cost_toffoli=11,
    )
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )

    estimate = estimate_qubitized_chemistry_qpe_from_model(
        model,
        precision=1,
        cost_model=cost,
    )

    assert (
        sp.simplify(summary.resource_values()[FTQCResourceQuantity.LAMBDA_NORM] - 5)
        == 0
    )
    assert model.resource_values()[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 11
    assert model.resource_values()[FTQCResourceQuantity.TRUNCATION_ERROR] == 0
    assert (
        cost.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL] == 100
    )
    assert (
        sp.simplify(estimate.resource_values()[FTQCResourceQuantity.TOFFOLI_GATES] - 55)
        == 0
    )
    assert estimate.resource_values()[FTQCResourceQuantity.TARGET_PRECISION] == 1
    assert estimate.to_quantity_table()[0]["quantity"] == "logical_qubits"


def test_surface_code_model_exposes_raw_and_derived_resource_values():
    """Surface-code models expose raw knobs and derived architecture values."""
    architecture = SurfaceCodeCostModel(
        code_distance=7,
        physical_cycle_time_seconds=sp.Float("2e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=3,
        factory_count=5,
        physical_qubits_per_factory=1000,
        factory_cycles_per_toffoli=4,
    )
    values = architecture.resource_values()

    assert values[FTQCResourceQuantity.CODE_DISTANCE] == 7
    assert values[FTQCResourceQuantity.FACTORY_COUNT] == 5
    assert values[FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL] == 98
    assert values[FTQCResourceQuantity.FACTORY_QUBITS] == 5000
    expected_throughput = sp.Float("5e6") / 168
    assert sp.Abs(
        values[FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND] - expected_throughput
    ) < sp.Float("1e-12")


def test_compare_ftqc_resource_estimates_reports_ratios_and_reductions():
    """Comparison rows quantify savings for a lower-cost FTQC candidate."""
    baseline = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    candidate = baseline.with_lambda_scale(sp.Rational(1, 2), source="compressed")
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    baseline_estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=baseline,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=10,
        ),
        precision=1,
        cost_model=cost,
    )
    candidate_estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=candidate,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=11,
        ),
        precision=1,
        cost_model=cost,
    )

    rows = compare_ftqc_resource_estimates(
        baseline_estimate,
        candidate_estimate,
        quantities=(
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
        ),
    )

    assert rows[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
    assert sp.simplify(rows[0].ratio - sp.Rational(1, 2)) == 0
    assert sp.simplify(rows[0].reduction - sp.Rational(1, 2)) == 0
    assert sp.simplify(rows[1].ratio - sp.Rational(11, 20)) == 0
    assert rows[1].to_dict()["unit"] == "Toffoli gates"


def test_compare_ftqc_resource_estimates_defaults_to_common_quantities():
    """Default comparison uses the canonical intersection of exposed values."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + qm_o.X(1))
    rows = compare_ftqc_resource_estimates(
        summary,
        summary.with_lambda_scale(sp.Rational(1, 2)),
    )

    assert [row.quantity for row in rows] == [
        FTQCResourceQuantity.N_SPIN_ORBITALS,
        FTQCResourceQuantity.N_PAULI_TERMS,
        FTQCResourceQuantity.LAMBDA_NORM,
        FTQCResourceQuantity.MAX_LOCALITY,
    ]
    lambda_row = rows[2]
    assert sp.simplify(lambda_row.ratio - sp.Rational(1, 2)) == 0


def test_compare_ftqc_resource_estimates_rejects_missing_or_zero_baseline():
    """Invalid comparison requests fail before returning misleading ratios."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="missing"):
        compare_ftqc_resource_estimates(
            summary,
            summary,
            quantities=(FTQCResourceQuantity.TOFFOLI_GATES,),
        )

    with pytest.raises(ValueError, match="zero baseline"):
        compare_ftqc_resource_estimates(
            summary.with_lambda_scale(0),
            summary,
            quantities=(FTQCResourceQuantity.LAMBDA_NORM,),
        )


def test_summarize_ftqc_resource_comparison_groups_review_drivers():
    """Comparison summaries group improvements, regressions, and ties."""
    baseline = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    candidate = baseline.with_lambda_scale(sp.Rational(1, 2), source="compressed")
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    baseline_estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=baseline,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=10,
        ),
        precision=1,
        cost_model=cost,
    )
    candidate_estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=candidate,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=30,
        ),
        precision=1,
        cost_model=cost,
    )

    summary = summarize_ftqc_resource_comparison(
        baseline_estimate,
        candidate_estimate,
        quantities=(
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
            FTQCResourceQuantity.LOGICAL_QUBITS,
        ),
    )

    assert [row.quantity for row in summary.smaller] == [
        FTQCResourceQuantity.QPE_ITERATIONS
    ]
    assert [row.quantity for row in summary.larger] == [
        FTQCResourceQuantity.TOFFOLI_GATES
    ]
    assert [row.quantity for row in summary.unchanged] == [
        FTQCResourceQuantity.LOGICAL_QUBITS
    ]
    assert summary.symbolic == ()
    assert summary.to_dict()["counts"] == {
        "smaller": 1,
        "larger": 1,
        "unchanged": 1,
        "symbolic": 0,
    }


def test_comparison_summary_keeps_undecidable_symbolic_changes_separate():
    """Symbolic comparison signs remain undecided until assumptions are added."""
    x = sp.Symbol("x")
    row = FTQCResourceComparisonRow(
        quantity=FTQCResourceQuantity.TOFFOLI_GATES,
        baseline=1,
        candidate=x,
        ratio=x,
        reduction=1 - x,
        label="Toffoli gates",
        unit="Toffoli gates",
        category=FTQCResourceCategory.LOGICAL,
    )

    summary = FTQCResourceComparisonSummary.from_rows((row,))

    assert summary.smaller == ()
    assert summary.larger == ()
    assert summary.unchanged == ()
    assert summary.symbolic == (row,)
