"""Tests for canonical FTQC resource quantity metadata."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCCostModel,
    FTQCResearchSignal,
    FTQCResourceCategory,
    FTQCResourceComparisonRow,
    FTQCResourceComparisonSummary,
    FTQCResourceFormula,
    FTQCResourceQuantity,
    SurfaceCodeCostModel,
    compare_ftqc_resource_estimates,
    describe_ftqc_resource_quantity,
    estimate_qubitized_chemistry_qpe,
    estimate_qubitized_chemistry_qpe_from_model,
    iter_ftqc_research_signals,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
    summarize_pauli_hamiltonian,
)
from qamomile.circuit.estimator.algorithmic.ftqc_block_encoding import (
    BlockEncodingResource,
    estimate_qubitized_qpe_from_block_encoding,
)


def test_ftqc_quantity_specs_cover_core_resource_layers():
    """The FTQC quantity catalog covers problem, logical, and physical layers."""
    specs = iter_ftqc_resource_quantity_specs()
    quantities = {spec.quantity for spec in specs}
    categories = {spec.category for spec in specs}

    assert FTQCResourceQuantity.LAMBDA_NORM in quantities
    assert FTQCResourceQuantity.TARGET_PRECISION in quantities
    assert FTQCResourceQuantity.TRUNCATION_ERROR in quantities
    assert FTQCResourceQuantity.SYSTEM_QUBITS in quantities
    assert FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS in quantities
    assert FTQCResourceQuantity.QPE_REGISTER_QUBITS in quantities
    assert FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY in quantities
    assert FTQCResourceQuantity.QPE_REPETITIONS in quantities
    assert FTQCResourceQuantity.STATE_PREPARATION_TOFFOLI in quantities
    assert FTQCResourceQuantity.PREPARE_COST_TOFFOLI in quantities
    assert FTQCResourceQuantity.SELECT_COST_TOFFOLI in quantities
    assert FTQCResourceQuantity.REFLECTION_COST_TOFFOLI in quantities
    assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME in quantities
    assert FTQCResourceQuantity.TOFFOLI_GATES in quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in quantities
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


def test_ftqc_research_signals_map_sources_to_quantity_catalog():
    """Research signals explain why FTQC quantities are tracked."""
    signals = iter_ftqc_research_signals()
    signal_by_key = {signal.reference_key: signal for signal in signals}

    assert all(isinstance(signal, FTQCResearchSignal) for signal in signals)
    assert len(signal_by_key) == len(signals)
    assert "arXiv:2403.03502" in signal_by_key
    assert "arXiv:2601.08533" in signal_by_key
    assert "arXiv:2603.22778" in signal_by_key
    assert all(signal.url.startswith("https://arxiv.org/abs/") for signal in signals)

    scdf = signal_by_key["arXiv:2403.03502"]
    assert FTQCResourceQuantity.LAMBDA_NORM in scdf.quantities
    assert FTQCResourceQuantity.TOFFOLI_GATES in scdf.quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in scdf.quantities

    state_preparation = signal_by_key["arXiv:2601.08533"]
    assert (
        FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY
        in state_preparation.quantities
    )
    assert FTQCResourceQuantity.QPE_REPETITIONS in state_preparation.quantities
    assert state_preparation.to_dict()["quantities"][0] == (
        "state_preparation_success_probability"
    )
    early_ftqc = signal_by_key["arXiv:2603.22778"]
    assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME in early_ftqc.quantities


def test_ftqc_resource_formula_serializes_symbolic_derivations():
    """Formula metadata records quantity dependencies and expression text."""
    formula = FTQCResourceFormula(
        quantity="qpe_iterations",
        expression=sp.Symbol("lambda_norm") / sp.Symbol("target_precision"),
        depends_on=("lambda_norm", "target_precision"),
        description="QPE walk calls.",
        reference_keys=("arXiv:1610.06546",),
    )

    assert formula.quantity == FTQCResourceQuantity.QPE_ITERATIONS
    assert formula.depends_on == (
        FTQCResourceQuantity.LAMBDA_NORM,
        FTQCResourceQuantity.TARGET_PRECISION,
    )
    assert formula.to_dict() == {
        "quantity": "qpe_iterations",
        "expression": "lambda_norm/target_precision",
        "depends_on": ["lambda_norm", "target_precision"],
        "description": "QPE walk calls.",
        "reference_keys": ["arXiv:1610.06546"],
    }


def test_ftqc_resource_formula_rejects_invalid_inputs():
    """Formula metadata rejects unknown quantities and unsympifiable values."""
    with pytest.raises(ValueError, match="Unknown FTQC resource quantity"):
        FTQCResourceFormula(quantity="not-a-resource", expression=1)

    with pytest.raises(TypeError, match="expression"):
        FTQCResourceFormula(quantity="qpe_iterations", expression=object())


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
    assert estimate.resource_values()[
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
    ] == (estimate.logical_qubits * estimate.logical_depth)
    assert estimate.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS] == (
        estimate.physical_qubits * estimate.runtime_seconds
    )
    assert estimate.to_quantity_table()[0]["quantity"] == "logical_qubits"

    formula_table = estimate.to_formula_table()
    formula_by_quantity = {row["quantity"]: row for row in formula_table}

    assert formula_by_quantity["qpe_iterations"]["expression"] == (
        "lambda_norm/target_precision"
    )
    assert formula_by_quantity["toffoli_gates"]["depends_on"] == [
        "qpe_iterations",
        "walk_cost_toffoli",
    ]
    assert formula_by_quantity["logical_spacetime_volume"]["depends_on"] == [
        "logical_qubits",
        "logical_depth",
    ]
    assert formula_by_quantity["physical_qubit_seconds"]["depends_on"] == [
        "physical_qubits",
        "runtime_seconds",
    ]
    assert estimate.to_dict()["formulas"][0]["quantity"] == "qpe_iterations"


def test_formula_tables_are_preserved_by_estimate_transforms():
    """Estimate transforms preserve and transform formula metadata."""
    lam, eps = sp.symbols("lambda_norm target_precision", positive=True)
    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=2,
        lambda_norm=lam,
        precision=eps,
        walk_cost_toffoli=11,
    )

    substituted = estimate.substitute(lambda_norm=4, target_precision=2)

    assert substituted.formulas[0].expression == 2
    assert substituted.to_formula_table()[0]["expression"] == "2"
    assert estimate.simplify().formulas[0].quantity == (
        FTQCResourceQuantity.QPE_ITERATIONS
    )


def test_block_encoding_estimates_expose_formula_provenance():
    """Block-encoding estimates expose walk and QPE formula provenance."""
    block = BlockEncodingResource(
        system_qubits=4,
        normalization=100,
        select_cost_toffoli=7,
        prepare_cost_toffoli=5,
        reflection_cost_toffoli=3,
    )

    estimate = estimate_qubitized_qpe_from_block_encoding(block, precision=2)
    formulas = {formula.quantity: formula for formula in estimate.formulas}

    assert formulas[FTQCResourceQuantity.WALK_COST_TOFFOLI].expression == (
        2 * sp.Symbol("prepare_cost_toffoli", nonnegative=True)
        + sp.Symbol("select_cost_toffoli", nonnegative=True)
        + sp.Symbol("reflection_cost_toffoli", nonnegative=True)
    )
    assert formulas[FTQCResourceQuantity.QPE_ITERATIONS].reference_keys == (
        "arXiv:1610.06546",
    )


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
