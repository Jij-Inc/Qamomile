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
    FTQCResearchSignalCoverage,
    FTQCResearchSignalCoverageReport,
    FTQCResourceAggregationRule,
    FTQCResourceBudgetReport,
    FTQCResourceCategory,
    FTQCResourceChangeDirection,
    FTQCResourceComparisonReport,
    FTQCResourceComparisonRow,
    FTQCResourceComparisonSummary,
    FTQCResourceConstraint,
    FTQCResourceConstraintResult,
    FTQCResourceConstraintSense,
    FTQCResourceConstraintStatus,
    FTQCResourceDriverReport,
    FTQCResourceFormula,
    FTQCResourceParetoReport,
    FTQCResourceParetoRow,
    FTQCResourcePlan,
    FTQCResourcePlanStep,
    FTQCResourceProfile,
    FTQCResourceProfileSpec,
    FTQCResourceQuantity,
    FTQCResourceReportBundle,
    FTQCResourceReportKind,
    FTQCResourceReportSnapshot,
    FTQCResourceReviewFinding,
    FTQCResourceScenario,
    FTQCResourceScenarioReport,
    FTQCResourceScenarioRow,
    HamiltonianResourceReduction,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    audit_ftqc_research_signal_catalog,
    audit_ftqc_research_signal_coverage,
    build_ftqc_research_signal_report,
    build_ftqc_resource_comparison_report,
    build_ftqc_resource_driver_report,
    build_ftqc_resource_pareto_report,
    build_ftqc_resource_report_bundle,
    build_ftqc_resource_report_snapshot,
    build_ftqc_resource_review_findings,
    build_ftqc_resource_scenario_report,
    compare_ftqc_resource_estimates,
    default_ftqc_resource_aggregation_rule,
    describe_ftqc_research_signal,
    describe_ftqc_resource_quantity,
    estimate_qubitized_chemistry_qpe,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    evaluate_ftqc_resource_constraints,
    ftqc_resource_profile_quantities,
    iter_ftqc_research_signals,
    iter_ftqc_resource_profile_specs,
    iter_ftqc_resource_quantity_specs,
    summarize_ftqc_resource_comparison,
    summarize_pauli_hamiltonian,
)
from qamomile.circuit.estimator.algorithmic.ftqc_block_encoding import (
    BlockEncodingResource,
    estimate_qubitized_qpe_from_block_encoding,
    plan_qubitized_qpe_from_block_encoding,
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
    profile_catalog = {
        profile.profile: set(profile.quantities)
        for profile in iter_ftqc_resource_profile_specs()
    }

    scdf = signal_by_key["arXiv:2403.03502"]
    assert FTQCResourceQuantity.LAMBDA_NORM in scdf.quantities
    assert FTQCResourceQuantity.TOFFOLI_GATES in scdf.quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in scdf.quantities
    assert scdf.profiles == (
        FTQCResourceProfile.BLOCK_ENCODING,
        FTQCResourceProfile.CHEMISTRY_QPE,
        FTQCResourceProfile.SPACETIME,
    )
    assert set(scdf.quantities).issubset(
        set().union(*(profile_catalog[profile] for profile in scdf.profiles))
    )

    state_preparation = signal_by_key["arXiv:2601.08533"]
    assert (
        FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY
        in state_preparation.quantities
    )
    assert FTQCResourceQuantity.QPE_REPETITIONS in state_preparation.quantities
    assert state_preparation.to_dict()["quantities"][0] == (
        "state_preparation_success_probability"
    )
    assert state_preparation.to_dict()["profiles"] == [
        "chemistry_qpe",
        "spacetime",
    ]
    assert set(state_preparation.quantities).issubset(
        set().union(
            *(profile_catalog[profile] for profile in state_preparation.profiles)
        )
    )
    early_ftqc = signal_by_key["arXiv:2603.22778"]
    assert FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME in early_ftqc.quantities
    assert early_ftqc.profiles == (
        FTQCResourceProfile.CHEMISTRY_QPE,
        FTQCResourceProfile.SPACETIME,
        FTQCResourceProfile.ARCHITECTURE,
    )
    assert set(early_ftqc.quantities).issubset(
        set().union(*(profile_catalog[profile] for profile in early_ftqc.profiles))
    )


def test_ftqc_resource_profiles_group_review_quantities():
    """Resource profiles provide reusable quantity bundles for reviews."""
    profiles = iter_ftqc_resource_profile_specs()
    profile_by_key = {profile.profile: profile for profile in profiles}
    quantity_catalog = {spec.quantity for spec in iter_ftqc_resource_quantity_specs()}

    assert all(isinstance(profile, FTQCResourceProfileSpec) for profile in profiles)
    assert set(profile_by_key) == {
        FTQCResourceProfile.CHEMISTRY_QPE,
        FTQCResourceProfile.BLOCK_ENCODING,
        FTQCResourceProfile.SPACETIME,
        FTQCResourceProfile.ERROR_BUDGET,
        FTQCResourceProfile.ARCHITECTURE,
    }
    assert all(
        quantity in quantity_catalog
        for profile in profiles
        for quantity in profile.quantities
    )

    chemistry_profile = profile_by_key[FTQCResourceProfile.CHEMISTRY_QPE]
    assert FTQCResourceQuantity.LAMBDA_NORM in chemistry_profile.quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in chemistry_profile.quantities
    assert chemistry_profile.to_dict()["profile"] == "chemistry_qpe"
    assert ftqc_resource_profile_quantities("spacetime") == (
        FTQCResourceQuantity.LOGICAL_QUBITS,
        FTQCResourceQuantity.LOGICAL_DEPTH,
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
        FTQCResourceQuantity.PHYSICAL_QUBITS,
        FTQCResourceQuantity.RUNTIME_SECONDS,
        FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    )


def test_ftqc_resource_profiles_match_concrete_resource_objects():
    """Focused resource profiles match the objects that expose their values."""
    distance_budget = SurfaceCodeDistanceBudget(
        physical_error_rate=sp.Float("1e-3"),
        threshold_error_rate=sp.Float("1e-2"),
        target_logical_failure_probability=sp.Float("1e-2"),
        logical_operation_budget=sp.Integer(10**8),
    )
    architecture = SurfaceCodeCostModel(
        code_distance=7,
        physical_cycle_time_seconds=sp.Float("2e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=3,
        factory_count=5,
        physical_qubits_per_factory=1000,
        factory_cycles_per_toffoli=4,
    )

    assert set(ftqc_resource_profile_quantities(FTQCResourceProfile.ERROR_BUDGET)) <= (
        set(distance_budget.resource_values())
    )
    assert set(ftqc_resource_profile_quantities(FTQCResourceProfile.ARCHITECTURE)) <= (
        set(architecture.resource_values())
    )


def test_ftqc_resource_profile_rejects_unknown_key():
    """Unknown resource profile keys fail with finite-set diagnostics."""
    with pytest.raises(ValueError, match="Unknown FTQC resource profile"):
        ftqc_resource_profile_quantities("not-a-profile")


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


def test_ftqc_resource_plan_composes_abstract_subroutines():
    """Resource plans compose repeated FTQC subroutines without circuit lowering."""
    prepare = FTQCResourcePlanStep(
        "prepare_filter",
        {
            FTQCResourceQuantity.LOGICAL_QUBITS: 12,
            FTQCResourceQuantity.TOFFOLI_GATES: 7,
            FTQCResourceQuantity.LOGICAL_DEPTH: 11,
            FTQCResourceQuantity.TARGET_PRECISION: sp.Float("0.001"),
        },
        repetitions=3,
        label="State preparation and filtering",
    )
    phase_estimation = FTQCResourcePlanStep(
        "phase_estimation",
        {
            "logical_qubits": 18,
            "toffoli_gates": 100,
            "logical_depth": 120,
            "runtime_seconds": 20,
            "physical_qubits": 4000,
            "target_precision": sp.Float("0.001"),
        },
        repetitions=2,
    )

    plan = FTQCResourcePlan(
        (prepare, phase_estimation),
        title="Filtered QPE plan",
    )
    values = plan.resource_values()

    assert values[FTQCResourceQuantity.TOFFOLI_GATES] == 221
    assert values[FTQCResourceQuantity.LOGICAL_DEPTH] == 273
    assert values[FTQCResourceQuantity.LOGICAL_QUBITS] == 18
    assert values[FTQCResourceQuantity.PHYSICAL_QUBITS] == 4000
    assert values[FTQCResourceQuantity.TARGET_PRECISION] == sp.Float("0.001")
    assert default_ftqc_resource_aggregation_rule("logical_qubits") == (
        FTQCResourceAggregationRule.PEAK
    )
    assert default_ftqc_resource_aggregation_rule("toffoli_gates") == (
        FTQCResourceAggregationRule.ADD
    )
    assert plan.to_quantity_table()[0]["quantity"] == "target_precision"
    assert plan.to_dict()["steps"][0]["label"] == "State preparation and filtering"
    assert plan.formulas() == ()
    assert plan.reference_keys() == ()

    budget = evaluate_ftqc_resource_constraints(
        plan,
        (
            FTQCResourceConstraint("logical_qubits", 20),
            FTQCResourceConstraint("toffoli_gates", 200),
        ),
    )

    assert budget.satisfied[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
    assert budget.violated[0].quantity == FTQCResourceQuantity.TOFFOLI_GATES


def test_ftqc_resource_plan_supports_symbolic_repetition_and_overrides():
    """Plans keep symbolic repetition factors and custom aggregation rules."""
    success_probability = sp.Symbol("p_success", positive=True)
    per_attempt_toffoli = sp.Symbol("t_attempt", nonnegative=True)
    filtering = FTQCResourcePlanStep(
        "filtering",
        {
            "toffoli_gates": per_attempt_toffoli,
            "runtime_seconds": 10,
            "logical_qubits": sp.Symbol("filter_qubits", positive=True),
        },
        repetitions=1 / success_probability,
    )
    verification = FTQCResourcePlanStep(
        "verification",
        {
            "runtime_seconds": 20,
            "logical_qubits": sp.Symbol("verify_qubits", positive=True),
        },
    )
    plan = FTQCResourcePlan(
        (filtering, verification),
        aggregation={"runtime_seconds": FTQCResourceAggregationRule.PEAK},
    )
    values = plan.resource_values()

    assert (
        sp.simplify(
            values[FTQCResourceQuantity.TOFFOLI_GATES]
            - per_attempt_toffoli / success_probability
        )
        == 0
    )
    assert values[FTQCResourceQuantity.RUNTIME_SECONDS] == sp.Max(
        20,
        10 / success_probability,
    )
    assert values[FTQCResourceQuantity.LOGICAL_QUBITS] == sp.Max(
        sp.Symbol("filter_qubits", positive=True),
        sp.Symbol("verify_qubits", positive=True),
    )


def test_ftqc_resource_plan_rejects_invalid_composition_inputs():
    """Resource plans reject unknown rules and inconsistent metadata."""
    with pytest.raises(ValueError, match="aggregation rule"):
        FTQCResourcePlanStep(
            "bad",
            {"toffoli_gates": 1},
            aggregation={"toffoli_gates": "not-a-rule"},
        )

    with pytest.raises(ValueError, match="repetitions"):
        FTQCResourcePlanStep("bad", {"toffoli_gates": 1}, repetitions=-1)

    plan = FTQCResourcePlan(
        (
            FTQCResourcePlanStep("a", {"target_precision": sp.Float("0.001")}),
            FTQCResourcePlanStep("b", {"target_precision": sp.Float("0.002")}),
        )
    )
    with pytest.raises(ValueError, match="Conflicting FTQC resource values"):
        plan.resource_values()

    with pytest.raises(TypeError, match="FTQCResourcePlanStep"):
        FTQCResourcePlan((object(),))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="formulas"):
        FTQCResourcePlanStep(
            "bad",
            {"toffoli_gates": 1},
            formulas=(object(),),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="reference_keys"):
        FTQCResourcePlanStep(
            "bad",
            {"toffoli_gates": 1},
            reference_keys=(object(),),  # type: ignore[arg-type]
        )


def test_ftqc_resource_plan_serializes_step_provenance():
    """Plans preserve formulas and reference keys for review reports."""
    formula = FTQCResourceFormula(
        quantity="toffoli_gates",
        expression=sp.Symbol("qpe_iterations") * sp.Symbol("walk_cost_toffoli"),
        depends_on=("qpe_iterations", "walk_cost_toffoli"),
        description="Multiply walk calls by per-walk Toffoli cost.",
        reference_keys=("arXiv:1610.06546",),
    )
    repeated_formula = FTQCResourceFormula(
        quantity="toffoli_gates",
        expression=sp.Symbol("qpe_iterations") * sp.Symbol("walk_cost_toffoli"),
        depends_on=("qpe_iterations", "walk_cost_toffoli"),
        description="Multiply walk calls by per-walk Toffoli cost.",
        reference_keys=("arXiv:1610.06546",),
    )
    prepare = FTQCResourcePlanStep(
        "prepare",
        {"toffoli_gates": 10},
        formulas=(formula,),
        reference_keys=("internal:loader",),
    )
    walk = FTQCResourcePlanStep(
        "walk",
        {"toffoli_gates": 20},
        formulas=(repeated_formula,),
        reference_keys=("arXiv:1610.06546",),
    )
    plan = FTQCResourcePlan((prepare, walk))

    assert plan.formulas() == (formula,)
    assert plan.reference_keys() == ("internal:loader", "arXiv:1610.06546")
    serialized = plan.to_dict()
    assert serialized["formulas"][0]["quantity"] == "toffoli_gates"
    assert serialized["steps"][0]["formulas"][0]["depends_on"] == [
        "qpe_iterations",
        "walk_cost_toffoli",
    ]
    assert serialized["reference_keys"] == ["internal:loader", "arXiv:1610.06546"]


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


def test_block_encoding_qpe_plan_matches_logical_estimate_resources():
    """Block-encoding plans expose subroutines matching logical estimates."""
    block = BlockEncodingResource(
        system_qubits=4,
        normalization=100,
        select_cost_toffoli=7,
        prepare_cost_toffoli=5,
        reflection_cost_toffoli=3,
        ancilla_qubits=2,
        name="toy_block",
    )

    plan = plan_qubitized_qpe_from_block_encoding(
        block,
        precision=2,
        qpe_register_qubits=3,
    )
    estimate = estimate_qubitized_qpe_from_block_encoding(
        block,
        precision=2,
        qpe_register_qubits=3,
    )
    plan_values = plan.resource_values()

    assert len(plan.steps) == 2
    assert plan.steps[0].name == "block_encoding_contract"
    assert plan.steps[1].name == "qubitized_walk_qpe"
    assert plan.steps[1].repetitions == 50
    assert plan_values[FTQCResourceQuantity.QPE_ITERATIONS] == (estimate.qpe_iterations)
    assert plan_values[FTQCResourceQuantity.TOFFOLI_GATES] == estimate.toffoli_gates
    assert plan_values[FTQCResourceQuantity.LOGICAL_DEPTH] == estimate.logical_depth
    assert plan_values[FTQCResourceQuantity.LOGICAL_QUBITS] == (estimate.logical_qubits)
    assert plan_values[FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME] == (
        estimate.logical_qubits * estimate.logical_depth
    )
    assert plan_values[FTQCResourceQuantity.WALK_COST_TOFFOLI] == 20
    assert plan.to_dict()["steps"][1]["label"] == "Repeated qubitized walk"
    assert plan.reference_keys() == ("arXiv:1610.06546",)
    assert [formula.quantity for formula in plan.formulas()] == [
        FTQCResourceQuantity.WALK_COST_TOFFOLI,
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
        FTQCResourceQuantity.LOGICAL_DEPTH,
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    ]
    assert plan.to_dict()["steps"][0]["formulas"][0]["quantity"] == (
        "walk_cost_toffoli"
    )


def test_block_encoding_qpe_plan_rejects_invalid_precision_inputs():
    """Block-encoding QPE plans reject invalid precision and readout sizes."""
    block = BlockEncodingResource(
        system_qubits=4,
        normalization=100,
        select_cost_toffoli=7,
    )

    with pytest.raises(ValueError, match="precision"):
        plan_qubitized_qpe_from_block_encoding(block, precision=0)

    with pytest.raises(ValueError, match="qpe_register_qubits"):
        plan_qubitized_qpe_from_block_encoding(
            block,
            precision=1,
            qpe_register_qubits=-1,
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


def test_compare_ftqc_resource_estimates_accepts_profile():
    """Comparison helpers accept standard profiles directly."""
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
        profile=FTQCResourceProfile.SPACETIME,
    )

    assert [row.quantity for row in rows] == list(
        ftqc_resource_profile_quantities("spacetime")
    )
    assert rows[2].quantity == FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
    assert rows[-1].quantity == FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS


def test_compare_ftqc_resource_estimates_appends_profile_without_duplicates():
    """Explicit quantities are kept before profile quantities without repeats."""
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
            FTQCResourceQuantity.LOGICAL_QUBITS,
        ),
        profile="spacetime",
    )

    assert [row.quantity for row in rows] == [
        FTQCResourceQuantity.QPE_ITERATIONS,
        FTQCResourceQuantity.LOGICAL_QUBITS,
        FTQCResourceQuantity.LOGICAL_DEPTH,
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
        FTQCResourceQuantity.PHYSICAL_QUBITS,
        FTQCResourceQuantity.RUNTIME_SECONDS,
        FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
    ]


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


def test_summarize_ftqc_resource_comparison_accepts_profile():
    """Comparison summaries accept standard review profiles directly."""
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

    summary = summarize_ftqc_resource_comparison(
        baseline_estimate,
        candidate_estimate,
        profile=FTQCResourceProfile.SPACETIME,
    )

    assert summary.rows[0].quantity == FTQCResourceQuantity.LOGICAL_QUBITS
    assert summary.rows[-1].quantity == FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS


def test_build_ftqc_resource_comparison_report_labels_profiled_rows():
    """Comparison reports preserve labels, profile, rows, and group counts."""
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

    report = build_ftqc_resource_comparison_report(
        baseline_estimate,
        candidate_estimate,
        title="Sparse versus compressed",
        baseline_label="Sparse",
        candidate_label="Compressed",
        quantities=(FTQCResourceQuantity.QPE_ITERATIONS,),
        profile="spacetime",
    )
    report_dict = report.to_dict()
    row_table = report.to_row_table()

    assert isinstance(report, FTQCResourceComparisonReport)
    assert report.profile == FTQCResourceProfile.SPACETIME
    assert report.summary.rows[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
    assert report.summary.rows[-1].quantity == (
        FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS
    )
    assert report_dict["title"] == "Sparse versus compressed"
    assert report_dict["profile"] == "spacetime"
    assert report_dict["quantities"][0] == "qpe_iterations"
    assert report_dict["counts"]["smaller"] == len(report.summary.smaller)
    assert row_table[0]["baseline_label"] == "Sparse"
    assert row_table[0]["candidate_label"] == "Compressed"
    assert row_table[0]["quantity"] == "qpe_iterations"


def test_build_ftqc_research_signal_report_selects_available_quantities():
    """Research-signal reports compare available quantities and audit gaps."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1) + 3 * qm_o.Z(2),
        source="plain_trotter_lcu",
    )
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    baseline = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        rotation_synthesis_t_gates=3,
        cost_model=cost,
    )
    candidate = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
        resource_reduction=HamiltonianResourceReduction(
            lambda_norm_factor=sp.Rational(1, 10)
        ),
        cost_model=cost,
    )

    signal = describe_ftqc_research_signal("arXiv:2603.22778")
    report = build_ftqc_research_signal_report(
        "arXiv:2603.22778",
        baseline,
        candidate,
        baseline_label="Plain Trotter",
        candidate_label="UWC Trotter",
    )
    row_quantities = [row.quantity for row in report.summary.rows]
    row_table = report.to_row_table()

    assert signal.profiles == (
        FTQCResourceProfile.CHEMISTRY_QPE,
        FTQCResourceProfile.SPACETIME,
        FTQCResourceProfile.ARCHITECTURE,
    )
    assert report.title == "arXiv:2603.22778 resource signal review"
    assert report.profile is None
    assert row_quantities[0] == FTQCResourceQuantity.LAMBDA_NORM
    assert row_quantities[1] == FTQCResourceQuantity.QPE_ITERATIONS
    assert FTQCResourceQuantity.T_GATES in row_quantities
    assert row_table[0]["baseline_label"] == "Plain Trotter"
    assert row_table[0]["candidate_label"] == "UWC Trotter"

    strict_report = build_ftqc_research_signal_report(
        "arXiv:2603.22778",
        baseline,
        candidate,
        require_all_quantities=True,
    )
    assert strict_report.summary.rows[0].quantity == FTQCResourceQuantity.LAMBDA_NORM

    with pytest.raises(ValueError, match="Unknown FTQC research signal"):
        describe_ftqc_research_signal("arXiv:0000.00000")


def test_build_ftqc_resource_driver_report_traces_formula_dependencies():
    """Driver reports compare target dependencies exposed by formulas."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1) + 3 * qm_o.Z(2),
        source="plain_trotter_lcu",
    )
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    baseline = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        rotation_synthesis_t_gates=3,
        cost_model=cost,
    )
    candidate = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
        resource_reduction=HamiltonianResourceReduction(
            lambda_norm_factor=sp.Rational(1, 10)
        ),
        cost_model=cost,
    )

    report = build_ftqc_resource_driver_report(
        baseline,
        candidate,
        targets=(FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,),
        title="UWC Trotter driver report",
        baseline_label="Plain Trotter",
        candidate_label="UWC Trotter",
    )
    row_by_quantity = {row.quantity: row for row in report.summary.rows}
    table = report.to_row_table()

    assert isinstance(report, FTQCResourceDriverReport)
    assert report.targets == (FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,)
    assert FTQCResourceQuantity.LAMBDA_NORM in row_by_quantity
    assert FTQCResourceQuantity.QPE_ITERATIONS in row_by_quantity
    assert FTQCResourceQuantity.T_GATES in row_by_quantity
    assert FTQCResourceQuantity.RUNTIME_SECONDS in row_by_quantity
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS in row_by_quantity
    assert sp.Abs(
        row_by_quantity[FTQCResourceQuantity.LAMBDA_NORM].ratio - sp.Rational(1, 10)
    ) < sp.Float("1e-12")
    assert table[-1]["quantity"] == "physical_qubit_seconds"
    assert table[-1]["is_target"] is True
    assert report.to_dict()["targets"] == ["physical_qubit_seconds"]

    drivers_only = build_ftqc_resource_driver_report(
        baseline,
        candidate,
        targets=("physical_qubit_seconds",),
        include_targets=False,
    )
    assert FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS not in {
        row.quantity for row in drivers_only.summary.rows
    }

    with pytest.raises(ValueError, match="targets must not be empty"):
        build_ftqc_resource_driver_report(baseline, candidate, targets=())


def test_build_ftqc_resource_pareto_report_marks_frontier_candidates():
    """Pareto reports keep resource tradeoffs and mark dominated candidates."""
    baseline = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "baseline",
                {
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 1000,
                    FTQCResourceQuantity.RUNTIME_SECONDS: 100,
                    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS: 100_000,
                },
            ),
        ),
        title="Baseline",
    )
    compressed = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "compressed",
                {
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 700,
                    FTQCResourceQuantity.RUNTIME_SECONDS: 80,
                    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS: 56_000,
                },
            ),
        ),
        title="Compressed",
    )
    tiny_slow = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "tiny_slow",
                {
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 450,
                    FTQCResourceQuantity.RUNTIME_SECONDS: 140,
                    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS: 63_000,
                },
            ),
        ),
        title="Tiny slow",
    )

    report = build_ftqc_resource_pareto_report(
        (
            ("baseline", baseline),
            ("compressed", compressed),
            ("tiny slow", tiny_slow),
        ),
        quantities=(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
        ),
        title="Toy early-FTQC frontier",
    )
    report_dict = report.to_dict()
    row_by_label = {row.label: row for row in report.rows}
    table_by_label = {row["label"]: row for row in report.to_row_table()}

    assert isinstance(report, FTQCResourceParetoReport)
    assert all(isinstance(row, FTQCResourceParetoRow) for row in report.rows)
    assert row_by_label["baseline"].dominated_by == ("compressed",)
    assert row_by_label["compressed"].is_frontier
    assert row_by_label["tiny slow"].is_frontier
    assert [row.label for row in report.frontier] == ["compressed", "tiny slow"]
    assert [row.label for row in report.dominated] == ["baseline"]
    assert table_by_label["baseline"]["is_frontier"] is False
    assert table_by_label["baseline"]["dominated_by"] == "compressed"
    assert report_dict["title"] == "Toy early-FTQC frontier"
    assert report_dict["quantities"] == ["physical_qubits", "runtime_seconds"]
    assert report_dict["counts"] == {"frontier": 2, "dominated": 1}


def test_build_ftqc_resource_pareto_report_keeps_symbolic_rows_on_frontier():
    """Undecidable symbolic dominance keeps candidates reviewable."""
    symbolic_runtime = sp.Symbol("runtime")
    concrete = FTQCResourcePlanStep(
        "concrete",
        {
            FTQCResourceQuantity.PHYSICAL_QUBITS: 100,
            FTQCResourceQuantity.RUNTIME_SECONDS: 10,
        },
    )
    symbolic = FTQCResourcePlanStep(
        "symbolic",
        {
            FTQCResourceQuantity.PHYSICAL_QUBITS: 90,
            FTQCResourceQuantity.RUNTIME_SECONDS: symbolic_runtime,
        },
    )

    report = build_ftqc_resource_pareto_report(
        (
            ("concrete", FTQCResourcePlan((concrete,))),
            ("symbolic", FTQCResourcePlan((symbolic,))),
        ),
        quantities=(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
        ),
    )

    assert [row.label for row in report.frontier] == ["concrete", "symbolic"]
    assert report.dominated == ()
    with pytest.raises(ValueError, match="at least two"):
        build_ftqc_resource_pareto_report((("only", FTQCResourcePlan((concrete,))),))
    with pytest.raises(ValueError, match="unique"):
        build_ftqc_resource_pareto_report(
            (
                ("duplicate", FTQCResourcePlan((concrete,))),
                ("duplicate", FTQCResourcePlan((symbolic,))),
            )
        )
    with pytest.raises(ValueError, match="missing"):
        build_ftqc_resource_pareto_report(
            (
                ("concrete", FTQCResourcePlan((concrete,))),
                ("symbolic", FTQCResourcePlan((symbolic,))),
            ),
            quantities=(FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,),
        )


def test_build_ftqc_resource_scenario_report_substitutes_symbolic_values():
    """Scenario reports evaluate symbolic estimates under assumption sets."""
    physical_qubits, runtime = sp.symbols("physical_qubits runtime", positive=True)
    plan = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "architecture_lift",
                {
                    FTQCResourceQuantity.PHYSICAL_QUBITS: physical_qubits,
                    FTQCResourceQuantity.RUNTIME_SECONDS: runtime,
                    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS: (
                        physical_qubits * runtime
                    ),
                },
            ),
        ),
        title="Symbolic architecture lift",
    )
    conservative = FTQCResourceScenario(
        "conservative",
        {
            "physical_qubits": 100_000,
            "runtime": 3600,
        },
    )
    optimistic = FTQCResourceScenario(
        "optimistic",
        {
            physical_qubits: 50_000,
            runtime: 900,
        },
    )

    report = build_ftqc_resource_scenario_report(
        plan,
        (conservative, optimistic),
        quantities=(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
        title="Architecture sensitivity",
    )
    row_table = report.to_row_table()
    report_dict = report.to_dict()

    assert isinstance(report, FTQCResourceScenarioReport)
    assert all(isinstance(row, FTQCResourceScenarioRow) for row in report.rows)
    assert report.scenarios == (conservative, optimistic)
    assert all(row.is_fully_resolved for row in report.rows)
    assert row_table[0]["label"] == "conservative"
    assert row_table[0]["physical_qubit_seconds"] == "360000000"
    assert row_table[1]["physical_qubit_seconds"] == "45000000"
    assert report_dict["counts"] == {"resolved": 2, "unresolved": 0}
    assert report_dict["scenarios"][0]["substitutions"]["runtime"] == "3600"


def test_build_ftqc_resource_scenario_report_keeps_unresolved_symbols():
    """Scenario reports expose remaining symbols for follow-up calibration."""
    runtime = sp.Symbol("runtime", positive=True)
    plan = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "runtime",
                {
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 100,
                    FTQCResourceQuantity.RUNTIME_SECONDS: runtime,
                },
            ),
        )
    )
    partial = FTQCResourceScenario("partial", {})

    report = build_ftqc_resource_scenario_report(
        plan,
        (partial,),
        quantities=(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
        ),
    )

    assert report.rows[0].unresolved_symbols == ("runtime",)
    assert not report.rows[0].is_fully_resolved
    assert report.to_dict()["counts"] == {"resolved": 0, "unresolved": 1}
    direct_report = FTQCResourceScenarioReport(
        title="direct",
        quantities=("runtime_seconds",),
        scenarios=(partial,),
        rows=(FTQCResourceScenarioRow("partial", {"runtime_seconds": runtime}),),
    )
    assert direct_report.quantities == (FTQCResourceQuantity.RUNTIME_SECONDS,)
    with pytest.raises(ValueError, match="scenarios must not be empty"):
        build_ftqc_resource_scenario_report(plan, ())
    with pytest.raises(TypeError, match="FTQCResourceScenario"):
        build_ftqc_resource_scenario_report(plan, (object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing"):
        build_ftqc_resource_scenario_report(
            plan,
            (partial,),
            quantities=(FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,),
        )
    with pytest.raises(ValueError, match="label"):
        FTQCResourceScenario("", {})
    with pytest.raises(TypeError, match="substitution keys"):
        FTQCResourceScenario("bad", {object(): 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="missing selected scenario quantities"):
        FTQCResourceScenarioReport(
            title="bad",
            quantities=(FTQCResourceQuantity.RUNTIME_SECONDS,),
            scenarios=(partial,),
            rows=(FTQCResourceScenarioRow("partial", {"physical_qubits": 100}),),
        )


def test_build_ftqc_resource_report_bundle_snapshots_reports():
    """Report bundles wrap heterogeneous FTQC reports for review artifacts."""
    runtime = sp.Symbol("runtime", positive=True)
    baseline = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "baseline",
                {
                    FTQCResourceQuantity.RUNTIME_SECONDS: 100,
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 1_000,
                },
            ),
        ),
        title="Baseline",
    )
    candidate = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "candidate",
                {
                    FTQCResourceQuantity.RUNTIME_SECONDS: 25,
                    FTQCResourceQuantity.PHYSICAL_QUBITS: 2_000,
                },
            ),
        ),
        title="Candidate",
    )
    comparison = build_ftqc_resource_comparison_report(
        baseline,
        candidate,
        title="Runtime comparison",
        quantities=(FTQCResourceQuantity.RUNTIME_SECONDS,),
    )
    scenario = build_ftqc_resource_scenario_report(
        FTQCResourcePlan(
            (
                FTQCResourcePlanStep(
                    "symbolic",
                    {
                        FTQCResourceQuantity.RUNTIME_SECONDS: runtime,
                    },
                ),
            ),
            title="Symbolic runtime",
        ),
        (
            FTQCResourceScenario("slow", {"runtime": 10}),
            FTQCResourceScenario("fast", {"runtime": 2}),
        ),
        quantities=(FTQCResourceQuantity.RUNTIME_SECONDS,),
    )

    snapshot = build_ftqc_resource_report_snapshot(comparison)
    bundle = build_ftqc_resource_report_bundle(
        "FTQC review bundle",
        (comparison, scenario),
    )
    bundle_dict = bundle.to_dict()
    bundle_rows = bundle.to_row_table()
    bundle_manifest = bundle.to_manifest()

    assert isinstance(snapshot, FTQCResourceReportSnapshot)
    assert snapshot.kind is FTQCResourceReportKind.COMPARISON
    assert snapshot.row_count == 1
    assert snapshot.counts == {"smaller": 1, "larger": 0, "unchanged": 0, "symbolic": 0}
    assert snapshot.to_dict()["payload"]["title"] == "Runtime comparison"
    assert isinstance(bundle, FTQCResourceReportBundle)
    assert bundle.counts_by_kind() == {"comparison": 1, "scenario": 1}
    assert bundle_manifest == bundle_dict["manifest"]
    assert bundle_manifest["counts"] == {"snapshots": 2, "rows": 3}
    assert bundle_manifest["counts_by_kind"] == {"comparison": 1, "scenario": 1}
    assert bundle_manifest["snapshots"] == [
        {
            "index": 0,
            "kind": "comparison",
            "title": "Runtime comparison",
            "row_count": 1,
            "counts": {
                "smaller": 1,
                "larger": 0,
                "unchanged": 0,
                "symbolic": 0,
            },
        },
        {
            "index": 1,
            "kind": "scenario",
            "title": "FTQC resource scenario report",
            "row_count": 2,
            "counts": {"resolved": 2, "unresolved": 0},
        },
    ]
    assert "payload" not in bundle_manifest["snapshots"][0]
    assert bundle_dict["counts"] == {"snapshots": 2, "rows": 3}
    assert bundle_dict["rows"] == bundle_rows
    assert bundle_dict["snapshots"][1]["kind"] == "scenario"
    assert bundle_dict["snapshots"][1]["counts"] == {"resolved": 2, "unresolved": 0}
    assert bundle_rows[0]["snapshot_index"] == 0
    assert bundle_rows[0]["report_kind"] == "comparison"
    assert bundle_rows[0]["report_title"] == "Runtime comparison"
    assert bundle_rows[0]["quantity"] == "runtime_seconds"
    assert bundle_rows[1]["snapshot_index"] == 1
    assert bundle_rows[1]["report_kind"] == "scenario"
    assert bundle_rows[1]["label"] == "slow"
    assert bundle_rows[2]["row_index"] == 1
    colliding = FTQCResourceReportBundle(
        "colliding",
        (
            FTQCResourceReportSnapshot(
                "comparison",
                "Colliding",
                {"title": "Colliding", "rows": [{"report_kind": "payload"}]},
                1,
            ),
        ),
    )
    assert colliding.to_row_table()[0]["report_kind"] == "comparison"

    overridden = build_ftqc_resource_report_snapshot(
        comparison,
        kind="driver",
        title="Override title",
    )
    assert overridden.kind is FTQCResourceReportKind.DRIVER
    assert overridden.title == "Override title"
    with pytest.raises(TypeError, match="Unsupported FTQC resource report type"):
        build_ftqc_resource_report_snapshot(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reports must not be empty"):
        build_ftqc_resource_report_bundle("empty", ())
    with pytest.raises(ValueError, match="counts must be non-negative"):
        FTQCResourceReportSnapshot(
            "scenario",
            "bad",
            {},
            0,
            {"resolved": -1},
        )
    malformed = FTQCResourceReportBundle(
        "malformed",
        (
            FTQCResourceReportSnapshot(
                "comparison",
                "Malformed",
                {"title": "Malformed", "rows": ["bad"]},
                1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="rows must contain dictionaries"):
        malformed.to_row_table()


def test_audit_ftqc_research_signal_coverage_marks_missing_quantities():
    """Research-signal coverage audits partial and complete FTQC estimates."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1),
        source="partial_lcu_summary",
    )
    partial = audit_ftqc_research_signal_coverage("arXiv:2603.22778", summary)

    assert isinstance(partial, FTQCResearchSignalCoverage)
    assert partial.reference_key == "arXiv:2603.22778"
    assert not partial.is_complete
    assert partial.available == (FTQCResourceQuantity.LAMBDA_NORM,)
    assert FTQCResourceQuantity.QPE_ITERATIONS in partial.missing
    assert partial.coverage_fraction == sp.Rational(1, 8)
    assert partial.to_dict()["missing"][0] == "qpe_iterations"
    with pytest.raises(ValueError, match="total must be positive"):
        FTQCResearchSignalCoverage(
            reference_key="arXiv:bad",
            title="Bad coverage",
            available=(),
            missing=(),
            total=0,
        )

    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    complete_estimate = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
        resource_reduction=HamiltonianResourceReduction(
            lambda_norm_factor=sp.Rational(1, 10)
        ),
        cost_model=cost,
    )
    complete = audit_ftqc_research_signal_coverage(
        "arXiv:2603.22778",
        complete_estimate,
    )

    assert complete.is_complete
    assert complete.missing == ()
    assert complete.coverage_fraction == 1
    assert complete.to_dict()["is_complete"] is True


def test_audit_ftqc_research_signal_catalog_groups_signal_coverage():
    """Coverage catalog reports complete and incomplete research signals."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1),
        source="catalog_lcu_summary",
    )
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    estimate = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
        resource_reduction=HamiltonianResourceReduction(
            lambda_norm_factor=sp.Rational(1, 10)
        ),
        cost_model=cost,
    )

    report = audit_ftqc_research_signal_catalog(
        estimate,
        reference_keys=("arXiv:2603.22778", "arXiv:2601.08533"),
        title="Toy signal coverage",
        estimate_label="UWC Trotter",
    )
    rows = report.to_row_table()

    assert isinstance(report, FTQCResearchSignalCoverageReport)
    assert report.complete[0].reference_key == "arXiv:2603.22778"
    assert report.incomplete[0].reference_key == "arXiv:2601.08533"
    assert rows[0]["estimate_label"] == "UWC Trotter"
    assert rows[0]["is_complete"] is True
    assert rows[1]["missing"].startswith("state_preparation_success_probability")
    assert report.to_dict()["counts"] == {"complete": 1, "incomplete": 1}
    with pytest.raises(ValueError, match="reference_keys must not be empty"):
        audit_ftqc_research_signal_catalog(estimate, reference_keys=())
    with pytest.raises(ValueError, match="coverages must not be empty"):
        FTQCResearchSignalCoverageReport("Empty", "estimate", ())


def test_ftqc_resource_review_findings_prioritize_savings_and_tradeoffs():
    """Review findings surface largest savings before resource tradeoffs."""
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
    report = build_ftqc_resource_comparison_report(
        baseline_estimate,
        candidate_estimate,
        quantities=(
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
            FTQCResourceQuantity.LOGICAL_QUBITS,
        ),
    )

    findings = report.to_review_findings(max_improvements=1, max_tradeoffs=1)
    report_dict = report.to_dict()

    assert all(isinstance(finding, FTQCResourceReviewFinding) for finding in findings)
    assert [finding.direction for finding in findings] == [
        FTQCResourceChangeDirection.SMALLER,
        FTQCResourceChangeDirection.LARGER,
    ]
    assert findings[0].quantity == FTQCResourceQuantity.QPE_ITERATIONS
    assert findings[0].headline == "Candidate reduces QPE iterations."
    assert findings[0].detail == (
        "qpe_iterations: baseline=5.00000000000000, "
        "candidate=2.50000000000000, ratio=0.500000000000000, "
        "reduction=0.500000000000000."
    )
    assert findings[1].quantity == FTQCResourceQuantity.TOFFOLI_GATES
    assert findings[1].to_dict()["direction"] == "larger"
    assert report_dict["findings"][0]["quantity"] == "qpe_iterations"


def test_ftqc_resource_review_findings_handle_symbolic_and_unchanged_rows():
    """Review findings can keep symbolic follow-ups and unchanged quantities."""
    symbolic = sp.Symbol("x")
    rows = (
        FTQCResourceComparisonRow(
            quantity=FTQCResourceQuantity.TOFFOLI_GATES,
            baseline=10,
            candidate=4,
            ratio=sp.Rational(2, 5),
            reduction=sp.Rational(3, 5),
            label="Toffoli gates",
            unit="Toffoli gates",
            category=FTQCResourceCategory.LOGICAL,
        ),
        FTQCResourceComparisonRow(
            quantity=FTQCResourceQuantity.LOGICAL_QUBITS,
            baseline=5,
            candidate=7,
            ratio=sp.Rational(7, 5),
            reduction=sp.Rational(-2, 5),
            label="Logical qubits",
            unit="logical qubits",
            category=FTQCResourceCategory.LOGICAL,
        ),
        FTQCResourceComparisonRow(
            quantity=FTQCResourceQuantity.RUNTIME_SECONDS,
            baseline=1,
            candidate=symbolic,
            ratio=symbolic,
            reduction=1 - symbolic,
            label="Runtime",
            unit="seconds",
            category=FTQCResourceCategory.PHYSICAL,
        ),
        FTQCResourceComparisonRow(
            quantity=FTQCResourceQuantity.PHYSICAL_QUBITS,
            baseline=100,
            candidate=100,
            ratio=1,
            reduction=0,
            label="Physical qubits",
            unit="physical qubits",
            category=FTQCResourceCategory.PHYSICAL,
        ),
    )
    summary = FTQCResourceComparisonSummary.from_rows(rows)

    findings = build_ftqc_resource_review_findings(
        summary,
        include_symbolic=True,
        include_unchanged=True,
    )

    assert [finding.direction for finding in findings] == [
        FTQCResourceChangeDirection.SMALLER,
        FTQCResourceChangeDirection.LARGER,
        FTQCResourceChangeDirection.SYMBOLIC,
        FTQCResourceChangeDirection.UNCHANGED,
    ]
    assert findings[2].headline == "Candidate change for Runtime remains symbolic."
    assert findings[3].headline == "Candidate leaves Physical qubits unchanged."
    with pytest.raises(ValueError, match="max_improvements"):
        build_ftqc_resource_review_findings(summary, max_improvements=-1)
    with pytest.raises(ValueError, match="max_tradeoffs"):
        build_ftqc_resource_review_findings(summary, max_tradeoffs=-1)


def test_evaluate_ftqc_resource_constraints_groups_budget_statuses():
    """Resource budget reports classify satisfied and violated constraints."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e5"),
    )
    estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=summary,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=10,
        ),
        precision=1,
        cost_model=cost,
    )
    constraints = (
        FTQCResourceConstraint(
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            estimate.physical_qubits + 1,
            label="Physical-qubit budget",
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.RUNTIME_SECONDS,
            estimate.runtime_seconds / 2,
        ),
        FTQCResourceConstraint(
            FTQCResourceQuantity.LOGICAL_QUBITS,
            estimate.logical_qubits,
            sense=FTQCResourceConstraintSense.AT_LEAST,
        ),
    )

    report = evaluate_ftqc_resource_constraints(
        estimate,
        constraints,
        title="Toy early-FTQC budget",
    )
    report_dict = report.to_dict()

    assert isinstance(report, FTQCResourceBudgetReport)
    assert all(
        isinstance(result, FTQCResourceConstraintResult) for result in report.results
    )
    assert [result.status for result in report.results] == [
        FTQCResourceConstraintStatus.SATISFIED,
        FTQCResourceConstraintStatus.VIOLATED,
        FTQCResourceConstraintStatus.SATISFIED,
    ]
    assert report.results[0].label == "Physical-qubit budget"
    assert report.results[0].margin == 1
    assert report.results[1].sense == FTQCResourceConstraintSense.AT_MOST
    assert report.results[2].sense == FTQCResourceConstraintSense.AT_LEAST
    assert report_dict["title"] == "Toy early-FTQC budget"
    assert report_dict["counts"] == {
        "satisfied": 2,
        "violated": 1,
        "symbolic": 0,
    }
    assert report_dict["violated"][0]["quantity"] == "runtime_seconds"


def test_evaluate_ftqc_resource_constraints_keeps_symbolic_budgets():
    """Symbolic budget margins stay undecided until architecture values bind."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    estimate = estimate_qubitized_chemistry_qpe_from_model(
        ChemistryQPEModel(
            hamiltonian=summary,
            method=ChemistryQPEMethod.SPARSE,
            walk_cost_toffoli=10,
        ),
        precision=1,
    )

    report = evaluate_ftqc_resource_constraints(
        estimate,
        (
            FTQCResourceConstraint(
                FTQCResourceQuantity.PHYSICAL_QUBITS,
                100_000,
            ),
        ),
    )

    assert report.satisfied == ()
    assert report.violated == ()
    assert report.symbolic[0].status == FTQCResourceConstraintStatus.SYMBOLIC
    assert report.symbolic[0].margin == (
        100_000 - estimate.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBITS]
    )


def test_ftqc_resource_constraints_reject_invalid_inputs():
    """Constraint helpers reject invalid senses, limits, and missing values."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="constraint sense"):
        FTQCResourceConstraint("physical_qubits", 10, sense="near")

    with pytest.raises(TypeError, match="limit"):
        FTQCResourceConstraint("physical_qubits", object())

    with pytest.raises(ValueError, match="missing"):
        evaluate_ftqc_resource_constraints(
            summary,
            (FTQCResourceConstraint(FTQCResourceQuantity.PHYSICAL_QUBITS, 10),),
        )


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
