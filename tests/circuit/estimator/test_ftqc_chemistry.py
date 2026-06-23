"""Tests for FTQC quantum-chemistry resource estimators."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCAccuracyBudget,
    FTQCCostModel,
    FTQCReference,
    FTQCResourcePlan,
    FTQCResourcePlanStep,
    FTQCResourceQuantity,
    HamiltonianResourceReduction,
    QPEStatePreparationBudget,
    SurfaceCodeCostModel,
    SurfaceCodeDistanceBudget,
    estimate_qubitized_chemistry_qpe,
    estimate_qubitized_chemistry_qpe_from_model,
    estimate_single_ancilla_trotter_qpe,
    estimate_single_ancilla_trotter_qpe_from_hamiltonian,
    hamiltonian_from_openfermion_qubit_operator,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
)


def test_qubitized_qpe_tracks_lambda_precision_and_walk_cost():
    """Qubitized QPE exposes the lambda-over-epsilon iteration contract."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)

    estimate = estimate_qubitized_chemistry_qpe(
        n,
        lam,
        eps,
        walk,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
    )

    assert estimate.logical_qubits == n
    assert estimate.qpe_iterations == lam / eps
    assert estimate.target_precision == eps
    assert estimate.toffoli_gates == lam * walk / eps
    assert {"lambda", "eps", "C_W"}.issubset(estimate.parameters)
    assert estimate.to_dict()["target_precision"] == "eps"
    assert estimate.resource_values()[FTQCResourceQuantity.TARGET_PRECISION] == eps


def test_ftqc_estimate_converts_to_common_logical_resource_estimate():
    """FTQC estimates expose logical work through the shared resource shape."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)

    estimate = estimate_qubitized_chemistry_qpe(
        n,
        lambda_norm=lam,
        precision=eps,
        walk_cost_toffoli=walk,
        logical_qubits=n,
    )

    logical = estimate.to_logical_resource_estimate()
    concrete = logical.substitute(**{"n": 10, "lambda": 100, "eps": 2, "C_W": 5})

    assert logical.qubits == estimate.logical_qubits
    assert logical.gates.total == lam * walk / eps
    assert logical.gates.multi_qubit == estimate.toffoli_gates
    assert logical.gates.t_gates == 0
    assert logical.gates.clifford_gates == 0
    assert logical.gates.oracle_calls["qpe_iterations"] == estimate.qpe_iterations
    assert set(logical.parameters) == {"C_W", "eps", "lambda", "n"}
    assert "physical_qubits_per_logical" not in logical.parameters
    assert concrete.qubits == 10
    assert concrete.gates.total == 250
    assert concrete.gates.oracle_calls["qpe_iterations"] == 50


def test_qubitized_qpe_uses_representation_specific_logical_qubits():
    """Representation defaults follow the chemistry-resource scaling table."""
    n, sparsity, rank = sp.symbols("n S Xi", positive=True)

    sparse = estimate_qubitized_chemistry_qpe(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        method="sparse",
        sparsity=sparsity,
    )
    single = estimate_qubitized_chemistry_qpe(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        method=ChemistryQPEMethod.SINGLE_FACTORIZATION,
    )
    double = estimate_qubitized_chemistry_qpe(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        second_factor_rank=rank,
    )

    assert sparse.logical_qubits == n + sp.sqrt(sparsity)
    assert single.logical_qubits == n ** sp.Rational(3, 2)
    assert double.logical_qubits == n * sp.sqrt(rank)


def test_qubitized_qpe_records_method_references():
    """Method-specific estimates expose the papers that motivate the model."""
    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=20,
        lambda_norm=100,
        precision=1,
        walk_cost_toffoli=10,
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        second_factor_rank=4,
    )

    reference_keys = {reference.key for reference in estimate.references}
    serialized_keys = {
        reference["key"] for reference in estimate.to_dict()["references"]
    }

    assert "arXiv:1902.02134" in reference_keys
    assert "arXiv:2403.03502" in reference_keys
    assert "arXiv:2412.01338" in reference_keys
    assert serialized_keys == reference_keys


def test_model_references_are_preserved_and_deduplicated():
    """Model-specific sources are merged with method defaults by key."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))
    custom_reference = FTQCReference(
        key="internal:toy-model",
        title="Toy molecule calibration",
        url="https://example.invalid/toy-model",
        note="Documents synthetic benchmark inputs.",
    )
    duplicate_reference = FTQCReference(
        key="arXiv:2403.03502",
        title="Duplicate SCDF citation",
        url="https://example.invalid/duplicate",
    )
    model = ChemistryQPEModel(
        hamiltonian=summary,
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=10,
        second_factor_rank=4,
        references=(custom_reference, duplicate_reference),
    )

    estimate = estimate_qubitized_chemistry_qpe_from_model(model, precision=1)
    relifted = estimate.with_cost_model(
        FTQCCostModel(
            physical_qubits_per_logical=2,
            logical_cycle_time_seconds=1,
            factory_qubits=0,
            toffoli_throughput_per_second=1,
        )
    )
    reference_keys = [reference.key for reference in estimate.references]

    assert reference_keys.count("arXiv:2403.03502") == 1
    assert reference_keys[-1] == "internal:toy-model"
    assert relifted.references == estimate.references
    assert relifted.target_precision == estimate.target_precision


def test_accuracy_budget_allocates_precision_to_qpe_and_truncation():
    """Accuracy budgets align total target precision with model truncation."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    model = ChemistryQPEModel(
        hamiltonian=summary,
        method=ChemistryQPEMethod.SPARSE,
        walk_cost_toffoli=11,
    )
    budget = FTQCAccuracyBudget(
        target_precision=sp.Float("1.6e-3"),
        truncation_error=sp.Float("1e-4"),
        safety_margin=sp.Float("5e-5"),
    )

    budgeted_model = budget.with_model(model)
    estimate = estimate_qubitized_chemistry_qpe_from_model(
        budgeted_model,
        precision=budget.qpe_precision,
    )

    assert sp.Abs(budget.qpe_precision - sp.Float("0.00145")) < sp.Float("1e-12")
    assert budgeted_model.truncation_error == sp.Float("1e-4")
    assert model.truncation_error == 0
    assert estimate.target_precision == budget.qpe_precision
    assert sp.sympify(estimate.assumptions["truncation_error"]) == sp.Float("1e-4")
    assert budget.resource_values()[FTQCResourceQuantity.TARGET_PRECISION] == sp.Float(
        "1.6e-3"
    )
    assert budget.to_dict()["qpe_precision"] == "0.00145000000000000"


def test_accuracy_budget_rejects_overallocated_error_budget():
    """Accuracy budgets reject non-positive precision left for QPE."""
    with pytest.raises(ValueError, match="qpe_precision"):
        FTQCAccuracyBudget(
            target_precision=sp.Float("1e-3"),
            truncation_error=sp.Float("8e-4"),
            safety_margin=sp.Float("2e-4"),
        )

    with pytest.raises(TypeError, match="ChemistryQPEModel"):
        FTQCAccuracyBudget(target_precision=1).with_model(object())


def test_hamiltonian_resource_reduction_scales_problem_drivers():
    """Hamiltonian reductions rescale lambda and term-count metadata."""
    summary = summarize_pauli_hamiltonian(
        2 * qm_o.Z(0) + 3 * qm_o.X(1) + qm_o.Y(2),
        source="plain_lcu",
    )
    reduction = HamiltonianResourceReduction(
        lambda_norm_factor=sp.Rational(3, 5),
        pauli_term_factor=sp.Rational(1, 3),
        source="symmetry_shifted_lcu",
        description="simultaneous symmetry shifts",
    )

    reduced = reduction.apply_to_hamiltonian(summary)
    factors = reduction.resource_factors()

    assert sp.simplify(summary.lambda_norm - 6) == 0
    assert reduced.source == "symmetry_shifted_lcu"
    assert sp.Abs(reduced.lambda_norm - sp.Rational(18, 5)) < sp.Float("1e-12")
    assert reduced.n_pauli_terms == 1
    assert reduced.max_locality == summary.max_locality
    assert factors[FTQCResourceQuantity.LAMBDA_NORM] == sp.Rational(3, 5)
    assert reduction.to_dict()["description"] == "simultaneous symmetry shifts"


def test_hamiltonian_resource_reduction_updates_chemistry_model_estimates():
    """Representation reductions turn research cost drivers into estimates."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1) + 3 * qm_o.Z(2) + 4 * qm_o.Z(3) + 5 * qm_o.Z(4),
        source="plain_sparse_lcu",
    )
    reference = FTQCReference(
        key="arXiv:2603.22778",
        title="Early FTQC chemistry resource model",
        url="https://arxiv.org/abs/2603.22778",
    )
    baseline_model = ChemistryQPEModel(
        hamiltonian=summary,
        method=ChemistryQPEMethod.SPARSE,
        walk_cost_toffoli=20,
        sparsity=50,
        truncation_error=sp.Float("1e-5"),
        description="plain sparse LCU",
    )
    reduction = HamiltonianResourceReduction(
        lambda_norm_factor=sp.Rational(1, 5),
        pauli_term_factor=sp.Rational(1, 2),
        walk_cost_factor=sp.Rational(1, 2),
        sparsity_factor=sp.Rational(2, 5),
        truncation_error=sp.Float("2e-5"),
        description="unitary weight concentration",
        references=(reference,),
    )

    reduced_model = reduction.apply_to_model(baseline_model)
    baseline = estimate_qubitized_chemistry_qpe_from_model(
        baseline_model,
        precision=1,
    )
    reduced = estimate_qubitized_chemistry_qpe_from_model(
        reduced_model,
        precision=1,
    )

    assert sp.simplify(baseline_model.hamiltonian.lambda_norm - 15) == 0
    assert sp.Abs(reduced_model.hamiltonian.lambda_norm - 3) < sp.Float("1e-12")
    assert reduced_model.hamiltonian.n_pauli_terms == sp.Rational(5, 2)
    assert reduced_model.walk_cost_toffoli == 10
    assert reduced_model.effective_sparsity == 20
    assert reduced_model.truncation_error == sp.Float("2e-5")
    assert reduced_model.description == (
        "plain sparse LCU; unitary weight concentration"
    )
    assert [item.key for item in reduced_model.references] == ["arXiv:2603.22778"]
    assert reduced.qpe_iterations == baseline.qpe_iterations / 5
    assert reduced.toffoli_gates == baseline.toffoli_gates / 10
    assert reduced.logical_qubits < baseline.logical_qubits


def test_state_preparation_budget_scales_expected_qpe_work():
    """State-preparation success budgets scale repeated QPE attempts."""
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
    budget = QPEStatePreparationBudget(
        success_probability=sp.Rational(1, 4),
        state_preparation_toffoli=20,
        state_preparation_t_gates=4,
        state_preparation_logical_depth=30,
        description="symmetry-filtered trial state",
    )

    repeated = budget.apply(estimate)

    assert budget.qpe_repetitions == 4
    assert repeated.qpe_iterations == estimate.qpe_iterations * 4
    assert repeated.toffoli_gates == (estimate.toffoli_gates + 20) * 4
    assert repeated.t_gates == 16
    assert repeated.logical_depth == (estimate.logical_depth + 30) * 4
    assert repeated.physical_qubits == estimate.physical_qubits
    assert sp.Abs(repeated.runtime_seconds - sp.Float("0.00296")) < sp.Float("1e-12")
    assert repeated.resource_values()[
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME
    ] == (repeated.logical_qubits * repeated.logical_depth)
    assert repeated.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS] == (
        repeated.physical_qubits * repeated.runtime_seconds
    )
    assert repeated.resource_values()[
        FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY
    ] == sp.Rational(1, 4)
    assert repeated.resource_values()[FTQCResourceQuantity.QPE_REPETITIONS] == 4
    assert repeated.to_dict()["algorithm_values"]["qpe_repetitions"] == "4"
    assert (
        repeated.assumptions["state_preparation_description"]
        == "symmetry-filtered trial state"
    )


def test_state_preparation_budget_scales_resource_plan_attempts():
    """State-preparation success budgets compose with abstract QPE plans."""
    base_plan = FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "qpe_attempt",
                {
                    FTQCResourceQuantity.QPE_ITERATIONS: 5,
                    FTQCResourceQuantity.TOFFOLI_GATES: 10,
                    FTQCResourceQuantity.T_GATES: 2,
                    FTQCResourceQuantity.LOGICAL_DEPTH: 11,
                    FTQCResourceQuantity.LOGICAL_QUBITS: 4,
                },
            ),
        ),
        title="Toy QPE plan",
    )
    reference = FTQCReference(
        key="internal:trial-state",
        title="Trial-state preparation model",
        url="https://example.invalid/trial-state",
    )
    budget = QPEStatePreparationBudget(
        success_probability=sp.Rational(1, 4),
        state_preparation_toffoli=3,
        state_preparation_t_gates=5,
        state_preparation_logical_depth=7,
        description="symmetry-filtered trial state",
        references=(reference,),
    )

    repeated_plan = budget.apply_to_plan(base_plan)
    values = repeated_plan.resource_values()

    assert repeated_plan.title == "Toy QPE plan with state-preparation budget"
    assert repeated_plan.steps[0].name == "state_preparation_budget"
    assert repeated_plan.steps[1].name == "repeated_qpe_attempt"
    assert values[FTQCResourceQuantity.QPE_REPETITIONS] == 4
    assert values[FTQCResourceQuantity.QPE_ITERATIONS] == 20
    assert values[FTQCResourceQuantity.TOFFOLI_GATES] == 52
    assert values[FTQCResourceQuantity.T_GATES] == 28
    assert values[FTQCResourceQuantity.LOGICAL_DEPTH] == 72
    assert values[FTQCResourceQuantity.LOGICAL_QUBITS] == 4
    assert values[FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME] == 288
    assert values[FTQCResourceQuantity.STATE_PREPARATION_TOFFOLI] == 3
    assert repeated_plan.reference_keys() == ("internal:trial-state",)
    assert repeated_plan.to_dict()["steps"][0]["label"] == (
        "symmetry-filtered trial state"
    )
    assert {formula.quantity for formula in repeated_plan.formulas()} >= {
        FTQCResourceQuantity.QPE_REPETITIONS,
        FTQCResourceQuantity.TOFFOLI_GATES,
        FTQCResourceQuantity.T_GATES,
        FTQCResourceQuantity.LOGICAL_DEPTH,
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    }


def test_state_preparation_budget_rejects_invalid_probabilities():
    """State-preparation success budgets reject invalid probability inputs."""
    with pytest.raises(ValueError, match="success_probability"):
        QPEStatePreparationBudget(success_probability=0)

    with pytest.raises(ValueError, match="at most one"):
        QPEStatePreparationBudget(success_probability=sp.Rational(3, 2))

    with pytest.raises(ValueError, match="state_preparation_toffoli"):
        QPEStatePreparationBudget(success_probability=1, state_preparation_toffoli=-1)

    with pytest.raises(TypeError, match="FTQCResourceEstimate"):
        QPEStatePreparationBudget(success_probability=1).apply(object())

    with pytest.raises(TypeError, match="FTQCResourcePlan"):
        QPEStatePreparationBudget(success_probability=1).apply_to_plan(object())


def test_hamiltonian_resource_reduction_rejects_invalid_inputs():
    """Hamiltonian reductions reject invalid scale factors and target types."""
    with pytest.raises(ValueError, match="lambda_norm_factor"):
        HamiltonianResourceReduction(lambda_norm_factor=0)

    with pytest.raises(ValueError, match="truncation_error"):
        HamiltonianResourceReduction(truncation_error=-1)

    with pytest.raises(TypeError, match="FTQCReference"):
        HamiltonianResourceReduction(references=(object(),))

    reduction = HamiltonianResourceReduction()
    with pytest.raises(TypeError, match="PauliHamiltonianResource"):
        reduction.apply_to_hamiltonian(object())

    with pytest.raises(TypeError, match="ChemistryQPEModel"):
        reduction.apply_to_model(object())

    with pytest.raises(TypeError, match="PauliHamiltonianResource"):
        estimate_single_ancilla_trotter_qpe_from_hamiltonian(
            object(),
            precision=1,
            trotter_steps_per_sample=1,
            samples=1,
        )

    with pytest.raises(TypeError, match="HamiltonianResourceReduction"):
        estimate_single_ancilla_trotter_qpe_from_hamiltonian(
            summarize_pauli_hamiltonian(qm_o.Z(0)),
            precision=1,
            trotter_steps_per_sample=1,
            samples=1,
            resource_reduction=object(),
        )


def test_cost_model_lifts_logical_estimates_to_physical_runtime():
    """Concrete architecture knobs produce concrete physical-qubit/runtime values."""
    model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=0.01,
        factory_qubits=20,
        toffoli_throughput_per_second=50,
    )

    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
        cost_model=model,
    )

    assert estimate.logical_qubits == 7
    assert estimate.physical_qubits == 720
    assert estimate.toffoli_gates == 40
    assert estimate.runtime_seconds == sp.Rational(4, 5)


def test_surface_code_cost_model_derives_architecture_knobs():
    """Surface-code knobs derive the cost model used by FTQC estimators."""
    architecture = SurfaceCodeCostModel(
        code_distance=5,
        physical_cycle_time_seconds=sp.Float("1e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=1,
        factory_count=4,
        physical_qubits_per_factory=1000,
        factory_cycles_per_toffoli=10,
    )

    cost_model = architecture.to_cost_model()
    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
        cost_model=cost_model,
    )

    assert architecture.physical_qubits_per_logical == 50
    assert sp.Abs(
        architecture.logical_cycle_time_seconds - sp.Float("5e-6")
    ) < sp.Float("1e-12")
    assert architecture.factory_qubits == 4000
    assert sp.Abs(
        architecture.toffoli_throughput_per_second - sp.Float("8e5") / 10
    ) < sp.Float("1e-12")
    assert estimate.physical_qubits == 4350
    assert sp.Abs(estimate.runtime_seconds - sp.Float("5e-4")) < sp.Float("1e-12")


def test_surface_code_model_is_preserved_on_resource_estimates():
    """Architecture-lifted estimates retain canonical surface-code quantities."""
    architecture = SurfaceCodeCostModel(
        code_distance=7,
        physical_cycle_time_seconds=sp.Float("2e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=3,
        factory_count=5,
        physical_qubits_per_factory=1000,
        factory_cycles_per_toffoli=4,
    )

    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
        cost_model=architecture,
    )

    values = estimate.resource_values()
    serialized = estimate.to_dict()

    assert estimate.physical_qubits == 5686
    assert values[FTQCResourceQuantity.CODE_DISTANCE] == 7
    assert values[FTQCResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS] == sp.Float("2e-6")
    assert values[FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL] == 98
    assert serialized["architecture_values"]["code_distance"] == "7"
    assert serialized["architecture_values"]["factory_count"] == "5"
    assert any(
        row["quantity"] == "code_distance" for row in estimate.to_quantity_table()
    )


def test_surface_code_distance_budget_selects_odd_distance():
    """Distance budgets select the smallest odd code distance from error inputs."""
    budget = SurfaceCodeDistanceBudget(
        physical_error_rate=sp.Float("1e-3"),
        threshold_error_rate=sp.Float("1e-2"),
        target_logical_failure_probability=sp.Float("1e-9"),
        logical_operation_budget=1000,
    )

    architecture = budget.to_surface_code_cost_model(
        physical_cycle_time_seconds=sp.Float("1e-6"),
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=1,
        factory_count=4,
        physical_qubits_per_factory=1000,
        factory_cycles_per_toffoli=10,
    )
    values = budget.resource_values()

    assert budget.code_distance == 21
    assert sp.Abs(
        budget.logical_failure_probability_per_operation - sp.Float("1e-12")
    ) < sp.Float("1e-24")
    assert sp.Abs(
        budget.logical_error_rate_for_distance(budget.code_distance) - sp.Float("1e-12")
    ) < sp.Float("1e-24")
    assert values[FTQCResourceQuantity.CODE_DISTANCE] == 21
    assert values[FTQCResourceQuantity.PHYSICAL_ERROR_RATE] == sp.Float("1e-3")
    assert architecture.code_distance == 21
    assert architecture.physical_qubits_per_logical == 882


def test_surface_code_distance_budget_rejects_invalid_error_model():
    """Distance budgets reject non-numeric or above-threshold error inputs."""
    with pytest.raises(ValueError, match="below threshold"):
        SurfaceCodeDistanceBudget(
            physical_error_rate=sp.Float("1e-2"),
            threshold_error_rate=sp.Float("1e-2"),
            target_logical_failure_probability=sp.Float("1e-9"),
            logical_operation_budget=1000,
        )

    with pytest.raises(ValueError, match="numeric"):
        SurfaceCodeDistanceBudget(
            physical_error_rate=sp.Symbol("p"),
            threshold_error_rate=sp.Float("1e-2"),
            target_logical_failure_probability=sp.Float("1e-9"),
            logical_operation_budget=1000,
        ).code_distance


def test_cost_model_relifts_existing_estimate_without_changing_logical_work():
    """Architecture relifting updates physical fields while preserving logical work."""
    original_cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e6"),
    )
    replacement_cost = FTQCCostModel(
        physical_qubits_per_logical=200,
        logical_cycle_time_seconds=sp.Float("5e-7"),
        factory_qubits=20,
        toffoli_throughput_per_second=sp.Float("2e6"),
    )
    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
        cost_model=original_cost,
    )

    relifted = estimate.with_cost_model(replacement_cost)
    relifted_via_model = replacement_cost.lift_estimate(estimate)

    assert relifted.logical_qubits == estimate.logical_qubits
    assert relifted.toffoli_gates == estimate.toffoli_gates
    assert relifted.t_gates == estimate.t_gates
    assert relifted.logical_depth == estimate.logical_depth
    assert relifted.physical_qubits == 1420
    assert sp.Abs(relifted.runtime_seconds - sp.Float("2e-5")) < sp.Float("1e-12")
    assert relifted_via_model.physical_qubits == relifted.physical_qubits
    assert relifted.assumptions["architecture_relift"].startswith("physical_qubits")


def test_cost_model_relifts_t_gate_estimates_with_t_throughput_demand():
    """Architecture relifting uses T counts when a model is not Toffoli-native."""
    cost = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e6"),
    )
    replacement = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-9"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e3"),
    )
    estimate = estimate_single_ancilla_trotter_qpe(
        n_spin_orbitals=2,
        n_pauli_terms=3,
        lambda_norm=4,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        rotation_synthesis_t_gates=7,
        cost_model=cost,
    )

    relifted = estimate.with_cost_model(replacement)

    assert estimate.toffoli_gates == 0
    assert estimate.t_gates == 840
    assert sp.Abs(relifted.runtime_seconds - sp.Float("0.84")) < sp.Float("1e-12")


def test_single_ancilla_trotter_qpe_models_unitary_weight_reduction():
    """Unitary-weight concentration reduces the lambda-driven QPE iterations."""
    baseline = estimate_single_ancilla_trotter_qpe(
        n_spin_orbitals=20,
        n_pauli_terms=100,
        lambda_norm=1000,
        precision=10,
        trotter_steps_per_sample=2,
        samples=5,
    )
    concentrated = estimate_single_ancilla_trotter_qpe(
        n_spin_orbitals=20,
        n_pauli_terms=100,
        lambda_norm=1000,
        precision=10,
        trotter_steps_per_sample=2,
        samples=5,
        unitary_weight_factor=sp.Rational(1, 10),
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
    )

    assert baseline.qpe_iterations == 100
    assert concentrated.qpe_iterations == 10
    assert concentrated.t_gates == 15000
    assert concentrated.logical_qubits == 21
    assert concentrated.logical_depth < baseline.logical_depth
    assert any(
        reference.key == "arXiv:2603.22778" for reference in concentrated.references
    )


def test_cost_model_rejects_zero_non_clifford_throughput():
    """A zero non-Clifford throughput is invalid because runtime divides by it."""
    with pytest.raises(ValueError, match="toffoli_throughput_per_second"):
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=1,
            factory_qubits=0,
            toffoli_throughput_per_second=0,
        )


def test_cost_model_rejects_negative_relift_inputs():
    """Cost-model lifting rejects negative logical work before producing resources."""
    model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=1,
        factory_qubits=0,
        toffoli_throughput_per_second=1,
    )

    with pytest.raises(ValueError, match="logical_qubits"):
        model.physical_qubits_for(-1)
    with pytest.raises(ValueError, match="logical_depth"):
        model.runtime_seconds_for(-1, 0)
    with pytest.raises(ValueError, match="non_clifford_gates"):
        model.runtime_seconds_for(0, -1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"code_distance": 0}, "code_distance"),
        ({"physical_cycle_time_seconds": 0}, "physical_cycle_time_seconds"),
        (
            {"physical_qubits_per_logical_factor": 0},
            "physical_qubits_per_logical_factor",
        ),
        ({"logical_cycle_factor": 0}, "logical_cycle_factor"),
        ({"factory_count": 0}, "factory_count"),
        ({"physical_qubits_per_factory": -1}, "physical_qubits_per_factory"),
        ({"factory_cycles_per_toffoli": 0}, "factory_cycles_per_toffoli"),
    ],
)
def test_surface_code_cost_model_rejects_invalid_architecture_knobs(
    kwargs: dict[str, object],
    match: str,
):
    """Invalid surface-code architecture knobs fail before estimator use."""
    with pytest.raises(ValueError, match=match):
        SurfaceCodeCostModel(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "n_spin_orbitals": 0,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
            },
            "n_spin_orbitals",
        ),
        (
            {
                "n_spin_orbitals": 1,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
                "method": "not-a-method",
            },
            "Unknown chemistry QPE method",
        ),
        (
            {
                "n_spin_orbitals": 1,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
                "method": "sparse",
            },
            "sparsity is required",
        ),
    ],
)
def test_qubitized_qpe_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
):
    """Invalid finite-set methods and non-positive quantities fail early."""
    with pytest.raises(ValueError, match=match):
        estimate_qubitized_chemistry_qpe(**kwargs)


def test_single_ancilla_trotter_qpe_rejects_negative_reduction_factor():
    """Negative reduction factors are rejected because they imply negative work."""
    with pytest.raises(ValueError, match="unitary_weight_factor"):
        estimate_single_ancilla_trotter_qpe(
            n_spin_orbitals=2,
            n_pauli_terms=3,
            lambda_norm=4,
            precision=1,
            trotter_steps_per_sample=1,
            samples=1,
            unitary_weight_factor=-1,
        )


def test_summarize_pauli_hamiltonian_extracts_lcu_quantities():
    """Hamiltonian summaries expose term count, locality, constant, and lambda."""
    hamiltonian = 0.5 * qm_o.Z(0) + (1.25 + 0.75j) * qm_o.X(1) * qm_o.Y(2) + 3

    summary = summarize_pauli_hamiltonian(hamiltonian, source="toy")
    with_constant = summarize_pauli_hamiltonian(
        hamiltonian,
        include_constant=True,
    )

    assert summary.source == "toy"
    assert summary.n_spin_orbitals == 3
    assert summary.n_pauli_terms == 2
    assert summary.max_locality == 2
    assert sp.simplify(summary.constant - 3) == 0
    expected_lambda = sp.Float("0.5") + sp.sqrt(sp.Float("2.125"))
    assert sp.Abs(summary.lambda_norm - expected_lambda) < sp.Float("1e-12")
    assert sp.Abs(with_constant.lambda_norm - summary.lambda_norm - 3) < sp.Float(
        "1e-12"
    )


def test_summarize_pauli_hamiltonian_rejects_non_hamiltonian_input():
    """The summary helper fails early on non-Qamomile Hamiltonian objects."""
    with pytest.raises(TypeError, match="qamomile.observable.Hamiltonian"):
        summarize_pauli_hamiltonian(object())


def test_qubitized_qpe_from_model_uses_hamiltonian_metadata():
    """Model-driven QPE estimates reuse Hamiltonian lambda and sparse term count."""
    summary = summarize_pauli_hamiltonian(2 * qm_o.Z(0) + 3 * qm_o.X(1))
    model = ChemistryQPEModel(
        hamiltonian=summary.with_lambda_scale(sp.Rational(1, 2), source="shifted"),
        method=ChemistryQPEMethod.SPARSE,
        walk_cost_toffoli=11,
        truncation_error=sp.Float("1e-5"),
        description="toy sparse model",
    )

    estimate = estimate_qubitized_chemistry_qpe_from_model(model, precision=1)

    assert model.effective_sparsity == 2
    assert estimate.logical_qubits == 2 + sp.sqrt(2)
    assert sp.Abs(estimate.qpe_iterations - sp.Rational(5, 2)) < sp.Float("1e-12")
    assert sp.Abs(estimate.toffoli_gates - sp.Rational(55, 2)) < sp.Float("1e-12")
    assert estimate.assumptions["hamiltonian_source"] == "shifted"
    assert sp.sympify(estimate.assumptions["truncation_error"]) == sp.Float("1e-5")
    assert model.resource_values()[FTQCResourceQuantity.TRUNCATION_ERROR] == sp.Float(
        "1e-5"
    )


def test_single_ancilla_trotter_qpe_from_hamiltonian_summary():
    """Trotter QPE estimates can be driven directly by a Hamiltonian summary."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + 2 * qm_o.X(1))

    estimate = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=3,
        rotation_synthesis_t_gates=5,
    )

    assert estimate.logical_qubits == 3
    assert sp.simplify(estimate.qpe_iterations - 3) == 0
    assert estimate.target_precision == 1
    assert sp.simplify(estimate.logical_depth - 36) == 0
    assert sp.simplify(estimate.t_gates - 180) == 0


def test_single_ancilla_trotter_qpe_accepts_hamiltonian_reduction():
    """Hamiltonian reductions feed early-FTQC Trotter QPE estimates."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + 2 * qm_o.Z(1) + 3 * qm_o.Z(2) + 4 * qm_o.Z(3),
        source="plain_trotter_lcu",
    )
    reference = FTQCReference(
        key="arXiv:2603.22778",
        title="Early FTQC chemistry resource model",
        url="https://arxiv.org/abs/2603.22778",
    )
    reduction = HamiltonianResourceReduction(
        lambda_norm_factor=sp.Rational(1, 10),
        pauli_term_factor=sp.Rational(1, 2),
        description="unitary weight concentration",
        references=(reference,),
    )

    baseline = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        rotation_synthesis_t_gates=3,
    )
    reduced = estimate_single_ancilla_trotter_qpe_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
        resource_reduction=reduction,
    )

    assert sp.Abs(reduced.qpe_iterations - baseline.qpe_iterations / 10) < sp.Float(
        "1e-12"
    )
    assert sp.Abs(reduced.logical_depth - baseline.logical_depth / 40) < sp.Float(
        "1e-12"
    )
    assert sp.Abs(reduced.t_gates - baseline.t_gates / 40) < sp.Float("1e-12")
    assert reduced.assumptions["hamiltonian_source"] == "plain_trotter_lcu:reduced"
    assert (
        reduced.assumptions["resource_reduction_description"]
        == "unitary weight concentration"
    )
    assert "lambda_norm=1/10" in reduced.assumptions["resource_reduction_factors"]
    assert "n_pauli_terms=1/2" in reduced.assumptions["resource_reduction_factors"]
    assert [item.key for item in reduced.references].count("arXiv:2603.22778") == 1


def test_chemistry_qpe_model_rejects_negative_truncation_error():
    """Representation error budgets cannot be negative."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="truncation_error"):
        ChemistryQPEModel(
            hamiltonian=summary,
            walk_cost_toffoli=1,
            truncation_error=-1,
        )


def test_openfermion_qubit_operator_conversion_preserves_terms():
    """OpenFermion qubit operators convert into Qamomile Hamiltonians."""
    openfermion = pytest.importorskip("openfermion")

    operator = (
        openfermion.QubitOperator("Z0", 0.5)
        + openfermion.QubitOperator("X1 Y2", -1.25j)
        + openfermion.QubitOperator((), 0.75)
    )

    hamiltonian = hamiltonian_from_openfermion_qubit_operator(
        operator,
        num_qubits=5,
    )

    assert hamiltonian.num_qubits == 5
    assert sp.sympify(hamiltonian.constant) == sp.Float("0.75")
    assert len(hamiltonian) == 2
    assert hamiltonian.terms[(qm_o.PauliOperator(qm_o.Pauli.Z, 0),)] == 0.5
    assert (
        hamiltonian.terms[
            (
                qm_o.PauliOperator(qm_o.Pauli.X, 1),
                qm_o.PauliOperator(qm_o.Pauli.Y, 2),
            )
        ]
        == -1.25j
    )


def test_openfermion_qubit_operator_summary_extracts_lcu_metadata():
    """OpenFermion summaries feed the FTQC Hamiltonian-resource model."""
    openfermion = pytest.importorskip("openfermion")
    operator = (
        openfermion.QubitOperator("Z0", 0.5)
        + openfermion.QubitOperator("X1 Y2", -1.25j)
        + openfermion.QubitOperator((), 0.75)
    )

    summary = summarize_openfermion_qubit_operator(
        operator,
        n_spin_orbitals=8,
        include_constant=True,
        source="h2_jordan_wigner",
    )

    assert summary.source == "h2_jordan_wigner"
    assert summary.n_spin_orbitals == 8
    assert summary.n_pauli_terms == 2
    assert summary.max_locality == 2
    assert sp.simplify(summary.lambda_norm - sp.Float("2.5")) == 0
    assert summary.constant_included is True


def test_openfermion_qubit_operator_conversion_rejects_malformed_input():
    """Malformed OpenFermion-style inputs fail before resource estimation."""
    with pytest.raises(TypeError, match="terms mapping"):
        hamiltonian_from_openfermion_qubit_operator(object())

    class BadOperator:
        """Hold malformed OpenFermion-like terms for validation tests."""

        terms = {((0, "A"),): 1.0}

    with pytest.raises(ValueError, match="Unsupported OpenFermion Pauli label"):
        hamiltonian_from_openfermion_qubit_operator(BadOperator())
