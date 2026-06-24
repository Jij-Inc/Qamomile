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
    ResourceParetoRow,
    ResourceQuantity,
    ResourceScenarioValueRow,
    ResourceSymbolDependencyRow,
    ResourceSymbolDriverRow,
    SurfaceCodeCostModel,
    audit_resource_value_drivers,
    audit_resource_value_symbols,
    compare_resource_values,
    describe_resource_quantity,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    evaluate_resource_value_scenarios,
    evaluate_resource_values,
    iter_resource_quantity_specs,
    pareto_resource_values,
    resource_values_from_estimate,
    summarize_pauli_hamiltonian,
)


def test_quantity_specs_cover_core_resource_layers():
    """The quantity catalog covers problem, logical, physical, and architecture layers."""
    specs = iter_resource_quantity_specs()
    quantities = {spec.quantity for spec in specs}
    categories = {spec.category for spec in specs}

    assert ResourceQuantity.LAMBDA_NORM in quantities
    assert ResourceQuantity.TARGET_PRECISION in quantities
    assert ResourceQuantity.ALGORITHMIC_PRECISION in quantities
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


def test_pareto_resource_values_marks_frontier_candidates():
    """Pareto rows mark dominated candidates while preserving tradeoffs."""
    baseline = {
        ResourceQuantity.PHYSICAL_QUBITS: 1000,
        ResourceQuantity.RUNTIME_SECONDS: 100,
        ResourceQuantity.PHYSICAL_QUBIT_SECONDS: 100_000,
    }
    compressed = {
        ResourceQuantity.PHYSICAL_QUBITS: 700,
        ResourceQuantity.RUNTIME_SECONDS: 80,
        ResourceQuantity.PHYSICAL_QUBIT_SECONDS: 56_000,
    }
    tiny_slow = {
        ResourceQuantity.PHYSICAL_QUBITS: 450,
        ResourceQuantity.RUNTIME_SECONDS: 140,
        ResourceQuantity.PHYSICAL_QUBIT_SECONDS: 63_000,
    }

    rows = pareto_resource_values(
        (
            ("baseline", baseline),
            ("compressed", compressed),
            ("tiny slow", tiny_slow),
        ),
        quantities=(
            ResourceQuantity.PHYSICAL_QUBITS,
            ResourceQuantity.RUNTIME_SECONDS,
        ),
    )
    by_label = {row.label: row for row in rows}

    assert all(isinstance(row, ResourceParetoRow) for row in rows)
    assert by_label["baseline"].dominated_by == ("compressed",)
    assert by_label["baseline"].is_frontier is False
    assert by_label["compressed"].is_frontier is True
    assert by_label["tiny slow"].is_frontier is True
    assert [row.label for row in rows if row.is_frontier] == [
        "compressed",
        "tiny slow",
    ]
    assert by_label["baseline"].to_dict()["values"] == {
        "physical_qubits": "1000",
        "runtime_seconds": "100",
    }


def test_pareto_resource_values_keeps_symbolic_rows_on_frontier():
    """Undecidable symbolic dominance keeps candidates reviewable."""
    symbolic_runtime = sp.Symbol("runtime")

    rows = pareto_resource_values(
        {
            "concrete": {
                ResourceQuantity.PHYSICAL_QUBITS: 100,
                ResourceQuantity.RUNTIME_SECONDS: 10,
            },
            "symbolic": {
                ResourceQuantity.PHYSICAL_QUBITS: 90,
                ResourceQuantity.RUNTIME_SECONDS: symbolic_runtime,
            },
        },
        quantities=(
            ResourceQuantity.PHYSICAL_QUBITS,
            ResourceQuantity.RUNTIME_SECONDS,
        ),
    )

    assert [row.label for row in rows if row.is_frontier] == [
        "concrete",
        "symbolic",
    ]
    assert all(not row.dominated_by for row in rows)
    with pytest.raises(ValueError, match="at least two"):
        pareto_resource_values({"only": {ResourceQuantity.RUNTIME_SECONDS: 1}})
    with pytest.raises(ValueError, match="unique"):
        pareto_resource_values(
            (
                ("duplicate", {ResourceQuantity.RUNTIME_SECONDS: 1}),
                ("duplicate", {ResourceQuantity.RUNTIME_SECONDS: 2}),
            )
        )
    with pytest.raises(ValueError, match="missing"):
        pareto_resource_values(
            {
                "concrete": {
                    ResourceQuantity.PHYSICAL_QUBITS: 100,
                    ResourceQuantity.RUNTIME_SECONDS: 10,
                },
                "symbolic": {ResourceQuantity.PHYSICAL_QUBITS: 90},
            },
            quantities=(ResourceQuantity.RUNTIME_SECONDS,),
        )
    with pytest.raises(ValueError, match="No common"):
        pareto_resource_values(
            {
                "qubits": {ResourceQuantity.PHYSICAL_QUBITS: 100},
                "runtime": {ResourceQuantity.RUNTIME_SECONDS: 10},
            }
        )


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
    assert ResourceQuantity.TARGET_PRECISION not in quantities
    assert ResourceQuantity.ALGORITHMIC_PRECISION not in quantities
    assert ResourceQuantity.LAMBDA_NORM in quantities
    assert ResourceQuantity.WALK_COST_TOFFOLI in quantities


def test_compare_resource_values_accepts_precision_aware_workload_values():
    """Precision-aware workload values expose target and QPE error budgets."""
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
    )
    rows = compare_resource_values(
        baseline.resource_values_for_precision(1),
        candidate.resource_values_for_precision(1),
        quantities=(
            ResourceQuantity.TARGET_PRECISION,
            ResourceQuantity.ALGORITHMIC_PRECISION,
        ),
    )

    assert rows[0].quantity == ResourceQuantity.TARGET_PRECISION
    assert rows[0].baseline == 1
    assert rows[0].candidate == 1
    assert rows[0].ratio == 1
    assert rows[1].quantity == ResourceQuantity.ALGORITHMIC_PRECISION
    assert rows[1].baseline == 1
    assert rows[1].candidate == sp.Rational(9, 10)
    assert rows[1].ratio == sp.Rational(9, 10)


def test_audit_resource_value_symbols_reports_unresolved_quantities():
    """Symbol audits expose which canonical quantities still depend on parameters."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
    estimate = estimate_qubitized_qpe_resources(
        n_qubits=n,
        lambda_norm=lam,
        precision=eps,
        walk_cost_toffoli=walk,
        logical_qubits=n + 2,
    )

    rows = audit_resource_value_symbols(
        estimate,
        quantities=(
            ResourceQuantity.LOGICAL_QUBITS,
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
        ),
    )

    assert all(isinstance(row, ResourceSymbolDependencyRow) for row in rows)
    assert rows[0].quantity == ResourceQuantity.LOGICAL_QUBITS
    assert rows[0].symbols == ("n",)
    assert rows[0].is_symbolic is True
    assert rows[1].quantity == ResourceQuantity.QPE_ITERATIONS
    assert rows[1].symbols == ("eps", "lambda")
    assert rows[2].quantity == ResourceQuantity.NON_CLIFFORD_COUNT
    assert rows[2].symbols == ("C_W", "eps", "lambda")
    assert rows[2].to_dict()["symbols"] == ["C_W", "eps", "lambda"]


def test_audit_resource_value_symbols_filters_resolved_values():
    """Symbol audits can focus on unresolved quantities only."""
    values = {
        ResourceQuantity.LOGICAL_QUBITS: 5,
        ResourceQuantity.RUNTIME_SECONDS: sp.Symbol("runtime_seconds"),
        ResourceQuantity.PHYSICAL_QUBITS: sp.Integer(1000),
    }

    rows = audit_resource_value_symbols(values, include_resolved=False)

    assert len(rows) == 1
    assert rows[0].quantity == ResourceQuantity.RUNTIME_SECONDS
    assert rows[0].symbols == ("runtime_seconds",)
    assert rows[0].to_dict()["is_symbolic"] is True
    with pytest.raises(ValueError, match="missing"):
        audit_resource_value_symbols(
            values,
            quantities=(ResourceQuantity.NON_CLIFFORD_COUNT,),
        )


def test_audit_resource_value_drivers_groups_impacted_quantities():
    """Driver audits invert symbol dependencies into symbol-centered rows."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
    estimate = estimate_qubitized_qpe_resources(
        n_qubits=n,
        lambda_norm=lam,
        precision=eps,
        walk_cost_toffoli=walk,
        logical_qubits=n + 2,
    )

    rows = audit_resource_value_drivers(
        estimate,
        quantities=(
            ResourceQuantity.LOGICAL_QUBITS,
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
        ),
    )
    by_symbol = {row.symbol: row for row in rows}

    assert all(isinstance(row, ResourceSymbolDriverRow) for row in rows)
    assert tuple(by_symbol) == ("C_W", "eps", "lambda", "n")
    assert by_symbol["lambda"].quantities == (
        ResourceQuantity.QPE_ITERATIONS,
        ResourceQuantity.NON_CLIFFORD_COUNT,
    )
    assert by_symbol["lambda"].categories == (
        ResourceCategory.ALGORITHM,
        ResourceCategory.LOGICAL,
    )
    assert by_symbol["lambda"].quantity_count == 2
    assert by_symbol["lambda"].to_dict()["quantities"] == [
        "qpe_iterations",
        "non_clifford_count",
    ]
    assert by_symbol["n"].quantities == (ResourceQuantity.LOGICAL_QUBITS,)


def test_audit_resource_value_drivers_crosses_architecture_and_physical_layers():
    """Driver audits expose architecture symbols that influence physical outputs."""
    d, cycle_time = sp.symbols("d cycle_time", positive=True)
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=1,
        walk_cost_toffoli=10,
        logical_qubits=12,
    )
    model = SurfaceCodeCostModel(
        code_distance=d,
        physical_cycle_time_seconds=cycle_time,
        physical_qubits_per_logical_factor=2,
        logical_cycle_factor=3,
        factory_count=4,
        physical_qubits_per_factory=1000,
        factory_cycles_per_non_clifford=2,
    )
    physical = estimate_physical_resources(logical, model)

    rows = audit_resource_value_drivers(
        physical,
        quantities=(
            ResourceQuantity.CODE_DISTANCE,
            ResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
            ResourceQuantity.PHYSICAL_QUBITS,
            ResourceQuantity.RUNTIME_SECONDS,
        ),
    )
    by_symbol = {row.symbol: row for row in rows}

    assert by_symbol["d"].quantities == (
        ResourceQuantity.CODE_DISTANCE,
        ResourceQuantity.PHYSICAL_QUBITS,
        ResourceQuantity.RUNTIME_SECONDS,
    )
    assert by_symbol["d"].categories == (
        ResourceCategory.ARCHITECTURE,
        ResourceCategory.PHYSICAL,
    )
    assert by_symbol["cycle_time"].quantities == (
        ResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
        ResourceQuantity.RUNTIME_SECONDS,
    )
    assert by_symbol["cycle_time"].categories == (
        ResourceCategory.ARCHITECTURE,
        ResourceCategory.PHYSICAL,
    )
    with pytest.raises(ValueError, match="missing"):
        audit_resource_value_drivers(
            physical,
            quantities=(ResourceQuantity.LAMBDA_NORM,),
        )


def test_evaluate_resource_values_resolves_one_scenario():
    """Scenario evaluation substitutes symbolic resource quantities."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
    estimate = estimate_qubitized_qpe_resources(
        n_qubits=n,
        lambda_norm=lam,
        precision=eps,
        walk_cost_toffoli=walk,
        logical_qubits=n + 2,
    )

    rows = evaluate_resource_values(
        estimate,
        {"n": 10, lam: 5, "eps": sp.Rational(1, 10), "C_W": 7},
        scenario="small",
        quantities=(
            ResourceQuantity.LOGICAL_QUBITS,
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
        ),
    )

    assert all(isinstance(row, ResourceScenarioValueRow) for row in rows)
    assert [row.scenario for row in rows] == ["small", "small", "small"]
    assert [row.quantity for row in rows] == [
        ResourceQuantity.LOGICAL_QUBITS,
        ResourceQuantity.QPE_ITERATIONS,
        ResourceQuantity.NON_CLIFFORD_COUNT,
    ]
    assert [row.value for row in rows] == [12, 50, 350]
    assert all(row.is_resolved for row in rows)
    assert rows[2].to_dict()["expression"] == "C_W*lambda/eps"
    assert rows[2].to_dict()["value"] == "350"


def test_evaluate_resource_value_scenarios_reports_remaining_symbols():
    """Scenario tables can either keep or reject unresolved symbols."""
    d, n, runtime = sp.symbols("d n runtime", positive=True)
    values = {
        ResourceQuantity.PHYSICAL_QUBITS: 2 * d**2 * n,
        ResourceQuantity.RUNTIME_SECONDS: runtime / d,
    }

    rows = evaluate_resource_value_scenarios(
        values,
        {
            "distance-7": {"d": 7},
            "distance-9": {"d": 9, runtime: 18},
        },
        quantities=(
            ResourceQuantity.PHYSICAL_QUBITS,
            ResourceQuantity.RUNTIME_SECONDS,
        ),
        require_resolved=False,
    )

    assert [row.scenario for row in rows] == [
        "distance-7",
        "distance-7",
        "distance-9",
        "distance-9",
    ]
    assert rows[0].value == 98 * n
    assert rows[0].symbols == ("n",)
    assert rows[1].symbols == ("runtime",)
    assert rows[3].value == 2
    assert rows[3].is_resolved is True
    with pytest.raises(ValueError, match="distance-7:physical_qubits"):
        evaluate_resource_value_scenarios(
            values,
            {"distance-7": {"d": 7}},
            quantities=(ResourceQuantity.PHYSICAL_QUBITS,),
        )
    with pytest.raises(ValueError, match="at least one"):
        evaluate_resource_value_scenarios(values, {})


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
