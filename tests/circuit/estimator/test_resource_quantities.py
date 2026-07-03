"""Tests for canonical resource quantity metadata and comparisons."""

from __future__ import annotations

from dataclasses import replace

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.resource_estimation import (
    ActiveVolumeCostModel,
    FTQCCostModel,
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    ResourceCategory,
    ResourceEstimate,
    ResourceQuantity,
    ResourceQuantitySpec,
    SupportsResourceValues,
    SurfaceCodeCostModel,
    compare_resource_values,
    describe_resource_quantity,
    estimate_active_volume_resources,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    iter_resource_quantity_specs,
    register_resource_quantity,
    resource_estimate_expressions,
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
    assert ResourceQuantity.LOGICAL_OPERATIONS in quantities
    assert ResourceQuantity.PHYSICAL_QUBITS in quantities
    assert ResourceQuantity.ACTIVE_VOLUME in quantities
    assert ResourceQuantity.ACTIVE_VOLUME_RUNTIME_SECONDS in quantities
    assert ResourceQuantity.DEPTH_LIMITED_RUNTIME_SECONDS in quantities
    assert ResourceQuantity.NON_CLIFFORD_LIMITED_RUNTIME_SECONDS in quantities
    assert ResourceQuantity.PHYSICAL_QUBIT_SECONDS in quantities
    assert ResourceQuantity.CODE_DISTANCE in quantities
    assert ResourceQuantity.FACTORY_COUNT in quantities
    assert ResourceQuantity.ACTIVE_VOLUME_THROUGHPUT_PER_SECOND in quantities
    assert {
        ResourceCategory.PROBLEM,
        ResourceCategory.ALGORITHM,
        ResourceCategory.LOGICAL,
        ResourceCategory.PHYSICAL,
        ResourceCategory.ARCHITECTURE,
    }.issubset(categories)
    assert len(quantities) == len(specs)


def test_resource_estimation_facade_exports_symbolic_value_helpers():
    """The public facade exposes value protocols and expression collectors."""
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=1,
        walk_cost_toffoli=3,
    )
    expressions = resource_estimate_expressions(logical)
    physical: SupportsResourceValues = estimate_physical_resources(
        logical,
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=sp.Rational(1, 1_000_000),
            factory_qubits=1000,
            non_clifford_throughput_per_second=100,
        ),
    )
    active_volume: SupportsResourceValues = estimate_active_volume_resources(
        logical,
        ActiveVolumeCostModel(
            active_volume_per_logical_gate=1,
            active_volume_per_non_clifford=0,
            active_volume_throughput_per_second=10,
        ),
    )

    assert logical.qubits in expressions
    assert logical.gates.total in expressions
    assert ResourceQuantity.LOGICAL_QUBITS.value in physical.resource_values()
    assert ResourceQuantity.ACTIVE_VOLUME.value in active_volume.resource_values()


def test_describe_resource_quantity_normalizes_strings():
    """String quantity keys resolve to metadata with stable units."""
    spec = describe_resource_quantity("lambda_norm")

    assert spec.quantity == ResourceQuantity.LAMBDA_NORM
    assert spec.category == ResourceCategory.PROBLEM
    assert spec.unit == "energy"


def test_describe_resource_quantity_rejects_unregistered_key():
    """Metadata lookup fails clearly for keys with no registered spec."""
    with pytest.raises(ValueError, match="No metadata registered"):
        describe_resource_quantity("not-a-resource")


def test_compare_accepts_unregistered_custom_quantity():
    """Custom string quantity keys compare without prior registration."""
    baseline = {"grover_oracle_calls": 40, "logical_qubits": 8}
    candidate = {"grover_oracle_calls": 10, "logical_qubits": 8}

    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=("grover_oracle_calls",),
    )

    assert rows[0].quantity == "grover_oracle_calls"
    assert sp.simplify(rows[0].ratio - sp.Rational(1, 4)) == 0
    assert rows[0].label == "grover_oracle_calls"
    assert rows[0].unit == ""
    assert rows[0].category is None
    assert rows[0].to_dict()["category"] == ""


def test_default_comparison_selection_includes_unregistered_keys():
    """Auto-selected quantities cover registered keys first, then custom keys."""
    baseline = {"logical_qubits": 8, "my_custom_counter": 4}
    candidate = {"logical_qubits": 8, "my_custom_counter": 2}

    rows = compare_resource_values(baseline, candidate)
    keys = [row.quantity for row in rows]

    assert keys == ["logical_qubits", "my_custom_counter"]


@pytest.fixture
def _restore_quantity_registry():
    """Snapshot and restore the process-global quantity registry."""
    from qamomile.resource_estimation.quantities import _QUANTITY_REGISTRY

    snapshot = dict(_QUANTITY_REGISTRY)
    yield
    _QUANTITY_REGISTRY.clear()
    _QUANTITY_REGISTRY.update(snapshot)


def test_register_resource_quantity_supplies_comparison_metadata(
    _restore_quantity_registry,
):
    """Registered custom specs drive labels; registration is last-wins."""
    spec = register_resource_quantity(
        ResourceQuantitySpec(
            quantity="test_registered_rounds",
            label="Test rounds",
            unit="rounds",
            category=ResourceCategory.ALGORITHM,
            description="Rounds used only by this test.",
        )
    )

    assert describe_resource_quantity("test_registered_rounds") == spec
    assert spec in iter_resource_quantity_specs()

    rows = compare_resource_values(
        {"test_registered_rounds": 6},
        {"test_registered_rounds": 3},
    )
    assert rows[0].label == "Test rounds"
    assert rows[0].unit == "rounds"
    assert rows[0].category == ResourceCategory.ALGORITHM

    # Registration is last-wins so notebook re-execution keeps working.
    replacement = register_resource_quantity(
        ResourceQuantitySpec(
            quantity="test_registered_rounds",
            label="Replaced label",
            unit="rounds",
            category=ResourceCategory.ALGORITHM,
            description="Replacement metadata.",
        )
    )
    assert describe_resource_quantity("test_registered_rounds") == replacement


def test_spec_rejects_non_string_quantity_key():
    """Non-string quantity keys fail loudly instead of mis-keying."""
    import enum

    class _OtherEnum(enum.Enum):
        ROUNDS = "rounds"

    with pytest.raises(TypeError, match="must be strings"):
        ResourceQuantitySpec(
            quantity=_OtherEnum.ROUNDS,  # type: ignore[arg-type]
            label="Rounds",
            unit="rounds",
            category=ResourceCategory.ALGORITHM,
            description="Mis-keyed spec.",
        )


def test_default_comparison_rejects_disjoint_inputs():
    """A typo that empties the default selection raises instead of ().."""
    with pytest.raises(ValueError, match="share no nonzero-baseline"):
        compare_resource_values(
            {"logical_qubit": 8},
            {"logical_qubits": 4},
        )


def test_oracle_collision_with_canonical_key_warns():
    """Colliding oracle counters warn instead of silently vanishing."""
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
    )
    renamed = ResourceEstimate(
        qubits=logical.qubits,
        gates=replace(
            logical.gates,
            oracle_calls={"t_gates": sp.Integer(9999)},
        ),
    )

    with pytest.warns(UserWarning, match="collides with a canonical"):
        values = resource_values_from_estimate(renamed)
    assert values["t_gates"] == renamed.gates.t_gates


def test_oracle_call_counters_pass_through_generically():
    """Every oracle-call counter surfaces in resource values by its own name."""
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
    )
    renamed = ResourceEstimate(
        qubits=logical.qubits,
        gates=replace(
            logical.gates,
            oracle_calls={"walk_calls": sp.Integer(4), "qpe_iterations": sp.Integer(4)},
        ),
    )

    values = resource_values_from_estimate(renamed)
    assert values["walk_calls"] == 4
    assert values["qpe_iterations"] == 4

    physical = estimate_physical_resources(
        renamed,
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=sp.Rational(1, 100),
            factory_qubits=20,
            non_clifford_throughput_per_second=50,
        ),
    )
    assert physical.resource_values()["walk_calls"] == 4

    rows = compare_resource_values(renamed, renamed, quantities=("walk_calls",))
    assert rows[0].ratio == 1


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


def test_active_volume_model_exposes_operation_volume_resources():
    """Active-volume models expose operation-volume runtime proxies."""
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=10,
        precision=2,
        walk_cost_toffoli=5,
        representation=HamiltonianRepresentation.TENSOR_HYPERCONTRACTION,
    )
    cost_model = ActiveVolumeCostModel(
        active_volume_per_logical_gate=3,
        active_volume_per_non_clifford=2,
        active_volume_throughput_per_second=10,
    )

    active_volume = estimate_active_volume_resources(logical, cost_model)
    values = active_volume.resource_values()

    assert active_volume.logical_gate_count == 25
    assert active_volume.non_clifford_count == 25
    assert active_volume.active_volume == 125
    assert active_volume.runtime_seconds == sp.Rational(25, 2)
    assert values["logical_operations"] == 25
    assert values["active_volume"] == 125
    assert values["active_volume_runtime_seconds"] == sp.Rational(25, 2)
    assert values["runtime_seconds"] == sp.Rational(25, 2)
    assert values["active_volume_per_logical_gate"] == 3


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


def test_physical_estimates_expose_runtime_bottleneck_components():
    """Physical estimates expose depth and non-Clifford runtime components."""
    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=20,
        precision=1,
        walk_cost_toffoli=5,
        representation=HamiltonianRepresentation.TENSOR_HYPERCONTRACTION,
    )
    cost_model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Rational(1, 1000),
        factory_qubits=0,
        non_clifford_throughput_per_second=10,
    )

    physical = estimate_physical_resources(
        logical,
        cost_model,
        logical_depth=50,
        non_clifford_count=200,
    )
    values = physical.resource_values()
    serialized = physical.to_dict()

    assert physical.depth_limited_runtime_seconds == sp.Rational(1, 20)
    assert physical.non_clifford_limited_runtime_seconds == 20
    assert physical.runtime_seconds == 20
    assert values[ResourceQuantity.DEPTH_LIMITED_RUNTIME_SECONDS.value] == sp.Rational(
        1, 20
    )
    assert values[ResourceQuantity.NON_CLIFFORD_LIMITED_RUNTIME_SECONDS.value] == 20
    assert serialized["depth_limited_runtime_seconds"] == "1/20"
    assert serialized["non_clifford_limited_runtime_seconds"] == "20"


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
