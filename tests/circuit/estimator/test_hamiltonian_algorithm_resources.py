"""Tests for Hamiltonian algorithm resource estimates."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.resource_estimation import (
    BlockEncodingResource,
    FTQCCostModel,
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    ResourceQuantity,
    SurfaceCodeCostModel,
    TrotterQPEWorkload,
    compare_resource_values,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    estimate_trotter_qpe_resources,
    estimate_trotter_qpe_resources_from_hamiltonian,
    estimate_trotter_qpe_resources_from_workload,
    qubitized_qpe_workload_from_openfermion,
    resource_values_from_estimate,
    summarize_pauli_hamiltonian,
    trotter_qpe_workload_from_openfermion,
)


class QubitOperatorStub:
    """Expose a minimal OpenFermion-style terms mapping for workload tests.

    Attributes:
        terms (dict[tuple[tuple[int, str], ...], int]): Pauli-string terms in
            the same shape as OpenFermion ``QubitOperator.terms``.
    """

    terms = {
        ((0, "Z"),): 2,
        ((1, "X"), (2, "Y")): 3,
        (): 4,
    }


def test_qubitized_qpe_tracks_lambda_precision_and_walk_cost():
    """Qubitized QPE exposes the lambda-over-epsilon iteration contract."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)

    estimate = estimate_qubitized_qpe_resources(
        n,
        lam,
        eps,
        walk,
        representation=HamiltonianRepresentation.TENSOR_HYPERCONTRACTION,
    )

    assert estimate.qubits == n
    assert estimate.gates.oracle_calls["qpe_iterations"] == lam / eps
    assert estimate.gates.multi_qubit == lam * walk / eps
    assert {"lambda", "eps", "C_W"}.issubset(estimate.parameters)


def test_qubitized_qpe_uses_representation_specific_logical_qubits():
    """Representation defaults follow the Hamiltonian workload scaling table."""
    n, sparsity, rank = sp.symbols("n S Xi", positive=True)

    sparse = estimate_qubitized_qpe_resources(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        representation="sparse_pauli_lcu",
        sparsity=sparsity,
    )
    single = estimate_qubitized_qpe_resources(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        representation=HamiltonianRepresentation.SINGLE_FACTORIZATION,
    )
    double = estimate_qubitized_qpe_resources(
        n,
        lambda_norm=10,
        precision=1,
        walk_cost_toffoli=3,
        representation=HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
        second_factor_rank=rank,
    )

    assert sparse.qubits == n + sp.sqrt(sparsity)
    assert single.qubits == n ** sp.Rational(3, 2)
    assert double.qubits == n * sp.sqrt(rank)


def test_hamiltonian_workload_composes_with_surface_code_model():
    """Hamiltonian workloads compose with generic physical resource lifts."""
    hamiltonian = 2 * qm_o.Z(0) + 3 * qm_o.X(1)
    summary = summarize_pauli_hamiltonian(hamiltonian)
    workload = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
    )
    logical = estimate_qubitized_qpe_resources_from_workload(
        workload,
        precision=1,
    )
    physical = estimate_physical_resources(
        logical,
        SurfaceCodeCostModel(
            code_distance=5,
            physical_cycle_time_seconds=1e-6,
            physical_qubits_per_logical_factor=2,
            logical_cycle_factor=3,
            factory_count=2,
            physical_qubits_per_factory=1000,
            factory_cycles_per_non_clifford=4,
        ),
    )

    assert logical.qubits == 2 + sp.sqrt(2)
    assert sp.Abs(logical.gates.multi_qubit - 50) < sp.Float("1e-12")
    assert physical.physical_qubits == 50 * (2 + sp.sqrt(2)) + 2000
    assert physical.non_clifford_count == logical.gates.multi_qubit


def test_hamiltonian_workload_builds_from_block_encoding_contract():
    """Block-encoding contracts become comparable Hamiltonian QPE workloads."""
    summary = summarize_pauli_hamiltonian(
        4 * qm_o.Z(0) + 3 * qm_o.X(1) + qm_o.Z(0) * qm_o.Z(1)
    )
    block = BlockEncodingResource(
        system_qubits=2,
        normalization=5,
        prepare_cost_toffoli=7,
        select_cost_toffoli=11,
        reflection_cost_toffoli=3,
        ancilla_qubits=4,
        name="compressed_df",
    )

    workload = HamiltonianQPEWorkload.from_block_encoding(
        summary,
        block,
        representation=HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF,
        second_factor_rank=2,
        qpe_register_qubits=3,
        representation_error=sp.Rational(1, 4),
    )
    logical = estimate_qubitized_qpe_resources_from_workload(workload, precision=1)

    assert workload.hamiltonian.lambda_norm == 5
    assert workload.hamiltonian.n_pauli_terms == summary.n_pauli_terms
    assert workload.hamiltonian.source == "compressed_df"
    assert workload.walk_cost_toffoli == 28
    assert workload.logical_qubits == 6
    assert workload.qpe_register_qubits == 3
    assert workload.resource_values()["qpe_register_qubits"] == 3
    assert logical.qubits == 9
    assert logical.gates.oracle_calls["qpe_iterations"] == sp.Rational(20, 3)
    assert logical.gates.multi_qubit == sp.Rational(560, 3)


def test_qubitized_workload_builds_from_openfermion_operator():
    """OpenFermion-style operators become auditable QPE workloads."""
    workload = qubitized_qpe_workload_from_openfermion(
        QubitOperatorStub(),
        walk_cost_toffoli=11,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        qpe_register_qubits=2,
        representation_error=sp.Rational(1, 5),
        description="openfermion sparse Pauli LCU",
    )
    logical = estimate_qubitized_qpe_resources_from_workload(workload, precision=1)

    assert workload.hamiltonian.n_qubits == 3
    assert workload.hamiltonian.n_pauli_terms == 2
    assert sp.simplify(workload.hamiltonian.lambda_norm - 5) == 0
    assert sp.simplify(workload.hamiltonian.constant - 4) == 0
    assert workload.hamiltonian.constant_included is False
    assert workload.effective_sparsity == 2
    assert workload.qpe_register_qubits == 2
    assert workload.resource_values_for_precision(1)["algorithmic_precision"] == (
        sp.Rational(4, 5)
    )
    assert logical.qubits == 5 + sp.sqrt(2)
    assert (
        sp.simplify(logical.gates.oracle_calls["qpe_iterations"] - sp.Rational(25, 4))
        == 0
    )
    assert sp.simplify(logical.gates.multi_qubit - sp.Rational(275, 4)) == 0


def test_trotter_qpe_models_unitary_weight_reduction():
    """Unitary-weight concentration reduces the lambda-driven QPE iterations."""
    baseline = estimate_trotter_qpe_resources(
        n_qubits=20,
        n_pauli_terms=100,
        lambda_norm=1000,
        precision=10,
        trotter_steps_per_sample=2,
        samples=5,
    )
    concentrated = estimate_trotter_qpe_resources(
        n_qubits=20,
        n_pauli_terms=100,
        lambda_norm=1000,
        precision=10,
        trotter_steps_per_sample=2,
        samples=5,
        unitary_weight_factor=sp.Rational(1, 10),
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=3,
    )

    assert baseline.gates.oracle_calls["qpe_iterations"] == 100
    assert concentrated.gates.oracle_calls["qpe_iterations"] == 10
    assert concentrated.gates.t_gates == 15000
    assert concentrated.qubits == 21
    assert concentrated.gates.total < baseline.gates.total


def test_trotter_workload_exposes_weight_concentration_drivers():
    """Trotter workloads expose algorithm assumptions before logical lifting."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + 2 * qm_o.X(1))
    workload = TrotterQPEWorkload(
        summary,
        trotter_steps_per_sample=2,
        samples=5,
        unitary_weight_factor=sp.Rational(1, 3),
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=7,
        representation_error=sp.Rational(1, 5),
        description="unitary weight concentration",
    )

    logical = estimate_trotter_qpe_resources_from_workload(workload, precision=1)
    workload_values = workload.resource_values_for_precision(1)
    logical_values = resource_values_from_estimate(logical)

    # The toy Hamiltonian has lambda_norm = |1| + |2| = 3. The
    # unitary-weight factor reduces this to 1, and representation error 1/5
    # leaves algorithmic precision 4/5, so QPE iterations are 1 / (4/5).
    assert sp.Abs(workload.effective_lambda_norm - 1) < sp.Float("1e-12")
    assert sp.Abs(workload_values["effective_lambda_norm"] - 1) < sp.Float("1e-12")
    assert workload_values["algorithmic_precision"] == sp.Rational(4, 5)
    assert sp.Abs(
        logical.gates.oracle_calls["qpe_iterations"] - sp.Rational(5, 4)
    ) < sp.Float("1e-12")
    assert logical.gates.rotation_gates == 10
    assert sp.Abs(logical.gates.t_gates - sp.Rational(175, 2)) < sp.Float("1e-12")
    assert logical_values["pauli_rotations"] == 10
    assert logical.qubits == 3


def test_trotter_workload_builds_from_reported_effective_lambda():
    """Reported effective lambda norms derive the unitary-weight factor."""
    summary = summarize_pauli_hamiltonian(
        3 * qm_o.Z(0) + 5 * qm_o.X(1) + 2 * qm_o.Z(0) * qm_o.Z(1)
    )

    workload = TrotterQPEWorkload.from_effective_lambda_norm(
        summary,
        effective_lambda_norm=2,
        trotter_steps_per_sample=3,
        samples=7,
        randomized_compilation_factor=sp.Rational(1, 3),
        rotation_synthesis_t_gates=11,
        description="reported chemistry estimate",
    )
    logical = estimate_trotter_qpe_resources_from_workload(workload, precision=1)

    assert sp.Abs(summary.lambda_norm - 10) < sp.Float("1e-12")
    assert sp.Abs(workload.unitary_weight_factor - sp.Rational(1, 5)) < sp.Float(
        "1e-12"
    )
    assert sp.Abs(workload.effective_lambda_norm - 2) < sp.Float("1e-12")
    assert sp.Abs(logical.gates.oracle_calls["qpe_iterations"] - 2) < sp.Float("1e-12")
    assert sp.Abs(logical.gates.rotation_gates - 21) < sp.Float("1e-12")
    assert sp.Abs(logical.gates.t_gates - 462) < sp.Float("1e-12")


def test_trotter_workload_builds_from_openfermion_effective_lambda():
    """OpenFermion Trotter workloads can use reported effective lambda."""
    workload = trotter_qpe_workload_from_openfermion(
        QubitOperatorStub(),
        effective_lambda_norm=1,
        trotter_steps_per_sample=2,
        samples=5,
        randomized_compilation_factor=sp.Rational(1, 2),
        rotation_synthesis_t_gates=7,
        description="openfermion unitary weight concentration",
    )
    logical = estimate_trotter_qpe_resources_from_workload(workload, precision=1)

    assert workload.hamiltonian.n_qubits == 3
    assert workload.hamiltonian.n_pauli_terms == 2
    assert sp.simplify(workload.hamiltonian.lambda_norm - 5) == 0
    assert sp.simplify(workload.unitary_weight_factor - sp.Rational(1, 5)) == 0
    assert sp.simplify(workload.effective_lambda_norm - 1) == 0
    assert logical.qubits == 4
    assert sp.simplify(logical.gates.oracle_calls["qpe_iterations"] - 1) == 0
    assert logical.gates.rotation_gates == 10
    assert sp.simplify(logical.gates.t_gates - 70) == 0


def test_trotter_openfermion_workload_rejects_conflicting_weight_inputs():
    """Effective lambda and direct unitary-weight factors are exclusive."""
    with pytest.raises(ValueError, match="effective_lambda_norm"):
        trotter_qpe_workload_from_openfermion(
            QubitOperatorStub(),
            effective_lambda_norm=1,
            unitary_weight_factor=sp.Rational(1, 2),
            trotter_steps_per_sample=1,
            samples=1,
        )


def test_trotter_effective_lambda_constructor_rejects_zero_original_lambda():
    """Effective-lambda workloads need a positive original Hamiltonian norm."""
    summary = summarize_pauli_hamiltonian(qm_o.Hamiltonian(num_qubits=1))

    with pytest.raises(ValueError, match="lambda_norm"):
        TrotterQPEWorkload.from_effective_lambda_norm(
            summary,
            effective_lambda_norm=1,
            trotter_steps_per_sample=1,
            samples=1,
        )


def test_trotter_qpe_from_hamiltonian_summary():
    """Trotter QPE estimates can be driven directly by a Hamiltonian summary."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + 2 * qm_o.X(1))

    estimate = estimate_trotter_qpe_resources_from_hamiltonian(
        summary,
        precision=1,
        trotter_steps_per_sample=2,
        samples=3,
        rotation_synthesis_t_gates=5,
    )

    assert estimate.qubits == 3
    assert sp.simplify(estimate.gates.oracle_calls["qpe_iterations"] - 3) == 0
    assert sp.simplify(estimate.gates.rotation_gates - 12) == 0
    assert sp.simplify(estimate.gates.t_gates - 180) == 0


def test_physical_estimation_lifts_logical_estimates_to_runtime():
    """Physical estimation applies architecture knobs after logical estimation."""
    model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=0.01,
        factory_qubits=20,
        non_clifford_throughput_per_second=50,
    )

    logical = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
    )
    physical = estimate_physical_resources(logical, model)

    assert logical.qubits == 7
    assert logical.gates.total == 40
    assert physical.physical_qubits == 720
    assert physical.non_clifford_count == 40
    assert physical.runtime_seconds == sp.Rational(4, 5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "n_qubits": 0,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
            },
            "n_qubits",
        ),
        (
            {
                "n_qubits": 1,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
                "representation": "not-a-representation",
            },
            "Unknown Hamiltonian representation",
        ),
        (
            {
                "n_qubits": 1,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
                "representation": "sparse_pauli_lcu",
            },
            "sparsity is required",
        ),
        (
            {
                "n_qubits": 1,
                "lambda_norm": 1,
                "precision": 1,
                "walk_cost_toffoli": 1,
                "qpe_register_qubits": -1,
            },
            "qpe_register_qubits",
        ),
    ],
)
def test_qubitized_qpe_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
):
    """Invalid finite-set representations and non-positive quantities fail."""
    with pytest.raises(ValueError, match=match):
        estimate_qubitized_qpe_resources(**kwargs)


def test_qubitized_qpe_adds_qpe_register_qubits_to_logical_footprint():
    """Explicit QPE readout registers are counted in logical qubits."""
    estimate = estimate_qubitized_qpe_resources(
        n_qubits=4,
        lambda_norm=8,
        precision=2,
        walk_cost_toffoli=10,
        logical_qubits=7,
        qpe_register_qubits=3,
    )

    assert estimate.qubits == 10
    assert estimate.gates.oracle_calls["qpe_iterations"] == 4
    assert estimate.gates.multi_qubit == 40


def test_workload_rejects_negative_representation_error():
    """Representation error budgets cannot be negative."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="representation_error"):
        HamiltonianQPEWorkload(
            hamiltonian=summary,
            walk_cost_toffoli=1,
            representation_error=-1,
        )


def test_workload_rejects_negative_qpe_register_qubits():
    """QPE register qubit metadata cannot be negative."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="qpe_register_qubits"):
        HamiltonianQPEWorkload(
            hamiltonian=summary,
            walk_cost_toffoli=1,
            qpe_register_qubits=-1,
        )


def test_workload_representation_error_consumes_precision_budget():
    """Representation error increases QPE iterations by reducing precision."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0) + 2 * qm_o.X(1))
    exact = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
    )
    approximate = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
        representation_error=sp.Rational(1, 4),
    )

    exact_logical = estimate_qubitized_qpe_resources_from_workload(
        exact,
        precision=1,
    )
    approximate_logical = estimate_qubitized_qpe_resources_from_workload(
        approximate,
        precision=1,
    )
    rows = compare_resource_values(
        approximate.resource_values_for_precision(1),
        exact.resource_values_for_precision(1),
        quantities=(ResourceQuantity.REPRESENTATION_ERROR,),
    )
    precision_rows = compare_resource_values(
        exact.resource_values_for_precision(1),
        approximate.resource_values_for_precision(1),
        quantities=(
            ResourceQuantity.TARGET_PRECISION,
            ResourceQuantity.ALGORITHMIC_PRECISION,
        ),
    )

    # The toy Hamiltonian has lambda_norm = |1| + |2| = 3. With no
    # representation error, qpe_iterations = 3 / 1. With error 1/4,
    # the remaining algorithmic precision is 3/4, so iterations = 3 / (3/4).
    assert sp.Abs(exact_logical.gates.oracle_calls["qpe_iterations"] - 3) < sp.Float(
        "1e-12"
    )
    assert approximate.algorithmic_precision(1) == sp.Rational(3, 4)
    assert sp.Abs(
        approximate_logical.gates.oracle_calls["qpe_iterations"] - 4
    ) < sp.Float("1e-12")
    assert rows[0].baseline == sp.Rational(1, 4)
    assert rows[0].candidate == 0
    assert precision_rows[0].quantity == ResourceQuantity.TARGET_PRECISION
    assert precision_rows[0].ratio == 1
    assert precision_rows[1].quantity == ResourceQuantity.ALGORITHMIC_PRECISION
    assert precision_rows[1].baseline == 1
    assert precision_rows[1].candidate == sp.Rational(3, 4)
    assert precision_rows[1].ratio == sp.Rational(3, 4)


def test_workload_rejects_exhausted_precision_budget():
    """Representation error must leave positive precision for QPE."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))
    workload = HamiltonianQPEWorkload(
        hamiltonian=summary,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        walk_cost_toffoli=10,
        representation_error=1,
    )

    with pytest.raises(ValueError, match="algorithmic_precision"):
        estimate_qubitized_qpe_resources_from_workload(workload, precision=1)

    with pytest.raises(ValueError, match="algorithmic_precision"):
        workload.resource_values_for_precision(1)


def test_logical_and_physical_substitute_recompute_free_parameters():
    """Substitution refreshes parameter metadata across logical and physical estimates."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
    overhead, cycle_time, factories, throughput = sp.symbols(
        "physical_qubits_per_logical "
        "logical_cycle_time_seconds "
        "factory_qubits "
        "non_clifford_throughput_per_second",
        positive=True,
    )

    logical = estimate_qubitized_qpe_resources(n, lam, eps, walk)
    concrete_logical = logical.substitute(
        **{
            "n": 4,
            "Xi": 9,
            "lambda": 10,
            "eps": 2,
            "C_W": 3,
        }
    )
    physical = estimate_physical_resources(
        logical,
        FTQCCostModel(
            physical_qubits_per_logical=overhead,
            logical_cycle_time_seconds=cycle_time,
            factory_qubits=factories,
            non_clifford_throughput_per_second=throughput,
        ),
    )
    concrete_physical = physical.substitute(
        **{
            "n": 4,
            "Xi": 9,
            "lambda": 10,
            "eps": 2,
            "C_W": 3,
        }
    )

    assert concrete_logical.gates.multi_qubit == 15
    assert concrete_logical.parameters == {}
    assert concrete_physical.logical.gates.multi_qubit == 15
    assert concrete_physical.non_clifford_count == 15
    assert set(concrete_physical.parameters) == {
        "factory_qubits",
        "logical_cycle_time_seconds",
        "non_clifford_throughput_per_second",
        "physical_qubits_per_logical",
    }
