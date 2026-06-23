"""Tests for Hamiltonian algorithm resource estimates."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.resource_estimation import (
    FTQCCostModel,
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    SurfaceCodeCostModel,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources,
    estimate_qubitized_qpe_resources_from_workload,
    estimate_trotter_qpe_resources,
    estimate_trotter_qpe_resources_from_hamiltonian,
    summarize_pauli_hamiltonian,
)


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
    ],
)
def test_qubitized_qpe_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
):
    """Invalid finite-set representations and non-positive quantities fail."""
    with pytest.raises(ValueError, match=match):
        estimate_qubitized_qpe_resources(**kwargs)


def test_workload_rejects_negative_representation_error():
    """Representation error budgets cannot be negative."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0))

    with pytest.raises(ValueError, match="representation_error"):
        HamiltonianQPEWorkload(
            hamiltonian=summary,
            walk_cost_toffoli=1,
            representation_error=-1,
        )


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
