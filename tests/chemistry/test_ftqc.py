"""Tests for FTQC quantum-chemistry resource estimators."""

from __future__ import annotations

import pytest
import sympy as sp

from qamomile.chemistry import (
    ChemistryQPEMethod,
    estimate_qubitized_chemistry_qpe,
    estimate_single_ancilla_trotter_qpe,
)
from qamomile.resource_estimation import (
    FTQCCostModel,
    estimate_physical_resources,
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

    assert estimate.qubits == n
    assert estimate.gates.oracle_calls["qpe_iterations"] == lam / eps
    assert estimate.gates.multi_qubit == lam * walk / eps
    assert {"lambda", "eps", "C_W"}.issubset(estimate.parameters)


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

    assert sparse.qubits == n + sp.sqrt(sparsity)
    assert single.qubits == n ** sp.Rational(3, 2)
    assert double.qubits == n * sp.sqrt(rank)


def test_physical_estimation_lifts_logical_estimates_to_runtime():
    """Physical estimation applies architecture knobs after logical estimation."""
    model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=0.01,
        factory_qubits=20,
        non_clifford_throughput_per_second=50,
    )

    logical = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=4,
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

    assert baseline.gates.oracle_calls["qpe_iterations"] == 100
    assert concentrated.gates.oracle_calls["qpe_iterations"] == 10
    assert concentrated.gates.t_gates == 15000
    assert concentrated.qubits == 21
    assert concentrated.gates.total < baseline.gates.total


def test_cost_model_rejects_zero_non_clifford_throughput():
    """A zero non-Clifford throughput is invalid because runtime divides by it."""
    with pytest.raises(ValueError, match="non_clifford_throughput_per_second"):
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=1,
            factory_qubits=0,
            non_clifford_throughput_per_second=0,
        )


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


def test_ftqc_substitute_recomputes_free_parameters():
    """Substitution refreshes parameter metadata after symbols become concrete."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)

    estimate = estimate_qubitized_chemistry_qpe(n, lam, eps, walk)
    concrete = estimate.substitute(
        **{
            "n": 4,
            "Xi": 9,
            "lambda": 10,
            "eps": 2,
            "C_W": 3,
        }
    )

    assert concrete.gates.multi_qubit == 15
    assert concrete.parameters == {}


def test_physical_substitute_recomputes_free_parameters():
    """Physical estimates keep architecture symbols separate until substituted."""
    n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
    physical_overhead, cycle_time, factories, throughput = sp.symbols(
        "physical_qubits_per_logical "
        "logical_cycle_time_seconds "
        "factory_qubits "
        "non_clifford_throughput_per_second",
        positive=True,
    )

    logical = estimate_qubitized_chemistry_qpe(n, lam, eps, walk)
    model = FTQCCostModel(
        physical_qubits_per_logical=physical_overhead,
        logical_cycle_time_seconds=cycle_time,
        factory_qubits=factories,
        non_clifford_throughput_per_second=throughput,
    )
    physical = estimate_physical_resources(logical, model)
    concrete = physical.substitute(
        **{
            "n": 4,
            "Xi": 9,
            "lambda": 10,
            "eps": 2,
            "C_W": 3,
        }
    )

    assert concrete.logical.gates.multi_qubit == 15
    assert concrete.non_clifford_count == 15
    assert set(concrete.parameters) == {
        "factory_qubits",
        "logical_cycle_time_seconds",
        "non_clifford_throughput_per_second",
        "physical_qubits_per_logical",
    }
