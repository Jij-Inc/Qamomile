"""Tests for FTQC quantum-chemistry resource estimators."""

from __future__ import annotations

import pytest
import sympy as sp

from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    FTQCCostModel,
    estimate_qubitized_chemistry_qpe,
    estimate_single_ancilla_trotter_qpe,
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
    assert estimate.toffoli_gates == lam * walk / eps
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

    assert sparse.logical_qubits == n + sp.sqrt(sparsity)
    assert single.logical_qubits == n ** sp.Rational(3, 2)
    assert double.logical_qubits == n * sp.sqrt(rank)


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


def test_cost_model_rejects_zero_non_clifford_throughput():
    """A zero non-Clifford throughput is invalid because runtime divides by it."""
    with pytest.raises(ValueError, match="toffoli_throughput_per_second"):
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=1,
            factory_qubits=0,
            toffoli_throughput_per_second=0,
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
