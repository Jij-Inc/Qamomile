"""Tests for FTQC quantum-chemistry resource estimators."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCCostModel,
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
    assert sp.simplify(estimate.logical_depth - 36) == 0
    assert sp.simplify(estimate.t_gates - 180) == 0


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
