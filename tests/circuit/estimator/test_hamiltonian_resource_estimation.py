"""Tests for Hamiltonian resource-estimation primitives."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.resource_estimation import (
    PauliHamiltonianResource,
    hamiltonian_from_openfermion_qubit_operator,
    summarize_openfermion_qubit_operator,
    summarize_pauli_hamiltonian,
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
    assert summary.n_qubits == 3
    assert summary.n_pauli_terms == 2
    assert summary.max_locality == 2
    assert sp.simplify(summary.constant - 3) == 0
    expected_lambda = sp.Float("0.5") + sp.sqrt(sp.Float("2.125"))
    assert sp.Abs(summary.lambda_norm - expected_lambda) < sp.Float("1e-12")
    assert sp.Abs(with_constant.lambda_norm - summary.lambda_norm - 3) < sp.Float(
        "1e-12"
    )


def test_summarize_pauli_hamiltonian_accepts_qubit_count_override():
    """Hamiltonian summaries can carry an externally chosen encoded width."""
    summary = summarize_pauli_hamiltonian(qm_o.Z(0), n_qubits=4)

    assert summary.n_qubits == 4
    assert summary.n_pauli_terms == 1


def test_pauli_hamiltonian_resource_rescales_lambda_norm():
    """Hamiltonian summaries support representation-level norm reductions."""
    summary = PauliHamiltonianResource(
        n_qubits=3,
        n_pauli_terms=5,
        lambda_norm=20,
        max_locality=2,
        source="original",
    )

    reduced = summary.with_lambda_scale(sp.Rational(1, 4), source="reduced")

    assert reduced.n_qubits == 3
    assert reduced.lambda_norm == 5
    assert reduced.source == "reduced"


def test_summarize_pauli_hamiltonian_rejects_non_hamiltonian_input():
    """The summary helper fails early on non-Qamomile Hamiltonian objects."""
    with pytest.raises(TypeError, match="qamomile.observable.Hamiltonian"):
        summarize_pauli_hamiltonian(object())


def test_openfermion_style_qubit_operator_converts_to_hamiltonian_summary():
    """OpenFermion-style term mappings feed the same Hamiltonian summary path."""

    class QubitOperatorStub:
        """Expose a minimal OpenFermion-style ``terms`` mapping."""

        terms = {
            ((0, "Z"),): 2,
            ((1, "X"), (2, "Y")): 3,
            (): 4,
        }

    hamiltonian = hamiltonian_from_openfermion_qubit_operator(QubitOperatorStub())
    summary = summarize_openfermion_qubit_operator(QubitOperatorStub())

    assert hamiltonian.num_qubits == 3
    assert hamiltonian.constant == 4
    assert summary.n_qubits == 3
    assert summary.n_pauli_terms == 2
    assert sp.Abs(summary.lambda_norm - 5) < sp.Float("1e-12")


def test_openfermion_style_qubit_operator_rejects_invalid_pauli_label():
    """OpenFermion conversion rejects unsupported Pauli labels early."""

    class QubitOperatorStub:
        """Expose an invalid OpenFermion-style ``terms`` mapping."""

        terms = {((0, "A"),): 1}

    with pytest.raises(ValueError, match="Unsupported OpenFermion Pauli label"):
        hamiltonian_from_openfermion_qubit_operator(QubitOperatorStub())
