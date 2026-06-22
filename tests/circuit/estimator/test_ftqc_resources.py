"""Tests for canonical FTQC resource quantity metadata."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCCostModel,
    FTQCResourceCategory,
    FTQCResourceQuantity,
    describe_ftqc_resource_quantity,
    estimate_qubitized_chemistry_qpe_from_model,
    iter_ftqc_resource_quantity_specs,
    summarize_pauli_hamiltonian,
)


def test_ftqc_quantity_specs_cover_core_resource_layers():
    """The FTQC quantity catalog covers problem, logical, and physical layers."""
    specs = iter_ftqc_resource_quantity_specs()
    quantities = {spec.quantity for spec in specs}
    categories = {spec.category for spec in specs}

    assert FTQCResourceQuantity.LAMBDA_NORM in quantities
    assert FTQCResourceQuantity.TOFFOLI_GATES in quantities
    assert FTQCResourceQuantity.PHYSICAL_QUBITS in quantities
    assert {
        FTQCResourceCategory.PROBLEM,
        FTQCResourceCategory.ALGORITHM,
        FTQCResourceCategory.LOGICAL,
        FTQCResourceCategory.PHYSICAL,
        FTQCResourceCategory.ARCHITECTURE,
    }.issubset(categories)
    assert len(quantities) == len(specs)


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
    assert (
        cost.resource_values()[FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL] == 100
    )
    assert (
        sp.simplify(estimate.resource_values()[FTQCResourceQuantity.TOFFOLI_GATES] - 55)
        == 0
    )
    assert estimate.to_quantity_table()[0]["quantity"] == "logical_qubits"
