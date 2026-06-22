"""Tests for block-encoding based FTQC resource estimators."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.observable as qm_o
from qamomile.circuit.estimator.algorithmic import (
    BlockEncodingResource,
    ChemistryQPEMethod,
    ChemistryQPEModel,
    FTQCCostModel,
    FTQCReference,
    FTQCResourceQuantity,
    block_encoding_from_chemistry_model,
    estimate_qubitized_qpe_from_block_encoding,
    summarize_pauli_hamiltonian,
)


def test_block_encoding_resource_separates_prepare_select_and_reflection():
    """Block-encoding metadata exposes walk cost without circuit lowering."""
    n, alpha, prep, select, reflect, ancilla = sp.symbols(
        "n alpha C_P C_S C_R a",
        positive=True,
    )

    block = BlockEncodingResource(
        system_qubits=n,
        normalization=alpha,
        prepare_cost_toffoli=prep,
        select_cost_toffoli=select,
        reflection_cost_toffoli=reflect,
        ancilla_qubits=ancilla,
        name="lcu_oracle",
    )

    assert block.logical_qubits == n + ancilla
    assert block.walk_cost_toffoli == 2 * prep + select + reflect
    assert block.resource_values()[FTQCResourceQuantity.LAMBDA_NORM] == alpha
    assert (
        block.resource_values()[FTQCResourceQuantity.WALK_COST_TOFFOLI]
        == 2 * prep + select + reflect
    )
    assert block.to_dict()["name"] == "lcu_oracle"


def test_qubitized_qpe_from_block_encoding_tracks_walk_calls_and_costs():
    """Qubitized QPE composes alpha-over-epsilon calls with walk cost."""
    cost_model = FTQCCostModel(
        physical_qubits_per_logical=100,
        logical_cycle_time_seconds=sp.Float("1e-6"),
        factory_qubits=10,
        toffoli_throughput_per_second=sp.Float("1e6"),
    )
    block = BlockEncodingResource(
        system_qubits=6,
        normalization=100,
        prepare_cost_toffoli=20,
        select_cost_toffoli=50,
        reflection_cost_toffoli=5,
        ancilla_qubits=3,
        name="toy_lcu",
    )

    estimate = estimate_qubitized_qpe_from_block_encoding(
        block,
        precision=2,
        qpe_register_qubits=4,
        cost_model=cost_model,
    )

    assert block.walk_cost_toffoli == 95
    assert estimate.logical_qubits == 13
    assert estimate.qpe_iterations == 50
    assert estimate.target_precision == 2
    assert estimate.resource_values()[FTQCResourceQuantity.TARGET_PRECISION] == 2
    assert estimate.toffoli_gates == 4750
    assert estimate.physical_qubits == 1310
    assert sp.Abs(estimate.runtime_seconds - sp.Float("0.00475")) < sp.Float("1e-12")
    assert estimate.assumptions["block_encoding"] == "toy_lcu"
    assert any(reference.key == "arXiv:1610.06546" for reference in estimate.references)


def test_qubitized_qpe_from_block_encoding_preserves_custom_references():
    """Block-encoding and estimate references are deduplicated by key."""
    duplicate = FTQCReference(
        key="arXiv:1610.06546",
        title="Duplicate qubitization citation",
        url="https://example.invalid/duplicate",
    )
    custom = FTQCReference(
        key="internal:block-cost-v1",
        title="Internal block-encoding cost model",
        url="https://example.invalid/block-cost-v1",
    )
    block = BlockEncodingResource(
        system_qubits=2,
        normalization=4,
        prepare_cost_toffoli=1,
        select_cost_toffoli=3,
        references=(duplicate, custom),
    )

    estimate = estimate_qubitized_qpe_from_block_encoding(
        block,
        precision=1,
        references=(custom,),
    )
    keys = [reference.key for reference in estimate.references]

    assert keys.count("arXiv:1610.06546") == 1
    assert keys.count("internal:block-cost-v1") == 1
    assert estimate.to_dict()["references"][-1]["key"] == "internal:block-cost-v1"


def test_block_encoding_from_chemistry_model_preserves_qpe_inputs():
    """Chemistry models translate into block-encoding contracts for QPE."""
    summary = summarize_pauli_hamiltonian(
        2 * qm_o.Z(0) + 3 * qm_o.X(1),
        n_spin_orbitals=8,
        source="toy_chemistry",
    )
    model = ChemistryQPEModel(
        hamiltonian=summary.with_lambda_scale(sp.Rational(1, 2)),
        method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
        walk_cost_toffoli=100,
        second_factor_rank=4,
        description="compressed chemistry toy",
    )

    block = block_encoding_from_chemistry_model(model)
    estimate = estimate_qubitized_qpe_from_block_encoding(
        block,
        precision=1,
    )

    assert model.logical_qubit_count == 16
    assert block.name == "compressed chemistry toy"
    assert sp.Abs(block.resource_values()[FTQCResourceQuantity.LAMBDA_NORM] - 2.5) < (
        sp.Float("1e-12")
    )
    assert block._select_cost_toffoli == 100
    assert block._ancilla_qubits == 8
    assert sp.Abs(estimate.qpe_iterations - 2.5) < sp.Float("1e-12")
    assert sp.Abs(estimate.toffoli_gates - 250) < sp.Float("1e-12")
    assert any(reference.key == "arXiv:2403.03502" for reference in block.references)


def test_block_encoding_from_chemistry_model_accepts_subroutine_split():
    """Chemistry walk costs can be re-expressed as PREPARE/SELECT/reflection."""
    summary = summarize_pauli_hamiltonian(
        qm_o.Z(0) + qm_o.Z(1),
        n_spin_orbitals=4,
    )
    model = ChemistryQPEModel(
        hamiltonian=summary,
        method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
        walk_cost_toffoli=100,
    )

    block = block_encoding_from_chemistry_model(
        model,
        prepare_cost_toffoli=10,
        select_cost_toffoli=70,
        reflection_cost_toffoli=5,
        name="split_walk",
    )

    assert block.name == "split_walk"
    assert block.logical_qubits == 4
    assert block.walk_cost_toffoli == 95


def test_block_encoding_from_chemistry_model_rejects_non_model():
    """The chemistry bridge fails early when called with unrelated objects."""
    with pytest.raises(TypeError, match="ChemistryQPEModel"):
        block_encoding_from_chemistry_model(object())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"system_qubits": 0}, "system_qubits"),
        ({"normalization": 0}, "normalization"),
        ({"select_cost_toffoli": -1}, "select_cost_toffoli"),
        ({"prepare_cost_toffoli": -1}, "prepare_cost_toffoli"),
        ({"reflection_cost_toffoli": -1}, "reflection_cost_toffoli"),
        ({"ancilla_qubits": -1}, "ancilla_qubits"),
    ],
)
def test_block_encoding_resource_rejects_invalid_quantities(
    kwargs: dict[str, object],
    match: str,
):
    """Invalid block-encoding quantities fail before estimate construction."""
    valid = {
        "system_qubits": 2,
        "normalization": 4,
        "select_cost_toffoli": 3,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=match):
        BlockEncodingResource(**valid)


@pytest.mark.parametrize(
    ("precision", "qpe_register_qubits", "match"),
    [
        (0, 0, "precision"),
        (1, -1, "qpe_register_qubits"),
    ],
)
def test_qubitized_qpe_from_block_encoding_rejects_invalid_inputs(
    precision: int,
    qpe_register_qubits: int,
    match: str,
):
    """Invalid QPE precision or readout register sizes fail early."""
    block = BlockEncodingResource(
        system_qubits=2,
        normalization=4,
        select_cost_toffoli=3,
    )

    with pytest.raises(ValueError, match=match):
        estimate_qubitized_qpe_from_block_encoding(
            block,
            precision=precision,
            qpe_register_qubits=qpe_register_qubits,
        )
