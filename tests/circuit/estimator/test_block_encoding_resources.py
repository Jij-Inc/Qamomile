"""Tests for block-encoding resource estimation primitives."""

from __future__ import annotations

import pytest
import sympy as sp

from qamomile.resource_estimation import (
    BlockEncodingResource,
    FTQCCostModel,
    ResourceQuantity,
    compare_resource_values,
    estimate_physical_resources,
    estimate_qubitized_qpe_resources_from_block_encoding,
)


def test_block_encoding_resource_exposes_walk_cost_drivers():
    """Block-encoding resources expose subroutine costs without IR lowering."""
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
    values = block.resource_values()

    assert block.logical_qubits == n + ancilla
    assert block.walk_cost_toffoli == 2 * prep + select + reflect
    assert values["system_qubits"] == n
    assert values["lambda_norm"] == alpha
    assert values["walk_cost_toffoli"] == 2 * prep + select + reflect
    assert block.to_dict()["name"] == "lcu_oracle"


def test_qubitized_qpe_from_block_encoding_returns_logical_estimate():
    """Block-encoding QPE returns logical resources before physical lifting."""
    block = BlockEncodingResource(
        system_qubits=6,
        normalization=100,
        prepare_cost_toffoli=20,
        select_cost_toffoli=50,
        reflection_cost_toffoli=5,
        ancilla_qubits=3,
    )

    logical = estimate_qubitized_qpe_resources_from_block_encoding(
        block,
        precision=2,
        qpe_register_qubits=4,
    )

    assert block.walk_cost_toffoli == 95
    assert logical.qubits == 13
    assert logical.gates.oracle_calls["qpe_iterations"] == 50
    assert logical.gates.total == 4750
    assert logical.gates.multi_qubit == 4750
    assert logical.gates.t_gates == 0


def test_block_encoding_logical_estimate_lifts_to_physical_resources():
    """Block-encoding logical estimates use the shared physical lift."""
    block = BlockEncodingResource(
        system_qubits=6,
        normalization=100,
        prepare_cost_toffoli=20,
        select_cost_toffoli=50,
        reflection_cost_toffoli=5,
        ancilla_qubits=3,
    )
    logical = estimate_qubitized_qpe_resources_from_block_encoding(
        block,
        precision=2,
        qpe_register_qubits=4,
    )
    physical = estimate_physical_resources(
        logical,
        FTQCCostModel(
            physical_qubits_per_logical=100,
            logical_cycle_time_seconds=sp.Float("1e-6"),
            factory_qubits=10,
            non_clifford_throughput_per_second=sp.Float("1e6"),
        ),
        non_clifford_count=logical.gates.multi_qubit,
    )

    assert physical.logical.qubits == 13
    assert physical.physical_qubits == 1310
    assert sp.Abs(physical.runtime_seconds - sp.Float("0.00475")) < sp.Float("1e-12")


def test_block_encoding_resource_values_compare_cost_drivers():
    """Block-encoding contracts compare through the shared quantity API."""
    baseline = BlockEncodingResource(
        system_qubits=8,
        normalization=100,
        prepare_cost_toffoli=20,
        select_cost_toffoli=60,
    )
    candidate = BlockEncodingResource(
        system_qubits=8,
        normalization=50,
        prepare_cost_toffoli=15,
        select_cost_toffoli=40,
    )

    rows = compare_resource_values(
        baseline,
        candidate,
        quantities=(
            ResourceQuantity.LAMBDA_NORM,
            ResourceQuantity.WALK_COST_TOFFOLI,
        ),
    )

    assert [row.quantity for row in rows] == [
        ResourceQuantity.LAMBDA_NORM,
        ResourceQuantity.WALK_COST_TOFFOLI,
    ]
    assert rows[0].ratio == sp.Rational(1, 2)
    assert rows[1].ratio == sp.Rational(7, 10)


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
        estimate_qubitized_qpe_resources_from_block_encoding(
            block,
            precision=precision,
            qpe_register_qubits=qpe_register_qubits,
        )
