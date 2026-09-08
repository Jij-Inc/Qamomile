"""Tests for shared callable resource-contract helpers."""

from __future__ import annotations

import pytest

from qamomile.circuit.ir._resource_contract import (
    ProductFormulaContract,
    QuantumOperandWidth,
    merge_product_formula_contract,
    merge_quantum_operand_widths,
    product_formula_contract,
)


def test_product_formula_contract_accepts_generic_family_and_roles() -> None:
    """IR validation accepts family-specific roles without interpreting them."""
    attrs = {
        "resource_contract": {
            "product_formula": {
                "kind": "custom_symmetric_formula",
                "generator_operand": 0,
                "segments_operand": 2,
            }
        }
    }

    contract = product_formula_contract(
        attrs,
        source="custom_formula",
        operand_count=3,
    )

    assert contract == ProductFormulaContract(
        kind="custom_symmetric_formula",
        operands={
            "generator_operand": 0,
            "segments_operand": 2,
        },
    )


def test_merge_product_formula_contract_preserves_flat_wire_shape() -> None:
    """Generic role storage retains the established flat serialized envelope."""
    merged = merge_product_formula_contract(
        {"kind": "qkernel"},
        ProductFormulaContract(
            kind="suzuki_trotter",
            operands={
                "hamiltonian_operand": 1,
                "order_operand": 2,
                "time_operand": 3,
                "steps_operand": 4,
            },
        ),
        source="_trotter_evolve",
        operand_count=5,
    )

    assert merged == {
        "kind": "qkernel",
        "resource_contract": {
            "product_formula": {
                "kind": "suzuki_trotter",
                "hamiltonian_operand": 1,
                "order_operand": 2,
                "time_operand": 3,
                "steps_operand": 4,
            }
        },
    }


def test_product_formula_contract_allows_family_specific_role_aliasing() -> None:
    """Generic IR does not impose one family's operand-distinctness rule."""
    attrs = {
        "resource_contract": {
            "product_formula": {
                "kind": "custom_formula",
                "left_operand": 1,
                "right_operand": 1,
            }
        }
    }

    contract = product_formula_contract(
        attrs,
        source="custom_formula",
        operand_count=2,
    )

    assert contract == ProductFormulaContract(
        kind="custom_formula",
        operands={"left_operand": 1, "right_operand": 1},
    )


def test_merge_quantum_operand_widths_completes_compatible_partial_contract() -> None:
    """Compatible partial widths merge canonically without dropping other attrs."""
    attrs = {
        "kind": "qkernel",
        "resource_contract": {
            "custom_contract": {"preserve": True},
            "quantum_operand_widths": [
                {"index": 1, "name": "system", "width": 3},
            ],
        },
    }

    merged = merge_quantum_operand_widths(
        attrs,
        (
            QuantumOperandWidth(index=1, name="system", width=3),
            QuantumOperandWidth(index=0, name="signal", width=2),
        ),
        source="unitary",
        operand_count=2,
    )

    assert merged == {
        "kind": "qkernel",
        "resource_contract": {
            "custom_contract": {"preserve": True},
            "quantum_operand_widths": [
                {"index": 0, "name": "signal", "width": 2},
                {"index": 1, "name": "system", "width": 3},
            ],
        },
    }
    assert attrs["resource_contract"]["quantum_operand_widths"] == [
        {"index": 1, "name": "system", "width": 3},
    ]


def test_merge_quantum_operand_widths_rejects_conflicting_partial_contract() -> None:
    """A second declaration cannot change a width at an existing operand index."""
    attrs = {
        "resource_contract": {
            "quantum_operand_widths": [
                {"index": 1, "name": "system", "width": 4},
            ],
        },
    }

    with pytest.raises(ValueError, match="quantum operand 1.*conflicts"):
        merge_quantum_operand_widths(
            attrs,
            (QuantumOperandWidth(index=1, name="system", width=3),),
            source="unitary",
            operand_count=2,
        )


@pytest.mark.parametrize("declaration_source", ["existing", "requested"])
def test_merge_quantum_operand_widths_rejects_out_of_range_entry(
    declaration_source: str,
) -> None:
    """Operand-count validation covers both old and newly requested entries."""
    out_of_range = {"index": 2, "name": "extra", "width": 1}
    attrs = (
        {
            "resource_contract": {
                "quantum_operand_widths": [out_of_range],
            },
        }
        if declaration_source == "existing"
        else {}
    )
    widths = (
        ()
        if declaration_source == "existing"
        else (QuantumOperandWidth(index=2, name="extra", width=1),)
    )

    with pytest.raises(ValueError, match="quantum operand 2.*only 2"):
        merge_quantum_operand_widths(
            attrs,
            widths,
            source="unitary",
            operand_count=2,
        )
