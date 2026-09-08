"""Tests for opaque-cost adaptation around generic IR serialization."""

from __future__ import annotations

from unittest.mock import patch

import qamomile.circuit.ir.serialize.decode as decode_module
from qamomile.circuit.ir.operation.callable import CallableDef, CallableRef
from qamomile.circuit.ir.serialize.decode import (
    _decode_callable_def,
    _DecodeContext,
)
from qamomile.circuit.ir.serialize.encode import (
    _encode_callable_def,
    _EncodeContext,
)


def test_ir_callable_cost_round_trips_as_generic_payload() -> None:
    """IR serialization preserves opaque payloads without estimator knowledge."""
    definition = CallableDef(
        ref=CallableRef("tests.serialization", "generic_cost"),
        opaque_cost={
            "model": "generic",
            "weights": [1, 2, 3],
        },
    )

    encoded = _encode_callable_def(definition, _EncodeContext())
    restored = _decode_callable_def(encoded, _DecodeContext([]))

    assert restored.opaque_cost == definition.opaque_cost


def test_callable_table_decodes_definition_attrs_once() -> None:
    """Prelinking reuses decoded callable attributes during phase two."""
    definition = CallableDef(
        ref=CallableRef("tests.serialization", "attrs_once"),
        attrs={"resource_contract": {"quantum_operand_widths": [2]}},
    )
    encoded = _encode_callable_def(definition, _EncodeContext())
    context = _DecodeContext(
        [],
        [{"id": "attrs_once", "definition": encoded}],
    )

    with patch.object(
        decode_module,
        "_decode_callable_definition_attrs",
        wraps=decode_module._decode_callable_definition_attrs,
    ) as decode_attrs:
        context.populate_definitions()

    assert decode_attrs.call_count == 1
    assert context.definition("attrs_once").attrs == definition.attrs
