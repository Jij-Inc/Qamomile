"""Tests for opaque-cost adaptation around generic IR serialization."""

from __future__ import annotations

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
