"""Tests for fixed opaque-cost protobuf serialization."""

from __future__ import annotations

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator.resource_estimator import (
    ResourceAssumption,
    ResourceTraceNode,
)
from qamomile.circuit.ir.operation.callable import (
    CallableBodyRef,
    CallableImplementation,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.circuit.serialization.resource_estimate import (
    decode_fixed_resource_estimate,
    encode_fixed_resource_estimate,
)

_FIXED_COST = qmc.ResourceEstimate(
    width=qmc.WidthResources(
        input_qubits=1,
        allocated_qubits=2,
        clean_ancilla_qubits=1,
        peak_qubits=3,
    ),
    gates=qmc.GateResources(
        total=7,
        single_qubit=3,
        two_qubit=4,
        clifford=5,
        rotation=2,
    ),
    depth=qmc.DepthResources(
        depth=6,
        clifford_depth=4,
        rotation_depth=2,
    ),
    calls=qmc.CallResources(
        calls_by_name={"fixed_cost_oracle": 1},
        queries_by_name={"fixed_cost_oracle": 1},
    ),
    assumptions=(
        ResourceAssumption(
            message="fixed implementation model",
            source="test",
        ),
    ),
    trace=ResourceTraceNode(
        name="fixed_cost",
        source_kind="opaque_cost",
        summary="seven gates",
    ),
    quality=qmc.EstimateQuality.UPPER_BOUND,
)
_FIXED_ORACLE = qmc.opaque(
    "fixed_cost_oracle",
    num_qubits=1,
    cost=_FIXED_COST,
)


def _context_cost(_: qmc.OpaqueCallContext) -> qmc.ResourceEstimate:
    """Return a context-dependent cost that cannot cross serialization."""
    return qmc.ResourceEstimate(gates=qmc.GateResources(total=1))


_CALLBACK_ORACLE = qmc.opaque(
    "callback_cost_oracle",
    num_qubits=1,
    cost=_context_cost,
)

_HIGH_PRECISION_VALUE = sp.Float(
    "1.234567890123456789012345678901234567890123456789",
    180,
)
_HIGH_PRECISION_ORACLE = qmc.opaque(
    "high_precision_cost_oracle",
    num_qubits=1,
    cost=qmc.ResourceEstimate(
        gates=qmc.GateResources(total=_HIGH_PRECISION_VALUE),
    ),
)


@qmc.qkernel
def _use_fixed_cost_oracle() -> qmc.Qubit:
    """Invoke an opaque oracle carrying a fixed resource cost."""
    (qubit,) = _FIXED_ORACLE(qmc.qubit("qubit"))
    return qubit


@qmc.qkernel
def _use_callback_cost_oracle() -> qmc.Qubit:
    """Invoke an opaque oracle carrying a Python cost callback."""
    (qubit,) = _CALLBACK_ORACLE(qmc.qubit("qubit"))
    return qubit


@qmc.qkernel
def _use_high_precision_cost_oracle() -> qmc.Qubit:
    """Invoke an opaque oracle carrying an arbitrary-precision numeric cost."""
    (qubit,) = _HIGH_PRECISION_ORACLE(qmc.qubit("qubit"))
    return qubit


def test_fixed_opaque_cost_round_trips_and_remains_estimable() -> None:
    """Fixed ResourceEstimate costs survive protobuf round trips."""
    payload = serialize(_use_fixed_cost_oracle)
    message = pb.QKernel.FromString(payload)
    [oracle_entry] = [
        entry
        for entry in message.callable_table
        if entry.definition.ref.name == "fixed_cost_oracle"
    ]
    assert oracle_entry.definition.HasField("opaque_cost")

    restored = deserialize(payload)
    assert serialize(restored) == payload
    [invoke] = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.definition is not None
    restored_cost = invoke.definition.opaque_cost
    assert isinstance(restored_cost, qmc.ResourceEstimate)
    assert restored_cost.width == _FIXED_COST.width
    assert restored_cost.gates == _FIXED_COST.gates
    assert restored_cost.depth == _FIXED_COST.depth
    assert restored_cost.calls == _FIXED_COST.calls
    assert restored_cost.assumptions == _FIXED_COST.assumptions
    assert restored_cost.trace == _FIXED_COST.trace
    assert restored_cost.quality is qmc.EstimateQuality.UPPER_BOUND

    estimate = qmc.estimate_resources(restored)
    assert estimate.gates.total == 7
    assert estimate.calls.calls_by_name == {"fixed_cost_oracle": 1}
    assert estimate.calls.queries_by_name == {"fixed_cost_oracle": 1}


def test_context_dependent_opaque_cost_callback_fails_serialization() -> None:
    """Python opaque-cost callbacks fail instead of disappearing silently."""
    with pytest.raises(
        TypeError,
        match="context-dependent opaque cost callback.*callback_cost_oracle",
    ):
        serialize(_use_callback_cost_oracle)


def test_body_backed_callable_cannot_declare_opaque_cost() -> None:
    """A definition with a semantic body cannot also carry an opaque cost."""

    @qmc.qkernel
    def helper(qubit: qmc.Qubit) -> qmc.Qubit:
        """Apply one body-backed gate."""
        return qmc.x(qubit)

    @qmc.qkernel
    def caller() -> qmc.Qubit:
        """Invoke the body-backed helper."""
        return helper(qmc.qubit("qubit"))

    [invoke] = [
        operation
        for operation in caller.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.definition is not None
    assert invoke.definition.body is not None
    invoke.definition.opaque_cost = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1),
    )

    with pytest.raises(ValueError, match="body-backed.*opaque_cost"):
        serialize(caller)


@pytest.mark.parametrize(
    "placement",
    [
        "definition_body_ref",
        "implementation_body",
        "implementation_body_ref",
    ],
)
def test_alternative_body_cannot_declare_opaque_cost(placement: str) -> None:
    """A deferred or alternative body is also exclusive with opaque cost.

    Args:
        placement (str): Callable body slot to populate.
    """

    @qmc.qkernel
    def helper(qubit: qmc.Qubit) -> qmc.Qubit:
        """Apply one body-backed gate."""
        return qmc.x(qubit)

    @qmc.qkernel
    def caller() -> qmc.Qubit:
        """Invoke the body-backed helper."""
        return helper(qmc.qubit("qubit"))

    [invoke] = [
        operation
        for operation in caller.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.definition is not None
    body = invoke.definition.body
    assert body is not None
    invoke.definition.body = None
    body_ref = CallableBodyRef(
        ref=CallableRef(
            namespace="tests.serialization",
            name=f"{placement}_body",
        )
    )
    if placement == "definition_body_ref":
        invoke.definition.body_ref = body_ref
    elif placement == "implementation_body":
        invoke.definition.implementations = [
            CallableImplementation(body=body),
        ]
    else:
        invoke.definition.implementations = [
            CallableImplementation(body_ref=body_ref),
        ]
    invoke.definition.opaque_cost = qmc.ResourceEstimate(
        gates=qmc.GateResources(total=1),
    )

    with pytest.raises(ValueError, match="body-backed.*opaque_cost"):
        serialize(caller)


def test_high_precision_sympy_float_round_trips_losslessly() -> None:
    """Arbitrary-precision SymPy floats retain their exact binary value."""
    payload = serialize(_use_high_precision_cost_oracle)
    restored = deserialize(payload)
    [invoke] = [
        operation
        for operation in restored.block.operations
        if isinstance(operation, InvokeOperation)
    ]
    assert invoke.definition is not None
    restored_cost = invoke.definition.opaque_cost
    assert isinstance(restored_cost, qmc.ResourceEstimate)
    restored_value = restored_cost.gates.total
    assert isinstance(restored_value, sp.Float)
    assert restored_value._mpf_ == _HIGH_PRECISION_VALUE._mpf_
    assert restored_value._prec == _HIGH_PRECISION_VALUE._prec
    assert serialize(restored) == payload


def test_high_precision_sympy_float_zero_round_trips_losslessly() -> None:
    """A zero SymPy Float remains a Float with its declared precision."""
    zero = sp.Float(0, precision=277)
    payload = encode_fixed_resource_estimate(
        qmc.ResourceEstimate(gates=qmc.GateResources(total=zero)),
        "high_precision_zero_oracle",
    )
    restored = decode_fixed_resource_estimate(
        payload,
        "high_precision_zero_oracle",
    )

    assert restored is not None
    restored_zero = restored.gates.total
    assert isinstance(restored_zero, sp.Float)
    assert restored_zero._mpf_ == zero._mpf_
    assert restored_zero._prec == zero._prec


@pytest.mark.parametrize(
    "cost",
    [
        qmc.ResourceEstimate(
            gates=qmc.GateResources(total=sp.Symbol("free_count") + 1),
        ),
        qmc.ResourceEstimate(
            parameters={"declared_count": sp.Symbol("declared_count")},
        ),
    ],
)
def test_symbolic_fixed_costs_are_rejected(cost: qmc.ResourceEstimate) -> None:
    """Free symbols and declared estimate parameters cannot be serialized."""
    with pytest.raises(TypeError, match="symbolic|fixed"):
        encode_fixed_resource_estimate(cost, "symbolic_cost_oracle")


def test_malformed_high_precision_float_payload_is_rejected() -> None:
    """Noncanonical arbitrary-precision float records fail closed."""
    payload = encode_fixed_resource_estimate(
        qmc.ResourceEstimate(
            gates=qmc.GateResources(total=_HIGH_PRECISION_VALUE),
        ),
        "malformed_float_oracle",
    )
    assert payload is not None
    encoded_float = payload["gates"]["total"]
    encoded_float["mantissa"] *= 2
    encoded_float["bitcount"] += 1

    with pytest.raises(ValueError, match="significand is not canonical"):
        decode_fixed_resource_estimate(payload, "malformed_float_oracle")


def test_malformed_resource_record_is_rejected() -> None:
    """Missing fixed-cost record fields fail closed during decoding."""
    payload = encode_fixed_resource_estimate(
        qmc.ResourceEstimate(),
        "malformed_record_oracle",
    )
    assert payload is not None
    del payload["depth"]["measurement_depth"]

    with pytest.raises(ValueError, match="requires exactly fields"):
        decode_fixed_resource_estimate(payload, "malformed_record_oracle")


def test_named_resource_entries_must_be_canonical() -> None:
    """Named resource maps must use unique, lexicographically sorted entries."""
    payload = encode_fixed_resource_estimate(
        qmc.ResourceEstimate(
            calls=qmc.CallResources(
                calls_by_name={"alpha": 1, "zeta": 2},
            ),
        ),
        "unordered_calls_oracle",
    )
    assert payload is not None
    payload["calls"]["calls_by_name"].reverse()

    with pytest.raises(ValueError, match="unique and sorted"):
        decode_fixed_resource_estimate(payload, "unordered_calls_oracle")


def test_equivalent_named_resource_maps_encode_canonically() -> None:
    """Equivalent cost maps encode identically regardless of insertion order."""
    left = qmc.ResourceEstimate(
        calls=qmc.CallResources(
            calls_by_name={"zeta": 2, "alpha": 1},
            queries_by_name={"target": 3, "oracle": 4},
        ),
    )
    right = qmc.ResourceEstimate(
        calls=qmc.CallResources(
            calls_by_name={"alpha": 1, "zeta": 2},
            queries_by_name={"oracle": 4, "target": 3},
        ),
    )

    assert encode_fixed_resource_estimate(
        left,
        "canonical_cost_oracle",
    ) == encode_fixed_resource_estimate(
        right,
        "canonical_cost_oracle",
    )
