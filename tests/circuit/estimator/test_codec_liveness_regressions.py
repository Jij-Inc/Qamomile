"""Regression tests for resource codec and boundary liveness fixes."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator._liveness import _liveness_width
from qamomile.circuit.estimator._resolver import ExprResolver
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.ir.types import QubitType
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    Value,
    ValueMetadata,
)


@qm.qkernel
def _measurement_source(target: qm.Qubit) -> qm.Bit:
    """Measure one target behind a callable boundary."""
    return qm.measure(target)


@qm.qkernel
def _branch_on_measurement_source() -> tuple[qm.Qubit, qm.Qubit]:
    """Use a callable result for runtime structure and disjoint work."""
    observed = _measurement_source(qm.qubit("observed"))
    branch_target = qm.qubit("branch_target")
    independent = qm.qubit("independent")
    if observed:
        branch_target = qm.h(branch_target)
    independent = qm.x(independent)
    return branch_target, independent


def test_invoke_fallback_liveness_caps_synthetic_carrier_owners() -> None:
    """A multi-owner runtime carrier cannot index a missing capacity."""
    left = Value(type=QubitType(), name="left")
    right = Value(type=QubitType(), name="right")
    carrier = ArrayValue(
        type=QubitType(),
        name="carrier",
        metadata=ValueMetadata(
            array_runtime=ArrayRuntimeMetadata(
                element_uuids=(left.uuid, right.uuid),
                element_logical_ids=(left.logical_id, right.logical_id),
            )
        ),
    )
    reference = CallableRef("test", "synthetic_carrier")
    operation = InvokeOperation(
        operands=[carrier],
        results=[carrier.next_version()],
        target=reference,
        definition=CallableDef(ref=reference),
    )

    summary = _liveness_width(
        [(operation, qm.ResourceEstimate())],
        {
            left.logical_id: sp.Integer(1),
            right.logical_id: sp.Integer(1),
        },
        ExprResolver(),
    )

    assert summary.final_live_by_owner == {
        left.logical_id: sp.Integer(1),
        right.logical_id: sp.Integer(1),
    }
    assert summary.width == qm.WidthResources.zero()


def test_resolver_normalizes_supported_real_numeric_scalars() -> None:
    """Supported real scalars stay numeric and preserve exact rationals."""
    resolver = ExprResolver()

    integer = resolver.resolve(np.int64(5))
    numpy_float = resolver.resolve(np.float32(1.5))
    decimal = resolver.resolve(Decimal("1.25"))
    rational = resolver.resolve(Fraction(1, 3))

    assert integer == sp.Integer(5)
    assert resolver.resolve_concrete(np.int64(5)) == 5
    assert numpy_float == sp.Float(1.5)
    assert decimal == sp.Float("1.25")
    assert rational == sp.Rational(1, 3)
    assert all(
        not expression.free_symbols
        for expression in (integer, numpy_float, decimal, rational)
    )


def test_measurement_opaque_cost_taints_only_its_classical_results() -> None:
    """Opaque observation results stay internal without fencing disjoint work."""
    operation = next(
        operation
        for operation in _branch_on_measurement_source.block.operations
        if isinstance(operation, InvokeOperation)
    )
    assert operation.definition is not None
    original_body = operation.definition.body
    original_implementations = operation.definition.implementations
    original_cost = operation.definition.opaque_cost
    operation.definition.body = None
    operation.definition.implementations = []
    operation.definition.opaque_cost = qm.ResourceEstimate(
        measurements=qm.MeasurementResources(total=1),
        depth=qm.DepthResources(depth=1, measurement_depth=1),
    )
    try:
        estimate = _branch_on_measurement_source.estimate_resources()
    finally:
        operation.definition.body = original_body
        operation.definition.implementations = original_implementations
        operation.definition.opaque_cost = original_cost

    assert estimate.parameters == {}
    assert estimate.measurements.total == 1
    assert estimate.gates.total == 2
    assert estimate.depth.depth == 2
