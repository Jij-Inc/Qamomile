"""Encode and decode nested records in resource-estimate wire payloads."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

import sympy as sp

from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    EstimateDerivation,
    EstimateQuality,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintRange,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
    _GuardedApproximation,
    _GuardedAssumption,
    _GuardedDerivation,
    _GuardedQuality,
)
from qamomile.circuit.estimator._wire_expression import (
    _boolean_expression_from_wire,
    _resource_expression_from_wire,
    _WireExpressionDecoder,
    _WireExpressionEncoder,
)

_MetricT = TypeVar(
    "_MetricT",
    WidthResources,
    GateResources,
    DepthResources,
    MeasurementResources,
    ResetResources,
)
_EnumT = TypeVar("_EnumT", bound=enum.Enum)
_MAX_TRACE_NODES = 100_000


def _metric_to_wire(
    metric: _MetricT,
    encoder: _WireExpressionEncoder,
) -> dict[str, str]:
    """Encode every dataclass field of one resource metric.

    Args:
        metric (_MetricT): Metric record to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, str]: Field names mapped to safe SymPy representations.
    """
    return {
        field.name: encoder.encode(getattr(metric, field.name))
        for field in dataclasses.fields(metric)
    }


def _metric_from_wire(
    payload: Any,
    metric_type: type[_MetricT],
    label: str,
    decoder: _WireExpressionDecoder,
) -> _MetricT:
    """Decode every field of one resource metric.

    Args:
        payload (Any): Serialized metric mapping.
        metric_type (type[_MetricT]): Dataclass type to construct.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _MetricT: Reconstructed metric record.

    Raises:
        ValueError: If fields are missing, unknown, or not numeric
            expressions.
    """
    record = _mapping(payload, label)
    field_names = {field.name for field in dataclasses.fields(metric_type)}
    if set(record) != field_names:
        raise ValueError(
            f"{label} fields must be {sorted(field_names)!r}, got {sorted(record)!r}"
        )
    values = {
        name: _resource_expression_from_wire(value, f"{label}.{name}", decoder)
        for name, value in record.items()
    }
    return metric_type(**values)


def _expression_map_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> dict[str, sp.Expr]:
    """Decode a name-to-resource-expression mapping.

    Args:
        payload (Any): Serialized mapping.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        dict[str, sp.Expr]: Decoded expression map in payload order.

    Raises:
        ValueError: If a key is not a string or a value is not a resource
            expression.
    """
    record = _mapping(payload, label)
    decoded: dict[str, sp.Expr] = {}
    for name, value in record.items():
        if not isinstance(name, str):
            raise ValueError(f"{label} keys must be strings")
        decoded[name] = _resource_expression_from_wire(
            value,
            f"{label}.{name}",
            decoder,
        )
    return decoded


def _assumption_to_wire(assumption: ResourceAssumption) -> dict[str, Any]:
    """Encode one resource assumption.

    Args:
        assumption (ResourceAssumption): Assumption to encode.

    Returns:
        dict[str, Any]: Message and optional source.
    """
    return {
        "message": assumption.message,
        "source": assumption.source,
    }


def _assumption_from_wire(payload: Any) -> ResourceAssumption:
    """Decode one resource assumption.

    Args:
        payload (Any): Serialized assumption mapping.

    Returns:
        ResourceAssumption: Reconstructed assumption.

    Raises:
        ValueError: If the message or source has the wrong type.
    """
    record = _mapping(payload, "resource assumption")
    message = record.get("message")
    source = record.get("source")
    if not isinstance(message, str):
        raise ValueError("resource assumption message must be a string")
    if source is not None and not isinstance(source, str):
        raise ValueError("resource assumption source must be a string or None")
    return ResourceAssumption(message=message, source=source)


def _trace_to_wire(
    trace: ResourceTraceNode | None,
    encoder: _WireExpressionEncoder,
) -> dict[str, Any] | None:
    """Encode one explanation trace as a flat indexed tree.

    Args:
        trace (ResourceTraceNode | None): Trace node to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, Any] | None: Serialized trace, or ``None``.

    Raises:
        ValueError: If the trace exceeds the supported node limit.
    """
    if trace is None:
        return None
    pending = [trace]
    nodes: list[dict[str, Any]] = []
    index = 0
    while index < len(pending):
        if len(pending) > _MAX_TRACE_NODES:
            raise ValueError(f"resource trace exceeds {_MAX_TRACE_NODES} nodes")
        node = pending[index]
        child_start = len(pending)
        pending.extend(node.children)
        nodes.append(
            {
                "name": node.name,
                "source_kind": node.source_kind,
                "strategy": node.strategy,
                "summary": node.summary,
                "assumptions": [
                    _assumption_to_wire(assumption) for assumption in node.assumptions
                ],
                "children": list(range(child_start, child_start + len(node.children))),
                "active_when": encoder.encode(node.active_when),
            }
        )
        index += 1
    return {"root": 0, "nodes": nodes}


def _trace_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> ResourceTraceNode | None:
    """Decode one flat indexed explanation trace.

    Args:
        payload (Any): Serialized trace mapping or ``None``.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        ResourceTraceNode | None: Reconstructed trace.

    Raises:
        ValueError: If trace fields or activation guards are malformed.
    """
    if payload is None:
        return None
    trace = _mapping(payload, "resource trace")
    if trace.get("root") != 0:
        raise ValueError("resource trace root must be node 0")
    raw_nodes = _sequence(trace.get("nodes"), "resource trace nodes")
    if not raw_nodes or len(raw_nodes) > _MAX_TRACE_NODES:
        raise ValueError(
            f"resource trace must contain between 1 and {_MAX_TRACE_NODES} nodes"
        )

    node_records = [
        _mapping(item, f"resource trace node {index}")
        for index, item in enumerate(raw_nodes)
    ]
    child_indices: list[tuple[int, ...]] = []
    referenced: list[int] = []
    for index, record in enumerate(node_records):
        children: list[int] = []
        for child in _sequence(
            record.get("children"),
            f"resource trace node {index} children",
        ):
            if (
                isinstance(child, bool)
                or not isinstance(child, int)
                or child <= index
                or child >= len(node_records)
            ):
                raise ValueError(
                    "resource trace child indices must refer to later nodes"
                )
            children.append(child)
            referenced.append(child)
        child_indices.append(tuple(children))
    if sorted(referenced) != list(range(1, len(node_records))):
        raise ValueError(
            "resource trace nodes must form one rooted tree without sharing"
        )

    decoded: list[ResourceTraceNode | None] = [None] * len(node_records)
    for index in reversed(range(len(node_records))):
        record = node_records[index]
        name = record.get("name")
        source_kind = record.get("source_kind")
        strategy = record.get("strategy")
        summary = record.get("summary")
        if not isinstance(name, str) or not isinstance(source_kind, str):
            raise ValueError("resource trace name and source_kind must be strings")
        if strategy is not None and not isinstance(strategy, str):
            raise ValueError("resource trace strategy must be a string or None")
        if not isinstance(summary, str):
            raise ValueError("resource trace summary must be a string")
        decoded_children = tuple(
            cast(ResourceTraceNode, decoded[child]) for child in child_indices[index]
        )
        decoded[index] = ResourceTraceNode(
            name=name,
            source_kind=source_kind,
            strategy=strategy,
            summary=summary,
            assumptions=tuple(
                _assumption_from_wire(item)
                for item in _sequence(
                    record.get("assumptions"),
                    f"resource trace node {index} assumptions",
                )
            ),
            children=decoded_children,
            active_when=_boolean_expression_from_wire(
                record.get("active_when"),
                f"resource trace node {index} active_when",
                decoder,
            ),
        )
    return cast(ResourceTraceNode, decoded[0])


def _requirement_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _ResourceConstraint:
    """Decode one structural resource requirement.

    Args:
        payload (Any): Serialized requirement mapping.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _ResourceConstraint: Reconstructed requirement.

    Raises:
        ValueError: If any requirement or quantified-range field is malformed.
    """
    record = _mapping(payload, "resource requirement")
    minimum = record.get("minimum")
    if minimum is not None and (
        isinstance(minimum, bool) or not isinstance(minimum, int)
    ):
        raise ValueError("resource requirement minimum must be an int or None")
    label = record.get("label")
    unit = record.get("unit")
    if not isinstance(label, str) or not isinstance(unit, str):
        raise ValueError("resource requirement label and unit must be strings")
    flags: dict[str, bool] = {}
    for name in ("integer", "minimum_inclusive", "finite"):
        value = record.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"resource requirement {name} must be a bool")
        flags[name] = value
    ranges: list[_ConstraintRange] = []
    for raw_range in _sequence(
        record.get("ranges"),
        "resource requirement ranges",
    ):
        range_record = _mapping(raw_range, "resource requirement range")
        symbol = decoder.decode(
            range_record.get("symbol"),
            "resource requirement range symbol",
        )
        if not isinstance(symbol, sp.Symbol):
            raise ValueError("resource requirement range symbol must be a Symbol")
        ranges.append(
            _ConstraintRange(
                symbol=symbol,
                start=_resource_expression_from_wire(
                    range_record.get("start"),
                    "resource requirement range start",
                    decoder,
                ),
                step=_resource_expression_from_wire(
                    range_record.get("step"),
                    "resource requirement range step",
                    decoder,
                ),
                iterations=_resource_expression_from_wire(
                    range_record.get("iterations"),
                    "resource requirement range iterations",
                    decoder,
                ),
            )
        )
    raw_expected = record.get("expected")
    requirement = _ResourceConstraint(
        expression=_resource_expression_from_wire(
            record.get("expression"),
            "resource requirement expression",
            decoder,
        ),
        minimum=minimum,
        label=label,
        unit=unit,
        integer=flags["integer"],
        minimum_inclusive=flags["minimum_inclusive"],
        finite=flags["finite"],
        expected=(
            _resource_expression_from_wire(
                raw_expected,
                "resource requirement expected",
                decoder,
            )
            if raw_expected is not None
            else None
        ),
        ranges=tuple(ranges),
        active_when=(
            _boolean_expression_from_wire(
                record.get("active_when"),
                "resource requirement activation",
                decoder,
            )
            if record.get("active_when") is not None
            else sp.true
        ),
    )
    requirement.validate()
    return requirement


def _guarded_assumption_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _GuardedAssumption:
    """Decode one guarded assumption fact.

    Args:
        payload (Any): Serialized provenance mapping.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _GuardedAssumption: Reconstructed guarded fact.

    Raises:
        ValueError: If the guard or assumption is malformed.
    """
    record = _mapping(payload, "guarded resource assumption")
    return _GuardedAssumption(
        active_when=_boolean_expression_from_wire(
            record.get("active_when"),
            "guarded resource assumption active_when",
            decoder,
        ),
        assumption=_assumption_from_wire(record.get("assumption")),
    )


def _guarded_derivation_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _GuardedDerivation:
    """Decode one guarded estimate-derivation fact.

    Args:
        payload (Any): Serialized provenance mapping.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _GuardedDerivation: Reconstructed guarded fact.

    Raises:
        ValueError: If the guard or derivation is malformed.
    """
    record = _mapping(payload, "guarded resource derivation")
    return _GuardedDerivation(
        active_when=_boolean_expression_from_wire(
            record.get("active_when"),
            "guarded resource derivation active_when",
            decoder,
        ),
        derivation=_enum_from_wire(
            EstimateDerivation,
            record.get("derivation"),
            "guarded resource derivation",
        ),
    )


def _guarded_quality_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _GuardedQuality:
    """Decode one guarded estimate-quality fact.

    Args:
        payload (Any): Serialized provenance mapping.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _GuardedQuality: Reconstructed guarded fact.

    Raises:
        ValueError: If the guard or quality is malformed.
    """
    record = _mapping(payload, "guarded resource quality")
    return _GuardedQuality(
        active_when=_boolean_expression_from_wire(
            record.get("active_when"),
            "guarded resource quality active_when",
            decoder,
        ),
        quality=_enum_from_wire(
            EstimateQuality,
            record.get("quality"),
            "guarded resource quality",
        ),
    )


def _guarded_approximation_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _GuardedApproximation:
    """Decode one guarded approximation fact.

    Args:
        payload (Any): Serialized provenance mapping.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        _GuardedApproximation: Reconstructed guarded fact.

    Raises:
        ValueError: If the guard or approximation is malformed.
    """
    record = _mapping(payload, "guarded resource approximation")
    return _GuardedApproximation(
        active_when=_boolean_expression_from_wire(
            record.get("active_when"),
            "guarded resource approximation active_when",
            decoder,
        ),
        approximation=_enum_from_wire(
            ApproximationStatus,
            record.get("approximation"),
            "guarded resource approximation",
        ),
    )


def _mapping(payload: Any, label: str) -> Mapping[Any, Any]:
    """Require one mapping payload.

    Args:
        payload (Any): Candidate value.
        label (str): Diagnostic label.

    Returns:
        Mapping[Any, Any]: Validated mapping.

    Raises:
        ValueError: If ``payload`` is not a mapping.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _sequence(payload: Any, label: str) -> Sequence[Any]:
    """Require one non-string sequence payload.

    Args:
        payload (Any): Candidate value.
        label (str): Diagnostic label.

    Returns:
        Sequence[Any]: Validated sequence.

    Raises:
        ValueError: If ``payload`` is not a non-string sequence.
    """
    if isinstance(payload, (str, bytes, bytearray)) or not isinstance(
        payload,
        Sequence,
    ):
        raise ValueError(f"{label} must be a sequence")
    return payload


def _enum_from_wire(
    enum_type: type[_EnumT],
    payload: Any,
    label: str,
) -> _EnumT:
    """Decode one string-valued enum member.

    Args:
        enum_type (type[_EnumT]): Enum class to construct.
        payload (Any): Serialized enum value.
        label (str): Diagnostic label.

    Returns:
        _EnumT: Reconstructed enum member.

    Raises:
        ValueError: If the value is not a string or names no enum member.
    """
    if not isinstance(payload, str):
        raise ValueError(f"{label} must be a string")
    for member in enum_type:
        if member.value == payload:
            return member
    raise ValueError(f"{label} has unknown value {payload!r}")
