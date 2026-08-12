"""Encode and decode nested records in resource-estimate wire payloads."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, cast

import sympy as sp

from qamomile.circuit.estimator._estimate_domain import (
    _DomainRewriteState,
    _PublicResourceSnapshot,
)
from qamomile.circuit.estimator._parameter_domain import _ConsumedDomainRequirement
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    EstimateDerivation,
    EstimateQuality,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_constraints import (
    _ConstraintOrigin,
    _ConstraintProvenance,
    _ConstraintRange,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._resource_types import (
    CallResources,
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
    _guarded_quality_with_reason,
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


def _require_fields(
    record: Mapping[Any, Any],
    fields: set[str],
    label: str,
) -> None:
    """Require an exact field set for one wire record.

    Args:
        record (Mapping[Any, Any]): Wire mapping to validate.
        fields (set[str]): Exact string field names required by the schema.
        label (str): Diagnostic label for malformed payloads.

    Raises:
        ValueError: If a field is missing, unknown, or not a string.
    """
    if set(record) != fields:
        raise ValueError(
            f"{label} fields must be {sorted(fields)!r}, got "
            f"{sorted(record, key=str)!r}"
        )


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
    _require_fields(record, field_names, label)
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


def _calls_to_wire(
    calls: CallResources,
    encoder: _WireExpressionEncoder,
) -> dict[str, dict[str, str]]:
    """Encode named opaque-call and query counts.

    Args:
        calls (CallResources): Named call resources to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, dict[str, str]]: Encoded call and query mappings.
    """
    return {
        "calls_by_name": {
            name: encoder.encode(value) for name, value in calls.calls_by_name.items()
        },
        "queries_by_name": {
            name: encoder.encode(value) for name, value in calls.queries_by_name.items()
        },
    }


def _calls_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> CallResources:
    """Decode named opaque-call and query counts.

    Args:
        payload (Any): Serialized calls mapping.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local decoder.

    Returns:
        CallResources: Reconstructed call resources.

    Raises:
        ValueError: If the calls record or an expression is malformed.
    """
    record = _mapping(payload, label)
    _require_fields(record, {"calls_by_name", "queries_by_name"}, label)
    return CallResources(
        calls_by_name=_expression_map_from_wire(
            record.get("calls_by_name"),
            f"{label}.calls_by_name",
            decoder,
        ),
        queries_by_name=_expression_map_from_wire(
            record.get("queries_by_name"),
            f"{label}.queries_by_name",
            decoder,
        ),
    )


def _public_snapshot_to_wire(
    snapshot: _PublicResourceSnapshot,
    encoder: _WireExpressionEncoder,
) -> dict[str, Any]:
    """Encode one immutable public-resource snapshot.

    Args:
        snapshot (_PublicResourceSnapshot): Snapshot to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, Any]: Encoded snapshot record.
    """
    return {
        "width": _metric_to_wire(snapshot.width_resources(), encoder),
        "gates": _metric_to_wire(snapshot.gate_resources(), encoder),
        "measurements": _metric_to_wire(
            snapshot.measurement_resources(),
            encoder,
        ),
        "resets": _metric_to_wire(snapshot.reset_resources(), encoder),
        "depth": _metric_to_wire(snapshot.depth_resources(), encoder),
        "calls": _calls_to_wire(snapshot.call_resources(), encoder),
    }


def _public_snapshot_from_resources(
    width: WidthResources,
    gates: GateResources,
    measurements: MeasurementResources,
    resets: ResetResources,
    depth: DepthResources,
    calls: CallResources,
) -> _PublicResourceSnapshot:
    """Freeze decoded public metric records scalar by scalar.

    Args:
        width (WidthResources): Decoded width resources.
        gates (GateResources): Decoded gate resources.
        measurements (MeasurementResources): Decoded measurement resources.
        resets (ResetResources): Decoded reset resources.
        depth (DepthResources): Decoded depth resources.
        calls (CallResources): Decoded call and query resources.

    Returns:
        _PublicResourceSnapshot: Immutable public-resource snapshot.
    """

    def values(metric: Any) -> tuple[ResourceExpr, ...]:
        """Copy dataclass fields in their declared order.

        Args:
            metric (object): Public metric record.

        Returns:
            tuple[ResourceExpr, ...]: Scalar values in field order.
        """
        return tuple(
            cast(ResourceExpr, getattr(metric, field.name))
            for field in dataclasses.fields(metric)
        )

    return _PublicResourceSnapshot(
        width=values(width),
        gates=values(gates),
        measurements=values(measurements),
        resets=values(resets),
        depth=values(depth),
        calls=tuple(sorted(calls.calls_by_name.items())),
        queries=tuple(sorted(calls.queries_by_name.items())),
    )


def _public_snapshot_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> _PublicResourceSnapshot:
    """Decode one immutable public-resource snapshot.

    Args:
        payload (Any): Serialized snapshot record.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local decoder.

    Returns:
        _PublicResourceSnapshot: Reconstructed immutable snapshot.

    Raises:
        ValueError: If snapshot fields or expressions are malformed.
    """
    record = _mapping(payload, label)
    _require_fields(
        record,
        {"width", "gates", "measurements", "resets", "depth", "calls"},
        label,
    )
    width = _metric_from_wire(
        record["width"], WidthResources, f"{label}.width", decoder
    )
    gates = _metric_from_wire(record["gates"], GateResources, f"{label}.gates", decoder)
    measurements = _metric_from_wire(
        record["measurements"],
        MeasurementResources,
        f"{label}.measurements",
        decoder,
    )
    resets = _metric_from_wire(
        record["resets"], ResetResources, f"{label}.resets", decoder
    )
    depth = _metric_from_wire(
        record["depth"], DepthResources, f"{label}.depth", decoder
    )
    calls = _calls_from_wire(record["calls"], f"{label}.calls", decoder)

    return _public_snapshot_from_resources(
        width,
        gates,
        measurements,
        resets,
        depth,
        calls,
    )


def _domain_requirement_to_wire(
    requirement: _ConsumedDomainRequirement,
    encoder: _WireExpressionEncoder,
) -> dict[str, Any]:
    """Encode consumed input-domain proof evidence.

    Args:
        requirement (_ConsumedDomainRequirement): Evidence to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, Any]: Encoded predicate and stable source labels.
    """
    return {
        "predicate": encoder.encode(requirement.predicate),
        "source_formals": list(requirement.source_formals),
        "labels": list(requirement.labels),
    }


def _domain_requirement_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _ConsumedDomainRequirement:
    """Decode consumed input-domain proof evidence.

    Args:
        payload (Any): Serialized domain requirement record.
        decoder (_WireExpressionDecoder): Shared payload-local decoder.

    Returns:
        _ConsumedDomainRequirement: Reconstructed proof evidence.

    Raises:
        ValueError: If the predicate or source labels are malformed.
    """
    record = _mapping(payload, "consumed domain requirement")
    _require_fields(
        record, {"predicate", "source_formals", "labels"}, "consumed domain requirement"
    )

    def strings(value: Any, label: str) -> tuple[str, ...]:
        """Decode one ordered string sequence.

        Args:
            value (Any): Serialized sequence.
            label (str): Diagnostic label.

        Returns:
            tuple[str, ...]: Decoded strings.

        Raises:
            ValueError: If any sequence item is not a string.
        """
        items = _sequence(value, label)
        if any(not isinstance(item, str) for item in items):
            raise ValueError(f"{label} must contain only strings")
        return tuple(cast(str, item) for item in items)

    return _ConsumedDomainRequirement(
        predicate=_boolean_expression_from_wire(
            record["predicate"],
            "consumed domain requirement predicate",
            decoder,
        ),
        source_formals=strings(
            record["source_formals"],
            "consumed domain requirement source_formals",
        ),
        labels=strings(record["labels"], "consumed domain requirement labels"),
    )


def _domain_state_to_wire(
    state: _DomainRewriteState | None,
    encoder: _WireExpressionEncoder,
) -> dict[str, Any] | None:
    """Encode retained input-domain rewrite state.

    Args:
        state (_DomainRewriteState | None): State to encode, or ``None``.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, Any] | None: Encoded rewrite state, or ``None``.
    """
    if state is None:
        return None
    return {
        "original": _public_snapshot_to_wire(state.original, encoder),
        "rewritten": _public_snapshot_to_wire(state.rewritten, encoder),
        "requirements": [
            _domain_requirement_to_wire(requirement, encoder)
            for requirement in state.requirements
        ],
    }


def _domain_state_from_wire(
    payload: Any,
    decoder: _WireExpressionDecoder,
) -> _DomainRewriteState | None:
    """Decode retained input-domain rewrite state.

    Serialized proof evidence is reconstructed only for comparison with a
    fresh canonical proof performed by the top-level decoder.

    Args:
        payload (Any): Serialized state record or ``None``.
        decoder (_WireExpressionDecoder): Shared payload-local decoder.

    Returns:
        _DomainRewriteState | None: Reconstructed untrusted state.

    Raises:
        ValueError: If state fields, snapshots, or evidence are malformed.
    """
    if payload is None:
        return None
    record = _mapping(payload, "resource domain rewrite state")
    _require_fields(
        record,
        {"original", "rewritten", "requirements"},
        "resource domain rewrite state",
    )
    return _DomainRewriteState(
        original=_public_snapshot_from_wire(
            record["original"],
            "resource domain rewrite original snapshot",
            decoder,
        ),
        rewritten=_public_snapshot_from_wire(
            record["rewritten"],
            "resource domain rewrite rewritten snapshot",
            decoder,
        ),
        requirements=tuple(
            _domain_requirement_from_wire(item, decoder)
            for item in _sequence(
                record["requirements"],
                "resource domain rewrite requirements",
            )
        ),
    )


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
    _require_fields(record, {"message", "source"}, "resource assumption")
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
    _require_fields(trace, {"root", "nodes"}, "resource trace")
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
    node_fields = {
        "name",
        "source_kind",
        "strategy",
        "summary",
        "assumptions",
        "children",
        "active_when",
    }
    for index, record in enumerate(node_records):
        _require_fields(record, node_fields, f"resource trace node {index}")
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


def _requirement_to_wire(
    requirement: _ResourceConstraint,
    encoder: _WireExpressionEncoder,
) -> dict[str, Any]:
    """Encode one structural resource requirement with typed provenance.

    Args:
        requirement (_ResourceConstraint): Requirement to encode.
        encoder (_WireExpressionEncoder): Shared canonical expression encoder.

    Returns:
        dict[str, Any]: Encoded requirement record.
    """
    return {
        "expression": encoder.encode(requirement.expression),
        "active_when": encoder.encode(requirement.active_when),
        "minimum": requirement.minimum,
        "label": requirement.label,
        "unit": requirement.unit,
        "integer": requirement.integer,
        "minimum_inclusive": requirement.minimum_inclusive,
        "finite": requirement.finite,
        "expected": (
            encoder.encode(requirement.expected)
            if requirement.expected is not None
            else None
        ),
        "ranges": [
            {
                "symbol": encoder.encode(loop_range.symbol),
                "start": encoder.encode(loop_range.start),
                "step": encoder.encode(loop_range.step),
                "iterations": encoder.encode(loop_range.iterations),
            }
            for loop_range in requirement.ranges
        ],
        "provenance": {
            "origin": requirement.provenance.origin.value,
            "source_expressions": [
                encoder.encode(expression)
                for expression in requirement.provenance.source_expressions
            ],
            "root_formal_names": list(requirement.provenance.root_formal_names),
        },
    }


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
    _require_fields(
        record,
        {
            "expression",
            "active_when",
            "minimum",
            "label",
            "unit",
            "integer",
            "minimum_inclusive",
            "finite",
            "expected",
            "ranges",
            "provenance",
        },
        "resource requirement",
    )
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
        _require_fields(
            range_record,
            {"symbol", "start", "step", "iterations"},
            "resource requirement range",
        )
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
    provenance_record = _mapping(
        record.get("provenance"),
        "resource requirement provenance",
    )
    _require_fields(
        provenance_record,
        {"origin", "source_expressions", "root_formal_names"},
        "resource requirement provenance",
    )
    source_expressions: list[sp.Basic] = []
    for index, raw_expression in enumerate(
        _sequence(
            provenance_record["source_expressions"],
            "resource requirement provenance source_expressions",
        )
    ):
        decoded_expression = decoder.decode(
            raw_expression,
            f"resource requirement provenance source expression {index}",
        )
        if not isinstance(decoded_expression, sp.Basic):
            raise ValueError(
                "resource requirement provenance source expressions must "
                "decode to SymPy expressions"
            )
        source_expressions.append(decoded_expression)
    raw_formal_names = _sequence(
        provenance_record["root_formal_names"],
        "resource requirement provenance root_formal_names",
    )
    if any(not isinstance(name, str) for name in raw_formal_names):
        raise ValueError(
            "resource requirement provenance root_formal_names must contain "
            "only strings"
        )
    provenance = _ConstraintProvenance(
        origin=_enum_from_wire(
            _ConstraintOrigin,
            provenance_record["origin"],
            "resource requirement provenance origin",
        ),
        source_expressions=tuple(source_expressions),
        root_formal_names=tuple(cast(str, name) for name in raw_formal_names),
    )
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
        active_when=_boolean_expression_from_wire(
            record["active_when"],
            "resource requirement activation",
            decoder,
        ),
        provenance=provenance,
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
    _require_fields(
        record,
        {"active_when", "assumption"},
        "guarded resource assumption",
    )
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
    _require_fields(
        record,
        {"active_when", "derivation"},
        "guarded resource derivation",
    )
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
        ValueError: If the guard, quality, or required reason is malformed.
    """
    record = _mapping(payload, "guarded resource quality")
    _require_fields(
        record,
        {"active_when", "quality", "reason"},
        "guarded resource quality",
    )
    return _guarded_quality_with_reason(
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
        reason=_assumption_from_wire(record.get("reason")),
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
    _require_fields(
        record,
        {"active_when", "approximation"},
        "guarded resource approximation",
    )
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
