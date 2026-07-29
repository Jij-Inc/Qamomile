"""Encode fixed opaque resource costs at the serialization boundary."""

from __future__ import annotations

import math
from typing import Any

import sympy as sp

from qamomile.circuit.estimator.resource_estimator import (
    CallResources,
    DepthResources,
    EstimateQuality,
    GateResources,
    ResourceAssumption,
    ResourceEstimate,
    ResourceTraceNode,
    WidthResources,
)

_FORMAT_VERSION = 1
_WIDTH_FIELDS = (
    "input_qubits",
    "allocated_qubits",
    "clean_ancilla_qubits",
    "dirty_ancilla_qubits",
    "peak_qubits",
)
_GATE_FIELDS = (
    "total",
    "single_qubit",
    "two_qubit",
    "multi_qubit",
    "clifford",
    "rotation",
    "t",
    "toffoli",
    "non_clifford",
)
_DEPTH_FIELDS = (
    "depth",
    "clifford_depth",
    "rotation_depth",
    "t_depth",
    "toffoli_depth",
    "non_clifford_depth",
    "measurement_depth",
)
_ESTIMATE_FIELDS = {
    "format_version",
    "width",
    "gates",
    "depth",
    "calls",
    "assumptions",
    "trace",
    "quality",
}
_CALL_FIELDS = {"calls_by_name", "queries_by_name"}
_TRACE_FIELDS = {
    "name",
    "source_kind",
    "strategy",
    "summary",
    "assumptions",
    "children",
}


def encode_fixed_resource_estimate(
    cost: Any,
    callable_name: str,
) -> dict[str, Any] | None:
    """Encode one fixed opaque cost without serializing Python behavior.

    Args:
        cost (Any): Opaque cost attached to a callable definition.
        callable_name (str): Callable name used in diagnostics.

    Returns:
        dict[str, Any] | None: Closed payload record for a fixed
            ``ResourceEstimate``, or ``None`` when no cost is attached.

    Raises:
        TypeError: If the cost is a callback, has an unsupported type, or
            contains symbolic or otherwise unsupported resource values.
        ValueError: If a resource value is negative or malformed.
    """
    if cost is None:
        return None
    if callable(cost):
        raise TypeError(
            f"Cannot serialize context-dependent opaque cost callback for "
            f"{callable_name!r}. Use a fixed ResourceEstimate; Python callback "
            "behavior is process-local."
        )
    if not isinstance(cost, ResourceEstimate):
        raise TypeError(
            f"Cannot serialize opaque cost for {callable_name!r}: expected a "
            f"ResourceEstimate, got {type(cost).__name__}."
        )
    if cost.parameters:
        raise TypeError(
            f"Cannot serialize symbolic opaque cost for {callable_name!r}: "
            "fixed ResourceEstimate costs cannot declare parameters."
        )
    return {
        "format_version": _FORMAT_VERSION,
        "width": _encode_resource_record(
            cost.width,
            _WIDTH_FIELDS,
            f"{callable_name}.width",
        ),
        "gates": _encode_resource_record(
            cost.gates,
            _GATE_FIELDS,
            f"{callable_name}.gates",
        ),
        "depth": _encode_resource_record(
            cost.depth,
            _DEPTH_FIELDS,
            f"{callable_name}.depth",
        ),
        "calls": {
            "calls_by_name": _encode_named_resources(
                cost.calls.calls_by_name,
                f"{callable_name}.calls.calls_by_name",
            ),
            "queries_by_name": _encode_named_resources(
                cost.calls.queries_by_name,
                f"{callable_name}.calls.queries_by_name",
            ),
        },
        "assumptions": [
            _encode_assumption(assumption, f"{callable_name}.assumptions")
            for assumption in cost.assumptions
        ],
        "trace": (
            _encode_trace(cost.trace, f"{callable_name}.trace")
            if cost.trace is not None
            else None
        ),
        "quality": cost.quality.value,
    }


def decode_fixed_resource_estimate(
    payload: Any,
    callable_name: str,
) -> ResourceEstimate | None:
    """Decode one fixed opaque cost from a closed semantic payload.

    Args:
        payload (Any): Decoded protobuf payload for ``opaque_cost``.
        callable_name (str): Callable name used in diagnostics.

    Returns:
        ResourceEstimate | None: Restored fixed cost, or ``None`` when absent.

    Raises:
        ValueError: If the payload is malformed, unsupported, or contains a
            negative resource value.
    """
    if payload is None:
        return None
    record = _require_record(payload, _ESTIMATE_FIELDS, callable_name)
    format_version = _require_plain_integer(
        record["format_version"],
        f"{callable_name}.format_version",
    )
    if format_version != _FORMAT_VERSION:
        raise ValueError(
            f"opaque cost for {callable_name!r} uses unsupported format version "
            f"{format_version!r}"
        )
    raw_calls = _require_record(
        record["calls"],
        _CALL_FIELDS,
        f"{callable_name}.calls",
    )
    raw_quality = record["quality"]
    if not isinstance(raw_quality, str):
        raise ValueError(f"{callable_name}.quality must be a string")
    try:
        quality = EstimateQuality(raw_quality)
    except ValueError as exc:
        raise ValueError(
            f"{callable_name}.quality is not a known EstimateQuality"
        ) from exc

    raw_assumptions = _require_list(
        record["assumptions"],
        f"{callable_name}.assumptions",
    )
    raw_trace = record["trace"]
    return ResourceEstimate(
        width=WidthResources(
            **_decode_resource_record(
                record["width"],
                _WIDTH_FIELDS,
                f"{callable_name}.width",
            )
        ),
        gates=GateResources(
            **_decode_resource_record(
                record["gates"],
                _GATE_FIELDS,
                f"{callable_name}.gates",
            )
        ),
        depth=DepthResources(
            **_decode_resource_record(
                record["depth"],
                _DEPTH_FIELDS,
                f"{callable_name}.depth",
            )
        ),
        calls=CallResources(
            calls_by_name=_decode_named_resources(
                raw_calls["calls_by_name"],
                f"{callable_name}.calls.calls_by_name",
            ),
            queries_by_name=_decode_named_resources(
                raw_calls["queries_by_name"],
                f"{callable_name}.calls.queries_by_name",
            ),
        ),
        assumptions=tuple(
            _decode_assumption(item, f"{callable_name}.assumptions[{index}]")
            for index, item in enumerate(raw_assumptions)
        ),
        trace=(
            None
            if raw_trace is None
            else _decode_trace(raw_trace, f"{callable_name}.trace")
        ),
        quality=quality,
    )


def _encode_resource_record(
    resource: Any,
    fields: tuple[str, ...],
    location: str,
) -> dict[str, Any]:
    """Encode the scalar fields of one resource category.

    Args:
        resource (Any): Resource category dataclass.
        fields (tuple[str, ...]): Ordered scalar field names.
        location (str): Human-readable payload location.

    Returns:
        dict[str, Any]: Encoded scalar fields.

    Raises:
        TypeError: If a field is absent or contains an unsupported value.
        ValueError: If a resource value is negative.
    """
    encoded: dict[str, Any] = {}
    for field in fields:
        if not hasattr(resource, field):
            raise TypeError(f"{location} is missing field {field!r}")
        encoded[field] = _encode_resource_number(
            getattr(resource, field),
            f"{location}.{field}",
        )
    return encoded


def _decode_resource_record(
    payload: Any,
    fields: tuple[str, ...],
    location: str,
) -> dict[str, sp.Expr]:
    """Decode the scalar fields of one resource category.

    Args:
        payload (Any): Decoded category payload.
        fields (tuple[str, ...]): Expected scalar field names.
        location (str): Human-readable payload location.

    Returns:
        dict[str, sp.Expr]: Decoded resource expressions by field name.

    Raises:
        ValueError: If the category or any scalar field is malformed.
    """
    record = _require_record(payload, set(fields), location)
    return {
        field: _decode_resource_number(record[field], f"{location}.{field}")
        for field in fields
    }


def _encode_named_resources(
    resources: Any,
    location: str,
) -> list[list[Any]]:
    """Encode a name-to-resource mapping in deterministic key order.

    Args:
        resources (Any): Mapping of callable names to resource values.
        location (str): Human-readable payload location.

    Returns:
        list[list[Any]]: Sorted ``[name, encoded value]`` entries.

    Raises:
        TypeError: If the mapping, a name, or a value is unsupported.
        ValueError: If a resource value is negative.
    """
    if not isinstance(resources, dict):
        raise TypeError(f"{location} must be a dict")
    for name in resources:
        if not isinstance(name, str):
            raise TypeError(f"{location} keys must be strings")
    return [
        [name, _encode_resource_number(resources[name], f"{location}[{name!r}]")]
        for name in sorted(resources)
    ]


def _decode_named_resources(
    payload: Any,
    location: str,
) -> dict[str, sp.Expr]:
    """Decode a canonical list of named resource values.

    Args:
        payload (Any): Sorted ``[name, encoded value]`` entries.
        location (str): Human-readable payload location.

    Returns:
        dict[str, sp.Expr]: Decoded resources keyed by callable name.

    Raises:
        ValueError: If entries are malformed, duplicated, or unsorted.
    """
    entries = _require_list(payload, location)
    decoded: dict[str, sp.Expr] = {}
    previous_name: str | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"{location}[{index}] must be [name, value]")
        name, raw_value = entry
        if not isinstance(name, str):
            raise ValueError(f"{location}[{index}] name must be a string")
        if previous_name is not None and name <= previous_name:
            raise ValueError(f"{location} names must be unique and sorted")
        decoded[name] = _decode_resource_number(
            raw_value,
            f"{location}[{name!r}]",
        )
        previous_name = name
    return decoded


def _encode_resource_number(value: Any, location: str) -> dict[str, Any]:
    """Encode one concrete nonnegative resource value.

    Args:
        value (Any): Python or SymPy numeric resource value.
        location (str): Human-readable payload location.

    Returns:
        dict[str, Any]: Tagged exact integer, rational, or arbitrary-precision
            binary floating-point record.

    Raises:
        TypeError: If the value is symbolic, Boolean, complex, or otherwise
            unsupported.
        ValueError: If the value is negative or non-finite.
    """
    if isinstance(value, bool):
        raise TypeError(f"{location} cannot be Boolean")
    if isinstance(value, int):
        number: sp.Expr = sp.Integer(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        number = sp.Float(value)
    elif isinstance(value, sp.Expr):
        number = value
    else:
        raise TypeError(
            f"{location} must be a concrete int, float, or SymPy number; got "
            f"{type(value).__name__}"
        )
    if number.free_symbols:
        raise TypeError(f"{location} must be fixed, not symbolic")
    if number.is_real is not True or number.is_finite is not True:
        raise ValueError(f"{location} must be a finite real number")
    if number.is_negative is True:
        raise ValueError(f"{location} cannot be negative")
    if isinstance(number, sp.Integer):
        return {"kind": "integer", "value": int(number)}
    if isinstance(number, sp.Rational):
        return {
            "kind": "rational",
            "numerator": int(number.p),
            "denominator": int(number.q),
        }
    if isinstance(number, sp.Float):
        sign, mantissa, exponent, bitcount = number._mpf_
        return {
            "kind": "float",
            "sign": sign,
            "mantissa": mantissa,
            "exponent": exponent,
            "bitcount": bitcount,
            "precision": number._prec,
        }
    raise TypeError(
        f"{location} uses unsupported concrete SymPy expression {type(number).__name__}"
    )


def _decode_resource_number(payload: Any, location: str) -> sp.Expr:
    """Decode one concrete nonnegative resource value.

    Args:
        payload (Any): Tagged numeric resource record.
        location (str): Human-readable payload location.

    Returns:
        sp.Expr: Exact integer, rational, or arbitrary-precision SymPy value.

    Raises:
        ValueError: If the numeric record is malformed, negative, or
            non-finite.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"{location} must be a numeric record")
    kind = payload.get("kind")
    if kind == "integer":
        record = _require_record(payload, {"kind", "value"}, location)
        value = record["value"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{location}.value must be an integer")
        number: sp.Expr = sp.Integer(value)
    elif kind == "rational":
        record = _require_record(
            payload,
            {"kind", "numerator", "denominator"},
            location,
        )
        numerator = record["numerator"]
        denominator = record["denominator"]
        if (
            not isinstance(numerator, int)
            or isinstance(numerator, bool)
            or not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator <= 1
        ):
            raise ValueError(
                f"{location} rational numerator and denominator are malformed"
            )
        if math.gcd(numerator, denominator) != 1:
            raise ValueError(f"{location} rational value is not canonical")
        number = sp.Rational(numerator, denominator)
    elif kind == "float":
        record = _require_record(
            payload,
            {
                "kind",
                "sign",
                "mantissa",
                "exponent",
                "bitcount",
                "precision",
            },
            location,
        )
        sign = _require_plain_integer(record["sign"], f"{location}.sign")
        mantissa = _require_plain_integer(
            record["mantissa"],
            f"{location}.mantissa",
        )
        exponent = _require_plain_integer(
            record["exponent"],
            f"{location}.exponent",
        )
        bitcount = _require_plain_integer(
            record["bitcount"],
            f"{location}.bitcount",
        )
        precision = _require_plain_integer(
            record["precision"],
            f"{location}.precision",
        )
        if sign not in (0, 1):
            raise ValueError(f"{location}.sign must be zero or one")
        if precision < 1:
            raise ValueError(f"{location}.precision must be positive")
        if mantissa == 0:
            if sign != 0 or exponent != 0 or bitcount != 0:
                raise ValueError(f"{location} zero float is not canonical")
        elif (
            mantissa < 0
            or mantissa % 2 == 0
            or bitcount != mantissa.bit_length()
            or bitcount > precision
        ):
            raise ValueError(f"{location} float significand is not canonical")
        raw_mpf = (sign, mantissa, exponent, bitcount)
        try:
            number = (
                sp.Float(0, precision=precision)
                if mantissa == 0
                else sp.Float(raw_mpf, precision=precision)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location} contains a malformed float") from exc
        if number._mpf_ != raw_mpf or number._prec != precision:
            raise ValueError(f"{location} float encoding is not canonical")
        if number.is_real is not True or number.is_finite is not True:
            raise ValueError(f"{location} must be a finite real number")
    else:
        raise ValueError(f"{location} has unknown numeric kind {kind!r}")
    if number.is_negative is True:
        raise ValueError(f"{location} cannot be negative")
    return number


def _require_plain_integer(value: Any, location: str) -> int:
    """Require one non-Boolean Python integer.

    Args:
        value (Any): Candidate decoded integer field.
        location (str): Human-readable payload location.

    Returns:
        int: Validated integer.

    Raises:
        ValueError: If ``value`` is not a plain Python integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    return value


def _encode_assumption(
    assumption: Any,
    location: str,
) -> dict[str, Any]:
    """Encode one resource-estimation assumption.

    Args:
        assumption (Any): Candidate ``ResourceAssumption``.
        location (str): Human-readable payload location.

    Returns:
        dict[str, Any]: Assumption record.

    Raises:
        TypeError: If the assumption fields are malformed.
    """
    if not isinstance(assumption, ResourceAssumption):
        raise TypeError(f"{location} entries must be ResourceAssumption objects")
    if not isinstance(assumption.message, str) or (
        assumption.source is not None and not isinstance(assumption.source, str)
    ):
        raise TypeError(f"{location} contains malformed assumption text")
    return {"message": assumption.message, "source": assumption.source}


def _decode_assumption(payload: Any, location: str) -> ResourceAssumption:
    """Decode one resource-estimation assumption.

    Args:
        payload (Any): Decoded assumption payload.
        location (str): Human-readable payload location.

    Returns:
        ResourceAssumption: Restored assumption.

    Raises:
        ValueError: If the assumption fields are malformed.
    """
    record = _require_record(payload, {"message", "source"}, location)
    message = record["message"]
    source = record["source"]
    if not isinstance(message, str) or (
        source is not None and not isinstance(source, str)
    ):
        raise ValueError(f"{location} contains malformed assumption text")
    return ResourceAssumption(message=message, source=source)


def _encode_trace(trace: Any, location: str) -> dict[str, Any]:
    """Encode one resource-estimation trace node recursively.

    Args:
        trace (Any): Candidate ``ResourceTraceNode``.
        location (str): Human-readable payload location.

    Returns:
        dict[str, Any]: Closed trace-node record.

    Raises:
        TypeError: If the trace node or one of its fields is malformed.
    """
    if not isinstance(trace, ResourceTraceNode):
        raise TypeError(f"{location} must be a ResourceTraceNode")
    if (
        not isinstance(trace.name, str)
        or not isinstance(trace.source_kind, str)
        or (trace.strategy is not None and not isinstance(trace.strategy, str))
        or not isinstance(trace.summary, str)
    ):
        raise TypeError(f"{location} contains malformed trace text")
    return {
        "name": trace.name,
        "source_kind": trace.source_kind,
        "strategy": trace.strategy,
        "summary": trace.summary,
        "assumptions": [
            _encode_assumption(assumption, f"{location}.assumptions")
            for assumption in trace.assumptions
        ],
        "children": [
            _encode_trace(child, f"{location}.children[{index}]")
            for index, child in enumerate(trace.children)
        ],
    }


def _decode_trace(payload: Any, location: str) -> ResourceTraceNode:
    """Decode one resource-estimation trace node recursively.

    Args:
        payload (Any): Decoded trace-node payload.
        location (str): Human-readable payload location.

    Returns:
        ResourceTraceNode: Restored trace node.

    Raises:
        ValueError: If the trace node or one of its fields is malformed.
    """
    record = _require_record(payload, _TRACE_FIELDS, location)
    name = record["name"]
    source_kind = record["source_kind"]
    strategy = record["strategy"]
    summary = record["summary"]
    if (
        not isinstance(name, str)
        or not isinstance(source_kind, str)
        or (strategy is not None and not isinstance(strategy, str))
        or not isinstance(summary, str)
    ):
        raise ValueError(f"{location} contains malformed trace text")
    raw_assumptions = _require_list(
        record["assumptions"],
        f"{location}.assumptions",
    )
    raw_children = _require_list(record["children"], f"{location}.children")
    return ResourceTraceNode(
        name=name,
        source_kind=source_kind,
        strategy=strategy,
        summary=summary,
        assumptions=tuple(
            _decode_assumption(item, f"{location}.assumptions[{index}]")
            for index, item in enumerate(raw_assumptions)
        ),
        children=tuple(
            _decode_trace(item, f"{location}.children[{index}]")
            for index, item in enumerate(raw_children)
        ),
    )


def _require_record(
    payload: Any,
    fields: set[str],
    location: str,
) -> dict[str, Any]:
    """Require one dictionary with exactly the expected string keys.

    Args:
        payload (Any): Candidate decoded protobuf payload.
        fields (set[str]): Exact accepted field names.
        location (str): Human-readable payload location.

    Returns:
        dict[str, Any]: Validated record.

    Raises:
        ValueError: If the payload is not a dict or has different keys.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"{location} must be a record")
    if not all(isinstance(field, str) for field in payload):
        raise ValueError(f"{location} record keys must be strings")
    if set(payload) != fields:
        raise ValueError(
            f"{location} requires exactly fields {sorted(fields)!r}; got "
            f"{sorted(payload)!r}"
        )
    return payload


def _require_list(payload: Any, location: str) -> list[Any]:
    """Require one decoded protobuf list.

    Args:
        payload (Any): Candidate decoded protobuf payload.
        location (str): Human-readable payload location.

    Returns:
        list[Any]: Validated list.

    Raises:
        ValueError: If the payload is not a list.
    """
    if not isinstance(payload, list):
        raise ValueError(f"{location} must be a list")
    return payload
