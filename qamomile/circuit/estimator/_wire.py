"""Encode fixed resource estimates for semantic IR serialization.

The user-facing :meth:`ResourceEstimate.to_dict` representation is intended
for reports.  Opaque callable costs need a stronger round-trip contract:
symbol domains, guarded provenance, requirements, and explanation traces must
remain usable after a qkernel is deserialized.  This module provides that
closed internal representation without serializing Python callback objects.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import sympy as sp
from sympy.core.function import AppliedUndef, FunctionClass
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._metrics import (
    ApproximationStatus,
    CallResources,
    ControlDecomposition,
    DepthResources,
    EstimateDerivation,
    EstimateQuality,
    GateBasis,
    GateResources,
    MeasurementResources,
    ResetResources,
    ResourceAssumption,
    ResourceTraceNode,
    WidthResources,
    _ConstraintRange,
    _GuardedApproximation,
    _GuardedAssumption,
    _GuardedDerivation,
    _GuardedQuality,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

_MetricT = TypeVar(
    "_MetricT",
    WidthResources,
    GateResources,
    DepthResources,
    MeasurementResources,
    ResetResources,
)
_EnumT = TypeVar("_EnumT", bound=enum.Enum)
_RESOURCE_ESTIMATE_WIRE_VERSION = 4
_MAX_EXPRESSION_NODES = 100_000
_MAX_NUMERIC_BITS = 4096
# ceil(4096 / log2(10)); bounds decimal mantissa parsing independently from a
# Float's compact decimal exponent.
_MAX_DECIMAL_FLOAT_DIGITS = 1234
_MAX_DECIMAL_FLOAT_EXPONENT_DIGITS = 4
_MAX_TRACE_NODES = 100_000
_SAFE_SYMPY_NAMES = frozenset(
    {
        "Abs",
        "Add",
        "And",
        "BooleanFalse",
        "BooleanTrue",
        "Dummy",
        "E",
        "Equality",
        "ExprCondPair",
        "Float",
        "Function",
        "GreaterThan",
        "Integer",
        "Lambda",
        "LessThan",
        "Max",
        "Min",
        "Mod",
        "Mul",
        "Not",
        "Or",
        "Piecewise",
        "Pow",
        "Rational",
        "StrictGreaterThan",
        "StrictLessThan",
        "Sum",
        "Symbol",
        "Tuple",
        "Unequality",
        "Xor",
        "ceiling",
        "cos",
        "exp",
        "floor",
        "log",
        "nan",
        "oo",
        "pi",
        "sign",
        "sin",
        "tan",
        "true",
        "false",
        "zoo",
    }
)


class _WireExpressionEncoder:
    """Canonicalize every expression in one resource wire payload.

    Ordinary symbols retain their public names. Identity-only ``Dummy``
    symbols instead receive payload-local slots in deterministic encounter
    order, removing SymPy's process-random ``dummy_index`` while preserving
    identity across every metric, requirement, and trace expression.

    Args:
        registry (SymbolRegistry): Symbol registry for one resource estimate.
        dummy_slots (dict[sp.Dummy, int] | None): Optional payload-wide mapping
            from source Dummy identities to deterministic slots. Defaults to a
            mapping local to this resource estimate.
    """

    def __init__(
        self,
        registry: SymbolRegistry,
        dummy_slots: dict[sp.Dummy, int] | None = None,
    ) -> None:
        """Build canonical replacements for every registered symbol.

        Args:
            registry (SymbolRegistry): Symbol registry for one resource
                estimate.
            dummy_slots (dict[sp.Dummy, int] | None): Optional payload-wide
                Dummy slot mapping. Defaults to ``None``.
        """
        replacements: dict[sp.Symbol, sp.Symbol] = {}
        resolved_dummy_slots = {} if dummy_slots is None else dummy_slots
        for symbol, public_name in registry.aliases().items():
            if isinstance(symbol, sp.Dummy):
                dummy_slot = resolved_dummy_slots.get(symbol)
                if dummy_slot is None:
                    dummy_slot = len(resolved_dummy_slots)
                    resolved_dummy_slots[symbol] = dummy_slot
                replacement = sp.Dummy(
                    public_name,
                    dummy_index=dummy_slot,
                    **symbol.assumptions0,
                )
            else:
                replacement = sp.Symbol(public_name, **symbol.assumptions0)
            replacements[symbol] = replacement
        self._replacements = replacements

    def encode(self, expression: Any) -> str:
        """Encode one expression with payload-canonical symbol identities.

        Args:
            expression (Any): SymPy-compatible expression.

        Returns:
            str: Deterministic SymPy structural representation.

        Raises:
            ValueError: If the expression cannot be decoded by the matching
                closed wire-expression language.
        """
        normalized = sp.sympify(expression).xreplace(self._replacements)
        _validate_sympy_expression_for_wire(normalized)
        payload = sp.srepr(normalized)
        return payload


class _WireExpressionDecoder:
    """Decode Dummy identities from one paired encoder stream.

    A decoder may span multiple resource records only when their encoders
    shared the same ``dummy_slots`` mapping. Independent wire payloads require
    independent decoder instances because their integer slots are local.
    """

    def __init__(self) -> None:
        """Initialize one encoder-stream-local Dummy mapping."""
        self._dummies: dict[int, sp.Dummy] = {}

    def decode(self, payload: Any, label: str) -> sp.Basic:
        """Decode one expression and freshen its canonical Dummy symbols.

        Args:
            payload (Any): Structural SymPy representation string.
            label (str): Diagnostic label.

        Returns:
            sp.Basic: Decoded expression sharing fresh Dummies only within this
                resource payload.

        Raises:
            ValueError: If the expression is outside the supported language.
        """
        expression = _expression_from_wire(payload, label)
        replacements: dict[sp.Dummy, sp.Dummy] = {}
        for symbol in expression.atoms(sp.Dummy):
            slot = cast(int, getattr(symbol, "dummy_index"))
            replacement = self._dummies.get(slot)
            if replacement is None:
                replacement = sp.Dummy(symbol.name, **symbol.assumptions0)
                self._dummies[slot] = replacement
            elif replacement.assumptions0 != symbol.assumptions0:
                raise ValueError(
                    f"{label} assigns conflicting assumptions to Dummy slot {slot}"
                )
            replacements[symbol] = replacement
        return expression.xreplace(replacements)


def resource_estimate_to_wire(
    estimate: ResourceEstimate,
    *,
    dummy_slots: dict[sp.Dummy, int] | None = None,
) -> dict[str, Any]:
    """Encode one fixed resource estimate as serializer-friendly data.

    Args:
        estimate (ResourceEstimate): Fixed resource estimate to encode.
        dummy_slots (dict[sp.Dummy, int] | None): Optional payload-wide Dummy
            slot mapping shared by every opaque cost in one serialized
            qkernel. Defaults to ``None``.

    Returns:
        dict[str, Any]: Closed payload containing metrics, requirements,
            guarded provenance, and symbolic expressions.

    Raises:
        TypeError: If ``estimate`` is not a ``ResourceEstimate``.
        ValueError: If its expression language or explanation trace exceeds
            the supported wire contract, or if it carries caller-scoped
            liveness state that has no meaning for an opaque definition.
    """
    from qamomile.circuit.estimator.resource_estimator import (
        ResourceEstimate,
        _serialization_expressions,
    )

    if not isinstance(estimate, ResourceEstimate):
        raise TypeError(
            "opaque_cost serialization requires a fixed ResourceEstimate; "
            f"got {type(estimate).__name__}"
        )
    if estimate._output_sizes or estimate._input_sizes or estimate._has_output_summary:
        raise ValueError(
            "opaque_cost serialization does not support caller-scoped "
            "input/output liveness summaries"
        )
    registry = SymbolRegistry.from_expressions(
        (
            *_serialization_expressions(estimate),
            *_trace_activation_guards(estimate.trace),
        ),
        estimate._symbol_aliases,
    )
    encoder = _WireExpressionEncoder(registry, dummy_slots)
    expression = encoder.encode
    guarded_assumptions = estimate._guarded_assumptions or ()
    guarded_derivations = estimate._guarded_derivations or ()
    guarded_qualities = estimate._guarded_qualities or ()
    guarded_approximations = estimate._guarded_approximations or ()
    return {
        "$type": "ResourceEstimate",
        "version": _RESOURCE_ESTIMATE_WIRE_VERSION,
        "width": _metric_to_wire(estimate.width, encoder),
        "gates": _metric_to_wire(estimate.gates, encoder),
        "depth": _metric_to_wire(estimate.depth, encoder),
        "measurements": _metric_to_wire(estimate.measurements, encoder),
        "resets": _metric_to_wire(estimate.resets, encoder),
        "calls": {
            "calls_by_name": {
                name: expression(value)
                for name, value in estimate.calls.calls_by_name.items()
            },
            "queries_by_name": {
                name: expression(value)
                for name, value in estimate.calls.queries_by_name.items()
            },
        },
        "assumptions": [
            _assumption_to_wire(assumption) for assumption in estimate.assumptions
        ],
        "trace": _trace_to_wire(estimate.trace, encoder),
        "symbol_aliases": {
            alias: expression(symbol) for symbol, alias in registry.aliases().items()
        },
        "parameters": {
            name: expression(symbol) for name, symbol in estimate.parameters.items()
        },
        "derivation": estimate.derivation.value,
        "quality": estimate.quality.value,
        "approximation": estimate.approximation.value,
        "basis": estimate.basis.value,
        "control_decomposition": estimate.control_decomposition.value,
        "precision": estimate.precision,
        "requirements": [
            {
                "expression": expression(constraint.expression),
                "minimum": constraint.minimum,
                "label": constraint.label,
                "unit": constraint.unit,
                "integer": constraint.integer,
                "minimum_inclusive": constraint.minimum_inclusive,
                "finite": constraint.finite,
                "expected": (
                    expression(constraint.expected)
                    if constraint.expected is not None
                    else None
                ),
                "ranges": [
                    {
                        "symbol": expression(loop_range.symbol),
                        "start": expression(loop_range.start),
                        "step": expression(loop_range.step),
                        "iterations": expression(loop_range.iterations),
                    }
                    for loop_range in constraint.ranges
                ],
            }
            for constraint in estimate._constraints
        ],
        "provenance": {
            "assumptions": [
                {
                    "active_when": expression(fact.active_when),
                    "assumption": _assumption_to_wire(fact.assumption),
                }
                for fact in guarded_assumptions
            ],
            "derivations": [
                {
                    "active_when": expression(fact.active_when),
                    "derivation": fact.derivation.value,
                }
                for fact in guarded_derivations
            ],
            "qualities": [
                {
                    "active_when": expression(fact.active_when),
                    "quality": fact.quality.value,
                }
                for fact in guarded_qualities
            ],
            "approximations": [
                {
                    "active_when": expression(fact.active_when),
                    "approximation": fact.approximation.value,
                }
                for fact in guarded_approximations
            ],
        },
    }


def resource_estimate_from_wire(
    payload: Any,
    *,
    decoder: _WireExpressionDecoder | None = None,
) -> ResourceEstimate:
    """Decode one fixed resource estimate from semantic IR data.

    Args:
        payload (Any): Payload produced by :func:`resource_estimate_to_wire`.
        decoder (_WireExpressionDecoder | None): Optional payload-wide
            expression decoder shared by every opaque cost in one serialized
            qkernel. Defaults to ``None``.

    Returns:
        ResourceEstimate: Reconstructed fixed resource estimate.

    Raises:
        ValueError: If the payload, an enum, a symbolic expression, or
            provenance metadata is malformed.
    """
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

    record = _mapping(payload, "opaque ResourceEstimate")
    if record.get("$type") != "ResourceEstimate":
        raise ValueError("opaque cost payload is not a ResourceEstimate")
    if record.get("version") != _RESOURCE_ESTIMATE_WIRE_VERSION:
        raise ValueError(
            "unsupported opaque ResourceEstimate wire version "
            f"{record.get('version')!r}"
        )

    decoder = _WireExpressionDecoder() if decoder is None else decoder
    calls = _mapping(record.get("calls"), "opaque ResourceEstimate calls")
    provenance = _mapping(
        record.get("provenance"),
        "opaque ResourceEstimate provenance",
    )
    assumptions = tuple(
        _assumption_from_wire(item)
        for item in _sequence(
            record.get("assumptions"),
            "opaque ResourceEstimate assumptions",
        )
    )
    requirements = tuple(
        _requirement_from_wire(item, decoder)
        for item in _sequence(
            record.get("requirements"),
            "opaque ResourceEstimate requirements",
        )
    )
    guarded_assumptions = tuple(
        _guarded_assumption_from_wire(item, decoder)
        for item in _sequence(
            provenance.get("assumptions"),
            "opaque ResourceEstimate assumption provenance",
        )
    )
    guarded_derivations = tuple(
        _guarded_derivation_from_wire(item, decoder)
        for item in _sequence(
            provenance.get("derivations"),
            "opaque ResourceEstimate derivation provenance",
        )
    )
    guarded_qualities = tuple(
        _guarded_quality_from_wire(item, decoder)
        for item in _sequence(
            provenance.get("qualities"),
            "opaque ResourceEstimate quality provenance",
        )
    )
    guarded_approximations = tuple(
        _guarded_approximation_from_wire(item, decoder)
        for item in _sequence(
            provenance.get("approximations"),
            "opaque ResourceEstimate approximation provenance",
        )
    )
    estimate = ResourceEstimate(
        width=_metric_from_wire(
            record.get("width"),
            WidthResources,
            "opaque ResourceEstimate width",
            decoder,
        ),
        gates=_metric_from_wire(
            record.get("gates"),
            GateResources,
            "opaque ResourceEstimate gates",
            decoder,
        ),
        depth=_metric_from_wire(
            record.get("depth"),
            DepthResources,
            "opaque ResourceEstimate depth",
            decoder,
        ),
        measurements=_metric_from_wire(
            record.get("measurements"),
            MeasurementResources,
            "opaque ResourceEstimate measurements",
            decoder,
        ),
        resets=_metric_from_wire(
            record.get("resets"),
            ResetResources,
            "opaque ResourceEstimate resets",
            decoder,
        ),
        calls=CallResources(
            calls_by_name=_expression_map_from_wire(
                calls.get("calls_by_name"),
                "opaque ResourceEstimate calls_by_name",
                decoder,
            ),
            queries_by_name=_expression_map_from_wire(
                calls.get("queries_by_name"),
                "opaque ResourceEstimate queries_by_name",
                decoder,
            ),
        ),
        assumptions=assumptions,
        trace=_trace_from_wire(record.get("trace"), decoder),
        derivation=_enum_from_wire(
            EstimateDerivation,
            record.get("derivation"),
            "opaque ResourceEstimate derivation",
        ),
        quality=_enum_from_wire(
            EstimateQuality,
            record.get("quality"),
            "opaque ResourceEstimate quality",
        ),
        approximation=_enum_from_wire(
            ApproximationStatus,
            record.get("approximation"),
            "opaque ResourceEstimate approximation",
        ),
        basis=_enum_from_wire(
            GateBasis,
            record.get("basis"),
            "opaque ResourceEstimate basis",
        ),
        control_decomposition=_enum_from_wire(
            ControlDecomposition,
            record.get("control_decomposition"),
            "opaque ResourceEstimate control decomposition",
        ),
        precision=_precision_from_wire(record.get("precision")),
        _constraints=requirements,
        _guarded_assumptions=guarded_assumptions,
        _guarded_derivations=guarded_derivations,
        _guarded_qualities=guarded_qualities,
        _guarded_approximations=guarded_approximations,
    )
    raw_parameters = _mapping(
        record.get("parameters"),
        "opaque ResourceEstimate parameters",
    )
    symbol_aliases = _symbol_aliases_from_wire(
        record.get("symbol_aliases"),
        estimate,
        decoder,
    )
    expected_parameters: dict[str, sp.Symbol] = {}
    for name, symbol_payload in raw_parameters.items():
        if not isinstance(name, str):
            raise ValueError("opaque ResourceEstimate parameter names must be strings")
        symbol = decoder.decode(
            symbol_payload,
            f"opaque ResourceEstimate parameter {name!r}",
        )
        if not isinstance(symbol, sp.Symbol):
            raise ValueError(
                "opaque ResourceEstimate parameter metadata must decode to symbols"
            )
        expected_parameters[name] = symbol
    if len(set(expected_parameters.values())) != len(expected_parameters):
        raise ValueError(
            "opaque ResourceEstimate parameter metadata assigns multiple public "
            "names to one symbol"
        )
    if any(
        symbol_aliases.get(symbol) != public_name
        for public_name, symbol in expected_parameters.items()
    ):
        raise ValueError(
            "opaque ResourceEstimate parameter metadata disagrees with its "
            "symbol aliases"
        )
    actual_symbols = set(estimate.parameters.values())
    if set(expected_parameters.values()) != actual_symbols:
        raise ValueError(
            "opaque ResourceEstimate parameter metadata disagrees with its "
            "symbolic expressions"
        )
    estimate._symbol_aliases = symbol_aliases
    estimate._refresh_symbol_metadata()
    return estimate


def _symbol_aliases_from_wire(
    payload: Any,
    estimate: ResourceEstimate,
    decoder: _WireExpressionDecoder,
) -> dict[sp.Symbol, str]:
    """Decode the complete identity-to-public-name symbol mapping.

    Args:
        payload (Any): Serialized alias mapping keyed by public name.
        estimate (ResourceEstimate): Decoded estimate whose expressions define
            the allowed symbol identities.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        dict[sp.Symbol, str]: Symbol identities mapped to public aliases.

    Raises:
        ValueError: If an alias is malformed, duplicated, missing, or refers
            to a symbol outside the estimate.
    """
    from qamomile.circuit.estimator.resource_estimator import (
        _serialization_expressions,
    )

    available: set[sp.Symbol] = set()
    for expression in (
        *_serialization_expressions(estimate),
        *_trace_activation_guards(estimate.trace),
    ):
        available.update(sp.sympify(expression).atoms(sp.Symbol))

    aliases: dict[sp.Symbol, str] = {}
    for public_name, symbol_payload in _mapping(
        payload,
        "opaque ResourceEstimate symbol_aliases",
    ).items():
        if not isinstance(public_name, str):
            raise ValueError("opaque ResourceEstimate symbol aliases must be strings")
        symbol = decoder.decode(
            symbol_payload,
            f"opaque ResourceEstimate symbol alias {public_name!r}",
        )
        if not isinstance(symbol, sp.Symbol):
            raise ValueError(
                "opaque ResourceEstimate symbol aliases must refer to symbols"
            )
        if symbol not in available:
            raise ValueError(
                "opaque ResourceEstimate symbol alias refers to a symbol "
                "outside its expressions"
            )
        if symbol in aliases:
            raise ValueError(
                "opaque ResourceEstimate symbol aliases assign multiple names "
                "to one symbol"
            )
        aliases[symbol] = public_name
    if set(aliases) != available:
        raise ValueError(
            "opaque ResourceEstimate symbol aliases do not cover every symbol"
        )
    return aliases


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


def _trace_activation_guards(
    trace: ResourceTraceNode | None,
) -> Sequence[sp.Basic]:
    """Collect every symbolic activation guard in one explanation trace.

    Args:
        trace (ResourceTraceNode | None): Trace tree to inspect.

    Returns:
        Sequence[sp.Basic]: Guards in deterministic pre-order.

    Raises:
        ValueError: If the trace exceeds the supported node limit.
    """
    if trace is None:
        return ()
    pending = [trace]
    guards: list[sp.Basic] = []
    index = 0
    while index < len(pending):
        if len(pending) > _MAX_TRACE_NODES:
            raise ValueError(f"resource trace exceeds {_MAX_TRACE_NODES} nodes")
        node = pending[index]
        guards.append(node.active_when)
        pending.extend(node.children)
        index += 1
    return guards


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


def _expression_from_wire(payload: Any, label: str) -> sp.Basic:
    """Decode one symbolic expression without evaluating arbitrary Python.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.

    Returns:
        sp.Basic: Reconstructed SymPy expression or Boolean.

    Raises:
        ValueError: If the syntax, constructor, node count, or result type is
            outside the supported closed expression language.
    """
    if not isinstance(payload, str):
        raise ValueError(f"{label} must be a symbolic-expression string")
    try:
        tree = ast.parse(payload, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} has invalid symbolic-expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise ValueError(f"{label} exceeds the symbolic-expression node limit")
    result = _evaluate_sympy_ast(tree.body)
    if not isinstance(result, sp.Basic):
        raise ValueError(f"{label} did not decode to a SymPy expression")
    return result


def _validate_sympy_expression_for_wire(expression: sp.Basic) -> None:
    """Validate an existing SymPy tree before persisting its representation.

    Args:
        expression (sp.Basic): Trusted, already-constructed SymPy expression.

    Raises:
        ValueError: If the expression contains an unsupported constructor or
            exceeds a numeric or structural wire budget.
    """
    for count, node in enumerate(sp.preorder_traversal(expression), start=1):
        if count > _MAX_EXPRESSION_NODES:
            raise ValueError(
                "resource expression exceeds the symbolic-expression node limit"
            )
        constructor_name = _wire_constructor_name(node)
        if constructor_name not in _SAFE_SYMPY_NAMES and constructor_name not in {
            "_CanonicalPhaseClass",
            "_ConditionIndicator",
            "_RangeAny",
            "_CappedRangeSum",
        }:
            raise ValueError(f"unsupported symbolic constructor {constructor_name!r}")
        if isinstance(node, AppliedUndef):
            _validate_undefined_function_constructor([node.func.__name__], {})
        if isinstance(node, sp.Float):
            _validate_float_constructor(
                [str(node)],
                {"precision": node._prec},
            )
        if isinstance(node, sp.Pow):
            _validate_sympy_constructor_call("Pow", list(node.args), {})
        _validate_sympy_numeric_size(node)


def _wire_constructor_name(expression: sp.Basic) -> str:
    """Return the closed-wire constructor name for one SymPy node.

    Args:
        expression (sp.Basic): One node from an expression tree.

    Returns:
        str: Constructor or singleton name emitted by ``sympy.srepr``.
    """
    if isinstance(expression, AppliedUndef):
        return "Function"
    if isinstance(expression, sp.Integer):
        return "Integer"
    if isinstance(expression, sp.Rational):
        return "Rational"
    if isinstance(expression, sp.Float):
        return "Float"
    special_names = (
        (sp.E, "E"),
        (sp.pi, "pi"),
        (sp.nan, "nan"),
        (sp.oo, "oo"),
        (-sp.oo, "oo"),
        (sp.zoo, "zoo"),
    )
    for singleton, name in special_names:
        if expression is singleton:
            return name
    return type(expression).__name__


def _resource_expression_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> sp.Expr:
    """Decode one numeric resource expression.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        sp.Expr: Reconstructed numeric expression.

    Raises:
        ValueError: If the payload decodes to a Boolean or non-expression.
    """
    expression = decoder.decode(payload, label)
    if not isinstance(expression, sp.Expr):
        raise ValueError(f"{label} must decode to a numeric SymPy expression")
    return expression


def _boolean_expression_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> Boolean:
    """Decode one symbolic Boolean guard.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        Boolean: Reconstructed Boolean condition.

    Raises:
        ValueError: If the payload does not decode to a SymPy Boolean.
    """
    expression = decoder.decode(payload, label)
    if not isinstance(expression, Boolean):
        raise ValueError(f"{label} must decode to a SymPy Boolean")
    return expression


def _evaluate_sympy_ast(node: ast.AST) -> Any:
    """Evaluate one validated SymPy-constructor AST node.

    Args:
        node (ast.AST): Expression node parsed from ``sympy.srepr`` output.

    Returns:
        Any: Primitive constructor argument, SymPy constructor, or constructed
            SymPy expression.

    Raises:
        ValueError: If the node uses executable Python syntax, an unsupported
            constructor, or invalid constructor arguments.
    """
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(
            node.value,
            (bool, int, float, str),
        ):
            return node.value
        raise ValueError("symbolic expression contains an unsupported literal")
    if isinstance(node, ast.Name):
        return _sympy_name(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_sympy_ast(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate_sympy_ast(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        operand = _evaluate_sympy_ast(node.operand)
        if not isinstance(operand, (int, float, sp.Expr)):
            raise ValueError("symbolic unary signs require a numeric operand")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if not isinstance(node, ast.Call):
        raise ValueError(
            "symbolic expression contains syntax outside safe constructors"
        )
    constructor = _evaluate_sympy_ast(node.func)
    if not _is_sympy_constructor(constructor):
        raise ValueError("symbolic expression tried to call a non-constructor")
    if any(keyword.arg is None for keyword in node.keywords):
        raise ValueError("symbolic expression cannot expand keyword mappings")
    args = [_evaluate_sympy_ast(item) for item in node.args]
    kwargs = {
        cast(str, keyword.arg): _evaluate_sympy_ast(keyword.value)
        for keyword in node.keywords
    }
    constructor_name = (
        node.func.id if isinstance(node.func, ast.Name) else "_AppliedUndefinedFunction"
    )
    _validate_sympy_constructor_call(constructor_name, args, kwargs)
    if constructor_name == "_CappedRangeSum":
        kwargs = {**kwargs, "evaluate": False}
    try:
        result = constructor(*args, **kwargs)
    except (TypeError, ValueError, sp.SympifyError) as exc:
        raise ValueError(
            "symbolic expression constructor arguments are invalid"
        ) from exc
    if not isinstance(result, (sp.Basic, FunctionClass)):
        raise ValueError("symbolic constructor produced an unsupported result")
    if isinstance(result, sp.Basic):
        _validate_sympy_numeric_size(result)
    return result


def _validate_sympy_constructor_call(
    name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Reject constructor inputs that can trigger unbounded eager arithmetic.

    Args:
        name (str): Validated SymPy constructor name.
        args (list[Any]): Recursively decoded positional arguments.
        kwargs (dict[str, Any]): Recursively decoded keyword arguments.

    Raises:
        ValueError: If a numeric constructor request exceeds the wire budget.
    """
    if name == "Float":
        _validate_float_constructor(args, kwargs)
    if name == "Function":
        _validate_undefined_function_constructor(args, kwargs)
    if name == "_AppliedUndefinedFunction":
        if kwargs or not all(isinstance(arg, sp.Basic) for arg in args):
            raise ValueError(
                "symbolic applied functions require SymPy positional arguments"
            )
    if name == "_CappedRangeSum" and (len(args) != 4 or kwargs):
        raise ValueError("symbolic _CappedRangeSum requires four positional arguments")
    if name != "Pow" or len(args) < 2:
        return
    base, exponent = args[:2]
    if not isinstance(exponent, sp.Integer):
        return
    exponent_value = int(exponent)
    if not isinstance(base, (sp.Integer, sp.Rational)):
        return
    if base in (sp.Integer(-1), sp.Integer(0), sp.Integer(1)):
        return
    magnitude = max(
        abs(int(base.p)).bit_length(),
        abs(int(base.q)).bit_length(),
    )
    if magnitude * max(1, abs(exponent_value)) > _MAX_NUMERIC_BITS:
        raise ValueError("symbolic numeric power exceeds the wire budget")


def _validate_undefined_function_constructor(
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Restrict undefined functions to canonical bounded identifiers.

    Args:
        args (list[Any]): Function-factory positional arguments.
        kwargs (dict[str, Any]): Function-factory keyword arguments.

    Raises:
        ValueError: If the factory request is not ``Function(identifier)``.
    """
    if len(args) != 1 or kwargs or not isinstance(args[0], str):
        raise ValueError("symbolic Function requires one identifier string")
    if re.fullmatch(r"[A-Za-z_]\w{0,255}", args[0], flags=re.ASCII) is None:
        raise ValueError("symbolic Function name must be a bounded identifier")


def _validate_float_constructor(args: list[Any], kwargs: dict[str, Any]) -> None:
    """Reject Float payloads whose parsing work exceeds the wire budget.

    The encoder's structural representation always uses one decimal string
    plus an optional ``precision`` keyword. Restricting the decoder to that
    canonical shape prevents positional ``dps`` or enormous exponent text from
    triggering expensive arbitrary-precision construction. A compact exponent
    may describe an arbitrarily small or large finite value without requiring
    a proportionally large mantissa, so its magnitude is not charged as
    decimal digits.

    Args:
        args (list[Any]): Recursively decoded Float positional arguments.
        kwargs (dict[str, Any]): Recursively decoded Float keyword arguments.

    Raises:
        ValueError: If the Float is noncanonical or its representation exceeds
            the wire budget.
    """
    if len(args) != 1 or not isinstance(args[0], str):
        raise ValueError("symbolic Float requires one decimal string")
    if set(kwargs) - {"precision"}:
        raise ValueError("symbolic Float contains unsupported keyword arguments")
    precision = kwargs.get("precision", 53)
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise ValueError("symbolic Float precision must be an integer")
    if precision < 1 or precision > _MAX_NUMERIC_BITS:
        raise ValueError("symbolic Float precision exceeds the wire budget")

    literal = args[0]
    if len(literal) > _MAX_DECIMAL_FLOAT_DIGITS + 16:
        raise ValueError("symbolic Float literal exceeds the wire budget")
    match = re.fullmatch(
        r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE]([+-]?\d+))?",
        literal,
    )
    if match is None:
        raise ValueError("symbolic Float literal is malformed")
    digits = sum(
        character.isdigit() for character in literal.split("e")[0].split("E")[0]
    )
    exponent_text = match.group(1)
    if exponent_text is not None:
        exponent_digits = exponent_text.lstrip("+-")
        if len(exponent_digits) > _MAX_DECIMAL_FLOAT_EXPONENT_DIGITS:
            raise ValueError("symbolic Float exponent exceeds the wire budget")
    if digits > _MAX_DECIMAL_FLOAT_DIGITS:
        raise ValueError("symbolic Float mantissa exceeds the wire budget")


def _validate_sympy_numeric_size(expression: sp.Basic) -> None:
    """Reject an exact numeric result that exceeds the wire arithmetic budget.

    Args:
        expression (sp.Basic): Newly constructed SymPy expression or number.

    Raises:
        ValueError: If an exact rational result exceeds the bit limit.
    """
    if not isinstance(expression, sp.Rational):
        return
    if (
        max(
            abs(int(expression.p)).bit_length(),
            abs(int(expression.q)).bit_length(),
        )
        > _MAX_NUMERIC_BITS
    ):
        raise ValueError("symbolic numeric value exceeds the wire budget")


def _sympy_name(name: str) -> Any:
    """Resolve one safe constructor or symbolic constant by name.

    Args:
        name (str): Name emitted by ``sympy.srepr``.

    Returns:
        Any: SymPy ``Basic`` constant or expression constructor.

    Raises:
        ValueError: If the name is not a safe SymPy expression constructor.
    """
    if name == "ExprCondPair":
        return ExprCondPair
    if name == "_CanonicalPhaseClass":
        from qamomile.circuit.estimator.resource_estimator import (
            _CanonicalPhaseClass,
        )

        return _CanonicalPhaseClass
    if name == "_ConditionIndicator":
        from qamomile.circuit.estimator._metrics import _ConditionIndicator

        return _ConditionIndicator
    if name == "_RangeAny":
        from qamomile.circuit.estimator._metrics import _RangeAny

        return _RangeAny
    if name == "_CappedRangeSum":
        from qamomile.circuit.estimator.resource_estimator import (
            _CappedRangeSum,
        )

        return _CappedRangeSum
    if name not in _SAFE_SYMPY_NAMES:
        raise ValueError(f"unsupported symbolic constructor {name!r}")
    candidate = getattr(sp, name, None)
    if isinstance(candidate, sp.Basic) or _is_sympy_constructor(candidate):
        return candidate
    raise ValueError(f"unsupported symbolic constructor {name!r}")


def _is_sympy_constructor(value: Any) -> bool:
    """Return whether a value is a safe SymPy expression constructor.

    Args:
        value (Any): Candidate object.

    Returns:
        bool: Whether the object can only construct ``sympy.Basic`` values.
    """
    return isinstance(value, FunctionClass) or (
        isinstance(value, type) and issubclass(value, sp.Basic)
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


def _precision_from_wire(payload: Any) -> float | None:
    """Decode optional rotation-synthesis precision.

    Args:
        payload (Any): Serialized precision.

    Returns:
        float | None: Reconstructed precision.

    Raises:
        ValueError: If the value is not a numeric scalar or ``None``.
    """
    if payload is None:
        return None
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError("opaque ResourceEstimate precision must be numeric or None")
    return float(payload)
