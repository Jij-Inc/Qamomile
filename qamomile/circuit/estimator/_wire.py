"""Encode fixed resource estimates for semantic IR serialization.

The user-facing :meth:`ResourceEstimate.to_dict` representation is intended
for reports.  Opaque callable costs need a stronger round-trip contract:
symbol domains, guarded provenance, requirements, and explanation traces must
remain usable after a qkernel is deserialized.  This module provides that
closed internal representation without serializing Python callback objects.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._estimate_domain import (
    _apply_domain_rewrite,
    _PublicResourceSnapshot,
    _validate_domain_rewrite_state,
)
from qamomile.circuit.estimator._parameter_domain import _DomainRewritePolicy
from qamomile.circuit.estimator._resource_base import (
    ApproximationStatus,
    ControlDecomposition,
    EstimateDerivation,
    EstimateQuality,
)
from qamomile.circuit.estimator._resource_types import (
    DepthResources,
    GateResources,
    MeasurementResources,
    ResetResources,
    WidthResources,
    _active_quality,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.estimator._symbol_discovery import _serialization_expressions
from qamomile.circuit.estimator._wire_expression import (
    _boolean_expression_from_wire,
    _WireExpressionDecoder,
    _WireExpressionEncoder,
)
from qamomile.circuit.estimator._wire_records import (
    _assumption_from_wire,
    _assumption_to_wire,
    _calls_from_wire,
    _calls_to_wire,
    _domain_state_from_wire,
    _domain_state_to_wire,
    _enum_from_wire,
    _guarded_approximation_from_wire,
    _guarded_assumption_from_wire,
    _guarded_derivation_from_wire,
    _guarded_quality_from_wire,
    _mapping,
    _metric_from_wire,
    _metric_to_wire,
    _public_snapshot_from_resources,
    _require_fields,
    _requirement_from_wire,
    _requirement_to_wire,
    _sequence,
    _trace_from_wire,
    _trace_to_wire,
)

_RESOURCE_ESTIMATE_WIRE_FIELDS = {
    "$type",
    "width",
    "gates",
    "depth",
    "measurements",
    "resets",
    "calls",
    "assumptions",
    "trace",
    "symbol_aliases",
    "parameters",
    "derivation",
    "quality",
    "approximation",
    "control_decomposition",
    "global_barrier_condition",
    "requirements",
    "domain_rewrite_policy",
    "domain_rewrite_state",
    "provenance",
}


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
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    if not isinstance(estimate, ResourceEstimate):
        raise TypeError(
            "opaque_cost serialization requires a fixed ResourceEstimate; "
            f"got {type(estimate).__name__}"
        )
    _validate_domain_rewrite_state(estimate)
    if estimate._output_sizes or estimate._input_sizes or estimate._has_output_summary:
        raise ValueError(
            "opaque_cost serialization does not support caller-scoped "
            "input/output liveness summaries"
        )
    registry = SymbolRegistry.from_expressions(
        _serialization_expressions(estimate),
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
        "width": _metric_to_wire(estimate.width, encoder),
        "gates": _metric_to_wire(estimate.gates, encoder),
        "depth": _metric_to_wire(estimate.depth, encoder),
        "measurements": _metric_to_wire(estimate.measurements, encoder),
        "resets": _metric_to_wire(estimate.resets, encoder),
        "calls": _calls_to_wire(estimate.calls, encoder),
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
        "control_decomposition": estimate.control_decomposition.value,
        "global_barrier_condition": expression(estimate._global_barrier_condition),
        "requirements": [
            _requirement_to_wire(constraint, encoder)
            for constraint in estimate._constraints
        ],
        "domain_rewrite_policy": estimate._domain_rewrite_policy.value,
        "domain_rewrite_state": _domain_state_to_wire(
            estimate._domain_rewrite_state,
            encoder,
        ),
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
                    "reason": _assumption_to_wire(fact.reason),
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
    record = _mapping(payload, "opaque ResourceEstimate")
    if record.get("$type") != "ResourceEstimate":
        raise ValueError("opaque cost payload is not a ResourceEstimate")
    _require_fields(
        record,
        _RESOURCE_ESTIMATE_WIRE_FIELDS,
        "opaque ResourceEstimate",
    )

    decoder = _WireExpressionDecoder() if decoder is None else decoder
    provenance = _mapping(
        record.get("provenance"),
        "opaque ResourceEstimate provenance",
    )
    _require_fields(
        provenance,
        {"assumptions", "derivations", "qualities", "approximations"},
        "opaque ResourceEstimate provenance",
    )
    serialized_assumptions = tuple(
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
    width = _metric_from_wire(
        record.get("width"),
        WidthResources,
        "opaque ResourceEstimate width",
        decoder,
    )
    gates = _metric_from_wire(
        record.get("gates"),
        GateResources,
        "opaque ResourceEstimate gates",
        decoder,
    )
    depth = _metric_from_wire(
        record.get("depth"),
        DepthResources,
        "opaque ResourceEstimate depth",
        decoder,
    )
    measurements = _metric_from_wire(
        record.get("measurements"),
        MeasurementResources,
        "opaque ResourceEstimate measurements",
        decoder,
    )
    resets = _metric_from_wire(
        record.get("resets"),
        ResetResources,
        "opaque ResourceEstimate resets",
        decoder,
    )
    calls = _calls_from_wire(
        record.get("calls"),
        "opaque ResourceEstimate calls",
        decoder,
    )
    visible_snapshot = _public_snapshot_from_resources(
        width,
        gates,
        measurements,
        resets,
        depth,
        calls,
    )
    serialized_state = _domain_state_from_wire(
        record.get("domain_rewrite_state"),
        decoder,
    )
    policy = _enum_from_wire(
        _DomainRewritePolicy,
        record.get("domain_rewrite_policy"),
        "opaque ResourceEstimate domain rewrite policy",
    )
    serialized_derivation = _enum_from_wire(
        EstimateDerivation,
        record.get("derivation"),
        "opaque ResourceEstimate derivation",
    )
    serialized_quality = _enum_from_wire(
        EstimateQuality,
        record.get("quality"),
        "opaque ResourceEstimate quality",
    )
    if _active_quality(guarded_qualities) is not serialized_quality:
        raise ValueError(
            "opaque ResourceEstimate public metadata disagrees with canonical "
            "guarded provenance"
        )
    serialized_approximation = _enum_from_wire(
        ApproximationStatus,
        record.get("approximation"),
        "opaque ResourceEstimate approximation",
    )
    original_snapshot = (
        serialized_state.original if serialized_state is not None else visible_snapshot
    )
    estimate = ResourceEstimate(
        width=original_snapshot.width_resources(),
        gates=original_snapshot.gate_resources(),
        measurements=original_snapshot.measurement_resources(),
        resets=original_snapshot.reset_resources(),
        depth=original_snapshot.depth_resources(),
        calls=original_snapshot.call_resources(),
        assumptions=(),
        trace=_trace_from_wire(record.get("trace"), decoder),
        derivation=serialized_derivation,
        quality=serialized_quality,
        approximation=serialized_approximation,
        control_decomposition=_enum_from_wire(
            ControlDecomposition,
            record.get("control_decomposition"),
            "opaque ResourceEstimate control decomposition",
        ),
        _global_barrier_condition=_boolean_expression_from_wire(
            record.get("global_barrier_condition"),
            "opaque ResourceEstimate global barrier condition",
            decoder,
        ),
        _constraints=requirements,
        _guarded_assumptions=guarded_assumptions,
        _guarded_derivations=guarded_derivations,
        _guarded_qualities=guarded_qualities,
        _guarded_approximations=guarded_approximations,
        _domain_rewrite_policy=policy,
    )
    estimate = _apply_domain_rewrite(estimate, policy=policy)
    _validate_domain_rewrite_state(estimate)
    if _PublicResourceSnapshot.capture(estimate) != visible_snapshot:
        raise ValueError(
            "opaque ResourceEstimate visible metrics disagree with its "
            "canonical domain rewrite"
        )
    if estimate._domain_rewrite_state != serialized_state:
        raise ValueError(
            "opaque ResourceEstimate domain rewrite state disagrees with "
            "canonical proof evidence"
        )
    if (
        estimate._guarded_derivations != guarded_derivations
        or estimate._guarded_qualities != guarded_qualities
        or estimate._guarded_approximations != guarded_approximations
        or estimate.derivation is not serialized_derivation
        or estimate.quality is not serialized_quality
        or estimate.approximation is not serialized_approximation
    ):
        raise ValueError(
            "opaque ResourceEstimate public metadata disagrees with canonical "
            "guarded provenance"
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
    if estimate.assumptions != serialized_assumptions:
        raise ValueError(
            "opaque ResourceEstimate assumptions disagree with canonical provenance"
        )
    _validate_domain_rewrite_state(estimate)
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
    available: set[sp.Symbol] = set()
    for expression in _serialization_expressions(estimate):
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
