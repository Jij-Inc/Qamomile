"""Typed native persistence bridge for symbolic resource studies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from google.protobuf.message import DecodeError

from qamomile._native import compiler as native
from qamomile.errors import CapabilityError, Diagnostic
from qamomile.proto import resource_analysis_pb2 as wire

from ._rule_wire import (
    _AUTHORITIES,
    _COMPOSITIONS,
    _COORDINATES,
    _DOMAINS,
    _ExpressionEncoder,
    decode_resource_expression_graph,
)
from .bundle import AnalysisBundle, AnalysisFact
from .expr import Expr, Unit
from .metrics import MetricDefinition
from .report import ResourceStudy
from .values import Evidence, Relation, ResourceValue

RESOURCE_STUDY_SCHEMA_MAJOR = 1
RESOURCE_STUDY_SCHEMA_MINOR = 0
_MAX_BYTES = 64 * 1024 * 1024
_DOMAIN_VALUES = {value: key for key, value in _DOMAINS.items()}
_COORDINATE_VALUES = {value: key for key, value in _COORDINATES.items()}
_COMPOSITION_VALUES = {value: key for key, value in _COMPOSITIONS.items()}
_RELATIONS = {
    Relation.EQUALITY: wire.RESOURCE_EQUALITY,
    Relation.LOWER_BOUND: wire.RESOURCE_LOWER_BOUND,
    Relation.UPPER_BOUND: wire.RESOURCE_UPPER_BOUND,
    Relation.ENCLOSURE: wire.RESOURCE_ENCLOSURE,
    Relation.PARTIAL_ENCLOSURE: wire.RESOURCE_PARTIAL_ENCLOSURE,
    Relation.UNKNOWN: wire.RESOURCE_UNKNOWN,
}
_RELATION_VALUES = {value: key for key, value in _RELATIONS.items()}
_AUTHORITY_VALUES = {value: key for key, value in _AUTHORITIES.items()}
_EnumKey = TypeVar("_EnumKey", bound=int)
_EnumValue = TypeVar("_EnumValue")


def encode_study(study: ResourceStudy, *, include_digest: bool = True) -> bytes:
    """Encode one study as deterministic typed Protobuf.

    Args:
        study: Immutable public study to encode.
        include_digest: Whether Rust should integrity-seal the canonical bytes.

    Returns:
        Canonical unsealed or integrity-sealed Protobuf bytes.

    Raises:
        CapabilityError: If the required native sealer is unavailable.
        TypeError: If the native extension violates its return ABI.
        ValueError: If Rust rejects the projected study.
    """

    encoded = wire.ResourceStudyProto(
        schema_major=RESOURCE_STUDY_SCHEMA_MAJOR,
        schema_minor=RESOURCE_STUDY_SCHEMA_MINOR,
        bundle=_encode_bundle(study._bundle),
        derivations=study.derivations,
    )
    for identifier, value in study.metrics:
        item = encoded.metrics.add(identifier=identifier)
        _encode_value(value, item.value)
    payload = encoded.SerializeToString(deterministic=True)
    if not include_digest:
        return payload
    sealer = getattr(native, "seal_resource_study", None)
    if not callable(sealer):
        raise CapabilityError(
            Diagnostic(
                "Q4214", "the native compiler does not provide seal_resource_study"
            )
        )
    result = sealer(payload)
    if not isinstance(result, bytes):
        raise TypeError("Q4257: native resource-study sealer returned malformed data")
    return result


def decode_study(payload: bytes) -> ResourceStudy:
    """Verify and decode one native-owned resource study.

    Args:
        payload: Integrity-sealed canonical Protobuf bytes.

    Returns:
        Reconstructed immutable public resource study.

    Raises:
        CapabilityError: If the required native inspector is unavailable.
        TypeError: If the native extension violates its return ABI.
        ValueError: If the study is malformed or fails native verification.
    """

    _inspect_digest(payload)
    if len(payload) > _MAX_BYTES:
        raise ValueError("resource study exceeds the wire-size limit")
    try:
        encoded = wire.ResourceStudyProto.FromString(payload)
    except DecodeError as error:
        raise ValueError("malformed resource study encoding") from error
    if encoded.SerializeToString(deterministic=True) != payload:
        raise TypeError("Q4257: native inspector accepted noncanonical study bytes")
    if not encoded.HasField("bundle"):
        raise TypeError("Q4257: native inspector accepted a study without a bundle")
    metrics = tuple(
        (item.identifier, _decode_value(item.value)) for item in encoded.metrics
    )
    bundle = _decode_bundle(encoded.bundle, dict(metrics))
    return ResourceStudy(metrics, bundle, tuple(encoded.derivations))


def study_digest(payload: bytes) -> str:
    """Return the Rust-verified digest of one sealed resource study."""

    return _inspect_digest(payload).hex()


def _inspect_digest(payload: bytes) -> bytes:
    """Invoke the mandatory native inspector and validate its digest ABI."""

    inspect = getattr(native, "inspect_resource_study", None)
    if not callable(inspect):
        raise CapabilityError(
            Diagnostic(
                "Q4214", "the native compiler does not provide inspect_resource_study"
            )
        )
    digest = inspect(payload)
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise TypeError(
            "Q4257: native resource-study inspector returned malformed data"
        )
    return digest


def _encode_bundle(bundle: AnalysisBundle) -> wire.ResourceAnalysisBundleProto:
    """Project one immutable analysis bundle to its native wire record."""

    encoded = wire.ResourceAnalysisBundleProto(
        entrypoint=bundle.entrypoint,
        analysis_version=bundle.analysis_version,
        qcore_payload=bundle.qcore_payload,
    )
    for fact in bundle.facts:
        target = encoded.facts.add(
            kind=fact.kind,
            owner=fact.owner,
            source=fact.source,
        )
        target.operation.opcode = fact.operation
        for identifier, value in fact.contributions:
            _encode_contribution(
                identifier, value, target.operation.contributions.add()
            )
    return encoded


def _decode_bundle(
    encoded: wire.ResourceAnalysisBundleProto,
    metrics: Mapping[str, ResourceValue],
) -> AnalysisBundle:
    """Reconstruct one native-verified analysis bundle."""

    return AnalysisBundle(
        encoded.entrypoint,
        encoded.analysis_version,
        bytes(encoded.qcore_payload),
        tuple(_decode_typed_fact(fact, metrics) for fact in encoded.facts),
    )


def _encode_contribution(
    identifier: str,
    value: ResourceValue,
    target: wire.ResourceContributionProto,
) -> None:
    """Encode one typed source-local metric contribution."""

    target.identifier = identifier
    target.relation = _RELATIONS[value.relation]
    for expression, graph in (
        (value.expression, target.expression),
        (value.lower, target.lower),
        (value.upper, target.upper),
    ):
        if expression is not None:
            _ExpressionEncoder(graph).encode(expression)
    target.unknown_contributors.extend(value.unknown_contributors)


def _decode_typed_fact(
    encoded: wire.ResourceAnalysisFactProto,
    metrics: Mapping[str, ResourceValue],
) -> AnalysisFact:
    """Restore one typed operation fact already verified by Rust."""

    if encoded.WhichOneof("value") != "operation":
        raise TypeError("Q4257: native study fact has no typed operation value")
    operation = encoded.operation
    contributions: list[tuple[str, ResourceValue]] = []
    for item in operation.contributions:
        value = metrics.get(item.identifier)
        if value is None:
            raise TypeError("Q4257: native study fact names an unknown metric")
        contributions.append(
            (
                item.identifier,
                _decode_contribution(item, value.metric),
            )
        )
    return AnalysisFact(
        encoded.kind,
        encoded.owner,
        encoded.source,
        operation.opcode,
        tuple(contributions),
    )


def _decode_contribution(
    encoded: wire.ResourceContributionProto,
    metric: MetricDefinition,
) -> ResourceValue:
    """Restore one typed source-local contribution already verified by Rust."""

    return ResourceValue(
        metric,
        _enum(_RELATION_VALUES, encoded.relation, "contribution relation"),
        _decode_optional_contribution_expression(encoded, "expression"),
        _decode_optional_contribution_expression(encoded, "lower"),
        _decode_optional_contribution_expression(encoded, "upper"),
        tuple(encoded.unknown_contributors),
    )


def _decode_optional_contribution_expression(
    encoded: wire.ResourceContributionProto,
    field: str,
) -> Expr | None:
    """Decode one optional source-local endpoint already verified by Rust."""

    if not encoded.HasField(field):
        return None
    value = decode_resource_expression_graph(
        getattr(encoded, field).SerializeToString(deterministic=True)
    )
    if not isinstance(value, Expr):
        raise TypeError("Q4257: native study contribution is not an expression")
    return value


def _encode_value(value: ResourceValue, target: wire.ResourceValueProto) -> None:
    """Project one complete mathematical resource value."""

    _encode_metric(value.metric, target.metric)
    target.relation = _RELATIONS[value.relation]
    for expression, graph in (
        (value.expression, target.expression),
        (value.lower, target.lower),
        (value.upper, target.upper),
    ):
        if expression is not None:
            _ExpressionEncoder(graph).encode(expression)
    target.unknown_contributors.extend(value.unknown_contributors)
    target.assumptions.extend(value.assumptions)
    for evidence in value.evidence:
        encoded = target.evidence.add(
            authority=_AUTHORITIES[evidence.authority],
            reference=evidence.reference,
        )
        for name, detail in evidence.details:
            encoded.details.add(name=name, value=detail)


def _decode_value(encoded: wire.ResourceValueProto) -> ResourceValue:
    """Reconstruct one native-verified mathematical resource value."""

    metric = _decode_metric(encoded.metric)
    relation = _enum(_RELATION_VALUES, encoded.relation, "relation")
    return ResourceValue(
        metric,
        relation,
        _decode_optional_expression(encoded, "expression"),
        _decode_optional_expression(encoded, "lower"),
        _decode_optional_expression(encoded, "upper"),
        tuple(encoded.unknown_contributors),
        tuple(
            Evidence(
                _enum(_AUTHORITY_VALUES, item.authority, "evidence authority"),
                item.reference,
                tuple((detail.name, detail.value) for detail in item.details),
            )
            for item in encoded.evidence
        ),
        tuple(encoded.assumptions),
    )


def _decode_optional_expression(
    encoded: wire.ResourceValueProto,
    field: str,
) -> Expr | None:
    """Decode one optional native-verified resource expression graph."""

    if not encoded.HasField(field):
        return None
    graph = getattr(encoded, field)
    value = decode_resource_expression_graph(
        graph.SerializeToString(deterministic=True)
    )
    if not isinstance(value, Expr):
        raise TypeError("Q4257: native study endpoint is not an expression")
    return value


def _encode_metric(
    metric: MetricDefinition,
    target: wire.ResourceMetricProto,
) -> None:
    """Project one complete metric definition."""

    target.identifier = metric.identifier
    target.coordinate = _COORDINATES[metric.coordinate]
    target.domain = _DOMAINS[metric.domain]
    target.unit = metric.unit.name
    target.composition = _COMPOSITIONS[metric.composition]
    target.version = metric.version


def _decode_metric(encoded: wire.ResourceMetricProto) -> MetricDefinition:
    """Reconstruct one native-verified metric definition."""

    return MetricDefinition(
        encoded.identifier,
        _enum(_COORDINATE_VALUES, encoded.coordinate, "coordinate"),
        _enum(_DOMAIN_VALUES, encoded.domain, "domain"),
        Unit._from_native(encoded.unit),
        _enum(_COMPOSITION_VALUES, encoded.composition, "composition"),
        encoded.version,
    )


def _enum(
    mapping: Mapping[_EnumKey, _EnumValue],
    value: _EnumKey,
    name: str,
) -> _EnumValue:
    """Decode one enum already accepted by the native verifier."""

    try:
        return mapping[value]
    except KeyError as error:
        raise TypeError(f"Q4257: native study has an invalid {name}") from error


__all__ = [
    "RESOURCE_STUDY_SCHEMA_MAJOR",
    "RESOURCE_STUDY_SCHEMA_MINOR",
    "decode_study",
    "encode_study",
    "study_digest",
]
