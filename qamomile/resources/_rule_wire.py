"""Canonical Protobuf requests for native external resource-rule analysis."""

from __future__ import annotations

import math
import struct
from fractions import Fraction

from google.protobuf.message import DecodeError

from qamomile._native import compiler as native
from qamomile.errors import CapabilityError, Diagnostic
from qamomile.proto import resource_analysis_pb2 as wire

from .expr import Domain, Expr, Number, Predicate, Symbol, Unit
from .metrics import Composition, Coordinate
from .rules import ResourceRule, TransformMode
from .values import EvidenceAuthority, Relation

RESOURCE_ANALYSIS_SCHEMA_MAJOR = 1
RESOURCE_ANALYSIS_SCHEMA_MINOR = 2
_MAX_EXPRESSION_GRAPH_BYTES = 64 * 1024 * 1024
_MAX_EXPRESSION_NODES = 100_000

_DOMAINS = {
    Domain.NATURAL: wire.RESOURCE_NATURAL,
    Domain.INTEGER: wire.RESOURCE_INTEGER,
    Domain.RATIONAL: wire.RESOURCE_RATIONAL,
    Domain.REAL: wire.RESOURCE_REAL,
    Domain.PROBABILITY: wire.RESOURCE_PROBABILITY,
    Domain.BOOLEAN: wire.RESOURCE_BOOLEAN,
}
_DOMAIN_VALUES = {value: key for key, value in _DOMAINS.items()}
_COORDINATES = {
    Coordinate.ALGORITHMIC: wire.RESOURCE_ALGORITHMIC,
    Coordinate.LOGICAL: wire.RESOURCE_LOGICAL,
    Coordinate.CIRCUIT: wire.RESOURCE_CIRCUIT,
    Coordinate.FAULT_TOLERANT: wire.RESOURCE_FAULT_TOLERANT,
    Coordinate.PHYSICAL: wire.RESOURCE_PHYSICAL,
    Coordinate.EMITTED: wire.RESOURCE_EMITTED,
    Coordinate.OBSERVED: wire.RESOURCE_OBSERVED_EXECUTION,
}
_COMPOSITIONS = {
    Composition.ADDITIVE_WORK: wire.RESOURCE_ADDITIVE_WORK,
    Composition.DEPENDENCY_SPAN: wire.RESOURCE_DEPENDENCY_SPAN,
    Composition.PEAK_LIVENESS: wire.RESOURCE_PEAK_LIVENESS,
    Composition.REPLACEMENT: wire.RESOURCE_REPLACEMENT,
    Composition.ERROR_BOUND: wire.RESOURCE_ERROR_BOUND,
}
_RELATIONS = {
    Relation.EQUALITY: wire.RESOURCE_EQUALITY,
    Relation.LOWER_BOUND: wire.RESOURCE_LOWER_BOUND,
    Relation.UPPER_BOUND: wire.RESOURCE_UPPER_BOUND,
}
_MODES = {
    TransformMode.DIRECT: wire.RESOURCE_DIRECT,
    TransformMode.CONTROLLED: wire.RESOURCE_CONTROLLED,
    TransformMode.ADJOINT: wire.RESOURCE_ADJOINT,
    TransformMode.CONTROLLED_ADJOINT: wire.RESOURCE_CONTROLLED_ADJOINT,
}
_AUTHORITIES = {
    EvidenceAuthority.STRUCTURALLY_DERIVED: wire.RESOURCE_STRUCTURALLY_DERIVED,
    EvidenceAuthority.TRUSTED_AXIOM: wire.RESOURCE_TRUSTED_AXIOM,
    EvidenceAuthority.HEURISTIC_MODEL: wire.RESOURCE_HEURISTIC_MODEL,
}


def encode_resource_analysis_request(
    program_family: bytes,
    rules: tuple[ResourceRule, ...],
) -> bytes:
    """Seal one selected QCore family and its declarative resource contracts.

    Args:
        program_family: Canonical integrity-sealed ``qamomile.analysis`` bytes.
        rules: Validated external callable rules in arbitrary input order.

    Returns:
        Canonical deterministic Protobuf bytes accepted by the Rust analyzer.

    Raises:
        CapabilityError: If the required native sealer is unavailable.
        TypeError: If the native sealer violates its result ABI.
        ValueError: If rule identities are duplicated or the request exceeds a
            resource wire budget.
    """

    ordered = tuple(sorted(rules, key=lambda rule: rule.identity))
    identities = tuple(rule.identity for rule in ordered)
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate resource-rule identity")
    request = wire.ResourceAnalysisRequestProto(
        schema_major=RESOURCE_ANALYSIS_SCHEMA_MAJOR,
        schema_minor=RESOURCE_ANALYSIS_SCHEMA_MINOR,
        program_family=program_family,
    )
    for rule in ordered:
        _encode_rule(rule, request.rules.add())
    sealer = getattr(native, "seal_resource_analysis_request", None)
    if not callable(sealer):
        raise CapabilityError(
            Diagnostic(
                "Q4214",
                "the native compiler does not provide seal_resource_analysis_request",
            )
        )
    payload = sealer(request.SerializeToString(deterministic=True))
    if not isinstance(payload, bytes):
        raise TypeError("Q4237: native resource-request sealer returned malformed data")
    return payload


def _encode_rule(rule: ResourceRule, target: wire.ResourceRuleProto) -> None:
    """Encode one rule with its complete metric and evidence semantics."""

    target.identity = rule.identity
    target.callable = str(rule.callable)
    target.metric.identifier = rule.metric.identifier
    target.metric.coordinate = _COORDINATES[rule.metric.coordinate]
    target.metric.domain = _DOMAINS[rule.metric.domain]
    target.metric.unit = rule.metric.unit.name
    target.metric.composition = _COMPOSITIONS[rule.metric.composition]
    target.metric.version = rule.metric.version
    target.relation = _RELATIONS[rule.relation]
    _ExpressionEncoder(target.expression).encode(rule.expression)
    target.modes.extend(sorted(_MODES[mode] for mode in rule.modes))
    target.assumptions.extend(rule.assumptions)
    target.validity_domain.extend(rule.validity_domain)
    target.authority = _AUTHORITIES[rule.authority]
    target.provenance = rule.provenance
    if rule.scheduling_footprint is not None:
        target.scheduling_footprint.parameters.extend(
            rule.scheduling_footprint.parameters
        )


class _ExpressionEncoder:
    """Hash-cons one expression and its predicates into a topological DAG."""

    def __init__(self, target: wire.ResourceExpressionGraphProto) -> None:
        """Initialize one graph encoder."""

        self.target = target
        self.object_identifiers: dict[tuple[str, int], int] = {}
        self.structural_identifiers: dict[tuple[object, ...], int] = {}

    def encode(self, expression: Expr) -> None:
        """Encode ``expression`` and record its root identifier."""

        self.target.root = self._expression(expression)

    def encode_root(self, value: Expr | Predicate) -> None:
        """Encode an expression or predicate as a standalone graph root."""

        self.target.root = self._intern(value)

    def _expression(self, expression: Expr) -> int:
        """Return the interned identifier of one expression node."""

        return self._intern(expression)

    def _intern(self, root: Expr | Predicate) -> int:
        """Intern one reachable DAG with an explicit depth-independent stack."""

        stack: list[tuple[Expr | Predicate, bool]] = [(root, False)]
        active: set[tuple[str, int]] = set()
        while stack:
            value, expanded = stack.pop()
            object_key = _expression_object_key(value)
            if object_key in self.object_identifiers:
                active.discard(object_key)
                continue
            if expanded:
                if isinstance(value, Expr):
                    self._finish_expression(value, object_key)
                else:
                    self._finish_predicate(value, object_key)
                active.remove(object_key)
                continue
            if object_key in active:
                raise ValueError("cyclic resource expression")
            if isinstance(value, Expr):
                if any(not isinstance(item, Expr) for item in value.arguments) or any(
                    not isinstance(item, Predicate) for item in value.predicates
                ):
                    raise TypeError("resource expression operands have invalid types")
            elif any(not isinstance(item, Expr) for item in value.arguments) or any(
                not isinstance(item, Predicate) for item in value.predicates
            ):
                raise TypeError("resource predicate operands have invalid types")
            active.add(object_key)
            stack.append((value, True))
            dependencies: tuple[Expr | Predicate, ...] = (
                *value.arguments,
                *value.predicates,
            )
            stack.extend((item, False) for item in reversed(dependencies))
        return self.object_identifiers[_expression_object_key(root)]

    def _finish_expression(
        self,
        expression: Expr,
        object_key: tuple[str, int],
    ) -> int:
        """Hash-cons one expression after all of its children are interned."""

        arguments = tuple(
            self.object_identifiers[_expression_object_key(item)]
            for item in expression.arguments
        )
        predicates = tuple(
            self.object_identifiers[_expression_object_key(item)]
            for item in expression.predicates
        )
        key = (
            "expression",
            expression.kind,
            expression.domain.value,
            expression.unit.name,
            arguments,
            _number_identity(expression.value),
            _symbol_identity(expression.symbol),
            tuple(sorted(expression.attributes)),
            predicates,
        )
        known = self.structural_identifiers.get(key)
        if known is not None:
            self.object_identifiers[object_key] = known
            return known
        node = self.target.nodes.add().expression
        node.kind = expression.kind
        node.domain = _DOMAINS[expression.domain]
        node.unit = expression.unit.name
        node.arguments.extend(arguments)
        node.predicates.extend(predicates)
        if expression.value is not None:
            _encode_number(expression.value, node.literal)
        if expression.symbol is not None:
            node.symbol.scope = expression.symbol.scope
            node.symbol.name = expression.symbol.name
            node.symbol.domain = _DOMAINS[expression.symbol.domain]
            node.symbol.unit = expression.symbol.unit.name
            node.symbol.origin = expression.symbol.origin
            node.symbol.constraints.extend(expression.symbol.constraints)
        for name, value in sorted(expression.attributes):
            node.attributes.add(name=name, value=value)
        return self._finish(key, object_key)

    def _finish_predicate(
        self,
        predicate: Predicate,
        object_key: tuple[str, int],
    ) -> int:
        """Hash-cons one predicate after all of its children are interned."""

        arguments = tuple(
            self.object_identifiers[_expression_object_key(item)]
            for item in predicate.arguments
        )
        predicates = tuple(
            self.object_identifiers[_expression_object_key(item)]
            for item in predicate.predicates
        )
        key = ("predicate", predicate.kind, arguments, predicates)
        known = self.structural_identifiers.get(key)
        if known is not None:
            self.object_identifiers[object_key] = known
            return known
        node = self.target.nodes.add().predicate
        node.kind = predicate.kind
        node.arguments.extend(arguments)
        node.predicates.extend(predicates)
        return self._finish(key, object_key)

    def _finish(
        self,
        structural_key: tuple[object, ...],
        object_key: tuple[str, int],
    ) -> int:
        """Assign the just-appended topological node its stable identifier."""

        if len(self.target.nodes) > _MAX_EXPRESSION_NODES:
            raise ValueError("resource expression exceeds the node budget")
        identifier = len(self.target.nodes) - 1
        self.structural_identifiers[structural_key] = identifier
        self.object_identifiers[object_key] = identifier
        return identifier


def _expression_object_key(value: Expr | Predicate) -> tuple[str, int]:
    """Return a type-separated process-local identity for one DAG node."""

    return ("expression" if isinstance(value, Expr) else "predicate", id(value))


def _number_identity(value: Number | None) -> tuple[object, ...] | None:
    """Return the shallow wire identity of one optional numeric literal."""

    if value is None:
        return None
    if isinstance(value, int):
        return ("integer", str(value))
    if isinstance(value, Fraction):
        return ("rational", str(value.numerator), str(value.denominator))
    return ("real", struct.unpack(">Q", struct.pack(">d", value))[0])


def _symbol_identity(symbol: Symbol | None) -> tuple[object, ...] | None:
    """Return the shallow wire identity of one optional symbol declaration."""

    if symbol is None:
        return None
    return (
        symbol.scope,
        symbol.name,
        symbol.domain.value,
        symbol.unit.name,
        symbol.origin,
        symbol.constraints,
    )


def _encode_number(value: Number, target: wire.ResourceNumberProto) -> None:
    """Encode an exact finite numeric literal without host-format ambiguity."""

    if isinstance(value, bool):
        raise TypeError("boolean is not a resource numeric literal")
    if isinstance(value, int):
        target.integer = str(value)
        return
    if isinstance(value, Fraction):
        target.rational.numerator = str(value.numerator)
        target.rational.denominator = str(value.denominator)
        return
    if not math.isfinite(value):
        raise ValueError("resource float must be finite")
    target.real_bits = struct.unpack(">Q", struct.pack(">d", value))[0]


def encode_resource_expression_graph(value: Expr | Predicate) -> bytes:
    """Encode one standalone expression or predicate for native evaluation.

    Args:
        value: Closed portable expression-language root.

    Returns:
        Deterministically encoded topological resource-expression graph.
    """

    graph = wire.ResourceExpressionGraphProto()
    _ExpressionEncoder(graph).encode_root(value)
    return graph.SerializeToString(deterministic=True)


def decode_resource_expression_graph(payload: bytes) -> Expr | Predicate:
    """Decode one native-produced standalone expression graph.

    Args:
        payload: Canonical graph bytes returned by the Rust compiler.
    Returns:
        Native-normalized typed expression or predicate root.

    Raises:
        TypeError: If the native graph violates the bridge ABI.
        ValueError: If the graph contains an invalid expression value.
    """

    if len(payload) > _MAX_EXPRESSION_GRAPH_BYTES:
        raise TypeError("Q4237: native resource-expression graph violates its budget")
    try:
        graph = wire.ResourceExpressionGraphProto.FromString(payload)
    except DecodeError as error:
        raise TypeError("Q4237: malformed native resource-expression graph") from error
    if graph.SerializeToString(deterministic=True) != payload:
        raise TypeError("Q4237: native resource-expression graph is not canonical")
    if not graph.nodes or len(graph.nodes) > _MAX_EXPRESSION_NODES:
        raise TypeError("Q4237: native resource-expression graph violates its budget")
    if graph.root >= len(graph.nodes):
        raise TypeError("Q4237: native resource-expression graph has no valid root")
    decoded: list[Expr | Predicate] = []
    encodings: set[bytes] = set()
    for identifier, raw in enumerate(graph.nodes):
        encoding = raw.SerializeToString(deterministic=True)
        if encoding in encodings:
            raise TypeError(
                "Q4237: native resource-expression graph is not hash-consed"
            )
        encodings.add(encoding)
        selected = raw.WhichOneof("node")
        if selected == "expression":
            node = raw.expression
            arguments = tuple(
                _expression_reference(item, identifier, decoded)
                for item in node.arguments
            )
            predicates = tuple(
                _predicate_reference(item, identifier, decoded)
                for item in node.predicates
            )
            domain = _DOMAIN_VALUES.get(node.domain)
            if domain is None:
                raise TypeError("Q4237: native expression has an invalid domain")
            unit = Unit._from_native(node.unit)
            attributes = tuple((item.name, item.value) for item in node.attributes)
            if node.kind == "literal":
                expression = Expr._from_native(
                    "literal",
                    domain,
                    unit,
                    value=_decode_number(node.literal),
                )
            elif node.kind == "symbol":
                declaration = node.symbol
                symbol_domain = _DOMAIN_VALUES.get(declaration.domain)
                if symbol_domain is None:
                    raise TypeError("Q4237: native symbol has an invalid domain")
                symbol = Symbol._from_native(
                    declaration.scope,
                    declaration.name,
                    symbol_domain,
                    Unit._from_native(declaration.unit),
                    declaration.origin,
                    tuple(declaration.constraints),
                )
                expression = Expr._from_native("symbol", domain, unit, symbol=symbol)
            else:
                expression = Expr._from_native(
                    node.kind,
                    domain,
                    unit,
                    arguments,
                    attributes=attributes,
                    predicates=predicates,
                )
            decoded.append(expression)
        elif selected == "predicate":
            node = raw.predicate
            decoded.append(
                Predicate._from_native(
                    node.kind,
                    tuple(
                        _expression_reference(item, identifier, decoded)
                        for item in node.arguments
                    ),
                    tuple(
                        _predicate_reference(item, identifier, decoded)
                        for item in node.predicates
                    ),
                )
            )
        else:
            raise TypeError("Q4237: native expression node has no constructor")
    return decoded[graph.root]


def _expression_reference(
    reference: int,
    identifier: int,
    decoded: list[Expr | Predicate],
) -> Expr:
    """Resolve one backward native expression reference."""

    if reference >= identifier:
        raise TypeError("Q4237: native expression reference is invalid")
    result = decoded[reference]
    if not isinstance(result, Expr):
        raise TypeError("Q4237: native expression reference is invalid")
    return result


def _predicate_reference(
    reference: int,
    identifier: int,
    decoded: list[Expr | Predicate],
) -> Predicate:
    """Resolve one backward native predicate reference."""

    if reference >= identifier:
        raise TypeError("Q4237: native predicate reference is invalid")
    result = decoded[reference]
    if not isinstance(result, Predicate):
        raise TypeError("Q4237: native predicate reference is invalid")
    return result


def _decode_number(value: wire.ResourceNumberProto) -> Number:
    """Decode one exact native resource-number payload."""

    selected = value.WhichOneof("value")
    if selected == "integer":
        result = int(value.integer)
        if str(result) != value.integer:
            raise TypeError("Q4237: native integer literal is not canonical")
        return result
    if selected == "rational":
        result = Fraction(
            int(value.rational.numerator),
            int(value.rational.denominator),
        )
        if (
            str(result.numerator) != value.rational.numerator
            or str(result.denominator) != value.rational.denominator
        ):
            raise TypeError("Q4237: native rational literal is not canonical")
        return result
    if selected == "real_bits":
        result = struct.unpack(">d", struct.pack(">Q", value.real_bits))[0]
        if not math.isfinite(result):
            raise TypeError("Q4237: native real literal is not finite")
        return result
    raise TypeError("Q4237: native literal has no numeric value")


__all__ = [
    "RESOURCE_ANALYSIS_SCHEMA_MAJOR",
    "RESOURCE_ANALYSIS_SCHEMA_MINOR",
    "decode_resource_expression_graph",
    "encode_resource_expression_graph",
    "encode_resource_analysis_request",
]
