"""Declarative semantic catalog for QCore primitive operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ._native import compiler as native
from .effects import Effect, Requirement
from .types import TypeSpec, as_type


@dataclass(frozen=True, slots=True)
class GateSpec:
    """The backend-neutral facts shared by all primitive-gate consumers."""

    name: str
    selectors: tuple[str, ...]
    selector_defaults: tuple[int, ...]
    families: frozenset[str]
    has_angle: bool = False

    @property
    def opcode(self) -> str:
        """Return the canonical QCore opcode."""

        return f"quantum.{self.name}"

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return source parameters after the owned state argument."""

        return (("angle",) if self.has_angle else ()) + self.selectors

    @property
    def default_indices(self) -> tuple[tuple[str, int], ...]:
        """Return the default logical selector positions."""

        return tuple(zip(self.selectors, self.selector_defaults, strict=True))

    @property
    def broadcasts_when_unselected(self) -> bool:
        """Return whether omission applies the gate to every logical lane.

        Returns:
            bool: True when this primitive has one selector and therefore
            broadcasts when that selector is omitted.
        """

        return len(self.selectors) == 1


_CatalogRow = tuple[
    str,
    str,
    bool,
    frozenset[Effect],
    frozenset[Requirement],
    GateSpec | None,
]


def _load_native_catalog() -> tuple[tuple[_CatalogRow, ...], frozenset[str]]:
    """Load and validate the compiler-owned operation vocabulary."""

    project = getattr(native, "semantic_operation_catalog", None)
    if not callable(project):
        raise RuntimeError("Q4501: native operation catalog is unavailable")
    try:
        projection = project()
        if not isinstance(projection, tuple) or len(projection) != 2:
            raise TypeError
        raw_operations, raw_terminators = projection
        if not isinstance(raw_operations, list) or not isinstance(
            raw_terminators, list
        ):
            raise TypeError
        rows: list[_CatalogRow] = []
        seen: set[str] = set()
        for raw in raw_operations:
            if not isinstance(raw, tuple) or len(raw) != 6:
                raise TypeError
            opcode, dialect, structured, raw_effects, raw_requirements, raw_gate = raw
            if (
                not isinstance(opcode, str)
                or not opcode
                or opcode in seen
                or dialect not in {"qcore", "classical", "quantum"}
                or type(structured) is not bool
                or not isinstance(raw_effects, list)
                or not isinstance(raw_requirements, list)
                or any(not isinstance(value, str) for value in raw_effects)
                or any(not isinstance(value, str) for value in raw_requirements)
                or len(set(raw_effects)) != len(raw_effects)
                or len(set(raw_requirements)) != len(raw_requirements)
            ):
                raise TypeError
            gate = None
            if raw_gate is not None:
                if not isinstance(raw_gate, tuple) or len(raw_gate) != 4:
                    raise TypeError
                name, raw_selectors, raw_families, has_angle = raw_gate
                if not isinstance(raw_selectors, list):
                    raise TypeError
                selectors: list[str] = []
                selector_defaults: list[int] = []
                for raw_selector in raw_selectors:
                    if (
                        not isinstance(raw_selector, tuple)
                        or len(raw_selector) != 2
                        or not isinstance(raw_selector[0], str)
                        or not raw_selector[0]
                        or type(raw_selector[1]) is not int
                        or raw_selector[1] < 0
                    ):
                        raise TypeError
                    selectors.append(raw_selector[0])
                    selector_defaults.append(raw_selector[1])
                if (
                    not isinstance(name, str)
                    or opcode != f"quantum.{name}"
                    or dialect != "quantum"
                    or not selectors
                    or len(set(selectors)) != len(selectors)
                    or len(set(selector_defaults)) != len(selector_defaults)
                    or not isinstance(raw_families, list)
                    or raw_families != sorted(set(raw_families))
                    or "gates" not in raw_families
                    or type(has_angle) is not bool
                ):
                    raise TypeError
                gate = GateSpec(
                    name,
                    tuple(selectors),
                    tuple(selector_defaults),
                    frozenset(raw_families),
                    has_angle,
                )
            rows.append(
                (
                    opcode,
                    dialect,
                    structured,
                    frozenset(Effect(value) for value in raw_effects),
                    frozenset(Requirement(value) for value in raw_requirements),
                    gate,
                )
            )
            seen.add(opcode)
        if (
            not rows
            or any(not isinstance(value, str) or not value for value in raw_terminators)
            or len(set(raw_terminators)) != len(raw_terminators)
        ):
            raise TypeError
        return tuple(rows), frozenset(raw_terminators)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Q4501: native operation catalog is malformed") from error


def evaluate_classical(
    opcode: str,
    operands: tuple[object, ...],
    *,
    attributes: Mapping[str, object] | Iterable[tuple[str, object]] = (),
) -> object:
    """Evaluate one pure classical operation through the Rust value algebra.

    Args:
        opcode: Closed QCore classical opcode.
        operands: Exact runtime operands in semantic order.
        attributes: Operation metadata in canonical or mapping form.

    Returns:
        The exact portable result restored to an ordinary Python value.

    Raises:
        TypeError: If the native compiler boundary is unavailable.
        ValueError: If an operand is non-portable or Rust rejects the operation.
    """

    from .ir import freeze_value, thaw_value
    from .proto import qcore_pb2 as wire
    from .serialization import decode_literal, encode_literal

    evaluate = getattr(native, "evaluate_classical", None)
    if not callable(evaluate):
        raise TypeError("Q4500: native classical evaluator is unavailable")

    def encode(value: object) -> bytes:
        """Encode one exact runtime value for the native boundary.

        Args:
            value: Python value from a verified classical operation.

        Returns:
            Canonical standalone ``LiteralProto`` bytes.

        Raises:
            TypeError: If the value is outside the portable literal domain.
        """

        literal = wire.LiteralProto()
        encode_literal(freeze_value(value), literal)
        return literal.SerializeToString(deterministic=True)

    entries = (
        tuple(attributes.items())
        if isinstance(attributes, Mapping)
        else tuple(attributes)
    )
    encoded = evaluate(
        opcode,
        tuple(encode(value) for value in operands),
        tuple((name, encode(value)) for name, value in sorted(entries)),
    )
    literal = wire.LiteralProto()
    literal.ParseFromString(bytes(encoded))
    return thaw_value(decode_literal(literal))


def infer_classical_result_type(
    opcode: str,
    operands: tuple[TypeSpec, ...],
    *,
    attributes: Mapping[str, object] | Iterable[tuple[str, object]] = (),
) -> TypeSpec:
    """Ask Rust to derive one classical operation's unique result type.

    Args:
        opcode: Closed classical opcode, including structural ``array``.
        operands: Operand types in semantic order.
        attributes: Operation metadata used by typed projections and casts.

    Returns:
        The uniquely derived staged result type.

    Raises:
        TypeError: If the native compiler boundary is unavailable.
        ValueError: If Rust rejects the opcode, arity, attributes, or operand
            types.
    """

    from google.protobuf.message import DecodeError

    from .errors import SerializationError
    from .ir import freeze_value
    from .proto import qcore_pb2 as wire
    from .serialization import decode_type, encode_literal, encode_type

    infer = getattr(native, "infer_classical_result_type", None)
    if not callable(infer):
        raise TypeError("Q4505: native classical type inference is unavailable")

    def encode_operand(type_spec: TypeSpec) -> bytes:
        """Encode one source type for the native local equation.

        Args:
            type_spec (TypeSpec): Source operand type to normalize and encode.

        Returns:
            bytes: Canonical persistent type payload.

        Raises:
            SerializationError: If the operand type is not serializable.
            TypeCheckError: If the operand is not a logical Qamomile type.
        """

        encoded = wire.TypeProto()
        encode_type(as_type(type_spec), encoded)
        return encoded.SerializeToString(deterministic=True)

    def encode_attribute(value: object) -> bytes:
        """Encode one operation attribute for the native local equation."""

        encoded = wire.LiteralProto()
        encode_literal(freeze_value(value), encoded)
        return encoded.SerializeToString(deterministic=True)

    entries = (
        tuple(attributes.items())
        if isinstance(attributes, Mapping)
        else tuple(attributes)
    )
    if len(dict(entries)) != len(entries):
        raise ValueError("classical operation attributes must be unique")
    result = infer(
        opcode,
        tuple(encode_operand(type_spec) for type_spec in operands),
        tuple((name, encode_attribute(value)) for name, value in sorted(entries)),
    )
    encoded = wire.TypeProto()
    try:
        payload = bytes(result)
        encoded.ParseFromString(payload)
        if encoded.SerializeToString(deterministic=True) != payload:
            raise DecodeError("noncanonical native type")
        return decode_type(encoded)
    except (DecodeError, SerializationError) as error:
        raise RuntimeError(
            "Q4506: native classical type inference returned a malformed type"
        ) from error


def normalize_aggregate_slice(
    extent: int,
    start: int | None,
    stop: int | None,
    step: int | None,
) -> tuple[int, ...]:
    """Normalize one exact aggregate slice through the native compiler.

    Args:
        extent: Fixed positive leading aggregate extent.
        start: Optional exact inclusive bound.
        stop: Optional exact exclusive bound.
        step: Optional exact nonzero stride.

    Returns:
        Canonical zero-based positions in source iteration order.

    Raises:
        TypeError: If the native compiler boundary is unavailable.
        ValueError: If the extent or stride is invalid.
        RuntimeError: If the native compiler returns malformed positions.
    """

    normalize = getattr(native, "normalize_aggregate_slice", None)
    if not callable(normalize):
        raise TypeError("Q4505: native aggregate slice normalization is unavailable")
    if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
        raise ValueError("aggregate slice extent must be a positive integer")
    result = normalize(
        extent,
        None if start is None else str(start),
        None if stop is None else str(stop),
        None if step is None else str(step),
    )
    if (
        not isinstance(result, list)
        or len(result) > extent
        or any(
            type(position) is not int or not 0 <= position < extent
            for position in result
        )
        or len(set(result)) != len(result)
    ):
        raise RuntimeError("Q4512: native aggregate slice normalization is malformed")
    return tuple(result)


_OPERATION_CATALOG, TERMINATOR_KINDS = _load_native_catalog()

GATES_BY_NAME = MappingProxyType(
    {gate.name: gate for _, _, _, _, _, gate in _OPERATION_CATALOG if gate is not None}
)

GATES_BY_OPCODE = MappingProxyType(
    {spec.opcode: spec for spec in GATES_BY_NAME.values()}
)
PRIMITIVE_GATES = frozenset(GATES_BY_OPCODE)
ADJOINTABLE_GATES = PRIMITIVE_GATES
ADJOINTABLE_OPERATIONS = ADJOINTABLE_GATES | {
    "quantum.global_phase",
    "quantum.pauli_evolution",
}
ROTATION_GATES = frozenset(
    spec.opcode for spec in GATES_BY_NAME.values() if spec.has_angle
)

STRUCTURED_OPCODES = frozenset(
    opcode for opcode, _, structured, _, _, _ in _OPERATION_CATALOG if structured
)
QCORE_OPCODES = frozenset(
    opcode for opcode, dialect, _, _, _, _ in _OPERATION_CATALOG if dialect == "qcore"
)
CLASSICAL_OPCODES = frozenset(
    opcode
    for opcode, dialect, _, _, _, _ in _OPERATION_CATALOG
    if dialect == "classical"
)
QUANTUM_OPCODES = frozenset(
    opcode for opcode, dialect, _, _, _, _ in _OPERATION_CATALOG if dialect == "quantum"
)
KNOWN_OPCODES = frozenset(opcode for opcode, _, _, _, _, _ in _OPERATION_CATALOG)

OPERATION_EFFECTS = MappingProxyType(
    {opcode: effects for opcode, _, _, effects, _, _ in _OPERATION_CATALOG if effects}
)

OPERATION_REQUIREMENTS = MappingProxyType(
    {
        opcode: requirements
        for opcode, _, _, _, requirements, _ in _OPERATION_CATALOG
        if requirements
    }
)

__all__ = [
    "ADJOINTABLE_GATES",
    "ADJOINTABLE_OPERATIONS",
    "CLASSICAL_OPCODES",
    "GATES_BY_NAME",
    "GATES_BY_OPCODE",
    "GateSpec",
    "KNOWN_OPCODES",
    "OPERATION_EFFECTS",
    "OPERATION_REQUIREMENTS",
    "PRIMITIVE_GATES",
    "QCORE_OPCODES",
    "QUANTUM_OPCODES",
    "ROTATION_GATES",
    "STRUCTURED_OPCODES",
    "TERMINATOR_KINDS",
    "evaluate_classical",
    "infer_classical_result_type",
    "normalize_aggregate_slice",
]
