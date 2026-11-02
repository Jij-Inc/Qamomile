"""Typed, deterministic symbolic expressions for Qamomile resource facts.

The wire language is deliberately closed.  It contains data, never Python
callbacks or host-language expression objects, and therefore remains safe to
inspect and evaluate after deserialization.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Mapping, TypeAlias, cast, overload


class Domain(str, Enum):
    """Scalar domains supported by portable resource expressions."""

    NATURAL = "natural"
    INTEGER = "integer"
    RATIONAL = "rational"
    REAL = "real"
    PROBABILITY = "probability"
    BOOLEAN = "boolean"


def _validate_native_unit(name: str) -> None:
    """Validate one public unit identity through the Rust compiler."""

    from qamomile._native import compiler

    validate = getattr(compiler, "validate_resource_unit", None)
    if not callable(validate):
        raise RuntimeError(
            "Q4238: the required native resource-unit validator is unavailable"
        )
    result = validate(name)
    if result is not None:
        raise TypeError("Q4238: native resource-unit validation violated the ABI")


@dataclass(frozen=True, slots=True, order=True, init=False)
class Unit:
    """A nominal unit; equal dimensions do not imply interchangeable metrics."""

    name: str

    def __init__(self, name: str) -> None:
        """Construct one Rust-validated nominal unit identity."""

        _validate_native_unit(name)
        object.__setattr__(self, "name", name)

    @classmethod
    def _from_native(cls, name: str) -> Unit:
        """Build one unit already checked by the native graph owner."""

        if not isinstance(name, str):
            raise TypeError("Q4237: native resource unit violated the ABI")
        result = object.__new__(cls)
        object.__setattr__(result, "name", name)
        return result

    @classmethod
    def _from_data_record(cls, name: str) -> Unit:
        """Project one host unit before whole-root native validation."""

        if not isinstance(name, str):
            raise TypeError("resource expression unit must be a string")
        result = object.__new__(cls)
        object.__setattr__(result, "name", name)
        return result

    @property
    def dimensionless(self) -> bool:
        """Return whether the unit is the multiplicative identity."""

        return self.name == "1"


ONE = Unit("1")
LOGICAL_QUBITS = Unit("logical_qubits")
PHYSICAL_QUBITS = Unit("physical_qubits")
PHYSICAL_QUBIT_SECONDS = Unit("physical_qubit_seconds")
SEMANTIC_OPS = Unit("semantic_operations")
DEPENDENCY_STEPS = Unit("dependency_steps")
MEASUREMENTS = Unit("measurements")
RESETS = Unit("resets")
INSTRUCTIONS = Unit("instructions")
BYTES = Unit("bytes")
SECONDS = Unit("seconds")
SHOTS = Unit("shots")
CALLS = Unit("calls")
QUERIES = Unit("queries")

Number: TypeAlias = int | float | Fraction


@dataclass(frozen=True, slots=True, order=True, init=False)
class Symbol:
    """A scoped variable with a stable identity and declared mathematical domain."""

    scope: str
    name: str
    domain: Domain = Domain.NATURAL
    unit: Unit = ONE
    origin: str = "program_parameter"
    constraints: tuple[str, ...] = ()

    def __init__(
        self,
        scope: str,
        name: str,
        domain: Domain = Domain.NATURAL,
        unit: Unit = ONE,
        origin: str = "program_parameter",
        constraints: tuple[str, ...] = (),
    ) -> None:
        """Construct one declaration validated by the Rust compiler."""

        expression = _native_symbol_construction(
            scope, name, domain, unit, origin, constraints
        )
        declaration = expression.symbol
        if declaration is None:
            raise TypeError("Q4238: native symbol construction returned no declaration")
        object.__setattr__(self, "scope", declaration.scope)
        object.__setattr__(self, "name", declaration.name)
        object.__setattr__(self, "domain", declaration.domain)
        object.__setattr__(self, "unit", declaration.unit)
        object.__setattr__(self, "origin", declaration.origin)
        object.__setattr__(self, "constraints", declaration.constraints)

    @classmethod
    def _from_native(
        cls,
        scope: str,
        name: str,
        domain: Domain,
        unit: Unit,
        origin: str,
        constraints: tuple[str, ...],
    ) -> Symbol:
        """Build one declaration already checked by the native graph owner."""

        if (
            not isinstance(scope, str)
            or not isinstance(name, str)
            or not isinstance(domain, Domain)
            or not isinstance(unit, Unit)
            or not isinstance(origin, str)
            or not isinstance(constraints, tuple)
            or any(not isinstance(constraint, str) for constraint in constraints)
        ):
            raise TypeError("Q4237: native resource symbol violated the ABI")
        result = object.__new__(cls)
        object.__setattr__(result, "scope", scope)
        object.__setattr__(result, "name", name)
        object.__setattr__(result, "domain", domain)
        object.__setattr__(result, "unit", unit)
        object.__setattr__(result, "origin", origin)
        object.__setattr__(result, "constraints", constraints)
        return result

    @classmethod
    def _from_data_record(
        cls,
        scope: str,
        name: str,
        domain: Domain,
        unit: Unit,
        origin: str,
        constraints: tuple[str, ...],
    ) -> Symbol:
        """Project one host declaration before whole-root native validation."""

        return cls._from_native(scope, name, domain, unit, origin, constraints)

    @property
    def identity(self) -> str:
        """Return the unambiguous wire identity of this symbol."""

        return f"{self.scope}::{self.name}"


_KINDS = {
    "literal",
    "symbol",
    "add",
    "multiply",
    "power",
    "absolute",
    "minimum",
    "maximum",
    "floor",
    "ceiling",
    "floor_divide",
    "ceiling_divide",
    "exp",
    "log",
    "log2",
    "square_root",
    "piecewise",
    "bounded_sum",
    "bounded_product",
    "contracted_function",
    "recurrence",
    "recurrence_equation",
    "recurrence_system",
}

_PREDICATE_KINDS = {
    "true",
    "false",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "divisible",
    "and",
    "or",
    "not",
}

_MAX_CONSTRUCTION_INPUTS = 100_000
_MAX_CONSTRUCTION_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True, init=False)
class Predicate:
    """A closed Boolean condition over resource expressions."""

    kind: str
    arguments: tuple[Expr, ...] = ()
    predicates: tuple[Predicate, ...] = ()

    def __init__(
        self,
        kind: str,
        arguments: tuple[Expr, ...] = (),
        predicates: tuple[Predicate, ...] = (),
    ) -> None:
        """Construct and validate one predicate through the Rust compiler."""

        result = _native_predicate_construction(kind, arguments, predicates)
        object.__setattr__(self, "kind", result.kind)
        object.__setattr__(self, "arguments", result.arguments)
        object.__setattr__(self, "predicates", result.predicates)

    @classmethod
    def _from_native(
        cls,
        kind: str,
        arguments: tuple[Expr, ...],
        predicates: tuple[Predicate, ...],
    ) -> Predicate:
        """Build one already checked native predicate projection."""

        valid_shape = (
            (kind in {"true", "false"} and not arguments and not predicates)
            or (
                kind
                in {
                    "equal",
                    "not_equal",
                    "less",
                    "less_equal",
                    "greater",
                    "greater_equal",
                    "divisible",
                }
                and len(arguments) == 2
                and not predicates
            )
            or (kind in {"and", "or"} and not arguments and bool(predicates))
            or (kind == "not" and not arguments and len(predicates) == 1)
        )
        if kind not in _PREDICATE_KINDS or not valid_shape:
            raise TypeError("Q4237: native predicate has an invalid constructor shape")
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "arguments", arguments)
        object.__setattr__(result, "predicates", predicates)
        return result

    @classmethod
    def _from_data_record(
        cls,
        kind: str,
        arguments: tuple[Expr, ...],
        predicates: tuple[Predicate, ...],
    ) -> Predicate:
        """Project one host record before whole-root native validation."""

        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "arguments", arguments)
        object.__setattr__(result, "predicates", predicates)
        return result

    def substitute(self, values: Mapping[str, Expr | Number]) -> Predicate:
        """Substitute all scoped expression symbols in this predicate."""

        result = _native_substitution(self, values)
        if not isinstance(result, Predicate):
            raise TypeError(
                "Q4237: native predicate substitution returned an expression"
            )
        return result

    def truth(self) -> bool | None:
        """Evaluate a fully literal predicate or return unknown."""

        result = _native_evaluation(self)
        if result[0] == "unknown":
            return None
        if result[0] != "boolean" or not isinstance(result[1], bool):
            raise TypeError("Q4234: native predicate evaluation returned a non-Boolean")
        return result[1]

    def to_data(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible predicate record."""

        data: dict[str, object] = {"kind": self.kind}
        if self.arguments:
            data["arguments"] = [argument.to_data() for argument in self.arguments]
        if self.predicates:
            data["predicates"] = [item.to_data() for item in self.predicates]
        return data

    @classmethod
    def from_data(cls, data: object, *, max_nodes: int = 100_000) -> Predicate:
        """Decode and validate one allowlisted predicate record in Rust."""

        result = _resource_from_data(data, max_nodes=max_nodes, predicate=True)
        if not isinstance(result, Predicate):
            raise TypeError("resource predicate data produced an expression")
        _native_normalization(result)
        return result


@dataclass(frozen=True, slots=True)
class Expr:
    """One node in the closed symbolic expression DAG."""

    kind: str
    domain: Domain
    unit: Unit = ONE
    arguments: tuple[Expr, ...] = ()
    value: Number | None = None
    symbol: Symbol | None = None
    attributes: tuple[tuple[str, str], ...] = ()
    predicates: tuple[Predicate, ...] = ()

    def __post_init__(self) -> None:
        """Validate constructor shape and scalar invariants."""

        if self.kind not in _KINDS:
            raise ValueError(f"unknown resource expression constructor {self.kind!r}")
        attribute_names = tuple(name for name, _ in self.attributes)
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("resource expression attributes must be unique")
        if self.kind == "piecewise":
            if not self.predicates or len(self.arguments) != len(self.predicates) + 1:
                raise ValueError(
                    "piecewise requires one value per predicate and a default"
                )
            if any(argument.unit != self.unit for argument in self.arguments):
                raise ValueError("piecewise branch units must agree")
        elif self.predicates and self.kind != "recurrence_equation":
            raise ValueError(
                "only piecewise and recurrence equations may contain predicates"
            )
        if self.kind == "literal":
            if self.value is None or self.arguments or self.symbol is not None:
                raise ValueError("literal expressions require exactly one value")
            _validate_number(self.value, self.domain)
        elif self.kind == "symbol":
            if self.symbol is None or self.arguments or self.value is not None:
                raise ValueError("symbol expressions require exactly one symbol")
            if self.symbol.domain is not self.domain or self.symbol.unit != self.unit:
                raise ValueError("symbol expression disagrees with its declaration")
        elif self.value is not None or self.symbol is not None:
            raise ValueError(f"{self.kind} cannot carry a literal or symbol payload")

    @classmethod
    def _from_native(
        cls,
        kind: str,
        domain: Domain,
        unit: Unit = ONE,
        arguments: tuple[Expr, ...] = (),
        value: Number | None = None,
        symbol: Symbol | None = None,
        attributes: tuple[tuple[str, str], ...] = (),
        predicates: tuple[Predicate, ...] = (),
    ) -> Expr:
        """Build one expression already checked by the Rust compiler.

        Args:
            kind (str): Closed native constructor spelling.
            domain (Domain): Rust-derived mathematical result domain.
            unit (Unit): Rust-derived nominal result unit.
            arguments (tuple[Expr, ...]): Earlier expression nodes referenced by
                the constructor.
            value (Number | None): Decoded literal value, when present.
            symbol (Symbol | None): Decoded symbol declaration, when present.
            attributes (tuple[tuple[str, str], ...]): Canonical constructor attributes.
            predicates (tuple[Predicate, ...]): Earlier predicates, including an
                equation's applicability condition.

        Returns:
            Expr: Immutable projection without a second semantic validation pass.

        Raises:
            TypeError: If the native projection violates the immutable facade ABI.
        """

        attribute_names = tuple(name for name, _ in attributes)
        valid_shape = kind in _KINDS and len(attribute_names) == len(
            set(attribute_names)
        )
        if kind == "piecewise":
            valid_shape = (
                valid_shape
                and bool(predicates)
                and len(arguments) == len(predicates) + 1
            )
        elif kind == "recurrence_equation":
            valid_shape = valid_shape and bool(arguments) and len(predicates) == 1
        elif kind == "recurrence_system":
            valid_shape = valid_shape and len(arguments) >= 2 and not predicates
        elif predicates:
            valid_shape = False
        if kind == "literal":
            valid_shape = (
                valid_shape and value is not None and not arguments and symbol is None
            )
        elif kind == "symbol":
            valid_shape = (
                valid_shape
                and symbol is not None
                and not arguments
                and value is None
                and symbol.domain is domain
                and symbol.unit == unit
            )
        elif value is not None or symbol is not None:
            valid_shape = False
        if not valid_shape:
            raise TypeError("Q4237: native expression has an invalid constructor shape")
        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "domain", domain)
        object.__setattr__(result, "unit", unit)
        object.__setattr__(result, "arguments", arguments)
        object.__setattr__(result, "value", value)
        object.__setattr__(result, "symbol", symbol)
        object.__setattr__(result, "attributes", attributes)
        object.__setattr__(result, "predicates", predicates)
        return result

    @classmethod
    def _from_data_record(
        cls,
        kind: str,
        domain: Domain,
        unit: Unit,
        arguments: tuple[Expr, ...],
        value: Number | None,
        symbol: Symbol | None,
        attributes: tuple[tuple[str, str], ...],
        predicates: tuple[Predicate, ...],
    ) -> Expr:
        """Project one host record before whole-root native validation."""

        result = object.__new__(cls)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "domain", domain)
        object.__setattr__(result, "unit", unit)
        object.__setattr__(result, "arguments", arguments)
        object.__setattr__(result, "value", value)
        object.__setattr__(result, "symbol", symbol)
        object.__setattr__(result, "attributes", attributes)
        object.__setattr__(result, "predicates", predicates)
        return result

    def __add__(self, other: Expr | Number) -> Expr:
        """Construct a domain- and unit-checked sum."""

        return add(self, as_expr(other))

    def __radd__(self, other: Expr | Number) -> Expr:
        """Construct a sum with a host numeric value on the left."""

        return add(as_expr(other), self)

    def __mul__(self, other: Expr | Number) -> Expr:
        """Construct a domain- and unit-checked product."""

        return multiply(self, as_expr(other))

    def __rmul__(self, other: Expr | Number) -> Expr:
        """Construct a product with a host numeric value on the left."""

        return multiply(as_expr(other), self)

    def evaluate(self, values: Mapping[str, Number]) -> Expr:
        """Substitute available symbols and simplify without changing semantics."""

        return self.substitute(resolve_substitutions((self,), values))

    def substitute(self, values: Mapping[str, Expr | Number]) -> Expr:
        """Substitute symbolic expressions while preserving scoped identities."""

        result = _native_substitution(self, values)
        if not isinstance(result, Expr):
            raise TypeError(
                "Q4237: native expression substitution returned a predicate"
            )
        return result

    def numeric(self) -> Number | None:
        """Return an exact numeric value without expanding bounded expressions."""

        result = _native_evaluation(self)
        if result[0] == "unknown":
            return None
        if result[0] == "boolean" or isinstance(result[1], bool):
            raise TypeError("Q4234: native numeric evaluation returned a Boolean")
        return cast(Number, result[1])

    def to_data(self) -> dict[str, object]:
        """Return a canonical JSON-compatible constructor record."""

        data: dict[str, object] = {
            "kind": self.kind,
            "domain": self.domain.value,
            "unit": self.unit.name,
        }
        if self.arguments:
            data["arguments"] = [argument.to_data() for argument in self.arguments]
        if self.value is not None:
            data["value"] = _number_to_data(self.value)
        if self.symbol is not None:
            data["symbol"] = {
                "scope": self.symbol.scope,
                "name": self.symbol.name,
                "domain": self.symbol.domain.value,
                "unit": self.symbol.unit.name,
                "origin": self.symbol.origin,
                "constraints": list(self.symbol.constraints),
            }
        if self.attributes:
            data["attributes"] = [list(item) for item in self.attributes]
        if self.predicates:
            data["predicates"] = [predicate.to_data() for predicate in self.predicates]
        return data

    @classmethod
    def from_data(cls, data: object, *, max_nodes: int = 100_000) -> Expr:
        """Decode and validate one allowlisted expression record in Rust."""

        result = _resource_from_data(data, max_nodes=max_nodes, predicate=False)
        if not isinstance(result, Expr):
            raise TypeError("resource expression data produced a predicate")
        _native_normalization(result)
        return result

    def __str__(self) -> str:
        """Render a stable compact mathematical representation."""

        if self.kind == "literal":
            return str(self.value)
        if self.kind == "symbol":
            assert self.symbol is not None
            return self.symbol.name
        if self.kind == "add":
            return " + ".join(str(argument) for argument in self.arguments)
        if self.kind == "multiply":
            return " * ".join(_parenthesize(argument) for argument in self.arguments)
        if self.kind in {"minimum", "maximum"}:
            return f"{self.kind}({', '.join(map(str, self.arguments))})"
        name = dict(self.attributes).get("name", self.kind)
        return f"{name}({', '.join(map(str, self.arguments))})"


def _native_evaluation(value: Expr | Predicate) -> tuple[str, Number | bool | None]:
    """Evaluate one closed expression-language root through the Rust compiler.

    Args:
        value: Numeric expression or Boolean predicate to evaluate.

    Returns:
        Native result kind and its validated Python scalar projection.

    Raises:
        RuntimeError: If the required native evaluation API is unavailable.
        TypeError: If the native projection violates the bridge ABI.
        ValueError: If native evaluation detects invalid numeric semantics.
        ZeroDivisionError: If a division-like expression has a zero divisor.
    """

    from qamomile._native import compiler

    from ._rule_wire import encode_resource_expression_graph

    evaluate = getattr(compiler, "evaluate_resource_expression", None)
    if evaluate is None:
        raise RuntimeError(
            "Q4234: the required native resource-expression evaluator is unavailable"
        )
    result = evaluate(encode_resource_expression_graph(value))
    kind = getattr(result, "kind", None)
    if kind == "unknown":
        return kind, None
    if kind == "boolean":
        boolean = getattr(result, "boolean", None)
        if not isinstance(boolean, bool):
            raise TypeError("Q4234: malformed native Boolean evaluation result")
        return kind, boolean
    if kind == "integer":
        integer = getattr(result, "integer", None)
        if not isinstance(integer, str):
            raise TypeError("Q4234: malformed native integer evaluation result")
        try:
            return kind, int(integer)
        except ValueError as error:
            raise TypeError("Q4234: malformed native integer spelling") from error
    if kind == "rational":
        numerator = getattr(result, "rational_numerator", None)
        denominator = getattr(result, "rational_denominator", None)
        if not isinstance(numerator, str) or not isinstance(denominator, str):
            raise TypeError("Q4234: malformed native rational evaluation result")
        try:
            return kind, Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError) as error:
            raise TypeError("Q4234: malformed native rational spelling") from error
    if kind == "real":
        bits = getattr(result, "real_bits", None)
        if (
            isinstance(bits, bool)
            or not isinstance(bits, int)
            or not 0 <= bits < 1 << 64
        ):
            raise TypeError("Q4234: malformed native real evaluation result")
        real = struct.unpack(">d", struct.pack(">Q", bits))[0]
        if not math.isfinite(real):
            raise TypeError("Q4234: native real evaluation result is not finite")
        return kind, real
    raise TypeError("Q4234: native resource-expression evaluation violated the ABI")


def _native_substitution(
    value: Expr | Predicate,
    values: Mapping[str, Expr | Number],
) -> Expr | Predicate:
    """Substitute qualified symbols through the Rust expression DAG transform.

    Args:
        value: Expression or predicate root to transform.
        values: Qualified replacement expressions or host numeric values.

    Returns:
        A typed root reconstructed from the native topological projection.

    Raises:
        RuntimeError: If the required native substitution API is unavailable.
        TypeError: If a host value or native projection violates the ABI.
        ValueError: If substitution changes a nominal unit or exceeds a budget.
    """

    from qamomile._native import compiler

    from ._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    substitute = getattr(compiler, "substitute_resource_expression", None)
    if substitute is None:
        raise RuntimeError(
            "Q4237: the required native resource-expression substituter is unavailable"
        )
    declarations = {symbol.identity: symbol for symbol in _root_free_symbols(value)}
    bindings: list[tuple[str, bytes, bool]] = []
    for identity in sorted(declarations):
        supplied = values.get(identity)
        if supplied is None:
            continue
        declaration = declarations[identity]
        replacement = (
            supplied
            if isinstance(supplied, Expr)
            else literal(supplied, declaration.unit, declaration.domain)
        )
        bindings.append(
            (
                identity,
                encode_resource_expression_graph(replacement),
                not isinstance(supplied, Expr),
            )
        )
    result = substitute(encode_resource_expression_graph(value), bindings)
    if not isinstance(result, bytes):
        raise TypeError(
            "Q4237: native resource-expression substitution violated the ABI"
        )
    return decode_resource_expression_graph(result)


@overload
def _native_normalization(value: Expr) -> Expr:
    """Normalize one resource expression through the native algebra.

    Args:
        value (Expr): Validated resource expression.

    Returns:
        Expr: Canonical native-normalized expression.

    Raises:
        RuntimeError: If the native normalization API is unavailable.
        TypeError: If the native boundary returns an invalid result.
    """

    ...


@overload
def _native_normalization(value: Predicate) -> Predicate:
    """Normalize one resource predicate through the native algebra.

    Args:
        value (Predicate): Validated resource predicate.

    Returns:
        Predicate: Canonical native-normalized predicate.

    Raises:
        RuntimeError: If the native normalization API is unavailable.
        TypeError: If the native boundary returns an invalid result.
    """

    ...


def _native_normalization(value: Expr | Predicate) -> Expr | Predicate:
    """Normalize one validated graph through the Rust constructor algebra.

    Args:
        value: Raw typed expression or predicate root with checked host inputs.

    Returns:
        Canonical native-normalized root of the same category.

    Raises:
        RuntimeError: If the required native normalization API is unavailable.
        TypeError: If the native projection violates the bridge ABI.
        ValueError: If native normalization rejects the expression or its budget.
    """

    from qamomile._native import compiler

    from ._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    normalize = getattr(compiler, "normalize_resource_expression", None)
    if normalize is None:
        raise RuntimeError(
            "Q4238: the required native resource-expression normalizer is unavailable"
        )
    payload = normalize(encode_resource_expression_graph(value))
    if not isinstance(payload, bytes):
        raise TypeError(
            "Q4238: native resource-expression normalization violated the ABI"
        )
    result = decode_resource_expression_graph(payload)
    if isinstance(result, Expr) != isinstance(value, Expr):
        raise TypeError("Q4238: native normalization changed the root category")
    return result


def _native_construction(
    kind: str,
    arguments: tuple[Expr, ...],
    predicates: tuple[Predicate, ...] = (),
    *,
    attributes: tuple[tuple[str, str], ...] = (),
    binder_unit: Unit | None = None,
    result_domain: Domain | None = None,
    result_unit: Unit | None = None,
) -> Expr:
    """Construct one typed algebraic node through the Rust expression owner.

    Args:
        kind: Closed resource-expression constructor name.
        arguments: Numeric expression roots consumed by the constructor.
        predicates: Boolean roots selecting ordered piecewise branches.
        attributes: Canonical constructor attributes.
        binder_unit: Ephemeral binder unit checked before bounded construction.
        result_domain: Explicit result domain for an opaque constructor.
        result_unit: Explicit nominal result unit for an opaque constructor.

    Returns:
        Canonical typed expression with its Rust-derived domain and unit.

    Raises:
        RuntimeError: If the required native constructor API is unavailable.
        TypeError: If the native projection violates the bridge ABI.
        ValueError: If Rust rejects the constructor contract or a graph budget.
    """

    from qamomile._native import compiler

    from ._rule_wire import (
        _DOMAINS,
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    construct = getattr(compiler, "construct_resource_expression", None)
    if construct is None:
        raise RuntimeError(
            "Q4238: the required native resource-expression constructor is unavailable"
        )
    if len(arguments) + len(predicates) > _MAX_CONSTRUCTION_INPUTS:
        raise ValueError("resource expression construction exceeds the input budget")
    encoded_arguments: list[bytes] = []
    encoded_predicates: list[bytes] = []
    encoded_bytes = 0
    for values, target in (
        (arguments, encoded_arguments),
        (predicates, encoded_predicates),
    ):
        for value in values:
            encoded = encode_resource_expression_graph(value)
            encoded_bytes += len(encoded)
            if encoded_bytes > _MAX_CONSTRUCTION_BYTES:
                raise ValueError(
                    "resource expression construction exceeds the wire budget"
                )
            target.append(encoded)
    if (result_domain is None) != (result_unit is None):
        raise ValueError(
            "Q4238: native resource-expression result contracts must be complete"
        )
    result_contract = (
        None
        if result_domain is None or result_unit is None
        else (_DOMAINS[result_domain], result_unit.name)
    )
    payload = construct(
        kind,
        encoded_arguments,
        encoded_predicates,
        list(attributes),
        None if binder_unit is None else binder_unit.name,
        result_contract,
    )
    if not isinstance(payload, bytes):
        raise TypeError(
            "Q4238: native resource-expression construction violated the ABI"
        )
    result = decode_resource_expression_graph(payload)
    if not isinstance(result, Expr):
        raise TypeError("Q4238: native expression construction returned a predicate")
    return result


def _native_predicate_construction(
    kind: str,
    arguments: tuple[Expr, ...],
    predicates: tuple[Predicate, ...],
) -> Predicate:
    """Construct one Boolean node through the Rust predicate contract.

    Args:
        kind: Closed resource-predicate constructor name.
        arguments: Numeric expression roots consumed by the predicate.
        predicates: Nested Boolean roots consumed by the predicate.

    Returns:
        Canonical checked predicate projected from Rust.

    Raises:
        RuntimeError: If the required native constructor API is unavailable.
        TypeError: If the native projection violates the bridge ABI.
        ValueError: If Rust rejects the predicate contract or graph budget.
    """

    from qamomile._native import compiler

    from ._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )

    construct = getattr(compiler, "construct_resource_predicate", None)
    if construct is None:
        raise RuntimeError(
            "Q4238: the required native resource-predicate constructor is unavailable"
        )
    if len(arguments) + len(predicates) > _MAX_CONSTRUCTION_INPUTS:
        raise ValueError("resource predicate construction exceeds the input budget")
    encoded_arguments: list[bytes] = []
    encoded_predicates: list[bytes] = []
    encoded_bytes = 0
    for values, target in (
        (arguments, encoded_arguments),
        (predicates, encoded_predicates),
    ):
        for value in values:
            encoded = encode_resource_expression_graph(value)
            encoded_bytes += len(encoded)
            if encoded_bytes > _MAX_CONSTRUCTION_BYTES:
                raise ValueError(
                    "resource predicate construction exceeds the wire budget"
                )
            target.append(encoded)
    payload = construct(kind, encoded_arguments, encoded_predicates)
    if not isinstance(payload, bytes):
        raise TypeError(
            "Q4238: native resource-predicate construction violated the ABI"
        )
    result = decode_resource_expression_graph(payload)
    if not isinstance(result, Predicate):
        raise TypeError("Q4238: native predicate construction returned an expression")
    return result


def _native_literal(
    value: Number,
    unit: Unit,
    domain: Domain | None,
) -> Expr:
    """Introduce one numeric literal through the Rust domain owner.

    Args:
        value: Host integer, rational, or binary64 value.
        unit: Nominal unit attached to the literal.
        domain: Optional explicit containing domain.

    Returns:
        Canonical literal with its Rust-derived or checked domain.

    Raises:
        RuntimeError: If the required native constructor API is unavailable.
        TypeError: If the host value or native projection violates the ABI.
        ValueError: If Rust rejects the numeric value, domain, unit, or budget.
    """

    from qamomile._native import compiler
    from qamomile.proto import resource_analysis_pb2 as wire

    from ._rule_wire import _DOMAINS, decode_resource_expression_graph

    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError("boolean is not a resource numeric literal")
    number = wire.ResourceNumberProto()
    if isinstance(value, int):
        number.integer = str(value)
    elif isinstance(value, Fraction):
        number.rational.numerator = str(value.numerator)
        number.rational.denominator = str(value.denominator)
    else:
        number.real_bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    construct = getattr(compiler, "construct_resource_literal", None)
    if construct is None:
        raise RuntimeError(
            "Q4238: the required native resource-literal constructor is unavailable"
        )
    payload = construct(
        number.SerializeToString(deterministic=True),
        unit.name,
        None if domain is None else _DOMAINS[domain],
    )
    if not isinstance(payload, bytes):
        raise TypeError("Q4238: native resource-literal construction violated the ABI")
    result = decode_resource_expression_graph(payload)
    if not isinstance(result, Expr) or result.kind != "literal":
        raise TypeError("Q4238: native literal construction returned a non-literal")
    return result


def _root_free_symbols(value: Expr | Predicate) -> tuple[Symbol, ...]:
    """Project free declarations through the Rust lexical-scope owner."""

    from qamomile._native import compiler

    from ._rule_wire import encode_resource_expression_graph

    inspect = getattr(compiler, "inspect_resource_expression_symbols", None)
    if inspect is None:
        raise RuntimeError(
            "Q4239: the required native resource-symbol inspector is unavailable"
        )
    projection = inspect(encode_resource_expression_graph(value))
    if not isinstance(projection, list):
        raise TypeError("Q4239: native resource-symbol inspection violated the ABI")

    symbols: list[Symbol] = []
    try:
        for item in projection:
            scope = item.scope
            name = item.name
            domain = item.domain
            unit = item.unit
            origin = item.origin
            constraints = item.constraints
            if (
                not isinstance(scope, str)
                or not isinstance(name, str)
                or not isinstance(domain, str)
                or not isinstance(unit, str)
                or not isinstance(origin, str)
                or not isinstance(constraints, list)
                or any(not isinstance(constraint, str) for constraint in constraints)
            ):
                raise TypeError
            symbols.append(
                Symbol._from_native(
                    scope=scope,
                    name=name,
                    domain=Domain(domain),
                    unit=Unit._from_native(unit),
                    origin=origin,
                    constraints=tuple(constraints),
                )
            )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "Q4239: native resource-symbol inspection violated the ABI"
        ) from error

    identities = [symbol_value.identity for symbol_value in symbols]
    if identities != sorted(set(identities)):
        raise TypeError("Q4239: native resource-symbol inspection violated the ABI")
    return tuple(symbols)


def resolve_substitutions(
    expressions: Iterable[Expr],
    values: Mapping[str, Expr | Number],
) -> dict[str, Expr | Number]:
    """Resolve public symbol names through the Rust lexical-scope owner.

    Args:
        expressions: Expressions sharing one public evaluation boundary.
        values: Qualified identities or convenient short-name substitutions.

    Returns:
        A mapping keyed only by stable qualified symbol identities.

    Raises:
        ValueError: If one supplied short name denotes symbols from multiple
            scopes.
    """

    from qamomile._native import compiler

    from ._rule_wire import encode_resource_expression_graph

    resolve = getattr(compiler, "resolve_resource_symbol_bindings", None)
    if not callable(resolve):
        raise RuntimeError(
            "Q4239: the required native resource-symbol resolver is unavailable"
        )
    names = list(values)
    projection = resolve(
        [encode_resource_expression_graph(expression) for expression in expressions],
        names,
    )
    if not isinstance(projection, list):
        raise TypeError("Q4239: native resource-symbol resolution violated the ABI")
    try:
        rows = tuple(projection)
        if any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or "::" not in row[0]
            or not isinstance(row[1], str)
            or row[1] not in values
            for row in rows
        ):
            raise TypeError
        identities = tuple(row[0] for row in rows)
        if identities != tuple(sorted(set(identities))):
            raise TypeError
        return {identity: values[source] for identity, source in rows}
    except TypeError as error:
        raise TypeError(
            "Q4239: native resource-symbol resolution violated the ABI"
        ) from error


def literal(
    value: Number,
    unit: Unit = ONE,
    domain: Domain | None = None,
) -> Expr:
    """Construct a validated literal expression."""

    return _native_literal(value, unit, domain)


def _native_symbol_payload(
    scope: str,
    name: str,
    domain: Domain,
    unit: Unit,
    origin: str,
    constraints: tuple[str, ...],
) -> bytes:
    """Validate and encode one symbol declaration through the Rust compiler."""

    from qamomile._native import compiler

    from ._rule_wire import _DOMAINS

    if (
        not isinstance(scope, str)
        or not isinstance(name, str)
        or not isinstance(domain, Domain)
        or not isinstance(unit, Unit)
        or not isinstance(origin, str)
        or not isinstance(constraints, tuple)
        or any(not isinstance(constraint, str) for constraint in constraints)
    ):
        raise TypeError("Q4238: resource symbol declaration has an invalid host shape")
    construct = getattr(compiler, "construct_resource_symbol", None)
    if not callable(construct):
        raise RuntimeError(
            "Q4238: the required native resource-symbol constructor is unavailable"
        )
    payload = construct(
        scope,
        name,
        _DOMAINS[domain],
        unit.name,
        origin,
        list(constraints),
    )
    if not isinstance(payload, bytes):
        raise TypeError("Q4238: native resource-symbol construction violated the ABI")
    return payload


def _native_symbol_construction(
    scope: str,
    name: str,
    domain: Domain,
    unit: Unit,
    origin: str,
    constraints: tuple[str, ...],
) -> Expr:
    """Return one checked symbol expression projected from the Rust compiler."""

    from ._rule_wire import decode_resource_expression_graph

    result = decode_resource_expression_graph(
        _native_symbol_payload(scope, name, domain, unit, origin, constraints)
    )
    if not isinstance(result, Expr) or result.kind != "symbol":
        raise TypeError("Q4238: native symbol construction returned a non-symbol")
    return result


def scoped_symbol(symbol: Symbol) -> Expr:
    """Construct a symbolic reference from its declaration."""

    if not isinstance(symbol, Symbol):
        raise TypeError("Q4238: scoped_symbol requires a Symbol declaration")
    return _native_symbol_construction(
        symbol.scope,
        symbol.name,
        symbol.domain,
        symbol.unit,
        symbol.origin,
        symbol.constraints,
    )


def symbol(
    name: str,
    *,
    scope: str = "user",
    domain: Domain = Domain.NATURAL,
    unit: Unit = ONE,
    origin: str = "program_parameter",
    constraints: tuple[str, ...] = (),
) -> Expr:
    """Construct a scoped symbolic resource parameter."""

    return _native_symbol_construction(scope, name, domain, unit, origin, constraints)


def as_expr(value: Expr | Number, *, unit: Unit = ONE) -> Expr:
    """Normalize an expression or host number to an expression."""

    return value if isinstance(value, Expr) else literal(value, unit)


def add(*operands: Expr) -> Expr:
    """Construct a canonical sum with exact literal folding."""

    if not operands:
        return literal(0)
    return _native_construction("add", operands)


def multiply(*operands: Expr) -> Expr:
    """Construct a canonical product with exact literal folding."""

    if not operands:
        return literal(1)
    return _native_construction("multiply", operands)


def minimum(*operands: Expr) -> Expr:
    """Construct a unit-checked pointwise minimum."""

    return _extremum("minimum", operands)


def maximum(*operands: Expr) -> Expr:
    """Construct a unit-checked pointwise maximum."""

    return _extremum("maximum", operands)


def power(base: Expr, exponent: Expr | Number) -> Expr:
    """Construct a domain-checked integer or rational power."""

    return _native_construction("power", (base, as_expr(exponent)))


def absolute(operand: Expr) -> Expr:
    """Construct an absolute-value expression preserving its nominal unit."""

    return _native_construction("absolute", (operand,))


def floor(operand: Expr) -> Expr:
    """Construct a mathematical floor expression."""

    return _native_construction("floor", (operand,))


def ceiling(operand: Expr) -> Expr:
    """Construct a mathematical ceiling expression."""

    return _native_construction("ceiling", (operand,))


def floor_divide(numerator: Expr, denominator: Expr) -> Expr:
    """Construct an integer floor division by a dimensionless denominator."""

    return _native_construction("floor_divide", (numerator, denominator))


def ceiling_divide(numerator: Expr, denominator: Expr) -> Expr:
    """Construct a safe integer ceiling division expression."""

    return _native_construction("ceiling_divide", (numerator, denominator))


def exponential(operand: Expr) -> Expr:
    """Construct an exponential of a dimensionless expression."""

    return _native_construction("exp", (operand,))


def logarithm(operand: Expr, *, base: Expr | Number | None = None) -> Expr:
    """Construct a logarithm with an explicit positive dimensionless base."""

    arguments = (operand,)
    if base is not None:
        arguments = (operand, as_expr(base))
    return _native_construction("log", arguments)


def log2(operand: Expr) -> Expr:
    """Construct a base-two logarithm of a positive dimensionless expression."""

    return _native_construction("log2", (operand,))


def square_root(operand: Expr) -> Expr:
    """Construct a real square root of a dimensionless expression."""

    return _native_construction("square_root", (operand,))


def comparison(kind: str, left: Expr, right: Expr) -> Predicate:
    """Construct a typed equality or order predicate."""

    return Predicate(kind, (left, right))


def divisible(value: Expr, divisor: Expr | Number) -> Predicate:
    """Construct an integral divisibility predicate."""

    return Predicate("divisible", (value, as_expr(divisor)))


def conjunction(*predicates: Predicate) -> Predicate:
    """Construct a Boolean conjunction."""

    return Predicate("and", predicates=predicates)


def disjunction(*predicates: Predicate) -> Predicate:
    """Construct a Boolean disjunction."""

    return Predicate("or", predicates=predicates)


def negate(predicate: Predicate) -> Predicate:
    """Construct Boolean negation."""

    return Predicate("not", predicates=(predicate,))


def piecewise(
    branches: tuple[tuple[Predicate, Expr], ...],
    *,
    otherwise: Expr,
) -> Expr:
    """Construct an exhaustive ordered piecewise expression."""

    if not branches:
        return otherwise
    values = tuple(value for _, value in branches)
    return _native_construction(
        "piecewise",
        (*values, otherwise),
        tuple(predicate for predicate, _ in branches),
    )


def bounded_sum(binder: Symbol, lower: Expr, upper: Expr, body: Expr) -> Expr:
    """Construct an inclusive bounded sum without unsafe expansion."""

    return _native_construction(
        "bounded_sum",
        (lower, upper, body),
        attributes=_binder_attributes(binder),
        binder_unit=binder.unit,
    )


def bounded_product(binder: Symbol, lower: Expr, upper: Expr, body: Expr) -> Expr:
    """Construct an inclusive dimensionless bounded product."""

    return _native_construction(
        "bounded_product",
        (lower, upper, body),
        attributes=_binder_attributes(binder),
        binder_unit=binder.unit,
    )


def recurrence(
    name: str,
    arguments: tuple[Expr, ...],
    *,
    domain: Domain,
    unit: Unit,
    version: str = "1",
) -> Expr:
    """Construct a namespaced versioned symbolic recurrence contract."""

    return _native_construction(
        "recurrence",
        arguments,
        attributes=(("name", name), ("version", version)),
        result_domain=domain,
        result_unit=unit,
    )


def contracted_function(
    name: str,
    arguments: tuple[Expr, ...],
    *,
    domain: Domain,
    unit: Unit,
    version: str = "1",
) -> Expr:
    """Construct a namespaced declarative function contract."""

    return _native_construction(
        "contracted_function",
        arguments,
        attributes=(("name", name), ("version", version)),
        result_domain=domain,
        result_unit=unit,
    )


def _extremum(kind: str, operands: tuple[Expr, ...]) -> Expr:
    """Build a simplified minimum or maximum."""

    if not operands:
        raise ValueError(f"{kind} requires at least one operand")
    return _native_construction(kind, operands)


def _binder_attributes(binder: Symbol) -> tuple[tuple[str, str], ...]:
    """Encode a binder declaration into canonical expression attributes."""

    return tuple(
        sorted(
            {
                "binder_domain": binder.domain.value,
                "binder_name": binder.name,
                "binder_origin": binder.origin,
                "binder_scope": binder.scope,
            }.items()
        )
    )


def _validate_number(value: Number, domain: Domain) -> None:
    """Validate one literal against its declared domain."""

    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError("resource literals must be finite numeric values")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("resource literals cannot contain NaN or infinity")
    if domain is Domain.NATURAL and (not isinstance(value, int) or value < 0):
        raise ValueError("natural resource literals must be non-negative integers")
    if domain is Domain.INTEGER and not isinstance(value, int):
        raise ValueError("integer resource literals must be integers")
    if domain is Domain.PROBABILITY and not 0 <= value <= 1:
        raise ValueError("probability resource literals must lie in [0, 1]")


def _resource_from_data(
    data: object,
    *,
    max_nodes: int,
    predicate: bool,
) -> Expr | Predicate:
    """Project one portable constructor tree before whole-root Rust validation.

    Args:
        data: JSON-compatible expression or predicate constructor record.
        max_nodes: Maximum combined expression and predicate node count.
        predicate: Whether the root must be a predicate rather than an expression.

    Returns:
        Raw immutable facade root ready for one native validation call.

    Raises:
        ValueError: If host containers, enum spellings, references, or budgets fail.
    """

    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    stack: list[tuple[object, bool, bool]] = [(data, predicate, False)]
    seen: set[int] = set()
    decoded: dict[int, Expr | Predicate] = {}
    count = 0
    while stack:
        raw, is_predicate, expanded = stack.pop()
        context = "predicate" if is_predicate else "expression"
        if not isinstance(raw, dict):
            raise ValueError(f"resource {context} node must be a mapping")
        identity = id(raw)
        if not expanded:
            if identity in seen:
                raise ValueError("cyclic resource expression encoding")
            seen.add(identity)
            count += 1
            if count > max_nodes:
                raise ValueError("resource expression exceeds the node budget")
            kind = raw.get("kind")
            kinds = _PREDICATE_KINDS if is_predicate else _KINDS
            if not isinstance(kind, str) or kind not in kinds:
                raise ValueError(f"unknown resource {context} constructor {kind!r}")
            arguments = raw.get("arguments", [])
            predicates = raw.get("predicates", [])
            if not isinstance(arguments, list) or not isinstance(predicates, list):
                raise ValueError(f"resource {context} operands must be lists")
            stack.append((raw, is_predicate, True))
            stack.extend((child, True, False) for child in reversed(predicates))
            stack.extend((child, False, False) for child in reversed(arguments))
            continue

        arguments_raw = cast(list[object], raw.get("arguments", []))
        predicates_raw = cast(list[object], raw.get("predicates", []))
        arguments = tuple(
            _data_expression_child(child, decoded) for child in arguments_raw
        )
        predicates = tuple(
            _data_predicate_child(child, decoded) for child in predicates_raw
        )
        kind = cast(str, raw["kind"])
        if is_predicate:
            decoded[identity] = Predicate._from_data_record(
                kind,
                arguments,
                predicates,
            )
            continue

        domain_raw = raw.get("domain")
        unit_raw = raw.get("unit")
        if not isinstance(domain_raw, str) or not isinstance(unit_raw, str):
            raise ValueError("resource expression domain and unit must be strings")
        domain = Domain(domain_raw)
        unit = Unit._from_data_record(unit_raw)
        attributes_raw = raw.get("attributes", [])
        if not isinstance(attributes_raw, list):
            raise ValueError("expression attributes must be a list")
        if any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(component, str) for component in item)
            for item in attributes_raw
        ):
            raise ValueError("malformed expression attributes")
        attributes = tuple((item[0], item[1]) for item in attributes_raw)
        value = _number_from_data(raw.get("value")) if kind == "literal" else None
        symbol = _symbol_from_data(raw.get("symbol")) if kind == "symbol" else None
        decoded[identity] = Expr._from_data_record(
            kind,
            domain,
            unit,
            arguments,
            value,
            symbol,
            attributes,
            predicates,
        )
    return decoded[id(data)]


def _data_expression_child(
    child: object,
    decoded: Mapping[int, Expr | Predicate],
) -> Expr:
    """Return one previously projected expression child."""

    value = decoded.get(id(child))
    if not isinstance(value, Expr):
        raise ValueError("resource expression argument is not an expression")
    return value


def _data_predicate_child(
    child: object,
    decoded: Mapping[int, Expr | Predicate],
) -> Predicate:
    """Return one previously projected predicate child."""

    value = decoded.get(id(child))
    if not isinstance(value, Predicate):
        raise ValueError("resource expression predicate is not Boolean")
    return value


def _symbol_from_data(value: object) -> Symbol:
    """Decode one portable symbol declaration as host data."""

    if not isinstance(value, dict):
        raise ValueError("symbol node has no declaration")
    scope = value.get("scope")
    name = value.get("name")
    domain = value.get("domain")
    unit = value.get("unit")
    origin = value.get("origin")
    constraints = value.get("constraints", [])
    if (
        not isinstance(scope, str)
        or not isinstance(name, str)
        or not isinstance(domain, str)
        or not isinstance(unit, str)
        or not isinstance(origin, str)
        or not isinstance(constraints, list)
        or any(not isinstance(item, str) for item in constraints)
    ):
        raise ValueError("malformed resource symbol declaration")
    return Symbol._from_data_record(
        scope,
        name,
        Domain(domain),
        Unit._from_data_record(unit),
        origin,
        tuple(constraints),
    )


def _parenthesize(expression: Expr) -> str:
    """Parenthesize sums inside a product rendering."""

    rendered = str(expression)
    return f"({rendered})" if expression.kind == "add" else rendered


def _number_to_data(value: Number) -> object:
    """Encode numeric literals without losing rational identity."""

    if isinstance(value, Fraction):
        return {"rational": [value.numerator, value.denominator]}
    return value


def _number_from_data(value: object) -> Number:
    """Decode one bounded numeric literal record."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a resource numeric literal")
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise ValueError("resource integer exceeds the numeric-size limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resource float must be finite")
        return value
    if isinstance(value, dict) and set(value) == {"rational"}:
        pair = value["rational"]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in pair)
        ):
            raise ValueError("malformed rational resource literal")
        return Fraction(pair[0], pair[1])
    raise ValueError("unsupported resource numeric literal")


__all__ = [
    "Predicate",
    "BYTES",
    "DEPENDENCY_STEPS",
    "Domain",
    "Expr",
    "INSTRUCTIONS",
    "LOGICAL_QUBITS",
    "MEASUREMENTS",
    "ONE",
    "PHYSICAL_QUBITS",
    "PHYSICAL_QUBIT_SECONDS",
    "RESETS",
    "SECONDS",
    "SEMANTIC_OPS",
    "SHOTS",
    "Symbol",
    "Unit",
    "absolute",
    "add",
    "as_expr",
    "bounded_product",
    "bounded_sum",
    "ceiling",
    "ceiling_divide",
    "comparison",
    "conjunction",
    "contracted_function",
    "disjunction",
    "divisible",
    "exponential",
    "floor",
    "floor_divide",
    "literal",
    "log2",
    "logarithm",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "piecewise",
    "power",
    "recurrence",
    "scoped_symbol",
    "square_root",
    "symbol",
]
