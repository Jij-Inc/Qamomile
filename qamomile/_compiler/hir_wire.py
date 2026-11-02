"""Private deterministic wire boundary for ephemeral typed source HIR."""

from __future__ import annotations

from .._native import compiler as native
from ..errors import CapabilityError, Diagnostic, SerializationError, VerificationError
from ..ir import Module, PortableValue, SourceSpan
from ..proto import hir_pb2 as hir_wire, qcore_pb2 as qcore_wire
from ..serialization import SCHEMA_MAJOR, SCHEMA_MINOR, encode_literal, encode_type
from .hir import HIRBlock, HIRExpression, HIRFunction, HIRModule, HIRStatement

_HIR_KIND = "qamomile.hir"


def encode_hir(module: HIRModule) -> bytes:
    """Encode one typed HIR compiler request deterministically.

    Args:
        module: Ephemeral typed source module produced by the Python parser.

    Returns:
        Integrity-sealed bytes for the private native HIR boundary.

    Raises:
        VerificationError: If the request exceeds the shared wire-size budget.
    """

    package = hir_wire.HIRPackageProto(
        schema_major=SCHEMA_MAJOR,
        schema_minor=SCHEMA_MINOR,
        artifact_kind=_HIR_KIND,
        module_name=module.module_name,
    )
    for function in sorted(module.functions, key=lambda item: item.symbol):
        _encode_function(function, package.functions.add())
    for constant in sorted(module.constants, key=lambda item: item.name):
        encoded = package.constants.add(name=constant.name)
        encode_type(constant.type, encoded.type)
        encode_literal(constant.value, encoded.value)
    sealer = getattr(native, "seal_hir_package", None)
    if not callable(sealer):
        raise CapabilityError(Diagnostic("Q7100", "native HIR sealing is unavailable"))
    try:
        payload = sealer(package.SerializeToString(deterministic=True))
    except ValueError as error:
        raise _native_hir_error(str(error)) from error
    if not isinstance(payload, bytes):
        raise VerificationError(
            Diagnostic("Q7003", "native HIR sealer returned malformed data")
        )
    return payload


def _native_hir_error(message: str) -> VerificationError:
    """Preserve one native HIR persistence diagnostic in Python."""

    code, separator, detail = message.partition(": ")
    if separator and len(code) == 5 and code.startswith("Q") and code[1:].isdigit():
        return VerificationError(Diagnostic(code, detail))
    return VerificationError(
        Diagnostic("Q7003", f"native HIR persistence failed: {message}")
    )


def lower_native_hir(module: HIRModule) -> Module:
    """Use authoritative native HIR lowering through the required compiler.

    Args:
        module: Typed source HIR ready for ownership-SSA lowering.

    Returns:
        A verified QCore module.

    Raises:
        CapabilityError: If the required compiler is unavailable, incomplete,
            or does not implement a valid HIR construct.
        VerificationError: If Rust rejects the HIR or its produced QCore.
    """

    payload = encode_hir(module)
    lower = getattr(native, "lower_hir", None)
    if not callable(lower):
        raise CapabilityError("Q7100: native HIR lowering is unavailable")
    try:
        return Module.from_bytes(bytes(lower(payload)))
    except SerializationError as error:
        raise VerificationError(str(error)) from error
    except ValueError as error:
        message = str(error)
        if message.startswith("Q7101:"):
            raise CapabilityError(message) from error
        raise VerificationError(message) from error


def _encode_function(
    function: HIRFunction,
    target: hir_wire.HIRFunctionProto,
) -> None:
    """Encode one HIR function declaration."""

    target.symbol = function.symbol
    encode_type(function.result_type, target.result_type)
    _encode_block(function.body, target.body)
    target.declared_kind = (
        function.declared_kind.value if function.declared_kind is not None else ""
    )
    _encode_span(function.span, target.span)
    target.external = function.external
    for name, value in function.intent:
        encode_literal(value, target.intent.add(name=name).value)
    for parameter in function.parameters:
        encoded = target.parameters.add(
            name=parameter.name,
            has_default=parameter.has_default,
        )
        encode_type(parameter.type, encoded.type)
        if parameter.has_default:
            encode_literal(parameter.default, encoded.default_value)


def _encode_block(block: HIRBlock, target: hir_wire.HIRBlockProto) -> None:
    """Encode one lexical HIR block in source order."""

    target.SetInParent()
    for statement in block.statements:
        _encode_statement(statement, target.statements.add())


def _encode_statement(
    statement: HIRStatement,
    target: hir_wire.HIRStatementProto,
) -> None:
    """Encode one typed HIR statement."""

    target.opcode = statement.opcode
    if statement.expression is not None:
        _encode_expression(statement.expression, target.expression)
    target.targets.extend(statement.targets)
    _encode_attributes(statement.attributes, target.attributes)
    for region in statement.regions:
        _encode_block(region, target.regions.add())
    _encode_span(statement.span, target.span)
    for result_type in statement.result_types:
        encode_type(result_type, target.result_types.add())


def _encode_expression(
    expression: HIRExpression,
    target: hir_wire.HIRExpressionProto,
) -> None:
    """Encode one typed HIR expression tree."""

    target.opcode = expression.opcode
    encode_type(expression.type, target.type)
    for operand in expression.operands:
        _encode_expression(operand, target.operands.add())
    _encode_attributes(expression.attributes, target.attributes)
    _encode_span(expression.span, target.span)
    for region in expression.regions:
        _encode_expression(region, target.regions.add())


def _encode_attributes(
    attributes: tuple[tuple[str, PortableValue], ...],
    target: object,
) -> None:
    """Encode canonical HIR attributes through QCore's literal algebra."""

    repeated = target
    for name, value in sorted(attributes, key=lambda item: item[0]):
        encoded = repeated.add(name=name)  # type: ignore[attr-defined]
        encode_literal(value, encoded.value)


def _encode_span(span: SourceSpan, target: qcore_wire.SourceSpanProto) -> None:
    """Encode one normalized source span."""

    target.uri = span.uri
    target.start_line = span.start_line
    target.start_column = span.start_column
    target.end_line = span.end_line
    target.end_column = span.end_column
