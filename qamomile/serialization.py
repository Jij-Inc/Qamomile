"""Canonical Protobuf serialization for QCore images and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

from google.protobuf.message import DecodeError

from ._native import compiler as native
from .effects import CallableContract, CallableKind, Capability, Effect, Requirement
from .errors import Diagnostic, SerializationError, TypeCheckError, VerificationError
from .ir import (
    Block,
    Constant,
    FrozenMapping,
    Function,
    Module,
    Operation,
    Parameter,
    PortableValue,
    Program,
    Region,
    RegionRef,
    SourceSpan,
    Terminator,
    Value,
)
from .proto import qcore_pb2 as wire
from .types import (
    ArrayType,
    FixedType,
    FunctionType,
    MapType,
    ProductType,
    QuantumType,
    RecordType,
    ScalarType,
    StagedType,
    TypeSpec,
    WordType,
    array_type,
)

SCHEMA_MAJOR = 5
SCHEMA_MINOR = 0
_MODULE_KIND = "qamomile.module"
_PROGRAM_KIND = "qamomile.program"
_ANALYSIS_KIND = "qamomile.analysis"
_SPECIALIZATION_KIND = "qamomile.specialization"
_TARGET_ADJOINT_KIND = "qamomile.target-adjoint"


@dataclass(frozen=True, slots=True)
class _TargetProgramView:
    """Read-only decoded view of one verified target-only QCore artifact."""

    module_name: str
    functions: tuple[Function, ...]
    constants: tuple[Constant, ...]
    entrypoint: str
    const_bindings: tuple[tuple[str, PortableValue], ...]

    @property
    def entry(self) -> Function:
        """Return the target artifact's closed entrypoint definition."""

        for function in self.functions:
            if function.symbol == self.entrypoint:
                return function
        raise VerificationError("Q1015: target artifact entrypoint is missing")

    def inspect(self) -> str:
        """Return a stable diagnostic rendering without persistence authority."""

        bindings = ", ".join(f"{name}={value!r}" for name, value in self.const_bindings)
        header = f"target-adjoint @{self.entrypoint}"
        if bindings:
            header += f" [{bindings}]"
        return (
            header
            + "\n\n"
            + "\n\n".join(function.inspect() for function in self.functions)
        )


def encode_module(module: Module) -> bytes:
    """Encode a verified Module deterministically.

    Args:
        module (Module): Verified source module to persist.

    Returns:
        bytes: Canonical integrity-sealed Module bytes.

    Raises:
        SerializationError: If native sealing rejects the encoded package.
    """

    package = _encode_package(
        artifact_kind=_MODULE_KIND,
        module_name=module.module_name,
        functions=module.functions,
        constants=module.constants,
        entrypoint="",
        bindings=(),
    )
    return _seal_package(package)


def decode_module(payload: bytes) -> Module:
    """Decode, integrity-check, and verify a canonical Module.

    Args:
        payload (bytes): Canonical integrity-sealed Module bytes.

    Returns:
        Module: Verified source module.

    Raises:
        SerializationError: If the payload is malformed or noncanonical.
        VerificationError: If native verification rejects the Module.
    """

    package = _parse_package(payload, _MODULE_KIND)
    try:
        return _decode_module_package(package, payload)
    except VerificationError as error:
        raise SerializationError(str(error)) from error


def _decode_module_package(
    package: wire.QCorePackageProto,
    payload: bytes,
) -> Module:
    """Decode a parsed package as one canonical Module.

    Args:
        package (wire.QCorePackageProto): Parsed Module wire message.
        payload (bytes): Original sealed bytes used for canonicality checking.

    Returns:
        Module: Verified source module.

    Raises:
        SerializationError: If re-encoding disagrees with ``payload``.
        VerificationError: If the Module violates semantic invariants.
    """

    module = Module(
        module_name=package.module_name,
        functions=tuple(_decode_function(value) for value in package.functions),
        constants=tuple(_decode_constant(value) for value in package.constants),
    )
    if encode_module(module) != payload:
        raise SerializationError("Q4013: Module is not canonically encoded")
    return module


def encode_program(program: Program) -> bytes:
    """Encode and natively verify a closed Program deterministically.

    Args:
        program (Program): Closed verified program to persist.

    Returns:
        bytes: Canonical integrity-sealed Program bytes.

    Raises:
        VerificationError: If native verification rejects the Program or its
            Python projection.
        SerializationError: If native sealing rejects the encoded package.
    """

    payload = _seal_package(_encode_program_package(program))
    try:
        header = _verify_native_program(payload)
        _validate_native_program_projection(program, header)
    except ValueError as error:
        raise VerificationError(str(error)) from error
    return payload


def _encode_program_package(program: Program) -> wire.QCorePackageProto:
    """Project one program facade into its existing unsealed wire message."""

    return _encode_package(
        artifact_kind=_PROGRAM_KIND,
        module_name=program.module_name,
        functions=program.functions,
        constants=program.constants,
        entrypoint=program.entrypoint,
        bindings=program.const_bindings,
    )


def replace_program_intent(
    program: Program,
    values: Mapping[str, PortableValue],
) -> Program:
    """Re-seal one program with intent rows merged onto its entry function.

    Args:
        program (Program): Verified closed program.
        values (Mapping[str, PortableValue]): Rows to replace or append.

    Returns:
        Program: Natively re-verified program with the merged entry intent.

    Raises:
        TypeError: If a value is not a portable literal.
        VerificationError: If native verification rejects the merged rows.
    """

    merged = list(program.entry.intent)
    for name, value in values.items():
        probe = wire.LiteralProto()
        try:
            encode_literal(value, probe)
        except SerializationError as error:
            raise TypeError(f"intent {name!r} is not a portable value") from error
        position = next(
            (index for index, (existing, _) in enumerate(merged) if existing == name),
            None,
        )
        if position is None:
            merged.append((name, value))
        else:
            merged[position] = (name, value)
    package = _encode_program_package(program)
    for function in package.functions:
        if function.symbol == program.entrypoint:
            del function.intent[:]
            _encode_intent(tuple(merged), function.intent)
    return decode_program(_seal_package(package))


def encode_specialization_request(program: Program) -> bytes:
    """Encode a selected fully bound program before native specialization.

    Args:
        program: Locally verified selected program retaining symbolic dimensions
            and source cleanup obligations.

    Returns:
        Canonical internal request bytes accepted only by the Rust
        specialization typestate.
    """

    package = _encode_package(
        artifact_kind=_SPECIALIZATION_KIND,
        module_name=program.module_name,
        functions=program.functions,
        constants=program.constants,
        entrypoint=program.entrypoint,
        bindings=program.const_bindings,
    )
    return _seal_package(package)


def encode_selection_request(
    module: Module,
    entrypoint: str,
    bindings: tuple[tuple[str, PortableValue], ...],
) -> bytes:
    """Encode a complete module and partial source-facing selection.

    Args:
        module: Verified source module before reachability linking.
        entrypoint: Local or fully qualified source selector.
        bindings: Portable values explicitly supplied by the caller.

    Returns:
        Canonical internal bytes accepted by the Rust selection typestate.
    """

    package = _encode_package(
        artifact_kind=_SPECIALIZATION_KIND,
        module_name=module.module_name,
        functions=module.functions,
        constants=module.constants,
        entrypoint=entrypoint,
        bindings=bindings,
    )
    return _seal_package(package)


def encode_analysis_request(module: Module, entrypoint: str) -> bytes:
    """Encode a selected symbolic program family for native analysis.

    Args:
        module (Module): Verified source module containing the selected
            entrypoint and its static dependencies.
        entrypoint (str): Local or fully qualified callable name to analyze.

    Returns:
        bytes: Canonical integrity-sealed analysis artifact. Const parameters
            intentionally remain unbound and symbolic.

    Raises:
        SerializationError: If the canonical package exceeds a wire budget.
    """

    package = _encode_package(
        artifact_kind=_ANALYSIS_KIND,
        module_name=module.module_name,
        functions=module.functions,
        constants=module.constants,
        entrypoint=entrypoint,
        bindings=(),
    )
    return _seal_package(package)


def decode_program(payload: bytes) -> Program:
    """Decode, integrity-check, and verify a canonical closed Program.

    Args:
        payload (bytes): Canonical integrity-sealed Program bytes.

    Returns:
        Program: Verified closed program.

    Raises:
        SerializationError: If the payload is malformed or noncanonical.
        VerificationError: If native verification rejects the Program.
    """

    package = _parse_package(payload, _PROGRAM_KIND)
    try:
        header = _verify_native_program(payload)
    except ValueError as error:
        raise VerificationError(str(error)) from error
    return _decode_program_package(package, payload, header)


def _verify_native_program(payload: object) -> object:
    """Inspect a retained verified owner or verify portable source bytes.

    Args:
        payload (object): Native verified owner or canonical ``Program`` bytes.

    Returns:
        object: The native verified header and callable-contract projection.

    Raises:
        ValueError: If the installed compiler is incomplete or rejects the
            program, including an invalid cleanup proof.
    """

    inspect = getattr(native, "inspect_program", None)
    if not callable(inspect):
        raise ValueError("Q4002: native program verification is unavailable")
    return inspect(payload)


def _validate_native_program_projection(
    program: Program,
    header: object,
) -> tuple[Function, ...]:
    """Fail closed when a native verified-program header violates its ABI."""

    try:
        digest_text = getattr(header, "package_digest")
        digest = bytes.fromhex(digest_text)
        valid = (
            getattr(header, "verified") is True
            and type(getattr(header, "schema_major")) is int
            and getattr(header, "schema_major") == SCHEMA_MAJOR
            and type(getattr(header, "schema_minor")) is int
            and getattr(header, "schema_minor") == SCHEMA_MINOR
            and getattr(header, "module_name") == program.module_name
            and getattr(header, "entrypoint") == program.entrypoint
            and type(getattr(header, "function_count")) is int
            and getattr(header, "function_count") == len(program.functions)
            and type(getattr(header, "constant_count")) is int
            and getattr(header, "constant_count") == len(program.constants)
            and type(getattr(header, "type_count")) is int
            and getattr(header, "type_count") > 0
            and type(getattr(header, "operation_count")) is int
            and getattr(header, "operation_count") >= 0
            and isinstance(digest_text, str)
            and len(digest_text) == 64
            and len(digest) == 32
            and digest_text == digest.hex()
        )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise VerificationError(
            "Q4002: native program verification returned a malformed header"
        )
    return _functions_with_native_contracts(program.functions, header)


def _decode_program_package(
    package: wire.QCorePackageProto,
    payload: bytes,
    header: object,
) -> Program:
    """Decode a parsed package as one canonical closed Program.

    Args:
        package (wire.QCorePackageProto): Parsed Program wire message.
        payload (bytes): Original sealed bytes used for canonicality checking.
        header (object): Native verified Program projection.

    Returns:
        Program: Verified closed program.

    Raises:
        SerializationError: If re-encoding disagrees with ``payload``.
        VerificationError: If the native projection violates its ABI.
    """

    functions = tuple(_decode_function(value) for value in package.functions)
    program = object.__new__(Program)
    object.__setattr__(program, "_owner", None)
    object.__setattr__(program, "_owner_lock", RLock())
    object.__setattr__(program, "module_name", package.module_name)
    object.__setattr__(program, "functions", functions)
    object.__setattr__(
        program,
        "constants",
        tuple(_decode_constant(value) for value in package.constants),
    )
    object.__setattr__(program, "entrypoint", package.entrypoint)
    object.__setattr__(
        program,
        "const_bindings",
        tuple(
            (binding.name, _decode_literal(binding.value))
            for binding in package.const_bindings
        ),
    )
    object.__setattr__(
        program,
        "functions",
        _validate_native_program_projection(program, header),
    )
    if _seal_package(_encode_program_package(program)) != payload:
        raise SerializationError("Q4014: Program is not canonically encoded")
    return program


def decode_target_adjoint(payload: bytes) -> _TargetProgramView:
    """Decode a Rust-verified target-adjoint artifact as a read-only view.

    Args:
        payload: Canonical bytes produced by the native target lowering pass.

    Returns:
        A non-persistent view suitable only for immediate target emission.

    Raises:
        SerializationError: If the envelope, kind, digest, or Python projection
            is malformed.
        VerificationError: If native target-artifact verification fails.
    """

    package = _parse_sealed_package(payload)
    if package.artifact_kind != _TARGET_ADJOINT_KIND:
        raise SerializationError(
            f"Q4011: expected {_TARGET_ADJOINT_KIND!r}, got {package.artifact_kind!r}"
        )
    inspect = getattr(native, "inspect_target_adjoint", None)
    if not callable(inspect):
        raise VerificationError("Q4002: native target verification is unavailable")
    try:
        header = inspect(payload)
        functions = _functions_with_native_contracts(
            tuple(_decode_function(value) for value in package.functions),
            header,
        )
    except ValueError as error:
        raise VerificationError(str(error)) from error
    return _TargetProgramView(
        module_name=package.module_name,
        functions=functions,
        constants=tuple(_decode_constant(value) for value in package.constants),
        entrypoint=package.entrypoint,
        const_bindings=tuple(
            (binding.name, _decode_literal(binding.value))
            for binding in package.const_bindings
        ),
    )


def decode_image(payload: bytes) -> Module | Program:
    """Decode a canonical QCore image using its integrity-checked wire kind."""

    package = _parse_package(payload, None)
    if package.artifact_kind == _MODULE_KIND:
        try:
            return _decode_module_package(package, payload)
        except VerificationError as error:
            raise SerializationError(str(error)) from error
    try:
        header = _verify_native_program(payload)
    except ValueError as error:
        raise VerificationError(str(error)) from error
    return _decode_program_package(package, payload, header)


def _functions_with_native_contracts(
    functions: tuple[Function, ...],
    header: object,
) -> tuple[Function, ...]:
    """Install compiler-derived contracts from one verified native header."""

    try:
        projections = tuple(getattr(header, "contracts"))
        contracts = {
            getattr(projection, "symbol"): CallableContract(
                CallableKind(getattr(projection, "kind")),
                frozenset(Effect(value) for value in getattr(projection, "effects")),
                frozenset(
                    Capability(value) for value in getattr(projection, "capabilities")
                ),
                frozenset(
                    Requirement(value) for value in getattr(projection, "requirements")
                ),
            )
            for projection in projections
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise VerificationError(
            "Q4002: native verification returned malformed callable contracts"
        ) from error
    if len(contracts) != len(projections) or contracts.keys() != {
        function.symbol for function in functions
    }:
        raise VerificationError(
            "Q4002: native verification returned incomplete callable contracts"
        )
    return tuple(
        replace(function, contract=contracts[function.symbol]) for function in functions
    )


def encode_type(type_spec: TypeSpec, target: wire.TypeProto) -> None:
    """Encode one semantic type into a wire message."""

    if isinstance(type_spec, ScalarType):
        target.kind = wire.TypeProto.SCALAR
        target.name = type_spec.name
        return
    if isinstance(type_spec, ArrayType):
        target.kind = wire.TypeProto.ARRAY
        target.name = "Array"
        encode_type(type_spec.element, target.arguments.add())
        for dimension in type_spec.shape:
            encoded = target.dimensions.add()
            if isinstance(dimension, int):
                encoded.fixed = dimension
            else:
                encoded.symbol = dimension
        return
    if isinstance(type_spec, MapType):
        target.kind = wire.TypeProto.ARRAY
        target.name = "Array"
        entry = RecordType(
            "__qamomile_mapping_entry__",
            (("key", type_spec.key), ("value", type_spec.value)),
        )
        encode_type(entry, target.arguments.add())
        encoded = target.dimensions.add()
        if isinstance(type_spec.size, int):
            encoded.fixed = type_spec.size
        else:
            encoded.symbol = type_spec.size
        return
    if isinstance(type_spec, WordType):
        target.kind = wire.TypeProto.ARRAY
        target.name = type_spec.kind
        encode_type(ScalarType("Bit"), target.arguments.add())
        encoded = target.dimensions.add()
        if isinstance(type_spec.width, int):
            encoded.fixed = type_spec.width
        else:
            encoded.symbol = type_spec.width
        return
    if isinstance(type_spec, FixedType):
        target.kind = wire.TypeProto.FIXED
        target.name = "UFixed"
        for width in (type_spec.integer_bits, type_spec.fractional_bits):
            dimension = target.dimensions.add()
            if isinstance(width, int):
                dimension.fixed = width
            else:
                dimension.symbol = width
        return
    if isinstance(type_spec, ProductType):
        target.kind = wire.TypeProto.PRODUCT
        for item in type_spec.elements:
            encode_type(item, target.arguments.add())
        return
    if isinstance(type_spec, RecordType):
        target.kind = wire.TypeProto.RECORD
        target.name = type_spec.name
        for name, item in type_spec.fields:
            target.field_names.append(name)
            encode_type(item, target.arguments.add())
        return
    if isinstance(type_spec, QuantumType):
        target.kind = wire.TypeProto.QUANTUM
        encode_type(type_spec.inner, target.arguments.add())
        return
    if isinstance(type_spec, StagedType):
        target.kind = wire.TypeProto.STAGED
        target.name = type_spec.stage
        encode_type(type_spec.inner, target.arguments.add())
        return
    if isinstance(type_spec, FunctionType):
        target.kind = wire.TypeProto.FUNCTION
        target.field_names.extend(type_spec.parameter_names)
        for input_type in type_spec.inputs:
            encode_type(input_type, target.arguments.add())
        encode_type(type_spec.output, target.arguments.add())
        return
    raise SerializationError(f"Q4001: cannot encode semantic type {type_spec!r}")


def decode_type(source: wire.TypeProto) -> TypeSpec:
    """Decode one semantic type from a wire message."""

    try:
        return _decode_type(source)
    except TypeCheckError as error:
        raise SerializationError(f"Q4002: invalid semantic type: {error}") from error


def _decode_type(source: wire.TypeProto) -> TypeSpec:
    """Decode one type after converting constructor failures at the boundary."""

    kind = source.kind
    if kind == wire.TypeProto.SCALAR:
        return ScalarType(source.name)
    if kind == wire.TypeProto.ARRAY:
        _expect_arity(source, 1)
        dimensions: list[int | str] = []
        for dimension in source.dimensions:
            selected = dimension.WhichOneof("value")
            if selected == "fixed":
                dimensions.append(int(dimension.fixed))
            elif selected == "symbol":
                dimensions.append(dimension.symbol)
            else:
                raise SerializationError("Q4002: array dimension has no value")
        element = decode_type(source.arguments[0])
        shape = tuple(dimensions)
        if (
            source.name == "Array"
            and len(shape) == 1
            and isinstance(element, RecordType)
            and element.name == "__qamomile_mapping_entry__"
            and tuple(name for name, _ in element.fields) == ("key", "value")
        ):
            return MapType(element.fields[0][1], element.fields[1][1], shape[0])
        if source.name == "Array":
            return array_type(element, shape)
        if (
            source.name in {"Bits", "UInt"}
            and element == ScalarType("Bit")
            and len(shape) == 1
        ):
            return WordType(source.name, shape[0])
        raise SerializationError(f"Q4002: unknown array encoding {source.name!r}")
    if kind == wire.TypeProto.FIXED:
        if source.name != "UFixed" or len(source.dimensions) != 2:
            raise SerializationError("Q4002: malformed UFixed encoding")
        widths: list[int | str] = []
        for dimension in source.dimensions:
            selected = dimension.WhichOneof("value")
            if selected == "fixed":
                widths.append(int(dimension.fixed))
            elif selected == "symbol":
                widths.append(dimension.symbol)
            else:
                raise SerializationError("Q4002: UFixed width has no value")
        return FixedType(widths[0], widths[1])
    if kind == wire.TypeProto.PRODUCT:
        return ProductType(tuple(decode_type(item) for item in source.arguments))
    if kind == wire.TypeProto.RECORD:
        if len(source.field_names) != len(source.arguments):
            raise SerializationError("Q4003: record field/type arity mismatch")
        return RecordType(
            source.name,
            tuple(
                (name, decode_type(item))
                for name, item in zip(
                    source.field_names,
                    source.arguments,
                    strict=True,
                )
            ),
        )
    if kind == wire.TypeProto.QUANTUM:
        _expect_arity(source, 1)
        return QuantumType(decode_type(source.arguments[0]))
    if kind == wire.TypeProto.STAGED:
        _expect_arity(source, 1)
        return StagedType(source.name, decode_type(source.arguments[0]))
    if kind == wire.TypeProto.FUNCTION:
        if not source.arguments:
            raise SerializationError("Q4003: function type expects an output type")
        return FunctionType(
            tuple(decode_type(item) for item in source.arguments[:-1]),
            decode_type(source.arguments[-1]),
            tuple(source.field_names),
        )
    raise SerializationError(f"Q4004: unknown semantic type kind {kind}")


def encode_literal(value: PortableValue, target: wire.LiteralProto) -> None:
    """Encode one closed portable literal."""

    if value is None:
        target.unit_value = True
    elif isinstance(value, bool):
        target.bool_value = value
    elif isinstance(value, int):
        if -(2**63) <= value < 2**63:
            target.int_value = value
        else:
            target.bigint_value = _encode_bigint(value)
    elif isinstance(value, float):
        target.real_value = value
    elif isinstance(value, complex):
        target.complex_value.real = value.real
        target.complex_value.imaginary = value.imag
    elif isinstance(value, str):
        target.string_value = value
    elif isinstance(value, bytes):
        target.bytes_value = value
    elif isinstance(value, tuple):
        # Selecting an empty protobuf message does not happen implicitly when
        # its repeated field remains empty.  Mark the oneof arm explicitly so
        # ``()`` has a representation distinct from a missing literal.
        target.tuple_value.SetInParent()
        for item in value:
            encode_literal(item, target.tuple_value.values.add())
    elif isinstance(value, FrozenMapping):
        target.mapping_value.SetInParent()
        for key, item in value.entries:
            entry = target.mapping_value.entries.add()
            entry.key = key
            encode_literal(item, entry.value)
    else:
        raise SerializationError(f"Q4005: cannot encode literal {value!r}")


def decode_literal(source: wire.LiteralProto) -> PortableValue:
    """Decode one closed portable literal."""

    return _decode_literal(source)


def _decode_literal(source: wire.LiteralProto) -> PortableValue:
    """Internal recursive portable literal decoder."""

    selected = source.WhichOneof("value")
    if selected == "unit_value":
        return None
    if selected == "bool_value":
        return source.bool_value
    if selected == "int_value":
        return source.int_value
    if selected == "bigint_value":
        return _decode_bigint(bytes(source.bigint_value))
    if selected == "real_value":
        return source.real_value
    if selected == "complex_value":
        return complex(
            source.complex_value.real,
            source.complex_value.imaginary,
        )
    if selected == "string_value":
        return source.string_value
    if selected == "bytes_value":
        return bytes(source.bytes_value)
    if selected == "tuple_value":
        return tuple(_decode_literal(item) for item in source.tuple_value.values)
    if selected == "mapping_value":
        entries = tuple(
            (entry.key, _decode_literal(entry.value))
            for entry in source.mapping_value.entries
        )
        keys = tuple(key for key, _ in entries)
        if len(keys) != len(set(keys)) or tuple(sorted(keys)) != keys:
            raise SerializationError("Q4006: mapping literal is not canonical")
        return FrozenMapping(entries)
    raise SerializationError("Q4007: portable literal has no selected value")


def _encode_bigint(value: int) -> bytes:
    """Encode an arbitrary-precision integer as minimal signed bytes."""

    width = max(1, (value.bit_length() + 8) // 8)
    encoded = value.to_bytes(width, "big", signed=True)
    while len(encoded) > 1:
        if encoded[0] == 0 and encoded[1] < 0x80:
            encoded = encoded[1:]
        elif encoded[0] == 0xFF and encoded[1] >= 0x80:
            encoded = encoded[1:]
        else:
            break
    return encoded


def _decode_bigint(encoded: bytes) -> int:
    """Decode and validate a canonical arbitrary-precision integer."""

    if not encoded:
        raise SerializationError("Q4008: bigint literal is empty")
    value = int.from_bytes(encoded, "big", signed=True)
    if _encode_bigint(value) != encoded or -(2**63) <= value < 2**63:
        raise SerializationError("Q4008: bigint literal is not canonical")
    return value


def _encode_package(
    *,
    artifact_kind: str,
    module_name: str,
    functions: tuple[Function, ...],
    constants: tuple[Constant, ...],
    entrypoint: str,
    bindings: tuple[tuple[str, PortableValue], ...],
) -> wire.QCorePackageProto:
    """Build an unsealed canonical package message."""

    package = wire.QCorePackageProto(
        schema_major=SCHEMA_MAJOR,
        schema_minor=SCHEMA_MINOR,
        artifact_kind=artifact_kind,
        module_name=module_name,
        entrypoint=entrypoint,
    )
    for function in sorted(functions, key=lambda item: item.symbol):
        _encode_function(function, package.functions.add())
    for constant in sorted(constants, key=lambda item: item.name):
        encoded = package.constants.add(name=constant.name)
        encode_type(constant.type, encoded.type)
        encode_literal(constant.value, encoded.value)
    for name, value in sorted(bindings, key=lambda item: item[0]):
        encoded = package.const_bindings.add(name=name)
        encode_literal(value, encoded.value)
    return package


def _seal_package(package: wire.QCorePackageProto) -> bytes:
    """Attach the exact package-integrity digest through Rust."""

    package.package_digest = b""
    sealer = getattr(native, "seal_qcore_package", None)
    if not callable(sealer):
        raise SerializationError(
            Diagnostic("Q4002", "native QCore package sealing is unavailable")
        )
    try:
        payload = sealer(package.SerializeToString(deterministic=True))
    except ValueError as error:
        raise _native_serialization_error(str(error)) from error
    if not isinstance(payload, bytes):
        raise SerializationError(
            Diagnostic("Q4008", "native QCore package sealer returned malformed data")
        )
    return payload


def _parse_package(
    payload: bytes,
    expected_kind: str | None,
) -> wire.QCorePackageProto:
    """Parse a package and fail closed on version, kind, or digest mismatch."""

    package = _parse_sealed_package(payload)
    if package.artifact_kind not in {_MODULE_KIND, _PROGRAM_KIND}:
        raise SerializationError(
            f"Q4011: unknown QCore image kind {package.artifact_kind!r}"
        )
    if expected_kind is not None and package.artifact_kind != expected_kind:
        raise SerializationError(
            f"Q4011: expected {expected_kind!r}, got {package.artifact_kind!r}"
        )
    return package


def _parse_sealed_package(payload: bytes) -> wire.QCorePackageProto:
    """Parse and integrity-check one canonical QCore artifact."""

    header = _inspect_native_envelope(payload)
    package = wire.QCorePackageProto()
    try:
        package.ParseFromString(payload)
    except (DecodeError, MemoryError, RecursionError) as error:
        raise SerializationError(
            Diagnostic("Q4008", "malformed QCore package")
        ) from error
    if (
        package.schema_major,
        package.schema_minor,
        package.artifact_kind,
        bytes(package.package_digest),
    ) != header:
        raise SerializationError(
            Diagnostic("Q4008", "native QCore envelope projection is malformed")
        )
    return package


def _inspect_native_envelope(payload: bytes) -> tuple[int, int, str, bytes]:
    """Validate one package before Python Protobuf allocation."""

    inspect = getattr(native, "inspect_qcore_envelope", None)
    if not callable(inspect):
        raise SerializationError(
            Diagnostic("Q4002", "native QCore envelope inspection is unavailable")
        )
    try:
        result = inspect(payload)
    except ValueError as error:
        raise _native_serialization_error(str(error)) from error
    if (
        type(result) is not tuple
        or len(result) != 4
        or type(result[0]) is not int
        or type(result[1]) is not int
        or type(result[2]) is not str
        or not isinstance(result[3], bytes)
        or len(result[3]) != 32
    ):
        raise SerializationError(
            Diagnostic("Q4008", "native QCore envelope projection is malformed")
        )
    return result[0], result[1], result[2], result[3]


def _native_serialization_error(message: str) -> SerializationError:
    """Preserve a native persistence diagnostic in the Python hierarchy."""

    code, separator, detail = message.partition(": ")
    if separator and len(code) == 5 and code.startswith("Q") and code[1:].isdigit():
        return SerializationError(Diagnostic(code, detail))
    return SerializationError(
        Diagnostic("Q4008", f"native QCore persistence failed: {message}")
    )


def _encode_function(function: Function, target: wire.FunctionProto) -> None:
    """Encode one QCore callable definition."""

    target.symbol = function.symbol
    target.asserted_kind = (
        function.asserted_kind.value if function.asserted_kind is not None else ""
    )
    target.external = function.external
    _encode_span(function.span, target.span)
    _encode_intent(function.intent, target.intent)
    encode_type(function.result_type, target.result_type)
    for parameter in function.parameters:
        encoded = target.parameters.add(
            name=parameter.name,
            has_default=parameter.has_default,
        )
        _encode_value(parameter.value, encoded.value)
        if parameter.has_default:
            encode_literal(parameter.default, encoded.default_value)
    _encode_block(function.body, target.body)


def _decode_function(source: wire.FunctionProto) -> Function:
    """Decode one QCore callable definition."""

    try:
        asserted_kind = (
            CallableKind(source.asserted_kind) if source.asserted_kind else None
        )
    except ValueError as error:
        raise SerializationError("Q4013: unknown callable contract enum") from error
    return Function(
        symbol=source.symbol,
        parameters=tuple(
            Parameter(
                value.name,
                _decode_value(value.value),
                _decode_literal(value.default_value) if value.has_default else None,
                value.has_default,
            )
            for value in source.parameters
        ),
        result_type=decode_type(source.result_type),
        body=_decode_block(source.body),
        span=_decode_span(source.span),
        asserted_kind=asserted_kind,
        external=source.external,
        intent=_decode_intent(source.intent),
    )


def _encode_block(block: Block, target: wire.BlockProto) -> None:
    """Encode one QCore block recursively."""

    for argument in block.arguments:
        _encode_value(argument, target.arguments.add())
    for operation in block.operations:
        _encode_operation(operation, target.operations.add())
    _encode_terminator(block.terminator, target.terminator)


def _decode_block(source: wire.BlockProto) -> Block:
    """Decode one QCore block recursively."""

    return Block(
        tuple(_decode_value(value) for value in source.arguments),
        tuple(_decode_operation(value) for value in source.operations),
        _decode_terminator(source.terminator),
    )


def _encode_operation(operation: Operation, target: wire.OperationProto) -> None:
    """Encode one QCore operation recursively."""

    target.opcode = operation.opcode
    target.operands.extend(operation.operands)
    _encode_span(operation.span, target.span)
    for result in operation.results:
        _encode_value(result, target.results.add())
    for name, value in operation.attributes:
        encoded = target.attributes.add(name=name)
        encode_literal(value, encoded.value)
    _encode_intent(operation.intent, target.intent)
    for region in operation.regions:
        encoded_region = target.regions.add()
        for block in region.blocks:
            _encode_block(block, encoded_region.blocks.add())


def _decode_operation(source: wire.OperationProto) -> Operation:
    """Decode one QCore operation recursively."""

    return Operation(
        opcode=source.opcode,
        operands=tuple(source.operands),
        results=tuple(_decode_value(value) for value in source.results),
        attributes=tuple(
            (attribute.name, _decode_literal(attribute.value))
            for attribute in source.attributes
        ),
        regions=tuple(
            Region(tuple(_decode_block(block) for block in region.blocks))
            for region in source.regions
        ),
        span=_decode_span(source.span),
        intent=_decode_intent(source.intent),
    )


def _encode_intent(
    intent: tuple[tuple[str, PortableValue], ...],
    target: Any,
) -> None:
    """Encode untrusted intent rows without interpreting them."""

    for name, value in intent:
        encoded = target.add(name=name)
        encode_literal(value, encoded.value)


def _decode_intent(source: Any) -> tuple[tuple[str, PortableValue], ...]:
    """Decode untrusted intent rows without interpreting them."""

    return tuple((row.name, _decode_literal(row.value)) for row in source)


def _encode_terminator(
    terminator: Terminator,
    target: wire.TerminatorProto,
) -> None:
    """Encode one block terminator."""

    target.kind = terminator.kind
    target.operands.extend(terminator.operands)
    _encode_span(terminator.span, target.span)
    for name, value in terminator.attributes:
        encoded = target.attributes.add(name=name)
        encode_literal(value, encoded.value)


def _decode_terminator(source: wire.TerminatorProto) -> Terminator:
    """Decode one block terminator."""

    return Terminator(
        source.kind,
        tuple(source.operands),
        tuple(
            (attribute.name, _decode_literal(attribute.value))
            for attribute in source.attributes
        ),
        _decode_span(source.span),
    )


def _encode_value(value: Value, target: wire.ValueProto) -> None:
    """Encode one SSA value definition."""

    target.id = value.id
    target.hint = value.hint
    encode_type(value.type, target.type)
    if value.region is not None:
        target.has_region = True
        target.region.root = value.region.root
        target.region.projections.extend(value.region.projections)


def _decode_value(source: wire.ValueProto) -> Value:
    """Decode one SSA value definition."""

    region = None
    if source.has_region:
        region = RegionRef(source.region.root, tuple(source.region.projections))
    return Value(source.id, decode_type(source.type), source.hint, region)


def _decode_constant(source: wire.ConstantProto) -> Constant:
    """Decode one module constant."""

    return Constant(
        source.name,
        decode_type(source.type),
        _decode_literal(source.value),
    )


def _encode_span(span: SourceSpan, target: wire.SourceSpanProto) -> None:
    """Encode one normalized source location."""

    target.uri = span.uri
    target.start_line = span.start_line
    target.start_column = span.start_column
    target.end_line = span.end_line
    target.end_column = span.end_column


def _decode_span(source: wire.SourceSpanProto) -> SourceSpan:
    """Decode one normalized source location."""

    return SourceSpan(
        source.uri or "qmi://unknown",
        source.start_line or 1,
        source.start_column,
        source.end_line or source.start_line or 1,
        source.end_column,
    )


def _expect_arity(source: wire.TypeProto, expected: int) -> None:
    """Validate a recursive type constructor arity."""

    if len(source.arguments) != expected:
        raise SerializationError(
            Diagnostic(
                "Q4015",
                f"type {source.name or source.kind} expects {expected} "
                f"arguments, got {len(source.arguments)}",
            )
        )
