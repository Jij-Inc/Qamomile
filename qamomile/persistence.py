"""Optional portable storage of compiled programs, separate from execution.

SDK codecs are loaded only on explicit export/import. The Rust artifact wire
schema remains the authority for integrity, ABI and resource consistency.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from google.protobuf.message import DecodeError

from ._execution_payload import ExecutablePayload
from ._native import compiler as native
from .compilation import (
    ARTIFACT_FAMILY_FORMAT,
    LAUNCH_MATERIALIZER_FORMAT,
    CompilationReport,
    CompiledProgram,
    ExecutionContract,
    ParameterABI,
    ParameterSlot,
    ResultABI,
    _ArtifactFamilyPayload,
    _ArtifactFamilyVariant,
    _encode_compilation_metadata,
    _LaunchMaterializerPayload,
    _validate_execution_interface,
)
from .effects import Requirement
from .errors import ArtifactError, Diagnostic, ExecutionError, QamomileError
from .ir import FrozenMapping, Program
from .proto import qcore_pb2 as wire
from .resources.report import ResourceStudy
from .serialization import (
    SCHEMA_MAJOR,
    SCHEMA_MINOR,
    decode_literal,
    decode_type,
)


def serialize(compiled: CompiledProgram) -> bytes:
    """Serialize and integrity-seal this artifact.

    Args:
        compiled (CompiledProgram): Live program whose backend supports export.

    Returns:
        bytes: Portable, native-validated artifact envelope.

    Raises:
        ArtifactError: If the artifact cannot be sealed.
        ExecutionError: If the backend executable cannot be exported.
        TypeError: If the backend does not support optional persistence.
    """

    encoded = _encode_compilation_metadata(compiled)
    encoded.schema_major = SCHEMA_MAJOR
    encoded.schema_minor = SCHEMA_MINOR
    encoded.payload = export_payload(compiled)
    encoded.report.stages.extend(compiled.report.stages)
    encoded.report.assumptions.extend(compiled.report.assumptions)
    encoded.report.warnings.extend(compiled.report.warnings)
    encoded.report.resource_study = compiled.report.resources.to_bytes()
    return _seal_artifact(encoded)


def deserialize(
    payload: bytes,
    *,
    payload_loader: Callable[[str, bytes], ExecutablePayload] | None = None,
) -> CompiledProgram:
    """Decode and verify an integrity-sealed CompiledProgram.

    Args:
        payload (bytes): Canonical integrity-sealed CompiledProgram bytes.
        payload_loader (Callable | None): Optional custom-backend codec, called
            only for formats not covered by built-in codecs. Defaults to None.

    Returns:
        CompiledProgram: Live executable restored after native archive verification.

    Raises:
        ArtifactError: If the artifact is malformed, noncanonical, or
            inconsistent with its resource and ABI metadata.
        ExecutionError: If the SDK payload cannot be restored.
        ImportError: If the selected SDK is unavailable.
        TypeError: If a custom loader does not return a live executable.
    """

    header = _inspect_artifact(payload)
    encoded = wire.ArtifactProto()
    try:
        encoded.ParseFromString(payload)
    except (DecodeError, MemoryError, RecursionError) as error:
        raise ArtifactError("Q7004: malformed artifact") from error
    if (
        encoded.schema_major,
        encoded.schema_minor,
        encoded.format,
        encoded.target_fingerprint,
        bytes(encoded.artifact_digest),
    ) != header:
        raise ArtifactError(
            Diagnostic("Q7004", "native artifact projection is malformed")
        )
    try:
        contract = ExecutionContract(
            encoded.format,
            frozenset(Requirement(item) for item in encoded.execution_requirements),
        )
    except ValueError as error:
        raise ArtifactError("Q7013: unknown execution requirement") from error
    metadata_entries = tuple(
        (attribute.name, decode_literal(attribute.value))
        for attribute in encoded.metadata
    )
    if len(metadata_entries) != len({name for name, _ in metadata_entries}):
        raise ArtifactError("Q7023: artifact metadata keys must be unique")
    try:
        resources = ResourceStudy.from_bytes(bytes(encoded.report.resource_study))
    except ValueError as error:
        raise ArtifactError("Q7021: invalid resource study in artifact") from error
    report = CompilationReport(
        stages=tuple(encoded.report.stages),
        resources=resources,
        assumptions=tuple(encoded.report.assumptions),
        warnings=tuple(encoded.report.warnings),
    )
    parameters = ParameterABI(
        tuple(
            ParameterSlot(
                slot.name,
                decode_type(slot.type),
                (decode_literal(slot.default_value) if slot.has_default else None),
                slot.has_default,
            )
            for slot in encoded.parameter_abi.slots
        )
    )
    result = ResultABI(
        decode_type(encoded.result_abi.type),
        tuple(encoded.result_abi.source_order),
        encoded.result_abi.native_width,
    )
    _validate_execution_interface(
        resources, parameters, result, FrozenMapping(tuple(sorted(metadata_entries)))
    )
    return CompiledProgram(
        payload=_restore_payload(
            encoded.format, bytes(encoded.payload), payload_loader
        ),
        target_fingerprint=encoded.target_fingerprint,
        parameter_abi=parameters,
        result_abi=result,
        report=report,
        execution_contract=contract,
        metadata=FrozenMapping(tuple(sorted(metadata_entries))),
    )


def _seal_artifact(package: wire.ArtifactProto) -> bytes:
    """Seal one canonical target artifact through the native wire policy."""

    package.artifact_digest = b""
    sealer = getattr(native, "seal_artifact", None)
    if not callable(sealer):
        raise ArtifactError(
            Diagnostic("Q7004", "native artifact sealing is unavailable")
        )
    try:
        payload = sealer(package.SerializeToString(deterministic=True))
    except ValueError as error:
        raise _native_artifact_error(str(error)) from error
    if not isinstance(payload, bytes):
        raise ArtifactError(
            Diagnostic("Q7004", "native artifact sealer returned malformed data")
        )
    return payload


def _inspect_artifact(payload: bytes) -> tuple[int, int, str, str, bytes]:
    """Preflight one target artifact before Python Protobuf allocation."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ArtifactError(Diagnostic("Q7004", "artifact must be bytes-like"))
    payload = bytes(payload)
    inspect = getattr(native, "inspect_artifact", None)
    if not callable(inspect):
        raise ArtifactError(
            Diagnostic("Q7004", "native artifact inspection is unavailable")
        )
    try:
        result = inspect(payload)
    except ValueError as error:
        raise _native_artifact_error(str(error)) from error
    if (
        type(result) is not tuple
        or len(result) != 5
        or type(result[0]) is not int
        or type(result[1]) is not int
        or type(result[2]) is not str
        or type(result[3]) is not str
        or not isinstance(result[4], bytes)
        or len(result[4]) != 32
    ):
        raise ArtifactError(
            Diagnostic("Q7004", "native artifact projection is malformed")
        )
    return result[0], result[1], result[2], result[3], result[4]


def _native_artifact_error(message: str) -> ArtifactError:
    """Preserve one native artifact diagnostic in the public hierarchy."""

    code, separator, detail = message.partition(": ")
    if separator and len(code) == 5 and code.startswith("Q") and code[1:].isdigit():
        return ArtifactError(Diagnostic(code, detail))
    return ArtifactError(
        Diagnostic("Q7004", f"native artifact persistence failed: {message}")
    )


def _encode_artifact_family_payload(family: _ArtifactFamilyPayload) -> bytes:
    """Encode one bounded artifact family through the native compiler.

    Args:
        family: Valid compiler-owned family payload.

    Returns:
        bytes: Canonical v3 payload containing sealed child artifacts.

    Raises:
        ArtifactError: If native persistence is unavailable or rejects the
            selector, decision paths, or child artifacts.
    """

    seal = getattr(native, "seal_artifact_family_payload", None)
    if not callable(seal):
        raise ArtifactError("Q7045: native artifact-family persistence is unavailable")
    try:
        payload = seal(
            family.target_format,
            list(family.dispatch_parameters),
            family.selection_plan,
            [
                (list(variant.decisions), serialize(variant.artifact))
                for variant in family.variants
            ],
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ArtifactError(
            f"Q7045: native artifact-family persistence failed: {error}"
        ) from error
    if not isinstance(payload, bytes):
        raise ArtifactError("Q7045: native artifact-family persistence is malformed")
    return payload


def _decode_artifact_family_payload(
    payload: bytes,
    *,
    payload_loader: Callable[[str, bytes], ExecutablePayload] | None = None,
) -> _ArtifactFamilyPayload:
    """Adopt one Rust-validated canonical artifact-family payload.

    Args:
        payload: Persistent bytes retained by the outer artifact.
        payload_loader (Callable | None): Optional codec for custom child targets.

    Returns:
        _ArtifactFamilyPayload: Typed, bounded dispatch data and child artifacts.

    Raises:
        ExecutionError: If native persistence is unavailable, the family is
            invalid, or the native bridge projection is malformed.
    """

    inspect = getattr(native, "inspect_artifact_family_payload", None)
    if not callable(inspect):
        raise ExecutionError("Q8130: native artifact-family persistence is unavailable")
    try:
        projection = inspect(payload)
        target_format = projection.target_format
        raw_parameters = projection.dispatch_parameters
        raw_selection_plan = projection.selection_plan
        raw_variants = projection.variants
        if (
            not isinstance(target_format, str)
            or not target_format
            or not isinstance(raw_parameters, list)
            or not raw_parameters
            or not isinstance(raw_selection_plan, (bytes, bytearray, memoryview))
            or not raw_selection_plan
            or not isinstance(raw_variants, list)
            or not raw_variants
        ):
            raise TypeError("invalid native artifact-family header")
        dispatch_parameters = tuple(
            (parameter.parameter_index, parameter.name) for parameter in raw_parameters
        )
        if any(
            type(index) is not int or index < 0 or not isinstance(name, str) or not name
            for index, name in dispatch_parameters
        ):
            raise TypeError("invalid native artifact-family parameters")
        indices = tuple(index for index, _ in dispatch_parameters)
        names = tuple(name for _, name in dispatch_parameters)
        if (
            len(indices) != len(set(indices))
            or indices != tuple(sorted(indices))
            or len(names) != len(set(names))
        ):
            raise TypeError("invalid native artifact-family parameter order")
        variants: list[_ArtifactFamilyVariant] = []
        for raw_variant in raw_variants:
            if not isinstance(raw_variant.decisions, list) or not isinstance(
                raw_variant.artifact, (bytes, bytearray, memoryview)
            ):
                raise TypeError("invalid native artifact-family variant")
            decisions = tuple(
                (decision.predicate_index, decision.value)
                for decision in raw_variant.decisions
            )
            if any(
                type(index) is not int or index < 0 or type(value) is not bool
                for index, value in decisions
            ) or len(decisions) != len({index for index, _ in decisions}):
                raise TypeError("invalid native artifact-family decisions")
            variants.append(
                _ArtifactFamilyVariant(
                    decisions,
                    deserialize(
                        bytes(raw_variant.artifact), payload_loader=payload_loader
                    ),
                )
            )
        return _ArtifactFamilyPayload(
            target_format=target_format,
            dispatch_parameters=dispatch_parameters,
            selection_plan=bytes(raw_selection_plan),
            variants=tuple(variants),
        )
    except ArtifactError as error:
        raise ExecutionError("Q8130: invalid child artifact") from error
    except (AttributeError, OverflowError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q8130: invalid native artifact-family payload: {error}"
        ) from error


def _serialize_launch(compiled: _LaunchMaterializerPayload) -> bytes:
    """Seal the launch recipe only at the explicit persistence boundary.

    Args:
        compiled (_LaunchMaterializerPayload): Live launch recipe to export.

    Returns:
        bytes: Native-sealed launch recipe.

    Raises:
        ArtifactError: If native persistence is unavailable or rejects the recipe.
    """
    seal = getattr(native, "seal_launch_materializer_payload", None)
    if not callable(seal):
        raise ArtifactError(
            "Q7047: native launch-materializer persistence is unavailable"
        )
    try:
        payload = seal(compiled.target_format, compiled.program.to_bytes())
    except (OverflowError, TypeError, ValueError) as error:
        raise ArtifactError(
            f"Q7047: native launch-materializer persistence failed: {error}"
        ) from error
    if not isinstance(payload, bytes):
        raise ArtifactError(
            "Q7047: native launch-materializer persistence is malformed"
        )
    return payload


def _deserialize_launch(payload: bytes) -> _LaunchMaterializerPayload:
    """Restore native-verified QCore once from a saved recipe.

    Args:
        payload (bytes): Saved launch-materializer payload.

    Returns:
        _LaunchMaterializerPayload: Verified program and empty executable slot.

    Raises:
        ExecutionError: If the saved recipe is invalid or native support is absent.
    """
    inspect = getattr(native, "inspect_launch_materializer_payload", None)
    if not callable(inspect):
        raise ExecutionError(
            "Q8133: native launch-materializer persistence is unavailable"
        )
    try:
        projection = inspect(payload)
        return _LaunchMaterializerPayload(
            projection.target_format, Program.from_bytes(bytes(projection.program))
        )
    except (ValueError, TypeError, QamomileError) as error:
        raise ExecutionError("Q8133: invalid launch-materializer payload") from error


def export_payload(compiled: CompiledProgram) -> bytes:
    """Explicitly export the backend representation without its ABI envelope.

    Args:
        compiled (CompiledProgram): Live compilation to export.

    Returns:
        bytes: Backend-specific persistent representation.

    Raises:
        TypeError: If the backend does not implement optional persistence.
    """
    owner = compiled.payload
    if isinstance(owner, _ArtifactFamilyPayload):
        return _encode_artifact_family_payload(owner)
    if isinstance(owner, _LaunchMaterializerPayload):
        return _serialize_launch(owner)
    export = getattr(owner, "to_bytes", None)
    if not callable(export):
        raise TypeError("This executable does not support persistence")
    return export()


def _restore_payload(
    format: str,
    payload: bytes,
    payload_loader: Callable[[str, bytes], ExecutablePayload] | None,
) -> ExecutablePayload:
    """Restore a built-in backend at the explicit import boundary.

    Args:
        format (str): Native-validated backend format identifier.
        payload (bytes): Backend bytes from the verified envelope.
        payload_loader (Callable | None): Optional decoder for a custom format.

    Returns:
        ExecutablePayload: Live owner including SDK lifetime dependencies.

    Raises:
        ArtifactError: If the format has no installed persistence codec.
    """
    if format == ARTIFACT_FAMILY_FORMAT:
        return _decode_artifact_family_payload(payload, payload_loader=payload_loader)
    if format == LAUNCH_MATERIALIZER_FORMAT:
        return _deserialize_launch(payload)
    codecs = {
        "qamomile.shot-execution.v1": ("qamomile._shot_execution", "ShotExecutable"),
        "qamomile.qcore.program": ("qamomile.compilation", "_ReferenceExecutable"),
        "qiskit.qpy.v1": ("qamomile.qiskit._payload", "QiskitExecutable"),
        "cirq.json.v1": ("qamomile.cirq._payload", "CirqExecutable"),
        "quri-parts.logical-circuit.v2": (
            "qamomile.quri_parts._payload",
            "QuriExecutable",
        ),
        "cudaq.python-source.v1": ("qamomile.cudaq._payload", "CudaqExecutable"),
        "braket.openqasm3.v1": ("qamomile.braket._payload", "BraketExecutable"),
        "hugr.package.v1": ("qamomile.hugr._payload", "HugrExecutable"),
        "quration.ir-json.v1": ("qamomile.quration._payload", "QurationExecutable"),
        "qir.bitcode.v1": ("qamomile.qir._payload", "QIRExecutable"),
    }
    if format not in codecs and payload_loader is not None:
        owner = payload_loader(format, payload)
        if not isinstance(owner, ExecutablePayload):
            raise TypeError("payload_loader must return a live executable")
        return owner
    if format not in codecs:
        raise ArtifactError(f"Q7004: no persistence codec for {format!r}")
    module, name = codecs[format]
    return getattr(import_module(module), name).from_bytes(payload)
