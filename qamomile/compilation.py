"""Compilation artifacts, multi-level optimization, and execution boundaries.

Qamomile drives compilation and SDK emission. Extensions compose verified symbolic/QCore transformations and portable late
strategies; Rust owns graph edits, source association, ABI checks, and
portable preparation receipts. Original logical resource studies stay intact.
``TargetProfile`` and its facets expose native-validated device facts through
``CapabilitySnapshot.profile`` and ``OptimizationContext.profile``; the profile
is target identity, never a capability check or an IR element.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

from google.protobuf.message import DecodeError

from ._compiler_optimization import (
    OPTIMIZATION_METADATA_KEY,
    CircuitSimplification as CircuitSimplification,
    CommonExpressions as CommonExpressions,
    FamilyCommonExpressions as FamilyCommonExpressions,
    FamilyOptimizationPass as FamilyOptimizationPass,
    OptimizationContext,
    OptimizationFamily as OptimizationFamily,
    OptimizationPass as OptimizationPass,
    OptimizationPipeline,
    OptimizationProgram as OptimizationProgram,
    StandardSimplification as StandardSimplification,
    _restore_optimization,
    _target_optimization_stages,
)
from ._execution_payload import ExecutablePayload
from ._native import compiler as native
from ._target_profile import (
    CouplingEdge as CouplingEdge,
    EdgeProperty as EdgeProperty,
    LogicalProfile as LogicalProfile,
    QubitProperty as QubitProperty,
    TargetCalibration as TargetCalibration,
    TargetProfile,
    TargetTopology as TargetTopology,
)
from .effects import Requirement
from .errors import (
    ArtifactError,
    CapabilityError,
    ExecutionError,
    QamomileError,
    SerializationError,
    VerificationError,
)
from .ir import FrozenMapping, PortableValue, Program, freeze_value, thaw_value
from .lowering import RealizationOptions
from .observable import Hamiltonian
from .proto import execution_snapshot_pb2 as snapshot_wire, qcore_pb2 as wire
from .resources.metrics import (
    CIRCUIT_VIRTUAL_GATES,
    CIRCUIT_VIRTUAL_QUBITS,
    EMITTED_INSTRUCTIONS,
    OBSERVED_SEGMENT_SHOTS,
    OBSERVED_SHOTS,
    OBSERVED_SUBMISSIONS,
)
from .resources.report import ResourceStudy
from .resources.values import (
    EvidenceAuthority,
    ResourceValue,
    _verified_evidence,
)
from .serialization import (
    SCHEMA_MAJOR,
    SCHEMA_MINOR,
    decode_literal,
    decode_type,
    encode_literal,
    encode_type,
)
from .types import (
    ArrayType,
    Observable,
    ProductType,
    RecordType,
    StagedType,
    TypeSpec,
    WordType,
    as_type,
    sequence_element_type,
    strip_stage,
)

_ExecutableT = TypeVar("_ExecutableT", bound=ExecutablePayload)

if TYPE_CHECKING:
    from .circuit import CircuitProgram


ARTIFACT_FAMILY_FORMAT = "qamomile.artifact-family.v3"
LAUNCH_MATERIALIZER_FORMAT = "qamomile.launch-materializer.v1"
_MAX_EXECUTION_BATCH_INVOCATIONS = 4096
_LAUNCH_CLOSABLE_REQUIREMENTS = frozenset(
    {
        Requirement.DYNAMIC_CLASSICAL_INDEX,
        Requirement.DYNAMIC_QUBIT_SELECTION,
        Requirement.HOST_DISPATCH,
        Requirement.PARAMETERIZED_EXECUTION,
    }
)
_CIRCUIT_LAUNCH_CLOSABLE_REQUIREMENTS = frozenset(
    {
        Requirement.DYNAMIC_CLASSICAL_INDEX,
        Requirement.DYNAMIC_QUBIT_SELECTION,
        Requirement.HOST_DISPATCH,
    }
)


def _hashable_result(value: object) -> object:
    """Convert a decoded record tree into an immutable mapping key."""

    if isinstance(value, Mapping):
        return tuple((name, _hashable_result(item)) for name, item in value.items())
    if isinstance(value, tuple):
        return tuple(_hashable_result(item) for item in value)
    return value


def _public_result_facade(type_spec: TypeSpec, value: object) -> object:
    """Build immutable public containers from a native-validated result.

    Args:
        type_spec: Declared semantic type of ``value``.
        value: Native-validated portable Python value.

    Returns:
        Corresponding public Python representation.

    Raises:
        TypeError: If the native projection has an invalid container shape.
    """

    base = strip_stage(type_spec)
    if base == Observable:
        if not isinstance(value, Mapping):
            raise TypeError("native Observable result is not a mapping")
        return Hamiltonian.from_portable(value)
    if isinstance(base, RecordType):
        if not isinstance(value, Mapping):
            raise TypeError("native record result is not a mapping")
        return MappingProxyType(
            {
                name: _public_result_facade(field_type, value[name])
                for name, field_type in base.fields
            }
        )
    if isinstance(base, ProductType):
        if not isinstance(value, tuple) or len(value) != len(base.elements):
            raise TypeError("native product result has an invalid shape")
        return tuple(
            _public_result_facade(item_type, item)
            for item_type, item in zip(base.elements, value, strict=True)
        )
    if isinstance(base, WordType) and base.kind == "UInt":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError("native UInt result is not a non-negative integer")
        return value
    if isinstance(base, (ArrayType, WordType)):
        if not isinstance(value, tuple):
            raise TypeError("native array result has an invalid shape")
        child_type = sequence_element_type(base)
        return tuple(_public_result_facade(child_type, item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ParameterSlot:
    """One ordered launch-time parameter in a target artifact ABI."""

    name: str
    type: TypeSpec
    default: PortableValue = None
    has_default: bool = False


@dataclass(frozen=True, slots=True)
class ParameterABI:
    """The ordered launch-time parameter contract for an artifact."""

    slots: tuple[ParameterSlot, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous launch parameter identities."""

        names = tuple(slot.name for slot in self.slots)
        if len(names) != len(set(names)):
            raise ArtifactError("Q7007: duplicate artifact parameter slot")

    @classmethod
    def from_program(cls, program: Program) -> ParameterABI:
        """Derive the launch ABI through the required Rust compiler.

        Args:
            program (Program): Closed verified program.

        Returns:
            ParameterABI: Canonical runtime parameter contract.

        Raises:
            VerificationError: If native ABI derivation is unavailable or
                rejects the Program.
        """

        return cls(_derive_native_artifact_abis(program)[0].slots)

    def _to_bytes(self) -> bytes:
        """Encode this ABI through its canonical persistent schema."""

        encoded = wire.ParameterABIProto()
        for slot in self.slots:
            target = encoded.slots.add(name=slot.name)
            encode_type(slot.type, target.type)
            target.has_default = slot.has_default
            if slot.has_default:
                encode_literal(slot.default, target.default_value)
        return encoded.SerializeToString(deterministic=True)

    def bind(self, values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
        """Bind defaults and validate exact values through the Rust ABI."""

        supplied = wire.LiteralProto()
        try:
            frozen = FrozenMapping.from_mapping(values)
            encode_literal(frozen, supplied)
        except (TypeError, SerializationError) as error:
            raise ExecutionError(
                f"Q8135: invocation parameters are not portable: {error}"
            ) from error
        binder = getattr(native, "bind_invocation", None)
        if not callable(binder):
            raise ExecutionError("Q8135: native invocation binding is unavailable")
        try:
            payload = bytes(
                binder(
                    self._to_bytes(),
                    supplied.SerializeToString(deterministic=True),
                )
            )
            resolved = wire.LiteralProto()
            resolved.ParseFromString(payload)
            if resolved.SerializeToString(deterministic=True) != payload:
                raise DecodeError("native binding is not canonical")
            value = decode_literal(resolved)
        except (
            DecodeError,
            OverflowError,
            SerializationError,
            TypeError,
            ValueError,
        ) as error:
            message = str(error)
            if message.startswith("Q8109:"):
                mismatched = next(
                    (
                        slot
                        for slot in self.slots
                        if f'"{slot.name}"' in message and slot.name in values
                    ),
                    None,
                )
                if mismatched is not None:
                    raise ExecutionError(
                        f"Q8109: parameter {mismatched.name!r} expects "
                        f"{mismatched.type}, got {values[mismatched.name]!r}"
                    ) from error
            if message.startswith(("Q8101:", "Q8102:", "Q8109:", "Q8135:")):
                raise ExecutionError(message) from error
            raise ExecutionError(
                f"Q8135: native invocation binding failed: {message}"
            ) from error
        if not isinstance(value, FrozenMapping):
            raise ExecutionError("Q8135: native invocation binding returned no mapping")
        resolved_values = {name: thaw_value(item) for name, item in value.entries}
        expected = {slot.name for slot in self.slots}
        if resolved_values.keys() != expected:
            raise ExecutionError("Q8135: native invocation binding changed the ABI")
        return tuple((slot.name, resolved_values[slot.name]) for slot in self.slots)


@dataclass(frozen=True, slots=True)
class ResultABI:
    """Describe logical result projection independently of an SDK.

    Attributes:
        type (TypeSpec): Logical source result type.
        source_order (tuple[int, ...]): Native measurement positions in logical
            source order.
        native_width (int): Total width of the native measurement row.
    """

    type: TypeSpec
    source_order: tuple[int, ...]
    native_width: int

    def __post_init__(self) -> None:
        """Validate logical-to-native measurement provenance.

        Raises:
            TypeCheckError: If ``type`` is not a logical Qamomile type.
            ArtifactError: If source positions are negative or outside the
                native measurement row.
        """

        object.__setattr__(self, "type", as_type(self.type))
        if any(index < 0 for index in self.source_order):
            raise ArtifactError(
                "Q7009: ResultABI measurement indices must be non-negative"
            )
        if self.native_width < 0 or any(
            index >= self.native_width for index in self.source_order
        ):
            raise ArtifactError("Q7010: ResultABI record width omits a logical result")

    def _to_bytes(self) -> bytes:
        """Encode this ABI through its canonical persistent schema."""

        encoded = wire.ResultABIProto(
            source_order=self.source_order,
            native_width=self.native_width,
        )
        encode_type(self.type, encoded.type)
        return encoded.SerializeToString(deterministic=True)

    @classmethod
    def from_program(cls, program: Program) -> ResultABI:
        """Derive the logical result layout through the Rust compiler.

        Args:
            program (Program): Closed verified program.

        Returns:
            ResultABI: Canonical result reconstruction contract.

        Raises:
            VerificationError: If native ABI derivation is unavailable or
                rejects the Program.
        """

        derived = _derive_native_artifact_abis(program)[1]
        return cls(derived.type, derived.source_order, derived.native_width)

    @property
    def width(self) -> int:
        """Return the fixed logical bit width or zero for non-bit results."""

        return len(self.source_order)

    def decode_big_endian(self, bitstring: str) -> tuple[int, ...]:
        """Decode an SDK ``bit_(n-1)...bit_0`` string through Rust."""

        return self._decode_native_bitstring(bitstring, q0_first=False)

    def decode_q0_first(self, bitstring: str) -> tuple[int, ...]:
        """Decode a q0-first native bitstring through Rust."""

        return self._decode_native_bitstring(bitstring, q0_first=True)

    def _decode_native_bitstring(
        self,
        bitstring: str,
        *,
        q0_first: bool,
    ) -> tuple[int, ...]:
        """Project one SDK textual row into logical source order."""

        decoder = getattr(native, "decode_native_result_bitstring", None)
        if not callable(decoder):
            raise ExecutionError("Q8144: native result decoding is unavailable")
        try:
            bits = decoder(self._to_bytes(), bitstring, q0_first)
        except ValueError as error:
            raise ExecutionError(str(error)) from error
        if (
            not isinstance(bits, list)
            or len(bits) != self.width
            or any(type(bit) is not int or bit not in {0, 1} for bit in bits)
        ):
            raise ExecutionError(
                "Q8144: native result decoding returned an invalid bit row"
            )
        return tuple(bits)

    def decode_value(self, bits: tuple[int, ...]) -> object:
        """Reconstruct a typed sampled value from flat source-ordered bits."""

        if len(bits) != self.width:
            raise ExecutionError(
                f"Q8123: ResultABI expects {self.width} logical bits, got {len(bits)}"
            )
        decoder = getattr(native, "decode_result_value", None)
        if not callable(decoder):
            raise ExecutionError("Q8144: native result decoding is unavailable")
        try:
            payload = decoder(self._to_bytes(), bytes(bits))
            if not isinstance(payload, bytes):
                raise TypeError("native result decoder returned a non-byte payload")
        except (
            OverflowError,
            QamomileError,
            TypeError,
            ValueError,
        ) as error:
            message = str(error)
            if message.startswith(("Q8121:", "Q8122:", "Q8123:", "Q8124:", "Q8144:")):
                raise ExecutionError(message) from error
            raise ExecutionError(
                f"Q8144: native result decoding failed: {message}"
            ) from error
        return self._decode_native_value(payload)

    def _decode_native_value(self, payload: bytes) -> object:
        """Project one canonical native literal into its public Python value."""

        validator = getattr(native, "validate_result_value", None)
        if not callable(validator):
            raise ExecutionError("Q8144: native result validation is unavailable")
        try:
            validator(self._to_bytes(), payload)
            encoded = wire.LiteralProto()
            encoded.ParseFromString(payload)
            if encoded.SerializeToString(deterministic=True) != payload:
                raise DecodeError("native result value is not canonical")
            public = thaw_value(decode_literal(encoded))
            return _public_result_facade(self.type, public)
        except (
            DecodeError,
            KeyError,
            OverflowError,
            QamomileError,
            TypeError,
            ValueError,
        ) as error:
            message = str(error)
            if message.startswith(("Q8121:", "Q8122:", "Q8123:", "Q8124:", "Q8144:")):
                raise ExecutionError(message) from error
            raise ExecutionError(
                f"Q8144: native result decoding failed: {message}"
            ) from error

    def encode_value(self, value: object) -> tuple[int, ...]:
        """Flatten a typed sampled value into logical source-bit order."""

        encoder = getattr(native, "encode_result_value", None)
        if not callable(encoder):
            raise ExecutionError("Q8144: native result encoding is unavailable")
        try:
            literal = wire.LiteralProto()
            encode_literal(freeze_value(value), literal)
            payload = encoder(
                self._to_bytes(),
                literal.SerializeToString(deterministic=True),
            )
            if not isinstance(payload, bytes):
                raise TypeError("native result encoder returned a non-byte payload")
            bits = tuple(payload)
            if len(bits) != self.width or any(bit not in {0, 1} for bit in bits):
                raise TypeError("native result encoder returned an invalid bit row")
            return bits
        except (OverflowError, QamomileError, TypeError, ValueError) as error:
            message = str(error)
            if message.startswith(("Q8122:", "Q8143:", "Q8144:")):
                raise ExecutionError(message) from error
            raise ExecutionError(
                f"Q8144: native result encoding failed: {message}"
            ) from error


def _derive_native_artifact_abis(
    program: Program,
) -> tuple[ParameterABI, ResultABI]:
    """Decode the ABIs derived by the required native compiler.

    Args:
        program: Closed verified program whose entrypoint defines both ABIs.

    Returns:
        The native parameter and result ABIs.

    Raises:
        ArtifactError: If an installed native compiler cannot derive or encode
            the ABI pair.
    """

    derive = getattr(native, "derive_artifact_abi", None)
    if not callable(derive):
        raise ArtifactError("Q7013: native artifact ABI derivation is unavailable")
    try:
        raw_parameter_abi, raw_result_abi = derive(program._native_program())
        parameter_payload = bytes(raw_parameter_abi)
        result_payload = bytes(raw_result_abi)
        encoded_parameters = wire.ParameterABIProto()
        encoded_parameters.ParseFromString(parameter_payload)
        encoded_result = wire.ResultABIProto()
        encoded_result.ParseFromString(result_payload)
        if (
            encoded_parameters.SerializeToString(deterministic=True)
            != parameter_payload
            or encoded_result.SerializeToString(deterministic=True) != result_payload
        ):
            raise DecodeError("native artifact ABI is not canonical")
        parameters = ParameterABI(
            tuple(
                ParameterSlot(
                    slot.name,
                    decode_type(slot.type),
                    decode_literal(slot.default_value) if slot.has_default else None,
                    slot.has_default,
                )
                for slot in encoded_parameters.slots
            )
        )
        result = ResultABI(
            decode_type(encoded_result.type),
            tuple(encoded_result.source_order),
            encoded_result.native_width,
        )
    except (
        DecodeError,
        OverflowError,
        QamomileError,
        TypeError,
        ValueError,
    ) as error:
        message = str(error)
        if message.startswith(("Q4002:", "Q7011:", "Q7012:", "Q7013:")):
            raise ArtifactError(message) from error
        raise ArtifactError(
            f"Q7013: native artifact ABI derivation failed: {message}"
        ) from error
    return parameters, result


def _validate_execution_interface(
    resources: ResourceStudy,
    parameters: ParameterABI,
    result: ResultABI,
    metadata: FrozenMapping = FrozenMapping(()),
) -> None:
    """Check the executable source and retained ABIs before loading SDK payloads.

    Args:
        resources (ResourceStudy): Report retaining the verified source.
        parameters (ParameterABI): Retained classical input contract.
        result (ResultABI): Retained classical output contract.
        metadata (FrozenMapping): Optional native raw segment interface.

    Raises:
        ArtifactError: If native validation is unavailable or rejects the boundary.
    """
    validator = getattr(native, "validate_execution_interface", None)
    optimization = dict(metadata.entries).get(OPTIMIZATION_METADATA_KEY)
    selected = None
    if optimization is not None:
        source = resources._semantic_source()
        if not isinstance(optimization, bytes) or not isinstance(source, Program):
            raise ArtifactError("Q7802: invalid optimization metadata")
        try:
            selected = source._native_program().restore_optimization(optimization)
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error
    receipt = dict(metadata.entries).get("qamomile.shot-segment-interface")
    if receipt is not None:
        validator = getattr(native, "validate_shot_segment_interface", None)
        if not callable(validator) or not isinstance(receipt, bytes):
            raise ArtifactError("Q4470: native segment interface is unavailable")
        program = resources._semantic_source()
        if not isinstance(program, Program):
            raise ArtifactError("Q4470: segment requires a verified source")
        try:
            validator(
                selected if selected is not None else program._native_program(),
                receipt,
                parameters._to_bytes(),
                result._to_bytes(),
            )
        except (TypeError, ValueError) as error:
            raise ArtifactError(str(error)) from error
        return
    if not callable(validator):
        raise ArtifactError(
            "Q7016: native execution-interface validation is unavailable"
        )
    source = resources._semantic_source()
    if not isinstance(source, Program):
        raise ArtifactError("Q7002: executable resources must retain a closed program")
    try:
        validator(
            source._native_program(),
            parameters._to_bytes(),
            result._to_bytes(),
        )
    except ValueError as error:
        raise ArtifactError(str(error)) from error


@dataclass(frozen=True, slots=True)
class CompilationReport:
    """Coordinate-safe resource study and target-emission provenance."""

    stages: tuple[str, ...]
    resources: ResourceStudy
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def metric(self, identifier: str) -> ResourceValue:
        """Return one coordinate-qualified metric from the retained study."""

        return self.resources.metric(identifier)

    def literal_int(self, identifier: str) -> int:
        """Return an exact integer metric or fail instead of guessing."""

        value = self.metric(identifier)
        lower, upper = value.numeric_enclosure()
        if lower != upper or isinstance(lower, bool) or not isinstance(lower, int):
            raise ArtifactError(
                f"Q7017: compilation metric {identifier!r} is not an exact integer"
            )
        return lower


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    """Minimal immutable runtime contract sealed into an artifact."""

    format: str
    requirements: frozenset[Requirement] = frozenset()

    def __post_init__(self) -> None:
        """Freeze and validate the complete runtime compatibility contract."""

        requirements = frozenset(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("execution requirements must be Requirement values")
        if not self.format:
            raise ValueError("execution contract identities must be nonempty")
        object.__setattr__(self, "requirements", requirements)

    @classmethod
    def from_program(cls, format: str, program: Program) -> ExecutionContract:
        """Build an execution contract from authoritative entry requirements.

        Args:
            format (str): Target artifact format identifier.
            program (Program): Closed verified program whose requirements are
                planned.

        Returns:
            ExecutionContract: Format and compiler-derived requirement set.

        Raises:
            ArtifactError: If native requirement planning is unavailable or
                inconsistent.
        """

        required, _, _ = _plan_target_requirements(
            program,
            frozenset(Requirement),
        )
        return cls(format, required)


class RealizationCapability(str, Enum):
    """Closed target-neutral composite realizations accepted by a consumer."""

    NAMED_CALL = "named_call"
    NATIVE_FOR = "native_for"
    PAULI_EVOLUTION = "pauli_evolution"
    RZZ = "rzz"
    ADD_INTO = "add_into"
    MOD_POW = "mod_pow"
    MCX = "mcx"
    CONTROLLED_SWAP = "controlled_swap"
    COHERENT_REGIONS = "coherent_regions"
    PAULI_MEASUREMENT = "pauli_measurement"
    INDEXED_FOR = "indexed_for"
    BOOLEAN_STORE = "boolean_store"


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """Freeze target support and checked late-realization choices.

    Args:
        format (str): Target-format identity.
        requirements (frozenset[Requirement]): Supported execution requirements.
        realization_capabilities (frozenset[RealizationCapability]): Accepted
            composite operations.
        realizations (RealizationOptions): Exact implementation choices.
        seeded_execution (bool): Whether the endpoint accepts explicit simulator
            streams. Device targets leave this transport fact false.
        circuit_optimization (bool): Whether a non-CircuitTarget emits through
            the checked native circuit projection, such as a static export target.
        profile (TargetProfile | None): Native-validated device facts, or None
            when the target supplies none. The profile is identity for
            optimization context and fingerprints; ``require`` ignores it, so
            an artifact compiled against one calibration may execute on an
            endpoint with another.

    Raises:
        TypeError: If capability values have invalid types.
        ValueError: If the format is empty.
        CapabilityError: If an override requires an unadvertised capability.
    """

    format: str
    requirements: frozenset[Requirement]
    realization_capabilities: frozenset[RealizationCapability] = frozenset()
    realizations: RealizationOptions = RealizationOptions()
    seeded_execution: bool = field(default=False, kw_only=True)
    circuit_optimization: bool = field(default=False, kw_only=True)
    profile: TargetProfile | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Freeze advertised sets and validate implementation requests.

        Raises:
            TypeError: If capability values have invalid types.
            ValueError: If the format is empty.
            CapabilityError: If an override cannot be honored.
        """

        if type(self.seeded_execution) is not bool:
            raise TypeError("seeded_execution must be bool")
        if type(self.circuit_optimization) is not bool:
            raise TypeError("circuit_optimization must be bool")
        if self.profile is not None and not isinstance(self.profile, TargetProfile):
            raise TypeError("profile must be a TargetProfile or None")
        requirements = frozenset(self.requirements)
        realization_capabilities = frozenset(self.realization_capabilities)
        if not isinstance(self.realizations, RealizationOptions):
            raise TypeError("realizations must be RealizationOptions")
        if not self.format:
            raise ValueError("capability snapshot identities must be nonempty")
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("capabilities must contain Requirement values")
        if any(
            not isinstance(item, RealizationCapability)
            for item in realization_capabilities
        ):
            raise TypeError(
                "realization capabilities must contain RealizationCapability values"
            )
        self.realizations.require_capabilities(
            tuple(sorted(item.value for item in realization_capabilities))
        )
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "realization_capabilities", realization_capabilities)

    def require(self, contract: ExecutionContract) -> None:
        """Raise a stable diagnostic for the first incompatibility."""

        if contract.format != self.format:
            raise ExecutionError(
                f"Q8126: endpoint does not accept artifact format {contract.format!r}"
            )
        missing = contract.requirements - self.requirements
        if missing:
            raise ExecutionError(
                "Q8127: endpoint lacks required capabilities: "
                + ", ".join(sorted(item.value for item in missing))
            )


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """Retain a backend executable with immutable semantic and ABI metadata.

    Compilation returns a process-local executable owner. Optional storage
    is provided separately by ``qamomile.persistence``.
    Copies share the executable owner, while each launch owns its bound inputs
    and quantum state. Equality includes executable-owner identity;
    ``execution_identity`` compares compilation provenance. Executable owners
    are never implicitly pickled.

    Attributes:
        payload (ExecutablePayload): Live backend executable and its SDK owners.
        target_fingerprint (str): Identity of the target configuration used
            for emission.
        parameter_abi (ParameterABI): Runtime parameter binding contract.
        result_abi (ResultABI): Normalized result reconstruction contract.
        report (CompilationReport): Compilation and resource evidence.
        execution_contract (ExecutionContract): Required endpoint format and
            capabilities.
        metadata (FrozenMapping): Canonical target-owned metadata.
    """

    payload: ExecutablePayload = field(repr=False)
    target_fingerprint: str
    parameter_abi: ParameterABI
    result_abi: ResultABI
    report: CompilationReport
    execution_contract: ExecutionContract
    metadata: FrozenMapping = FrozenMapping(())

    def execution_payload(self, kind: type[_ExecutableT]) -> _ExecutableT:
        """Return the live backend owner without restoring any saved bytes.

        Args:
            kind (type): Expected backend executable class.

        Returns:
            ExecutablePayload: Retained SDK owner for this compilation.

        Raises:
            ExecutionError: If the executable belongs to another backend.
        """
        if not isinstance(self.payload, kind):
            raise ExecutionError("Q7001: incompatible executable payload")
        return self.payload

    def __reduce__(self) -> tuple[object, ...]:
        """Reject implicit persistence of process-local execution resources.

        Raises:
            TypeError: Always; use qamomile.persistence for explicit persistence.
        """
        raise TypeError(
            "CompiledProgram is process-local; use qamomile.persistence.serialize"
        )

    def __copy__(self) -> CompiledProgram:
        """Share this immutable facade and its synchronized executable owner.

        Returns:
            CompiledProgram: This immutable facade.
        """
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> CompiledProgram:
        """Share immutable metadata while preserving the SDK lifetime owner.

        Args:
            memo (dict[int, object]): Deep-copy identity map.

        Returns:
            CompiledProgram: This immutable facade, without duplicating native resources.
        """
        memo[id(self)] = self
        return self

    def __post_init__(self) -> None:
        """Require a live backend executable and consistent compilation facts."""

        if not isinstance(self.payload, ExecutablePayload):
            raise ArtifactError("Q7001: artifact payload must be a live executable")
        try:
            source_digest = self.source_digest
        except ValueError as error:
            raise ArtifactError(
                "Q7002: artifact resources must retain one closed program"
            ) from error
        if len(source_digest) != 32:
            raise ArtifactError("Q7002: artifact program identity must be SHA-256")
        if not self.format or not self.target_fingerprint:
            raise ArtifactError("Q7003: artifact format and fingerprint are required")
        _validate_execution_interface(
            self.report.resources, self.parameter_abi, self.result_abi, self.metadata
        )
        metadata_names = tuple(name for name, _ in self.metadata.entries)
        if len(metadata_names) != len(set(metadata_names)):
            raise ArtifactError("Q7023: artifact metadata keys must be unique")

    @property
    def format(self) -> str:
        """Return the sole artifact format from its execution contract."""

        return self.execution_contract.format

    @property
    def source_digest(self) -> bytes:
        """Derive the sole program identity from the retained resource study."""

        return bytes.fromhex(self.report.resources.semantic_digest)

    @property
    def execution_identity(self) -> str:
        """Identify the compilation contract without exporting the SDK circuit.

        This is a provenance key for remote job matching, not an integrity hash
        of SDK bytes or a proof of quantum equivalence. Source meaning and ABI
        facts remain compiler-owned; the target fingerprint identifies emission
        choices. Launch-specialized values are retained in metadata.

        Returns:
            str: SHA-256 of canonical source, target, ABI and execution metadata.
        """
        identity = _encode_compilation_metadata(self)
        return hashlib.sha256(
            b"qamomile.compilation-identity.v1\0"
            + self.source_digest
            + identity.SerializeToString(deterministic=True)
        ).hexdigest()


def _encode_compilation_metadata(compiled: CompiledProgram) -> wire.ArtifactProto:
    """Project checked ABI and target facts without exporting an SDK payload.

    Args:
        compiled (CompiledProgram): Live owner of compiler-derived facts.

    Returns:
        wire.ArtifactProto: Canonical metadata projection shared by job matching
        and optional persistence; payload, report and integrity seal are absent.
    """
    encoded = wire.ArtifactProto(
        format=compiled.format, target_fingerprint=compiled.target_fingerprint
    )
    encoded.execution_requirements.extend(
        sorted(item.value for item in compiled.execution_contract.requirements)
    )
    for slot in compiled.parameter_abi.slots:
        target = encoded.parameter_abi.slots.add(name=slot.name)
        encode_type(slot.type, target.type)
        target.has_default = slot.has_default
        if slot.has_default:
            encode_literal(slot.default, target.default_value)
    encode_type(compiled.result_abi.type, encoded.result_abi.type)
    encoded.result_abi.source_order.extend(compiled.result_abi.source_order)
    encoded.result_abi.native_width = compiled.result_abi.native_width
    for name, value in sorted(compiled.metadata.entries):
        encode_literal(value, encoded.metadata.add(name=name).value)
    return encoded


@dataclass(frozen=True, slots=True)
class Invocation:
    """Launch-time values and finite execution policy, separate from artifacts."""

    parameters: FrozenMapping = FrozenMapping(())
    shots: int = 1
    seed: int = 0

    @classmethod
    def create(
        cls,
        *,
        parameters: Mapping[str, object] | None = None,
        shots: int = 1,
        seed: int = 0,
    ) -> Invocation:
        """Create an immutable invocation from ordinary Python values."""

        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ExecutionError("Q8105: shots must be a positive integer")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= (1 << 64) - 1
        ):
            raise ExecutionError("Q8138: seed must be an unsigned 64-bit integer")
        return cls(
            FrozenMapping.from_mapping(parameters or {}),
            shots,
            seed,
        )

    def values(self) -> dict[str, object]:
        """Return a fresh launch-parameter mapping."""

        return {name: thaw_value(value) for name, value in self.parameters.entries}


@dataclass(frozen=True, slots=True)
class ValueResult:
    """A non-sampled typed execution result."""

    backend: str
    value: object
    worker_pid: int | None = None


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Immutable logical samples and optional native physical execution counts.

    Attributes:
        execution_statistics (FrozenMapping): Native logical/physical shot and
            submission counts for a segmented invocation; empty otherwise.
    """

    backend: str
    _counts: tuple[tuple[tuple[int, ...], int], ...]
    worker_pid: int | None = None
    result_abi: ResultABI | None = field(default=None, repr=False, compare=False)
    execution_statistics: FrozenMapping = FrozenMapping(())
    _shots: int = field(init=False, repr=False, compare=False)
    _most_likely: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _native_summary: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Protect result invariants even when constructed without the factory."""

        if not self.backend:
            raise ExecutionError("Q8114: sample backend identity cannot be empty")
        normalized, shots, most_likely, native_summary = _native_sample_count_summary(
            self._counts,
            expected_width=None,
        )
        if normalized != self._counts:
            raise ExecutionError("Q8115: sampling counts must be nonempty and ordered")
        object.__setattr__(self, "_shots", shots)
        object.__setattr__(self, "_most_likely", most_likely)
        object.__setattr__(self, "_native_summary", native_summary)
        statistics = dict(self.execution_statistics.entries)
        if statistics and (
            statistics.keys()
            != {"logical_shots", "physical_segment_shots", "submissions"}
            or any(type(value) is not int or value < 0 for value in statistics.values())
            or statistics["logical_shots"] != shots
        ):
            raise ExecutionError("Q4470: invalid shot execution statistics")

    def __copy__(self) -> SampleResult:
        """Return this immutable result for a shallow copy request."""

        return self

    def __deepcopy__(self, memo: dict[int, object]) -> SampleResult:
        """Return this immutable result without copying its native cache."""

        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize public state and rebuild the derived native summary."""

        return (
            type(self),
            (
                self.backend,
                self._counts,
                self.worker_pid,
                self.result_abi,
                self.execution_statistics,
            ),
        )

    @classmethod
    def from_counts(
        cls,
        *,
        backend: str,
        counts: Mapping[tuple[int, ...], int],
        worker_pid: int | None = None,
        expected_width: int | None = None,
    ) -> SampleResult:
        """Normalize a logical count mapping at the ResultABI boundary."""

        if expected_width is not None and (
            isinstance(expected_width, bool)
            or not isinstance(expected_width, int)
            or expected_width < 0
        ):
            raise ExecutionError("Q8110: expected sample width must be non-negative")
        rows: list[tuple[tuple[int, ...], int]] = []
        for raw_bits, raw_count in counts.items():
            bits = tuple(raw_bits)
            if any(type(bit) is not int or bit not in {0, 1} for bit in bits):
                raise ExecutionError("Q8111: sample keys must contain only 0 or 1")
            if type(raw_count) is not int:
                raise ExecutionError("Q8112: sampling counts must be integers")
            rows.append((bits, raw_count))
        normalized, shots, most_likely, native_summary = _native_sample_count_summary(
            tuple(rows),
            expected_width=expected_width,
        )
        return cls._from_native_summary(
            backend=backend,
            counts=normalized,
            shots=shots,
            most_likely=most_likely,
            native_summary=native_summary,
            worker_pid=worker_pid,
        )

    @classmethod
    def _from_native_summary(
        cls,
        *,
        backend: str,
        counts: tuple[tuple[tuple[int, ...], int], ...],
        shots: int,
        most_likely: tuple[int, ...],
        native_summary: object,
        worker_pid: int | None,
    ) -> SampleResult:
        """Construct one result from an already validated native summary."""

        if not backend:
            raise ExecutionError("Q8114: sample backend identity cannot be empty")
        result = object.__new__(cls)
        object.__setattr__(result, "backend", backend)
        object.__setattr__(result, "_counts", counts)
        object.__setattr__(result, "worker_pid", worker_pid)
        object.__setattr__(result, "result_abi", None)
        object.__setattr__(result, "_shots", shots)
        object.__setattr__(result, "_most_likely", most_likely)
        object.__setattr__(result, "_native_summary", native_summary)
        object.__setattr__(result, "execution_statistics", FrozenMapping(()))
        return result

    @property
    def counts(self) -> Mapping[tuple[int, ...], int]:
        """Return a read-only logical count mapping."""

        return MappingProxyType(dict(self._counts))

    @property
    def shots(self) -> int:
        """Return the observation count derived from the canonical histogram."""

        return self._shots

    @property
    def width(self) -> int:
        """Return the uniform logical sample width."""

        return len(self._counts[0][0])

    @property
    def most_likely(self) -> tuple[int, ...]:
        """Return the most frequent result with deterministic tie-breaking."""

        return self._most_likely

    @property
    def most_likely_value(self) -> object:
        """Return the most frequent result decoded by the declared result type."""

        if self.result_abi is None:
            return self.most_likely
        return self.result_abi.decode_value(self.most_likely)

    def probability(self, bits: tuple[int, ...]) -> float:
        """Return the empirical probability of a logical bit value."""

        if len(bits) != self.width or any(
            type(bit) is not int or bit not in {0, 1} for bit in bits
        ):
            raise ValueError(f"probability expects {self.width} logical bits")
        probability = getattr(self._native_summary, "probability", None)
        if not callable(probability):
            raise ExecutionError("Q8128: native sample probability is unavailable")
        try:
            value = probability(bytes(bits))
        except (TypeError, ValueError) as error:
            raise ExecutionError(
                f"Q8128: native sample probability failed: {error}"
            ) from error
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ExecutionError("Q8128: native sample probability is malformed")
        return value

    def expectation_z(self, *indices: int) -> float:
        """Estimate a Pauli-Z product from logical samples."""

        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(
                type(index) is not int or not 0 <= index < self.width
                for index in indices
            )
        ):
            raise ValueError(
                f"expectation_z expects distinct indices in [0, {self.width})"
            )
        estimate = getattr(self._native_summary, "expectation_z", None)
        if not callable(estimate):
            raise CapabilityError(
                "Q7300: native Pauli expectation aggregation is unavailable"
            )
        try:
            value = estimate(list(indices))
        except ValueError as error:
            raise ExecutionError(str(error)) from error
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -1.0 <= value <= 1.0
        ):
            raise ExecutionError("Q7310: malformed native Pauli expectation estimate")
        return value

    @property
    def values(self) -> Mapping[object, int]:
        """Return counts decoded into the declared typed result shape."""

        if self.result_abi is None:
            flat: dict[object, int] = {bits: count for bits, count in self._counts}
            return MappingProxyType(flat)
        decode_values = getattr(self._native_summary, "decode_values", None)
        if not callable(decode_values):
            raise ExecutionError("Q8144: native typed sample decoding is unavailable")
        try:
            rows = decode_values(self.result_abi._to_bytes())
        except (TypeError, ValueError) as error:
            raise ExecutionError(
                f"Q8144: native typed sample decoding failed: {error}"
            ) from error
        if not isinstance(rows, list):
            raise ExecutionError("Q8144: native typed sample decoding is malformed")
        decoded: dict[object, int] = {}
        for row in rows:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], bytes)
                or not isinstance(row[1], str)
                or not row[1].isdecimal()
            ):
                raise ExecutionError("Q8144: native typed sample decoding is malformed")
            value = self.result_abi._decode_native_value(row[0])
            key = _hashable_result(value)
            decoded[key] = int(row[1])
        return MappingProxyType(decoded)


def _native_sample_count_summary(
    rows: tuple[tuple[tuple[int, ...], int], ...],
    *,
    expected_width: int | None,
) -> tuple[
    tuple[tuple[tuple[int, ...], int], ...],
    int,
    tuple[int, ...],
    object,
]:
    """Normalize and summarize one logical histogram through Rust."""

    normalize = getattr(native, "normalize_sample_counts", None)
    summary_type = getattr(native, "SampleCountSummary", None)
    if not callable(normalize) or not isinstance(summary_type, type):
        raise ExecutionError("Q8128: native sample-count normalization is unavailable")
    prepared: list[tuple[list[int], str]] = []
    try:
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError
            bits, count = row
            if not isinstance(bits, tuple):
                raise TypeError
            if any(type(bit) is not int for bit in bits) or type(count) is not int:
                raise TypeError
            prepared.append((list(bits), str(count)))
    except (TypeError, ValueError) as error:
        raise ExecutionError(
            "Q8117: sampling keys must be uniform logical bits"
        ) from error
    try:
        projection = normalize(
            prepared,
            None if expected_width is None else str(expected_width),
        )
        if not isinstance(projection, summary_type):
            raise TypeError
        raw_entries = projection.entries
        raw_shots = projection.shots
        raw_most_likely = projection.most_likely
        if (
            not isinstance(raw_entries, list)
            or not isinstance(raw_shots, str)
            or not raw_shots.isdecimal()
            or not isinstance(raw_most_likely, list)
            or any(type(bit) is not int or bit not in {0, 1} for bit in raw_most_likely)
        ):
            raise TypeError
        entries: list[tuple[tuple[int, ...], int]] = []
        for row in raw_entries:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError
            bits, count = row
            if (
                not isinstance(bits, list)
                or any(type(bit) is not int or bit not in {0, 1} for bit in bits)
                or not isinstance(count, str)
                or not count.isdecimal()
            ):
                raise TypeError
            entries.append((tuple(bits), int(count)))
        return tuple(entries), int(raw_shots), tuple(raw_most_likely), projection
    except ValueError as error:
        raise ExecutionError(str(error)) from error
    except (AttributeError, TypeError) as error:
        raise ExecutionError(
            "Q8128: native sample-count normalization returned a malformed summary"
        ) from error


ExecutionResult: TypeAlias = ValueResult | SampleResult


class ExecutionSnapshotKind(str, Enum):
    """Identify the closed shapes of persistent execution state."""

    REMOTE = "remote"
    LOCAL = "local"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class ExecutionReference:
    """Store secret-free identifiers needed to restore remote execution.

    Args:
        provider (str): Stable provider adapter identity.
        job_ids (tuple[str, ...]): One or more nonempty provider job IDs.
        target (str | None): Optional provider target or device identity.
        group_id (str | None): Optional session, batch, or parent identity.
        context (tuple[tuple[str, str], ...]): Sorted provider-specific decoding
            metadata containing no credentials.

    Raises:
        TypeError: If an identifier or context entry has an incompatible type.
        ValueError: If required identifiers are empty or context is not sorted.
    """

    provider: str
    job_ids: tuple[str, ...]
    target: str | None = None
    group_id: str | None = None
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate and detach all provider reference fields.

        Raises:
            TypeError: If an identifier or context entry has an incompatible type.
            ValueError: If required identifiers are empty or context is not sorted.
        """

        if type(self.provider) is not str or any(
            type(item) is not str for item in self.job_ids
        ):
            raise TypeError("execution provider and job IDs must be strings")
        if (
            not self.provider
            or not self.job_ids
            or any(not item for item in self.job_ids)
        ):
            raise ValueError("execution provider and job IDs must be nonempty")
        if self.target is not None and (
            type(self.target) is not str or not self.target
        ):
            raise ValueError("execution target must be nonempty when provided")
        if self.group_id is not None and (
            type(self.group_id) is not str or not self.group_id
        ):
            raise ValueError("execution group ID must be nonempty when provided")
        entries = tuple(self.context)
        if any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            or not entry[0]
            for entry in entries
        ):
            raise TypeError(
                "execution context must contain nonempty string keys and values"
            )
        if tuple(sorted(entries)) != entries or len({key for key, _ in entries}) != len(
            entries
        ):
            raise ValueError("execution context keys must be unique and sorted")
        object.__setattr__(self, "job_ids", tuple(self.job_ids))
        object.__setattr__(self, "context", entries)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        job_ids: tuple[str, ...],
        target: str | None = None,
        group_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> ExecutionReference:
        """Create a reference from an ordinary context mapping.

        Args:
            provider (str): Stable provider adapter identity.
            job_ids (tuple[str, ...]): One or more provider job IDs.
            target (str | None): Optional provider target or device identity.
            group_id (str | None): Optional session, batch, or parent identity.
            context (Mapping[str, str] | None): Provider decoding metadata.

        Returns:
            ExecutionReference: Immutable, lexically ordered reference.

        Raises:
            TypeError: If context is not a string mapping.
            ValueError: If any required identifier is empty.
        """

        if context is not None and (
            not isinstance(context, Mapping)
            or any(
                type(key) is not str or type(value) is not str
                for key, value in context.items()
            )
        ):
            raise TypeError("execution context must map strings to strings")
        return cls(
            provider,
            tuple(job_ids),
            target,
            group_id,
            tuple(sorted((context or {}).items())),
        )

    @property
    def context_values(self) -> Mapping[str, str]:
        """Return provider decoding metadata as a read-only mapping.

        Returns:
            Mapping[str, str]: Detached immutable view of provider context.
        """

        return MappingProxyType(dict(self.context))


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """Persist local, remote, or grouped execution without SDK objects.

    Args:
        kind (ExecutionSnapshotKind): Selected node shape.
        reference (ExecutionReference | None): Remote provider reference.
        value (PortableValue): Canonical local value when ``has_value`` is true.
        has_value (bool): Whether this node owns a local value, including ``None``.
        children (tuple[ExecutionSnapshot, ...]): Ordered composite children.

    Raises:
        TypeError: If fields do not use immutable portable representations.
        ValueError: If fields disagree with the selected node kind.
    """

    kind: ExecutionSnapshotKind
    reference: ExecutionReference | None = None
    value: PortableValue = None
    has_value: bool = False
    children: tuple[ExecutionSnapshot, ...] = ()

    def __post_init__(self) -> None:
        """Validate one immutable execution node before native sealing.

        Raises:
            TypeError: If fields do not use immutable portable representations.
            ValueError: If fields disagree with the selected node kind.
        """

        if not isinstance(self.kind, ExecutionSnapshotKind):
            raise TypeError("execution snapshot kind is invalid")
        children = tuple(self.children)
        if any(not isinstance(child, ExecutionSnapshot) for child in children):
            raise TypeError("execution snapshot children must be snapshots")
        if self.kind is ExecutionSnapshotKind.REMOTE:
            if self.reference is None or self.has_value or children:
                raise ValueError("remote snapshots contain exactly one reference")
        elif self.kind is ExecutionSnapshotKind.LOCAL:
            if self.reference is not None or not self.has_value or children:
                raise ValueError("local snapshots contain exactly one value")
            object.__setattr__(self, "value", freeze_value(self.value))
        elif self.reference is not None or self.has_value or not children:
            raise ValueError("composite snapshots contain only nonempty children")
        object.__setattr__(self, "children", children)

    @classmethod
    def remote(cls, reference: ExecutionReference) -> ExecutionSnapshot:
        """Create one remote execution leaf.

        Args:
            reference (ExecutionReference): Secret-free provider reference.

        Returns:
            ExecutionSnapshot: Immutable remote leaf.

        Raises:
            TypeError: If ``reference`` is not an execution reference.
        """

        if not isinstance(reference, ExecutionReference):
            raise TypeError("remote snapshot expects an ExecutionReference")
        return cls(ExecutionSnapshotKind.REMOTE, reference=reference)

    @classmethod
    def local(cls, value: object) -> ExecutionSnapshot:
        """Create one local portable-value leaf.

        Args:
            value (object): Value in Qamomile's closed portable literal domain.

        Returns:
            ExecutionSnapshot: Immutable local leaf.

        Raises:
            TypeError: If ``value`` is not portable.
        """

        return cls(
            ExecutionSnapshotKind.LOCAL, value=freeze_value(value), has_value=True
        )

    @classmethod
    def completed(cls, result: ExecutionResult) -> ExecutionSnapshot:
        """Capture one completed typed execution result.

        Args:
            result (ExecutionResult): Validated sample or value result.

        Returns:
            ExecutionSnapshot: Local leaf retaining the complete public result.

        Raises:
            TypeError: If ``result`` is not a supported execution result.
        """

        return cls.local(_freeze_execution_result(result))

    @classmethod
    def composite(cls, children: tuple[ExecutionSnapshot, ...]) -> ExecutionSnapshot:
        """Create one ordered composite execution node.

        Args:
            children (tuple[ExecutionSnapshot, ...]): Nonempty ordered child row.

        Returns:
            ExecutionSnapshot: Immutable composite node.

        Raises:
            TypeError: If a child is not an execution snapshot.
            ValueError: If ``children`` is empty.
        """

        return cls(ExecutionSnapshotKind.COMPOSITE, children=tuple(children))

    @property
    def result(self) -> ExecutionResult:
        """Restore the typed result held by a local completed snapshot.

        Returns:
            ExecutionResult: Revalidated sample or value result.

        Raises:
            ValueError: If this snapshot is not a local completed result.
            SerializationError: If the local value is not a result record.
        """

        if self.kind is not ExecutionSnapshotKind.LOCAL:
            raise ValueError("only local snapshots contain completed results")
        return _thaw_execution_result(self.value)

    def references(self) -> tuple[ExecutionReference, ...]:
        """Collect remote references in stable tree order.

        Returns:
            tuple[ExecutionReference, ...]: Remote leaf references in tree order.
        """

        pending: list[ExecutionSnapshot] = [self]
        references: list[ExecutionReference] = []
        while pending:
            node = pending.pop()
            if node.reference is not None:
                references.append(node.reference)
            pending.extend(reversed(node.children))
        return tuple(references)

    def to_bytes(self) -> bytes:
        """Serialize and integrity-seal this execution tree through Rust.

        Returns:
            bytes: Canonical integrity-sealed snapshot bytes.

        Raises:
            SerializationError: If encoding or native validation fails.
        """

        package = snapshot_wire.ExecutionSnapshotProto(
            schema_major=SCHEMA_MAJOR,
            schema_minor=SCHEMA_MINOR,
            artifact_kind="qamomile.execution-snapshot",
        )
        _encode_execution_snapshot_node(self, package.root)
        seal = getattr(native, "seal_execution_snapshot", None)
        if not callable(seal):
            raise SerializationError(
                "Q8402: native execution snapshot sealing is unavailable"
            )
        try:
            return bytes(seal(package.SerializeToString(deterministic=True)))
        except (TypeError, ValueError) as error:
            raise SerializationError(str(error)) from error

    @classmethod
    def from_bytes(cls, payload: bytes) -> ExecutionSnapshot:
        """Decode one bounded integrity-sealed execution snapshot.

        Args:
            payload (bytes): Canonical snapshot bytes produced by :meth:`to_bytes`.

        Returns:
            ExecutionSnapshot: Reconstructed immutable execution tree.

        Raises:
            SerializationError: If native preflight, integrity, or decoding fails.
        """

        inspect = getattr(native, "inspect_execution_snapshot", None)
        if not callable(inspect):
            raise SerializationError(
                "Q8402: native execution snapshot inspection is unavailable"
            )
        try:
            header = inspect(payload)
            if (
                not isinstance(header, tuple)
                or len(header) != 3
                or not isinstance(header[0], bytes)
                or len(header[0]) != 32
                or type(header[1]) is not int
                or type(header[2]) is not int
            ):
                raise TypeError("native snapshot header is malformed")
            package = snapshot_wire.ExecutionSnapshotProto()
            package.ParseFromString(payload)
            if package.SerializeToString(deterministic=True) != payload:
                raise DecodeError("execution snapshot is not canonical")
            return _decode_execution_snapshot_node(package.root)
        except (DecodeError, TypeError, ValueError) as error:
            raise SerializationError(str(error)) from error


def _freeze_execution_result(result: ExecutionResult) -> FrozenMapping:
    """Encode one validated public result as a canonical portable record.

    Args:
        result (ExecutionResult): Sample or deterministic value result.

    Returns:
        FrozenMapping: Tagged portable execution-result record.

    Raises:
        TypeError: If ``result`` is not a supported execution result.
    """

    if isinstance(result, SampleResult):
        return FrozenMapping.from_mapping(
            {
                "backend": result.backend,
                "counts": tuple((bits, count) for bits, count in result.counts.items()),
                "kind": "sample",
                "result_abi": result.result_abi._to_bytes()
                if result.result_abi
                else None,
                "worker_pid": result.worker_pid,
                **(
                    {"execution_statistics": result.execution_statistics}
                    if result.execution_statistics.entries
                    else {}
                ),
            }
        )
    if isinstance(result, ValueResult):
        return FrozenMapping.from_mapping(
            {
                "backend": result.backend,
                "kind": "value",
                "value": result.value,
                "worker_pid": result.worker_pid,
            }
        )
    raise TypeError("completed snapshot expects an execution result")


def _thaw_execution_result(value: PortableValue) -> ExecutionResult:
    """Revalidate one portable completed-result record.

    Args:
        value (PortableValue): Tagged result record from a local snapshot.

    Returns:
        ExecutionResult: Reconstructed sample or deterministic value result.

    Raises:
        SerializationError: If the record shape or value is invalid.
    """

    try:
        if not isinstance(value, FrozenMapping):
            raise TypeError("completed result is not a mapping")
        record = {name: item for name, item in value.entries}
        kind = record["kind"]
        backend = record["backend"]
        worker_pid = record.get("worker_pid")
        if type(backend) is not str or not backend:
            raise TypeError("completed result backend is invalid")
        if worker_pid is not None and type(worker_pid) is not int:
            raise TypeError("completed result worker PID is invalid")
        if kind == "value" and record.keys() == {
            "backend",
            "kind",
            "value",
            "worker_pid",
        }:
            return ValueResult(backend, thaw_value(record["value"]), worker_pid)
        statistics = record.pop("execution_statistics", FrozenMapping(()))
        if not isinstance(statistics, FrozenMapping):
            raise TypeError("execution statistics are invalid")
        if kind != "sample" or record.keys() != {
            "backend",
            "counts",
            "kind",
            "result_abi",
            "worker_pid",
        }:
            raise TypeError("completed result kind or fields are invalid")
        counts_value = record["counts"]
        if not isinstance(counts_value, tuple):
            raise TypeError("sample counts are invalid")
        counts: dict[tuple[int, ...], int] = {}
        for row in counts_value:
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], tuple)
                or any(type(bit) is not int or bit not in {0, 1} for bit in row[0])
                or type(row[1]) is not int
            ):
                raise TypeError("sample count row is invalid")
            bits = cast(tuple[int, ...], row[0])
            count = cast(int, row[1])
            counts[bits] = count
        result = SampleResult.from_counts(
            backend=backend, counts=counts, worker_pid=worker_pid
        )
        result = replace(result, execution_statistics=statistics)
        raw_abi = record["result_abi"]
        if raw_abi is not None:
            if not isinstance(raw_abi, bytes):
                raise TypeError("sample ResultABI is invalid")
            encoded = wire.ResultABIProto()
            encoded.ParseFromString(raw_abi)
            if encoded.SerializeToString(deterministic=True) != raw_abi:
                raise DecodeError("sample ResultABI is not canonical")
            result = replace(
                result,
                result_abi=ResultABI(
                    decode_type(encoded.type),
                    tuple(encoded.source_order),
                    encoded.native_width,
                ),
            )
        return result
    except (DecodeError, KeyError, TypeError, ValueError) as error:
        raise SerializationError(
            f"Q8409: invalid completed execution result: {error}"
        ) from error


def _encode_execution_snapshot_node(
    snapshot: ExecutionSnapshot,
    target: snapshot_wire.ExecutionSnapshotNodeProto,
) -> None:
    """Encode one snapshot tree node into its canonical wire form.

    Args:
        snapshot (ExecutionSnapshot): Immutable source node.
        target (ExecutionSnapshotNodeProto): Mutable protobuf target node.
    """

    if snapshot.kind is ExecutionSnapshotKind.REMOTE:
        target.kind = snapshot_wire.ExecutionSnapshotNodeProto.REMOTE
        reference = cast(ExecutionReference, snapshot.reference)
        target.reference.provider = reference.provider
        target.reference.job_ids.extend(reference.job_ids)
        if reference.target is not None:
            target.reference.target = reference.target
            target.reference.has_target = True
        if reference.group_id is not None:
            target.reference.group_id = reference.group_id
            target.reference.has_group_id = True
        for key, value in reference.context:
            target.reference.context.add(key=key, value=value)
        return
    if snapshot.kind is ExecutionSnapshotKind.LOCAL:
        target.kind = snapshot_wire.ExecutionSnapshotNodeProto.LOCAL
        encode_literal(snapshot.value, target.local_value)
        return
    target.kind = snapshot_wire.ExecutionSnapshotNodeProto.COMPOSITE
    for child in snapshot.children:
        _encode_execution_snapshot_node(child, target.children.add())


def _decode_execution_snapshot_node(
    source: snapshot_wire.ExecutionSnapshotNodeProto,
) -> ExecutionSnapshot:
    """Decode one Rust-validated snapshot node.

    Args:
        source (ExecutionSnapshotNodeProto): Native-validated protobuf node.

    Returns:
        ExecutionSnapshot: Immutable public snapshot tree.

    Raises:
        SerializationError: If the checked projection is internally malformed.
    """

    try:
        if source.kind == snapshot_wire.ExecutionSnapshotNodeProto.REMOTE:
            reference = source.reference
            return ExecutionSnapshot.remote(
                ExecutionReference(
                    reference.provider,
                    tuple(reference.job_ids),
                    reference.target if reference.has_target else None,
                    reference.group_id if reference.has_group_id else None,
                    tuple((entry.key, entry.value) for entry in reference.context),
                )
            )
        if source.kind == snapshot_wire.ExecutionSnapshotNodeProto.LOCAL:
            return ExecutionSnapshot.local(decode_literal(source.local_value))
        if source.kind == snapshot_wire.ExecutionSnapshotNodeProto.COMPOSITE:
            return ExecutionSnapshot.composite(
                tuple(
                    _decode_execution_snapshot_node(child) for child in source.children
                )
            )
        raise ValueError("unknown snapshot kind")
    except (TypeError, ValueError) as error:
        raise SerializationError(
            f"Q8407: malformed checked execution snapshot: {error}"
        ) from error


class JobStatus(str, Enum):
    """Portable job states exposed independently of provider SDK objects."""

    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@runtime_checkable
class Job(Protocol):
    """A stateful execution submission owned by an endpoint."""

    @property
    def job_id(self) -> str:
        """Return the endpoint-scoped job identity."""

        ...

    @property
    def status(self) -> JobStatus:
        """Return the current portable job status."""

        ...

    def result(self, timeout: float | None = None) -> ExecutionResult:
        """Wait for and return the typed backend result."""

        ...

    def cancel(self) -> None:
        """Request best-effort cancellation without changing result semantics."""

        ...

    def snapshot(self) -> ExecutionSnapshot:
        """Capture restorable execution state without fetching a remote result.

        Returns:
            ExecutionSnapshot: Local values or secret-free provider references.
        """

        ...


@dataclass(frozen=True, slots=True)
class CompletedJob:
    """An eager local job implementing the same interface as remote jobs."""

    job_id: str
    value: ExecutionResult

    @property
    def status(self) -> JobStatus:
        """Return the terminal completed status."""

        return JobStatus.COMPLETED

    def result(self, timeout: float | None = None) -> ExecutionResult:
        """Return the already completed result."""

        del timeout
        return self.value

    def cancel(self) -> None:
        """Leave an already completed local job unchanged."""

    def snapshot(self) -> ExecutionSnapshot:
        """Capture the completed typed result in canonical portable form.

        Returns:
            ExecutionSnapshot: Local snapshot containing this validated result.
        """

        return ExecutionSnapshot.completed(self.value)


@dataclass(frozen=True, slots=True)
class _ValidatedJob:
    """A job wrapper enforcing the artifact ResultABI at retrieval time."""

    inner: Job
    artifact: CompiledProgram
    shots: int

    @property
    def job_id(self) -> str:
        """Return the underlying endpoint job identity."""

        return self.inner.job_id

    @property
    def status(self) -> JobStatus:
        """Return the underlying endpoint job status."""

        return self.inner.status

    def result(self, timeout: float | None = None) -> ExecutionResult:
        """Retrieve and validate the endpoint result against the sealed ABI."""

        result = self.inner.result(timeout)
        if self.artifact.result_abi.width:
            if not isinstance(result, SampleResult):
                raise ExecutionError("Q8118: sampled ResultABI received a value result")
            if (
                result.shots != self.shots
                or result.width != self.artifact.result_abi.width
            ):
                raise ExecutionError(
                    "Q8119: execution result disagrees with the invocation or ResultABI"
                )
            return replace(result, result_abi=self.artifact.result_abi)
        if isinstance(result, SampleResult):
            raise ExecutionError("Q8120: value ResultABI received a sample result")
        return result

    def cancel(self) -> None:
        """Delegate best-effort cancellation to the endpoint job."""

        self.inner.cancel()

    def snapshot(self) -> ExecutionSnapshot:
        """Capture endpoint state while retaining validated local results.

        Returns:
            ExecutionSnapshot: Remote reference tree or typed completed result.
        """

        snapshot = self.inner.snapshot()
        if snapshot.kind is ExecutionSnapshotKind.LOCAL:
            return ExecutionSnapshot.completed(self.result())
        return snapshot


@dataclass(frozen=True, slots=True)
class RunRecord:
    """The visible composition of compilation, invocation, and execution."""

    artifact: CompiledProgram
    invocation: Invocation
    result: ExecutionResult

    def __post_init__(self) -> None:
        """Reject records whose immutable lifecycle values disagree."""

        if not isinstance(self.artifact, CompiledProgram):
            raise TypeError("run record artifact must be an CompiledProgram")
        if not isinstance(self.invocation, Invocation):
            raise TypeError("run record invocation must be an Invocation")
        if not isinstance(self.result, (SampleResult, ValueResult)):
            raise TypeError("run record result must be an execution result")
        if (
            isinstance(self.result, SampleResult)
            and self.result.shots != self.invocation.shots
        ):
            raise ExecutionError(
                "Q8119: execution result disagrees with the recorded invocation"
            )

    @property
    def resources(self) -> ResourceStudy:
        """Return compilation facts extended with observed execution demand."""

        if not isinstance(self.result, SampleResult):
            return self.artifact.report.resources
        evidence = (
            _verified_evidence(
                EvidenceAuthority.STRUCTURALLY_DERIVED,
                f"{self.result.backend}:validated-sample-result.v1",
            ),
        )
        metrics = {
            OBSERVED_SHOTS.identifier: ResourceValue.equality(
                OBSERVED_SHOTS,
                self.result.shots,
                evidence=evidence,
            )
        }
        statistics = dict(self.result.execution_statistics.entries)
        for definition, name in (
            (OBSERVED_SEGMENT_SHOTS, "physical_segment_shots"),
            (OBSERVED_SUBMISSIONS, "submissions"),
        ):
            if name in statistics:
                metrics[definition.identifier] = ResourceValue.equality(
                    definition, cast(int, statistics[name]), evidence=evidence
                )
        return self.artifact.report.resources.with_metrics(
            metrics,
            derivation=f"observe:{self.result.backend}",
        )


@dataclass(frozen=True, slots=True)
class LoweredProgram:
    """Backend executable and emission facts before metadata composition.

    This is an implementation boundary, not a user lifecycle value. A target
    owns SDK-specific executable lifetime; the compiler core owns resource-study
    composition and compatibility contracts. Persistence is a separate consumer.

    Targets report the stages performed while producing this result. Shared
    projection helpers contribute their compiler stages; compilation aggregates
    those records without inferring which optimizations a target performed.
    """

    payload: ExecutablePayload = field(repr=False)
    parameter_abi: ParameterABI
    result_abi: ResultABI
    stages: tuple[str, ...]
    metadata: FrozenMapping = FrozenMapping(())
    circuit_qubits: int | None = None
    circuit_gates: int | None = None
    emitted_instructions: int | None = None
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require a live executable and emission provenance."""

        if not isinstance(self.payload, ExecutablePayload):
            raise ArtifactError("Q7040: lowered payload must be a live executable")
        if not self.stages or any(not stage for stage in self.stages):
            raise ArtifactError("Q7041: lowering stages must be nonempty")


@dataclass(frozen=True, slots=True)
class _ArtifactFamilyVariant:
    """One backend artifact and its decision-tree path."""

    decisions: tuple[tuple[int, bool], ...]
    artifact: CompiledProgram


@dataclass(frozen=True, slots=True)
class _ArtifactFamilyPayload(ExecutablePayload):
    """Decoded canonical payload for one backend artifact family."""

    target_format: str
    dispatch_parameters: tuple[tuple[int, str], ...]
    selection_plan: bytes
    variants: tuple[_ArtifactFamilyVariant, ...]


@dataclass(frozen=True, slots=True)
class _LaunchMaterializerPayload(ExecutablePayload):
    """Own verified QCore and one bounded invocation-specific executable slot.

    Args:
        target_format (str): Target expected by the retained launch recipe.
        program (Program): Verified program with unresolved launch topology.
    """

    target_format: str
    program: Program
    _last: Any = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=RLock, init=False, repr=False, compare=False)


@runtime_checkable
class CompilationTarget(Protocol):
    """An immutable profile describing one artifact and capability target."""

    @property
    def name(self) -> str:
        """Return the stable artifact format name."""

        ...

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the immutable compilation capability snapshot."""

        ...

    @property
    def fingerprint(self) -> str:
        """Return the versioned toolchain and target-configuration identity."""

        ...

    def lower(self, program: Program) -> LoweredProgram:
        """Lower a closed Program without sealing it."""

        ...


@runtime_checkable
class CircuitTarget(CompilationTarget, Protocol):
    """An artifact target that emits an already verified static circuit."""

    def lower_circuit(self, circuit: CircuitProgram, /) -> LoweredProgram:
        """Emit one target-neutral static circuit without projecting QCore.

        Args:
            circuit (CircuitProgram): Verified native circuit projection.

        Returns:
            LoweredProgram: Unsealed target payload and ABI facts.

        Raises:
            CapabilityError: If the target cannot emit the circuit.
        """

        ...


@runtime_checkable
class CircuitResultPostprocessingTarget(Protocol):
    """An opt-in target that evaluates retained static-circuit result trees."""

    @property
    def supports_circuit_result_postprocessing(self) -> bool:
        """Return true only when emission preserves result expressions."""

        ...


@runtime_checkable
class LaunchTarget(CompilationTarget, Protocol):
    """A structured target that closes selected requirements at launch.

    This boundary is for target formats whose execution engine cannot accept
    the source launch ABI directly, but whose emitter can consume the verified
    QCore program once exact invocation values are known.  It deliberately
    differs from :class:`CircuitTarget`: structured targets need not
    flatten QCore into a circuit merely to participate in the artifact
    lifecycle.
    """

    @property
    def launch_materialized_requirements(self) -> frozenset[Requirement]:
        """Return requirements discharged by exact launch-time lowering."""

        ...

    def lower_at(
        self,
        program: Program,
        parameters: Mapping[str, object],
    ) -> LoweredProgram:
        """Emit one transient target artifact for exact invocation values."""

        ...


@runtime_checkable
class ExecutionEndpoint(Protocol):
    """A runtime accepting one explicit artifact format."""

    @property
    def name(self) -> str:
        """Return the stable endpoint identity."""

        ...

    @property
    def target(self) -> CompilationTarget:
        """Return the compiler target accepted by this endpoint."""

        ...

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the immutable live endpoint capability snapshot."""

        ...

    def launch(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Launch an already materialized artifact without compiling it."""

        ...


@runtime_checkable
class BatchExecutionEndpoint(ExecutionEndpoint, Protocol):
    """An execution endpoint that can launch an ordered invocation batch."""

    def launch_batch(
        self,
        artifact: CompiledProgram,
        invocations: tuple[Invocation, ...],
    ) -> tuple[Job, ...]:
        """Launch one artifact for multiple normalized invocations.

        Args:
            artifact (CompiledProgram): Validated artifact reused by every invocation.
            invocations (tuple[Invocation, ...]): ABI-normalized invocations in
                source order.

        Returns:
            tuple[Job, ...]: One job per invocation in the same order.

        Raises:
            ExecutionError: If the endpoint cannot execute the complete batch.
        """

        ...


def compile(
    program: object,
    *,
    to: CompilationTarget,
    optimizer: OptimizationPipeline | None = None,
    **bindings: object,
) -> CompiledProgram:
    """Compile a classical-input, classical-output execution entry.

    Args:
        program (object): Qamomile source, closed program, or retained resource
            study whose entry has no quantum values in its inputs or output.
            Internal quantum helpers are allowed. Use circuit inspection for
            quantum-returning components.
        to (CompilationTarget): Immutable artifact target profile used for
            planning and emission.
        optimizer (OptimizationPipeline | None): Explicit multi-level pass pipeline.
            None retains ordinary target preparation. An explicit empty pipeline
            skips optional simplification while retaining native verification.
        **bindings (object): Const values, or BlockEncoding values for a
            source qkernel, keyed by parameter name.

    Returns:
        CompiledProgram: Sealed target artifact with ABI and resource evidence.

    Raises:
        TypeError: If a public input has an invalid shape.
        ArtifactError: If target planning or emission fails.
        VerificationError: If native program verification fails.
    """

    if not isinstance(to, CompilationTarget):
        raise TypeError("to= must implement CompilationTarget")
    if optimizer is not None and not isinstance(optimizer, OptimizationPipeline):
        raise TypeError("optimizer= must be an OptimizationPipeline")
    if optimizer is not None and optimizer.circuit is not None:
        if (
            not isinstance(to, (CircuitTarget, Reference))
            and not to.capabilities.circuit_optimization
        ):
            raise ArtifactError(
                "Q7802: selected late optimization requires a circuit target or Reference"
            )
    if optimizer is not None and optimizer.family_passes:
        from .source import QKernel

        if isinstance(program, QKernel):
            source, bindings = program._bind_source_descriptors(bindings)
            program = source.family()
    optimization_source = program
    from .resources import logical
    from .serialization import _verify_native_program

    if isinstance(program, ResourceStudy):
        if bindings:
            raise TypeError("a ResourceStudy does not accept Const bindings")
        if "display-evaluation" in program.derivations:
            raise TypeError(
                "cannot compile an evaluated ResourceStudy; specialize the source "
                "program so control, liveness, and resources are re-analyzed"
            )
        study: ResourceStudy | None = program
        retained = program._semantic_source()
        if not isinstance(retained, Program):
            raise TypeError("cannot compile an unspecialized ResourceStudy")
        program = retained
    else:
        from .source import _close_program

        program = _close_program(program, bindings)
        study = None
    capabilities = to.capabilities
    if to.name != capabilities.format:
        raise ArtifactError(
            f"Q7014: target {to.name!r} advertises format {capabilities.format!r}"
        )
    try:
        _verify_native_program(program._native_program())
    except ValueError as error:
        raise VerificationError(str(error)) from error
    if optimizer is not None:
        if study is None:
            study = logical(program)
        context = OptimizationContext(
            to.fingerprint,
            frozenset(item.value for item in capabilities.requirements),
            profile=capabilities.profile,
        )
        candidate = None
        if optimizer.family_passes:
            family = optimizer.optimize_family(optimization_source, context=context)
            candidate = family.module.specialize(family.entrypoint, **bindings)
        program = optimizer._prepare(program, context, candidate=candidate)
    (
        required,
        missing,
        _materialized,
        unsupported,
        lowering_route,
        planning_stage,
    ) = _plan_target_lowering(program, to)
    if unsupported:
        raise ArtifactError(
            "Q7015: target lacks required capabilities: "
            + ", ".join(sorted(item.value for item in unsupported))
        )
    if study is None:
        study = logical(program)
    if lowering_route == "shot_execution":
        from ._shot_execution import compile_shot_program

        if not isinstance(to, CircuitTarget):
            raise ArtifactError("Q7016: shot execution requires a circuit target")
        return _with_optimization_provenance(
            compile_shot_program(program, to, required, study, planning_stage), program
        )
    if lowering_route == "circuit_family":
        if not isinstance(to, CircuitTarget):
            raise ArtifactError(
                "Q7016: native target planner selected an invalid circuit-family route"
            )
        return _with_optimization_provenance(
            _compile_artifact_family(
                program,
                to=to,
                required=required,
                planning_stage=planning_stage,
                study=study,
            ),
            program,
        )
    if lowering_route == "launch_materialized":
        return _with_optimization_provenance(
            _compile_launch_materializer(
                program,
                to=to,
                required=required,
                planning_stage=planning_stage,
                study=study,
            ),
            program,
        )
    if lowering_route != "direct" or missing:
        raise ArtifactError("Q7016: native target planner returned an invalid route")
    contract = ExecutionContract(to.name, required)
    lowered = to.lower(program)
    if not isinstance(lowered, LoweredProgram):
        raise ArtifactError("Q7042: target lower() did not return LoweredProgram")
    report = compilation_report(
        program,
        target=to.name,
        stages=(planning_stage, *lowered.stages),
        circuit_qubits=lowered.circuit_qubits,
        circuit_gates=lowered.circuit_gates,
        emitted_instructions=lowered.emitted_instructions,
        assumptions=tuple(
            dict.fromkeys(
                (
                    *lowered.assumptions,
                    *(
                        "user asserts "
                        f"{row.implementation_symbol} implements {row.target_symbol}"
                        for row in capabilities.realizations._custom
                    ),
                )
            )
        ),
        warnings=lowered.warnings,
        resources=study,
    )
    return _with_optimization_provenance(
        CompiledProgram(
            payload=lowered.payload,
            target_fingerprint=to.fingerprint,
            parameter_abi=lowered.parameter_abi,
            result_abi=lowered.result_abi,
            report=report,
            execution_contract=contract,
            metadata=lowered.metadata,
        ),
        program,
    )


def _with_optimization_provenance(
    artifact: CompiledProgram, program: Program
) -> CompiledProgram:
    """Retain native preparation identity independently of source resources.

    Args:
        artifact (CompiledProgram): Completed emission or deferred execution recipe.
        program (Program): Selected native preparation used by the target.

    Returns:
        CompiledProgram: Artifact retaining portable preparation and performed stages.

    Raises:
        ArtifactError: If target metadata shadows compiler-owned optimization facts.
    """
    receipt = program._native_program().optimization_receipt()
    if receipt is None:
        return artifact
    metadata = dict(artifact.metadata.entries)
    if OPTIMIZATION_METADATA_KEY in metadata:
        raise ArtifactError(
            "Q7802: target metadata shadows compiler optimization receipt"
        )
    metadata[OPTIMIZATION_METADATA_KEY] = receipt
    payload = artifact.payload
    if isinstance(payload, _ArtifactFamilyPayload):
        payload = replace(
            payload,
            variants=tuple(
                replace(
                    variant,
                    artifact=_with_optimization_provenance(variant.artifact, program),
                )
                for variant in payload.variants
            ),
        )
    return replace(
        artifact,
        payload=payload,
        metadata=FrozenMapping.from_mapping(metadata),
        report=replace(
            artifact.report,
            stages=(
                *program._native_program().optimization_stages(),
                *artifact.report.stages,
            ),
        ),
    )


def _compile_artifact_family(
    program: Program,
    *,
    to: CircuitTarget,
    required: frozenset[Requirement],
    planning_stage: str,
    study: ResourceStudy,
) -> CompiledProgram:
    """Emit and seal every finite Rust-derived host-dispatch variant.

    Args:
        program: Verified source program retaining the full launch ABI.
        to: Static-circuit backend target used for every family member.
        required: Authoritative whole-program execution requirements.
        planning_stage: Capability-planning stage retained in reports.
        study: Logical resource study computed once for the source program.

    Returns:
        CompiledProgram: Portable family artifact selected during submission.

    Raises:
        ArtifactError: If family projection or target emission fails, or a
            target returns inconsistent ABI facts for a variant.
    """

    from .circuit import lower_to_circuit_family
    from .errors import CapabilityError

    try:
        family = lower_to_circuit_family(
            program,
            realization_capabilities=to.capabilities.realization_capabilities,
            realizations=to.capabilities.realizations,
        )
    except CapabilityError as error:
        if error.diagnostic.code == "Q4306":
            return _compile_launch_materializer(
                program,
                to=to,
                required=required,
                planning_stage=planning_stage,
                study=study,
            )
        raise ArtifactError(
            f"Q7043: artifact-family lowering failed: {error}"
        ) from error
    child_contract = ExecutionContract(
        to.name,
        required - {Requirement.HOST_DISPATCH},
    )
    children: list[_ArtifactFamilyVariant] = []
    lowered_variants: list[LoweredProgram] = []
    for variant in family.variants:
        if variant.circuit.requires_result_postprocessing and not (
            isinstance(to, CircuitResultPostprocessingTarget)
            and to.supports_circuit_result_postprocessing is True
        ):
            raise ArtifactError(
                "Q7048: target must opt in to static-circuit result postprocessing"
            )
        lowered = to.lower_circuit(variant.circuit)
        if not isinstance(lowered, LoweredProgram):
            raise ArtifactError(
                "Q7042: target lower_circuit() did not return LoweredProgram"
            )
        if (
            lowered.parameter_abi != variant.circuit.parameter_abi
            or lowered.result_abi != family.result_abi
        ):
            raise ArtifactError("Q7044: target changed a circuit-family variant ABI")
        child_report = compilation_report(
            program,
            target=to.name,
            stages=(
                planning_stage,
                *_target_optimization_stages(program),
                "rust-circuit-family-projection",
                "host-dispatch-variant",
                *lowered.stages,
            ),
            circuit_qubits=lowered.circuit_qubits,
            circuit_gates=lowered.circuit_gates,
            emitted_instructions=lowered.emitted_instructions,
            assumptions=lowered.assumptions,
            warnings=lowered.warnings,
            resources=study,
        )
        child = CompiledProgram(
            payload=lowered.payload,
            target_fingerprint=to.fingerprint,
            parameter_abi=lowered.parameter_abi,
            result_abi=lowered.result_abi,
            report=child_report,
            execution_contract=child_contract,
            metadata=lowered.metadata,
        )
        children.append(
            _ArtifactFamilyVariant(
                tuple(
                    (decision.predicate_index, decision.value)
                    for decision in variant.decisions
                ),
                child,
            )
        )
        lowered_variants.append(lowered)
    if not family._native_selection_plan:
        raise ArtifactError("Q7045: compiler omitted the native family selector")
    payload = _ArtifactFamilyPayload(
        target_format=to.name,
        dispatch_parameters=tuple(
            (parameter.parameter_index, parameter.name)
            for parameter in family.dispatch_parameters
        ),
        selection_plan=family._native_selection_plan,
        variants=tuple(children),
    )
    assumptions = tuple(
        sorted({item for lowered in lowered_variants for item in lowered.assumptions})
    )
    warnings = tuple(
        sorted({item for lowered in lowered_variants for item in lowered.warnings})
    )
    stages = lowered_variants[0].stages if lowered_variants else ()
    circuit_qubits = tuple(item.circuit_qubits for item in lowered_variants)
    circuit_gates = tuple(item.circuit_gates for item in lowered_variants)
    emitted_instructions = tuple(item.emitted_instructions for item in lowered_variants)
    report = compilation_report(
        program,
        target=to.name,
        stages=(
            planning_stage,
            *_target_optimization_stages(program),
            "rust-circuit-family-projection",
            "host-dispatch-artifact-family",
            *stages,
        ),
        circuit_qubits=(
            max(cast(tuple[int, ...], circuit_qubits))
            if all(item is not None for item in circuit_qubits)
            else None
        ),
        circuit_gates=(
            sum(cast(tuple[int, ...], circuit_gates))
            if all(item is not None for item in circuit_gates)
            else None
        ),
        emitted_instructions=(
            sum(cast(tuple[int, ...], emitted_instructions))
            if all(item is not None for item in emitted_instructions)
            else None
        ),
        assumptions=assumptions,
        warnings=warnings,
        resources=study,
    )
    return CompiledProgram(
        payload=payload,
        target_fingerprint=to.fingerprint,
        parameter_abi=family.parameter_abi,
        result_abi=family.result_abi,
        report=report,
        execution_contract=ExecutionContract(ARTIFACT_FAMILY_FORMAT, required),
        metadata=FrozenMapping.from_mapping(
            {
                "dispatch_target_format": to.name,
                "dispatch_variants": len(children),
            }
        ),
    )


def _compile_launch_materializer(
    program: Program,
    *,
    to: CompilationTarget,
    required: frozenset[Requirement],
    planning_stage: str,
    study: ResourceStudy,
) -> CompiledProgram:
    """Retain a verified program for bounded target materialization at launch.

    Args:
        program: Verified source program retaining launch-dependent topology.
        to: Target used after invocation values are known.
        required: Authoritative whole-program execution requirements.
        planning_stage: Capability-planning stage retained in reports.
        study: Logical resource study computed once for the source program.

    Returns:
        CompiledProgram: Target-bound materialization recipe.

    Raises:
        ArtifactError: If the compilation metadata or retained source is invalid.
    """

    payload = _LaunchMaterializerPayload(to.name, program)
    report = compilation_report(
        program,
        target=to.name,
        stages=(
            planning_stage,
            "rust-launch-materialization-plan",
            "retained-launch-materializer",
        ),
        resources=study,
    )
    return CompiledProgram(
        payload=payload,
        target_fingerprint=to.fingerprint,
        parameter_abi=ParameterABI.from_program(program),
        result_abi=ResultABI.from_program(program),
        report=report,
        execution_contract=ExecutionContract(LAUNCH_MATERIALIZER_FORMAT, required),
        metadata=FrozenMapping.from_mapping({"materialization_target_format": to.name}),
    )


def _launch_materialized_requirements(
    target: CompilationTarget,
    required: frozenset[Requirement],
) -> frozenset[Requirement]:
    """Return required capabilities an exact launch can discharge.

    Args:
        target: Candidate artifact target.
        required: Whole-program execution requirements.

    Returns:
        Required capabilities that the target promises to close before its
        execution endpoint receives the transient artifact.

    Raises:
        ArtifactError: If a structured target advertises an invalid empty or
            non-requirement materialization contract.
    """

    materialized, _ = _target_launch_profile(target)
    return materialized & (required - target.capabilities.requirements)


def _target_launch_profile(
    target: CompilationTarget,
) -> tuple[frozenset[Requirement], bool]:
    """Normalize one target's transient launch-lowering protocols.

    Args:
        target: Candidate artifact target.

    Returns:
        The declared materializable requirements and whether the target can
        emit finite circuit families.

    Raises:
        ArtifactError: If the target advertises ambiguous or invalid launch
            protocols.
    """

    if isinstance(target, CircuitTarget) and isinstance(target, LaunchTarget):
        raise ArtifactError("Q7046: target has ambiguous launch protocols")
    if isinstance(target, CircuitTarget):
        return _CIRCUIT_LAUNCH_CLOSABLE_REQUIREMENTS, True
    if not isinstance(target, LaunchTarget):
        return frozenset(), False
    materialized = target.launch_materialized_requirements
    if (
        not isinstance(materialized, frozenset)
        or not materialized
        or any(not isinstance(item, Requirement) for item in materialized)
        or not materialized.issubset(_LAUNCH_CLOSABLE_REQUIREMENTS)
    ):
        raise ArtifactError("Q7046: invalid launch-materialization contract")
    return materialized, False


def _select_artifact_family_variant(
    artifact: CompiledProgram,
    parameters: Mapping[str, object],
    *,
    target_format: str,
) -> CompiledProgram:
    """Validate a family artifact and select one child backend artifact.

    Args:
        artifact: Outer family artifact carrying the complete launch ABI.
        parameters: Already validated and default-filled launch values.
        target_format: Format expected by the selected execution endpoint.

    Returns:
        CompiledProgram: Exact static child artifact for this invocation.

    Raises:
        ExecutionError: If the family disagrees with its outer artifact or no
            verified predicate path matches the launch values.
    """

    family = artifact.execution_payload(_ArtifactFamilyPayload)
    if family.target_format != target_format:
        raise ExecutionError(
            "Q8131: artifact-family target format does not match the endpoint"
        )
    dispatch_names = tuple(name for _, name in family.dispatch_parameters)
    outer_slots = {
        slot.name: (index, slot)
        for index, slot in enumerate(artifact.parameter_abi.slots)
    }
    dispatch_positions: list[int] = []
    for name in dispatch_names:
        outer = outer_slots.get(name)
        if outer is None:
            raise ExecutionError("Q8131: artifact-family dispatch ABI mismatch")
        slot = outer[1]
        if not isinstance(slot.type, StagedType) or slot.type.stage != "Param":
            raise ExecutionError("Q8131: artifact-family dispatch ABI mismatch")
        dispatch_positions.append(outer[0])
    if dispatch_positions != sorted(dispatch_positions):
        raise ExecutionError("Q8131: artifact-family dispatch ABI mismatch")
    expected_requirements = artifact.execution_contract.requirements - {
        Requirement.HOST_DISPATCH
    }
    if Requirement.HOST_DISPATCH not in artifact.execution_contract.requirements:
        raise ExecutionError("Q8131: artifact-family contract omits host dispatch")
    for variant in family.variants:
        child = variant.artifact
        child_positions: list[int] = []
        child_abi_valid = True
        for child_slot in child.parameter_abi.slots:
            outer = outer_slots.get(child_slot.name)
            if outer is None or outer[1] != child_slot:
                child_abi_valid = False
                break
            child_positions.append(outer[0])
        if (
            child.format != family.target_format
            or child.target_fingerprint != artifact.target_fingerprint
            or not child_abi_valid
            or child_positions != sorted(child_positions)
            or child.result_abi != artifact.result_abi
            or child.source_digest != artifact.source_digest
            or child.execution_contract.requirements != expected_requirements
        ):
            raise ExecutionError("Q8131: child artifact disagrees with its family")
    select = getattr(native, "select_circuit_family_variant", None)
    if not callable(select):
        raise ExecutionError("Q8131: native artifact-family selection is unavailable")
    try:
        selected_index = select(
            family.selection_plan,
            {name: parameters[name] for name in dispatch_names},
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q8131: native artifact-family selection failed: {error}"
        ) from error
    if type(selected_index) is not int or not 0 <= selected_index < len(
        family.variants
    ):
        raise ExecutionError(
            "Q8131: native artifact-family selector returned an invalid variant"
        )
    return family.variants[selected_index].artifact


def _materialize_launch_artifact(
    artifact: CompiledProgram,
    parameters: Mapping[str, object],
    *,
    target: CompilationTarget,
) -> CompiledProgram:
    """Reuse the most recent exact launch materialization without persistence.

    Args:
        artifact (CompiledProgram): Verified target-bound launch recipe.
        parameters (Mapping): Complete bound invocation values.
        target (CompilationTarget): Endpoint's exact target profile.

    Returns:
        CompiledProgram: Retained child for these values, with independent shot state.
    """
    recipe = artifact.execution_payload(_LaunchMaterializerPayload)
    # Structured targets may bake every launch value into their executable.
    # Circuit projection keeps non-control values in the child parameter ABI.
    if isinstance(target, CircuitTarget):
        from .circuit import launch_control_bindings

        consumed = launch_control_bindings(recipe.program, parameters)
    else:
        consumed = parameters
    key = (
        target.name,
        target.fingerprint,
        FrozenMapping.from_mapping(consumed),
        artifact.parameter_abi,
        artifact.result_abi,
        artifact.source_digest,
        artifact.execution_contract,
    )
    with recipe._lock:
        if recipe._last is not None and recipe._last[0] == key:
            return recipe._last[1]
        child = _materialize_launch_artifact_uncached(
            artifact, parameters, target=target
        )
        object.__setattr__(recipe, "_last", (key, child))
        return child


def _materialize_launch_artifact_uncached(
    artifact: CompiledProgram,
    parameters: Mapping[str, object],
    *,
    target: CompilationTarget,
) -> CompiledProgram:
    """Materialize one sealed launch-controlled program for a static target.

    Args:
        artifact: Outer target-bound materialization recipe.
        parameters: Validated and default-filled launch parameter values.
        target: Endpoint target whose fingerprint already matches the recipe.

    Returns:
        CompiledProgram: Transient backend artifact for this invocation.

    Raises:
        ExecutionError: If the recipe, native projection, or target emission
            violates the sealed source and ABI contracts.
    """

    materialized_requirements = _launch_materialized_requirements(
        target,
        artifact.execution_contract.requirements,
    )
    if not materialized_requirements:
        raise ExecutionError("Q8133: endpoint cannot materialize launch requirements")
    program = _decode_launch_materializer_program(
        artifact,
        target_format=target.name,
        materialized_requirements=materialized_requirements,
    )
    try:
        if isinstance(target, CircuitTarget):
            from .circuit import launch_control_bindings, lower_to_circuit_at

            closed_parameters = launch_control_bindings(program, parameters)

            circuit = lower_to_circuit_at(
                program,
                parameters,
                realization_capabilities=target.capabilities.realization_capabilities,
                realizations=target.capabilities.realizations,
            )
            if circuit.requires_result_postprocessing and not (
                isinstance(target, CircuitResultPostprocessingTarget)
                and target.supports_circuit_result_postprocessing is True
            ):
                raise CapabilityError(
                    "Q7246: target must opt in to static-circuit result postprocessing"
                )
            lowered = target.lower_circuit(circuit)
            expected_parameter_abi = circuit.parameter_abi
            preparation_stages: tuple[str, ...] = _target_optimization_stages(program)
            materialization_stage = "rust-launch-circuit-materialization"
        elif isinstance(target, LaunchTarget):
            closed_parameters = parameters
            lowered = target.lower_at(program, parameters)
            expected_parameter_abi = (
                ParameterABI()
                if Requirement.PARAMETERIZED_EXECUTION in materialized_requirements
                else lowered.parameter_abi
            )
            preparation_stages = ()
            materialization_stage = "rust-launch-qcore-materialization"
        else:  # pragma: no cover - guarded by the contract helper above
            raise TypeError("target has no launch materialization boundary")
    except (ArtifactError, CapabilityError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q8134: launch materialization failed: {error}"
        ) from error
    if not isinstance(lowered, LoweredProgram):
        raise ExecutionError(
            "Q8134: target lower_circuit() did not return LoweredProgram"
        )
    outer_slots = {
        slot.name: (index, slot)
        for index, slot in enumerate(artifact.parameter_abi.slots)
    }
    child_positions: list[int] = []
    for slot in lowered.parameter_abi.slots:
        outer = outer_slots.get(slot.name)
        if outer is None or outer[1] != slot:
            raise ExecutionError("Q8134: materialized target changed the parameter ABI")
        child_positions.append(outer[0])
    if (
        lowered.parameter_abi != expected_parameter_abi
        or child_positions != sorted(child_positions)
        or lowered.result_abi != artifact.result_abi
    ):
        raise ExecutionError("Q8134: materialized target changed the circuit ABI")
    report = compilation_report(
        program,
        target=target.name,
        stages=(
            *artifact.report.stages,
            *preparation_stages,
            materialization_stage,
            *lowered.stages,
        ),
        circuit_qubits=lowered.circuit_qubits,
        circuit_gates=lowered.circuit_gates,
        emitted_instructions=lowered.emitted_instructions,
        assumptions=lowered.assumptions,
        warnings=lowered.warnings,
        resources=artifact.report.resources,
    )
    return CompiledProgram(
        payload=lowered.payload,
        target_fingerprint=artifact.target_fingerprint,
        parameter_abi=lowered.parameter_abi,
        result_abi=lowered.result_abi,
        report=report,
        execution_contract=ExecutionContract(
            target.name,
            artifact.execution_contract.requirements - materialized_requirements,
        ),
        metadata=FrozenMapping(
            tuple(
                sorted(
                    (
                        *lowered.metadata.entries,
                        *(
                            (name, value)
                            for name, value in artifact.metadata.entries
                            if name == OPTIMIZATION_METADATA_KEY
                        ),
                        (
                            "qamomile.launch_parameters",
                            FrozenMapping.from_mapping(closed_parameters),
                        ),
                    )
                )
            )
        ),
    )


def _decode_launch_materializer_program(
    artifact: CompiledProgram,
    *,
    target_format: str,
    materialized_requirements: frozenset[Requirement],
) -> Program:
    """Decode and validate the program sealed into a launch materializer.

    Args:
        artifact: Outer target-bound materialization recipe.
        target_format: Static child format expected by the endpoint target.
        materialized_requirements: Capabilities the target closes before
            execution.

    Returns:
        Program: Canonical verified program retained by the recipe.

    Raises:
        ExecutionError: If payload or artifact identities disagree.
    """

    recipe = artifact.execution_payload(_LaunchMaterializerPayload)
    program = recipe.program
    receipt = dict(artifact.metadata.entries).get(OPTIMIZATION_METADATA_KEY)
    if receipt is not None:
        source = artifact.report.resources._semantic_source()
        if not isinstance(source, Program) or not isinstance(receipt, bytes):
            raise ExecutionError("Q7802: malformed optimization recipe")
        program = _restore_optimization(source, program, receipt)
    required, _, _ = _plan_target_requirements(program, frozenset(Requirement))
    if (
        recipe.target_format != target_format
        or program._native_program().compilation_source_digest()
        != artifact.source_digest
        or ParameterABI.from_program(program) != artifact.parameter_abi
        or ResultABI.from_program(program) != artifact.result_abi
        or required != artifact.execution_contract.requirements
        or not materialized_requirements
        or not materialized_requirements <= required
    ):
        raise ExecutionError(
            "Q8133: launch recipe disagrees with its artifact contract"
        )
    return program


def _plan_target_requirements(
    program: Program,
    supported: frozenset[Requirement],
) -> tuple[frozenset[Requirement], frozenset[Requirement], str]:
    """Plan semantic target compatibility through the native compiler.

    Args:
        program: Closed verified program whose entry contract is authoritative.
        supported: Immutable target capability profile.

    Returns:
        Required and missing capabilities plus the retained planning-stage
        identity.

    Raises:
        ArtifactError: If an installed native planner rejects the request or
            returns an internally inconsistent capability plan.
    """

    planner = getattr(native, "plan_target", None)
    if not callable(planner):
        raise ArtifactError("Q7016: installed native compiler has no target planner")
    labels = [item.value for item in Requirement if item in supported]
    try:
        plan = planner(program._native_program(), labels)
        required = frozenset(Requirement(item) for item in plan.required)
        native_supported = frozenset(Requirement(item) for item in plan.supported)
        missing = frozenset(Requirement(item) for item in plan.missing)
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"Q7016: native target planning failed: {error}") from error
    if (
        native_supported != supported
        or bool(plan.compatible) != (not missing)
        or missing != required - native_supported
    ):
        raise ArtifactError(
            "Q7016: native target plan disagrees with its input contract"
        )
    return required, missing, "native-capability-planning"


def _plan_target_lowering(
    program: Program,
    target: CompilationTarget,
) -> tuple[
    frozenset[Requirement],
    frozenset[Requirement],
    frozenset[Requirement],
    frozenset[Requirement],
    str,
    str,
]:
    """Select one transient target-lowering route through Rust.

    Args:
        program: Closed verified program whose requirements are authoritative.
        target: Target supplying capabilities and launch protocol facts.

    Returns:
        Required, missing, materialized, and unsupported requirement sets,
        followed by the selected route and planning-stage identity.

    Raises:
        ArtifactError: If target protocols are invalid or an installed native
            planner rejects or contradicts the request.
    """

    supported = target.capabilities.requirements
    promised, circuit_family = _target_launch_profile(target)
    planner = getattr(native, "plan_target_lowering", None)
    if not callable(planner):
        raise ArtifactError("Q7016: installed native compiler has no lowering planner")
    supported_labels = [item.value for item in Requirement if item in supported]
    promised_labels = [item.value for item in Requirement if item in promised]
    try:
        plan = planner(
            program._native_program(),
            supported_labels,
            promised_labels,
            circuit_family,
        )
        required = frozenset(Requirement(item) for item in plan.required)
        native_supported = frozenset(Requirement(item) for item in plan.supported)
        missing = frozenset(Requirement(item) for item in plan.missing)
        materialized = frozenset(Requirement(item) for item in plan.materialized)
        unsupported = frozenset(Requirement(item) for item in plan.unsupported)
        route = plan.route
    except (TypeError, ValueError) as error:
        raise ArtifactError(
            f"Q7016: native lowering planning failed: {error}"
        ) from error
    if (
        native_supported != supported
        or missing != required - supported
        or materialized != missing & promised
        or unsupported != missing - materialized
        or route
        not in {
            "direct",
            "circuit_family",
            "launch_materialized",
            "shot_execution",
            "unsupported",
        }
    ):
        raise ArtifactError("Q7016: native lowering plan is internally inconsistent")
    return (
        required,
        missing,
        materialized,
        unsupported,
        route,
        "native-lowering-planning",
    )


def execute(
    artifact: CompiledProgram,
    *,
    on: ExecutionEndpoint,
    parameters: Mapping[str, object] | None = None,
    shots: int = 1,
    seed: int = 0,
) -> ExecutionResult:
    """Execute an already compiled CompiledProgram and return its raw result.

    This low-level synchronous path is useful when callers deliberately keep
    compilation and execution separate. Most applications should use
    :func:`run`, which also retains the CompiledProgram and normalized Invocation in a
    :class:`RunRecord`.

    Args:
        artifact (CompiledProgram): Compiled immutable target artifact.
        on (ExecutionEndpoint): Endpoint whose target and capabilities accept
            ``artifact``.
        parameters (Mapping[str, object] | None): Runtime Param values keyed by
            source parameter name.
        shots (int): Positive number of observations to request.
        seed (int): Unsigned deterministic execution seed.

    Returns:
        ExecutionResult: Normalized sampled or deterministic result.

    Raises:
        TypeError: If the artifact, endpoint, or parameters have an invalid
            public shape.
        ValueError: If an invocation value is outside its accepted range.
        ExecutionError: If launch or result validation fails.
    """

    return submit(
        artifact,
        on=on,
        parameters=parameters,
        shots=shots,
        seed=seed,
    ).result()


def execute_batch(
    artifact: CompiledProgram,
    *,
    on: ExecutionEndpoint,
    parameters: Sequence[Mapping[str, object]],
    shots: int = 1,
    seed: int = 0,
) -> tuple[ExecutionResult, ...]:
    """Execute an ordered parameter batch against one compiled artifact.

    Each row is an independent invocation. Seeds advance from ``seed`` in
    source order, and returned results retain that same order. Reference
    execution reuses the retained verified owner; other endpoints retain
    their ordinary per-invocation launch behavior.

    Args:
        artifact (CompiledProgram): Compiled immutable target artifact reused
            for every invocation.
        on (ExecutionEndpoint): Endpoint whose target and capabilities accept
            ``artifact``.
        parameters (Sequence[Mapping[str, object]]): Ordered runtime parameter
            mappings. At most 4096 invocations may be submitted together.
        shots (int): Positive observation count used for every invocation.
        seed (int): Unsigned 64-bit seed for the first row. Each later row uses
            the next seed.

    Returns:
        tuple[ExecutionResult, ...]: Validated results in parameter-row order.

    Raises:
        TypeError: If the artifact, endpoint, or parameter batch has an invalid
            public shape.
        ValueError: If the batch is too large or an invocation value is outside
            its accepted range.
        ExecutionError: If the artifact cannot run on the endpoint or execution
            fails or returns a result inconsistent with the sealed ABI.
    """

    if not isinstance(artifact, CompiledProgram):
        raise TypeError("execute_batch expects an CompiledProgram")
    if not isinstance(on, ExecutionEndpoint):
        raise TypeError("on= must implement ExecutionEndpoint")
    if not isinstance(parameters, Sequence):
        raise TypeError("parameters= must be an ordered sequence of mappings")
    if len(parameters) > _MAX_EXECUTION_BATCH_INVOCATIONS:
        raise ValueError(
            "execute_batch accepts at most "
            f"{_MAX_EXECUTION_BATCH_INVOCATIONS} invocations"
        )
    if artifact.target_fingerprint != on.target.fingerprint:
        raise ExecutionError(
            "Q8129: artifact target does not match execution endpoint target"
        )

    rows = tuple(parameters)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("parameters= rows must be mappings")
    if (
        not isinstance(on, BatchExecutionEndpoint)
        or artifact.format == "qamomile.shot-execution.v1"
    ):
        return tuple(
            execute(
                artifact,
                on=on,
                parameters=row,
                shots=shots,
                seed=seed + index,
            )
            for index, row in enumerate(rows)
        )

    invocations: list[Invocation] = []
    for index, row in enumerate(rows):
        invocation = Invocation.create(
            parameters=row,
            shots=shots,
            seed=seed + index,
        )
        bound = artifact.parameter_abi.bind(invocation.values())
        invocations.append(
            Invocation(
                FrozenMapping.from_mapping(dict(bound)),
                invocation.shots,
                invocation.seed,
            )
        )
    jobs = on.launch_batch(artifact, tuple(invocations))
    if (
        not isinstance(jobs, tuple)
        or len(jobs) != len(invocations)
        or any(not isinstance(job, Job) for job in jobs)
    ):
        raise ExecutionError(
            "Q8128: endpoint batch launch did not return one Job per invocation"
        )
    return tuple(
        _ValidatedJob(job, artifact, invocation.shots).result()
        for job, invocation in zip(jobs, invocations, strict=True)
    )


def submit(
    artifact: CompiledProgram,
    *,
    on: ExecutionEndpoint,
    parameters: Mapping[str, object] | None = None,
    shots: int = 1,
    seed: int = 0,
) -> Job:
    """Validate and submit an existing artifact without recompiling source."""

    return _submit_with_invocation(
        artifact,
        on=on,
        parameters=parameters,
        shots=shots,
        seed=seed,
    )[0]


def _submit_with_invocation(
    artifact: CompiledProgram,
    *,
    on: ExecutionEndpoint,
    parameters: Mapping[str, object] | None = None,
    shots: int = 1,
    seed: int = 0,
) -> tuple[Job, Invocation]:
    """Submit once and retain the Rust-normalized outer invocation."""

    if not isinstance(artifact, CompiledProgram):
        raise TypeError("submit expects an CompiledProgram")
    if not isinstance(on, ExecutionEndpoint):
        raise TypeError("on= must implement ExecutionEndpoint")
    if artifact.target_fingerprint != on.target.fingerprint:
        raise ExecutionError(
            "Q8129: artifact target does not match execution endpoint target"
        )
    invocation = Invocation.create(parameters=parameters, shots=shots, seed=seed)
    bound = artifact.parameter_abi.bind(invocation.values())
    recorded_invocation = Invocation(
        FrozenMapping.from_mapping(dict(bound)),
        invocation.shots,
        invocation.seed,
    )
    from ._shot_execution import ShotExecutable, ShotJob

    if isinstance(artifact.payload, ShotExecutable):
        if artifact.payload.target_format != on.target.name:
            raise ExecutionError("Q8129: shot plan target does not match endpoint")
        on.capabilities.require(
            ExecutionContract(on.target.name, artifact.execution_contract.requirements)
        )
        job = ShotJob(artifact, on, recorded_invocation)
        return _ValidatedJob(job, artifact, invocation.shots), recorded_invocation
    selected_artifact = artifact
    if artifact.format == LAUNCH_MATERIALIZER_FORMAT:
        selected_artifact = _materialize_launch_artifact(
            artifact,
            dict(bound),
            target=on.target,
        )
        selected_names = {slot.name for slot in selected_artifact.parameter_abi.slots}
        bound = tuple((name, value) for name, value in bound if name in selected_names)
    elif artifact.format == ARTIFACT_FAMILY_FORMAT:
        selected_artifact = _select_artifact_family_variant(
            artifact,
            dict(bound),
            target_format=on.target.name,
        )
        selected_names = {slot.name for slot in selected_artifact.parameter_abi.slots}
        bound = tuple((name, value) for name, value in bound if name in selected_names)
    on.capabilities.require(selected_artifact.execution_contract)
    normalized = Invocation(
        FrozenMapping.from_mapping(dict(bound)),
        invocation.shots,
        invocation.seed,
    )
    job = on.launch(selected_artifact, normalized)
    if not isinstance(job, Job):
        raise ExecutionError("Q8128: endpoint launch did not return a Job")
    return _ValidatedJob(job, artifact, invocation.shots), recorded_invocation


def run(
    program: object,
    *,
    on: ExecutionEndpoint,
    parameters: Mapping[str, object] | None = None,
    shots: int = 1,
    seed: int = 0,
    optimizer: OptimizationPipeline | None = None,
    **bindings: object,
) -> RunRecord:
    """Compile or reuse one artifact and retain the completed lifecycle.

    Args:
        program (object): Qamomile source, closed program, resource study, or
            existing artifact.
        on (ExecutionEndpoint): Endpoint used for target selection and launch.
        parameters (Mapping[str, object] | None): Runtime Param values keyed by
            source parameter name.
        shots (int): Positive number of observations to request.
        seed (int): Unsigned deterministic execution seed.
        optimizer (OptimizationPipeline | None): Pipeline used when compiling
            source; existing artifacts cannot select another optimizer.
        **bindings (object): Const values, or BlockEncoding values for a
            source qkernel, keyed by parameter name.

    Returns:
        RunRecord: Immutable artifact, invocation, and normalized result.

    Raises:
        TypeError: If an existing artifact receives Const bindings or an input
            has an invalid public shape.
        ArtifactError: If compilation or target lowering fails.
        ExecutionError: If launch or result validation fails.
    """

    if isinstance(program, CompiledProgram):
        if bindings or optimizer is not None:
            raise TypeError("an CompiledProgram does not accept Const bindings")
        artifact = program
    else:
        artifact = compile(program, to=on.target, optimizer=optimizer, **bindings)
    job, invocation = _submit_with_invocation(
        artifact,
        on=on,
        parameters=parameters,
        shots=shots,
        seed=seed,
    )
    return RunRecord(artifact=artifact, invocation=invocation, result=job.result())


def evaluate(
    program: Program,
    /,
    *,
    seed: int = 0,
    **parameters: object,
) -> object:
    """Evaluate one program shot through the verified Reference runtime.

    Args:
        program (Program): Closed verified program to compile and execute.
        seed (int): Deterministic unsigned 64-bit seed for quantum execution.
        **parameters (object): Launch-time values keyed by source parameter
            name.

    Returns:
        object: A typed sampled value for a shot result, or the deterministic
        value produced by a non-sampled program.

    Raises:
        ExecutionError: If invocation binding or native execution fails.
        VerificationError: If native program verification rejects the input.

    Notes:
        This convenience API shares native program verification, ABI binding,
        execution, and typed result reconstruction with :func:`run` without
        constructing a resource-bearing artifact. The required compiler's
        unified Reference execution result is authoritative.
    """

    if not isinstance(program, Program):
        raise TypeError("evaluate expects a Program")
    from .serialization import _verify_native_program

    payload = program.to_bytes()
    invocation = Invocation.create(parameters=parameters, shots=1, seed=seed)
    try:
        _verify_native_program(payload)
    except ValueError as error:
        raise VerificationError(str(error)) from error
    bound = dict(ParameterABI.from_program(program).bind(invocation.values()))
    result_abi = ResultABI.from_program(program)
    execution = _native_reference_execution(
        payload,
        bound,
        result_abi,
        shots=1,
        seed=invocation.seed,
    )
    counts, value = execution
    if counts is None:
        return value
    sample = SampleResult.from_counts(
        backend="Reference",
        counts=counts,
        expected_width=result_abi.width,
    )
    if sample.shots != 1:
        raise ExecutionError(
            "Q8142: native one-shot Reference execution returned "
            "an invalid observation count"
        )
    return result_abi.decode_value(sample.most_likely)


def compilation_report(
    program: Program,
    *,
    target: str,
    stages: tuple[str, ...],
    circuit_qubits: int | None = None,
    circuit_gates: int | None = None,
    emitted_instructions: int | None = None,
    assumptions: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    resources: ResourceStudy | None = None,
) -> CompilationReport:
    """Create a coordinate-safe report for one backend emission pipeline."""

    from .resources import logical

    study = resources or logical(program)
    if (
        study.semantic_digest
        != program._native_program().compilation_source_digest().hex()
    ):
        raise ArtifactError("Q7043: compilation resources describe another program")
    evidence = (
        _verified_evidence(
            EvidenceAuthority.STRUCTURALLY_DERIVED,
            f"{target}:emission-report.v1",
        ),
    )
    metrics: dict[str, ResourceValue] = {}
    for definition, value in (
        (CIRCUIT_VIRTUAL_QUBITS, circuit_qubits),
        (CIRCUIT_VIRTUAL_GATES, circuit_gates),
        (EMITTED_INSTRUCTIONS, emitted_instructions),
    ):
        if value is not None:
            metrics[definition.identifier] = ResourceValue.equality(
                definition, value, evidence=evidence
            )
    if metrics:
        study = study.with_metrics(
            metrics,
            derivation=f"emit:{target}",
        )
    return CompilationReport(
        stages,
        study,
        assumptions=assumptions,
        warnings=warnings,
    )


class _ReferenceExecutable(ExecutablePayload):
    """Own native verified execution state and its immutable ABI projection.

    Args:
        payload (bytes): Optimized canonical QCore at the compiler boundary.
    """

    def __init__(self, payload: bytes) -> None:
        """Restore the native handle and facade once, before any invocation.

        Args:
            payload (bytes): Optimized canonical QCore program.

        Raises:
            ExecutionError: If the native Reference handle is unavailable.
            ValueError: If native verification rejects the program.
        """
        handle = getattr(native, "ReferenceProgramHandle", None)
        if not callable(handle):
            raise ExecutionError("Q8145: native Reference handle is unavailable")
        self.handle = handle(payload)
        self.program = Program.from_bytes(payload)
        self.parameter_abi = ParameterABI.from_program(self.program)
        self.result_abi = ResultABI.from_program(self.program)

    def to_bytes(self) -> bytes:
        """Export the optimized canonical program for explicit persistence.

        Returns:
            bytes: Canonical QCore program owned by the executable.
        """
        return self.program.to_bytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> _ReferenceExecutable:
        """Restore one saved Reference program before repeated execution.

        Args:
            payload (bytes): Saved canonical QCore program.

        Returns:
            _ReferenceExecutable: Native handle with cached verified ABI projections.
        """
        return cls(payload)


def _native_reference_execution(
    payload: bytes | _ReferenceExecutable,
    parameters: Mapping[str, object],
    result_abi: ResultABI,
    *,
    shots: int,
    seed: int,
) -> tuple[dict[tuple[int, ...], int] | None, object]:
    """Execute one Reference invocation through the unified native bridge.

    Args:
        payload (bytes | _ReferenceExecutable): Canonical QCore or retained native owner.
        parameters: Complete ABI-normalized launch values.
        result_abi: Verified logical result ABI used to validate the native
            execution route and restore record values.
        shots: Positive observation count.
        seed: Unsigned 64-bit deterministic seed for quantum effects.

    Returns:
        A pair containing either canonical sample counts or an externalized
        value.

    Raises:
        ExecutionError: If an installed bridge is incomplete, rejects the
            invocation, chooses a route inconsistent with the ABI, or returns
            malformed data.
    """

    executor = getattr(native, "execute_reference_at", None)
    if not callable(executor):
        raise ExecutionError("Q8145: native Reference execution is unavailable")
    try:
        execution = executor(
            payload.handle if isinstance(payload, _ReferenceExecutable) else payload,
            dict(parameters),
            shots,
            seed,
        )
    except NotImplementedError as error:
        raise ExecutionError(
            f"Q8145: native Reference execution is unsupported: {error}"
        ) from error
    except (OverflowError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q8145: native Reference execution failed: {error}"
        ) from error
    return _native_reference_projection(execution, result_abi)


def _native_reference_batch_execution(
    payload: bytes | _ReferenceExecutable,
    parameters: tuple[Mapping[str, object], ...],
    result_abi: ResultABI,
    *,
    shots: int,
    seed: int,
) -> tuple[tuple[dict[tuple[int, ...], int] | None, object], ...]:
    """Execute ordered Reference invocations through one native restoration.

    Args:
        payload (bytes | _ReferenceExecutable): Canonical QCore or retained native owner.
        parameters (tuple[Mapping[str, object], ...]): ABI-normalized parameter
            rows in invocation order.
        result_abi (ResultABI): Verified result ABI used to check every native
            projection.
        shots (int): Positive observation count shared by every row.
        seed (int): Unsigned 64-bit seed for the first row.

    Returns:
        tuple[tuple[dict[tuple[int, ...], int] | None, object], ...]: Ordered
        native counts-or-value pairs.

    Raises:
        ExecutionError: If the native batch boundary is absent, rejects an
            invocation, or returns malformed or ABI-inconsistent results.
    """

    executor = getattr(native, "execute_reference_batch_at", None)
    if not callable(executor):
        raise ExecutionError("Q8145: native Reference batch execution is unavailable")
    try:
        executions = executor(
            payload.handle if isinstance(payload, _ReferenceExecutable) else payload,
            [dict(row) for row in parameters],
            shots,
            seed,
        )
    except NotImplementedError as error:
        raise ExecutionError(
            f"Q8145: native Reference batch execution is unsupported: {error}"
        ) from error
    except (OverflowError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q8145: native Reference batch execution failed: {error}"
        ) from error
    if not isinstance(executions, list) or len(executions) != len(parameters):
        raise ExecutionError(
            "Q8145: native Reference batch execution returned a malformed result"
        )
    return tuple(
        _native_reference_projection(execution, result_abi) for execution in executions
    )


def _native_reference_projection(
    execution: object,
    result_abi: ResultABI,
) -> tuple[dict[tuple[int, ...], int] | None, object]:
    """Validate one native Reference execution projection.

    Args:
        execution (object): Native projection containing kind, counts, and
            optional encoded value.
        result_abi (ResultABI): Sealed logical result ABI for validation.

    Returns:
        tuple[dict[tuple[int, ...], int] | None, object]: Canonical counts or a
        decoded value.

    Raises:
        ExecutionError: If the native projection is malformed or disagrees with
            the result ABI.
    """

    try:
        projection = cast(Any, execution)
        kind = projection.kind
        raw_counts = projection.counts
        payload_value = projection.value
    except (AttributeError, TypeError) as error:
        raise ExecutionError(
            "Q8145: native Reference execution returned a malformed result"
        ) from error
    if result_abi.width:
        if kind != "samples" or payload_value is not None:
            raise ExecutionError(
                "Q8145: native Reference execution disagrees with the sampled ABI"
            )
        try:
            entries = tuple((tuple(bits), count) for bits, count in raw_counts)
            keys = tuple(bits for bits, _ in entries)
            canonical = bool(entries) and keys == tuple(sorted(keys))
            unique = len(keys) == len(set(keys))
        except (TypeError, ValueError) as error:
            raise ExecutionError(
                "Q8145: native Reference execution returned malformed counts"
            ) from error
        if not canonical or not unique:
            raise ExecutionError(
                "Q8145: native Reference execution returned malformed counts"
            )
        counts = dict(entries)
        try:
            SampleResult.from_counts(
                backend="Reference",
                counts=counts,
                expected_width=result_abi.width,
            )
        except ExecutionError as error:
            raise ExecutionError(
                f"Q8145: native Reference execution returned malformed counts: {error}"
            ) from error
        return counts, None
    if kind != "value" or raw_counts:
        raise ExecutionError(
            "Q8145: native Reference execution disagrees with the value ABI"
        )
    if not isinstance(payload_value, bytes):
        raise ExecutionError("Q8145: native Reference value result is not bytes")
    try:
        return None, result_abi._decode_native_value(payload_value)
    except ExecutionError as error:
        raise ExecutionError(
            f"Q8145: native Reference value disagrees with its ABI: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class Reference:
    """Portable QCore target and source-independent reference endpoint."""

    @property
    def name(self) -> str:
        """Return the reference artifact format and endpoint identity."""

        return "qamomile.qcore.program"

    @property
    def target(self) -> Reference:
        """Return this target accepted by the reference endpoint."""

        return self

    @property
    def fingerprint(self) -> str:
        """Return the reference lowering and execution identity."""

        return "qamomile.reference.v7"

    @property
    def capabilities(self) -> CapabilitySnapshot:
        """Return the reference runtime's stable support contract."""

        return CapabilitySnapshot(
            self.name,
            frozenset(Requirement) - {Requirement.EXTERNAL_CALL},
        )

    def lower(self, program: Program) -> LoweredProgram:
        """Retain standard-prepared native execution state for the Reference endpoint.

        Args:
            program (Program): Verified specialized program to optimize and
                retain as the portable Reference artifact.

        Returns:
            LoweredProgram: Immutable optimized QCore payload with ABIs
            derived from the original verified program.

        Raises:
            ArtifactError: If the required native optimizer is unavailable,
                rejects the program, or returns a malformed payload.
        """

        optimize = getattr(native, "optimize_standard_program", None)
        if not callable(optimize):
            raise ArtifactError(
                "Q7049: Reference lowering requires standard QCore optimization"
            )
        try:
            optimized = optimize(program._native_program())
        except ValueError as error:
            raise ArtifactError(
                f"Q7049: standard QCore optimization failed: {error}"
            ) from error
        if not isinstance(optimized, bytes):
            raise ArtifactError(
                "Q7049: standard QCore optimization returned malformed bytes"
            )
        return LoweredProgram(
            payload=_ReferenceExecutable(optimized),
            parameter_abi=ParameterABI.from_program(program),
            result_abi=ResultABI.from_program(program),
            stages=(
                "qcore-restored",
                *_target_optimization_stages(program),
                "reference-artifact",
            ),
        )

    def launch(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Job:
        """Execute the retained verified program with fresh invocation state.

        Args:
            artifact (CompiledProgram): Compiled Reference program and verified ABI.
            invocation (Invocation): Bound launch values, shot count, and seed.

        Returns:
            Job: Completed native execution result.

        Raises:
            ExecutionError: If the artifact, invocation, or native execution is invalid.
        """

        self._validate_artifact(artifact)
        values = self._validate_invocation(artifact, invocation)
        counts, value = _native_reference_execution(
            artifact.execution_payload(_ReferenceExecutable),
            values,
            artifact.result_abi,
            shots=invocation.shots,
            seed=invocation.seed,
        )
        result = self._result(artifact, counts, value)
        return CompletedJob(self._job_id(artifact, invocation), result)

    def launch_batch(
        self,
        artifact: CompiledProgram,
        invocations: tuple[Invocation, ...],
    ) -> tuple[Job, ...]:
        """Interpret ordered invocations through one restored Program.

        Args:
            artifact (CompiledProgram): Sealed Reference artifact reused by
                every invocation.
            invocations (tuple[Invocation, ...]): ABI-normalized invocations
                with one shared shot count and consecutive seeds.

        Returns:
            tuple[Job, ...]: Completed local jobs in invocation order.

        Raises:
            ExecutionError: If the artifact or an invocation is inconsistent,
                seeds are not consecutive, shot counts differ, or native batch
                execution fails.
        """

        self._validate_artifact(artifact)
        if not invocations:
            return ()
        shots = invocations[0].shots
        base_seed = invocations[0].seed
        rows: list[Mapping[str, object]] = []
        for index, invocation in enumerate(invocations):
            if invocation.shots != shots or invocation.seed != base_seed + index:
                raise ExecutionError(
                    "Q8145: Reference batch requires one shot count and "
                    "consecutive seeds"
                )
            rows.append(self._validate_invocation(artifact, invocation))
        executions = _native_reference_batch_execution(
            artifact.execution_payload(_ReferenceExecutable),
            tuple(rows),
            artifact.result_abi,
            shots=shots,
            seed=base_seed,
        )
        return tuple(
            CompletedJob(
                self._job_id(artifact, invocation),
                self._result(artifact, counts, value),
            )
            for invocation, (counts, value) in zip(invocations, executions, strict=True)
        )

    def _validate_artifact(self, artifact: CompiledProgram) -> None:
        """Validate one Reference artifact against its restored payload.

        Args:
            artifact (CompiledProgram): Reference artifact to validate.

        Raises:
            ExecutionError: If the artifact format or sealed ABI disagrees with
                the restored program.
        """

        if artifact.format != self.name:
            raise ExecutionError(
                f"Q8108: Reference accepts {self.name!r}, got {artifact.format!r}"
            )
        executable = artifact.execution_payload(_ReferenceExecutable)
        payload_parameter_abi = executable.parameter_abi
        payload_result_abi = executable.result_abi
        if (
            artifact.parameter_abi != payload_parameter_abi
            or artifact.result_abi != payload_result_abi
        ):
            raise ExecutionError(
                "Q8136: reference artifact payload disagrees with its metadata"
            )

    def _validate_invocation(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> Mapping[str, object]:
        """Validate one normalized invocation against a Reference artifact.

        Args:
            artifact (CompiledProgram): Validated Reference artifact.
            invocation (Invocation): Runtime invocation to check.

        Returns:
            Mapping[str, object]: ABI-normalized parameter values.

        Raises:
            ExecutionError: If the invocation parameter names are not exactly
                those declared by the artifact ABI.
        """

        values = invocation.values()
        parameter_names = tuple(slot.name for slot in artifact.parameter_abi.slots)
        if values.keys() != set(parameter_names):
            raise ExecutionError(
                "Q8137: reference invocation is not normalized to its artifact ABI"
            )
        return values

    def _result(
        self,
        artifact: CompiledProgram,
        counts: dict[tuple[int, ...], int] | None,
        value: object,
    ) -> ExecutionResult:
        """Build one typed Reference result from a native projection.

        Args:
            artifact (CompiledProgram): CompiledProgram carrying the result ABI.
            counts (dict[tuple[int, ...], int] | None): Canonical sampled counts
                or ``None`` for a value result.
            value (object): Decoded value when ``counts`` is ``None``.

        Returns:
            ExecutionResult: Typed sampled or value result.
        """

        if counts is not None:
            result: ExecutionResult = SampleResult.from_counts(
                backend=self.name,
                counts=counts,
                expected_width=artifact.result_abi.width,
            )
        else:
            result = ValueResult(self.name, value)
        return result

    def _job_id(
        self,
        artifact: CompiledProgram,
        invocation: Invocation,
    ) -> str:
        """Derive one deterministic local Reference job identifier.

        Args:
            artifact (CompiledProgram): Executed sealed artifact.
            invocation (Invocation): Normalized invocation represented by the
                identifier.

        Returns:
            str: Endpoint-scoped deterministic job identifier.
        """

        return (
            "reference:"
            + hashlib.sha256(
                artifact.source_digest + repr(invocation).encode()
            ).hexdigest()[:20]
        )
