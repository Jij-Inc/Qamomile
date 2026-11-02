"""Immutable, native-validated device facts for one compilation target.

A target profile is an input to extension passes and to target identity. It is
never an IR element and no lowering stage reads it. Rust owns the schema, its
bounds, canonical row order, and the digest; this module keeps the canonical
bytes behind frozen facades. Absent facets are unknown, not empty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.protobuf.message import DecodeError

from ._native import compiler as native
from .errors import ArtifactError
from .proto import target_profile_pb2 as wire

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CouplingEdge:
    """One native two-qubit coupling.

    Attributes:
        control (int): First qubit of the directed pair.
        target (int): Second qubit of the directed pair.
        symmetric (bool): Whether the reverse direction is native too. A
            symmetric edge lists the smaller qubit first and the reverse pair
            never appears as its own edge.
    """

    control: int
    target: int
    symmetric: bool = False


@dataclass(frozen=True, slots=True)
class QubitProperty:
    """One measured per-qubit quantity, such as ``t1`` in seconds.

    Attributes:
        qubit (int): Qubit the value describes.
        name (str): Quantity name, unique per qubit.
        value (float): Finite measured value.
        unit (str): Unit label, empty when dimensionless.
    """

    qubit: int
    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True, slots=True)
class EdgeProperty:
    """One measured per-edge quantity, such as ``cx.error``.

    Attributes:
        control (int): First qubit of a declared coupling edge.
        target (int): Second qubit of that edge; either direction of a
            symmetric edge may be named.
        name (str): Quantity name, unique per directed pair.
        value (float): Finite measured value.
        unit (str): Unit label, empty when dimensionless.
    """

    control: int
    target: int
    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True, slots=True)
class TargetTopology:
    """Structural facet: qubit count, coupling graph, and native operations.

    Attributes:
        num_qubits (int): Positive physical qubit count.
        edges (tuple[CouplingEdge, ...]): Native couplings in canonical order.
        one_qubit_operations (tuple[str, ...]): Sorted native one-qubit names.
        two_qubit_operations (tuple[str, ...]): Sorted native two-qubit names.
    """

    num_qubits: int
    edges: tuple[CouplingEdge, ...] = ()
    one_qubit_operations: tuple[str, ...] = ()
    two_qubit_operations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TargetCalibration:
    """Measured facet: per-qubit and per-edge property rows.

    Attributes:
        qubit_properties (tuple[QubitProperty, ...]): Rows sorted by qubit and name.
        edge_properties (tuple[EdgeProperty, ...]): Rows sorted by pair and name.
        provenance (str): Free-form statement of where the values came from.
    """

    qubit_properties: tuple[QubitProperty, ...] = ()
    edge_properties: tuple[EdgeProperty, ...] = ()
    provenance: str = ""


@dataclass(frozen=True, slots=True)
class LogicalProfile:
    """Fault-tolerant cost facet for magic-state and cycle accounting.

    Attributes:
        code_family (str): Error-correcting code family name.
        code_distance (int): Positive code distance.
        cycle_time (float): Nonnegative logical cycle time.
        cycle_time_unit (str): Unit of ``cycle_time``.
        magic_state_cost (float): Nonnegative cost per T state.
        magic_state_cost_unit (str): Unit of ``magic_state_cost``.
        rotation_synthesis_model (str): Rotation-synthesis model name.
        rotation_synthesis_constants (tuple[tuple[str, float], ...]): Named
            finite model constants sorted by name.
    """

    code_family: str
    code_distance: int
    cycle_time: float = 0.0
    cycle_time_unit: str = ""
    magic_state_cost: float = 0.0
    magic_state_cost_unit: str = ""
    rotation_synthesis_model: str = ""
    rotation_synthesis_constants: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Immutable device facts validated and canonicalized by Rust.

    Construction encodes the facets, validates them natively, and re-projects
    the canonical row order onto this facade, so equal facts in any input order
    compare and digest equal. The profile is identity, not capability:
    ``CapabilitySnapshot.require`` ignores it and the target fingerprint folds
    in ``digest``.

    Attributes:
        format (str): Target format the profile describes.
        topology (TargetTopology): Required structural facet.
        calibration (TargetCalibration | None): Optional measured facet.
        logical (LogicalProfile | None): Optional fault-tolerant cost facet.
        backend_name (str): Optional backend identity.
        calibration_time (str): Optional operator-supplied timestamp, opaque.
        provenance (str): Optional statement of where the facts came from.

    Raises:
        TypeError: If a field does not have its declared type.
        ArtifactError: If native validation rejects the facts (``Q7030`` to
            ``Q7033``) or the native validator is unavailable.
    """

    format: str
    topology: TargetTopology
    calibration: TargetCalibration | None = None
    logical: LogicalProfile | None = None
    backend_name: str = ""
    calibration_time: str = ""
    provenance: str = ""
    _canonical: bytes = field(default=b"", init=False, repr=False, compare=False)
    _digest: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate through Rust and retain the canonical projection.

        Raises:
            TypeError: If a field does not have its declared type.
            ArtifactError: If native validation rejects the facts (``Q7030`` to
                ``Q7033``), is unavailable, or returns malformed data.
        """

        _native_validator()
        try:
            payload = _encode(self).SerializeToString(deterministic=True)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"target profile fields must have their declared types: {error}"
            ) from error
        canonical, digest = _validate_native(payload)
        for name, value in _project(wire.TargetProfileProto.FromString(canonical)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_canonical", canonical)
        object.__setattr__(self, "_digest", digest.hex())

    @property
    def digest(self) -> str:
        """Return the lowercase hexadecimal SHA-256 identity of the profile."""

        return self._digest

    def to_bytes(self) -> bytes:
        """Return the canonical portable encoding."""

        return self._canonical

    @classmethod
    def from_bytes(cls, payload: bytes) -> TargetProfile:
        """Natively validate one canonical profile before decoding its fields.

        Args:
            payload (bytes): Bytes produced by :meth:`to_bytes`.

        Returns:
            TargetProfile: Validated profile equal to the encoded one.

        Raises:
            ArtifactError: If the payload is malformed or noncanonical
                (``Q7030``), exceeds native bounds (``Q7031``), or fails native
                fact validation (``Q7032`` or ``Q7033``). Also raised if native
                validation is unavailable or returns malformed data.
        """

        canonical, _ = _validate_native(payload)
        if canonical != payload:
            raise ArtifactError("Q7030: target profile is not canonically encoded")
        try:
            decoded = wire.TargetProfileProto.FromString(canonical)
        except (DecodeError, MemoryError, RecursionError) as error:
            raise ArtifactError("Q7030: malformed target profile") from error
        profile = cls(**dict(_project(decoded)))
        if profile.to_bytes() != payload:
            raise ArtifactError("Q7030: target profile is not canonically encoded")
        return profile


def _native_validator() -> Callable[[bytes], object]:
    """Require native support for target-profile validation.

    Returns:
        Callable[[bytes], object]: Native validator for encoded profile bytes.

    Raises:
        ArtifactError: If native validation is unavailable (``Q7030``).
    """

    validate = getattr(native, "validate_target_profile", None)
    if not callable(validate):
        raise ArtifactError("Q7030: native target profile validation is unavailable")
    return validate


def _validate_native(payload: bytes) -> tuple[bytes, bytes]:
    """Validate encoded facts through Rust and check its returned data.

    Args:
        payload (bytes): Untrusted encoded target profile.

    Returns:
        tuple[bytes, bytes]: Canonical profile bytes and the 32-byte SHA-256 digest.

    Raises:
        ArtifactError: If native validation rejects the payload (``Q7030`` to
            ``Q7033``), is unavailable, or returns malformed data.
    """

    validate = _native_validator()
    try:
        result = validate(payload)
    except ValueError as error:
        raise ArtifactError(str(error)) from error
    if (
        type(result) is not tuple
        or len(result) != 2
        or not isinstance(result[0], bytes)
        or not isinstance(result[1], bytes)
        or len(result[1]) != 32
    ):
        raise ArtifactError(
            "Q7030: native target profile validation returned malformed data"
        )
    return result


def _encode(profile: TargetProfile) -> wire.TargetProfileProto:
    """Project one facade into its unvalidated wire message."""

    message = wire.TargetProfileProto(
        schema_version=_SCHEMA_VERSION,
        format=profile.format,
        backend_name=profile.backend_name,
        calibration_time=profile.calibration_time,
        provenance=profile.provenance,
    )
    topology = profile.topology
    if not isinstance(topology, TargetTopology):
        raise TypeError("topology must be a TargetTopology")
    message.topology.num_qubits = topology.num_qubits
    for edge in topology.edges:
        message.topology.edges.add(
            control=edge.control, target=edge.target, symmetric=edge.symmetric
        )
    message.topology.one_qubit_operations.extend(topology.one_qubit_operations)
    message.topology.two_qubit_operations.extend(topology.two_qubit_operations)
    calibration = profile.calibration
    if calibration is not None:
        if not isinstance(calibration, TargetCalibration):
            raise TypeError("calibration must be a TargetCalibration or None")
        message.calibration.SetInParent()
        message.calibration.provenance = calibration.provenance
        for row in calibration.qubit_properties:
            message.calibration.qubit_properties.add(
                qubit=row.qubit, name=row.name, value=row.value, unit=row.unit
            )
        for edge_row in calibration.edge_properties:
            message.calibration.edge_properties.add(
                control=edge_row.control,
                target=edge_row.target,
                name=edge_row.name,
                value=edge_row.value,
                unit=edge_row.unit,
            )
    logical = profile.logical
    if logical is not None:
        if not isinstance(logical, LogicalProfile):
            raise TypeError("logical must be a LogicalProfile or None")
        message.logical.SetInParent()
        message.logical.code_family = logical.code_family
        message.logical.code_distance = logical.code_distance
        message.logical.cycle_time = logical.cycle_time
        message.logical.cycle_time_unit = logical.cycle_time_unit
        message.logical.magic_state_cost = logical.magic_state_cost
        message.logical.magic_state_cost_unit = logical.magic_state_cost_unit
        message.logical.rotation_synthesis_model = logical.rotation_synthesis_model
        for name, value in logical.rotation_synthesis_constants:
            message.logical.rotation_synthesis_constants.add(name=name, value=value)
    return message


def _project(message: wire.TargetProfileProto) -> tuple[tuple[str, Any], ...]:
    """Project one canonical wire message onto facade constructor fields."""

    topology = TargetTopology(
        message.topology.num_qubits,
        tuple(
            CouplingEdge(edge.control, edge.target, edge.symmetric)
            for edge in message.topology.edges
        ),
        tuple(message.topology.one_qubit_operations),
        tuple(message.topology.two_qubit_operations),
    )
    calibration = None
    if message.HasField("calibration"):
        calibration = TargetCalibration(
            tuple(
                QubitProperty(row.qubit, row.name, row.value, row.unit)
                for row in message.calibration.qubit_properties
            ),
            tuple(
                EdgeProperty(row.control, row.target, row.name, row.value, row.unit)
                for row in message.calibration.edge_properties
            ),
            message.calibration.provenance,
        )
    logical = None
    if message.HasField("logical"):
        logical = LogicalProfile(
            message.logical.code_family,
            message.logical.code_distance,
            message.logical.cycle_time,
            message.logical.cycle_time_unit,
            message.logical.magic_state_cost,
            message.logical.magic_state_cost_unit,
            message.logical.rotation_synthesis_model,
            tuple(
                (constant.name, constant.value)
                for constant in message.logical.rotation_synthesis_constants
            ),
        )
    return (
        ("format", message.format),
        ("topology", topology),
        ("calibration", calibration),
        ("logical", logical),
        ("backend_name", message.backend_name),
        ("calibration_time", message.calibration_time),
        ("provenance", message.provenance),
    )
