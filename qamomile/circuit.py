"""Logical circuit-family projection for circuit-shaped QCore programs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias, cast

from google.protobuf.message import DecodeError

from ._compiler_optimization import _preserve_preparation
from ._native import compiler as native
from .compilation import (
    CircuitTarget,
    LoweredProgram,
    ParameterABI,
    RealizationCapability,
    ResultABI,
    _target_optimization_stages,
)
from .errors import ArtifactError, CapabilityError, ExecutionError, SerializationError
from .ir import (
    FrozenMapping,
    PortableValue,
    Program,
    thaw_value,
)
from .lowering import RealizationOptions
from .observable import Hamiltonian
from .types import (
    Bool,
    Int,
    Real,
    ScalarType,
    strip_stage,
)


@dataclass(frozen=True, slots=True)
class ParameterValue:
    """A launch parameter retained symbolically in a circuit family."""

    name: str
    path: tuple[int | str, ...] = ()


@dataclass(frozen=True, slots=True)
class BooleanPlan:
    """An immutable projection of the compiler's verified Boolean ANF.

    Args:
        input_width (int): Ordered source register width.
        outputs (tuple[tuple[tuple[int, ...], ...], ...]): Output polynomials;
            each polynomial XORs its conjunctions of source bit indices.
    """

    input_width: int
    outputs: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class DeferredDefinition:
    """Keep a checked native definition alive until an SDK needs its body.

    Args:
        identity (int): Process-local identity owned by the native handle.
        owner (Any): Immutable native owner; never part of portable encoding.
    """

    identity: int
    owner: Any = field(compare=False, repr=False)

    @property
    def scalars(self) -> tuple[CircuitValue, ...]:
        """Return the compact scalar bindings without materializing gates.

        Returns:
            tuple[CircuitValue, ...]: Checked scalar arena of this definition.
        """
        return _native_scalar_values(self.owner)

    def materialize(self) -> CircuitBlock:
        """Request a bounded gate body from the existing Rust implementation.

        Returns:
            CircuitBlock: Immutable local-wire definition.
        """
        scalars = self.scalars
        return CircuitBlock(
            tuple(
                _native_instruction(row, scalars, {})
                for row in self.owner.materialize()
            )
        )


@dataclass(frozen=True, slots=True)
class LoopIndex:
    """A scoped retained-loop index, independent of the launch ABI.

    Args:
        index (int): Compiler-assigned scope identity.
    """

    index: int


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    """One scalar leaf of a fixed-shape launch parameter."""

    value: ParameterValue
    type: ScalarType


@dataclass(frozen=True, slots=True)
class ClassicalExpr:
    """A target-neutral symbolic classical scalar expression."""

    opcode: str
    operands: tuple[CircuitValue, ...]
    attributes: FrozenMapping = FrozenMapping(())


@dataclass(frozen=True, slots=True)
class QuantumWires:
    """Logical wires realizing one owned QCore region version."""

    region: int
    wires: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MeasurementBit:
    """One source-ordered classical bit produced inside a circuit."""

    index: int


@dataclass(frozen=True, slots=True)
class CallValue:
    """One typed scalar returned by a structured runtime call."""

    index: int
    type: ScalarType


@dataclass(frozen=True, slots=True)
class MeasurementTuple:
    """A complete source-ordered terminal measurement record."""

    bits: tuple[MeasurementBit, ...]


@dataclass(frozen=True, slots=True)
class CircuitRecordValue:
    """One named result aggregate in semantic declaration order."""

    fields: tuple[tuple[str, "CircuitValue"], ...]


CircuitValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | bytes
    | FrozenMapping
    | tuple["CircuitValue", ...]
    | ParameterValue
    | LoopIndex
    | BooleanPlan
    | DeferredDefinition
    | ClassicalExpr
    | QuantumWires
    | MeasurementBit
    | CallValue
    | MeasurementTuple
    | CircuitRecordValue
    | Hamiltonian
)


@dataclass(frozen=True, slots=True)
class CircuitInstruction:
    """One logical circuit operation with optional structured regions."""

    opcode: str
    qubits: tuple[int, ...] = ()
    parameters: tuple[CircuitValue, ...] = ()
    condition: CircuitValue | None = None
    regions: tuple[CircuitBlock, ...] = ()
    controls: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CircuitBlock:
    """A structured sequence of logical circuit operations."""

    instructions: tuple[CircuitInstruction, ...]
    has_return: bool = False
    return_value: CircuitValue = None


@dataclass(frozen=True, slots=True)
class CircuitProgram:
    """Retain a closed circuit projection and its native-derived ABI.

    Args:
        name (str): Entrypoint display name.
        semantic_id (str): Original QCore semantic digest.
        width (int): Logical wire count.
        measurement_count (int): Private and returned measurement width.
        body (CircuitBlock): Structured target-neutral instructions.
        parameter_abi (ParameterABI): Native-derived runtime parameter layout.
        result_abi (ResultABI): Native-derived result layout.
        result_value (CircuitValue): Retained symbolic result expression.
        output_wires (tuple[int, ...] | None): Source-order wires of a quantum
            component result; absent for classical execution results.
        realizations (tuple[str, ...]): Selected versioned implementation IDs.
        optimization_stages (tuple[str, ...]): Late native passes actually run.
        _native_result_route (str | None): Checked native result route.
        _native_result_postprocessing (bytes | None): Sealed native result plan.
        _native_operation_count (int | None): Verified instruction count.
    """

    name: str
    semantic_id: str
    width: int
    measurement_count: int
    body: CircuitBlock
    parameter_abi: ParameterABI
    result_abi: ResultABI
    result_value: CircuitValue
    realizations: tuple[str, ...] = ()
    optimization_stages: tuple[str, ...] = ()
    output_wires: tuple[int, ...] | None = None
    _native_result_route: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _native_result_postprocessing: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _native_operation_count: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def realization_stages(self) -> tuple[str, ...]:
        """Project compiler-selected identities into the compilation report.

        Returns:
            tuple[str, ...]: Stable versioned realization stage labels.
        """

        return (
            *self.optimization_stages,
            *(f"realization:{identity}" for identity in self.realizations),
        )

    @property
    def operation_count(self) -> int:
        """Return the compiler-derived logical instruction count."""

        if self._native_operation_count is None:
            raise CapabilityError(
                "Q7248: circuit operation count requires a native projection"
            )
        return self._native_operation_count

    @property
    def requires_result_postprocessing(self) -> bool:
        """Return whether a backend must evaluate the retained result tree."""

        if self._native_result_route in {"value", "direct_measurements"}:
            return False
        if self._native_result_route in {"postprocessing", "runtime_local"}:
            return True
        raise CapabilityError(
            "Q7249: circuit result route requires a native projection"
        )

    def to_bytes(self) -> bytes:
        """Encode the target-neutral circuit projection canonically."""

        data = {
            "schema": 2,
            "width": self.width,
            "measurement_count": self.measurement_count,
            "body": _block_to_data(self.body),
            "result": _value_to_data(self.result_value),
        }
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")


RESULT_POSTPROCESSING_METADATA = "circuit_result_postprocessing"
RESULT_POSTPROCESSING_FORMAT = "qamomile.result-postprocessing.protobuf.v1"


def circuit_result_postprocessing(program: CircuitProgram) -> bytes | None:
    """Encode the retained result evaluator required by a static backend."""

    if not program.requires_result_postprocessing:
        return None
    if program._native_result_postprocessing is None:
        raise CapabilityError(
            "Q7247: native result plan cannot represent a runtime-local result"
        )
    return program._native_result_postprocessing


def postprocess_circuit_counts(
    payload: bytes,
    counts: Mapping[tuple[int, ...], int],
    parameters: Mapping[str, object],
    result_abi: ResultABI,
) -> dict[tuple[int, ...], int]:
    """Evaluate a retained result tree over backend measurement counts."""

    rows: list[tuple[bytes, int]] = []
    try:
        for raw_bits, count in counts.items():
            bits = tuple(raw_bits)
            if (
                type(count) is not int
                or count < 0
                or len(bits) > 1_000_000
                or any(type(bit) is not int or bit not in {0, 1} for bit in bits)
            ):
                raise TypeError
            rows.append((bytes(bits), count))
    except (OverflowError, TypeError, ValueError) as error:
        raise ExecutionError("Q7247: invalid circuit result postprocessing") from error
    evaluator = getattr(native, "postprocess_circuit_counts", None)
    if not callable(evaluator):
        raise ExecutionError("Q7247: native result postprocessing is unavailable")
    try:
        result = evaluator(
            payload,
            rows,
            _native_launch_value(parameters),
            result_abi._to_bytes(),
        )
        logical: dict[tuple[int, ...], int] = {}
        for raw_bits, count in result:
            bits = tuple(bytes(raw_bits))
            if (
                len(bits) != result_abi.width
                or any(bit not in {0, 1} for bit in bits)
                or type(count) is not int
                or count < 0
                or bits in logical
            ):
                raise TypeError
            logical[bits] = count
        return logical
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ExecutionError(
            f"Q7247: invalid circuit result postprocessing: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class CircuitDispatchParameter:
    """One finite entry parameter selected before circuit execution."""

    parameter_index: int
    name: str


@dataclass(frozen=True, slots=True)
class CircuitDispatchPredicate:
    """One Rust-verified Boolean expression evaluated before execution."""

    expression: CircuitValue


@dataclass(frozen=True, slots=True)
class CircuitDispatchDecision:
    """One predicate truth assignment selecting a static circuit variant."""

    predicate_index: int
    value: bool


@dataclass(frozen=True, slots=True)
class CircuitVariant:
    """One static circuit and its exact launch-time dispatch key."""

    circuit: CircuitProgram
    decisions: tuple[CircuitDispatchDecision, ...]


@dataclass(frozen=True, slots=True)
class CircuitFamilyProgram:
    """Finite static circuits implementing launch-time scalar control."""

    name: str
    semantic_id: str
    dispatch_parameters: tuple[CircuitDispatchParameter, ...]
    variants: tuple[CircuitVariant, ...]
    parameter_abi: ParameterABI
    result_abi: ResultABI
    dispatch_predicates: tuple[CircuitDispatchPredicate, ...]
    _native_selection_plan: bytes | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def select(self, parameters: Mapping[str, object]) -> CircuitProgram:
        """Select one variant from launch parameter values.

        Args:
            parameters: Launch parameter mapping. Non-dispatch parameters are
                ignored and remain in the selected circuit's parameter ABI.

        Returns:
            CircuitProgram: Exact static circuit selected by the dispatch key.

        Raises:
            CapabilityError: If a dispatch value is missing, violates its
                declared type, or does not select one closed family variant.
        """

        selected = _bind_dispatch_parameters(
            self.parameter_abi,
            tuple(parameter.name for parameter in self.dispatch_parameters),
            parameters,
        )
        if self._native_selection_plan is None:
            raise CapabilityError("Q7224: circuit family has no native selection plan")
        select = getattr(native, "select_circuit_family_variant", None)
        if not callable(select):
            raise CapabilityError(
                "Q7224: native circuit-family selection is unavailable"
            )
        try:
            index = select(self._native_selection_plan, selected)
        except (TypeError, ValueError) as error:
            raise CapabilityError(
                f"Q7224: native circuit-family selection failed: {error}"
            ) from error
        if type(index) is not int or not 0 <= index < len(self.variants):
            raise CapabilityError(
                "Q7224: native circuit-family selector returned an invalid variant"
            )
        return self.variants[index].circuit


def lower_to_circuit_family(
    program: Program,
    *,
    realization_capabilities: frozenset[RealizationCapability] = frozenset(),
    realizations: RealizationOptions = RealizationOptions(),
) -> CircuitFamilyProgram:
    """Lower finite launch-time scalar control into static circuit variants.

    Args:
        program (Program): Verified closed program to lower.
        realizations (RealizationOptions): Operation implementations.
        realization_capabilities (frozenset[RealizationCapability]): Composite instructions accepted by
            the target-neutral consumer.

    Returns:
        CircuitFamilyProgram: Rust-derived dispatch keys and static circuits.

    Raises:
        CapabilityError: If the native compiler is unavailable, control is not
            a finite scalar-predicate tree, the family exceeds its work budget,
            or a selected variant is not circuit-shaped.
        TypeError: If an installed native extension violates its bridge ABI.
    """

    inspect = getattr(native, "inspect_static_circuit_family", None)
    if inspect is None:
        raise CapabilityError(
            "Q7220: circuit-family lowering requires the native compiler"
        )
    try:
        realized_payload = (
            _preserve_preparation(
                program,
                Program.from_bytes(
                    realizations._apply(program.to_bytes(), boundary="program")
                ),
            )._native_program()
            if realizations._custom
            else program._native_program()
        )
        projection = inspect(
            realized_payload,
            _realization_labels(realization_capabilities),
            realizations.overrides,
        )
    except ValueError as error:
        raise CapabilityError(str(error)) from error
    semantic_id = program.semantic_digest
    parameter_abi = ParameterABI.from_program(program)
    result_abi = ResultABI.from_program(program)
    try:
        dispatch_parameters = tuple(
            CircuitDispatchParameter(parameter.parameter_index, parameter.name)
            for parameter in projection.dispatch_parameters
        )
        dispatch_names = {parameter.name for parameter in dispatch_parameters}
        if len(dispatch_names) != len(dispatch_parameters):
            raise TypeError("native dispatch parameters are not unique")
        for parameter in dispatch_parameters:
            if (
                type(parameter.parameter_index) is not int
                or parameter.parameter_index < 0
                or parameter.parameter_index >= len(parameter_abi.slots)
                or parameter_abi.slots[parameter.parameter_index].name != parameter.name
            ):
                raise TypeError("native dispatch parameter violates the source ABI")
        dispatch_predicates_list: list[CircuitDispatchPredicate] = []
        predicate_names: set[str] = set()
        for predicate in projection.predicates:
            raw_parameter_names = predicate.parameter_names
            if (
                not isinstance(raw_parameter_names, list)
                or any(
                    type(name) is not str or not name for name in raw_parameter_names
                )
                or len(raw_parameter_names) != len(set(raw_parameter_names))
            ):
                raise TypeError("native dispatch predicate has invalid parameter names")
            predicate_names.update(raw_parameter_names)
            dispatch_predicates_list.append(_native_dispatch_predicate(predicate))
        dispatch_predicates = tuple(dispatch_predicates_list)
        variants_list: list[CircuitVariant] = []
        for variant in projection.variants:
            circuit = _circuit_from_native_projection(
                program,
                variant.circuit,
                semantic_id=semantic_id,
                parameter_abi=parameter_abi,
                conditionally_eliminated_parameters=frozenset(dispatch_names),
            )
            variants_list.append(
                CircuitVariant(
                    circuit,
                    tuple(
                        CircuitDispatchDecision(
                            decision.predicate_index,
                            decision.value,
                        )
                        for decision in variant.decisions
                    ),
                )
            )
        variants = tuple(variants_list)
        if predicate_names != dispatch_names:
            raise TypeError("native dispatch parameters disagree with predicate leaves")
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError(
            "native circuit-family projection violates the Python ABI"
        ) from error
    return CircuitFamilyProgram(
        name=program.entrypoint.rsplit(".", 1)[-1],
        semantic_id=semantic_id,
        dispatch_parameters=dispatch_parameters,
        variants=variants,
        parameter_abi=parameter_abi,
        result_abi=result_abi,
        dispatch_predicates=dispatch_predicates,
        _native_selection_plan=bytes(projection.selection_plan),
    )


def lower_to_circuit_at(
    program: Program,
    parameters: Mapping[str, object],
    *,
    realization_capabilities: frozenset[RealizationCapability] = frozenset(),
    realizations: RealizationOptions = RealizationOptions(),
) -> CircuitProgram:
    """Materialize launch-controlled topology for one exact invocation.

    Args:
        program (Program): Verified closed program to lower.
        parameters (Mapping[str, object]): Values for each parameter reaching host
            control. Unrelated runtime parameters are ignored.
        realizations (RealizationOptions): Operation implementations.
        realization_capabilities (frozenset[RealizationCapability]): Composite instructions accepted by
            the target-neutral consumer.

    Returns:
        CircuitProgram: Target-neutral circuit for the selected invocation.

    Raises:
        CapabilityError: If the native compiler is unavailable, a control
            value is missing or invalid, or the invocation exceeds a fixed
            circuit-materialization budget.
        TypeError: If an installed native extension violates its bridge ABI.
    """

    inspect = getattr(native, "inspect_static_circuit_at", None)
    inspect_program = getattr(native, "inspect_program", None)
    if inspect is None or inspect_program is None:
        raise CapabilityError(
            "Q7220: launch circuit materialization requires the native compiler"
        )
    parameter_abi = ParameterABI.from_program(program)
    selected = launch_control_bindings(program, parameters)
    try:
        realized_payload = (
            _preserve_preparation(
                program,
                Program.from_bytes(
                    realizations._apply(program.to_bytes(), boundary="program")
                ),
            )._native_program()
            if realizations._custom
            else program._native_program()
        )
        projection = inspect(
            realized_payload,
            {name: _native_launch_value(value) for name, value in selected.items()},
            _realization_labels(realization_capabilities),
            realizations.overrides,
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise CapabilityError(str(error)) from error
    circuit = _circuit_from_native_projection(
        program,
        projection,
        semantic_id=program.semantic_digest,
        parameter_abi=parameter_abi,
        conditionally_eliminated_parameters=frozenset(selected),
    )
    return circuit


def launch_control_bindings(
    program: Program, parameters: Mapping[str, object]
) -> dict[str, object]:
    """Select validated launch values consumed by native circuit materialization.

    Args:
        program (Program): Verified source owning control-dependence facts.
        parameters (Mapping[str, object]): Complete invocation values.

    Returns:
        dict[str, object]: Source-ordered values affecting circuit topology.

    Raises:
        CapabilityError: If native inspection or required bindings are unavailable.
        TypeError: If the native control projection is malformed.
    """
    inspect = getattr(native, "inspect_program", None)
    if not callable(inspect):
        raise CapabilityError(
            "Q7220: launch circuit materialization requires the native compiler"
        )
    try:
        header = inspect(program._native_program())
        names = tuple(
            program.entry.parameters[index].name
            for index in header.entry_control_parameters
        )
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("native control ABI violates the Python ABI") from error
    return _bind_dispatch_parameters(
        ParameterABI.from_program(program), names, parameters
    )


def _bind_dispatch_parameters(
    parameter_abi: ParameterABI,
    names: tuple[str, ...],
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Bind only host-control parameters through the native artifact ABI.

    Args:
        parameter_abi: Complete Rust-derived entrypoint parameter ABI.
        names: Source-ordered parameters proven to reach host control.
        parameters: User-supplied launch values; unrelated entries are ignored.

    Returns:
        Native-validated dispatch values with defaults materialized.

    Raises:
        CapabilityError: If a required dispatch value is absent, invalid, or
            cannot be checked by the native compiler.
        TypeError: If the native control ABI disagrees with the parameter ABI.
    """

    slots = {slot.name: slot for slot in parameter_abi.slots}
    try:
        dispatch_slots = tuple(slots[name] for name in names)
    except KeyError as error:
        raise TypeError(
            "native control ABI disagrees with the parameter ABI"
        ) from error
    supplied = {name: parameters[name] for name in names if name in parameters}
    try:
        return dict(ParameterABI(dispatch_slots).bind(supplied))
    except ExecutionError as error:
        code = error.diagnostic.code
        message = error.diagnostic.message
        if code == "Q8102":
            missing = next(
                (
                    slot.name
                    for slot in dispatch_slots
                    if slot.name not in supplied and not slot.has_default
                ),
                None,
            )
            if missing is not None:
                raise CapabilityError(
                    f"Q7221: missing host-dispatch parameter {missing!r}"
                ) from error
        if code == "Q8109" or (
            code == "Q8135"
            and message.startswith("invocation parameters are not portable:")
        ):
            rejected = next(
                (
                    slot
                    for slot in dispatch_slots
                    if slot.name in supplied
                    and (repr(slot.name) in message or f'"{slot.name}"' in message)
                ),
                next((slot for slot in dispatch_slots if slot.name in supplied), None),
            )
            if rejected is not None:
                expectation = (
                    "must be Bool"
                    if strip_stage(rejected.type) == Bool
                    else f"expects {rejected.type}"
                )
                raise CapabilityError(
                    f"Q7222: host-dispatch parameter {rejected.name!r} {expectation}"
                ) from error
        raise CapabilityError(error.diagnostic) from error


def validate_launch_parameters(
    program: Program,
    parameters: Mapping[str, object],
) -> None:
    """Validate a complete invocation through the native entrypoint ABI.

    Args:
        program: Verified closed program whose launch ABI is authoritative.
        parameters: Exact values for every unresolved entry ``Param`` slot.

    Raises:
        CapabilityError: If the native compiler is unavailable or rejects a
            name, scalar type, record field, or fixed aggregate shape.
        ExecutionError: If the public parameter ABI is incomplete or contains
            an unknown value.
    """

    bound = dict(ParameterABI.from_program(program).bind(parameters))
    validate = getattr(native, "validate_launch_parameters", None)
    if not callable(validate):
        raise CapabilityError(
            "Q7223: launch parameter validation requires the native compiler"
        )
    try:
        validate(
            program._native_program(),
            {name: _native_launch_value(value) for name, value in bound.items()},
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise CapabilityError(str(error)) from error


def _native_launch_value(value: object) -> object:
    """Normalize one validated launch value for the native bridge.

    Args:
        value: Scalar or fixed aggregate accepted by the public ABI.

    Returns:
        object: Equivalent built-in scalar, tuple, or dictionary tree.
    """

    if isinstance(value, Mapping):
        return {
            name: _native_launch_value(component) for name, component in value.items()
        }
    if isinstance(value, (tuple, list)):
        return tuple(_native_launch_value(component) for component in value)
    return value


def lower_to_circuit(
    program: Program,
    *,
    realization_capabilities: frozenset[RealizationCapability] = frozenset(),
    realizations: RealizationOptions = RealizationOptions(),
) -> CircuitProgram:
    """Project a restored QCore program into a logical circuit family.

    Args:
        program (Program): Verified closed program to lower.
        realizations (RealizationOptions): Operation implementations.
        realization_capabilities (frozenset[RealizationCapability]): Composite instructions accepted by
            the target-neutral consumer.

    Returns:
        CircuitProgram: Target-neutral logical circuit family.

    Raises:
        CapabilityError: If Rust cannot represent the program as a circuit.
    """

    semantic_id = program.semantic_digest
    return _native_circuit(
        program,
        semantic_id=semantic_id,
        realization_capabilities=realization_capabilities,
        realizations=realizations,
    )


def lower_circuit_target(
    program: Program,
    target: CircuitTarget,
) -> LoweredProgram:
    """Project verified QCore once and emit one direct circuit artifact.

    Args:
        program (Program): Closed verified source program.
        target (CircuitTarget): Format emitter with exact capabilities.

    Returns:
        LoweredProgram: Target payload with the native-derived ABIs and stages.

    Raises:
        CapabilityError: If native circuit projection cannot represent the program.
        ArtifactError: If emission returns an invalid artifact or changes an ABI.
    """

    capabilities = target.capabilities
    logical = lower_to_circuit(
        program,
        realization_capabilities=capabilities.realization_capabilities,
        realizations=capabilities.realizations,
    )
    lowered = target.lower_circuit(logical)
    if not isinstance(lowered, LoweredProgram):
        raise ArtifactError(
            "Q7042: target lower_circuit() did not return LoweredProgram"
        )
    if (
        lowered.parameter_abi != logical.parameter_abi
        or lowered.result_abi != logical.result_abi
    ):
        raise ArtifactError("Q7044: target changed a direct circuit ABI")
    return replace(
        lowered,
        stages=(
            "qcore-restored",
            *_target_optimization_stages(program),
            "circuit-projection",
            *lowered.stages,
        ),
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
    )


def _native_circuit(
    program: Program,
    *,
    semantic_id: str,
    realization_capabilities: frozenset[RealizationCapability],
    realizations: RealizationOptions,
) -> CircuitProgram:
    """Adopt a successful native circuit projection.

    Args:
        program (Program): Verified program after shared semantic
            normalization passes.
        semantic_id (str): Digest of the pre-normalized source program.
        realizations (RealizationOptions): Operation implementations.
        realization_capabilities (frozenset[RealizationCapability]): Composite instructions accepted by
            the target-neutral consumer.

    Returns:
        CircuitProgram: Native target-neutral circuit projection.

    Raises:
        CapabilityError: If an installed native compiler lacks the projection
            boundary or rejects the verified program.
        TypeError: If an installed native extension violates its bridge ABI.
    """

    inspect = getattr(native, "inspect_static_circuit", None)
    if not callable(inspect):
        raise CapabilityError("Q7220: circuit lowering requires the native compiler")
    try:
        realized_payload = (
            _preserve_preparation(
                program,
                Program.from_bytes(
                    realizations._apply(program.to_bytes(), boundary="program")
                ),
            )._native_program()
            if realizations._custom
            else program._native_program()
        )
        projection = inspect(
            realized_payload,
            _realization_labels(realization_capabilities),
            realizations.overrides,
        )
    except ValueError as error:
        raise CapabilityError(str(error)) from error
    return _circuit_from_native_projection(
        program,
        projection,
        semantic_id=semantic_id,
        parameter_abi=ParameterABI.from_program(program),
    )


def _realization_labels(
    capabilities: frozenset[RealizationCapability],
) -> list[str]:
    """Project one immutable closed realization set to the native vocabulary."""

    capabilities = frozenset(capabilities)
    if any(not isinstance(item, RealizationCapability) for item in capabilities):
        raise TypeError(
            "realization_capabilities must contain RealizationCapability values"
        )
    return sorted(item.value for item in capabilities)


def _circuit_from_native_projection(
    program: Program,
    projection: Any,
    *,
    semantic_id: str,
    parameter_abi: ParameterABI,
    conditionally_eliminated_parameters: frozenset[str] = frozenset(),
    result_abi: ResultABI | None = None,
) -> CircuitProgram:
    """Restore one public circuit from an independently verified projection.

    Args:
        program: Verified source program supplying logical ABI information.
        projection: Native bridge circuit projection.
        semantic_id: Digest of the unspecialized semantic program.
        parameter_abi: Complete launch parameter ABI before circuit projection.
        conditionally_eliminated_parameters: Parameters consumed by static
            control unless the projected circuit still references them.
        result_abi (ResultABI | None): Native-checked raw child transport layout,
            or None to use the source result layout.

    Returns:
        CircuitProgram: Public target-neutral circuit value.

    Raises:
        TypeError: If the native projection violates its bridge ABI.
    """

    try:
        scalars = _native_scalar_values(projection)
        call_values: dict[int, CallValue] = {}
        for value in scalars:
            if isinstance(value, CallValue):
                if value.index in call_values:
                    raise TypeError("native circuit has duplicate call value slots")
                call_values[value.index] = value
        instructions = tuple(
            _native_instruction(instruction, scalars, call_values)
            for instruction in projection.instructions
        )
        operation_count = projection.operation_count
        if type(operation_count) is not int or operation_count < 0:
            raise TypeError("native circuit has an invalid operation count")
        raw_parameter_names = projection.parameter_names
        if not isinstance(raw_parameter_names, list) or any(
            type(name) is not str for name in raw_parameter_names
        ):
            raise TypeError("native circuit has invalid retained parameter names")
        retained_parameter_names = frozenset(raw_parameter_names)
        if len(retained_parameter_names) != len(raw_parameter_names):
            raise TypeError("native circuit has duplicate retained parameter names")
        known_parameter_names = frozenset(slot.name for slot in parameter_abi.slots)
        if not retained_parameter_names <= known_parameter_names:
            raise TypeError("native circuit retained an unknown parameter")
        projected_parameter_abi = ParameterABI(
            tuple(
                slot
                for slot in parameter_abi.slots
                if slot.name not in conditionally_eliminated_parameters
                or slot.name in retained_parameter_names
            )
        )
        declared_result = result_abi or ResultABI.from_program(program)
        projected_result = projection.result_value
        if projected_result is None and projection.result_measurements:
            result_value: CircuitValue = MeasurementTuple(
                tuple(MeasurementBit(index) for index in projection.result_measurements)
            )
        elif projected_result is None:
            result_value = None
        else:
            result_value = _native_result_value(projected_result, scalars)
        result_route = projection.result_route
        if result_route not in {
            "value",
            "direct_measurements",
            "postprocessing",
            "runtime_local",
        }:
            raise TypeError("native circuit has an invalid result route")
        result_postprocessing = (
            bytes(projection.result_postprocessing)
            if projection.result_postprocessing is not None
            else None
        )
        if declared_result.width:
            if result_value is None:
                raise TypeError("native circuit omitted a sampled result value")
            result_measurements = tuple(projection.result_measurements)
            if result_route == "direct_measurements":
                if (
                    len(result_measurements) != declared_result.width
                    or result_postprocessing is not None
                ):
                    raise TypeError("native direct result violates the Python ABI")
                result_abi = ResultABI(
                    declared_result.type,
                    result_measurements,
                    projection.measurement_count,
                )
            elif result_route == "postprocessing":
                if result_measurements or result_postprocessing is None:
                    raise TypeError(
                        "native postprocessed result violates the Python ABI"
                    )
                result_abi = declared_result
            elif result_route == "runtime_local":
                if result_measurements or result_postprocessing is not None:
                    raise TypeError(
                        "native runtime-local result violates the Python ABI"
                    )
                result_abi = declared_result
            else:
                raise TypeError("native sampled result uses the value route")
        else:
            if (
                result_route != "value"
                or projection.result_measurements
                or result_postprocessing is not None
            ):
                raise TypeError("native value result violates the Python ABI")
            result_abi = declared_result
        realizations = tuple(projection.realizations)
        if any(not isinstance(item, str) for item in realizations) or len(
            set(realizations)
        ) != len(realizations):
            raise TypeError("native realization trace violates the Python ABI")
        return CircuitProgram(
            name=program.entrypoint.rsplit(".", 1)[-1],
            semantic_id=semantic_id,
            width=projection.width,
            measurement_count=projection.measurement_count,
            body=CircuitBlock(instructions),
            parameter_abi=projected_parameter_abi,
            result_abi=result_abi,
            result_value=result_value,
            output_wires=(
                tuple(projection.output_wires)
                if projection.output_wires is not None
                else None
            ),
            _native_result_route=result_route,
            _native_result_postprocessing=result_postprocessing,
            _native_operation_count=operation_count,
            realizations=realizations,
            optimization_stages=tuple(projection.optimization_stages),
        )
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError("native circuit projection violates the Python ABI") from error


def _native_result_value(
    value: Any,
    scalars: tuple[CircuitValue, ...],
) -> CircuitValue:
    """Restore one recursive result tree from the native bridge.

    Args:
        value: Native tagged result node.
        scalars: Restored scalar arena indexed by native scalar ID.

    Returns:
        CircuitValue: Public circuit result expression.

    Raises:
        TypeError: If the native result node violates the bridge ABI.
    """

    kind = value.kind
    if kind == "unit":
        return None
    if kind == "literal":
        try:
            from .proto import qcore_pb2 as wire
            from .serialization import decode_literal

            literal = wire.LiteralProto.FromString(bytes(value.literal))
            return _to_circuit_value(decode_literal(literal))
        except (DecodeError, SerializationError, TypeError, ValueError) as error:
            raise TypeError("native circuit result literal is invalid") from error
    if kind == "scalar":
        if value.scalar is None:
            raise TypeError("native circuit result scalar is missing its root")
        return scalars[value.scalar]
    elements = tuple(_native_result_value(item, scalars) for item in value.elements)
    if kind == "aggregate":
        return elements
    if kind == "record":
        names = tuple(value.field_names)
        if len(names) != len(elements) or len(names) != len(set(names)):
            raise TypeError("native circuit result record has invalid fields")
        return CircuitRecordValue(tuple(zip(names, elements, strict=True)))
    raise TypeError(f"native circuit result has unknown kind {kind!r}")


def _native_instruction(
    instruction: Any,
    scalars: tuple[CircuitValue, ...],
    call_values: Mapping[int, CallValue],
) -> CircuitInstruction:
    """Restore one possibly structured native circuit instruction.

    Args:
        instruction (Any): Native bridge instruction with tagged parameters,
            an optional scalar condition, and zero or more nested blocks.
        scalars (tuple[CircuitValue, ...]): Restored scalar arena indexed by
            native scalar ID.
        call_values (Mapping[int, CallValue]): Call-result scalars indexed by their stable storage slot.

    Returns:
        CircuitInstruction: Public recursive circuit instruction.

    Raises:
        TypeError: If a condition index or nested block violates the bridge ABI.
    """

    raw_condition = instruction.condition
    if raw_condition is None:
        condition: CircuitValue | None = None
    else:
        if (
            isinstance(raw_condition, bool)
            or not isinstance(raw_condition, int)
            or raw_condition < 0
        ):
            raise TypeError("native circuit condition has an invalid scalar index")
        try:
            condition = scalars[raw_condition]
        except IndexError as error:
            raise TypeError("native circuit condition scalar is unavailable") from error
    regions_list = []
    for region in instruction.regions:
        has_return = region.has_return
        if not isinstance(has_return, bool):
            raise TypeError("native circuit block return flag is invalid")
        return_value = (
            _native_result_value(region.return_value, scalars)
            if has_return and region.return_value is not None
            else None
        )
        if not has_return and region.return_value is not None:
            raise TypeError("native circuit block return value is inconsistent")
        regions_list.append(
            CircuitBlock(
                tuple(
                    _native_instruction(nested, scalars, call_values)
                    for nested in region.instructions
                ),
                has_return=has_return,
                return_value=return_value,
            )
        )
    regions = tuple(regions_list)
    return CircuitInstruction(
        instruction.opcode,
        tuple(instruction.qubits),
        tuple(
            _native_parameter_value(parameter, scalars, call_values)
            for parameter in instruction.parameters
        ),
        condition=condition,
        regions=regions,
        controls=tuple(instruction.controls),
    )


def _native_parameter_value(
    parameter: Any,
    scalars: tuple[CircuitValue, ...],
    call_values: Mapping[int, CallValue],
) -> CircuitValue:
    """Restore one typed native instruction parameter.

    Args:
        parameter (Any): Native tagged instruction parameter.
        scalars (tuple[CircuitValue, ...]): Restored scalar arena indexed by
            native scalar ID.
        call_values (Mapping[int, CallValue]): Call-result scalars indexed by their stable storage slot.

    Returns:
        CircuitValue: Measurement destination or scalar circuit value.

    Raises:
        TypeError: If the tag or index violates the native bridge ABI.
    """

    kind = getattr(parameter, "kind", None)
    index = getattr(parameter, "index", None)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError("native circuit parameter has an invalid index")
    if kind == "definition":
        owner = parameter.definition
        if owner is None:
            raise TypeError("native definition owner is missing")
        return DeferredDefinition(owner.identity, owner)
    if kind == "boolean_plan":
        outputs = tuple(
            tuple(tuple(term) for term in output)
            for output in parameter.boolean_outputs
        )
        if any(
            type(bit) is not int or not 0 <= bit < index
            for output in outputs
            for term in output
            for bit in term
        ):
            raise TypeError("native Boolean plan has an invalid source bit")
        return BooleanPlan(index, outputs)
    if kind == "measurement":
        return index
    if kind == "scalar":
        try:
            return scalars[index]
        except IndexError as error:
            raise TypeError("native circuit parameter scalar is unavailable") from error
    if kind == "call_result":
        raw_scalar_type = getattr(parameter, "scalar_type", None)
        if not isinstance(raw_scalar_type, str):
            raise TypeError("native call result target has an invalid type")
        scalar_type = {
            "Bool": Bool,
            "Int": Int,
            "Real": Real,
        }.get(raw_scalar_type)
        if scalar_type is None:
            raise TypeError("native call result target has an invalid type")
        value = call_values.get(index, CallValue(index, scalar_type))
        if value.type is not scalar_type:
            raise TypeError("native call result target type is inconsistent")
        return value
    if kind == "hamiltonian":
        payload = getattr(parameter, "payload", None)
        if not isinstance(payload, (bytes, bytearray, list)):
            raise TypeError("native Hamiltonian parameter payload is invalid")
        try:
            from .proto import qcore_pb2 as wire
            from .serialization import decode_literal

            literal = wire.LiteralProto.FromString(bytes(payload))
            return _to_circuit_value(decode_literal(literal))
        except (DecodeError, SerializationError, TypeError, ValueError) as error:
            raise TypeError("native Hamiltonian parameter is invalid") from error
    raise TypeError(f"unknown native circuit parameter kind {kind!r}")


def _native_scalar_values(projection: Any) -> tuple[CircuitValue, ...]:
    """Restore a native hash-consed scalar DAG into Python circuit values.

    Args:
        projection (Any): Native bridge projection carrying topologically
            ordered scalar nodes.

    Returns:
        tuple[CircuitValue, ...]: Scalar values indexed by native scalar ID.

    Raises:
        TypeError: If a scalar node is malformed, references a forward node,
            or carries an unknown kind.
    """

    values: list[CircuitValue] = []
    for scalar in projection.scalars:
        if scalar.kind == "bool":
            value: CircuitValue = scalar.bool_value
        elif scalar.kind == "int":
            value = scalar.int_value
        elif scalar.kind == "loop_index":
            if scalar.int_value is None or scalar.int_value < 0:
                raise TypeError("invalid native loop index")
            value = LoopIndex(scalar.int_value)
        elif scalar.kind == "real":
            value = scalar.real_value
        elif scalar.kind == "measurement":
            if (
                isinstance(scalar.measurement_index, bool)
                or not isinstance(scalar.measurement_index, int)
                or scalar.measurement_index < 0
            ):
                raise TypeError("native measurement node has an invalid index")
            value = MeasurementBit(scalar.measurement_index)
        elif scalar.kind == "call_value":
            if (
                isinstance(scalar.call_index, bool)
                or not isinstance(scalar.call_index, int)
                or scalar.call_index < 0
            ):
                raise TypeError("native call value node has an invalid index")
            scalar_type = {
                "Bool": Bool,
                "Int": Int,
                "Real": Real,
            }.get(scalar.scalar_type)
            if scalar_type is None:
                raise TypeError("native call value node has an invalid type")
            value = CallValue(scalar.call_index, scalar_type)
        elif scalar.kind == "parameter":
            if not isinstance(scalar.name, str):
                raise TypeError("native parameter node has no name")
            path: list[int | str] = []
            for component in scalar.path:
                if component.kind == "index":
                    if (
                        isinstance(component.index, bool)
                        or not isinstance(component.index, int)
                        or component.index < 0
                        or component.field is not None
                    ):
                        raise TypeError("native parameter path has an invalid index")
                    path.append(component.index)
                elif component.kind == "field":
                    if (
                        not isinstance(component.field, str)
                        or not component.field
                        or component.index is not None
                    ):
                        raise TypeError("native parameter path has an invalid field")
                    path.append(component.field)
                else:
                    raise TypeError(
                        f"unknown native parameter path kind {component.kind!r}"
                    )
            value = ParameterValue(scalar.name, tuple(path))
        elif scalar.kind == "apply":
            if not isinstance(scalar.opcode, str):
                raise TypeError("native apply node has no opcode")
            try:
                operands = tuple(values[index] for index in scalar.operands)
            except IndexError as error:
                raise TypeError("native scalar DAG has a forward reference") from error
            attributes = tuple(
                (attribute.name, _native_scalar_attribute_value(attribute))
                for attribute in scalar.attributes
            )
            if any(not isinstance(name, str) or not name for name, _ in attributes):
                raise TypeError("native scalar attribute has an invalid name")
            if len(dict(attributes)) != len(attributes):
                raise TypeError("native scalar attributes are not unique")
            value = ClassicalExpr(
                scalar.opcode,
                operands,
                FrozenMapping(attributes),
            )
        else:
            raise TypeError(f"unknown native scalar kind {scalar.kind!r}")
        values.append(value)
    return tuple(values)


def _native_dispatch_predicate(projection: Any) -> CircuitDispatchPredicate:
    """Restore one verified native launch predicate.

    Args:
        projection: Native predicate projection with a scalar arena and root.

    Returns:
        Public immutable launch predicate.

    Raises:
        TypeError: If the scalar root is unavailable or not Boolean-valued.
    """

    values = _native_scalar_values(projection)
    root = projection.root
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise TypeError("native dispatch predicate has an invalid root")
    try:
        expression = values[root]
    except IndexError as error:
        raise TypeError("native dispatch predicate root is unavailable") from error
    if isinstance(expression, (ParameterValue, ClassicalExpr, bool)):
        return CircuitDispatchPredicate(expression)
    raise TypeError("native dispatch predicate root is not Boolean-valued")


def _native_scalar_attribute_value(attribute: Any) -> PortableValue:
    """Restore one typed native scalar-operation attribute value.

    Args:
        attribute (Any): Native bridge attribute with one closed value tag.

    Returns:
        PortableValue: Immutable Python representation of the native value.

    Raises:
        TypeError: If the tag or tagged payload violates the bridge ABI.
    """

    if attribute.kind == "int":
        value = attribute.int_value
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("native scalar int attribute has an invalid value")
        return value
    if attribute.kind == "string":
        value = attribute.string_value
        if not isinstance(value, str):
            raise TypeError("native scalar string attribute has an invalid value")
        return value
    if attribute.kind == "strings":
        values = tuple(attribute.string_values)
        if any(not isinstance(value, str) for value in values):
            raise TypeError("native scalar string tuple attribute has an invalid value")
        return values
    raise TypeError(f"unknown native scalar attribute kind {attribute.kind!r}")


def _to_circuit_value(value: object) -> CircuitValue:
    """Validate a runtime-free value at the circuit projection boundary."""

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, FrozenMapping):
        thawed = thaw_value(value)
        if isinstance(thawed, dict) and thawed.get("kind") == "qamomile.pauli_sum.v1":
            try:
                return Hamiltonian.from_portable(thawed)
            except (TypeError, ValueError) as error:
                raise CapabilityError("Q7244: malformed Pauli Hamiltonian") from error
        return value
    if isinstance(value, Hamiltonian):
        return value
    if isinstance(value, dict) and value.get("kind") == "qamomile.pauli_sum.v1":
        try:
            return Hamiltonian.from_portable(value)
        except (TypeError, ValueError) as error:
            raise CapabilityError("Q7244: malformed Pauli Hamiltonian") from error
    if isinstance(value, tuple):
        return tuple(_to_circuit_value(item) for item in value)
    raise CapabilityError(f"Q7233: {value!r} is not a circuit-family value")


def scalar_parameters(abi: ParameterABI) -> tuple[ScalarParameter, ...]:
    """Expand a launch ABI into deterministic scalar leaves.

    Fixed arrays and products are an ABI-level user convenience. Circuit SDKs
    may expose only scalar inputs, so Rust provides one bounded, row-major
    structural rule without weakening the public aggregate ABI.

    Raises:
        CapabilityError: If the ABI has a dependent shape, an unsupported
            scalar representation, or exceeds the native work budget.
        TypeError: If the installed native compiler violates its bridge ABI.
    """

    project = getattr(native, "project_scalar_parameters", None)
    if not callable(project):
        raise CapabilityError(
            "Q7245: native circuit scalar parameter projection is unavailable"
        )
    try:
        projection = project(abi._to_bytes())
    except (OverflowError, TypeError, ValueError) as error:
        raise CapabilityError(str(error)) from error
    try:
        if type(projection) is not list:
            raise TypeError("scalar parameter projection must be a list")
        slot_names = {slot.name for slot in abi.slots}
        seen: set[tuple[str, tuple[int | str, ...]]] = set()
        return tuple(
            _decode_scalar_parameter_row(row, slot_names, seen) for row in projection
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "native scalar parameter projection violates the Python ABI"
        ) from error


def _decode_scalar_parameter_row(
    row: object,
    slot_names: set[str],
    seen: set[tuple[str, tuple[int | str, ...]]],
) -> ScalarParameter:
    """Validate one native scalar parameter identity row.

    Args:
        row: Native bridge row containing a name, path, and scalar type.
        slot_names: Parameter names admitted by the public ABI.
        seen: Identities already returned by the native projection.

    Returns:
        The checked public scalar parameter identity.

    Raises:
        TypeError: If the native row violates the bridge ABI.
        ValueError: If the native row names an unknown scalar type.
    """

    if not isinstance(row, tuple) or len(row) != 3:
        raise TypeError("scalar parameter row must be a triple")
    name, encoded_path, scalar_name = row
    if (
        type(name) is not str
        or not name
        or name not in slot_names
        or type(scalar_name) is not str
        or type(encoded_path) is not list
    ):
        raise TypeError("scalar parameter names and types must be strings")
    if scalar_name not in {
        "Unit",
        "Bit",
        "Bool",
        "Int",
        "Real",
        "Complex",
        "Probability",
        "Observable",
    }:
        raise ValueError("unknown scalar parameter type")
    path: list[int | str] = []
    for component in encoded_path:
        if not isinstance(component, tuple) or len(component) != 2:
            raise TypeError("scalar parameter path component must be a pair")
        index, field_name = component
        if type(index) is int and index >= 0 and field_name is None:
            path.append(index)
        elif index is None and type(field_name) is str and field_name:
            path.append(field_name)
        else:
            raise TypeError("scalar parameter path component is ambiguous")
    identity = (name, tuple(path))
    if identity in seen:
        raise TypeError("scalar parameter path is duplicated")
    seen.add(identity)
    return ScalarParameter(ParameterValue(name, identity[1]), ScalarType(scalar_name))


def scalar_parameter_bindings(
    abi: ParameterABI,
    values: Mapping[str, object],
) -> tuple[tuple[ParameterValue, object], ...]:
    """Bind and flatten aggregate values through the native scalar ABI.

    Args:
        abi: Portable artifact parameter ABI.
        values: Supplied launch values, which may omit parameters with defaults.

    Returns:
        Scalar parameter identities paired with ordinary Python values.

    Raises:
        CapabilityError: If the ABI has no bounded circuit-scalar projection.
        ExecutionError: If supplied values violate the artifact ABI.
        TypeError: If the installed native compiler violates its bridge ABI.
    """

    from .proto import qcore_pb2 as wire
    from .serialization import decode_literal, encode_literal

    supplied = wire.LiteralProto()
    try:
        encode_literal(FrozenMapping.from_mapping(values), supplied)
    except (TypeError, SerializationError) as error:
        raise ExecutionError(
            f"Q8135: invocation parameters are not portable: {error}"
        ) from error
    binder = getattr(native, "bind_scalar_parameters", None)
    if not callable(binder):
        raise CapabilityError("Q7245: native scalar parameter binding is unavailable")
    try:
        projection = binder(
            abi._to_bytes(),
            supplied.SerializeToString(deterministic=True),
        )
    except (OverflowError, TypeError, ValueError) as error:
        message = str(error)
        if message.startswith("Q724"):
            raise CapabilityError(message) from error
        raise ExecutionError(message) from error
    try:
        if type(projection) is not list:
            raise TypeError("scalar parameter bindings must be a list")
        slot_names = {slot.name for slot in abi.slots}
        seen: set[tuple[str, tuple[int | str, ...]]] = set()
        bindings: list[tuple[ParameterValue, object]] = []
        for row in projection:
            if not isinstance(row, tuple) or len(row) != 4:
                raise TypeError("scalar parameter binding row must be a quadruple")
            parameter = _decode_scalar_parameter_row(row[:3], slot_names, seen)
            payload = row[3]
            if type(payload) is not bytes:
                raise TypeError("scalar parameter literal must be bytes")
            literal = wire.LiteralProto.FromString(payload)
            if literal.SerializeToString(deterministic=True) != payload:
                raise DecodeError("scalar parameter literal is not canonical")
            bindings.append((parameter.value, thaw_value(decode_literal(literal))))
        return tuple(bindings)
    except (DecodeError, SerializationError, TypeError, ValueError) as error:
        raise TypeError(
            "native scalar parameter binding violates the Python ABI"
        ) from error


def _block_to_data(block: CircuitBlock) -> list[dict[str, object]]:
    """Convert a circuit block to canonical JSON data."""

    if block.has_return:
        raise CapabilityError(
            "Q7247: runtime-return circuit blocks require a structured target"
        )
    return [
        {
            "op": instruction.opcode,
            "qubits": list(instruction.qubits),
            "parameters": [_value_to_data(value) for value in instruction.parameters],
            "condition": (
                _value_to_data(instruction.condition)
                if instruction.condition is not None
                else None
            ),
            "regions": [_block_to_data(region) for region in instruction.regions],
            "controls": list(instruction.controls),
        }
        for instruction in block.instructions
    ]


def _value_to_data(value: CircuitValue) -> object:
    """Convert a circuit value to canonical JSON data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, FrozenMapping):
        return {
            "mapping": [
                [name, _value_to_data(cast(CircuitValue, item))]
                for name, item in value.entries
            ]
        }
    if isinstance(value, Hamiltonian):
        return value.to_portable()
    if isinstance(value, tuple):
        return {"tuple": [_value_to_data(item) for item in value]}
    if isinstance(value, CircuitRecordValue):
        return {"record": [[name, _value_to_data(item)] for name, item in value.fields]}
    if isinstance(value, BooleanPlan):
        return {
            "boolean_plan": {"input_width": value.input_width, "outputs": value.outputs}
        }
    if isinstance(value, LoopIndex):
        return {"loop_index": value.index}
    if isinstance(value, ParameterValue):
        return {"parameter": value.name, "path": list(value.path)}
    if isinstance(value, ClassicalExpr):
        return {
            "expr": value.opcode,
            "operands": [_value_to_data(item) for item in value.operands],
            "attributes": {
                name: thaw_value(item) for name, item in value.attributes.entries
            },
        }
    if isinstance(value, QuantumWires):
        return {"region": value.region, "wires": list(value.wires)}
    if isinstance(value, MeasurementBit):
        return {"measurement": value.index}
    if isinstance(value, CallValue):
        return {"call_value": value.index, "type": str(value.type)}
    if isinstance(value, MeasurementTuple):
        return {"measurements": [item.index for item in value.bits]}
    raise TypeError(f"unsupported circuit value {value!r}")
