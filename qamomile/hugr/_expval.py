"""Prepare hardware-compatible Pauli measurements for HUGR expectations.

The transformation is owned by the HUGR execution target. Shared semantic IR
keeps its abstract expectation operation, while every measurement program
retains the original runtime inputs and compile-time bindings.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import operator
from collections.abc import Mapping, Sequence
from decimal import Decimal
from numbers import Real
from typing import Any
from uuid import UUID

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
)
from qamomile.circuit.ir.operation.operation import CInitOperation, OperationKind
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types import BitType, QubitType, UIntType, ValueType
from qamomile.circuit.ir.value import (
    ArrayValue,
    TupleValue,
    Value,
    ValueLike,
    array_static_length,
)
from qamomile.circuit.transpiler import (
    ClassicalExecutor,
    ClassicalSegment,
    ExecutionContext,
    PreparedModule,
    ProgramABI,
    TargetCapabilityError,
    inline_callables,
    lower_compile_time_ifs_preserving_loop_conditions,
    prepare_module,
)
from qamomile.observable import Hamiltonian, Pauli, PauliOperator
from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL


@dataclasses.dataclass(frozen=True)
class ShotExpectationPlan:
    """Hold independent Pauli measurement programs and their reconstruction.

    Args:
        circuits (tuple[PreparedModule, ...]): One program per nonzero,
            nonidentity Pauli term, in coefficient order. Each returns all
            measured qubits as separate scalar bits in observable order.
        coefficients (tuple[float, ...]): Real coefficients of those terms.
        parity_indices (tuple[tuple[int, ...], ...]): Nonidentity bit positions
            used for each term's eigenvalue.
        constant (float): Exact identity contribution.
        num_qubits (int): Width of every decoded measurement row.
    """

    circuits: tuple[PreparedModule, ...]
    coefficients: tuple[float, ...]
    parity_indices: tuple[tuple[int, ...], ...]
    constant: float
    num_qubits: int

    def aggregate(self, rows_by_term: Sequence[Sequence[Any]]) -> float:
        """Compute a Hamiltonian estimate from independent measurement shots.

        Args:
            rows_by_term (Sequence[Sequence[Any]]): Decoded shots for each
                circuit. A one-qubit shot is a scalar bit or a one-element
                sequence; wider shots are sequences in public output order.

        Returns:
            float: Identity contribution plus coefficient-weighted mean
            Pauli eigenvalues. Each supplied shot has equal weight.

        Raises:
            ValueError: If a term is missing, a shot batch is empty, or a
                decoded row has invalid width or contains a non-bit value.
        """
        if len(rows_by_term) != len(self.circuits):
            raise ValueError("HUGR expectation result count does not match Pauli terms")
        contributions = [self.constant]
        for coefficient, indices, rows in zip(
            self.coefficients, self.parity_indices, rows_by_term, strict=True
        ):
            if len(rows) == 0:
                raise ValueError("HUGR expectation requires at least one shot per term")
            eigenvalue_sum = 0
            for row in rows:
                bits = _measurement_bits(row, self.num_qubits)
                parity = sum(bits[index] for index in indices) % 2
                eigenvalue_sum += 1 - 2 * parity
            contributions.append(coefficient * eigenvalue_sum / len(rows))
        return math.fsum(contributions)


@dataclasses.dataclass(frozen=True)
class ShotExpectationWorkflow:
    """Combine Pauli estimates with the shared classical interpreter.

    Args:
        estimates (tuple[ShotExpectationPlan, ...]): Measurement plans in
            semantic expectation order, including identity-only estimates.
        result_values (tuple[Value, ...]): Values populated by the estimates.
        classical (ClassicalSegment): Required host operations in source order.
        abi (ProgramABI): Input identities and structured public outputs.
        bindings (Mapping[str, Any]): Owned compile-time parameter snapshots.
    """

    estimates: tuple[ShotExpectationPlan, ...]
    result_values: tuple[Value, ...]
    classical: ClassicalSegment
    abi: ProgramABI
    bindings: Mapping[str, Any]

    @property
    def circuits(self) -> tuple[PreparedModule, ...]:
        """Return measurement modules in provider submission order.

        Returns:
            tuple[PreparedModule, ...]: Flattened Pauli measurement programs.
        """
        return tuple(circuit for plan in self.estimates for circuit in plan.circuits)

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Return the coefficients associated with provider submissions.

        Returns:
            tuple[float, ...]: Nonidentity coefficients in submission order.
        """
        return tuple(value for plan in self.estimates for value in plan.coefficients)

    def aggregate(
        self,
        rows_by_term: Sequence[Sequence[Any]],
        bindings: Mapping[str, Any] | None = None,
    ) -> Any:
        """Resolve estimates and interpret the public classical result.

        Args:
            rows_by_term (Sequence[Sequence[Any]]): Decoded measurement batches
                in flattened Pauli-term order.
            bindings (Mapping[str, Any] | None): Validated runtime arguments.

        Returns:
            Any: Public scalar or structured result, with all classical
            computation performed once on the expectation estimates.

        Raises:
            ValueError: If measurement batches are missing or malformed.
            ExecutionError: If classical values or operations cannot be resolved.
        """
        if len(rows_by_term) != len(self.circuits):
            raise ValueError("HUGR expectation result count does not match Pauli terms")
        values = {**self.bindings, **(bindings or {})}
        context = ExecutionContext(dict(values))
        for name, value in self.abi.public_inputs.items():
            if name in values:
                _bind_input(value, values[name], context)
        offset = 0
        for plan, value in zip(self.estimates, self.result_values, strict=True):
            end = offset + len(plan.circuits)
            context.set(value.uuid, plan.aggregate(rows_by_term[offset:end]))
            offset = end
        interpreter = ClassicalExecutor()
        context.update(interpreter.execute(self.classical, context))
        outputs = tuple(
            interpreter.resolve_value(value, context)
            for value in self.abi.output_values
        )
        return outputs[0] if len(outputs) == 1 else outputs

    def recipe(self) -> Any:
        """Identify estimation and host semantics without transient IR UUIDs.

        Returns:
            Any: JSON-compatible estimator, input, and output reconstruction.

        Raises:
            TypeError: If a host semantic payload has no supported encoding.
        """
        payload = (
            tuple(
                (plan.coefficients, plan.parity_indices, plan.constant, plan.num_qubits)
                for plan in self.estimates
            ),
            self.result_values,
            self.classical,
            self.abi,
            {
                name: value
                for name, value in self.bindings.items()
                if not isinstance(value, Hamiltonian)
            },
        )
        return _semantic_identity(payload, {})


def _bind_input(value: ValueLike, data: Any, context: ExecutionContext) -> None:
    """Seed public input identities, including tuple element aliases.

    Args:
        value (ValueLike): Public input or nested tuple element.
        data (Any): Validated concrete input value.
        context (ExecutionContext): Host context to populate.

    Raises:
        ValueError: If a tuple input does not match its validated arity.
    """
    context.set(value.uuid, data)
    if isinstance(value, TupleValue):
        for element, item in zip(value.elements, data, strict=True):
            _bind_input(element, item, context)


def _semantic_identity(value: Any, identities: dict[str, int]) -> Any:
    """Encode host semantics with stable identity relationships.

    Args:
        value (Any): Classical IR, metadata, or supported binding payload.
        identities (dict[str, int]): Canonical UUID numbers assigned in order.

    Returns:
        Any: JSON-compatible structure independent of generated UUIDs.

    Raises:
        TypeError: If a payload cannot be encoded deterministically.
    """
    if isinstance(value, enum.Enum):
        return [type(value).__qualname__, value.name]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            type(value).__qualname__,
            [
                (field.name, _semantic_identity(getattr(value, field.name), identities))
                for field in dataclasses.fields(value)
            ],
        ]
    if isinstance(value, ValueType):
        return value.label()
    if isinstance(value, Mapping):
        return [
            "mapping",
            [
                (
                    _semantic_identity(key, identities),
                    _semantic_identity(item, identities),
                )
                for key, item in value.items()
            ],
        ]
    if isinstance(value, (tuple, list)):
        return [
            type(value).__name__,
            [_semantic_identity(item, identities) for item in value],
        ]
    if isinstance(value, str):
        try:
            UUID(value)
        except ValueError:
            return value
        return ["identity", identities.setdefault(value, len(identities))]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (Real, Decimal)):
        return float(value)
    if hasattr(value, "tolist"):
        return _semantic_identity(value.tolist(), identities)
    raise TypeError(
        f"Unsupported HUGR expectation recipe payload: {type(value).__name__}"
    )


def _measurement_bits(row: Any, width: int) -> tuple[int, ...]:
    """Validate decoded bits without assuming a service's bit-string order.

    Args:
        row (Any): Decoded public output for one shot.
        width (int): Required output width.

    Returns:
        tuple[int, ...]: Binary values in observable qubit order.

    Raises:
        ValueError: If a row has the wrong width or contains non-binary data.
    """
    if width == 1 and not isinstance(row, (list, tuple)):
        row = (row,)
    if not isinstance(row, (list, tuple)) or len(row) != width:
        raise ValueError(f"HUGR expectation measurement row must contain {width} bits")
    result: list[int] = []
    for value in row:
        try:
            bit = operator.index(value)
        except TypeError as error:
            raise ValueError(
                "HUGR expectation measurement contains a non-bit value"
            ) from error
        if bit not in (0, 1):
            raise ValueError("HUGR expectation measurement contains a non-bit value")
        result.append(bit)
    return tuple(result)


def prepare_expval(
    program: PreparedModule,
) -> ShotExpectationPlan | ShotExpectationWorkflow | None:
    """Prepare Pauli measurements and shared classical result computation.

    Quantum preparation remains a direct program graph in each measured
    package. Only inline-policy helpers containing expectations or host
    classical work lose their call boundaries. The supported execution shape
    is classical preparation, one quantum region, expectations, and classical
    postprocessing; quantum work cannot resume after an expectation.

    Args:
        program (PreparedModule): Prepared semantic module to inspect.

    Returns:
        ShotExpectationPlan | ShotExpectationWorkflow | None: Pure scalar
        estimate, structured classical workflow, or ``None`` without expvals.

    Raises:
        TargetCapabilityError: If expectations remain inside control flow or
            boxed helpers, quantum work resumes after an estimate, or an
            observable or state carrier cannot be measured on the target.
    """
    located = _find_expvals(program)
    if not located:
        return None
    if any(owner is not program.entrypoint for owner, _ in located) or any(
        isinstance(op, InvokeOperation) and _host_inline_body(op) is not None
        for op in program.entrypoint.operations
    ):
        entrypoint = inline_callables(
            program.owned_snapshot().entrypoint, body_selector=_host_inline_body
        )
        program = prepare_module(entrypoint, program.bindings)
        located = _find_expvals(program)
    if any(owner is not program.entrypoint for owner, _ in located):
        entrypoint = lower_compile_time_ifs_preserving_loop_conditions(
            program.owned_snapshot().entrypoint, dict(program.bindings)
        )
        program = prepare_module(entrypoint, program.bindings)
        located = _find_expvals(program)
        if not located:
            return None
    if any(owner is not program.entrypoint for owner, _ in located):
        raise _unsupported(
            "requires expectations outside control-flow regions and boxed helpers"
        )
    expvals = tuple(expval for _, expval in located)
    first = program.entrypoint.operations.index(expvals[0])
    prefix = program.entrypoint.operations[:first]
    suffix = program.entrypoint.operations[first:]
    for operation in suffix:
        if not isinstance(operation, (ExpvalOp, ReturnOperation)) and not _is_classical(
            operation
        ):
            raise _unsupported("cannot resume quantum work after an expectation")
    estimates = tuple(_prepare_estimate(program, expval, prefix) for expval in expvals)
    outputs = program.entrypoint.output_values
    if (
        len(expvals) == 1
        and len(outputs) == 1
        and outputs[0].uuid == expvals[0].output.uuid
    ):
        return estimates[0]
    classical = [
        operation
        for operation in program.entrypoint.operations
        if _is_classical(operation)
        and not isinstance(operation, ReturnOperation)
        and not (
            isinstance(operation, CInitOperation)
            and all(value.is_parameter() for value in operation.results)
        )
    ]
    return ShotExpectationWorkflow(
        estimates=estimates,
        result_values=tuple(expval.output for expval in expvals),
        classical=ClassicalSegment(operations=classical),
        abi=program.abi,
        bindings=program.bindings,
    )


def _host_inline_body(operation: InvokeOperation) -> Block | None:
    """Select expectation-bearing or classical helpers for host legalization.

    Args:
        operation (InvokeOperation): Callable boundary under consideration.

    Returns:
        Block | None: Body needed by the host workflow, or ``None`` to
        preserve an unrelated quantum helper as a native graph function.
    """
    body = operation.effective_body()
    if body is None:
        return None
    module = prepare_module(body)
    bodies = [
        body,
        *(
            definition.body
            for definition in module.definitions.values()
            if definition.body is not None
        ),
    ]
    classical = all(
        isinstance(op, (InvokeOperation, HasNestedOps)) or _is_classical(op)
        for candidate in bodies
        for op in _walk_operations(candidate)
    )
    if _find_expvals(module) or classical:
        return body
    return None


def _is_classical(operation: Operation) -> bool:
    """Identify host operations without crossing quantum callable boundaries.

    Args:
        operation (Operation): Operation whose executable contents are examined.

    Returns:
        bool: Whether the operation contains only classical computation.
    """
    if isinstance(operation, InvokeOperation):
        return False
    if isinstance(operation, HasNestedOps):
        return all(
            _is_classical(nested)
            for region in operation.nested_regions()
            for nested in region.operations
        )
    return operation.operation_kind is OperationKind.CLASSICAL


def _prepare_estimate(
    program: PreparedModule, expval: ExpvalOp, prefix: list[Operation]
) -> ShotExpectationPlan:
    """Prepare one expectation while retaining the shared quantum prefix.

    Args:
        program (PreparedModule): Prepared expectation workflow.
        expval (ExpvalOp): Expectation to replace with Pauli measurements.
        prefix (list[Operation]): Direct quantum preparation before all estimates.

    Returns:
        ShotExpectationPlan: Independent measured packages and Pauli weights.

    Raises:
        TargetCapabilityError: If the state or observable is unsupported.
    """
    state = _state_qubits(program.entrypoint, expval)
    observable = _bound_observable(program, expval)
    width = len(state)
    if observable.num_qubits > width:
        raise _unsupported(
            "observable addresses more qubits than the expectation state"
        )
    constant = _real_coefficient(observable.constant)
    circuits: list[PreparedModule] = []
    coefficients: list[float] = []
    parity_indices: list[tuple[int, ...]] = []
    for operators, coefficient in observable.terms.items():
        real = _real_coefficient(coefficient)
        active = tuple(op for op in operators if op.pauli is not Pauli.I)
        if not active:
            constant += real
        elif real != 0.0:
            circuits.append(_measurement_program(program, expval, prefix, active))
            coefficients.append(real)
            parity_indices.append(tuple(op.index for op in active))
    return ShotExpectationPlan(
        circuits=tuple(circuits),
        coefficients=tuple(coefficients),
        parity_indices=tuple(parity_indices),
        constant=constant,
        num_qubits=width,
    )


def _unsupported(message: str) -> TargetCapabilityError:
    """Construct an expectation-specific target capability diagnosis.

    Args:
        message (str): Unsupported capability or malformed input description.

    Returns:
        TargetCapabilityError: Error identifying the HUGR execution target.
    """
    return TargetCapabilityError(
        f"HUGR shot expectation {message}", target="hugr", operation="ExpvalOp"
    )


def _walk_operations(block: Block) -> list[Operation]:
    """Collect operations including nested control-flow and select bodies.

    Args:
        block (Block): Block to inspect.

    Returns:
        list[Operation]: Operations in recursive execution-tree order.
    """
    result: list[Operation] = []
    for operation in block.operations:
        result.append(operation)
        if isinstance(operation, HasNestedOps):
            for region in operation.nested_regions():
                result.extend(
                    _walk_operations(Block(operations=list(region.operations)))
                )
        if isinstance(operation, SelectOperation):
            for case in operation.case_blocks:
                result.extend(_walk_operations(case))
    return result


def _find_expvals(program: PreparedModule) -> list[tuple[Block | None, ExpvalOp]]:
    """Locate expectations while distinguishing entrypoint operation ownership.

    Args:
        program (PreparedModule): Module whose reachable bodies are inspected.

    Returns:
        list[tuple[Block | None, ExpvalOp]]: Unique expectation operations with
        their top-level block owner, or ``None`` for region-owned operations.
    """
    blocks = [program.entrypoint]
    for variants in program.definition_variants.values():
        for definition in variants:
            if definition.body is not None:
                blocks.append(definition.body)
            blocks.extend(
                implementation.body
                for implementation in definition.implementations
                if implementation.body is not None
            )
    found: list[tuple[Block | None, ExpvalOp]] = []
    visited: set[int] = set()
    for block in blocks:
        immediate = {id(operation) for operation in block.operations}
        for operation in _walk_operations(block):
            if isinstance(operation, ExpvalOp) and id(operation) not in visited:
                visited.add(id(operation))
                found.append((block if id(operation) in immediate else None, operation))
    return found


def _bound_observable(program: PreparedModule, expval: ExpvalOp) -> Hamiltonian:
    """Resolve an independent compile-time Hamiltonian snapshot.

    Args:
        program (PreparedModule): Program holding compile-time bindings.
        expval (ExpvalOp): Expectation whose observable is resolved.

    Returns:
        Hamiltonian: Independent observable copy.

    Raises:
        TargetCapabilityError: If the observable has no bound Hamiltonian.
    """
    names = [
        name
        for name, value in program.entrypoint.parameters.items()
        if value.uuid == expval.observable.uuid
    ]
    parameter_name = expval.observable.parameter_name()
    if parameter_name is not None:
        names.append(parameter_name)
    names.append(expval.observable.name)
    for name in names:
        if name in program.bindings:
            observable = program.bindings[name]
            if isinstance(observable, Hamiltonian):
                return observable.copy()
    raise _unsupported("requires a Hamiltonian in compile-time bindings")


def _real_coefficient(coefficient: Any) -> float:
    """Normalize finite Hermitian coefficients using the shared tolerance.

    Args:
        coefficient (Any): Constant or Pauli-term coefficient.

    Returns:
        float: Finite real coefficient.

    Raises:
        TargetCapabilityError: If conversion fails or the coefficient is
            nonfinite or has a non-negligible imaginary component.
    """
    try:
        value = complex(coefficient)
    except (TypeError, ValueError, OverflowError) as error:
        raise _unsupported("requires finite real Hamiltonian coefficients") from error
    if (
        not (math.isfinite(value.real) and math.isfinite(value.imag))
        or abs(value.imag) > HERMITIAN_IMAG_ATOL
    ):
        raise _unsupported("requires finite real Hamiltonian coefficients")
    return float(value.real)


def _state_qubits(block: Block, expval: ExpvalOp) -> tuple[Value, ...]:
    """Resolve scalar qubits in the exact observable-local operand order.

    Args:
        block (Block): Entrypoint containing the expectation.
        expval (ExpvalOp): Expectation carrying a scalar, vector, or tuple.

    Returns:
        tuple[Value, ...]: Scalar qubits including fixed array-element views.

    Raises:
        TargetCapabilityError: If the quantum carrier is symbolic, unsupported,
            or lacks the metadata required to recover tuple order.
    """
    carrier = expval.qubits
    if not isinstance(carrier, Value) or not isinstance(carrier.type, QubitType):
        raise _unsupported("requires a qubit, fixed qubit vector, or tuple of qubits")
    if not isinstance(carrier, ArrayValue):
        return (carrier,)
    if carrier.shape:
        size = array_static_length(carrier)
        if size is None:
            raise _unsupported("requires statically sized expectation qubits")
        return tuple(_array_element(carrier, index) for index in range(size))
    runtime = carrier.metadata.array_runtime
    if runtime is None or not runtime.element_uuids:
        raise _unsupported("cannot resolve the tuple-form expectation qubits")
    values: dict[str, Value] = {}
    for operation in _walk_operations(block):
        for candidate in (*operation.all_input_values(), *operation.results):
            if isinstance(candidate, Value):
                values[candidate.uuid] = candidate
    qubits: list[Value] = []
    for position, uuid in enumerate(runtime.element_uuids):
        value = values.get(uuid)
        if isinstance(value, Value) and not isinstance(value, ArrayValue):
            qubits.append(value)
            continue
        if position < len(runtime.element_parent_uuids) and position < len(
            runtime.element_parent_indices
        ):
            parent = values.get(runtime.element_parent_uuids[position])
            index = runtime.element_parent_indices[position]
            size = (
                array_static_length(parent) if isinstance(parent, ArrayValue) else None
            )
            if (
                isinstance(parent, ArrayValue)
                and isinstance(parent.type, QubitType)
                and size is not None
                and 0 <= index < size
            ):
                qubits.append(_array_element(parent, index))
                continue
        raise _unsupported("cannot resolve every tuple-form expectation qubit")
    return tuple(qubits)


def _array_element(array: ArrayValue, index: int) -> Value:
    """Build a scalar qubit view at a fixed array-local index.

    Args:
        array (ArrayValue): Quantum array or strided view.
        index (int): Index in the array's local order.

    Returns:
        Value: Scalar qubit with preserved parent addressing.
    """
    return Value(
        type=QubitType(),
        name=f"{array.name}[{index}]",
        parent_array=array,
        element_indices=(Value(type=UIntType(), name="index").with_const(index),),
    )


def _measurement_program(
    program: PreparedModule,
    expval: ExpvalOp,
    prefix: list[Operation],
    operators: tuple[PauliOperator, ...],
) -> PreparedModule:
    """Create an independent measurement program for one Pauli product.

    Args:
        program (PreparedModule): Original expectation workflow.
        expval (ExpvalOp): Expectation selected for this Pauli measurement.
        prefix (list[Operation]): Quantum preparation before the first estimate.
        operators (tuple[PauliOperator, ...]): Nonidentity Pauli operators.

    Returns:
        PreparedModule: Program preserving runtime inputs and returning bits.

    Raises:
        TargetCapabilityError: If copied expectation qubits cannot be resolved.
    """
    # Copy prefix and selected expectation together so array aliases remain
    # shared, then rebuild the reachable registry after dropping host work.
    selected = dataclasses.replace(
        program.entrypoint,
        operations=[*prefix, expval],
        input_values=[
            program.abi.public_inputs.get(name, value)
            for name, value in zip(
                program.entrypoint.label_args,
                program.entrypoint.input_values,
                strict=True,
            )
        ]
        if program.entrypoint.label_args
        else program.entrypoint.input_values,
        output_values=[expval.output],
        output_names=[expval.output.name],
    )
    lowered = prepare_module(selected, program.bindings).owned_snapshot()
    copied_expval = lowered.entrypoint.operations[-1]
    assert isinstance(copied_expval, ExpvalOp)
    qubits = _state_qubits(lowered.entrypoint, copied_expval)
    axes = {op.index: op.pauli for op in operators}
    replacement: list[Operation] = []
    outputs: list[Value] = []
    for index, qubit in enumerate(qubits):
        axis = axes.get(index, Pauli.Z)
        gates = {
            Pauli.X: (GateOperationType.H,),
            Pauli.Y: (GateOperationType.SDG, GateOperationType.H),
            Pauli.Z: (),
        }[axis]
        for gate in gates:
            rotated = qubit.next_version()
            replacement.append(GateOperation.fixed(gate, [qubit], [rotated]))
            qubit = rotated
        bit = Value(type=BitType(), name=f"__qamomile_expval_bit_{index}")
        replacement.append(MeasureOperation(operands=[qubit], results=[bit]))
        outputs.append(bit)
    replacement.append(ReturnOperation(operands=outputs, results=[]))
    prefix = lowered.entrypoint.operations[
        : lowered.entrypoint.operations.index(copied_expval)
    ]
    entrypoint = dataclasses.replace(
        lowered.entrypoint,
        operations=[*prefix, *replacement],
        output_values=list(outputs),
        output_names=[value.name for value in outputs],
    )
    return dataclasses.replace(
        lowered,
        entrypoint=entrypoint,
        abi=ProgramABI(
            public_inputs=dict(lowered.abi.public_inputs), output_values=list(outputs)
        ),
    )
