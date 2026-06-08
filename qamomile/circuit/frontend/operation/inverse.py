"""Frontend helpers for applying inverse quantum operations."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.array import ArrayBase, VectorView
from qamomile.circuit.frontend.operation.control import _qkernel_for_callable
from qamomile.circuit.frontend.qkernel import (
    QKernel,
    _promote_literal_to_handle,
)
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    InverseBlockOperation,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.ir.value_mapping import ValueSubstitutor

if TYPE_CHECKING:
    from inspect import BoundArguments


_SELF_INVERSE_GATES: frozenset[GateOperationType] = frozenset(
    {
        GateOperationType.H,
        GateOperationType.X,
        GateOperationType.Y,
        GateOperationType.Z,
        GateOperationType.CX,
        GateOperationType.CZ,
        GateOperationType.SWAP,
        GateOperationType.TOFFOLI,
    }
)

_DAGGER_GATES: dict[GateOperationType, GateOperationType] = {
    GateOperationType.S: GateOperationType.SDG,
    GateOperationType.SDG: GateOperationType.S,
    GateOperationType.T: GateOperationType.TDG,
    GateOperationType.TDG: GateOperationType.T,
}

_ROTATION_GATES: frozenset[GateOperationType] = frozenset(
    {
        GateOperationType.P,
        GateOperationType.RX,
        GateOperationType.RY,
        GateOperationType.RZ,
        GateOperationType.CP,
        GateOperationType.RZZ,
    }
)


def _normalize_zero(value: float) -> float:
    """Normalize signed floating zero to positive zero.

    Args:
        value (float): Floating-point value to normalize.

    Returns:
        float: `0.0` when `value` is either signed zero, otherwise `value`.
    """
    return abs(value) if value == 0.0 else value


@dataclasses.dataclass(frozen=True)
class _InverseRotationCallable:
    """Apply the inverse of a native rotation gate callable.

    Args:
        rotation_callable (Callable[..., Any]): Native rotation gate
            callable whose signature should be mirrored for argument
            binding and invocation.
        angle_param (str): Name of the angle parameter to negate before
            invoking `rotation_callable`.
    """

    rotation_callable: Callable[..., Any]
    angle_param: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the native inverse rotation operation.

        Args:
            *args (Any): Positional arguments accepted by
                `rotation_callable`.
            **kwargs (Any): Keyword arguments accepted by
                `rotation_callable`.

        Returns:
            Any: Result from `rotation_callable` with the angle negated.

        Raises:
            TypeError: If the supplied arguments do not match the rotation
                gate signature.
        """
        signature = inspect.signature(self.rotation_callable)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        negated = -bound.arguments[self.angle_param]
        if isinstance(negated, Real):
            negated = _normalize_zero(float(negated))
        bound.arguments[self.angle_param] = negated
        return self.rotation_callable(*bound.args, **bound.kwargs)


@dataclasses.dataclass
class _InputBinding:
    """Track a bound inverse-call argument.

    Args:
        name (str): Python parameter name in the wrapped kernel.
        handle (Handle): Original frontend handle supplied by the caller.
        active_handle (Handle): Handle whose value should be used in IR.
            Quantum non-view handles are consumed before they become active.
        block_input (ValueBase): The corresponding input value in the
            selected block.
    """

    name: str
    handle: Handle
    active_handle: Handle
    block_input: ValueBase

    @property
    def is_quantum(self) -> bool:
        """Return whether this binding carries quantum state.

        Returns:
            bool: True when the active handle's IR value is quantum typed.
        """
        return self.active_handle.value.type.is_quantum()


def _substitute_value(value: ValueBase, value_map: dict[str, ValueBase]) -> ValueBase:
    """Resolve a value through an inverse-construction mapping.

    Args:
        value (ValueBase): Value to substitute.
        value_map (dict[str, ValueBase]): UUID-keyed value mapping.

    Returns:
        ValueBase: The substituted value, with nested array metadata also
            rewritten when needed.
    """
    return ValueSubstitutor(value_map, transitive=True).substitute_value(value)


def _as_value(value: ValueBase, context: str) -> Value:
    """Return an IR Value or raise a clear error.

    Args:
        value (ValueBase): Value candidate.
        context (str): Human-readable context for the error message.

    Returns:
        Value: The same value narrowed to `Value`.

    Raises:
        TypeError: If `value` is not a `Value`.
    """
    if isinstance(value, Value):
        return value
    raise TypeError(f"{context} requires a Value, got {type(value).__name__}.")


def _static_quantum_width(value: ValueBase) -> int | None:
    """Return the compile-time scalar qubit width of a quantum value.

    Args:
        value (ValueBase): Scalar qubit or ``Vector[Qubit]`` value.

    Returns:
        int | None: Number of scalar qubits represented by ``value`` when
            statically known, or None for unresolved vector lengths.
    """
    if isinstance(value, ArrayValue):
        if not value.shape:
            return None
        dim = value.shape[0]
        if not dim.is_constant():
            return None
        return int(dim.get_const())
    return 1


def _fresh_result_value(
    value: ValueBase,
    value_map: dict[str, ValueBase],
) -> ValueBase:
    """Create a next-version result with mapped metadata fields.

    Args:
        value (ValueBase): Original result value to clone.
        value_map (dict[str, ValueBase]): Current inverse value mapping.

    Returns:
        ValueBase: A fresh next-version value with parent arrays, shape
            values, and slice metadata resolved through `value_map`.
    """
    return _substitute_value(value.next_version(), value_map)


def _const_float(name: str, value: float) -> Value:
    """Create a constant float IR value.

    Args:
        name (str): Display name for the constant.
        value (float): Numeric value to store.

    Returns:
        Value: A `FloatType` value carrying `value` as metadata.
    """
    return Value(type=FloatType(), name=name).with_const(_normalize_zero(value))


def _const_uint(name: str, value: int) -> Value:
    """Create a constant UInt IR value.

    Args:
        name (str): Display name for the constant.
        value (int): Integer value to store. Negative sentinel values are
            accepted because existing loop IR uses `UIntType` for Python
            range bounds such as `-1`.

    Returns:
        Value: A `UIntType` value carrying `value` as metadata.
    """
    return Value(type=UIntType(), name=name).with_const(value)


def _validate_input_shape(
    name: str,
    block_input: ValueBase,
    actual: ValueBase,
) -> None:
    """Validate that an inverse call argument matches the wrapped input shape.

    Args:
        name (str): Python argument name being checked.
        block_input (ValueBase): Input value declared by the wrapped block.
        actual (ValueBase): Caller-side value supplied to the inverse wrapper.

    Returns:
        None.

    Raises:
        TypeError: If one side is an array value and the other is scalar.
    """
    expected_array = isinstance(block_input, ArrayValue)
    actual_array = isinstance(actual, ArrayValue)
    if expected_array != actual_array:
        expected = "Vector" if expected_array else "scalar"
        got = "Vector" if actual_array else "scalar"
        raise TypeError(
            f"inverse(): argument {name!r} shape does not match the wrapped "
            f"kernel input; expected {expected}, got {got}."
        )


def _copy_resource_metadata(
    resource_metadata: ResourceMetadata | None,
) -> ResourceMetadata | None:
    """Copy resource metadata for an inverse opaque composite.

    Args:
        resource_metadata (ResourceMetadata | None): Metadata attached to
            the original composite operation.

    Returns:
        ResourceMetadata | None: Independent metadata object with the same
            resource values, or None when the original had no metadata.
    """
    if resource_metadata is None:
        return None
    return dataclasses.replace(
        resource_metadata,
        custom_metadata=copy.deepcopy(resource_metadata.custom_metadata),
    )


def _inverse_stub_name(name: str) -> str:
    """Return an opaque inverse-stub name.

    Args:
        name (str): Original custom composite name.

    Returns:
        str: Name with an `_inv` suffix toggled.
    """
    return name[:-4] if name.endswith("_inv") else f"{name}_inv"


def _operation_result_map(
    op: Operation,
    value_map: dict[str, ValueBase],
) -> dict[str, ValueBase]:
    """Build fresh result substitutions for an operation.

    Args:
        op (Operation): Operation whose results should be cloned.
        value_map (dict[str, ValueBase]): Current inverse value mapping.

    Returns:
        dict[str, ValueBase]: Mapping from original result UUIDs to fresh
            result values.
    """
    return {
        result.uuid: _fresh_result_value(result, value_map) for result in op.results
    }


class _BlockInverter:
    """Invert supported Qamomile IR blocks.

    The inverter emits classical helper operations in original order, then
    walks quantum operations in reverse order. It intentionally rejects
    non-unitary operations and control flow whose inverse is not yet defined.

    Args:
        None.
    """

    def __init__(self) -> None:
        """Initialize an empty recursion guard."""
        self._active_blocks: set[int] = set()

    def invert_block(
        self,
        block: Block,
        extra_value_map: dict[str, ValueBase] | None = None,
    ) -> Block:
        """Create a standalone inverse block.

        Args:
            block (Block): Block to invert. Its quantum outputs must be
                pass-through versions of its quantum inputs.
            extra_value_map (dict[str, ValueBase] | None): Optional
                UUID-keyed substitutions for auxiliary values such as
                call-site-resolved vector shape dimensions. Defaults to
                None.

        Returns:
            Block: A hierarchical block containing the inverse operations.

        Raises:
            NotImplementedError: If the block contains an unsupported
                operation or a recursive cycle.
            TypeError: If the block output contract is not unitary-like.
        """
        self._reject_unsupported_control_flow(block.operations)
        value_map: dict[str, ValueBase] = {
            value.uuid: value for value in block.input_values
        }
        if extra_value_map is not None:
            value_map.update(extra_value_map)
        self._seed_output_values(block, value_map)
        operations = self._invert_block_operations(block, value_map)
        output_values = [
            cast(Value, value_map[value.uuid])
            for value in block.input_values
            if value.type.is_quantum()
        ]
        return Block(
            name=f"{block.name}_inverse",
            label_args=list(block.label_args),
            input_values=list(block.input_values),
            output_values=output_values,
            operations=operations,
            kind=BlockKind.HIERARCHICAL,
            parameters=dict(block.parameters),
            param_slots=block.param_slots,
        )

    def invert_call_site(
        self,
        block: Block,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a block into the caller's current operation list.

        Args:
            block (Block): Block to invert.
            value_map (dict[str, ValueBase]): Caller-side mapping seeded
                with block inputs and outputs.

        Returns:
            list[Operation]: Inverse operations ready to append to the
                active tracer.

        Raises:
            NotImplementedError: If an unsupported operation is reached.
            TypeError: If the block output contract is not unitary-like.
        """
        self._reject_unsupported_control_flow(block.operations)
        self._seed_output_values(block, value_map)
        return self._invert_block_operations(block, value_map)

    def _reject_unsupported_control_flow(self, operations: list[Operation]) -> None:
        """Raise early for control-flow forms whose inverse is undefined.

        Args:
            operations (list[Operation]): Operations to scan.

        Returns:
            None.

        Raises:
            NotImplementedError: If an unsupported control-flow operation is
                found.
        """
        for op in operations:
            if isinstance(op, (IfOperation, WhileOperation, ForItemsOperation)):
                raise NotImplementedError(
                    f"inverse() does not support {type(op).__name__} yet."
                )
            if isinstance(op, ForOperation):
                self._reject_unsupported_control_flow(op.operations)
            if isinstance(op, CallBlockOperation) and op.block is not None:
                self._reject_unsupported_control_flow(op.block.operations)

    def _seed_output_values(
        self,
        block: Block,
        value_map: dict[str, ValueBase],
    ) -> None:
        """Map block outputs to the current caller-side quantum values.

        Args:
            block (Block): Block whose outputs are being used as inverse
                inputs.
            value_map (dict[str, ValueBase]): Mapping to mutate.

        Returns:
            None.

        Raises:
            TypeError: If any output is non-quantum or does not correspond
                to exactly one quantum input.
        """
        quantum_inputs = [
            value for value in block.input_values if value.type.is_quantum()
        ]
        quantum_outputs = list(block.output_values)
        if len(quantum_outputs) != len(quantum_inputs):
            raise TypeError(
                "inverse() can only invert kernels whose quantum outputs "
                "preserve every quantum input."
            )
        output_logical_ids = [output.logical_id for output in quantum_outputs]
        input_logical_ids = [input_value.logical_id for input_value in quantum_inputs]
        if set(output_logical_ids) != set(input_logical_ids) or len(
            output_logical_ids
        ) != len(set(output_logical_ids)):
            raise TypeError(
                "inverse() can only invert kernels whose quantum outputs "
                "preserve the logical identity of every input quantum value."
            )
        quantum_inputs_by_logical_id = {
            value.logical_id: value for value in quantum_inputs
        }
        positional_values = [
            value_map[input_value.uuid] for input_value in quantum_inputs
        ]
        for output, positional_value in zip(quantum_outputs, positional_values):
            if not output.type.is_quantum():
                raise TypeError(
                    "inverse() can only invert kernels whose outputs are "
                    "quantum values corresponding to quantum inputs."
                )
            input_value = quantum_inputs_by_logical_id.get(output.logical_id)
            if input_value is None:
                raise TypeError(
                    "inverse() can only invert kernels whose quantum outputs "
                    "preserve the logical identity of an input quantum value."
                )
            value_map[input_value.uuid] = positional_value
            value_map[output.uuid] = positional_value

    def _invert_block_operations(
        self,
        block: Block,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert all operations in a block.

        Args:
            block (Block): Block being inverted.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: Inverted operation list.

        Raises:
            NotImplementedError: If a recursive block cycle or unsupported
                operation is encountered.
        """
        block_id = id(block)
        if block_id in self._active_blocks:
            raise NotImplementedError("inverse() does not support recursive kernels.")
        self._active_blocks.add(block_id)
        try:
            return self._invert_operations(block.operations, value_map)
        finally:
            self._active_blocks.remove(block_id)

    def _invert_operations(
        self,
        operations: list[Operation],
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert an operation list.

        Args:
            operations (list[Operation]): Operations to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: Classical clones followed by reversed quantum
                inverses.

        Raises:
            NotImplementedError: If an unsupported operation is encountered.
        """
        inverted: list[Operation] = []
        for op in operations:
            if isinstance(op, ReturnOperation):
                continue
            if op.operation_kind is OperationKind.CLASSICAL:
                inverted.append(self._clone_classical_operation(op, value_map))

        for op in reversed(operations):
            if isinstance(op, ReturnOperation):
                continue
            if op.operation_kind is OperationKind.CLASSICAL:
                continue
            inverted.extend(self._invert_operation(op, value_map))
        return inverted

    def _clone_classical_operation(
        self,
        op: Operation,
        value_map: dict[str, ValueBase],
    ) -> Operation:
        """Clone a classical operation with substituted operands.

        Args:
            op (Operation): Classical operation to clone.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            Operation: The cloned operation.
        """
        result_map = _operation_result_map(op, value_map)
        substitutor = ValueSubstitutor({**value_map, **result_map}, transitive=True)
        cloned = substitutor.substitute_operation(op)
        value_map.update(result_map)
        return cloned

    def _invert_operation(
        self,
        op: Operation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a single non-classical operation.

        Args:
            op (Operation): Operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: One or more inverse operations.

        Raises:
            NotImplementedError: If the operation is unsupported or
                non-unitary.
        """
        if isinstance(op, GateOperation):
            return self._invert_gate(op, value_map)
        if isinstance(op, InverseBlockOperation):
            return self._invert_inverse_block(op, value_map)
        if isinstance(op, CompositeGateOperation):
            return self._invert_composite_gate(op, value_map)
        if isinstance(op, PauliEvolveOp):
            return self._invert_pauli_evolve(op, value_map)
        if isinstance(op, ControlledUOperation):
            return self._invert_controlled_u(op, value_map)
        if isinstance(op, CallBlockOperation):
            return self._invert_call_block(op, value_map)
        if isinstance(op, ForOperation):
            return self._invert_for(op, value_map)
        if isinstance(op, (IfOperation, WhileOperation, ForItemsOperation)):
            raise NotImplementedError(
                f"inverse() does not support {type(op).__name__} yet."
            )
        if isinstance(
            op,
            (
                MeasureOperation,
                MeasureVectorOperation,
                MeasureQFixedOperation,
            ),
        ):
            raise NotImplementedError(
                f"inverse() cannot invert non-unitary {type(op).__name__}."
            )
        if isinstance(op, QInitOperation):
            raise NotImplementedError(
                "inverse() cannot invert kernels that allocate qubits internally."
            )
        raise NotImplementedError(
            f"inverse() does not know how to invert {type(op).__name__}."
        )

    def _negate_angle(
        self,
        rotation_angle: Value,
        value_map: dict[str, ValueBase],
    ) -> tuple[list[Operation], Value]:
        """Create the IR value representing `-theta`.

        Args:
            rotation_angle (Value): Angle value to negate.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            tuple[list[Operation], Value]: Extra classical operations and
                the resulting negated angle value.
        """
        from qamomile.circuit.ir.operation.arithmetic_operations import (
            BinOp,
            BinOpKind,
        )

        mapped_rotation_angle = _as_value(
            _substitute_value(rotation_angle, value_map),
            "angle",
        )
        if mapped_rotation_angle.is_constant():
            const = mapped_rotation_angle.get_const()
            assert const is not None
            return [], _const_float(
                f"{mapped_rotation_angle.name}_inverse",
                -float(const),
            )

        minus_one = _const_float("inverse_minus_one", -1.0)
        result = Value(type=FloatType(), name=f"{mapped_rotation_angle.name}_inverse")
        op = BinOp(
            operands=[mapped_rotation_angle, minus_one],
            results=[result],
            kind=BinOpKind.MUL,
        )
        return [op], result

    def _invert_gate(
        self,
        op: GateOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a native gate operation.

        Args:
            op (GateOperation): Gate operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: The inverse gate operation, plus any needed
                classical angle-negation operation.

        Raises:
            NotImplementedError: If the gate type is unknown.
        """
        if op.gate_type in _SELF_INVERSE_GATES:
            inverse_gate_type = op.gate_type
        elif op.gate_type in _DAGGER_GATES:
            inverse_gate_type = _DAGGER_GATES[op.gate_type]
        elif op.gate_type in _ROTATION_GATES:
            inverse_gate_type = op.gate_type
        else:
            raise NotImplementedError(
                f"inverse() does not know how to invert {op.gate_type}."
            )

        current_qubits = [
            _as_value(_substitute_value(result, value_map), "gate result")
            for result in op.results
        ]
        new_results = [qubit.next_version() for qubit in current_qubits]
        extra_ops: list[Operation] = []
        rotation_angle = op.theta
        if rotation_angle is None:
            inverse_op = GateOperation.fixed(
                inverse_gate_type,
                current_qubits,
                new_results,
            )
        else:
            angle_ops, inverse_rotation_angle = self._negate_angle(
                rotation_angle,
                value_map,
            )
            extra_ops.extend(angle_ops)
            inverse_op = GateOperation.rotation(
                inverse_gate_type,
                current_qubits,
                inverse_rotation_angle,
                new_results,
            )

        for operand, result in zip(op.qubit_operands, new_results):
            value_map[operand.uuid] = result
        return [*extra_ops, inverse_op]

    def _invert_composite_gate(
        self,
        op: CompositeGateOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a composite gate operation.

        Args:
            op (CompositeGateOperation): Composite gate operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: The inverse composite operation.

        Raises:
            NotImplementedError: If the composite gate has no known inverse
                and no implementation block.
        """
        current_qubits = [
            _as_value(_substitute_value(result, value_map), "composite result")
            for result in op.results
        ]
        new_results = [qubit.next_version() for qubit in current_qubits]
        mapped_params = [
            _as_value(_substitute_value(param, value_map), "composite parameter")
            for param in op.parameters
        ]

        gate_type = op.gate_type
        custom_name = op.custom_name
        implementation_block = op.implementation_block
        has_implementation = op.has_implementation
        resource_metadata = _copy_resource_metadata(op.resource_metadata)
        strategy_name = op.strategy_name
        source_block = None

        if op.gate_type is CompositeGateType.QFT:
            from qamomile.circuit.stdlib.qft import IQFT

            gate_type = CompositeGateType.IQFT
            custom_name = "iqft"
            implementation_block = None
            has_implementation = False
            if strategy_name is not None and IQFT.get_strategy(strategy_name) is None:
                raise NotImplementedError(
                    "inverse() cannot invert QFT with strategy "
                    f"{strategy_name!r} because IQFT does not define the "
                    "same strategy."
                )
        elif op.gate_type is CompositeGateType.IQFT:
            from qamomile.circuit.stdlib.qft import QFT

            gate_type = CompositeGateType.QFT
            custom_name = "qft"
            implementation_block = None
            has_implementation = False
            if strategy_name is not None and QFT.get_strategy(strategy_name) is None:
                raise NotImplementedError(
                    "inverse() cannot invert IQFT with strategy "
                    f"{strategy_name!r} because QFT does not define the "
                    "same strategy."
                )
        elif op.implementation is not None:
            gate_type = CompositeGateType.CUSTOM
            source_block = op.implementation
            implementation_block = self.invert_block(op.implementation)
            has_implementation = True
            resource_metadata = None
            custom_name = f"{op.name}_inverse"
        elif op.gate_type is not CompositeGateType.CUSTOM:
            raise NotImplementedError(
                "inverse() cannot invert native CompositeGateOperation "
                f"{op.gate_type.value!r}. Use an explicit inverse operation "
                "or provide an implementation block."
            )
        else:
            gate_type = CompositeGateType.CUSTOM
            source_block = None
            implementation_block = None
            has_implementation = False
            custom_name = _inverse_stub_name(op.name)

        if source_block is not None and implementation_block is not None:
            inverse_op: Operation = InverseBlockOperation(
                operands=[*current_qubits, *mapped_params],
                results=new_results,
                num_control_qubits=op.num_control_qubits,
                num_target_qubits=op.num_target_qubits,
                custom_name=custom_name,
                source_block=source_block,
                implementation_block=implementation_block,
            )
        else:
            inverse_op = CompositeGateOperation(
                operands=[*current_qubits, *mapped_params],
                results=new_results,
                gate_type=gate_type,
                num_control_qubits=op.num_control_qubits,
                num_target_qubits=op.num_target_qubits,
                custom_name=custom_name,
                resource_metadata=resource_metadata,
                has_implementation=has_implementation,
                implementation_block=implementation_block,
                composite_gate_instance=None,
                strategy_name=strategy_name,
            )
        for operand, result in zip(op.control_qubits + op.target_qubits, new_results):
            value_map[operand.uuid] = result
        return [inverse_op]

    def _invert_inverse_block(
        self,
        op: InverseBlockOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert an existing first-class inverse block operation.

        Args:
            op (InverseBlockOperation): Inverse block operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: Forward source-block operations cloned into
                the current inverse construction site.

        Raises:
            NotImplementedError: If the inverse op lacks the source block
                needed to reconstruct the forward operation.
        """
        if op.source_block is None:
            raise NotImplementedError(
                "inverse() cannot invert an InverseBlockOperation without "
                "a source block."
            )
        self._reject_unsupported_control_flow(op.source_block.operations)
        current_controls = [
            _as_value(
                _substitute_value(result, value_map),
                "inverse block control result",
            )
            for result in op.results[: op.num_control_qubits]
        ]
        current_targets = [
            _as_value(
                _substitute_value(result, value_map),
                "inverse block target result",
            )
            for result in op.results[op.num_control_qubits :]
        ]
        mapped_params = [_substitute_value(param, value_map) for param in op.parameters]
        if len(current_targets) != len(op.target_qubits) or len(
            op.source_block.output_values
        ) != len(op.target_qubits):
            raise TypeError(
                "inverse() cannot restore an InverseBlockOperation whose "
                "source block outputs do not match its stored target operands."
            )

        if op.num_control_qubits > 0:
            return self._restore_controlled_inverse_block(
                op,
                current_controls,
                current_targets,
                mapped_params,
                value_map,
            )

        local_map = dict(value_map)
        self._bind_forward_block_inputs(
            op.source_block,
            current_targets,
            mapped_params,
            local_map,
        )
        operations = self._clone_forward_operations(
            op.source_block.operations, local_map
        )

        for output, operand in zip(op.source_block.output_values, op.target_qubits):
            resolved = _substitute_value(output, local_map)
            value_map[operand.uuid] = resolved
            if isinstance(operand, ArrayValue) and isinstance(resolved, ArrayValue):
                for operand_dim, resolved_dim in zip(operand.shape, resolved.shape):
                    if operand_dim.uuid != resolved_dim.uuid:
                        value_map[operand_dim.uuid] = resolved_dim
        return operations

    def _restore_controlled_inverse_block(
        self,
        op: InverseBlockOperation,
        current_controls: Sequence[ValueBase],
        current_targets: Sequence[ValueBase],
        mapped_params: Sequence[ValueBase],
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Restore a controlled forward block from an inverse block.

        Args:
            op (InverseBlockOperation): Controlled inverse block being
                inverted.
            current_controls (Sequence[ValueBase]): Current control values
                flowing out of `op`.
            current_targets (Sequence[ValueBase]): Current target values
                flowing out of `op`.
            mapped_params (Sequence[ValueBase]): Classical/object operands
                after applying `value_map`.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map
                to update with the restored operation results.

        Returns:
            list[Operation]: A single controlled forward operation.

        Raises:
            NotImplementedError: If the controlled inverse block has no
                source block to restore.
        """
        if op.source_block is None:
            raise NotImplementedError(
                "inverse() cannot restore a controlled inverse block without "
                "a source block."
            )

        new_controls = [control.next_version() for control in current_controls]
        new_targets = [target.next_version() for target in current_targets]
        restored = ConcreteControlledU(
            operands=cast(
                list[Value],
                [*current_controls, *current_targets, *mapped_params],
            ),
            results=cast(list[Value], [*new_controls, *new_targets]),
            num_controls=op.num_control_qubits,
            block=op.source_block,
        )

        self._update_quantum_value_map(value_map, op.control_qubits, new_controls)
        self._update_quantum_value_map(value_map, op.target_qubits, new_targets)
        return [restored]

    def _bind_forward_block_inputs(
        self,
        block: Block,
        quantum_operands: Sequence[ValueBase],
        parameter_operands: Sequence[ValueBase],
        value_map: dict[str, ValueBase],
    ) -> None:
        """Bind a forward source block to inverse-of-inverse operands.

        Args:
            block (Block): Source block being restored.
            quantum_operands (Sequence[ValueBase]): Current target quantum
                values that feed the source block.
            parameter_operands (Sequence[ValueBase]): Classical/object
                operands for the source block.
            value_map (dict[str, ValueBase]): Local mapping to mutate.

        Returns:
            None.

        Raises:
            TypeError: If the stored inverse block operands no longer match
                the source block input contract.
        """
        quantum_inputs = [
            value for value in block.input_values if value.type.is_quantum()
        ]
        parameter_inputs = [
            value for value in block.input_values if not value.type.is_quantum()
        ]
        if len(quantum_inputs) != len(quantum_operands) or len(parameter_inputs) != len(
            parameter_operands
        ):
            raise TypeError(
                "inverse() cannot restore an InverseBlockOperation whose "
                "source block inputs do not match its stored operands."
            )

        for block_input, operand in [
            *zip(quantum_inputs, quantum_operands),
            *zip(parameter_inputs, parameter_operands),
        ]:
            resolved = _substitute_value(operand, value_map)
            value_map[block_input.uuid] = resolved
            if isinstance(block_input, ArrayValue) and isinstance(resolved, ArrayValue):
                for block_dim, operand_dim in zip(block_input.shape, resolved.shape):
                    value_map[block_dim.uuid] = _substitute_value(
                        operand_dim,
                        value_map,
                    )

    def _clone_forward_operations(
        self,
        operations: list[Operation],
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Clone forward operations into the current inverse construction.

        Args:
            operations (list[Operation]): Source-block operations in
                forward order.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: Cloned forward operations.
        """
        cloned: list[Operation] = []
        for op in operations:
            if isinstance(op, ReturnOperation):
                continue
            if isinstance(op, (IfOperation, WhileOperation, ForItemsOperation)):
                raise NotImplementedError(
                    f"inverse() does not support {type(op).__name__} yet."
                )
            if isinstance(op, ForOperation):
                cloned.append(self._clone_forward_for(op, value_map))
                continue
            cloned.append(self._clone_forward_operation(op, value_map))
        return cloned

    def _clone_forward_operation(
        self,
        op: Operation,
        value_map: dict[str, ValueBase],
    ) -> Operation:
        """Clone one forward operation and advance the local value map.

        Args:
            op (Operation): Source operation to clone.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            Operation: Cloned operation with substituted operands and fresh
                results.
        """
        result_map = _operation_result_map(op, value_map)
        substitutor = ValueSubstitutor({**value_map, **result_map}, transitive=True)
        cloned = substitutor.substitute_operation(op)
        value_map.update(result_map)
        return cloned

    def _clone_forward_for(
        self,
        op: ForOperation,
        value_map: dict[str, ValueBase],
    ) -> ForOperation:
        """Clone a forward compile-time range loop.

        Args:
            op (ForOperation): Source loop operation to clone.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            ForOperation: Cloned loop with its body cloned in forward order.

        Raises:
            TypeError: If the loop variable cannot be represented as a
                scalar IR value.
        """
        body_map = dict(value_map)
        loop_var_value = None
        if op.loop_var_value is not None:
            loop_var_value = _as_value(
                _fresh_result_value(op.loop_var_value, value_map),
                "forward loop variable",
            )
            body_map[op.loop_var_value.uuid] = loop_var_value

        body = self._clone_forward_operations(op.operations, body_map)
        excluded_uuids = {op.loop_var_value.uuid} if op.loop_var_value else set()
        self._merge_loop_body_map(value_map, body_map, excluded_uuids)

        cloned = dataclasses.replace(
            op,
            loop_var_value=loop_var_value,
            operations=body,
        )
        substitutor = ValueSubstitutor(value_map, transitive=True)
        result = substitutor.substitute_operation(cloned)
        assert isinstance(result, ForOperation)
        return result

    def _invert_pauli_evolve(
        self,
        op: PauliEvolveOp,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a Pauli evolution operation.

        Args:
            op (PauliEvolveOp): Pauli evolution operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: The inverse Pauli evolution operation, plus
                any needed angle-negation operation.
        """
        current_qubits = _as_value(
            _substitute_value(op.evolved_qubits, value_map),
            "pauli_evolve result",
        )
        observable = _as_value(
            _substitute_value(op.observable, value_map),
            "pauli_evolve observable",
        )
        angle_ops, inverse_evolution_time = self._negate_angle(op.gamma, value_map)
        result = current_qubits.next_version()
        inverse_op = PauliEvolveOp(
            operands=[current_qubits, observable, inverse_evolution_time],
            results=[result],
        )
        value_map[op.qubits.uuid] = result
        return [*angle_ops, inverse_op]

    def _invert_controlled_u(
        self,
        op: ControlledUOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a controlled-U operation.

        Args:
            op (ControlledUOperation): Controlled operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: The inverse controlled operation.

        Raises:
            NotImplementedError: If the controlled block is missing.
        """
        if op.block is None:
            raise NotImplementedError("inverse() cannot invert unresolved ControlledU.")
        inverse_block = self.invert_block(op.block)
        current_results = [
            _as_value(_substitute_value(result, value_map), "ControlledU result")
            for result in op.results
        ]
        new_results = [result.next_version() for result in current_results]
        mapped_params = [
            _as_value(_substitute_value(param, value_map), "ControlledU parameter")
            for param in op.param_operands
        ]
        power: int | Value = op.power
        if isinstance(power, Value):
            power = _as_value(_substitute_value(power, value_map), "ControlledU power")

        if isinstance(op, SymbolicControlledU):
            num_controls = _as_value(
                _substitute_value(op.num_controls, value_map),
                "ControlledU num_controls",
            )
            control_indices = (
                tuple(
                    _as_value(
                        _substitute_value(control_index, value_map),
                        "ControlledU control index",
                    )
                    for control_index in op.control_indices
                )
                if op.control_indices is not None
                else None
            )
            operands = [*current_results, *mapped_params]
            inverse_op = SymbolicControlledU(
                operands=operands,
                results=new_results,
                num_controls=num_controls,
                control_indices=control_indices,
                power=power,
                block=inverse_block,
                num_control_args=op.num_control_args,
            )
        elif isinstance(op, ConcreteControlledU):
            operands = [*current_results, *mapped_params]
            inverse_op = ConcreteControlledU(
                operands=operands,
                results=new_results,
                num_controls=op.num_controls,
                power=power,
                block=inverse_block,
            )
        else:
            raise NotImplementedError(f"inverse() cannot invert {type(op).__name__}.")

        self._update_quantum_value_map(
            value_map,
            op.control_operands + op.target_operands,
            new_results,
        )
        return [inverse_op]

    def _invert_call_block(
        self,
        op: CallBlockOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Inline the inverse of a nested QKernel call.

        Args:
            op (CallBlockOperation): Call operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: Inlined inverse operations.

        Raises:
            NotImplementedError: If the call has no block target.
        """
        if op.block is None:
            raise NotImplementedError(
                "inverse() cannot invert unresolved qkernel calls."
            )
        local_map = dict(value_map)
        for block_input, call_operand in zip(op.block.input_values, op.operands):
            resolved = _substitute_value(call_operand, value_map)
            local_map[block_input.uuid] = resolved
            if isinstance(block_input, ArrayValue) and isinstance(resolved, ArrayValue):
                for block_dim, arg_dim in zip(block_input.shape, resolved.shape):
                    local_map[block_dim.uuid] = _substitute_value(arg_dim, value_map)
        for block_output, call_result in zip(op.block.output_values, op.results):
            local_map[block_output.uuid] = _substitute_value(call_result, value_map)

        self._reject_unsupported_control_flow(op.block.operations)
        operations = self._invert_block_operations(op.block, local_map)
        for block_input, call_operand in zip(op.block.input_values, op.operands):
            if block_input.type.is_quantum():
                value_map[call_operand.uuid] = local_map[block_input.uuid]
        return operations

    def _invert_for(
        self,
        op: ForOperation,
        value_map: dict[str, ValueBase],
    ) -> list[Operation]:
        """Invert a compile-time range loop.

        Args:
            op (ForOperation): Loop operation to invert.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            list[Operation]: A reversed `ForOperation`, or an empty list for
                an empty forward range.

        Raises:
            NotImplementedError: If the bounds are symbolic or invalid.
        """
        start, stop, step = self._resolve_range_constants(op, value_map)
        sequence = list(range(start, stop, step))
        if not sequence:
            return []
        reverse_start = sequence[-1]
        reverse_stop = sequence[0] - step
        reverse_step = -step
        loop_var = Value(type=UIntType(), name=op.loop_var or "_inverse_loop_idx")
        body_map = dict(value_map)
        if op.loop_var_value is not None:
            body_map[op.loop_var_value.uuid] = loop_var
        inverse_body = self._invert_operations(op.operations, body_map)
        excluded_uuids = {op.loop_var_value.uuid} if op.loop_var_value else set()
        self._merge_loop_body_map(value_map, body_map, excluded_uuids)
        return [
            ForOperation(
                # Match control_flow._value_to_ir_value: Python range sentinels
                # such as -1 are represented with UIntType today.
                operands=[
                    _const_uint("inverse_loop_start", reverse_start),
                    _const_uint("inverse_loop_stop", reverse_stop),
                    _const_uint("inverse_loop_step", reverse_step),
                ],
                loop_var=op.loop_var,
                loop_var_value=loop_var,
                operations=inverse_body,
            )
        ]

    def _merge_loop_body_map(
        self,
        value_map: dict[str, ValueBase],
        body_map: dict[str, ValueBase],
        excluded_uuids: set[str],
    ) -> None:
        """Propagate loop-body substitutions back to the surrounding scope.

        Args:
            value_map (dict[str, ValueBase]): Surrounding UUID-keyed value
                map to update.
            body_map (dict[str, ValueBase]): Loop-body map after cloning or
                inverting the body.
            excluded_uuids (set[str]): Loop-local UUIDs that must not leak
                into the surrounding scope.

        Returns:
            None.
        """
        for uuid, value in body_map.items():
            if uuid not in excluded_uuids:
                value_map[uuid] = value

    def _update_quantum_value_map(
        self,
        value_map: dict[str, ValueBase],
        operands: Sequence[ValueBase],
        results: Sequence[ValueBase],
    ) -> None:
        """Map quantum operands to their current result values.

        Args:
            value_map (dict[str, ValueBase]): UUID-keyed current-value map
                to update.
            operands (Sequence[ValueBase]): Original quantum operands.
            results (Sequence[ValueBase]): Current result values replacing
                the operands.

        Returns:
            None.
        """
        for operand, result in zip(operands, results):
            value_map[operand.uuid] = result
            if isinstance(operand, ArrayValue) and isinstance(result, ArrayValue):
                for operand_dim, result_dim in zip(operand.shape, result.shape):
                    if operand_dim.uuid != result_dim.uuid:
                        value_map[operand_dim.uuid] = result_dim

    def _resolve_range_constants(
        self,
        op: ForOperation,
        value_map: dict[str, ValueBase],
    ) -> tuple[int, int, int]:
        """Resolve loop bounds to Python integers.

        Args:
            op (ForOperation): Loop whose bounds should be resolved.
            value_map (dict[str, ValueBase]): UUID-keyed current-value map.

        Returns:
            tuple[int, int, int]: Start, stop, and step.

        Raises:
            NotImplementedError: If a bound is symbolic or step is zero.
        """
        resolved: list[int] = []
        for bound_name, operand in zip(("start", "stop", "step"), op.operands):
            value = _as_value(
                _substitute_value(operand, value_map),
                f"ForOperation {bound_name}",
            )
            const = value.get_const()
            if const is None:
                raise NotImplementedError(
                    "inverse() only supports ForOperation with compile-time "
                    f"constant {bound_name} bounds."
                )
            resolved.append(int(const))
        if resolved[2] == 0:
            raise NotImplementedError("inverse() cannot invert a zero-step loop.")
        return resolved[0], resolved[1], resolved[2]


class InverseGate:
    """Callable wrapper that applies a QKernel's inverse.

    Args:
        qkernel (QKernel): Kernel whose inverse should be emitted.
    """

    def __init__(self, qkernel: QKernel) -> None:
        """Initialize the inverse wrapper.

        Args:
            qkernel (QKernel): Kernel whose inverse should be emitted.
        """
        self._qkernel = qkernel

    def _bind_arguments(self, *args: Any, **kwargs: Any) -> "BoundArguments":
        """Bind and literal-promote call arguments.

        Args:
            *args (Any): Positional arguments supplied by the caller.
            **kwargs (Any): Keyword arguments supplied by the caller.

        Returns:
            BoundArguments: Bound and default-filled argument mapping.

        Raises:
            TypeError: If any final argument is not a frontend `Handle`.
        """
        bound_args = self._qkernel.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in list(bound_args.arguments.items()):
            expected_type = self._qkernel.input_types.get(name)
            if expected_type is not None:
                bound_args.arguments[name] = _promote_literal_to_handle(
                    value,
                    expected_type,
                )
        for name, value in bound_args.arguments.items():
            if not isinstance(value, Handle):
                raise TypeError(
                    f"inverse(): argument {name!r} must be a Handle instance, "
                    f"got {type(value).__name__}."
                )
        return bound_args

    def _select_block(self, arguments: dict[str, Any]) -> Block:
        """Select a cached or call-time-specialized block.

        Args:
            arguments (dict[str, Any]): Bound call arguments.

        Returns:
            Block: Block whose operations should be inverted.
        """
        block_ir = None
        if not self._qkernel._specializing:
            spec = self._qkernel._extract_calltime_specialization(arguments)
            if spec is not None:
                sub_parameters, sub_bindings, sub_qubit_sizes = spec
                self._qkernel._specializing = True
                try:
                    block_ir = self._qkernel._build_specialized(
                        parameters=sub_parameters,
                        bindings=sub_bindings,
                        qubit_sizes=sub_qubit_sizes,
                    )
                finally:
                    self._qkernel._specializing = False
        if block_ir is None:
            block_ir = self._qkernel.block
        return block_ir

    def _prepare_inputs(
        self,
        block: Block,
        arguments: dict[str, Any],
    ) -> list[_InputBinding]:
        """Consume quantum arguments and pair them with block inputs.

        Args:
            block (Block): Selected block.
            arguments (dict[str, Any]): Bound call arguments.

        Returns:
            list[_InputBinding]: One binding per block input.
        """
        bindings: list[_InputBinding] = []
        for name, block_input in zip(block.label_args, block.input_values):
            handle = cast(Handle, arguments[name])
            active_handle = handle
            if handle._should_enforce_linear() and not isinstance(handle, VectorView):
                active_handle = handle.consume(
                    operation_name=f"Inverse[{self._qkernel.name}]"
                )
            _validate_input_shape(name, block_input, active_handle.value)
            bindings.append(
                _InputBinding(
                    name=name,
                    handle=handle,
                    active_handle=active_handle,
                    block_input=block_input,
                )
            )
        return bindings

    def _initial_value_map(
        self,
        bindings: list[_InputBinding],
    ) -> dict[str, ValueBase]:
        """Build the input-side value map for inverse expansion.

        Args:
            bindings (list[_InputBinding]): Prepared input bindings.

        Returns:
            dict[str, ValueBase]: UUID-keyed value mapping.
        """
        value_map: dict[str, ValueBase] = {}
        for binding in bindings:
            actual = binding.active_handle.value
            value_map[binding.block_input.uuid] = actual
            if isinstance(binding.block_input, ArrayValue) and isinstance(
                actual,
                ArrayValue,
            ):
                for block_dim, actual_dim in zip(
                    binding.block_input.shape,
                    actual.shape,
                ):
                    value_map[block_dim.uuid] = actual_dim
        return value_map

    def _wrap_quantum_result(
        self,
        binding: _InputBinding,
        value: ValueBase,
    ) -> Handle:
        """Wrap an inverse output value as a frontend handle.

        Args:
            binding (_InputBinding): Original input binding.
            value (ValueBase): Final inverse output value.

        Returns:
            Handle: Frontend handle carrying `value`.

        Raises:
            TypeError: If an array input maps to a scalar output or vice
                versa.
        """
        active = binding.active_handle
        if isinstance(active, VectorView):
            if not isinstance(value, ArrayValue):
                raise TypeError("inverse(): VectorView input produced scalar output.")
            new_view = VectorView._wrap_unregistered(
                parent=active._slice_parent,
                sliced_av=value,
                length=active._shape[0],
                start_uint=active._slice_start,
                step_uint=active._slice_step,
            )
            active._transfer_borrow_to(new_view, f"Inverse[{self._qkernel.name}]")
            return new_view
        if isinstance(active, ArrayBase):
            if not isinstance(value, ArrayValue):
                raise TypeError("inverse(): array input produced scalar output.")
            return type(active)._create_from_value(
                value=value,
                shape=active.shape,
                name=active.value.name,
            )
        if not isinstance(value, Value) or isinstance(value, ArrayValue):
            raise TypeError("inverse(): scalar input produced array output.")
        return type(active)(
            value=value,
            parent=active.parent,
            indices=active.indices,
            name=active.name,
        )

    def _can_emit_atomic_inverse(
        self,
        block: Block,
        bindings: list[_InputBinding],
    ) -> bool:
        """Return whether this inverse call can stay atomic until emit.

        Args:
            block (Block): Wrapped qkernel block.
            bindings (list[_InputBinding]): Prepared call-site bindings.

        Returns:
            bool: True when every quantum input has a statically known
                scalar qubit width and the inverse result order matches the
                wrapped input order.
        """
        quantum_inputs = [
            value for value in block.input_values if value.type.is_quantum()
        ]
        quantum_outputs = list(block.output_values)
        preserves_output_order = len(quantum_inputs) == len(quantum_outputs) and all(
            input_value.logical_id == output.logical_id
            for input_value, output in zip(quantum_inputs, quantum_outputs)
        )
        return preserves_output_order and all(
            _static_quantum_width(binding.active_handle.value) is not None
            for binding in bindings
            if binding.is_quantum
        )

    def _emit_atomic_inverse(
        self,
        block: Block,
        bindings: list[_InputBinding],
    ) -> Any:
        """Emit an inverse qkernel as an atomic inverse composite.

        Args:
            block (Block): Wrapped qkernel block.
            bindings (list[_InputBinding]): Prepared call-site bindings.

        Returns:
            Any: Quantum output handle, or a tuple of handles when the
                wrapped kernel has multiple quantum inputs.
        """
        shape_value_map: dict[str, ValueBase] = {}
        for binding in bindings:
            if isinstance(binding.block_input, ArrayValue) and isinstance(
                binding.active_handle.value,
                ArrayValue,
            ):
                for block_dim, actual_dim in zip(
                    binding.block_input.shape,
                    binding.active_handle.value.shape,
                ):
                    shape_value_map[block_dim.uuid] = actual_dim
        inverse_block = _BlockInverter().invert_block(block, shape_value_map)
        quantum_bindings = [binding for binding in bindings if binding.is_quantum]
        quantum_values = [
            _as_value(binding.active_handle.value, "inverse qkernel input")
            for binding in quantum_bindings
        ]
        # `InverseBlockOperation` stores the scalar backend width separately
        # from operand/results lists: a Vector[Qubit] contributes many scalar
        # qubits here but remains a single operand/result value.
        target_width = sum(
            width
            for value in quantum_values
            if (width := _static_quantum_width(value)) is not None
        )
        parameter_values = [
            binding.active_handle.value
            for binding in bindings
            if not binding.is_quantum
        ]
        result_values = [value.next_version() for value in quantum_values]
        op = InverseBlockOperation(
            operands=cast(list[Value], [*quantum_values, *parameter_values]),
            results=result_values,
            num_control_qubits=0,
            num_target_qubits=target_width,
            custom_name=f"{block.name}_inverse",
            source_block=block,
            implementation_block=inverse_block,
        )
        get_current_tracer().add_operation(op)

        outputs = [
            self._wrap_quantum_result(binding, value)
            for binding, value in zip(quantum_bindings, result_values)
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the inverse at the current trace site.

        Args:
            *args (Any): Positional arguments for the wrapped kernel.
            **kwargs (Any): Keyword arguments for the wrapped kernel.

        Returns:
            Any: Quantum output handle, or a tuple of handles when the
                wrapped kernel has multiple quantum inputs.
        """
        bound_args = self._bind_arguments(*args, **kwargs)
        block = self._select_block(bound_args.arguments)
        bindings = self._prepare_inputs(block, bound_args.arguments)
        if self._can_emit_atomic_inverse(block, bindings):
            return self._emit_atomic_inverse(block, bindings)

        value_map = self._initial_value_map(bindings)
        operations = _BlockInverter().invert_call_site(block, value_map)

        tracer = get_current_tracer()
        for op in operations:
            tracer.add_operation(op)

        outputs = [
            self._wrap_quantum_result(binding, value_map[binding.block_input.uuid])
            for binding in bindings
            if binding.is_quantum
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def _inverse_known_qft_target(target: Any) -> Any | None:
    """Return the direct QFT/IQFT function counterpart for known targets.

    Args:
        target (Any): Object supplied to `inverse`.

    Returns:
        Any | None: The opposite stdlib function, or `None` when `target`
            is not a known QFT/IQFT function.
    """
    from qamomile.circuit.stdlib.qft import iqft, qft

    if target is qft:
        return iqft
    if target is iqft:
        return qft
    return None


def _inverse_native_gate_target(target: Any) -> Any | None:
    """Return a native frontend inverse callable when `target` is known.

    Args:
        target (Any): Object supplied to `inverse`.

    Returns:
        Any | None: Callable native inverse wrapper, or None when `target` is
            not a recognized native gate callable.
    """
    from qamomile.circuit.frontend.operation import qubit_gates

    direct_map: dict[Callable[..., Any], Callable[..., Any]] = {
        qubit_gates.h: qubit_gates.h,
        qubit_gates.x: qubit_gates.x,
        qubit_gates.y: qubit_gates.y,
        qubit_gates.z: qubit_gates.z,
        qubit_gates.cx: qubit_gates.cx,
        qubit_gates.cz: qubit_gates.cz,
        qubit_gates.swap: qubit_gates.swap,
        qubit_gates.ccx: qubit_gates.ccx,
        qubit_gates.s: qubit_gates.sdg,
        qubit_gates.sdg: qubit_gates.s,
        qubit_gates.t: qubit_gates.tdg,
        qubit_gates.tdg: qubit_gates.t,
    }
    for forward_callable, inverse_callable in direct_map.items():
        if target is forward_callable:
            return inverse_callable

    rotation_map: dict[Callable[..., Any], str] = {
        qubit_gates.p: "theta",
        qubit_gates.rx: "angle",
        qubit_gates.ry: "angle",
        qubit_gates.rz: "angle",
        qubit_gates.cp: "theta",
        qubit_gates.rzz: "angle",
    }
    for forward_callable, angle_param in rotation_map.items():
        if target is forward_callable:
            return _InverseRotationCallable(
                rotation_callable=forward_callable,
                angle_param=angle_param,
            )
    return None


def inverse(target: QKernel | Callable[..., Any]) -> Any:
    """Create an inverse operation wrapper.

    Native Qamomile gate functions are first synthesized into tiny
    `QKernel` objects, then inverted with the same block walker used for
    user-defined kernels. Known QFT/IQFT functions map directly to their
    counterpart so backend-native composite emission remains available.

    Args:
        target (QKernel | Callable[..., Any]): Native gate function,
            `QKernel`, or supported stdlib function to invert.

    Returns:
        Any: A callable inverse wrapper, or the opposite QFT/IQFT function.

    Raises:
        TypeError: If `target` cannot be interpreted as a gate-like
            callable, or if a `CompositeGate` instance is passed directly.
        NotImplementedError: If an inverted kernel uses unsupported
            operations such as `if`/`while`/`for items` control flow,
            `QInit`, or a `ForOperation` whose bounds are not compile-time
            constants when the inverse wrapper is traced.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def layer(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
        ...     q = qmc.h(q)
        ...     q = qmc.rz(q, angle)
        ...     return q
        >>> @qmc.qkernel
        ... def circuit(angle: qmc.Float) -> qmc.Qubit:
        ...     q = qmc.qubit("q")
        ...     q = layer(q, angle)
        ...     q = qmc.inverse(layer)(q, angle)
        ...     return q
    """
    known_inverse = _inverse_known_qft_target(target)
    if known_inverse is not None:
        return known_inverse
    native_inverse = _inverse_native_gate_target(target)
    if native_inverse is not None:
        return native_inverse
    if isinstance(target, CompositeGate):
        raise TypeError(
            "inverse() does not support direct CompositeGate instances. "
            "Use qmc.inverse(qmc.qft) or qmc.inverse(qmc.iqft) for QFT/IQFT, "
            "or invert a QKernel that contains the composite gate."
        )
    qkernel = _qkernel_for_callable(target, caller="inverse")
    return InverseGate(qkernel)
