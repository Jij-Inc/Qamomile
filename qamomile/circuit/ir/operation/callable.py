"""Callable operation model for composite and oracle calls."""

from __future__ import annotations

import dataclasses
import enum
import uuid as uuid_module
from collections.abc import Mapping, Sequence
from typing import Any, cast

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    remap_value_metadata_references,
)

from .control_value import normalize_control_value
from .operation import Operation, OperationKind, ParamHint, Signature


class CallTransform(enum.Enum):
    """Describe the requested transform of a callable implementation."""

    DIRECT = "direct"
    INVERSE = "inverse"
    CONTROLLED = "controlled"


class CallPolicy(enum.Enum):
    """Describe the default lowering policy for a callable call."""

    INLINE = "inline"
    PRESERVE_BOX = "preserve_box"
    NATIVE_FIRST = "native_first"


def _policy_from_attrs(attrs: dict[str, Any]) -> CallPolicy:
    """Infer a default policy from serialized callable attributes.

    Args:
        attrs (dict[str, Any]): Invocation attributes.

    Returns:
        CallPolicy: Explicit ``default_policy`` when present; otherwise
            ``INLINE`` for qkernel calls and ``PRESERVE_BOX`` for boxed calls.
    """
    raw = attrs.get("default_policy")
    if isinstance(raw, str):
        try:
            return CallPolicy[raw]
        except KeyError:
            try:
                return CallPolicy(raw)
            except ValueError:
                pass
    if attrs.get("kind") in {"composite", "oracle"}:
        return CallPolicy.PRESERVE_BOX
    return CallPolicy.INLINE


class CompositeGateType(enum.Enum):
    """Classify standard boxed quantum callables."""

    QPE = "qpe"
    QFT = "qft"
    IQFT = "iqft"
    CUSTOM = "custom"


@dataclasses.dataclass(frozen=True)
class CallableRef:
    """Identify a callable independently of its Python object.

    Args:
        namespace (str): Stable namespace such as ``"qamomile.stdlib"`` or
            ``"user"``.
        name (str): Stable callable name within the namespace.
        version (str): Schema or behavior version for the callable.
    """

    namespace: str
    name: str
    version: str = "1"


@dataclasses.dataclass
class CallableBodyRef:
    """Reference a callable body that can be materialized later.

    Args:
        ref (CallableRef): Callable whose standard body is referenced.
        kind (str): Body-reference kind, such as ``"standard"`` or
            ``"symbolic_vector"``. Defaults to ``"standard"``.
        attrs (dict[str, Any]): Serializer-friendly body-materialization
            attributes. Defaults to an empty dict.
    """

    ref: CallableRef
    kind: str = "standard"
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CallableImplementation:
    """Describe one implementation candidate for a callable.

    Args:
        transform (CallTransform): Transform this implementation realizes.
        backend (str | None): Backend name for native implementations.
        strategy (str | None): Strategy name such as ``"standard"``.
        body (Block | None): IR implementation body. A transform-specific body
            realizes that transform completely; a controlled body therefore
            includes control operands in its signature.
        body_ref (CallableBodyRef | None): Reference to a body that should be
            materialized by a later resolver. Defaults to ``None``.
        emitter (Any): Backend-native emitter object.
        attrs (dict[str, Any]): Serializer-friendly implementation metadata.
    """

    transform: CallTransform = CallTransform.DIRECT
    backend: str | None = None
    strategy: str | None = None
    body: Block | None = None
    body_ref: CallableBodyRef | None = None
    emitter: Any = None
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CallableDef:
    """Describe a compiler-facing callable definition.

    Args:
        ref (CallableRef): Stable callable identity.
        signature (Signature | None): Optional callable signature.
        body (Block | None): Standard IR body, or ``None`` for opaque calls.
        body_ref (CallableBodyRef | None): Reference to a standard body that is
            intentionally deferred. Defaults to ``None``.
        implementations (list[CallableImplementation]): Alternative native or
            strategy-specific implementations.
        opaque_cost (Any | None): Explicit cost contract for a bodyless
            callable. Body-backed callables must leave this as ``None``.
        default_policy (CallPolicy): Default call lowering policy.
        attrs (dict[str, Any]): Serializer-friendly definition metadata.
    """

    ref: CallableRef
    signature: Signature | None = None
    body: Block | None = None
    body_ref: CallableBodyRef | None = None
    implementations: list[CallableImplementation] = dataclasses.field(
        default_factory=list
    )
    opaque_cost: Any | None = None
    default_policy: CallPolicy = CallPolicy.INLINE
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def implementation_for(
        self,
        *,
        transform: CallTransform = CallTransform.DIRECT,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> CallableImplementation | None:
        """Return the best matching implementation candidate.

        Args:
            transform (CallTransform): Requested call transform.
            backend (str | None): Requested backend name.
            strategy (str | None): Requested strategy name.

        Returns:
            CallableImplementation | None: Matching implementation, if any.
        """
        candidates = [
            impl
            for impl in self.implementations
            if impl.transform == transform
            and (
                (backend is None and impl.backend is None)
                or (backend is not None and impl.backend in (None, backend))
            )
            and (
                (strategy is None and impl.strategy is None)
                or (strategy is not None and impl.strategy in (None, strategy))
            )
        ]
        if not candidates:
            return None

        def score(impl: CallableImplementation) -> int:
            """Score exact backend and strategy matches above generic ones.

            Args:
                impl (CallableImplementation): Candidate implementation.

            Returns:
                int: Match score.
            """
            return int(impl.backend == backend) + int(impl.strategy == strategy)

        return max(candidates, key=score)


def signature_from_values(
    operands: Sequence[ValueLike],
    results: Sequence[ValueLike],
    *,
    operand_names: Sequence[str] | None = None,
    result_names: Sequence[str] | None = None,
) -> Signature:
    """Build a callable signature from concrete operand and result values.

    Args:
        operands (Sequence[ValueLike]): Values consumed by the callable.
        results (Sequence[ValueLike]): Values produced by the callable.
        operand_names (Sequence[str] | None): Optional names for operands.
            Missing entries fall back to ``arg_<index>``. Defaults to ``None``.
        result_names (Sequence[str] | None): Optional names for results.
            Missing entries fall back to ``result_<index>``. Defaults to
            ``None``.

    Returns:
        Signature: IR signature with typed parameter hints.
    """

    def name_at(names: Sequence[str] | None, index: int, fallback: str) -> str:
        """Return a provided name or a positional fallback.

        Args:
            names (Sequence[str] | None): Optional candidate names.
            index (int): Name index to read.
            fallback (str): Name to use when no candidate exists.

        Returns:
            str: Selected name.
        """
        if names is not None and index < len(names) and names[index]:
            return names[index]
        return fallback

    return Signature(
        operands=[
            ParamHint(name=name_at(operand_names, i, f"arg_{i}"), type=operand.type)
            for i, operand in enumerate(operands)
        ],
        results=[
            ParamHint(name=name_at(result_names, i, f"result_{i}"), type=result.type)
            for i, result in enumerate(results)
        ],
    )


def signature_from_block(block: Block) -> Signature:
    """Build a callable signature from a traced implementation block.

    Args:
        block (Block): Callable implementation block whose inputs and outputs
            define the signature.

    Returns:
        Signature: IR signature using ``Block.label_args`` and
        ``Block.output_names`` when available.
    """
    return signature_from_values(
        block.input_values,
        block.output_values,
        operand_names=block.label_args,
        result_names=block.output_names,
    )


class _CallResultMaterializer:
    """Create caller-local values for a block invocation's results.

    The materializer first reserves UUID and logical-ID mappings for the
    complete callee value graph, then rebuilds output values. Reserving first
    lets metadata on one output safely reference values owned by another
    output or by an operation visited later in the block.

    Args:
        block (Block): Callee block whose outputs are being materialized.
        actuals (list[ValueLike]): Caller-side arguments in formal-input order.
    """

    def __init__(self, block: Block, actuals: list[ValueLike]) -> None:
        """Initialize formal bindings and reserve the callee value graph.

        Args:
            block (Block): Callee block whose values define the source graph.
            actuals (list[ValueLike]): Caller-side arguments aligned with
                ``block.input_values``.

        Raises:
            ValueError: If the formal and actual argument counts differ.
        """
        if len(block.input_values) != len(actuals):
            raise ValueError(
                f"Block '{block.name}' expects {len(block.input_values)} "
                f"arguments, got {len(actuals)}"
            )

        self._input_by_uuid: dict[str, ValueLike] = {}
        self._input_by_logical_id: dict[str, ValueLike] = {}
        self._uuid_remap: dict[str, str] = {}
        self._logical_id_remap: dict[str, str] = {}
        self._reserved_uuids: set[str] = set()
        self._materialized: dict[str, ValueLike] = {}
        self._pass_through: dict[str, ValueLike] = {}

        for formal, actual in zip(block.input_values, actuals, strict=True):
            self._bind_input(formal, actual)

        self._reserve_block(block)
        for output in block.output_values:
            self._reserve_output_pass_throughs(output)

    def materialize(self, value: ValueLike) -> ValueLike:
        """Return a caller-local materialization of one callee output value.

        Args:
            value (ValueLike): Callee-side output value.

        Returns:
            ValueLike: Caller-local value whose graph and metadata contain no
                callee-local UUIDs when those UUIDs belong to the block.
        """
        if value.uuid in self._pass_through:
            return self._pass_through[value.uuid]
        if value.uuid in self._materialized:
            return self._materialized[value.uuid]

        actual = self._bound_actual(value)
        if actual is not None:
            result = cast(ValueLike, actual.next_version())
            self._pass_through[value.uuid] = result
            self._uuid_remap[value.uuid] = result.uuid
            self._logical_id_remap[value.logical_id] = result.logical_id
            return result

        self._reserve_value(value)
        new_uuid = self._uuid_remap[value.uuid]
        new_logical_id = self._logical_id_remap[value.logical_id]
        metadata = remap_value_metadata_references(
            value.metadata,
            self._remap_uuid,
            self._remap_logical_id,
        )

        if isinstance(value, TupleValue):
            result = dataclasses.replace(
                value,
                elements=tuple(self.materialize(element) for element in value.elements),
                metadata=metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )
        elif isinstance(value, DictValue):
            entries: list[tuple[TupleValue | Value, Value]] = []
            for key, entry_value in value.entries:
                entries.append(
                    (
                        cast(TupleValue | Value, self.materialize(key)),
                        cast(Value, self.materialize(entry_value)),
                    )
                )
            result = dataclasses.replace(
                value,
                entries=tuple(entries),
                metadata=metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )
        elif isinstance(value, ArrayValue):
            result = dataclasses.replace(
                value,
                parent_array=(
                    cast(ArrayValue, self.materialize(value.parent_array))
                    if value.parent_array is not None
                    else None
                ),
                element_indices=tuple(
                    cast(Value, self.materialize(index))
                    for index in value.element_indices
                ),
                shape=tuple(
                    cast(Value, self.materialize(dimension))
                    for dimension in value.shape
                ),
                slice_of=(
                    cast(ArrayValue, self.materialize(value.slice_of))
                    if value.slice_of is not None
                    else None
                ),
                slice_start=(
                    cast(Value, self.materialize(value.slice_start))
                    if value.slice_start is not None
                    else None
                ),
                slice_step=(
                    cast(Value, self.materialize(value.slice_step))
                    if value.slice_step is not None
                    else None
                ),
                metadata=metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )
        else:
            result = dataclasses.replace(
                value,
                parent_array=(
                    cast(ArrayValue, self.materialize(value.parent_array))
                    if value.parent_array is not None
                    else None
                ),
                element_indices=tuple(
                    cast(Value, self.materialize(index))
                    for index in value.element_indices
                ),
                metadata=metadata,
                uuid=new_uuid,
                logical_id=new_logical_id,
            )

        self._materialized[value.uuid] = result
        return result

    def _bind_input(
        self,
        formal: ValueLike,
        actual: ValueLike,
        seen: set[str] | None = None,
    ) -> None:
        """Bind one formal value graph to its caller-side value graph.

        Dict containers are intentionally bound at the root only. A symbolic
        formal dict may have no entries while its bound caller value does, so
        positional entry zipping would invent an invalid correspondence.

        Args:
            formal (ValueLike): Callee-side formal value.
            actual (ValueLike): Caller-side actual value.
            seen (set[str] | None): Formal UUIDs already visited. Defaults to
                ``None``.
        """
        if seen is None:
            seen = set()
        if formal.uuid in seen:
            return
        seen.add(formal.uuid)

        self._input_by_uuid[formal.uuid] = actual
        self._input_by_logical_id[formal.logical_id] = actual
        self._uuid_remap[formal.uuid] = actual.uuid
        self._logical_id_remap[formal.logical_id] = actual.logical_id

        if isinstance(formal, TupleValue) and isinstance(actual, TupleValue):
            if len(formal.elements) == len(actual.elements):
                for formal_element, actual_element in zip(
                    formal.elements,
                    actual.elements,
                    strict=True,
                ):
                    self._bind_input(formal_element, actual_element, seen)
            return
        if isinstance(formal, DictValue):
            return
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            for formal_dimension, actual_dimension in zip(
                formal.shape,
                actual.shape,
            ):
                self._bind_input(formal_dimension, actual_dimension, seen)
            if formal.slice_of is not None and actual.slice_of is not None:
                self._bind_input(formal.slice_of, actual.slice_of, seen)
            if formal.slice_start is not None and actual.slice_start is not None:
                self._bind_input(formal.slice_start, actual.slice_start, seen)
            if formal.slice_step is not None and actual.slice_step is not None:
                self._bind_input(formal.slice_step, actual.slice_step, seen)
        if isinstance(formal, Value) and isinstance(actual, Value):
            if formal.parent_array is not None and actual.parent_array is not None:
                self._bind_input(formal.parent_array, actual.parent_array, seen)
            for formal_index, actual_index in zip(
                formal.element_indices,
                actual.element_indices,
            ):
                self._bind_input(formal_index, actual_index, seen)

    def _reserve_block(self, block: Block) -> None:
        """Reserve identifiers for every concrete value owned by a block.

        Args:
            block (Block): Callee block to scan recursively.
        """
        for value in block.input_values:
            self._reserve_value(value)
        for value in block.output_values:
            self._reserve_value(value)
        for value in block.parameters.values():
            if isinstance(value, ValueBase):
                self._reserve_value(cast(ValueLike, value))
        self._reserve_operations(block.operations)

    def _reserve_operations(self, operations: list[Operation]) -> None:
        """Reserve values owned or referenced by an operation tree.

        Args:
            operations (list[Operation]): Operations to scan recursively.
        """
        for operation in operations:
            for value in operation.all_input_values():
                if isinstance(value, ValueBase):
                    self._reserve_value(cast(ValueLike, value))
            for value in operation.results:
                if isinstance(value, ValueBase):
                    self._reserve_value(cast(ValueLike, value))
            nested_op_lists = getattr(operation, "nested_op_lists", None)
            if callable(nested_op_lists):
                for nested in nested_op_lists():
                    self._reserve_operations(nested)

    def _reserve_value(self, value: ValueLike) -> None:
        """Reserve identifiers for one value and its object references.

        Args:
            value (ValueLike): Value graph node to reserve.
        """
        if value.uuid in self._reserved_uuids:
            return
        self._reserved_uuids.add(value.uuid)

        actual = self._bound_actual(value)
        if actual is not None:
            self._uuid_remap[value.uuid] = actual.uuid
            self._logical_id_remap[value.logical_id] = actual.logical_id
            return

        self._uuid_remap.setdefault(value.uuid, str(uuid_module.uuid4()))
        self._logical_id_remap.setdefault(
            value.logical_id,
            str(uuid_module.uuid4()),
        )
        for child in self._value_children(value):
            self._reserve_value(child)

    def _reserve_output_pass_throughs(self, value: ValueLike) -> None:
        """Reserve next-version placeholders for input-derived output nodes.

        Args:
            value (ValueLike): Output graph node to inspect.
        """
        actual = self._bound_actual(value)
        if actual is not None:
            result = cast(ValueLike, actual.next_version())
            self._pass_through[value.uuid] = result
            self._uuid_remap[value.uuid] = result.uuid
            self._logical_id_remap[value.logical_id] = result.logical_id
            return
        for child in self._value_children(value):
            self._reserve_output_pass_throughs(child)

    @staticmethod
    def _value_children(value: ValueLike) -> tuple[ValueLike, ...]:
        """Return object-linked child values for one graph node.

        Args:
            value (ValueLike): Value graph node.

        Returns:
            tuple[ValueLike, ...]: Child values referenced by dataclass fields.
        """
        if isinstance(value, TupleValue):
            return value.elements
        if isinstance(value, DictValue):
            return tuple(
                child
                for key, entry_value in value.entries
                for child in (key, entry_value)
            )

        children: list[ValueLike] = []
        if value.parent_array is not None:
            children.append(value.parent_array)
        children.extend(value.element_indices)
        if isinstance(value, ArrayValue):
            children.extend(value.shape)
            if value.slice_of is not None:
                children.append(value.slice_of)
            if value.slice_start is not None:
                children.append(value.slice_start)
            if value.slice_step is not None:
                children.append(value.slice_step)
        return tuple(children)

    def _bound_actual(self, value: ValueLike) -> ValueLike | None:
        """Return the caller value corresponding to an input-derived value.

        Args:
            value (ValueLike): Callee-side value.

        Returns:
            ValueLike | None: Bound caller value, or ``None``.
        """
        return self._input_by_uuid.get(
            value.uuid,
            self._input_by_logical_id.get(value.logical_id),
        )

    def _remap_uuid(self, uuid: str) -> str:
        """Return the caller-local UUID for a callee UUID reference.

        Args:
            uuid (str): Callee UUID.

        Returns:
            str: Caller-local UUID when reserved, otherwise ``uuid``.
        """
        return self._uuid_remap.get(uuid, uuid)

    def _remap_logical_id(self, logical_id: str) -> str:
        """Return the caller-local logical ID for a callee reference.

        Args:
            logical_id (str): Callee logical ID.

        Returns:
            str: Caller-local logical ID when reserved, otherwise
                ``logical_id``.
        """
        return self._logical_id_remap.get(logical_id, logical_id)


def block_call_operands_and_results(
    block: Block,
    inputs_map: Mapping[str, ValueLike],
) -> tuple[list[ValueLike], list[ValueLike]]:
    """Materialize one block invocation's operands and results.

    Args:
        block (Block): Callee block.
        inputs_map (Mapping[str, ValueLike]): Caller values keyed by formal
            label.

    Returns:
        tuple[list[ValueLike], list[ValueLike]]: Ordered caller operands and
        caller-local result values.

    Raises:
        KeyError: If a formal label is missing from ``inputs_map``.
        ValueError: If the resulting argument count does not match the block.
    """
    inputs = [inputs_map[label] for label in block.label_args]
    materializer = _CallResultMaterializer(block, inputs)
    return inputs, [materializer.materialize(output) for output in block.output_values]


@dataclasses.dataclass(init=False)
class InvokeOperation(Operation):
    """Represent a composite, stdlib, or oracle call.

    Args:
        operands (list[ValueLike]): Input values consumed by the call.
        results (list[ValueLike]): Output values produced by the call.
        target (CallableRef): Callable identity.
        transform (CallTransform): Direct, inverse, or controlled invocation.
        attrs (dict[str, Any]): Compile-time attributes for strategy, arity,
            and resource/lowering decisions. ``control_value`` is reserved for
            a controlled invocation's LSB-first activation value. Values must
            be serializer-friendly.
        definition (CallableDef | None): Optional callable definition.
    """

    target: CallableRef = dataclasses.field(
        default_factory=lambda: CallableRef(namespace="user", name="anonymous")
    )
    transform: CallTransform = CallTransform.DIRECT
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    definition: CallableDef | None = None

    def __init__(
        self,
        operands: Sequence[ValueLike] | None = None,
        results: Sequence[ValueLike] | None = None,
        *,
        target: CallableRef | None = None,
        transform: CallTransform = CallTransform.DIRECT,
        attrs: dict[str, Any] | None = None,
        definition: CallableDef | None = None,
    ) -> None:
        """Initialize an invocation operation.

        Args:
            operands (Sequence[ValueLike] | None): Input values consumed by the call.
                Defaults to ``None``, meaning no operands.
            results (Sequence[ValueLike] | None): Output values produced by the call.
                Defaults to ``None``, meaning no results.
            target (CallableRef | None): Callable identity. Defaults to an
                anonymous user callable when omitted.
            transform (CallTransform): Requested call transform. Defaults to
                ``CallTransform.DIRECT``.
            attrs (dict[str, Any] | None): Serializer-friendly call
                attributes. Defaults to an empty dict.
            definition (CallableDef | None): Callable definition. Defaults to
                ``None``, in which case one is created from ``target``.

        Raises:
            TypeError: If a controlled invocation's ``control_value`` is not
                a Python ``int`` or ``None``.
            ValueError: If ``control_value`` is used on a non-controlled call
                or does not fit the controlled invocation's width.
        """
        self.operands = cast(
            list[Value],
            list(operands) if operands is not None else [],
        )
        self.results = cast(
            list[Value],
            list(results) if results is not None else [],
        )
        self.target = (
            target
            if target is not None
            else CallableRef(namespace="user", name="anonymous")
        )
        self.transform = transform
        self.attrs = dict(attrs) if attrs is not None else {}
        raw_control_value = self.attrs.pop("control_value", None)
        if raw_control_value is not None:
            if self.transform is not CallTransform.CONTROLLED:
                raise ValueError(
                    "control_value is only valid for a controlled invocation."
                )
            normalized_control_value = normalize_control_value(
                raw_control_value,
                self.num_control_qubits,
            )
            if normalized_control_value is not None:
                self.attrs["control_value"] = normalized_control_value
        self.definition = definition
        self._ensure_definition()

    def _ensure_definition(self) -> None:
        """Ensure the invocation has a compiler-facing callable definition."""
        if self.definition is None:
            self.definition = CallableDef(
                ref=self.target,
                signature=(
                    signature_from_values(self.operands, self.results)
                    if self.operands or self.results
                    else None
                ),
                default_policy=_policy_from_attrs(self.attrs),
                attrs=dict(self.attrs),
            )
        else:
            self.target = self.definition.ref

    @property
    def body(self) -> Block | None:
        """Return the callable's default body from its definition.

        Returns:
            Block | None: The direct-call body stored on ``CallableDef``, or
            ``None`` for opaque callables.
        """
        if self.definition is None:
            return None
        return self.definition.body

    @body.setter
    def body(self, value: Block | None) -> None:
        """Set the callable's default body on its definition.

        Args:
            value (Block | None): Replacement default callable body.
        """
        if self.definition is None:
            self.definition = CallableDef(
                ref=self.target,
                signature=(
                    signature_from_values(self.operands, self.results)
                    if self.operands or self.results
                    else None
                ),
                default_policy=_policy_from_attrs(self.attrs),
                attrs=dict(self.attrs),
            )
        self.definition.body = value

    @property
    def body_ref(self) -> CallableBodyRef | None:
        """Return the callable's deferred body reference.

        Returns:
            CallableBodyRef | None: Selected implementation body reference, the
            definition-level body reference, or ``None`` when no deferred body is
            available.
        """
        impl = self.implementation_for()
        if impl is not None and impl.body_ref is not None:
            return impl.body_ref
        if self.definition is None:
            return None
        return self.definition.body_ref

    @property
    def name(self) -> str:
        """Return the display name for this invocation.

        Returns:
            str: The callable name, optionally prefixed for transforms.
        """
        if self.transform == CallTransform.INVERSE:
            return f"{self.target.name}†"
        return self.target.name

    @property
    def num_control_qubits(self) -> int:
        """Return the number of leading control-qubit operands.

        Returns:
            int: Control arity recorded in ``attrs``. Defaults to ``0``.
        """
        return int(self.attrs.get("num_control_qubits", 0))

    @property
    def control_value(self) -> int | None:
        """Return the controlled invocation's activation value.

        Returns:
            int | None: LSB-first activation value, or ``None`` for the
            ordinary all-ones control state.
        """
        return cast(int | None, self.attrs.get("control_value"))

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target-qubit operands.

        Returns:
            int: Target arity recorded in ``attrs``. Defaults to the operand
            count of quantum operands after controls.
        """
        default = sum(
            operand.type.is_quantum()
            for operand in self.operands[self.num_control_qubits :]
        )
        return int(self.attrs.get("num_target_qubits", default))

    @property
    def control_qubits(self) -> list["Value"]:
        """Return the control-qubit operands.

        Returns:
            list[Value]: Leading control operands.
        """
        return list(self.operands[: self.num_control_qubits])

    @property
    def target_qubits(self) -> list["Value"]:
        """Return the target-qubit operands.

        Returns:
            list[Value]: Quantum target operands after any controls, preserving
            their relative order even when classical parameters are interleaved
            in the callable signature. A vector target counts as one operand even when
            ``num_target_qubits`` records its scalar backend width.
        """
        start = self.num_control_qubits
        return [
            operand for operand in self.operands[start:] if operand.type.is_quantum()
        ]

    @property
    def parameters(self) -> list["Value"]:
        """Return non-qubit parameter operands.

        Returns:
            list[Value]: Classical/object operands after the control prefix,
            preserving their relative declaration order.
        """
        start = self.num_control_qubits
        return [
            operand
            for operand in self.operands[start:]
            if operand.type.is_classical() or operand.type.is_object()
        ]

    @property
    def gate_type(self) -> CompositeGateType:
        """Return the standard composite classification for this invocation.

        Returns:
            CompositeGateType: Type encoded in ``attrs["gate_type"]``. Unknown
            values map to ``CompositeGateType.CUSTOM``.
        """
        raw = str(self.attrs.get("gate_type", CompositeGateType.CUSTOM.name))
        try:
            return CompositeGateType[raw]
        except KeyError:
            return CompositeGateType.CUSTOM

    @property
    def custom_name(self) -> str:
        """Return the display name for custom callable boxes.

        Returns:
            str: ``attrs["custom_name"]`` when present, else target name.
        """
        return str(self.attrs.get("custom_name", self.target.name))

    @property
    def strategy_name(self) -> str | None:
        """Return the selected lowering/resource strategy name.

        Returns:
            str | None: Strategy name stored in attrs, or ``None``.
        """
        value = self.attrs.get("strategy_name")
        if value is None:
            return None
        return str(value)

    @property
    def default_policy(self) -> CallPolicy:
        """Return the callable's default lowering policy.

        Returns:
            CallPolicy: Policy from the callable definition, or ``INLINE``.
        """
        if self.definition is None:
            return CallPolicy.INLINE
        return self.definition.default_policy

    def implementation_for(
        self,
        *,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> CallableImplementation | None:
        """Return the selected implementation for this invocation.

        Args:
            backend (str | None): Backend name to match. Defaults to ``None``,
                which only selects backend-generic implementations.
            strategy (str | None): Strategy name to match. Defaults to
                ``None``, meaning the invocation's ``strategy_name`` attribute
                is used.

        Returns:
            CallableImplementation | None: Matching implementation candidate,
            or ``None`` when the callable definition has no match.
        """
        if self.definition is None:
            return None
        requested_strategy = self.strategy_name if strategy is None else strategy
        return self.definition.implementation_for(
            transform=self.transform,
            backend=backend,
            strategy=requested_strategy,
        )

    def effective_body(
        self,
        *,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> Block | None:
        """Return the implementation body selected for this invocation.

        Args:
            backend (str | None): Backend name to match. Defaults to ``None``.
            strategy (str | None): Strategy name to match. Defaults to the
                invocation's ``strategy_name`` attribute.

        Returns:
            Block | None: Selected implementation body, or the callable's
            default body when no transform-specific implementation exists.
            A compiler may synthesize inverse or controlled behavior from this
            fallback body.
        """
        impl = self.implementation_for(backend=backend, strategy=strategy)
        if impl is not None and impl.body is not None:
            return impl.body
        return self.body

    @property
    def signature(self) -> Signature:
        """Return the operation signature.

        Returns:
            Signature: Callable definition signature when available,
            otherwise best-effort operand and result hints from concrete
            values.
        """
        if self.definition is not None and self.definition.signature is not None:
            return self.definition.signature
        return signature_from_values(self.operands, self.results)

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind.

        Returns:
            OperationKind: ``QUANTUM`` because callables in this model consume
            or produce quantum values.
        """
        return OperationKind.QUANTUM
