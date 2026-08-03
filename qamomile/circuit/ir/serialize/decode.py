"""Private graph record → semantic IR block decoder.

Reconstructs a ``Block`` from the graph envelope produced by
:mod:`qamomile.circuit.ir.serialize.encode`. The decoder NEVER
performs dynamic class resolution: every ``$type`` tag is routed
through a hard-coded factory table, and unknown tags raise ``ValueError``.
This closed dispatch is the load-bearing security invariant for protobuf
deserialization.

Values are materialized lazily via depth-first recursion so any
referenced ``parent_array`` / ``element_indices`` / shape Value is
instantiated before the Value that points at it. A cycle (defensive;
not produced by the canonical encoder) raises ``ValueError``.
"""

from __future__ import annotations

from typing import Any, Callable, cast

from qamomile._utils import is_plain_int
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    CallableBodyRef,
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    GateOperationType,
    GlobalPhaseOperation,
    InverseBlockOperation,
    InvokeOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    Operation,
    ProjectOperation,
    ResetOperation,
    ReturnOperation,
    SelectOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    RegionArg,
    WhileOperation,
    validate_region_args,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    ParamHint,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.static_binding import StaticBindingField, StaticBindingSlot
from qamomile.circuit.ir.types.hamiltonian import ObservableType
from qamomile.circuit.ir.types.primitives import (
    BitType,
    BlockType,
    DictType,
    FloatType,
    QubitType,
    TupleType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.types.q_register import QFixedType, QUIntType
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    CastMetadata,
    DictRuntimeMetadata,
    DictValue,
    QFixedMetadata,
    ScalarMetadata,
    TupleValue,
    Value,
    ValueBase,
    ValueLike,
    ValueMetadata,
    _freeze_data,
)

from .hamiltonian_io import dict_to_hamiltonian, is_hamiltonian_wrapper
from .numpy_io import (
    dict_to_array,
    dict_to_scalar,
    is_array_wrapper,
    is_scalar_wrapper,
)

# ---------------------------------------------------------------------------
# Decoding context: lazy Value materialization
# ---------------------------------------------------------------------------


class _DecodeContext:
    """Materialize module-wide Value and CallableDef registries.

    Holds the value-table dicts keyed by UUID so the recursive
    materializer can resolve cross-references depth-first.
    """

    def __init__(
        self,
        value_table: list[dict[str, Any]],
        callable_table: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a decode context.

        Args:
            value_table (list[dict[str, Any]]): The list of Value
                dicts from the block envelope. Each entry must have a
                ``uuid`` field; duplicates are an error.
            callable_table (list[dict[str, Any]] | None): Callable definitions
                keyed by module-local IDs. Defaults to an empty registry.

        Raises:
            ValueError: If a value-table dict lacks a ``uuid`` or if
                two entries share the same UUID, or if callable IDs are
                malformed or duplicated.
        """
        if callable_table is None:
            callable_table = []
        self._by_uuid: dict[str, dict[str, Any]] = {}
        for entry in value_table:
            uuid = entry.get("uuid")
            if not isinstance(uuid, str):
                raise ValueError("value_table entry is missing a 'uuid' string")
            if uuid in self._by_uuid:
                raise ValueError(f"duplicate UUID in value_table: {uuid!r}")
            self._by_uuid[uuid] = entry
        self._built: dict[str, ValueBase] = {}
        self._building: set[str] = set()
        self._blocks: list[Block] = []
        self._definition_entries: dict[str, dict[str, Any]] = {}
        self._definitions: dict[str, CallableDef] = {}
        for entry in callable_table:
            if not isinstance(entry, dict):
                raise ValueError("callable_table entries must be dicts")
            definition_id = entry.get("id")
            definition_payload = entry.get("definition")
            if not isinstance(definition_id, str):
                raise ValueError("callable_table entry is missing a string 'id'")
            if definition_id in self._definition_entries:
                raise ValueError(f"duplicate callable definition id {definition_id!r}")
            if not isinstance(definition_payload, dict):
                raise ValueError(
                    f"callable_table entry {definition_id!r} is missing a "
                    "'definition' dict"
                )
            ref = _decode_callable_ref(definition_payload.get("ref"))
            self._definition_entries[definition_id] = definition_payload
            self._definitions[definition_id] = CallableDef(ref=ref)

    def register_block(self, block: Block) -> Block:
        """Register a decoded block for post-link metadata refresh.

        Args:
            block (Block): Newly decoded semantic block.

        Returns:
            Block: The same block for convenient decoder composition.
        """
        self._blocks.append(block)
        return block

    def refresh_block_effects(self) -> None:
        """Refresh derived effects after callable placeholders are linked.

        Callable bodies are decoded through shared placeholders so recursive
        and forward references can be reconstructed. Blocks created before a
        referenced placeholder is populated initially have an incomplete
        effect summary. Repeated refresh reaches the finite fixed point across
        the decoded block graph without rescanning at later API access sites.
        """
        from qamomile.circuit.ir.effect import refresh_block_effects

        while True:
            previous = tuple(
                (block.effects, block.measurement_result_indices)
                for block in self._blocks
            )
            for block in self._blocks:
                refresh_block_effects(block)
            current = tuple(
                (block.effects, block.measurement_result_indices)
                for block in self._blocks
            )
            if current == previous:
                return

    def materialize(self, uuid: str) -> ValueBase:
        """Return the ``ValueBase`` for ``uuid``, instantiating on demand.

        Args:
            uuid (str): The Value UUID to materialize.

        Returns:
            ValueBase: The reconstructed Value, possibly with its
                referenced sub-Values already materialized.

        Raises:
            ValueError: If ``uuid`` is not present in the value table
                or if a reference cycle is detected.
        """
        if uuid in self._built:
            return self._built[uuid]
        if uuid in self._building:
            raise ValueError(f"cycle detected while materializing Value uuid {uuid!r}")
        entry = self._by_uuid.get(uuid)
        if entry is None:
            raise ValueError(f"value_table is missing entry for uuid {uuid!r}")
        self._building.add(uuid)
        try:
            built = _decode_value(entry, self)
        finally:
            self._building.discard(uuid)
        self._built[uuid] = built
        return built

    def definition(self, definition_id: str) -> CallableDef:
        """Resolve a module-local callable-definition reference.

        Args:
            definition_id (str): Definition ID from an InvokeOperation.

        Returns:
            CallableDef: Shared reconstructed definition object.

        Raises:
            ValueError: If the ID is absent from the callable table.
        """
        definition = self._definitions.get(definition_id)
        if definition is None:
            raise ValueError(
                f"callable_table is missing definition id {definition_id!r}"
            )
        return definition

    def populate_definitions(self) -> None:
        """Populate callable placeholders after every graph node has an ID."""
        for definition_id, payload in self._definition_entries.items():
            decoded = _decode_callable_def(payload, self)
            placeholder = self._definitions[definition_id]
            if placeholder.ref != decoded.ref:
                raise ValueError(
                    f"callable definition {definition_id!r} changed ref while decoding"
                )
            placeholder.signature = decoded.signature
            placeholder.body = decoded.body
            placeholder.body_ref = decoded.body_ref
            placeholder.implementations = decoded.implementations
            placeholder.opaque_cost = decoded.opaque_cost
            placeholder.default_policy = decoded.default_policy
            placeholder.attrs = decoded.attrs


# ---------------------------------------------------------------------------
# Block decoder
# ---------------------------------------------------------------------------


def _decode_block(d: dict[str, Any], ctx: _DecodeContext) -> Block:
    """Decode a Block dict.

    Args:
        d (dict[str, Any]): Internal graph record for one block.
        ctx (_DecodeContext): Module-wide Value and CallableDef registry.

    Returns:
        Block: The reconstructed Block.

    Raises:
        ValueError: For any structural / dispatch errors.
    """
    if d.get("$type") != "Block":
        raise ValueError(f"expected $type='Block', got {d.get('$type')!r}")
    kind_name = d.get("kind")
    if not isinstance(kind_name, str):
        raise ValueError("block dict is missing 'kind'")
    try:
        kind = BlockKind[kind_name]
    except KeyError as exc:
        raise ValueError(f"unknown BlockKind {kind_name!r}") from exc
    # Block I/O may legitimately carry ``DictValue`` / ``TupleValue``:
    # a ``qmc.Dict`` / ``qmc.Tuple`` kernel argument lands in
    # ``input_values`` and a container pass-through return lands in
    # ``output_values``. Parameters are scalar / array ``Value``s plus the
    # one structural exception: a runtime-parameter ``Dict`` keeps its
    # ``DictValue`` in ``Block.parameters`` (paired with a ``DictType``
    # slot in ``param_slots``).
    input_values = [
        _materialize_as_value_like(ctx, ref) for ref in d["input_value_refs"]
    ]
    output_values = [
        _materialize_as_value_like(ctx, ref) for ref in d["output_value_refs"]
    ]
    # The cast mirrors the frontend: ``Block.parameters`` is annotated
    # ``dict[str, Value]`` while a runtime-parameter ``Dict`` stores its
    # ``DictValue`` there (the same widening #562 applies at build time).
    parameters = cast(
        "dict[str, Value]",
        {k: _materialize_as_parameter(ctx, ref) for k, ref in d["parameters"].items()},
    )
    static_bindings: list[StaticBindingSlot] = []
    for raw_slot in d.get("static_bindings", ()):
        if not isinstance(raw_slot, dict):
            raise ValueError("static binding record must be a dictionary")
        name = raw_slot.get("name")
        type_key = raw_slot.get("type_key")
        raw_fields = raw_slot.get("fields")
        if (
            not isinstance(name, str)
            or not isinstance(type_key, str)
            or not isinstance(raw_fields, list)
        ):
            raise ValueError("static binding record is malformed")
        fields: list[StaticBindingField] = []
        for raw_field in raw_fields:
            if not isinstance(raw_field, dict):
                raise ValueError("static binding field must be a dictionary")
            field_name = raw_field.get("name")
            value_ref = raw_field.get("value_ref")
            if not isinstance(field_name, str) or not isinstance(value_ref, str):
                raise ValueError("static binding field is malformed")
            fields.append(
                StaticBindingField(
                    name=field_name,
                    value=_materialize_as_value(ctx, value_ref),
                )
            )
        static_bindings.append(
            StaticBindingSlot(name=name, type_key=type_key, fields=tuple(fields))
        )

    operations = [_decode_operation(op_dict, ctx) for op_dict in d["operations"]]

    inferred_slots = [
        ParamSlot(
            name=name,
            type=value.type,
            kind=ParamKind.RUNTIME_PARAMETER,
            ndim=len(value.shape) if isinstance(value, ArrayValue) else 0,
        )
        for name, value in parameters.items()
    ]
    inferred_names = set(parameters)
    for name, value in zip(d.get("label_args", ()), input_values, strict=True):
        if (
            not isinstance(name, str)
            or name in inferred_names
            or isinstance(value, TupleValue)
            or value.type.is_quantum()
            or not value.is_parameter()
        ):
            continue
        inferred_slots.append(
            ParamSlot(
                name=name,
                type=value.type,
                kind=ParamKind.RUNTIME_PARAMETER,
                ndim=len(value.shape) if isinstance(value, ArrayValue) else 0,
            )
        )
        inferred_names.add(name)

    return ctx.register_block(
        Block(
            name=d.get("name", ""),
            kind=kind,
            label_args=list(d.get("label_args", ())),
            input_values=input_values,
            output_values=output_values,
            output_names=list(d.get("output_names", ())),
            operations=operations,
            parameters=parameters,
            param_slots=tuple(inferred_slots),
            static_bindings=tuple(static_bindings),
        )
    )


def _materialize_as_value(ctx: _DecodeContext, uuid: str) -> Value:
    """Materialize a UUID reference and assert it resolves to a ``Value``.

    Used in positions that must hold a scalar / array ``Value`` (e.g.
    operation results, ``ForOperation.loop_var_value``, and slice refs).
    Block-level I/O goes through ``_materialize_as_value_like`` and
    ``Block.parameters`` through ``_materialize_as_parameter`` instead,
    because those positions may carry structural container values.

    Args:
        ctx (_DecodeContext): The active decode context.
        uuid (str): The Value UUID.

    Returns:
        Value: The materialized Value (also accepts ``ArrayValue`` as
            a subclass).

    Raises:
        ValueError: If the materialized object is not a ``Value``.
    """
    v = ctx.materialize(uuid)
    if not isinstance(v, Value):
        raise ValueError(
            f"expected Value or ArrayValue at uuid {uuid!r}, got {type(v).__name__}"
        )
    return v


def _materialize_as_parameter(ctx: _DecodeContext, uuid: str) -> Value | DictValue:
    """Materialize a UUID reference for a ``Block.parameters`` entry.

    Parameters are scalar / array ``Value``s with one structural exception:
    a ``Dict`` kept as a runtime parameter stays in ``Block.parameters`` as
    a ``DictValue`` (its per-key values are rebound per call, mirrored by a
    ``DictType`` ``RUNTIME_PARAMETER`` entry in ``param_slots``).
    ``TupleValue`` never appears here — tuples cannot be runtime parameters.

    Args:
        ctx (_DecodeContext): The active decode context.
        uuid (str): The parameter Value UUID.

    Returns:
        Value | DictValue: The materialized scalar, array, or dict parameter
            value.

    Raises:
        ValueError: If the materialized object is neither a ``Value`` nor a
            ``DictValue``.
    """
    v = ctx.materialize(uuid)
    if not isinstance(v, (Value, DictValue)):
        raise ValueError(
            f"expected Value, ArrayValue, or DictValue at parameter uuid "
            f"{uuid!r}, got {type(v).__name__}"
        )
    return v


def _materialize_as_value_like(ctx: _DecodeContext, uuid: str) -> ValueLike:
    """Materialize a UUID reference as any block-output value type.

    Block inputs and outputs can be structural values such as ``TupleValue``
    and ``DictValue`` in addition to scalar ``Value`` / ``ArrayValue``.
    ``Block.parameters`` goes through ``_materialize_as_parameter`` instead,
    which additionally rejects ``TupleValue``.

    Args:
        ctx (_DecodeContext): The active decode context.
        uuid (str): The Value-like UUID.

    Returns:
        ValueLike: The materialized scalar, array, tuple, or dict value.

    Raises:
        ValueError: If the materialized object is not a supported output value
            type.
    """
    v = ctx.materialize(uuid)
    if not isinstance(v, (Value, TupleValue, DictValue)):
        raise ValueError(
            f"expected Value, ArrayValue, TupleValue, or DictValue at uuid "
            f"{uuid!r}, got {type(v).__name__}"
        )
    return v


# ---------------------------------------------------------------------------
# Value decoder
# ---------------------------------------------------------------------------


def _decode_value(d: dict[str, Any], ctx: _DecodeContext) -> ValueBase:
    """Reconstruct a ``ValueBase`` from its value-table dict.

    Args:
        d (dict[str, Any]): A value dict from ``value_table``.
        ctx (_DecodeContext): The active decode context (used to
            resolve nested Value references via :meth:`_DecodeContext.materialize`).

    Returns:
        ValueBase: The reconstructed Value (concrete subclass matches
            the ``$type`` tag).

    Raises:
        ValueError: If ``$type`` is not a known Value tag.
    """
    tag = d.get("$type")
    if tag == "Value":
        return Value(
            type=_decode_value_type(d["value_type"], ctx),
            name=d.get("name", ""),
            version=int(d.get("version", 0)),
            metadata=_decode_metadata(d.get("metadata"), ctx),
            uuid=d["uuid"],
            logical_id=d["logical_id"],
            parent_array=(
                _materialize_array(ctx, d["parent_array_ref"])
                if d.get("parent_array_ref")
                else None
            ),
            element_indices=tuple(
                _materialize_as_value(ctx, ref)
                for ref in d.get("element_index_refs", ())
            ),
        )
    if tag == "ArrayValue":
        return ArrayValue(
            type=_decode_value_type(d["value_type"], ctx),
            name=d.get("name", ""),
            version=int(d.get("version", 0)),
            metadata=_decode_metadata(d.get("metadata"), ctx),
            uuid=d["uuid"],
            logical_id=d["logical_id"],
            shape=tuple(
                _materialize_as_value(ctx, ref) for ref in d.get("shape_refs", ())
            ),
            slice_of=(
                _materialize_array(ctx, d["slice_of_ref"])
                if d.get("slice_of_ref")
                else None
            ),
            slice_start=(
                _materialize_as_value(ctx, d["slice_start_ref"])
                if d.get("slice_start_ref")
                else None
            ),
            slice_step=(
                _materialize_as_value(ctx, d["slice_step_ref"])
                if d.get("slice_step_ref")
                else None
            ),
        )
    if tag == "TupleValue":
        return TupleValue(
            name=d.get("name", ""),
            elements=tuple(
                _materialize_as_value_like(ctx, ref)
                for ref in d.get("element_refs", ())
            ),
            metadata=_decode_metadata(d.get("metadata"), ctx),
            uuid=d["uuid"],
            logical_id=d["logical_id"],
        )
    if tag == "DictValue":
        entries: list[tuple[TupleValue | Value, Value]] = []
        for k_ref, v_ref in d.get("entry_refs", ()):
            key = ctx.materialize(k_ref)
            if not isinstance(key, (TupleValue, Value)):
                raise ValueError(
                    f"DictValue key at uuid {k_ref!r} must be TupleValue or "
                    f"Value, got {type(key).__name__}"
                )
            entries.append((key, _materialize_as_value(ctx, v_ref)))
        return DictValue(
            name=d.get("name", ""),
            entries=tuple(entries),
            metadata=_decode_metadata(d.get("metadata"), ctx),
            uuid=d["uuid"],
            logical_id=d["logical_id"],
        )
    raise ValueError(f"unknown Value $type tag: {tag!r}")


def _materialize_array(ctx: _DecodeContext, uuid: str) -> ArrayValue:
    """Materialize a UUID and assert it resolves to an ``ArrayValue``.

    Args:
        ctx (_DecodeContext): The active decode context.
        uuid (str): The Value UUID.

    Returns:
        ArrayValue: The materialized array.

    Raises:
        ValueError: If the materialized object is not an
            ``ArrayValue``.
    """
    v = ctx.materialize(uuid)
    if not isinstance(v, ArrayValue):
        raise ValueError(
            f"expected ArrayValue at uuid {uuid!r}, got {type(v).__name__}"
        )
    return v


# ---------------------------------------------------------------------------
# Metadata decoder
# ---------------------------------------------------------------------------


def _decode_metadata(d: Any, ctx: _DecodeContext) -> ValueMetadata:
    """Decode the ``ValueMetadata`` dict.

    Args:
        d (Any): A metadata dict (or ``None`` for an empty bundle).
        ctx (_DecodeContext): The active decode context (unused
            currently; reserved for future metadata that holds
            Value references).

    Returns:
        ValueMetadata: The reconstructed metadata.
    """
    if d is None:
        return ValueMetadata()
    return ValueMetadata(
        scalar=_decode_scalar_metadata(d.get("scalar")),
        cast=_decode_cast_metadata(d.get("cast")),
        qfixed=_decode_qfixed_metadata(d.get("qfixed")),
        array_runtime=_decode_array_runtime_metadata(d.get("array_runtime")),
        dict_runtime=_decode_dict_runtime_metadata(d.get("dict_runtime")),
    )


def _decode_scalar_metadata(d: Any) -> ScalarMetadata | None:
    """Decode :class:`ScalarMetadata` (or ``None``).

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        ScalarMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    return ScalarMetadata(
        const_value=_decode_payload(d.get("const_value")),
        parameter_name=d.get("parameter_name"),
    )


def _decode_cast_metadata(d: Any) -> CastMetadata | None:
    """Decode :class:`CastMetadata` (or ``None``).

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        CastMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    return CastMetadata(
        source_uuid=d["source_uuid"],
        qubit_uuids=tuple(d.get("qubit_uuids", ())),
        source_logical_id=d.get("source_logical_id"),
        qubit_logical_ids=tuple(d.get("qubit_logical_ids", ())),
    )


def _decode_qfixed_metadata(d: Any) -> QFixedMetadata | None:
    """Decode :class:`QFixedMetadata` (or ``None``).

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        QFixedMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    return QFixedMetadata(
        qubit_uuids=tuple(d.get("qubit_uuids", ())),
        num_bits=int(d.get("num_bits", 0)),
        int_bits=int(d.get("int_bits", 0)),
    )


def _decode_array_parent_uuids(values: Any) -> tuple[str, ...]:
    """Decode array-parent UUIDs and restore the standalone sentinel.

    Args:
        values (Any): Sequence of parent UUID strings or ``None`` entries from
            optional protobuf fields.

    Returns:
        tuple[str, ...]: Parent UUIDs with missing entries normalized to ``""``.

    Raises:
        ValueError: If the collection is not a list or tuple, or an entry is
            neither ``str`` nor ``None``.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError("element_parent_uuids must be a list or tuple")
    decoded: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            decoded.append("")
        elif isinstance(value, str):
            decoded.append(value)
        else:
            raise ValueError(
                f"element_parent_uuids[{index}] must be str or None, "
                f"got {type(value).__name__}"
            )
    return tuple(decoded)


def _decode_array_parent_indices(values: Any) -> tuple[int, ...]:
    """Decode array-parent indices and restore the standalone sentinel.

    Args:
        values (Any): Sequence of plain integer indices or ``None`` entries
            from optional protobuf fields.

    Returns:
        tuple[int, ...]: Parent indices with missing entries normalized to
            ``-1``.

    Raises:
        ValueError: If the collection is not a list or tuple, or an entry is
            neither a plain ``int`` nor ``None``.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError("element_parent_indices must be a list or tuple")
    decoded: list[int] = []
    for index, value in enumerate(values):
        if value is None:
            decoded.append(-1)
        elif is_plain_int(value):
            decoded.append(value)
        else:
            raise ValueError(
                f"element_parent_indices[{index}] must be int or None, "
                f"got {type(value).__name__}"
            )
    return tuple(decoded)


def _decode_array_runtime_metadata(d: Any) -> ArrayRuntimeMetadata | None:
    """Decode :class:`ArrayRuntimeMetadata` (or ``None``).

    ``const_array`` is re-frozen after payload decoding: the in-memory
    canonical form is all-tuples (``with_array_runtime_metadata`` runs
    ``_freeze_data`` at construction), while the intermediate graph can only
    carry lists. Without re-freezing, a decoded block would silently
    hold lists where the original held tuples.

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        ArrayRuntimeMetadata | None: The metadata, or ``None`` if absent.

    Raises:
        ValueError: If array-parent metadata has an invalid collection or
            entry type.
    """
    if d is None:
        return None
    return ArrayRuntimeMetadata(
        const_array=_freeze_data(_decode_payload(d.get("const_array"))),
        element_uuids=tuple(d.get("element_uuids", ())),
        element_logical_ids=tuple(d.get("element_logical_ids", ())),
        element_parent_uuids=_decode_array_parent_uuids(
            d.get("element_parent_uuids", ())
        ),
        element_parent_indices=_decode_array_parent_indices(
            d.get("element_parent_indices", ())
        ),
    )


def _decode_dict_runtime_metadata(d: Any) -> DictRuntimeMetadata | None:
    """Decode :class:`DictRuntimeMetadata` (or ``None``).

    Entries are re-frozen after payload decoding: bound dict keys are
    tuples in the in-memory canonical form (``_freeze_data`` runs at
    construction time), while the intermediate graph flattens them to lists.
    Without re-freezing, ``DictValue.get_bound_data()`` and
    ``content_hash`` would fail on a decoded block with
    ``TypeError: unhashable type: 'list'``.

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        DictRuntimeMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    raw_items = d.get("bound_data", [])
    items = tuple(
        (_freeze_data(_decode_payload(k)), _freeze_data(_decode_payload(v)))
        for k, v in raw_items
    )
    return DictRuntimeMetadata(bound_data=items)


# ---------------------------------------------------------------------------
# Payload decoder for arbitrary Python data (numpy arrays, primitives, ...)
# ---------------------------------------------------------------------------


def _decode_payload(value: Any) -> Any:
    """Reverse :func:`encode._encode_payload`.

    Args:
        value (Any): The wire-form payload.

    Returns:
        Any: Reconstructed Python value with tuple/list, set/frozenset,
            arbitrary-key mapping, complex-number, numpy, and Hamiltonian
            identity preserved.

    Raises:
        ValueError: If a tagged container wrapper is malformed, contains an
            unhashable mapping key, or repeats a mapping key.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if is_array_wrapper(value):
        return dict_to_array(value)
    if is_scalar_wrapper(value):
        return dict_to_scalar(value)
    if is_hamiltonian_wrapper(value):
        return dict_to_hamiltonian(value)
    if isinstance(value, list):
        return [_decode_payload(x) for x in value]
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(
                "payload tagged dictionaries must contain exactly one wrapper key"
            )
        if "$tuple" in value:
            raw_items = value["$tuple"]
            if not isinstance(raw_items, list):
                raise ValueError("$tuple payload must contain a list")
            return tuple(_decode_payload(item) for item in raw_items)
        if "$set" in value:
            raw_items = value["$set"]
            if not isinstance(raw_items, list):
                raise ValueError("$set payload must contain a list")
            try:
                return set(_decode_payload(item) for item in raw_items)
            except TypeError as error:
                raise ValueError("$set payload contains an unhashable item") from error
        if "$frozenset" in value:
            raw_items = value["$frozenset"]
            if not isinstance(raw_items, list):
                raise ValueError("$frozenset payload must contain a list")
            try:
                return frozenset(_decode_payload(item) for item in raw_items)
            except TypeError as error:
                raise ValueError(
                    "$frozenset payload contains an unhashable item"
                ) from error
        if "$complex_number" in value:
            raw_parts = value["$complex_number"]
            if not isinstance(raw_parts, list) or len(raw_parts) != 2:
                raise ValueError("$complex_number payload must contain [real, imag]")
            real, imag = raw_parts
            if not isinstance(real, (int, float)) or not isinstance(imag, (int, float)):
                raise ValueError("complex real and imaginary parts must be numeric")
            return complex(real, imag)
        if "$map" in value:
            raw_entries = value["$map"]
            if not isinstance(raw_entries, list):
                raise ValueError("$map payload must contain a list of pairs")
            decoded: dict[Any, Any] = {}
            for entry in raw_entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("$map entries must be [key, value] pairs")
                key = _decode_payload(entry[0])
                item = _decode_payload(entry[1])
                try:
                    if key in decoded:
                        raise ValueError(f"$map contains duplicate key {key!r}")
                    decoded[key] = item
                except TypeError as exc:
                    raise ValueError(f"$map key {key!r} is not hashable") from exc
            return decoded
        raise ValueError(f"unknown payload wrapper {next(iter(value))!r}")
    return value


# ---------------------------------------------------------------------------
# ValueType decoder (closed dispatch)
# ---------------------------------------------------------------------------


def _decode_value_type(d: dict[str, Any], ctx: _DecodeContext) -> ValueType:
    """Decode a ``ValueType`` tagged dict.

    Args:
        d (dict[str, Any]): The tagged dict produced by
            :func:`encode._encode_value_type`.
        ctx (_DecodeContext): The active decode context (needed for
            ``QFixedType`` / ``QUIntType`` which may carry symbolic
            width Values).

    Returns:
        ValueType: The reconstructed type instance.

    Raises:
        ValueError: If ``$type`` is not a known ValueType tag.
    """
    tag = d.get("$type")
    if not isinstance(tag, str):
        raise ValueError(f"ValueType $type tag must be a string, got {tag!r}")
    factory = _VALUE_TYPE_DECODERS.get(tag)
    if factory is None:
        raise ValueError(f"unknown ValueType $type tag: {tag!r}")
    return factory(d, ctx)


def _decode_tuple_type(d: dict[str, Any], ctx: _DecodeContext) -> TupleType:
    """Decode :class:`TupleType`.

    Args:
        d (dict[str, Any]): The tagged dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        TupleType: The reconstructed type.
    """
    return TupleType(
        element_types=tuple(
            _decode_value_type(et, ctx) for et in d.get("element_types", ())
        )
    )


def _decode_dict_type(d: dict[str, Any], ctx: _DecodeContext) -> DictType:
    """Decode :class:`DictType`.

    Args:
        d (dict[str, Any]): The tagged dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        DictType: The reconstructed type.
    """
    return DictType(
        key_type=_decode_value_type(d["key_type"], ctx) if d.get("key_type") else None,
        value_type=(
            _decode_value_type(d["value_type"], ctx) if d.get("value_type") else None
        ),
    )


def _decode_qfixed_type(d: dict[str, Any], ctx: _DecodeContext) -> QFixedType:
    """Decode :class:`QFixedType`.

    Args:
        d (dict[str, Any]): The tagged dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        QFixedType: The reconstructed type.
    """
    return QFixedType(
        integer_bits=_decode_qreg_width(d["integer_bits"], ctx),
        fractional_bits=_decode_qreg_width(d["fractional_bits"], ctx),
    )


def _decode_quint_type(d: dict[str, Any], ctx: _DecodeContext) -> QUIntType:
    """Decode :class:`QUIntType`.

    Args:
        d (dict[str, Any]): The tagged dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        QUIntType: The reconstructed type.
    """
    return QUIntType(width=_decode_qreg_width(d["width"], ctx))


def _decode_qreg_width(value: Any, ctx: _DecodeContext) -> Any:
    """Decode the width field of a QUInt / QFixed type.

    Concrete widths come back as ``int``; symbolic widths come back
    as the materialized ``Value`` they reference.

    Args:
        value (Any): The encoded width (``int`` or
            ``{"$value_ref": <uuid>}``).
        ctx (_DecodeContext): The active decode context.

    Returns:
        Any: ``int`` or ``Value``.

    Raises:
        ValueError: If the width has an unrecognized shape.
    """
    if is_plain_int(value):
        return value
    if isinstance(value, dict) and "$value_ref" in value:
        return _materialize_as_value(ctx, value["$value_ref"])
    raise ValueError(f"unrecognized width payload: {value!r}")


_VALUE_TYPE_DECODERS: dict[
    str, Callable[[dict[str, Any], _DecodeContext], ValueType]
] = {
    "UIntType": lambda d, ctx: UIntType(),
    "FloatType": lambda d, ctx: FloatType(),
    "BitType": lambda d, ctx: BitType(),
    "QubitType": lambda d, ctx: QubitType(),
    "BlockType": lambda d, ctx: BlockType(),
    "ObservableType": lambda d, ctx: ObservableType(),
    "TupleType": _decode_tuple_type,
    "DictType": _decode_dict_type,
    "QFixedType": _decode_qfixed_type,
    "QUIntType": _decode_quint_type,
}


# ---------------------------------------------------------------------------
# Operation decoder (closed dispatch)
# ---------------------------------------------------------------------------


def _decode_operation(d: dict[str, Any], ctx: _DecodeContext) -> Operation:
    """Decode an operation tagged dict.

    Args:
        d (dict[str, Any]): The operation dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        Operation: The reconstructed operation (concrete subclass
            matching the ``$type`` tag).

    Raises:
        ValueError: If ``$type`` is not in the dispatch table.
    """
    tag = d.get("$type")
    if not isinstance(tag, str):
        raise ValueError(f"Operation $type tag must be a string, got {tag!r}")
    decoder = _OP_DECODERS.get(tag)
    if decoder is None:
        raise ValueError(f"unknown Operation $type tag: {tag!r}")
    return decoder(d, ctx)


def _enum_by_name(enum_cls: type, name: Any, field_label: str) -> Any:
    """Resolve an Enum member by its ``name`` attribute, with type checks.

    Args:
        enum_cls (type): The Enum class to look up (e.g.,
            ``GateOperationType``).
        name (Any): The dict-loaded value. Must be a ``str`` matching
            an Enum member name.
        field_label (str): A short human-readable label used in error
            messages (e.g., ``"GateOperationType"``).

    Returns:
        Any: The resolved Enum member.

    Raises:
        ValueError: If ``name`` is not a string or does not match any
            member of ``enum_cls``.
    """
    if not isinstance(name, str):
        raise ValueError(f"{field_label} name must be a string, got {name!r}")
    try:
        return enum_cls[name]  # type: ignore
    except KeyError as exc:
        raise ValueError(f"unknown {field_label} {name!r}") from exc


def _operands_results(
    d: dict[str, Any], ctx: _DecodeContext
) -> tuple[list[Value], list[Value]]:
    """Materialize an operation's operand / result Value lists.

    Args:
        d (dict[str, Any]): The operation dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        tuple[list[Value], list[Value]]: ``(operands, results)``.
    """
    operands = [_materialize_as_value(ctx, ref) for ref in d.get("operand_refs", ())]
    results = [_materialize_as_value(ctx, ref) for ref in d.get("result_refs", ())]
    return operands, results


def _value_like_operands_results(
    d: dict[str, Any], ctx: _DecodeContext
) -> tuple[list[ValueLike], list[ValueLike]]:
    """Materialize structural operation operands and results.

    Args:
        d (dict[str, Any]): The operation dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        tuple[list[ValueLike], list[ValueLike]]: Materialized operands and
        results, including TupleValue and DictValue containers.
    """
    operands = [
        _materialize_as_value_like(ctx, ref) for ref in d.get("operand_refs", ())
    ]
    results = [_materialize_as_value_like(ctx, ref) for ref in d.get("result_refs", ())]
    return operands, results


def _container_operands_results(
    d: dict[str, Any], ctx: _DecodeContext
) -> tuple[list[Value], list[Value]]:
    """Materialize operands permitting container values, results strictly.

    Some operations legitimately carry ``DictValue`` / ``TupleValue``
    operands at trace time: ``ForItemsOperation`` iterates a ``Dict``
    kernel argument and ``InverseBlockOperation`` keeps the container
    parameters of the inverted kernel as operands. Their results are
    still plain ``Value`` / ``ArrayValue`` instances, so result decoding
    keeps the strict check.

    Args:
        d (dict[str, Any]): The operation dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        tuple[list[Value], list[Value]]: ``(operands, results)``. The
            operand list is typed ``list[Value]`` to match
            ``Operation.operands``, but entries may be ``DictValue`` /
            ``TupleValue``, mirroring what the frontend stores there.
    """
    operands = cast(
        list[Value],
        [ctx.materialize(ref) for ref in d.get("operand_refs", ())],
    )
    results = [_materialize_as_value(ctx, ref) for ref in d.get("result_refs", ())]
    return operands, results


def _decode_gate_operation(d: dict[str, Any], ctx: _DecodeContext) -> GateOperation:
    """Decode :class:`GateOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        GateOperation: The reconstructed op.

    Raises:
        ValueError: If ``gate_type`` is not a known
            :class:`GateOperationType` name.
    """
    operands, results = _operands_results(d, ctx)
    gate_type = _enum_by_name(
        GateOperationType, d.get("gate_type"), "GateOperationType"
    )
    return GateOperation(operands=operands, results=results, gate_type=gate_type)


def _decode_measure(d: dict[str, Any], ctx: _DecodeContext) -> MeasureOperation:
    """Decode :class:`MeasureOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        MeasureOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return MeasureOperation(operands=operands, results=results)


def _decode_project(d: dict[str, Any], ctx: _DecodeContext) -> ProjectOperation:
    """Decode :class:`ProjectOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ProjectOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return ProjectOperation(
        operands=operands,
        results=results,
        axis=str(d.get("axis", "z")),
    )


def _decode_reset(d: dict[str, Any], ctx: _DecodeContext) -> ResetOperation:
    """Decode :class:`ResetOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ResetOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return ResetOperation(operands=operands, results=results)


def _decode_measure_vector(
    d: dict[str, Any], ctx: _DecodeContext
) -> MeasureVectorOperation:
    """Decode :class:`MeasureVectorOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        MeasureVectorOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return MeasureVectorOperation(operands=operands, results=results)


def _decode_measure_qfixed(
    d: dict[str, Any], ctx: _DecodeContext
) -> MeasureQFixedOperation:
    """Decode :class:`MeasureQFixedOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        MeasureQFixedOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return MeasureQFixedOperation(
        operands=operands,
        results=results,
        num_bits=int(d.get("num_bits", 0)),
        int_bits=int(d.get("int_bits", 0)),
    )


def _decode_decode_qfixed(
    d: dict[str, Any], ctx: _DecodeContext
) -> DecodeQFixedOperation:
    """Decode :class:`DecodeQFixedOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        DecodeQFixedOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return DecodeQFixedOperation(
        operands=operands,
        results=results,
        num_bits=int(d.get("num_bits", 0)),
        int_bits=int(d.get("int_bits", 0)),
    )


def _decode_store_array_element(
    d: dict[str, Any], ctx: _DecodeContext
) -> StoreArrayElementOperation:
    """Decode :class:`StoreArrayElementOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        StoreArrayElementOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return StoreArrayElementOperation(operands=operands, results=results)


def _decode_return_quantum_array_element(
    d: dict[str, Any], ctx: _DecodeContext
) -> ReturnQuantumArrayElementOperation:
    """Decode :class:`ReturnQuantumArrayElementOperation`.

    Args:
        d (dict[str, Any]): Serialized operation dictionary.
        ctx (_DecodeContext): Active decoding context.

    Returns:
        ReturnQuantumArrayElementOperation: Reconstructed operation.
    """
    operands, results = _operands_results(d, ctx)
    return ReturnQuantumArrayElementOperation(operands=operands, results=results)


def _decode_dict_getitem(
    d: dict[str, Any], ctx: _DecodeContext
) -> DictGetItemOperation:
    """Decode :class:`DictGetItemOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        DictGetItemOperation: The reconstructed op.
    """
    operands, results = _container_operands_results(d, ctx)
    return DictGetItemOperation(
        operands=operands,
        results=results,
        key_arity=int(d.get("key_arity", 1)),
    )


def _decode_cast(d: dict[str, Any], ctx: _DecodeContext) -> CastOperation:
    """Decode :class:`CastOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        CastOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return CastOperation(
        operands=operands,
        results=results,
        source_type=(
            _decode_value_type(d["source_type"], ctx)
            if d.get("source_type") is not None
            else None
        ),
        target_type=(
            _decode_value_type(d["target_type"], ctx)
            if d.get("target_type") is not None
            else None
        ),
        qubit_mapping=list(d.get("qubit_mapping", ())),
    )


def _decode_qinit(d: dict[str, Any], ctx: _DecodeContext) -> QInitOperation:
    """Decode :class:`QInitOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        QInitOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return QInitOperation(operands=operands, results=results)


def _decode_cinit(d: dict[str, Any], ctx: _DecodeContext) -> CInitOperation:
    """Decode :class:`CInitOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        CInitOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return CInitOperation(operands=operands, results=results)


def _decode_slice_array(d: dict[str, Any], ctx: _DecodeContext) -> SliceArrayOperation:
    """Decode :class:`SliceArrayOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        SliceArrayOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return SliceArrayOperation(operands=operands, results=results)


def _decode_release_slice_view(
    d: dict[str, Any], ctx: _DecodeContext
) -> ReleaseSliceViewOperation:
    """Decode :class:`ReleaseSliceViewOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ReleaseSliceViewOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return ReleaseSliceViewOperation(operands=operands, results=results)


def _decode_return(d: dict[str, Any], ctx: _DecodeContext) -> ReturnOperation:
    """Decode :class:`ReturnOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ReturnOperation: The reconstructed op.
    """
    operands, results = _container_operands_results(d, ctx)
    return ReturnOperation(operands=operands, results=results)


def _decode_expval(d: dict[str, Any], ctx: _DecodeContext) -> ExpvalOp:
    """Decode :class:`ExpvalOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ExpvalOp: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return ExpvalOp(operands=operands, results=results)


def _decode_pauli_evolve(d: dict[str, Any], ctx: _DecodeContext) -> PauliEvolveOp:
    """Decode :class:`PauliEvolveOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        PauliEvolveOp: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return PauliEvolveOp(operands=operands, results=results)


def _decode_binop(d: dict[str, Any], ctx: _DecodeContext) -> BinOp:
    """Decode :class:`BinOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        BinOp: The reconstructed op.

    Raises:
        ValueError: If ``kind`` is not a known :class:`BinOpKind`
            name.
    """
    operands, results = _operands_results(d, ctx)
    kind = _enum_by_name(BinOpKind, d.get("kind"), "BinOpKind")
    return BinOp(operands=operands, results=results, kind=kind)


def _decode_unary_math(
    d: dict[str, Any],
    ctx: _DecodeContext,
) -> UnaryMathOp:
    """Decode a unary mathematical operation.

    Args:
        d (dict[str, Any]): Encoded operation dictionary.
        ctx (_DecodeContext): Active decoding context.

    Returns:
        UnaryMathOp: Reconstructed unary mathematical operation.

    Raises:
        ValueError: If ``kind`` is not a known ``UnaryMathOpKind`` name.
    """
    operands, results = _operands_results(d, ctx)
    kind = _enum_by_name(UnaryMathOpKind, d.get("kind"), "UnaryMathOpKind")
    return UnaryMathOp(operands=operands, results=results, kind=kind)


def _decode_compop(d: dict[str, Any], ctx: _DecodeContext) -> CompOp:
    """Decode :class:`CompOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        CompOp: The reconstructed op.

    Raises:
        ValueError: If ``kind`` is not a known :class:`CompOpKind`
            name.
    """
    operands, results = _operands_results(d, ctx)
    kind = _enum_by_name(CompOpKind, d.get("kind"), "CompOpKind")
    return CompOp(operands=operands, results=results, kind=kind)


def _decode_condop(d: dict[str, Any], ctx: _DecodeContext) -> CondOp:
    """Decode :class:`CondOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        CondOp: The reconstructed op.

    Raises:
        ValueError: If ``kind`` is not a known :class:`CondOpKind`
            name.
    """
    operands, results = _operands_results(d, ctx)
    kind = _enum_by_name(CondOpKind, d.get("kind"), "CondOpKind")
    return CondOp(operands=operands, results=results, kind=kind)


def _decode_notop(d: dict[str, Any], ctx: _DecodeContext) -> NotOp:
    """Decode :class:`NotOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        NotOp: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return NotOp(operands=operands, results=results)


def _decode_runtime_classical(
    d: dict[str, Any], ctx: _DecodeContext
) -> RuntimeClassicalExpr:
    """Decode :class:`RuntimeClassicalExpr`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        RuntimeClassicalExpr: The reconstructed op.

    Raises:
        ValueError: If ``kind`` is not a known :class:`RuntimeOpKind`
            name.
    """
    operands, results = _operands_results(d, ctx)
    kind = _enum_by_name(RuntimeOpKind, d.get("kind"), "RuntimeOpKind")
    return RuntimeClassicalExpr(operands=operands, results=results, kind=kind)


def _decode_loop_carried_rebinds(
    d: dict[str, Any],
    ctx: _DecodeContext,
) -> tuple[LoopCarriedRebind, ...]:
    """Decode loop-carried rebind records from a loop op dict.

    Args:
        d (dict[str, Any]): The loop op dict, possibly carrying a
            ``loop_carried_rebinds`` list.
        ctx (_DecodeContext): The active decode context.

    Returns:
        tuple[LoopCarriedRebind, ...]: The reconstructed records; empty
            when the key is absent.
    """
    return tuple(
        LoopCarriedRebind(
            var_name=str(r.get("var_name", "")),
            # Rebind diagnostics also cover structural containers.  Keep
            # their concrete ``ValueBase`` subtype instead of forcing the
            # scalar-only helper used by executable operands/results.
            before=ctx.materialize(r["before_ref"]),
            after=ctx.materialize(r["after_ref"]),
            before_synthesized=bool(r.get("before_synthesized", False)),
        )
        for r in d.get("loop_carried_rebinds", ())
    )


def _decode_region_args(
    d: dict[str, Any],
    ctx: _DecodeContext,
    results: list[Value],
) -> tuple[RegionArg, ...]:
    """Decode loop region-argument records from a loop op dict.

    Args:
        d (dict[str, Any]): The loop op dict, possibly carrying a
            ``region_args`` list.
        ctx (_DecodeContext): The active decode context.
        results (list[Value]): The loop operation's materialized
            results. Every region argument must own the corresponding
            result by both position and UUID.

    Returns:
        tuple[RegionArg, ...]: The reconstructed records; empty when
            the key is absent.

    Raises:
        ValueError: If a removed parallel-list carry payload is present,
            a region argument's four values have different types, or its
            result does not align with the loop operation's results.
    """
    legacy_keys = {
        "carried_names",
        "iter_arg_refs",
        "body_arg_refs",
        "body_yield_refs",
    }
    present_legacy_keys = sorted(legacy_keys.intersection(d))
    if present_legacy_keys:
        raise ValueError(
            "Legacy parallel-list loop carry fields are not supported: "
            f"{', '.join(present_legacy_keys)}. Re-encode the block with "
            "RegionArg records."
        )

    region_args = tuple(
        RegionArg(
            var_name=str(r.get("var_name", "")),
            init=_materialize_as_value(ctx, r["init_ref"]),
            block_arg=_materialize_as_value(ctx, r["block_arg_ref"]),
            yielded=_materialize_as_value(ctx, r["yielded_ref"]),
            result=_materialize_as_value(ctx, r["result_ref"]),
        )
        for r in d.get("region_args", ())
    )
    if len(region_args) != len(results):
        raise ValueError(
            "Loop RegionArg payload is inconsistent: "
            f"{len(region_args)} region args for {len(results)} results."
        )
    for index, (arg, result) in enumerate(zip(region_args, results, strict=True)):
        if arg.result.uuid != result.uuid:
            raise ValueError(
                "Loop RegionArg payload is inconsistent: "
                f"region_args[{index}].result_ref does not match "
                f"result_refs[{index}]."
            )
        if not (
            arg.init.type == arg.block_arg.type == arg.yielded.type == arg.result.type
        ):
            raise ValueError(
                f"Loop RegionArg '{arg.var_name}' has mismatched slot types: "
                f"init={arg.init.type}, block_arg={arg.block_arg.type}, "
                f"yielded={arg.yielded.type}, result={arg.result.type}. "
                "All four values of a RegionArg must share a type."
            )
    return region_args


def _decode_for(d: dict[str, Any], ctx: _DecodeContext) -> ForOperation:
    """Decode :class:`ForOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ForOperation: The reconstructed op, including the materialized
            loop variable, region arguments, and recursively-decoded
            loop body.

    Raises:
        ValueError: If the region arguments are inconsistent (see
            :func:`_decode_region_args`).
    """
    operands, results = _operands_results(d, ctx)
    loop_var_ref = d.get("loop_var_value_ref")
    loop_var_value = _materialize_as_value(ctx, loop_var_ref) if loop_var_ref else None
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    op = ForOperation(
        operands=operands,
        results=results,
        loop_var=d.get("loop_var", ""),
        loop_var_value=loop_var_value,
        operations=body,
        loop_carried_rebinds=_decode_loop_carried_rebinds(d, ctx),
        region_args=_decode_region_args(d, ctx, results),
        captures=tuple(
            _materialize_as_value_like(ctx, ref) for ref in d.get("capture_refs", ())
        ),
    )
    validate_region_args(op)
    return op


def _decode_for_items(d: dict[str, Any], ctx: _DecodeContext) -> ForItemsOperation:
    """Decode :class:`ForItemsOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ForItemsOperation: The reconstructed op, including key /
            value identity Values and the recursively-decoded body.

    Raises:
        ValueError: If the region arguments are inconsistent (see
            :func:`_decode_region_args`).
    """
    operands, results = _container_operands_results(d, ctx)
    key_refs = d.get("key_var_value_refs")
    key_var_values = (
        tuple(_materialize_as_value(ctx, ref) for ref in key_refs)
        if key_refs is not None
        else None
    )
    value_ref = d.get("value_var_value_ref")
    value_var_value = _materialize_as_value(ctx, value_ref) if value_ref else None
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    op = ForItemsOperation(
        operands=operands,
        results=results,
        key_vars=list(d.get("key_vars", ())),
        value_var=d.get("value_var", ""),
        key_is_vector=bool(d.get("key_is_vector", False)),
        key_var_values=key_var_values,
        value_var_value=value_var_value,
        operations=body,
        loop_carried_rebinds=_decode_loop_carried_rebinds(d, ctx),
        region_args=_decode_region_args(d, ctx, results),
        captures=tuple(
            _materialize_as_value_like(ctx, ref) for ref in d.get("capture_refs", ())
        ),
    )
    validate_region_args(op)
    return op


def _decode_while(d: dict[str, Any], ctx: _DecodeContext) -> WhileOperation:
    """Decode :class:`WhileOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        WhileOperation: The reconstructed op, including the
            recursively-decoded loop body.

    Raises:
        ValueError: If the region arguments are inconsistent (see
            :func:`_decode_region_args`).
    """
    operands, results = _operands_results(d, ctx)
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    op = WhileOperation(
        operands=operands,
        results=results,
        operations=body,
        max_iterations=d.get("max_iterations"),
        loop_carried_rebinds=_decode_loop_carried_rebinds(d, ctx),
        region_args=_decode_region_args(d, ctx, results),
        captures=tuple(
            _materialize_as_value_like(ctx, ref) for ref in d.get("capture_refs", ())
        ),
    )
    validate_region_args(op)
    return op


def _decode_branch_rebinds(
    d: dict[str, Any],
    ctx: _DecodeContext,
) -> tuple[BranchRebind, ...]:
    """Decode branch rebind records from an if op dict.

    Args:
        d (dict[str, Any]): The if op dict, possibly carrying a
            ``branch_rebinds`` list.
        ctx (_DecodeContext): The active decode context.

    Returns:
        tuple[BranchRebind, ...]: The reconstructed records; empty when
            the key is absent.
    """
    return tuple(
        BranchRebind(
            var_name=str(r.get("var_name", "")),
            before=_materialize_as_value(ctx, r["before_ref"]),
            rebound_in_true=bool(r.get("rebound_in_true", False)),
            rebound_in_false=bool(r.get("rebound_in_false", False)),
        )
        for r in d.get("branch_rebinds", ())
    )


def _decode_if(d: dict[str, Any], ctx: _DecodeContext) -> IfOperation:
    """Decode :class:`IfOperation`.

    Branch merges arrive as ``true_yield_refs`` / ``false_yield_refs``
    UUID lists parallel to ``results`` and are re-attached through
    ``add_merge`` so the decoded op satisfies the merge invariants
    ``iter_merges`` checks.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        IfOperation: The reconstructed op, including true / false
            branches, the branch-merge slots, and the branch rebind
            records.

    Raises:
        ValueError: If the yield-reference lists disagree with each other or
            with ``results`` in length.
        KeyError: If a required current-format field is absent.
    """
    operands, results = _operands_results(d, ctx)
    true_body = [_decode_operation(child, ctx) for child in d["true_body"]]
    false_body = [_decode_operation(child, ctx) for child in d["false_body"]]
    true_refs = list(d["true_yield_refs"])
    false_refs = list(d["false_yield_refs"])
    if len(true_refs) != len(false_refs) or len(true_refs) != len(results):
        raise ValueError(
            "IfOperation merge data is inconsistent: "
            f"{len(true_refs)} true_yield_refs / {len(false_refs)} "
            f"false_yield_refs for {len(results)} results."
        )
    op = IfOperation(
        operands=operands,
        true_operations=true_body,
        false_operations=false_body,
        branch_rebinds=_decode_branch_rebinds(d, ctx),
        true_captures=tuple(
            _materialize_as_value_like(ctx, ref)
            for ref in d.get("true_capture_refs", ())
        ),
        false_captures=tuple(
            _materialize_as_value_like(ctx, ref)
            for ref in d.get("false_capture_refs", ())
        ),
    )
    for true_ref, false_ref, result in zip(true_refs, false_refs, results, strict=True):
        op.add_merge(
            _materialize_as_value(ctx, true_ref),
            _materialize_as_value(ctx, false_ref),
            result,
        )
    return op


def _decode_control_value(d: dict[str, Any], operation_name: str) -> int | None:
    """Decode an optional coherent-control activation value.

    Args:
        d (dict[str, Any]): Encoded operation payload.
        operation_name (str): Operation name used in malformed-payload errors.

    Returns:
        int | None: Decoded activation value, or ``None`` for all-ones control.

    Raises:
        ValueError: If the encoded value is not a plain Python integer or null.
    """
    control_value = d.get("control_value")
    if control_value is not None and not is_plain_int(control_value):
        raise ValueError(
            f"{operation_name}.control_value must be a Python int or null, "
            f"got {control_value!r}."
        )
    return cast(int | None, control_value)


def _decode_concrete_controlled(
    d: dict[str, Any], ctx: _DecodeContext
) -> ConcreteControlledU:
    """Decode :class:`ConcreteControlledU`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ConcreteControlledU: The reconstructed op, including its
            nested unitary block.

    Raises:
        ValueError: If ``control_value`` is present but is not a Python
            ``int``. Width validation is performed by ``ConcreteControlledU``.
    """
    operands, results = _operands_results(d, ctx)
    block = (
        _decode_block(d["unitary_block"], ctx)
        if d.get("unitary_block") is not None
        else None
    )
    callable_attrs = _decode_callable_attrs(d.get("callable_attrs"))
    control_value = _decode_control_value(d, "ConcreteControlledU")
    return ConcreteControlledU(
        operands=operands,
        results=results,
        num_controls=int(d["num_controls"]),
        control_value=control_value,
        power=_decode_power(d["power"], ctx),
        block=block,
        callable_ref=(
            _decode_callable_ref(d.get("callable_ref"))
            if d.get("callable_ref") is not None
            else None
        ),
        callable_attrs=callable_attrs,
    )


def _decode_select(d: dict[str, Any], ctx: _DecodeContext) -> SelectOperation:
    """Decode a quantum multiplexer and its callable bodies.

    Args:
        d (dict[str, Any]): Encoded operation payload.
        ctx (_DecodeContext): Active decode context.

    Returns:
        SelectOperation: Reconstructed operation.

    Raises:
        ValueError: If the concrete/reference width union, index-argument
            count, or case list is malformed.
    """
    operands, results = _operands_results(d, ctx)
    has_concrete_width = "num_index_qubits" in d
    has_symbolic_width = "num_index_qubits_ref" in d
    if has_concrete_width == has_symbolic_width:
        raise ValueError(
            "SelectOperation requires exactly one of num_index_qubits and "
            "num_index_qubits_ref."
        )
    if has_concrete_width:
        num_index_qubits = d["num_index_qubits"]
        if not is_plain_int(num_index_qubits):
            raise ValueError(
                "SelectOperation.num_index_qubits must be a Python int, "
                f"got {num_index_qubits!r}."
            )
        num_index_args = d.get("num_index_args", num_index_qubits)
    else:
        width_ref = d["num_index_qubits_ref"]
        if not isinstance(width_ref, str):
            raise ValueError(
                "SelectOperation.num_index_qubits_ref must be a string, "
                f"got {width_ref!r}."
            )
        num_index_qubits = _materialize_as_value(ctx, width_ref)
        if "num_index_args" not in d:
            raise ValueError("A symbolic SelectOperation requires num_index_args.")
        num_index_args = d["num_index_args"]
    if not is_plain_int(num_index_args) or num_index_args < 1:
        raise ValueError(
            "SelectOperation.num_index_args must be a positive Python int, "
            f"got {num_index_args!r}."
        )
    raw_case_blocks = d.get("case_blocks")
    if not isinstance(raw_case_blocks, list):
        raise ValueError("SelectOperation.case_blocks must be a list.")
    return SelectOperation(
        operands=operands,
        results=results,
        num_index_qubits=cast("int | Value", num_index_qubits),
        case_blocks=[_decode_block(block, ctx) for block in raw_case_blocks],
        num_index_args=cast(int, num_index_args),
    )


def _decode_symbolic_controlled(
    d: dict[str, Any], ctx: _DecodeContext
) -> SymbolicControlledU:
    """Decode :class:`SymbolicControlledU`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        SymbolicControlledU: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    block = (
        _decode_block(d["unitary_block"], ctx)
        if d.get("unitary_block") is not None
        else None
    )
    callable_attrs = _decode_callable_attrs(d.get("callable_attrs"))
    controlled_refs = d["control_index_refs"]
    control_indices: tuple[Value, ...] | None
    if controlled_refs is None:
        control_indices = None
    else:
        control_indices = tuple(
            _materialize_as_value(ctx, ref) for ref in controlled_refs
        )
    return SymbolicControlledU(
        operands=operands,
        results=results,
        num_controls=_materialize_as_value(ctx, d["num_controls_ref"]),
        control_indices=control_indices,
        power=_decode_power(d["power"], ctx),
        block=block,
        num_control_args=int(d["num_control_args"]),
        callable_ref=(
            _decode_callable_ref(d.get("callable_ref"))
            if d.get("callable_ref") is not None
            else None
        ),
        callable_attrs=callable_attrs,
    )


def _decode_power(value: Any, ctx: _DecodeContext) -> Any:
    """Decode ControlledU's ``power`` field.

    Args:
        value (Any): The encoded power (``int`` or
            ``{"$value_ref": <uuid>}``).
        ctx (_DecodeContext): The active decode context.

    Returns:
        Any: ``int`` or materialized ``Value``.

    Raises:
        ValueError: If the payload shape is unrecognized.
    """
    if is_plain_int(value):
        return value
    if isinstance(value, dict) and "$value_ref" in value:
        return _materialize_as_value(ctx, value["$value_ref"])
    raise ValueError(f"unrecognized power payload: {value!r}")


def _decode_callable_ref(d: Any) -> CallableRef:
    """Decode a callable reference.

    Args:
        d (Any): Serialized callable reference payload.

    Returns:
        CallableRef: Reconstructed callable reference.

    Raises:
        ValueError: If the payload is not a dict with namespace and name.
    """
    if not isinstance(d, dict):
        raise ValueError("CallableRef payload must be a dict")
    namespace = d.get("namespace")
    name = d.get("name")
    version = d.get("version", "1")
    if not isinstance(namespace, str) or not isinstance(name, str):
        raise ValueError("CallableRef payload requires string namespace and name")
    if not isinstance(version, str):
        raise ValueError("CallableRef payload requires string version")
    return CallableRef(namespace=namespace, name=name, version=version)


def _decode_callable_attrs(d: Any) -> dict[str, Any]:
    """Decode optional serializer-friendly callable attrs.

    Args:
        d (Any): Encoded payload or ``None``.

    Returns:
        dict[str, Any]: Decoded attrs. Missing attrs decode to an empty dict.

    Raises:
        ValueError: If the payload does not decode to a dict.
    """
    attrs = _decode_payload(d)
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        raise ValueError("ControlledU callable_attrs must decode to a dict")
    return attrs


def _decode_callable_body_ref(d: Any) -> CallableBodyRef | None:
    """Decode a deferred callable body reference.

    Args:
        d (Any): Serialized body-reference payload.

    Returns:
        CallableBodyRef | None: Reconstructed body reference, or ``None`` when
        absent.

    Raises:
        ValueError: If the payload is malformed.
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError("CallableBodyRef payload must be a dict")
    kind = d.get("kind", "standard")
    if not isinstance(kind, str):
        raise ValueError("CallableBodyRef kind must be a string")
    attrs = _decode_payload(d.get("attrs"))
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise ValueError("CallableBodyRef attrs must decode to a dict")
    return CallableBodyRef(
        ref=_decode_callable_ref(d.get("ref")),
        kind=kind,
        attrs=attrs,
    )


def _decode_callable_implementation(
    d: Any,
    ctx: _DecodeContext,
) -> CallableImplementation:
    """Decode a callable implementation candidate.

    Args:
        d (Any): Serialized implementation payload.
        ctx (_DecodeContext): Module-wide graph registry.

    Returns:
        CallableImplementation: Reconstructed implementation.

    Raises:
        ValueError: If the payload is not a dict.
    """
    if not isinstance(d, dict):
        raise ValueError("CallableImplementation payload must be a dict")
    attrs = _decode_payload(d.get("attrs"))
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise ValueError("CallableImplementation attrs must decode to a dict")
    return CallableImplementation(
        transform=_enum_by_name(CallTransform, d.get("transform"), "CallTransform"),
        backend=d.get("backend"),
        strategy=d.get("strategy"),
        body=(_decode_block(d["body"], ctx) if d.get("body") is not None else None),
        body_ref=_decode_callable_body_ref(d.get("body_ref")),
        attrs=attrs,
    )


def _decode_signature(d: Any, ctx: _DecodeContext) -> Signature | None:
    """Decode a callable definition signature.

    Args:
        d (Any): Serialized signature payload.
        ctx (_DecodeContext): Active decode context used for value-type
            decoding.

    Returns:
        Signature | None: Decoded signature, or ``None`` when absent.

    Raises:
        ValueError: If the signature payload is malformed.
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError("Signature payload must be a dict")

    operands: list[ParamHint | None] = []
    for hint in d.get("operands", []):
        if hint is None:
            operands.append(None)
            continue
        if not isinstance(hint, dict):
            raise ValueError("Signature operand hint must be a dict or None")
        name = hint.get("name")
        if not isinstance(name, str):
            raise ValueError("Signature operand hint requires a string name")
        operands.append(
            ParamHint(name=name, type=_decode_value_type(hint["type"], ctx))
        )

    results: list[ParamHint] = []
    for hint in d.get("results", []):
        if not isinstance(hint, dict):
            raise ValueError("Signature result hint must be a dict")
        name = hint.get("name")
        if not isinstance(name, str):
            raise ValueError("Signature result hint requires a string name")
        results.append(ParamHint(name=name, type=_decode_value_type(hint["type"], ctx)))

    return Signature(operands=operands, results=results)


def _decode_callable_def(d: Any, ctx: _DecodeContext) -> CallableDef:
    """Decode a callable definition.

    Args:
        d (Any): Serialized definition payload.
        ctx (_DecodeContext): Active decode context.

    Returns:
        CallableDef: Reconstructed definition.

    Raises:
        ValueError: If the payload is malformed.
    """
    if not isinstance(d, dict):
        raise ValueError("CallableDef payload must be a dict")
    attrs = _decode_payload(d.get("attrs"))
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise ValueError("CallableDef attrs must decode to a dict")
    raw_policy = d.get("default_policy", CallPolicy.INLINE.name)
    return CallableDef(
        ref=_decode_callable_ref(d.get("ref")),
        signature=_decode_signature(d.get("signature"), ctx),
        body=(_decode_block(d["body"], ctx) if d.get("body") is not None else None),
        body_ref=_decode_callable_body_ref(d.get("body_ref")),
        implementations=[
            _decode_callable_implementation(impl, ctx)
            for impl in d.get("implementations", [])
        ],
        opaque_cost=None,
        default_policy=_enum_by_name(CallPolicy, raw_policy, "CallPolicy"),
        attrs=attrs,
    )


def _decode_invoke_operation(d: dict[str, Any], ctx: _DecodeContext) -> InvokeOperation:
    """Decode :class:`InvokeOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        InvokeOperation: The reconstructed invocation operation.

    Raises:
        ValueError: If transform names are unknown.
    """
    operands, results = _value_like_operands_results(d, ctx)
    transform = _enum_by_name(CallTransform, d.get("transform"), "CallTransform")
    attrs = _decode_payload(d.get("attrs"))
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise ValueError("InvokeOperation attrs must decode to a dict")
    raw_definition_ref = d.get("definition_ref")
    if not isinstance(raw_definition_ref, str):
        raise ValueError("InvokeOperation definition_ref must be a string")
    definition = ctx.definition(raw_definition_ref)
    return InvokeOperation(
        operands=operands,
        results=results,
        target=_decode_callable_ref(d.get("target")),
        transform=transform,
        attrs=attrs,
        definition=definition,
    )


def _decode_inverse_block(
    d: dict[str, Any], ctx: _DecodeContext
) -> InverseBlockOperation:
    """Decode :class:`InverseBlockOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        InverseBlockOperation: The reconstructed inverse block op with its
            source and fallback blocks.
    """
    operands, results = _container_operands_results(d, ctx)
    source_block = (
        _decode_block(d["source_block"], ctx)
        if d.get("source_block") is not None
        else None
    )
    implementation_block = (
        _decode_block(d["implementation_block"], ctx)
        if d.get("implementation_block") is not None
        else None
    )
    return InverseBlockOperation(
        operands=operands,
        results=results,
        num_control_qubits=int(d.get("num_control_qubits", 0)),
        num_target_qubits=int(d.get("num_target_qubits", 0)),
        custom_name=d.get("custom_name", ""),
        source_block=source_block,
        implementation_block=implementation_block,
        callable_ref=(
            _decode_callable_ref(d.get("callable_ref"))
            if d.get("callable_ref") is not None
            else None
        ),
        callable_attrs=_decode_callable_attrs(d.get("callable_attrs")),
        control_value=_decode_control_value(d, "InverseBlockOperation"),
    )


def _decode_global_phase_operation(
    d: dict[str, Any], ctx: _DecodeContext
) -> GlobalPhaseOperation:
    """Decode a zero-qubit global-phase operation.

    Args:
        d (dict[str, Any]): Serialized operation dictionary.
        ctx (_DecodeContext): Active decode context.

    Returns:
        GlobalPhaseOperation: Reconstructed operation.
    """
    operands, results = _operands_results(d, ctx)
    return GlobalPhaseOperation(operands=operands, results=results)


_OP_DECODERS: dict[str, Callable[[dict[str, Any], _DecodeContext], Operation]] = {
    "GateOperation": _decode_gate_operation,
    "MeasureOperation": _decode_measure,
    "ProjectOperation": _decode_project,
    "ResetOperation": _decode_reset,
    "MeasureVectorOperation": _decode_measure_vector,
    "MeasureQFixedOperation": _decode_measure_qfixed,
    "DecodeQFixedOperation": _decode_decode_qfixed,
    "DictGetItemOperation": _decode_dict_getitem,
    "StoreArrayElementOperation": _decode_store_array_element,
    "ReturnQuantumArrayElementOperation": _decode_return_quantum_array_element,
    "CastOperation": _decode_cast,
    "QInitOperation": _decode_qinit,
    "CInitOperation": _decode_cinit,
    "SliceArrayOperation": _decode_slice_array,
    "ReleaseSliceViewOperation": _decode_release_slice_view,
    "ReturnOperation": _decode_return,
    "ExpvalOp": _decode_expval,
    "PauliEvolveOp": _decode_pauli_evolve,
    "BinOp": _decode_binop,
    "UnaryMathOp": _decode_unary_math,
    "CompOp": _decode_compop,
    "CondOp": _decode_condop,
    "NotOp": _decode_notop,
    "RuntimeClassicalExpr": _decode_runtime_classical,
    "ForOperation": _decode_for,
    "ForItemsOperation": _decode_for_items,
    "WhileOperation": _decode_while,
    "IfOperation": _decode_if,
    "ConcreteControlledU": _decode_concrete_controlled,
    "SymbolicControlledU": _decode_symbolic_controlled,
    "SelectOperation": _decode_select,
    "InvokeOperation": _decode_invoke_operation,
    "InverseBlockOperation": _decode_inverse_block,
    "GlobalPhaseOperation": _decode_global_phase_operation,
}
