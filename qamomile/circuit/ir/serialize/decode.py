"""Intermediate dict → IR decoder.

Reconstructs a ``Block`` from the dict envelope produced by
:mod:`qamomile.circuit.ir.serialize.encode`. The decoder NEVER
performs dynamic class resolution: every ``$type`` tag is routed
through a hard-coded factory table, and unknown tags raise
``ValueError``. This is the load-bearing security invariant — see
:mod:`qamomile.circuit.ir.serialize.schema` for the rationale.

Values are materialized lazily via depth-first recursion so any
referenced ``parent_array`` / ``element_indices`` / shape Value is
instantiated before the Value that points at it. A cycle (defensive;
not produced by the canonical encoder) raises ``ValueError``.
"""

from __future__ import annotations

from typing import Any, Callable

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    CompositeGateOperation,
    CompositeGateType,
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    Operation,
    ReturnOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    PhiOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
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
    ValueMetadata,
)

from .numpy_io import dict_to_array, is_array_wrapper
from .schema import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_dict(envelope: dict[str, Any]) -> Block:
    """Reconstruct a ``Block`` from a dict envelope.

    Args:
        envelope (dict[str, Any]): The dict produced by
            :func:`qamomile.circuit.ir.serialize.encode.to_dict` (or
            an equivalent producer that respects the schema).

    Returns:
        Block: The reconstructed Block at ``AFFINE`` or ``ANALYZED``.

    Raises:
        ValueError: If the envelope is malformed, the
            ``schema_version`` does not match, or any ``$type`` tag
            is unknown to the dispatch table.
        TypeError: If a payload has an unexpected Python type.
    """
    if not isinstance(envelope, dict):
        raise ValueError(
            f"from_dict() expected a dict envelope, got {type(envelope).__name__}"
        )
    version = envelope.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: payload reports {version!r}, this loader "
            f"supports {SCHEMA_VERSION}. Cross-version migration is not yet "
            f"implemented."
        )
    block_dict = envelope.get("block")
    if not isinstance(block_dict, dict):
        raise ValueError("envelope is missing a 'block' dict")
    return _decode_block(block_dict, enforce_top_kind=True)


# ---------------------------------------------------------------------------
# Decoding context: lazy Value materialization
# ---------------------------------------------------------------------------


class _DecodeContext:
    """Working state for one Block decode.

    Holds the value-table dicts keyed by UUID so the recursive
    materializer can resolve cross-references depth-first.
    """

    def __init__(self, value_table: list[dict[str, Any]]) -> None:
        """Initialize a decode context.

        Args:
            value_table (list[dict[str, Any]]): The list of Value
                dicts from the block envelope. Each entry must have a
                ``uuid`` field; duplicates are an error.

        Raises:
            ValueError: If a value-table dict lacks a ``uuid`` or if
                two entries share the same UUID.
        """
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


# ---------------------------------------------------------------------------
# Block decoder
# ---------------------------------------------------------------------------


def _decode_block(d: dict[str, Any], *, enforce_top_kind: bool = False) -> Block:
    """Decode a Block dict.

    Args:
        d (dict[str, Any]): The block dict (see schema).
        enforce_top_kind (bool): When ``True`` (only at the top-level
            ``from_dict`` entry), reject any kind other than
            ``AFFINE`` / ``ANALYZED``. Nested blocks (those embedded in
            ``ControlledUOperation.block`` or
            ``CompositeGateOperation.implementation_block``) may
            legitimately be ``HIERARCHICAL`` — e.g., the cached
            ``kernel.block`` form of a leaf kernel passed to
            ``qmc.control``. Skip the kind check there.

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
    if enforce_top_kind and kind not in (BlockKind.AFFINE, BlockKind.ANALYZED):
        raise ValueError(
            f"from_dict only supports AFFINE / ANALYZED blocks at the top "
            f"level; got {kind.name}"
        )

    value_table = d.get("value_table")
    if not isinstance(value_table, list):
        raise ValueError("block dict is missing 'value_table' list")
    ctx = _DecodeContext(value_table)

    input_values = [_materialize_as_value(ctx, ref) for ref in d["input_value_refs"]]
    output_values = [_materialize_as_value(ctx, ref) for ref in d["output_value_refs"]]
    parameters = {
        k: _materialize_as_value(ctx, ref) for k, ref in d["parameters"].items()
    }
    param_slots = tuple(_decode_param_slot(s, ctx) for s in d.get("param_slots", ()))

    operations = [_decode_operation(op_dict, ctx) for op_dict in d["operations"]]

    return Block(
        name=d.get("name", ""),
        kind=kind,
        label_args=list(d.get("label_args", ())),
        input_values=input_values,
        output_values=output_values,
        output_names=list(d.get("output_names", ())),
        operations=operations,
        parameters=parameters,
        param_slots=param_slots,
    )


def _materialize_as_value(ctx: _DecodeContext, uuid: str) -> Value:
    """Materialize a UUID reference and assert it resolves to a ``Value``.

    Used in positions like ``Block.input_values`` whose static type is
    ``list[Value]``.

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
        )
    if tag == "TupleValue":
        return TupleValue(
            name=d.get("name", ""),
            elements=tuple(
                _materialize_as_value(ctx, ref) for ref in d.get("element_refs", ())
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


def _decode_array_runtime_metadata(d: Any) -> ArrayRuntimeMetadata | None:
    """Decode :class:`ArrayRuntimeMetadata` (or ``None``).

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        ArrayRuntimeMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    return ArrayRuntimeMetadata(
        const_array=_decode_payload(d.get("const_array")),
        element_uuids=tuple(d.get("element_uuids", ())),
        element_logical_ids=tuple(d.get("element_logical_ids", ())),
        element_parent_uuids=tuple(d.get("element_parent_uuids", ())),
        element_parent_indices=tuple(d.get("element_parent_indices", ())),
    )


def _decode_dict_runtime_metadata(d: Any) -> DictRuntimeMetadata | None:
    """Decode :class:`DictRuntimeMetadata` (or ``None``).

    Args:
        d (Any): The serialized form (dict or ``None``).

    Returns:
        DictRuntimeMetadata | None: The metadata, or ``None`` if absent.
    """
    if d is None:
        return None
    raw_items = d.get("bound_data", [])
    items = tuple((_decode_payload(k), _decode_payload(v)) for k, v in raw_items)
    return DictRuntimeMetadata(bound_data=items)


# ---------------------------------------------------------------------------
# ParamSlot decoder
# ---------------------------------------------------------------------------


def _decode_param_slot(d: dict[str, Any], ctx: _DecodeContext) -> ParamSlot:
    """Decode a ``ParamSlot`` dict.

    Args:
        d (dict[str, Any]): The slot dict.
        ctx (_DecodeContext): The active decode context (passed
            through to ``_decode_value_type`` in case the type carries
            a symbolic-width Value reference).

    Returns:
        ParamSlot: The reconstructed slot.

    Raises:
        ValueError: If ``kind`` is not a known ``ParamKind`` value.
    """
    kind_str = d.get("kind")
    try:
        kind = ParamKind(kind_str)
    except ValueError as exc:
        raise ValueError(f"unknown ParamKind value {kind_str!r}") from exc
    return ParamSlot(
        name=d["name"],
        type=_decode_value_type(d["type"], ctx),
        kind=kind,
        ndim=int(d.get("ndim", 0)),
        default=_decode_payload(d.get("default")),
        bound_value=_decode_payload(d.get("bound_value")),
        differentiable=bool(d.get("differentiable", False)),
    )


# ---------------------------------------------------------------------------
# Payload decoder for arbitrary Python data (numpy arrays, primitives, ...)
# ---------------------------------------------------------------------------


def _decode_payload(value: Any) -> Any:
    """Reverse :func:`encode._encode_payload`.

    Args:
        value (Any): The wire-form payload.

    Returns:
        Any: The reconstructed Python value (primitives unchanged,
            list / dict recursed, numpy wrappers expanded into
            ``np.ndarray``).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if is_array_wrapper(value):
        return dict_to_array(value)
    if isinstance(value, list):
        return [_decode_payload(x) for x in value]
    if isinstance(value, dict):
        # Plain dict (not a numpy wrapper); recurse on values.
        return {k: _decode_payload(v) for k, v in value.items()}
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
    if isinstance(value, int):
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


def _decode_return(d: dict[str, Any], ctx: _DecodeContext) -> ReturnOperation:
    """Decode :class:`ReturnOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ReturnOperation: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
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


def _decode_phi(d: dict[str, Any], ctx: _DecodeContext) -> PhiOp:
    """Decode :class:`PhiOp`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        PhiOp: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    return PhiOp(operands=operands, results=results)


def _decode_for(d: dict[str, Any], ctx: _DecodeContext) -> ForOperation:
    """Decode :class:`ForOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ForOperation: The reconstructed op, including the
            materialized loop variable and the recursively-decoded
            loop body.
    """
    operands, results = _operands_results(d, ctx)
    loop_var_ref = d.get("loop_var_value_ref")
    loop_var_value = _materialize_as_value(ctx, loop_var_ref) if loop_var_ref else None
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    return ForOperation(
        operands=operands,
        results=results,
        loop_var=d.get("loop_var", ""),
        loop_var_value=loop_var_value,
        operations=body,
    )


def _decode_for_items(d: dict[str, Any], ctx: _DecodeContext) -> ForItemsOperation:
    """Decode :class:`ForItemsOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        ForItemsOperation: The reconstructed op, including key /
            value identity Values and the recursively-decoded body.
    """
    operands, results = _operands_results(d, ctx)
    key_refs = d.get("key_var_value_refs")
    key_var_values = (
        tuple(_materialize_as_value(ctx, ref) for ref in key_refs)
        if key_refs is not None
        else None
    )
    value_ref = d.get("value_var_value_ref")
    value_var_value = _materialize_as_value(ctx, value_ref) if value_ref else None
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    return ForItemsOperation(
        operands=operands,
        results=results,
        key_vars=list(d.get("key_vars", ())),
        value_var=d.get("value_var", ""),
        key_is_vector=bool(d.get("key_is_vector", False)),
        key_var_values=key_var_values,
        value_var_value=value_var_value,
        operations=body,
    )


def _decode_while(d: dict[str, Any], ctx: _DecodeContext) -> WhileOperation:
    """Decode :class:`WhileOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        WhileOperation: The reconstructed op, including the
            recursively-decoded loop body.
    """
    operands, results = _operands_results(d, ctx)
    body = [_decode_operation(child, ctx) for child in d.get("body", ())]
    return WhileOperation(
        operands=operands,
        results=results,
        operations=body,
        max_iterations=d.get("max_iterations"),
    )


def _decode_if(d: dict[str, Any], ctx: _DecodeContext) -> IfOperation:
    """Decode :class:`IfOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        IfOperation: The reconstructed op, including true / false
            branches and the phi-op list.
    """
    operands, results = _operands_results(d, ctx)
    true_body = [_decode_operation(child, ctx) for child in d.get("true_body", ())]
    false_body = [_decode_operation(child, ctx) for child in d.get("false_body", ())]
    phi_ops: list[PhiOp] = []
    for raw in d.get("phi_ops", ()):
        decoded = _decode_operation(raw, ctx)
        if not isinstance(decoded, PhiOp):
            raise ValueError(
                f"IfOperation.phi_ops must be PhiOp, got {type(decoded).__name__}"
            )
        phi_ops.append(decoded)
    return IfOperation(
        operands=operands,
        results=results,
        true_operations=true_body,
        false_operations=false_body,
        phi_ops=phi_ops,
    )


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
    """
    operands, results = _operands_results(d, ctx)
    block = (
        _decode_block(d["unitary_block"])
        if d.get("unitary_block") is not None
        else None
    )
    return ConcreteControlledU(
        operands=operands,
        results=results,
        num_controls=int(d.get("num_controls", 1)),
        power=_decode_power(d.get("power", 1), ctx),
        block=block,
    )


def _decode_symbolic_controlled(
    d: dict[str, Any], ctx: _DecodeContext
) -> SymbolicControlledU:
    """Decode :class:`SymbolicControlledU`.

    The ``num_control_args`` field is treated as additive: payloads
    that omit it (either pre-multi-arg encoders or the legacy single-
    pool form, which the encoder skips for compactness) decode with
    the dataclass default ``1``.  Newer payloads carry the actual
    count so the emit pass can split ``operands`` at the correct
    boundary between the control prefix and the sub-kernel quantum
    tail.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        SymbolicControlledU: The reconstructed op.
    """
    operands, results = _operands_results(d, ctx)
    block = (
        _decode_block(d["unitary_block"])
        if d.get("unitary_block") is not None
        else None
    )
    controlled_refs = d.get("control_index_refs")
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
        power=_decode_power(d.get("power", 1), ctx),
        block=block,
        num_control_args=int(d.get("num_control_args", 1)),
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
    if isinstance(value, int):
        return value
    if isinstance(value, dict) and "$value_ref" in value:
        return _materialize_as_value(ctx, value["$value_ref"])
    raise ValueError(f"unrecognized power payload: {value!r}")


def _decode_composite_gate(
    d: dict[str, Any], ctx: _DecodeContext
) -> CompositeGateOperation:
    """Decode :class:`CompositeGateOperation`.

    Args:
        d (dict[str, Any]): The op dict.
        ctx (_DecodeContext): The active decode context.

    Returns:
        CompositeGateOperation: The reconstructed op with its nested
            implementation block.

    Raises:
        ValueError: If ``gate_type`` is not a known
            :class:`CompositeGateType` name.
    """
    operands, results = _operands_results(d, ctx)
    gate_type = _enum_by_name(
        CompositeGateType, d.get("gate_type"), "CompositeGateType"
    )
    implementation = (
        _decode_block(d["implementation_block"])
        if d.get("implementation_block") is not None
        else None
    )
    return CompositeGateOperation(
        operands=operands,
        results=results,
        gate_type=gate_type,
        num_control_qubits=int(d.get("num_control_qubits", 0)),
        num_target_qubits=int(d.get("num_target_qubits", 0)),
        custom_name=d.get("custom_name", ""),
        resource_metadata=_decode_resource_metadata(d.get("resource_metadata")),
        has_implementation=bool(d.get("has_implementation", True)),
        implementation_block=implementation,
        composite_gate_instance=None,
        strategy_name=d.get("strategy_name"),
    )


def _decode_resource_metadata(d: Any) -> ResourceMetadata | None:
    """Decode :class:`ResourceMetadata` (or ``None``).

    Args:
        d (Any): The serialized form.

    Returns:
        ResourceMetadata | None: ``None`` when absent; else the
            reconstructed metadata.
    """
    if d is None:
        return None
    custom = _decode_payload(d.get("custom_metadata"))
    if custom is None:
        custom = {}
    return ResourceMetadata(
        query_complexity=d.get("query_complexity"),
        t_gates=d.get("t_gates"),
        ancilla_qubits=int(d.get("ancilla_qubits", 0)),
        total_gates=d.get("total_gates"),
        single_qubit_gates=d.get("single_qubit_gates"),
        two_qubit_gates=d.get("two_qubit_gates"),
        multi_qubit_gates=d.get("multi_qubit_gates"),
        clifford_gates=d.get("clifford_gates"),
        rotation_gates=d.get("rotation_gates"),
        custom_metadata=custom,
    )


_OP_DECODERS: dict[str, Callable[[dict[str, Any], _DecodeContext], Operation]] = {
    "GateOperation": _decode_gate_operation,
    "MeasureOperation": _decode_measure,
    "MeasureVectorOperation": _decode_measure_vector,
    "MeasureQFixedOperation": _decode_measure_qfixed,
    "DecodeQFixedOperation": _decode_decode_qfixed,
    "CastOperation": _decode_cast,
    "QInitOperation": _decode_qinit,
    "CInitOperation": _decode_cinit,
    "ReturnOperation": _decode_return,
    "ExpvalOp": _decode_expval,
    "PauliEvolveOp": _decode_pauli_evolve,
    "BinOp": _decode_binop,
    "CompOp": _decode_compop,
    "CondOp": _decode_condop,
    "NotOp": _decode_notop,
    "RuntimeClassicalExpr": _decode_runtime_classical,
    "PhiOp": _decode_phi,
    "ForOperation": _decode_for,
    "ForItemsOperation": _decode_for_items,
    "WhileOperation": _decode_while,
    "IfOperation": _decode_if,
    "ConcreteControlledU": _decode_concrete_controlled,
    "SymbolicControlledU": _decode_symbolic_controlled,
    "CompositeGateOperation": _decode_composite_gate,
}
