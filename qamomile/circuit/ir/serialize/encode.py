"""IR → intermediate dict encoder.

Walks a ``Block`` (AFFINE or ANALYZED) and produces the dict shape
documented in :mod:`qamomile.circuit.ir.serialize.schema`. Values are
deduplicated into ``value_table`` and referenced elsewhere by UUID.

Every encoder branch is dispatched through a hard-coded table keyed
on the runtime class; there is no dynamic resolution, no ``getattr``
on user data, and no ``importlib`` use. The decoder mirrors this
discipline (see :mod:`qamomile.circuit.ir.serialize.decode`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qamomile._utils import is_plain_int
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    CompositeGateOperation,
    ControlledUOperation,
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    InverseBlockOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    Operation,
    ReturnOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    PhiOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.parameter import ParamSlot
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
from qamomile.observable.hamiltonian import Hamiltonian

from .hamiltonian_io import hamiltonian_to_dict
from .numpy_io import array_to_dict
from .schema import SCHEMA_VERSION

_SUPPORTED_KINDS = frozenset({BlockKind.AFFINE, BlockKind.ANALYZED})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_dict(block: Block) -> dict[str, Any]:
    """Encode a ``Block`` into the intermediate dict envelope.

    Args:
        block (Block): The block to encode. Must be at
            ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED``.

    Returns:
        dict[str, Any]: ``{"schema_version": SCHEMA_VERSION, "block":
            <block dict>}`` ready to be passed to a wire-format
            encoder (JSON / msgpack).

    Raises:
        ValueError: If ``block.kind`` is not ``AFFINE`` / ``ANALYZED``.
        NotImplementedError: If the block contains a
            ``CallBlockOperation`` (HIERARCHICAL holdover).
        TypeError: If a payload (e.g., ``ParamSlot.bound_value``)
            cannot be encoded with a known wire representation.
    """
    if block.kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"to_dict() requires BlockKind.AFFINE or BlockKind.ANALYZED; "
            f"got {block.kind.name}. Inline first."
        )
    ctx = _EncodeContext()
    block_dict = _encode_block(block, ctx)
    return {"schema_version": SCHEMA_VERSION, "block": block_dict}


# ---------------------------------------------------------------------------
# Context: collects unique Values into a flat table
# ---------------------------------------------------------------------------


class _EncodeContext:
    """Working state for one ``to_dict`` invocation.

    Collects every Value reachable from the Block under encoding into
    ``value_table_dicts`` in first-visit order, keyed by UUID for
    deduplication.
    """

    def __init__(self) -> None:
        """Initialize an empty encode context."""
        self.value_table_dicts: list[dict[str, Any]] = []
        self._seen_uuids: set[str] = set()

    def register_value(self, v: ValueBase) -> str:
        """Record ``v`` in the value table if not already present.

        Args:
            v (ValueBase): A Value reachable from the block.

        Returns:
            str: ``v.uuid``. Always returns the UUID so callers can use
                the result inline in a ``_ref`` field.
        """
        if v.uuid in self._seen_uuids:
            return v.uuid
        self._seen_uuids.add(v.uuid)
        # Reserve a slot before recursing so cycles (defensive) and
        # multiple references resolve to a single entry. The slot is
        # filled in-place after the encoder returns.
        slot_index = len(self.value_table_dicts)
        self.value_table_dicts.append({})
        self.value_table_dicts[slot_index] = _encode_value(v, self)
        return v.uuid


# ---------------------------------------------------------------------------
# Block / Operation / Value encoders
# ---------------------------------------------------------------------------


def _encode_block(block: Block, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode a Block dict.

    Args:
        block (Block): The block to encode.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: The Block dict; see schema documentation.
    """
    # Walk all reachable Values into the table. Order matters for the
    # output dict's ``value_table`` field but the actual ordering is
    # not semantically load-bearing — references are by UUID.
    for v in block.input_values:
        ctx.register_value(v)
    for v in block.parameters.values():
        ctx.register_value(v)
    for op in block.operations:
        _walk_op_values(op, ctx)
    for v in block.output_values:
        ctx.register_value(v)

    return {
        "$type": "Block",
        "kind": block.kind.name,
        "name": block.name,
        "label_args": list(block.label_args),
        "input_value_refs": [v.uuid for v in block.input_values],
        "output_value_refs": [v.uuid for v in block.output_values],
        "output_names": list(block.output_names),
        "parameters": {k: v.uuid for k, v in block.parameters.items()},
        "param_slots": [_encode_param_slot(s) for s in block.param_slots],
        "value_table": ctx.value_table_dicts,
        "operations": [_encode_operation(op, ctx) for op in block.operations],
    }


def _walk_op_values(op: Operation, ctx: _EncodeContext) -> None:
    """Recursively register every Value referenced by ``op``.

    Covers operands, results, subclass-extra Value fields (via
    ``all_input_values``), nested control-flow op bodies, and nested
    Blocks inside ``CompositeGateOperation`` /
    ``InverseBlockOperation`` /
    ``ControlledUOperation``.

    Args:
        op (Operation): The op to walk.
        ctx (_EncodeContext): The active encoding context.
    """
    from qamomile.circuit.ir.operation.control_flow import HasNestedOps

    for v in op.all_input_values():
        ctx.register_value(v)
    for v in op.results:
        if isinstance(v, ValueBase):
            ctx.register_value(v)
    if isinstance(op, HasNestedOps):
        for child_list in op.nested_op_lists():
            for child in child_list:
                _walk_op_values(child, ctx)
    if isinstance(op, CompositeGateOperation):
        if op.implementation_block is not None:
            _walk_block_values(op.implementation_block, ctx)
    if isinstance(op, InverseBlockOperation):
        if op.source_block is not None:
            _walk_block_values(op.source_block, ctx)
        if op.implementation_block is not None:
            _walk_block_values(op.implementation_block, ctx)
    if isinstance(op, ControlledUOperation) and op.block is not None:
        _walk_block_values(op.block, ctx)


def _walk_block_values(sub: Block, ctx: _EncodeContext) -> None:
    """Walk Values inside a nested Block embedded in an Operation.

    Args:
        sub (Block): The nested Block.
        ctx (_EncodeContext): The active encoding context.
    """
    for v in sub.input_values:
        ctx.register_value(v)
    for v in sub.parameters.values():
        ctx.register_value(v)
    for op in sub.operations:
        _walk_op_values(op, ctx)
    for v in sub.output_values:
        ctx.register_value(v)


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------


def _encode_value(v: ValueBase, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode a single ``Value`` / ``ArrayValue`` / ``TupleValue`` / ``DictValue``.

    Args:
        v (ValueBase): The IR value to encode.
        ctx (_EncodeContext): The active encoding context (used to
            register nested Value references that appear inside
            container values).

    Returns:
        dict[str, Any]: A Value dict with ``$type`` distinguishing
            the concrete class and UUID-based references for any
            nested Values.

    Raises:
        TypeError: If ``v`` is an unrecognized ValueBase subclass.
    """
    if isinstance(v, TupleValue):
        for elem in v.elements:
            ctx.register_value(elem)
        return {
            "$type": "TupleValue",
            "uuid": v.uuid,
            "logical_id": v.logical_id,
            "name": v.name,
            "metadata": _encode_metadata(v.metadata),
            "element_refs": [e.uuid for e in v.elements],
        }
    if isinstance(v, DictValue):
        for key, val in v.entries:
            ctx.register_value(key)
            ctx.register_value(val)
        return {
            "$type": "DictValue",
            "uuid": v.uuid,
            "logical_id": v.logical_id,
            "name": v.name,
            "metadata": _encode_metadata(v.metadata),
            "entry_refs": [(k.uuid, val.uuid) for k, val in v.entries],
        }
    if isinstance(v, ArrayValue):
        for dim in v.shape:
            ctx.register_value(dim)
        if v.slice_of is not None:
            ctx.register_value(v.slice_of)
        if v.slice_start is not None:
            ctx.register_value(v.slice_start)
        if v.slice_step is not None:
            ctx.register_value(v.slice_step)
        return {
            "$type": "ArrayValue",
            "uuid": v.uuid,
            "logical_id": v.logical_id,
            "name": v.name,
            "version": v.version,
            "value_type": _encode_value_type(v.type),
            "metadata": _encode_metadata(v.metadata),
            "shape_refs": [d.uuid for d in v.shape],
            "slice_of_ref": v.slice_of.uuid if v.slice_of is not None else None,
            "slice_start_ref": (
                v.slice_start.uuid if v.slice_start is not None else None
            ),
            "slice_step_ref": (v.slice_step.uuid if v.slice_step is not None else None),
        }
    if isinstance(v, Value):
        if v.parent_array is not None:
            ctx.register_value(v.parent_array)
        for idx in v.element_indices:
            ctx.register_value(idx)
        return {
            "$type": "Value",
            "uuid": v.uuid,
            "logical_id": v.logical_id,
            "name": v.name,
            "version": v.version,
            "value_type": _encode_value_type(v.type),
            "metadata": _encode_metadata(v.metadata),
            "parent_array_ref": (
                v.parent_array.uuid if v.parent_array is not None else None
            ),
            "element_index_refs": [idx.uuid for idx in v.element_indices],
        }
    raise TypeError(
        f"Cannot encode value of type {type(v).__name__}; "
        f"expected Value / ArrayValue / TupleValue / DictValue"
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _encode_metadata(metadata: ValueMetadata) -> dict[str, Any]:
    """Encode ``ValueMetadata`` into a dict.

    Sub-records that are ``None`` are emitted as ``None`` so the
    decoder can faithfully reconstruct the sparse shape.

    Args:
        metadata (ValueMetadata): The metadata bundle.

    Returns:
        dict[str, Any]: ``{"scalar": ..., "cast": ..., "qfixed": ...,
            "array_runtime": ..., "dict_runtime": ...}``.
    """
    return {
        "scalar": _encode_scalar_metadata(metadata.scalar),
        "cast": _encode_cast_metadata(metadata.cast),
        "qfixed": _encode_qfixed_metadata(metadata.qfixed),
        "array_runtime": _encode_array_runtime_metadata(metadata.array_runtime),
        "dict_runtime": _encode_dict_runtime_metadata(metadata.dict_runtime),
    }


def _encode_scalar_metadata(m: ScalarMetadata | None) -> dict[str, Any] | None:
    """Encode ``ScalarMetadata``.

    Args:
        m (ScalarMetadata | None): The scalar metadata or ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when the metadata is absent;
            otherwise a dict with ``const_value`` (primitive or
            ``None``) and ``parameter_name``.
    """
    if m is None:
        return None
    return {
        "const_value": _encode_payload(m.const_value),
        "parameter_name": m.parameter_name,
    }


def _encode_cast_metadata(m: CastMetadata | None) -> dict[str, Any] | None:
    """Encode ``CastMetadata`` (UUID references stay as strings).

    Args:
        m (CastMetadata | None): The cast metadata or ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when absent; else the dict
            form with the original UUIDs.
    """
    if m is None:
        return None
    return {
        "source_uuid": m.source_uuid,
        "qubit_uuids": list(m.qubit_uuids),
        "source_logical_id": m.source_logical_id,
        "qubit_logical_ids": list(m.qubit_logical_ids),
    }


def _encode_qfixed_metadata(m: QFixedMetadata | None) -> dict[str, Any] | None:
    """Encode ``QFixedMetadata``.

    Args:
        m (QFixedMetadata | None): The QFixed metadata or ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when absent; else the dict
            with ``qubit_uuids``, ``num_bits``, ``int_bits``.
    """
    if m is None:
        return None
    return {
        "qubit_uuids": list(m.qubit_uuids),
        "num_bits": m.num_bits,
        "int_bits": m.int_bits,
    }


def _encode_array_runtime_metadata(
    m: ArrayRuntimeMetadata | None,
) -> dict[str, Any] | None:
    """Encode ``ArrayRuntimeMetadata``.

    Args:
        m (ArrayRuntimeMetadata | None): The metadata bundle or
            ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when absent; else dict form
            with ``const_array`` (Python data, possibly numpy
            wrapper), UUID-string lists for element UUIDs /
            logical_ids, and the parallel per-element root
            ``(array_uuid, index)`` lists.
    """
    if m is None:
        return None
    return {
        "const_array": _encode_payload(m.const_array),
        "element_uuids": list(m.element_uuids),
        "element_logical_ids": list(m.element_logical_ids),
        "element_parent_uuids": list(m.element_parent_uuids),
        "element_parent_indices": list(m.element_parent_indices),
    }


def _encode_dict_runtime_metadata(
    m: DictRuntimeMetadata | None,
) -> dict[str, Any] | None:
    """Encode ``DictRuntimeMetadata``.

    Args:
        m (DictRuntimeMetadata | None): The metadata bundle or ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when absent; else dict with
            ``bound_data`` as a list of ``[key, value]`` pairs (both
            encoded via :func:`_encode_payload`).
    """
    if m is None:
        return None
    return {
        "bound_data": [
            [_encode_payload(k), _encode_payload(v)] for k, v in m.bound_data
        ],
    }


# ---------------------------------------------------------------------------
# ParamSlot
# ---------------------------------------------------------------------------


def _encode_param_slot(slot: ParamSlot) -> dict[str, Any]:
    """Encode a ``ParamSlot``.

    Args:
        slot (ParamSlot): The slot to encode.

    Returns:
        dict[str, Any]: Dict form with the slot's type encoded via
            :func:`_encode_value_type` and ``bound_value`` /
            ``default`` routed through :func:`_encode_payload`.
    """
    return {
        "name": slot.name,
        "type": _encode_value_type(slot.type),
        "kind": slot.kind.value,
        "ndim": slot.ndim,
        "default": _encode_payload(slot.default),
        "bound_value": _encode_payload(slot.bound_value),
        "differentiable": slot.differentiable,
    }


# ---------------------------------------------------------------------------
# Payload encoding for arbitrary Python data in metadata / param slots
# ---------------------------------------------------------------------------


def _encode_payload(value: Any) -> Any:
    """Encode a Python payload (binding value / const) into a JSON-able form.

    Supports primitives (``None``, ``bool``, ``int``, ``float``,
    ``str``), homogeneous containers (``list``, ``tuple``, ``dict``),
    numpy arrays, ``numpy`` scalar types, and
    ``qamomile.observable.Hamiltonian`` (the bound value of an
    ``Observable`` kernel parameter). Falls through to raising
    ``TypeError`` for unknown types so an unencodable binding never
    silently slips into the wire format.

    Args:
        value (Any): The Python value to encode.

    Returns:
        Any: A JSON / msgpack-friendly representation of ``value``.

    Raises:
        TypeError: If ``value`` has no known wire representation.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        # bytes pass through; JSON encoder converts to base64 at the boundary.
        return value
    if isinstance(value, np.ndarray):
        return array_to_dict(value)
    if isinstance(value, np.generic):
        # Cast numpy scalar to its closest Python primitive.
        return value.item()
    if isinstance(value, Hamiltonian):
        return hamiltonian_to_dict(value)
    if isinstance(value, (list, tuple)):
        return [_encode_payload(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _encode_payload(v) for k, v in value.items()}
    raise TypeError(
        f"Cannot encode payload of type {type(value).__name__!r}; supported types "
        f"are primitives, bytes, list/tuple, dict, np.ndarray, np.generic, "
        f"Hamiltonian."
    )


# ---------------------------------------------------------------------------
# ValueType encoding (closed dispatch)
# ---------------------------------------------------------------------------


def _encode_value_type(t: ValueType) -> dict[str, Any]:
    """Encode a ``ValueType`` instance into a tagged dict.

    Parameterless types serialize as ``{"$type": "FloatType"}``;
    parametric types ( ``TupleType``, ``DictType``, ``QFixedType``,
    ``QUIntType``) include their parameters.

    Args:
        t (ValueType): The IR type.

    Returns:
        dict[str, Any]: A tagged dict.

    Raises:
        TypeError: If ``t`` is an unknown ``ValueType`` subclass.
    """
    if isinstance(t, TupleType):
        return {
            "$type": "TupleType",
            "element_types": [_encode_value_type(et) for et in t.element_types],
        }
    if isinstance(t, DictType):
        return {
            "$type": "DictType",
            "key_type": _encode_value_type(t.key_type) if t.key_type else None,
            "value_type": _encode_value_type(t.value_type) if t.value_type else None,
        }
    if isinstance(t, QFixedType):
        return {
            "$type": "QFixedType",
            "integer_bits": _encode_qreg_width(t.integer_bits),
            "fractional_bits": _encode_qreg_width(t.fractional_bits),
        }
    if isinstance(t, QUIntType):
        return {
            "$type": "QUIntType",
            "width": _encode_qreg_width(t.width),
        }
    # Parameterless types
    for cls, tag in (
        (UIntType, "UIntType"),
        (FloatType, "FloatType"),
        (BitType, "BitType"),
        (QubitType, "QubitType"),
        (BlockType, "BlockType"),
        (ObservableType, "ObservableType"),
    ):
        if isinstance(t, cls):
            return {"$type": tag}
    raise TypeError(f"Cannot encode ValueType of type {type(t).__name__}")


def _encode_qreg_width(width: Any) -> Any:
    """Encode a QUIntType / QFixedType width field.

    The width is either a concrete ``int`` or a symbolic ``Value``.
    Concrete widths pass through as integers. Symbolic widths are
    encoded as a UUID reference so the decoder can resolve them
    against ``value_table`` like any other Value reference.

    Args:
        width (Any): An ``int`` or a symbolic ``Value``.

    Returns:
        Any: ``int`` for concrete widths; ``{"$value_ref": <uuid>}``
            for symbolic widths.

    Raises:
        TypeError: If the width is neither.
    """
    if is_plain_int(width):
        return width
    if isinstance(width, Value):
        return {"$value_ref": width.uuid}
    raise TypeError(
        f"QUInt/QFixed width must be int or Value, got {type(width).__name__}"
    )


# ---------------------------------------------------------------------------
# Operation encoding (closed dispatch on the concrete class)
# ---------------------------------------------------------------------------


def _encode_operation(op: Operation, ctx: _EncodeContext) -> dict[str, Any]:
    """Dispatch on the operation's concrete class to its encoder.

    Args:
        op (Operation): The operation to encode.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: A tagged op dict.

    Raises:
        NotImplementedError: If ``op`` is a ``CallBlockOperation``
            (HIERARCHICAL-only).
        TypeError: If ``op`` has no encoder in the dispatch table.
    """
    if isinstance(op, CallBlockOperation):
        raise NotImplementedError(
            "Cannot serialize CallBlockOperation; the block must be inlined first."
        )
    encoder = _OP_ENCODERS.get(type(op))
    if encoder is None:
        raise TypeError(
            f"Cannot encode operation of type {type(op).__name__}; no encoder "
            f"registered in the serializer's dispatch table."
        )
    return encoder(op, ctx)


def _operand_refs(op: Operation) -> list[str]:
    """Return the UUIDs of an operation's positional operands.

    Args:
        op (Operation): The op.

    Returns:
        list[str]: One UUID per entry in ``op.operands``.
    """
    return [v.uuid for v in op.operands]


def _result_refs(op: Operation) -> list[str]:
    """Return the UUIDs of an operation's results.

    Args:
        op (Operation): The op.

    Returns:
        list[str]: One UUID per entry in ``op.results``.
    """
    return [v.uuid for v in op.results]


def _base_op_dict(tag: str, op: Operation) -> dict[str, Any]:
    """Build the shared ``{$type, operand_refs, result_refs}`` skeleton.

    Args:
        tag (str): The ``$type`` tag for the op.
        op (Operation): The op being encoded.

    Returns:
        dict[str, Any]: A new dict with the shared base fields.
    """
    return {
        "$type": tag,
        "operand_refs": _operand_refs(op),
        "result_refs": _result_refs(op),
    }


def _encode_gate_operation(op: GateOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`GateOperation`.

    Args:
        op (GateOperation): The gate op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``gate_type`` enum name.
    """
    d = _base_op_dict("GateOperation", op)
    assert op.gate_type is not None
    d["gate_type"] = op.gate_type.name
    return d


def _encode_measure_operation(
    op: MeasureOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`MeasureOperation`.

    Args:
        op (MeasureOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("MeasureOperation", op)


def _encode_measure_vector(
    op: MeasureVectorOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`MeasureVectorOperation`.

    Args:
        op (MeasureVectorOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("MeasureVectorOperation", op)


def _encode_measure_qfixed(
    op: MeasureQFixedOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`MeasureQFixedOperation`.

    Args:
        op (MeasureQFixedOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``num_bits`` and ``int_bits``.
    """
    d = _base_op_dict("MeasureQFixedOperation", op)
    d["num_bits"] = op.num_bits
    d["int_bits"] = op.int_bits
    return d


def _encode_decode_qfixed(
    op: DecodeQFixedOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`DecodeQFixedOperation`.

    Args:
        op (DecodeQFixedOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``num_bits`` and ``int_bits``.
    """
    d = _base_op_dict("DecodeQFixedOperation", op)
    d["num_bits"] = op.num_bits
    d["int_bits"] = op.int_bits
    return d


def _encode_store_array_element(
    op: StoreArrayElementOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`StoreArrayElementOperation`.

    Args:
        op (StoreArrayElementOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict (the op carries no extra fields).
    """
    return _base_op_dict("StoreArrayElementOperation", op)


def _encode_dict_getitem(
    op: DictGetItemOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`DictGetItemOperation`.

    Args:
        op (DictGetItemOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``key_arity``.
    """
    d = _base_op_dict("DictGetItemOperation", op)
    d["key_arity"] = op.key_arity
    return d


def _encode_cast(op: CastOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`CastOperation`.

    Args:
        op (CastOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus source/target types and the
            UUID-string qubit mapping list.
    """
    d = _base_op_dict("CastOperation", op)
    d["source_type"] = (
        _encode_value_type(op.source_type) if op.source_type is not None else None
    )
    d["target_type"] = (
        _encode_value_type(op.target_type) if op.target_type is not None else None
    )
    d["qubit_mapping"] = list(op.qubit_mapping)
    return d


def _encode_qinit(op: QInitOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`QInitOperation`.

    Args:
        op (QInitOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("QInitOperation", op)


def _encode_cinit(op: CInitOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`CInitOperation`.

    Args:
        op (CInitOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("CInitOperation", op)


def _encode_slice_array(op: SliceArrayOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`SliceArrayOperation`.

    Args:
        op (SliceArrayOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("SliceArrayOperation", op)


def _encode_release_slice_view(
    op: ReleaseSliceViewOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`ReleaseSliceViewOperation`.

    Args:
        op (ReleaseSliceViewOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("ReleaseSliceViewOperation", op)


def _encode_return(op: ReturnOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ReturnOperation`.

    Args:
        op (ReturnOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("ReturnOperation", op)


def _encode_expval(op: ExpvalOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ExpvalOp`.

    Args:
        op (ExpvalOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("ExpvalOp", op)


def _encode_pauli_evolve(op: PauliEvolveOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`PauliEvolveOp`.

    Args:
        op (PauliEvolveOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("PauliEvolveOp", op)


def _encode_binop(op: BinOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`BinOp`.

    Args:
        op (BinOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus the ``BinOpKind`` enum name.
    """
    d = _base_op_dict("BinOp", op)
    assert op.kind is not None
    d["kind"] = op.kind.name
    return d


def _encode_compop(op: CompOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`CompOp`.

    Args:
        op (CompOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus the ``CompOpKind`` enum name.
    """
    d = _base_op_dict("CompOp", op)
    assert op.kind is not None
    d["kind"] = op.kind.name
    return d


def _encode_condop(op: CondOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`CondOp`.

    Args:
        op (CondOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus the ``CondOpKind`` enum name.
    """
    d = _base_op_dict("CondOp", op)
    assert op.kind is not None
    d["kind"] = op.kind.name
    return d


def _encode_notop(op: NotOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`NotOp`.

    Args:
        op (NotOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("NotOp", op)


def _encode_runtime_classical(
    op: RuntimeClassicalExpr, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`RuntimeClassicalExpr`.

    Args:
        op (RuntimeClassicalExpr): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``RuntimeOpKind`` name.
    """
    d = _base_op_dict("RuntimeClassicalExpr", op)
    assert op.kind is not None
    d["kind"] = op.kind.name
    return d


def _encode_phi(op: PhiOp, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`PhiOp`.

    Args:
        op (PhiOp): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("PhiOp", op)


def _encode_loop_carried_rebinds(
    rebinds: tuple[LoopCarriedRebind, ...],
) -> list[dict[str, Any]]:
    """Encode loop-carried rebind records as value references.

    Args:
        rebinds (tuple[LoopCarriedRebind, ...]): Records attached to a
            loop operation. Their ``before`` / ``after`` values are
            already registered in the value table via
            ``all_input_values``.

    Returns:
        list[dict[str, Any]]: One dict per record with ``var_name``,
            ``before_ref`` / ``after_ref`` UUIDs, and
            ``before_synthesized``.
    """
    return [
        {
            "var_name": r.var_name,
            "before_ref": r.before.uuid,
            "after_ref": r.after.uuid,
            "before_synthesized": r.before_synthesized,
        }
        for r in rebinds
    ]


def _encode_for(op: ForOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ForOperation`.

    Args:
        op (ForOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``loop_var`` display name,
            ``loop_var_value_ref`` (or ``None``), the
            ``loop_carried_rebinds`` record list, and ``body`` op list.
    """
    d = _base_op_dict("ForOperation", op)
    d["loop_var"] = op.loop_var
    d["loop_var_value_ref"] = (
        op.loop_var_value.uuid if op.loop_var_value is not None else None
    )
    d["loop_carried_rebinds"] = _encode_loop_carried_rebinds(op.loop_carried_rebinds)
    d["body"] = [_encode_operation(child, ctx) for child in op.operations]
    return d


def _encode_for_items(op: ForItemsOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ForItemsOperation`.

    Args:
        op (ForItemsOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus display key/value var names,
            their UUID refs (or ``None`` for legacy IR), the
            ``key_is_vector`` flag, and the ``body`` op list.
    """
    d = _base_op_dict("ForItemsOperation", op)
    d["key_vars"] = list(op.key_vars)
    d["value_var"] = op.value_var
    d["key_is_vector"] = op.key_is_vector
    d["key_var_value_refs"] = (
        [v.uuid for v in op.key_var_values] if op.key_var_values is not None else None
    )
    d["value_var_value_ref"] = (
        op.value_var_value.uuid if op.value_var_value is not None else None
    )
    d["loop_carried_rebinds"] = _encode_loop_carried_rebinds(op.loop_carried_rebinds)
    d["body"] = [_encode_operation(child, ctx) for child in op.operations]
    return d


def _encode_while(op: WhileOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`WhileOperation`.

    Args:
        op (WhileOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``max_iterations`` and the
            loop ``body``.
    """
    d = _base_op_dict("WhileOperation", op)
    d["max_iterations"] = op.max_iterations
    d["loop_carried_rebinds"] = _encode_loop_carried_rebinds(op.loop_carried_rebinds)
    d["body"] = [_encode_operation(child, ctx) for child in op.operations]
    return d


def _encode_branch_rebinds(
    rebinds: tuple[BranchRebind, ...],
) -> list[dict[str, Any]]:
    """Encode branch rebind records as value references.

    Args:
        rebinds (tuple[BranchRebind, ...]): Records attached to an
            ``IfOperation``. Their ``before`` values are already
            registered in the value table via ``all_input_values``.

    Returns:
        list[dict[str, Any]]: One dict per record with ``var_name``,
            ``before_ref`` UUID, and the ``rebound_in_true`` /
            ``rebound_in_false`` flags.
    """
    return [
        {
            "var_name": r.var_name,
            "before_ref": r.before.uuid,
            "rebound_in_true": r.rebound_in_true,
            "rebound_in_false": r.rebound_in_false,
        }
        for r in rebinds
    ]


def _encode_if(op: IfOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`IfOperation`.

    Args:
        op (IfOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``true_body`` /
            ``false_body`` operation lists, a parallel ``phi_ops``
            list, and the ``branch_rebinds`` record list.
    """
    d = _base_op_dict("IfOperation", op)
    d["true_body"] = [_encode_operation(child, ctx) for child in op.true_operations]
    d["false_body"] = [_encode_operation(child, ctx) for child in op.false_operations]
    d["phi_ops"] = [_encode_operation(p, ctx) for p in op.phi_ops]
    d["branch_rebinds"] = _encode_branch_rebinds(op.branch_rebinds)
    return d


def _encode_concrete_controlled(
    op: ConcreteControlledU, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`ConcreteControlledU`.

    Args:
        op (ConcreteControlledU): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``num_controls`` (concrete
            int), ``power`` (int or value-ref), and a nested
            ``unitary_block`` dict.
    """
    d = _base_op_dict("ConcreteControlledU", op)
    d["num_controls"] = op.num_controls
    d["power"] = _encode_power(op.power)
    # ``control_values`` carries the zero-control (anti-control) pattern.
    # Persist only when non-empty so standard-control payloads stay
    # byte-compatible with pre-zero-control encoders.
    if op.control_values:
        d["control_values"] = list(op.control_values)
    d["unitary_block"] = _encode_block(op.block, ctx) if op.block is not None else None
    return d


def _encode_symbolic_controlled(
    op: SymbolicControlledU, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`SymbolicControlledU`.

    Args:
        op (SymbolicControlledU): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``num_controls_ref``
            (symbolic Value's UUID), ``power``, ``control_index_refs``
            (per-element ``UInt`` Value UUIDs, or ``None`` when the op
            uses the entire pool), ``num_control_args`` (count of
            positional control arguments at the call site -- the legacy
            single-pool form is ``1``, the multi-arg control prefix
            stores the actual N), and nested ``unitary_block`` dict.
    """
    d = _base_op_dict("SymbolicControlledU", op)
    ctx.register_value(op.num_controls)
    d["num_controls_ref"] = op.num_controls.uuid
    d["power"] = _encode_power(op.power)
    if op.control_indices is not None:
        for v in op.control_indices:
            ctx.register_value(v)
        d["control_index_refs"] = [v.uuid for v in op.control_indices]
    else:
        d["control_index_refs"] = None
    # ``num_control_args`` tracks how many positional control arguments
    # the call site supplied (one ArrayBase pool in the legacy form;
    # any sequence of scalar Qubits and / or ArrayBases in the multi-
    # arg form).  The emit pass uses it to split ``operands`` into the
    # control prefix vs the sub-kernel quantum tail, so a wrong default
    # at decode time shifts the boundary and corrupts the operand
    # layout.  Persist the field whenever it differs from the legacy
    # default of 1 so existing v1 payloads stay readable.
    if op.num_control_args != 1:
        d["num_control_args"] = op.num_control_args
    d["unitary_block"] = _encode_block(op.block, ctx) if op.block is not None else None
    return d


def _encode_power(power: Any) -> Any:
    """Encode ControlledU's ``power`` field.

    Args:
        power (Any): An ``int`` or a symbolic ``Value``.

    Returns:
        Any: ``int`` for concrete powers; ``{"$value_ref": <uuid>}``
            for symbolic powers.

    Raises:
        TypeError: If ``power`` is neither.
    """
    if is_plain_int(power):
        return power
    if isinstance(power, Value):
        return {"$value_ref": power.uuid}
    raise TypeError(
        f"ControlledU.power must be int or Value, got {type(power).__name__}"
    )


def _encode_select(op: SelectOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`SelectOperation`.

    Args:
        op (SelectOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``num_index_qubits`` and the
            list of nested per-case unitary block dicts.
    """
    d = _base_op_dict("SelectOperation", op)
    d["num_index_qubits"] = op.num_index_qubits
    d["case_blocks"] = [_encode_block(b, ctx) for b in op.case_blocks]
    return d


def _encode_composite_gate(
    op: CompositeGateOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`CompositeGateOperation`.

    Args:
        op (CompositeGateOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus gate-type enum, control /
            target counts, custom name, strategy, resource metadata,
            and the nested implementation Block dict (or ``None``).
    """
    d = _base_op_dict("CompositeGateOperation", op)
    d["gate_type"] = op.gate_type.name
    d["num_control_qubits"] = op.num_control_qubits
    d["num_target_qubits"] = op.num_target_qubits
    d["custom_name"] = op.custom_name
    d["has_implementation"] = op.has_implementation
    d["strategy_name"] = op.strategy_name
    d["resource_metadata"] = _encode_resource_metadata(op.resource_metadata)
    d["implementation_block"] = (
        _encode_block(op.implementation_block, ctx)
        if op.implementation_block is not None
        else None
    )
    # ``composite_gate_instance`` is an opaque Python callable; it is
    # intentionally not serialized. The receiver reconstructs the gate
    # from the implementation block (or via a registry, future work).
    return d


def _encode_inverse_block(
    op: InverseBlockOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`InverseBlockOperation`.

    Args:
        op (InverseBlockOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus control / target counts, custom
            name, source block, and fallback implementation block.
    """
    d = _base_op_dict("InverseBlockOperation", op)
    d["num_control_qubits"] = op.num_control_qubits
    d["num_target_qubits"] = op.num_target_qubits
    d["custom_name"] = op.custom_name
    d["source_block"] = (
        _encode_block(op.source_block, ctx) if op.source_block is not None else None
    )
    d["implementation_block"] = (
        _encode_block(op.implementation_block, ctx)
        if op.implementation_block is not None
        else None
    )
    return d


def _encode_resource_metadata(
    m: ResourceMetadata | None,
) -> dict[str, Any] | None:
    """Encode :class:`ResourceMetadata`.

    Args:
        m (ResourceMetadata | None): The resource metadata or ``None``.

    Returns:
        dict[str, Any] | None: ``None`` when absent; else a dict with
            all numeric fields plus ``custom_metadata`` routed
            through :func:`_encode_payload`.
    """
    if m is None:
        return None
    return {
        "query_complexity": m.query_complexity,
        "t_gates": m.t_gates,
        "ancilla_qubits": m.ancilla_qubits,
        "total_gates": m.total_gates,
        "single_qubit_gates": m.single_qubit_gates,
        "two_qubit_gates": m.two_qubit_gates,
        "multi_qubit_gates": m.multi_qubit_gates,
        "clifford_gates": m.clifford_gates,
        "rotation_gates": m.rotation_gates,
        "custom_metadata": _encode_payload(m.custom_metadata),
    }


_OP_ENCODERS: dict[type, Callable[[Any, _EncodeContext], dict[str, Any]]] = {
    GateOperation: _encode_gate_operation,
    MeasureOperation: _encode_measure_operation,
    MeasureVectorOperation: _encode_measure_vector,
    MeasureQFixedOperation: _encode_measure_qfixed,
    DecodeQFixedOperation: _encode_decode_qfixed,
    DictGetItemOperation: _encode_dict_getitem,
    StoreArrayElementOperation: _encode_store_array_element,
    CastOperation: _encode_cast,
    QInitOperation: _encode_qinit,
    CInitOperation: _encode_cinit,
    SliceArrayOperation: _encode_slice_array,
    ReleaseSliceViewOperation: _encode_release_slice_view,
    ReturnOperation: _encode_return,
    ExpvalOp: _encode_expval,
    PauliEvolveOp: _encode_pauli_evolve,
    BinOp: _encode_binop,
    CompOp: _encode_compop,
    CondOp: _encode_condop,
    NotOp: _encode_notop,
    RuntimeClassicalExpr: _encode_runtime_classical,
    PhiOp: _encode_phi,
    ForOperation: _encode_for,
    ForItemsOperation: _encode_for_items,
    WhileOperation: _encode_while,
    IfOperation: _encode_if,
    ConcreteControlledU: _encode_concrete_controlled,
    SymbolicControlledU: _encode_symbolic_controlled,
    SelectOperation: _encode_select,
    CompositeGateOperation: _encode_composite_gate,
    InverseBlockOperation: _encode_inverse_block,
}
