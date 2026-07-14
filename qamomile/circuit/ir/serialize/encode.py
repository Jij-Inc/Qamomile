"""Semantic IR block → private graph-record encoder.

Walks a :class:`Block` and produces the internal graph records consumed by
:mod:`qamomile.circuit.serialization.graph_protobuf`. Values and callable
definitions are deduplicated into module-wide tables and referenced elsewhere
by stable IDs. Keeping those registries outside individual blocks preserves
recursive call graphs and shared callable identity without expanding the
high-level IR into backend-specific instructions.

Every encoder branch is dispatched through a hard-coded table keyed
on the runtime class; there is no dynamic resolution, no ``getattr``
on user data, and no ``importlib`` use. The decoder mirrors this
discipline (see :mod:`qamomile.circuit.ir.serialize.decode`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qamomile._utils import is_plain_int
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    CallableBodyRef,
    CallableDef,
    CallableImplementation,
    CallableRef,
    ControlledUOperation,
    ExpvalOp,
    ForItemsOperation,
    GateOperation,
    InverseBlockOperation,
    InvokeOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    Operation,
    ProjectOperation,
    ResetOperation,
    ReturnOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import (
    DecodeQFixedOperation,
    DictGetItemOperation,
    StoreArrayElementOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    RegionArg,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import (
    CInitOperation,
    QInitOperation,
    Signature,
)
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
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
from .numpy_io import array_to_dict, scalar_to_dict

# ---------------------------------------------------------------------------
# Context: collects unique Values into a flat table
# ---------------------------------------------------------------------------


class _EncodeContext:
    """Collect module-wide Value and CallableDef registries.

    Values are keyed by semantic UUID. Callable definitions are keyed by
    object identity because one :class:`CallableRef` may intentionally identify
    more than one semantic body in a surrounding module.
    """

    def __init__(self) -> None:
        """Initialize an empty encode context."""
        self.value_table_dicts: list[dict[str, Any]] = []
        self._seen_uuids: set[str] = set()
        self._definition_ids: dict[int, str] = {}
        self._definitions: list[CallableDef] = []

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

    def register_definition(self, definition: CallableDef | None) -> str | None:
        """Register a callable definition and every reachable semantic body.

        The ID is assigned before walking bodies so direct and mutual recursion
        terminate naturally and serialize as graph edges.

        Args:
            definition (CallableDef | None): Definition to register, or
                ``None`` for an invocation without an explicit definition.

        Returns:
            str | None: Module-local definition ID, or ``None``.
        """
        if definition is None:
            return None
        object_id = id(definition)
        existing = self._definition_ids.get(object_id)
        if existing is not None:
            return existing
        definition_id = f"callable_{len(self._definitions)}"
        self._definition_ids[object_id] = definition_id
        self._definitions.append(definition)
        if definition.body is not None:
            _walk_block_values(definition.body, self)
        for implementation in definition.implementations:
            if implementation.body is not None:
                _walk_block_values(implementation.body, self)
        return definition_id

    def encode_callable_table(self) -> list[dict[str, Any]]:
        """Encode every registered callable definition exactly once.

        Returns:
            list[dict[str, Any]]: Definition-table entries in deterministic
                first-visit order.
        """
        encoded: list[dict[str, Any]] = []
        index = 0
        while index < len(self._definitions):
            definition = self._definitions[index]
            definition_id = self._definition_ids[id(definition)]
            encoded.append(
                {
                    "id": definition_id,
                    "definition": _encode_callable_def(definition, self),
                }
            )
            index += 1
        return encoded


# ---------------------------------------------------------------------------
# Block / Operation / Value encoders
# ---------------------------------------------------------------------------


def _encode_block(block: Block, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode a Block dict.

    Args:
        block (Block): The block to encode.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Internal graph record for the block.
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
        "operations": [_encode_operation(op, ctx) for op in block.operations],
    }


def _walk_op_values(op: Operation, ctx: _EncodeContext) -> None:
    """Recursively register every Value referenced by ``op``.

    Covers operands, results, subclass-extra Value fields (via
    ``all_input_values``), nested control-flow op bodies, and nested
    Blocks inside ``InvokeOperation`` /
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
    if isinstance(op, InvokeOperation):
        ctx.register_definition(op.definition)
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
# Payload encoding for arbitrary Python data in metadata and defaults
# ---------------------------------------------------------------------------


def _encode_payload(value: Any) -> Any:
    """Encode a Python payload without collapsing container identity.

    Supports primitives (``None``, ``bool``, ``int``, ``float``,
    ``str``), bytes, complex numbers, lists, tuples, sets, arbitrary-key
    dictionaries, numpy arrays, ``numpy`` scalar types, and
    ``qamomile.observable.Hamiltonian`` (the bound value of an
    ``Observable`` kernel parameter). Dictionaries use a list-of-pairs
    ``$map`` wrapper, so arbitrary key types and reserved names remain distinct.

    Args:
        value (Any): The Python value to encode.

    Returns:
        Any: A closed intermediate representation for protobuf conversion.

    Raises:
        TypeError: If ``value`` has no known lossless wire representation.
    """
    # NumPy float64 subclasses Python float on some NumPy versions, so scalar
    # dispatch must precede the primitive branch to retain dtype and bits.
    if isinstance(value, np.generic):
        return scalar_to_dict(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        # Bytes pass through into protobuf's native bytes field.
        return value
    if isinstance(value, np.ndarray):
        return array_to_dict(value)
    if isinstance(value, Hamiltonian):
        return hamiltonian_to_dict(value)
    if isinstance(value, complex):
        return {
            "$complex_number": [float(value.real), float(value.imag)],
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode_payload(item) for item in value]}
    if isinstance(value, list):
        return [_encode_payload(x) for x in value]
    if isinstance(value, set):
        return {"$set": [_encode_payload(item) for item in value]}
    if isinstance(value, frozenset):
        return {"$frozenset": [_encode_payload(item) for item in value]}
    if isinstance(value, dict):
        return {
            "$map": [
                [_encode_payload(key), _encode_payload(item)]
                for key, item in value.items()
            ]
        }
    if callable(value):
        raise TypeError(
            "Cannot serialize a Python callable payload. Evaluate it while "
            "tracing or replace it with serializer-friendly semantic metadata."
        )
    raise TypeError(
        f"Cannot encode payload of type {type(value).__name__!r}; supported types "
        f"are primitives, bytes, complex, list/tuple/set/frozenset, dict, "
        f"np.ndarray, np.generic, and Hamiltonian."
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
        TypeError: If ``op`` has no encoder in the dispatch table.
    """
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


def _encode_project_operation(
    op: ProjectOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`ProjectOperation`.

    Args:
        op (ProjectOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus the projection axis.
    """
    d = _base_op_dict("ProjectOperation", op)
    d["axis"] = op.axis
    return d


def _encode_reset_operation(op: ResetOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ResetOperation`.

    Args:
        op (ResetOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict.
    """
    return _base_op_dict("ResetOperation", op)


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


def _encode_region_args(
    region_args: tuple[RegionArg, ...],
) -> list[dict[str, Any]]:
    """Encode loop region arguments as value references.

    Args:
        region_args (tuple[RegionArg, ...]): Records attached to a loop
            operation. Their ``init`` / ``block_arg`` / ``yielded`` /
            ``result`` values are already registered in the value table
            via ``all_input_values`` (and ``result`` via ``results``).

    Returns:
        list[dict[str, Any]]: One dict per record with ``var_name`` and
            the four UUID refs.
    """
    return [
        {
            "var_name": a.var_name,
            "init_ref": a.init.uuid,
            "block_arg_ref": a.block_arg.uuid,
            "yielded_ref": a.yielded.uuid,
            "result_ref": a.result.uuid,
        }
        for a in region_args
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
    d["region_args"] = _encode_region_args(op.region_args)
    d["body"] = [_encode_operation(child, ctx) for child in op.operations]
    return d


def _encode_for_items(op: ForItemsOperation, ctx: _EncodeContext) -> dict[str, Any]:
    """Encode :class:`ForItemsOperation`.

    Args:
        op (ForItemsOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus display key/value var names,
            their optional UUID refs, the
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
    d["region_args"] = _encode_region_args(op.region_args)
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
    d["region_args"] = _encode_region_args(op.region_args)
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

    Branch merges are encoded as two value-reference lists parallel to
    ``results``: ``true_yield_refs[i]`` / ``false_yield_refs[i]`` are the
    branch sources merged into ``results[i]``. The referenced Values are
    already registered in the value table by the block-wide value walk.

    Args:
        op (IfOperation): The op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus ``true_body`` /
            ``false_body`` operation lists, the parallel
            ``true_yield_refs`` / ``false_yield_refs`` UUID lists, and
            the ``branch_rebinds`` record list.
    """
    d = _base_op_dict("IfOperation", op)
    d["true_body"] = [_encode_operation(child, ctx) for child in op.true_operations]
    d["false_body"] = [_encode_operation(child, ctx) for child in op.false_operations]
    merges = list(op.iter_merges())
    d["true_yield_refs"] = [m.true_value.uuid for m in merges]
    d["false_yield_refs"] = [m.false_value.uuid for m in merges]
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
    if op.callable_ref is not None:
        d["callable_ref"] = _encode_callable_ref(op.callable_ref)
    if op.callable_attrs:
        d["callable_attrs"] = _encode_payload(op.callable_attrs)
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
            positional control arguments at the call site), and nested
            ``unitary_block`` dict.
    """
    d = _base_op_dict("SymbolicControlledU", op)
    ctx.register_value(op.num_controls)
    d["num_controls_ref"] = op.num_controls.uuid
    d["power"] = _encode_power(op.power)
    if op.callable_ref is not None:
        d["callable_ref"] = _encode_callable_ref(op.callable_ref)
    if op.callable_attrs:
        d["callable_attrs"] = _encode_payload(op.callable_attrs)
    if op.control_indices is not None:
        for v in op.control_indices:
            ctx.register_value(v)
        d["control_index_refs"] = [v.uuid for v in op.control_indices]
    else:
        d["control_index_refs"] = None
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


def _encode_callable_ref(ref: CallableRef) -> dict[str, str]:
    """Encode a callable reference.

    Args:
        ref (CallableRef): Callable reference to encode.

    Returns:
        dict[str, str]: Dict with namespace, name, and version.
    """
    return {
        "namespace": ref.namespace,
        "name": ref.name,
        "version": ref.version,
    }


def _encode_callable_body_ref(
    body_ref: CallableBodyRef | None,
) -> dict[str, Any] | None:
    """Encode a deferred callable body reference.

    Args:
        body_ref (CallableBodyRef | None): Body reference to encode.

    Returns:
        dict[str, Any] | None: Encoded reference payload, or ``None`` when
        absent.
    """
    if body_ref is None:
        return None
    return {
        "ref": _encode_callable_ref(body_ref.ref),
        "kind": body_ref.kind,
        "attrs": _encode_payload(body_ref.attrs),
    }


def _encode_callable_implementation(
    impl: CallableImplementation,
    ctx: _EncodeContext,
) -> dict[str, Any]:
    """Encode a callable implementation candidate.

    Args:
        impl (CallableImplementation): Implementation to encode.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Serialized implementation.
    """
    if impl.emitter is not None:
        raise TypeError(
            "CallableImplementation.emitter contains a process-local extension "
            "object and cannot be represented in semantic Qamomile IR. Standard "
            "Qamomile backend emitters are registered outside the serialized "
            "module. Store a backend/strategy/body_ref contract instead."
        )
    return {
        "transform": impl.transform.name,
        "backend": impl.backend,
        "strategy": impl.strategy,
        "body": _encode_block(impl.body, ctx) if impl.body is not None else None,
        "body_ref": _encode_callable_body_ref(impl.body_ref),
        "attrs": _encode_payload(impl.attrs),
    }


def _encode_signature(signature: Signature | None) -> dict[str, Any] | None:
    """Encode an operation signature carried by a callable definition.

    Args:
        signature (Signature | None): Signature to encode.

    Returns:
        dict[str, Any] | None: Encoded signature or ``None``.
    """
    if signature is None:
        return None
    return {
        "operands": [
            None
            if hint is None
            else {
                "name": hint.name,
                "type": _encode_value_type(hint.type),
            }
            for hint in signature.operands
        ],
        "results": [
            {
                "name": hint.name,
                "type": _encode_value_type(hint.type),
            }
            for hint in signature.results
        ],
    }


def _encode_callable_def(
    definition: CallableDef,
    ctx: _EncodeContext,
) -> dict[str, Any]:
    """Encode a callable definition.

    Args:
        definition (CallableDef): Definition to encode.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Serialized definition.
    """
    return {
        "ref": _encode_callable_ref(definition.ref),
        "signature": _encode_signature(definition.signature),
        "body": (
            _encode_block(definition.body, ctx) if definition.body is not None else None
        ),
        "body_ref": _encode_callable_body_ref(definition.body_ref),
        "implementations": [
            _encode_callable_implementation(impl, ctx)
            for impl in definition.implementations
        ],
        "default_policy": definition.default_policy.name,
        "attrs": _encode_payload(definition.attrs),
    }


def _encode_invoke_operation(
    op: InvokeOperation, ctx: _EncodeContext
) -> dict[str, Any]:
    """Encode :class:`InvokeOperation`.

    Args:
        op (InvokeOperation): The invocation op.
        ctx (_EncodeContext): The active encoding context.

    Returns:
        dict[str, Any]: Base op dict plus callable identity, transform,
            attrs, definition, and optional nested body.
    """
    d = _base_op_dict("InvokeOperation", op)
    d["target"] = _encode_callable_ref(op.target)
    d["transform"] = op.transform.name
    d["attrs"] = _encode_payload(op.attrs)
    d["definition_ref"] = ctx.register_definition(op.definition)
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
    if op.callable_ref is not None:
        d["callable_ref"] = _encode_callable_ref(op.callable_ref)
    if op.callable_attrs:
        d["callable_attrs"] = _encode_payload(op.callable_attrs)
    d["source_block"] = (
        _encode_block(op.source_block, ctx) if op.source_block is not None else None
    )
    d["implementation_block"] = (
        _encode_block(op.implementation_block, ctx)
        if op.implementation_block is not None
        else None
    )
    return d


_OP_ENCODERS: dict[type, Callable[[Any, _EncodeContext], dict[str, Any]]] = {
    GateOperation: _encode_gate_operation,
    MeasureOperation: _encode_measure_operation,
    ProjectOperation: _encode_project_operation,
    ResetOperation: _encode_reset_operation,
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
    ForOperation: _encode_for,
    ForItemsOperation: _encode_for_items,
    WhileOperation: _encode_while,
    IfOperation: _encode_if,
    ConcreteControlledU: _encode_concrete_controlled,
    SymbolicControlledU: _encode_symbolic_controlled,
    InvokeOperation: _encode_invoke_operation,
    InverseBlockOperation: _encode_inverse_block,
}
