"""Canonical form for IR blocks: deterministic UUIDs and content hashing.

This module provides a normalization pass that re-numbers every Value
UUID (and ``logical_id``) in a Block from a deterministic counter,
rewriting every UUID reference embedded in operations and value
metadata so the resulting Block is structurally invariant across
independent builds of the same kernel.

The canonical form is intended for IR-level equality checks, debugging
diffs, and a content-addressable identity (``content_hash``) suitable
for caching and (later) for serialization keyed on IR contents rather
than build-local Python state.

Supported scope:
    Only ``BlockKind.AFFINE`` and ``BlockKind.ANALYZED`` are accepted.
    ``HIERARCHICAL`` Blocks still contain ``CallBlockOperation``s that
    reference sibling Blocks by Python identity; their canonical
    treatment is deferred (see backlog ``[IR design] Add named/versioned
    module references for cross-process kernel composition``).

Output guarantees:
    1. ``canonicalize`` does not change ``Block.kind``; it is a
       normalization, not a pipeline stage advance.
    2. For two builds of the same kernel that produce structurally
       identical IR, ``canonicalize`` returns Blocks that are equal
       under ``to_canonical_bytes`` (and therefore under
       ``content_hash``).
    3. ``canonicalize`` is idempotent: running it twice on the same
       Block yields the same canonical bytes as running it once.

Canonical-bytes scope:
    Display-only fields (``Block.name``, ``Block.output_names``,
    ``Value.name``) are excluded from ``to_canonical_bytes``;
    functional fields are included. ``Block.label_args`` (input port
    names by position) and ``Block.param_slots`` (the kernel's
    classical parameter contract) are both functional: two Blocks
    whose operations match but whose parameter manifests differ (e.g.,
    a slot rebound from ``RUNTIME_PARAMETER`` to
    ``COMPILE_TIME_BOUND``) hash differently. ``param_slots`` holds no
    Value/UUID references, so ``canonicalize`` carries the tuple over
    verbatim with nothing to remap.

Limitations:
    - ``Value.parent_array`` cycles (a Value whose ``parent_array`` is
      itself reachable from the Value's siblings) are not constructed
      by the frontend today; the canonicalizer assumes the
      Value-reference graph through ``parent_array`` and
      ``element_indices`` is acyclic.
    - ``ValueMetadata.dict_runtime.bound_data``,
      ``ArrayRuntimeMetadata.const_array``, and
      ``ParamSlot.default`` / ``ParamSlot.bound_value`` may carry
      arbitrary frozen Python data. ``numpy.ndarray`` payloads (except
      object-dtype arrays) are hashed from their raw buffer
      (dtype + shape + bytes) and are therefore exempt; hash stability
      for any other object type requires a stable ``repr``.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import uuid as _uuid
from typing import Any, cast

import numpy as np

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.global_phase_block import GlobalPhaseBlockOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.types.primitives import ValueType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
    ValueMetadata,
    remap_indexed_identifier,
    remap_value_metadata_references,
)

_SUPPORTED_KINDS = frozenset({BlockKind.AFFINE, BlockKind.ANALYZED})


def canonicalize(block: Block) -> Block:
    """Return a canonical-form clone of ``block``.

    The returned Block has the same structure as ``block`` but with
    every Value UUID and ``logical_id`` re-issued from a deterministic
    counter. All UUID references inside operations and value metadata
    are rewritten consistently. ``Block.kind`` is preserved.

    Args:
        block (Block): The block to canonicalize. Must be at
            ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED``.

    Returns:
        Block: A new Block with canonical UUIDs. ``Block.kind`` matches
            the input. Existing input/output ordering, operation
            ordering, metadata structure, and the ``param_slots``
            manifest (carried over verbatim) are preserved.

    Raises:
        ValueError: If ``block.kind`` is not in ``{AFFINE, ANALYZED}``.
        NotImplementedError: If a ``CallBlockOperation`` is encountered
            (typically because ``inline`` has not been run yet).

    Example:
        >>> from qamomile.qiskit import QiskitTranspiler
        >>> transpiler = QiskitTranspiler()
        >>> affine = transpiler.inline(transpiler.to_block(my_kernel))
        >>> canon = canonicalize(affine)
        >>> canon.kind is affine.kind
        True
    """
    return canonicalize_and_remap(block)[0]


def canonicalize_and_remap(
    block: Block,
) -> tuple[Block, dict[str, str], dict[str, str]]:
    """Return canonical-form Block plus the UUID and logical_id remap tables.

    Useful when the caller holds external references keyed on the
    original Value UUIDs or logical_ids (e.g., a host-side port map)
    and needs to update those references to match the canonical form.

    Args:
        block (Block): The block to canonicalize. Must be at
            ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED``.

    Returns:
        tuple[Block, dict[str, str], dict[str, str]]: A triple
            ``(canonical_block, uuid_remap, logical_id_remap)`` where
            each remap maps every original identifier encountered
            during the walk to its canonical counterpart. ``uuid`` and
            ``logical_id`` share the same monotonic counter but are
            tracked in separate maps.

    Raises:
        ValueError: If ``block.kind`` is not in ``{AFFINE, ANALYZED}``.
        NotImplementedError: If a ``CallBlockOperation`` is encountered.
    """
    if block.kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"canonicalize() requires BlockKind.AFFINE or BlockKind.ANALYZED; "
            f"got {block.kind.name}. Run `inline` (and optionally `analyze`) "
            f"first."
        )
    canon = _Canonicalizer()
    new_block = canon.canonical_block(block)
    return (
        new_block,
        dict(canon.uuid_remap),
        dict(canon.logical_id_remap),
    )


def to_canonical_bytes(block: Block) -> bytes:
    """Serialize ``block`` to a deterministic byte representation.

    The byte format is the **internal** representation backing
    ``content_hash`` and is not stable across qamomile versions. It is
    suitable for hashing and equality checks within a single
    deployment but should not be relied upon as a serialization
    format. (A stable, versioned serialization format is tracked
    separately.)

    Args:
        block (Block): The block to serialize. Must be at
            ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED``. The block
            is canonicalized first; passing an already-canonical block
            is harmless.

    Returns:
        bytes: A UTF-8-encoded byte string. Two structurally-equal
            Blocks produce the same bytes; changing the IR yields
            different bytes.

    Raises:
        ValueError: If ``block.kind`` is not in ``{AFFINE, ANALYZED}``.
    """
    canon = canonicalize(block)
    lines: list[str] = []
    _emit_block(canon, lines, indent=0)
    return "\n".join(lines).encode("utf-8")


def content_hash(block: Block) -> str:
    """Compute a content-addressable hash of ``block``.

    Two Blocks that canonicalize to the same form (structurally equal
    after UUID remapping) produce the same hash. Any IR-level change
    (gate added, parameter renamed, operand reordered) produces a
    different hash.

    Args:
        block (Block): The block to hash. Must be at
            ``BlockKind.AFFINE`` or ``BlockKind.ANALYZED``.

    Returns:
        str: The SHA-256 hex digest of ``to_canonical_bytes(block)``.

    Raises:
        ValueError: If ``block.kind`` is not in ``{AFFINE, ANALYZED}``.

    Example:
        >>> h1 = content_hash(canonicalize(affine_a))
        >>> h2 = content_hash(canonicalize(affine_b))
        >>> # If the two kernels are structurally identical, h1 == h2.
    """
    return hashlib.sha256(to_canonical_bytes(block)).hexdigest()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _Canonicalizer:
    """Walker that assigns deterministic UUIDs by traversal order.

    A single ``_Canonicalizer`` covers one ``canonicalize`` invocation
    and may recurse into nested Blocks (for example via
    ``CompositeGateOperation.implementation_block`` and
    ``InverseBlockOperation`` source/fallback blocks). A shared counter
    across nested Blocks guarantees no UUID string collisions within
    the returned canonical tree.
    """

    def __init__(self) -> None:
        """Initialize an empty Canonicalizer state.

        Sets up the shared monotonic counter that drives ``_next_id`` and
        empty remap / cache dictionaries that fill in as Values, Blocks,
        and metadata UUID references are visited during the walk.
        """
        self._counter = 0
        self._uuid_remap: dict[str, str] = {}
        self._logical_id_remap: dict[str, str] = {}
        self._value_cache: dict[str, ValueBase] = {}
        self._block_cache: dict[int, Block] = {}

    @property
    def uuid_remap(self) -> dict[str, str]:
        """Return the accumulated old-UUID to canonical-UUID map.

        Returns:
            dict[str, str]: Snapshot reference (not a copy) of the live
                remap table; callers that need a stable copy should
                wrap it themselves.
        """
        return self._uuid_remap

    @property
    def logical_id_remap(self) -> dict[str, str]:
        """Return the accumulated old-logical_id to canonical-logical_id map.

        Returns:
            dict[str, str]: Snapshot reference (not a copy) of the live
                remap table; callers that need a stable copy should
                wrap it themselves.
        """
        return self._logical_id_remap

    def _next_id(self) -> str:
        """Mint the next deterministic UUID string from the counter.

        Returns:
            str: A canonical UUID string of the form
                ``00000000-0000-0000-0000-XXXXXXXXXXXX`` where the
                trailing portion encodes the current counter value.
        """
        ident = str(_uuid.UUID(int=self._counter))
        self._counter += 1
        return ident

    def _remap_uuid(self, old: str) -> str:
        """Resolve (or assign) the canonical UUID for ``old``.

        Args:
            old (str): A pre-canonical (e.g., ``uuid4()``-issued) UUID
                string captured from an IR Value or metadata field.

        Returns:
            str: The canonical UUID assigned to ``old``. Subsequent
                calls with the same ``old`` return the same canonical
                string.
        """
        if old not in self._uuid_remap:
            self._uuid_remap[old] = self._next_id()
        return self._uuid_remap[old]

    def _remap_logical_id(self, old: str) -> str:
        """Resolve (or assign) the canonical logical_id for ``old``.

        Shares the counter with ``_remap_uuid`` so canonical IDs across
        both namespaces are unique within a single canonicalize() run.

        Args:
            old (str): A pre-canonical ``logical_id`` value.

        Returns:
            str: The canonical ``logical_id`` assigned to ``old``.
        """
        if old not in self._logical_id_remap:
            self._logical_id_remap[old] = self._next_id()
        return self._logical_id_remap[old]

    # -------------------------------------------------------------------
    # Block / Operation walk
    # -------------------------------------------------------------------

    def canonical_block(self, block: Block) -> Block:
        """Return a canonical clone of ``block``.

        Sub-Blocks reached via nested operation fields are
        canonicalized through the same Canonicalizer instance and are
        cached by Python ``id`` so repeated references share a single
        canonical Block.

        Args:
            block (Block): The IR Block to canonicalize. Sub-Blocks
                reachable from this Block are canonicalized recursively.

        Returns:
            Block: A new Block with canonical UUIDs and rewritten
                metadata references. ``Block.kind`` matches the input.
        """
        cached = self._block_cache.get(id(block))
        if cached is not None:
            return cached

        new_input_values: list[Value] = [
            cast(Value, self.canonical_value(v)) for v in block.input_values
        ]
        new_parameters: dict[str, Value] = {
            key: cast(Value, self.canonical_value(block.parameters[key]))
            for key in sorted(block.parameters)
        }
        new_operations: list[Operation] = [
            self.canonical_operation(op) for op in block.operations
        ]
        new_output_values: list[Value] = [
            cast(Value, self.canonical_value(v)) for v in block.output_values
        ]

        new_block = Block(
            name=block.name,
            label_args=list(block.label_args),
            input_values=new_input_values,
            output_values=new_output_values,
            output_names=list(block.output_names),
            operations=new_operations,
            kind=block.kind,
            parameters=new_parameters,
            # ParamSlot is frozen and holds no Value/UUID references, so
            # the manifest is shared verbatim — there is nothing to remap.
            param_slots=block.param_slots,
        )
        self._block_cache[id(block)] = new_block
        return new_block

    def canonical_operation(self, op: Operation) -> Operation:
        """Return a canonical clone of ``op`` with remapped Values.

        Uses ``Operation.all_input_values`` / ``Operation.replace_values``
        so subclass-extra Value fields (e.g., ``ControlledU.power``,
        ``ForOperation.loop_var_value``) are rewritten consistently.
        Recurses into nested control-flow ops via ``HasNestedOps`` and
        into nested operation Blocks.

        Args:
            op (Operation): The Operation to canonicalize.

        Returns:
            Operation: A new Operation of the same concrete type with
                canonical Values, rewritten ``CastOperation.qubit_mapping``,
                and canonicalized nested op lists / implementation Blocks.

        Raises:
            NotImplementedError: If ``op`` is a ``CallBlockOperation``.
                Canonicalize requires the input Block to be inlined first.
        """
        if isinstance(op, CallBlockOperation):
            raise NotImplementedError(
                "canonicalize() does not support HIERARCHICAL blocks "
                "(CallBlockOperation present). Run inline() first."
            )

        sub_map: dict[str, ValueBase] = {}
        for v in op.all_input_values():
            sub_map[v.uuid] = self.canonical_value(v)
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid not in sub_map:
                sub_map[v.uuid] = self.canonical_value(v)

        new_op: Operation = op.replace_values(sub_map) if sub_map else op

        if isinstance(new_op, HasNestedOps):
            new_lists = [
                [self.canonical_operation(child) for child in op_list]
                for op_list in new_op.nested_op_lists()
            ]
            new_op = new_op.rebuild_nested(new_lists)

        if isinstance(new_op, CastOperation) and new_op.qubit_mapping:
            new_op = dataclasses.replace(
                new_op,
                qubit_mapping=[
                    remap_indexed_identifier(u, self._remap_uuid)
                    for u in new_op.qubit_mapping
                ],
            )

        if (
            isinstance(new_op, CompositeGateOperation)
            and new_op.implementation_block is not None
        ):
            sub = self.canonical_block(new_op.implementation_block)
            new_op = dataclasses.replace(new_op, implementation_block=sub)
        if isinstance(new_op, InverseBlockOperation):
            if new_op.source_block is not None:
                sub = self.canonical_block(new_op.source_block)
                new_op = dataclasses.replace(new_op, source_block=sub)
            if new_op.implementation_block is not None:
                sub = self.canonical_block(new_op.implementation_block)
                new_op = dataclasses.replace(new_op, implementation_block=sub)
        if isinstance(new_op, GlobalPhaseBlockOperation):
            if new_op.source_block is not None:
                sub = self.canonical_block(new_op.source_block)
                new_op = dataclasses.replace(new_op, source_block=sub)

        # ControlledUOperation carries the unitary as a nested ``block``
        # field. Canonicalize it through the same canonicalizer so UUIDs
        # inside the unitary body are rewritten in lockstep with the
        # parent Block.
        if isinstance(new_op, ControlledUOperation) and new_op.block is not None:
            sub_block = self.canonical_block(new_op.block)
            new_op = dataclasses.replace(new_op, block=sub_block)

        return new_op

    # -------------------------------------------------------------------
    # Value walk
    # -------------------------------------------------------------------

    def canonical_value(self, value: ValueBase) -> ValueBase:
        """Return a canonical clone of ``value`` with remapped UUIDs.

        Caches by source UUID so repeated references (e.g., the same
        ``parent_array`` shared by multiple element Values) resolve to
        the same canonical instance.

        Args:
            value (ValueBase): An IR Value (any ``Value`` /
                ``ArrayValue`` / ``TupleValue`` / ``DictValue``).

        Returns:
            ValueBase: A canonical clone of the same concrete Value
                type, with ``uuid`` and ``logical_id`` reassigned and
                metadata UUID references rewritten consistently. Nested
                Value references (``TupleValue.elements``,
                ``DictValue.entries``, ``ArrayValue.shape``,
                ``Value.parent_array``, ``Value.element_indices``) are
                canonicalized recursively.
        """
        if value.uuid in self._value_cache:
            return self._value_cache[value.uuid]

        new_uuid = self._remap_uuid(value.uuid)
        new_logical_id = self._remap_logical_id(value.logical_id)
        new_metadata = self._canonical_metadata(value.metadata)

        cloned: ValueBase
        if isinstance(value, TupleValue):
            new_elements = tuple(
                cast(Value, self.canonical_value(e)) for e in value.elements
            )
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
                elements=new_elements,
            )
        elif isinstance(value, DictValue):
            new_entries: list[tuple[TupleValue | Value, Value]] = []
            for k, v in value.entries:
                new_k_base = self.canonical_value(k)
                new_v_base = self.canonical_value(v)
                new_entries.append(
                    (
                        cast("TupleValue | Value", new_k_base),
                        cast(Value, new_v_base),
                    )
                )
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
                entries=tuple(new_entries),
            )
        elif isinstance(value, ArrayValue):
            new_slice_of: ArrayValue | None = None
            if value.slice_of is not None:
                new_slice_of = cast(ArrayValue, self.canonical_value(value.slice_of))
            cloned = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
                shape=tuple(
                    cast(Value, self.canonical_value(dim)) for dim in value.shape
                ),
                slice_of=new_slice_of,
                slice_start=(
                    cast(Value, self.canonical_value(value.slice_start))
                    if value.slice_start is not None
                    else None
                ),
                slice_step=(
                    cast(Value, self.canonical_value(value.slice_step))
                    if value.slice_step is not None
                    else None
                ),
            )
        elif isinstance(value, Value):
            # Cache a placeholder before recursing through parent_array /
            # element_indices so any cycle (not expected in practice, but
            # defensive) terminates with a partially-built canonical
            # instance rather than infinite recursion.
            placeholder = dataclasses.replace(
                value,
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
            )
            self._value_cache[value.uuid] = placeholder
            new_parent_array: ArrayValue | None = None
            if value.parent_array is not None:
                new_parent_array = cast(
                    ArrayValue, self.canonical_value(value.parent_array)
                )
            new_element_indices: tuple[Value, ...] = ()
            if value.element_indices:
                new_element_indices = tuple(
                    cast(Value, self.canonical_value(idx))
                    for idx in value.element_indices
                )
            cloned = dataclasses.replace(
                placeholder,
                parent_array=new_parent_array,
                element_indices=new_element_indices,
            )
        else:
            cloned = dataclasses.replace(
                cast(Any, value),
                uuid=new_uuid,
                logical_id=new_logical_id,
                metadata=new_metadata,
            )

        self._value_cache[value.uuid] = cloned
        return cloned

    def _canonical_metadata(self, metadata: ValueMetadata) -> ValueMetadata:
        """Rewrite UUID and logical_id references inside ValueMetadata.

        ``ScalarMetadata`` and ``DictRuntimeMetadata`` carry no UUID
        references; ``CastMetadata``, ``QFixedMetadata``, and
        ``ArrayRuntimeMetadata`` do and are rewritten through the
        active remap tables.

        Args:
            metadata (ValueMetadata): Original metadata bundle.

        Returns:
            ValueMetadata: A new bundle with rewritten UUID / logical_id
                references and untouched scalar / dict-runtime sections.
        """
        return remap_value_metadata_references(
            metadata,
            self._remap_uuid,
            self._remap_logical_id,
        )


# ---------------------------------------------------------------------------
# Deterministic byte serialization (for content_hash only)
# ---------------------------------------------------------------------------


def _token(obj: Any) -> str:
    """Render an arbitrary Python value into a stable string token.

    Handles the small set of types appearing in IR metadata: scalars,
    enums, ``numpy.ndarray`` (rendered as dtype + shape + a digest of
    the raw buffer), tuples/lists, dicts (sorted by key), ``ValueBase``
    instances (rendered as a compact ``<ClassName:UUID>`` reference
    since the full state is emitted in the value-declaration section),
    and ``ValueType`` instances (rendered via ``.label()`` to avoid
    embedding memory addresses from the default ``object.__repr__``).
    Falls back to ``repr`` for anything else; callers that store
    opaque Python objects in metadata must ensure those objects have a
    stable ``repr`` for the hash to be reliable.

    Args:
        obj (Any): A Python value reachable from canonical IR data
            (Value metadata field, operation dataclass field,
            ``ParamSlot`` payload, etc.).

    Returns:
        str: A deterministic string token suitable for inclusion in
            ``to_canonical_bytes``.
    """
    if obj is None:
        return "None"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float, str)):
        return repr(obj)
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__name__}.{obj.name}"
    if isinstance(obj, np.ndarray) and not obj.dtype.hasobject:
        # ``repr`` is unusable as array identity: it truncates large
        # arrays (different arrays collide) and obeys process-global
        # ``np.printoptions`` (equal arrays diverge). Hash the raw
        # buffer instead. Object-dtype arrays have no content-defined
        # buffer and fall through to the ``repr`` fallback below.
        data = np.ascontiguousarray(obj)
        # Feed the contiguous buffer to the hasher as a flat ``uint8``
        # memoryview (buffer protocol) so no extra ``bytes`` copy of a
        # potentially large array is allocated; ``ascontiguousarray``
        # already guarantees a C-contiguous buffer.
        digest = hashlib.sha256(memoryview(data.reshape(-1)).cast("B")).hexdigest()
        return f"ndarray<dtype={data.dtype.str},shape={data.shape},sha256={digest}>"
    if isinstance(obj, ValueBase):
        return _value_token(obj)
    if isinstance(obj, ValueType):
        return f"Type<{obj.label()}>"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_token(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _token(kv[0]))
        return "{" + ",".join(f"{_token(k)}:{_token(v)}" for k, v in items) + "}"
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Dataclass instances (e.g. ``ResourceMetadata`` on
        # ``CompositeGateOperation``) get serialized field-by-field in
        # name-sorted order so canonical bytes are independent of the
        # declared-field order and of any nested ``dict``'s insertion
        # order (handled transitively via the dict branch above).
        fields = sorted(dataclasses.fields(obj), key=lambda f: f.name)
        body = ",".join(f"{f.name}={_token(getattr(obj, f.name))}" for f in fields)
        return f"{type(obj).__name__}({body})"
    return repr(obj)


def _value_token(v: ValueBase) -> str:
    """Render the canonical identity of ``v`` as a structural token.

    Only the canonical UUID and the value's type label are included;
    full metadata is emitted separately in the value declaration block
    so each Value's full state appears once and references stay
    compact.

    Args:
        v (ValueBase): A canonical IR Value.

    Returns:
        str: A compact ``<ClassName<uuid:TypeLabel>>`` token. Falls
            back to ``repr(v.type)`` if the type does not expose a
            stable ``label()``.
    """
    type_obj = getattr(v, "type", None)
    if isinstance(type_obj, ValueType):
        type_label = type_obj.label()
    else:
        type_label = repr(type_obj)
    return f"{type(v).__name__}<{v.uuid}:{type_label}>"


def _metadata_token(metadata: ValueMetadata) -> str:
    """Serialize ``ValueMetadata`` into a deterministic token string.

    Args:
        metadata (ValueMetadata): The metadata bundle to render.

    Returns:
        str: A bracketed, comma-separated list of sub-record tokens
            (``scalar(...)``, ``cast(...)``, ``qfixed(...)``,
            ``array_runtime(...)``, ``dict_runtime(...)``). Sub-records
            with ``None`` values are omitted.
    """
    parts: list[str] = []
    if metadata.scalar is not None:
        parts.append(
            f"scalar(const={_token(metadata.scalar.const_value)},"
            f"param={_token(metadata.scalar.parameter_name)})"
        )
    if metadata.cast is not None:
        parts.append(
            f"cast(src={metadata.cast.source_uuid},"
            f"qubits={_token(list(metadata.cast.qubit_uuids))},"
            f"src_lid={_token(metadata.cast.source_logical_id)},"
            f"qubit_lids={_token(list(metadata.cast.qubit_logical_ids))})"
        )
    if metadata.qfixed is not None:
        parts.append(
            f"qfixed(qubits={_token(list(metadata.qfixed.qubit_uuids))},"
            f"num_bits={metadata.qfixed.num_bits},"
            f"int_bits={metadata.qfixed.int_bits})"
        )
    if metadata.array_runtime is not None:
        parts.append(
            f"array_runtime(const={_token(metadata.array_runtime.const_array)},"
            f"elem_uuids={_token(list(metadata.array_runtime.element_uuids))},"
            f"elem_lids={_token(list(metadata.array_runtime.element_logical_ids))},"
            f"elem_par_uuids={_token(list(metadata.array_runtime.element_parent_uuids))},"
            f"elem_par_idxs={_token(list(metadata.array_runtime.element_parent_indices))})"
        )
    if metadata.dict_runtime is not None:
        # ``bound_data`` is stored as tuple-of-pairs in the Mapping's
        # original iteration order. Two semantically-equal dicts with
        # different insertion order must hash identically, so emit via
        # ``_token(dict(...))`` so the dict-handling branch sorts keys.
        parts.append(
            "dict_runtime(items=" + _token(dict(metadata.dict_runtime.bound_data)) + ")"
        )
    return "[" + ",".join(parts) + "]"


def _collect_values(block: Block, out: list[ValueBase], seen: set[str]) -> None:
    """Walk ``block`` deterministically and collect every Value reachable.

    Each Value appears at most once in ``out``; ordering follows the
    canonical walk used for UUID assignment so the value-declaration
    section of the canonical bytes is itself deterministic.

    Args:
        block (Block): The canonical Block to walk.
        out (list[ValueBase]): Output list, populated in walk order
            with each unique Value encountered.
        seen (set[str]): Mutable membership set keyed by Value UUID; an
            incoming UUID short-circuits visitation. The caller passes
            an empty set on the top-level call.
    """

    def visit(v: ValueBase) -> None:
        if v.uuid in seen:
            return
        seen.add(v.uuid)
        # Visit nested Value references first to keep declaration order
        # close to the canonical walk in _Canonicalizer.
        if isinstance(v, TupleValue):
            for e in v.elements:
                visit(e)
        elif isinstance(v, DictValue):
            for k, val in v.entries:
                visit(k)
                visit(val)
        elif isinstance(v, ArrayValue):
            for dim in v.shape:
                visit(dim)
            if v.slice_of is not None:
                visit(v.slice_of)
            if v.slice_start is not None:
                visit(v.slice_start)
            if v.slice_step is not None:
                visit(v.slice_step)
        elif isinstance(v, Value):
            if v.parent_array is not None:
                visit(v.parent_array)
            for idx in v.element_indices:
                visit(idx)
        out.append(v)

    for v in block.input_values:
        visit(v)
    for key in sorted(block.parameters):
        visit(block.parameters[key])
    for op in block.operations:
        _collect_from_operation(op, visit)
    for v in block.output_values:
        visit(v)


def _collect_from_operation(op: Operation, visit: Any) -> None:
    """Apply ``visit`` to every Value referenced by ``op`` (including nested).

    Args:
        op (Operation): The Operation whose Value references to walk.
        visit (Callable[[ValueBase], None]): A callable that receives
            each Value once. Typed as ``Any`` because the nested
            ``visit`` closure carries internal capture state and is not
            an exported protocol.
    """
    for v in op.all_input_values():
        visit(v)
    for v in op.results:
        if isinstance(v, ValueBase):
            visit(v)
    if isinstance(op, HasNestedOps):
        for child_list in op.nested_op_lists():
            for child in child_list:
                _collect_from_operation(child, visit)
    if isinstance(op, CompositeGateOperation):
        if op.implementation_block is not None:
            # Nested block's values participate in the same canonical
            # universe; recurse to ensure they are declared too.
            _collect_from_subblock(op.implementation_block, visit)
    if isinstance(op, InverseBlockOperation):
        if op.source_block is not None:
            _collect_from_subblock(op.source_block, visit)
        if op.implementation_block is not None:
            _collect_from_subblock(op.implementation_block, visit)
    if isinstance(op, GlobalPhaseBlockOperation) and op.source_block is not None:
        _collect_from_subblock(op.source_block, visit)
    if isinstance(op, ControlledUOperation) and op.block is not None:
        _collect_from_subblock(op.block, visit)


def _collect_from_subblock(sub: Block, visit: Any) -> None:
    """Walk the Values declared inside a nested Block.

    Used for nested operation Blocks such as
    ``CompositeGateOperation.implementation_block``,
    ``InverseBlockOperation.source_block``, and
    ``ControlledUOperation.block``. Mirrors the top-level
    ``_collect_values`` walk order so declarations remain
    deterministic.

    Args:
        sub (Block): A nested Block embedded in an operation.
        visit (Callable[[ValueBase], None]): The same ``visit`` closure
            passed by ``_collect_values`` so the nested Block's Values
            land in the parent declaration list.
    """
    for v in sub.input_values:
        visit(v)
    for key in sorted(sub.parameters):
        visit(sub.parameters[key])
    for nested_op in sub.operations:
        _collect_from_operation(nested_op, visit)
    for v in sub.output_values:
        visit(v)


_INLINE_INDENT = "  "

# Operation dataclass fields whose values do not affect IR semantics for
# hashing purposes (or whose Python identity changes between builds).
_OP_FIELD_EXCLUDES: frozenset[str] = frozenset(
    {
        "operands",
        "results",
        # CompositeGateOperation extras: opaque Python references that
        # do not reflect IR-level structure.
        "composite_gate_instance",
        # Nested-Block fields. These are emitted separately by
        # ``_emit_operation`` so they are not rendered through ``repr``
        # here.
        "implementation_block",
        "source_block",
        "block",
    }
)


def _emit_block(block: Block, out: list[str], indent: int) -> None:
    """Append a deterministic textual representation of ``block`` to ``out``.

    ``Block.name`` and ``Block.output_names`` are display-only labels
    and are intentionally omitted from canonical bytes; two
    structurally-identical kernels with different function names hash
    equally. ``Block.label_args`` (input parameter names by position)
    and ``Block.param_slots`` (the kernel's classical parameter
    contract) are functional and are included.

    Args:
        block (Block): The canonical Block to render.
        out (list[str]): Output line buffer; appended to in place.
        indent (int): Current indentation level in
            ``_INLINE_INDENT``-wide units.
    """
    pad = _INLINE_INDENT * indent
    out.append(f"{pad}BLOCK kind={block.kind.name}")
    out.append(f"{pad}{_INLINE_INDENT}label_args={_token(block.label_args)}")
    out.append(f"{pad}{_INLINE_INDENT}param_slots={_token(block.param_slots)}")

    declared: list[ValueBase] = []
    seen: set[str] = set()
    _collect_values(block, declared, seen)
    out.append(f"{pad}{_INLINE_INDENT}values:")
    for v in declared:
        out.append(f"{pad}{_INLINE_INDENT * 2}{_value_declaration(v)}")

    out.append(
        f"{pad}{_INLINE_INDENT}inputs={_token([v.uuid for v in block.input_values])}"
    )
    out.append(
        f"{pad}{_INLINE_INDENT}outputs={_token([v.uuid for v in block.output_values])}"
    )
    out.append(
        f"{pad}{_INLINE_INDENT}parameters="
        + _token({k: block.parameters[k].uuid for k in sorted(block.parameters)})
    )

    out.append(f"{pad}{_INLINE_INDENT}operations:")
    for op in block.operations:
        _emit_operation(op, out, indent + 2)


def _value_declaration(v: ValueBase) -> str:
    """Build a one-line, fully-self-describing declaration for ``v``.

    ``Value.name`` is display-only (per the docstring on ``Value``) and
    is intentionally omitted so that two structurally-identical kernels
    whose intermediate values were given different debug labels still
    canonicalize to the same bytes.

    Args:
        v (ValueBase): The canonical Value to declare.

    Returns:
        str: A pipe-separated, single-line declaration including class
            name, canonical UUID / logical_id, type label, metadata,
            and Value-kind-specific extras (parent_array, shape,
            elements, dict entries, element_indices, version).
    """
    type_obj = getattr(v, "type", None)
    if isinstance(type_obj, ValueType):
        type_label = type_obj.label()
    else:
        type_label = repr(type_obj)
    parts = [
        type(v).__name__,
        f"uuid={v.uuid}",
        f"lid={v.logical_id}",
        f"type={type_label}",
        f"meta={_metadata_token(v.metadata)}",
    ]
    if isinstance(v, Value):
        parts.append(f"version={v.version}")
        if v.parent_array is not None:
            parts.append(f"parent={v.parent_array.uuid}")
        if v.element_indices:
            parts.append("indices=" + _token([idx.uuid for idx in v.element_indices]))
    if isinstance(v, ArrayValue):
        parts.append("shape=" + _token([dim.uuid for dim in v.shape]))
        if v.slice_of is not None:
            parts.append(f"slice_of={v.slice_of.uuid}")
        if v.slice_start is not None:
            parts.append(f"slice_start={v.slice_start.uuid}")
        if v.slice_step is not None:
            parts.append(f"slice_step={v.slice_step.uuid}")
    if isinstance(v, TupleValue):
        parts.append("elements=" + _token([e.uuid for e in v.elements]))
    if isinstance(v, DictValue):
        parts.append("entries=" + _token([(k.uuid, val.uuid) for k, val in v.entries]))
    return "|".join(parts)


def _emit_operation(op: Operation, out: list[str], indent: int) -> None:
    """Append a deterministic textual representation of ``op`` to ``out``.

    Args:
        op (Operation): The canonical Operation to render.
        out (list[str]): Output line buffer; appended to in place.
        indent (int): Current indentation level in
            ``_INLINE_INDENT``-wide units.
    """
    pad = _INLINE_INDENT * indent
    header = (
        f"{pad}{type(op).__name__} "
        f"operands={_token([v.uuid for v in op.operands])} "
        f"results={_token([v.uuid for v in op.results])}"
    )
    out.append(header)

    extras = _extra_field_tokens(op)
    if extras:
        out.append(f"{pad}{_INLINE_INDENT}fields={{{','.join(extras)}}}")

    if isinstance(op, HasNestedOps):
        for i, child_list in enumerate(op.nested_op_lists()):
            out.append(f"{pad}{_INLINE_INDENT}nested[{i}]:")
            for child in child_list:
                _emit_operation(child, out, indent + 2)

    if isinstance(op, CompositeGateOperation) and op.implementation_block is not None:
        out.append(f"{pad}{_INLINE_INDENT}implementation:")
        _emit_block(op.implementation_block, out, indent + 2)
    if isinstance(op, InverseBlockOperation):
        if op.source_block is not None:
            out.append(f"{pad}{_INLINE_INDENT}source:")
            _emit_block(op.source_block, out, indent + 2)
        if op.implementation_block is not None:
            out.append(f"{pad}{_INLINE_INDENT}implementation:")
            _emit_block(op.implementation_block, out, indent + 2)
    if isinstance(op, GlobalPhaseBlockOperation) and op.source_block is not None:
        out.append(f"{pad}{_INLINE_INDENT}source:")
        _emit_block(op.source_block, out, indent + 2)

    if isinstance(op, ControlledUOperation) and op.block is not None:
        out.append(f"{pad}{_INLINE_INDENT}unitary_block:")
        _emit_block(op.block, out, indent + 2)


def _extra_field_tokens(op: Operation) -> list[str]:
    """Render the IR-meaningful dataclass fields of ``op`` (excluding Values).

    The base ``Operation`` fields (``operands`` / ``results``) and a
    handful of opaque-Python fields are excluded; everything else is
    serialized through ``_token`` and emitted in field-name-sorted
    order so the token list is stable regardless of dataclass
    declaration ordering.

    Args:
        op (Operation): The canonical Operation to introspect.

    Returns:
        list[str]: Zero or more ``name=token`` strings, one per
            included dataclass field, sorted by field name.
    """
    parts: list[str] = []
    for f in sorted(dataclasses.fields(op), key=lambda field: field.name):
        if f.name in _OP_FIELD_EXCLUDES:
            continue
        value = getattr(op, f.name)
        # Skip nested-op fields handled by HasNestedOps emission to avoid
        # double-printing operation bodies.
        if isinstance(value, list) and value and isinstance(value[0], Operation):
            continue
        parts.append(f"{f.name}={_token(value)}")
    return parts
