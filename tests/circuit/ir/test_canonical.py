"""Tests for the IR canonical form and content hash.

These tests verify that ``canonicalize`` re-numbers UUIDs
deterministically so two structurally-identical builds of the same
kernel produce byte-identical canonical IR and identical hashes, while
genuine IR differences yield different hashes.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.canonical import (
    canonicalize,
    canonicalize_and_remap,
    content_fingerprint,
    content_hash,
    to_canonical_bytes,
)
from qamomile.circuit.ir.operation import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.types.primitives import (
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    CastMetadata,
    DictValue,
    QFixedMetadata,
    ScalarMetadata,
    TupleValue,
    Value,
    ValueMetadata,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.observable.hamiltonian import Hamiltonian, Pauli, PauliOperator

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _to_affine(kernel: qmc.QKernel) -> Block:
    """Return an AFFINE block for ``kernel`` without instantiating a backend.

    Args:
        kernel (qmc.QKernel): A ``@qkernel``-decorated function.

    Returns:
        Block: The kernel's traced block after running ``InlinePass``,
            so all inline-policy ``InvokeOperation``s are removed and the block is
            ready for canonicalize().
    """
    return InlinePass().run(kernel.block)


@qmc.qkernel
def _h_then_rx(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Simple single-qubit kernel with one parameter."""
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _h_then_rx_twin(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Structurally identical twin of ``_h_then_rx`` for cross-build equality tests.

    A second ``@qkernel`` decoration produces an independent build with
    fresh UUIDs, simulating the "two processes built the same kernel"
    scenario without having to clear the kernel's cached block.
    """
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _h_then_ry(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Variant of ``_h_then_rx`` that uses RY instead of RX."""
    q = qmc.h(q)
    q = qmc.ry(q, theta)
    return q


@qmc.qkernel
def _loop_h(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Kernel that exercises a symbolic ForOperation over a Vector input."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _loop_h_twin(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Twin of ``_loop_h`` for cross-build determinism on symbolic ForOperation."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _carried_sum(n: qmc.UInt) -> qmc.UInt:
    """Kernel whose ForOperation carries a region argument (``total``)."""
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def _carried_sum_twin(n: qmc.UInt) -> qmc.UInt:
    """Twin of ``_carried_sum`` for cross-build determinism on region args."""
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def _carried_sum_renamed(n: qmc.UInt) -> qmc.UInt:
    """Structurally match ``_carried_sum`` using different local labels."""
    accumulator = qmc.uint(0)
    for j in qmc.range(n):
        accumulator = accumulator + j
    return accumulator


@qmc.qkernel
def _measure_after_h(q: qmc.Qubit) -> qmc.Bit:
    """Kernel that exercises a measurement-derived classical bit."""
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _measure_after_h_twin(q: qmc.Qubit) -> qmc.Bit:
    """Twin of ``_measure_after_h`` for cross-build determinism with measurement."""
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _nested_if_merge() -> qmc.Bit:
    """Nested measurement-conditioned ifs whose branches rebind a qubit.

    The outer if's true-branch yield is the inner if's merge output, so
    canonicalize must renumber UUIDs consistently through the
    ``true_yields`` / ``false_yields`` lists at both nesting levels.
    """
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    outer = qmc.measure(q)
    p = qmc.qubit(name="p")
    p = qmc.h(p)
    inner = qmc.measure(p)
    s = qmc.qubit(name="s")
    if outer:
        if inner:
            s = qmc.z(s)
        else:
            s = qmc.x(s)
    else:
        s = qmc.h(s)
    return qmc.measure(s)


@qmc.qkernel
def _nested_if_merge_twin() -> qmc.Bit:
    """Twin of ``_nested_if_merge`` for cross-build determinism on merges."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    outer = qmc.measure(q)
    p = qmc.qubit(name="p")
    p = qmc.h(p)
    inner = qmc.measure(p)
    s = qmc.qubit(name="s")
    if outer:
        if inner:
            s = qmc.z(s)
        else:
            s = qmc.x(s)
    else:
        s = qmc.h(s)
    return qmc.measure(s)


# Two structurally-identical controlled-U scenarios. Each top-level
# kernel applies controlled(_phase_a/_phase_b) so each top-level Block
# contains a ``ControlledUOperation`` carrying a nested unitary
# ``block`` field. The canonical form must rewrite UUIDs inside that
# nested block in lockstep with the parent so the two builds agree.


@qmc.qkernel
def _phase_a(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit phase rotation used as a controlled-U body."""
    return qmc.p(q, theta)


@qmc.qkernel
def _phase_b(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Structurally identical twin of ``_phase_a``."""
    return qmc.p(q, theta)


@qmc.qkernel
def _controlled_phase_a(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Top-level kernel that embeds ``ControlledUOperation`` via controlled(_phase_a)."""
    op = qmc.control(_phase_a)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


@qmc.qkernel
def _callable_body_x_a(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X in the first independently-built callable body."""
    return qmc.x(q)


@qmc.qkernel
def _callable_body_x_b(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X in the structurally identical callable body twin."""
    return qmc.x(q)


@qmc.qkernel
def _callable_impl_h_a(q: qmc.Qubit) -> qmc.Qubit:
    """Apply H in the first transform-specific implementation body."""
    return qmc.h(q)


@qmc.qkernel
def _callable_impl_h_b(q: qmc.Qubit) -> qmc.Qubit:
    """Apply H in the implementation-body twin."""
    return qmc.h(q)


@qmc.qkernel
def _callable_impl_z(q: qmc.Qubit) -> qmc.Qubit:
    """Apply Z in a semantically different implementation body."""
    return qmc.z(q)


def _boxed_callable_case(body: Block, implementation_body: Block) -> Block:
    """Build one SELECT case containing a boxed callable definition.

    Args:
        body (Block): Default callable body.
        implementation_body (Block): Transform-specific implementation body.

    Returns:
        Block: AFFINE one-qubit case block containing an ``InvokeOperation``.
    """
    q = Value(type=QubitType(), name="case_q")
    result = q.next_version()
    ref = CallableRef(namespace="test.canonical", name="boxed_case")
    definition = CallableDef(
        ref=ref,
        body=body,
        implementations=[
            CallableImplementation(
                transform=CallTransform.DIRECT,
                strategy="alternate",
                body=implementation_body,
                attrs={"variant": "stable"},
            )
        ],
        default_policy=CallPolicy.PRESERVE_BOX,
        attrs={"kind": "composite"},
    )
    invoke = InvokeOperation(
        operands=[q],
        results=[result],
        target=ref,
        attrs={"kind": "composite", "default_policy": "PRESERVE_BOX"},
        definition=definition,
    )
    return Block(
        input_values=[q],
        output_values=[result],
        operations=[invoke],
        kind=BlockKind.AFFINE,
        label_args=["q"],
    )


def _select_with_boxed_case(body: Block, implementation_body: Block) -> Block:
    """Build an AFFINE SELECT whose selected case contains a boxed callable.

    Args:
        body (Block): Default boxed-callable body.
        implementation_body (Block): Alternate boxed-callable body.

    Returns:
        Block: Two-qubit SELECT block with independent random Value UUIDs.
    """
    identity_q = Value(type=QubitType(), name="identity_q")
    identity_case = Block(
        input_values=[identity_q],
        output_values=[identity_q],
        kind=BlockKind.AFFINE,
        label_args=["q"],
    )
    idx = Value(type=QubitType(), name="idx")
    target = Value(type=QubitType(), name="target")
    idx_result = idx.next_version()
    target_result = target.next_version()
    select_op = SelectOperation(
        operands=[idx, target],
        results=[idx_result, target_result],
        num_index_qubits=1,
        case_blocks=[identity_case, _boxed_callable_case(body, implementation_body)],
    )
    return Block(
        input_values=[idx, target],
        output_values=[idx_result, target_result],
        operations=[select_op],
        kind=BlockKind.AFFINE,
        label_args=["idx", "target"],
    )


@qmc.qkernel
def _controlled_phase_b(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Variant that controls a different callable with the same structure."""
    op = qmc.control(_phase_b)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


@qmc.qkernel
def _controlled_phase_a_twin(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Twin that controls the same callable as ``_controlled_phase_a``."""
    op = qmc.control(_phase_a)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


class TestCanonicalizePreservesStructure:
    """Verify that canonicalize preserves kind, shape, and does not mutate."""

    def test_preserves_block_kind(self):
        """Canonicalize keeps the input ``BlockKind`` (does not advance the stage)."""
        block = _to_affine(_h_then_rx)
        canon = canonicalize(block)
        assert canon.kind == block.kind == BlockKind.AFFINE

    def test_does_not_mutate_input(self):
        """Original block UUIDs are untouched by canonicalize."""
        block = _to_affine(_h_then_rx)
        original_input_uuid = block.input_values[0].uuid
        canonicalize(block)
        assert block.input_values[0].uuid == original_input_uuid

    def test_preserves_operation_count_and_order(self):
        """Number and order of operations is preserved through canonicalize."""
        block = _to_affine(_h_then_rx)
        canon = canonicalize(block)
        assert len(canon.operations) == len(block.operations)
        assert [type(op).__name__ for op in canon.operations] == [
            type(op).__name__ for op in block.operations
        ]


class TestCanonicalizeDeterminism:
    """Cross-build determinism: same kernel built twice produces matching IR."""

    def test_twin_kernels_same_canonical_bytes(self):
        """Two structurally-identical kernels canonicalize to identical bytes."""
        a = _to_affine(_h_then_rx)
        b = _to_affine(_h_then_rx_twin)
        # Sanity: independent builds have different raw UUIDs.
        assert a.input_values[0].uuid != b.input_values[0].uuid
        # But canonical forms align byte-for-byte.
        assert to_canonical_bytes(a) == to_canonical_bytes(b)

    def test_twin_kernels_same_hash(self):
        """``content_hash`` agrees across two structurally-identical kernels."""
        assert content_hash(_to_affine(_h_then_rx)) == content_hash(
            _to_affine(_h_then_rx_twin)
        )

    def test_loop_kernel_twins_same_hash(self):
        """Determinism holds for kernels containing symbolic ForOperation."""
        a = _to_affine(_loop_h)
        b = _to_affine(_loop_h_twin)
        assert content_hash(a) == content_hash(b)

    def test_measurement_kernel_twins_same_hash(self):
        """Determinism holds for kernels with measurement-backed bits."""
        a = _to_affine(_measure_after_h)
        b = _to_affine(_measure_after_h_twin)
        assert content_hash(a) == content_hash(b)

    def test_region_arg_loop_twins_same_hash(self):
        """Determinism holds for loops carrying region arguments.

        The ``RegionArg`` values (init / block_arg / yielded / result)
        must be remapped in lockstep with the body operations and the
        loop results for two independent builds to hash equally.
        """
        a = _to_affine(_carried_sum)
        b = _to_affine(_carried_sum_twin)
        assert a.operations[-1].region_args, "fixture must carry region args"
        assert to_canonical_bytes(a) == to_canonical_bytes(b)
        assert content_hash(a) == content_hash(b)

    def test_region_arg_display_names_do_not_affect_hash(self):
        """Loop-variable and carry labels are excluded from content identity."""
        original = _to_affine(_carried_sum)
        renamed = _to_affine(_carried_sum_renamed)

        assert to_canonical_bytes(original) == to_canonical_bytes(renamed)
        assert content_hash(original) == content_hash(renamed)

    def test_controlled_u_twins_same_canonical_bytes(self):
        """Cross-build determinism through ``ControlledUOperation.block``.

        Each top-level kernel embeds a ``ControlledUOperation`` whose
        ``block`` field is itself an independently-traced sub-Block.
        Without nested canonicalization, UUIDs inside that sub-Block
        would differ between builds and the canonical bytes would
        disagree.
        """
        a = _to_affine(_controlled_phase_a)
        b = _to_affine(_controlled_phase_a_twin)
        assert to_canonical_bytes(a) == to_canonical_bytes(b)
        assert content_hash(a) == content_hash(b)

    def test_controlled_u_callable_ref_affects_canonical_bytes(self):
        """Controlled-U callable identity participates in content hashing.

        ``_phase_a`` and ``_phase_b`` have the same body shape, but they are
        distinct qkernel callables.  QPE and other higher-level routines rely
        on this identity, so the canonical form must not collapse the two
        controlled calls merely because their decompositions happen to match.
        """
        a = _to_affine(_controlled_phase_a)
        b = _to_affine(_controlled_phase_b)
        assert to_canonical_bytes(a) != to_canonical_bytes(b)
        assert content_hash(a) != content_hash(b)

    def test_select_boxed_implementation_twins_have_same_hash(self):
        """Callable implementation bodies inside SELECT canonicalize fully.

        Transform-specific bodies under ``CallableDef.implementations`` must
        share the enclosing canonical UUID universe; otherwise independently
        built but identical boxed cases leak random UUIDs into
        ``content_hash``.
        """
        a = _select_with_boxed_case(
            _callable_body_x_a.block,
            _callable_impl_h_a.block,
        )
        b = _select_with_boxed_case(
            _callable_body_x_b.block,
            _callable_impl_h_b.block,
        )

        assert to_canonical_bytes(a) == to_canonical_bytes(b)
        assert content_hash(a) == content_hash(b)

    def test_symbolic_select_width_canonicalizes_with_its_input(self):
        """Independent symbolic SELECT widths reuse canonical input identity."""

        @qmc.qkernel
        def identity(target: qmc.Qubit) -> qmc.Qubit:
            """Return a SELECT target unchanged."""
            return target

        @qmc.qkernel
        def flipped(target: qmc.Qubit) -> qmc.Qubit:
            """Apply X to a SELECT target."""
            return qmc.x(target)

        @qmc.qkernel
        def first(width: qmc.UInt) -> qmc.Bit:
            """Build the first symbolic-width SELECT twin."""
            index = qmc.qubit_array(width, "index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [identity, flipped],
                num_index_qubits=width,
            )(index, target)
            return qmc.measure(target)

        @qmc.qkernel
        def second(width: qmc.UInt) -> qmc.Bit:
            """Build the second symbolic-width SELECT twin."""
            index = qmc.qubit_array(width, "index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [identity, flipped],
                num_index_qubits=width,
            )(index, target)
            return qmc.measure(target)

        first_block = _to_affine(first)
        second_block = _to_affine(second)
        canonical = canonicalize(first_block)
        select = next(
            operation
            for operation in canonical.operations
            if isinstance(operation, SelectOperation)
        )

        assert isinstance(select.num_index_qubits, Value)
        assert select.num_index_qubits.uuid == canonical.input_values[0].uuid
        assert to_canonical_bytes(first_block) == to_canonical_bytes(second_block)

    def test_select_boxed_implementation_body_affects_hash(self):
        """A semantic change in an alternate implementation changes the hash."""
        h_impl = _select_with_boxed_case(
            _callable_body_x_a.block,
            _callable_impl_h_a.block,
        )
        z_impl = _select_with_boxed_case(
            _callable_body_x_a.block,
            _callable_impl_z.block,
        )

        assert content_hash(h_impl) != content_hash(z_impl)


@pytest.mark.parametrize(
    "unsupported",
    [
        {"alpha", "beta"},
        object(),
        np.array([object()], dtype=object),
    ],
    ids=["set", "object", "object-dtype-array"],
)
def test_content_fingerprint_rejects_unsupported_nested_values(
    unsupported: object,
) -> None:
    """Strict lowered-IR fingerprints never use an implicit repr fallback."""
    with pytest.raises(TypeError, match=r"content_fingerprint\(\)"):
        content_fingerprint({"nested": [unsupported]})


def test_content_fingerprint_accepts_structural_range_values() -> None:
    """Concrete loop index ranges have a stable strict representation."""
    assert len(content_fingerprint(range(1, 7, 2))) == 64


class TestCanonicalizeIdempotence:
    """Canonicalizing twice must equal canonicalizing once."""

    def test_idempotent_bytes(self):
        """``to_canonical_bytes(canonicalize(b)) == to_canonical_bytes(b)``."""
        block = _to_affine(_h_then_rx)
        once = to_canonical_bytes(block)
        canon_once = canonicalize(block)
        twice = to_canonical_bytes(canon_once)
        assert once == twice

    def test_idempotent_hash(self):
        """``content_hash`` is stable under repeated canonicalize."""
        block = _to_affine(_h_then_rx)
        h1 = content_hash(block)
        canon = canonicalize(block)
        h2 = content_hash(canon)
        h3 = content_hash(canonicalize(canon))
        assert h1 == h2 == h3


class TestCanonicalizeIfMerges:
    """Canonical form covers IfOperation merge yields (nested ifs included)."""

    def test_nested_if_twins_same_canonical_form(self):
        """Two identical nested-if kernels agree byte-for-byte and by hash."""
        a = _to_affine(_nested_if_merge)
        b = _to_affine(_nested_if_merge_twin)
        assert to_canonical_bytes(a) == to_canonical_bytes(b)
        assert content_hash(a) == content_hash(b)

    def test_nested_if_idempotent(self):
        """Canonicalize is idempotent on a block with nested if merges."""
        block = _to_affine(_nested_if_merge)
        once = to_canonical_bytes(block)
        canon = canonicalize(block)
        assert to_canonical_bytes(canon) == once
        assert content_hash(canonicalize(canon)) == content_hash(block)

    def test_canonicalize_renumbers_yield_uuids(self):
        """Yield values carry counter-based UUIDs after canonicalize.

        An unmapped yield would keep its random uuid4, so asserting the
        counter prefix on every yield proves the UUID rewrite reaches the
        ``true_yields`` / ``false_yields`` lists at every nesting level.
        """

        def collect_if_ops(operations) -> list[IfOperation]:
            """Recursively collect IfOperations from an operation list."""
            found: list[IfOperation] = []
            for op in operations:
                if isinstance(op, IfOperation):
                    found.append(op)
                    found.extend(collect_if_ops(op.true_operations))
                    found.extend(collect_if_ops(op.false_operations))
            return found

        canon = canonicalize(_to_affine(_nested_if_merge))
        if_ops = collect_if_ops(canon.operations)
        assert len(if_ops) == 2
        for if_op in if_ops:
            merges = list(if_op.iter_merges())
            assert merges
            for merge in merges:
                assert merge.true_value.uuid.startswith("00000000-0000-0000-0000-")
                assert merge.false_value.uuid.startswith("00000000-0000-0000-0000-")


# ---------------------------------------------------------------------------
# Counter-based UUID format
# ---------------------------------------------------------------------------


class TestCounterBasedUUIDs:
    """The first Value visited gets ``UUID(int=0)``; subsequent values continue."""

    def test_first_input_has_zero_uuid(self):
        """First canonicalized Value carries the counter-zero UUID."""
        canon = canonicalize(_to_affine(_h_then_rx))
        assert canon.input_values[0].uuid == "00000000-0000-0000-0000-000000000000"

    def test_remap_dict_covers_every_original_uuid(self):
        """``canonicalize_and_remap`` returns a key for every original Value UUID."""
        block = _to_affine(_h_then_rx)
        _, uuid_remap, _ = canonicalize_and_remap(block)
        for v in block.input_values:
            assert v.uuid in uuid_remap

    def test_remap_exposes_logical_id_table(self):
        """``canonicalize_and_remap`` also exposes the logical_id remap table."""
        block = _to_affine(_h_then_rx)
        _, _, logical_id_remap = canonicalize_and_remap(block)
        for v in block.input_values:
            assert v.logical_id in logical_id_remap


# ---------------------------------------------------------------------------
# Differentiation: structurally different kernels hash differently
# ---------------------------------------------------------------------------


class TestHashDifferentiation:
    """Different IR shapes must produce different ``content_hash`` values."""

    def test_different_gate_types_differ(self):
        """RX-based and RY-based variants of the same kernel disagree."""
        rx_hash = content_hash(_to_affine(_h_then_rx))
        ry_hash = content_hash(_to_affine(_h_then_ry))
        assert rx_hash != ry_hash

    def test_extra_gate_changes_hash(self):
        """Appending a gate to the operation list changes the hash."""
        block = _to_affine(_h_then_rx)
        # Append a fixed X gate referencing the existing output value.
        last_result = block.operations[-1].results[0]
        new_q = last_result.next_version()
        block_extra = dataclasses.replace(
            block,
            operations=[
                *block.operations,
                GateOperation.fixed(
                    GateOperationType.X,
                    qubits=[last_result],
                    results=[new_q],
                ),
            ],
            output_values=[new_q],
        )
        assert content_hash(block) != content_hash(block_extra)


# ---------------------------------------------------------------------------
# Kind enforcement
# ---------------------------------------------------------------------------


class TestUnsupportedBlockKind:
    """``canonicalize`` rejects Blocks that have not been inlined yet."""

    def test_rejects_hierarchical(self):
        """A raw HIERARCHICAL block raises ``ValueError``."""
        hierarchical = dataclasses.replace(
            _h_then_rx.block, kind=BlockKind.HIERARCHICAL
        )
        with pytest.raises(ValueError, match="AFFINE"):
            canonicalize(hierarchical)

    def test_rejects_traced(self):
        """A TRACED block also raises (must be inlined first)."""
        traced = dataclasses.replace(_h_then_rx.block, kind=BlockKind.TRACED)
        with pytest.raises(ValueError, match="AFFINE"):
            canonicalize(traced)


# ---------------------------------------------------------------------------
# Direct-construction tests for metadata UUID rewrite
# ---------------------------------------------------------------------------


def _make_minimal_block(
    *,
    input_values: list[Value] | None = None,
    output_values: list[Value] | None = None,
    operations: list | None = None,
    parameters: dict[str, Value] | None = None,
) -> Block:
    """Build a minimal AFFINE Block from explicit Value lists.

    Used by metadata-rewrite tests that need precise control over IR
    contents that the frontend does not produce by itself.

    Args:
        input_values (list[Value] | None): Values to expose as
            ``Block.input_values``. Defaults to ``None`` (empty list).
        output_values (list[Value] | None): Values to expose as
            ``Block.output_values``. Defaults to ``None`` (empty list).
        operations (list | None): Operations to place inside the
            block. Defaults to ``None`` (no operations).
        parameters (dict[str, Value] | None): Parameter slot mapping.
            Defaults to ``None`` (no parameters).

    Returns:
        Block: A freshly-constructed Block with ``BlockKind.AFFINE``,
            ready to be passed to ``canonicalize``.
    """
    return Block(
        name="manual",
        label_args=[v.name for v in (input_values or [])],
        input_values=list(input_values or []),
        output_values=list(output_values or []),
        output_names=[v.name for v in (output_values or [])],
        operations=list(operations or []),
        kind=BlockKind.AFFINE,
        parameters=dict(parameters or {}),
    )


class TestMetadataUUIDRewrite:
    """``ValueMetadata`` UUID fields must be remapped in lockstep with Values."""

    def test_cast_metadata_uuid_rewritten(self):
        """CastMetadata.source_uuid / qubit_uuids land on canonical UUIDs."""
        source = Value(type=QubitType(), name="src")
        qubit = Value(type=QubitType(), name="q0")
        cast_result = Value(
            type=FloatType(),
            name="cast",
            metadata=ValueMetadata(
                cast=CastMetadata(
                    source_uuid=source.uuid,
                    qubit_uuids=(qubit.uuid,),
                    source_logical_id=source.logical_id,
                    qubit_logical_ids=(qubit.logical_id,),
                ),
            ),
        )
        block = _make_minimal_block(
            input_values=[source, qubit],
            output_values=[cast_result],
        )
        canon = canonicalize(block)
        canon_source, canon_qubit = canon.input_values
        canon_cast = canon.output_values[0]
        assert canon_cast.metadata.cast is not None
        assert canon_cast.metadata.cast.source_uuid == canon_source.uuid
        assert canon_cast.metadata.cast.qubit_uuids == (canon_qubit.uuid,)
        assert canon_cast.metadata.cast.source_logical_id == canon_source.logical_id
        assert canon_cast.metadata.cast.qubit_logical_ids == (canon_qubit.logical_id,)

    def test_qfixed_metadata_qubits_rewritten(self):
        """QFixedMetadata.qubit_uuids are rewritten to canonical UUIDs."""
        q0 = Value(type=QubitType(), name="q0")
        q1 = Value(type=QubitType(), name="q1")
        qf = Value(
            type=FloatType(),
            name="qf",
            metadata=ValueMetadata(
                qfixed=QFixedMetadata(
                    qubit_uuids=(q0.uuid, q1.uuid),
                    num_bits=2,
                    int_bits=0,
                ),
            ),
        )
        block = _make_minimal_block(input_values=[q0, q1], output_values=[qf])
        canon = canonicalize(block)
        cq0, cq1 = canon.input_values
        cqf = canon.output_values[0]
        assert cqf.metadata.qfixed is not None
        assert cqf.metadata.qfixed.qubit_uuids == (cq0.uuid, cq1.uuid)

    def test_indexed_carrier_keys_preserve_suffixes(self):
        """Legacy carrier keys remap their root UUID without losing indices."""
        source = ArrayValue(type=QubitType(), name="q")
        qf = Value(
            type=FloatType(),
            name="qf",
            metadata=ValueMetadata(
                cast=CastMetadata(
                    source_uuid=source.uuid,
                    source_logical_id=source.logical_id,
                    qubit_uuids=(f"{source.uuid}_1", f"{source.uuid}_3"),
                    qubit_logical_ids=(
                        f"{source.logical_id}_1",
                        f"{source.logical_id}_3",
                    ),
                ),
                qfixed=QFixedMetadata(
                    qubit_uuids=(f"{source.uuid}_1", f"{source.uuid}_3"),
                    num_bits=2,
                    int_bits=0,
                ),
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(f"{source.uuid}_1", f"{source.uuid}_3"),
                    element_logical_ids=(
                        f"{source.logical_id}_1",
                        f"{source.logical_id}_3",
                    ),
                ),
            ),
        )
        block = _make_minimal_block(input_values=[source], output_values=[qf])

        canon = canonicalize(block)

        canon_source = canon.input_values[0]
        canon_qf = canon.output_values[0]
        assert canon_qf.metadata.cast is not None
        assert canon_qf.metadata.cast.qubit_uuids == (
            f"{canon_source.uuid}_1",
            f"{canon_source.uuid}_3",
        )
        assert canon_qf.metadata.cast.qubit_logical_ids == (
            f"{canon_source.logical_id}_1",
            f"{canon_source.logical_id}_3",
        )
        assert canon_qf.metadata.qfixed is not None
        assert canon_qf.metadata.qfixed.qubit_uuids == (
            f"{canon_source.uuid}_1",
            f"{canon_source.uuid}_3",
        )
        assert canon_qf.metadata.array_runtime is not None
        assert canon_qf.metadata.array_runtime.element_uuids == (
            f"{canon_source.uuid}_1",
            f"{canon_source.uuid}_3",
        )

    def test_cast_operation_mapping_preserves_index_suffixes(self):
        """CastOperation.qubit_mapping remaps composite-key bases only."""
        source = ArrayValue(type=QubitType(), name="q")
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        qf = Value(type=result_type, name="qf")
        op = CastOperation(
            operands=[source],
            results=[qf],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{source.uuid}_1", f"{source.uuid}_3"],
        )
        block = _make_minimal_block(
            input_values=[source],
            output_values=[qf],
            operations=[op],
        )

        canon = canonicalize(block)

        canon_source = canon.input_values[0]
        canon_op = canon.operations[0]
        assert isinstance(canon_op, CastOperation)
        assert canon_op.qubit_mapping == [
            f"{canon_source.uuid}_1",
            f"{canon_source.uuid}_3",
        ]

    def test_array_runtime_element_uuids_rewritten(self):
        """ArrayRuntimeMetadata element-UUID lists track canonical UUIDs."""
        e0 = Value(type=FloatType(), name="e0")
        e1 = Value(type=FloatType(), name="e1")
        parent = ArrayValue(type=QubitType(), name="parent")
        arr = ArrayValue(
            type=FloatType(),
            name="arr",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(e0.uuid, e1.uuid),
                    element_logical_ids=(e0.logical_id, e1.logical_id),
                    element_parent_uuids=(parent.uuid, ""),
                    element_parent_indices=(1, -1),
                ),
            ),
        )
        block = _make_minimal_block(
            input_values=[e0, e1, parent],
            output_values=[arr],
        )
        canon = canonicalize(block)
        ce0, ce1, cparent = canon.input_values
        carr = canon.output_values[0]
        assert carr.metadata.array_runtime is not None
        assert carr.metadata.array_runtime.element_uuids == (ce0.uuid, ce1.uuid)
        assert carr.metadata.array_runtime.element_logical_ids == (
            ce0.logical_id,
            ce1.logical_id,
        )
        assert carr.metadata.array_runtime.element_parent_uuids == (
            cparent.uuid,
            "",
        )
        assert carr.metadata.array_runtime.element_parent_indices == (1, -1)

    def test_scalar_metadata_preserved_verbatim(self):
        """ScalarMetadata (no UUID refs) carries through unchanged."""
        v = Value(
            type=FloatType(),
            name="theta",
            metadata=ValueMetadata(scalar=ScalarMetadata(parameter_name="theta")),
        )
        block = _make_minimal_block(input_values=[v], output_values=[v])
        canon = canonicalize(block)
        assert canon.input_values[0].metadata.scalar == ScalarMetadata(
            parameter_name="theta"
        )


class TestCarrierKeyCanonicalStability:
    """Canonical-form stability for blocks carrying composite carrier keys."""

    @staticmethod
    def _build_cast_block() -> Block:
        """Build a structurally fixed cast block with fresh UUIDs.

        Returns:
            Block: Minimal AFFINE block with a CastOperation whose result
                carries composite carrier keys in cast / qfixed metadata and
                in ``CastOperation.qubit_mapping``.
        """
        source = ArrayValue(type=QubitType(), name="q")
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        qf = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=source.uuid,
                source_logical_id=source.logical_id,
                qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                qubit_logical_ids=[
                    f"{source.logical_id}_0",
                    f"{source.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        op = CastOperation(
            operands=[source],
            results=[qf],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{source.uuid}_0", f"{source.uuid}_1"],
        )
        return _make_minimal_block(
            input_values=[source],
            output_values=[qf],
            operations=[op],
        )

    def test_content_hash_deterministic_across_builds(self):
        """Two fresh builds of the same cast kernel hash identically."""
        assert content_hash(self._build_cast_block()) == content_hash(
            self._build_cast_block()
        )

    def test_canonicalize_idempotent_with_carrier_keys(self):
        """Re-canonicalizing a canonical block with carrier keys is stable."""
        canon = canonicalize(self._build_cast_block())
        again = canonicalize(canon)
        assert to_canonical_bytes(canon) == to_canonical_bytes(again)


# ---------------------------------------------------------------------------
# Composite Value types: Tuple / Dict
# ---------------------------------------------------------------------------


class TestCompositeValueTypes:
    """Canonicalize must descend into TupleValue / DictValue / ArrayValue."""

    def test_tuple_value_elements_rewritten(self):
        """Each element of a ``TupleValue`` gets its own canonical UUID."""
        a = Value(type=FloatType(), name="a")
        b = Value(type=UIntType(), name="b")
        tup = TupleValue(name="pair", elements=(a, b))
        block = _make_minimal_block(
            input_values=[a, b],
            output_values=cast(list[Value], [tup]),
        )
        canon = canonicalize(block)
        ca, cb = canon.input_values
        ctup = canon.output_values[0]
        assert isinstance(ctup, TupleValue)
        assert tuple(e.uuid for e in ctup.elements) == (ca.uuid, cb.uuid)

    def test_dict_value_entries_rewritten(self):
        """Each key/value of a ``DictValue`` gets canonical UUIDs."""
        k = Value(type=UIntType(), name="k")
        v = Value(type=FloatType(), name="v")
        dv = DictValue(name="m", entries=((k, v),))
        block = _make_minimal_block(
            input_values=[k, v],
            output_values=cast(list[Value], [dv]),
        )
        canon = canonicalize(block)
        ck, cv = canon.input_values
        cdv = canon.output_values[0]
        assert isinstance(cdv, DictValue)
        assert cdv.entries[0][0].uuid == ck.uuid
        assert cdv.entries[0][1].uuid == cv.uuid

    def test_array_value_shape_rewritten(self):
        """``ArrayValue.shape`` dimension Values are canonicalized too."""
        n = Value(type=UIntType(), name="n")
        arr = ArrayValue(type=QubitType(), name="qs", shape=(n,))
        block = _make_minimal_block(input_values=[n], output_values=[arr])
        canon = canonicalize(block)
        cn = canon.input_values[0]
        carr = canon.output_values[0]
        assert isinstance(carr, ArrayValue)
        assert carr.shape[0].uuid == cn.uuid


# ---------------------------------------------------------------------------
# Parameters ordering
# ---------------------------------------------------------------------------


class TestParametersOrdering:
    """``Block.parameters`` is walked in sorted-key order, so insertion order
    of the underlying dict does not affect canonical form.
    """

    def test_parameters_walked_in_sorted_key_order(self):
        """Same parameters with different insertion order produce equal bytes."""
        alpha = Value(
            type=FloatType(),
            name="alpha",
            metadata=ValueMetadata(scalar=ScalarMetadata(parameter_name="alpha")),
        )
        beta = Value(
            type=FloatType(),
            name="beta",
            metadata=ValueMetadata(scalar=ScalarMetadata(parameter_name="beta")),
        )
        block_in_order = _make_minimal_block(
            parameters={"alpha": alpha, "beta": beta},
            input_values=[alpha, beta],
            output_values=[alpha, beta],
        )
        block_reversed = _make_minimal_block(
            parameters={"beta": beta, "alpha": alpha},
            input_values=[alpha, beta],
            output_values=[alpha, beta],
        )
        assert to_canonical_bytes(block_in_order) == to_canonical_bytes(block_reversed)


@qmc.qkernel
def _slice_view_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Vector layer inverted on a slice view by the twin kernels below."""
    qs = qmc.h(qs)
    qs = qmc.s(qs)
    return qs


@qmc.qkernel
def _slice_view_inverse_a() -> qmc.Vector[qmc.Bit]:
    """Kernel whose inverse block operand is a strided slice view."""
    qs = qmc.qubit_array(4, "qs")
    view = qs[1:4]
    view = qmc.inverse(_slice_view_layer)(view)
    qs[1:4] = view
    return qmc.measure(qs)


@qmc.qkernel
def _slice_view_inverse_b() -> qmc.Vector[qmc.Bit]:
    """Structural twin of ``_slice_view_inverse_a``."""
    qs = qmc.qubit_array(4, "qs")
    view = qs[1:4]
    view = qmc.inverse(_slice_view_layer)(view)
    qs[1:4] = view
    return qmc.measure(qs)


class TestCanonicalizeSliceViews:
    """Slice-view refs are remapped into the canonical UUID space."""

    def test_slice_refs_remapped_to_canonical_values(self):
        """``slice_of`` / ``slice_start`` / ``slice_step`` are canonicalized.

        The view operand of an inverse block references its root array
        and slice bounds by Value identity; canonicalize must rewrite
        those references with the same mapping as every other use of
        the root array, otherwise the canonical block leaks build-time
        UUIDs and content hashes diverge between identical builds.
        """
        from qamomile.circuit.ir.operation.inverse_block import (
            InverseBlockOperation,
        )
        from qamomile.circuit.ir.operation.operation import QInitOperation

        block = _to_affine(_slice_view_inverse_a)
        original_operand = cast(
            ArrayValue,
            next(
                op for op in block.operations if isinstance(op, InverseBlockOperation)
            ).target_qubits[0],
        )
        assert original_operand.slice_of is not None

        canonical = canonicalize(block)
        operand = cast(
            ArrayValue,
            next(
                op
                for op in canonical.operations
                if isinstance(op, InverseBlockOperation)
            ).target_qubits[0],
        )
        root = next(
            op.results[0]
            for op in canonical.operations
            if isinstance(op, QInitOperation)
        )

        assert operand.slice_of is not None
        assert operand.slice_of.uuid != original_operand.slice_of.uuid
        assert operand.slice_of.uuid == root.uuid
        assert operand.slice_start is not None
        assert operand.slice_start.get_const() == 1
        assert operand.slice_step is not None
        assert operand.slice_step.get_const() == 1

    def test_slice_view_twins_same_hash(self):
        """Two identical builds with slice-view operands hash equally."""
        hash_a = content_hash(_to_affine(_slice_view_inverse_a))
        hash_b = content_hash(_to_affine(_slice_view_inverse_b))
        assert hash_a == hash_b


# ---------------------------------------------------------------------------
# param_slots: preservation, serialize round-trip, hash participation
# ---------------------------------------------------------------------------


def _block_with_bound_slot(bound_value: np.ndarray) -> Block:
    """Build a minimal AFFINE Block whose manifest carries ``bound_value``.

    The Block has no values or operations, so any content-hash
    difference between two such Blocks is attributable to the
    ``param_slots`` manifest alone.

    Args:
        bound_value (np.ndarray): Payload stored on the single
            ``COMPILE_TIME_BOUND`` slot.

    Returns:
        Block: An otherwise-empty AFFINE Block with a one-slot
            ``param_slots`` manifest.
    """
    return Block(
        name="manual",
        kind=BlockKind.AFFINE,
        param_slots=(
            ParamSlot(
                name="weights",
                type=FloatType(),
                kind=ParamKind.COMPILE_TIME_BOUND,
                ndim=1,
                bound_value=bound_value,
            ),
        ),
    )


class TestParamSlotsPreservation:
    """``Block.param_slots`` must survive canonicalize and serialization."""

    def test_canonicalize_preserves_param_slots_verbatim(self):
        """A non-empty manifest is carried over unchanged.

        Regression: the canonical Block used to be rebuilt without
        ``param_slots``, silently resetting the manifest to ``()``.
        """
        block = _to_affine(_h_then_rx)
        assert block.param_slots, "fixture kernel must produce a non-empty manifest"
        canon = canonicalize(block)
        assert canon.param_slots == block.param_slots

    def test_canonicalize_and_remap_preserves_param_slots(self):
        """The remap-table variant carries the manifest too."""
        block = _to_affine(_h_then_rx)
        canon, _, _ = canonicalize_and_remap(block)
        assert canon.param_slots == block.param_slots


class TestParamSlotsHashParticipation:
    """``param_slots`` is functional and participates in ``content_hash``."""

    def test_dropping_manifest_changes_hash(self):
        """Stripping a non-empty manifest changes the content hash."""
        block = _to_affine(_h_then_rx)
        stripped = dataclasses.replace(block, param_slots=())
        assert content_hash(block) != content_hash(stripped)

    def test_slot_kind_changes_hash(self):
        """Rebinding a slot from RUNTIME_PARAMETER to COMPILE_TIME_BOUND changes the hash."""
        block = _to_affine(_h_then_rx)
        rebound = dataclasses.replace(
            block,
            param_slots=tuple(
                dataclasses.replace(
                    slot, kind=ParamKind.COMPILE_TIME_BOUND, bound_value=0.5
                )
                for slot in block.param_slots
            ),
        )
        assert content_hash(block) != content_hash(rebound)

    def test_equal_numpy_bound_values_hash_equal(self):
        """Separately constructed arrays with equal content hash identically."""
        hash_a = content_hash(_block_with_bound_slot(np.array([0.1, 0.2, 0.3])))
        hash_b = content_hash(_block_with_bound_slot(np.array([0.1, 0.2, 0.3])))
        assert hash_a == hash_b

    def test_numpy_bound_value_shape_changes_hash(self):
        """A scalar array and a length-one array have distinct identities."""
        scalar = np.array(0.5)
        vector = np.array([0.5])
        assert scalar.tobytes() == vector.tobytes()
        assert content_hash(_block_with_bound_slot(scalar)) != content_hash(
            _block_with_bound_slot(vector)
        )

    def test_large_numpy_bound_values_hash_by_content(self):
        """Arrays indistinguishable under truncated ``repr`` still hash apart."""
        base = np.zeros(2048)
        tweaked = base.copy()
        tweaked[1500] = 1.0
        # Pin printoptions so the repr-collision precondition is
        # deterministic regardless of any global ``np.set_printoptions``:
        # 2048 > threshold elides the middle (where index 1500 differs),
        # so both arrays render identically and repr-based hashing would
        # collide on this pair.
        with np.printoptions(threshold=1000, edgeitems=3):
            assert repr(base) == repr(tweaked)
        assert content_hash(_block_with_bound_slot(base)) != content_hash(
            _block_with_bound_slot(tweaked)
        )


# ---------------------------------------------------------------------------
# Structural payload hashing (I1): no repr dependence for known payload types
# ---------------------------------------------------------------------------


def _block_with_payload_slot(payload: object) -> Block:
    """Build a minimal AFFINE Block whose manifest carries ``payload``.

    Generic variant of ``_block_with_bound_slot`` for non-ndarray
    payloads (Hamiltonians, numpy scalars, plain floats). Any
    content-hash difference between two such Blocks is attributable to
    the slot's ``bound_value`` alone.

    Args:
        payload (object): Payload stored on the single
            ``COMPILE_TIME_BOUND`` slot.

    Returns:
        Block: An otherwise-empty AFFINE Block with a one-slot
            ``param_slots`` manifest.
    """
    return Block(
        name="manual",
        kind=BlockKind.AFFINE,
        param_slots=(
            ParamSlot(
                name="payload",
                type=FloatType(),
                kind=ParamKind.COMPILE_TIME_BOUND,
                ndim=0,
                bound_value=payload,
            ),
        ),
    )


def _block_with_metadata_value(value: Value) -> Block:
    """Build a minimal AFFINE Block whose sole input carries ``value``.

    Args:
        value (Value): A Value whose metadata (``array_runtime`` /
            ``dict_runtime``) is the payload under test.

    Returns:
        Block: An otherwise-empty AFFINE Block with one input value.
    """
    return Block(
        name="manual",
        label_args=[value.name],
        input_values=[value],
        kind=BlockKind.AFFINE,
    )


def _xz_hamiltonian(*, reverse: bool = False, z_coeff: float = 0.5) -> "Hamiltonian":
    """Build a two-term Hamiltonian with controllable term insertion order.

    Args:
        reverse (bool): Insert the Z term before the X term when True, so
            the term-dict iteration order (and thus ``repr``) differs
            while ``Hamiltonian.__eq__`` still holds.
        z_coeff (float): Coefficient of the Z term. Vary it to produce a
            genuinely different Hamiltonian.

    Returns:
        Hamiltonian: ``1.0 * X0 + z_coeff * Z1`` with the requested
            insertion order.
    """
    h = Hamiltonian()
    terms = [
        ((PauliOperator(Pauli.X, 0),), 1.0),
        ((PauliOperator(Pauli.Z, 1),), z_coeff),
    ]
    if reverse:
        terms.reverse()
    for ops, coeff in terms:
        h.add_term(ops, coeff)
    return h


class TestStructuralPayloadHashing:
    """Known payload types hash structurally, without ``repr`` dependence.

    Regression class for I1: ``content_hash`` used to stringify
    Hamiltonian (and other opaque) payloads via ``repr``, making the
    hash depend on term insertion order and on ``__repr__`` formatting
    stability. Payloads the wire format supports are now emitted
    structurally.
    """

    def test_hamiltonian_bound_value_term_order_independent(self):
        """Equal Hamiltonians with different term insertion order hash equally."""
        h_fwd = _xz_hamiltonian()
        h_rev = _xz_hamiltonian(reverse=True)
        # Precondition: the two are ==-equal but repr-distinct, so a
        # repr-based hash would tell them apart.
        assert h_fwd == h_rev
        assert repr(h_fwd) != repr(h_rev)
        assert content_hash(_block_with_payload_slot(h_fwd)) == content_hash(
            _block_with_payload_slot(h_rev)
        )

    def test_hamiltonian_bound_value_content_sensitive(self):
        """Hamiltonians with different coefficients hash differently."""
        assert content_hash(
            _block_with_payload_slot(_xz_hamiltonian(z_coeff=0.5))
        ) != content_hash(_block_with_payload_slot(_xz_hamiltonian(z_coeff=0.7)))

    def test_hamiltonian_declared_width_changes_hash(self):
        """The declared register width participates in the hash.

        ``Hamiltonian.__eq__`` ignores ``num_qubits``, but the declared
        width changes circuit behavior, so the hash includes it.
        """
        narrow = _xz_hamiltonian()
        wide = Hamiltonian(num_qubits=5)
        for ops, coeff in narrow.terms.items():
            wide.add_term(ops, coeff)
        assert narrow == wide
        assert content_hash(_block_with_payload_slot(narrow)) != content_hash(
            _block_with_payload_slot(wide)
        )

    def test_hamiltonian_in_const_array_term_order_independent(self):
        """``const_array`` Hamiltonian payloads hash structurally too."""

        def block_for(h: Hamiltonian) -> Block:
            value = Value(FloatType(), name="obs_vec").with_array_runtime_metadata(
                const_array=(h,)
            )
            return _block_with_metadata_value(value)

        block_fwd = block_for(_xz_hamiltonian())
        block_rev = block_for(_xz_hamiltonian(reverse=True))
        assert content_hash(block_fwd) == content_hash(block_rev)
        block_other = block_for(_xz_hamiltonian(z_coeff=0.7))
        assert content_hash(block_fwd) != content_hash(block_other)

    def test_hamiltonian_in_dict_runtime_term_order_independent(self):
        """``dict_runtime.bound_data`` Hamiltonian payloads hash structurally."""

        def block_for(h: Hamiltonian) -> Block:
            value = Value(FloatType(), name="coeffs").with_dict_runtime_metadata(
                {"obs": h}
            )
            return _block_with_metadata_value(value)

        block_fwd = block_for(_xz_hamiltonian())
        block_rev = block_for(_xz_hamiltonian(reverse=True))
        assert content_hash(block_fwd) == content_hash(block_rev)
        block_other = block_for(_xz_hamiltonian(z_coeff=0.7))
        assert content_hash(block_fwd) != content_hash(block_other)

    def test_numpy_scalar_bound_value_matches_python_float(self):
        """``np.float64(x)`` and ``x`` hash identically (both structural)."""
        assert content_hash(_block_with_payload_slot(np.float64(0.5))) == content_hash(
            _block_with_payload_slot(0.5)
        )
