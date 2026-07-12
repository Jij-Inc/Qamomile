"""Tests for the IR serialization pipeline (JSON + msgpack).

The intermediate dict schema lives in
:mod:`qamomile.circuit.ir.serialize.schema`. Encoders and decoders
should produce byte-stable round-trips for ``AFFINE`` / ``ANALYZED``
Blocks across a representative spread of IR features (gates,
measurement, control flow, ControlledU, composite gates).
"""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o

# Real algorithm kernels used by TestRealAlgorithmRoundTrip below.
from qamomile.circuit.algorithm.basic import (  # noqa: E402
    cz_entangling_layer,
    phase_gadget,
    rx_layer,
    superposition_vector,
)
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.canonical import content_hash
from qamomile.circuit.ir.operation import (
    ForItemsOperation,
    GateOperation,
    GateOperationType,
    InverseBlockOperation,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.callable import CallTransform, CompositeGateType
from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.serialize import (
    SCHEMA_VERSION,
    dump_json,
    dump_msgpack,
    from_dict,
    load_json,
    load_msgpack,
    to_dict,
)
from qamomile.circuit.ir.serialize.hamiltonian_io import (
    dict_to_hamiltonian,
    hamiltonian_to_dict,
)
from qamomile.circuit.ir.serialize.numpy_io import array_to_dict, dict_to_array
from qamomile.circuit.ir.types.primitives import (
    FloatType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    DictValue,
    ScalarMetadata,
    TupleValue,
    Value,
    ValueMetadata,
)
from qamomile.circuit.stdlib.qft import iqft, qft  # noqa: E402
from qamomile.circuit.transpiler.passes.affine_validate import AffineValidationPass
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.partial_eval import PartialEvaluationPass
from qamomile.circuit.transpiler.passes.slice_borrow_check import SliceBorrowCheckPass
from qamomile.circuit.transpiler.passes.strip_slice_ops import StripSliceArrayOpsPass

# ---------------------------------------------------------------------------
# Fixture kernels (representative IR shapes)
# ---------------------------------------------------------------------------


def _to_affine(kernel: qmc.QKernel) -> Block:
    """Return an AFFINE block for ``kernel`` without a backend transpiler.

    Args:
        kernel (qmc.QKernel): A ``@qkernel``-decorated function.

    Returns:
        Block: The kernel's traced block after running ``InlinePass``.
    """
    return InlinePass().run(kernel.block)


def _to_analyzed(block: Block) -> Block:
    """Advance an AFFINE block to ANALYZED without a backend transpiler.

    Mirrors the canonical pipeline order (affine_validate →
    partial_eval → slice_borrow_check → strip_slice_ops → analyze) so
    the resulting block matches what ``Transpiler.transpile`` hands to
    serialization-stage consumers.

    Args:
        block (Block): An AFFINE block (e.g. from :func:`_to_affine`).

    Returns:
        Block: The block with ``BlockKind.ANALYZED``.
    """
    block = AffineValidationPass().run(block)
    block = PartialEvaluationPass().run(block)
    block = SliceBorrowCheckPass().run(block)
    block = StripSliceArrayOpsPass().run(block)
    return AnalyzePass().run(block)


@qmc.qkernel
def _scalar_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit kernel with a Float parameter."""
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _measure_kernel(q: qmc.Qubit) -> qmc.Bit:
    """Kernel that exercises a measurement-derived classical bit."""
    q = qmc.h(q)
    return qmc.measure(q)


@qmc.qkernel
def _if_merge_kernel() -> qmc.Bit:
    """Measurement-conditioned if-else whose branches rebind a qubit."""
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    r = qmc.qubit(name="r")
    if bit:
        r = qmc.x(r)
    else:
        r = qmc.h(r)
    return qmc.measure(r)


@qmc.qkernel
def _inverse_kernel(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Kernel that emits a first-class inverse block operation."""
    q = qmc.inverse(_scalar_gate)(q, theta)
    return q


@qmc.qkernel
def _nested_inverse_inner(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single nested helper for inverse source-block serialization."""
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _nested_inverse_outer(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Outer helper that leaves an inline InvokeOperation before inline."""
    q = qmc.h(q)
    q = _nested_inverse_inner(q, theta)
    return q


@qmc.qkernel
def _nested_inverse_kernel(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Kernel whose inverse source block contains a nested call."""
    q = qmc.inverse(_nested_inverse_outer)(q, theta)
    return q


@qmc.qkernel
def _view_layer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Vector layer applied to a slice view by ``_view_inverse_kernel``."""
    qs = qmc.h(qs)
    qs = qmc.s(qs)
    return qs


@qmc.qkernel
def _view_inverse_kernel() -> qmc.Vector[qmc.Bit]:
    """Kernel whose inverse block operand is a strided slice view."""
    qs = qmc.qubit_array(4, "qs")
    view = qs[1:4]
    view = qmc.inverse(_view_layer)(view)
    qs[1:4] = view
    return qmc.measure(qs)


@qmc.qkernel
def _loop_kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Symbolic ``ForOperation`` over a Vector input."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.h(qs[i])
    return qs


@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit phase rotation used inside a controlled-U body."""
    return qmc.p(q, theta)


@qmc.composite_gate(name="boxed_phase")
def _boxed_phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit phase rotation exposed as a composite callable."""
    return qmc.p(q, theta)


@qmc.qkernel
def _controlled_phase(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Top-level kernel that embeds a ``ControlledUOperation``."""
    op = qmc.control(_phase)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


@qmc.qkernel
def _controlled_boxed_phase(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Top-level kernel that controls a composite callable."""
    op = qmc.control(_boxed_phase)
    ctrl, target = op(ctrl, target, theta=theta)
    return ctrl, target


@qmc.qkernel
def _inverse_boxed_phase(target: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Top-level kernel that inverts a composite callable."""
    inverse_phase = qmc.inverse(_boxed_phase)
    target = inverse_phase(target, theta=theta)
    return target


@qmc.qkernel
def _controlled_phase_symbolic_power(
    ctrl: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float, power: qmc.UInt
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Top-level kernel with a controlled-U symbolic power operand."""
    op = qmc.control(_phase)
    ctrl, target = op(ctrl, target, theta=theta, power=power)
    return ctrl, target


@qmc.qkernel
def _symbolic_multi_arg_controlled(
    ctrl_main: qmc.Qubit,
    prefix: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    nc: qmc.UInt,
) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Top-level kernel that embeds a multi-arg ``SymbolicControlledU``.

    The call site supplies two positional control arguments
    (``ctrl_main`` scalar Qubit + ``prefix`` Vector) ahead of the
    sub-kernel's positional ``target``.  This forces
    ``num_control_args == 2`` on the emitted ``SymbolicControlledU``,
    which the serializer must round-trip; without persisting the
    field, the decoded op would re-split ``operands`` at the default
    boundary of ``num_control_args == 1`` and route ``prefix`` into
    the sub-kernel's target slot.
    """
    op = qmc.control(qmc.x, num_controls=nc)
    ctrl_main, prefix, target = op(ctrl_main, prefix, target)
    return ctrl_main, prefix, target


@qmc.qkernel
def _symbolic_control_indices_controlled(
    pool: qmc.Vector[qmc.Qubit],
    target: qmc.Qubit,
    nc: qmc.UInt,
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
    """Top-level kernel that embeds ``control_indices`` on SymbolicControlledU."""
    op = qmc.control(qmc.x, num_controls=nc)
    pool, target = op(pool, target, control_indices=[0, 1, 3])
    return pool, target


@qmc.qkernel
def _while_measure_kernel() -> qmc.Bit:
    """Kernel that exercises a loop-carried measurement condition."""
    q = qmc.qubit("q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    while bit:
        q2 = qmc.qubit("q2")
        q2 = qmc.h(q2)
        bit = qmc.measure(q2)
    return bit


@qmc.qkernel
def _if_branch_rebind_kernel() -> qmc.Bit:
    """Kernel whose IfOperation carries a non-empty ``branch_rebinds``.

    ``q`` is measured before the if, so rebinding it to the external
    ``fresh`` register in both branches is legal variable reuse (the
    pre-branch value is owned outside the if) rather than a discard.
    Because both branches rebind ``q`` to a value whose old binding is
    dead, the pre-branch value survives only on the recorded
    ``BranchRebind``, giving the serializer a non-empty record list to
    round-trip.
    """
    q = qmc.qubit("q")
    fresh = qmc.qubit("fresh")
    qmc.measure(q)
    p = qmc.qubit("p")
    cond = qmc.measure(p)
    if cond:
        q = fresh
    else:
        q = fresh
    return qmc.measure(q)


@qmc.qkernel
def _loop_quantum_rebind_kernel(n: qmc.UInt) -> qmc.Bit:
    """Kernel whose ForOperation carries a quantum ``LoopCarriedRebind``.

    Gives the serializer a quantum-``before`` loop record to round-trip.
    The kernel is only built to AFFINE here — transpiling it would be
    rejected by the control-flow discard check (unrolled loops carry no
    register between iterations), which is irrelevant to the wire
    format under test.
    """
    q = qmc.qubit("q")
    b = qmc.bit(0)
    for _ in qmc.range(n):
        b = qmc.measure(q)
        q = qmc.qubit("fresh")
    return b


@qmc.qkernel
def _for_items_ising_kernel(
    n_qubits: qmc.UInt,
    ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Kernel that exercises ForItemsOperation over a DictValue operand."""
    q = qmc.qubit_array(n_qubits, name="q")
    for (i, j), jij in qmc.items(ising):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * jij)
    return q


@qmc.qkernel
def _slice_assignment_kernel() -> qmc.Vector[qmc.Bit]:
    """Kernel that exercises literal slice assignment markers."""
    q = qmc.qubit_array(4, name="q")
    q[1::2] = qmc.h(q[1::2])
    return qmc.measure(q)


@qmc.qkernel
def _nested_slice_assignment_kernel() -> qmc.Vector[qmc.Bit]:
    """Kernel that exercises nested slice views and releases."""
    q = qmc.qubit_array(8, name="q")
    evens = q[0::2]
    evens[0:2] = qmc.x(evens[0:2])
    q[0::2] = evens
    return qmc.measure(q)


@qmc.qkernel
def _for_items_vector_key_kernel(
    n_qubits: qmc.UInt,
    interactions: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Kernel that exercises ``key_is_vector=True`` ForItemsOperation."""
    q = qmc.qubit_array(n_qubits, name="q")
    for key, coeff in qmc.items(interactions):
        for step in qmc.range(key.shape[0] - 1):
            q[key[step]], q[key[step + 1]] = qmc.rzz(
                q[key[step]], q[key[step + 1]], gamma * coeff
            )
    return q


@qmc.qkernel
def _for_items_classical_kernel(
    n_qubits: qmc.UInt,
    ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Measure-terminated for-items kernel (classical I/O for analyze)."""
    q = qmc.qubit_array(n_qubits, name="q")
    for (i, j), jij in qmc.items(ising):
        q[i], q[j] = qmc.rzz(q[i], q[j], gamma * jij)
    return qmc.measure(q)


@qmc.qkernel
def _dict_getitem_kernel(
    n_qubits: qmc.UInt,
    ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    gammas: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    biases: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Kernel that exercises DictGetItemOperation with 1- and 2-ary keys."""
    q = qmc.qubit_array(n_qubits, name="q")
    for (i, j), jij in qmc.items(ising):
        q[i] = qmc.rz(q[i], angle=biases[i])
        q[i], q[j] = qmc.rzz(q[i], q[j], jij * gammas[(i, j)])
    return qmc.measure(q)


@qmc.qkernel
def _inverse_dict_param_layer(
    q: qmc.Qubit,
    angles: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Qubit:
    """Layer whose unused dict parameter rides along as a DictValue operand."""
    q = qmc.h(q)
    return q


@qmc.qkernel
def _inverse_dict_param_kernel(
    angles: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Qubit:
    """Kernel whose InverseBlockOperation carries a DictValue operand."""
    q = qmc.qubit("q")
    q = qmc.inverse(_inverse_dict_param_layer)(q, angles)
    return q


@qmc.qkernel
def _controlled_nonleaf_inner(q: qmc.Qubit) -> qmc.Qubit:
    """Leaf helper called by the non-leaf controlled unitary."""
    q = qmc.h(q)
    return q


@qmc.qkernel
def _controlled_nonleaf_unitary(q: qmc.Qubit) -> qmc.Qubit:
    """Non-leaf unitary: its cached block contains an inline InvokeOperation."""
    q = _controlled_nonleaf_inner(q)
    q = qmc.x(q)
    return q


@qmc.qkernel
def _controlled_nonleaf_kernel(
    ctrl: qmc.Qubit, target: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Kernel controlling a non-leaf sub-kernel.

    Before InlinePass descended into ``ControlledUOperation.block``,
    the nested call survived inlining, the block was still reported as
    AFFINE, and serialization died on the residual inline InvokeOperation.
    """
    op = qmc.control(_controlled_nonleaf_unitary)
    ctrl, target = op(ctrl, target)
    return ctrl, target


@qmc.qkernel
def _trotter_observable_kernel(
    q: qmc.Vector[qmc.Qubit], Hs: qmc.Vector[qmc.Observable], t: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Trotter-style kernel whose ``Vector[Observable]`` is bound at build time.

    Binding ``Hs`` bakes the ``Hamiltonian`` objects into
    ``ParamSlot.bound_value`` and ``ArrayRuntimeMetadata.const_array``,
    which the serializer must carry through the ``$hamiltonian``
    payload wrapper.
    """
    for i in qmc.range(Hs.shape[0]):
        q = qmc.pauli_evolve(q, Hs[i], t)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Smoke + structure preservation
# ---------------------------------------------------------------------------


class TestRoundTripStructure:
    """Encoder + decoder preserve kind, operation order, and param_slots."""

    def test_json_round_trip_preserves_kind_and_op_count(self):
        """``load_json(dump_json(block))`` produces structurally similar IR."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert restored.kind == block.kind
        assert len(restored.operations) == len(block.operations)
        assert [type(op).__name__ for op in restored.operations] == [
            type(op).__name__ for op in block.operations
        ]

    def test_msgpack_round_trip_preserves_kind_and_op_count(self):
        """msgpack round-trip mirrors the JSON one."""
        block = _to_affine(_scalar_gate)
        restored = load_msgpack(dump_msgpack(block))
        assert restored.kind == block.kind
        assert len(restored.operations) == len(block.operations)

    def test_label_args_preserved(self):
        """``label_args`` round-trips byte-for-byte."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert restored.label_args == block.label_args

    def test_param_slots_preserved(self):
        """``param_slots`` round-trips with name, kind, ndim intact."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert len(restored.param_slots) == len(block.param_slots)
        for original, restored_slot in zip(block.param_slots, restored.param_slots):
            assert original.name == restored_slot.name
            assert original.kind == restored_slot.kind
            assert original.ndim == restored_slot.ndim
            assert isinstance(restored_slot.type, type(original.type))

    def test_inverse_block_round_trip_preserves_source_and_fallback(self):
        """InverseBlockOperation persists its source and fallback blocks."""
        block = _to_affine(_inverse_kernel)
        restored = load_json(dump_json(block))
        inverse_ops = [
            op for op in restored.operations if isinstance(op, InverseBlockOperation)
        ]

        assert len(inverse_ops) == 1
        assert inverse_ops[0].source_block is not None
        assert inverse_ops[0].source_block.name == "_scalar_gate"
        assert inverse_ops[0].implementation_block is not None
        assert inverse_ops[0].implementation_block.name.endswith("_inverse")

    def test_inverse_block_nested_blocks_are_inlined(self):
        """Nested inverse source and fallback blocks can serialize and hash."""
        block = _to_affine(_nested_inverse_kernel)
        inverse_op = next(
            op for op in block.operations if isinstance(op, InverseBlockOperation)
        )

        assert inverse_op.source_block is not None
        assert inverse_op.implementation_block is not None
        assert inverse_op.source_block.kind is BlockKind.AFFINE
        assert inverse_op.implementation_block.kind is BlockKind.AFFINE
        assert not any(
            isinstance(op, InvokeOperation) and op.attrs.get("kind") == "qkernel"
            for op in inverse_op.source_block.operations
        )
        assert not any(
            isinstance(op, InvokeOperation) and op.attrs.get("kind") == "qkernel"
            for op in inverse_op.implementation_block.operations
        )

        restored = load_json(dump_json(block))
        assert load_msgpack(dump_msgpack(block)).kind == block.kind
        assert content_hash(restored) == content_hash(block)

    def test_inverse_block_view_operand_round_trips_slice_fields(self):
        """A sliced view operand keeps its slice refs across the round-trip.

        ``ArrayValue.slice_of`` / ``slice_start`` / ``slice_step`` carry
        the affine map that resolves view elements to root-array qubits
        at emit time; dropping them on the wire would make the restored
        inverse block unresolvable.
        """
        block = _to_affine(_view_inverse_kernel)
        original = next(
            op for op in block.operations if isinstance(op, InverseBlockOperation)
        )
        original_operand = cast(ArrayValue, original.target_qubits[0])
        assert original_operand.slice_of is not None

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            inverse_op = next(
                op
                for op in restored.operations
                if isinstance(op, InverseBlockOperation)
            )
            operand = cast(ArrayValue, inverse_op.target_qubits[0])
            assert operand.slice_of is not None
            assert operand.slice_of.uuid == original_operand.slice_of.uuid
            assert operand.slice_start is not None
            assert operand.slice_start.get_const() == 1
            assert operand.slice_step is not None
            assert operand.slice_step.get_const() == 1
            assert content_hash(restored) == content_hash(block)

    def test_input_value_types_preserved(self):
        """Input Value types match across the round-trip."""
        block = _to_affine(_scalar_gate)
        restored = load_json(dump_json(block))
        assert [type(v.type).__name__ for v in restored.input_values] == [
            type(v.type).__name__ for v in block.input_values
        ]


class TestRoundTripIRFeatures:
    """Round-trip works for IR features beyond plain gate sequences."""

    def test_measurement_round_trip(self):
        """Kernels with ``MeasureOperation`` survive both wire formats."""
        block = _to_affine(_measure_kernel)
        assert load_json(dump_json(block)).kind == block.kind
        assert load_msgpack(dump_msgpack(block)).kind == block.kind

    def test_for_loop_round_trip(self):
        """Symbolic ``ForOperation`` survives serialization."""
        block = _to_affine(_loop_kernel)
        restored = load_json(dump_json(block))
        # The top-level operation list should retain the for loop wrapper.
        op_names = [type(op).__name__ for op in restored.operations]
        assert "ForOperation" in op_names

    def test_controlled_u_round_trip(self):
        """``ControlledUOperation`` with nested unitary block round-trips."""
        block = _to_affine(_controlled_phase)
        restored = load_json(dump_json(block))
        op_names = [type(op).__name__ for op in restored.operations]
        assert any(name.endswith("ControlledU") for name in op_names), op_names

    def test_controlled_u_callable_ref_and_attrs_round_trip(self):
        """Controlled-U keeps the wrapped qkernel callable identity and attrs."""
        from qamomile.circuit.ir.operation.gate import ControlledUOperation

        block = _to_affine(_controlled_phase)
        controlled_ops = [
            op for op in block.operations if isinstance(op, ControlledUOperation)
        ]
        assert controlled_ops
        assert {op.callable_ref.name for op in controlled_ops if op.callable_ref} == {
            "_phase"
        }
        assert {op.callable_attrs["kind"] for op in controlled_ops} == {"qkernel"}
        assert {op.callable_attrs["default_policy"] for op in controlled_ops} == {
            "INLINE"
        }

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            restored_ops = [
                op for op in restored.operations if isinstance(op, ControlledUOperation)
            ]
            assert [op.callable_ref.name for op in restored_ops if op.callable_ref] == [
                "_phase"
            ]
            assert [op.callable_attrs["kind"] for op in restored_ops] == ["qkernel"]
            assert [op.callable_attrs["default_policy"] for op in restored_ops] == [
                "INLINE"
            ]

    def test_controlled_composite_callable_ref_and_attrs_round_trip(self):
        """Controlled composite Invoke keeps one callable definition."""

        block = _to_affine(_controlled_boxed_phase)
        controlled_ops = [
            op for op in block.operations if isinstance(op, InvokeOperation)
        ]
        assert controlled_ops
        assert {op.target.namespace for op in controlled_ops} == {"user.composite"}
        assert {op.target.name for op in controlled_ops} == {"boxed_phase"}
        assert {op.transform for op in controlled_ops} == {CallTransform.CONTROLLED}
        assert {op.attrs["kind"] for op in controlled_ops} == {"composite"}
        assert {op.attrs["default_policy"] for op in controlled_ops} == {"PRESERVE_BOX"}

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            restored_ops = [
                op for op in restored.operations if isinstance(op, InvokeOperation)
            ]
            assert [op.target.namespace for op in restored_ops] == ["user.composite"]
            assert [op.target.name for op in restored_ops] == ["boxed_phase"]
            assert [op.transform for op in restored_ops] == [CallTransform.CONTROLLED]
            assert [op.attrs["kind"] for op in restored_ops] == ["composite"]
            assert [op.attrs["default_policy"] for op in restored_ops] == [
                "PRESERVE_BOX"
            ]

    def test_inverse_composite_callable_ref_and_attrs_round_trip(self):
        """Inverse composite Invoke keeps one callable definition."""
        block = _to_affine(_inverse_boxed_phase)
        inverse_ops = [op for op in block.operations if isinstance(op, InvokeOperation)]
        assert inverse_ops
        assert {op.target.namespace for op in inverse_ops} == {"user.composite"}
        assert {op.target.name for op in inverse_ops} == {"boxed_phase"}
        assert {op.transform for op in inverse_ops} == {CallTransform.INVERSE}
        assert {op.attrs["kind"] for op in inverse_ops} == {"composite"}
        assert {op.attrs["default_policy"] for op in inverse_ops} == {"PRESERVE_BOX"}

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            restored_ops = [
                op for op in restored.operations if isinstance(op, InvokeOperation)
            ]
            assert [op.target.namespace for op in restored_ops] == ["user.composite"]
            assert [op.target.name for op in restored_ops] == ["boxed_phase"]
            assert [op.transform for op in restored_ops] == [CallTransform.INVERSE]
            assert [op.attrs["kind"] for op in restored_ops] == ["composite"]
            assert [op.attrs["default_policy"] for op in restored_ops] == [
                "PRESERVE_BOX"
            ]

    def test_controlled_u_symbolic_power_round_trip(self):
        """Controlled-U symbolic ``power`` survives both wire formats."""
        block = _to_affine(_controlled_phase_symbolic_power)
        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original

    def test_symbolic_controlled_u_preserves_num_control_args(self):
        """Multi-arg ``SymbolicControlledU`` keeps its operand layout across encode/decode.

        Without persisting ``num_control_args``, the decoder defaults
        the field to ``1`` -- correct for the legacy single-pool form
        but wrong whenever the call site supplied a multi-arg control
        prefix.  The downstream emit pass then splits ``operands`` at
        the wrong boundary, so the regression must catch the field on
        both wire formats and on a non-trivial source op (here,
        ``num_control_args == 2`` for a scalar + Vector prefix).
        """
        from qamomile.circuit.ir.operation.gate import SymbolicControlledU

        block = _to_affine(_symbolic_multi_arg_controlled)
        sym_ops = [op for op in block.operations if isinstance(op, SymbolicControlledU)]
        assert sym_ops, [type(op).__name__ for op in block.operations]
        original_args = [op.num_control_args for op in sym_ops]
        assert all(n > 1 for n in original_args), original_args

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            restored_sym = [
                op for op in restored.operations if isinstance(op, SymbolicControlledU)
            ]
            assert [op.num_control_args for op in restored_sym] == original_args

    def test_symbolic_controlled_u_preserves_control_indices(self):
        """``control_indices`` keeps its selected control slots across round-trip."""
        from qamomile.circuit.ir.operation.gate import SymbolicControlledU

        block = _to_affine(_symbolic_control_indices_controlled)
        sym_ops = [op for op in block.operations if isinstance(op, SymbolicControlledU)]
        assert sym_ops, [type(op).__name__ for op in block.operations]
        original_indices = [
            tuple(v.get_const() for v in op.control_indices or ()) for op in sym_ops
        ]
        assert original_indices == [(0, 1, 3)]

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            restored_sym = [
                op for op in restored.operations if isinstance(op, SymbolicControlledU)
            ]
            restored_indices = [
                tuple(v.get_const() for v in op.control_indices or ())
                for op in restored_sym
            ]
            assert restored_indices == original_indices

    def test_symbolic_controlled_u_legacy_payload_decodes_as_single_pool(self):
        """A payload missing ``num_control_args`` decodes as the legacy form.

        Pre-existing v1 payloads (and any single-pool op the encoder
        omits the field for, for compactness) must keep working: the
        decoder defaults ``num_control_args = 1`` so the operand split
        matches the legacy single-pool form's expectations.
        """
        from qamomile.circuit.ir.operation.gate import SymbolicControlledU

        block = _to_affine(_symbolic_multi_arg_controlled)
        payload = to_dict(block)
        # Strip the field everywhere the encoder wrote it; the legacy
        # contract is "absent field <=> single-pool form".
        for op_dict in payload["block"]["operations"]:
            op_dict.pop("num_control_args", None)
        restored = from_dict(payload)
        restored_sym = [
            op for op in restored.operations if isinstance(op, SymbolicControlledU)
        ]
        assert restored_sym, [type(op).__name__ for op in restored.operations]
        assert all(op.num_control_args == 1 for op in restored_sym)

    def test_msgpack_smaller_than_json_for_loop_kernel(self):
        """msgpack should produce at least as compact bytes as JSON.

        Acts as a regression guard against accidental textification of
        binary fields in the msgpack path.
        """
        block = _to_affine(_loop_kernel)
        assert len(dump_msgpack(block)) <= len(dump_json(block))

    def test_while_operation_round_trip(self):
        """Loop-carried while conditions survive both wire formats."""
        block = _to_affine(_while_measure_kernel)
        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original
        assert content_hash(load_json(dump_json(block))) == content_hash(block)

    def test_for_items_operation_round_trip(self):
        """Dict-backed ForItemsOperation survives both wire formats."""
        block = _to_affine(_for_items_ising_kernel)
        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original
        assert content_hash(load_json(dump_json(block))) == content_hash(block)

    def test_if_operation_merge_round_trip(self):
        """IfOperation branch merges survive both wire formats as yield refs.

        The wire form encodes merges as ``true_yield_refs`` /
        ``false_yield_refs`` UUID lists parallel to ``result_refs`` (no
        embedded phi operation dicts); the decoded op must reproduce the
        original ``iter_merges`` view exactly.
        """
        block = _to_affine(_if_merge_kernel)
        if_ops = [op for op in block.operations if isinstance(op, IfOperation)]
        assert if_ops, [type(op).__name__ for op in block.operations]
        original_merges = [
            [
                (m.true_value.uuid, m.false_value.uuid, m.result.uuid)
                for m in op.iter_merges()
            ]
            for op in if_ops
        ]
        assert all(original_merges), original_merges

        payload = to_dict(block)
        if_dicts = [
            d for d in payload["block"]["operations"] if d.get("$type") == "IfOperation"
        ]
        assert if_dicts
        for d in if_dicts:
            assert "phi_ops" not in d
            assert (
                len(d["true_yield_refs"])
                == len(d["false_yield_refs"])
                == len(d["result_refs"])
            )

        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original

        restored = load_json(dump_json(block))
        restored_merges = [
            [
                (m.true_value.uuid, m.false_value.uuid, m.result.uuid)
                for m in op.iter_merges()
            ]
            for op in restored.operations
            if isinstance(op, IfOperation)
        ]
        assert restored_merges == original_merges
        assert content_hash(restored) == content_hash(block)

    def test_if_operation_mismatched_yield_refs_rejected(self):
        """Corrupted yield-ref lists are rejected loudly at decode time."""
        block = _to_affine(_if_merge_kernel)
        payload = to_dict(block)
        corrupted = False
        for op_dict in payload["block"]["operations"]:
            if op_dict.get("$type") == "IfOperation":
                op_dict["true_yield_refs"] = op_dict["true_yield_refs"][:-1]
                corrupted = True
        assert corrupted
        with pytest.raises(ValueError, match="merge data is inconsistent"):
            from_dict(payload)

    def test_if_operation_legacy_phi_ops_payload_rejected(self):
        """A pre-yields payload carrying the removed 'phi_ops' field fails loud.

        The length check only catches a legacy op that also kept its merge
        outputs in ``results``; an explicit ``phi_ops`` guard is what keeps
        the "old payloads are rejected, never silently down-decoded"
        contract true for a ``phi_ops``-only shape.
        """
        block = _to_affine(_if_merge_kernel)
        payload = to_dict(block)
        injected = False
        for op_dict in payload["block"]["operations"]:
            if op_dict.get("$type") == "IfOperation":
                # Simulate an old-revision IfOperation: the merges lived
                # under the removed ``phi_ops`` field, not the yield refs.
                op_dict.pop("true_yield_refs", None)
                op_dict.pop("false_yield_refs", None)
                op_dict["phi_ops"] = []
                injected = True
        assert injected
        with pytest.raises(ValueError, match="phi_ops"):
            from_dict(payload)

    def test_if_branch_rebinds_round_trip(self):
        """IfOperation branch-rebind records survive both wire formats."""
        block = _to_affine(_if_branch_rebind_kernel)
        if_ops = [op for op in block.operations if isinstance(op, IfOperation)]
        assert if_ops and if_ops[0].branch_rebinds, (
            "expected a non-empty branch_rebinds to exercise the round-trip"
        )

        original = to_dict(block)
        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == original
            restored_ifs = [
                op for op in restored.operations if isinstance(op, IfOperation)
            ]
            assert restored_ifs
            restored_records = restored_ifs[0].branch_rebinds
            assert [
                (r.var_name, r.before.uuid, r.rebound_in_true, r.rebound_in_false)
                for r in restored_records
            ] == [
                (r.var_name, r.before.uuid, r.rebound_in_true, r.rebound_in_false)
                for r in if_ops[0].branch_rebinds
            ]
            assert content_hash(restored) == content_hash(block)

    def test_loop_quantum_rebind_records_round_trip(self):
        """Quantum-``before`` loop rebind records survive both wire formats."""
        block = _to_affine(_loop_quantum_rebind_kernel)
        for_ops = [op for op in block.operations if isinstance(op, ForOperation)]
        assert for_ops and any(
            r.before.type.is_quantum() for r in for_ops[0].loop_carried_rebinds
        ), "expected a quantum loop rebind record to exercise the round-trip"

        original = to_dict(block)
        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == original
            restored_fors = [
                op for op in restored.operations if isinstance(op, ForOperation)
            ]
            assert restored_fors
            assert [
                (r.var_name, r.before.uuid, r.after.uuid, r.before_synthesized)
                for r in restored_fors[0].loop_carried_rebinds
            ] == [
                (r.var_name, r.before.uuid, r.after.uuid, r.before_synthesized)
                for r in for_ops[0].loop_carried_rebinds
            ]
            assert content_hash(restored) == content_hash(block)

    def test_for_items_vector_key_round_trip(self):
        """``key_is_vector=True`` ForItemsOperation survives both formats."""
        block = _to_affine(_for_items_vector_key_kernel)
        for_items_ops = [
            op for op in block.operations if isinstance(op, ForItemsOperation)
        ]
        assert for_items_ops and for_items_ops[0].key_is_vector

        original = to_dict(block)
        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == original
            restored_ops = [
                op for op in restored.operations if isinstance(op, ForItemsOperation)
            ]
            assert restored_ops and restored_ops[0].key_is_vector

    def test_dict_getitem_round_trip(self):
        """DictGetItemOperation survives both wire formats with key_arity."""
        from qamomile.circuit.ir.operation.classical_ops import (
            DictGetItemOperation,
        )

        block = _to_affine(_dict_getitem_kernel)

        def collect_arities(blk: Block) -> list[int]:
            arities: list[int] = []
            for op in blk.operations:
                if isinstance(op, ForItemsOperation):
                    arities.extend(
                        nested.key_arity
                        for nested in op.operations
                        if isinstance(nested, DictGetItemOperation)
                    )
            return sorted(arities)

        assert collect_arities(block) == [1, 2]

        original = to_dict(block)
        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == original
            assert collect_arities(restored) == [1, 2]
            assert content_hash(restored) == content_hash(block)

    def test_bound_dict_metadata_round_trip(self):
        """Bound-dict tuple keys survive decode as real tuples.

        The wire formats flatten tuples to lists; the decoder must
        re-freeze ``dict_runtime.bound_data`` so a deserialized block
        still supports ``DictValue.get_bound_data()`` (dict keys must
        be hashable) and ``content_hash``.
        """
        bound = {(0, 1): 0.5, (1, 2): -0.3}
        block = InlinePass().run(
            _for_items_classical_kernel.build(n_qubits=3, ising=bound, gamma=0.7)
        )

        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == to_dict(block)
            dict_values = [v for v in restored.input_values if isinstance(v, DictValue)]
            assert dict_values
            assert dict_values[0].get_bound_data() == bound
            assert content_hash(restored) == content_hash(block)

    def test_inverse_block_dict_parameter_round_trip(self):
        """InverseBlockOperation with a DictValue operand survives decode."""
        block = _to_affine(_inverse_dict_param_kernel)
        inverse_ops = [
            op for op in block.operations if isinstance(op, InverseBlockOperation)
        ]
        assert inverse_ops
        assert any(
            isinstance(operand, DictValue) for operand in inverse_ops[0].operands
        )

        original = to_dict(block)
        for restored in (
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert to_dict(restored) == original
            restored_inverse = [
                op
                for op in restored.operations
                if isinstance(op, InverseBlockOperation)
            ]
            assert restored_inverse
            assert any(
                isinstance(operand, DictValue)
                for operand in restored_inverse[0].operands
            )

    def test_controlled_u_nonleaf_kernel_round_trip(self):
        """A controlled non-leaf kernel inlines fully and round-trips.

        InlinePass must descend into ``ControlledUOperation.block`` so
        AFFINE keeps meaning "no residual inline InvokeOperation anywhere
        reachable"; the serializer rejects residual calls.
        """
        from qamomile.circuit.ir.operation.gate import ControlledUOperation

        block = _to_affine(_controlled_nonleaf_kernel)
        assert block.kind is BlockKind.AFFINE
        controlled_ops = [
            op for op in block.operations if isinstance(op, ControlledUOperation)
        ]
        assert controlled_ops
        nested = controlled_ops[0].block
        assert nested is not None
        assert not any(
            isinstance(op, InvokeOperation) and op.attrs.get("kind") == "qkernel"
            for op in nested.operations
        )

        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original
        assert content_hash(load_json(dump_json(block))) == content_hash(block)

    @pytest.mark.parametrize(
        "kernel",
        [
            _slice_assignment_kernel,
            _nested_slice_assignment_kernel,
        ],
    )
    def test_slice_view_operations_round_trip(self, kernel: qmc.QKernel):
        """Frontend slice views survive both wire formats."""
        block = _to_affine(kernel)
        op_names = [type(op).__name__ for op in block.operations]
        assert "SliceArrayOperation" in op_names
        assert "ReleaseSliceViewOperation" in op_names

        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original
        assert content_hash(load_json(dump_json(block))) == content_hash(block)

    @pytest.mark.parametrize(
        "kernel",
        [
            _while_measure_kernel,
            _for_items_classical_kernel,
        ],
    )
    def test_round_trip_at_analyzed_stage(self, kernel: qmc.QKernel):
        """The ANALYZED half of the serialization contract also round-trips.

        ``from_dict`` accepts AFFINE and ANALYZED; the while / for-items
        fixtures must survive both formats after the analyze pass, not
        just right after inline.
        """
        block = _to_analyzed(_to_affine(kernel))
        assert block.kind is BlockKind.ANALYZED
        original = to_dict(block)
        assert to_dict(load_json(dump_json(block))) == original
        assert to_dict(load_msgpack(dump_msgpack(block))) == original
        assert content_hash(load_json(dump_json(block))) == content_hash(block)


class TestCastCarrierKeyRoundTrip:
    """Legacy composite cast/QFixed carrier keys survive serialization.

    Carrier keys use the ``"<root_uuid>_<index>"`` spelling and live in
    ``CastOperation.qubit_mapping`` and in the cast / QFixed metadata of the
    cast result. The wire formats treat them as opaque strings, so a
    round-trip must preserve them verbatim — and therefore preserve the
    ``content_hash`` of a block carrying them.
    """

    @staticmethod
    def _carrier_keys(block: Block) -> list[tuple[str, tuple[str, ...]]]:
        """Collect every composite carrier key in a block, order-stable.

        Args:
            block (Block): Block to scan for cast/QFixed carrier keys.

        Returns:
            list[tuple[str, tuple[str, ...]]]: ``(source, keys)`` pairs for the
                ``CastOperation.qubit_mapping`` and the cast / QFixed metadata
                of each cast result, in operation order.
        """
        from qamomile.circuit.ir.operation.cast import CastOperation

        collected: list[tuple[str, tuple[str, ...]]] = []
        for op in block.operations:
            if isinstance(op, CastOperation):
                collected.append(("qubit_mapping", tuple(op.qubit_mapping)))
            for result in op.results:
                metadata = getattr(result, "metadata", None)
                if metadata is None:
                    continue
                if metadata.cast is not None:
                    collected.append(("cast", tuple(metadata.cast.qubit_uuids)))
                if metadata.qfixed is not None:
                    collected.append(("qfixed", tuple(metadata.qfixed.qubit_uuids)))
        return collected

    def test_whole_array_cast_carrier_keys_round_trip(self):
        """A sub-kernel cast carrier survives all three wire formats.

        Inlining ``sub(q)`` rebases the carrier keys onto the caller's array;
        the round-trip must keep those rebased keys byte-identical.
        """

        @qmc.qkernel
        def sub(q: qmc.Vector[qmc.Qubit]) -> qmc.Float:
            return qmc.measure(qmc.cast(q, qmc.QFixed, int_bits=0))

        @qmc.qkernel
        def circuit() -> qmc.Float:
            anc = qmc.qubit("anc")
            q = qmc.qubit_array(2, "q")
            anc = qmc.x(anc)
            return sub(q)

        block = _to_affine(circuit)
        original_keys = self._carrier_keys(block)
        # The cast carries two carrier qubits, so the keys are non-empty.
        assert any(keys for _, keys in original_keys), original_keys

        for restored in (
            from_dict(to_dict(block)),
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert content_hash(restored) == content_hash(block)
            assert self._carrier_keys(restored) == original_keys

    def test_slice_view_cast_carrier_keys_round_trip(self):
        """Root-space carrier keys from a strided-view cast round-trip intact.

        ``sub(q[1::2])`` folds the carrier indices into the root array's index
        space (``_1`` / ``_3``); the round-trip must preserve that folding.
        """

        @qmc.qkernel
        def sub(q: qmc.Vector[qmc.Qubit]) -> qmc.Float:
            return qmc.measure(qmc.cast(q, qmc.QFixed, int_bits=0))

        @qmc.qkernel
        def circuit() -> qmc.Float:
            q = qmc.qubit_array(4, "q")
            return sub(q[1::2])

        block = _to_affine(circuit)
        original_keys = self._carrier_keys(block)
        # Every carrier key should be a composite ``<uuid>_<index>`` key whose
        # index is in root space (1 or 3), proving the fold happened pre-encode.
        flat = [key for _, keys in original_keys for key in keys]
        assert flat, original_keys
        assert all(key.rsplit("_", 1)[1] in {"1", "3"} for key in flat), flat

        for restored in (
            from_dict(to_dict(block)),
            load_json(dump_json(block)),
            load_msgpack(dump_msgpack(block)),
        ):
            assert content_hash(restored) == content_hash(block)
            assert self._carrier_keys(restored) == original_keys


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    """Schema-version envelope is enforced."""

    def test_envelope_has_current_version(self):
        """``to_dict`` emits the current ``SCHEMA_VERSION`` in the envelope."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        assert d["schema_version"] == SCHEMA_VERSION

    def test_load_rejects_future_version(self):
        """A higher ``schema_version`` than the loader supports raises."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["schema_version"] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema_version"):
            from_dict(d)

    def test_load_rejects_past_version(self):
        """A lower ``schema_version`` raises until migration is implemented."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["schema_version"] = 0
        with pytest.raises(ValueError, match="schema_version"):
            from_dict(d)


# ---------------------------------------------------------------------------
# numpy wrapper
# ---------------------------------------------------------------------------


class TestNumpyWrapper:
    """The numpy ndarray wrapper preserves dtype, shape, and values."""

    @pytest.mark.parametrize(
        "dtype",
        ["float64", "float32", "int64", "int32", "uint8", "complex128"],
    )
    def test_array_round_trip_dtype(self, dtype):
        """Each allow-listed dtype round-trips losslessly."""
        rng = np.random.default_rng(seed=0)
        arr = (rng.random(size=(2, 3)) * 100).astype(dtype)
        restored = dict_to_array(array_to_dict(arr))
        assert restored.dtype.name == arr.dtype.name
        assert restored.shape == arr.shape
        assert np.array_equal(restored, arr)

    def test_disallowed_dtype_rejected(self):
        """A dtype outside the allow-list raises on encode."""
        arr = np.array(["a", "b"])
        with pytest.raises(ValueError, match="allow-list"):
            array_to_dict(arr)

    def test_decoder_rejects_corrupt_length(self):
        """A length mismatch in the wrapper raises on decode."""
        wrapper = array_to_dict(np.array([1.0, 2.0, 3.0]))
        wrapper["data"] = wrapper["data"][:-1]  # truncate one byte
        with pytest.raises(ValueError, match="data length"):
            dict_to_array(wrapper)

    @pytest.mark.parametrize("flag", [True, False])
    def test_decoder_rejects_bool_shape_dim(self, flag):
        """A boolean shape dimension is rejected (bool is not a plain int).

        ``json.loads('[true, 2]')`` yields ``[True, 2]`` and ``bool``
        subclasses ``int``, so a bare ``isinstance(x, int)`` would accept
        the dim and silently reshape with ``True == 1``.
        """
        wrapper = array_to_dict(np.array([1.0, 2.0]))
        wrapper["shape"] = [flag, 2]
        with pytest.raises(ValueError, match="list of ints"):
            dict_to_array(wrapper)

    def test_numpy_array_in_param_slot_bound_value(self):
        """A numpy ``bound_value`` on a ParamSlot round-trips via JSON.

        Construct a Block manually carrying a numpy bound_value so we
        can verify both the encoder routes through ``array_to_dict``
        and the JSON path converts bytes ↔ base64 transparently.
        """
        original = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        slot = ParamSlot(
            name="thetas",
            type=FloatType(),
            kind=ParamKind.COMPILE_TIME_BOUND,
            ndim=1,
            bound_value=original,
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            param_slots=(slot,),
        )
        restored = load_json(dump_json(block))
        restored_slot = restored.param_slots[0]
        assert isinstance(restored_slot.bound_value, np.ndarray)
        assert np.array_equal(restored_slot.bound_value, original)
        assert restored_slot.bound_value.dtype == original.dtype


# ---------------------------------------------------------------------------
# Hamiltonian payloads (bound Observable parameters)
# ---------------------------------------------------------------------------


class TestHamiltonianPayloadRoundTrip:
    """A bound ``Vector[Observable]`` kernel round-trips its Hamiltonians."""

    @pytest.mark.parametrize(
        "dump, load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
        ids=["json", "msgpack"],
    )
    def test_bound_observable_vector_round_trip(self, dump, load):
        """``to_dict`` / ``content_hash`` equality plus payload fidelity.

        The documented Trotter pattern binds a list of Hamiltonians at
        build time; both wire formats must reproduce the exact
        intermediate dict, the canonical content hash, and Hamiltonian
        payloads equal to the originals in both ``ParamSlot.bound_value``
        and the Hs array's ``const_array`` metadata.
        """
        hs = [1.2 * qm_o.Z(0), 0.8 * qm_o.X(0)]
        block = InlinePass().run(
            _trotter_observable_kernel.build(Hs=hs, parameters=["t"])
        )
        restored = load(dump(block))

        assert to_dict(restored) == to_dict(block)
        assert content_hash(restored) == content_hash(block)

        slot = next(s for s in restored.param_slots if s.name == "Hs")
        assert slot.kind == ParamKind.COMPILE_TIME_BOUND
        assert isinstance(slot.bound_value, list)
        assert slot.bound_value == hs
        for restored_h, original_h in zip(slot.bound_value, hs):
            assert isinstance(restored_h, qm_o.Hamiltonian)
            assert repr(restored_h) == repr(original_h)
            assert restored_h.num_qubits == original_h.num_qubits

        hs_value = next(v for v in restored.input_values if v.name == "Hs")
        assert hs_value.metadata.array_runtime is not None
        const_array = hs_value.metadata.array_runtime.const_array
        assert const_array is not None
        assert list(const_array) == hs


class TestHamiltonianWrapper:
    """Direct unit tests for the ``$hamiltonian`` payload codec."""

    def test_wire_shape_matches_schema(self):
        """The wrapper dict has exactly the documented shape."""
        wrapper = hamiltonian_to_dict(1.2 * qm_o.Z(0))
        assert wrapper == {
            "$hamiltonian": True,
            "terms": [[[["Z", 0]], 1.2]],
            "constant": 0.0,
            "num_qubits": 1,
        }

    def test_round_trip_preserves_term_order_and_complex_coeffs(self):
        """Complex coefficients and term insertion order survive the wrapper.

        ``X(0) * Y(0)`` folds to ``1j * Z(0)`` and is inserted after the
        ``Y(2)`` term, so the term dict is NOT sorted by qubit index —
        order preservation is what keeps ``repr`` (and thus
        ``content_hash``) stable across the round-trip.
        """
        h = (1 + 0.5j) * qm_o.Y(2) + qm_o.X(0) * qm_o.Y(0)
        restored = dict_to_hamiltonian(hamiltonian_to_dict(h))
        assert restored == h
        assert repr(restored) == repr(h)
        assert list(restored.terms) == list(h.terms)
        for original_coeff, restored_coeff in zip(
            h.terms.values(), restored.terms.values()
        ):
            assert type(restored_coeff) is type(original_coeff)
            assert restored_coeff == original_coeff

    def test_multi_qubit_product_round_trip(self):
        """A multi-operator Pauli product keeps every factor."""
        h = 0.7 * qm_o.Z(0) * qm_o.Z(1) + 0.3 * qm_o.X(0)
        restored = dict_to_hamiltonian(hamiltonian_to_dict(h))
        assert restored == h
        assert repr(restored) == repr(h)

    def test_float_coeff_stays_float(self):
        """A real coefficient is not widened to complex by the round-trip."""
        h = 1.2 * qm_o.Z(0)
        restored = dict_to_hamiltonian(hamiltonian_to_dict(h))
        (coeff,) = restored.terms.values()
        assert type(coeff) is float
        assert coeff == 1.2

    def test_constant_and_declared_num_qubits_preserved(self):
        """An int constant and the declared register width round-trip."""
        h = qm_o.Hamiltonian.identity(2, num_qubits=5)
        restored = dict_to_hamiltonian(hamiltonian_to_dict(h))
        assert restored.constant == 2
        assert type(restored.constant) is int
        assert restored.num_qubits == 5

    def test_numpy_scalar_coefficients_serialize_as_plain_numbers(self):
        """numpy scalar coefficients (np.float64 / np.complex128) round-trip.

        Real Hamiltonian builders carry numpy scalars — e.g. a coefficient
        from ``np.sqrt(...)`` is an ``np.float64`` — which are not Python
        ``float`` / ``complex`` instances. The wrapper coerces them via
        ``.item()`` so they serialize like plain numbers (and decode back to
        Python scalars) instead of being rejected as unencodable.
        """
        h = qm_o.Hamiltonian()
        h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), np.float64(np.sqrt(2.0)))
        h.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 1),), np.complex128(0.5 + 0.25j))
        h.constant = np.float64(2.0)
        restored = dict_to_hamiltonian(hamiltonian_to_dict(h))
        real_coeff, complex_coeff = restored.terms.values()
        assert type(real_coeff) is float
        assert type(complex_coeff) is complex
        assert type(restored.constant) is float
        # value fidelity: numpy scalars coerce to the equal Python scalar,
        # so the reconstructed Hamiltonian compares equal term-for-term.
        assert restored == h

    def test_numpy_bool_constant_rejected(self):
        """A ``numpy.bool_`` constant is rejected like a Python ``bool``.

        Hamiltonian coefficients are never boolean. Setting the constant to a
        ``numpy.bool_`` reaches the guard directly (unlike ``add_term``, whose
        internal ``phase * coeff`` would coerce a bool to a float first), and
        ``.item()`` collapses it to a Python ``bool`` so the guard fires.
        """
        h = qm_o.Hamiltonian()
        h.constant = np.bool_(True)
        with pytest.raises(TypeError, match="must not be bool"):
            hamiltonian_to_dict(h)

    def test_negative_num_qubits_rejected(self):
        """A negative declared register width is rejected on decode."""
        wrapper = hamiltonian_to_dict(qm_o.Hamiltonian.identity(1.0, num_qubits=2))
        wrapper["num_qubits"] = -1
        with pytest.raises(ValueError, match="non-negative int"):
            dict_to_hamiltonian(wrapper)

    def test_numpy_int_declared_num_qubits_coerced_on_encode(self):
        """A numpy integer declared width is coerced to a Python int on encode.

        ``Hamiltonian(num_qubits=np.int64(...))`` stores a numpy scalar; without
        coercion the wrapper would carry a non-JSON-serializable value and
        ``dump_json`` would fail. The encoder ``.item()``-coerces it to a plain
        int and the width round-trips.
        """
        h = qm_o.Hamiltonian(num_qubits=np.int64(3))
        h.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)
        wrapper = hamiltonian_to_dict(h)
        assert type(wrapper["num_qubits"]) is int
        assert wrapper["num_qubits"] == 3
        restored = dict_to_hamiltonian(wrapper)
        assert restored.num_qubits == h.num_qubits

    def test_negative_declared_num_qubits_rejected_on_encode(self):
        """A negative declared register width is rejected when encoding."""
        h = qm_o.Hamiltonian(num_qubits=-1)
        with pytest.raises(ValueError, match="non-negative"):
            hamiltonian_to_dict(h)

    def test_empty_operator_list_rejected(self):
        """A term with an empty operator list is rejected on decode.

        The encoder never emits an empty operator list — the constant lives in
        the dedicated ``constant`` field — so an empty list is malformed wire
        data that would otherwise fold into the constant via ``add_term`` and
        double-encode it.
        """
        wrapper = hamiltonian_to_dict(1.2 * qm_o.Z(0))
        wrapper["terms"].append([[], 3.0])
        with pytest.raises(ValueError, match="non-empty list"):
            dict_to_hamiltonian(wrapper)

    def test_negative_qubit_index_rejected(self):
        """A negative qubit index in a term operator is rejected on decode."""
        wrapper = hamiltonian_to_dict(1.2 * qm_o.Z(0))
        wrapper["terms"][0][0][0][1] = -1
        with pytest.raises(ValueError, match="non-negative int"):
            dict_to_hamiltonian(wrapper)

    def test_unknown_pauli_name_rejected(self):
        """A Pauli name outside the allow-map raises on decode."""
        wrapper = hamiltonian_to_dict(1.2 * qm_o.Z(0))
        wrapper["terms"][0][0][0][0] = "Q"
        with pytest.raises(ValueError, match="allow-map"):
            dict_to_hamiltonian(wrapper)

    def test_malformed_coefficient_rejected(self):
        """A non-numeric coefficient raises on decode."""
        wrapper = hamiltonian_to_dict(1.2 * qm_o.Z(0))
        wrapper["terms"][0][1] = "not-a-number"
        with pytest.raises(ValueError, match="coefficient"):
            dict_to_hamiltonian(wrapper)


# ---------------------------------------------------------------------------
# Safety: closed dispatch tables refuse unknown tags
# ---------------------------------------------------------------------------


class TestDecoderSafety:
    """The decoder refuses unknown ``$type`` tags rather than resolve them."""

    def test_unknown_operation_tag_rejected(self):
        """An unrecognized operation tag raises ``ValueError``."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["block"]["operations"][0]["$type"] = "__rogue__"
        with pytest.raises(ValueError, match="Operation"):
            from_dict(d)

    def test_unknown_value_type_tag_rejected(self):
        """An unrecognized value-type tag raises ``ValueError``."""
        block = _to_affine(_scalar_gate)
        d = to_dict(block)
        d["block"]["value_table"][0]["value_type"]["$type"] = "__rogue__"
        with pytest.raises(ValueError, match="ValueType"):
            from_dict(d)


# ---------------------------------------------------------------------------
# bool rejected where a plain int is required (bool subclasses int)
# ---------------------------------------------------------------------------


class TestRejectBoolInIntFields:
    """Width and power fields reject ``bool`` via ``is_plain_int``.

    ``bool`` is an ``int`` subclass, so a bare ``isinstance(x, int)``
    accepts ``True`` / ``False``. Register widths and ControlledU powers
    are integers where a boolean is meaningless, and on the decode side a
    malformed payload could otherwise smuggle a bool into an int slot.
    """

    @pytest.mark.parametrize("flag", [True, False])
    def test_encode_qreg_width_rejects_bool(self, flag):
        """``_encode_qreg_width`` refuses a bool width."""
        from qamomile.circuit.ir.serialize.encode import _encode_qreg_width

        with pytest.raises(TypeError, match="width must be int or Value"):
            _encode_qreg_width(flag)

    @pytest.mark.parametrize("flag", [True, False])
    def test_encode_power_rejects_bool(self, flag):
        """``_encode_power`` refuses a bool power."""
        from qamomile.circuit.ir.serialize.encode import _encode_power

        with pytest.raises(TypeError, match="power must be int or Value"):
            _encode_power(flag)

    @pytest.mark.parametrize("flag", [True, False])
    def test_decode_qreg_width_rejects_bool(self, flag):
        """``_decode_qreg_width`` refuses a bool width payload."""
        from qamomile.circuit.ir.serialize.decode import (
            _decode_qreg_width,
            _DecodeContext,
        )

        with pytest.raises(ValueError, match="unrecognized width payload"):
            _decode_qreg_width(flag, _DecodeContext([]))

    @pytest.mark.parametrize("flag", [True, False])
    def test_decode_power_rejects_bool_payload(self, flag):
        """A bool ``power`` in a ControlledU payload is rejected by ``from_dict``."""
        payload = to_dict(_to_affine(_controlled_phase))
        controlled = [
            op
            for op in payload["block"]["operations"]
            if op["$type"].endswith("ControlledU")
        ]
        assert controlled, [op["$type"] for op in payload["block"]["operations"]]
        controlled[0]["power"] = flag
        with pytest.raises(ValueError, match="unrecognized power payload"):
            from_dict(payload)


# ---------------------------------------------------------------------------
# Unsupported scope: HIERARCHICAL
# ---------------------------------------------------------------------------


class TestUnsupportedKind:
    """``to_dict`` rejects unsupported block kinds."""

    def test_rejects_hierarchical(self):
        """HIERARCHICAL blocks must be inlined first."""
        hierarchical = dataclasses.replace(
            _scalar_gate.block, kind=BlockKind.HIERARCHICAL
        )
        with pytest.raises(ValueError, match="AFFINE"):
            to_dict(hierarchical)


# ---------------------------------------------------------------------------
# Deterministic byte stability under canonical UUIDs
# ---------------------------------------------------------------------------


class TestEncodedShape:
    """The encoded dict has the documented top-level shape."""

    def test_envelope_keys(self):
        """The envelope carries exactly ``schema_version`` and ``block``."""
        d = to_dict(_to_affine(_scalar_gate))
        assert set(d) == {"schema_version", "block"}

    def test_block_has_value_table_and_operations(self):
        """The block dict includes ``value_table`` and ``operations`` lists."""
        block_dict = to_dict(_to_affine(_scalar_gate))["block"]
        assert isinstance(block_dict["value_table"], list)
        assert isinstance(block_dict["operations"], list)
        assert block_dict["$type"] == "Block"


# ---------------------------------------------------------------------------
# Manual construction with metadata UUID references
# ---------------------------------------------------------------------------


class TestManualConstruction:
    """Hand-built Blocks with metadata round-trip without going through tracing."""

    def test_array_runtime_metadata_round_trip(self):
        """``ArrayRuntimeMetadata`` element UUIDs round-trip exactly."""
        e0 = Value(type=FloatType(), name="e0")
        e1 = Value(type=FloatType(), name="e1")
        arr = ArrayValue(
            type=FloatType(),
            name="arr",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(e0.uuid, e1.uuid),
                    element_logical_ids=(e0.logical_id, e1.logical_id),
                    element_parent_uuids=("parent-array", ""),
                    element_parent_indices=(2, -1),
                ),
            ),
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[e0, e1],
            output_values=cast(list[Value], [arr]),
            label_args=["e0", "e1"],
            output_names=["arr"],
        )
        restored = load_json(dump_json(block))
        restored_arr = restored.output_values[0]
        assert restored_arr.metadata.array_runtime is not None
        assert restored_arr.metadata.array_runtime.element_uuids == (e0.uuid, e1.uuid)
        assert restored_arr.metadata.array_runtime.element_parent_uuids == (
            "parent-array",
            "",
        )
        assert restored_arr.metadata.array_runtime.element_parent_indices == (2, -1)

    def test_scalar_metadata_round_trip(self):
        """``ScalarMetadata`` parameter_name / const_value round-trip."""
        v = Value(
            type=FloatType(),
            name="theta",
            metadata=ValueMetadata(
                scalar=ScalarMetadata(const_value=0.5, parameter_name="theta")
            ),
        )
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[v],
            output_values=[v],
            label_args=["theta"],
        )
        restored = load_json(dump_json(block))
        assert restored.input_values[0].metadata.scalar == ScalarMetadata(
            const_value=0.5, parameter_name="theta"
        )

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_tuple_value_output_round_trip(self, dump, load):
        """Structured ``TupleValue`` block outputs survive serialization."""
        a = Value(type=UIntType(), name="a")
        b = Value(type=UIntType(), name="b")
        pair = TupleValue(name="pair", elements=(a, b))
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[a, b],
            output_values=[pair],
            label_args=["a", "b"],
            output_names=["pair"],
        )

        restored = load(dump(block))
        assert to_dict(restored) == to_dict(block)
        assert isinstance(restored.output_values[0], TupleValue)

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_nested_tuple_return_round_trip(self, dump, load):
        """Nested tuple outputs and Return operands survive serialization."""
        a = Value(type=UIntType(), name="a")
        b = Value(type=UIntType(), name="b")
        inner = TupleValue(name="inner", elements=(a, b))
        outer = TupleValue(name="outer", elements=(inner, a))
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[outer],
            output_values=[outer],
            label_args=["outer"],
            output_names=["outer"],
            operations=[ReturnOperation(operands=cast(list[Value], [outer]))],
        )

        restored = load(dump(block))

        assert to_dict(restored) == to_dict(block)
        restored_outer = restored.output_values[0]
        assert isinstance(restored_outer, TupleValue)
        assert isinstance(restored_outer.elements[0], TupleValue)
        restored_return = restored.operations[0]
        assert isinstance(restored_return, ReturnOperation)
        assert isinstance(restored_return.operands[0], TupleValue)

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_dict_value_output_round_trip(self, dump, load):
        """Structured ``DictValue`` block outputs survive serialization."""
        key = Value(type=UIntType(), name="key")
        value = Value(type=FloatType(), name="value")
        mapping = DictValue(name="mapping", entries=((key, value),))
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[key, value],
            output_values=[mapping],
            label_args=["key", "value"],
            output_names=["mapping"],
        )

        restored = load(dump(block))
        assert to_dict(restored) == to_dict(block)
        assert isinstance(restored.output_values[0], DictValue)

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_tuple_value_input_round_trip(self, dump, load):
        """Structured ``TupleValue`` block inputs survive serialization."""
        a = Value(type=UIntType(), name="pair_0")
        b = Value(type=UIntType(), name="pair_1")
        pair = TupleValue(name="pair", elements=(a, b))
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[pair],
            output_values=[a],
            label_args=["pair"],
        )

        restored = load(dump(block))
        assert to_dict(restored) == to_dict(block)
        assert isinstance(restored.input_values[0], TupleValue)

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_dict_value_input_round_trip(self, dump, load):
        """Structured ``DictValue`` block inputs survive serialization."""
        key = Value(type=UIntType(), name="key")
        value = Value(type=FloatType(), name="value")
        mapping = DictValue(name="mapping", entries=((key, value),))
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[mapping],
            output_values=[value],
            label_args=["mapping"],
        )

        restored = load(dump(block))
        assert to_dict(restored) == to_dict(block)
        assert isinstance(restored.input_values[0], DictValue)

    @pytest.mark.parametrize(
        "dump,load",
        [(dump_json, load_json), (dump_msgpack, load_msgpack)],
    )
    def test_structural_invoke_round_trip(self, dump, load):
        """Invoke operands and results preserve TupleValue and DictValue."""
        key = Value(type=UIntType(), name="key")
        value = Value(type=FloatType(), name="value")
        pair = TupleValue(name="pair", elements=(key, value))
        mapping = DictValue(name="mapping", entries=((key, value),))
        invoke = InvokeOperation(operands=[pair], results=[mapping])
        block = Block(
            name="manual",
            kind=BlockKind.AFFINE,
            input_values=[pair],
            output_values=[mapping],
            label_args=["pair"],
            output_names=["mapping"],
            operations=[invoke],
        )

        restored = load(dump(block))

        assert to_dict(restored) == to_dict(block)
        restored_invoke = restored.operations[0]
        assert isinstance(restored_invoke, InvokeOperation)
        assert isinstance(restored_invoke.operands[0], TupleValue)
        assert isinstance(restored_invoke.results[0], DictValue)

    def test_extra_x_gate_changes_bytes(self):
        """Adding a gate changes the serialized bytes (regression guard)."""
        block = _to_affine(_scalar_gate)
        baseline = dump_json(block)
        last_result = block.operations[-1].results[0]
        new_q = last_result.next_version()
        modified = dataclasses.replace(
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
        assert baseline != dump_json(modified)


# ---------------------------------------------------------------------------
# Real-algorithm round-trip
# ---------------------------------------------------------------------------


@qmc.qkernel
def _bell_state() -> qmc.Vector[qmc.Qubit]:
    """Bell state: H on q0, then CX(q0, q1). Two-qubit, two-op kernel."""
    qs = qmc.qubit_array(2, "qs")
    qs[0] = qmc.h(qs[0])
    qs[0], qs[1] = qmc.cx(qs[0], qs[1])
    return qs


@qmc.qkernel
def _ansatz_h_rx_cz(thetas: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Qubit]:
    """A small variational ansatz: superposition + Rx layer + CZ ladder.

    Stresses ``ForOperation`` over a symbolic ``Vector[Float]`` plus
    multi-qubit ``cz`` tuple assignment, both common shapes in real
    QAOA / VQE ansatze.
    """
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], thetas[i])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cz(q[i], q[i + 1])
    return q


@qmc.qkernel
def _algo_superposition_5() -> qmc.Vector[qmc.Qubit]:
    """Exercise the ``superposition_vector`` stdlib kernel via composition."""
    return superposition_vector(5)  # type: ignore[arg-type]


@qmc.qkernel
def _algo_rx_then_cz(
    thetas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """Compose two ``qamomile.circuit.algorithm.basic`` kernels.

    Forces ``InlinePass`` to flatten two distinct inline InvokeOperations
    and confirms the serialized form still survives.
    """
    n = thetas.shape[0]
    q = qmc.qubit_array(n, "q")
    q = rx_layer(q, thetas, 0)  # type: ignore[arg-type]
    q = cz_entangling_layer(q)
    return q


@qmc.qkernel
def _algo_phase_gadget(
    indices: qmc.Vector[qmc.UInt], angle: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Exercise the ``phase_gadget`` stdlib kernel (CX ladder + RZ).

    ``indices`` and ``angle`` are bound at build time so the gadget's
    ``k`` (number of qubits in the term) is concrete.
    """
    q = qmc.qubit_array(3, "q")
    return phase_gadget(q, indices, angle)


@qmc.qkernel
def _qft_3() -> qmc.Vector[qmc.Qubit]:
    """QFT on 3 qubits — exercises ``InvokeOperation`` + nested body."""
    qs = qmc.qubit_array(3, "qs")
    return qft(qs)


@qmc.qkernel
def _qft_then_iqft_4() -> qmc.Vector[qmc.Qubit]:
    """QFT followed by IQFT — two composite gates in sequence."""
    qs = qmc.qubit_array(4, "qs")
    qs = qft(qs)
    qs = iqft(qs)
    return qs


@qmc.qkernel
def _symbolic_qft(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Symbolic-width QFT — exercises deferred stdlib body references."""
    qs = qmc.qubit_array(n, "qs")
    qs = qft(qs)
    return qmc.measure(qs)


_REAL_KERNELS = [
    pytest.param(_bell_state, {}, id="bell"),
    pytest.param(
        _ansatz_h_rx_cz,
        {"thetas": np.array([0.1, 0.2, 0.3, 0.4])},
        id="ansatz_h_rx_cz",
    ),
    pytest.param(_algo_superposition_5, {}, id="superposition_5"),
    pytest.param(
        _algo_rx_then_cz,
        {"thetas": np.array([0.5, 0.6, 0.7])},
        id="rx_then_cz",
    ),
    pytest.param(
        _algo_phase_gadget,
        {"indices": np.array([0, 1, 2], dtype=np.int64), "angle": 0.7},
        id="phase_gadget",
    ),
    pytest.param(_qft_3, {}, id="qft_3"),
    pytest.param(_qft_then_iqft_4, {}, id="qft_then_iqft_4"),
]


def _build_real(kernel, build_kwargs):
    """Build an AFFINE block for a real-algorithm kernel.

    Uses ``kernel.build(**kwargs)`` when bindings are provided
    (typical for kernels with ``Vector[Float]`` parameters) and falls
    back to the cached ``kernel.block`` otherwise; either way the
    block is then run through ``InlinePass`` so the result is AFFINE.

    Args:
        kernel: A ``@qkernel``-decorated function.
        build_kwargs: Bindings for ``kernel.build``. Empty dict means
            use ``kernel.block`` directly.

    Returns:
        Block: The AFFINE block ready for serialization.
    """
    raw = kernel.build(**build_kwargs) if build_kwargs else kernel.block
    return InlinePass().run(raw)


class TestRealAlgorithmRoundTrip:
    """End-to-end round-trip on representative real-world kernels.

    Each kernel is built into an AFFINE block, serialized through one
    of the wire formats, decoded back, and re-encoded into the
    intermediate dict. ``to_dict`` equality on the two dicts is the
    strongest equivalence we can express without depending on the
    canonical-form pass (PR #389, separate branch): UUIDs round-trip
    verbatim, every operation field is preserved, and the value table
    is rebuilt in the same order.
    """

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_json_round_trip_to_dict_equal(self, kernel, build_kwargs):
        """``to_dict(load_json(dump_json(b))) == to_dict(b)`` for real kernels."""
        block = _build_real(kernel, build_kwargs)
        original = to_dict(block)
        restored = to_dict(load_json(dump_json(block)))
        assert restored == original

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_msgpack_round_trip_to_dict_equal(self, kernel, build_kwargs):
        """``to_dict(load_msgpack(dump_msgpack(b))) == to_dict(b)`` for real kernels."""
        block = _build_real(kernel, build_kwargs)
        original = to_dict(block)
        restored = to_dict(load_msgpack(dump_msgpack(block)))
        assert restored == original

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_round_trip_preserves_op_sequence(self, kernel, build_kwargs):
        """Operation type and order are byte-stable across the round-trip."""
        block = _build_real(kernel, build_kwargs)
        restored = load_json(dump_json(block))
        assert [type(op).__name__ for op in restored.operations] == [
            type(op).__name__ for op in block.operations
        ]

    @pytest.mark.parametrize("kernel,build_kwargs", _REAL_KERNELS)
    def test_round_trip_preserves_value_uuids(self, kernel, build_kwargs):
        """Every Value UUID present in the original survives the round-trip."""
        block = _build_real(kernel, build_kwargs)
        restored = load_json(dump_json(block))
        original_uuids = {v.uuid for v in block.input_values} | {
            v.uuid for v in block.output_values
        }
        restored_uuids = {v.uuid for v in restored.input_values} | {
            v.uuid for v in restored.output_values
        }
        assert restored_uuids == original_uuids

    def test_qft_round_trip_preserves_invoke_body(self):
        """QFT's nested invoke body survives byte-for-byte.

        This is a focused regression test: the encoder must emit the
        nested unitary block separately rather than via ``repr`` (the
        same fix that addressed Copilot's review on PR #389 for
        ControlledU's nested block).
        """
        block = _build_real(_qft_3, {})
        restored = load_msgpack(dump_msgpack(block))
        original_d = to_dict(block)
        restored_d = to_dict(restored)
        # Find the QFT InvokeOperation in operations.
        invoke = next(
            op
            for op in original_d["block"]["operations"]
            if op["$type"] == "InvokeOperation"
            and op["attrs"]["gate_type"] == CompositeGateType.QFT.name
        )
        invoke_r = next(
            op
            for op in restored_d["block"]["operations"]
            if op["$type"] == "InvokeOperation"
            and op["attrs"]["gate_type"] == CompositeGateType.QFT.name
        )
        assert invoke["definition"]["body"] == invoke_r["definition"]["body"]

    def test_symbolic_qft_round_trip_preserves_body(self):
        """Symbolic-width QFT serializes its ordinary qkernel body."""
        block = _build_real(_symbolic_qft, {})
        original_d = to_dict(block)
        restored_d = to_dict(load_msgpack(dump_msgpack(block)))
        invoke = next(
            op
            for op in original_d["block"]["operations"]
            if op["$type"] == "InvokeOperation"
            and op["attrs"]["gate_type"] == CompositeGateType.QFT.name
        )
        invoke_r = next(
            op
            for op in restored_d["block"]["operations"]
            if op["$type"] == "InvokeOperation"
            and op["attrs"]["gate_type"] == CompositeGateType.QFT.name
        )

        assert invoke["definition"]["body"] is not None
        assert invoke["definition"]["body_ref"] is None
        assert invoke_r["definition"]["body"] == invoke["definition"]["body"]


# ---------------------------------------------------------------------------
# Dict runtime parameter round-trip
# ---------------------------------------------------------------------------


@qmc.qkernel
def _dict_runtime_param_layer(
    n: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Ising-style layer whose coefficient dicts are runtime parameters."""
    q = qmc.qubit_array(n, name="q")
    q = qmc.h(q)
    for i in qmc.range(n):
        q[i] = qmc.rz(q[i], angle=2.0 * linear[i])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.rzz(q[i], q[i + 1], angle=2.0 * quad[(i, i + 1)])
    return qmc.measure(q)


def _analyzed_with_params(kernel: qmc.QKernel, parameters: list[str], **bindings):
    """Return an ANALYZED block for a kernel built with runtime parameters.

    Unlike :func:`_to_affine`, which uses ``kernel.block`` (auto-detected
    parameters), this builds the block with an explicit ``parameters``
    list so a ``Dict`` argument is created on the runtime-parameter path.

    Args:
        kernel (qmc.QKernel): The kernel to build.
        parameters (list[str]): Argument names kept as runtime parameters.
        **bindings (Any): Compile-time bindings (e.g. register sizes).

    Returns:
        Block: The kernel's block advanced to ``BlockKind.ANALYZED``.
    """
    block = kernel.build(parameters=parameters, **bindings)
    return _to_analyzed(InlinePass().run(block))


class TestDictRuntimeParameterRoundTrip:
    """A runtime-parameter Dict survives serialize -> deserialize intact.

    The runtime-parameter identity is carried on two channels, both of
    which serialize: the ``DictValue`` in ``Block.parameters`` (parameter
    marker plus absence of ``dict_runtime`` bound data) and the
    ``DictType``-typed ``RUNTIME_PARAMETER`` entry in
    ``Block.param_slots``. These tests pin that both round-trip. In
    particular ``dict_runtime`` must stay ``None`` across the round-trip;
    were it to decode as an empty-but-present ``DictRuntimeMetadata``,
    re-emit would misread the dict as compile-time-bound and drop the
    per-key parameters.
    """

    _PARAMS = ["quad", "linear"]

    def _assert_dict_param_signature(self, block) -> None:
        """Assert both dict arguments kept their runtime-parameter signature.

        Checks both channels that carry the runtime-parameter identity: the
        ``DictValue`` in ``Block.parameters`` (marker plus absent
        ``dict_runtime``) and the ``DictType``-typed ``RUNTIME_PARAMETER``
        entry in ``Block.param_slots``.

        Args:
            block (Block): The deserialized block to inspect.
        """
        from qamomile.circuit.ir.types.primitives import DictType

        slots = {s.name: s for s in block.param_slots}
        for name in self._PARAMS:
            dv = block.parameters.get(name)
            assert dv is not None, f"{name} missing from Block.parameters"
            assert dv.is_parameter(), f"{name} lost its parameter marker"
            assert dv.metadata.dict_runtime is None, (
                f"{name} gained dict_runtime metadata (would be misread as a "
                f"compile-time-bound dict on re-emit)"
            )
            assert name in slots, f"{name} missing from Block.param_slots"
            assert slots[name].kind is ParamKind.RUNTIME_PARAMETER
            assert isinstance(slots[name].type, DictType), (
                f"{name} param slot type is {type(slots[name].type).__name__}, "
                f"expected DictType"
            )

    def test_json_round_trip_preserves_runtime_dict(self):
        """JSON round-trip keeps content hash and the dict parameter signature."""
        block = _analyzed_with_params(_dict_runtime_param_layer, self._PARAMS, n=3)
        restored = load_json(dump_json(block))
        assert content_hash(restored) == content_hash(block)
        self._assert_dict_param_signature(restored)

    def test_msgpack_round_trip_preserves_runtime_dict(self):
        """msgpack round-trip mirrors the JSON one."""
        block = _analyzed_with_params(_dict_runtime_param_layer, self._PARAMS, n=3)
        restored = load_msgpack(dump_msgpack(block))
        assert content_hash(restored) == content_hash(block)
        self._assert_dict_param_signature(restored)

    def test_reemit_after_round_trip_matches_parameters(self):
        """Re-emitting a deserialized block yields identical per-key parameters."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        t = QiskitTranspiler()
        exe_ref = t.transpile(
            _dict_runtime_param_layer, bindings={"n": 3}, parameters=self._PARAMS
        )
        block = _analyzed_with_params(_dict_runtime_param_layer, self._PARAMS, n=3)
        restored = load_json(dump_json(block))
        lowered = t.validate_symbolic_shapes(t.classical_lowering(restored))
        exe = t.emit(t.plan(lowered), bindings={"n": 3}, parameters=self._PARAMS)
        assert sorted(exe.parameter_names) == sorted(exe_ref.parameter_names)
