"""Tests for emit_support helpers — ResourceAllocator merge allocation & LoopAnalyzer.

Section 1: ResourceAllocator if-merge allocation (Bug #6).
    ResourceAllocator._allocate_recursive() registers IfOperation merge
    slots after processing the branches, delegating to the shared
    map_merge_outputs() utility which:
    - Maps scalar QubitType merge outputs to the same physical qubit as
      the branch source value.
    - Copies composite element keys ``{source_uuid}_{i}`` →
      ``{output_uuid}_{i}`` for ArrayValue merge outputs.
    - Maps scalar BitType merge outputs to the same classical bit index.

Section 2: LoopAnalyzer BinOp dependency detection.
    LoopAnalyzer.should_unroll correctly identifies ForOperation loops
    containing BinOps that depend on the loop variable (directly or
    inside nested control-flow), and theta array-element access
    referencing the loop variable triggers unrolling.

Section 3: Integration tests for UInt BinOp (``//``, ``%``, ``**``) folding
    into loop bounds via the constant folding pass.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_support import (
    LoopAnalyzer,
    QubitAddress,
    ResourceAllocator,
    map_merge_outputs,
    resolve_qubit_key,
)

# ---------------------------------------------------------------------------
# Helpers — merge-slot tests
# ---------------------------------------------------------------------------


def _make_value(name: str, type_cls: type = UIntType) -> Value:
    """Create a simple Value with the given name and type."""
    return Value(type=type_cls(), name=name)


def _make_const_value(
    name: str, const: int | float, type_cls: type = UIntType
) -> Value:
    """Create a constant Value."""
    return Value(type=type_cls(), name=name).with_const(const)


def _make_array_value(
    name: str,
    shape_vals: tuple[Value, ...] = (),
    type_cls: type = QubitType,
) -> ArrayValue:
    """Create an ArrayValue with the given shape dimension Values."""
    return ArrayValue(type=type_cls(), name=name, shape=shape_vals)


def _make_array_element(
    parent: ArrayValue,
    index: int,
    name: str,
    type_cls: type = QubitType,
) -> Value:
    """Create a constant-index element Value for an ArrayValue.

    Args:
        parent (ArrayValue): Parent vector or slice view.
        index (int): Constant element index.
        name (str): Display name assigned to the element.
        type_cls (type): Element type class. Defaults to QubitType.

    Returns:
        Value: Element value referencing ``parent[index]``.
    """
    idx_value = _make_const_value(f"{name}_idx", index)
    return Value(
        type=type_cls(),
        name=name,
        parent_array=parent,
        element_indices=(idx_value,),
    )


def _make_if_with_merge(
    cond: Value,
    true_value: Value,
    false_value: Value,
    result: Value,
) -> IfOperation:
    """Build a minimal IfOperation carrying a single branch merge.

    Args:
        cond (Value): Condition operand (Bit).
        true_value (Value): Value selected when the condition is true.
        false_value (Value): Value selected when the condition is false.
        result (Value): Merged output value.

    Returns:
        IfOperation: If-else with empty branches and one merge slot.
    """
    if_op = IfOperation(operands=[cond])
    if_op.add_merge(true_value, false_value, result)
    return if_op


# ---------------------------------------------------------------------------
# Helpers — LoopAnalyzer tests
# ---------------------------------------------------------------------------


def _qubit(name: str = "q") -> Value:
    """Create a qubit Value."""
    return Value(type=QubitType(), name=name)


def _float_val(name: str = "theta", *, const: float | None = None) -> Value:
    """Create a float Value, optionally constant."""
    value = Value(type=FloatType(), name=name)
    return value.with_const(const) if const is not None else value


def _uint_val(name: str, *, const: int | None = None) -> Value:
    """Create a UInt Value, optionally constant."""
    value = Value(type=UIntType(), name=name)
    return value.with_const(const) if const is not None else value


def _make_binop(
    lhs: Value,
    rhs: Value,
    kind: BinOpKind,
    result_name: str = "binop_result",
) -> tuple[BinOp, Value]:
    """Create a BinOp and its result Value.

    The result type is inferred from the left-hand operand.
    """
    result = Value(type=lhs.type, name=result_name)
    op = BinOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    return op, result


def _make_gate(
    gate_type: GateOperationType,
    qubits: list[Value],
    theta: float | Value | None = None,
) -> GateOperation:
    """Create a GateOperation with given qubits and theta."""
    results = [q.next_version() for q in qubits]
    if theta is not None:
        # Wrap float to Value for operands-based theta storage
        if isinstance(theta, (int, float)):
            from qamomile.circuit.ir.types.primitives import FloatType

            theta = Value(type=FloatType(), name="theta").with_const(theta)
        return GateOperation.rotation(
            gate_type=gate_type, qubits=qubits, theta=theta, results=results
        )
    return GateOperation.fixed(gate_type=gate_type, qubits=qubits, results=results)


# ===========================================================================
# Bug #6: Merge output allocation
# ===========================================================================


class TestIfMergeAllocation:
    """Tests that ResourceAllocator processes IfOperation merge slots."""

    def test_resolve_qubit_key_uses_root_for_nested_slice_element(self) -> None:
        """Constant nested slice elements resolve to the root qubit address."""
        root_size = _make_const_value("q_dim0", 5)
        q_array = _make_array_value("q", shape_vals=(root_size,))
        first_view = ArrayValue(
            type=QubitType(),
            name="q_view",
            shape=(_make_const_value("q_view_dim0", 2),),
            slice_of=q_array,
            slice_start=_make_const_value("start_1", 1),
            slice_step=_make_const_value("step_2", 2),
        )
        nested_view = ArrayValue(
            type=QubitType(),
            name="q_nested_view",
            shape=(_make_const_value("q_nested_view_dim0", 2),),
            slice_of=first_view,
            slice_start=_make_const_value("nested_start_0", 0),
            slice_step=_make_const_value("nested_step_1", 1),
        )
        element = _make_array_element(nested_view, 1, "q_nested_view[1]")

        key, is_array = resolve_qubit_key(element)

        assert is_array
        assert key == QubitAddress(q_array.uuid, 3)

    def test_resolve_qubit_key_defers_symbolic_slice_bounds(self) -> None:
        """Symbolic slice bounds defer instead of using the slice UUID."""
        root_size = _make_const_value("q_dim0", 4)
        q_array = _make_array_value("q", shape_vals=(root_size,))
        view = ArrayValue(
            type=QubitType(),
            name="q_view",
            shape=(_make_const_value("q_view_dim0", 2),),
            slice_of=q_array,
            slice_start=_make_value("offset", UIntType),
            slice_step=_make_const_value("step_1", 1),
        )
        element = _make_array_element(view, 0, "q_view[0]")

        key, is_array = resolve_qubit_key(element)

        assert is_array
        assert key is None

    def test_scalar_merges_root_and_slice_element_aliases(self) -> None:
        """Scalar qubit merge should recognize root and slice element aliases."""
        q_array = _make_array_value("q", shape_vals=(_make_const_value("q_dim0", 3),))
        view = ArrayValue(
            type=QubitType(),
            name="q_view",
            shape=(_make_const_value("q_view_dim0", 2),),
            slice_of=q_array,
            slice_start=_make_const_value("start_1", 1),
            slice_step=_make_const_value("step_1", 1),
        )
        cond = _make_value("cond", BitType)
        root_element = _make_array_element(q_array, 1, "q[1]")
        view_element = _make_array_element(view, 0, "q_view[0]")
        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(cond, root_element, view_element, merge_output)
        qubit_map = {QubitAddress(q_array.uuid, i): i for i in range(3)}

        map_merge_outputs(if_op, qubit_map, {})

        assert qubit_map[QubitAddress(merge_output.uuid)] == 1
        assert QubitAddress(view.uuid, 0) not in qubit_map

    def test_scalar_merge_defers_unresolved_symbolic_slice_element(self) -> None:
        """Unresolved symbolic slice elements must not create slice-address aliases."""
        q_array = _make_array_value("q", shape_vals=(_make_const_value("q_dim0", 3),))
        view = ArrayValue(
            type=QubitType(),
            name="q_view",
            shape=(_make_const_value("q_view_dim0", 2),),
            slice_of=q_array,
            slice_start=_make_value("offset", UIntType),
            slice_step=_make_const_value("step_1", 1),
        )
        cond = _make_value("cond", BitType)
        view_element = _make_array_element(view, 0, "q_view[0]")
        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(cond, view_element, view_element, merge_output)
        qubit_map = {QubitAddress(q_array.uuid, i): i for i in range(3)}

        map_merge_outputs(if_op, qubit_map, {})

        assert QubitAddress(merge_output.uuid) not in qubit_map
        assert QubitAddress(view.uuid, 0) not in qubit_map

    def test_merge_output_qubit_is_allocated(self) -> None:
        """Merge output for a qubit type should be registered in qubit_map."""
        # QInit → q
        q_init = _make_value("q", QubitType)
        q_init_out = q_init.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_init_out])

        # H → q_after_h
        q_after_h = q_init_out.next_version()
        h_gate = GateOperation(
            operands=[q_init_out],
            results=[q_after_h],
            gate_type=GateOperationType.H,
        )

        # Measure → bit
        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_after_h], results=[bit])

        # If-else with merge
        q_true = q_after_h.next_version()
        x_gate = GateOperation(
            operands=[q_after_h],
            results=[q_true],
            gate_type=GateOperationType.X,
        )
        q_false = q_after_h  # identity in false branch

        merge_output = _make_value("q_merge", QubitType)
        if_op = IfOperation(
            operands=[bit],
            true_operations=[x_gate],
            false_operations=[],
        )
        if_op.add_merge(q_true, q_false, merge_output)

        # Measure merge output
        bit2 = _make_value("bit2", BitType)
        measure2 = MeasureOperation(operands=[merge_output], results=[bit2])

        operations = [qinit_op, h_gate, measure_op, if_op, measure2]

        allocator = ResourceAllocator()
        qubit_map, clbit_map = allocator.allocate(operations, bindings={})

        # merge_output should be in qubit_map, mapped to same physical qubit
        assert QubitAddress(merge_output.uuid) in qubit_map
        assert (
            qubit_map[QubitAddress(merge_output.uuid)]
            == qubit_map[QubitAddress(q_init_out.uuid)]
        )

    def test_merge_output_bit_is_allocated(self) -> None:
        """Merge output for a bit type should be registered in clbit_map."""
        # Setup: condition bit
        cond = _make_value("cond", BitType)

        # Create a qubit so we have something to measure
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        # Measure to get condition
        measure_op = MeasureOperation(operands=[q_out], results=[cond])

        # Bit merge: both branches produce the same bit
        true_bit = _make_value("true_bit", BitType)
        false_bit = _make_value("false_bit", BitType)
        true_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[true_bit]
        )
        false_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[false_bit]
        )

        merge_bit = _make_value("merge_bit", BitType)
        if_op = IfOperation(
            operands=[cond],
            true_operations=[true_measure],
            false_operations=[false_measure],
        )
        if_op.add_merge(true_bit, false_bit, merge_bit)

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        _, clbit_map = allocator.allocate(operations, bindings={})

        # merge_bit should be in clbit_map
        assert QubitAddress(merge_bit.uuid) in clbit_map

    @pytest.mark.parametrize("array_size", [1, 2, 4])
    def test_merge_output_array_composite_keys_are_allocated(
        self, array_size: int
    ) -> None:
        """Merge output for an ArrayValue should copy composite element keys."""
        size_val = _make_const_value("size", array_size)
        q_array = _make_array_value("q", shape_vals=(size_val,))
        qinit_op = QInitOperation(operands=[], results=[q_array])

        # Measure q[0] → bit (condition)
        q0_idx = _make_const_value("idx_0", 0)
        q0_elem = Value(
            type=QubitType(),
            name="q[0]",
            parent_array=q_array,
            element_indices=(q0_idx,),
        )
        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q0_elem], results=[bit])

        # If-else with merge on the whole array
        merge_array = _make_array_value("q_merge", shape_vals=(size_val,))
        if_op = _make_if_with_merge(bit, q_array, q_array, merge_array)

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        # All composite keys for the merge output array should exist
        for i in range(array_size):
            merge_addr = QubitAddress(merge_array.uuid, i)
            src_addr = QubitAddress(q_array.uuid, i)
            assert merge_addr in qubit_map, f"merge array element {i} not allocated"
            assert qubit_map[merge_addr] == qubit_map[src_addr]

    def test_identity_merge_maps_to_same_qubit(self) -> None:
        """Merge with true_val == false_val (identity) still maps correctly."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_out], results=[bit])

        # Identity merge: both branches refer to the same qubit
        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(bit, q_out, q_out, merge_output)

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        assert QubitAddress(merge_output.uuid) in qubit_map
        assert (
            qubit_map[QubitAddress(merge_output.uuid)]
            == qubit_map[QubitAddress(q_out.uuid)]
        )

    def test_merge_output_already_registered_is_skipped(self) -> None:
        """Merge output UUID already in qubit_map is not overwritten."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_out], results=[bit])

        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(bit, q_out, q_out, merge_output)

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, clbit_map = allocator.allocate(operations, bindings={})

        # Pre-register the merge output with a sentinel value
        sentinel_idx = 999
        qubit_map[QubitAddress(merge_output.uuid)] = sentinel_idx

        # Re-running allocation should not overwrite it
        map_merge_outputs(if_op, qubit_map, clbit_map)
        assert qubit_map[QubitAddress(merge_output.uuid)] == sentinel_idx

    def test_multiple_merges_all_allocated(self) -> None:
        """Multiple merge slots in a single IfOperation are all allocated."""
        q0 = _make_value("q0", QubitType)
        q0_out = q0.next_version()
        q1 = _make_value("q1", QubitType)
        q1_out = q1.next_version()
        qinit0 = QInitOperation(operands=[], results=[q0_out])
        qinit1 = QInitOperation(operands=[], results=[q1_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q0_out], results=[bit])

        # Two merge slots
        merge_out0 = _make_value("merge0", QubitType)
        merge_out1 = _make_value("merge1", QubitType)

        if_op = IfOperation(
            operands=[bit],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(q0_out, q0_out, merge_out0)
        if_op.add_merge(q1_out, q1_out, merge_out1)

        operations = [qinit0, qinit1, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        assert QubitAddress(merge_out0.uuid) in qubit_map
        assert QubitAddress(merge_out1.uuid) in qubit_map
        assert (
            qubit_map[QubitAddress(merge_out0.uuid)]
            == qubit_map[QubitAddress(q0_out.uuid)]
        )
        assert (
            qubit_map[QubitAddress(merge_out1.uuid)]
            == qubit_map[QubitAddress(q1_out.uuid)]
        )

    def test_merge_bit_consolidates_both_branches(self) -> None:
        """Both branches' clbits must be consolidated to the same physical clbit.

        Under Qiskit's ``if_test``, only one branch executes, so both
        branches must measure into the same physical clbit. Otherwise
        the merge output always reads the true branch's result.
        """
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        cond = _make_value("cond", BitType)
        measure_cond = MeasureOperation(operands=[q_out], results=[cond])

        true_bit = _make_value("true_bit", BitType)
        false_bit = _make_value("false_bit", BitType)
        true_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[true_bit]
        )
        false_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[false_bit]
        )

        merge_bit = _make_value("merge_bit", BitType)
        if_op = IfOperation(
            operands=[cond],
            true_operations=[true_measure],
            false_operations=[false_measure],
        )
        if_op.add_merge(true_bit, false_bit, merge_bit)

        operations = [qinit_op, measure_cond, if_op]
        allocator = ResourceAllocator()
        _, clbit_map = allocator.allocate(operations, bindings={})

        # All three — true_bit, false_bit, merge_bit — must share the same physical clbit
        assert (
            clbit_map[QubitAddress(true_bit.uuid)]
            == clbit_map[QubitAddress(false_bit.uuid)]
        )
        assert (
            clbit_map[QubitAddress(merge_bit.uuid)]
            == clbit_map[QubitAddress(true_bit.uuid)]
        )

    def test_clbit_compaction_preserves_seeded_indices(self) -> None:
        """Remove new merge holes without renumbering caller-owned clbits."""
        cond = _make_value("cond", BitType)
        true_bit = _make_value("true_bit", BitType)
        false_bit = _make_value("false_bit", BitType)
        merged = _make_value("merged", BitType)
        tail = _make_value("tail", BitType)
        qubit = _make_value("q", QubitType)
        if_op = IfOperation(
            operands=[cond],
            true_operations=[MeasureOperation(operands=[qubit], results=[true_bit])],
            false_operations=[MeasureOperation(operands=[qubit], results=[false_bit])],
        )
        if_op.add_merge(true_bit, false_bit, merged)

        seeded_address = QubitAddress(cond.uuid)
        allocator = ResourceAllocator()
        _, clbit_map = allocator.allocate(
            [
                MeasureOperation(operands=[qubit], results=[cond]),
                if_op,
                MeasureOperation(operands=[qubit], results=[tail]),
            ],
            initial_clbit_map={seeded_address: 7},
        )

        assert clbit_map[seeded_address] == 7
        assert clbit_map[QubitAddress(true_bit.uuid)] == 8
        assert clbit_map[QubitAddress(false_bit.uuid)] == 8
        assert clbit_map[QubitAddress(merged.uuid)] == 8
        assert clbit_map[QubitAddress(tail.uuid)] == 9
        assert set(clbit_map.values()) == {7, 8, 9}

    @pytest.mark.parametrize("array_size", [1, 2, 4])
    def test_merge_bit_array_consolidates_both_branches(self, array_size: int) -> None:
        """BitType ArrayValue merge must consolidate per-element clbits across branches."""
        size_val = _make_const_value("size", array_size)

        q_array = _make_array_value("q", shape_vals=(size_val,))
        _ = QInitOperation(operands=[], results=[q_array])

        # Measure q[0] → condition
        q0_idx = _make_const_value("idx_0", 0)
        q0_elem = Value(
            type=QubitType(),
            name="q[0]",
            parent_array=q_array,
            element_indices=(q0_idx,),
        )
        cond = _make_value("cond", BitType)
        _ = MeasureOperation(operands=[q0_elem], results=[cond])

        # True branch: measure into true_bits array
        true_bits = _make_array_value(
            "true_bits", shape_vals=(size_val,), type_cls=BitType
        )
        false_bits = _make_array_value(
            "false_bits", shape_vals=(size_val,), type_cls=BitType
        )

        # Simulate allocation of individual array element clbits
        # (MeasureVectorOperation would do this, but for unit test we pre-fill)

        merge_bits = _make_array_value(
            "merge_bits", shape_vals=(size_val,), type_cls=BitType
        )
        if_op = _make_if_with_merge(cond, true_bits, false_bits, merge_bits)

        # Pre-fill clbit_map with element keys for both arrays
        clbit_map = {QubitAddress(cond.uuid): 0}
        for i in range(array_size):
            clbit_map[QubitAddress(true_bits.uuid, i)] = i + 1
            clbit_map[QubitAddress(false_bits.uuid, i)] = array_size + i + 1

        map_merge_outputs(if_op, {}, clbit_map)

        # Merge output elements should exist AND both branches consolidated
        for i in range(array_size):
            merge_addr = QubitAddress(merge_bits.uuid, i)
            true_addr = QubitAddress(true_bits.uuid, i)
            false_addr = QubitAddress(false_bits.uuid, i)
            assert merge_addr in clbit_map, f"merge element {i} not allocated"
            assert clbit_map[merge_addr] == clbit_map[true_addr]
            assert clbit_map[false_addr] == clbit_map[true_addr], (
                f"element {i}: false branch clbit not consolidated"
            )

    # --- Quantum merge validation tests ---

    def test_quantum_merge_scalar_different_resources_raises_emit_error(self) -> None:
        """Scalar qubit merge with different physical resources must raise EmitError."""
        q0 = _make_value("q0", QubitType)
        q0_out = q0.next_version()
        q1 = _make_value("q1", QubitType)
        q1_out = q1.next_version()
        cond = _make_value("cond", BitType)

        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(cond, q0_out, q1_out, merge_output)

        qubit_map = {QubitAddress(q0_out.uuid): 0, QubitAddress(q1_out.uuid): 1}
        with pytest.raises(EmitError, match="merge requires identical"):
            map_merge_outputs(if_op, qubit_map, {})

    def test_quantum_merge_array_different_resources_raises_emit_error(self) -> None:
        """Array qubit merge with different physical resources must raise EmitError."""
        size_val = _make_const_value("size", 2)
        arr_a = _make_array_value("arr_a", shape_vals=(size_val,))
        arr_b = _make_array_value("arr_b", shape_vals=(size_val,))
        cond = _make_value("cond", BitType)

        merge_output = _make_array_value("merge_arr", shape_vals=(size_val,))
        if_op = _make_if_with_merge(cond, arr_a, arr_b, merge_output)

        qubit_map = {
            QubitAddress(arr_a.uuid, 0): 0,
            QubitAddress(arr_a.uuid, 1): 1,
            QubitAddress(arr_b.uuid, 0): 2,
            QubitAddress(arr_b.uuid, 1): 3,
        }
        with pytest.raises(EmitError, match="merge requires identical"):
            map_merge_outputs(if_op, qubit_map, {})

    def test_quantum_merge_scalar_unresolved_branch_raises_emit_error(self) -> None:
        """Scalar qubit merge with one unresolved branch must raise EmitError."""
        q0 = _make_value("q0", QubitType)
        q0_out = q0.next_version()
        q1 = _make_value("q1", QubitType)
        q1_out = q1.next_version()
        cond = _make_value("cond", BitType)

        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(cond, q0_out, q1_out, merge_output)

        # Only q0_out is in qubit_map; q1_out is unresolved
        qubit_map = {QubitAddress(q0_out.uuid): 0}
        with pytest.raises(EmitError, match="merge requires identical"):
            map_merge_outputs(if_op, qubit_map, {})

    def test_quantum_merge_array_unresolved_suffix_raises_emit_error(self) -> None:
        """Array qubit merge with missing suffix in one branch must raise EmitError."""
        size_val = _make_const_value("size", 2)
        arr_a = _make_array_value("arr_a", shape_vals=(size_val,))
        arr_b = _make_array_value("arr_b", shape_vals=(size_val,))
        cond = _make_value("cond", BitType)

        merge_output = _make_array_value("merge_arr", shape_vals=(size_val,))
        if_op = _make_if_with_merge(cond, arr_a, arr_b, merge_output)

        # arr_a has both elements, arr_b only has element 0
        qubit_map = {
            QubitAddress(arr_a.uuid, 0): 0,
            QubitAddress(arr_a.uuid, 1): 1,
            QubitAddress(arr_b.uuid, 0): 0,
        }
        with pytest.raises(EmitError, match="merge requires identical"):
            map_merge_outputs(if_op, qubit_map, {})

    def test_quantum_merge_scalar_identity_still_allowed(self) -> None:
        """Identity scalar merge (same physical resource) must succeed."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        cond = _make_value("cond", BitType)

        merge_output = _make_value("q_merge", QubitType)
        if_op = _make_if_with_merge(cond, q_out, q_out, merge_output)

        qubit_map = {QubitAddress(q_out.uuid): 0}
        map_merge_outputs(if_op, qubit_map, {})
        assert QubitAddress(merge_output.uuid) in qubit_map
        assert qubit_map[QubitAddress(merge_output.uuid)] == 0

    def test_quantum_merge_array_identity_still_allowed(self) -> None:
        """Identity array merge (same physical resources) must succeed."""
        size_val = _make_const_value("size", 3)
        arr = _make_array_value("arr", shape_vals=(size_val,))
        cond = _make_value("cond", BitType)

        merge_output = _make_array_value("merge_arr", shape_vals=(size_val,))
        if_op = _make_if_with_merge(cond, arr, arr, merge_output)

        qubit_map = {
            QubitAddress(arr.uuid, 0): 0,
            QubitAddress(arr.uuid, 1): 1,
            QubitAddress(arr.uuid, 2): 2,
        }
        map_merge_outputs(if_op, qubit_map, {})
        for i in range(3):
            assert qubit_map[QubitAddress(merge_output.uuid, i)] == i


# ===========================================================================
# LoopAnalyzer._has_loop_var_binop
# ===========================================================================


class TestLoopAnalyzerBinOp:
    """Tests for LoopAnalyzer detecting BinOps dependent on loop variables."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_direct_binop_dependency_triggers_unroll(self) -> None:
        """A BinOp using the loop variable directly should trigger unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.5)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=0.1)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=5)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_no_binop_no_unroll(self) -> None:
        """A loop with no BinOps and no array access should not unroll."""
        loop_var_val = _uint_val("i")
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_not_using_loop_var_no_unroll(self) -> None:
        """A BinOp not referencing the loop variable should not trigger unrolling."""
        loop_var_val = _uint_val("i")
        a = _float_val("a", const=1.0)
        b = _float_val("b", const=2.0)
        binop, _ = _make_binop(a, b, BinOpKind.ADD)

        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_in_nested_for_triggers_unroll(self) -> None:
        """A BinOp inside a nested ForOperation referencing the outer loop var."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.1)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        inner_start = _uint_val("is", const=0)
        inner_stop = _uint_val("ie", const=2)
        inner_step = _uint_val("ist", const=1)

        inner_for = ForOperation(
            operands=[inner_start, inner_stop, inner_step],
            results=[],
            loop_var="j",
            loop_var_value=_uint_val("j"),
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[inner_for],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_true_branch_triggers_unroll(self) -> None:
        """A BinOp in the true branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=2.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.ADD)

        cond = Value(type=FloatType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[binop],
            false_operations=[],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_if_false_branch_triggers_unroll(self) -> None:
        """A BinOp in the false branch of an IfOperation triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=3.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.SUB)

        cond = Value(type=FloatType(), name="cond")
        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[],
            false_operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[if_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_while_triggers_unroll(self) -> None:
        """A BinOp inside a WhileOperation referencing loop var triggers unrolling."""
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=1.0)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.MUL)

        cond = Value(type=FloatType(), name="cond")
        while_op = WhileOperation(
            operands=[cond],
            results=[],
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[while_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_binop_in_for_items_triggers_unroll(self) -> None:
        """A BinOp inside ForItemsOperation referencing outer loop var triggers unrolling.

        ForItemsOperation is unrolled for its own iteration (dict items),
        but BinOps inside it may still reference the *outer* ForOperation's
        loop variable.  The outer loop must be unrolled so the BinOp gets
        concrete values.
        """
        loop_var_val = _uint_val("i")
        const_val = _float_val("c", const=0.5)
        binop, _ = _make_binop(loop_var_val, const_val, BinOpKind.ADD)

        dict_val = Value(type=FloatType(), name="d")
        for_items_op = ForItemsOperation(
            operands=[dict_val],
            results=[],
            key_vars=["k"],
            value_var="v",
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_var_val,
            operations=[for_items_op],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True


# ===========================================================================
# LoopAnalyzer — theta array element access
# ===========================================================================


class TestLoopAnalyzerThetaArrayAccess:
    """Tests that theta array element access with loop var triggers unrolling."""

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def test_theta_array_element_with_loop_var_triggers_unroll(self) -> None:
        """Gate with theta = gammas[i] where i is the loop var should unroll."""
        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        loop_idx = Value(type=UIntType(), name="i")
        theta_elem = Value(
            type=FloatType(),
            name="gammas_i",
            parent_array=gammas_array,
            element_indices=(loop_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_idx,
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_theta_array_element_without_loop_var_no_unroll(self) -> None:
        """Gate with theta = gammas[0] (constant index) should not unroll."""
        loop_idx = Value(type=UIntType(), name="i")
        gammas_array = ArrayValue(type=FloatType(), name="gammas")
        const_idx = _uint_val("idx", const=0)
        theta_elem = Value(
            type=FloatType(),
            name="gammas_0",
            parent_array=gammas_array,
            element_indices=(const_idx,),
        )

        q = _qubit()
        gate = _make_gate(GateOperationType.RZ, [q], theta=theta_elem)

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            loop_var_value=loop_idx,
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_missing_loop_identity_fails_closed_to_unroll(self) -> None:
        """Legacy IR without an index Value cannot select native emission."""
        q = _qubit()
        for_op = ForOperation(
            operands=[
                _uint_val("start", const=0),
                _uint_val("stop", const=1),
                _uint_val("step", const=1),
            ],
            results=[],
            loop_var="i",
            operations=[_make_gate(GateOperationType.H, [q])],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True


# ===========================================================================
# LoopAnalyzer — measurement / reset / projection array element access (M2)
# ===========================================================================


class TestLoopAnalyzerMeasurementArrayAccess:
    """Loop-var element access by measurement/reset/projection forces unroll.

    Regression for the silent-measurement-drop bug: ``MeasureOperation``,
    ``MeasureVectorOperation``, ``ProjectOperation`` and ``ResetOperation``
    are not ``GateOperation`` subclasses, so the old type-enumerated
    ``_has_array_element_access`` skipped them. A native backend loop whose
    only loop-var element access was ``measure(q[i])`` then took the native
    path and dropped the measurement at emit. The scan is now generic over
    ``all_input_values()``.
    """

    def setup_method(self) -> None:
        self.analyzer = LoopAnalyzer()

    def _loop_with(self, op, loop_idx) -> ForOperation:
        """Wrap a single op in a ``for i in range(0, 3)`` loop."""
        return ForOperation(
            operands=[
                _uint_val("start", const=0),
                _uint_val("stop", const=3),
                _uint_val("step", const=1),
            ],
            results=[],
            loop_var="i",
            loop_var_value=loop_idx,
            operations=[op],
        )

    def _element_qubit(self, loop_idx) -> Value:
        """Build ``q[i]`` — a qubit element indexed by the loop variable."""
        q_array = ArrayValue(type=QubitType(), name="q")
        return Value(
            type=QubitType(),
            name="q_i",
            parent_array=q_array,
            element_indices=(loop_idx,),
        )

    def test_measure_of_loop_var_element_triggers_unroll(self) -> None:
        """``measure(q[i])`` as the only body op must force unrolling."""
        loop_idx = Value(type=UIntType(), name="i")
        q_elem = self._element_qubit(loop_idx)
        measure = MeasureOperation(
            operands=[q_elem],
            results=[Value(type=BitType(), name="b")],
        )
        assert (
            self.analyzer.should_unroll(self._loop_with(measure, loop_idx), {}) is True
        )

    def test_reset_of_loop_var_element_triggers_unroll(self) -> None:
        """``reset(q[i])`` must force unrolling."""
        loop_idx = Value(type=UIntType(), name="i")
        q_elem = self._element_qubit(loop_idx)
        reset = ResetOperation(
            operands=[q_elem],
            results=[Value(type=QubitType(), name="q_i2")],
        )
        assert self.analyzer.should_unroll(self._loop_with(reset, loop_idx), {}) is True

    def test_project_of_loop_var_element_triggers_unroll(self) -> None:
        """``project_z(q[i])`` must force unrolling."""
        loop_idx = Value(type=UIntType(), name="i")
        q_elem = self._element_qubit(loop_idx)
        project = ProjectOperation(
            operands=[q_elem],
            results=[
                Value(type=QubitType(), name="q_i2"),
                Value(type=BitType(), name="b"),
            ],
            axis="z",
        )
        assert (
            self.analyzer.should_unroll(self._loop_with(project, loop_idx), {}) is True
        )

    def test_measure_vector_of_loop_var_element_triggers_unroll(self) -> None:
        """A vector measurement whose operand is ``q[i]`` must unroll.

        This builds an ``element_indices``-based operand (``q[i]``), not a
        ``slice_of``-chain view — the generic ``all_input_values`` scan flags
        it because the operand carries ``parent_array`` + ``element_indices``
        depending on the loop variable.
        """
        loop_idx = Value(type=UIntType(), name="i")
        q_elem = self._element_qubit(loop_idx)
        mvec = MeasureVectorOperation(
            operands=[q_elem],
            results=[ArrayValue(type=BitType(), name="bits")],
        )
        assert self.analyzer.should_unroll(self._loop_with(mvec, loop_idx), {}) is True

    def test_measure_of_constant_index_does_not_unroll(self) -> None:
        """``measure(q[0])`` (constant index) does not need unrolling."""
        loop_idx = Value(type=UIntType(), name="i")
        q_array = ArrayValue(type=QubitType(), name="q")
        q_const = Value(
            type=QubitType(),
            name="q_0",
            parent_array=q_array,
            element_indices=(_uint_val("idx", const=0),),
        )
        measure = MeasureOperation(
            operands=[q_const],
            results=[Value(type=BitType(), name="b")],
        )
        assert (
            self.analyzer.should_unroll(self._loop_with(measure, loop_idx), {}) is False
        )


class TestLoopAnalyzerGenericValueDependency:
    """Test recursive loop-index discovery across generic operation inputs."""

    def setup_method(self) -> None:
        """Create a fresh analyzer for each generic dependency test."""
        self.analyzer = LoopAnalyzer()

    def _loop_with(self, operation: Operation, loop_idx: Value) -> ForOperation:
        """Wrap one operation in a valid identity-bearing loop.

        Args:
            operation (Operation): Body operation whose structural inputs are
                analyzed.
            loop_idx (Value): UUID-bearing loop-index value.

        Returns:
            ForOperation: Minimal two-trip loop containing ``operation``.
        """
        return ForOperation(
            operands=[
                _uint_val("start", const=0),
                _uint_val("stop", const=2),
                _uint_val("step", const=1),
            ],
            loop_var="i",
            loop_var_value=loop_idx,
            operations=[operation],
        )

    def test_loop_index_comparison_triggers_unroll(self) -> None:
        """A comparison input participates in generic dependency analysis."""
        loop_idx = _uint_val("i")
        comparison = CompOp(
            operands=[loop_idx, _uint_val("zero", const=0)],
            results=[Value(type=BitType(), name="condition")],
            kind=CompOpKind.EQ,
        )

        assert self.analyzer.should_unroll(self._loop_with(comparison, loop_idx), {})

    def test_array_slice_bound_with_loop_var_triggers_unroll(self) -> None:
        """A vector view whose start is ``i`` requires emit-time unrolling."""
        loop_idx = _uint_val("i")
        one = _uint_val("one", const=1)
        root = ArrayValue(
            type=QubitType(),
            name="root",
            shape=(_uint_val("size", const=2),),
        )
        view = ArrayValue(
            type=QubitType(),
            name="view",
            shape=(one,),
            slice_of=root,
            slice_start=loop_idx,
            slice_step=one,
        )
        measure = MeasureVectorOperation(
            operands=[view],
            results=[ArrayValue(type=BitType(), name="bits", shape=(one,))],
        )

        assert self.analyzer.should_unroll(self._loop_with(measure, loop_idx), {})

    def test_symbolic_control_index_with_loop_var_triggers_unroll(self) -> None:
        """Subclass-specific ``control_indices`` participate in analysis."""
        loop_idx = _uint_val("i")
        pool = ArrayValue(
            type=QubitType(),
            name="pool",
            shape=(_uint_val("pool_size", const=2),),
        )
        target = _qubit("target")
        controlled = SymbolicControlledU(
            operands=[pool, target],
            results=[pool.next_version(), target.next_version()],
            num_controls=_uint_val("num_controls", const=1),
            control_indices=(loop_idx,),
            block=Block(name="unitary"),
        )

        assert self.analyzer.should_unroll(self._loop_with(controlled, loop_idx), {})


# ===========================================================================
# Integration tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ===========================================================================

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from qamomile.qiskit.transpiler import QiskitTranspiler  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level @qkernel definitions (required for inspect.getsource)
# ---------------------------------------------------------------------------


@qmc.qkernel
def broken_merge_bit_array_example() -> qmc.Vector[qmc.Bit]:
    """Regression: different qubit arrays in true/false branches must not silently merge.

    Before the fix, this returned ``[((0, 0, 0, 0, 1), 200)]`` because the merge
    always picked the true branch's physical resources regardless of the runtime
    condition. the kernel transpiles and executes successfully.
    """
    c = qmc.qubit_array(1, "c")
    true_q = qmc.qubit_array(2, "true")
    false_q = qmc.qubit_array(2, "false")

    m = qmc.measure(c[0])
    false_q[1] = qmc.x(false_q[1])

    out = None
    if m:
        out = qmc.measure(true_q)
    else:
        out = qmc.measure(false_q)

    return out


@qmc.qkernel
def broken_merge_bit_scalar_example() -> qmc.Bit:
    """Regression: scalar bit merge with different source qubits.

    Before the fix, the merge always read the true branch's clbit regardless of
    the runtime condition.  ``b`` is 0, so the else branch (``measure(q[2])``,
    which is ``|1>``) must be chosen → result ``(1, 200)``.
    Before the fix this returned ``(0, 200)``.
    """
    q = qmc.qubit_array(3, "q")

    b = qmc.measure(q[0])  # Always zero
    q[2] = qmc.x(q[2])  # q[2] is one

    if b:
        b = qmc.measure(q[1])  # 0
    else:
        b = qmc.measure(q[2])  # 1

    return b


@qmc.qkernel
def frontend_target_vars_leak_example() -> qmc.Bit:
    """Regression: Store-only reassignment of ``b`` leaked from merge.

    Before the fix, ``b = qmc.bit(False)`` was never updated by the merge
    (Store-only reassignment excluded from ``target_vars``), so the second
    ``if b`` always saw the stale ``False`` value and skipped the X gate.
    Before the fix this returned ``(0, 200)``.
    """
    q = qmc.qubit_array(4, "q")
    b = qmc.bit(False)

    c = qmc.measure(q[0])  # 0

    if c:
        b = qmc.measure(q[1])
    else:
        q[2] = qmc.x(q[2])  # |1>
        b = qmc.measure(q[2])  # 1

    if b:
        q[3] = qmc.x(q[3])  # |1>

    return qmc.measure(q[3])  # 1


@qmc.qkernel
def binop_floordiv_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n // 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n // 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_pow_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n ** 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n**2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_mod_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n % 3 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n % 3
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_rmod_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first 7 % n qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = 7 % n  # exercises UInt.__rmod__
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def array_element_loop_bound_circuit(
    bounds: qmc.Vector[qmc.UInt], theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) using ``bounds[0]`` as an emit-time loop bound."""
    q = qmc.qubit_array(4, "q")
    for i in qmc.range(bounds[0]):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def apply_rx_helper(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply RX(theta) through a helper kernel."""
    return qmc.rx(q, angle=theta)


@qmc.qkernel
def vector_view_element_helper_circuit(
    slopes_p: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Pass ``view[0]`` from ``slopes_p[1:3]`` into a helper kernel."""
    q = qmc.qubit_array(1, "q")
    view = slopes_p[1:3]
    q[0] = apply_rx_helper(q[0], view[0])
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _transpile_and_get_circuit(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
) -> tuple[Any, "QuantumCircuit"]:
    """Transpile a kernel and return ``(executable, circuit)``."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kernel, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    return exe, qc


def _extract_rx_angles(qc: "QuantumCircuit") -> list[float]:
    """Extract all RX gate angles from a Qiskit circuit."""
    return [
        float(inst.operation.params[0])
        for inst in qc.data
        if inst.operation.name == "rx"
    ]


# ---------------------------------------------------------------------------
# Tests — UInt BinOp (floordiv, pow) folding into loop bounds
# ---------------------------------------------------------------------------


class TestMergeAliasRegression:
    """End-to-end regression for quantum merge alias miscompile.

    ``broken_merge_bit_array_example`` uses distinct qubit arrays (``true_q``,
    ``false_q``) in the two branches.  Before the fix the transpiler silently
    picked the true branch's physical qubits, producing wrong results
    ``[((0, 0, 0, 0, 1), 200)]``.  After the fix the classical bit
    consolidation path correctly merges the measurement results and the
    circuit returns ``(0, 1)`` — the else branch's ``false_q`` after X on
    index 1.
    """

    def test_merge_bit_array_produces_correct_result(self) -> None:
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(broken_merge_bit_array_example)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=200, bindings={})
        results = job.result().results
        # m = measure(c[0]) is always 0 → else branch taken
        # false_q[1] = X(false_q[1]) → false_q = [|0>, |1>]
        # out = measure(false_q) → (0, 1)
        assert len(results) == 1
        bitstring, count = results[0]
        assert bitstring == (0, 1), (
            f"Expected (0, 1) but got {bitstring}; "
            "before the fix this was (0, 0, 0, 0, 1)"
        )
        assert count == 200

    def test_merge_bit_scalar_produces_correct_result(self) -> None:
        """Scalar bit merge must pick the else branch (measure q[2] = |1>)."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(broken_merge_bit_scalar_example)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=200, bindings={})
        results = job.result().results
        # b = measure(q[0]) = 0 → else branch → b = measure(q[2]) = 1
        assert len(results) == 1
        bitstring, count = results[0]
        assert bitstring == 1, (
            f"Expected 1 but got {bitstring}; before the fix this was 0"
        )
        assert count == 200

    def test_frontend_target_vars_leak_produces_correct_result(self) -> None:
        """Store-only reassignment of ``b`` must be merged so second if sees new value."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(frontend_target_vars_leak_example)
        executor = transpiler.executor()
        job = exe.sample(executor, shots=200, bindings={})
        results = job.result().results
        # c = 0 → else: X(q[2]), b = measure(q[2]) = 1
        # b = 1 → if b: X(q[3]) → measure(q[3]) = 1
        assert len(results) == 1
        bitstring, count = results[0]
        assert bitstring == 1, (
            f"Expected 1 but got {bitstring}; before the fix this was 0 due to stale b"
        )
        assert count == 200


class TestUIntBinOpFolding:
    """Tests for UInt BinOp kinds (``//``, ``%``, ``**``) that affect loop bounds.

    At tracing time ``n`` is a symbolic ``UInt`` handle, so ``n // 2``
    emits a ``BinOp(FLOORDIV)`` into the IR.  The constant folding pass
    resolves the parameter binding and evaluates the BinOp, producing
    a concrete loop bound that the emit pass can unroll.
    """

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 2),  # 4 // 2 = 2
            (6, 0.3, 3),  # 6 // 2 = 3
            (2, 1.0, 1),  # 2 // 2 = 1
        ],
        ids=["4//2=2", "6//2=3", "2//2=1"],
    )
    def test_floordiv_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n // 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_floordiv_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n//2={n // 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (2, 0.3, 4),  # 2 ** 2 = 4
            (1, 0.5, 1),  # 1 ** 2 = 1
        ],
        ids=["2**2=4", "1**2=1"],
    )
    def test_pow_loop_bound(self, n: int, theta: float, expected_rx_count: int) -> None:
        """``n ** 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_pow_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n**2={n**2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 1),  # 4 % 3 = 1
            (5, 0.3, 2),  # 5 % 3 = 2
            (6, 1.0, 0),  # 6 % 3 = 0 (boundary: empty loop)
            (2, 0.2, 2),  # 2 % 3 = 2
        ],
        ids=["4%3=1", "5%3=2", "6%3=0", "2%3=2"],
    )
    def test_mod_loop_bound(self, n: int, theta: float, expected_rx_count: int) -> None:
        """``n % 3`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_mod_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n%3={n % 3}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (2, 0.5, 1),  # 7 % 2 = 1
            (3, 0.3, 1),  # 7 % 3 = 1
            (4, 1.0, 3),  # 7 % 4 = 3
        ],
        ids=["7%2=1", "7%3=1", "7%4=3"],
    )
    def test_rmod_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``7 % n`` (UInt.__rmod__) correctly folded as loop bound."""
        _, qc = _transpile_and_get_circuit(
            binop_rmod_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, 7%n={7 % n}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )


class TestEmitArrayElementResolution:
    """Integration coverage for array elements resolved during emit."""

    def test_bound_uint_array_element_executes_as_loop_bound(self) -> None:
        """``qmc.range(bounds[0])`` executes the expected deterministic flips."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            array_element_loop_bound_circuit,
            bindings={"bounds": np.array([3], dtype=np.uint64), "theta": np.pi},
        )
        results = exe.sample(transpiler.executor(), shots=200).result().results

        assert results == [((1, 1, 1, 0), 200)]

    def test_vector_view_element_executes_through_helper_kernel(self) -> None:
        """``view[0]`` passed to a helper executes with the root array element."""
        transpiler = QiskitTranspiler()
        slopes = np.array([0.1, np.pi, 0.7], dtype=np.float64)
        exe = transpiler.transpile(
            vector_view_element_helper_circuit,
            bindings={"slopes_p": slopes},
        )
        results = exe.sample(transpiler.executor(), shots=200).result().results

        assert results == [((1,), 200)]


class TestAllocatorAnalysisStateIsolation:
    """Nested allocations must not leak analysis state into the segment.

    ``allocate`` recomputes the measurement-taint set, the safe
    mixed-merge allowlist, and the monotonic counters for the operation
    list it receives. Controlled-block / native-inverse emission reuses
    the segment allocator on a sub-block mid-emission; without the
    ``preserving_analysis_state`` snapshot, later
    ``resolve_iteration_maps`` replays of the enclosing segment consult
    the sub-block's (typically empty) sets — silently disarming the
    runtime-mux guards.
    """

    def test_preserving_analysis_state_restores_segment_sets(self) -> None:
        """The context manager restores every analysis field on exit."""
        from qamomile.circuit.transpiler.passes.emit_support.resource_allocator import (
            ResourceAllocator,
        )

        allocator = ResourceAllocator()
        allocator._measurement_tainted = {"segment-tainted-uuid"}
        allocator._safe_mixed_bit_merge_outputs = frozenset({"segment-safe-uuid"})
        allocator._next_qubit_index = 5
        allocator._next_clbit_index = 3

        with allocator.preserving_analysis_state():
            allocator._measurement_tainted = set()
            allocator._safe_mixed_bit_merge_outputs = frozenset()
            allocator._next_qubit_index = 0
            allocator._next_clbit_index = 0

        assert allocator._measurement_tainted == {"segment-tainted-uuid"}
        assert allocator._safe_mixed_bit_merge_outputs == frozenset(
            {"segment-safe-uuid"}
        )
        assert allocator._next_qubit_index == 5
        assert allocator._next_clbit_index == 3

    def test_preserving_analysis_state_restores_on_error(self) -> None:
        """State is restored even when the nested work raises."""
        from qamomile.circuit.transpiler.passes.emit_support.resource_allocator import (
            ResourceAllocator,
        )

        allocator = ResourceAllocator()
        allocator._measurement_tainted = {"segment-tainted-uuid"}

        with pytest.raises(RuntimeError, match="nested failure"):
            with allocator.preserving_analysis_state():
                allocator._measurement_tainted = set()
                raise RuntimeError("nested failure")

        assert allocator._measurement_tainted == {"segment-tainted-uuid"}
