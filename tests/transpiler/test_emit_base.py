"""Tests for emit_base pass — ResourceAllocator phi_ops & LoopAnalyzer.

Section 1: ResourceAllocator phi_ops allocation (Bug #6).
    ResourceAllocator._allocate_recursive() calls _allocate_phi_ops()
    after processing IfOperation branches.  _allocate_phi_ops() delegates
    to the shared map_phi_outputs() utility which:
    - Maps scalar QubitType phi outputs to the same physical qubit as
      the branch source value.
    - Copies composite element keys ``{source_uuid}_{i}`` →
      ``{output_uuid}_{i}`` for ArrayValue phi outputs.
    - Maps scalar BitType phi outputs to the same classical bit index.

Section 2: LoopAnalyzer BinOp dependency detection.
    LoopAnalyzer.should_unroll correctly identifies ForOperation loops
    containing BinOps that depend on the loop variable (directly or
    inside nested control-flow), and theta array-element access
    referencing the loop variable triggers unrolling.

Section 3: Integration tests for UInt BinOp (``//``, ``**``) folding
    into loop bounds via the constant folding pass.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    PhiOp,
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
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.emit_base import (
    LoopAnalyzer,
    ResourceAllocator,
)


# ---------------------------------------------------------------------------
# Helpers — phi_ops tests
# ---------------------------------------------------------------------------


def _make_value(name: str, type_cls: type = UIntType) -> Value:
    """Create a simple Value with the given name and type."""
    return Value(type=type_cls(), name=name)


def _make_const_value(name: str, const: int | float, type_cls: type = UIntType) -> Value:
    """Create a constant Value with a ``const`` param entry."""
    return Value(type=type_cls(), name=name, params={"const": const})


def _make_array_value(
    name: str,
    shape_vals: tuple[Value, ...] = (),
    type_cls: type = QubitType,
) -> ArrayValue:
    """Create an ArrayValue with the given shape dimension Values."""
    return ArrayValue(type=type_cls(), name=name, shape=shape_vals)


# ---------------------------------------------------------------------------
# Helpers — LoopAnalyzer tests
# ---------------------------------------------------------------------------


def _qubit(name: str = "q") -> Value:
    """Create a qubit Value."""
    return Value(type=QubitType(), name=name)


def _float_val(name: str = "theta", *, const: float | None = None) -> Value:
    """Create a float Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=FloatType(), name=name, params=params)


def _uint_val(name: str, *, const: int | None = None) -> Value:
    """Create a UInt Value, optionally constant."""
    params: dict[str, object] = {}
    if const is not None:
        params["const"] = const
    return Value(type=UIntType(), name=name, params=params)


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
    return GateOperation(
        operands=qubits,
        results=results,
        gate_type=gate_type,
        theta=theta,
    )


# ===========================================================================
# Bug #6: Phi output allocation
# ===========================================================================


class TestPhiOpsAllocation:
    """Tests that ResourceAllocator processes phi_ops for IfOperation."""

    def test_phi_output_qubit_is_allocated(self) -> None:
        """Phi output for a qubit type should be registered in qubit_map."""
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

        # If-else with phi merge
        q_true = q_after_h.next_version()
        x_gate = GateOperation(
            operands=[q_after_h],
            results=[q_true],
            gate_type=GateOperationType.X,
        )
        q_false = q_after_h  # identity in false branch

        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[bit, q_true, q_false],
            results=[phi_output],
        )

        if_op = IfOperation(
            operands=[bit],
            results=[phi_output],
            true_operations=[x_gate],
            false_operations=[],
            phi_ops=[phi],
        )

        # Measure phi output
        bit2 = _make_value("bit2", BitType)
        measure2 = MeasureOperation(operands=[phi_output], results=[bit2])

        operations = [qinit_op, h_gate, measure_op, if_op, measure2]

        allocator = ResourceAllocator()
        qubit_map, clbit_map = allocator.allocate(operations, bindings={})

        # phi_output should be in qubit_map, mapped to same physical qubit
        assert phi_output.uuid in qubit_map
        assert qubit_map[phi_output.uuid] == qubit_map[q_init_out.uuid]

    def test_phi_output_bit_is_allocated(self) -> None:
        """Phi output for a bit type should be registered in clbit_map."""
        # Setup: condition bit
        cond = _make_value("cond", BitType)

        # Create a qubit so we have something to measure
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        # Measure to get condition
        measure_op = MeasureOperation(operands=[q_out], results=[cond])

        # Bit phi: both branches produce the same bit
        true_bit = _make_value("true_bit", BitType)
        false_bit = _make_value("false_bit", BitType)
        true_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[true_bit]
        )
        false_measure = MeasureOperation(
            operands=[q_out.next_version()], results=[false_bit]
        )

        phi_bit = _make_value("phi_bit", BitType)
        phi = PhiOp(
            operands=[cond, true_bit, false_bit],
            results=[phi_bit],
        )

        if_op = IfOperation(
            operands=[cond],
            results=[phi_bit],
            true_operations=[true_measure],
            false_operations=[false_measure],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        _, clbit_map = allocator.allocate(operations, bindings={})

        # phi_bit should be in clbit_map
        assert phi_bit.uuid in clbit_map

    @pytest.mark.parametrize("array_size", [1, 2, 4])
    def test_phi_output_array_composite_keys_are_allocated(
        self, array_size: int
    ) -> None:
        """Phi output for an ArrayValue should copy composite element keys."""
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

        # If-else with phi on the whole array
        phi_array = _make_array_value("q_phi", shape_vals=(size_val,))
        phi = PhiOp(
            operands=[bit, q_array, q_array],
            results=[phi_array],
        )

        if_op = IfOperation(
            operands=[bit],
            results=[phi_array],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        # All composite keys for the phi output array should exist
        for i in range(array_size):
            phi_key = f"{phi_array.uuid}_{i}"
            src_key = f"{q_array.uuid}_{i}"
            assert phi_key in qubit_map, f"phi array element {i} not allocated"
            assert qubit_map[phi_key] == qubit_map[src_key]

    def test_identity_phi_maps_to_same_qubit(self) -> None:
        """Phi with true_val == false_val (identity) still maps correctly."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_out], results=[bit])

        # Identity phi: both branches refer to the same qubit
        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[bit, q_out, q_out],
            results=[phi_output],
        )

        if_op = IfOperation(
            operands=[bit],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        assert phi_output.uuid in qubit_map
        assert qubit_map[phi_output.uuid] == qubit_map[q_out.uuid]

    def test_phi_output_already_registered_is_skipped(self) -> None:
        """Phi output UUID already in qubit_map is not overwritten."""
        q = _make_value("q", QubitType)
        q_out = q.next_version()
        qinit_op = QInitOperation(operands=[], results=[q_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q_out], results=[bit])

        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[bit, q_out, q_out],
            results=[phi_output],
        )

        if_op = IfOperation(
            operands=[bit],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, clbit_map = allocator.allocate(operations, bindings={})

        # Pre-register the phi output with a sentinel value
        sentinel_idx = 999
        qubit_map[phi_output.uuid] = sentinel_idx

        # Re-running allocation should not overwrite it
        allocator._allocate_phi_ops(
            if_op.phi_ops, qubit_map, clbit_map
        )
        assert qubit_map[phi_output.uuid] == sentinel_idx

    def test_multiple_phi_ops_all_allocated(self) -> None:
        """Multiple phi_ops in a single IfOperation are all allocated."""
        q0 = _make_value("q0", QubitType)
        q0_out = q0.next_version()
        q1 = _make_value("q1", QubitType)
        q1_out = q1.next_version()
        qinit0 = QInitOperation(operands=[], results=[q0_out])
        qinit1 = QInitOperation(operands=[], results=[q1_out])

        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q0_out], results=[bit])

        # Two phi merges
        phi_out0 = _make_value("phi0", QubitType)
        phi_out1 = _make_value("phi1", QubitType)
        phi0 = PhiOp(operands=[bit, q0_out, q0_out], results=[phi_out0])
        phi1 = PhiOp(operands=[bit, q1_out, q1_out], results=[phi_out1])

        if_op = IfOperation(
            operands=[bit],
            results=[phi_out0, phi_out1],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi0, phi1],
        )

        operations = [qinit0, qinit1, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        assert phi_out0.uuid in qubit_map
        assert phi_out1.uuid in qubit_map
        assert qubit_map[phi_out0.uuid] == qubit_map[q0_out.uuid]
        assert qubit_map[phi_out1.uuid] == qubit_map[q1_out.uuid]

    def test_phi_bit_consolidates_both_branches(self) -> None:
        """Both branches' clbits must be consolidated to the same physical clbit.

        Under Qiskit's ``if_test``, only one branch executes, so both
        branches must measure into the same physical clbit. Otherwise
        the phi output always reads the true branch's result.
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

        phi_bit = _make_value("phi_bit", BitType)
        phi = PhiOp(
            operands=[cond, true_bit, false_bit],
            results=[phi_bit],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_bit],
            true_operations=[true_measure],
            false_operations=[false_measure],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_cond, if_op]
        allocator = ResourceAllocator()
        _, clbit_map = allocator.allocate(operations, bindings={})

        # All three — true_bit, false_bit, phi_bit — must share the same physical clbit
        assert clbit_map[true_bit.uuid] == clbit_map[false_bit.uuid]
        assert clbit_map[phi_bit.uuid] == clbit_map[true_bit.uuid]

    @pytest.mark.parametrize("array_size", [1, 2, 4])
    def test_phi_bit_array_consolidates_both_branches(
        self, array_size: int
    ) -> None:
        """BitType ArrayValue phi must consolidate per-element clbits across branches."""
        size_val = _make_const_value("size", array_size)

        q_array = _make_array_value("q", shape_vals=(size_val,))
        qinit_op = QInitOperation(operands=[], results=[q_array])

        # Measure q[0] → condition
        q0_idx = _make_const_value("idx_0", 0)
        q0_elem = Value(
            type=QubitType(), name="q[0]",
            parent_array=q_array, element_indices=(q0_idx,),
        )
        cond = _make_value("cond", BitType)
        measure_cond = MeasureOperation(operands=[q0_elem], results=[cond])

        # True branch: measure into true_bits array
        true_bits = _make_array_value("true_bits", shape_vals=(size_val,), type_cls=BitType)
        false_bits = _make_array_value("false_bits", shape_vals=(size_val,), type_cls=BitType)

        # Simulate allocation of individual array element clbits
        # (MeasureVectorOperation would do this, but for unit test we pre-fill)

        phi_bits = _make_array_value("phi_bits", shape_vals=(size_val,), type_cls=BitType)
        phi = PhiOp(
            operands=[cond, true_bits, false_bits],
            results=[phi_bits],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_bits],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        # Pre-fill clbit_map with element keys for both arrays
        clbit_map: dict[str, int] = {cond.uuid: 0}
        for i in range(array_size):
            clbit_map[f"{true_bits.uuid}_{i}"] = i + 1
            clbit_map[f"{false_bits.uuid}_{i}"] = array_size + i + 1

        from qamomile.circuit.transpiler.passes.emit_base import map_phi_outputs

        map_phi_outputs(if_op.phi_ops, {}, clbit_map)

        # Phi output elements should exist AND both branches consolidated
        for i in range(array_size):
            phi_key = f"{phi_bits.uuid}_{i}"
            true_key = f"{true_bits.uuid}_{i}"
            false_key = f"{false_bits.uuid}_{i}"
            assert phi_key in clbit_map, f"phi element {i} not allocated"
            assert clbit_map[phi_key] == clbit_map[true_key]
            assert clbit_map[false_key] == clbit_map[true_key], (
                f"element {i}: false branch clbit not consolidated"
            )


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
            operations=[binop, gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_no_binop_no_unroll(self) -> None:
        """A loop with no BinOps and no array access should not unroll."""
        q = _qubit()
        gate = _make_gate(GateOperationType.H, [q])

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False

    def test_binop_not_using_loop_var_no_unroll(self) -> None:
        """A BinOp not referencing the loop variable should not trigger unrolling."""
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
            operations=[binop],
        )

        start = _uint_val("start", const=0)
        stop = _uint_val("stop", const=3)
        step = _uint_val("step", const=1)

        for_op = ForOperation(
            operands=[start, stop, step],
            results=[],
            loop_var="i",
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
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is True

    def test_theta_array_element_without_loop_var_no_unroll(self) -> None:
        """Gate with theta = gammas[0] (constant index) should not unroll."""
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
            operations=[gate],
        )

        assert self.analyzer.should_unroll(for_op, {}) is False


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
def binop_floordiv_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n // 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n // 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
    return qmc.measure(q)


@qmc.qkernel
def binop_pow_circuit(
    n: qmc.UInt, theta: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Apply RX(theta) to first n ** 2 qubits of a 4-qubit register."""
    q = qmc.qubit_array(4, "q")
    count = n ** 2
    for i in qmc.range(count):
        q[i] = qmc.rx(q[i], angle=theta)
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


class TestUIntBinOpFolding:
    """Tests for UInt BinOp kinds (``//``, ``**``) that affect loop bounds.

    At tracing time ``n`` is a symbolic ``UInt`` handle, so ``n // 2``
    emits a ``BinOp(FLOORDIV)`` into the IR.  The constant folding pass
    resolves the parameter binding and evaluates the BinOp, producing
    a concrete loop bound that the emit pass can unroll.
    """

    @pytest.mark.parametrize(
        "n, theta, expected_rx_count",
        [
            (4, 0.5, 2),   # 4 // 2 = 2
            (6, 0.3, 3),   # 6 // 2 = 3
            (2, 1.0, 1),   # 2 // 2 = 1
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
            (2, 0.3, 4),   # 2 ** 2 = 4
            (1, 0.5, 1),   # 1 ** 2 = 1
        ],
        ids=["2**2=4", "1**2=1"],
    )
    def test_pow_loop_bound(
        self, n: int, theta: float, expected_rx_count: int
    ) -> None:
        """``n ** 2`` correctly folded as loop bound; angles verified."""
        _, qc = _transpile_and_get_circuit(
            binop_pow_circuit, bindings={"n": n, "theta": theta}
        )
        rx_angles = _extract_rx_angles(qc)

        assert len(rx_angles) == expected_rx_count, (
            f"Expected {expected_rx_count} RX gates (n={n}, n**2={n ** 2}), "
            f"got {len(rx_angles)}"
        )
        for i, angle in enumerate(rx_angles):
            assert np.isclose(angle, theta), (
                f"RX[{i}] angle {angle} != expected {theta}"
            )
