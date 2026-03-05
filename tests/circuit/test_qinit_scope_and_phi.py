"""Tests for QInitOperation scope (Bug #5) and IfOperation phi_ops (Bug #6).

Bug #5: ArrayValue.shape was not cloned/substituted during sub-kernel
inlining, causing QInitOperation to fail resolving array size at emit time.

Bug #6: IfOperation.phi_ops were never processed by the resource allocator
or emitter, causing phi output UUIDs to be missing from qubit_map.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.emit_base import ResourceAllocator
from qamomile.circuit.transpiler.passes.value_mapping import (
    UUIDRemapper,
    ValueSubstitutor,
)


# ---------------------------------------------------------------------------
# Helpers
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


# ===========================================================================
# Bug #5: ArrayValue.shape cloning and substitution
# ===========================================================================


class TestArrayValueShapeCloning:
    """Tests that UUIDRemapper clones ArrayValue.shape values."""

    @pytest.mark.parametrize("ndims", [1, 2, 3])
    def test_shape_values_get_fresh_uuids(self, ndims: int) -> None:
        """Cloned ArrayValue shape dims get new UUIDs for any dimensionality."""
        dims = tuple(_make_value(f"dim{i}") for i in range(ndims))
        arr = _make_array_value("q", shape_vals=dims)

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert cloned.uuid != arr.uuid
        assert len(cloned.shape) == ndims
        for i in range(ndims):
            assert cloned.shape[i].uuid != dims[i].uuid
            assert cloned.shape[i].name == dims[i].name

    def test_shape_uuids_appear_in_remap_table(self) -> None:
        """Cloned shape dimension UUIDs should be recorded in uuid_remap."""
        dim = _make_value("n")
        arr = _make_array_value("q", shape_vals=(dim,))

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert dim.uuid in remapper.uuid_remap
        assert isinstance(cloned, ArrayValue)
        assert remapper.uuid_remap[dim.uuid] == cloned.shape[0].uuid

    def test_empty_shape_is_preserved(self) -> None:
        """ArrayValue with empty shape is cloned without error."""
        arr = _make_array_value("q", shape_vals=())

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert cloned.shape == ()

    def test_random_dimensionality_shape_cloning(self) -> None:
        """ArrayValue with random-count dimensions clones all of them."""
        rng = np.random.default_rng(seed=42)
        ndims = int(rng.integers(2, 6))
        dims = tuple(_make_value(f"d{i}") for i in range(ndims))
        arr = _make_array_value("tensor", shape_vals=dims)

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert len(cloned.shape) == ndims
        for orig, clone in zip(dims, cloned.shape):
            assert clone.uuid != orig.uuid
            assert orig.uuid in remapper.uuid_remap


class TestArrayValueShapeSubstitution:
    """Tests that ValueSubstitutor substitutes ArrayValue.shape values."""

    def test_shape_dimension_is_substituted(self) -> None:
        """Shape dimension referenced in value_map is substituted."""
        old_dim = _make_value("num_qubits")
        new_dim = _make_const_value("n", const=4)
        arr = _make_array_value("q", shape_vals=(old_dim,))

        sub = ValueSubstitutor({old_dim.uuid: new_dim})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert len(result.shape) == 1
        assert result.shape[0].uuid == new_dim.uuid
        assert result.shape[0].get_const() == 4

    def test_shape_not_in_map_is_unchanged(self) -> None:
        """Shape dimension not in value_map is preserved."""
        dim = _make_value("n")
        arr = _make_array_value("q", shape_vals=(dim,))

        sub = ValueSubstitutor({})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert result.shape[0].uuid == dim.uuid

    def test_partial_shape_substitution(self) -> None:
        """Only matched shape dimensions are substituted."""
        dim0 = _make_value("rows")
        dim1 = _make_value("cols")
        new_dim0 = _make_const_value("r", const=3)
        arr = _make_array_value("m", shape_vals=(dim0, dim1))

        sub = ValueSubstitutor({dim0.uuid: new_dim0})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert result.shape[0].uuid == new_dim0.uuid
        assert result.shape[1].uuid == dim1.uuid


# ===========================================================================
# Bug #6: IfOperation phi_ops handling
# ===========================================================================


class TestIfOperationPhiOpsCloning:
    """Tests that UUIDRemapper clones IfOperation.phi_ops."""

    def test_phi_ops_are_cloned(self) -> None:
        """Cloning IfOperation should also clone phi_ops."""
        cond = _make_value("cond", BitType)
        true_q = _make_value("q_true", QubitType)
        false_q = _make_value("q_false", QubitType)
        phi_output = _make_value("q_phi", QubitType)

        phi = PhiOp(
            operands=[cond, true_q, false_q],
            results=[phi_output],
        )

        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        assert len(cloned.phi_ops) == 1
        cloned_phi = cloned.phi_ops[0]
        assert isinstance(cloned_phi, PhiOp)
        # Phi output should have a fresh UUID
        assert cloned_phi.results[0].uuid != phi_output.uuid

    def test_phi_ops_empty_is_preserved(self) -> None:
        """IfOperation with no phi_ops is cloned without error."""
        cond = _make_value("cond", BitType)

        if_op = IfOperation(
            operands=[cond],
            results=[],
            true_operations=[],
            false_operations=[],
            phi_ops=[],
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        assert len(cloned.phi_ops) == 0

    def test_multiple_phi_ops_are_all_cloned(self) -> None:
        """IfOperation with multiple phi_ops clones each one."""
        cond = _make_value("cond", BitType)

        phi_outputs = []
        phis = []
        for i in range(3):
            t = _make_value(f"t{i}", QubitType)
            f = _make_value(f"f{i}", QubitType)
            out = _make_value(f"phi{i}", QubitType)
            phi_outputs.append(out)
            phis.append(PhiOp(operands=[cond, t, f], results=[out]))

        if_op = IfOperation(
            operands=[cond],
            results=phi_outputs,
            true_operations=[],
            false_operations=[],
            phi_ops=phis,
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        assert len(cloned.phi_ops) == 3
        for orig_phi, cloned_phi in zip(phis, cloned.phi_ops):
            assert isinstance(cloned_phi, PhiOp)
            assert cloned_phi.results[0].uuid != orig_phi.results[0].uuid


class TestPhiOpsSubstitution:
    """Tests that ValueSubstitutor substitutes phi_ops inside IfOperation."""

    def test_phi_ops_operands_are_substituted(self) -> None:
        """ValueSubstitutor should substitute values inside phi_ops."""
        cond = _make_value("cond", BitType)
        old_q = _make_value("q_old", QubitType)
        new_q = _make_value("q_new", QubitType)
        false_q = _make_value("q_false", QubitType)
        phi_output = _make_value("q_phi", QubitType)

        phi = PhiOp(
            operands=[cond, old_q, false_q],
            results=[phi_output],
        )

        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        sub = ValueSubstitutor({old_q.uuid: new_q})
        result = sub.substitute_operation(if_op)

        assert isinstance(result, IfOperation)
        assert len(result.phi_ops) == 1
        subst_phi = result.phi_ops[0]
        # The true_val operand (index 1) should be substituted
        assert subst_phi.operands[1].uuid == new_q.uuid


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


# ===========================================================================
# ControlFlowVisitor / OperationTransformer phi_ops handling
# ===========================================================================


class TestControlFlowVisitorPhiOps:
    """Tests that ControlFlowVisitor base classes visit/transform phi_ops."""

    def test_visitor_visits_phi_ops(self) -> None:
        """ControlFlowVisitor.visit_operations visits phi_ops."""
        from qamomile.circuit.ir.operation import Operation
        from qamomile.circuit.transpiler.passes.control_flow_visitor import (
            ControlFlowVisitor,
        )

        visited: list[Operation] = []

        class Collector(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                visited.append(op)

        cond = _make_value("cond", BitType)
        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[cond, _make_value("t", QubitType), _make_value("f", QubitType)],
            results=[phi_output],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        collector = Collector()
        collector.visit_operations([if_op])

        # Should visit: if_op itself + the phi op inside
        assert if_op in visited
        assert phi in visited

    def test_transformer_transforms_phi_ops(self) -> None:
        """OperationTransformer.transform_operations transforms phi_ops."""
        import dataclasses
        from qamomile.circuit.ir.operation import Operation
        from qamomile.circuit.transpiler.passes.control_flow_visitor import (
            OperationTransformer,
        )

        class NoopTransformer(OperationTransformer):
            """Identity transform — just returns ops unchanged."""

            def transform_operation(self, op: Operation) -> Operation:
                return op

        cond = _make_value("cond", BitType)
        phi_output = _make_value("q_phi", QubitType)
        phi = PhiOp(
            operands=[cond, _make_value("t", QubitType), _make_value("f", QubitType)],
            results=[phi_output],
        )
        if_op = IfOperation(
            operands=[cond],
            results=[phi_output],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        transformer = NoopTransformer()
        result = transformer.transform_operations([if_op])

        assert len(result) == 1
        assert isinstance(result[0], IfOperation)
        # phi_ops should be preserved through transform
        assert len(result[0].phi_ops) == 1
        assert isinstance(result[0].phi_ops[0], PhiOp)


# Integration tests are in test_qinit_scope_and_phi_integration.py
# (separate file without ``from __future__ import annotations``
# because @qkernel requires resolved type annotations).
