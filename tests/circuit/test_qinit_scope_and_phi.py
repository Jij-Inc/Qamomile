"""Tests for QInitOperation scope (Bug #5) and IfOperation phi_ops (Bug #6).

Bug #5: ArrayValue.shape was not cloned/substituted during sub-kernel
inlining, causing QInitOperation to fail resolving array size at emit time.

Bug #6: IfOperation.phi_ops were never processed by the resource allocator
or emitter, causing phi output UUIDs to be missing from qubit_map.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import pytest

from qamomile.circuit.ir.block import Block, BlockKind
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
    FloatType,
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
    """Create a simple Value."""
    return Value(type=type_cls(), name=name)


def _make_const_value(name: str, const: int | float, type_cls: type = UIntType) -> Value:
    """Create a constant Value."""
    return Value(type=type_cls(), name=name, params={"const": const})


def _make_array_value(
    name: str,
    shape_vals: tuple[Value, ...] = (),
    type_cls: type = QubitType,
) -> ArrayValue:
    """Create an ArrayValue with given shape."""
    return ArrayValue(type=type_cls(), name=name, shape=shape_vals)


# ===========================================================================
# Bug #5: ArrayValue.shape cloning and substitution
# ===========================================================================


class TestArrayValueShapeCloning:
    """Tests that UUIDRemapper clones ArrayValue.shape values."""

    def test_shape_values_get_fresh_uuids(self) -> None:
        """Cloned ArrayValue should have shape values with new UUIDs."""
        dim = _make_value("num_qubits")
        arr = _make_array_value("q", shape_vals=(dim,))

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert cloned.uuid != arr.uuid
        assert len(cloned.shape) == 1
        assert cloned.shape[0].uuid != dim.uuid
        assert cloned.shape[0].name == dim.name

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

    def test_multidimensional_shape_is_cloned(self) -> None:
        """ArrayValue with multiple shape dimensions clones all of them."""
        dim0 = _make_value("rows")
        dim1 = _make_value("cols")
        arr = _make_array_value("matrix", shape_vals=(dim0, dim1))

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert len(cloned.shape) == 2
        assert cloned.shape[0].uuid != dim0.uuid
        assert cloned.shape[1].uuid != dim1.uuid


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

    def test_phi_output_array_composite_keys_are_allocated(self) -> None:
        """Phi output for an ArrayValue should copy composite element keys."""
        # QInit → qubit array of size 2
        size_val = _make_const_value("size", 2)
        q_array = _make_array_value("q", shape_vals=(size_val,))
        qinit_op = QInitOperation(operands=[], results=[q_array])

        # Measure q[0] → bit (condition)
        q0_idx = _make_const_value("idx_0", 0)
        q0_elem = Value(
            type=QubitType(), name="q[0]",
            parent_array=q_array, element_indices=(q0_idx,),
        )
        bit = _make_value("bit", BitType)
        measure_op = MeasureOperation(operands=[q0_elem], results=[bit])

        # If-else with phi on the whole array
        # true branch: X gate on q[1]
        q1_idx = _make_const_value("idx_1", 1)
        q1_elem = Value(
            type=QubitType(), name="q[1]",
            parent_array=q_array, element_indices=(q1_idx,),
        )
        q1_after_x = Value(
            type=QubitType(), name="q[1]_x",
            parent_array=q_array, element_indices=(q1_idx,),
        )
        x_gate = GateOperation(
            operands=[q1_elem],
            results=[q1_after_x],
            gate_type=GateOperationType.X,
        )

        # Phi merges the whole array
        phi_array = _make_array_value("q_phi", shape_vals=(size_val,))
        phi = PhiOp(
            operands=[bit, q_array, q_array],
            results=[phi_array],
        )

        if_op = IfOperation(
            operands=[bit],
            results=[phi_array],
            true_operations=[x_gate],
            false_operations=[],
            phi_ops=[phi],
        )

        operations = [qinit_op, measure_op, if_op]

        allocator = ResourceAllocator()
        qubit_map, _ = allocator.allocate(operations, bindings={})

        # Composite keys for the phi output array should exist
        phi_key_0 = f"{phi_array.uuid}_0"
        phi_key_1 = f"{phi_array.uuid}_1"
        src_key_0 = f"{q_array.uuid}_0"
        src_key_1 = f"{q_array.uuid}_1"

        assert phi_key_0 in qubit_map, "phi array element 0 not allocated"
        assert phi_key_1 in qubit_map, "phi array element 1 not allocated"
        assert qubit_map[phi_key_0] == qubit_map[src_key_0]
        assert qubit_map[phi_key_1] == qubit_map[src_key_1]


# Integration tests are in test_qinit_scope_and_phi_integration.py
# (separate file without ``from __future__ import annotations``
# because @qkernel requires resolved type annotations).
