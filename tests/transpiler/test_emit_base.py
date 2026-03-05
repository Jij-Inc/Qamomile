"""Unit tests for emit_base.py — ResourceAllocator phi_ops allocation.

Covers fixes for Bug #6 (IfOperation phi_ops not allocated):
    ResourceAllocator._allocate_recursive() now calls _allocate_phi_ops()
    after processing IfOperation branches.  _allocate_phi_ops() delegates
    to the shared map_phi_outputs() utility which:
    - Maps scalar QubitType phi outputs to the same physical qubit as
      the branch source value.
    - Copies composite element keys ``{source_uuid}_{i}`` →
      ``{output_uuid}_{i}`` for ArrayValue phi outputs so subsequent
      element accesses resolve correctly.
    - Maps scalar BitType phi outputs to the same classical bit index.
"""

from __future__ import annotations

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
