"""Unit tests for value_mapping.py — UUIDRemapper and ValueSubstitutor.

Covers fixes for:
- Bug #5 (ArrayValue.shape cloning/substitution):
    UUIDRemapper.clone_value() now clones ArrayValue.shape dimension Values
    with fresh UUIDs so sub-kernel dimension parameters survive inlining.
    ValueSubstitutor.substitute_value() now substitutes shape dimension UUIDs
    through the value_map so callers' concrete sizes propagate correctly.

- Bug #6 (IfOperation phi_ops cloning/substitution):
    UUIDRemapper.clone_operation() now recurses into IfOperation.phi_ops,
    giving each PhiOp fresh UUIDs.
    ValueSubstitutor.substitute_operation() now substitutes values inside
    IfOperation.phi_ops so inlined phi operands are correctly remapped.
"""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.value import ArrayValue, Value
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
# Bug #6: IfOperation phi_ops cloning and substitution
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
