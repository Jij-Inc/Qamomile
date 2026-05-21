"""Tests for constant folding of ControlledUOperation fields."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.gate import ControlledUOperation

# -- Helper kernels ----------------------------------------------------------


@qm.qkernel
def _zgate(q: qm.Qubit) -> qm.Qubit:
    return qm.z(q)


# -- Unit tests: constant folding pass directly ------------------------------


class TestConstantFoldControlledUFields:
    """Verify that ConstantFoldingPass folds ControlledUOperation fields."""

    @staticmethod
    def _find_controlled_u(ops):
        for op in ops:
            if isinstance(op, ControlledUOperation):
                return op
        return None

    def test_fold_num_controls_from_binop(self):
        """num_controls=n-1 with bindings={'n':4} should fold to int(3)."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=n - 1)
            qs = cg(qs, target_indices=[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None, "ControlledUOperation not found after folding"
        assert isinstance(cu.num_controls, int), (
            f"num_controls should be int, got {type(cu.num_controls)}"
        )
        assert cu.num_controls == 3

    def test_fold_target_indices_from_binop(self):
        """target_indices=[n-1] with bindings={'n':4} should have const=3."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=n - 1)
            qs = cg(qs, target_indices=[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert cu.target_indices is not None
        assert len(cu.target_indices) == 1
        const_val = cu.target_indices[0].get_const()
        assert const_val == 3, f"Expected target_indices[0] const=3, got {const_val}"

    def test_concrete_num_controls_unchanged(self):
        """Concrete num_controls=3 should pass through unmodified."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, target_indices=[3])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel)
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated)

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert cu.num_controls == 3
        assert isinstance(cu.num_controls, int)

    def test_fold_controlled_indices_from_binop(self):
        """controlled_indices=[0, 1, n-1] with bindings={'n':4} should fold n-1 to const=3."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert cu.controlled_indices is not None
        assert len(cu.controlled_indices) == 3
        const_val = cu.controlled_indices[2].get_const()
        assert const_val == 3, (
            f"Expected controlled_indices[2] const=3, got {const_val}"
        )

    def test_fold_controlled_indices_concrete(self):
        """Concrete controlled_indices=[0,1,2] should pass through unmodified."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel)
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated)

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert cu.controlled_indices is not None
        assert len(cu.controlled_indices) == 3
        for i, expected in enumerate([0, 1, 2]):
            const_val = cu.controlled_indices[i].get_const()
            assert const_val == expected, (
                f"Expected controlled_indices[{i}] const={expected}, got {const_val}"
            )

    @pytest.mark.parametrize("n_value", [1, 2, 3])
    def test_symbolic_promotion_expands_control_vector_operands(self, n_value):
        """SymbolicControlledU -> ConcreteControlledU promotion must expand
        the control Vector operand into ``num_controls`` per-qubit Values.

        Without this expansion the promoted ``ConcreteControlledU`` keeps a
        ``[ctrl_vector, tgt, ...]`` operand layout, so
        ``control_operands = operands[:nc]`` accidentally aliases the
        target into the control slice and the emit path produces a
        partial-arity controlled gate (Qiskit raises ``CircuitError``).
        """
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU
        from qamomile.circuit.ir.value import ArrayValue

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(n, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.controlled(_zgate, num_controls=n)
            ctrls, tgt = cg(ctrls, tgt)  # type: ignore
            return qm.measure(ctrls)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": n_value})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": n_value})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None, "ControlledUOperation not found after folding"
        assert isinstance(cu, ConcreteControlledU), (
            f"Expected promotion to ConcreteControlledU, "
            f"got {type(cu).__name__}"
        )
        assert cu.num_controls == n_value
        # Per-qubit control operands: each ``operands[i]`` for i<nc is a
        # scalar Qubit Value whose ``parent_array`` references the original
        # control Vector.
        controls = cu.control_operands
        assert len(controls) == n_value
        for i, ctrl in enumerate(controls):
            assert ctrl.parent_array is not None, (
                f"control_operands[{i}] should carry parent_array; got "
                f"a non-element Value instead"
            )
            assert not isinstance(ctrl, ArrayValue), (
                f"control_operands[{i}] should be a scalar Qubit Value, "
                f"not the bare control Vector"
            )
            assert ctrl.element_indices, (
                f"control_operands[{i}] should have element_indices "
                f"populated"
            )
            assert ctrl.element_indices[0].get_const() == i
        # Per-qubit control results carry the matching parent_array — the
        # next-version control Vector — so downstream MeasureVectorOperation
        # on that Vector resolves each element.
        ctrl_results = cu.results[: cu.num_controls]
        for i, ctrl_out in enumerate(ctrl_results):
            assert ctrl_out.parent_array is not None
            assert ctrl_out.element_indices[0].get_const() == i


# -- Integration tests: full transpilation -----------------------------------


class TestControlledUTranspileIntegration:
    """Verify that symbolic ControlledU transpiles successfully."""

    def test_case_a_symbolic_index_spec(self):
        """Case A: num_controls=n-1, target_indices=[n-1], bindings={'n':4}."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=n - 1)
            qs = cg(qs, target_indices=[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4})
        assert result is not None

    def test_case_b_concrete_index_spec(self):
        """Case B: concrete num_controls=3, target_indices=[3]."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, target_indices=[3])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel)
        assert result is not None

    @pytest.mark.parametrize("n_value", [1, 2, 3])
    def test_case_c_symbolic_non_index(self, n_value):
        """Case C: num_controls=n (non-index), parametrized over n_value."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(n, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.controlled(_zgate, num_controls=n)
            ctrls, tgt = cg(ctrls, tgt)  # type: ignore
            return qm.measure(ctrls)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": n_value})
        assert result is not None

    def test_case_d_concrete_nc_symbolic_index(self):
        """Case D: num_controls=3, target_indices=[n-1], bindings={'n':4}."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, target_indices=[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4})
        assert result is not None

    def test_case_e_controlled_indices_concrete(self):
        """Case E: concrete controlled_indices=[0,1,2], num_controls=3."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel)
        assert result is not None

    def test_case_f_controlled_indices_symbolic(self):
        """Case F: controlled_indices=[0, 1, n-1], num_controls=3, bindings={'n':4}."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.controlled(_zgate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4})
        assert result is not None


# -- Power field strict-int-cast unit tests ----------------------------------


from qamomile.circuit.transpiler.passes.constant_fold import (  # noqa: E402
    ConstantFoldingPass,
)


class TestConstantFoldPowerStrictValidation:
    """Verify that _strict_int_cast rejects invalid power types."""

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            ConstantFoldingPass._strict_int_cast(True)

    def test_bool_false_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            ConstantFoldingPass._strict_int_cast(False)

    def test_non_integer_float_rejected(self):
        with pytest.raises(ValueError, match="non-integer float"):
            ConstantFoldingPass._strict_int_cast(1.5)

    def test_zero_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ConstantFoldingPass._strict_int_cast(0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            ConstantFoldingPass._strict_int_cast(-1)

    def test_valid_integer_accepted(self):
        assert ConstantFoldingPass._strict_int_cast(4) == 4

    def test_whole_float_accepted(self):
        assert ConstantFoldingPass._strict_int_cast(4.0) == 4
