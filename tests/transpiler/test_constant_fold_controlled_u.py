"""Tests for constant folding of ControlledUOperation fields."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass


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
        validated = transpiler.linear_validate(inlined)
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
        validated = transpiler.linear_validate(inlined)
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
        validated = transpiler.linear_validate(inlined)
        folded = transpiler.constant_fold(validated)

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert cu.num_controls == 3
        assert isinstance(cu.num_controls, int)


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

    def test_case_c_symbolic_non_index(self):
        """Case C: num_controls=n (non-index), bindings={'n':2}."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(n, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.controlled(_zgate, num_controls=n)
            ctrls, tgt = cg(ctrls, tgt)  # type: ignore
            return qm.measure(ctrls)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 2})
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
