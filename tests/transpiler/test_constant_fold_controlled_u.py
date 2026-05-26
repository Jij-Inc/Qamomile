"""Tests for constant folding of ControlledUOperation fields.

The legacy ``IndexSpecControlledU``-specific tests were removed when
the API redesign deleted the ``target_indices`` parameter and routed
``controlled_indices`` to symbolic-mode only.  Surviving coverage:

* ``ConcreteControlledU`` and ``SymbolicControlledU`` field folding
  (``num_controls``, ``controlled_indices``).
* The symbolic-to-concrete promotion in ``ConstantFoldingPass`` —
  including the new "skip promotion when ``controlled_indices`` is
  set" branch added in Step 5 of the redesign.
* Length-mismatch validation between the control vector and
  ``num_controls`` during promotion.
* End-to-end transpilation of the supported call shapes.
"""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    SymbolicControlledU,
)

# -- Helper kernels ----------------------------------------------------------


@qm.qkernel
def _zgate(q: qm.Qubit) -> qm.Qubit:
    return qm.z(q)


# -- Unit tests: constant folding pass directly ------------------------------


class TestConstantFoldControlledUFields:
    """Verify that ``ConstantFoldingPass`` folds ``ControlledUOperation`` fields."""

    @staticmethod
    def _find_controlled_u(ops):
        for op in ops:
            if isinstance(op, ControlledUOperation):
                return op
        return None

    def test_fold_num_controls_from_binop_promotes_to_concrete(self):
        """``num_controls=n-1`` folds to an ``int`` and promotes to ``ConcreteControlledU``.

        With ``controlled_indices`` left at its default (``None``) the
        constant-folding pass is free to expand the control vector into
        per-qubit operands and switch the op subclass.
        """

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.control(_zgate, num_controls=n - 1)
            qs[0 : n - 1], qs[n - 1] = cg(qs[0 : n - 1], qs[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None, "ControlledUOperation not found after folding"
        assert isinstance(cu, ConcreteControlledU), (
            f"Expected promotion to ConcreteControlledU, got {type(cu).__name__}"
        )
        assert cu.num_controls == 3

    def test_concrete_num_controls_unchanged(self):
        """A natively concrete ``num_controls`` stays unchanged through folding."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.control(_zgate, num_controls=3)
            qs[0:3], qs[3] = cg(qs[0:3], qs[3])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel)
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated)

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert isinstance(cu, ConcreteControlledU)
        assert cu.num_controls == 3

    def test_fold_controlled_indices_in_symbolic_mode(self):
        """Symbolic-mode ``controlled_indices`` entries fold to constants.

        With ``num_controls`` symbolic (``UInt``) and a mixed-literal /
        ``UInt`` ``controlled_indices`` tuple, the literal entries
        survive folding as constant ``Value``\\ s, the ``UInt`` entries
        are folded against the supplied bindings, and the op stays a
        ``SymbolicControlledU`` because the design forbids promoting
        out of the pass-through-aware emit path while
        ``controlled_indices`` is set.
        """

        @qm.qkernel
        def kernel(n: qm.UInt, k: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            target = qm.qubit("target")
            cg = qm.control(_zgate, num_controls=k)
            _qs_out, _target_out = cg(qs, target, controlled_indices=[0, 1, k - 1])
            return qm.measure(_qs_out)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4, "k": 3})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4, "k": 3})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert isinstance(cu, SymbolicControlledU), (
            f"Expected the op to stay SymbolicControlledU (controlled_indices "
            f"blocks promotion), got {type(cu).__name__}"
        )
        assert cu.controlled_indices is not None
        assert len(cu.controlled_indices) == 3
        # Literal int entries: already constant at frontend lift time.
        assert cu.controlled_indices[0].get_const() == 0
        assert cu.controlled_indices[1].get_const() == 1
        # Symbolic UInt entry: folded against the bindings.
        const_val = cu.controlled_indices[2].get_const()
        assert const_val == 2, (
            f"Expected controlled_indices[2] const=2 after folding "
            f"k=3 → k-1=2; got {const_val}"
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
        from qamomile.circuit.ir.value import ArrayValue

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(n, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=n)
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
            f"Expected promotion to ConcreteControlledU, got {type(cu).__name__}"
        )
        assert cu.num_controls == n_value
        controls = cu.control_operands
        assert len(controls) == n_value
        for i, ctrl in enumerate(controls):
            assert ctrl.parent_array is not None
            assert not isinstance(ctrl, ArrayValue)
            assert ctrl.element_indices
            assert ctrl.element_indices[0].get_const() == i
        ctrl_results = cu.results[: cu.num_controls]
        for i, ctrl_out in enumerate(ctrl_results):
            assert ctrl_out.parent_array is not None
            assert ctrl_out.element_indices[0].get_const() == i

    @pytest.mark.parametrize(
        ("vector_len", "num_controls"),
        [
            (2, 4),  # oversized num_controls (would OOB without check)
            (4, 2),  # undersized num_controls (would silently drop qubits)
        ],
    )
    def test_symbolic_promotion_rejects_vector_length_mismatch(
        self, vector_len, num_controls
    ):
        """Length mismatch between control Vector and num_controls must
        raise ValidationError during the symbolic→concrete promotion.

        Without the check, the oversized case fails later with a
        misleading ``QInit not allocated`` AssertionError from the
        allocator, and the undersized case silently uses only the first
        ``num_controls`` qubits of the Vector — both are confusing
        failure modes that the explicit check turns into a single
        clear error.
        """
        from qamomile.circuit.transpiler.errors import ValidationError

        @qm.qkernel
        def kernel(m: qm.UInt, n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(m, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=n)
            ctrls, tgt = cg(ctrls, tgt)  # type: ignore
            return qm.measure(ctrls)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        with pytest.raises(ValidationError, match="control Vector"):
            transpiler.transpile(kernel, bindings={"m": vector_len, "n": num_controls})

    def test_controlled_indices_blocks_promotion(self):
        """``controlled_indices`` set keeps the op as ``SymbolicControlledU``.

        Even when ``num_controls`` resolves to a concrete int, the
        non-``None`` ``controlled_indices`` slot prevents promotion to
        ``ConcreteControlledU`` (which has no scalar operand slot for
        a pass-through pool element).
        """

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            pool = qm.qubit_array(n, "pool")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=n - 1)
            _pool_out, _tgt_out = cg(pool, tgt, controlled_indices=[0, 1, 2])
            return qm.measure(_pool_out)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(kernel, bindings={"n": 4})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"n": 4})

        cu = self._find_controlled_u(folded.operations)
        assert cu is not None
        assert isinstance(cu, SymbolicControlledU)
        assert cu.controlled_indices is not None


# -- Integration tests: full transpilation -----------------------------------


class TestControlledUTranspileIntegration:
    """Verify that the surviving call shapes transpile successfully."""

    def test_concrete_num_controls_pool_view(self):
        """Concrete ``num_controls`` with a ``VectorView`` control pool."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            cg = qm.control(_zgate, num_controls=3)
            qs[0:3], qs[3] = cg(qs[0:3], qs[3])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel)
        assert result is not None

    def test_symbolic_num_controls_pool_view_promotion(self):
        """``num_controls=n-1`` with a ``VectorView`` pool promotes and runs."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(n, "qs")
            cg = qm.control(_zgate, num_controls=n - 1)
            qs[0 : n - 1], qs[n - 1] = cg(qs[0 : n - 1], qs[n - 1])
            return qm.measure(qs)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4})
        assert result is not None

    @pytest.mark.parametrize("n_value", [1, 2, 3])
    def test_symbolic_num_controls_separate_pool(self, n_value):
        """``num_controls=n`` with a separate control Vector and target."""

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            ctrls = qm.qubit_array(n, "ctrls")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=n)
            ctrls, tgt = cg(ctrls, tgt)  # type: ignore
            return qm.measure(ctrls)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": n_value})
        assert result is not None

    def test_symbolic_mode_controlled_indices(self):
        """Symbolic ``num_controls`` with literal ``controlled_indices`` transpiles."""

        @qm.qkernel
        def kernel(n: qm.UInt, k: qm.UInt) -> qm.Vector[qm.Bit]:
            pool = qm.qubit_array(n, "pool")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=k)
            _pool_out, _tgt_out = cg(pool, tgt, controlled_indices=[0, 1, 2])
            return qm.measure(_pool_out)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4, "k": 3})
        assert result is not None

    def test_symbolic_mode_controlled_indices_with_uint_entry(self):
        """Symbolic ``num_controls`` with a ``UInt`` entry in ``controlled_indices``."""

        @qm.qkernel
        def kernel(n: qm.UInt, k: qm.UInt) -> qm.Vector[qm.Bit]:
            pool = qm.qubit_array(n, "pool")
            tgt = qm.qubit("tgt")
            cg = qm.control(_zgate, num_controls=k)
            _pool_out, _tgt_out = cg(pool, tgt, controlled_indices=[0, 1, k - 1])
            return qm.measure(_pool_out)

        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        result = transpiler.transpile(kernel, bindings={"n": 4, "k": 3})
        assert result is not None
