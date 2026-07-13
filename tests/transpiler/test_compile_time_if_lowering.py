"""Direct tests for CompileTimeIfLoweringPass.

Tests the pass-internal contracts that are not observable from backend
circuit success alone: exact Block rewrites, merge substitution, recursive
lowering, dead-op elimination, and runtime IfOperation preservation.
"""

import numpy as np

import qamomile.circuit as qm
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation, SliceArrayOperation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForOperation,
    IfOperation,
    LoopCarriedRebind,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    CompileTimeIfLoweringPass,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower(kernel, bindings=None):
    """Run pipeline up through lower_compile_time_ifs and return the block."""
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    block = transpiler.to_block(kernel, bindings=bindings)
    inlined = transpiler.inline(transpiler.substitute(block))
    validated = transpiler.affine_validate(inlined)
    folded = transpiler.constant_fold(validated, bindings=bindings)
    return transpiler.lower_compile_time_ifs(folded, bindings=bindings)


def _find_ops(ops, op_type):
    """Find all operations of a given type in a flat operation list."""
    return [op for op in ops if isinstance(op, op_type)]


def _find_gates(ops, gate_type=None):
    """Find GateOperations, optionally filtering by gate_type."""
    gates = _find_ops(ops, GateOperation)
    if gate_type is not None:
        gates = [g for g in gates if g.gate_type == gate_type]
    return gates


# ---------------------------------------------------------------------------
# Synthetic IR helpers
# ---------------------------------------------------------------------------


def _uint_val(name, *, const=None):
    value = Value(type=UIntType(), name=name)
    return value.with_const(const) if const is not None else value


def _bit_val(name):
    return Value(type=BitType(), name=name)


def _float_val(name, *, const=None):
    value = Value(type=FloatType(), name=name)
    return value.with_const(const) if const is not None else value


def _qubit_val(name="q"):
    return Value(type=QubitType(), name=name)


def _run_pass(block, bindings=None):
    return CompileTimeIfLoweringPass(bindings=bindings or {}).run(block)


def _make_if_with_x_gate(condition_val, qubit_in):
    """Build IfOperation(condition, [X gate], []) with a merge slot for the qubit."""
    q_true = qubit_in.next_version()
    x_gate = GateOperation(
        operands=[qubit_in],
        results=[q_true],
        gate_type=GateOperationType.X,
    )
    q_false = qubit_in
    merge_out = qubit_in.next_version()
    if_op = IfOperation(
        operands=[condition_val],
        true_operations=[x_gate],
        false_operations=[],
    )
    if_op.add_merge(q_true, q_false, merge_out)
    return if_op, merge_out


def _make_comparison_if_with_x_gate(
    selector: Value,
    qubit: Value,
) -> tuple[CompOp, IfOperation]:
    """Build ``selector == 0`` followed by a conditional X gate.

    Args:
        selector (Value): UInt value compared with zero.
        qubit (Value): Qubit consumed by the true-branch X gate.

    Returns:
        tuple[CompOp, IfOperation]: The comparison producer and its
            condition-consuming if operation.
    """
    zero = _uint_val("zero", const=0)
    condition = _bit_val("condition")
    comparison = CompOp(
        operands=[selector, zero],
        results=[condition],
        kind=CompOpKind.EQ,
    )
    if_op, _ = _make_if_with_x_gate(condition, qubit)
    return comparison, if_op


# ---------------------------------------------------------------------------
# Test: simple if flag lowering (true/false)
# ---------------------------------------------------------------------------


class TestSimpleIfLowering:
    """Compile-time `if flag:` removes IfOperation and inlines the branch."""

    def test_flag_true_selects_true_branch(self):
        """flag=1: IfOperation removed, X gate from true branch present."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 1})

        assert not _find_ops(lowered.operations, IfOperation), (
            "IfOperation should be removed after lowering"
        )
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) >= 1, "X gate from true branch should be present"

    def test_flag_false_selects_false_branch(self):
        """flag=0: IfOperation removed, no X gate."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 0})

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 0, "No X gate when flag=0"


# ---------------------------------------------------------------------------
# Test: expression-derived conditions
# ---------------------------------------------------------------------------


class TestExpressionConditions:
    """Conditions derived from CompOp, NotOp are resolved and lowered."""

    def test_comp_op_greater_than_true(self):
        """flag > 0 with flag=1: resolves to true, X gate present."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag > 0:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 1})

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) >= 1

    def test_comp_op_greater_than_false(self):
        """flag > 0 with flag=0: resolves to false, no X gate."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag > 0:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 0})

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 0


# ---------------------------------------------------------------------------
# Test: nested compile-time ifs
# ---------------------------------------------------------------------------


class TestNestedCompileTimeIf:
    """Nested compile-time ifs are recursively lowered."""

    def test_nested_both_true(self):
        """Both outer and inner flags true: both branches inlined."""

        @qm.qkernel
        def kernel(a: qm.UInt, b: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if a:
                q[0] = qm.x(q[0])
                if b:
                    q[0] = qm.h(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"a": 1, "b": 1})

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1
        assert len(_find_gates(lowered.operations, GateOperationType.H)) >= 1

    def test_nested_outer_true_inner_false(self):
        """Outer true, inner false: X present, H absent."""

        @qm.qkernel
        def kernel(a: qm.UInt, b: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if a:
                q[0] = qm.x(q[0])
                if b:
                    q[0] = qm.h(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"a": 1, "b": 0})

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1
        assert len(_find_gates(lowered.operations, GateOperationType.H)) == 0

    def test_nested_outer_false(self):
        """Outer false: nothing from either branch."""

        @qm.qkernel
        def kernel(a: qm.UInt, b: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if a:
                q[0] = qm.x(q[0])
                if b:
                    q[0] = qm.h(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"a": 0, "b": 1})

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) == 0
        assert len(_find_gates(lowered.operations, GateOperationType.H)) == 0


# ---------------------------------------------------------------------------
# Test: compile-time ifs inside a controlled-U's nested block
# ---------------------------------------------------------------------------


def _first_controlled_block_ops(block: Block) -> list[Operation]:
    """Return the operations of the first ControlledUOperation's nested block.

    Args:
        block (Block): A lowered block expected to contain a controlled-U.

    Returns:
        list[Operation]: The nested ``block.operations`` of the first
            ``ControlledUOperation`` found.
    """
    for op in block.operations:
        if isinstance(op, ControlledUOperation):
            assert op.block is not None
            return op.block.operations
    raise AssertionError("no ControlledUOperation found in block")


def _controlled_invoke_ops(block: Block) -> list[InvokeOperation]:
    """Return the top-level controlled invocation operations in a block.

    Args:
        block (Block): Lowered block to inspect.

    Returns:
        list[InvokeOperation]: Controlled boxed invocations in program order.
    """
    return [
        op
        for op in block.operations
        if isinstance(op, InvokeOperation) and op.transform is CallTransform.CONTROLLED
    ]


class TestControlledBlockIfLowering:
    """Compile-time ifs inside a ``qm.control`` body are lowered too.

    The body of a controlled-U is a self-contained block with its own value
    namespace, so its compile-time ``if sel == k`` must be resolved by
    recursing into the block with bindings reconstructed from the controlled
    operands — never the outer bindings, which could collide by name. Before
    this lowering the comparison survives as a ``CompOp`` + ``IfOperation``
    pair that the QURI Parts / CUDA-Q controlled-emission walks reject.
    """

    @staticmethod
    def _comp_if_body() -> qm.QKernel:
        """Build a one-qubit kernel that X-flips its target when ``sel == 0``.

        Returns:
            QKernel: A kernel ``(q, sel) -> q`` whose body is a single
                comparison-conditioned compile-time ``if sel == 0``.
        """

        @qm.qkernel
        def x_if_cmp(q: qm.Qubit, sel: qm.UInt) -> qm.Qubit:
            if sel == 0:
                q = qm.x(q)
            return q

        return x_if_cmp

    def test_comparison_true_branch_inlined_in_block(self):
        """sel=0: the controlled block holds the X gate, no if/CompOp remain."""
        x_if_cmp = self._comp_if_body()

        @qm.qkernel
        def circ(sel: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_cmp)(qs[0], qs[1], sel)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ, bindings={"sel": 0}))

        assert not _find_ops(block_ops, IfOperation), (
            "IfOperation should be lowered out of the controlled block"
        )
        assert not _find_ops(block_ops, CompOp), (
            "CompOp condition should be dead-eliminated from the controlled block"
        )
        assert len(_find_gates(block_ops, GateOperationType.X)) == 1, (
            "true branch X gate should remain in the controlled block"
        )

    def test_comparison_false_branch_empty_in_block(self):
        """sel=1: the controlled block is empty of gates, no if/CompOp remain."""
        x_if_cmp = self._comp_if_body()

        @qm.qkernel
        def circ(sel: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_cmp)(qs[0], qs[1], sel)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ, bindings={"sel": 1}))

        assert not _find_ops(block_ops, IfOperation)
        assert not _find_ops(block_ops, CompOp)
        assert len(_find_gates(block_ops, GateOperationType.X)) == 0, (
            "false branch leaves no X gate in the controlled block"
        )

    def test_inner_binding_comes_from_operand_not_outer_name(self):
        """A literal operand wins over a same-named outer binding.

        The outer kernel binds ``sel=5`` while the controlled call passes a
        literal ``0`` for the body's own ``sel`` parameter. The inner ``if
        sel == 0`` must resolve against the operand (0 → true branch), proving
        the recursion seeds from operands rather than leaking the colliding
        outer ``sel`` name.
        """
        x_if_cmp = self._comp_if_body()

        @qm.qkernel
        def circ(sel: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_cmp)(qs[0], qs[1], 0)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ, bindings={"sel": 5}))

        assert not _find_ops(block_ops, IfOperation)
        assert len(_find_gates(block_ops, GateOperationType.X)) == 1, (
            "inner sel=0 (operand) must select the true branch despite outer sel=5"
        )

    def test_pairs_multiple_classical_params_by_position(self):
        """Two classical params: the operand-to-input zip must stay aligned.

        The body branches on its SECOND classical parameter (``b``). Only a
        correctly ordered pairing of the ``[a, b]`` inputs with the
        ``[a_operand, b_operand]`` operands selects the right branch, so this
        is the discriminating case the single-parameter tests cannot give:
        with one parameter a mis-order is unobservable.
        """

        @qm.qkernel
        def x_if_b_zero(q: qm.Qubit, a: qm.UInt, b: qm.UInt) -> qm.Qubit:
            if b == 0:
                q = qm.x(q)
            return q

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            # a=1 (non-zero), b=0 (zero): branching on b must fire the X.
            qs[0], qs[1] = qm.control(x_if_b_zero)(qs[0], qs[1], 1, 0)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ))

        assert not _find_ops(block_ops, IfOperation)
        assert not _find_ops(block_ops, CompOp)
        assert len(_find_gates(block_ops, GateOperationType.X)) == 1, (
            "branch on the 2nd param (b=0) must select the X; pairing b with "
            "the a operand (=1) would drop it"
        )

    def test_multi_param_pairing_is_not_swapped(self):
        """Swapping the two operands flips the outcome — pins pairing order.

        Same ``if b == 0`` body, now called with ``a=0, b=1``. The X must NOT
        appear; if the pairing bound ``b`` to the ``a`` operand (=0) it would
        wrongly fire.
        """

        @qm.qkernel
        def x_if_b_zero(q: qm.Qubit, a: qm.UInt, b: qm.UInt) -> qm.Qubit:
            if b == 0:
                q = qm.x(q)
            return q

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_b_zero)(qs[0], qs[1], 0, 1)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ))

        assert not _find_ops(block_ops, IfOperation)
        assert len(_find_gates(block_ops, GateOperationType.X)) == 0, (
            "a=0 must not fire the b==0 branch; a swapped pairing would add an X"
        )

    def test_bare_truthiness_condition_lowered_in_block(self):
        """A bare ``if flag:`` inside a controlled body lowers via the UUID key.

        The inner bindings are keyed only by the inner input UUID (no name
        key), so this pins that a non-comparison truthiness condition still
        resolves from the operand through ``resolve_if_condition``'s
        ``uuid in bindings`` path rather than a name shortcut.
        """

        @qm.qkernel
        def x_if_flag(q: qm.Qubit, flag: qm.UInt) -> qm.Qubit:
            if flag:
                q = qm.x(q)
            return q

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_flag)(qs[0], qs[1], 1)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(_lower(circ))

        assert not _find_ops(block_ops, IfOperation), (
            "bare truthiness condition should lower without a name key"
        )
        assert len(_find_gates(block_ops, GateOperationType.X)) == 1

    def test_array_element_condition_uses_operand_bound_container(self):
        """An inner Vector element resolves from the call-site array operand."""

        @qm.qkernel
        def x_if_first_zero(
            q: qm.Qubit,
            selectors: qm.Vector[qm.UInt],
        ) -> qm.Qubit:
            if selectors[0] == 0:
                q = qm.x(q)
            return q

        @qm.qkernel
        def circ(selectors: qm.Vector[qm.UInt]) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_first_zero)(qs[0], qs[1], selectors)
            return qm.measure(qs)

        block_ops = _first_controlled_block_ops(
            _lower(circ, bindings={"selectors": np.array([0], dtype=np.int64)})
        )

        assert not _find_ops(block_ops, IfOperation)
        assert not _find_ops(block_ops, CompOp)
        assert len(_find_gates(block_ops, GateOperationType.X)) == 1

    def test_shared_body_is_lowered_independently_per_call(self):
        """Two calls to one body keep their different operand bindings isolated."""
        x_if_cmp = self._comp_if_body()

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(3, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_cmp)(qs[0], qs[1], 0)
            qs[0], qs[2] = qm.control(x_if_cmp)(qs[0], qs[2], 1)
            return qm.measure(qs)

        lowered = _lower(circ)
        controlled_ops = _find_ops(lowered.operations, ControlledUOperation)

        assert len(controlled_ops) == 2
        first_block = controlled_ops[0].block
        second_block = controlled_ops[1].block
        assert first_block is not None
        assert second_block is not None
        first_ops = first_block.operations
        second_ops = second_block.operations
        assert len(_find_gates(first_ops, GateOperationType.X)) == 1
        assert not _find_ops(first_ops, IfOperation)
        assert len(_find_gates(second_ops, GateOperationType.X)) == 0
        assert not _find_ops(second_ops, IfOperation)

    def test_nested_controlled_block_lowered(self):
        """A controlled-U whose body wraps another controlled-U + inner if.

        The recursion must reach the innermost block: the deepest controlled
        body's ``if sel == 0`` folds to a single X, with no residual
        if/CompOp at either level.
        """

        @qm.qkernel
        def x_if_sel(q: qm.Qubit, sel: qm.UInt) -> qm.Qubit:
            if sel == 0:
                q = qm.x(q)
            return q

        @qm.qkernel
        def inner_ctrl(
            c: qm.Qubit, t: qm.Qubit, sel: qm.UInt
        ) -> tuple[qm.Qubit, qm.Qubit]:
            c, t = qm.control(x_if_sel)(c, t, sel)
            return c, t

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(3, "qs")
            qs[0] = qm.x(qs[0])
            qs[1] = qm.x(qs[1])
            qs[0], qs[1], qs[2] = qm.control(inner_ctrl)(qs[0], qs[1], qs[2], 0)
            return qm.measure(qs)

        outer_ops = _first_controlled_block_ops(_lower(circ))
        inner_ctrls = _find_ops(outer_ops, ControlledUOperation)
        assert len(inner_ctrls) == 1, (
            "outer controlled body should hold the nested controlled-U"
        )
        inner_ops = inner_ctrls[0].block.operations
        assert not _find_ops(inner_ops, IfOperation)
        assert not _find_ops(inner_ops, CompOp)
        assert len(_find_gates(inner_ops, GateOperationType.X)) == 1, (
            "innermost sel=0 must fold to a single X through nested recursion"
        )

    def test_unresolved_operand_leaves_inner_if_unlowered(self):
        """An unbound (runtime) operand keeps the inner if symbolic — fail-safe.

        When the controlled operand does not resolve to a compile-time
        constant, the recursion must leave the inner ``if`` in place (as a
        CompOp + IfOperation pair) rather than folding a branch, so a backend
        that cannot handle it fails loudly instead of silently miscompiling.
        """
        x_if_cmp = self._comp_if_body()

        @qm.qkernel
        def circ(sel: qm.UInt) -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(x_if_cmp)(qs[0], qs[1], sel)
            return qm.measure(qs)

        # No binding for ``sel``: it stays a runtime parameter, so the inner
        # comparison cannot be resolved at compile time.
        block_ops = _first_controlled_block_ops(_lower(circ))

        assert _find_ops(block_ops, IfOperation), (
            "unresolved operand must leave the inner if in place"
        )
        assert _find_ops(block_ops, CompOp), (
            "the comparison condition must survive as a CompOp"
        )

    def test_if_inside_controlled_for_body_removes_condition_producer(self):
        """A folded if nested in a controlled For leaves no dead CompOp.

        Controlled emission supports ``ForOperation`` by walking its body,
        but rejects standalone ``CompOp`` nodes. Dead-condition elimination
        must therefore recurse into the loop body after lowering the if.
        """
        inner_q = _qubit_val("inner_q")
        inner_sel = _uint_val("inner_sel")
        comparison, if_op = _make_comparison_if_with_x_gate(inner_sel, inner_q)
        loop = ForOperation(
            operands=[
                _uint_val("start", const=0),
                _uint_val("stop", const=2),
                _uint_val("step", const=1),
            ],
            loop_var="i",
            loop_var_value=_uint_val("i"),
            operations=[comparison, if_op],
        )
        unitary = Block(
            name="unitary",
            input_values=[inner_q, inner_sel],
            operations=[loop],
            kind=BlockKind.AFFINE,
        )
        control = _qubit_val("control")
        target = _qubit_val("target")
        controlled = ConcreteControlledU(
            operands=[control, target, _uint_val("actual_sel", const=0)],
            results=[control.next_version(), target.next_version()],
            num_controls=1,
            block=unitary,
        )
        outer = Block(
            name="outer",
            operations=[controlled],
            kind=BlockKind.AFFINE,
        )

        nested_ops = _first_controlled_block_ops(_run_pass(outer))

        [lowered_loop] = _find_ops(nested_ops, ForOperation)
        assert not _find_ops(lowered_loop.operations, IfOperation)
        assert not _find_ops(lowered_loop.operations, CompOp)
        assert len(_find_gates(lowered_loop.operations, GateOperationType.X)) == 1

    def test_if_inside_controlled_while_body_is_lowered_in_fresh_scope(self):
        """A controlled-owned While body receives isolated compile-time values."""
        inner_q = _qubit_val("inner_q")
        inner_sel = _uint_val("inner_sel")
        comparison, if_op = _make_comparison_if_with_x_gate(inner_sel, inner_q)
        while_op = WhileOperation(
            operands=[_bit_val("runtime_condition")],
            operations=[comparison, if_op],
        )
        unitary = Block(
            name="unitary",
            input_values=[inner_q, inner_sel],
            operations=[while_op],
            kind=BlockKind.AFFINE,
        )
        control = _qubit_val("control")
        target = _qubit_val("target")
        controlled = ConcreteControlledU(
            operands=[control, target, _uint_val("actual_sel", const=0)],
            results=[control.next_version(), target.next_version()],
            num_controls=1,
            block=unitary,
        )
        outer = Block(
            name="outer",
            operations=[controlled],
            kind=BlockKind.AFFINE,
        )

        nested_ops = _first_controlled_block_ops(_run_pass(outer))

        [lowered_while] = _find_ops(nested_ops, WhileOperation)
        assert not _find_ops(lowered_while.operations, IfOperation)
        assert not _find_ops(lowered_while.operations, CompOp)
        assert len(_find_gates(lowered_while.operations, GateOperationType.X)) == 1


# ---------------------------------------------------------------------------
# Test: compile-time ifs inside controlled boxed invocation bodies
# ---------------------------------------------------------------------------


class TestControlledInvokeIfLowering:
    """Controlled InvokeOperation bodies use the same isolated lowering."""

    @staticmethod
    def _boxed_comp_if_body() -> qm.QKernel:
        """Build a boxed one-qubit callable conditioned on ``sel == 0``.

        Returns:
            QKernel: Composite callable with a compile-time conditional X gate.
        """

        @qm.composite_gate(name="boxed_x_if_zero")
        def boxed_x_if_zero(q: qm.Qubit, sel: qm.UInt) -> qm.Qubit:
            """Apply X when the compile-time selector equals zero."""
            if sel == 0:
                q = qm.x(q)
            return q

        return boxed_x_if_zero

    def test_shared_definition_is_lowered_independently_per_call(self):
        """Two controlled calls clone and fold one shared composite body."""
        boxed_x_if_zero = self._boxed_comp_if_body()

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(3, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(boxed_x_if_zero)(qs[0], qs[1], 0)
            qs[0], qs[2] = qm.control(boxed_x_if_zero)(qs[0], qs[2], 1)
            return qm.measure(qs)

        [true_call, false_call] = _controlled_invoke_ops(_lower(circ))
        assert true_call.definition is not None
        assert false_call.definition is not None
        true_body = true_call.definition.body
        false_body = false_call.definition.body
        assert true_body is not None
        assert false_body is not None
        assert true_body is not false_body
        assert not _find_ops(true_body.operations, IfOperation)
        assert not _find_ops(true_body.operations, CompOp)
        assert len(_find_gates(true_body.operations, GateOperationType.X)) == 1
        assert not _find_ops(false_body.operations, IfOperation)
        assert not _find_ops(false_body.operations, CompOp)
        assert not _find_gates(false_body.operations, GateOperationType.X)

        source_ops = boxed_x_if_zero.block.operations
        assert _find_ops(source_ops, IfOperation), (
            "per-call lowering must not mutate the shared source definition"
        )
        assert _find_ops(source_ops, CompOp)

    def test_multiple_parameters_keep_declaration_order(self):
        """A controlled boxed body pairs its second parameter correctly."""

        @qm.composite_gate(name="boxed_x_if_b_zero")
        def boxed_x_if_b_zero(
            q: qm.Qubit,
            a: qm.UInt,
            b: qm.UInt,
        ) -> qm.Qubit:
            """Apply X when the second compile-time parameter is zero."""
            if b == 0:
                q = qm.x(q)
            return q

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(boxed_x_if_b_zero)(
                qs[0],
                qs[1],
                1,
                0,
            )
            return qm.measure(qs)

        [invoke] = _controlled_invoke_ops(_lower(circ))
        assert invoke.body is not None
        assert not _find_ops(invoke.body.operations, IfOperation)
        assert not _find_ops(invoke.body.operations, CompOp)
        assert len(_find_gates(invoke.body.operations, GateOperationType.X)) == 1

    def test_direct_boxed_helper_inside_controlled_body_is_lowered(self):
        """Controlled decomposition descends through a direct boxed helper."""
        boxed_x_if_zero = self._boxed_comp_if_body()

        @qm.qkernel
        def wrapper(q: qm.Qubit, sel: qm.UInt) -> qm.Qubit:
            return boxed_x_if_zero(q, sel)

        @qm.qkernel
        def circ() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            qs[0] = qm.x(qs[0])
            qs[0], qs[1] = qm.control(wrapper)(qs[0], qs[1], 0)
            return qm.measure(qs)

        outer_ops = _first_controlled_block_ops(_lower(circ))
        [invoke] = _find_ops(outer_ops, InvokeOperation)
        assert invoke.transform is CallTransform.DIRECT
        assert invoke.body is not None
        assert not _find_ops(invoke.body.operations, IfOperation)
        assert not _find_ops(invoke.body.operations, CompOp)
        assert len(_find_gates(invoke.body.operations, GateOperationType.X)) == 1

    def test_controlled_implementation_body_is_lowered(self):
        """A transform-specific controlled body is folded before selection."""

        def conditional_body(name: str, include_control: bool) -> Block:
            """Build a fresh-namespace conditional implementation block.

            Args:
                name (str): Block name used for diagnostics.
                include_control (bool): Whether to include a control-qubit
                    formal before the target formal.

            Returns:
                Block: Affine block containing ``if sel == 0: X(target)``.
            """
            target = _qubit_val(f"{name}_target")
            selector = _uint_val(f"{name}_selector")
            comparison, if_op = _make_comparison_if_with_x_gate(selector, target)
            inputs = [target, selector]
            if include_control:
                inputs.insert(0, _qubit_val(f"{name}_control"))
            return Block(
                name=name,
                input_values=inputs,
                operations=[comparison, if_op],
                kind=BlockKind.AFFINE,
            )

        default_body = conditional_body("default", include_control=False)
        controlled_body = conditional_body("controlled", include_control=True)
        definition = CallableDef(
            ref=CallableRef(namespace="test", name="conditional_box"),
            body=default_body,
            implementations=[
                CallableImplementation(
                    transform=CallTransform.CONTROLLED,
                    backend="test-backend",
                    body=controlled_body,
                )
            ],
        )
        control = _qubit_val("actual_control")
        target = _qubit_val("actual_target")
        selector = _uint_val("actual_selector", const=0)
        invoke = InvokeOperation(
            operands=[control, target, selector],
            results=[control.next_version(), target.next_version()],
            transform=CallTransform.CONTROLLED,
            attrs={"num_control_qubits": 1, "num_target_qubits": 1},
            definition=definition,
        )
        outer = Block(
            name="outer",
            operations=[invoke],
            kind=BlockKind.AFFINE,
        )

        [lowered] = _controlled_invoke_ops(_run_pass(outer))
        assert lowered.definition is not None
        lowered_default = lowered.definition.body
        assert lowered_default is not None
        [lowered_impl] = lowered.definition.implementations
        assert lowered_impl.body is not None
        for body in (lowered_default, lowered_impl.body):
            assert not _find_ops(body.operations, IfOperation)
            assert not _find_ops(body.operations, CompOp)
            assert len(_find_gates(body.operations, GateOperationType.X)) == 1

    def test_recursive_callable_body_stops_at_active_boundary(self):
        """A self-referential boxed definition does not recurse forever."""
        inner_target = _qubit_val("inner_target")
        inner_selector = _uint_val("inner_selector")
        comparison, if_op = _make_comparison_if_with_x_gate(
            inner_selector,
            inner_target,
        )
        body = Block(
            name="recursive_body",
            input_values=[inner_target, inner_selector],
            operations=[comparison, if_op],
            kind=BlockKind.AFFINE,
        )
        ref = CallableRef(namespace="test", name="recursive_box")
        definition = CallableDef(ref=ref, body=body)
        recursive_call = InvokeOperation(
            operands=[inner_target, inner_selector],
            results=[inner_target.next_version()],
            definition=definition,
        )
        body.operations.append(recursive_call)

        control = _qubit_val("control")
        target = _qubit_val("target")
        selector = _uint_val("selector", const=0)
        controlled_call = InvokeOperation(
            operands=[control, target, selector],
            results=[control.next_version(), target.next_version()],
            transform=CallTransform.CONTROLLED,
            attrs={"num_control_qubits": 1, "num_target_qubits": 1},
            definition=definition,
        )
        outer = Block(
            name="outer",
            operations=[controlled_call],
            kind=BlockKind.AFFINE,
        )

        [lowered] = _controlled_invoke_ops(_run_pass(outer))

        assert lowered.body is not None
        assert not _find_ops(lowered.body.operations, IfOperation)
        assert len(_find_ops(lowered.body.operations, InvokeOperation)) == 1

    def test_inverse_block_inside_controlled_body_is_lowered(self):
        """Controlled decomposition lowers an inverse operation's owned bodies."""

        def inverse_body(name: str) -> Block:
            """Build a conditional inverse recipe block.

            Args:
                name (str): Diagnostic block name.

            Returns:
                Block: Fresh affine body with ``if selector == 0: X``.
            """
            target = _qubit_val(f"{name}_target")
            selector = _uint_val(f"{name}_selector")
            comparison, if_op = _make_comparison_if_with_x_gate(selector, target)
            return Block(
                name=name,
                input_values=[target, selector],
                operations=[comparison, if_op],
                kind=BlockKind.AFFINE,
            )

        unitary_target = _qubit_val("unitary_target")
        selector = _uint_val("actual_selector", const=0)
        inverse = InverseBlockOperation(
            operands=[unitary_target, selector],
            results=[unitary_target.next_version()],
            num_target_qubits=1,
            source_block=inverse_body("source"),
            implementation_block=inverse_body("implementation"),
        )
        unitary = Block(
            name="unitary",
            input_values=[unitary_target],
            operations=[inverse],
            kind=BlockKind.AFFINE,
        )
        control = _qubit_val("control")
        target = _qubit_val("target")
        controlled = ConcreteControlledU(
            operands=[control, target],
            results=[control.next_version(), target.next_version()],
            num_controls=1,
            block=unitary,
        )
        outer = Block(
            name="outer",
            operations=[controlled],
            kind=BlockKind.AFFINE,
        )

        [lowered_inverse] = _find_ops(
            _first_controlled_block_ops(_run_pass(outer)),
            InverseBlockOperation,
        )

        assert lowered_inverse.source_block is not None
        assert lowered_inverse.implementation_block is not None
        for body in (
            lowered_inverse.source_block,
            lowered_inverse.implementation_block,
        ):
            assert not _find_ops(body.operations, IfOperation)
            assert not _find_ops(body.operations, CompOp)
            assert len(_find_gates(body.operations, GateOperationType.X)) == 1


# ---------------------------------------------------------------------------
# Test: merge substitution into GateOperation.theta
# ---------------------------------------------------------------------------


class TestMergeSubstitutionTheta:
    """Symbolic parameter alias through compile-time if survives in gate theta."""

    @staticmethod
    def _extract_theta_const(rx_gate):
        """Extract the concrete theta value from an RX gate."""
        theta = rx_gate.theta
        if isinstance(theta, (int, float)):
            return float(theta)
        if isinstance(theta, Value):
            c = theta.get_const()
            if c is not None:
                return float(c)
            if theta.is_parameter():
                return theta.parameter_name()
        return theta

    def test_parameter_alias_true_branch(self):
        """flag=1: true branch angle (theta_a=1.5) selected in RX theta."""

        @qm.qkernel
        def kernel(
            flag: qm.UInt, theta_a: qm.Float, theta_b: qm.Float
        ) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                angle = theta_a
            else:
                angle = theta_b
            q[0] = qm.rx(q[0], angle)
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 1, "theta_a": 1.5, "theta_b": 2.5})

        assert not _find_ops(lowered.operations, IfOperation)
        rx_gates = _find_gates(lowered.operations, GateOperationType.RX)
        assert len(rx_gates) >= 1, "RX gate should be present after lowering"
        theta_val = self._extract_theta_const(rx_gates[0])
        assert theta_val == 1.5 or theta_val == "theta_a", (
            f"True branch should select theta_a (1.5), got {theta_val}"
        )

    def test_parameter_alias_false_branch(self):
        """flag=0: false branch angle (theta_b=2.5) selected in RX theta."""

        @qm.qkernel
        def kernel(
            flag: qm.UInt, theta_a: qm.Float, theta_b: qm.Float
        ) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                angle = theta_a
            else:
                angle = theta_b
            q[0] = qm.rx(q[0], angle)
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 0, "theta_a": 1.5, "theta_b": 2.5})

        assert not _find_ops(lowered.operations, IfOperation)
        rx_gates = _find_gates(lowered.operations, GateOperationType.RX)
        assert len(rx_gates) >= 1
        theta_val = self._extract_theta_const(rx_gates[0])
        assert theta_val == 2.5 or theta_val == "theta_b", (
            f"False branch should select theta_b (2.5), got {theta_val}"
        )


# ---------------------------------------------------------------------------
# Test: merge substitution into block output_values
# ---------------------------------------------------------------------------


class TestMergeSubstitutionOutputs:
    """Block output_values are updated when merge values are substituted."""

    def test_output_references_resolved_true(self):
        """flag=1: output UUIDs differ from merge output (substituted)."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered_true = _lower(kernel, bindings={"flag": 1})
        lowered_false = _lower(kernel, bindings={"flag": 0})

        assert not _find_ops(lowered_true.operations, IfOperation)
        assert not _find_ops(lowered_false.operations, IfOperation)
        assert len(lowered_true.output_values) > 0
        assert len(lowered_false.output_values) > 0
        # The two branches should produce different output value identities
        # because the true branch applies X (new qubit version) while false doesn't.
        true_uuids = {ov.uuid for ov in lowered_true.output_values}
        false_uuids = {ov.uuid for ov in lowered_false.output_values}
        assert true_uuids != false_uuids, (
            "True and false branch outputs should have different value identities"
        )

    def test_output_substitution_synthetic(self):
        """Synthetic: merge output UUID in block outputs is replaced by branch value."""
        q = _qubit_val()
        flag = _uint_val("flag", const=1)

        if_op, merge_out = _make_if_with_x_gate(flag, q)

        block = Block(
            name="test",
            operations=[if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(lowered.output_values) == 1
        # Output should NOT be the merge_out UUID (it was substituted)
        assert lowered.output_values[0].uuid != merge_out.uuid, (
            "Block output should be substituted away from merge output UUID"
        )


# ---------------------------------------------------------------------------
# Test: dead-op elimination
# ---------------------------------------------------------------------------


class TestDeadOpElimination:
    """Condition producers are removed after the if is lowered."""

    def test_unreferenced_classical_expression_chain_is_eliminated(self):
        """Pure condition arithmetic is removed when nothing else reads it."""
        one = _uint_val("one", const=1)
        two = _uint_val("two", const=2)
        total = _uint_val("total")
        condition = _bit_val("condition")
        add_op = BinOp(operands=[one, one], results=[total], kind=BinOpKind.ADD)
        comp_op = CompOp(
            operands=[total, two],
            results=[condition],
            kind=CompOpKind.EQ,
        )
        if_op = IfOperation(
            operands=[condition],
            true_operations=[],
            false_operations=[],
        )
        block = Block(
            name="test",
            operations=[add_op, comp_op, if_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        assert lowered.operations == []

    def test_output_element_index_keeps_classical_producer_alive(self):
        """A producer read only by an output element index remains live."""
        one = _uint_val("one", const=1)
        two = _uint_val("two", const=2)
        index = _uint_val("index")
        condition = _bit_val("condition")
        add_op = BinOp(operands=[one, one], results=[index], kind=BinOpKind.ADD)
        comp_op = CompOp(
            operands=[index, two],
            results=[condition],
            kind=CompOpKind.EQ,
        )
        if_op = IfOperation(
            operands=[condition],
            true_operations=[],
            false_operations=[],
        )
        length = _uint_val("length", const=4)
        array = ArrayValue(type=UIntType(), name="array", shape=(length,))
        output = Value(
            type=UIntType(),
            name="array[index]",
            parent_array=array,
            element_indices=(index,),
        )
        block = Block(
            name="test",
            input_values=[array],
            operations=[add_op, comp_op, if_op],
            output_values=[output],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        assert _find_ops(lowered.operations, BinOp) == [add_op]
        assert not _find_ops(lowered.operations, CompOp)
        assert not _find_ops(lowered.operations, IfOperation)

    def test_output_view_shape_and_slice_bound_keep_producer_alive(self):
        """Output view metadata keeps its shared dimension producer live."""
        one = _uint_val("one", const=1)
        two = _uint_val("two", const=2)
        dimension = _uint_val("dimension")
        condition = _bit_val("condition")
        add_op = BinOp(
            operands=[one, one],
            results=[dimension],
            kind=BinOpKind.ADD,
        )
        comp_op = CompOp(
            operands=[dimension, two],
            results=[condition],
            kind=CompOpKind.EQ,
        )
        if_op = IfOperation(
            operands=[condition],
            true_operations=[],
            false_operations=[],
        )
        root_length = _uint_val("root_length", const=8)
        root = ArrayValue(type=UIntType(), name="root", shape=(root_length,))
        step = _uint_val("step", const=1)
        view = ArrayValue(
            type=UIntType(),
            name="view",
            shape=(dimension,),
            slice_of=root,
            slice_start=dimension,
            slice_step=step,
        )
        block = Block(
            name="test",
            input_values=[root],
            operations=[add_op, comp_op, if_op],
            output_values=[view],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        assert _find_ops(lowered.operations, BinOp) == [add_op]
        assert not _find_ops(lowered.operations, CompOp)
        assert not _find_ops(lowered.operations, IfOperation)

    def test_losing_external_measurement_source_is_preserved(self):
        """A dead merge source never erases its destructive measurement."""
        q = _qubit_val()
        measured = _bit_val("measured")
        measure = MeasureOperation(operands=[q], results=[measured])

        condition = _bit_val("condition").with_const(True)
        selected = _bit_val("selected").with_const(False)
        merged = _bit_val("merged")
        if_op = IfOperation(
            operands=[condition],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(selected, measured, merged)

        block = Block(
            name="test",
            input_values=[q],
            operations=[measure, if_op],
            output_values=[merged],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert _find_ops(lowered.operations, MeasureOperation) == [measure]
        assert lowered.output_values == [selected]

    def test_comp_op_eliminated(self):
        """CompOp producing the if condition should be eliminated."""

        @qm.qkernel
        def kernel(flag: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag > 0:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": 1})

        assert not _find_ops(lowered.operations, IfOperation)
        # The CompOp that produced the condition should be dead-eliminated
        comp_ops = _find_ops(lowered.operations, CompOp)
        assert len(comp_ops) == 0, (
            "CompOp should be eliminated when its result is only used "
            "by the lowered IfOperation"
        )

    def test_condition_producer_removed_inside_runtime_if_branch(self):
        """Dead producers are removed inside a surviving runtime-if branch."""
        comparison, compile_time_if = _make_comparison_if_with_x_gate(
            _uint_val("selector", const=0), _qubit_val()
        )
        runtime_if = IfOperation(
            operands=[_bit_val("runtime_condition")],
            true_operations=[comparison, compile_time_if],
            false_operations=[],
        )
        block = Block(
            name="test",
            operations=[runtime_if],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        [remaining_if] = _find_ops(lowered.operations, IfOperation)
        assert not _find_ops(remaining_if.true_operations, IfOperation)
        assert not _find_ops(remaining_if.true_operations, CompOp)
        assert len(_find_gates(remaining_if.true_operations, GateOperationType.X)) == 1

    def test_condition_producer_removed_inside_while_body(self):
        """Dead producers are removed inside a surviving runtime while body."""
        comparison, compile_time_if = _make_comparison_if_with_x_gate(
            _uint_val("selector", const=0), _qubit_val()
        )
        while_op = WhileOperation(
            operands=[_bit_val("runtime_condition")],
            operations=[comparison, compile_time_if],
        )
        block = Block(
            name="test",
            operations=[while_op],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        [remaining_while] = _find_ops(lowered.operations, WhileOperation)
        assert not _find_ops(remaining_while.operations, IfOperation)
        assert not _find_ops(remaining_while.operations, CompOp)
        assert len(_find_gates(remaining_while.operations, GateOperationType.X)) == 1

    def test_runtime_if_yield_keeps_producer_alive(self):
        """A value read only as a runtime-if merge yield survives dead-op elimination.

        Lowering the compile-time if marks its condition chain dead; the
        chain reaches ``m``, whose only remaining reader is the runtime
        if's false yield. The yield must count as a use — otherwise the
        BinOp producing ``m`` is deleted and the surviving runtime if is
        left with a dangling merge source.
        """
        n = _uint_val("n", const=3)
        one = _uint_val("one", const=1)
        m = _uint_val("m")
        add_op = BinOp(operands=[n, one], results=[m], kind=BinOpKind.ADD)

        three = _uint_val("three", const=3)
        flag = _bit_val("flag")
        comp_op = CompOp(operands=[m, three], results=[flag], kind=CompOpKind.EQ)

        # Compile-time if: (3 + 1) == 3 resolves to False, so lowering
        # drops it and cascades dead-uuid marking through flag into m.
        q = _qubit_val()
        ct_if, _ = _make_if_with_x_gate(flag, q)

        # Runtime if: unresolvable Bit condition; its merge reads m as
        # the false yield — the only surviving reference to m.
        bit = _bit_val("bit")
        true_source = _uint_val("t")
        merged = _uint_val("merged")
        rt_if = IfOperation(
            operands=[bit],
            true_operations=[],
            false_operations=[],
        )
        rt_if.add_merge(true_source, m, merged)

        block = Block(
            name="test",
            operations=[add_op, comp_op, ct_if, rt_if],
            output_values=[merged],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        # The dead-op cascade ran (condition producer removed) ...
        assert not _find_ops(lowered.operations, CompOp)
        # ... but the yield-only producer must survive.
        assert len(_find_ops(lowered.operations, BinOp)) == 1, (
            "BinOp producing the runtime-if merge yield must survive "
            "dead-op elimination — the yield is its only reader"
        )
        [runtime_if] = _find_ops(lowered.operations, IfOperation)
        [merge] = runtime_if.iter_merges()
        assert merge.false_value.uuid == m.uuid

    def test_collect_used_uuids_excludes_loop_rebind_records(self):
        """Values referenced only by a loop rebind record are not liveness reads.

        The rebind record's before/after ride along ``all_input_values``
        for cloning but are not genuine reads, so a producer referenced
        solely through the record must remain eligible for elimination.
        """
        cond = _bit_val("cond")
        before = _uint_val("before")
        after = _uint_val("after")
        while_op = WhileOperation(
            operands=[cond],
            operations=[],
            loop_carried_rebinds=(
                LoopCarriedRebind(var_name="acc", before=before, after=after),
            ),
        )

        used: set[str] = set()
        CompileTimeIfLoweringPass._collect_used_uuids(while_op, used)

        assert cond.uuid in used, "the while condition is a genuine read"
        assert before.uuid not in used, "rebind-record before is not a read"
        assert after.uuid not in used, "rebind-record after is not a read"

    def test_collect_used_uuids_keeps_yield_shared_with_branch_rebind(self):
        """A value that is both a false yield and a branch-rebind before stays used.

        The canonical branch-discard shape (``if cond: q = fresh`` with no
        rebinding else) yields the pre-branch ``q`` on the false side, so
        that value is simultaneously a ``false_yields`` entry and the
        ``branch_rebinds`` before. It must stay used via the yield; only
        the record occurrence is dropped.
        """
        cond = _bit_val("cond")
        q_pre = _qubit_val("q_pre")
        fresh = _qubit_val("fresh")
        merged = _qubit_val("merged")
        if_op = IfOperation(operands=[cond], true_operations=[], false_operations=[])
        if_op.add_merge(fresh, q_pre, merged)
        if_op.branch_rebinds = (
            BranchRebind(
                var_name="q",
                before=q_pre,
                rebound_in_true=True,
                rebound_in_false=False,
            ),
        )

        used: set[str] = set()
        CompileTimeIfLoweringPass._collect_used_uuids(if_op, used)

        assert q_pre.uuid in used, "kept via the false yield despite the record"
        assert fresh.uuid in used, "the true yield is a genuine read"


# ---------------------------------------------------------------------------
# Test: runtime IfOperation preservation
# ---------------------------------------------------------------------------


class TestRuntimeIfPreservation:
    """Non-compile-time IfOperation should remain after the pass."""

    def test_measurement_dependent_if_preserved(self):
        """if based on measurement result stays as IfOperation."""

        @qm.qkernel
        def kernel() -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(2, "q")
            q[0] = qm.h(q[0])
            b = qm.measure(q[0])
            if b:
                q[1] = qm.x(q[1])
            return qm.measure(q)

        lowered = _lower(kernel)

        if_ops = _find_ops(lowered.operations, IfOperation)
        assert len(if_ops) >= 1, (
            "Measurement-dependent IfOperation should remain "
            "after compile-time if lowering"
        )


# ---------------------------------------------------------------------------
# Test: reserved measurement-result binding names
# ---------------------------------------------------------------------------


class TestMeasurementNameDoesNotCaptureBinding:
    """A binding sharing a measurement result's generated name does not capture it.

    The frontend generates measurement-result names like ``q_measured``. A
    kernel parameter can legitimately share that name (here the unused
    ``q_measured: UInt``). Resolution against ``bindings`` is keyed only on
    parameter provenance (``ScalarMetadata.parameter_name``) — never on the
    display ``Value.name`` — so binding the parameter must NOT hijack the
    runtime ``measure(q)`` condition that happens to carry the same display
    name. Before the bare-name fallback was removed, ``q_measured=1`` silently
    resolved ``if bit:`` to the true branch, pruning the runtime control flow
    (a miscompilation). This test pins the corrected behavior.
    """

    def test_binding_named_like_measurement_result_stays_runtime(self):
        """A same-named binding leaves the runtime ``measure(q)`` condition intact."""

        @qm.qkernel
        def kernel(q_measured: qm.UInt) -> qm.Bit:
            q = qm.qubit("q")
            target = qm.qubit("target")
            q = qm.h(q)
            bit = qm.measure(q)
            if bit:
                target = qm.x(target)
            return qm.measure(target)

        lowered = _lower(kernel, bindings={"q_measured": 1})

        # The IfOperation is preserved: the runtime measurement condition is
        # not captured by the same-named parameter binding.
        assert _find_ops(lowered.operations, IfOperation), (
            "A binding sharing the measurement result's generated name must "
            "not statically resolve the runtime condition"
        )
        # The X gate stays nested inside the if body, not hoisted to top level
        # by a spurious static branch selection.
        top_level_x = _find_gates(lowered.operations, GateOperationType.X)
        assert len(top_level_x) == 0, (
            "The X gate must remain inside the runtime branch, not be hoisted"
        )

    def test_inlined_callee_measurement_not_captured_by_caller_binding(self):
        """An inlined callee's measurement condition survives a same-named caller binding.

        This is the cross-inlining form of the capture bug: the callee's
        ``measure(a)`` result carries the generated name ``a_measured``, and the
        caller binds an unrelated parameter of the same name. After inlining,
        the bare-name fallback would have seeded the callee's runtime condition
        from the caller binding and pruned the branch. Provenance-only
        resolution keeps the ``if`` runtime.
        """
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def helper(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            a = qm.h(a)
            bit = qm.measure(a)
            if bit:
                b = qm.x(b)
            return b

        @qm.qkernel
        def top(a_measured: qm.UInt) -> qm.Bit:
            a = qm.qubit("a")
            b = qm.qubit("b")
            b = helper(a, b)
            return qm.measure(b)

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(top, bindings={"a_measured": 1})
        inlined = transpiler.inline(transpiler.substitute(block))
        validated = transpiler.affine_validate(inlined)
        folded = transpiler.constant_fold(validated, bindings={"a_measured": 1})
        lowered = transpiler.lower_compile_time_ifs(folded, bindings={"a_measured": 1})

        assert _find_ops(lowered.operations, IfOperation), (
            "An inlined callee's runtime measurement condition must not be "
            "captured by a caller binding that shares the generated name"
        )


# ---------------------------------------------------------------------------
# Test: CondOp (AND/OR) condition lowering (synthetic IR)
# ---------------------------------------------------------------------------


class TestCondOpLowering:
    """CondOp-derived conditions are resolved and lowered."""

    def test_cond_and_both_true(self):
        """CondOp(AND, 1, 1) → true: X gate present."""
        flag_a = _uint_val("a", const=1)
        flag_b = _uint_val("b", const=1)
        cond_result = _bit_val("cond")
        cond_op = CondOp(
            operands=[flag_a, flag_b],
            results=[cond_result],
            kind=CondOpKind.AND,
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1

    def test_cond_and_one_false(self):
        """CondOp(AND, 1, 0) → false: no X gate."""
        flag_a = _uint_val("a", const=1)
        flag_b = _uint_val("b", const=0)
        cond_result = _bit_val("cond")
        cond_op = CondOp(
            operands=[flag_a, flag_b],
            results=[cond_result],
            kind=CondOpKind.AND,
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) == 0

    def test_cond_or_one_true(self):
        """CondOp(OR, 0, 1) → true: X gate present."""
        flag_a = _uint_val("a", const=0)
        flag_b = _uint_val("b", const=1)
        cond_result = _bit_val("cond")
        cond_op = CondOp(
            operands=[flag_a, flag_b],
            results=[cond_result],
            kind=CondOpKind.OR,
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1


# ---------------------------------------------------------------------------
# Test: NotOp condition lowering (synthetic IR)
# ---------------------------------------------------------------------------


class TestNotOpLowering:
    """NotOp-derived conditions are resolved and lowered."""

    def test_not_of_true_selects_false_branch(self):
        """NotOp(1) → false: no X gate."""
        flag = _uint_val("flag", const=1)
        not_result = _bit_val("not_result")
        not_op = NotOp(
            operands=[flag],
            results=[not_result],
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(not_result, q)

        block = Block(
            name="test",
            operations=[not_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) == 0

    def test_not_of_false_selects_true_branch(self):
        """NotOp(0) → true: X gate present."""
        flag = _uint_val("flag", const=0)
        not_result = _bit_val("not_result")
        not_op = NotOp(
            operands=[flag],
            results=[not_result],
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(not_result, q)

        block = Block(
            name="test",
            operations=[not_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1


# ---------------------------------------------------------------------------
# Test: BinOp-fed condition lowering (synthetic IR)
# ---------------------------------------------------------------------------


class TestBinOpFedCondition:
    """Conditions derived from BinOp + CompOp chains are resolved."""

    def test_binop_add_then_comp_true(self):
        """(a + b) > 0 with a=2, b=3 → 5 > 0 → true: X gate present."""
        a = _uint_val("a", const=2)
        b = _uint_val("b", const=3)
        sum_result = _uint_val("sum")
        binop = BinOp(
            operands=[a, b],
            results=[sum_result],
            kind=BinOpKind.ADD,
        )
        zero = _uint_val("zero", const=0)
        cond_result = _bit_val("cond")
        comp_op = CompOp(
            operands=[sum_result, zero],
            results=[cond_result],
            kind=CompOpKind.GT,
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[binop, comp_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) >= 1

    def test_binop_sub_then_comp_false(self):
        """(a - b) > 0 with a=1, b=3 → -2 > 0 → false: no X gate."""
        a = _uint_val("a", const=1)
        b = _uint_val("b", const=3)
        diff_result = _uint_val("diff")
        binop = BinOp(
            operands=[a, b],
            results=[diff_result],
            kind=BinOpKind.SUB,
        )
        zero = _uint_val("zero", const=0)
        cond_result = _bit_val("cond")
        comp_op = CompOp(
            operands=[diff_result, zero],
            results=[cond_result],
            kind=CompOpKind.GT,
        )
        q = _qubit_val()
        if_op, merge_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[binop, comp_op, if_op],
            output_values=[merge_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) == 0


# ---------------------------------------------------------------------------
# Test: parent_array substitution (synthetic IR)
# ---------------------------------------------------------------------------


class TestParentArraySubstitution:
    """Merge slot merging ArrayValues propagates selected parent_array."""

    def test_parent_array_substituted_true_branch(self):
        """True branch ArrayValue becomes parent_array after lowering."""
        arr_true = ArrayValue(type=QubitType(), name="q_true")
        arr_false = ArrayValue(type=QubitType(), name="q_false")

        flag = _uint_val("flag", const=1)

        # Merge slot merges two array values
        merge_out_arr = ArrayValue(type=QubitType(), name="q_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(arr_true, arr_false, merge_out_arr)

        # Downstream value references the merge_out as parent_array
        idx = _uint_val("idx", const=0)
        elem = Value(
            type=QubitType(),
            name="q[0]",
            parent_array=merge_out_arr,
            element_indices=(idx,),
        )
        # Use elem as operand in a gate
        gate = GateOperation(
            operands=[elem],
            results=[elem.next_version()],
            gate_type=GateOperationType.X,
        )

        block = Block(
            name="test",
            operations=[if_op, gate],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        # The gate operand should now reference arr_true (not merge_out_arr)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 1
        operand = x_gates[0].operands[0]
        assert isinstance(operand, Value) and operand.parent_array is not None
        assert operand.parent_array.uuid == arr_true.uuid, (
            f"parent_array should be arr_true ({arr_true.uuid}), "
            f"got {operand.parent_array.uuid}"
        )


# ---------------------------------------------------------------------------
# Test: element_indices substitution (synthetic IR)
# ---------------------------------------------------------------------------


class TestElementIndicesSubstitution:
    """Merge slot merging index Values propagates selected index in element_indices."""

    def test_element_index_substituted_true_branch(self):
        """True branch index Value replaces merge index after lowering."""
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)

        flag = _uint_val("flag", const=1)

        merge_idx = _uint_val("idx_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(idx_true, idx_false, merge_idx)

        # Downstream value uses merge_idx as element_indices
        arr = ArrayValue(type=QubitType(), name="q")
        elem = Value(
            type=QubitType(),
            name="q[merge]",
            parent_array=arr,
            element_indices=(merge_idx,),
        )
        gate = GateOperation(
            operands=[elem],
            results=[elem.next_version()],
            gate_type=GateOperationType.X,
        )

        block = Block(
            name="test",
            operations=[if_op, gate],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 1
        operand = x_gates[0].operands[0]
        assert isinstance(operand, Value) and len(operand.element_indices) == 1
        assert operand.element_indices[0].uuid == idx_true.uuid, (
            f"element_indices should be idx_true ({idx_true.uuid}), "
            f"got {operand.element_indices[0].uuid}"
        )

    def test_element_index_substituted_false_branch(self):
        """False branch index Value replaces merge index when flag=0."""
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)

        flag = _uint_val("flag", const=0)

        merge_idx = _uint_val("idx_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(idx_true, idx_false, merge_idx)

        arr = ArrayValue(type=QubitType(), name="q")
        elem = Value(
            type=QubitType(),
            name="q[merge]",
            parent_array=arr,
            element_indices=(merge_idx,),
        )
        gate = GateOperation(
            operands=[elem],
            results=[elem.next_version()],
            gate_type=GateOperationType.X,
        )

        block = Block(
            name="test",
            operations=[if_op, gate],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 1
        operand = x_gates[0].operands[0]
        assert isinstance(operand, Value) and len(operand.element_indices) == 1
        assert operand.element_indices[0].uuid == idx_false.uuid, (
            f"element_indices should be idx_false ({idx_false.uuid}), "
            f"got {operand.element_indices[0].uuid}"
        )


# ---------------------------------------------------------------------------
# CC1: Nested scalar merge chain (transitive substitution)
# ---------------------------------------------------------------------------


class TestNestedScalarMergeChain:
    """Nested compile-time if: merge_outer -> merge_inner -> terminal must resolve."""

    def test_transitive_chain_resolves_to_terminal(self):
        """Gate operand UUID must equal terminal qubit UUID after 2-hop chain."""
        # Outer if: flag1=1 selects q_a
        q_a = _qubit_val("q_a")
        q_b = _qubit_val("q_b")
        flag1 = _uint_val("flag1", const=1)
        merge_inner = _qubit_val("q_merge_inner")
        if1 = IfOperation(
            operands=[flag1],
            true_operations=[],
            false_operations=[],
        )
        if1.add_merge(q_a, q_b, merge_inner)

        # Inner if: flag2=1 selects merge_inner (which should transitively be q_a)
        q_c = _qubit_val("q_c")
        flag2 = _uint_val("flag2", const=1)
        merge_outer = _qubit_val("q_merge_outer")
        if2 = IfOperation(
            operands=[flag2],
            true_operations=[],
            false_operations=[],
        )
        if2.add_merge(merge_inner, q_c, merge_outer)

        # Downstream gate uses merge_outer
        gate = GateOperation(
            operands=[merge_outer],
            results=[merge_outer.next_version()],
            gate_type=GateOperationType.X,
        )

        block = Block(
            name="test",
            operations=[if1, if2, gate],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 1
        assert x_gates[0].operands[0].uuid == q_a.uuid, (
            f"Expected terminal q_a ({q_a.uuid}), got {x_gates[0].operands[0].uuid}"
        )


# ---------------------------------------------------------------------------
# CC2: Combined parent_array + element_indices substitution
# ---------------------------------------------------------------------------


class TestCombinedArrayAndIndexSubstitution:
    """Merge slots on both parent_array and element_indices must both resolve."""

    def test_both_parent_and_index_substituted(self):
        """After lowering, both parent_array and element_indices[0] are replaced."""
        arr_true = ArrayValue(type=QubitType(), name="qa")
        arr_false = ArrayValue(type=QubitType(), name="qb")
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)
        flag = _uint_val("flag", const=1)

        # Merge slot for array
        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")

        # Merge slot for index
        merge_idx = _uint_val("idx_merge")

        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(arr_true, arr_false, merge_arr)
        if_op.add_merge(idx_true, idx_false, merge_idx)

        # Downstream element uses both merge array and merge index
        elem = Value(
            type=QubitType(),
            name="elem_merge",
            parent_array=merge_arr,
            element_indices=(merge_idx,),
        )
        gate = GateOperation(
            operands=[elem],
            results=[elem.next_version()],
            gate_type=GateOperationType.X,
        )

        block = Block(
            name="test",
            operations=[if_op, gate],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(x_gates) == 1
        operand = x_gates[0].operands[0]
        assert isinstance(operand, Value)
        assert operand.parent_array is not None
        assert operand.parent_array.uuid == arr_true.uuid, (
            f"parent_array should be arr_true ({arr_true.uuid}), "
            f"got {operand.parent_array.uuid}"
        )
        assert len(operand.element_indices) == 1
        assert operand.element_indices[0].uuid == idx_true.uuid, (
            f"element_indices[0] should be idx_true ({idx_true.uuid}), "
            f"got {operand.element_indices[0].uuid}"
        )


class TestNestedSliceMetadataSubstitution:
    """Lowered merge indices must reach nested slice-result metadata."""

    def test_slice_result_and_output_drop_nested_merge_index(self):
        """A slice bound ``bounds[phi]`` rewrites phi everywhere."""
        selected_index = _uint_val("selected", const=0)
        losing_index = _uint_val("losing", const=1)
        merge_index = _uint_val("merge_index")
        condition = _uint_val("condition", const=1)
        if_op = IfOperation(operands=[condition])
        if_op.add_merge(selected_index, losing_index, merge_index)

        two = _uint_val("two", const=2)
        four = _uint_val("four", const=4)
        bounds = ArrayValue(type=UIntType(), name="bounds", shape=(two,))
        start = Value(
            type=UIntType(),
            name="bounds[merge_index]",
            parent_array=bounds,
            element_indices=(merge_index,),
        )
        step = _uint_val("step", const=1)
        root = ArrayValue(type=QubitType(), name="q", shape=(four,))
        view = ArrayValue(
            type=QubitType(),
            name="view",
            shape=(two,),
            slice_of=root,
            slice_start=start,
            slice_step=step,
        )
        slice_op = SliceArrayOperation(
            operands=[root, start, step],
            results=[view],
        )
        block = Block(
            name="test",
            input_values=[root, bounds],
            operations=[if_op, slice_op],
            output_values=[view],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        [lowered_slice] = _find_ops(lowered.operations, SliceArrayOperation)
        operand_start = lowered_slice.operands[1]
        result_view = lowered_slice.results[0]
        output_view = lowered.output_values[0]
        assert operand_start.element_indices[0] is selected_index
        assert isinstance(result_view, ArrayValue)
        assert result_view.slice_start is not None
        assert result_view.slice_start.element_indices[0] is selected_index
        assert isinstance(output_view, ArrayValue)
        assert output_view.slice_start is not None
        assert output_view.slice_start.element_indices[0] is selected_index


# ---------------------------------------------------------------------------
# CC3: SymbolicControlledU field substitution (num_controls + control_indices)
# ---------------------------------------------------------------------------


class TestSymbolicControlledUFieldSubstitution:
    """Merge-substituted ``SymbolicControlledU`` fields must all resolve.

    The legacy ``IndexSpecControlledU`` version of this test covered
    ``target_indices`` too; that field was deleted alongside the
    index-spec API.  The redesigned ``SymbolicControlledU`` carries
    ``num_controls``, ``power``, and ``control_indices``, so this
    test pins merge resolution on the three remaining fields.
    """

    def test_three_fields_substituted(self):
        """``num_controls``, ``power``, ``control_indices`` all resolve."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.gate import (
            ControlledUOperation,
            SymbolicControlledU,
        )
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import ArrayValue

        flag = _uint_val("flag", const=1)

        # True branch values.  Both branches use ``num_controls=1`` to
        # keep the constructed ``SymbolicControlledU`` well-formed
        # against its ``control_indices=(ci_merge,)`` (length 1)
        # field — the API contract is
        # ``len(control_indices) == num_controls`` and the emit
        # pass enforces it.  The two branches are still
        # distinguishable via their UUIDs (which is what the merge
        # substitution assertions below check).
        nc_true = _uint_val("nc_true", const=1)
        power_true = _uint_val("power_true", const=4)
        ci_true = _uint_val("ci_true", const=1)

        # False branch values
        nc_false = _uint_val("nc_false", const=1)
        power_false = _uint_val("power_false", const=1)
        ci_false = _uint_val("ci_false", const=3)

        # Merge outputs
        nc_merge = _uint_val("nc_merge")
        power_merge = _uint_val("power_merge")
        ci_merge = _uint_val("ci_merge")

        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(nc_true, nc_false, nc_merge)
        if_op.add_merge(power_true, power_false, power_merge)
        if_op.add_merge(ci_true, ci_false, ci_merge)

        # SymbolicControlledU: operands = [pool ArrayValue, target Value]
        unitary_block = Block(name="U")
        pool_shape = (_uint_val("pool_len", const=4),)
        pool_av = ArrayValue(type=QubitType(), name="pool", shape=pool_shape)
        target_q = _qubit_val("target")
        ctrl_u = SymbolicControlledU(
            operands=[pool_av, target_q],
            results=[pool_av.next_version(), target_q.next_version()],
            num_controls=nc_merge,
            control_indices=(ci_merge,),
            power=power_merge,
            block=unitary_block,
        )

        block = Block(
            name="test",
            operations=[if_op, ctrl_u],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        ctrl_u_ops = [
            op for op in lowered.operations if isinstance(op, ControlledUOperation)
        ]
        assert len(ctrl_u_ops) == 1
        op = ctrl_u_ops[0]
        assert isinstance(op, SymbolicControlledU)

        assert isinstance(op.num_controls, Value)
        assert op.num_controls.uuid == nc_true.uuid
        assert isinstance(op.power, Value)
        assert op.power.uuid == power_true.uuid
        assert op.control_indices is not None and len(op.control_indices) == 1
        assert op.control_indices[0].uuid == ci_true.uuid


# ---------------------------------------------------------------------------
# CC4: ForItemsOperation body substitution
# ---------------------------------------------------------------------------


class TestForItemsOperationBodySubstitution:
    """Merge output used in ForItemsOperation body must resolve after lowering."""

    def test_shape_less_vector_key_keeps_one_entry_loop_boxed(self):
        """A vector key without a dimension identity is never flattened."""
        from qamomile.circuit.ir.operation.control_flow import ForItemsOperation
        from qamomile.circuit.ir.value import DictValue

        key = ArrayValue(type=UIntType(), name="key", shape=())
        item_value = _float_val("value")
        q = _qubit_val()
        q_after = q.next_version()
        body_gate = GateOperation(
            operands=[q],
            results=[q_after],
            gate_type=GateOperationType.X,
        )
        iterable = DictValue(name="data").with_dict_runtime_metadata({(1, 2): 0.5})
        for_items = ForItemsOperation(
            operands=[iterable],
            results=[],
            key_vars=["key"],
            key_is_vector=True,
            key_var_values=(key,),
            value_var="value",
            value_var_value=item_value,
            operations=[body_gate],
        )
        block = Block(
            name="test",
            input_values=[q],
            operations=[for_items],
            output_values=[q_after],
            kind=BlockKind.AFFINE,
        )

        first = _run_pass(block)
        second = _run_pass(block)

        assert len(first.operations) == 1
        assert isinstance(first.operations[0], ForItemsOperation)
        assert first.operations[0].key_var_values == (key,)
        assert first.operations[0].operations == [body_gate]
        assert first == second

    def test_body_operand_substituted(self):
        """Body operand UUID equals terminal value UUID after lowering."""
        from qamomile.circuit.ir.operation.control_flow import ForItemsOperation
        from qamomile.circuit.ir.value import DictValue

        q_true = _qubit_val("q_true")
        q_false = _qubit_val("q_false")
        flag = _uint_val("flag", const=1)

        merge_q = _qubit_val("q_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(q_true, q_false, merge_q)

        # ForItemsOperation body uses merge_q
        body_gate = GateOperation(
            operands=[merge_q],
            results=[merge_q.next_version()],
            gate_type=GateOperationType.X,
        )
        dict_val = DictValue(name="data")
        for_items = ForItemsOperation(
            operands=[dict_val],
            results=[],
            key_vars=["k"],
            value_var="v",
            operations=[body_gate],
        )

        block = Block(
            name="test",
            operations=[if_op, for_items],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        for_items_ops = [
            op for op in lowered.operations if isinstance(op, ForItemsOperation)
        ]
        assert len(for_items_ops) == 1
        body_ops = for_items_ops[0].operations
        x_gates = _find_gates(body_ops, GateOperationType.X)
        assert len(x_gates) == 1
        assert x_gates[0].operands[0].uuid == q_true.uuid, (
            f"Body operand should be q_true ({q_true.uuid}), "
            f"got {x_gates[0].operands[0].uuid}"
        )


# ---------------------------------------------------------------------------
# CC5: Cast source provenance sync after compile-time if
# ---------------------------------------------------------------------------


class TestCastSourceProvenanceSync:
    """cast_source_uuid/cast_source_logical_id must sync with selected branch."""

    def test_cast_source_uuid_synced_after_lowering(self):
        """After lowering, CastOperation result params match selected array."""
        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.types.q_register import QFixedType

        arr_true = ArrayValue(type=QubitType(), name="qa")
        arr_false = ArrayValue(type=QubitType(), name="qb")
        flag = _uint_val("flag", const=1)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(arr_true, arr_false, merge_arr)

        # CastOperation with stale provenance from merge
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                qubit_logical_ids=[
                    f"{merge_arr.logical_id}_0",
                    f"{merge_arr.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
        )

        block = Block(
            name="test",
            operations=[if_op, cast_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        cast_ops = [op for op in lowered.operations if isinstance(op, CastOperation)]
        assert len(cast_ops) == 1
        result = cast_ops[0].results[0]
        assert result.get_cast_source_uuid() == arr_true.uuid, (
            f"cast_source_uuid should be arr_true ({arr_true.uuid}), "
            f"got {result.get_cast_source_uuid()}"
        )
        assert result.get_cast_source_logical_id() == arr_true.logical_id
        # Carrier keys should also be rebuilt
        assert list(result.get_cast_qubit_uuids() or ()) == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]
        assert cast_ops[0].qubit_mapping == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]

    def test_cast_selected_slice_rebuilds_root_space_carriers(self):
        """Selected slice views keep source provenance but use root carriers."""
        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.types.q_register import QFixedType

        root_true = ArrayValue(
            type=QubitType(),
            name="qa",
            shape=(_uint_val("root_len", const=4),),
        )
        view_true = ArrayValue(
            type=QubitType(),
            name="qa_view",
            shape=(_uint_val("view_len", const=2),),
            slice_of=root_true,
            slice_start=_uint_val("start", const=1),
            slice_step=_uint_val("step", const=2),
        )
        arr_false = ArrayValue(type=QubitType(), name="qb")
        flag = _uint_val("flag", const=1)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(view_true, arr_false, merge_arr)

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                qubit_logical_ids=[
                    f"{merge_arr.logical_id}_0",
                    f"{merge_arr.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
        )
        block = Block(
            name="test",
            operations=[if_op, cast_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        cast_ops = [op for op in lowered.operations if isinstance(op, CastOperation)]
        assert len(cast_ops) == 1
        result = cast_ops[0].results[0]
        assert result.get_cast_source_uuid() == view_true.uuid
        assert result.get_cast_source_logical_id() == view_true.logical_id
        assert result.get_cast_qubit_uuids() == (
            f"{root_true.uuid}_1",
            f"{root_true.uuid}_3",
        )
        assert result.get_cast_qubit_logical_ids() == (
            f"{root_true.logical_id}_1",
            f"{root_true.logical_id}_3",
        )
        assert result.get_qfixed_qubit_uuids() == (
            f"{root_true.uuid}_1",
            f"{root_true.uuid}_3",
        )
        assert cast_ops[0].qubit_mapping == [
            f"{root_true.uuid}_1",
            f"{root_true.uuid}_3",
        ]

    def test_cast_false_branch_slice_rebuilds_root_space_carriers(self):
        """False-branch selection replaces stale trace-time true-branch keys.

        The frontend bakes the TRUE branch's root-space carrier keys into the
        cast metadata at trace time. When lowering selects the FALSE branch,
        the carriers must be rebuilt for the actually-selected view instead
        of keeping the true-branch indices.
        """
        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.types.q_register import QFixedType

        root = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(_uint_val("root_len", const=4),),
        )
        view_true = ArrayValue(
            type=QubitType(),
            name="q_odd",
            shape=(_uint_val("vt_len", const=2),),
            slice_of=root,
            slice_start=_uint_val("vt_start", const=1),
            slice_step=_uint_val("vt_step", const=2),
        )
        view_false = ArrayValue(
            type=QubitType(),
            name="q_even",
            shape=(_uint_val("vf_len", const=2),),
            slice_of=root,
            slice_start=_uint_val("vf_start", const=0),
            slice_step=_uint_val("vf_step", const=2),
        )
        flag = _uint_val("flag", const=0)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(view_true, view_false, merge_arr)

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        # Trace-time metadata: source points at the merge output, carriers
        # carry the TRUE branch's root-space indices (q_1, q_3).
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{root.uuid}_1", f"{root.uuid}_3"],
                qubit_logical_ids=[
                    f"{root.logical_id}_1",
                    f"{root.logical_id}_3",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{root.uuid}_1", f"{root.uuid}_3"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{root.uuid}_1", f"{root.uuid}_3"],
        )
        block = Block(
            name="test",
            operations=[if_op, cast_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        cast_ops = [op for op in lowered.operations if isinstance(op, CastOperation)]
        assert len(cast_ops) == 1
        result = cast_ops[0].results[0]
        assert result.get_cast_source_uuid() == view_false.uuid
        assert result.get_cast_qubit_uuids() == (
            f"{root.uuid}_0",
            f"{root.uuid}_2",
        )
        assert result.get_qfixed_qubit_uuids() == (
            f"{root.uuid}_0",
            f"{root.uuid}_2",
        )
        assert cast_ops[0].qubit_mapping == [
            f"{root.uuid}_0",
            f"{root.uuid}_2",
        ]

    def test_cast_rebuilt_carriers_propagate_to_downstream_measure(self):
        """Downstream MeasureQFixed operands pick up rebuilt carrier keys.

        Plan-time lowering reads carrier keys from the measure operand's own
        metadata, not from the CastOperation result, so the rebuilt result
        must be registered in the substitution map for downstream consumers.
        """
        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.operation.gate import MeasureQFixedOperation
        from qamomile.circuit.ir.types.q_register import QFixedType

        root = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(_uint_val("root_len", const=4),),
        )
        view_true = ArrayValue(
            type=QubitType(),
            name="q_odd",
            shape=(_uint_val("vt_len", const=2),),
            slice_of=root,
            slice_start=_uint_val("vt_start", const=1),
            slice_step=_uint_val("vt_step", const=2),
        )
        view_false = ArrayValue(
            type=QubitType(),
            name="q_even",
            shape=(_uint_val("vf_len", const=2),),
            slice_of=root,
            slice_start=_uint_val("vf_start", const=0),
            slice_step=_uint_val("vf_step", const=2),
        )
        flag = _uint_val("flag", const=0)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(view_true, view_false, merge_arr)

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{root.uuid}_1", f"{root.uuid}_3"],
                qubit_logical_ids=[
                    f"{root.logical_id}_1",
                    f"{root.logical_id}_3",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{root.uuid}_1", f"{root.uuid}_3"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{root.uuid}_1", f"{root.uuid}_3"],
        )
        float_out = Value(type=FloatType(), name="f")
        measure_op = MeasureQFixedOperation(
            operands=[cast_result],
            results=[float_out],
            num_bits=2,
            int_bits=0,
        )
        block = Block(
            name="test",
            operations=[if_op, cast_op, measure_op],
            output_values=[float_out],
            kind=BlockKind.AFFINE,
        )

        lowered = _run_pass(block)

        measure_ops = [
            op for op in lowered.operations if isinstance(op, MeasureQFixedOperation)
        ]
        assert len(measure_ops) == 1
        measured_qfixed = measure_ops[0].operands[0]
        assert measured_qfixed.get_cast_qubit_uuids() == (
            f"{root.uuid}_0",
            f"{root.uuid}_2",
        )
        assert measured_qfixed.get_qfixed_qubit_uuids() == (
            f"{root.uuid}_0",
            f"{root.uuid}_2",
        )

    def test_cast_selected_symbolic_slice_view_raises(self):
        """A selected slice view with symbolic bounds fails fast, not silently.

        When the chosen branch is a strided view whose ``slice_start`` /
        ``slice_step`` are not compile-time constants, the root index space
        cannot be resolved, so the carriers cannot be rebuilt in root space.
        Emitting view-local keys would silently drop the measurement, so the
        pass must raise instead.
        """
        import pytest

        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.types.q_register import QFixedType
        from qamomile.circuit.transpiler.errors import ValidationError

        root = ArrayValue(
            type=QubitType(),
            name="q",
            shape=(_uint_val("root_len", const=4),),
        )
        # Symbolic slice_start (no const) makes the root index unresolvable.
        view = ArrayValue(
            type=QubitType(),
            name="q_view",
            shape=(_uint_val("view_len", const=2),),
            slice_of=root,
            slice_start=_uint_val("start"),
            slice_step=_uint_val("step", const=2),
        )
        arr_false = ArrayValue(type=QubitType(), name="qb")
        flag = _uint_val("flag", const=1)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(view, arr_false, merge_arr)

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                qubit_logical_ids=[
                    f"{merge_arr.logical_id}_0",
                    f"{merge_arr.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
        )
        block = Block(
            name="test",
            operations=[if_op, cast_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )

        with pytest.raises(ValidationError, match="symbolic slice"):
            _run_pass(block)


# ---------------------------------------------------------------------------
# CC6: Cast provenance through separate() (frontend integration)
# ---------------------------------------------------------------------------


class TestCastProvenanceThroughSeparate:
    """cast_source_* must persist correctly through separate() after lowering."""

    def test_separate_preserves_synced_provenance(self):
        """CastOperation result provenance survives through separate()."""
        from qamomile.circuit.ir.operation.cast import CastOperation
        from qamomile.circuit.ir.types.q_register import QFixedType

        # Same setup as CC5, but run through separate()
        arr_true = ArrayValue(type=QubitType(), name="qa")
        arr_false = ArrayValue(type=QubitType(), name="qb")
        flag = _uint_val("flag", const=1)

        merge_arr = ArrayValue(type=QubitType(), name="arr_merge")
        if_op = IfOperation(
            operands=[flag],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(arr_true, arr_false, merge_arr)

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=merge_arr.uuid,
                source_logical_id=merge_arr.logical_id,
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                qubit_logical_ids=[
                    f"{merge_arr.logical_id}_0",
                    f"{merge_arr.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        cast_op = CastOperation(
            operands=[merge_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{merge_arr.uuid}_0", f"{merge_arr.uuid}_1"],
        )

        block = Block(
            name="test",
            operations=[if_op, cast_op],
            output_values=[],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        # Verify lowering synced the provenance
        cast_ops = [op for op in lowered.operations if isinstance(op, CastOperation)]
        assert len(cast_ops) == 1
        result = cast_ops[0].results[0]
        assert result.get_cast_source_uuid() == arr_true.uuid
        assert list(result.get_qfixed_qubit_uuids()) == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]


# ---------------------------------------------------------------------------
# Test: bool parameter binding
# ---------------------------------------------------------------------------


class TestBoolBinding:
    """Compile-time binding of ``bool`` parameters folds `if flag:` branches."""

    def test_bool_true_selects_true_branch(self):
        """flag=True: IfOperation removed, H gate from true branch present."""

        @qm.qkernel
        def kernel(flag: bool) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.h(q[0])
            else:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": True})

        assert not _find_ops(lowered.operations, IfOperation), (
            "IfOperation should be removed after lowering"
        )
        h_gates = _find_gates(lowered.operations, GateOperationType.H)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(h_gates) >= 1, "H gate from true branch should be present"
        assert len(x_gates) == 0, "X gate from false branch must be dropped"

    def test_bool_false_selects_false_branch(self):
        """flag=False: IfOperation removed, X gate from false branch present."""

        @qm.qkernel
        def kernel(flag: bool) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.h(q[0])
            else:
                q[0] = qm.x(q[0])
            return qm.measure(q)

        lowered = _lower(kernel, bindings={"flag": False})

        assert not _find_ops(lowered.operations, IfOperation)
        h_gates = _find_gates(lowered.operations, GateOperationType.H)
        x_gates = _find_gates(lowered.operations, GateOperationType.X)
        assert len(h_gates) == 0, "H gate from true branch must be dropped"
        assert len(x_gates) >= 1, "X gate from false branch should be present"

    def test_bool_build_only_no_bindings_error(self):
        """``kernel.build(flag=True)`` must accept a Python bool without error."""

        @qm.qkernel
        def kernel(flag: bool) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(1, "q")
            if flag:
                q[0] = qm.h(q[0])
            return qm.measure(q)

        kernel.build(flag=True)
        kernel.build(flag=False)


# ---------------------------------------------------------------------------
# End-to-end emission tests for compile-time if folded inside unrolled loops
# ---------------------------------------------------------------------------


class TestCompileTimeIfInsideLoops:
    """End-to-end checks that ``if`` conditions derived from loop-iterated
    classical values are folded at emit time when the iterable is bound.

    These exercise the emit-time evaluation of ``CompOp``/``CondOp``/``NotOp``
    in ``StandardEmitPass._emit_operations``: ``emit_for_unrolled`` and
    ``emit_for_items`` bind the loop variable in ``loop_bindings`` by name,
    and then the predicate that branches on it must be evaluated so that
    ``resolve_if_condition`` can fold the surrounding ``IfOperation``.

    Without this, a kernel like ``for _, e in qmc.items(d): if e == K: ...``
    would fail at emit time with ``Runtime if-conditions must come from
    measurement results``.
    """

    def test_if_on_items_loop_value(self):
        """``if etype == K`` inside a ``qmc.items`` loop should fold per
        iteration based on the dict's bound values."""
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def kernel(errors: qm.Dict[qm.UInt, qm.UInt]) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(3, name="q")
            for idx, etype in qm.items(errors):
                if etype == 1:
                    q[idx] = qm.x(q[idx])
                elif etype == 2:
                    q[idx] = qm.y(q[idx])
                elif etype == 3:
                    q[idx] = qm.z(q[idx])
            return qm.measure(q)

        tp = QiskitTranspiler()
        bindings = {"errors": {0: 1, 1: 3, 2: 2}}
        exe = tp.transpile(kernel, bindings=bindings)
        qc = exe.compiled_quantum[0].circuit

        gate_names = [inst.operation.name for inst in qc.data]
        # Each error type should match its qubit position deterministically.
        assert gate_names.count("x") == 1
        assert gate_names.count("y") == 1
        assert gate_names.count("z") == 1
        # No surviving runtime if/else
        assert "if_else" not in gate_names

    def test_if_on_range_loop_value(self):
        """``if i == k`` inside a ``qmc.range`` loop should fold to the
        appropriate gate per iteration."""
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def kernel(n: qm.UInt) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                if i == 1:
                    q[i] = qm.x(q[i])
                else:
                    q[i] = qm.h(q[i])
            return qm.measure(q)

        tp = QiskitTranspiler()
        exe = tp.transpile(kernel, bindings={"n": 3})
        qc = exe.compiled_quantum[0].circuit

        gate_names = [inst.operation.name for inst in qc.data]
        # i=0 → H, i=1 → X, i=2 → H
        assert gate_names.count("h") == 2
        assert gate_names.count("x") == 1

    def test_inequality_predicate_in_items_loop(self):
        """An ``!=`` comparison on a loop value should also fold."""
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def kernel(spec: qm.Dict[qm.UInt, qm.UInt]) -> qm.Vector[qm.Bit]:
            q = qm.qubit_array(3, name="q")
            for idx, flag in qm.items(spec):
                if flag != 0:
                    q[idx] = qm.x(q[idx])
            return qm.measure(q)

        tp = QiskitTranspiler()
        exe = tp.transpile(kernel, bindings={"spec": {0: 1, 1: 0, 2: 5}})
        qc = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in qc.data]
        # Only entries with non-zero flag (idx 0 and 2) emit X.
        assert gate_names.count("x") == 2

    def test_runtime_if_on_measurement_still_works(self):
        """A runtime if backed by a measurement must NOT be folded — the
        emit-time predicate evaluation must skip when operands are unbound."""
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def kernel() -> qm.Bit:
            q0 = qm.qubit("q0")
            q1 = qm.qubit("q1")
            q0 = qm.x(q0)
            bit = qm.measure(q0)
            if bit:
                q1 = qm.x(q1)
            return qm.measure(q1)

        tp = QiskitTranspiler()
        # Should compile without raising — the runtime if-test is preserved.
        exe = tp.transpile(kernel)
        assert exe is not None


class TestEvaluateClassicalPredicate:
    """Unit tests for the emit-time predicate evaluator.

    The frontend currently produces only ``CompOp``, but ``CondOp`` and
    ``NotOp`` are part of the IR contract — emitted by other passes and
    reserved for future ``&`` / ``|`` / ``~`` handle overloads.
    ``compile_time_if_lowering`` and ``classical_executor`` already handle
    all three; the emit pass must too, otherwise a partially-resolvable
    ``CondOp`` / ``NotOp`` reaching emit would fail to fold.
    """

    def _bindings_after_eval(self, op):
        """Run evaluate_classical_predicate against a stub emit_pass and
        return the populated bindings dict."""
        from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
            evaluate_classical_predicate,
        )
        from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
            ValueResolver as EmitResolver,
        )

        class _StubEmitPass:
            def __init__(self):
                self._resolver = EmitResolver(parameters=set())

        bindings = {op.operands[0].uuid: 5}
        if len(op.operands) > 1 and hasattr(op.operands[1], "name"):
            bindings[op.operands[1].uuid] = 3
        evaluate_classical_predicate(_StubEmitPass(), op, bindings)
        return bindings

    def test_compop_lt_folds(self):
        lhs = Value(type=UIntType(), name="lhs")
        rhs = Value(type=UIntType(), name="rhs")
        out = Value(type=BitType(), name="out")
        op = CompOp(operands=[lhs, rhs], results=[out], kind=CompOpKind.LT)
        bindings = self._bindings_after_eval(op)
        # 5 < 3 is False
        assert bindings[out.uuid] is False
        # Name-keyed write is intentionally NOT performed — tmp values share
        # generic names ("bit_tmp" etc.) which would collide across tmps.
        assert out.name not in bindings

    def test_condop_and_folds(self):
        lhs = Value(type=BitType(), name="lhs")
        rhs = Value(type=BitType(), name="rhs")
        out = Value(type=BitType(), name="out")
        op = CondOp(operands=[lhs, rhs], results=[out], kind=CondOpKind.AND)
        bindings = self._bindings_after_eval(op)
        # bool(5 and 3) is True
        assert bindings[out.uuid] is True

    def test_condop_or_folds(self):
        lhs = Value(type=BitType(), name="lhs")
        rhs = Value(type=BitType(), name="rhs")
        out = Value(type=BitType(), name="out")
        op = CondOp(operands=[lhs, rhs], results=[out], kind=CondOpKind.OR)
        bindings = self._bindings_after_eval(op)
        assert bindings[out.uuid] is True

    def test_notop_folds(self):
        operand = Value(type=BitType(), name="lhs")
        out = Value(type=BitType(), name="out")
        op = NotOp(operands=[operand], results=[out])
        bindings = self._bindings_after_eval(op)
        # not bool(5) is False
        assert bindings[out.uuid] is False

    def test_unresolved_operand_is_noop(self):
        """If an operand is not in bindings, evaluate is a no-op so the
        downstream IfOperation falls through to its runtime path."""
        lhs = Value(type=UIntType(), name="lhs")
        rhs = Value(type=UIntType(), name="rhs")
        out = Value(type=BitType(), name="out")
        op = CompOp(operands=[lhs, rhs], results=[out], kind=CompOpKind.EQ)

        from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
            evaluate_classical_predicate,
        )
        from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
            ValueResolver as EmitResolver,
        )

        class _StubEmitPass:
            def __init__(self):
                self._resolver = EmitResolver(parameters=set())

        bindings: dict = {}  # neither lhs nor rhs bound
        evaluate_classical_predicate(_StubEmitPass(), op, bindings)
        assert out.uuid not in bindings
        assert out.name not in bindings


# ---------------------------------------------------------------------------
# Fix B regression tests: name-keyed writes for tmp values must be skipped.
# ---------------------------------------------------------------------------


class TestNoNameCollisionInTmpWrites:
    """Auto-generated tmp values (``"bit_tmp"``, ``"uint_tmp"``, etc.) must
    not be written to ``bindings`` by name. ``evaluate_binop`` /
    ``evaluate_classical_predicate`` previously did so, causing chained
    expressions with multiple tmps of the same name to overwrite each other
    and mis-resolve. The fix writes by UUID only.

    Manifesting bug: a runtime ``(~a) & (~b) & c`` Steane decoder branch
    folded the inner ``(~a) & (~b)`` to literal ``true`` because the second
    NotOp's bool result clobbered the first NotOp's expr value under the
    shared name ``"bit_tmp"``.
    """

    def test_chained_runtime_predicate_emits_full_expr(self):
        """End-to-end: ``(~a) & (~b) & c`` over three measurement bits emits
        a compound classical-expression condition, not a folded literal."""
        from qamomile.qiskit import QiskitTranspiler

        @qm.qkernel
        def kernel() -> qm.Bit:
            q0 = qm.qubit("q0")
            q1 = qm.qubit("q1")
            q2 = qm.qubit("q2")
            q3 = qm.qubit("q3")
            # Measurements with a, b → 0 and c → 1.
            a = qm.measure(q0)
            b = qm.measure(q1)
            q2 = qm.x(q2)
            c = qm.measure(q2)
            if (~a) & (~b) & c:
                q3 = qm.x(q3)
            return qm.measure(q3)

        from qiskit.circuit.classical import expr
        from qiskit.circuit.controlflow import IfElseOp

        tp = QiskitTranspiler()
        exe = tp.transpile(kernel)
        qc = exe.compiled_quantum[0].circuit
        if_ops = [i.operation for i in qc.data if isinstance(i.operation, IfElseOp)]
        assert if_ops, "Expected an IfElseOp in the runtime circuit"
        condition = if_ops[0].condition
        # The whole compound expression must be present — not a literal True.
        assert isinstance(condition, expr.Expr)
        # And it must reference the measurement clbits (not be a constant).
        identifiers = list(expr.iter_identifiers(condition))
        # The compound contains both the c bit and the negated a, b bits.
        assert len(identifiers) >= 3, (
            f"Expected ≥3 clbit refs in compound condition; got {len(identifiers)}. "
            f"Likely the inner (~a) & (~b) collapsed due to a name-collision."
        )

    def test_evaluate_binop_writes_only_by_uuid(self):
        """White-box: a successful BinOp evaluation must NOT write
        ``bindings[output.name]`` — only ``bindings[output.uuid]``."""
        from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
        from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
            evaluate_binop,
        )
        from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
            ValueResolver as EmitResolver,
        )

        lhs = Value(type=UIntType(), name="lhs")
        rhs = Value(type=UIntType(), name="rhs")
        out = Value(type=UIntType(), name="uint_tmp")  # generic tmp name
        op = BinOp(operands=[lhs, rhs], results=[out], kind=BinOpKind.ADD)

        class _StubEmitPass:
            def __init__(self):
                self._resolver = EmitResolver(parameters=set())

        bindings = {lhs.uuid: 5, rhs.uuid: 3}
        evaluate_binop(_StubEmitPass(), op, bindings)
        assert bindings[out.uuid] == 8
        # The tmp name "uint_tmp" must NOT have been written — that would
        # collide with any other UInt tmp produced earlier in the program.
        assert out.name not in bindings

    def test_evaluate_classical_predicate_writes_only_by_uuid(self):
        """White-box analog for predicate evaluation."""
        from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
            evaluate_classical_predicate,
        )
        from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
            ValueResolver as EmitResolver,
        )

        lhs = Value(type=UIntType(), name="lhs")
        rhs = Value(type=UIntType(), name="rhs")
        out = Value(type=BitType(), name="bit_tmp")
        op = CompOp(operands=[lhs, rhs], results=[out], kind=CompOpKind.LT)

        class _StubEmitPass:
            def __init__(self):
                self._resolver = EmitResolver(parameters=set())

        bindings = {lhs.uuid: 5, rhs.uuid: 3}
        evaluate_classical_predicate(_StubEmitPass(), op, bindings)
        assert bindings[out.uuid] is False
        assert out.name not in bindings


# ---------------------------------------------------------------------------
# Block-kind acceptance (TRACED is accepted for the circuit drawer)
# ---------------------------------------------------------------------------


class TestBlockKindAcceptance:
    """The pass accepts TRACED/AFFINE/HIERARCHICAL and rejects ANALYZED."""

    def test_traced_block_is_lowered(self):
        """A TRACED block (the drawer's input kind) lowers and keeps its kind."""
        q = _qubit_val()
        flag = _uint_val("flag", const=1)
        if_op, merge_out = _make_if_with_x_gate(flag, q)
        block = Block(
            name="traced",
            operations=[if_op],
            output_values=[merge_out],
            kind=BlockKind.TRACED,
        )

        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert _find_gates(lowered.operations, GateOperationType.X)
        # Kind is a normalization-neutral attribute: it must survive lowering.
        assert lowered.kind == BlockKind.TRACED

    def test_analyzed_block_is_rejected(self):
        """ANALYZED is rejected: the pass must run before dependency analysis."""
        import pytest

        from qamomile.circuit.transpiler.errors import ValidationError

        q = _qubit_val()
        flag = _uint_val("flag", const=1)
        if_op, merge_out = _make_if_with_x_gate(flag, q)
        block = Block(
            name="analyzed",
            operations=[if_op],
            output_values=[merge_out],
            kind=BlockKind.ANALYZED,
        )

        with pytest.raises(ValidationError, match="TRACED, AFFINE, or"):
            _run_pass(block)
