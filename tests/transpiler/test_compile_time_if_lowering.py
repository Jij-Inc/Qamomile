"""Direct tests for CompileTimeIfLoweringPass.

Tests the pass-internal contracts that are not observable from backend
circuit success alone: exact Block rewrites, phi substitution, recursive
lowering, dead-op elimination, and runtime IfOperation preservation.
"""

import qamomile.circuit as qm
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    PhiOp,
)
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
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
    params = {"const": const} if const is not None else {}
    return Value(type=UIntType(), name=name, params=params)


def _bit_val(name):
    return Value(type=BitType(), name=name)


def _float_val(name, *, const=None):
    params = {"const": const} if const is not None else {}
    return Value(type=FloatType(), name=name, params=params)


def _qubit_val(name="q"):
    return Value(type=QubitType(), name=name)


def _run_pass(block, bindings=None):
    return CompileTimeIfLoweringPass(bindings=bindings or {}).run(block)


def _make_if_with_x_gate(condition_val, qubit_in):
    """Build IfOperation(condition, [X gate], []) with PhiOp for qubit."""
    q_true = qubit_in.next_version()
    x_gate = GateOperation(
        operands=[qubit_in],
        results=[q_true],
        gate_type=GateOperationType.X,
    )
    q_false = qubit_in
    phi_out = qubit_in.next_version()
    phi = PhiOp(
        operands=[condition_val, q_true, q_false],
        results=[phi_out],
    )
    if_op = IfOperation(
        operands=[condition_val],
        results=[phi_out],
        true_operations=[x_gate],
        false_operations=[],
        phi_ops=[phi],
    )
    return if_op, phi_out


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
# Test: phi substitution into GateOperation.theta
# ---------------------------------------------------------------------------


class TestPhiSubstitutionTheta:
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
# Test: phi substitution into block output_values
# ---------------------------------------------------------------------------


class TestPhiSubstitutionOutputs:
    """Block output_values are updated when phi values are substituted."""

    def test_output_references_resolved_true(self):
        """flag=1: output UUIDs differ from phi output (substituted)."""

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
        """Synthetic: phi output UUID in block outputs is replaced by branch value."""
        q = _qubit_val()
        flag = _uint_val("flag", const=1)

        if_op, phi_out = _make_if_with_x_gate(flag, q)

        block = Block(
            name="test",
            operations=[if_op],
            output_values=[phi_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(lowered.output_values) == 1
        # Output should NOT be the phi_out UUID (it was substituted)
        assert lowered.output_values[0].uuid != phi_out.uuid, (
            "Block output should be substituted away from phi output UUID"
        )


# ---------------------------------------------------------------------------
# Test: dead-op elimination
# ---------------------------------------------------------------------------


class TestDeadOpElimination:
    """Condition producers are removed after the if is lowered."""

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
        if_op, phi_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[cond_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(not_result, q)

        block = Block(
            name="test",
            operations=[not_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(not_result, q)

        block = Block(
            name="test",
            operations=[not_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[binop, comp_op, if_op],
            output_values=[phi_out],
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
        if_op, phi_out = _make_if_with_x_gate(cond_result, q)

        block = Block(
            name="test",
            operations=[binop, comp_op, if_op],
            output_values=[phi_out],
            kind=BlockKind.AFFINE,
        )
        lowered = _run_pass(block)

        assert not _find_ops(lowered.operations, IfOperation)
        assert len(_find_gates(lowered.operations, GateOperationType.X)) == 0


# ---------------------------------------------------------------------------
# Test: parent_array substitution (synthetic IR)
# ---------------------------------------------------------------------------


class TestParentArraySubstitution:
    """PhiOp merging ArrayValues propagates selected parent_array."""

    def test_parent_array_substituted_true_branch(self):
        """True branch ArrayValue becomes parent_array after lowering."""
        arr_true = ArrayValue(type=QubitType(), name="q_true")
        arr_false = ArrayValue(type=QubitType(), name="q_false")

        flag = _uint_val("flag", const=1)

        # PhiOp merges two array values
        phi_out_arr = ArrayValue(type=QubitType(), name="q_phi")
        phi = PhiOp(
            operands=[flag, arr_true, arr_false],
            results=[phi_out_arr],
        )
        if_op = IfOperation(
            operands=[flag],
            results=[phi_out_arr],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        # Downstream value references the phi_out as parent_array
        idx = _uint_val("idx", const=0)
        elem = Value(
            type=QubitType(),
            name="q[0]",
            parent_array=phi_out_arr,
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
        # The gate operand should now reference arr_true (not phi_out_arr)
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
    """PhiOp merging index Values propagates selected index in element_indices."""

    def test_element_index_substituted_true_branch(self):
        """True branch index Value replaces phi index after lowering."""
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)

        flag = _uint_val("flag", const=1)

        phi_idx = _uint_val("idx_phi")
        phi = PhiOp(
            operands=[flag, idx_true, idx_false],
            results=[phi_idx],
        )
        if_op = IfOperation(
            operands=[flag],
            results=[phi_idx],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        # Downstream value uses phi_idx as element_indices
        arr = ArrayValue(type=QubitType(), name="q")
        elem = Value(
            type=QubitType(),
            name="q[phi]",
            parent_array=arr,
            element_indices=(phi_idx,),
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
        """False branch index Value replaces phi index when flag=0."""
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)

        flag = _uint_val("flag", const=0)

        phi_idx = _uint_val("idx_phi")
        phi = PhiOp(
            operands=[flag, idx_true, idx_false],
            results=[phi_idx],
        )
        if_op = IfOperation(
            operands=[flag],
            results=[phi_idx],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        arr = ArrayValue(type=QubitType(), name="q")
        elem = Value(
            type=QubitType(),
            name="q[phi]",
            parent_array=arr,
            element_indices=(phi_idx,),
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
# CC1: Nested scalar phi chain (transitive substitution)
# ---------------------------------------------------------------------------


class TestNestedScalarPhiChain:
    """Nested compile-time if: phi_outer -> phi_inner -> terminal must resolve."""

    def test_transitive_chain_resolves_to_terminal(self):
        """Gate operand UUID must equal terminal qubit UUID after 2-hop chain."""
        # Outer if: flag1=1 selects q_a
        q_a = _qubit_val("q_a")
        q_b = _qubit_val("q_b")
        flag1 = _uint_val("flag1", const=1)
        phi_inner = _qubit_val("q_phi_inner")
        phi1 = PhiOp(operands=[flag1, q_a, q_b], results=[phi_inner])
        if1 = IfOperation(
            operands=[flag1],
            results=[phi_inner],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi1],
        )

        # Inner if: flag2=1 selects phi_inner (which should transitively be q_a)
        q_c = _qubit_val("q_c")
        flag2 = _uint_val("flag2", const=1)
        phi_outer = _qubit_val("q_phi_outer")
        phi2 = PhiOp(operands=[flag2, phi_inner, q_c], results=[phi_outer])
        if2 = IfOperation(
            operands=[flag2],
            results=[phi_outer],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi2],
        )

        # Downstream gate uses phi_outer
        gate = GateOperation(
            operands=[phi_outer],
            results=[phi_outer.next_version()],
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
    """PhiOp on both parent_array and element_indices must both resolve."""

    def test_both_parent_and_index_substituted(self):
        """After lowering, both parent_array and element_indices[0] are replaced."""
        arr_true = ArrayValue(type=QubitType(), name="qa")
        arr_false = ArrayValue(type=QubitType(), name="qb")
        idx_true = _uint_val("idx_true", const=0)
        idx_false = _uint_val("idx_false", const=1)
        flag = _uint_val("flag", const=1)

        # PhiOp for array
        phi_arr = ArrayValue(type=QubitType(), name="arr_phi")
        phi_arr_op = PhiOp(operands=[flag, arr_true, arr_false], results=[phi_arr])

        # PhiOp for index
        phi_idx = _uint_val("idx_phi")
        phi_idx_op = PhiOp(operands=[flag, idx_true, idx_false], results=[phi_idx])

        if_op = IfOperation(
            operands=[flag],
            results=[phi_arr, phi_idx],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi_arr_op, phi_idx_op],
        )

        # Downstream element uses both phi array and phi index
        elem = Value(
            type=QubitType(),
            name="elem_phi",
            parent_array=phi_arr,
            element_indices=(phi_idx,),
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


# ---------------------------------------------------------------------------
# CC3: ControlledUOperation 4-field substitution
# ---------------------------------------------------------------------------


class TestControlledUOperationFieldSubstitution:
    """Phi-substituted fields on ControlledUOperation must all resolve."""

    def test_four_fields_substituted(self):
        """num_controls, power, target_indices, controlled_indices all resolve."""
        from qamomile.circuit.ir.block_value import BlockValue
        from qamomile.circuit.ir.operation.gate import ControlledUOperation
        from qamomile.circuit.ir.types.primitives import BlockType

        flag = _uint_val("flag", const=1)

        # True branch values
        nc_true = _uint_val("nc_true", const=2)
        power_true = _uint_val("power_true", const=4)
        ti_true = _uint_val("ti_true", const=0)
        ci_true = _uint_val("ci_true", const=1)

        # False branch values
        nc_false = _uint_val("nc_false", const=1)
        power_false = _uint_val("power_false", const=1)
        ti_false = _uint_val("ti_false", const=2)
        ci_false = _uint_val("ci_false", const=3)

        # Phis
        nc_phi = _uint_val("nc_phi")
        power_phi = _uint_val("power_phi")
        ti_phi = _uint_val("ti_phi")
        ci_phi = _uint_val("ci_phi")

        phi_nc = PhiOp(operands=[flag, nc_true, nc_false], results=[nc_phi])
        phi_power = PhiOp(operands=[flag, power_true, power_false], results=[power_phi])
        phi_ti = PhiOp(operands=[flag, ti_true, ti_false], results=[ti_phi])
        phi_ci = PhiOp(operands=[flag, ci_true, ci_false], results=[ci_phi])

        if_op = IfOperation(
            operands=[flag],
            results=[nc_phi, power_phi, ti_phi, ci_phi],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi_nc, phi_power, phi_ti, phi_ci],
        )

        # ControlledUOperation with phi fields
        block_val = BlockValue(type=BlockType(), name="U")
        ctrl_q = _qubit_val("ctrl")
        target_q = _qubit_val("target")
        ctrl_u = ControlledUOperation(
            operands=[block_val, ctrl_q, target_q],
            results=[ctrl_q.next_version(), target_q.next_version()],
            num_controls=nc_phi,
            power=power_phi,
            target_indices=[ti_phi],
            controlled_indices=[ci_phi],
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

        assert isinstance(op.num_controls, Value)
        assert op.num_controls.uuid == nc_true.uuid
        assert isinstance(op.power, Value)
        assert op.power.uuid == power_true.uuid
        assert op.target_indices is not None and len(op.target_indices) == 1
        assert op.target_indices[0].uuid == ti_true.uuid
        assert op.controlled_indices is not None and len(op.controlled_indices) == 1
        assert op.controlled_indices[0].uuid == ci_true.uuid


# ---------------------------------------------------------------------------
# CC4: ForItemsOperation body substitution
# ---------------------------------------------------------------------------


class TestForItemsOperationBodySubstitution:
    """Phi output used in ForItemsOperation body must resolve after lowering."""

    def test_body_operand_substituted(self):
        """Body operand UUID equals terminal value UUID after lowering."""
        from qamomile.circuit.ir.operation.control_flow import ForItemsOperation
        from qamomile.circuit.ir.value import DictValue

        q_true = _qubit_val("q_true")
        q_false = _qubit_val("q_false")
        flag = _uint_val("flag", const=1)

        phi_q = _qubit_val("q_phi")
        phi = PhiOp(operands=[flag, q_true, q_false], results=[phi_q])
        if_op = IfOperation(
            operands=[flag],
            results=[phi_q],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        # ForItemsOperation body uses phi_q
        body_gate = GateOperation(
            operands=[phi_q],
            results=[phi_q.next_version()],
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

        phi_arr = ArrayValue(type=QubitType(), name="arr_phi")
        phi = PhiOp(operands=[flag, arr_true, arr_false], results=[phi_arr])
        if_op = IfOperation(
            operands=[flag],
            results=[phi_arr],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        # CastOperation with stale provenance from phi
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = Value(
            type=result_type,
            name="qf",
            params={
                "cast_source_uuid": phi_arr.uuid,
                "cast_source_logical_id": phi_arr.logical_id,
                "cast_qubit_uuids": [f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
                "cast_qubit_logical_ids": [
                    f"{phi_arr.logical_id}_0",
                    f"{phi_arr.logical_id}_1",
                ],
                "num_bits": 2,
                "int_bits": 0,
                "qubit_values": [f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
            },
        )
        cast_op = CastOperation(
            operands=[phi_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
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
        assert result.params["cast_source_uuid"] == arr_true.uuid, (
            f"cast_source_uuid should be arr_true ({arr_true.uuid}), "
            f"got {result.params['cast_source_uuid']}"
        )
        assert result.params["cast_source_logical_id"] == arr_true.logical_id
        # Carrier keys should also be rebuilt
        assert result.params["cast_qubit_uuids"] == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]
        assert cast_ops[0].qubit_mapping == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]


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

        phi_arr = ArrayValue(type=QubitType(), name="arr_phi")
        phi = PhiOp(operands=[flag, arr_true, arr_false], results=[phi_arr])
        if_op = IfOperation(
            operands=[flag],
            results=[phi_arr],
            true_operations=[],
            false_operations=[],
            phi_ops=[phi],
        )

        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = Value(
            type=result_type,
            name="qf",
            params={
                "cast_source_uuid": phi_arr.uuid,
                "cast_source_logical_id": phi_arr.logical_id,
                "cast_qubit_uuids": [f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
                "cast_qubit_logical_ids": [
                    f"{phi_arr.logical_id}_0",
                    f"{phi_arr.logical_id}_1",
                ],
                "num_bits": 2,
                "int_bits": 0,
                "qubit_values": [f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
            },
        )
        cast_op = CastOperation(
            operands=[phi_arr],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{phi_arr.uuid}_0", f"{phi_arr.uuid}_1"],
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
        assert result.params["cast_source_uuid"] == arr_true.uuid
        assert result.params["qubit_values"] == [
            f"{arr_true.uuid}_0",
            f"{arr_true.uuid}_1",
        ]
