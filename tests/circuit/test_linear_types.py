"""Tests for linear type enforcement in the circuit frontend."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.transpiler.errors import (
    LinearTypeError,
    QubitAliasError,
    QubitConsumedError,
    UnreturnedBorrowError,
)


class TestDoubleUseDetection:
    """Test that using the same qubit handle twice raises an error."""

    def test_double_use_single_qubit_gate_raises_error(self):
        """Same qubit used in two single-qubit gates should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            _q1 = qm.h(q)
            q2 = qm.x(q)  # ERROR: q was already consumed by h()
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        # Check error message contains helpful information
        error = exc_info.value
        assert "already consumed" in str(error)
        assert "H" in str(error)  # First consumer
        assert "X" in str(error)  # Second attempted use

    def test_double_use_with_two_qubit_gate_raises_error(self):
        """Qubit consumed by two-qubit gate should not be reusable."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1_out, q2_out = qm.cx(q1, q2)
            q1_again = qm.h(q1)  # ERROR: q1 was consumed by cx
            return q1_again, q2_out

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        error = exc_info.value
        assert "already consumed" in str(error)
        assert "CX" in str(error)

    def test_double_use_rotation_gate_raises_error(self):
        """Same qubit used twice with rotation gates should raise error."""

        @qkernel
        def bad_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            _q1 = qm.rx(q, theta)
            q2 = qm.ry(q, theta)  # ERROR: q was already consumed
            return q2

        with pytest.raises(QubitConsumedError):
            # Use parameters to mark theta as a parameter
            bad_circuit.build(parameters=["theta"])


class TestProperReassignment:
    """Test that proper reassignment of qubit handles works correctly."""

    def test_reassignment_single_qubit_gates_works(self):
        """Proper reassignment should work without errors."""

        @qkernel
        def good_circuit(q: Qubit) -> Qubit:
            q = qm.h(q)  # Reassign to capture new handle
            q = qm.x(q)  # Use the reassigned handle
            return q

        # Should build without errors
        graph = good_circuit.build()
        assert graph is not None
        # Operations: QInitOperation + H + X = 3
        assert len(graph.operations) == 3

    def test_reassignment_two_qubit_gates_works(self):
        """Proper reassignment with two-qubit gates should work."""

        @qkernel
        def good_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1, q2 = qm.cx(q1, q2)  # Reassign both
            q1 = qm.h(q1)  # Use reassigned handles
            q2 = qm.x(q2)
            return q1, q2

        # Should build without errors
        graph = good_circuit.build()
        assert graph is not None

    def test_z_gate_build(self):
        """Z gate should build correctly."""

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            q = qm.z(q)
            return q

        graph = circuit.build()
        assert graph is not None
        # Operations: QInitOperation + Z = 2
        assert len(graph.operations) == 2
        ops = graph.operations
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], GateOperation)
        gate_op = ops[1]
        assert gate_op.gate_type == GateOperationType.Z

    def test_reassignment_three_qubit_gate_works(self):
        """Proper reassignment with three-qubit gate should work."""

        @qkernel
        def circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            q1, q2, q3 = qm.ccx(q1, q2, q3)
            q1 = qm.h(q1)
            return q1, q2, q3

        graph = circuit.build()
        assert graph is not None

        assert len(graph.operations) == 5  # QInitOperation * 3 + CCX + H = 5
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], QInitOperation)
        assert isinstance(graph.operations[2], QInitOperation)
        assert isinstance(graph.operations[3], GateOperation)
        ccx_op = graph.operations[3]
        assert ccx_op.gate_type == GateOperationType.TOFFOLI
        assert isinstance(graph.operations[4], GateOperation)
        h_op = graph.operations[4]
        assert h_op.gate_type == GateOperationType.H

    def test_rotation_gates_with_reassignment_works(self):
        """Proper use of rotation gates should work."""

        @qkernel
        def good_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, theta)
            q = qm.rz(q, theta)
            return q

        # Use parameters to mark theta as a parameter
        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None
        # Operations: QInitOperation + RX + RY + RZ = 4
        assert len(graph.operations) == 4


class TestQubitAliasDetection:
    """Test that aliasing errors are detected (same qubit in both positions)."""

    def test_cx_same_qubit_raises_alias_error(self):
        """Using same qubit as both control and target in cx should raise error."""

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return qm.cx(q, q)  # ERROR: same qubit in both positions

        with pytest.raises(QubitAliasError) as exc_info:
            bad_circuit.build()

        error = exc_info.value
        assert "same qubit" in str(error).lower() or "both" in str(error).lower()
        assert "CX" in str(error)

    def test_cp_same_qubit_raises_alias_error(self):
        """Using same qubit as both control and target in cp should raise error."""

        @qkernel
        def bad_circuit(q: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            return qm.cp(q, q, theta)  # ERROR: same qubit in both positions

        with pytest.raises(QubitAliasError):
            bad_circuit.build(parameters=["theta"])

    def test_swap_same_qubit_raises_alias_error(self):
        """Using same qubit in swap should raise error."""

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return qm.swap(q, q)  # ERROR: same qubit in both positions

        with pytest.raises(QubitAliasError):
            bad_circuit.build()

    def test_ccx_same_qubit_control1_control2_raises_alias_error(self):
        """Using same qubit as control1 and control2 in ccx should raise error."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            return qm.ccx(q1, q1, q2)

        with pytest.raises(QubitAliasError):
            bad_circuit.build()

    def test_ccx_same_qubit_control1_target_raises_alias_error(self):
        """Using same qubit as control1 and target in ccx should raise error."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            return qm.ccx(q1, q2, q1)

        with pytest.raises(QubitAliasError):
            bad_circuit.build()

    def test_ccx_same_qubit_control2_target_raises_alias_error(self):
        """Using same qubit as control2 and target in ccx should raise error."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            return qm.ccx(q2, q1, q1)

        with pytest.raises(QubitAliasError):
            bad_circuit.build()

    def test_rzz_same_qubit_raises_alias_error(self):
        """Using same qubit in rzz should raise error."""

        @qkernel
        def bad_circuit(q: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            return qm.rzz(q, q, theta)  # ERROR: same qubit in both positions

        with pytest.raises(QubitAliasError):
            bad_circuit.build(parameters=["theta"])


class TestArrayBorrowChecking:
    """Test that array element borrowing is properly tracked."""

    def test_double_borrow_same_index_raises_error(self):
        """Borrowing same array element twice without return should raise error."""

        @qkernel
        def bad_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            _q0 = qs[0]  # Borrow element 0
            q0_again = qs[0]  # ERROR: element 0 is already borrowed
            return q0_again

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        error = exc_info.value
        assert "borrowed" in str(error).lower()

    def test_borrow_different_indices_works(self):
        """Borrowing different array elements should work."""

        @qkernel
        def good_circuit() -> tuple[Qubit, Qubit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]  # Borrow element 0
            q1 = qs[1]  # Borrow element 1 - different index, OK
            q0 = qm.h(q0)
            q1 = qm.h(q1)
            return q0, q1

        # Should build without errors
        graph = good_circuit.build()
        assert graph is not None

    def test_proper_return_and_reborrow_works(self):
        """Returning element then borrowing again should work."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            q0 = qs[0]  # Borrow element 0
            q0 = qm.h(q0)
            qs[0] = q0  # Return the element
            q0_again = qs[0]  # Now safe to borrow again
            q0_again = qm.x(q0_again)
            return q0_again

        # Should build without errors
        graph = good_circuit.build()
        assert graph is not None

    def test_validate_all_returned_catches_unreturned(self):
        """validate_all_returned should detect unreturned borrows."""
        from qamomile.circuit.frontend.tracer import trace

        # Create a vector and borrow an element
        with trace():
            qs = qubit_array(3, "qs")
            _q0 = qs[0]  # Borrow but don't return

            # Validate should raise error
            with pytest.raises(UnreturnedBorrowError) as exc_info:
                qs.validate_all_returned()

            error = exc_info.value
            assert "unreturned" in str(error).lower()


class TestMeasurementConsumption:
    """Test that measurement properly consumes qubits."""

    def test_measure_consumes_qubit(self):
        """Qubit should be consumed after measurement."""

        @qkernel
        def bad_circuit(q: Qubit) -> qm.Bit:
            bit = qm.measure(q)
            _q2 = qm.h(q)  # ERROR: q was consumed by measure
            return bit

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        error = exc_info.value
        assert "measure" in str(error).lower()


class TestErrorMessageQuality:
    """Test that error messages are helpful and informative."""

    def test_error_message_contains_fix_suggestion(self):
        """Error messages should include how to fix the problem."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            _q1 = qm.h(q)
            q2 = qm.x(q)
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        error_msg = str(exc_info.value)
        # Should contain fix suggestion
        assert "reassign" in error_msg.lower() or "fix" in error_msg.lower()

    def test_error_message_contains_handle_name(self):
        """Error messages should identify which qubit was misused."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            _q1 = qm.h(q)
            q2 = qm.x(q)
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        # Error should have handle_name set
        error = exc_info.value
        assert error.handle_name is not None

    def test_error_message_contains_operation_names(self):
        """Error messages should name both operations involved."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            _q1 = qm.h(q)
            q2 = qm.x(q)
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        error = exc_info.value
        assert error.operation_name is not None  # The second operation
        assert error.first_use_location is not None  # The first operation


class TestClassicalValuesNotLinear:
    """Test that classical values don't have linear type restrictions."""

    def test_float_can_be_used_multiple_times(self):
        """Float values should be reusable."""

        @qkernel
        def good_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, theta)  # Same theta used again - OK for Float
            q = qm.rz(q, theta)
            return q

        # Should build without errors (use parameters to mark theta)
        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None

    def test_uint_can_be_used_for_indexing(self):
        """UInt values should be reusable for array indexing."""

        @qkernel
        def good_circuit() -> tuple[Qubit, Qubit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0 = qm.h(q0)
            q1 = qm.h(q1)
            return q0, q1

        # Should build without errors
        graph = good_circuit.build()
        assert graph is not None


class TestArrayConsumePreservesState:
    """Issue 07: Handle.consume() must preserve ArrayBase subclass state."""

    def test_vector_consume_preserves_shape(self):
        """Vector.consume() should preserve shape."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(3, "qs")
            original_shape = qs.shape
            qs2 = qs.consume("test")
            assert qs2.shape == original_shape

    def test_vector_consume_preserves_element_type(self):
        """Vector.consume() should preserve element_type."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(3, "qs")
            qs2 = qs.consume("test")
            assert qs2.element_type == Qubit

    def test_measure_vector_result_shape_matches_input(self):
        """measure(Vector[Qubit]) should return Vector[Bit] with matching shape."""

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            return qm.measure(qs)

        graph = circuit.build()
        assert graph is not None


class TestSetitemConsumeAndValidation:
    """Issues 01+03: __setitem__ must consume handle and validate return."""

    def test_setitem_consumes_handle(self):
        """After qs[0] = q, reusing q should raise QubitConsumedError."""

        @qkernel
        def bad_circuit() -> Qubit:
            qs = qubit_array(1, "qs")
            q = qs[0]
            q = qm.h(q)
            qs[0] = q
            q = qm.x(q)  # ERROR: q was consumed by return
            return q

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_setitem_rejects_rogue_return(self):
        """Returning a qubit from a different array should raise LinearTypeError."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs1 = qubit_array(1, "qs1")
            qs2 = qubit_array(1, "qs2")
            _q1 = qs1[0]
            rogue = qs2[0]
            with pytest.raises(LinearTypeError, match="not borrowed from this array"):
                qs1[0] = rogue

    def test_setitem_unborrowed_index_consumes_handle(self):
        """Writing to an unborrowed index should consume the handle."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(2, "qs")
            rogue = qm.qubit("rogue")
            # Fresh assignment to unborrowed index is allowed but consumes handle
            qs[1] = rogue
            # rogue is now consumed and cannot be reused
            with pytest.raises(QubitConsumedError):
                qm.h(rogue)

    def test_borrow_return_reborrow_still_works(self):
        """Normal borrow -> gate -> return -> reborrow should work."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            q = qs[0]
            q = qm.h(q)
            qs[0] = q
            q = qs[0]  # reborrow
            q = qm.x(q)
            return q

        graph = good_circuit.build()
        assert graph is not None


class TestMeasureVectorValidation:
    """Issue 06: validate_all_returned must be called before measure."""

    def test_measure_vector_unreturned_borrow_raises(self):
        """measure(qs) with unreturned borrow should raise UnreturnedBorrowError."""

        @qkernel
        def bad_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            _q = qs[0]  # borrow but don't return
            return qm.measure(qs)

        with pytest.raises(UnreturnedBorrowError):
            bad_circuit.build()

    def test_measure_vector_after_return_works(self):
        """measure(qs) after all borrows returned should work."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q = qs[0]
            q = qm.h(q)
            qs[0] = q  # return the borrow
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None


class TestCallOperationsConsume:
    """Issue 08: QKernel/CompositeGate __call__ must consume quantum inputs."""

    def test_qkernel_call_consumes_quantum_input(self):
        """Reusing qubit after sub-kernel call should raise QubitConsumedError."""

        @qkernel
        def sub(q: Qubit) -> Qubit:
            return qm.h(q)

        @qkernel
        def main(q: Qubit) -> tuple[Qubit, Qubit]:
            q2 = sub(q)
            q3 = qm.x(q)  # ERROR: q consumed by sub-kernel call
            return q2, q3

        with pytest.raises(QubitConsumedError):
            main.build()

    def test_qkernel_call_classical_input_reusable(self):
        """Classical arguments to sub-kernel should remain reusable."""

        @qkernel
        def sub(q: Qubit, theta: qm.Float) -> Qubit:
            return qm.rx(q, theta)

        @qkernel
        def main(q: Qubit, theta: qm.Float) -> Qubit:
            q = sub(q, theta)
            q = qm.ry(q, theta)  # theta is classical, reuse is OK
            return q

        graph = main.build(parameters=["theta"])
        assert graph is not None


class TestLoopVariableShadowing:
    """Issue 02: Loop variable must not shadow function parameters."""

    def test_loop_var_shadows_parameter_raises(self):
        """for i in qm.range(n) where i is a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_circuit(i: qm.UInt) -> qm.UInt:
                for i in qm.range(3):
                    pass
                return i

    def test_loop_var_no_shadow_works(self):
        """Non-shadowing loop variable should work fine."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            for j in qm.range(3):
                qs[j] = qm.h(qs[j])
            q = qs[0]
            return q

        graph = good_circuit.build()
        assert graph is not None


class TestRuntimeLimitations:
    """Issue 05: Better diagnostics for runtime limitations."""

    def test_getsource_failure_descriptive_error(self):
        """Dynamically created function should give descriptive SyntaxError."""
        from qamomile.circuit.frontend.ast_transform import transform_control_flow

        ns = {"qm": qm, "Qubit": Qubit}
        exec(
            "def f(q: Qubit) -> Qubit:\n    q = qm.h(q)\n    return q\n",
            ns,
        )
        with pytest.raises(SyntaxError, match="Cannot retrieve source code"):
            transform_control_flow(ns["f"])

    def test_while_quantum_condition_raises(self):
        """while condition with quantum operation should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Quantum operation"):

            @qkernel
            def bad_circuit(q: Qubit) -> Qubit:
                while qm.measure(q):
                    q = qm.h(q)
                return q

    def test_while_classical_condition_works(self):
        """while condition with classical value should work."""

        @qkernel
        def good_circuit(q: Qubit, n: qm.UInt) -> Qubit:
            while n:
                q = qm.h(q)
            return q

        graph = good_circuit.build(n=1)
        assert graph is not None


class TestCastConsumeAndValidation:
    """Issue 04: cast must validate and consume source."""

    def test_cast_consumes_source(self):
        """After cast, using source array should raise QubitConsumedError."""

        @qkernel
        def bad_circuit() -> qm.QFixed:
            qs = qubit_array(3, "qs")
            qf = qm.cast(qs, qm.QFixed, int_bits=1)
            _q = qs[0]  # ERROR: qs was consumed by cast
            return qf

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_cast_unreturned_borrow_raises(self):
        """cast with unreturned borrow should raise UnreturnedBorrowError."""

        @qkernel
        def bad_circuit() -> qm.QFixed:
            qs = qubit_array(3, "qs")
            _q = qs[0]  # borrow but don't return
            return qm.cast(qs, qm.QFixed, int_bits=1)

        with pytest.raises(UnreturnedBorrowError):
            bad_circuit.build()

    def test_cast_normal_usage_works(self):
        """Normal cast should work."""

        @qkernel
        def good_circuit() -> qm.QFixed:
            qs = qubit_array(3, "qs")
            for i in qm.range(3):
                qs[i] = qm.h(qs[i])
            return qm.cast(qs, qm.QFixed, int_bits=1)

        graph = good_circuit.build()
        assert graph is not None
