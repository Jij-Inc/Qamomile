"""Tests for linear type enforcement in the circuit frontend."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    GateOperationType,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.stdlib.qft import iqft, qft
from qamomile.circuit.stdlib.qpe import qpe
from qamomile.circuit.transpiler.errors import (
    LinearTypeError,
    QubitAliasError,
    QubitConsumedError,
    QubitRebindError,
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
    """Handle.consume() must preserve ArrayBase subclass state."""

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
        ops = graph.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], MeasureVectorOperation)


class TestSetitemConsumeAndValidation:
    """__setitem__ must consume handle and validate return."""

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

    def test_setitem_after_measure_raises_consumed_error(self):
        """Writing to measured array should raise QubitConsumedError."""

        @qkernel
        def bad_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(1, "qs")
            bits = qm.measure(qs)
            qs[0] = qm.qubit("fresh")  # ERROR: qs consumed by measure
            return bits

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_setitem_on_moved_array_raises_consumed_error(self):
        """Writing to moved array handle should raise QubitConsumedError."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(1, "qs")
            _moved = qs.consume("move")
            with pytest.raises(QubitConsumedError):
                qs[0] = qm.qubit("fresh")

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
    """validate_all_returned must be called before measure."""

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
    """QKernel/CompositeGate __call__ must consume quantum inputs."""

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

    def test_composite_gate_call_consumes_target_qubit(self):
        """Reusing qubit after CompositeGate call should raise QubitConsumedError."""
        from qamomile.circuit.frontend.composite_gate import CompositeGate

        class SimpleH(CompositeGate):
            custom_name = "simple_h"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                (q,) = qubits
                return (qm.h(q),)

        gate = SimpleH()

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            (q2,) = gate(q)
            q3 = qm.x(q)  # ERROR: q consumed by gate call
            return q2, q3

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_composite_gate_call_consumes_control_qubit(self):
        """Reusing control qubit after CompositeGate call should raise QubitConsumedError."""
        from qamomile.circuit.frontend.composite_gate import CompositeGate

        class ControlledH(CompositeGate):
            custom_name = "controlled_h"
            num_control_qubits = 1

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                (q,) = qubits
                return (qm.h(q),)

        gate = ControlledH()

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, tgt_out = gate(tgt, controls=(ctrl,))
            ctrl_reuse = qm.x(ctrl)  # ERROR: ctrl consumed by gate call
            return ctrl_out, tgt_out, ctrl_reuse

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()


class TestCastConsumeAndValidation:
    """cast must validate and consume source."""

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

    def test_setitem_after_cast_raises_consumed_error(self):
        """Writing to source array after cast should raise QubitConsumedError."""

        @qkernel
        def bad_circuit() -> qm.QFixed:
            qs = qubit_array(3, "qs")
            qf = qm.cast(qs, qm.QFixed, int_bits=1)
            qs[0] = qm.qubit("fresh")  # ERROR: qs consumed by cast
            return qf

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()


class TestIterationProhibition:
    """Verify direct iteration over Vector is prohibited at both AST and runtime layers."""

    def test_for_sequence_raises_syntax_error(self):
        """'for q in qs:' in @qkernel should raise SyntaxError at AST stage."""
        with pytest.raises(SyntaxError, match="Direct iteration over sequences"):

            @qkernel
            def bad_circuit() -> Qubit:
                qs = qubit_array(3, "qs")
                for q in qs:
                    q = qm.h(q)
                return qs[0]

    def test_vector_iter_raises_type_error(self):
        """iter(Vector) should raise TypeError at runtime."""
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.tracer import trace
        from qamomile.circuit.ir.types import QubitType
        from qamomile.circuit.ir.value import Value

        with trace():
            v = Vector._create_from_value(
                value=Value(type=QubitType(), name="qs"),
                shape=(),
            )
            with pytest.raises(TypeError, match="Direct iteration over Vector"):
                iter(v)

    def test_range_iteration_allowed(self):
        """'for i in qmc.range(n):' should be allowed."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            for i in qm.range(3):
                qs[i] = qm.h(qs[i])
            return qs[0]

        graph = good_circuit.build()
        assert graph is not None


class TestComputedIndexBorrowReturn:
    """Computed symbolic indices (e.g. i+1) must not cause false UnreturnedBorrowError."""

    def test_computed_index_plus_one_borrow_return(self):
        """GHZ pattern: q[i], q[i+1] = qm.cx(q[i], q[i+1]) should succeed."""

        @qkernel
        def ghz(n: int) -> qm.Vector[qm.Bit]:
            qubits = qubit_array(n, "q")
            qubits[0] = qm.h(qubits[0])
            for i in qm.range(n - 1):
                qubits[i], qubits[i + 1] = qm.cx(qubits[i], qubits[i + 1])
            return qm.measure(qubits)

        graph = ghz.build(n=3)
        assert graph is not None

    def test_genuine_unreturned_borrow_still_detected(self):
        """Borrowing without return must still raise UnreturnedBorrowError."""

        @qkernel
        def bad_circuit(n: int) -> qm.Vector[qm.Bit]:
            qubits = qubit_array(n, "q")
            for i in qm.range(n - 1):
                q = qubits[i]
                q = qm.h(q)
                # Missing: qubits[i] = q
            return qm.measure(qubits)

        with pytest.raises(UnreturnedBorrowError):
            bad_circuit.build(n=3)


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

_SINGLE_QUBIT_GATES = [
    ("h", qm.h),
    ("x", qm.x),
    ("y", qm.y),
    ("z", qm.z),
    ("t", qm.t),
    ("s", qm.s),
    ("sdg", qm.sdg),
    ("tdg", qm.tdg),
]

_ROTATION_GATES = [
    ("rx", qm.rx),
    ("ry", qm.ry),
    ("rz", qm.rz),
    ("p", qm.p),
]

_TWO_QUBIT_GATES_NO_PARAM = [
    ("cx", qm.cx),
    ("cz", qm.cz),
    ("swap", qm.swap),
]

_TWO_QUBIT_GATES_WITH_PARAM = [
    ("cp", qm.cp),
    ("rzz", qm.rzz),
]


class TestAllSingleQubitGatesDoubleUse:
    """Every single-qubit gate must enforce linearity (double-use → error, reassign → OK)."""

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_double_use_raises(self, name, gate):
        """Using original qubit handle after gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            _q1 = gate(q)
            q2 = qm.h(q)  # q already consumed
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build()

        assert name.upper() in str(exc_info.value)

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_reassign_works(self, name, gate):
        """Proper reassignment after single-qubit gate should build without errors."""

        @qkernel
        def good_circuit(q: Qubit) -> Qubit:
            q = gate(q)
            return q

        graph = good_circuit.build()
        assert graph is not None


class TestAllRotationGatesDoubleUse:
    """Every rotation gate must enforce linearity."""

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_double_use_raises(self, name, gate):
        """Using original qubit handle after rotation gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            _q1 = gate(q, theta)
            q2 = qm.h(q)  # q already consumed
            return q2

        with pytest.raises(QubitConsumedError) as exc_info:
            bad_circuit.build(parameters=["theta"])

        assert name.upper() in str(exc_info.value)

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_reassign_works(self, name, gate):
        """Proper reassignment after rotation gate should build without errors."""

        @qkernel
        def good_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            q = gate(q, theta)
            return q

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_float_angle_reusable(self, name, gate):
        """Float angle should be reusable across multiple rotation gates."""

        @qkernel
        def good_circuit(q: Qubit, theta: qm.Float) -> Qubit:
            q = gate(q, theta)
            q = qm.rx(q, theta)  # reuse theta — OK for Float
            return q

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None


class TestAllTwoQubitGatesDoubleUse:
    """Every two-qubit gate must consume both qubits."""

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_reuse_first_qubit_raises(self, name, gate):
        """Reusing first qubit after two-qubit gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            _q1_out, q2_out = gate(q1, q2)
            q1_bad = qm.h(q1)  # q1 already consumed
            return q1_bad, q2_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_reuse_second_qubit_raises(self, name, gate):
        """Reusing second qubit after two-qubit gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1_out, _q2_out = gate(q1, q2)
            q2_bad = qm.h(q2)  # q2 already consumed
            return q1_out, q2_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_reassign_both_works(self, name, gate):
        """Proper reassignment of both qubits after two-qubit gate should succeed."""

        @qkernel
        def good_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1, q2 = gate(q1, q2)
            return q1, q2

        graph = good_circuit.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_reuse_first_qubit_param_gate_raises(self, name, gate):
        """Reusing first qubit after parameterized two-qubit gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            _q1_out, q2_out = gate(q1, q2, theta)
            q1_bad = qm.h(q1)  # q1 already consumed
            return q1_bad, q2_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build(parameters=["theta"])

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_reuse_second_qubit_param_gate_raises(self, name, gate):
        """Reusing second qubit after parameterized two-qubit gate should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            q1_out, _q2_out = gate(q1, q2, theta)
            q2_bad = qm.h(q2)  # q2 already consumed
            return q1_out, q2_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build(parameters=["theta"])

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_reassign_both_param_gate_works(self, name, gate):
        """Proper reassignment of both qubits after parameterized two-qubit gate should succeed."""

        @qkernel
        def good_circuit(q1: Qubit, q2: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            q1, q2 = gate(q1, q2, theta)
            return q1, q2

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None


class TestAllTwoQubitGatesAlias:
    """Every two-qubit gate must detect aliasing (same qubit in both positions)."""

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_alias_same_qubit_raises(self, name, gate):
        """Same qubit in both positions should raise QubitAliasError."""

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return gate(q, q)

        with pytest.raises(QubitAliasError) as exc_info:
            bad_circuit.build()

        assert name.upper() in str(exc_info.value)

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_alias_same_qubit_param_gate_raises(self, name, gate):
        """Same qubit in both positions of parameterized gate should raise QubitAliasError."""

        @qkernel
        def bad_circuit(q: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            return gate(q, q, theta)

        with pytest.raises(QubitAliasError):
            bad_circuit.build(parameters=["theta"])


class TestThreeQubitGateDoubleUse:
    """ccx must consume all three qubits; reusing any one raises QubitConsumedError."""

    def test_reuse_control1_raises(self):
        """Reusing control1 qubit after ccx should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            _q1_out, q2_out, q3_out = qm.ccx(q1, q2, q3)
            q1_bad = qm.h(q1)  # q1 already consumed
            return q1_bad, q2_out, q3_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_reuse_control2_raises(self):
        """Reusing control2 qubit after ccx should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            q1_out, _q2_out, q3_out = qm.ccx(q1, q2, q3)
            q2_bad = qm.h(q2)  # q2 already consumed
            return q1_out, q2_bad, q3_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_reuse_target_raises(self):
        """Reusing target qubit after ccx should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            q1_out, q2_out, _q3_out = qm.ccx(q1, q2, q3)
            q3_bad = qm.h(q3)  # q3 already consumed
            return q1_out, q2_out, q3_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_reassign_all_works(self):
        """Proper reassignment of all three qubits after ccx should succeed."""

        @qkernel
        def good_circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            q1, q2, q3 = qm.ccx(q1, q2, q3)
            return q1, q2, q3

        graph = good_circuit.build()
        assert graph is not None


class TestVectorQubitPatterns:
    """Gates used via Vector[Qubit] borrow-operate-return pattern."""

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_single_qubit_gate_borrow_return_works(self, name, gate):
        """Borrow from vector, apply single-qubit gate, return to vector should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q = qs[0]
            q = gate(q)
            qs[0] = q
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_gate_borrow_return_works(self, name, gate):
        """Borrow from vector, apply rotation gate, return to vector should succeed."""

        @qkernel
        def good_circuit(theta: qm.Float) -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q = qs[0]
            q = gate(q, theta)
            qs[0] = q
            return qm.measure(qs)

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_two_qubit_gate_borrow_return_works(self, name, gate):
        """Borrow two elements from vector, apply two-qubit gate, return both should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = gate(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_two_qubit_param_gate_borrow_return_works(self, name, gate):
        """Borrow two elements, apply parameterized two-qubit gate, return both should succeed."""

        @qkernel
        def good_circuit(theta: qm.Float) -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = gate(q0, q1, theta)
            qs[0] = q0
            qs[1] = q1
            return qm.measure(qs)

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None

    def test_three_qubit_gate_borrow_return_works(self):
        """Borrow three elements from vector, apply ccx, return all should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q2 = qs[2]
            q0, q1, q2 = qm.ccx(q0, q1, q2)
            qs[0] = q0
            qs[1] = q1
            qs[2] = q2
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_unreturned_borrow_after_gate_raises(self, name, gate):
        """Borrowing, applying gate, not returning before measure should raise UnreturnedBorrowError."""

        @qkernel
        def bad_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            q = qs[0]
            q = gate(q)
            # Missing: qs[0] = q
            return qm.measure(qs)

        with pytest.raises(UnreturnedBorrowError):
            bad_circuit.build()

    def test_double_borrow_after_gate_raises(self):
        """Borrowing an element, applying gate, then borrowing same element again should raise."""

        @qkernel
        def bad_circuit() -> Qubit:
            qs = qubit_array(2, "qs")
            q = qs[0]
            q = qm.h(q)
            # qs[0] still borrowed — trying to borrow again raises
            q_again = qs[0]
            return q_again

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()


class TestStubCompositeGateLinearity:
    """Stub composite gates must enforce linearity on their qubit arguments."""

    def _make_single_qubit_composite(self):
        class StubSingleQubit(CompositeGate):
            custom_name = "stub_h"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                (q,) = qubits
                return (qm.h(q),)

        return StubSingleQubit()

    def _make_two_qubit_composite(self):
        class StubTwoQubit(CompositeGate):
            custom_name = "stub_cx"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits):
                q0, q1 = qubits
                q0, q1 = qm.cx(q0, q1)
                return (q0, q1)

        return StubTwoQubit()

    def _make_controlled_composite(self):
        class StubControlled(CompositeGate):
            custom_name = "stub_ctrl_h"
            num_control_qubits = 1

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                (q,) = qubits
                return (qm.h(q),)

        return StubControlled()

    def test_single_qubit_composite_double_use_raises(self):
        """Reusing qubit after single-qubit composite gate should raise QubitConsumedError."""
        gate = self._make_single_qubit_composite()

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            (q2,) = gate(q)
            q3 = qm.x(q)  # q already consumed
            return q2, q3

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_single_qubit_composite_proper_use_works(self):
        """Proper use of single-qubit composite gate should succeed."""
        gate = self._make_single_qubit_composite()

        @qkernel
        def good_circuit(q: Qubit) -> Qubit:
            (q,) = gate(q)
            q = qm.x(q)
            return q

        graph = good_circuit.build()
        assert graph is not None

    def test_two_qubit_composite_double_use_first_raises(self):
        """Reusing first qubit after two-qubit composite gate should raise QubitConsumedError."""
        gate = self._make_two_qubit_composite()

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            _q1_out, q2_out = gate(q1, q2)
            q1_bad = qm.h(q1)  # q1 already consumed
            return q1_bad, q2_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_two_qubit_composite_double_use_second_raises(self):
        """Reusing second qubit after two-qubit composite gate should raise QubitConsumedError."""
        gate = self._make_two_qubit_composite()

        @qkernel
        def bad_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1_out, _q2_out = gate(q1, q2)
            q2_bad = qm.h(q2)  # q2 already consumed
            return q1_out, q2_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_two_qubit_composite_alias_raises(self):
        """Same qubit in both positions of two-qubit composite gate should raise a LinearTypeError.

        Note: CompositeGate consumes targets sequentially, so the second use of
        an already-consumed handle raises QubitConsumedError (not QubitAliasError
        as with built-in gates that do an explicit alias check first).
        """
        gate = self._make_two_qubit_composite()

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return gate(q, q)

        with pytest.raises(LinearTypeError):
            bad_circuit.build()

    def test_two_qubit_composite_proper_use_works(self):
        """Proper use of two-qubit composite gate should succeed."""
        gate = self._make_two_qubit_composite()

        @qkernel
        def good_circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1, q2 = gate(q1, q2)
            return q1, q2

        graph = good_circuit.build()
        assert graph is not None

    def test_controlled_composite_reuse_control_raises(self):
        """Reusing control qubit after composite gate call should raise QubitConsumedError."""
        gate = self._make_controlled_composite()

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, tgt_out = gate(tgt, controls=(ctrl,))
            ctrl_bad = qm.x(ctrl)  # ctrl already consumed
            return ctrl_out, tgt_out, ctrl_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_composite_reuse_target_raises(self):
        """Reusing target qubit after composite gate call should raise QubitConsumedError."""
        gate = self._make_controlled_composite()

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, tgt_out = gate(tgt, controls=(ctrl,))
            tgt_bad = qm.x(tgt)  # tgt already consumed
            return ctrl_out, tgt_out, tgt_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_composite_proper_use_works(self):
        """Proper use of controlled composite gate should succeed."""
        gate = self._make_controlled_composite()

        @qkernel
        def good_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = gate(tgt, controls=(ctrl,))
            return ctrl, tgt

        graph = good_circuit.build()
        assert graph is not None


class TestStdlibGatesLinearity:
    """QFT, IQFT, and QPE must properly consume qubits."""

    def test_qft_proper_use_works(self):
        """qft on a vector should succeed when vector is properly reassigned."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            qs = qft(qs)
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    def test_iqft_proper_use_works(self):
        """iqft on a vector should succeed when vector is properly reassigned."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            qs = iqft(qs)
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    def test_qft_then_iqft_works(self):
        """Applying qft followed by iqft should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    def test_qpe_proper_use_works(self):
        """qpe with a valid unitary kernel should succeed."""

        @qkernel
        def phase_gate(q: Qubit, theta: qm.Float) -> Qubit:
            return qm.p(q, theta)

        @qkernel
        def good_circuit(theta: qm.Float) -> qm.Float:
            counting = qubit_array(3, "counting")
            target = qm.qubit("target")
            target = qm.x(target)
            phase = qpe(target, counting, phase_gate, theta=theta)
            return qm.measure(phase)

        graph = good_circuit.build(parameters=["theta"])
        assert graph is not None


class TestControlledGateLinearity:
    """qm.controlled() wrappers must enforce linearity on both control and target."""

    def _make_sub_kernel(self):
        @qkernel
        def sub(q: Qubit) -> Qubit:
            return qm.h(q)

        return sub

    def test_controlled_proper_use_works(self):
        """Controlled gate with reassignment of both outputs should succeed."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.controlled(sub)

        @qkernel
        def good_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = ctrl_h(ctrl, tgt)
            return ctrl, tgt

        graph = good_circuit.build()
        assert graph is not None

    def test_controlled_reuse_control_raises(self):
        """Reusing control qubit after controlled gate should raise QubitConsumedError."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.controlled(sub)

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            _ctrl_out, tgt_out = ctrl_h(ctrl, tgt)
            ctrl_bad = qm.x(ctrl)  # ctrl already consumed
            return _ctrl_out, tgt_out, ctrl_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_reuse_target_raises(self):
        """Reusing target qubit after controlled gate should raise QubitConsumedError."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.controlled(sub)

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, _tgt_out = ctrl_h(ctrl, tgt)
            tgt_bad = qm.x(tgt)  # tgt already consumed
            return ctrl_out, tgt_bad, _tgt_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_alias_raises(self):
        """Same qubit as both control and target should raise QubitAliasError."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.controlled(sub)

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return ctrl_h(q, q)  # same qubit in both positions

        with pytest.raises(QubitAliasError):
            bad_circuit.build()

    def test_double_controlled_proper_use_works(self):
        """Double-controlled gate (num_controls=2) with reassignment should succeed."""
        sub = self._make_sub_kernel()
        cc_h = qm.controlled(sub, num_controls=2)

        @qkernel
        def good_circuit(
            c0: Qubit, c1: Qubit, tgt: Qubit
        ) -> tuple[Qubit, Qubit, Qubit]:
            c0, c1, tgt = cc_h(c0, c1, tgt)
            return c0, c1, tgt

        graph = good_circuit.build()
        assert graph is not None

    def test_double_controlled_reuse_control_raises(self):
        """Reusing a control qubit after double-controlled gate should raise QubitConsumedError."""
        sub = self._make_sub_kernel()
        cc_h = qm.controlled(sub, num_controls=2)

        @qkernel
        def bad_circuit(c0: Qubit, c1: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            _c0_out, c1_out, tgt_out = cc_h(c0, c1, tgt)
            c0_bad = qm.x(c0)  # c0 already consumed
            return c0_bad, c1_out, tgt_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()


class TestMeasurementAllPatterns:
    """Measurement patterns: single qubit, vector, and linearity after measurement."""

    def test_measure_single_after_gate_works(self):
        """Measuring a qubit after a gate should succeed."""

        @qkernel
        def good_circuit(q: Qubit) -> qm.Bit:
            q = qm.h(q)
            return qm.measure(q)

        graph = good_circuit.build()
        assert graph is not None

    def test_measure_vector_works(self):
        """Measuring a vector of qubits should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    def test_measure_vector_after_gates_works(self):
        """Measuring a vector after applying gates to each element should succeed."""

        @qkernel
        def good_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            for i in qm.range(3):
                qs[i] = qm.h(qs[i])
            return qm.measure(qs)

        graph = good_circuit.build()
        assert graph is not None

    def test_measure_vector_unreturned_borrow_raises(self):
        """Measuring a vector with an unreturned borrow should raise UnreturnedBorrowError."""

        @qkernel
        def bad_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            _q = qs[0]  # borrow but don't return
            return qm.measure(qs)

        with pytest.raises(UnreturnedBorrowError):
            bad_circuit.build()

    def test_reuse_qubit_after_measure_raises(self):
        """Using a qubit after measure should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q: Qubit) -> qm.Bit:
            bit = qm.measure(q)
            _q2 = qm.h(q)  # q consumed by measure
            return bit

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_gate_then_measure_works(self, name, gate):
        """Every single-qubit gate followed by measure should succeed."""

        @qkernel
        def good_circuit(q: Qubit) -> qm.Bit:
            q = gate(q)
            return qm.measure(q)

        graph = good_circuit.build()
        assert graph is not None


class TestArrayConsumeUnreturnedBorrow:
    """Test that ArrayBase.consume() rejects quantum arrays with unreturned borrows.

    Validates the common borrow-return contract in consume() covers all sinks:
    qkernel calls, controlled gates (index-spec and symbolic), measure, and cast.
    """

    def test_qkernel_call_with_unreturned_vector_borrow_raises(self):
        """QKernel call consuming a Vector[Qubit] with unreturned borrow should raise."""

        @qkernel
        def id_arr(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            return qs

        @qkernel
        def bad_outer() -> qm.Vector[Qubit]:
            qs = qubit_array(3, "qs")
            _q = qs[0]  # borrow but don't return
            return id_arr(qs)

        with pytest.raises(UnreturnedBorrowError):
            bad_outer.build()

    def test_qkernel_call_with_returned_vector_borrow_works(self):
        """QKernel call after all borrows are returned should succeed."""

        @qkernel
        def id_arr(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            return qs

        @qkernel
        def good_outer() -> qm.Vector[Qubit]:
            qs = qubit_array(3, "qs")
            q = qs[0]
            q = qm.h(q)
            qs[0] = q  # return the borrow
            return id_arr(qs)

        graph = good_outer.build()
        assert graph is not None

    def test_controlled_index_spec_with_unreturned_vector_borrow_raises(self):
        """controlled() with target_indices on a Vector with unreturned borrow should raise."""

        @qkernel
        def x_gate(q: Qubit) -> Qubit:
            return qm.x(q)

        @qkernel
        def bad_controlled() -> qm.Vector[Qubit]:
            qs = qubit_array(3, "qs")
            _q = qs[0]  # borrow but don't return
            cx = qm.controlled(x_gate)
            qs = cx(qs, target_indices=[2])
            return qs

        with pytest.raises(UnreturnedBorrowError):
            bad_controlled.build()

    def test_controlled_symbolic_controls_with_unreturned_borrow_raises(self):
        """controlled() with symbolic num_controls on a Vector with unreturned borrow should raise."""
        from qamomile.circuit.frontend.handle.primitives import UInt
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        @qkernel
        def x_gate(q: Qubit) -> Qubit:
            return qm.x(q)

        n = UInt(value=Value(type=UIntType(), name="n"))

        @qkernel
        def bad_controlled_sym() -> tuple[qm.Vector[Qubit], Qubit]:
            qs = qubit_array(3, "qs")
            tgt = qubit_array(1, "tgt")
            _q = qs[0]  # borrow but don't return
            cx = qm.controlled(x_gate, num_controls=n)
            qs, t = cx(qs, tgt[0])
            return qs, t

        with pytest.raises(UnreturnedBorrowError):
            bad_controlled_sym.build()

    def test_measure_cast_unreturned_borrow_behavior_unchanged(self):
        """measure/cast with unreturned borrow should still raise (regression guard)."""

        @qkernel
        def bad_measure() -> qm.Vector[qm.Bit]:
            qs = qubit_array(3, "qs")
            _q = qs[0]  # borrow but don't return
            return qm.measure(qs)

        with pytest.raises(UnreturnedBorrowError):
            bad_measure.build()

    def test_non_quantum_array_consume_unchanged(self):
        """Classical array consume should not raise even with 'borrows'."""

        @qkernel
        def classical_circuit() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            return qm.measure(qs)

        graph = classical_circuit.build()
        assert graph is not None


class TestQuantumRebindDetection:
    """Test AST-level detection of forbidden quantum variable reassignment."""

    # ---- Forbidden patterns ----

    def test_build_scalar_overwrite_call_rejected(self):
        """a = h(b) where a is existing quantum should raise QubitRebindError."""

        @qkernel
        def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            a = qm.h(b)
            return a

        with pytest.raises(QubitRebindError):
            bad.build()

    def test_build_scalar_overwrite_direct_rejected(self):
        """a = b where a is existing quantum should raise QubitRebindError."""

        @qkernel
        def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            a = b
            return a

        with pytest.raises(QubitRebindError):
            bad.build()

    def test_transpile_scalar_overwrite_rebind_rejected(self):
        """Rebind violation should also raise when accessing .block."""

        @qkernel
        def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            a = qm.h(b)
            return a

        with pytest.raises(QubitRebindError):
            _ = bad.block

    def test_build_vector_overwrite_direct_rejected(self):
        """as_ = bs where both are Vector[Qubit] should raise."""

        @qkernel
        def bad(qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            qs1 = qs2
            return qs1

        with pytest.raises(QubitRebindError):
            bad.build()

    def test_build_vector_overwrite_call_rejected(self):
        """Vector quantum var overwritten from call with different source."""

        @qkernel
        def bad(qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            qs1 = qft(qs2)
            return qs1

        with pytest.raises(QubitRebindError):
            bad.build()

    # ---- Allowed patterns ----

    def test_scalar_new_binding_allowed(self):
        """q3 = h(q1) where q3 is new should be allowed."""

        @qkernel
        def ok(q1: qm.Qubit) -> qm.Qubit:
            q3 = qm.h(q1)
            return q3

        graph = ok.build()
        assert graph is not None

    def test_scalar_new_alias_allowed(self):
        """alias = q where alias is new should be allowed."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Qubit:
            alias = q
            alias = qm.h(alias)
            return alias

        graph = ok.build()
        assert graph is not None

    def test_vector_new_binding_allowed(self):
        """New variable from vector call is allowed."""

        @qkernel
        def ok(qs: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            result = qft(qs)
            return result

        graph = ok.build()
        assert graph is not None

    def test_self_update_allowed(self):
        """q = h(q) self-update should be allowed."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            q = qm.x(q)
            return q

        graph = ok.build()
        assert graph is not None

    def test_same_origin_alias_allowed(self):
        """alias = q; q = h(alias) should be allowed (same origin)."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Qubit:
            alias = q
            q = qm.h(alias)
            return q

        graph = ok.build()
        assert graph is not None

    def test_tuple_self_update_allowed(self):
        """q1, q2 = cx(q1, q2) should be allowed."""

        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = qm.cx(q1, q2)
            return q1, q2

        graph = ok.build()
        assert graph is not None

    def test_implicit_discard_still_allowed(self):
        """Not returning a qubit (implicit discard) should still be allowed."""

        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> qm.Qubit:
            q1 = qm.h(q1)
            return q1

        graph = ok.build()
        assert graph is not None

    def test_error_not_raised_at_definition(self):
        """@qkernel definition should succeed even with violations."""

        @qkernel
        def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            a = qm.h(b)
            return a

        # Definition succeeded, error only on build/block access
        assert bad.name == "bad"
        with pytest.raises(QubitRebindError):
            bad.build()
