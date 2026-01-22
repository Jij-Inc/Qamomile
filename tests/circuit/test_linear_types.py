"""Tests for linear type enforcement in the circuit frontend."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.constructors import qubit, qubit_array
from qamomile.circuit.transpiler.errors import (
    QubitConsumedError,
    QubitAliasError,
    UnreturnedBorrowError,
)


class TestDoubleUseDetection:
    """Test that using the same qubit handle twice raises an error."""

    def test_double_use_single_qubit_gate_raises_error(self):
        """Same qubit used in two single-qubit gates should raise QubitConsumedError."""

        @qkernel
        def bad_circuit(q: Qubit) -> Qubit:
            q1 = qm.h(q)
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
            q1 = qm.rx(q, theta)
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
            q0 = qs[0]  # Borrow element 0
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
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.tracer import trace

        # Create a vector and borrow an element
        with trace():
            qs = qubit_array(3, "qs")
            q0 = qs[0]  # Borrow but don't return

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
            q2 = qm.h(q)  # ERROR: q was consumed by measure
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
            q1 = qm.h(q)
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
            q1 = qm.h(q)
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
            q1 = qm.h(q)
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
