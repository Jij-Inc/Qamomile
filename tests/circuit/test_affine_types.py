"""Tests for affine type enforcement in the circuit frontend."""

import numpy as np
import pytest

import qamomile.circuit as qm
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
    AffineTypeError,
    QubitAliasError,
    QubitBorrowConflictError,
    QubitConsumedError,
    QubitRebindError,
    UnreturnedBorrowError,
)
from tests.circuit._gate_catalog import (
    ROTATION_GATES as _ROTATION_GATES,
    SINGLE_QUBIT_GATES as _SINGLE_QUBIT_GATES,
    STDLIB_GATES as _STDLIB_GATES,
    THREE_QUBIT_GATES as _THREE_QUBIT_GATES,
    TWO_QUBIT_GATES_NO_PARAM as _TWO_QUBIT_GATES_NO_PARAM,
    TWO_QUBIT_GATES_WITH_PARAM as _TWO_QUBIT_GATES_WITH_PARAM,
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


class TestOperationExceptionSafety:
    """Primitive operations avoid partial affine ownership commits."""

    @pytest.mark.parametrize(
        "operation",
        [qm.h, qm.project_z, qm.reset, qm.measure],
        ids=["gate", "project", "reset", "measure"],
    )
    def test_missing_tracer_leaves_scalar_input_unconsumed(self, operation):
        """A missing tracer is diagnosed before scalar ownership moves."""
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        qubit = Qubit(value=Value(type=QubitType(), name="q"))

        with pytest.raises(RuntimeError, match="No active tracer"):
            operation(qubit)

        assert not qubit._consumed

    def test_late_multi_gate_validation_leaves_earlier_input_unconsumed(self):
        """A consumed second operand cannot partially consume the first."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            control = qm.qubit("control")
            target = qm.qubit("target")
            _target_successor = qm.h(target)

            with pytest.raises(QubitConsumedError):
                qm.cx(control, target)

            assert not control._consumed

    def test_missing_tracer_leaves_vector_input_unconsumed(self):
        """Broadcast and vector measurement preflight the tracer."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            gate_register = qubit_array(2, "gate_register")
            measure_register = qubit_array(2, "measure_register")

        with pytest.raises(RuntimeError, match="No active tracer"):
            qm.h(gate_register)
        with pytest.raises(RuntimeError, match="No active tracer"):
            qm.measure(measure_register)

        assert not gate_register._consumed
        assert not measure_register._consumed


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

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ((0, 2, 4), (1, 2, 4), False),
            ((0, 2, 4), (4, 3, 2), True),
            ((10, 5, 1_000_000_000), (12, 7, 1_000_000_000), True),
            ((0, 1, 0), (0, 1, 1), False),
            (None, (0, 1, 1), False),
        ],
    )
    def test_affine_region_overlap_is_exact_and_size_independent(
        self,
        left: tuple[int, int, int] | None,
        right: tuple[int, int, int] | None,
        expected: bool,
    ):
        """Finite strided regions intersect without enumerating their slots."""
        from qamomile.circuit.frontend.qkernel_utils import _regions_may_overlap

        assert _regions_may_overlap(left, right) is expected

    def test_affine_region_overlap_matches_random_explicit_sets(self):
        """Modular overlap agrees with enumerated finite regions for seed 8421."""
        from qamomile.circuit.frontend.qkernel_utils import _regions_may_overlap

        rng = np.random.default_rng(8421)
        for _ in range(1_000):
            left = (
                int(rng.integers(0, 30)),
                int(rng.integers(1, 9)),
                int(rng.integers(0, 15)),
            )
            right = (
                int(rng.integers(0, 30)),
                int(rng.integers(1, 9)),
                int(rng.integers(0, 15)),
            )
            left_slots = {left[0] + left[1] * index for index in range(left[2])}
            right_slots = {right[0] + right[1] * index for index in range(right[2])}

            assert _regions_may_overlap(left, right) is bool(left_slots & right_slots)

    def test_distinct_fresh_subkernel_qubits_do_not_alias(self):
        """Fresh allocations returned by separate calls have distinct logical IDs."""

        @qkernel
        def make_q() -> Qubit:
            return qm.qubit("q")

        @qkernel
        def top() -> qm.Bit:
            a = make_q()
            b = make_q()
            a, b = qm.cx(a, b)
            return qm.measure(a)

        graph = top.block
        assert any(
            isinstance(op, GateOperation) and op.gate_type == GateOperationType.CX
            for op in graph.operations
        )

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

        with pytest.raises(QubitBorrowConflictError) as exc_info:
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
    """Test that classical values don't have affine type restrictions."""

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

    def test_setitem_rejects_return_to_different_index(self):
        """Returning a borrowed element to a different index should raise."""

        @qkernel
        def bad_circuit() -> tuple[Qubit, Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            qs[0] = q1  # ERROR: q1 was borrowed from qs[1], not qs[0]
            qs[1] = q0
            return q0, q1

        with pytest.raises(AffineTypeError, match="same index"):
            bad_circuit.build()

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
        """Returning a qubit from a different array should raise AffineTypeError."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs1 = qubit_array(1, "qs1")
            qs2 = qubit_array(1, "qs2")
            _q1 = qs1[0]
            rogue = qs2[0]
            with pytest.raises(AffineTypeError, match="not borrowed from this array"):
                qs1[0] = rogue

    def test_setitem_unborrowed_index_rejects_fresh_handle(self):
        """Writing a fresh qubit to an unborrowed slot fails without mutation."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(2, "qs")
            rogue = qm.qubit("rogue")
            with pytest.raises(AffineTypeError, match="not representable"):
                qs[1] = rogue
            assert rogue._consumed is False

    def test_setitem_unborrowed_rejects_foreign_array_handle(self):
        """Writing a handle borrowed from another array to an unborrowed index should raise."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            foreign = qs2[0]  # borrowed from qs2, parent=qs2
            with pytest.raises(AffineTypeError, match="not borrowed from this array"):
                qs1[1] = foreign  # unborrowed index, but foreign handle

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

        @qm.composite_gate(name="simple_h")
        def gate(q: Qubit) -> Qubit:
            return qm.h(q)

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            q2 = gate(q)
            q3 = qm.x(q)  # ERROR: q consumed by gate call
            return q2, q3

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_composite_gate_call_consumes_control_qubit(self):
        """Reusing control qubit after CompositeGate call should raise QubitConsumedError."""

        @qm.composite_gate(name="controlled_h")
        def base_gate(q: Qubit) -> Qubit:
            return qm.h(q)

        gate = qm.control(base_gate)

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, tgt_out = gate(ctrl, tgt)
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

    def test_symbolic_cross_index_write_is_rejected(self):
        """A symbolic target cannot hide a write to a different quantum slot."""

        @qkernel
        def bad_cross_index(n: int) -> qm.Vector[qm.Bit]:
            qubits = qubit_array(n, "q")
            for i in qm.range(1):
                qubits[i + 2] = qm.x(qubits[i])
            return qm.measure(qubits)

        with pytest.raises(
            AffineTypeError, match="different symbolic index expression"
        ) as exc_info:
            bad_cross_index.build(n=4)

        message = str(exc_info.value)
        assert "'q[i]'" in message
        assert "'q[(i + 2)]'" in message


class TestAllSingleQubitGatesDoubleUse:
    """Every single-qubit gate must enforce affine usage (double-use → error, reassign → OK)."""

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
    """Every rotation gate must enforce affine usage."""

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

        with pytest.raises(QubitBorrowConflictError):
            bad_circuit.build()


class TestCustomCompositeGateAffine:
    """Custom composite gates must enforce affine usage on their qubit arguments."""

    def _make_single_qubit_composite(self):
        @qm.composite_gate(name="custom_h")
        def custom_h(q: Qubit) -> Qubit:
            return qm.h(q)

        return custom_h

    def _make_two_qubit_composite(self):
        @qm.composite_gate(name="custom_cx")
        def custom_cx(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            return qm.cx(q0, q1)

        return custom_cx

    def _make_controlled_composite(self):
        return qm.control(self._make_single_qubit_composite())

    def test_single_qubit_composite_double_use_raises(self):
        """Reusing qubit after single-qubit composite gate should raise QubitConsumedError."""
        gate = self._make_single_qubit_composite()

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            q2 = gate(q)
            q3 = qm.x(q)  # q already consumed
            return q2, q3

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_single_qubit_composite_proper_use_works(self):
        """Proper use of single-qubit composite gate should succeed."""
        gate = self._make_single_qubit_composite()

        @qkernel
        def good_circuit(q: Qubit) -> Qubit:
            q = gate(q)
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
        """Same qubit in both positions of two-qubit composite gate should raise an AffineTypeError.

        Note: CompositeGate consumes targets sequentially, so the second use of
        an already-consumed handle raises QubitConsumedError (not QubitAliasError
        as with built-in gates that do an explicit alias check first).
        """
        gate = self._make_two_qubit_composite()

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return gate(q, q)

        with pytest.raises(AffineTypeError):
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
            ctrl_out, tgt_out = gate(ctrl, tgt)
            ctrl_bad = qm.x(ctrl)  # ctrl already consumed
            return ctrl_out, tgt_out, ctrl_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_composite_reuse_target_raises(self):
        """Reusing target qubit after composite gate call should raise QubitConsumedError."""
        gate = self._make_controlled_composite()

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, tgt_out = gate(ctrl, tgt)
            tgt_bad = qm.x(tgt)  # tgt already consumed
            return ctrl_out, tgt_out, tgt_bad

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_composite_proper_use_works(self):
        """Proper use of controlled composite gate should succeed."""
        gate = self._make_controlled_composite()

        @qkernel
        def good_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = gate(ctrl, tgt)
            return ctrl, tgt

        graph = good_circuit.build()
        assert graph is not None


class TestStdlibGatesAffine:
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


class TestControlledGateAffine:
    """qm.control() wrappers must enforce affine usage on both control and target."""

    def _make_sub_kernel(self):
        @qkernel
        def sub(q: Qubit) -> Qubit:
            return qm.h(q)

        return sub

    def test_controlled_proper_use_works(self):
        """Controlled gate with reassignment of both outputs should succeed."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.control(sub)

        @qkernel
        def good_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = ctrl_h(ctrl, tgt)
            return ctrl, tgt

        graph = good_circuit.build()
        assert graph is not None

    def test_controlled_reuse_control_raises(self):
        """Reusing control qubit after controlled gate should raise QubitConsumedError."""
        sub = self._make_sub_kernel()
        ctrl_h = qm.control(sub)

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
        ctrl_h = qm.control(sub)

        @qkernel
        def bad_circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            ctrl_out, _tgt_out = ctrl_h(ctrl, tgt)
            tgt_bad = qm.x(tgt)  # tgt already consumed
            return ctrl_out, tgt_bad, _tgt_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_controlled_alias_raises(self):
        """Reusing the same qubit as control + target raises QubitConsumedError.

        Step 6 of the controlled-API redesign dropped the bespoke
        entry-point ``_validate_no_alias_or_overlap`` check; the
        ``Handle.consume()`` linear-type layer catches the duplicate
        on the second consume, so the error class is
        ``QubitConsumedError`` instead of ``QubitAliasError``.
        """
        sub = self._make_sub_kernel()
        ctrl_h = qm.control(sub)

        @qkernel
        def bad_circuit(q: Qubit) -> tuple[Qubit, Qubit]:
            return ctrl_h(q, q)  # same qubit in both positions

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()

    def test_double_controlled_proper_use_works(self):
        """Double-controlled gate (num_controls=2) with reassignment should succeed."""
        sub = self._make_sub_kernel()
        cc_h = qm.control(sub, num_controls=2)

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
        cc_h = qm.control(sub, num_controls=2)

        @qkernel
        def bad_circuit(c0: Qubit, c1: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            _c0_out, c1_out, tgt_out = cc_h(c0, c1, tgt)
            c0_bad = qm.x(c0)  # c0 already consumed
            return c0_bad, c1_out, tgt_out

        with pytest.raises(QubitConsumedError):
            bad_circuit.build()


class TestMeasurementAllPatterns:
    """Measurement patterns: single qubit, vector, and affine usage after measurement."""

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

    def test_controlled_with_unreturned_vector_borrow_raises(self):
        """control() called on a Vector with an unreturned borrow should raise.

        Migrated from the old ``target_indices``-on-Vector form: the
        regression concern (an unreturned element borrow blocks any
        whole-Vector consume) is now exercised through the new
        ``Vector[Qubit]`` sub-kernel argument path, which is the
        moral equivalent — both shapes hand the whole Vector to the
        controlled call and trip ``validate_all_returned()``.
        """

        @qkernel
        def x_gate_broadcast(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs = qm.x(qs)
            return qs

        @qkernel
        def bad_controlled() -> qm.Vector[Qubit]:
            qs = qubit_array(3, "qs")
            ctrl = qm.qubit(name="ctrl")
            _q = qs[0]  # borrow but don't return
            cx = qm.control(x_gate_broadcast)
            _ctrl_out, qs = cx(ctrl, qs)  # type: ignore
            return qs

        with pytest.raises(UnreturnedBorrowError):
            bad_controlled.build()

    def test_controlled_symbolic_controls_with_unreturned_borrow_raises(self):
        """control() with symbolic num_controls on a Vector with unreturned borrow should raise."""
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
            cx = qm.control(x_gate, num_controls=n)
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


_DIRECT_ELEMENT_FORBIDDEN_ERRORS = (
    QubitRebindError,
    QubitConsumedError,
    AffineTypeError,
)


@qm.composite_gate(name="single_rebind_comp")
def _single_rebind_composite(q0: qm.Qubit) -> qm.Qubit:
    return qm.h(q0)


@qm.composite_gate(name="two_rebind_comp")
def _two_rebind_composite(
    q0: qm.Qubit,
    q1: qm.Qubit,
) -> tuple[qm.Qubit, qm.Qubit]:
    return qm.cx(q0, q1)


_controlled_rebind_composite = qm.control(_single_rebind_composite)


def _make_rebind_subkernels():
    @qkernel
    def k1(q: qm.Qubit) -> qm.Qubit:
        return qm.h(q)

    @qkernel
    def k2(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
        return qm.cx(q1, q2)

    @qkernel
    def kv(qs: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
        return qft(qs)

    return k1, k2, kv


class TestQuantumRebindDetectionBasics:
    """Core AST-level detection and compatibility behavior."""

    def test_scalar_overwrite_call_rejected(self):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = qm.h(b)
                return a

    def test_scalar_overwrite_direct_rejected(self):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = b
                return a

    def test_vector_overwrite_direct_rejected(self):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]
            ) -> qm.Vector[qm.Qubit]:
                qs1 = qs2
                return qs1

    def test_error_raised_at_definition(self):
        """A rebind violation aborts ``@qkernel`` itself; no QKernel object is created."""
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = qm.h(b)
                return a

        assert "bad" not in locals()

    def test_implicit_discard_still_allowed(self):
        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> qm.Qubit:
            q1 = qm.h(q1)
            return q1

        graph = ok.build()
        assert graph is not None

    def test_rebind_error_message_deterministic_with_multiple_quantum_args(self):
        """Error message should consistently reference the first traversed quantum arg."""

        def define_bad() -> None:
            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit, c: qm.Qubit) -> qm.Qubit:
                a = qm.cx(b, c)  # a's origin is a; neither b nor c matches
                return a

        with pytest.raises(QubitRebindError) as exc_info:
            define_bad()
        msg = str(exc_info.value)
        assert "b" in msg or "c" in msg
        for _ in range(5):
            with pytest.raises(QubitRebindError) as exc2:
                define_bad()
            assert str(exc2.value) == msg

    def test_rebind_error_message_includes_kernel_name(self):
        """Error message identifies the offending kernel by name."""
        with pytest.raises(QubitRebindError) as exc_info:

            @qkernel
            def my_uniquely_named_kernel(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = qm.h(b)
                return a

        assert "Kernel 'my_uniquely_named_kernel'" in str(exc_info.value)


class TestQuantumRebindDetectionSingleQubit:
    """Single-qubit and rotation gates with scalar + vector-element patterns."""

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_scalar_overwrite_from_other_qubit_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = gate(b)
                return a

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_scalar_self_update_allowed(self, name, gate):
        @qkernel
        def ok(a: qm.Qubit) -> qm.Qubit:
            a = gate(a)
            return a

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_scalar_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            new_q = gate(b)
            return new_q

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_vector_element_direct_assign_from_other_vector_rejected(self, name, gate):
        @qkernel
        def bad() -> qm.Vector[qm.Qubit]:
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            qs1[0] = gate(qs2[0])
            qs2[1] = gate(qs2[0])  # Re-borrowing same source element should fail.
            return qs1

        with pytest.raises(_DIRECT_ELEMENT_FORBIDDEN_ERRORS):
            bad.build()

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_vector_element_self_update_allowed(self, name, gate):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            qs[0] = gate(qs[0])
            return qs

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _SINGLE_QUBIT_GATES, ids=[g[0] for g in _SINGLE_QUBIT_GATES]
    )
    def test_vector_element_new_binding_allowed(self, name, gate):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            new_q = gate(qs[0])
            qs[0] = new_q
            return qs

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_scalar_overwrite_from_other_qubit_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit, theta: qm.Float) -> qm.Qubit:
                a = gate(b, theta)
                return a

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_scalar_self_update_allowed(self, name, gate):
        @qkernel
        def ok(a: qm.Qubit, theta: qm.Float) -> qm.Qubit:
            a = gate(a, theta)
            return a

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_scalar_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(a: qm.Qubit, b: qm.Qubit, theta: qm.Float) -> qm.Qubit:
            new_q = gate(b, theta)
            return new_q

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_vector_element_direct_assign_from_other_vector_rejected(
        self, name, gate
    ):
        @qkernel
        def bad(theta: qm.Float) -> qm.Vector[qm.Qubit]:
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            qs1[0] = gate(qs2[0], theta)
            qs2[1] = gate(qs2[0], theta)
            return qs1

        with pytest.raises(_DIRECT_ELEMENT_FORBIDDEN_ERRORS):
            bad.build(parameters=["theta"])

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_vector_element_self_update_allowed(self, name, gate):
        @qkernel
        def ok(theta: qm.Float) -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            qs[0] = gate(qs[0], theta)
            return qs

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _ROTATION_GATES, ids=[g[0] for g in _ROTATION_GATES]
    )
    def test_rotation_vector_element_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(theta: qm.Float) -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            new_q = gate(qs[0], theta)
            qs[0] = new_q
            return qs

        graph = ok.build(parameters=["theta"])
        assert graph is not None


class TestQuantumRebindDetectionTwoQubit:
    """Two/three-qubit gate tuple assignment and vector-element tuple patterns."""

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_scalar_tuple_overwrite_first_mismatch_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = gate(q1, q3)
                return q1, q2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_scalar_tuple_overwrite_second_mismatch_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = gate(q3, q2)
                return q1, q2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_scalar_tuple_overwrite_all_mismatch_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, q4: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = gate(q3, q4)
                return q1, q2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_scalar_tuple_self_update_allowed(self, name, gate):
        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = gate(q1, q2)
            return q1, q2

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_scalar_tuple_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(
            q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, q4: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit]:
            new_q1, new_q2 = gate(q3, q4)
            return new_q1, new_q2

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_scalar_tuple_mismatch_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, theta: qm.Float
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = gate(q1, q3, theta)
                return q1, q2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_scalar_tuple_self_update_allowed(self, name, gate):
        @qkernel
        def ok(
            q1: qm.Qubit, q2: qm.Qubit, theta: qm.Float
        ) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = gate(q1, q2, theta)
            return q1, q2

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_scalar_tuple_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(
            q1: qm.Qubit,
            q2: qm.Qubit,
            q3: qm.Qubit,
            q4: qm.Qubit,
            theta: qm.Float,
        ) -> tuple[qm.Qubit, qm.Qubit]:
            new_q1, new_q2 = gate(q3, q4, theta)
            return new_q1, new_q2

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_vector_element_tuple_overwrite_from_other_vector_rejected(
        self, name, gate
    ):
        # No quantum-typed kernel arguments, but the analyzer's constructor
        # tracking promotes ``qs1`` / ``qs2`` to quantum origins, so the
        # cross-array rebind is now flagged at decoration time as a
        # ``QubitRebindError`` (a subclass of ``AffineTypeError``).
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad() -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
                qs1 = qubit_array(2, "qs1")
                qs2 = qubit_array(2, "qs2")
                q0, q1 = qs1[0], qs1[1]
                p0, p1 = qs2[0], qs2[1]
                q0, q1 = gate(p0, p1)
                qs1[0] = q0
                qs1[1] = q1
                return qs1, qs2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_vector_element_tuple_self_update_allowed(self, name, gate):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            q0, q1 = qs[0], qs[1]
            q0, q1 = gate(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_NO_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_NO_PARAM],
    )
    def test_vector_element_tuple_new_binding_allowed(self, name, gate):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            p0, p1 = qs[0], qs[1]
            new_q0, new_q1 = gate(p0, p1)
            qs[0] = new_q0
            qs[1] = new_q1
            return qs

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_vector_element_tuple_overwrite_from_other_vector_rejected(
        self, name, gate
    ):
        # ``theta`` is a classical parameter, so it does not seed any
        # quantum origin — but constructor tracking on ``qs1`` / ``qs2``
        # still catches the cross-array rebind at decoration time.
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                theta: qm.Float,
            ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
                qs1 = qubit_array(2, "qs1")
                qs2 = qubit_array(2, "qs2")
                q0, q1 = qs1[0], qs1[1]
                p0, p1 = qs2[0], qs2[1]
                q0, q1 = gate(p0, p1, theta)
                qs1[0] = q0
                qs1[1] = q1
                return qs1, qs2

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_vector_element_tuple_self_update_allowed(self, name, gate):
        @qkernel
        def ok(theta: qm.Float) -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            q0, q1 = qs[0], qs[1]
            q0, q1 = gate(q0, q1, theta)
            qs[0] = q0
            qs[1] = q1
            return qs

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate",
        _TWO_QUBIT_GATES_WITH_PARAM,
        ids=[g[0] for g in _TWO_QUBIT_GATES_WITH_PARAM],
    )
    def test_param_vector_element_tuple_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(theta: qm.Float) -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            p0, p1 = qs[0], qs[1]
            new_q0, new_q1 = gate(p0, p1, theta)
            qs[0] = new_q0
            qs[1] = new_q1
            return qs

        graph = ok.build(parameters=["theta"])
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _THREE_QUBIT_GATES, ids=[g[0] for g in _THREE_QUBIT_GATES]
    )
    def test_three_qubit_tuple_mixed_origin_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                a: qm.Qubit, b: qm.Qubit, c: qm.Qubit, d: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
                a, b, c = gate(a, b, d)
                return a, b, c

    @pytest.mark.parametrize(
        "name,gate", _THREE_QUBIT_GATES, ids=[g[0] for g in _THREE_QUBIT_GATES]
    )
    def test_three_qubit_tuple_self_update_allowed(self, name, gate):
        @qkernel
        def ok(
            a: qm.Qubit, b: qm.Qubit, c: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
            a, b, c = gate(a, b, c)
            return a, b, c

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _THREE_QUBIT_GATES, ids=[g[0] for g in _THREE_QUBIT_GATES]
    )
    def test_three_qubit_tuple_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(
            a: qm.Qubit,
            b: qm.Qubit,
            c: qm.Qubit,
        ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
            r0, r1, r2 = gate(a, b, c)
            return r0, r1, r2

        graph = ok.build()
        assert graph is not None


class TestQuantumRebindDetectionComposite:
    """Composite gate rebind behavior and controlled-call regression."""

    def test_single_qubit_composite_overwrite_rejected(self):
        gate = _single_rebind_composite

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = gate(b)
                return a

    def test_single_qubit_composite_self_update_allowed(self):
        gate = _single_rebind_composite

        @qkernel
        def ok(a: qm.Qubit) -> qm.Qubit:
            a = gate(a)
            return a

        graph = ok.build()
        assert graph is not None

    def test_single_qubit_composite_new_binding_allowed(self):
        gate = _single_rebind_composite

        @qkernel
        def ok(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            out = gate(b)
            return out

        graph = ok.build()
        assert graph is not None

    def test_two_qubit_composite_tuple_mismatch_rejected(self):
        gate = _two_rebind_composite

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = gate(q1, q3)
                return q1, q2

    def test_two_qubit_composite_self_update_allowed(self):
        gate = _two_rebind_composite

        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = gate(q1, q2)
            return q1, q2

        graph = ok.build()
        assert graph is not None

    def test_two_qubit_composite_new_binding_allowed(self):
        gate = _two_rebind_composite

        @qkernel
        def ok(
            q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, q4: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit]:
            r1, r2 = gate(q3, q4)
            return r1, r2

        graph = ok.build()
        assert graph is not None

    def test_controlled_composite_uses_control_transform(self):
        gate = _controlled_rebind_composite

        @qkernel
        def circuit(ctrl: qm.Qubit, tgt: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            ctrl, tgt = gate(ctrl, tgt)
            return ctrl, tgt

        graph = circuit.build()
        assert graph is not None


class TestQuantumRebindDetectionViaQKernel:
    """Rebind behavior through qkernel calls (scalar/tuple/vector/element)."""

    def test_scalar_overwrite_rejected(self):
        k1, _, _ = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = k1(b)
                return a

    def test_scalar_self_update_allowed(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def ok(a: qm.Qubit) -> qm.Qubit:
            a = k1(a)
            return a

        graph = ok.build()
        assert graph is not None

    def test_scalar_new_binding_allowed(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def ok(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
            out = k1(b)
            return out

        graph = ok.build()
        assert graph is not None

    def test_tuple_overwrite_first_mismatch_rejected(self):
        _, k2, _ = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = k2(q1, q3)
                return q1, q2

    def test_tuple_overwrite_second_mismatch_rejected(self):
        _, k2, _ = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = k2(q3, q2)
                return q1, q2

    def test_tuple_overwrite_all_mismatch_rejected(self):
        _, k2, _ = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, q4: qm.Qubit
            ) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = k2(q3, q4)
                return q1, q2

    def test_tuple_self_update_allowed(self):
        _, k2, _ = _make_rebind_subkernels()

        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = k2(q1, q2)
            return q1, q2

        graph = ok.build()
        assert graph is not None

    def test_tuple_new_binding_allowed(self):
        _, k2, _ = _make_rebind_subkernels()

        @qkernel
        def ok(
            q1: qm.Qubit, q2: qm.Qubit, q3: qm.Qubit, q4: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit]:
            r1, r2 = k2(q3, q4)
            return r1, r2

        graph = ok.build()
        assert graph is not None

    def test_tuple_intermediate_alias_rebind_rejected(self):
        """Tuple result alias used to overwrite an existing quantum var should be rejected."""
        _, k2, _ = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit, c: qm.Qubit) -> qm.Qubit:
                x, _y = k2(b, c)
                a = x
                return a

    def test_vector_whole_overwrite_rejected(self):
        _, _, kv = _make_rebind_subkernels()

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]
            ) -> qm.Vector[qm.Qubit]:
                qs1 = kv(qs2)
                return qs1

    def test_vector_whole_self_update_allowed(self):
        _, _, kv = _make_rebind_subkernels()

        @qkernel
        def ok(qs1: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            qs1 = kv(qs1)
            return qs1

        graph = ok.build()
        assert graph is not None

    def test_vector_whole_new_binding_allowed(self):
        _, _, kv = _make_rebind_subkernels()

        @qkernel
        def ok(
            qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]
        ) -> qm.Vector[qm.Qubit]:
            out = kv(qs2)
            return out

        graph = ok.build()
        assert graph is not None

    def test_scalar_overwrite_from_vector_element_const_index_rejected(self):
        """Overwriting an existing scalar qubit from qs[0] should raise rebind error."""

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, qs: qm.Vector[qm.Qubit]) -> qm.Qubit:
                a = qs[0]
                return a

    def test_scalar_overwrite_from_vector_element_symbolic_index_rejected(self):
        """Overwriting an existing scalar qubit from qs[i] should raise rebind error."""

        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, qs: qm.Vector[qm.Qubit], i: qm.UInt) -> qm.Qubit:
                a = qs[i]
                return a

    def test_scalar_new_binding_from_vector_element_symbolic_index_allowed(self):
        """Binding qs[i] to a new name should remain allowed."""

        @qkernel
        def ok(qs: qm.Vector[qm.Qubit], i: qm.UInt) -> qm.Qubit:
            new_q = qs[i]
            return new_q

        graph = ok.build(i=0)
        assert graph is not None

    def test_vector_element_direct_assign_via_qkernel_rejected(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def bad() -> qm.Vector[qm.Qubit]:
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            qs1[0] = k1(qs2[0])
            qs2[1] = k1(qs2[0])
            return qs1

        with pytest.raises(_DIRECT_ELEMENT_FORBIDDEN_ERRORS):
            bad.build()

    def test_vector_element_self_update_via_qkernel_allowed(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            qs[0] = k1(qs[0])
            return qs

        graph = ok.build()
        assert graph is not None

    def test_vector_element_new_binding_via_qkernel_allowed(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            new_q = k1(qs[0])
            qs[0] = new_q
            return qs

        graph = ok.build()
        assert graph is not None


class TestQuantumRebindDetectionStdlib:
    """Rebind behavior through stdlib APIs (qft/iqft) and element-level patterns."""

    @pytest.mark.parametrize(
        "name,gate", _STDLIB_GATES, ids=[g[0] for g in _STDLIB_GATES]
    )
    def test_vector_whole_overwrite_rejected(self, name, gate):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(
                qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]
            ) -> qm.Vector[qm.Qubit]:
                qs1 = gate(qs2)
                return qs1

    @pytest.mark.parametrize(
        "name,gate", _STDLIB_GATES, ids=[g[0] for g in _STDLIB_GATES]
    )
    def test_vector_whole_self_update_allowed(self, name, gate):
        @qkernel
        def ok(qs1: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            qs1 = gate(qs1)
            return qs1

        graph = ok.build()
        assert graph is not None

    @pytest.mark.parametrize(
        "name,gate", _STDLIB_GATES, ids=[g[0] for g in _STDLIB_GATES]
    )
    def test_vector_whole_new_binding_allowed(self, name, gate):
        @qkernel
        def ok(
            qs1: qm.Vector[qm.Qubit], qs2: qm.Vector[qm.Qubit]
        ) -> qm.Vector[qm.Qubit]:
            out = gate(qs2)
            return out

        graph = ok.build()
        assert graph is not None

    def test_vector_element_direct_assign_via_frontend_gate_rejected(self):
        @qkernel
        def bad() -> qm.Vector[qm.Qubit]:
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            qs1[0] = qm.h(qs2[0])
            qs2[1] = qm.h(qs2[0])
            return qs1

        with pytest.raises(_DIRECT_ELEMENT_FORBIDDEN_ERRORS):
            bad.build()

    def test_vector_element_direct_assign_via_qkernel_rejected(self):
        k1, _, _ = _make_rebind_subkernels()

        @qkernel
        def bad() -> qm.Vector[qm.Qubit]:
            qs1 = qubit_array(2, "qs1")
            qs2 = qubit_array(2, "qs2")
            qs1[0] = k1(qs2[0])
            qs2[1] = k1(qs2[0])
            return qs1

        with pytest.raises(_DIRECT_ELEMENT_FORBIDDEN_ERRORS):
            bad.build()

    def test_vector_element_self_update_allowed(self):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            qs[0] = qm.h(qs[0])
            return qs

        graph = ok.build()
        assert graph is not None

    def test_vector_element_new_binding_allowed(self):
        @qkernel
        def ok() -> qm.Vector[qm.Qubit]:
            qs = qubit_array(2, "qs")
            new_q = qm.h(qs[0])
            qs[0] = new_q
            return qs

        graph = ok.build()
        assert graph is not None


class TestQuantumRebindFreshAllocation:
    """Rebinding an existing quantum variable to a freshly allocated qubit."""

    def test_scalar_fresh_allocation_rejected(self):
        """`q = qm.qubit("s")` silently discards the parameter — rejected."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q = qm.qubit("s")
                return q

        msg = str(exc.value)
        assert "freshly allocated quantum value" in msg
        assert "'q'" in msg

    def test_array_fresh_allocation_rejected(self):
        """`qs = qm.qubit_array(...)` over an existing Vector param — rejected."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(qs: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
                qs = qm.qubit_array(2, "qs2")
                return qs

        assert "freshly allocated quantum value" in str(exc.value)

    def test_unqualified_constructor_rejected(self):
        """`qubit_array(...)` without `qm.` prefix is also recognized."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(qs: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
                qs = qubit_array(2, "fresh")
                return qs

        assert "freshly allocated quantum value" in str(exc.value)

    def test_fresh_then_alias_rejected(self):
        """Constructor result tracked as origin: `tmp = qubit(...); q = tmp` rejected."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                tmp = qm.qubit("s")
                q = tmp
                return q

        assert "alias of a different quantum variable" in str(exc.value)

    def test_two_back_to_back_fresh_allocations_rejected(self):
        """`tmp = qubit("a"); tmp = qubit("b")` rejected (origin tracking on tmp)."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad() -> qm.Qubit:
                tmp = qm.qubit("a")
                tmp = qm.qubit("b")
                return tmp

        assert "freshly allocated quantum value" in str(exc.value)

    def test_throwaway_underscore_allowed_for_repeated_allocation(self):
        """`_ = qubit(...)` may repeat; the underscore is a Python throwaway."""

        @qkernel
        def ok() -> qm.Bit:
            _ = qm.qubit("ancilla_a")
            _ = qm.qubit("ancilla_b")
            q = qm.qubit("real")
            q = qm.x(q)
            return qm.measure(q)

        # Decoration must succeed; build must succeed too (the IR-level
        # check rightly allows the unused qubits to be allocated).
        assert ok.build() is not None

    def test_throwaway_underscore_in_tuple_unpack_allowed(self):
        """``_, real = some_call(...)`` does not flag and does not track ``_``."""

        @qkernel
        def two_qubits() -> tuple[qm.Qubit, qm.Qubit]:
            q1 = qm.qubit("q1")
            q2 = qm.qubit("q2")
            return q1, q2

        @qkernel
        def ok() -> qm.Bit:
            _, real = two_qubits()
            real = qm.x(real)
            return qm.measure(real)

        assert ok.build() is not None

    def test_throwaway_underscore_consumes_measure_arg(self):
        """``_ = qm.measure(q)`` still consumes ``q`` so a later fresh ``q`` is OK."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Bit:
            _ = qm.measure(q)  # consumes q's origin even though LHS is throwaway
            q = qm.qubit("fresh")
            return qm.measure(q)

        assert ok.name == "ok"


class TestQuantumRebindAnnAssign:
    """Annotated assignment (`q: qm.Qubit = ...`) goes through the same rules."""

    def test_annotated_fresh_allocation_rejected(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q: qm.Qubit = qm.qubit("s")
                return q

        assert "freshly allocated quantum value" in str(exc.value)

    def test_annotated_quantum_arg_rejected(self):
        with pytest.raises(QubitRebindError):

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a: qm.Qubit = qm.h(b)
                return a

    def test_annotated_self_update_allowed(self):
        @qkernel
        def ok(q: qm.Qubit) -> qm.Qubit:
            q: qm.Qubit = qm.h(q)
            return q

        assert ok.build() is not None


class TestQuantumRebindChainedAssignment:
    """`q1 = q2 = expr` cannot be statically verified as self-update."""

    def test_chained_with_existing_quantum_rejected(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = b = qm.qubit("s")  # noqa: F841 — `b` is the rebind target under test
                return a

        msg = str(exc.value)
        assert "chained assignment" in msg

    def test_chained_with_no_existing_quantum_allowed(self):
        """No-existing-quantum chained assignment falls through silently."""

        @qkernel
        def ok() -> qm.Qubit:
            a = b = qm.qubit("c")  # noqa: F841 — exercising chained-assign analysis
            return a

        # Currently allowed: neither a nor b was previously quantum.
        # (Tracking through chained assigns is intentionally out of
        # scope for this analyzer pass — see RebindViolation docstring.)
        # build() may still fail later for unrelated reasons; the
        # important assertion is that @qkernel itself does not raise
        # QubitRebindError.
        assert ok.name == "ok"


class TestQuantumRebindTupleLiteralRHS:
    """`a, b = (e1, e2)` element-wise check with permutation special case."""

    def test_fresh_allocations_rejected(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = (qm.qubit("a"), qm.qubit("b"))
                return q1, q2

        assert "freshly allocated quantum value" in str(exc.value)

    def test_name_only_swap_allowed(self):
        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = (q2, q1)
            return q1, q2

        assert ok.build() is not None

    def test_call_swap_allowed(self):
        @qkernel
        def ok(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q1, q2 = (qm.h(q2), qm.h(q1))
            return q1, q2

        assert ok.build() is not None

    def test_mixed_fresh_and_self_rejected(self):
        """`q1, q2 = (qm.qubit(...), qm.h(q1))` flags the fresh allocation on q1."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q1: qm.Qubit, q2: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
                q1, q2 = (qm.qubit("fresh"), qm.h(q1))
                return q1, q2

        # First violation reported should be the fresh allocation on q1.
        assert "freshly allocated quantum value" in str(exc.value)


class TestQuantumRebindMeasureConsumesAliases:
    """`measure` / `expval` pops every alias of the consumed origin."""

    def test_measure_then_reassign_param_allowed(self):
        """`bit = measure(q); q = qm.qubit(...)` must not trigger a false rebind."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Bit:
            bit = qm.measure(q)
            q = qm.qubit("fresh")
            _ = qm.measure(q)
            return bit

        # The check must complete without QubitRebindError; later
        # passes may still reject for unrelated reasons (e.g. unused
        # second measurement), so only assert decoration succeeded.
        assert ok.name == "ok"

    def test_measure_via_alias_consumes_origin(self):
        """`alias = q; measure(alias); q = qm.qubit(...)` is allowed (alias-aware pop)."""

        @qkernel
        def ok(q: qm.Qubit) -> qm.Bit:
            alias = q
            bit = qm.measure(alias)
            q = qm.qubit("fresh")
            _ = qm.measure(q)
            return bit

        assert ok.name == "ok"

    def test_measure_does_not_pop_unrelated_quantum(self):
        """Measuring `q1` must not silence subsequent rebind detection on `q2`."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q1: qm.Qubit, q2: qm.Qubit) -> qm.Bit:
                bit = qm.measure(q1)
                q2 = qm.qubit("fresh")  # noqa: F841 — rebind target under test
                return bit

        assert "freshly allocated quantum value" in str(exc.value)

    def test_per_element_measure_does_not_drop_array_origin(self):
        """Subscript-arg ``measure(qs[0])`` consumes only the element,
        so a later whole-array rebind still raises."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad() -> qm.Bit:
                qs = qubit_array(2, "qs")
                bit = qm.measure(qs[0])
                qs = qubit_array(2, "fresh")  # noqa: F841 — rebind target under test
                return bit

        assert "freshly allocated quantum value" in str(exc.value)

    def test_whole_array_measure_then_rebind_allowed(self):
        """Name-arg ``measure(qs)`` consumes the whole array; the
        subsequent ``qs = qubit_array(...)`` is allowed because the
        original array was fully consumed, not silently discarded."""

        @qkernel
        def ok() -> qm.Vector[qm.Bit]:
            qs = qubit_array(2, "qs")
            _ = qm.measure(qs)
            qs = qubit_array(2, "fresh")
            return qm.measure(qs)

        assert ok.name == "ok"


class TestQuantumRebindBranchScopeContract:
    """Lock in the documented branch-scope behavior for ``if`` / ``for`` / ``while``.

    The analyzer suppresses violations detected inside an ``if`` /
    ``for`` / ``while`` body so that legitimate compile-time-if
    dead-branch patterns (``if flag: ... ; else: alt = qubit_array(...);
    q = alt``) decorate successfully. Runtime-branch and loop-body
    silent discards are rejected later, at the IR layer, by
    ``reject_control_flow_quantum_discard`` (see
    ``tests/circuit/test_branch_quantum_discard.py``); decoration time
    must stay silent for both so the compile-time idiom keeps working —
    see ``QubitRebindError`` and
    ``QuantumRebindAnalyzer._visit_branch_scope`` docstrings. These
    tests lock that contract in so a future change that re-enables
    branch-internal raising (or breaks the legitimate dead-branch
    pattern) is caught.
    """

    def test_top_level_rebind_still_flagged(self):
        """Outside any branch, a fresh-allocation rebind still raises."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q = qm.qubit("fresh")
                return q

        assert "freshly allocated quantum value" in str(exc.value)

    def test_rebind_inside_if_branch_silenced(self):
        """A rebind nested in an ``if`` body is NOT raised at decoration time."""

        @qkernel
        def silenced(q: qm.Qubit, flag: bool) -> qm.Bit:
            if flag:
                # If raised, this would be a fresh_allocation violation
                # on the parameter ``q``. The analyzer silences it so
                # the compile-time-if dead-branch case still works.
                q = qm.qubit("fresh")
            return qm.measure(q)

        # Decoration must succeed; .build() is not asserted here because
        # the kernel has a runtime ``flag`` parameter and the IR-level
        # pipeline rightly cannot resolve it without bindings.
        assert silenced.name == "silenced"

    def test_compile_time_if_dead_branch_array_rebind_allowed(self):
        """Compile-time-if dead-branch ``q = alt`` (the failing-CI pattern) decorates."""
        flag = True  # noqa: F841 — closure-captured compile-time constant

        @qkernel
        def circuit() -> qm.Bit:
            q = qm.qubit("q")
            if flag:
                q = qm.x(q)
            else:
                alt = qm.qubit("alt")
                q = alt
            return qm.measure(q)

        # Decoration must succeed; whether downstream lowering can fold
        # the if is a transpiler-level concern, not asserted here.
        assert circuit.name == "circuit"

    def test_rebind_inside_for_body_silenced(self):
        """A rebind nested in a ``for`` body is NOT raised at decoration time."""

        @qkernel
        def silenced(q: qm.Qubit) -> qm.Bit:
            for _ in qm.range(2):
                q = qm.qubit("fresh")
            return qm.measure(q)

        assert silenced.name == "silenced"

    def test_rebind_after_if_still_flagged(self):
        """Branch suppression does not leak out: a rebind after ``if`` is raised."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit, flag: bool) -> qm.Qubit:
                if flag:
                    q = qm.h(q)  # self-update — OK
                q = qm.qubit("fresh")  # outside the if — rebind, flag this
                return q

        assert "freshly allocated quantum value" in str(exc.value)


class TestQuantumRebindErrorMessageDispatch:
    """Each source_kind formats its own (pattern, reason, fix) triple."""

    def test_direct_alias_message(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = b
                return a

        msg = str(exc.value)
        assert "an alias of a different quantum variable 'b'" in msg
        assert "Use a new variable: new_var = b" in msg

    def test_quantum_arg_message(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(a: qm.Qubit, b: qm.Qubit) -> qm.Qubit:
                a = qm.h(b)
                return a

        msg = str(exc.value)
        assert "different quantum variable 'b'" in msg
        assert "Use self-update: a = h(a, ...)" in msg

    def test_fresh_allocation_message(self):
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q = qm.qubit("s")
                return q

        msg = str(exc.value)
        assert "freshly allocated quantum value" in msg
        assert "Bind the new allocation to a new name" in msg

    def test_message_line_is_body_relative(self):
        """The ``line N`` in the rendered error message is 1-based
        relative to the function body (first body statement is line 1),
        not relative to the ``inspect.getsource`` snippet which would
        also count the decorator / ``def`` lines."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q = qm.qubit("s")  # body line 1
                return q

        msg = str(exc.value)
        # First body statement is line 1; never 3 (snippet line of the
        # assign for a typical decorator + def + body kernel).
        assert "at body line 1 " in msg
        assert "first statement of the function body as line 1" in msg

    def test_message_line_offsets_for_later_body_statements(self):
        """A violation on the second body statement reports body line 2."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit, r: qm.Qubit) -> qm.Qubit:
                r = qm.h(r)  # body line 1 — self-update, OK
                q = qm.qubit("s")  # body line 2 — flagged
                return q

        assert "at body line 2 " in str(exc.value)

    def test_message_line_robust_to_blank_lines_before_body(self):
        """Blank / comment-only lines between the ``def`` and the
        first body statement must not shift the reported body line:
        the actual first body statement is still line 1."""
        with pytest.raises(QubitRebindError) as exc:
            # fmt: off
            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:

                # leading comment / blank lines before the first body
                # statement — these must not be counted as body lines.

                q = qm.qubit("s")  # first actual body statement
                return q
            # fmt: on

        assert "at body line 1 " in str(exc.value)

    def test_subscript_alias_pattern_renders_full_rhs(self):
        """A ``Subscript`` source like ``a = qs[0]`` renders the full
        subscript in the offending-code pattern, not just the base
        array name. The underlying quantum variable name in the
        reason wording is still the base array (``qs``)."""
        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(a: qm.Qubit, qs: qm.Vector[qm.Qubit]) -> qm.Qubit:
                a = qs[0]  # noqa: F841 — rebind target
                return a

        msg = str(exc.value)
        # The pattern shows the actual RHS source, including the index.
        assert "'a = qs[0]' overwrites" in msg
        # The reason mentions the underlying array name (the quantum
        # variable being aliased from), not the indexed expression.
        assert "an alias of a different quantum variable 'qs'" in msg
        # The fix suggestion uses the full RHS too.
        assert "Use a new variable: new_var = qs[0]" in msg

    def test_unknown_call_message(self):
        """A call with no quantum args and no recognized constructor
        kind triggers ``UNKNOWN_CALL`` and renders the matching reason
        / fix wording."""

        def some_classical_func() -> int:
            """Stand-in for an arbitrary non-Qamomile call referenced
            from a kernel body."""
            return 0

        with pytest.raises(QubitRebindError) as exc:

            @qkernel
            def bad(q: qm.Qubit) -> qm.Qubit:
                q = some_classical_func()  # noqa: F841 — rebind target
                return q

        msg = str(exc.value)
        assert "does not thread the original quantum variable" in msg
        assert (
            "Pass 'q' into the call so it is self-updated: "
            "q = some_classical_func(q, ...)"
        ) in msg
        assert "Or bind the new value to a different name" in msg


class TestDuplicateQuantumCallArgs:
    """Binding the same qubit register to two sub-kernel parameters must raise.

    Regression tests for the deferred-view-consumption hole: identical
    ``VectorView`` arguments used to collapse to a single
    ``input_view_metas`` entry in ``QKernel.__call__``, silently aliasing
    both formal registers onto the same physical qubits (or crashing with
    a raw backend error once the callee entangled them).
    """

    @staticmethod
    def _pair_kernel():
        """Build a fresh two-register sub-kernel (H on ``a``, X on ``b``)."""

        @qkernel
        def pair(
            a: qm.Vector[qm.Qubit], b: qm.Vector[qm.Qubit]
        ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
            a = qm.h(a)
            b = qm.x(b)
            return a, b

        return pair

    def test_same_view_twice_raises(self):
        """The same VectorView bound to two parameters raises instead of
        silently aliasing both formal registers onto the same qubits."""
        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            v = q[0:2]
            x, _y = pair(v, v)  # same view twice
            q[0:2] = x
            return qm.measure(q)

        with pytest.raises(
            QubitConsumedError, match="backed by the same qubit register"
        ) as exc_info:
            circuit.build()
        assert "'a' and 'b'" in str(exc_info.value)

    def test_same_view_twice_entangling_callee_raises(self):
        """An entangling callee with an aliased view pair raises a Qamomile
        affine error at trace time, not a raw backend error deep in emit."""

        @qkernel
        def entangle(
            a: qm.Vector[qm.Qubit], b: qm.Vector[qm.Qubit]
        ) -> tuple[qm.Vector[qm.Qubit], qm.Vector[qm.Qubit]]:
            a0 = a[0]
            b0 = b[0]
            a0, b0 = qm.cx(a0, b0)
            a[0] = a0
            b[0] = b0
            return a, b

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            v = q[0:2]
            x, _y = entangle(v, v)  # same view twice
            q[0:2] = x
            return qm.measure(q)

        with pytest.raises(
            QubitConsumedError, match="backed by the same qubit register"
        ):
            circuit.build()

    def test_stale_and_live_view_versions_raise(self):
        """A stale SSA version and its live successor passed together raise.

        A broadcast gate on a full view keeps the same backing ``Value``
        (uuid preserved), so ``v`` and ``v2`` collide on the same-uuid
        alias guard; either way the program is rejected (previously it
        emitted the caller's gate twice on the same wires)."""
        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            v = q[0:2]
            v2 = qm.h(v)
            x, _y = pair(v, v2)  # stale + live version of one register
            q[0:2] = x
            return qm.measure(q)

        with pytest.raises(
            QubitConsumedError, match="backed by the same qubit register"
        ):
            circuit.build()

    def test_consumed_view_arg_raises(self):
        """An already-consumed view passed as a call argument raises the
        standard consumed error instead of slipping past deferred
        consumption."""
        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            v = q[0:2]
            _v2 = qm.h(v)  # consumes v
            x, y = pair(v, q[2:4])
            q[0:2] = x
            q[2:4] = y
            return qm.measure(q)

        with pytest.raises(QubitConsumedError, match="already consumed by 'H'"):
            circuit.build()

    def test_same_vector_twice_raises(self):
        """A whole Vector bound to two parameters is rejected with the
        argument-aliasing wording."""
        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            x, _y = pair(q, q)  # same Vector twice
            return qm.measure(x)

        with pytest.raises(
            QubitConsumedError, match="backed by the same qubit register"
        ):
            circuit.build()

    @pytest.mark.parametrize("use_phase_wrapper", [False, True])
    def test_parent_vector_and_borrowed_element_raise_without_consuming(
        self,
        use_phase_wrapper: bool,
    ):
        """A whole register and one of its elements cannot cross one call."""

        @qkernel
        def mixed(
            register: qm.Vector[qm.Qubit],
            element: qm.Qubit,
        ) -> tuple[qm.Vector[qm.Qubit], qm.Qubit]:
            """Return a register and scalar argument unchanged.

            Args:
                register (qm.Vector[qm.Qubit]): Register argument.
                element (qm.Qubit): Scalar argument.

            Returns:
                tuple[qm.Vector[qm.Qubit], qm.Qubit]: Original arguments.
            """
            return register, element

        from qamomile.circuit.frontend.tracer import trace

        with trace():
            register = qubit_array(2, "q")
            element = register[0]
            callable_ = qm.global_phase(mixed, 0.25) if use_phase_wrapper else mixed
            with pytest.raises(
                QubitConsumedError,
                match="overlapping physical region",
            ):
                callable_(register, element)
            assert not register._consumed
            assert not element._consumed

    def test_specialization_failure_does_not_consume_input(self, monkeypatch):
        """A failed call-site specialization leaves affine ownership intact."""
        from qamomile.circuit.frontend import qkernel_invocation
        from qamomile.circuit.frontend.tracer import trace

        @qkernel
        def identity(q: qm.Qubit) -> qm.Qubit:
            """Return one qubit unchanged.

            Args:
                q (qm.Qubit): Input qubit.

            Returns:
                qm.Qubit: Original input qubit.
            """
            return q

        def fail_specialization(*_args, **_kwargs):
            """Raise the synthetic specialization failure.

            Args:
                *_args (object): Ignored positional arguments.
                **_kwargs (object): Ignored keyword arguments.

            Raises:
                RuntimeError: Always, to exercise pre-consume validation.
            """
            raise RuntimeError("specialization failed")

        monkeypatch.setattr(
            qkernel_invocation,
            "select_specialized_block",
            fail_specialization,
        )
        with trace():
            q = qm.qubit("q")
            with pytest.raises(RuntimeError, match="specialization failed"):
                qm.global_phase(identity, 0.25)(q)
            assert not q._consumed

    def test_disjoint_views_accepted(self):
        """Disjoint views of one array remain a valid argument pair."""
        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            x, y = pair(q[0:2], q[2:4])
            q[0:2] = x
            q[2:4] = y
            return qm.measure(q)

        block = circuit.build()
        assert block is not None

    def test_inline_rejects_duplicate_quantum_call_operands(self):
        """InlinePass rejects a hand-built InvokeOperation with duplicate args.

        This defends against IR that did not come from ``QKernel.__call__``.
        """
        from qamomile.circuit.ir.operation.callable import InvokeOperation
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.transpiler.passes.inline import InlinePass

        pair = self._pair_kernel()

        @qkernel
        def circuit() -> qm.Vector[qm.Bit]:
            q = qubit_array(4, "q")
            x, y = pair(q[0:2], q[2:4])
            q[0:2] = x
            q[2:4] = y
            return qm.measure(q)

        block = circuit.block
        call_ops = [op for op in block.operations if isinstance(op, InvokeOperation)]
        assert len(call_ops) == 1
        call_op = call_ops[0]

        # Positive control: the untampered block inlines cleanly.
        InlinePass().run(block)

        quantum_indices = [
            i
            for i, operand in enumerate(call_op.operands)
            if isinstance(operand, Value) and operand.type.is_quantum()
        ]
        assert len(quantum_indices) == 2
        call_op.operands[quantum_indices[1]] = call_op.operands[quantum_indices[0]]

        with pytest.raises(QubitConsumedError, match="binds the same quantum value"):
            InlinePass().run(block)


class TestDirectElementBorrowHandoff:
    """Operation results remain the owner of a directly borrowed slot."""

    @pytest.mark.parametrize("operation", ["gate", "project", "reset"])
    def test_destructive_followup_marks_the_slot_destroyed(self, operation: str):
        """A returned scalar can be measured without leaving a stale owner."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            register = qubit_array(1, "q")
            element = register[0]
            if operation == "gate":
                element = qm.h(element)
            elif operation == "project":
                element, _ = qm.project_z(element)
            else:
                element = qm.reset(element)
            qm.measure(element)
            with pytest.raises(QubitConsumedError, match="destructive"):
                _ = register[0]


class TestArrayIndexValidation:
    """Array indexing rejects aliases and non-integral index types early."""

    @pytest.mark.parametrize("index_kind", ["python_float", "float_handle"])
    def test_non_integer_element_index_has_clear_type_error(
        self,
        index_kind: str,
    ) -> None:
        """Floats cannot silently floor or leak an AttributeError."""

        @qkernel
        def invalid_index() -> Qubit:
            register = qubit_array(2, "q")
            index = 1.0 if index_kind == "python_float" else qm.float_(1.0)
            return register[index]  # type: ignore[index]

        with pytest.raises(TypeError, match="plain int or qmc.UInt"):
            invalid_index.build()

    def test_nested_slice_rejects_slot_borrowed_from_outer_view(self) -> None:
        """A nested slice cannot duplicate an outstanding element borrow."""

        @qkernel
        def aliased_nested_slice() -> qm.Vector[Qubit]:
            register = qubit_array(4, "q")
            outer = register[0:4]
            borrowed = outer[1]
            _ = borrowed
            return outer[0:2]

        with pytest.raises(QubitBorrowConflictError, match="already borrowed"):
            aliased_nested_slice.build()

    def test_vector_iteration_message_uses_shape(self) -> None:
        """The suggested index loop only uses APIs Vector implements."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            register = qubit_array(2, "q")
            with pytest.raises(TypeError, match=r"range\(vector\.shape\[0\]\)"):
                iter(register)
