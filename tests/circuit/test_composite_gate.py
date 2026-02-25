"""Tests for the CompositeGate API."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.operation import QInitOperation


class TestCompositeGateDefinition:
    """Test that CompositeGate subclasses can be defined correctly."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_simple_composite_gate_definition(self, n):
        """A simple CompositeGate can be defined with _decompose()."""

        class MyHadamardAll(CompositeGate):
            """Apply H to all qubits."""

            custom_name = "hadamard_all"

            def __init__(self, num_qubits: int):
                self._num_qubits = num_qubits

            @property
            def num_target_qubits(self) -> int:
                return self._num_qubits

            def _decompose(
                self, qubits: Vector[Qubit] | tuple[Qubit, ...]
            ) -> tuple[Qubit, ...]:
                result = []
                for q in qubits:
                    result.append(qmc.h(q))
                return tuple(result)

        # Create instance
        gate = MyHadamardAll(n)
        assert gate.num_target_qubits == n
        assert gate.num_control_qubits == 0
        assert gate.gate_type == CompositeGateType.CUSTOM
        assert gate.custom_name == "hadamard_all"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_composite_gate_with_resources(self, n):
        """CompositeGate can define _resources() for metadata."""

        class MyGate(CompositeGate):
            custom_name = "my_gate"

            def __init__(self, n: int):
                self._n = n

            @property
            def num_target_qubits(self) -> int:
                return self._n

            def _decompose(
                self, qubits: Vector[Qubit] | tuple[Qubit, ...]
            ) -> tuple[Qubit, ...]:
                return tuple(qmc.h(q) for q in qubits)

            def _resources(self) -> ResourceMetadata:
                return ResourceMetadata(
                    t_gate_count=10,
                    query_complexity=5,
                    custom_metadata={"num_qubits": self._n},
                )

        gate = MyGate(n)
        assert gate.num_target_qubits == n
        assert gate.num_control_qubits == 0
        assert gate.get_resource_metadata() is not None
        assert gate.custom_name == "my_gate"

        metadata = gate.get_resource_metadata()

        assert metadata is not None
        assert metadata.t_gate_count == 10
        assert metadata.query_complexity == 5
        assert metadata.custom_metadata["num_qubits"] == n


class TestCompositeGateApplication:
    """Test that CompositeGate can be applied to qubits in qkernels."""

    def test_apply_composite_gate_in_qkernel(self):
        """CompositeGate can be used inside a qkernel."""

        class DoubleH(CompositeGate):
            """Apply H twice."""

            custom_name = "double_h"

            def __init__(self):
                pass

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(
                self, qubits: Vector[Qubit] | tuple[Qubit, ...]
            ) -> tuple[Qubit, ...]:
                q = qubits[0]
                q = qmc.h(q)
                q = qmc.h(q)
                return (q,)

        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        # Should build successfully
        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2  # Should have 2 operations from the decomposition
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "double_h"
        assert ops[1].num_target_qubits == 1
        assert ops[1].num_control_qubits == 0

    def test_apply_multi_qubit_composite_gate(self):
        """Multi-qubit CompositeGate works correctly."""

        class BellPair(CompositeGate):
            """Create Bell pair."""

            custom_name = "bell_pair"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(
                self, qubits: Vector[Qubit] | tuple[Qubit, ...]
            ) -> tuple[Qubit, ...]:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return (q0, q1)

        bell = BellPair()

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0, q1 = bell(q0, q1)
            return q0, q1

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 3  # Should have 2 operations from the decomposition
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], QInitOperation)
        assert isinstance(ops[2], CompositeGateOperation)
        assert ops[2].custom_name == "bell_pair"
        assert ops[2].num_target_qubits == 2
        assert ops[2].num_control_qubits == 0


class TestQFTAndIQFTClasses:
    """Test the built-in QFT and IQFT CompositeGate classes."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_qft_class_attributes(self, n):
        """QFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import QFT

        qft = QFT(n)
        assert qft.num_target_qubits == n
        assert qft.num_control_qubits == 0
        assert qft.gate_type == CompositeGateType.QFT
        assert qft.custom_name == "qft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_iqft_class_attributes(self, n):
        """IQFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import IQFT

        iqft = IQFT(n)
        assert iqft.num_target_qubits == n
        assert iqft.num_control_qubits == 0
        assert iqft.gate_type == CompositeGateType.IQFT
        assert iqft.custom_name == "iqft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_qft_resources(self, n):
        """QFT returns correct resource metadata."""
        from qamomile.circuit.stdlib.qft import QFT

        qft = QFT(n)
        metadata = qft.get_resource_metadata()

        assert metadata is not None
        assert metadata.t_gate_count == 0  # Standard QFT uses no T gates
        assert "num_h_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_h_gates"] == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_qft_in_qkernel(self, n):
        """QFT can be used in a qkernel."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2  # Should have 2 operations from the decomposition
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "qft"
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_iqft_in_qkernel(self, n):
        """iqft() factory function works in qkernel."""
        from qamomile.circuit.stdlib.qft import iqft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2  # Should have 2 operations from the decomposition
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "iqft"
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0


class TestCompositeGateTranspilation:
    """Test that CompositeGate IR generation works correctly.

    Note: Full transpilation to Qiskit requires circuits with classical I/O.
    These tests verify that the IR is correctly generated for composite gates.
    """

    @pytest.fixture
    def qiskit_transpiler(self):
        """Get Qiskit transpiler."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()

    def test_custom_gate_builds_ir(self, qiskit_transpiler):
        """Custom CompositeGate builds correct IR."""

        class MyGate(CompositeGate):
            custom_name = "my_gate"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return (q0, q1)

        my_gate = MyGate()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = my_gate(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        # Build to Block and inline
        block = qiskit_transpiler.to_block(circuit)
        inlined = qiskit_transpiler.inline(block)

        # After inline, should have H and CX operations
        op_types = [type(op).__name__ for op in inlined.operations]
        assert "GateOperation" in op_types  # H and CX are GateOperations

    def test_qft_builds_ir(self, qiskit_transpiler):
        """QFT builds correct IR with native gate type."""
        from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
        from qamomile.circuit.stdlib.qft import QFT

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q2 = qs[2]
            qft = QFT(3)
            q0, q1, q2 = qft(q0, q1, q2)
            qs[0] = q0
            qs[1] = q1
            qs[2] = q2
            return qs

        # Build to Block (before inline)
        block = qiskit_transpiler.to_block(circuit)

        # Should have CompositeGateOperation with QFT type
        from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.QFT

    def test_no_phantom_qubits_with_array_elements(self, qiskit_transpiler):
        """qubit_array elements + CompositeGate should not create phantom qubits."""

        class BellPair(CompositeGate):
            custom_name = "bell_pair"

            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
                q0, q1 = qubits
                q0 = qmc.h(q0)
                q0, q1 = qmc.cx(q0, q1)
                return (q0, q1)

        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Bit:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = bell(q0, q1)
            return qmc.measure(q0)

        qc = qiskit_transpiler.to_circuit(circuit)
        qc.remove_final_measurements()
        assert qc.num_qubits == 2, f"Expected 2 qubits, got {qc.num_qubits}"

    def test_iqft_builds_ir(self, qiskit_transpiler):
        """IQFT builds correct IR with native gate type."""
        from qamomile.circuit.ir.operation.composite_gate import CompositeGateType
        from qamomile.circuit.stdlib.qft import IQFT

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            iqft = IQFT(2)
            q0, q1 = iqft(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        # Build to Block (before inline)
        block = qiskit_transpiler.to_block(circuit)

        # Should have CompositeGateOperation with IQFT type
        from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.IQFT


class TestCompositeGateEdgeCases:
    """Test edge cases and error handling for CompositeGate."""

    def test_wrong_num_qubits_raises_error(self):
        """Passing wrong number of qubits raises ValueError."""

        class TwoQubitGate(CompositeGate):
            @property
            def num_target_qubits(self) -> int:
                return 2

            def _decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
                return qubits

        gate = TwoQubitGate()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            # Passing 1 qubit to a 2-qubit gate
            (q,) = gate(q)  # This should raise
            return q

        with pytest.raises(ValueError, match="requires 2 target qubits"):
            circuit.build()

    def test_stub_gate_without_decomposition(self):
        """Stub gate (no _decompose) has no implementation."""
        from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation

        class StubOracle(CompositeGate):
            custom_name = "oracle"

            @property
            def num_target_qubits(self) -> int:
                return 2

            # No _decompose defined - stub gate

        oracle = StubOracle()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = oracle(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        # Build to block
        block = circuit.build()

        # Find the CompositeGateOperation
        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        # Stub gate should have no implementation
        assert composite_ops[0].has_implementation is False
        assert composite_ops[0].implementation is None
