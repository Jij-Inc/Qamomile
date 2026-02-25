"""Tests for the built-in QFT and IQFT CompositeGate classes."""

import pytest

from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.stdlib.qft import QFT, IQFT, qft, iqft


@pytest.fixture
def qiskit_transpiler():
    """Get Qiskit transpiler."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


class TestQFT:
    """Test the QFT CompositeGate class."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_class_attributes(self, n):
        """QFT class has correct attributes."""
        gate = QFT(n)
        assert gate.num_target_qubits == n
        assert gate.num_control_qubits == 0
        assert gate.gate_type == CompositeGateType.QFT
        assert gate.custom_name == "qft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_resources(self, n):
        """QFT returns correct resource metadata."""
        gate = QFT(n)
        metadata = gate.get_resource_metadata()

        assert metadata is not None
        assert metadata.t_gate_count == 0
        assert "num_h_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_h_gates"] == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel(self, n):
        """QFT can be used in a qkernel via qft() factory."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "qft"
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0

    def test_builds_ir(self, qiskit_transpiler):
        """QFT builds correct IR with native gate type."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q2 = qs[2]
            gate = QFT(3)
            q0, q1, q2 = gate(q0, q1, q2)
            qs[0] = q0
            qs[1] = q1
            qs[2] = q2
            return qs

        block = qiskit_transpiler.to_block(circuit)

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.QFT


class TestIQFT:
    """Test the IQFT CompositeGate class."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_class_attributes(self, n):
        """IQFT class has correct attributes."""
        gate = IQFT(n)
        assert gate.num_target_qubits == n
        assert gate.num_control_qubits == 0
        assert gate.gate_type == CompositeGateType.IQFT
        assert gate.custom_name == "iqft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_resources(self, n):
        """IQFT returns correct resource metadata."""
        gate = IQFT(n)
        metadata = gate.get_resource_metadata()

        assert metadata is not None
        assert metadata.t_gate_count == 0
        assert "num_h_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_h_gates"] == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel(self, n):
        """IQFT can be used in a qkernel via iqft() factory."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "iqft"
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0

    def test_builds_ir(self, qiskit_transpiler):
        """IQFT builds correct IR with native gate type."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            gate = IQFT(2)
            q0, q1 = gate(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        block = qiskit_transpiler.to_block(circuit)

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.IQFT
