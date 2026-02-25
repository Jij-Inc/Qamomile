"""Tests for the built-in QFT and IQFT CompositeGate classes."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.stdlib.qft import IQFT, QFT, iqft, qft


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

        assert block is not None

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].gate_type == CompositeGateType.QFT
        assert isinstance(ops[2], ReturnOperation)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_transpile_circuit(self, qiskit_transpiler, n):
        """QFT transpiles to a valid Qiskit circuit with correct qubit count."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == n
        assert len(qc.data) == n + 1  # |measurement| + QFT

        assert qc.data[0].operation.name.lower() == "qft"
        for datum in qc.data[1:]:
            assert datum.operation.name == "measure"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_uniform_sampling(self, qiskit_transpiler, n):
        """QFT on |0...0> produces approximately uniform distribution."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 2048
        job = executable.sample(qiskit_transpiler.executor(), shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        total = sum(counts.values())
        assert total == shots

        # All 2^n outcomes should appear (uniform superposition)
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

        # Each outcome should be within 3-sigma of expected (shots / 2^n)
        expected = shots / num_outcomes
        sigma = (expected * (1 - 1 / num_outcomes)) ** 0.5
        for outcome, count in counts.items():
            assert abs(count - expected) < 4 * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {4 * sigma:.0f}"
            )


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
            qs = qubit_array(3, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q2 = qs[2]
            gate = IQFT(3)
            q0, q1, q2 = gate(q0, q1, q2)
            qs[0] = q0
            qs[1] = q1
            qs[2] = q2
            return qs

        block = qiskit_transpiler.to_block(circuit)

        assert block is not None

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].gate_type == CompositeGateType.IQFT
        assert isinstance(ops[2], ReturnOperation)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_transpile_circuit(self, qiskit_transpiler, n):
        """IQFT transpiles to a valid Qiskit circuit with correct qubit count."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == n
        assert len(qc.data) == n + 1  # |measurement| + IQFT

        assert qc.data[0].operation.name.lower() == "iqft"
        for datum in qc.data[1:]:
            assert datum.operation.name == "measure"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_iqft_identity(self, qiskit_transpiler, n):
        """QFT followed by IQFT on |0...0> returns all zeros."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"n={n}: expected all zeros, got {bits} (count={count})"
            )
