"""Tests for the built-in QFT and IQFT CompositeGate classes."""

import numpy as np
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
from tests.circuit.conftest import run_statevector


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
        assert "num_cp_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_cp_gates"] == n * (n - 1) // 2
        assert "num_swap_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_swap_gates"] == n // 2
        assert "total_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["total_gates"] == n + n * (n - 1) // 2 + n // 2
        assert "depth" in metadata.custom_metadata
        assert metadata.custom_metadata["depth"] == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_resources_symbolic(self, n):
        """QFT resources are correct when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build(num=n)

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.QFT
        assert ops[1].custom_name == "qft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel(self, n):
        """QFT can be used in a qkernel via qft() factory."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.QFT
        assert ops[1].custom_name == "qft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel_symbolic(self, n):
        """QFT works in a qkernel when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build(num=n)

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.QFT
        assert ops[1].custom_name == "qft"

    def test_builds_ir(self, qiskit_transpiler):
        """QFT builds correct IR with native gate type."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(3, "qs")
            gate = QFT(3)
            qs[0], qs[1], qs[2] = gate(qs[0], qs[1], qs[2])
            return qs

        block = qiskit_transpiler.to_block(circuit)

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == 3
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.QFT
        assert ops[1].custom_name == "qft"
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

        from qiskit.circuit.library import QFTGate
        from qiskit.circuit.measure import Measure

        assert isinstance(qc.data[0].operation, QFTGate)
        for datum in qc.data[1:]:
            assert isinstance(datum.operation, Measure)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_uniform_statevector(self, qiskit_transpiler, n):
        """QFT on |0...0> produces uniform superposition (statevector check)."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        # QFT on |0...0> gives equal amplitudes 1/sqrt(2^n)
        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_transpile_circuit_symbolic(self, qiskit_transpiler, n):
        """QFT transpiles correctly when n is bound at transpile time."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        assert qc.num_qubits == n
        assert len(qc.data) == n + 1  # |measurement| + QFT

        from qiskit.circuit.library import QFTGate
        from qiskit.circuit.measure import Measure

        assert isinstance(qc.data[0].operation, QFTGate)
        for datum in qc.data[1:]:
            assert isinstance(datum.operation, Measure)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_uniform_statevector_symbolic(self, qiskit_transpiler, n):
        """QFT on |0...0> produces uniform superposition (symbolic n)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        sv = run_statevector(qc)

        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
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
        assert "num_cp_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_cp_gates"] == n * (n - 1) // 2
        assert "num_swap_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["num_swap_gates"] == n // 2
        assert "total_gates" in metadata.custom_metadata
        assert metadata.custom_metadata["total_gates"] == n + n * (n - 1) // 2 + n // 2
        assert "depth" in metadata.custom_metadata
        assert metadata.custom_metadata["depth"] == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_resources_symbolic(self, n):
        """IQFT resources are correct when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build(num=n)

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.IQFT
        assert ops[1].custom_name == "iqft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel(self, n):
        """IQFT can be used in a qkernel via iqft() factory."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(n, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.IQFT
        assert ops[1].custom_name == "iqft"

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel_symbolic(self, n):
        """IQFT works in a qkernel when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build(num=n)

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == n
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.IQFT
        assert ops[1].custom_name == "iqft"

    def test_builds_ir(self, qiskit_transpiler):
        """IQFT builds correct IR with native gate type."""

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(3, "qs")
            gate = IQFT(3)
            qs[0], qs[1], qs[2] = gate(qs[0], qs[1], qs[2])
            return qs

        block = qiskit_transpiler.to_block(circuit)

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].num_target_qubits == 3
        assert ops[1].num_control_qubits == 0
        assert ops[1].gate_type == CompositeGateType.IQFT
        assert ops[1].custom_name == "iqft"
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

        from qiskit.circuit import AnnotatedOperation, InverseModifier
        from qiskit.circuit.library import QFTGate
        from qiskit.circuit.measure import Measure

        iqft_op = qc.data[0].operation
        assert isinstance(iqft_op, AnnotatedOperation)
        assert isinstance(iqft_op.base_op, QFTGate)
        assert any(isinstance(m, InverseModifier) for m in iqft_op.modifiers)
        for datum in qc.data[1:]:
            assert isinstance(datum.operation, Measure)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_zero_statevector(self, qiskit_transpiler, n):
        """IQFT on uniform superposition produces |0...0> (statevector check)."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            for i in range(n):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        # IQFT on H^n|0> should give |0...0>
        expected = np.zeros(2**n)
        expected[0] = 1.0
        assert np.allclose(np.abs(sv), expected, atol=1e-8), (
            f"n={n}: expected |0...0>, got {sv}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_transpile_circuit_symbolic(self, qiskit_transpiler, n):
        """IQFT transpiles correctly when n is bound at transpile time."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        assert qc.num_qubits == n
        assert len(qc.data) == n + 1  # |measurement| + IQFT

        from qiskit.circuit import AnnotatedOperation, InverseModifier
        from qiskit.circuit.library import QFTGate
        from qiskit.circuit.measure import Measure

        iqft_op = qc.data[0].operation
        assert isinstance(iqft_op, AnnotatedOperation)
        assert isinstance(iqft_op.base_op, QFTGate)
        assert any(isinstance(m, InverseModifier) for m in iqft_op.modifiers)
        for datum in qc.data[1:]:
            assert isinstance(datum.operation, Measure)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_zero_statevector_symbolic(self, qiskit_transpiler, n):
        """IQFT on uniform superposition produces |0...0> (symbolic n)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            for i in qmc.range(num):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        sv = run_statevector(qc)

        expected = np.zeros(2**n)
        expected[0] = 1.0
        assert np.allclose(np.abs(sv), expected, atol=1e-8), (
            f"n={n}: expected |0...0>, got {sv}"
        )



class TestQFTIQFT:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_iqft_identity_statevector(self, qiskit_transpiler, n):
        """QFT followed by IQFT on |0...0> returns |0...0> (statevector check)."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        # QFT followed by IQFT should be identity
        expected = np.zeros(2**n)
        expected[0] = 1.0
        assert np.allclose(np.abs(sv), expected, atol=1e-8), (
            f"n={n}: expected |0...0>, got {sv}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_iqft_identity(self, qiskit_transpiler, seeded_executor, n):
        """QFT followed by IQFT on |0...0> returns all zeros."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        job = executable.sample(seeded_executor, shots=1024)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"n={n}: expected all zeros, got {bits} (count={count})"
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_iqft_identity_statevector_symbolic(self, qiskit_transpiler, n):
        """QFT then IQFT is identity (symbolic n, statevector check)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        sv = run_statevector(qc)

        expected = np.zeros(2**n)
        expected[0] = 1.0
        assert np.allclose(np.abs(sv), expected, atol=1e-8), (
            f"n={n}: expected |0...0>, got {sv}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_qft_iqft_identity_symbolic(self, qiskit_transpiler, seeded_executor, n):
        """QFT then IQFT returns all zeros (symbolic n, sampling)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            qs = iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit, bindings={"num": n})
        job = executable.sample(seeded_executor, shots=1024)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"n={n}: expected all zeros, got {bits} (count={count})"
            )
