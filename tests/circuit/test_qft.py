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


@pytest.fixture
def qiskit_transpiler():
    """Get Qiskit transpiler."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@pytest.fixture
def seeded_executor(qiskit_transpiler):
    """Executor with fixed seed for reproducible sampling."""
    from qiskit_aer import AerSimulator

    return qiskit_transpiler.executor(backend=AerSimulator(seed_simulator=901))


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
    def test_resources_symbolic(self, n):
        """QFT resources are correct when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build(num=n)
        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].num_target_qubits == n

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

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel_symbolic(self, n):
        """QFT works in a qkernel when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qs

        block = circuit.build(num=n)
        assert block is not None

        ops = block.operations
        assert isinstance(ops[0], QInitOperation)
        composite_ops = [op for op in ops if isinstance(op, CompositeGateOperation)]
        assert len(composite_ops) == 1
        assert composite_ops[0].custom_name == "qft"
        assert composite_ops[0].num_target_qubits == n
        assert composite_ops[0].num_control_qubits == 0

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
    def test_uniform_sampling(self, qiskit_transpiler, seeded_executor, n):
        """QFT on |0...0> produces approximately uniform distribution."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 4096
        job = executable.sample(seeded_executor, shots=shots)
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
            assert abs(count - expected) < 3 * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {3 * sigma:.0f}"
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

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_uniform_sampling_symbolic(self, qiskit_transpiler, seeded_executor, n):
        """QFT on |0...0> produces uniform distribution (symbolic n)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = qft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit, bindings={"num": n})
        shots = 4096
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        total = sum(counts.values())
        assert total == shots

        assert len([c for c in counts.values() if c > 0]) == num_outcomes

        expected = shots / num_outcomes
        sigma = (expected * (1 - 1 / num_outcomes)) ** 0.5
        for outcome, count in counts.items():
            assert abs(count - expected) < 3 * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {3 * sigma:.0f}"
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
    def test_resources_symbolic(self, n):
        """IQFT resources are correct when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build(num=n)
        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].num_target_qubits == n

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

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_in_qkernel_symbolic(self, n):
        """IQFT works in a qkernel when n is symbolic (bound at build time)."""

        @qkernel
        def circuit(num: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(num, "qs")
            qs = iqft(qs)
            return qs

        block = circuit.build(num=n)
        assert block is not None

        ops = block.operations
        assert isinstance(ops[0], QInitOperation)
        composite_ops = [op for op in ops if isinstance(op, CompositeGateOperation)]
        assert len(composite_ops) == 1
        assert composite_ops[0].custom_name == "iqft"
        assert composite_ops[0].num_target_qubits == n
        assert composite_ops[0].num_control_qubits == 0

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
    def test_zero_sampling(self, qiskit_transpiler, seeded_executor, n):
        """IQFT on the uniform distribution produces |0...0>."""

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            for i in range(n):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        total = sum(counts.values())
        assert total == shots

        assert len([c for c in counts.values() if c > 0]) == 1

        # Each outcome should be within 3-sigma of expected (shots / 2^n)
        for outcome, count in counts.items():
            if count == 0:
                assert outcome != 0, (
                    f"n={n}: expected outcome 0, got {outcome} (count={count})"
                )
            else:
                assert outcome == 0, (
                    f"n={n}: expected outcome 0, got {outcome} (count={count})"
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

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_zero_sampling_symbolic(self, qiskit_transpiler, seeded_executor, n):
        """IQFT on uniform distribution produces |0...0> (symbolic n)."""

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            for i in qmc.range(num):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit, bindings={"num": n})
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        total = sum(counts.values())
        assert total == shots

        assert len([c for c in counts.values() if c > 0]) == 1

        for outcome, count in counts.items():
            if count == 0:
                assert outcome != 0, (
                    f"n={n}: expected outcome 0, got {outcome} (count={count})"
                )
            else:
                assert outcome == 0, (
                    f"n={n}: expected outcome 0, got {outcome} (count={count})"
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
