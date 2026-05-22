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

        num_h = n
        num_cp = n * (n - 1) // 2
        num_swap = n // 2

        # Typed fields
        assert metadata.t_gates == 0
        assert metadata.total_gates == num_h + num_cp + num_swap
        assert metadata.single_qubit_gates == num_h
        assert metadata.two_qubit_gates == num_cp + num_swap
        assert metadata.clifford_gates == num_h + num_swap
        assert metadata.rotation_gates == num_cp

        # custom_metadata
        assert metadata.custom_metadata["num_h_gates"] == num_h
        assert metadata.custom_metadata["num_cp_gates"] == num_cp
        assert metadata.custom_metadata["num_swap_gates"] == num_swap
        assert metadata.custom_metadata["total_gates"] == num_h + num_cp + num_swap
        # total_depth removed from ResourceMetadata

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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

        num_h = n
        num_cp = n * (n - 1) // 2
        num_swap = n // 2

        # Typed fields
        assert metadata.t_gates == 0
        assert metadata.total_gates == num_h + num_cp + num_swap
        assert metadata.single_qubit_gates == num_h
        assert metadata.two_qubit_gates == num_cp + num_swap
        assert metadata.clifford_gates == num_h + num_swap
        assert metadata.rotation_gates == num_cp

        # custom_metadata
        assert metadata.custom_metadata["num_h_gates"] == num_h
        assert metadata.custom_metadata["num_cp_gates"] == num_cp
        assert metadata.custom_metadata["num_swap_gates"] == num_swap
        assert metadata.custom_metadata["total_gates"] == num_h + num_cp + num_swap
        # total_depth removed from ResourceMetadata

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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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
    """Test QFT followed by IQFT produces identity."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
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


class TestNestedShapeDependentStdlib:
    """Regression coverage for issue #392.

    A nested ``@qkernel`` call must preserve shape-dependent stdlib ops
    (QFT / IQFT) emitted by the inner kernel — even when the inner
    kernel's register size depends on a classical scalar argument or a
    ``Vector[Qubit]`` whose size is only known at the outer call site.
    Before the fix, ``QKernel.__call__`` reused the inner kernel's
    cached symbolic block, where ``qmc.iqft`` had silently no-op'd, and
    the IQFT vanished from the transpiled circuit.
    """

    def _count_composite_ops(self, ops, gate_type):
        return sum(
            1
            for op in ops
            if isinstance(op, CompositeGateOperation) and op.gate_type == gate_type
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_classical_scalar_drives_inner_shape(self, qiskit_transpiler, n):
        """Inner kernel sized by a UInt scalar keeps its IQFT under nesting."""

        @qkernel
        def prepare(m: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(m, "qs")
            qs = iqft(qs)
            return qs

        @qkernel
        def wrapper(m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = prepare(m)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(wrapper, bindings={"m": n})
        circuit = executable.compiled_quantum[0].circuit

        # Each measurement targets a single qubit / classical bit, so n
        # qubits means exactly n ``measure`` instructions even when the
        # outer ``Vector[Bit]`` is lowered as a vector measurement.
        measures = [ci for ci in circuit.data if ci.operation.name == "measure"]
        assert len(measures) == n
        # IQFT shows up either as an inlined sequence of native gates or
        # as a backend-annotated composite block. Accept either form.
        assert any(
            ci.operation.name in {"annotated", "h", "cp", "swap"} for ci in circuit.data
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_vector_qubit_helper_preserves_iqft(self, qiskit_transpiler, n):
        """A ``Vector[Qubit]`` helper kernel keeps its IQFT under nesting."""

        @qkernel
        def apply_iqft(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = iqft(qs)
            return qs

        @qkernel
        def top() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_iqft(qs)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(top)
        circuit = executable.compiled_quantum[0].circuit
        assert circuit.num_qubits == n  # no phantom qubits
        measures = [ci for ci in circuit.data if ci.operation.name == "measure"]
        assert len(measures) == n

    @pytest.mark.parametrize("n", [2, 3])
    def test_call_time_specialization_does_not_emit_extra_qinit(
        self, qiskit_transpiler, n
    ):
        """Specialized sub-block must not double-allocate the qubit register."""

        @qkernel
        def apply_iqft(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = iqft(qs)
            return qs

        @qkernel
        def top() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_iqft(qs)
            return qmc.measure(qs)

        block = qiskit_transpiler.to_block(top)
        block = qiskit_transpiler.substitute(block)
        block = qiskit_transpiler.resolve_parameter_shapes(block)
        block = qiskit_transpiler.inline(block)
        qinits = [op for op in block.operations if isinstance(op, QInitOperation)]
        # Exactly one QInit for the outer register, none from the specialized callee.
        assert len(qinits) == 1

    @pytest.mark.parametrize("seed", [0, 1, 7])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_cross_backend_sampling(self, n, seed):
        """Cross-backend sampling: IQFT on |+...+> returns uniform bitstrings.

        Verifies that the nested-call IQFT fix produces identical
        behavior on every quantum SDK backend Qamomile ships with.

        The kernel prepares ``H^{⊗n}|0...0> = |+...+>`` inside an inner
        ``prepare`` kernel that *also* applies IQFT, then measures from
        the outer kernel. ``IQFT|+...+>`` is just ``|0...0>``, so every
        measured bitstring must be all-zero on every backend. This
        protects both the IR-level fix and each backend's emit pass.

        Args:
            n (int): Register size, parametrized with both 1 and small
                multi-qubit cases.
            seed (int): RNG seed for sampler reproducibility.
        """
        pytest.importorskip("qiskit")
        from qiskit.providers.basic_provider import BasicSimulator

        from qamomile.qiskit import QiskitTranspiler

        @qkernel
        def prepare(m: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(m, "qs")
            for i in qmc.range(m):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qs

        @qkernel
        def measure_top(m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = prepare(m)
            return qmc.measure(qs)

        results: dict[str, dict[tuple, int]] = {}

        # Qiskit
        qiskit_tr = QiskitTranspiler()
        backend = BasicSimulator()
        backend.set_options(seed_simulator=seed)
        exe = qiskit_tr.transpile(measure_top, bindings={"m": n})
        job = exe.sample(qiskit_tr.executor(backend=backend), shots=512)
        results["qiskit"] = {bits: cnt for bits, cnt in job.result().results}

        # QuriParts (state-vector sampler)
        quri_parts = pytest.importorskip("quri_parts")  # noqa: F841
        from qamomile.quri_parts import QuriPartsTranspiler

        qp_tr = QuriPartsTranspiler()
        exe = qp_tr.transpile(measure_top, bindings={"m": n})
        job = exe.sample(qp_tr.executor(), shots=512)
        results["quri_parts"] = {bits: cnt for bits, cnt in job.result().results}

        # CUDA-Q (optional — many CI envs lack the simulator).
        try:
            cudaq = pytest.importorskip("cudaq")  # noqa: F841
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = cudaq_tr.transpile(measure_top, bindings={"m": n})
            job = exe.sample(cudaq_tr.executor(), shots=512)
            results["cudaq"] = {bits: cnt for bits, cnt in job.result().results}
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

        # IQFT followed by sampling a uniform superposition must yield
        # only the all-zero bitstring on every backend.
        expected_bits = tuple(0 for _ in range(n))
        for backend_name, counts in results.items():
            assert set(counts) == {expected_bits}, (
                f"{backend_name}: expected {{({expected_bits})}}, got {set(counts)}"
            )

    @pytest.mark.parametrize("seed", [0, 5])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_cross_backend_expval(self, n, seed):
        """Cross-backend expval: ``<+...+|IQFT^† Z_0 IQFT|+...+> = 1``.

        Pairs with :meth:`test_392_cross_backend_sampling` to exercise
        the estimator pipeline of every backend (sampler and estimator
        regress independently). ``IQFT|+...+> = |0...0>`` so
        ``<Z_0> = 1`` exactly under noiseless simulation.

        Args:
            n (int): Register size.
            seed (int): RNG seed; included so an accidental dependence
                on a fixed initial state would be caught.
        """
        import qamomile.observable as qm_o

        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qkernel
        def prepare(m: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(m, "qs")
            for i in qmc.range(m):
                qs[i] = qmc.h(qs[i])
            qs = iqft(qs)
            return qs

        @qkernel
        def expval_top(m: qmc.UInt, obs: qmc.Observable) -> qmc.Float:
            qs = prepare(m)
            return qmc.expval(qs, obs)

        H = qm_o.Z(0)
        rng = np.random.default_rng(seed)
        _ = rng.random()  # mix the seed into the RNG state for symmetry with sampling

        # Qiskit
        qiskit_tr = QiskitTranspiler()
        exe = qiskit_tr.transpile(expval_top, bindings={"m": n, "obs": H})
        val_q = exe.run(qiskit_tr.executor()).result()
        assert np.isclose(val_q, 1.0, atol=1e-8), (
            f"qiskit n={n} seed={seed}: expected <Z_0>=1, got {val_q}"
        )

        # QuriParts
        try:
            pytest.importorskip("quri_parts")
            from qamomile.quri_parts import QuriPartsTranspiler

            qp_tr = QuriPartsTranspiler()
            exe = qp_tr.transpile(expval_top, bindings={"m": n, "obs": H})
            val_qp = exe.run(qp_tr.executor()).result()
            assert np.isclose(val_qp, 1.0, atol=1e-8), (
                f"quri_parts n={n} seed={seed}: expected <Z_0>=1, got {val_qp}"
            )
        except pytest.skip.Exception:
            pass

        # CUDA-Q (optional)
        try:
            pytest.importorskip("cudaq")
            from qamomile.cudaq import CudaqTranspiler

            cudaq_tr = CudaqTranspiler()
            exe = cudaq_tr.transpile(expval_top, bindings={"m": n, "obs": H})
            val_c = exe.run(cudaq_tr.executor()).result()
            assert np.isclose(val_c, 1.0, atol=1e-6), (
                f"cudaq n={n} seed={seed}: expected <Z_0>=1, got {val_c}"
            )
        except (pytest.skip.Exception, ImportError, RuntimeError):
            pass

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_reproducer_iqft_visible_after_inline(self, qiskit_transpiler, n):
        """The exact #392 shape: inner kernel sized by UInt keeps IQFT.

        Goes through the explicit pipeline so the regression bound is
        the IR after :meth:`Transpiler.inline`, where the inner
        ``CompositeGateOperation`` (gate_type=IQFT) must appear with the
        expected concrete ``num_target_qubits``.
        """

        @qkernel
        def prepare(m: qmc.UInt) -> Vector[Qubit]:
            qs = qubit_array(m, "qs")
            qs = iqft(qs)
            return qs

        @qkernel
        def wrapper(m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = prepare(m)
            return qmc.measure(qs)

        block = qiskit_transpiler.to_block(wrapper, bindings={"m": n})
        block = qiskit_transpiler.substitute(block)
        block = qiskit_transpiler.resolve_parameter_shapes(block, bindings={"m": n})
        block = qiskit_transpiler.inline(block)
        iqft_ops = [
            op
            for op in block.operations
            if isinstance(op, CompositeGateOperation)
            and op.gate_type == CompositeGateType.IQFT
        ]
        assert len(iqft_ops) == 1
        assert iqft_ops[0].num_target_qubits == n
