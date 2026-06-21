"""Tests for the built-in QFT and IQFT CompositeGate classes."""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Matrix, Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
)
from qamomile.circuit.ir.operation.gate import MeasureVectorOperation
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.stdlib.qft import IQFT, QFT, iqft, qft
from tests.circuit.conftest import run_statevector

# ---------------------------------------------------------------------------
# Module-level kernels for the self-recursive + shape-dependent stdlib
# regression. Self-recursive @qkernel definitions must be at module level
# so the AST transformer can resolve the recursive name (function-local
# closures bind the name only after the decorator has run, which fires
# the "Closure variable is not yet bound" branch).
# ---------------------------------------------------------------------------


@qmc.qkernel
def _rec_iqft(depth: qmc.UInt, qs: Vector[Qubit]) -> Vector[Qubit]:
    """Apply IQFT and recurse ``depth`` more times. Used by the self-
    recursion regression test in ``TestNestedShapeDependentStdlib``."""
    qs = iqft(qs)
    if depth > qmc.uint(0):
        qs = _rec_iqft(depth - qmc.uint(1), qs)
    return qs


@qmc.qkernel
def _rec_iqft_top(depth: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Wrap ``_rec_iqft`` with a concrete-size register and measure."""
    qs = qmc.qubit_array(3, "qs")
    qs = _rec_iqft(depth, qs)
    return qmc.measure(qs)


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

    @pytest.mark.parametrize("depth", [0, 1, 2])
    def test_self_recursive_with_shape_dependent_stdlib_limitation(
        self, qiskit_transpiler, depth
    ):
        """Documents a known limitation of call-time specialization.

        ``_rec_iqft(depth, qs)`` applies ``qmc.iqft`` once and then
        recurses ``depth`` more times, so the program semantically
        applies IQFT ``(depth + 1)`` times. The call-time
        specialization in :meth:`QKernel.__call__` only fires for the
        *outermost* call: the body's recursive self-call sees
        ``self._specializing == True`` and falls back to the cached
        symbolic ``self.block`` where the inner ``qmc.iqft`` had no-op'd
        on the symbolic-shape register. After ``inline ↔ partial_eval``
        unrolling the recursion, only the outer IQFT remains in the IR.

        This test pins that limitation down. If a future change relaxes
        the ``_specializing`` guard (e.g. allows bounded
        re-specialization for self-recursive callees with concrete
        bindings), the expected IQFT count will rise to ``depth + 1``
        and this test should be updated to match — the goal here is to
        flag the behavior change consciously, not to enforce the
        current shortcut forever.
        """
        block = qiskit_transpiler.to_block(_rec_iqft_top, bindings={"depth": depth})
        block = qiskit_transpiler.substitute(block)
        block = qiskit_transpiler.resolve_parameter_shapes(
            block, bindings={"depth": depth}
        )
        block = qiskit_transpiler.inline(block)
        block = qiskit_transpiler.unroll_recursion(block, bindings={"depth": depth})
        iqft_ops = [
            op
            for op in block.operations
            if isinstance(op, CompositeGateOperation)
            and op.gate_type == CompositeGateType.IQFT
        ]
        # Documented limitation: exactly one IQFT survives regardless
        # of ``depth``. Mathematically the kernel applies IQFT
        # ``depth + 1`` times.
        assert len(iqft_ops) == 1
        assert iqft_ops[0].num_target_qubits == 3

    @pytest.mark.parametrize("n", [2, 3])
    def test_runtime_vector_bit_arg_does_not_block_other_specialization(
        self, qiskit_transpiler, n
    ):
        """A runtime ``Vector[Bit]`` co-argument must not block IQFT spec.

        ``_extract_calltime_specialization`` previously aborted the
        whole call when it saw a non-parameterizable runtime classical
        without a compile-time constant — re-introducing the original
        #392 failure mode for callees that take both a shape-relevant
        argument and a runtime classical. The extractor now
        ``continue``s past such arguments so specialization on the
        shape-relevant arg still fires, and inline-time substitution
        supplies the runtime classical Value.

        The kernel below feeds a ``Vector[Bit]`` produced by a
        measurement (so it carries no compile-time constant) into a
        helper that also takes a concrete ``Vector[Qubit]``. The
        IQFT inside the helper must still emit with the resolved
        qubit-register size.
        """

        @qkernel
        def helper(
            qs: Vector[Qubit], flags: qmc.Vector[qmc.Bit]
        ) -> tuple[Vector[Qubit], qmc.Vector[qmc.Bit]]:
            qs = iqft(qs)
            return qs, flags

        @qkernel
        def top() -> qmc.Vector[qmc.Bit]:
            q_aux = qmc.qubit_array(2, "q_aux")
            aux_bits = qmc.measure(q_aux)  # runtime Vector[Bit]
            qs = qmc.qubit_array(n, "qs")
            qs, _ = helper(qs, aux_bits)
            return qmc.measure(qs)

        block = qiskit_transpiler.to_block(top)
        block = qiskit_transpiler.substitute(block)
        block = qiskit_transpiler.resolve_parameter_shapes(block)
        block = qiskit_transpiler.inline(block)
        iqft_ops = [
            op
            for op in block.operations
            if isinstance(op, CompositeGateOperation)
            and op.gate_type == CompositeGateType.IQFT
        ]
        assert len(iqft_ops) == 1, (
            f"expected IQFT specialization to fire despite runtime "
            f"Vector[Bit] co-argument, got {len(iqft_ops)} IQFT op(s)"
        )
        assert iqft_ops[0].num_target_qubits == n

    def test_matrix_qubit_sub_kernel_is_rejected(self):
        """A nested call allocating a ``Matrix[Qubit]`` raises loudly.

        Rank>1 quantum registers are explicitly rejected (see the
        "Rank>1 quantum registers are explicitly rejected" entry of
        ``LIMITATIONS.md``): the quantum addressing path is rank-1,
        so a ``(2, 3)`` register would silently alias distinct
        elements onto the same physical qubit. The ``qubit_array``
        guard fires while the outer kernel's body is traced, before
        the nested call is reached.
        """

        @qkernel
        def inner(qs: Matrix[Qubit]) -> Matrix[Qubit]:
            return qs

        @qkernel
        def outer() -> Matrix[Qubit]:
            qs = qubit_array((2, 3), "qs")
            qs = inner(qs)
            return qs

        with pytest.raises(NotImplementedError, match="rank-2"):
            outer.build()

    @pytest.mark.parametrize("seed", [0, 1, 7])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_cross_backend_sampling(self, sdk_transpiler, n, seed):
        """Cross-backend sampling: ``IQFT|+...+>`` collapses to ``|0...0>``.

        Verifies that the nested-call IQFT fix produces identical
        behavior on every quantum SDK backend Qamomile ships with,
        parametrized via ``sdk_transpiler`` so each backend leg runs in
        the matching ``-m`` session (the cudaq leg must never load
        cudaq into a default session — see tests/_cudaq_isolation.py).

        The kernel prepares ``H^{⊗n}|0...0> = |+...+>`` inside an inner
        ``prepare`` kernel that *also* applies IQFT, then measures
        from the outer kernel. Because ``IQFT|+...+> = |0...0>``
        exactly, every measured bitstring must be the all-zero string
        on every backend (any other bitstring would indicate the
        nested-call IQFT was lost or emitted incorrectly). This
        protects both the IR-level fix and each backend's emit pass.

        Args:
            sdk_transpiler (SdkTranspilerCase): Backend label plus
                transpiler instance from the shared fixture.
            n (int): Register size, parametrized with both 1 and small
                multi-qubit cases.
            seed (int): RNG seed for sampler reproducibility (used by
                the seedable Qiskit simulator leg).
        """

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

        transpiler = sdk_transpiler.transpiler
        if sdk_transpiler.backend_name == "qiskit":
            from qiskit.providers.basic_provider import BasicSimulator

            backend = BasicSimulator()
            backend.set_options(seed_simulator=seed)
            executor = transpiler.executor(backend=backend)
        else:
            executor = transpiler.executor()

        exe = transpiler.transpile(measure_top, bindings={"m": n})
        job = exe.sample(executor, shots=512)
        counts = {bits: cnt for bits, cnt in job.result().results}

        # IQFT followed by sampling a uniform superposition must yield
        # only the all-zero bitstring on every backend.
        expected_bits = tuple(0 for _ in range(n))
        assert set(counts) == {expected_bits}, (
            f"{sdk_transpiler.backend_name}: "
            f"expected {{({expected_bits})}}, got {set(counts)}"
        )

    @pytest.mark.parametrize("seed", [0, 5])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_cross_backend_expval(self, sdk_transpiler, n, seed):
        """Cross-backend expval: ``<+...+|IQFT^† Z_0 IQFT|+...+> = 1``.

        Pairs with :meth:`test_392_cross_backend_sampling` to exercise
        the estimator pipeline of every backend (sampler and estimator
        regress independently), parametrized via ``sdk_transpiler`` so
        each backend leg runs in the matching ``-m`` session (see
        tests/_cudaq_isolation.py for the cudaq constraint).
        ``IQFT|+...+> = |0...0>`` so ``<Z_0> = 1`` exactly under
        noiseless simulation.

        Args:
            sdk_transpiler (SdkTranspilerCase): Backend label plus
                transpiler instance from the shared fixture.
            n (int): Register size.
            seed (int): RNG seed; included so an accidental dependence
                on a fixed initial state would be caught.
        """
        import qamomile.observable as qm_o

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

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(expval_top, bindings={"m": n, "obs": H})
        val = exe.run(transpiler.executor()).result()
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(val, 1.0, atol=atol), (
            f"{sdk_transpiler.backend_name} n={n} seed={seed}: "
            f"expected <Z_0>=1, got {val}"
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_392_reproducer_iqft_visible_after_inline(self, qiskit_transpiler, n):
        """The exact #392 shape: inner kernel sized by UInt keeps IQFT.

        Goes through the explicit pipeline so the regression bound is
        the IR after :meth:`Transpiler.inline`. Asserts both halves of
        the original failure mode:

        - The inner ``CompositeGateOperation`` (``gate_type=IQFT``) must
          appear exactly once with the expected concrete
          ``num_target_qubits`` (the original symptom).
        - The outer ``MeasureVectorOperation`` must survive inlining,
          with its operand pointing at the same logical register as
          the IQFT's results. An earlier intermediate fix recovered
          the IQFT but dropped the measure because the specialized
          block had no ``ReturnOperation`` for the inline pass to use
          when remapping cloned return Values; this assertion guards
          that regression.
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

        measure_ops = [
            op for op in block.operations if isinstance(op, MeasureVectorOperation)
        ]
        assert len(measure_ops) == 1, (
            f"expected exactly one MeasureVectorOperation, got {len(measure_ops)}"
        )
        # The measure must consume the post-IQFT register. Without the
        # ``ReturnOperation`` emitted by the specialized callee trace,
        # the inline pass would leave the measure's operand pointing
        # at the pre-inline ArrayValue while the IQFT's qubit results
        # point at the cloned post-inline ArrayValue — a UUID mismatch
        # that drops the measure in the final emit. Compare register
        # ``logical_id``s (a stable identity preserved across cloning)
        # to pin down the operand mapping at the IR level.
        measure_operand = measure_ops[0].operands[0]
        iqft_parent_logical_ids = {
            res.parent_array.logical_id
            for res in iqft_ops[0].results
            if res.parent_array is not None
        }
        assert iqft_parent_logical_ids, (
            "IQFT result Values should carry a ``parent_array`` "
            "pointing at the qubit register after inline"
        )
        assert measure_operand.logical_id in iqft_parent_logical_ids
