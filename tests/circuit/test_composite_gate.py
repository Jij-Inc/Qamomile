"""Tests for the CompositeGate API."""

import numpy as np
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


# =============================================================================
# Reusable CompositeGate definitions
# =============================================================================


class HadamardAll(CompositeGate):
    """Apply H to all n qubits."""

    custom_name = "hadamard_all"

    def __init__(self, n: int):
        self._n = n

    @property
    def num_target_qubits(self) -> int:
        return self._n

    def _decompose(
        self, qubits: Vector[Qubit] | tuple[Qubit, ...]
    ) -> tuple[Qubit, ...]:
        return tuple(qmc.h(q) for q in qubits)


class BellPair(CompositeGate):
    """H + CX on 2 qubits -> Bell state."""

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


class DoubleH(CompositeGate):
    """Apply H twice -> identity."""

    custom_name = "double_h"

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


def run_statevector(qc):
    """Run circuit and return statevector (measurements removed)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc.remove_final_measurements()
    simulator = AerSimulator(method="statevector")
    qc = transpile(qc, simulator)
    qc.save_statevector()
    result = simulator.run(qc).result()
    return np.array(result.get_statevector())


def apply_gate_to_array(
    qs: Vector[Qubit], gate: CompositeGate
) -> Vector[Qubit]:
    """Apply a CompositeGate to all qubits in an array.

    This is a regular function (not a qkernel) so range() produces Python ints,
    allowing tuple indexing on the gate's return value.
    """
    n = gate.num_target_qubits
    qubit_list = [qs[i] for i in range(n)]
    result = gate(*qubit_list)
    for i in range(n):
        qs[i] = result[i]
    return qs


class TestCompositeGate:
    """Test the CompositeGate API: definition, application, and error handling."""

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

        block = circuit.build()
        assert block is not None

        ops = block.operations
        assert len(ops) == 2
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
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], QInitOperation)
        assert isinstance(ops[2], CompositeGateOperation)
        assert ops[2].custom_name == "bell_pair"
        assert ops[2].num_target_qubits == 2
        assert ops[2].num_control_qubits == 0

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
            (q,) = gate(q)
            return q

        with pytest.raises(ValueError, match="requires 2 target qubits"):
            circuit.build()

    def test_stub_gate_without_decomposition(self):
        """Stub gate (no _decompose) has no implementation."""

        class StubOracle(CompositeGate):
            custom_name = "oracle"

            @property
            def num_target_qubits(self) -> int:
                return 2

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

        block = circuit.build()

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].has_implementation is False
        assert composite_ops[0].implementation is None


class TestCompositeGateTranspilation:
    """Test CompositeGate IR generation and transpilation (requires Qiskit)."""

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

        block = qiskit_transpiler.to_block(circuit)
        inlined = qiskit_transpiler.inline(block)

        op_types = [type(op).__name__ for op in inlined.operations]
        assert "GateOperation" in op_types

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

    # -----------------------------------------------------------------
    # Transpile circuit correctness
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_transpile_circuit(self, qiskit_transpiler, n):
        """HadamardAll transpiles to a circuit with correct qubit count."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == n

    def test_bell_pair_transpile_circuit(self, qiskit_transpiler):
        """BellPair transpiles to a 2-qubit circuit."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == 2

    def test_double_h_transpile_circuit(self, qiskit_transpiler):
        """DoubleH transpiles to a 1-qubit circuit."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        assert qc.num_qubits == 1

    # -----------------------------------------------------------------
    # Statevector verification
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_statevector(self, qiskit_transpiler, n):
        """HadamardAll on |0...0> produces uniform superposition."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
        )

    def test_bell_pair_statevector(self, qiskit_transpiler):
        """BellPair produces (|00> + |11>) / sqrt(2)."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        # Bell state: |00> and |11> each have amplitude 1/sqrt(2)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-8), (
            f"Expected Bell state, got {sv}"
        )

    def test_double_h_statevector(self, qiskit_transpiler):
        """DoubleH (H*H = I) leaves |0> unchanged."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit)
        sv = run_statevector(qc)

        expected = np.array([1, 0], dtype=complex)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-8), (
            f"Expected |0>, got {sv}"
        )

    # -----------------------------------------------------------------
    # Sampling verification
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_sampling(self, qiskit_transpiler, n):
        """HadamardAll on |0...0> produces approximately uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 4096
        job = executable.sample(qiskit_transpiler.executor(), shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

        expected = shots / num_outcomes
        sigma = (expected * (1 - 1 / num_outcomes)) ** 0.5
        for outcome, count in counts.items():
            assert abs(count - expected) < 3 * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {3 * sigma:.0f}"
            )

    def test_bell_pair_sampling(self, qiskit_transpiler):
        """BellPair produces only |00> and |11> outcomes."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(qiskit_transpiler.executor(), shots=shots)
        result = job.result()

        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        # Only |00> (0) and |11> (3) should appear
        assert counts[1] == 0, f"Unexpected |01> count: {counts[1]}"
        assert counts[2] == 0, f"Unexpected |10> count: {counts[2]}"
        assert counts[0] > 0, "Expected |00> outcomes"
        assert counts[3] > 0, "Expected |11> outcomes"

    def test_double_h_sampling(self, qiskit_transpiler):
        """DoubleH (identity) always measures 0."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(qiskit_transpiler.executor(), shots=shots)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"Expected all zeros, got {bits} (count={count})"
            )

    # -----------------------------------------------------------------
    # Symbolic n tests
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_statevector_symbolic(self, qiskit_transpiler, n):
        """HadamardAll with symbolic n produces uniform superposition."""
        gate = HadamardAll(n)

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": n})
        sv = run_statevector(qc)

        expected_amp = 1.0 / np.sqrt(2**n)
        assert np.allclose(np.abs(sv), expected_amp, atol=1e-8), (
            f"n={n}: expected uniform amplitudes {expected_amp}, got {np.abs(sv)}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_sampling_symbolic(self, qiskit_transpiler, n):
        """HadamardAll with symbolic n produces uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = apply_gate_to_array(qs, gate)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit, bindings={"num": n})
        shots = 4096
        job = executable.sample(qiskit_transpiler.executor(), shots=shots)
        result = job.result()

        num_outcomes = 2**n
        counts = {i: 0 for i in range(num_outcomes)}
        for bits, count in result.results:
            val = sum(b << i for i, b in enumerate(bits))
            counts[val] += count

        assert sum(counts.values()) == shots
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

    def test_no_phantom_qubits_symbolic(self, qiskit_transpiler):
        """No phantom qubits with symbolic n and CompositeGate."""
        bell = BellPair()

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Bit:
            qs = qubit_array(num, "qs")
            q0, q1 = bell(qs[0], qs[1])
            qs[0] = q0
            qs[1] = q1
            return qmc.measure(qs[0])

        qc = qiskit_transpiler.to_circuit(circuit, bindings={"num": 2})
        qc.remove_final_measurements()
        assert qc.num_qubits == 2, f"Expected 2 qubits, got {qc.num_qubits}"
