"""Tests for the CompositeGate API."""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.frontend.tracer import Tracer
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value
from tests.circuit.conftest import run_statevector

# =============================================================================
# Reusable CompositeGate definitions
# =============================================================================


class HadamardAll(CompositeGate):
    """Apply H to all n qubits.

    Args:
        n: Number of target qubits.
    """

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
    """Apply H + CX on 2 qubits to create a Bell state."""

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
    """Apply H twice (identity operation) on a single qubit."""

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


def apply_gate_to_array(qs: Vector[Qubit], gate: CompositeGate) -> Vector[Qubit]:
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
        gate = HadamardAll(n)
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
                    t_gates=10,
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
        assert metadata.t_gates == 10
        assert metadata.query_complexity == 5
        assert metadata.custom_metadata["num_qubits"] == n
        # Unspecified typed fields default to None
        assert metadata.total_gates is None
        assert metadata.single_qubit_gates is None
        assert metadata.two_qubit_gates is None
        assert metadata.multi_qubit_gates is None
        assert metadata.clifford_gates is None
        assert metadata.rotation_gates is None

    def test_apply_composite_gate_in_qkernel(self):
        """CompositeGate can be used inside a qkernel."""
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        assert ops[1].custom_name == "double_h"
        assert ops[1].num_target_qubits == 1
        assert ops[1].num_control_qubits == 0

    def test_apply_multi_qubit_composite_gate(self):
        """Multi-qubit CompositeGate works correctly."""
        bell = BellPair()

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0, q1 = bell(q0, q1)
            return q0, q1

        block = circuit.build()

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

    def test_decompose_called_during_call(self, mocker):
        """_decompose is called exactly once when gate is invoked."""
        spy = mocker.spy(DoubleH, "_decompose")
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        circuit.build()
        spy.assert_called_once()

    def test_strategy_overrides_decompose(self, mocker):
        """When a strategy is registered and selected, _decompose is NOT called."""

        class TestGateForStrategy(CompositeGate):
            custom_name = "test_strat_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                q = qubits[0]
                q = qmc.h(q)
                return (q,)

        gate = TestGateForStrategy()

        mock_strategy = mocker.MagicMock()
        mock_strategy.name = "mock_strat"
        mock_strategy.decompose.return_value = None
        mock_strategy.resources.return_value = None

        TestGateForStrategy.register_strategy("mock_strat", mock_strategy)
        decompose_spy = mocker.spy(TestGateForStrategy, "_decompose")

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = gate(q, strategy="mock_strat")
            return q

        circuit.build()

        mock_strategy.decompose.assert_called_once()
        decompose_spy.assert_not_called()

        del TestGateForStrategy._strategies["mock_strat"]

    def test_decomposition_uses_fresh_tracer(self):
        """Decomposition operations don't leak into the outer block."""
        bell = BellPair()

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            q0, q1 = bell(q0, q1)
            return q0, q1

        block = circuit.build()
        # Block should have QInit ops + 1 CompositeGateOperation only
        # (no inner H/CX ops leaked from decomposition)
        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].custom_name == "bell_pair"

    def test_output_qubit_versions_incremented(self):
        """Output qubits have version = input version + 1 with same logical_id."""
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        op = ops[1]
        # Target operand is the last operand (after BlockValue)
        target_operand = op.operands[-1]
        target_result = op.results[-1]
        assert target_result.version == target_operand.version + 1
        assert target_result.logical_id == target_operand.logical_id

    def test_operands_order_in_ir(self):
        """Operands are ordered: [BlockValue, ...controls, ...targets]."""
        from qamomile.circuit.ir.block_value import BlockValue

        class ControlledGate(CompositeGate):
            custom_name = "ctrl_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            @property
            def num_control_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                q = qubits[0]
                q = qmc.h(q)
                return (q,)

        gate = ControlledGate()

        @qkernel
        def circuit(ctrl: Qubit, tgt: Qubit) -> tuple[Qubit, Qubit]:
            ctrl, tgt = gate(tgt, controls=(ctrl,))
            return ctrl, tgt

        block = circuit.build()

        ops = block.operations
        assert len(ops) == 3
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], QInitOperation)
        assert isinstance(ops[2], CompositeGateOperation)
        op = ops[2]
        # operands[0] = BlockValue (implementation)
        assert isinstance(op.operands[0], BlockValue)
        # operands[1] = control qubit value, operands[2] = target qubit value
        ctrl_value = ops[0].results[0]
        tgt_value = ops[1].results[0]
        assert op.operands[1].logical_id == ctrl_value.logical_id
        assert op.operands[2].logical_id == tgt_value.logical_id
        assert op.num_control_qubits == 1
        assert op.num_target_qubits == 1

    def test_resource_metadata_in_operation(self):
        """_resources() metadata flows through to the emitted operation."""

        class GateWithResources(CompositeGate):
            custom_name = "res_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                return (qmc.h(qubits[0]),)

            def _resources(self) -> ResourceMetadata:
                return ResourceMetadata(
                    t_gates=42,
                    query_complexity=7,
                    custom_metadata={"key": "value"},
                )

        gate = GateWithResources()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = gate(q)
            return q

        block = circuit.build()
        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        op = ops[1]
        assert op.resource_metadata is not None
        assert op.resource_metadata.t_gates == 42
        assert op.resource_metadata.query_complexity == 7
        assert op.resource_metadata.custom_metadata["key"] == "value"
        # Unspecified typed fields default to None
        assert op.resource_metadata.total_gates is None
        assert op.resource_metadata.single_qubit_gates is None
        assert op.resource_metadata.two_qubit_gates is None
        assert op.resource_metadata.multi_qubit_gates is None
        assert op.resource_metadata.clifford_gates is None
        assert op.resource_metadata.rotation_gates is None

    def test_strategy_resource_metadata(self, mocker):
        """Strategy's resources() is used instead of _resources() when strategy is active."""

        class GateForStratRes(CompositeGate):
            custom_name = "strat_res_gate"

            @property
            def num_target_qubits(self) -> int:
                return 1

            def _decompose(self, qubits):
                return (qmc.h(qubits[0]),)

            def _resources(self) -> ResourceMetadata:
                return ResourceMetadata(t_gates=100)

        gate = GateForStratRes()

        strategy_meta = ResourceMetadata(t_gates=5, query_complexity=3)
        mock_strategy = mocker.MagicMock()
        mock_strategy.name = "efficient"
        mock_strategy.decompose.return_value = None
        mock_strategy.resources.return_value = strategy_meta

        GateForStratRes.register_strategy("efficient", mock_strategy)

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = gate(q, strategy="efficient")
            return q

        block = circuit.build()
        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        op = ops[1]
        assert op.resource_metadata is strategy_meta
        assert op.resource_metadata.t_gates == 5

        del GateForStratRes._strategies["efficient"]

    def test_strategy_registry_isolation(self):
        """Strategies registered on one subclass are not visible to another."""

        class GateA(CompositeGate):
            custom_name = "gate_a"

            @property
            def num_target_qubits(self) -> int:
                return 1

        class GateB(CompositeGate):
            custom_name = "gate_b"

            @property
            def num_target_qubits(self) -> int:
                return 1

        class FakeStrategy:
            @property
            def name(self) -> str:
                return "fake"

            def decompose(self, qubits):
                return qubits

            def resources(self, num_qubits):
                return None

        GateA.register_strategy("fake", FakeStrategy())
        assert GateA.get_strategy("fake") is not None
        assert GateB.get_strategy("fake") is None
        del GateA._strategies["fake"]

    def test_tracer_add_operation_called(self, mocker):
        """Gate invocation calls tracer.add_operation with CompositeGateOperation."""
        spy = mocker.spy(Tracer, "add_operation")
        double_h = DoubleH()

        @qkernel
        def circuit(q: Qubit) -> Qubit:
            (q,) = double_h(q)
            return q

        circuit.build()

        # add_operation is called for QInitOperation + CompositeGateOperation
        composite_calls = [
            c
            for c in spy.call_args_list
            if isinstance(c.args[1], CompositeGateOperation)
        ]
        assert len(composite_calls) == 1
        assert composite_calls[0].args[1].custom_name == "double_h"

    def test_output_preserves_parent_indices(self):
        """Output qubit handles preserve parent_array from array elements."""
        bell = BellPair()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0, q1 = bell(qs[0], qs[1])
            qs[0] = q0
            qs[1] = q1
            return qs

        block = circuit.build()
        ops = block.operations
        assert len(ops) == 2
        assert isinstance(ops[0], QInitOperation)
        assert isinstance(ops[1], CompositeGateOperation)
        op = ops[1]
        # Result values should have parent_array set (from array elements)
        for index, r in enumerate(op.results):
            assert isinstance(r.parent_array, ArrayValue)
            assert len(r.element_indices) == 1
            assert isinstance(r.element_indices[0], Value)
            assert r.element_indices[0].params["const"] == index


class TestQFTAndIQFTClasses:
    """Test the built-in QFT and IQFT CompositeGate classes."""

    def test_qft_class_attributes(self):
        """QFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import QFT

        qft = QFT(4)
        assert qft.num_target_qubits == 4
        assert qft.gate_type == CompositeGateType.QFT
        assert qft.custom_name == "qft"

    def test_iqft_class_attributes(self):
        """IQFT class has correct attributes."""
        from qamomile.circuit.stdlib.qft import IQFT

        iqft = IQFT(3)
        assert iqft.num_target_qubits == 3
        assert iqft.gate_type == CompositeGateType.IQFT
        assert iqft.custom_name == "iqft"

    def test_qft_resources(self):
        """QFT returns correct resource metadata."""
        from qamomile.circuit.stdlib.qft import QFT

        qft = QFT(4)
        metadata = qft.get_resource_metadata()

        assert metadata is not None
        assert metadata.t_gates == 0
        # n=4: num_h=4, num_cp=6, num_swap=2
        assert metadata.total_gates == 12
        assert metadata.single_qubit_gates == 4
        assert metadata.two_qubit_gates == 8
        assert metadata.clifford_gates == 6
        assert metadata.rotation_gates == 6
        assert metadata.custom_metadata["num_h_gates"] == 4

    def test_qft_in_qkernel(self):
        """QFT can be used in a qkernel."""
        from qamomile.circuit.stdlib.qft import QFT

        @qkernel
        def circuit(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
            qft = QFT(2)
            q0, q1 = qft(q0, q1)
            return q0, q1

        block = circuit.build()
        assert block is not None

    def test_qft_factory_function(self):
        """qft() factory function works in qkernel."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qubits = qubit_array(3, "q")
            qubits = qft(qubits)
            return qubits

        block = circuit.build()
        assert block is not None

    def test_iqft_factory_function(self):
        """iqft() factory function works in qkernel."""
        from qamomile.circuit.stdlib.qft import iqft

        @qkernel
        def circuit() -> Vector[Qubit]:
            qubits = qubit_array(3, "q")
            qubits = iqft(qubits)
            return qubits

        block = circuit.build()
        assert block is not None


class TestCompositeGateTranspilation:
    """Test CompositeGate IR generation and transpilation (requires Qiskit)."""

    def test_custom_gate_builds_ir(self, qiskit_transpiler):
        """Custom CompositeGate builds correct IR."""
        bell = BellPair()

        @qkernel
        def circuit() -> Vector[Qubit]:
            qs = qubit_array(2, "qs")
            q0 = qs[0]
            q1 = qs[1]
            q0, q1 = bell(q0, q1)
            qs[0] = q0
            qs[1] = q1
            return qs

        block = qiskit_transpiler.to_block(circuit)
        inlined = qiskit_transpiler.inline(block)

        op_types = [type(op).__name__ for op in inlined.operations]
        assert "GateOperation" in op_types

    def test_qft_builds_ir(self, qiskit_transpiler):
        """QFT builds correct IR with native gate type."""
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

        block = qiskit_transpiler.to_block(circuit)

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.QFT

    def test_iqft_builds_ir(self, qiskit_transpiler):
        """IQFT builds correct IR with native gate type."""
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

        block = qiskit_transpiler.to_block(circuit)

        composite_ops = [
            op for op in block.operations if isinstance(op, CompositeGateOperation)
        ]
        assert len(composite_ops) == 1
        assert composite_ops[0].gate_type == CompositeGateType.IQFT

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
        assert len(qc.data) == n * 2  # n H gates + |measurement|, no extra qubits
        data = qc.data
        from qiskit.circuit import Measure
        from qiskit.circuit.library import HGate

        for i in range(n):
            assert isinstance(data[i].operation, HGate)
        for i in range(n, 2 * n):
            assert isinstance(data[i].operation, Measure)

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
        assert len(qc.data) == 4  # H + CX + |measurement|, no extra qubits
        data = qc.data
        from qiskit.circuit import Measure
        from qiskit.circuit.library import CXGate, HGate

        assert isinstance(data[0].operation, HGate)
        assert isinstance(data[1].operation, CXGate)
        assert isinstance(data[2].operation, Measure)
        assert isinstance(data[3].operation, Measure)

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
        assert len(qc.data) == 3  # H + H + |measurement|, no extra qubits
        data = qc.data
        from qiskit.circuit import Measure
        from qiskit.circuit.library import HGate

        assert isinstance(data[0].operation, HGate)
        assert isinstance(data[1].operation, HGate)
        assert isinstance(data[2].operation, Measure)

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

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hadamard_all_sampling(self, qiskit_transpiler, seeded_executor, n):
        """HadamardAll on |0...0> produces approximately uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(n, "qs")
            qs = apply_gate_to_array(qs, gate)
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

        assert sum(counts.values()) == shots
        assert len([c for c in counts.values() if c > 0]) == num_outcomes

        expected = shots / num_outcomes
        sigma = (expected * (1 - 1 / num_outcomes)) ** 0.5
        for outcome, count in counts.items():
            assert abs(count - expected) < 3 * sigma, (
                f"n={n}, outcome={outcome}: count={count}, "
                f"expected={expected:.0f} +/- {3 * sigma:.0f}"
            )

    def test_bell_pair_sampling(self, qiskit_transpiler, seeded_executor):
        """BellPair produces only |00> and |11> outcomes."""
        bell = BellPair()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(2, "qs")
            qs = apply_gate_to_array(qs, bell)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
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

    def test_double_h_sampling(self, qiskit_transpiler, seeded_executor):
        """DoubleH (identity) always measures 0."""
        double_h = DoubleH()

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(1, "qs")
            qs = apply_gate_to_array(qs, double_h)
            return qmc.measure(qs)

        executable = qiskit_transpiler.transpile(circuit)
        shots = 1024
        job = executable.sample(seeded_executor, shots=shots)
        result = job.result()

        for bits, count in result.results:
            assert all(b == 0 for b in bits), (
                f"Expected all zeros, got {bits} (count={count})"
            )

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
    def test_hadamard_all_sampling_symbolic(
        self, qiskit_transpiler, seeded_executor, n
    ):
        """HadamardAll with symbolic n produces uniform distribution."""
        gate = HadamardAll(n)

        @qkernel
        def circuit(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(num, "qs")
            qs = apply_gate_to_array(qs, gate)
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


class TestNestedQKernelNoPhantomQubits:
    """Verify no phantom qubits when nesting qkernels with array arguments."""

    def test_2_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 2-level nested qkernel passing array."""

        @qkernel
        def inner(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def outer() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = inner(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(outer)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_3_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 3-level nested qkernel passing array."""

        @qkernel
        def level_c(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def level_b(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = level_c(qs)
            return qs

        @qkernel
        def level_a() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = level_b(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(level_a)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_4_level_nest(self, qiskit_transpiler):
        """No phantom qubits with 4-level nested qkernel passing array."""

        @qkernel
        def d(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def c(qs: Vector[Qubit]) -> Vector[Qubit]:
            return d(qs)

        @qkernel
        def b(qs: Vector[Qubit]) -> Vector[Qubit]:
            return c(qs)

        @qkernel
        def a() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = b(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(a)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_inner_creates_array(self, qiskit_transpiler):
        """No phantom qubits when inner qkernel creates its own array."""

        @qkernel
        def inner_create() -> Vector[Qubit]:
            qs = qubit_array(3, "inner_qs")
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def outer_create() -> qmc.Vector[qmc.Bit]:
            qs = inner_create()
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(outer_create)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_3_level_nest_with_composite_gate(self, qiskit_transpiler):
        """No phantom qubits with 3-level nest and CompositeGate (QFT)."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def apply_qft(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = qft(qs)
            return qs

        @qkernel
        def middle(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = apply_qft(qs)
            return qs

        @qkernel
        def top() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = middle(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(top)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_multi_element_access_nested(self, qiskit_transpiler):
        """No phantom qubits when multiple array elements are accessed in nested call."""

        @qkernel
        def process(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.h(qs[2])
            return qs

        @qkernel
        def wrapper(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs = process(qs)
            return qs

        @qkernel
        def main_circuit() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(4, "qs")
            qs = wrapper(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(main_circuit)
        assert qc.num_qubits == 4, f"Expected 4 qubits, got {qc.num_qubits}"

    def test_same_qkernel_called_twice(self, qiskit_transpiler):
        """No phantom qubits when same qkernel is called twice on same array."""

        @qkernel
        def apply_h_first(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def call_twice() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h_first(qs)
            qs = apply_h_first(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(call_twice)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_multiple_array_parameters(self, qiskit_transpiler):
        """No phantom qubits with qkernel taking two array parameters."""

        @qkernel
        def entangle_arrays(
            qs1: Vector[Qubit], qs2: Vector[Qubit]
        ) -> tuple[Vector[Qubit], Vector[Qubit]]:
            qs1[0] = qmc.h(qs1[0])
            qs1[0], qs2[0] = qmc.cx(qs1[0], qs2[0])
            return qs1, qs2

        @qkernel
        def two_arrays() -> qmc.Vector[qmc.Bit]:
            a = qubit_array(2, "a")
            b = qubit_array(2, "b")
            a, b = entangle_arrays(a, b)
            return qmc.measure(a)

        qc = qiskit_transpiler.to_circuit(two_arrays)
        assert qc.num_qubits == 4, f"Expected 4 qubits, got {qc.num_qubits}"

    def test_scalar_qubit_and_vector_mixed(self, qiskit_transpiler):
        """No phantom qubits with mixed scalar Qubit and Vector parameters."""

        @qkernel
        def mixed_params(
            q: Qubit, qs: Vector[Qubit]
        ) -> tuple[Qubit, Vector[Qubit]]:
            q, qs[0] = qmc.cx(q, qs[0])
            return q, qs

        @qkernel
        def scalar_and_vector() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            q = qs[0]
            q = qmc.h(q)
            qs[0] = q
            q2 = qs[1]
            q2, qs = mixed_params(q2, qs)
            qs[1] = q2
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(scalar_and_vector)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_operations_before_and_after_nested_call(self, qiskit_transpiler):
        """No phantom qubits with array ops before and after nested call."""

        @qkernel
        def inner_op(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def before_and_after() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs[0] = qmc.h(qs[0])
            qs = inner_op(qs)
            qs[2] = qmc.h(qs[2])
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(before_and_after)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_two_different_nested_calls(self, qiskit_transpiler):
        """No phantom qubits with two different nested qkernel calls."""

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def apply_x(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def two_nested() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h(qs)
            qs = apply_x(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(two_nested)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_nested_call_then_composite_gate(self, qiskit_transpiler):
        """No phantom qubits with nested call followed by CompositeGate."""
        from qamomile.circuit.stdlib.qft import qft

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def nested_then_qft() -> qmc.Vector[qmc.Bit]:
            qs = qubit_array(3, "qs")
            qs = apply_h(qs)
            qs = qft(qs)
            return qmc.measure(qs)

        qc = qiskit_transpiler.to_circuit(nested_then_qft)
        assert qc.num_qubits == 3, f"Expected 3 qubits, got {qc.num_qubits}"

    def test_two_independent_arrays_with_nested_calls(self, qiskit_transpiler):
        """No phantom qubits with two independent arrays and separate nested calls."""

        @qkernel
        def apply_h(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def apply_x(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[1] = qmc.x(qs[1])
            return qs

        @qkernel
        def two_independent() -> qmc.Vector[qmc.Bit]:
            a = qubit_array(2, "a")
            b = qubit_array(3, "b")
            a = apply_h(a)
            b = apply_x(b)
            return qmc.measure(b)

        qc = qiskit_transpiler.to_circuit(two_independent)
        assert qc.num_qubits == 5, f"Expected 5 qubits, got {qc.num_qubits}"


class TestAllocateGateInvariant:
    """Test that _allocate_gate enforces invariants on malformed operations."""

    def test_missing_qinit_for_array_element(self):
        """GateOperation on array element without QInitOperation raises AssertionError."""
        from qamomile.circuit.ir.operation.gate import (
            GateOperation,
            GateOperationType,
        )
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.transpiler.passes.emit_base import ResourceAllocator

        parent_array = ArrayValue(
            type=QubitType(),
            name="q_array",
            shape=(Value(type=UIntType(), name="dim0", params={"const": 3}),),
        )

        idx_value = Value(
            type=UIntType(),
            name="idx",
            params={"const": 0},
        )

        element_qubit = Value(
            type=QubitType(),
            name="q_element",
            parent_array=parent_array,
            element_indices=(idx_value,),
        )

        result_qubit = element_qubit.next_version()

        gate_op = GateOperation(
            gate_type=GateOperationType.H,
            operands=[element_qubit],
            results=[result_qubit],
        )

        allocator = ResourceAllocator()
        with pytest.raises(AssertionError, match="Array element key"):
            allocator.allocate([gate_op])


class TestBackwardsCompatibility:
    """Test that existing APIs still work."""

    def test_old_iqft_import_still_works(self):
        """Old import path for iqft/qft still works."""
        from qamomile.circuit.stdlib import iqft, qft

        assert callable(iqft)
        assert callable(qft)

    def test_stdlib_exports_classes(self):
        """stdlib exports both class and function APIs."""
        from qamomile.circuit.stdlib import QFT, IQFT, qft, iqft, qpe

        assert QFT is not None
        assert IQFT is not None
        assert qft is not None
        assert iqft is not None
        assert qpe is not None
