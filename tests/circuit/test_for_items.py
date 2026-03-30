"""Tests for for_items iteration over Dict types."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation


class TestForItemsIR:
    """Tests for ForItemsOperation IR node."""

    def test_for_items_operation_creation(self):
        """Test creating a ForItemsOperation."""
        op = ForItemsOperation(
            key_vars=["i", "j"],
            value_var="Jij",
            operations=[],
        )
        assert op.key_vars == ["i", "j"]
        assert op.value_var == "Jij"
        assert len(op.operations) == 0

    def test_for_items_operation_kind(self):
        """Test ForItemsOperation has CONTROL operation kind."""
        from qamomile.circuit.ir.operation.operation import OperationKind

        op = ForItemsOperation(
            key_vars=["i"],
            value_var="v",
            operations=[],
        )
        assert op.operation_kind == OperationKind.CONTROL


class TestForItemsContextManager:
    """Tests for for_items context manager."""

    def test_for_items_builds_operation(self):
        """Test that for_items context manager builds ForItemsOperation."""
        from qamomile.circuit.frontend.handle.containers import Dict
        from qamomile.circuit.frontend.operation.control_flow import for_items
        from qamomile.circuit.frontend.tracer import Tracer, trace
        from qamomile.circuit.ir.value import DictValue

        # Create a dummy Dict handle
        dv = DictValue(name="ising", entries=[], params={"parameter": "ising"})
        dict_handle = Dict(value=dv, _entries=[])

        # Create tracer and use for_items
        tracer = Tracer()
        with trace(tracer):
            with for_items(dict_handle, ["i", "j"], "Jij") as (key, val):
                # Loop body would go here
                pass

        # Check that ForItemsOperation was added
        assert len(tracer.operations) == 1
        op = tracer.operations[0]
        assert isinstance(op, ForItemsOperation)
        assert op.key_vars == ["i", "j"]
        assert op.value_var == "Jij"


class TestForItemsQKernel:
    """Tests for for_items in qkernel."""

    def test_for_items_with_dict_iteration(self):
        """Test for-items loop over Dict in qkernel."""
        from qamomile.circuit.frontend.func_to_block import func_to_block

        @qmc.qkernel
        def ising_cost(
            n_qubits: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return q

        # Build the qkernel to IR
        block = func_to_block(ising_cost.func)

        # Verify ForItemsOperation is in the block
        found_for_items = False
        for op in block.operations:
            if isinstance(op, ForItemsOperation):
                found_for_items = True
                assert op.key_vars == ["i", "j"]
                assert op.value_var == "Jij"
                break

        assert found_for_items, "ForItemsOperation not found in block"


class TestForItemsTranspile:
    """Tests for transpiling for_items."""

    def test_for_items_unroll_with_qiskit(self):
        """Test for-items loop unrolling with Qiskit backend."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def ising_rzz(
            n_qubits: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # Transpile with bound ising dict
        ising_data = {(0, 1): 1.0, (1, 2): -0.5}
        executor = transpiler.transpile(
            ising_rzz,
            bindings={"n_qubits": 3, "ising": ising_data, "gamma": 0.5},
        )

        # Verify circuit was generated
        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None

        # Check that RZZ gates were applied
        # ising has 2 entries, so we expect 2 RZZ gates
        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count == 2, f"Expected 2 RZZ gates, got {rzz_count}"

    def test_for_items_with_parameters(self):
        """Test for-items loop with parametric angle."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def ising_cost(
            n_qubits: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], gamma * Jij)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # Transpile with gamma as parameter
        ising_data = {(0, 1): 1.0, (0, 2): 0.5}
        executor = transpiler.transpile(
            ising_cost,
            bindings={"n_qubits": 3, "ising": ising_data},
            parameters=["gamma"],
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None

        # Check that circuit has parameters
        assert len(circuit.parameters) > 0, "Expected parametric circuit"


class TestForItemsEdgeCases:
    """Tests for edge cases in for_items."""

    def test_for_items_empty_dict(self):
        """Test for-items loop with empty dict."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def empty_ising(
            n_qubits: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for (i, j), Jij in qmc.items(ising):
                q[i], q[j] = qmc.rzz(q[i], q[j], Jij)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # Transpile with empty ising dict
        executor = transpiler.transpile(
            empty_ising,
            bindings={"n_qubits": 2, "ising": {}},
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None

        # No RZZ gates should be applied (empty loop)
        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count == 0, f"Expected 0 RZZ gates for empty dict, got {rzz_count}"

    def test_for_items_single_key(self):
        """Test for-items loop with single key (non-tuple)."""
        from qamomile.circuit.frontend.func_to_block import func_to_block

        @qmc.qkernel
        def single_key_dict(
            n_qubits: qmc.UInt,
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for i, theta in qmc.items(angles):
                q[i] = qmc.rz(q[i], theta)
            return q

        # Just verify it builds without error
        block = func_to_block(single_key_dict.func)

        found_for_items = False
        for op in block.operations:
            if isinstance(op, ForItemsOperation):
                found_for_items = True
                assert op.key_vars == ["i"]
                assert op.value_var == "theta"
                break

        assert found_for_items, "ForItemsOperation not found"
