"""Tests for Dict[Vector[UInt], Float] support in for_items."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation


class TestForItemsVectorKeyIR:
    """Tests for Vector key IR generation."""

    def test_vector_key_builds_ir_with_flag(self):
        """Test that Dict[Vector[UInt], Float] produces key_is_vector=True."""
        from qamomile.circuit.frontend.func_to_block import func_to_block

        @qmc.qkernel
        def higher_order(
            n_qubits: qmc.UInt,
            interactions: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for key, coeff in qmc.items(interactions):
                for step in qmc.range(key.shape[0] - 1):
                    q[key[step]], q[key[step + 1]] = qmc.rzz(
                        q[key[step]], q[key[step + 1]], gamma * coeff
                    )
            return q

        block = func_to_block(higher_order.func)

        found = False
        for op in block.operations:
            if isinstance(op, ForItemsOperation):
                found = True
                assert op.key_is_vector is True
                assert op.key_vars == ["key"]
                assert op.value_var == "coeff"
                break

        assert found, "ForItemsOperation not found in block"

    def test_tuple_key_keeps_flag_false(self):
        """Test that Dict[Tuple[UInt, UInt], Float] keeps key_is_vector=False."""
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

        block = func_to_block(ising_cost.func)

        for op in block.operations:
            if isinstance(op, ForItemsOperation):
                assert op.key_is_vector is False
                break


class TestForItemsVectorKeyTranspile:
    """Tests for transpiling Dict[Vector[UInt], Float]."""

    def test_vector_key_transpile_with_qiskit(self):
        """Test end-to-end transpile of Dict[Vector[UInt], Float] kernel."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def higher_order(
            n_qubits: qmc.UInt,
            interactions: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for key, coeff in qmc.items(interactions):
                for step in qmc.range(key.shape[0] - 1):
                    q[key[step]], q[key[step + 1]] = qmc.rzz(
                        q[key[step]], q[key[step + 1]], gamma * coeff
                    )
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # 2-body: (0,1) has 1 RZZ, 3-body: (0,1,2) has 2 RZZs → total 3
        interactions = {(0, 1): 1.0, (0, 1, 2): 0.3}
        executor = transpiler.transpile(
            higher_order,
            bindings={
                "n_qubits": 3,
                "interactions": interactions,
                "gamma": 0.5,
            },
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None

        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count == 3, f"Expected 3 RZZ gates, got {rzz_count}"

    def test_vector_key_single_element(self):
        """Test Dict[Vector[UInt], Float] with single-element keys."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def single_elem(
            n_qubits: qmc.UInt,
            interactions: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for key, coeff in qmc.items(interactions):
                for step in qmc.range(key.shape[0] - 1):
                    q[key[step]], q[key[step + 1]] = qmc.rzz(
                        q[key[step]], q[key[step + 1]], gamma * coeff
                    )
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        # Single-element keys: shape[0]-1 = 0 → no RZZ gates
        interactions = {(0,): 1.0, (1,): 0.5}
        executor = transpiler.transpile(
            single_elem,
            bindings={
                "n_qubits": 2,
                "interactions": interactions,
                "gamma": 0.5,
            },
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None

        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count == 0, f"Expected 0 RZZ gates, got {rzz_count}"

    def test_vector_key_empty_dict(self):
        """Test Dict[Vector[UInt], Float] with empty dict."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def empty_interactions(
            n_qubits: qmc.UInt,
            interactions: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gamma: qmc.Float,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for key, coeff in qmc.items(interactions):
                for step in qmc.range(key.shape[0] - 1):
                    q[key[step]], q[key[step + 1]] = qmc.rzz(
                        q[key[step]], q[key[step + 1]], gamma * coeff
                    )
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        executor = transpiler.transpile(
            empty_interactions,
            bindings={"n_qubits": 2, "interactions": {}, "gamma": 0.5},
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        assert circuit is not None


class TestForItemsRegression:
    """Regression tests to ensure existing Dict types still work."""

    def test_tuple_key_still_works(self):
        """Test Dict[Tuple[UInt, UInt], Float] still transpiles correctly."""
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

        ising_data = {(0, 1): 1.0, (1, 2): -0.5}
        executor = transpiler.transpile(
            ising_rzz,
            bindings={"n_qubits": 3, "ising": ising_data, "gamma": 0.5},
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        rzz_count = sum(1 for inst in circuit.data if inst.operation.name == "rzz")
        assert rzz_count == 2

    def test_scalar_key_still_works(self):
        """Test Dict[UInt, Float] still transpiles correctly."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def single_key(
            n_qubits: qmc.UInt,
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n_qubits, name="q")
            for i, theta in qmc.items(angles):
                q[i] = qmc.rz(q[i], theta)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()

        angles_data = {0: 0.5, 1: 1.0, 2: 1.5}
        executor = transpiler.transpile(
            single_key,
            bindings={"n_qubits": 3, "angles": angles_data},
        )

        assert len(executor.compiled_quantum) > 0
        circuit = executor.compiled_quantum[0].circuit
        rz_count = sum(1 for inst in circuit.data if inst.operation.name == "rz")
        assert rz_count == 3
