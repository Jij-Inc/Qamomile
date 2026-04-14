"""Integration tests for QInit scope (Bug #5) and phi_ops (Bug #6).

Bug #5: Sub-kernel QInit scope — ArrayValue.shape was not cloned/substituted
    during inlining, so QInitOperation failed to resolve array size at emit
    time.  Fixed in value_mapping.py (clone_value / substitute_value).

Bug #6: IfOperation phi_ops — phi outputs were never allocated or emitted,
    causing phi output UUIDs to be missing from qubit_map.  Fixed in
    emit_support/resource_allocator.py (ResourceAllocator), standard_emit.py (_register_phi_outputs,
    _emit_measure), and control_flow_visitor.py (visit/transform phi_ops).

Unit tests for the individual passes live in ``tests/transpiler/``:
    - ``test_value_mapping.py``  (shape cloning/substitution, phi_ops cloning)
    - ``test_emit_support.py``   (phi output allocation)
    - ``test_control_flow_visitor.py`` (phi_ops visitor/transformer traversal)

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations to identify
Float vs UInt etc.  PEP 563 deferred annotations turn them into strings, which
breaks ``_create_bound_input``.
"""

import pytest

# ===========================================================================
# Bug #5: Sub-kernel QInit scope integration test
# ===========================================================================


class TestFQAOAStateIntegration:
    """Integration test for fqaoa_state sub-kernel transpilation (Bug #5)."""

    @pytest.fixture(autouse=True)
    def _skip_without_qiskit(self):
        pytest.importorskip("qiskit")

    def test_fqaoa_state_transpiles(self):
        """Full fqaoa_state transpiles successfully with 4 qubits."""
        import qamomile.circuit as qmc
        from qamomile.circuit.algorithm.fqaoa import fqaoa_state
        from qamomile.qiskit.transpiler import QiskitTranspiler

        @qmc.qkernel
        def circuit(
            p: qmc.UInt,
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            n: qmc.UInt,
            n_f: qmc.UInt,
            givens_ij: qmc.Matrix[qmc.UInt],
            givens_theta: qmc.Vector[qmc.Float],
            hopping_val: qmc.Float,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = fqaoa_state(
                p,
                linear,
                quad,
                n,
                n_f,
                givens_ij,
                givens_theta,
                hopping_val,
                gammas,
                betas,
            )
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(
            circuit,
            bindings={
                "p": 1,
                "linear": {0: 0.5, 1: -0.3},
                "quad": {(0, 1): 1.0},
                "n": 4,
                "n_f": 2,
                "givens_ij": [[0, 1]],
                "givens_theta": [0.3],
                "hopping_val": 1.0,
                "gammas": [0.5],
                "betas": [0.3],
            },
        )
        qc = exe.compiled_quantum[0].circuit
        assert qc.num_qubits == 4


# ===========================================================================
# Bug #6: IfOperation phi_ops integration test
# ===========================================================================


class TestIfElseArrayExecution:
    """Integration test for if-else with qubit arrays (Bug #6)."""

    @pytest.fixture(autouse=True)
    def _skip_without_qiskit(self):
        pytest.importorskip("qiskit")
        pytest.importorskip("qiskit_aer")

    def test_if_else_array_execution(self):
        """Shot-based execution of if-else on qubit_array(2)."""
        import qamomile.circuit as qmc
        from qamomile.qiskit.transpiler import QiskitTranspiler

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit_array(2, "q")
            q[0] = qmc.x(q[0])
            bit = qmc.measure(q[0])
            if bit:
                q[1] = qmc.x(q[1])
            else:
                q[1] = q[1]
            return qmc.measure(q[1])

        transpiler = QiskitTranspiler()
        executor = transpiler.executor()
        exe = transpiler.transpile(circuit)
        job = exe.sample(executor, bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        for value, count in result.results:
            assert value in (0, 1)
            assert count > 0
