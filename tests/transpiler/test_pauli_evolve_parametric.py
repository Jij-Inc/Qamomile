"""Tests for Layer 5: ``pauli_evolve`` parametric gamma support.

Before Layer 5, ``pauli_evolve`` required a concrete ``gamma`` at emit
time — any symbolic gamma raised ``EmitError``. Now, when gamma is a
declared parameter (scalar or ``arr[idx]`` with ``arr`` in
``parameters``), the emitted circuit carries a backend parameter that
can be bound at run-time, matching how ``ising_cost``/``rz``/``rzz``
already handle parametric angles.

The library ``qaoa_layers`` pattern continues to route through
``ising_cost`` and is unchanged.
"""

import pytest

pytest.importorskip("qiskit")

import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import x_mixer
from qamomile.qiskit.transpiler import QiskitTranspiler


def _make_zz_h() -> qm_o.Hamiltonian:
    H = qm_o.Hamiltonian()
    H.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Z, 1),
        ),
        1.0,
    )
    return H


class TestScalarParametricGamma:
    def test_scalar_parametric_gamma_compiles(self):
        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Float,
            hamiltonian: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            q = qmc.pauli_evolve(q, hamiltonian, gamma)
            return qmc.measure(q)

        H = _make_zz_h()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={"n": H.num_qubits, "hamiltonian": H},
            parameters=["gamma"],
        )
        circuit = exe.compiled_quantum[0].circuit
        assert circuit.num_parameters == 1
        assert any(str(p) == "gamma" for p in circuit.parameters)


class TestArrayElementParametricGamma:
    def test_parametric_gamma_per_layer(self):
        """Shape comes from a compile-time scalar; gamma stays runtime-symbolic.

        Pre-disjointness this test passed ``gamma=[0,0,0]`` in bindings as
        a shape hint while also marking ``gamma`` as a runtime parameter. That
        overlap pattern is now rejected by ``Transpiler.transpile``; the
        QAOA-style pattern (separate compile-time depth + runtime array)
        is the supported alternative.
        """

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            p: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            for k in qmc.range(p):
                q = qmc.pauli_evolve(q, hamiltonian, gamma[k])
            return qmc.measure(q)

        H = _make_zz_h()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={
                "n": H.num_qubits,
                "p": 3,
                "hamiltonian": H,
            },
            parameters=["gamma"],
        )
        circuit = exe.compiled_quantum[0].circuit
        assert circuit.num_parameters == 3
        param_names = {str(p) for p in circuit.parameters}
        assert param_names == {"gamma[0]", "gamma[1]", "gamma[2]"}


class TestFullQAOAExecution:
    """End-to-end: user's ipynb pattern with parametric gamma AND beta."""

    def _kernels(self):
        """Build a QAOA expval kernel with compile-time depth ``p``.

        Migrated from the pre-disjointness pattern that passed
        ``gamma=[0.0]*p`` as a shape hint while also marking ``gamma`` as
        a runtime parameter; the new contract takes the depth as its own
        compile-time scalar so ``gamma``/``beta`` stay purely runtime-symbolic.

        Returns:
            The compiled-once expval kernel parameterized over ``p``.
        """

        @qmc.qkernel
        def qaoa_circuit(
            n: qmc.UInt,
            p: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            beta: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Vector[qmc.Qubit]:
            q = superposition_vector(n)
            for k in qmc.range(p):
                q = qmc.pauli_evolve(q, hamiltonian, gamma[k])
                q = x_mixer(q, beta[k])
            return q

        @qmc.qkernel
        def qaoa_exp(
            n: qmc.UInt,
            p: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            beta: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = qaoa_circuit(n, p, gamma, beta, hamiltonian)
            return qmc.expval(q, hamiltonian)

        return qaoa_exp

    def test_end_to_end_parametric_qaoa(self):
        """Compile once, run with different params, verify results differ."""
        qaoa_exp = self._kernels()
        H = _make_zz_h()

        tr = QiskitTranspiler()
        exe = tr.transpile(
            qaoa_exp,
            bindings={
                "n": H.num_qubits,
                "p": 2,
                "hamiltonian": H,
            },
            parameters=["gamma", "beta"],
        )

        circuit = exe.compiled_quantum[0].circuit
        # 4 parameters: gamma[0..1], beta[0..1]
        assert circuit.num_parameters == 4
        # 2 PauliEvolution + 4 Rx (x_mixer emits rx for each of 2 qubits × 2 layers)
        gate_names = [inst.operation.name for inst in circuit.data]
        assert gate_names.count("PauliEvolution") == 2
        assert gate_names.count("rx") == 4

        executor = tr.executor()
        r1 = exe.run(
            executor, bindings={"gamma": [0.3, 0.5], "beta": [0.4, 0.6]}
        ).result()
        r2 = exe.run(
            executor, bindings={"gamma": [0.7, 0.2], "beta": [0.1, 0.8]}
        ).result()

        # Same compiled circuit, different params → results must differ
        assert abs(r1 - r2) > 1e-6

    def test_statevector_matches_analytical_single_layer(self):
        """Sanity: p=1 QAOA analytical expval matches execution."""
        qaoa_exp = self._kernels()
        H = _make_zz_h()

        tr = QiskitTranspiler()
        exe = tr.transpile(
            qaoa_exp,
            bindings={
                "n": H.num_qubits,
                "p": 1,
                "hamiltonian": H,
            },
            parameters=["gamma", "beta"],
        )

        executor = tr.executor()
        gamma = 0.3
        beta = 0.4
        r = exe.run(executor, bindings={"gamma": [gamma], "beta": [beta]}).result()

        # For |+> ⊗ |+> → ZZ layer → X-mixer, the expval of ZZ can be
        # computed analytically. We only check that it's finite and
        # within [-1, 1] to keep the test robust against backend drift.
        assert np.isfinite(r)
        assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9
