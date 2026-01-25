"""Tests for resource estimation functionality."""

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator import (
    estimate_resources,
    count_gates,
    estimate_depth,
    qubits_counter,
)
from qamomile.circuit.estimator.algorithmic import (
    estimate_qaoa,
    estimate_qpe,
    estimate_trotter,
    estimate_qsvt,
    estimate_qdrift,
)


class TestBasicCircuitEstimation:
    """Test basic circuit resource estimation."""

    def test_bell_state_estimation(self):
        """Test resource estimation for Bell state circuit."""

        @qm.qkernel
        def bell_state() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            return q

        est = estimate_resources(bell_state.block)

        # Check qubits
        assert est.qubits == 2

        # Check gates: 1 H + 1 CX = 2 total
        assert est.gates.total == 2
        assert est.gates.single_qubit == 1  # H gate
        assert est.gates.two_qubit == 1  # CX gate
        assert est.gates.clifford_gates == 2  # Both H and CX are Clifford

    def test_parametric_ghz_estimation(self):
        """Test resource estimation with parametric circuit size."""

        @qm.qkernel
        def ghz_state(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            q[0] = qm.h(q[0])
            for i in qm.range(n - 1):
                q[i], q[i + 1] = qm.cx(q[i], q[i + 1])
            return q

        est = estimate_resources(ghz_state.block)

        # Check that estimation produces symbolic expressions
        # (exact form depends on IR details, so we just check they exist)
        n = sp.Symbol("n")
        assert est.qubits == n
        assert isinstance(est.gates.total, sp.Expr)
        assert isinstance(est.gates.single_qubit, sp.Expr)
        assert isinstance(est.gates.two_qubit, sp.Expr)

        # The most reliable check: verify with concrete values
        # by building a circuit with concrete size
        @qm.qkernel
        def ghz10() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(10, name="q")
            q[0] = qm.h(q[0])
            for i in qm.range(9):
                q[i], q[i + 1] = qm.cx(q[i], q[i + 1])
            return q

        concrete = estimate_resources(ghz10.block)
        assert concrete.qubits == 10
        assert concrete.gates.total == 10  # 1 H + 9 CX
        assert concrete.gates.single_qubit == 1
        assert concrete.gates.two_qubit == 9

    def test_qaoa_circuit_estimation(self):
        """Test resource estimation for QAOA circuit."""

        # Test with concrete size to avoid IR complexity
        @qm.qkernel
        def qaoa_layer_concrete() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(5, name="q")
            # Cost layer: ZZ rotations on pairs
            for i in qm.range(4):
                q[i], q[i + 1] = qm.rzz(q[i], q[i + 1], angle=1.0)
            # Mixer layer: X rotations
            for i in qm.range(5):
                q[i] = qm.rx(q[i], angle=0.5)
            return q

        est = estimate_resources(qaoa_layer_concrete.block)

        # Check concrete values
        assert est.qubits == 5
        # Gates: 4 RZZ + 5 RX = 9 total
        assert est.gates.total == 9
        assert est.gates.single_qubit == 5  # RX gates
        assert est.gates.two_qubit == 4  # RZZ gates


class TestAlgorithmicEstimators:
    """Test algorithmic (theoretical) resource estimators."""

    def test_qaoa_maxcut_complete_graph(self):
        """Test QAOA estimation for MaxCut on complete graph."""
        n, p = sp.symbols("n p", positive=True, integer=True)

        # Complete graph K_n has n*(n-1)/2 edges
        edges = n * (n - 1) / 2

        est = estimate_qaoa(n, p, num_edges=edges)

        # Qubits should be n
        assert est.qubits == n

        # Total gates: n H + p*(n*(n-1)/2 RZZ + n RX)
        # = n + p*(n^2/2 - n/2 + n) = n + p*n^2/2 + p*n/2
        # Simplified: n(1 + p(n+1)/2)

        # Test concrete values
        concrete = est.substitute(n=10, p=3)
        assert concrete.qubits == 10

        # 10 H + 3*(45 RZZ + 10 RX) = 10 + 3*55 = 175
        assert concrete.gates.total == 175

    def test_qpe_qubitization(self):
        """Test QPE resource estimation with qubitization."""
        n = sp.Symbol("n", positive=True, integer=True)
        m = sp.Symbol("m", positive=True, integer=True)  # precision
        alpha = sp.Symbol("alpha", positive=True)  # ||H||

        est = estimate_qpe(n, m, hamiltonian_norm=alpha, method="qubitization")

        # Qubits: n + m + O(log n)
        # Check that qubits contains n, m terms
        assert n in est.qubits.free_symbols
        assert m in est.qubits.free_symbols

        # Gates should scale with alpha * 2^m
        # Just check it's a valid expression with these symbols
        assert alpha in est.gates.total.free_symbols or alpha in str(est.gates.total)

    def test_hamiltonian_simulation_methods(self):
        """Test different Hamiltonian simulation methods."""
        n, L, t, eps = sp.symbols("n L t eps", positive=True)

        # Trotter (second-order)
        trotter2 = estimate_trotter(n, L, t, eps, order=2)
        assert trotter2.qubits == n

        # Trotter (fourth-order)
        trotter4 = estimate_trotter(n, L, t, eps, order=4)
        assert trotter4.qubits == n

        # QSVT
        qsvt = estimate_qsvt(n, hamiltonian_norm=L, time=t, error=eps)
        assert qsvt.qubits == n

        # qDRIFT
        qdrift = estimate_qdrift(L, hamiltonian_1norm=L, time=t, error=eps)

        # All should have the right free symbols
        for est in [trotter2, trotter4, qsvt]:
            assert n in est.gates.total.free_symbols

    def test_concrete_hamiltonian_simulation(self):
        """Test Hamiltonian simulation with concrete parameters."""
        # 100 qubits, 1000 terms, time=10, error=0.001

        trotter = estimate_trotter(
            n=100, L=1000, time=10, error=0.001, order=2, hamiltonian_1norm=1000
        )

        # Should get concrete numbers
        assert trotter.qubits == 100
        assert isinstance(float(trotter.gates.total), float)

        qsvt = estimate_qsvt(n=100, hamiltonian_norm=1000, time=10, error=0.001)

        assert qsvt.qubits == 100
        assert isinstance(float(qsvt.gates.total), float)


class TestResourceEstimateOperations:
    """Test operations on ResourceEstimate objects."""

    def test_substitute(self):
        """Test parameter substitution."""
        n, p = sp.symbols("n p", positive=True, integer=True)
        est = estimate_qaoa(n, p, num_edges=n * (n - 1) / 2)

        # Substitute n=10, p=3
        concrete = est.substitute(n=10, p=3)

        assert concrete.qubits == 10
        # All expressions should be concrete numbers
        assert isinstance(int(concrete.gates.total), int)

    def test_simplify(self):
        """Test expression simplification."""
        n = sp.Symbol("n", positive=True, integer=True)

        # Create estimate with complex expression
        est = estimate_qaoa(n, p=1, num_edges=n * (n - 1) / 2)

        simplified = est.simplify()

        # Should still have same values (just simplified)
        assert simplified.qubits == n

    def test_to_dict(self):
        """Test conversion to dictionary."""
        est = estimate_qaoa(n=10, p=3, num_edges=45)

        data = est.to_dict()

        assert "qubits" in data
        assert "gates" in data
        assert "depth" in data
        assert data["qubits"] == "10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
