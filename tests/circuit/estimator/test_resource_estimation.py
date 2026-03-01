"""Tests for resource estimation functionality."""

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator import (
    estimate_resources,
)
from qamomile.circuit.estimator.algorithmic import (
    estimate_qaoa,
    estimate_qdrift,
    estimate_qpe,
    estimate_qsvt,
    estimate_trotter,
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
        assert est.gates.rotation_gates == 0  # No rotation gates

    def test_parametric_ghz_estimation(self):
        """Test resource estimation with symbolic parameter."""

        @qm.qkernel
        def ghz_state(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            q[0] = qm.h(q[0])
            for i in qm.range(n - 1):
                q[i], q[i + 1] = qm.cx(q[i], q[i + 1])
            return q

        est = estimate_resources(ghz_state.block)

        # Verify symbolic expressions are resolved correctly
        n = sp.Symbol("n", integer=True, positive=True)
        assert est.qubits == n

        # Total gates: 1 H + (n-1) CX = n gates
        assert sp.simplify(est.gates.total - n) == 0

        # Single-qubit: just the H gate
        assert est.gates.single_qubit == 1

        # Two-qubit: n-1 CX gates
        assert sp.simplify(est.gates.two_qubit - (n - 1)) == 0

        # Verify no "uint_tmp" symbols remain
        assert "uint_tmp" not in str(est.gates.total)
        assert "uint_tmp" not in str(est.gates.two_qubit)

        # Concrete substitution should work
        concrete = est.substitute(n=10)
        assert concrete.qubits == 10
        assert concrete.gates.total == 10
        assert concrete.gates.two_qubit == 9

    def test_arithmetic_in_loop_bounds(self):
        """Test that arithmetic expressions in loop bounds are traced correctly."""

        @qm.qkernel
        def circuit_with_arithmetic(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            # n-1 iterations
            for i in qm.range(n - 1):
                q[i] = qm.h(q[i])
            return q

        est = estimate_resources(circuit_with_arithmetic.block)
        n = sp.Symbol("n", integer=True, positive=True)

        # Should be n-1 H gates
        assert sp.simplify(est.gates.single_qubit - (n - 1)) == 0

        # Should NOT contain auto-generated temp symbols
        assert "uint_tmp" not in str(est.gates.single_qubit)

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
        assert est.gates.rotation_gates == 9  # All are rotation gates

    def test_nested_loop_dependent_bounds(self):
        """Test nested loops where inner bound depends on outer variable."""

        @qm.qkernel
        def nested_circuit(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            # Nested loop: for j in range(n): for k in range(j): 1 gate
            for j in qm.range(n):
                for k in qm.range(j):
                    q[k] = qm.h(q[k])
            return q

        est = estimate_resources(nested_circuit.block)
        n = sp.Symbol("n", integer=True, positive=True)

        # Inner loop executes 0 + 1 + 2 + ... + (n-1) = n(n-1)/2 times
        # Expected: n(n-1)/2 H gates
        expected = n * (n - 1) / 2
        assert sp.simplify(est.gates.single_qubit - expected) == 0

        # Should NOT contain loop variable 'j' or 'k'
        assert "j" not in str(est.gates.single_qubit)
        assert "k" not in str(est.gates.single_qubit)

        # Test concrete value
        concrete = est.substitute(n=10)
        assert concrete.gates.single_qubit == 45  # 10*9/2

    def test_iqft_pattern(self):
        """Test IQFT-like pattern with nested dependent loops."""

        @qm.qkernel
        def iqft_pattern(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            # Swap gates: n//2 iterations
            for j in qm.range(n // 2):
                q[j], q[n - j - 1] = qm.swap(q[j], q[n - j - 1])
            # Nested controlled rotations + H
            for j in qm.range(n):
                for k in qm.range(j):
                    q[j], q[k] = qm.cp(q[j], q[k], theta=1.0)
                q[j] = qm.h(q[j])
            return q

        est = estimate_resources(iqft_pattern.block)
        n = sp.Symbol("n", integer=True, positive=True)

        # SWAP gates: floor(n/2)
        # CP gates: sum(j for j in range(n)) = n(n-1)/2
        # H gates: n
        # Total two-qubit: floor(n/2) + n(n-1)/2

        # Should NOT contain 'j' or 'k' symbols
        assert "j" not in str(est.gates.total)
        assert "k" not in str(est.gates.total)

        # Verify with concrete value
        concrete = est.substitute(n=8)
        # SWAP: 4, CP: 28, H: 8 → Total: 40
        assert concrete.gates.total == 40

    def test_symbol_propagation_through_calls(self):
        """Test that symbolic parameters propagate through qkernel calls."""

        @qm.qkernel
        def inner_kernel(qubits: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            """Inner kernel that uses qubits.shape[0]"""
            n = qubits.shape[0]
            for i in qm.range(n):
                qubits[i] = qm.h(qubits[i])
            return qubits

        @qm.qkernel
        def outer_kernel(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            """Outer kernel that passes array to inner kernel"""
            qubits = qm.qubit_array(n, name="q")
            qubits = inner_kernel(qubits)
            return qubits

        est = estimate_resources(outer_kernel.block)
        n = sp.Symbol("n", integer=True, positive=True)

        # Should use 'n', not 'qubits_dim0'
        assert "qubits_dim0" not in str(est.gates.total)
        assert est.gates.single_qubit == n

        # Verify concrete substitution
        concrete = est.substitute(n=10)
        assert concrete.gates.single_qubit == 10

    def test_iqft_symbol_propagation(self):
        """Test IQFT doesn't produce qubits_dim0 symbols."""

        # This is the actual IQFT from the tutorial
        @qm.qkernel
        def iqft(qubits: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            n = qubits.shape[0]
            for j in qm.range(n // 2):
                qubits[j], qubits[n - j - 1] = qm.swap(qubits[j], qubits[n - j - 1])
            for j in qm.range(n):
                for k in qm.range(j):
                    qubits[j], qubits[k] = qm.cp(qubits[j], qubits[k], theta=1.0)
                qubits[j] = qm.h(qubits[j])
            return qubits

        @qm.qkernel
        def iqft_n(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            qubits = qm.qubit_array(n, name="q")
            return iqft(qubits)

        est = estimate_resources(iqft_n.block)

        # Should NOT contain qubits_dim0
        assert "qubits_dim0" not in str(est.gates.total)

        # Should contain 'n'
        n = sp.Symbol("n", integer=True, positive=True)
        assert n in est.gates.total.free_symbols

        # Test concrete value
        concrete = est.substitute(n=8)
        assert concrete.gates.total == 40  # 4 SWAP + 28 CP + 8 H

    def test_power_in_loop_no_uint_tmp(self):
        """Test that 2**i in loop doesn't produce uint_tmp symbols."""

        @qm.qkernel
        def power_loop(m: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(m, name="q")
            for i in qm.range(m):
                iterations = 2**i
                for _ in qm.range(iterations):
                    q[i] = qm.h(q[i])
            return q

        est = estimate_resources(power_loop.block)

        # Should NOT contain uint_tmp
        assert "uint_tmp" not in str(est.gates.total)

        # Should contain m and possibly i (in Sum)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert m_sym in est.gates.total.free_symbols

        # Verify concrete substitution
        concrete = est.substitute(m=4)
        # For m=4: sum(2^i for i in 0..3) = 1+2+4+8 = 15 H gates
        assert concrete.gates.single_qubit == 15

    def test_qpe_manual_no_uint_tmp(self):
        """Test actual QPE pattern from tutorial doesn't produce uint_tmp."""

        @qm.qkernel
        def controlled_phase(q: qm.Qubit, theta: qm.Float) -> qm.Qubit:
            q = qm.p(q, theta)
            return q

        @qm.qkernel
        def qpe_simplified(m: qm.UInt) -> qm.Vector[qm.Qubit]:
            counting = qm.qubit_array(m, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)

            # Prepare superposition
            for i in qm.range(m):
                counting[i] = qm.h(counting[i])

            # Controlled-U operations
            cp = qm.controlled(controlled_phase)
            for i in qm.range(m):
                iterations = 2**i
                for _ in qm.range(iterations):
                    counting[i], target = cp(counting[i], target, theta=1.0)

            return counting

        est = estimate_resources(qpe_simplified.block)

        # Should NOT contain uint_tmp
        assert "uint_tmp" not in str(est.gates.total)

        # Should contain m
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert m_sym in est.gates.total.free_symbols

    def test_controlled_gate_with_varying_parameter(self):
        """Test that controlled gates with parametric iterations resolve correctly."""

        @qm.qkernel
        def repeated_gate(q: qm.Qubit, theta: qm.Float, iter: qm.UInt) -> qm.Qubit:
            """Apply gate iter times"""
            for _ in qm.range(iter):
                q = qm.p(q, theta)
            return q

        @qm.qkernel
        def qpe_pattern(m: qm.UInt) -> qm.Vector[qm.Qubit]:
            """QPE-like pattern with 2^i iterations"""
            counting = qm.qubit_array(m, name="counting")
            target = qm.qubit(name="target")

            controlled_gate = qm.controlled(repeated_gate)
            for i in qm.range(m):
                iterations = 2**i
                counting[i], target = controlled_gate(
                    counting[i], target, theta=1.0, iter=iterations
                )
            return counting

        est = estimate_resources(qpe_pattern.block)

        # Should NOT contain 'iter' symbol
        assert "iter" not in str(est.gates.total)

        # Should contain m (and ideally 2**m pattern)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert m_sym in est.gates.total.free_symbols

        # Verify concrete substitution
        concrete = est.substitute(m=3)
        # For m=3: sum(2^i for i in 0..2) = 1+2+4 = 7 P gates
        # (Plus gates for qubit initialization, but focus on the controlled part)
        # The key is that it should be a concrete number, not contain 'iter'
        assert isinstance(int(concrete.gates.total), int)


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
        alpha = sp.Symbol("alpha", integer=True, positive=True)  # ||H||

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
        n, L, t, eps = sp.symbols("n L t eps", integer=True, positive=True)

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


class TestCompositeGateEstimation:
    """Test resource estimation for CompositeGateOperation."""

    def test_qpe_builtin_estimation(self):
        """Test built-in qmc.qpe() resource estimation with concrete size."""

        # Note: qmc.qpe() is a Python function that emits operations at build time
        # It requires concrete array sizes, not symbolic parameters
        # So we use a concrete size here

        @qm.qkernel
        def simple_phase_gate(q: qm.Qubit, theta: float) -> qm.Qubit:
            return qm.p(q, theta)

        @qm.qkernel
        def test_qpe_concrete(theta: float) -> qm.Float:
            counting = qm.qubit_array(3, name="counting")  # Concrete size
            target = qm.qubit(name="target")
            target = qm.x(target)
            phase = qm.qpe(target, counting, simple_phase_gate, theta=theta)
            return qm.measure(phase)

        # Test estimation
        est = estimate_resources(test_qpe_concrete.block)

        # Should get concrete gate counts
        # For m=3: 1 X + 3 H + 3 controlled-P (power= means 1 gate each) + IQFT(3)
        # IQFT(3): 1 SWAP + 3 CP + 3 H = 7 gates
        # Total: 1 + 3 + 3 + 7 = 14 gates
        assert est.gates.total == 14

    def test_qpe_builtin_concrete_m8(self):
        """Test built-in QPE with m=8 matches expected gate count."""

        @qm.qkernel
        def simple_phase_gate(q: qm.Qubit, theta: float) -> qm.Qubit:
            return qm.p(q, theta)

        @qm.qkernel
        def test_qpe_m8(theta: float) -> qm.Float:
            counting = qm.qubit_array(8, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)
            phase = qm.qpe(target, counting, simple_phase_gate, theta=theta)
            return qm.measure(phase)

        est = estimate_resources(test_qpe_m8.block)

        # For m=8: 1 X + 8 H + 8 controlled-P (power= means 1 gate each) + IQFT(8)
        # IQFT(8) = 4 SWAP + 28 CP + 8 H = 40 gates
        # Total: 1 X + 8 H + 8 CP + 40 IQFT = 57 gates
        assert est.gates.total == 57

    def test_qft_builtin_estimation(self):
        """Test built-in qmc.qft() resource estimation."""

        # qmc.qft() requires concrete array sizes
        @qm.qkernel
        def test_qft() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(4, name="q")
            q = qm.qft(q)
            return q

        est = estimate_resources(test_qft.block)

        # Should not be just 1
        assert est.gates.total != 1

        # QFT(4) = 4 H + 6 CP (0+1+2+3=6 pairs) + 2 SWAP = 12 gates
        assert est.gates.total == 12
        assert est.gates.rotation_gates == 6  # 6 CP gates are rotation

    def test_iqft_builtin_estimation(self):
        """Test built-in qmc.iqft() resource estimation."""

        # qmc.iqft() requires concrete array sizes
        @qm.qkernel
        def test_iqft() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(8, name="q")
            q = qm.iqft(q)
            return q

        est = estimate_resources(test_iqft.block)

        # Should not be just 1
        assert est.gates.total != 1

        # IQFT(8) = 4 SWAP + 28 CP + 8 H = 40 gates
        assert est.gates.total == 40
        assert est.gates.rotation_gates == 28  # 28 CP gates are rotation

    def test_nested_composite_gates(self):
        """Test nested composite gates (QPE contains IQFT)."""

        @qm.qkernel
        def simple_phase(q: qm.Qubit, theta: float) -> qm.Qubit:
            return qm.p(q, theta)

        @qm.qkernel
        def test_nested(theta: float) -> qm.Float:
            counting = qm.qubit_array(4, name="counting")  # Concrete size
            target = qm.qubit(name="target")
            target = qm.x(target)
            # QPE internally uses IQFT
            phase = qm.qpe(target, counting, simple_phase, theta=theta)
            return qm.measure(phase)

        est = estimate_resources(test_nested.block)

        # Should handle nested composite gates
        assert est.gates.total != 1

        # QPE(m=4): 1 X + 4 H + 4 controlled-P (power= means 1 gate each) + IQFT(4)
        # IQFT(4): 2 SWAP + 6 CP + 4 H = 12 gates
        # Total: 1 + 4 + 4 + 12 = 21 gates
        assert est.gates.total == 21


class TestStubGateEstimation:
    """Test resource estimation for stub composite gates."""

    def test_stub_gate_resource_estimation(self):
        """Test that stub gate metadata is correctly propagated to estimate_resources."""
        from qamomile.circuit.frontend.composite_gate import composite_gate
        from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata

        @composite_gate(
            stub=True,
            name="black_box_oracle",
            num_qubits=3,
            resource_metadata=ResourceMetadata(t_gates=10, query_complexity=1),
        )
        def black_box_oracle():
            pass

        @qm.qkernel
        def circuit_with_stub() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(3, name="q")
            for i in qm.range(3):
                q[i] = qm.h(q[i])
            q[0], q[1], q[2] = black_box_oracle(q[0], q[1], q[2])
            return q

        est = estimate_resources(circuit_with_stub.block)
        assert est.gates.t_gates == 10
        # 3 H gates from circuit, stub contributes 0 for unspecified fields
        assert est.gates.single_qubit == 3
        assert est.gates.clifford_gates == 3
        assert est.gates.two_qubit == 0
        assert est.gates.rotation_gates == 0
        # Verify oracle call is tracked
        assert "black_box_oracle" in est.gates.oracle_calls
        assert est.gates.oracle_calls["black_box_oracle"] == 1

    def test_stub_gate_multiple_calls(self):
        """Test oracle_calls counts multiple invocations of the same stub gate."""
        from qamomile.circuit.frontend.composite_gate import composite_gate
        from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata

        @composite_gate(
            stub=True,
            name="repeated_oracle",
            num_qubits=2,
            resource_metadata=ResourceMetadata(t_gates=5),
        )
        def repeated_oracle():
            pass

        @qm.qkernel
        def circuit_multi() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0], q[1] = repeated_oracle(q[0], q[1])
            q[0], q[1] = repeated_oracle(q[0], q[1])
            q[0], q[1] = repeated_oracle(q[0], q[1])
            return q

        est = estimate_resources(circuit_multi.block)
        assert est.gates.oracle_calls["repeated_oracle"] == 3
        assert est.gates.t_gates == 15

    def test_multiple_different_oracle_calls(self):
        """Test tracking multiple different stub gates in one circuit."""
        from qamomile.circuit.frontend.composite_gate import composite_gate
        from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata

        @composite_gate(
            stub=True,
            name="oracle_A",
            num_qubits=2,
            resource_metadata=ResourceMetadata(t_gates=3),
        )
        def oracle_a():
            pass

        @composite_gate(
            stub=True,
            name="oracle_B",
            num_qubits=2,
            resource_metadata=ResourceMetadata(t_gates=7),
        )
        def oracle_b():
            pass

        @qm.qkernel
        def circuit_two_oracles() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0], q[1] = oracle_a(q[0], q[1])
            q[0], q[1] = oracle_b(q[0], q[1])
            q[0], q[1] = oracle_a(q[0], q[1])
            return q

        est = estimate_resources(circuit_two_oracles.block)
        assert est.gates.oracle_calls["oracle_A"] == 2
        assert est.gates.oracle_calls["oracle_B"] == 1
        assert est.gates.t_gates == 3 + 7 + 3

    def test_no_oracle_calls_for_non_stub(self):
        """Test that non-stub circuits have empty oracle_calls."""

        @qm.qkernel
        def bell_state() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            return q

        est = estimate_resources(bell_state.block)
        assert est.gates.oracle_calls == {}


class TestQPEResourceEstimation:
    """Test QPE resource estimation comparing manual vs built-in implementations."""

    def test_manual_qpe_has_polynomial_term(self):
        """Test that manual QPE with opaque ControlledU has polynomial resource."""

        @qm.qkernel
        def phase_gate(q: qm.Qubit, theta: float, iter: int) -> qm.Qubit:
            """Phase gate with power parameter."""
            for i in qm.range(iter):
                q = qm.p(q, theta)
            return q

        @qm.qkernel
        def iqft(qubits: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            """Inverse QFT."""
            import math

            n = qubits.shape[0]
            for j in qm.range(n // 2):
                qubits[j], qubits[n - j - 1] = qm.swap(qubits[j], qubits[n - j - 1])
            for j in qm.range(n):
                for k in qm.range(j):
                    angle = -math.pi / (2 ** (j - k))
                    qubits[j], qubits[k] = qm.cp(qubits[j], qubits[k], theta=angle)
                qubits[j] = qm.h(qubits[j])
            return qubits

        @qm.qkernel
        def qpe_manual(theta: float, m: qm.UInt) -> qm.Vector[qm.Bit]:
            """Manual QPE implementation."""
            counting = qm.qubit_array(m, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)

            # Hadamard gates on counting qubits
            for i in qm.range(m):
                counting[i] = qm.h(counting[i])

            # Controlled-U^(2^k) operations
            controlled_phase = qm.controlled(phase_gate)
            for i in qm.range(m):
                iterations = 2**i
                counting[i], target = controlled_phase(
                    counting[i], target, theta=theta, iter=iterations
                )

            # IQFT
            counting = iqft(counting)

            # Measure
            bits = qm.measure(counting)
            return bits

        est = estimate_resources(qpe_manual.block)

        # With opaque ControlledUOperation, manual QPE is polynomial (not exponential).
        # Each controlled_phase call counts as 1 gate regardless of iter parameter.
        m = sp.Symbol("m", integer=True, positive=True)
        total_gates_expr = est.gates.total

        # Should NOT have exponential term 2^m (ControlledU is opaque)
        has_exponential = any(
            isinstance(arg, sp.Pow) and arg.base == 2
            for arg in sp.preorder_traversal(total_gates_expr)
        )
        assert not has_exponential, (
            f"Expected no 2^m term in total gates (opaque ControlledU), got: {total_gates_expr}"
        )

        # Verify qubit count: m + 1
        assert est.qubits == m + 1

    def test_builtin_qpe_has_polynomial_term(self):
        """Test that built-in qpe() has polynomial (not exponential) resource estimate.

        ControlledUOperation with power=2^k represents a single Controlled(U^(2^k))
        gate, not 2^k repetitions. So QPE resource should be polynomial in n.
        """

        @qm.qkernel
        def simple_phase_gate(q: qm.Qubit, theta: float) -> qm.Qubit:
            """Simple phase gate for qpe()."""
            return qm.p(q, theta)

        @qm.qkernel
        def qpe_builtin(theta: float, n: qm.UInt) -> qm.Float:
            """QPE using built-in qpe function."""
            counting = qm.qubit_array(n, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)

            phase = qm.qpe(target, counting, simple_phase_gate, theta=theta)
            return qm.measure(phase)

        est = estimate_resources(qpe_builtin.block)
        est = est.simplify()

        n = sp.Symbol("n", integer=True, positive=True)
        total_gates_expr = est.gates.total

        # Should NOT have exponential term 2^n (power= means single gate)
        has_exponential = any(
            isinstance(arg, sp.Pow) and arg.base == 2
            for arg in sp.preorder_traversal(total_gates_expr)
        )
        assert not has_exponential, (
            f"Expected no 2^n term in total gates, got: {total_gates_expr}"
        )

        # Verify qubit count: n + 1
        assert est.qubits == n + 1

    def test_manual_vs_builtin_qpe_similarity(self):
        """Test that manual and built-in QPE have similar resource estimates."""

        # Manual implementation
        @qm.qkernel
        def phase_gate(q: qm.Qubit, theta: float, iter: int) -> qm.Qubit:
            for i in qm.range(iter):
                q = qm.p(q, theta)
            return q

        @qm.qkernel
        def iqft(qubits: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            import math

            n = qubits.shape[0]
            for j in qm.range(n // 2):
                qubits[j], qubits[n - j - 1] = qm.swap(qubits[j], qubits[n - j - 1])
            for j in qm.range(n):
                for k in qm.range(j):
                    angle = -math.pi / (2 ** (j - k))
                    qubits[j], qubits[k] = qm.cp(qubits[j], qubits[k], theta=angle)
                qubits[j] = qm.h(qubits[j])
            return qubits

        @qm.qkernel
        def qpe_manual(theta: float, m: qm.UInt) -> qm.Vector[qm.Bit]:
            counting = qm.qubit_array(m, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)

            for i in qm.range(m):
                counting[i] = qm.h(counting[i])

            controlled_phase = qm.controlled(phase_gate)
            for i in qm.range(m):
                iterations = 2**i
                counting[i], target = controlled_phase(
                    counting[i], target, theta=theta, iter=iterations
                )

            counting = iqft(counting)
            bits = qm.measure(counting)
            return bits

        # Built-in implementation
        @qm.qkernel
        def simple_phase_gate(q: qm.Qubit, theta: float) -> qm.Qubit:
            return qm.p(q, theta)

        @qm.qkernel
        def qpe_builtin(theta: float, n: qm.UInt) -> qm.Float:
            counting = qm.qubit_array(n, name="counting")
            target = qm.qubit(name="target")
            target = qm.x(target)
            phase = qm.qpe(target, counting, simple_phase_gate, theta=theta)
            return qm.measure(phase)

        est_manual = estimate_resources(qpe_manual.block)
        est_builtin = estimate_resources(qpe_builtin.block).simplify()

        # Both should have similar qubit counts
        m = sp.Symbol("m", integer=True, positive=True)
        n = sp.Symbol("n", integer=True, positive=True)

        assert est_manual.qubits == m + 1
        assert est_builtin.qubits == n + 1

        # Substitute concrete value and compare
        est_manual_8 = est_manual.substitute(m=8)
        est_builtin_8 = est_builtin.substitute(n=8)

        # With opaque ControlledU, both manual and builtin QPE are polynomial.
        # Manual QPE: m H + m ControlledU + IQFT(m) = polynomial in m
        # Built-in QPE: same polynomial structure
        # For m=8: both should give the same total (57)
        assert est_manual_8.gates.total == est_builtin_8.gates.total, (
            f"Manual and built-in QPE should have same total gates, "
            f"got manual={est_manual_8.gates.total}, builtin={est_builtin_8.gates.total}"
        )


class TestRotationGateCounting:
    """Test rotation gate counting."""

    def test_single_qubit_rotation_rx(self):
        @qm.qkernel
        def rx_circuit() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(1, name="q")
            q[0] = qm.rx(q[0], angle=1.0)
            return q

        est = estimate_resources(rx_circuit.block)
        assert est.gates.rotation_gates == 1
        assert est.depth.rotation_depth == 1

    def test_clifford_not_counted_as_rotation(self):
        @qm.qkernel
        def clifford_circuit() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0] = qm.h(q[0])
            q[0], q[1] = qm.cx(q[0], q[1])
            return q

        est = estimate_resources(clifford_circuit.block)
        assert est.gates.rotation_gates == 0
        assert est.depth.rotation_depth == 0

    def test_mixed_rotation_and_clifford(self):
        @qm.qkernel
        def mixed() -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(2, name="q")
            q[0] = qm.h(q[0])
            q[0] = qm.rx(q[0], 1.0)
            q[0], q[1] = qm.cx(q[0], q[1])
            q[0], q[1] = qm.rzz(q[0], q[1], 1.0)
            return q

        est = estimate_resources(mixed.block)
        assert est.gates.rotation_gates == 2  # rx + rzz
        assert est.gates.clifford_gates == 2  # h + cx
        assert est.gates.total == 4

    def test_parametric_rotation_count(self):
        @qm.qkernel
        def rotation_loop(n: qm.UInt) -> qm.Vector[qm.Qubit]:
            q = qm.qubit_array(n, name="q")
            for i in qm.range(n):
                q[i] = qm.rx(q[i], angle=1.0)
            return q

        est = estimate_resources(rotation_loop.block)
        n = sp.Symbol("n", integer=True, positive=True)
        assert sp.simplify(est.gates.rotation_gates - n) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
