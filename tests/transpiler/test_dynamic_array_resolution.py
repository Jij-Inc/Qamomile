"""Tests for dynamic array size resolution in transpiler.

These tests verify that the transpiler correctly handles:
1. Dynamic qubit array sizes derived from input array shapes (hi.shape[0] → hi_dim0)
2. Nested array access patterns (edges[e, 0])
3. Measurement of dynamically-sized qubit arrays
4. Numpy integer types in array element resolution

These tests were added to prevent regression of bugs fixed in the emit pass.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.qiskit.transpiler import QiskitTranspiler


# ==============================================================================
# Kernel definitions at module level (required for inspect.getsource to work)
# ==============================================================================


@qmc.qkernel
def kernel_dynamic_size(
    hi: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Kernel with qubit array sized by input vector shape."""
    n = hi.shape[0]
    q = qmc.qubit_array(n, name="q")
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return qmc.measure(q)


@qmc.qkernel
def kernel_matrix_size(
    edges: qmc.Matrix[qmc.UInt],
) -> qmc.Vector[qmc.Bit]:
    """Kernel with qubit array sized by matrix dimension."""
    n = edges.shape[0]
    q = qmc.qubit_array(n, name="q")
    return qmc.measure(q)


@qmc.qkernel
def kernel_nested_index(
    edges: qmc.Matrix[qmc.UInt],
    n_edges: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Kernel with nested array index access."""
    q = qmc.qubit_array(3, name="q")
    for e in qmc.range(n_edges):
        i = edges[e, 0]
        j = edges[e, 1]
        q[i], q[j] = qmc.cx(q[i], q[j])
    return qmc.measure(q)


@qmc.qkernel
def kernel_ising_cost(
    q: qmc.Vector[qmc.Qubit],
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """QAOA cost layer with nested array access."""
    num_e = edges.shape[0]
    for e in qmc.range(num_e):
        i = edges[e, 0]
        j = edges[e, 1]
        angle = 2 * Jij[e] * gamma
        q[i], q[j] = qmc.rzz(q[i], q[j], angle)
    n = hi.shape[0]
    for i in qmc.range(n):
        angle = 2 * hi[i] * gamma
        q[i] = qmc.rz(q[i], angle)
    return q


@qmc.qkernel
def kernel_x_mixer(
    q: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """QAOA mixer layer."""
    n = q.shape[0]
    for i in qmc.range(n):
        angle = 2 * beta
        q[i] = qmc.rx(q[i], angle)
    return q


@qmc.qkernel
def kernel_qaoa_state(
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    p: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Qubit]:
    """QAOA state preparation with dynamic qubit array."""
    n = hi.shape[0]
    q = qmc.qubit_array(n, name="qaoa_state")
    for iter in qmc.range(p):
        gamma = gammas[iter]
        beta = betas[iter]
        q = kernel_ising_cost(q, edges, Jij, hi, gamma)
        q = kernel_x_mixer(q, beta)
    return q


@qmc.qkernel
def kernel_qaoa(
    edges: qmc.Matrix[qmc.UInt],
    Jij: qmc.Vector[qmc.Float],
    hi: qmc.Vector[qmc.Float],
    p: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Full QAOA circuit with measurement."""
    q = kernel_qaoa_state(edges, Jij, hi, p, gammas, betas)
    return qmc.measure(q)


# ==============================================================================
# Test classes
# ==============================================================================


class TestDynamicArraySizeResolution:
    """Tests for dynamic qubit array size resolution.

    Verifies that qubit arrays sized by input array shapes (e.g., hi.shape[0])
    are correctly allocated and can be used in circuits.
    """

    def test_qubit_array_size_from_vector_shape(self):
        """Test that qubit array size can be determined from input vector shape.

        This tests the hi_dim0 naming pattern resolution in ResourceAllocator._resolve_size().
        """
        transpiler = QiskitTranspiler()
        hi = np.array([0.1, 0.2, 0.3])  # 3 elements → 3 qubits

        executor = transpiler.transpile(kernel_dynamic_size, bindings={"hi": hi})

        # Run the circuit to verify it executes without error
        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        # Should have measurement results for 3 qubits
        for bitstring, _count in result.results:
            assert len(bitstring) == 3

    def test_qubit_array_size_from_matrix_shape(self):
        """Test that qubit array size works with 2D array shapes."""
        transpiler = QiskitTranspiler()
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64)  # 3 edges

        executor = transpiler.transpile(kernel_matrix_size, bindings={"edges": edges})

        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        for bitstring, _count in result.results:
            assert len(bitstring) == 3


class TestNestedArrayAccess:
    """Tests for nested array access resolution.

    Verifies that qubit indices computed from array lookups (e.g., edges[e, 0])
    are correctly resolved during loop unrolling.
    """

    def test_nested_array_index_resolution(self):
        """Test that nested array access like q[edges[e, 0]] works correctly.

        This tests the numpy integer type handling in _resolve_array_element_value().
        """
        transpiler = QiskitTranspiler()
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint64)

        executor = transpiler.transpile(
            kernel_nested_index,
            bindings={"edges": edges, "n_edges": 2},
        )

        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None


class TestDynamicMeasurement:
    """Tests for measurement of dynamically-sized qubit arrays.

    Verifies that _emit_measure_vector correctly handles dynamic sizes.
    """

    def test_measure_dynamic_qubit_array(self):
        """Test that measurement works with dynamically-sized qubit arrays.

        This tests the _emit_measure_vector fix for dynamic size resolution.
        """
        transpiler = QiskitTranspiler()
        hi = np.array([0.1, 0.2, 0.3, 0.4])  # 4 elements

        executor = transpiler.transpile(kernel_dynamic_size, bindings={"hi": hi})

        # This should not raise "No counts for experiment" error
        job = executor.sample(transpiler.executor(), bindings={}, shots=100)
        result = job.result()
        assert result is not None
        assert len(result.results) > 0
        # Verify all bitstrings have correct length
        for bitstring, _count in result.results:
            assert len(bitstring) == 4


class TestQAOAPattern:
    """Integration tests for QAOA-like circuit patterns.

    These tests verify that the combination of all dynamic resolution features
    works correctly in realistic QAOA circuits.
    """

    def test_qaoa_full_pattern(self):
        """Test full QAOA pattern with dynamic sizes and nested array access."""
        # Test data - triangle graph
        edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64)
        Jij = np.array([0.5, 0.3, 0.2])
        hi = np.array([0.1, 0.2, 0.15])  # 3 qubits

        transpiler = QiskitTranspiler()
        executor = transpiler.transpile(
            kernel_qaoa,
            bindings={"edges": edges, "Jij": Jij, "hi": hi, "p": 2},
            parameters=["betas", "gammas"],
        )

        # Run with concrete parameter values
        job = executor.sample(
            transpiler.executor(),
            bindings={"betas": [0.1, 0.2], "gammas": [0.3, 0.4]},
            shots=100,
        )
        result = job.result()

        assert result is not None
        assert len(result.results) > 0
        # Verify all bitstrings have correct length (3 qubits)
        for bitstring, _count in result.results:
            assert len(bitstring) == 3
