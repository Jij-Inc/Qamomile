"""Theoretical resource estimates for QAOA (Quantum Approximate Optimization Algorithm).

Based on Section 4 and Section 20 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.depth_estimator import CircuitDepth


def estimate_qaoa(
    n: sp.Expr | int,
    p: sp.Expr | int,
    num_edges: sp.Expr | int,
    mixer_type: str = "x",
) -> ResourceEstimate:
    """Estimate resources for QAOA circuit.

    QAOA alternates between cost and mixer unitaries for p layers.
    For Ising/QUBO problems:
    - Cost layer: RZZ gates on edges (controlled-Z rotations)
    - Mixer layer: RX gates on all qubits

    Based on standard QAOA formulation (Farhi et al. 2014).

    Args:
        n: Number of qubits (problem size)
        p: Number of QAOA layers
        num_edges: Number of edges in problem graph (for cost Hamiltonian)
        mixer_type: Type of mixer ("x" for standard X-mixer, future: "xy", "grover")

    Returns:
        ResourceEstimate with:
            qubits: n
            gates.total: 2*p*n + p*num_edges (RX + RZZ gates)
            gates.single_qubit: 2*p*n (RX gates)
            gates.two_qubit: p*num_edges (RZZ gates)
            depth.total_depth: O(p * (num_edges + n))

    Example:
        >>> import sympy as sp
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> # Complete graph K_n has n*(n-1)/2 edges
        >>> edges = n * (n - 1) / 2
        >>> est = estimate_qaoa(n, p, edges)
        >>> print(est.qubits)  # n
        >>> print(est.gates.total)  # n*p*(n + 3)/2
        >>>
        >>> # MaxCut on K_10 with p=3
        >>> concrete = est.substitute(n=10, p=3)
        >>> print(concrete.gates.total)  # 195

    References:
        - Farhi et al. "A Quantum Approximate Optimization Algorithm"
          arXiv:1411.4028
        - Section 20 of arXiv:2310.03011v2 for variational algorithms
    """
    # Convert to SymPy if needed
    n_expr = sp.Integer(n) if isinstance(n, int) else n
    p_expr = sp.Integer(p) if isinstance(p, int) else p
    edges_expr = sp.Integer(num_edges) if isinstance(num_edges, int) else num_edges

    # Gate counts
    # Each layer has:
    # - RZZ gates on all edges (cost)
    # - 2 RX gates per qubit (mixer: 1 for prep if p=0, 1 for each layer)

    if mixer_type != "x":
        raise NotImplementedError(f"Mixer type '{mixer_type}' not yet implemented")

    # Initial superposition: n H gates
    initial_gates = n_expr

    # Per layer:
    # - num_edges RZZ gates (cost)
    # - n RX gates (mixer)
    gates_per_layer = edges_expr + n_expr
    total_gates = initial_gates + p_expr * gates_per_layer

    # Single-qubit: n H gates + p*n RX gates
    single_qubit = n_expr + p_expr * n_expr

    # Two-qubit: p * num_edges RZZ gates
    two_qubit = p_expr * edges_expr

    # Depth estimation (conservative, assumes sequential)
    # - Initial: n H gates (depth n or 1 if parallel)
    # - Per layer: num_edges RZZ + n RX (sequential)
    # Conservative: sum everything
    total_depth = n_expr + p_expr * (edges_expr + n_expr)

    # Two-qubit depth: only RZZ gates
    two_qubit_depth = p_expr * edges_expr

    # T gates and Clifford gates:
    # Standard QAOA uses only rotations (Rx, RZZ), no T gates
    # Clifford gates: only the initial H gates
    t_gates = sp.Integer(0)
    clifford = n_expr  # H gates

    return ResourceEstimate(
        qubits=n_expr,
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.simplify(single_qubit),
            two_qubit=sp.simplify(two_qubit),
            t_gates=t_gates,
            clifford_gates=clifford,
        ),
        depth=CircuitDepth(
            total_depth=sp.simplify(total_depth),
            t_depth=sp.Integer(0),  # No T gates
            two_qubit_depth=sp.simplify(two_qubit_depth),
        ),
        parameters={
            str(s): s
            for s in [n_expr, p_expr, edges_expr]
            if isinstance(s, sp.Symbol)
        },
    )


def estimate_qaoa_ising(
    n: sp.Expr | int,
    p: sp.Expr | int,
    quadratic_terms: sp.Expr | int,
    linear_terms: sp.Expr | int | None = None,
) -> ResourceEstimate:
    """Estimate resources for QAOA on Ising model.

    Ising Hamiltonian: H = Σ J_ij Z_i Z_j + Σ h_i Z_i

    This is a convenience wrapper around estimate_qaoa() that accepts
    the number of quadratic and linear terms directly.

    Args:
        n: Number of qubits
        p: Number of QAOA layers
        quadratic_terms: Number of J_ij terms (edges in interaction graph)
        linear_terms: Number of h_i terms (defaults to n if None)

    Returns:
        ResourceEstimate for QAOA

    Example:
        >>> # 3-regular graph with n vertices: 3n/2 edges
        >>> import sympy as sp
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> est = estimate_qaoa_ising(n, p, quadratic_terms=3*n/2)
        >>> print(est.gates.total)  # 2*n*p + 3*n*p/2
    """
    linear = sp.Integer(n) if linear_terms is None else linear_terms

    # Ising cost layer uses:
    # - RZZ for each quadratic term
    # - RZ for each linear term (single-qubit)

    # Call base QAOA estimator
    base_est = estimate_qaoa(n, p, num_edges=quadratic_terms)

    # Add linear term contributions
    linear_expr = sp.Integer(linear) if isinstance(linear, int) else linear
    p_expr = sp.Integer(p) if isinstance(p, int) else p

    # Each linear term adds one RZ gate per layer
    extra_single_qubit = p_expr * linear_expr

    return ResourceEstimate(
        qubits=base_est.qubits,
        gates=GateCount(
            total=base_est.gates.total + extra_single_qubit,
            single_qubit=base_est.gates.single_qubit + extra_single_qubit,
            two_qubit=base_est.gates.two_qubit,
            t_gates=base_est.gates.t_gates,
            clifford_gates=base_est.gates.clifford_gates,
        ),
        depth=CircuitDepth(
            # Linear terms can be done in parallel with mixer, so doesn't add depth
            total_depth=base_est.depth.total_depth,
            t_depth=base_est.depth.t_depth,
            two_qubit_depth=base_est.depth.two_qubit_depth,
        ),
        parameters=base_est.parameters,
    )
