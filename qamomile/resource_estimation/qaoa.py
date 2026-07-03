"""Theoretical resource estimates for QAOA (Quantum Approximate Optimization Algorithm).

Based on Section 4 and Section 20 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.resource_estimator import ResourceEstimate


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
        n (sp.Expr | int): Number of qubits (problem size).
        p (sp.Expr | int): Number of QAOA layers.
        num_edges (sp.Expr | int): Number of edges in the problem graph
            (quadratic terms of the cost Hamiltonian).
        mixer_type (str): Type of mixer. Only ``"x"`` (standard X-mixer)
            is currently supported; ``"xy"`` and ``"grover"`` are
            planned. Defaults to ``"x"``.

    Returns:
        ResourceEstimate: Estimate with:
            qubits: n
            gates.total: n + p*(num_edges + n) (initial H layer, then
                RZZ + RX gates per layer)
            gates.single_qubit: n + p*n (H gates + RX gates)
            gates.two_qubit: p*num_edges (RZZ gates)
            gates.clifford_gates: n (initial H gates)
            gates.rotation_gates: p*(num_edges + n) (RZZ + RX gates)

    Raises:
        NotImplementedError: If ``mixer_type`` is not ``"x"``.

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_qaoa
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> # Complete graph K_n has n*(n-1)/2 edges
        >>> edges = n * (n - 1) / 2
        >>> est = estimate_qaoa(n, p, edges)
        >>> print(est.qubits)  # n
        >>> print(est.gates.total)  # n*(p*(n + 1) + 2)/2
        >>>
        >>> # MaxCut on K_10 with p=3
        >>> concrete = est.substitute(n=10, p=3)
        >>> print(concrete.gates.total)  # 175

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
            multi_qubit=sp.Integer(0),
            t_gates=t_gates,
            clifford_gates=clifford,
            rotation_gates=sp.simplify(p_expr * (edges_expr + n_expr)),
        ),
        parameters={
            str(s): s for s in [n_expr, p_expr, edges_expr] if isinstance(s, sp.Symbol)
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
        n (sp.Expr | int): Number of qubits.
        p (sp.Expr | int): Number of QAOA layers.
        quadratic_terms (sp.Expr | int): Number of J_ij terms (edges in
            the interaction graph), each costing one RZZ gate per layer.
        linear_terms (sp.Expr | int | None): Number of h_i terms, each
            costing one RZ gate per layer. Defaults to None, meaning n
            linear terms (one per qubit); this default requires ``n`` to
            be a plain int — pass ``linear_terms`` explicitly when ``n``
            is symbolic.

    Returns:
        ResourceEstimate: The estimate_qaoa() result for ``n``, ``p``,
            and ``quadratic_terms``, with p*linear_terms RZ gates added
            to the total, single-qubit, and rotation counts.

    Raises:
        TypeError: If ``linear_terms`` is None while ``n`` is a symbolic
            expression (the default of n linear terms cannot be
            constructed from a symbol).

    Example:
        >>> # 3-regular graph with n vertices: 3n/2 edges
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_qaoa_ising
        >>> n, p = sp.symbols('n p', positive=True, integer=True)
        >>> est = estimate_qaoa_ising(
        ...     n, p, quadratic_terms=3 * n / 2, linear_terms=n
        ... )
        >>> print(sp.simplify(est.gates.total))  # n*(7*p + 2)/2
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
            multi_qubit=sp.Integer(0),
            t_gates=base_est.gates.t_gates,
            clifford_gates=base_est.gates.clifford_gates,
            rotation_gates=base_est.gates.rotation_gates + extra_single_qubit,
        ),
        parameters=base_est.parameters,
    )
