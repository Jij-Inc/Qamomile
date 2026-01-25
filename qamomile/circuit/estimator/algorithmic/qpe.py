"""Theoretical resource estimates for QPE (Quantum Phase Estimation).

Based on Section 13 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.depth_estimator import CircuitDepth


def estimate_qpe(
    n_system: sp.Expr | int,
    precision: sp.Expr | int,
    hamiltonian_norm: sp.Expr | float | None = None,
    method: str = "qubitization",
) -> ResourceEstimate:
    """Estimate resources for Quantum Phase Estimation.

    QPE estimates the eigenvalue of a unitary operator U = e^(iHt).
    Two main approaches:
    1. Trotter-based: Approximate e^(iHt) using Trotter formulas
    2. Qubitization: Use block-encoding with quantum walk operator

    Args:
        n_system: Number of system qubits (qubits in the state being analyzed)
        precision: Number of bits of precision in phase estimate (ε = 2^(-precision))
        hamiltonian_norm: Normalization ||H|| for block-encoding (required for qubitization)
        method: "qubitization" (recommended) or "trotter"

    Returns:
        ResourceEstimate with qubit and gate counts

    For qubitization method (Section 13, arXiv:2310.03011):
        qubits: n_system + precision + O(log n_system) ancillas
        calls to block-encoding: O(||H|| * 2^precision)
        gates per call: depends on Hamiltonian structure

    For Trotter method:
        qubits: n_system + precision
        depth: O(2^precision * Trotter_depth)

    Example:
        >>> import sympy as sp
        >>> n = sp.Symbol('n', positive=True, integer=True)
        >>> m = sp.Symbol('m', positive=True, integer=True)  # precision bits
        >>> alpha = sp.Symbol('alpha', positive=True)  # ||H||
        >>>
        >>> est = estimate_qpe(n, m, hamiltonian_norm=alpha)
        >>> print(est.qubits)  # n + m + O(log n)
        >>> print(est.gates.total)  # O(alpha * 2^m)
        >>>
        >>> # Concrete example: 100 qubits, 10 bits precision
        >>> concrete = est.substitute(n=100, m=10, alpha=50)
        >>> print(concrete.qubits)  # ~117 (100 + 10 + log2(100))

    References:
        - Nielsen & Chuang, Section 5.2: Original QPE
        - Section 13 of arXiv:2310.03011v2: Modern variants
        - Low & Chuang arXiv:1610.06546: Qubitization-based QPE
    """
    # Convert to SymPy
    n_expr = sp.Integer(n_system) if isinstance(n_system, int) else n_system
    prec_expr = sp.Integer(precision) if isinstance(precision, int) else precision

    if method == "qubitization":
        if hamiltonian_norm is None:
            raise ValueError("hamiltonian_norm required for qubitization method")

        alpha = (
            sp.Float(hamiltonian_norm)
            if isinstance(hamiltonian_norm, (int, float))
            else hamiltonian_norm
        )

        # Qubits: n_system + precision + ancillas for block-encoding
        # Conservative estimate: log2(n_system) ancillas
        ancilla_qubits = sp.ceiling(sp.log(n_expr, 2))
        total_qubits = n_expr + prec_expr + ancilla_qubits

        # Calls to block-encoding: O(α * 2^m) where m is precision
        # This is from the analysis in Section 13
        block_encoding_calls = alpha * sp.Pow(2, prec_expr)

        # Each block-encoding call requires some gates
        # This depends heavily on Hamiltonian structure
        # Conservative: assume O(n) gates per call
        gates_per_call = n_expr

        total_gates = block_encoding_calls * gates_per_call

        # Circuit depth: sequential execution of controlled-U operations
        # Depth ≈ number of calls * depth per call
        depth_per_call = n_expr  # Conservative
        total_depth = block_encoding_calls * depth_per_call

        return ResourceEstimate(
            qubits=sp.simplify(total_qubits),
            gates=GateCount(
                total=sp.simplify(total_gates),
                single_qubit=sp.simplify(total_gates / 2),  # Rough estimate
                two_qubit=sp.simplify(total_gates / 2),
                t_gates=sp.Integer(0),  # Depends on gate decomposition
                clifford_gates=sp.Integer(0),
            ),
            depth=CircuitDepth(
                total_depth=sp.simplify(total_depth),
                t_depth=sp.Integer(0),
                two_qubit_depth=sp.simplify(total_depth / 2),
            ),
            parameters={
                str(s): s
                for s in [n_expr, prec_expr, alpha]
                if isinstance(s, sp.Symbol)
            },
        )

    elif method == "trotter":
        # Simpler Trotter-based QPE
        # Qubits: n_system + precision (no extra ancillas needed)
        total_qubits = n_expr + prec_expr

        # Number of Trotter steps depends on target precision
        # Conservative: O(2^m) controlled-U operations
        num_operations = sp.Pow(2, prec_expr)

        # Gates: each controlled-U implemented via Trotter
        # Assume O(n) gates per Trotter step
        total_gates = num_operations * n_expr

        return ResourceEstimate(
            qubits=sp.simplify(total_qubits),
            gates=GateCount(
                total=sp.simplify(total_gates),
                single_qubit=sp.simplify(total_gates / 2),
                two_qubit=sp.simplify(total_gates / 2),
                t_gates=sp.Integer(0),
                clifford_gates=sp.Integer(0),
            ),
            depth=CircuitDepth(
                total_depth=sp.simplify(num_operations * n_expr),
                t_depth=sp.Integer(0),
                two_qubit_depth=sp.simplify(num_operations * n_expr / 2),
            ),
            parameters={
                str(s): s for s in [n_expr, prec_expr] if isinstance(s, sp.Symbol)
            },
        )

    else:
        raise ValueError(f"Unknown QPE method: {method}")


def estimate_eigenvalue_filtering(
    n_system: sp.Expr | int,
    target_overlap: sp.Expr | float,
    gap: sp.Expr | float | None = None,
) -> ResourceEstimate:
    """Estimate resources for eigenstate filtering (QSVT-based).

    Uses quantum singular value transformation to filter eigenstates
    based on their eigenvalues.

    Based on Section 2.1 (Fermi-Hubbard) and general eigenstate preparation
    methods in arXiv:2310.03011v2.

    Args:
        n_system: Number of system qubits
        target_overlap: Desired overlap γ with target eigenstate
        gap: Spectral gap Δ (if known, improves estimates)

    Returns:
        ResourceEstimate

    Complexity: O(1/γ) calls to block-encoding if no gap known
                O(1/√γΔ) if gap Δ is known

    Example:
        >>> import sympy as sp
        >>> n = sp.Symbol('n', positive=True, integer=True)
        >>> gamma = sp.Symbol('gamma', positive=True)  # overlap
        >>>
        >>> est = estimate_eigenvalue_filtering(n, gamma)
        >>> print(est.gates.total)  # O(n/gamma)

    References:
        - Lin & Tong arXiv:1910.14596: Eigenstate filtering via QSVT
    """
    n_expr = sp.Integer(n_system) if isinstance(n_system, int) else n_system
    gamma = (
        sp.Float(target_overlap)
        if isinstance(target_overlap, (int, float))
        else target_overlap
    )

    # Number of calls to block-encoding
    if gap is not None:
        delta = sp.Float(gap) if isinstance(gap, (int, float)) else gap
        # With gap: O(1/√γΔ)
        num_calls = 1 / sp.sqrt(gamma * delta)
    else:
        # Without gap: O(1/γ)
        num_calls = 1 / gamma

    # Gates per call: O(n) for Hamiltonian block-encoding
    total_gates = num_calls * n_expr

    return ResourceEstimate(
        qubits=n_expr,
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.simplify(total_gates / 2),
            two_qubit=sp.simplify(total_gates / 2),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=sp.simplify(total_gates),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.simplify(total_gates / 2),
        ),
        parameters={str(s): s for s in [n_expr, gamma] if isinstance(s, sp.Symbol)},
    )
