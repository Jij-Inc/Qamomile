"""Theoretical resource estimates for Hamiltonian simulation.

Provides estimates for multiple simulation methods:
- Product formulas (Trotter/Suzuki)
- qDRIFT
- QSVT/QSP-based methods

Based on Section 11 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.depth_estimator import CircuitDepth


def estimate_trotter(
    n: sp.Expr | int,
    L: sp.Expr | int,
    time: sp.Expr | float,
    error: sp.Expr | float,
    order: int = 2,
    hamiltonian_1norm: sp.Expr | float | None = None,
) -> ResourceEstimate:
    """Estimate resources for Trotter/Suzuki formula Hamiltonian simulation.

    Product formula methods approximate e^(iHt) by decomposing H into
    a sum of L terms and using the formula:
        e^(iHt) ≈ (e^(iH_1 Δt) ... e^(iH_L Δt))^r

    Args:
        n: Number of qubits
        L: Number of terms in Hamiltonian decomposition
        time: Evolution time t
        error: Target error ε
        order: Trotter order (2, 4, 6, ...). Higher order = fewer steps but more complex
        hamiltonian_1norm: ||H||_1 = Σ|coefficients| (if None, assumes ||H||_1 ~ L)

    Returns:
        ResourceEstimate for Trotter simulation

    Complexity (pth-order formula, Section 11.1):
        Number of steps: r = O((||H||_1 * t)^(1+1/p) / ε^(1/p))
        Total gates: O(r * L * n) where n is gates per Hamiltonian term

    Example:
        >>> import sympy as sp
        >>> n, L, t, eps = sp.symbols('n L t eps', positive=True)
        >>>
        >>> # Second-order Trotter
        >>> est2 = estimate_trotter(n, L, t, eps, order=2)
        >>> print(est2.gates.total)  # O(L * (||H||_1 * t)^1.5 / eps^0.5)
        >>>
        >>> # Fourth-order Trotter (fewer steps, better scaling)
        >>> est4 = estimate_trotter(n, L, t, eps, order=4)
        >>> print(est4.gates.total)  # O(L * (||H||_1 * t)^1.25 / eps^0.25)
        >>>
        >>> # Concrete: 100 qubits, 1000 terms, time=10, error=0.001
        >>> concrete = est2.substitute(n=100, L=1000, t=10, eps=0.001)

    References:
        - Section 11.1 of arXiv:2310.03011v2
        - Childs et al. arXiv:1912.08854: Improved Trotter bounds
    """
    # Convert to SymPy
    n_expr = sp.Integer(n) if isinstance(n, int) else n
    L_expr = sp.Integer(L) if isinstance(L, int) else L
    t_expr = sp.Float(time) if isinstance(time, (int, float)) else time
    eps_expr = sp.Float(error) if isinstance(error, (int, float)) else error

    # ||H||_1 estimate
    if hamiltonian_1norm is None:
        h1norm = L_expr  # Conservative: assume unit coefficients
    else:
        h1norm = (
            sp.Float(hamiltonian_1norm)
            if isinstance(hamiltonian_1norm, (int, float))
            else hamiltonian_1norm
        )

    # Number of Trotter steps for pth-order formula
    # r = O((||H||_1 * t)^(1 + 1/p) / ε^(1/p))
    p = order
    exponent_time = sp.Rational(1 + 1 / p)
    exponent_error = sp.Rational(1 / p)

    num_steps = sp.Pow(h1norm * t_expr, exponent_time) / sp.Pow(eps_expr, exponent_error)

    # Each step applies all L Hamiltonian terms
    # Each term typically requires O(1) to O(n) gates
    # Conservative: O(n) gates per term
    gates_per_step = L_expr * n_expr

    total_gates = num_steps * gates_per_step

    # Depth: sequential application of all terms, all steps
    total_depth = num_steps * L_expr

    return ResourceEstimate(
        qubits=n_expr,
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.simplify(total_gates / 2),  # Rough split
            two_qubit=sp.simplify(total_gates / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),  # Depends on decomposition
            clifford_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=sp.simplify(total_depth),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.simplify(total_depth / 2),
            multi_qubit_depth=sp.Integer(0),
        ),
        parameters={
            str(s): s
            for s in [n_expr, L_expr, t_expr, eps_expr, h1norm]
            if isinstance(s, sp.Symbol)
        },
    )


def estimate_qsvt(
    n: sp.Expr | int,
    hamiltonian_norm: sp.Expr | float,
    time: sp.Expr | float,
    error: sp.Expr | float,
) -> ResourceEstimate:
    """Estimate resources for QSVT-based Hamiltonian simulation.

    Quantum Singular Value Transformation (QSVT) provides near-optimal
    Hamiltonian simulation with complexity linear in time and logarithmic in error.

    Args:
        n: Number of qubits
        hamiltonian_norm: α where ||H|| ≤ α (block-encoding normalization)
        time: Evolution time t
        error: Target error ε

    Returns:
        ResourceEstimate for QSVT simulation

    Complexity (Section 11.4, arXiv:2310.03011v2):
        Calls to block-encoding: O(α*t + log(1/ε) / log(log(1/ε)))

    This is nearly optimal: Ω(α*t + log(1/ε)) lower bound known.

    Example:
        >>> import sympy as sp
        >>> n, alpha, t, eps = sp.symbols('n alpha t eps', positive=True)
        >>>
        >>> est = estimate_qsvt(n, alpha, t, eps)
        >>> print(est.gates.total)  # O(α*t + log(1/ε)/log(log(1/ε)))
        >>>
        >>> # Much better than Trotter for small error!
        >>> trotter = estimate_trotter(n, L=alpha, time=t, error=eps)
        >>> # QSVT: O(αt + log 1/ε), Trotter: O(α t^1.5 / √ε)

    References:
        - Section 11.4 of arXiv:2310.03011v2
        - Low & Chuang arXiv:1610.06546: QSVT framework
        - Gilyen et al. arXiv:1806.01838: QSP/QSVT
    """
    # Convert to SymPy
    n_expr = sp.Integer(n) if isinstance(n, int) else n
    alpha = (
        sp.Float(hamiltonian_norm)
        if isinstance(hamiltonian_norm, (int, float))
        else hamiltonian_norm
    )
    t_expr = sp.Float(time) if isinstance(time, (int, float)) else time
    eps_expr = sp.Float(error) if isinstance(error, (int, float)) else error

    # Number of block-encoding calls
    # First term: linear in time
    linear_term = alpha * t_expr

    # Second term: logarithmic in error (with log-log factor)
    # log(1/ε) / log(log(1/ε))
    log_error = sp.log(1 / eps_expr)
    loglog_error = sp.log(log_error)
    log_term = log_error / loglog_error

    num_calls = linear_term + log_term

    # Each block-encoding call requires O(n) gates
    # (Exact cost depends on Hamiltonian structure)
    gates_per_call = n_expr
    total_gates = num_calls * gates_per_call

    return ResourceEstimate(
        qubits=n_expr,
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.simplify(total_gates / 2),
            two_qubit=sp.simplify(total_gates / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            total_depth=sp.simplify(num_calls * n_expr),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.simplify(num_calls * n_expr / 2),
            multi_qubit_depth=sp.Integer(0),
        ),
        parameters={
            str(s): s
            for s in [n_expr, alpha, t_expr, eps_expr]
            if isinstance(s, sp.Symbol)
        },
    )


def estimate_qdrift(
    L: sp.Expr | int,
    hamiltonian_1norm: sp.Expr | float,
    time: sp.Expr | float,
    error: sp.Expr | float,
) -> ResourceEstimate:
    """Estimate resources for qDRIFT Hamiltonian simulation.

    qDRIFT is a randomized algorithm that samples Hamiltonian terms
    proportionally to their coefficients. Simpler than Trotter but
    requires more samples.

    Args:
        L: Number of terms in Hamiltonian
        hamiltonian_1norm: ||H||_1 = Σ|coefficients|
        time: Evolution time t
        error: Target error ε

    Returns:
        ResourceEstimate for qDRIFT

    Complexity (Section 11.2, arXiv:2310.03011v2):
        Number of samples: N = O(||H||_1^2 * t^2 / ε)

    Note quadratic dependence on time (bad) but linear in error (good).
    Best for small t or when simplicity is valued over gate count.

    Example:
        >>> import sympy as sp
        >>> L, h1, t, eps = sp.symbols('L h1 t eps', positive=True)
        >>>
        >>> est = estimate_qdrift(L, h1, t, eps)
        >>> print(est.gates.total)  # O(h1^2 * t^2 / ε)
        >>>
        >>> # Compare to Trotter (order 2): O((h1*t)^1.5 / √ε)
        >>> # qDRIFT worse for large t, better for small ε

    References:
        - Section 11.2 of arXiv:2310.03011v2
        - Campbell arXiv:1811.08017: qDRIFT algorithm
    """
    # Convert to SymPy
    L_expr = sp.Integer(L) if isinstance(L, int) else L
    h1_expr = (
        sp.Float(hamiltonian_1norm)
        if isinstance(hamiltonian_1norm, (int, float))
        else hamiltonian_1norm
    )
    t_expr = sp.Float(time) if isinstance(time, (int, float)) else time
    eps_expr = sp.Float(error) if isinstance(error, (int, float)) else error

    # Number of random samples
    num_samples = (h1_expr**2) * (t_expr**2) / eps_expr

    # Each sample applies one Hamiltonian term
    # Assume O(1) gates per term (local Hamiltonian)
    # For n-qubit systems, could be O(n) gates
    # Here we report just the number of term applications
    total_operations = num_samples

    return ResourceEstimate(
        qubits=sp.Symbol("n"),  # Not determined by this formula
        gates=GateCount(
            total=sp.simplify(total_operations),
            single_qubit=sp.Integer(0),  # Unknown without knowing term structure
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
        ),
        depth=CircuitDepth(
            # Sequential application
            total_depth=sp.simplify(total_operations),
            t_depth=sp.Integer(0),
            two_qubit_depth=sp.Integer(0),
            multi_qubit_depth=sp.Integer(0),
        ),
        parameters={
            str(s): s
            for s in [L_expr, h1_expr, t_expr, eps_expr]
            if isinstance(s, sp.Symbol)
        },
    )
