"""Theoretical resource estimates for Hamiltonian simulation.

Provides estimates for multiple simulation methods:
- Product formulas (Trotter/Suzuki)
- qDRIFT
- QSVT/QSP-based methods

Based on Section 11 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.resource_estimation._common import (
    _as_expr,
    _validate_positive,
)


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

    This prices a single approximate time evolution at a fixed time and
    error. For a full phase-estimation workload driven by a
    normalization/precision budget — with per-sample step counts and
    rotation-synthesis costs — use ``estimate_trotter_qpe_resources`` or
    ``TrotterQPEWorkload`` from ``qamomile.resource_estimation``.

    Args:
        n (sp.Expr | int): Number of qubits in the simulated system. Also
            used as the assumed O(n) gate cost per Hamiltonian term.
        L (sp.Expr | int): Number of terms in the Hamiltonian
            decomposition H = Σ H_j.
        time (sp.Expr | float): Evolution time t. Must be positive.
        error (sp.Expr | float): Target simulation error ε (distance to
            the exact evolution). Must be in (0, 1).
        order (int): Trotter/Suzuki order p (2, 4, 6, ...). Higher order
            means fewer steps but a more complex per-step circuit.
            Defaults to 2.
        hamiltonian_1norm (sp.Expr | float | None): Hamiltonian 1-norm
            ||H||_1 = Σ|coefficients|. Defaults to None, in which case
            the conservative assumption ||H||_1 ~ L (unit coefficients)
            is used.

    Returns:
        ResourceEstimate: Estimate with ``qubits = n`` and total gate
            count r * L * n, where r is the pth-order step count below.
            All free symbols appearing in the resulting expressions are
            collected into ``parameters`` for later ``substitute`` calls.

    Raises:
        TypeError: If ``n``, ``L``, ``time``, ``error``, or
            ``hamiltonian_1norm`` cannot be converted to a SymPy
            expression.
        ValueError: If SymPy can prove that ``n``, ``L``, ``time``,
            ``error``, or ``hamiltonian_1norm`` is not positive.

    Complexity (pth-order formula, Section 11.1):
        Number of steps: r = O((||H||_1 * t)^(1+1/p) / ε^(1/p))
        Total gates: O(r * L * n) where n is gates per Hamiltonian term

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_trotter
        >>>
        >>> # Second-order Trotter
        >>> n, L, t, eps = sp.symbols('n L t eps', positive=True)
        >>> est2 = estimate_trotter(n, L, t, eps, order=2)
        >>> est2.gates.total
        L**(5/2)*n*t**(3/2)/sqrt(eps)
        >>>
        >>> # Fourth-order Trotter (fewer steps, better scaling)
        >>> est4 = estimate_trotter(n, L, t, eps, order=4)
        >>> est4.gates.total
        L**(9/4)*n*t**(5/4)/eps**(1/4)
        >>>
        >>> # Concrete: 100 qubits, 1000 terms, time=10, error=0.001
        >>> concrete = est2.substitute(n=100, L=1000, t=10, eps=0.001)

    References:
        - Section 11.1 of arXiv:2310.03011v2
        - Childs et al. arXiv:1912.08854: Improved Trotter bounds
    """
    # Convert to SymPy and validate
    n_expr = _as_expr(n, "n")
    L_expr = _as_expr(L, "L")
    t_expr = _as_expr(time, "time")
    eps_expr = _as_expr(error, "error")
    _validate_positive(n_expr, "n")
    _validate_positive(L_expr, "L")
    _validate_positive(t_expr, "time")
    _validate_positive(eps_expr, "error")

    # ||H||_1 estimate
    if hamiltonian_1norm is None:
        h1norm = L_expr  # Conservative: assume unit coefficients
    else:
        h1norm = _as_expr(hamiltonian_1norm, "hamiltonian_1norm")
        _validate_positive(h1norm, "hamiltonian_1norm")

    # Number of Trotter steps for pth-order formula
    # r = O((||H||_1 * t)^(1 + 1/p) / ε^(1/p))
    p = order
    exponent_time = sp.Rational(1 + 1 / p)
    exponent_error = sp.Rational(1 / p)

    num_steps = sp.Pow(h1norm * t_expr, exponent_time) / sp.Pow(
        eps_expr, exponent_error
    )

    # Each step applies all L Hamiltonian terms
    # Each term typically requires O(1) to O(n) gates
    # Conservative: O(n) gates per term
    gates_per_step = L_expr * n_expr

    total_gates = num_steps * gates_per_step

    return ResourceEstimate(
        qubits=n_expr,
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.simplify(total_gates / 2),  # Rough split
            two_qubit=sp.simplify(total_gates / 2),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),  # Depends on decomposition
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        ),
    ).with_collected_parameters()


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
        n (sp.Expr | int): Number of qubits. Also used as the assumed
            O(n) gate cost per block-encoding call.
        hamiltonian_norm (sp.Expr | float): Block-encoding normalization
            α with ||H|| ≤ α. Must be positive.
        time (sp.Expr | float): Evolution time t. Must be positive.
        error (sp.Expr | float): Target simulation error ε. Must be in
            (0, 1).

    Returns:
        ResourceEstimate: Estimate with ``qubits = n`` and total gate
            count (α*t + log(1/ε)/log(log(1/ε))) * n. All free symbols
            appearing in the resulting expressions are collected into
            ``parameters`` for later ``substitute`` calls.

    Raises:
        TypeError: If ``n``, ``hamiltonian_norm``, ``time``, or ``error``
            cannot be converted to a SymPy expression.
        ValueError: If SymPy can prove that ``n``, ``hamiltonian_norm``,
            ``time``, or ``error`` is not positive.

    Complexity (Section 11.4, arXiv:2310.03011v2):
        Calls to block-encoding: O(α*t + log(1/ε) / log(log(1/ε)))

    This is nearly optimal: Ω(α*t + log(1/ε)) lower bound known.

    QSVT scales much better than Trotter for small error:
    QSVT is O(αt + log 1/ε) while second-order Trotter is O(α t^1.5 / √ε).

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_qsvt
        >>> n, alpha, t, eps = sp.symbols('n alpha t eps', positive=True)
        >>> est = estimate_qsvt(n, alpha, t, eps)
        >>> est.gates.total
        alpha*n*t - n*log(eps)/log(-log(eps))

    References:
        - Section 11.4 of arXiv:2310.03011v2
        - Low & Chuang arXiv:1610.06546: QSVT framework
        - Gilyen et al. arXiv:1806.01838: QSP/QSVT
    """
    # Convert to SymPy and validate
    n_expr = _as_expr(n, "n")
    alpha = _as_expr(hamiltonian_norm, "hamiltonian_norm")
    t_expr = _as_expr(time, "time")
    eps_expr = _as_expr(error, "error")
    _validate_positive(n_expr, "n")
    _validate_positive(alpha, "hamiltonian_norm")
    _validate_positive(t_expr, "time")
    _validate_positive(eps_expr, "error")

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
            rotation_gates=sp.Integer(0),
        ),
    ).with_collected_parameters()


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
        L (sp.Expr | int): Number of terms in the Hamiltonian. Validated
            but not used by the sample-count formula, which depends only
            on ||H||_1, t, and ε; a symbolic ``L`` therefore does not
            appear in the collected ``parameters``.
        hamiltonian_1norm (sp.Expr | float): Hamiltonian 1-norm
            ||H||_1 = Σ|coefficients|. Must be positive.
        time (sp.Expr | float): Evolution time t. Must be positive.
        error (sp.Expr | float): Target simulation error ε. Must be in
            (0, 1).

    Returns:
        ResourceEstimate: Estimate whose total gate count is the number
            of sampled term applications ||H||_1^2 * t^2 / ε (each term
            assumed O(1) gates). The qubit count is left as a fresh
            symbolic ``n`` because it is not determined by this formula.
            All free symbols appearing in the resulting expressions
            (including the fresh ``n``) are collected into
            ``parameters`` for later ``substitute`` calls.

    Raises:
        TypeError: If ``L``, ``hamiltonian_1norm``, ``time``, or
            ``error`` cannot be converted to a SymPy expression.
        ValueError: If SymPy can prove that ``L``,
            ``hamiltonian_1norm``, ``time``, or ``error`` is not
            positive.

    Complexity (Section 11.2, arXiv:2310.03011v2):
        Number of samples: N = O(||H||_1^2 * t^2 / ε)

    Note quadratic dependence on time (bad) but linear in error (good).
    Best for small t or when simplicity is valued over gate count.
    Compared to second-order Trotter, O((h1*t)^1.5 / √ε), qDRIFT is
    worse for large t but better for small ε.

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_qdrift
        >>> L, h1, t, eps = sp.symbols('L h1 t eps', positive=True)
        >>> est = estimate_qdrift(L, h1, t, eps)
        >>> est.gates.total
        h1**2*t**2/eps

    References:
        - Section 11.2 of arXiv:2310.03011v2
        - Campbell arXiv:1811.08017: qDRIFT algorithm
    """
    # Convert to SymPy and validate
    L_expr = _as_expr(L, "L")
    h1_expr = _as_expr(hamiltonian_1norm, "hamiltonian_1norm")
    t_expr = _as_expr(time, "time")
    eps_expr = _as_expr(error, "error")
    _validate_positive(L_expr, "L")
    _validate_positive(h1_expr, "hamiltonian_1norm")
    _validate_positive(t_expr, "time")
    _validate_positive(eps_expr, "error")

    # Number of random samples
    num_samples = (h1_expr**2) * (t_expr**2) / eps_expr

    # Each sample applies one Hamiltonian term
    # Assume O(1) gates per term (local Hamiltonian)
    # For n-qubit systems, could be O(n) gates
    # Here we report just the number of term applications
    total_operations = num_samples

    return ResourceEstimate(
        qubits=sp.Symbol(
            "n", integer=True, positive=True
        ),  # Not determined by this formula
        gates=GateCount(
            total=sp.simplify(total_operations),
            single_qubit=sp.Integer(0),  # Unknown without knowing term structure
            two_qubit=sp.Integer(0),
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        ),
    ).with_collected_parameters()
