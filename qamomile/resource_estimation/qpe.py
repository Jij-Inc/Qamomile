"""Theoretical resource estimates for QPE (Quantum Phase Estimation).

Based on Section 13 of arXiv:2310.03011v2.
"""

from __future__ import annotations

import sympy as sp

from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.resource_estimator import ResourceEstimate
from qamomile.resource_estimation._common import (
    _as_expr,
    _validate_positive,
)


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

    This is a coarse textbook-level bound parameterized by precision bits.
    For qubitized-QPE workloads with explicit walk costs, Hamiltonian
    representations, and error budgets, use
    ``estimate_qubitized_qpe_resources`` or ``HamiltonianQPEWorkload`` from
    ``qamomile.resource_estimation``.

    Args:
        n_system (sp.Expr | int): Number of system qubits (qubits in the
            state being analyzed). Also used as the assumed O(n) gate
            cost per controlled evolution / block-encoding call.
        precision (sp.Expr | int): Number of bits of precision in the
            phase estimate (ε = 2^(-precision)).
        hamiltonian_norm (sp.Expr | float | None): Block-encoding
            normalization ||H||. Must be positive. Required for
            ``method="qubitization"``; ignored for ``method="trotter"``.
            Defaults to None.
        method (str): Estimation method, either ``"qubitization"``
            (recommended) or ``"trotter"``. Defaults to
            ``"qubitization"``.

    Returns:
        ResourceEstimate: Estimate with qubit and gate counts for the
            chosen method (see the per-method breakdown below). All free
            symbols appearing in the resulting expressions are collected
            into ``parameters`` for later ``substitute`` calls.

    Raises:
        TypeError: If ``n_system``, ``precision``, or a required
            ``hamiltonian_norm`` cannot be converted to a SymPy
            expression.
        ValueError: If ``method="qubitization"`` and ``hamiltonian_norm``
            is None, if ``method`` is neither ``"qubitization"`` nor
            ``"trotter"``, or if SymPy can prove that ``n_system``,
            ``precision``, or ``hamiltonian_norm`` is not positive.

    For qubitization method (Section 13, arXiv:2310.03011):
        qubits: n_system + precision + O(log n_system) ancillas
        calls to block-encoding: O(||H|| * 2^precision)
        gates per call: depends on Hamiltonian structure

    For Trotter method:
        qubits: n_system + precision
        depth: O(2^precision * Trotter_depth)

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import estimate_qpe
        >>> n = sp.Symbol('n', positive=True, integer=True)
        >>> m = sp.Symbol('m', positive=True, integer=True)  # precision bits
        >>> alpha = sp.Symbol('alpha', positive=True)  # ||H||
        >>>
        >>> est = estimate_qpe(n, m, hamiltonian_norm=alpha)
        >>> est.qubits
        m + n + ceiling(log(n)/log(2))
        >>> est.gates.total
        2**m*alpha*n
        >>>
        >>> # Concrete example: 100 qubits, 10 bits precision
        >>> concrete = est.substitute(n=100, m=10, alpha=50)
        >>> concrete.qubits
        117

    References:
        - Nielsen & Chuang, Section 5.2: Original QPE
        - Section 13 of arXiv:2310.03011v2: Modern variants
        - Low & Chuang arXiv:1610.06546: Qubitization-based QPE
    """
    # Convert to SymPy and validate
    n_expr = _as_expr(n_system, "n_system")
    prec_expr = _as_expr(precision, "precision")
    _validate_positive(n_expr, "n_system")
    _validate_positive(prec_expr, "precision")

    if method == "qubitization":
        if hamiltonian_norm is None:
            raise ValueError("hamiltonian_norm required for qubitization method")

        alpha = _as_expr(hamiltonian_norm, "hamiltonian_norm")
        _validate_positive(alpha, "hamiltonian_norm")

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

        return ResourceEstimate(
            qubits=sp.simplify(total_qubits),
            gates=GateCount(
                total=sp.simplify(total_gates),
                single_qubit=sp.simplify(total_gates / 2),  # Rough estimate
                two_qubit=sp.simplify(total_gates / 2),
                multi_qubit=sp.Integer(0),
                t_gates=sp.Integer(0),  # Depends on gate decomposition
                clifford_gates=sp.Integer(0),
                rotation_gates=sp.Integer(0),
            ),
        ).with_collected_parameters()

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
                multi_qubit=sp.Integer(0),
                t_gates=sp.Integer(0),
                clifford_gates=sp.Integer(0),
                rotation_gates=sp.Integer(0),
            ),
        ).with_collected_parameters()

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
        n_system (sp.Expr | int): Number of system qubits. Also used as
            the assumed O(n) gate cost per block-encoding call.
        target_overlap (sp.Expr | float): Desired overlap γ with the
            target eigenstate. Must be in (0, 1].
        gap (sp.Expr | float | None): Spectral gap Δ. Must be positive
            when provided. Defaults to None; when provided, the tighter
            O(1/√(γΔ)) call count is used.

    Returns:
        ResourceEstimate: Estimate with ``qubits = n_system`` and total
            gate count n_system/γ (no gap) or n_system/√(γΔ) (with gap).
            All free symbols appearing in the resulting expressions are
            collected into ``parameters`` for later ``substitute`` calls.

    Raises:
        TypeError: If ``n_system``, ``target_overlap``, or a provided
            ``gap`` cannot be converted to a SymPy expression.
        ValueError: If SymPy can prove that ``n_system``,
            ``target_overlap``, or a provided ``gap`` is not positive.

    Complexity: O(1/γ) calls to block-encoding if no gap known
                O(1/√γΔ) if gap Δ is known

    Example:
        >>> import sympy as sp
        >>> from qamomile.resource_estimation import (
        ...     estimate_eigenvalue_filtering,
        ... )
        >>> n = sp.Symbol('n', positive=True, integer=True)
        >>> gamma = sp.Symbol('gamma', positive=True)  # overlap
        >>>
        >>> est = estimate_eigenvalue_filtering(n, gamma)
        >>> est.gates.total
        n/gamma

    References:
        - Lin & Tong arXiv:1910.14596: Eigenstate filtering via QSVT
    """
    n_expr = _as_expr(n_system, "n_system")
    gamma = _as_expr(target_overlap, "target_overlap")
    _validate_positive(n_expr, "n_system")
    _validate_positive(gamma, "target_overlap")

    # Number of calls to block-encoding
    if gap is not None:
        delta = _as_expr(gap, "gap")
        _validate_positive(delta, "gap")
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
            multi_qubit=sp.Integer(0),
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
        ),
    ).with_collected_parameters()
