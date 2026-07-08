"""Convert logical resource estimates into physical surface-code estimates.

**Experimental.** This module is intentionally not re-exported from
``qamomile.circuit.estimator``; import it explicitly
(``from qamomile.circuit.estimator.physical import ...``). It layers a
fault-tolerance model on top of the logical estimator and is kept separate so
logical estimation and physical assumptions do not blur together. The model and
its API may change.

The logical :class:`ResourceEstimate` produced by
:mod:`qamomile.circuit.estimator.resource_estimator` counts logical qubits and
(non-Clifford) gates. Turning those into *physical* qubit counts and wall-clock
runtime requires a fault-tolerance model. This module implements the toy
surface-code / lattice-surgery back-of-the-envelope model used for high-level
resource estimates such as the RSA-2048 factoring numbers in the literature:

    d               ~= ceil(2 * log(alpha * N * M) / log(p_th / p))
    physical_qubits ~= 4 * N * d**2
    runtime         ~= M * d * tau

where ``N`` is the logical qubit count, ``M`` is the non-Clifford (magic-state)
gate count, ``p`` is the physical error rate, ``p_th`` is the surface-code
threshold, ``alpha`` is a constant prefactor, and ``tau`` is the syndrome-cycle
time. Every quantity is kept symbolic (``sympy``) so a symbolic logical estimate
(e.g. ``non_clifford = 0.3 * n**3``) flows straight through to a symbolic
physical estimate.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

ResourceExpr = sp.Expr


@dataclasses.dataclass(frozen=True)
class PhysicalResourceEstimate:
    """Hold a surface-code physical-resource estimate.

    Args:
        logical_qubits (ResourceExpr): Logical qubit count ``N`` used as input.
        non_clifford_gates (ResourceExpr): Non-Clifford (magic-state) gate count
            ``M`` used as input.
        code_distance (ResourceExpr): Surface-code distance ``d``.
        physical_qubits (ResourceExpr): Estimated physical qubit count.
        runtime_seconds (ResourceExpr): Estimated wall-clock runtime in seconds.
        qubit_seconds (ResourceExpr): Spacetime volume in physical
            qubit-seconds (``physical_qubits * runtime_seconds``).
        physical_error_rate (float): Physical gate error rate ``p`` assumed.
        threshold (float): Surface-code threshold ``p_th`` assumed.
        alpha (float): Prefactor ``alpha`` in the code-distance formula.
        syndrome_cycle_seconds (float): Syndrome-cycle time ``tau`` in seconds.
    """

    logical_qubits: ResourceExpr
    non_clifford_gates: ResourceExpr
    code_distance: ResourceExpr
    physical_qubits: ResourceExpr
    runtime_seconds: ResourceExpr
    qubit_seconds: ResourceExpr
    physical_error_rate: float
    threshold: float
    alpha: float
    syndrome_cycle_seconds: float

    @property
    def runtime_hours(self) -> ResourceExpr:
        """Return the estimated runtime in hours.

        Returns:
            ResourceExpr: ``runtime_seconds / 3600``.
        """
        return self.runtime_seconds / sp.Integer(3600)


def surface_code_estimate(
    logical_qubits: ResourceExpr | float | int,
    non_clifford_gates: ResourceExpr | float | int,
    *,
    physical_error_rate: float = 1e-3,
    threshold: float = 1e-2,
    alpha: float = 0.05,
    syndrome_cycle_seconds: float = 1e-6,
) -> PhysicalResourceEstimate:
    """Estimate physical surface-code resources from logical counts.

    Implements the Chapter-27-style toy model: choose a code distance large
    enough to suppress the logical error rate below ``1 / (N * M)``, then read
    off physical qubits and runtime.

    Args:
        logical_qubits (ResourceExpr | float | int): Logical qubit count ``N``.
            May be symbolic.
        non_clifford_gates (ResourceExpr | float | int): Non-Clifford gate count
            ``M`` (magic states consumed). May be symbolic.
        physical_error_rate (float): Physical gate error rate ``p``. Defaults to
            ``1e-3``.
        threshold (float): Surface-code threshold ``p_th``. Defaults to
            ``1e-2``.
        alpha (float): Prefactor ``alpha`` in the distance formula. Defaults to
            ``0.05``.
        syndrome_cycle_seconds (float): Syndrome-cycle time ``tau`` in seconds.
            Defaults to ``1e-6`` (1 microsecond).

    Returns:
        PhysicalResourceEstimate: Physical estimate with a possibly-symbolic
        code distance, physical qubit count, and runtime.

    Raises:
        ValueError: If ``threshold`` is not strictly greater than
            ``physical_error_rate`` (the code cannot suppress errors otherwise).

    Example:
        >>> import sympy as sp
        >>> n = sp.Symbol("n", positive=True)
        >>> est = surface_code_estimate(3 * n, sp.Rational(3, 10) * n**3)
        >>> est.physical_qubits.subs(n, 2048).evalf()  # doctest: +ELLIPSIS
        1...e+7
    """
    if threshold <= physical_error_rate:
        raise ValueError(
            "surface_code_estimate requires threshold > physical_error_rate; "
            f"got threshold={threshold}, physical_error_rate={physical_error_rate}."
        )
    n = sp.sympify(logical_qubits)
    m = sp.sympify(non_clifford_gates)
    ratio = sp.log(sp.Float(threshold) / sp.Float(physical_error_rate))
    # For small circuits ``alpha * N * M`` can fall below 1, making the raw
    # distance formula non-positive; clamp to the smallest sensible odd surface
    # code distance so downstream qubit/runtime figures stay meaningful.
    raw_distance = sp.ceiling(2 * sp.log(sp.Float(alpha) * n * m) / ratio)
    distance = sp.Max(raw_distance, sp.Integer(3))
    physical_qubits = 4 * n * distance**2
    runtime_seconds = m * distance * sp.Float(syndrome_cycle_seconds)
    return PhysicalResourceEstimate(
        logical_qubits=n,
        non_clifford_gates=m,
        code_distance=distance,
        physical_qubits=physical_qubits,
        runtime_seconds=runtime_seconds,
        qubit_seconds=physical_qubits * runtime_seconds,
        physical_error_rate=physical_error_rate,
        threshold=threshold,
        alpha=alpha,
        syndrome_cycle_seconds=syndrome_cycle_seconds,
    )


def estimate_physical_resources(
    estimate: "ResourceEstimate",
    *,
    logical_qubits: ResourceExpr | float | int | None = None,
    non_clifford_gates: ResourceExpr | float | int | None = None,
    physical_error_rate: float = 1e-3,
    threshold: float = 1e-2,
    alpha: float = 0.05,
    syndrome_cycle_seconds: float = 1e-6,
) -> PhysicalResourceEstimate:
    """Estimate physical resources directly from a logical resource estimate.

    Reads the logical qubit count and non-Clifford gate count from a
    :class:`ResourceEstimate` and feeds them into :func:`surface_code_estimate`.
    The non-Clifford count falls back to ``t + toffoli`` when the estimate does
    not populate ``gates.non_clifford`` explicitly.

    Args:
        estimate (ResourceEstimate): Logical resource estimate to convert.
        logical_qubits (ResourceExpr | float | int | None): Override for the
            logical qubit count ``N``. Defaults to ``None``, meaning
            ``estimate.qubits`` is used.
        non_clifford_gates (ResourceExpr | float | int | None): Override for the
            non-Clifford gate count ``M``. Defaults to ``None``, meaning the
            value is read from ``estimate.gates``.
        physical_error_rate (float): Physical gate error rate ``p``. Defaults to
            ``1e-3``.
        threshold (float): Surface-code threshold ``p_th``. Defaults to
            ``1e-2``.
        alpha (float): Prefactor ``alpha``. Defaults to ``0.05``.
        syndrome_cycle_seconds (float): Syndrome-cycle time ``tau`` in seconds.
            Defaults to ``1e-6``.

    Returns:
        PhysicalResourceEstimate: Physical estimate derived from the logical
        estimate.

    Raises:
        ValueError: If ``threshold`` is not strictly greater than
            ``physical_error_rate``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> # est = kernel.estimate_resources(bindings={"n": 2048})
        >>> # phys = estimate_physical_resources(est)
    """
    n = estimate.qubits if logical_qubits is None else logical_qubits
    if non_clifford_gates is None:
        gates = estimate.gates
        m: ResourceExpr = sp.sympify(gates.non_clifford)
        if m == 0:
            m = sp.sympify(gates.t) + sp.sympify(gates.toffoli)
    else:
        m = sp.sympify(non_clifford_gates)
    return surface_code_estimate(
        n,
        m,
        physical_error_rate=physical_error_rate,
        threshold=threshold,
        alpha=alpha,
        syndrome_cycle_seconds=syndrome_cycle_seconds,
    )
