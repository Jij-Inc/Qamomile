"""Tests for the surface-code physical resource model."""

from __future__ import annotations

import pytest
import sympy as sp

from qamomile.circuit.estimator import (
    GateResources,
    ResourceEstimate,
    WidthResources,
)
from qamomile.circuit.estimator.physical import (
    estimate_physical_resources,
    surface_code_estimate,
)


def test_surface_code_estimate_rounds_rsa2048_distance_up_to_odd() -> None:
    """The RSA-2048 worksheet distance rounds up to an odd code distance."""
    n = sp.Symbol("n", positive=True)
    est = surface_code_estimate(3 * n, sp.Rational(3, 10) * n**3)

    # The raw worksheet value d=24 must use the next valid odd distance, d=25.
    assert int(est.code_distance.subs(n, 2048)) == 25
    assert float(est.physical_qubits.subs(n, 2048)) == pytest.approx(
        15_360_000,
        rel=1e-12,
    )
    assert abs(float(est.runtime_hours.subs(n, 2048)) - 17.9) < 0.1


def test_surface_code_estimate_is_symbolic_in_n() -> None:
    """The code distance and qubit count stay symbolic expressions in n."""
    n = sp.Symbol("n", positive=True)
    est = surface_code_estimate(3 * n, sp.Rational(3, 10) * n**3)

    assert n in est.physical_qubits.free_symbols
    assert n in est.code_distance.free_symbols


def test_estimate_physical_resources_reads_logical_estimate() -> None:
    """The helper applies its documented logical-family heuristic."""
    n = sp.Symbol("n", positive=True)
    logical = ResourceEstimate(
        width=WidthResources(peak_qubits=3 * n),
        gates=GateResources(non_clifford=sp.Rational(3, 10) * n**3),
    )
    phys = estimate_physical_resources(logical)

    assert phys.logical_qubits == 3 * n
    assert int(phys.code_distance.subs(n, 2048)) == 25
    assert float(phys.physical_qubits.subs(n, 2048)) == pytest.approx(
        15_360_000,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    ("logical_qubits", "non_clifford_gates"),
    [(1, 1), (3 * 2048, sp.Rational(3, 10) * 2048**3), (100, 10**6)],
)
def test_surface_code_estimate_always_selects_an_odd_distance(
    logical_qubits: int,
    non_clifford_gates: int | sp.Expr,
) -> None:
    """Every concrete physical estimate uses an odd code distance."""
    estimate = surface_code_estimate(logical_qubits, non_clifford_gates)

    assert int(estimate.code_distance) >= 3
    assert int(estimate.code_distance) % 2 == 1


def test_estimate_physical_resources_falls_back_to_t_plus_toffoli() -> None:
    """The heuristic falls back to literal T plus Toffoli family counts."""
    logical = ResourceEstimate(
        width=WidthResources(peak_qubits=sp.Integer(10)),
        gates=GateResources(t=sp.Integer(100), toffoli=sp.Integer(50)),
    )
    phys = estimate_physical_resources(logical)

    assert phys.non_clifford_gates == 150


def test_surface_code_estimate_handles_zero_non_clifford_work() -> None:
    """A zero magic-state count uses the minimum distance and zero runtime."""
    phys = surface_code_estimate(5, 0)

    assert phys.code_distance == 3
    assert phys.physical_qubits == 180
    assert phys.runtime_seconds == 0
    assert phys.qubit_seconds == 0


def test_estimate_physical_resources_handles_clifford_only_estimate() -> None:
    """A Clifford-only logical estimate converts without a SymPy crash."""
    logical = ResourceEstimate(
        width=WidthResources(peak_qubits=sp.Integer(5)),
        gates=GateResources(clifford=sp.Integer(10), total=sp.Integer(10)),
    )

    phys = estimate_physical_resources(logical)

    assert phys.non_clifford_gates == 0
    assert phys.code_distance == 3
    assert phys.runtime_seconds == 0


@pytest.mark.parametrize(
    ("logical_qubits", "non_clifford_gates"),
    [(-1, 1), (1, -1), (sp.oo, 1), (1, sp.nan)],
)
def test_surface_code_estimate_rejects_invalid_logical_counts(
    logical_qubits: object,
    non_clifford_gates: object,
) -> None:
    """Physical estimates reject negative and non-finite logical counts."""
    with pytest.raises(ValueError, match="finite nonnegative real value"):
        surface_code_estimate(logical_qubits, non_clifford_gates)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("physical_error_rate", 0.0),
        ("threshold", float("inf")),
        ("alpha", -0.1),
        ("syndrome_cycle_seconds", 0.0),
    ],
)
def test_surface_code_estimate_rejects_invalid_model_coefficients(
    keyword: str,
    value: float,
) -> None:
    """Physical model coefficients must be strictly positive and finite."""
    with pytest.raises(ValueError, match=f"{keyword}.*positive and finite"):
        surface_code_estimate(1, 1, **{keyword: value})


def test_surface_code_estimate_rejects_string_counts() -> None:
    """Physical counts do not parse strings as symbolic expressions."""
    with pytest.raises(TypeError, match="logical_qubits.*numeric or symbolic"):
        surface_code_estimate("1", 1)  # type: ignore[arg-type]


def test_surface_code_estimate_varies_with_architecture() -> None:
    """A faster syndrome cycle shortens the estimated runtime proportionally."""
    n = 2048
    base = surface_code_estimate(3 * n, 0.3 * n**3, syndrome_cycle_seconds=1e-6)
    fast = surface_code_estimate(3 * n, 0.3 * n**3, syndrome_cycle_seconds=1e-7)

    ratio = float(fast.runtime_seconds) / float(base.runtime_seconds)
    assert abs(ratio - 0.1) < 1e-6
