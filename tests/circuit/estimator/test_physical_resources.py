"""Tests for the surface-code physical resource model."""

from __future__ import annotations

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


def test_surface_code_estimate_reproduces_rsa2048_worksheet() -> None:
    """Symbolic 3n / 0.3n^3 inputs reproduce the RSA-2048 worksheet numbers."""
    n = sp.Symbol("n", positive=True)
    est = surface_code_estimate(3 * n, sp.Rational(3, 10) * n**3)

    # Book/worksheet figures at n = 2048: d=24, ~1.42e7 physical qubits, ~17.2h.
    assert int(est.code_distance.subs(n, 2048)) == 24
    assert abs(float(est.physical_qubits.subs(n, 2048)) - 1.4156e7) / 1.4156e7 < 0.01
    assert abs(float(est.runtime_hours.subs(n, 2048)) - 17.2) < 0.1


def test_surface_code_estimate_is_symbolic_in_n() -> None:
    """The code distance and qubit count stay symbolic expressions in n."""
    n = sp.Symbol("n", positive=True)
    est = surface_code_estimate(3 * n, sp.Rational(3, 10) * n**3)

    assert n in est.physical_qubits.free_symbols
    assert n in est.code_distance.free_symbols


def test_estimate_physical_resources_reads_logical_estimate() -> None:
    """estimate_physical_resources pulls N and M from a ResourceEstimate."""
    n = sp.Symbol("n", positive=True)
    logical = ResourceEstimate(
        width=WidthResources(peak_qubits=3 * n),
        gates=GateResources(non_clifford=sp.Rational(3, 10) * n**3),
    )
    phys = estimate_physical_resources(logical)

    assert phys.logical_qubits == 3 * n
    assert int(phys.code_distance.subs(n, 2048)) == 24
    assert abs(float(phys.physical_qubits.subs(n, 2048)) - 1.4156e7) / 1.4156e7 < 0.01


def test_estimate_physical_resources_falls_back_to_t_plus_toffoli() -> None:
    """Missing non_clifford falls back to t + toffoli for the magic-state count."""
    logical = ResourceEstimate(
        width=WidthResources(peak_qubits=sp.Integer(10)),
        gates=GateResources(t=sp.Integer(100), toffoli=sp.Integer(50)),
    )
    phys = estimate_physical_resources(logical)

    assert phys.non_clifford_gates == 150


def test_surface_code_estimate_varies_with_architecture() -> None:
    """A faster syndrome cycle shortens the estimated runtime proportionally."""
    n = 2048
    base = surface_code_estimate(3 * n, 0.3 * n**3, syndrome_cycle_seconds=1e-6)
    fast = surface_code_estimate(3 * n, 0.3 * n**3, syndrome_cycle_seconds=1e-7)

    ratio = float(fast.runtime_seconds) / float(base.runtime_seconds)
    assert abs(ratio - 0.1) < 1e-6
