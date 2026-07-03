"""Tests for the declarative Hamiltonian workload mixin."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sympy as sp

from qamomile.resource_estimation import (
    HamiltonianQPEWorkload,
    HamiltonianRepresentation,
    HamiltonianWorkloadMixin,
    PauliHamiltonianResource,
    TrotterQPEWorkload,
    compare_resource_values,
)
from qamomile.resource_estimation._common import _as_expr, _SympyLike


@dataclass(frozen=True)
class _ToyAmplitudeWorkload(HamiltonianWorkloadMixin):
    """Model a minimal amplitude-amplification workload for mixin tests."""

    hamiltonian: PauliHamiltonianResource
    grover_rounds: _SympyLike
    oracle_cost_toffoli: _SympyLike = 1
    representation_error: _SympyLike = 0
    description: str = ""

    _POSITIVE_FIELDS = ("grover_rounds", "oracle_cost_toffoli")
    _NONNEGATIVE_FIELDS = ("representation_error",)

    def _own_resource_values(self) -> dict[str, sp.Expr]:
        """Return the toy workload's algorithm-level values.

        Returns:
            dict[str, sp.Expr]: Round and oracle-cost quantities.
        """
        return {
            "grover_rounds": _as_expr(self.grover_rounds, "grover_rounds"),
            "oracle_cost_toffoli": _as_expr(
                self.oracle_cost_toffoli,
                "oracle_cost_toffoli",
            ),
        }


def _summary() -> PauliHamiltonianResource:
    """Build a small Hamiltonian summary shared by the mixin tests.

    Returns:
        PauliHamiltonianResource: Four-qubit toy summary.
    """
    return PauliHamiltonianResource(
        n_qubits=4,
        n_pauli_terms=10,
        lambda_norm=20,
        max_locality=2,
    )


def test_toy_workload_inherits_full_workload_surface():
    """A ~30-line subclass gets validation, values, precision, and to_dict."""
    workload = _ToyAmplitudeWorkload(
        _summary(),
        grover_rounds=8,
        oracle_cost_toffoli=50,
        representation_error=sp.Rational(1, 10),
    )

    values = workload.resource_values()
    assert values["n_qubits"] == 4
    assert values["grover_rounds"] == 8
    assert values["oracle_cost_toffoli"] == 50

    precise = workload.resource_values_for_precision(1)
    assert precise["target_precision"] == 1
    assert precise["algorithmic_precision"] == sp.Rational(9, 10)

    data = workload.to_dict()
    assert data["hamiltonian"]["n_qubits"] == "4"
    assert data["grover_rounds"] == "8"
    assert data["description"] == ""

    rows = compare_resource_values(
        workload,
        workload,
        quantities=("grover_rounds",),
    )
    assert rows[0].ratio == 1


def test_toy_workload_validates_declared_fields():
    """Declared validation tuples reject invalid values at construction."""
    with pytest.raises(ValueError, match="grover_rounds"):
        _ToyAmplitudeWorkload(_summary(), grover_rounds=0)
    with pytest.raises(ValueError, match="representation_error"):
        _ToyAmplitudeWorkload(
            _summary(),
            grover_rounds=1,
            representation_error=-1,
        )
    with pytest.raises(TypeError, match="PauliHamiltonianResource"):
        _ToyAmplitudeWorkload(object(), grover_rounds=1)  # type: ignore[arg-type]


def test_workload_missing_representation_error_fails_at_construction():
    """A subclass without representation_error fails fast with guidance."""

    @dataclass(frozen=True)
    class _NoBudgetWorkload(HamiltonianWorkloadMixin):
        hamiltonian: PauliHamiltonianResource
        rounds: _SympyLike = 1

        _POSITIVE_FIELDS = ("rounds",)

        def _own_resource_values(self) -> dict[str, sp.Expr]:
            """Return the toy workload's values.

            Returns:
                dict[str, sp.Expr]: Round count.
            """
            return {"rounds": _as_expr(self.rounds, "rounds")}

    with pytest.raises(TypeError, match="representation_error"):
        _NoBudgetWorkload(_summary())


def test_builtin_workloads_share_the_mixin_surface():
    """Both built-in workloads inherit the mixin's precision accounting."""
    qpe = HamiltonianQPEWorkload(
        _summary(),
        walk_cost_toffoli=100,
        representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        representation_error=sp.Rational(1, 5),
    )
    trotter = TrotterQPEWorkload(
        _summary(),
        trotter_steps_per_sample=2,
        samples=5,
        representation_error=sp.Rational(1, 5),
    )

    assert isinstance(qpe, HamiltonianWorkloadMixin)
    assert isinstance(trotter, HamiltonianWorkloadMixin)
    assert qpe.algorithmic_precision(1) == sp.Rational(4, 5)
    assert trotter.algorithmic_precision(1) == sp.Rational(4, 5)
    assert qpe.resource_values()["sparsity"] == 10
    assert trotter.resource_values()["trotter_samples"] == 5
