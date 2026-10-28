"""Prove narrow dependency shapes whose scalar depth composes exactly."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy as sp

from qamomile.circuit.estimator._constants import _ONE, _ZERO
from qamomile.circuit.estimator._dependency_footprints import (
    _is_classical_dependency_key,
)
from qamomile.circuit.estimator._resource_base import ResourceExpr
from qamomile.circuit.estimator._resource_expressions import _safe_simplify

if TYPE_CHECKING:
    from qamomile.circuit.estimator._estimate import ResourceEstimate


def _has_single_serializing_quantum_dependency(
    estimate: ResourceEstimate,
) -> bool:
    """Return whether one physical qubit serializes the whole estimate.

    A ``None`` wire index may mean either one scalar qubit or a wildcard for
    an entire register. It is therefore accepted only when captured-input
    metadata proves that the allocation owner has width one.

    Args:
        estimate (ResourceEstimate): Estimate whose dependency footprint and
            captured-input widths should be inspected.

    Returns:
        bool: Whether exactly one caller-visible physical qubit is used and no
            hidden reusable workspace participates in the estimate.
    """
    if any(
        demand != _ZERO
        for demand in (
            estimate.width.allocated_qubits,
            estimate.width.clean_ancilla_qubits,
            estimate.width.dirty_ancilla_qubits,
        )
    ):
        return False
    keys = estimate._dependency_keys
    if keys is None:
        return False
    quantum_keys = tuple(key for key in keys if not _is_classical_dependency_key(key))
    if len(quantum_keys) != 1:
        return False
    owner, index = quantum_keys[0]
    if isinstance(index, (int, sp.Expr)):
        return True
    if index is not None:
        return False
    owner_size = estimate._input_sizes.get(owner)
    return (
        owner_size is not None
        and _safe_simplify(cast(ResourceExpr, owner_size - _ONE)) == _ZERO
    )
