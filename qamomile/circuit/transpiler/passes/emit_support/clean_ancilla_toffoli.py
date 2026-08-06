"""Pure arithmetic for the named clean-ancilla Toffoli ladder recipe."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar, overload

import sympy as sp

_CountT = TypeVar("_CountT")


@dataclasses.dataclass(frozen=True, slots=True)
class CleanAncillaToffoliLadder(Generic[_CountT]):
    """Describe the size of one clean-ancilla Toffoli ladder.

    Args:
        compute_toffolis (_CountT): Toffoli count in the compute half.
        total_toffolis (_CountT): Combined compute and uncompute count.
        clean_ancillas (_CountT): Simultaneously required clean ancillas.
    """

    compute_toffolis: _CountT
    total_toffolis: _CountT
    clean_ancillas: _CountT


@overload
def clean_ancilla_toffoli_ladder(
    num_controls: int,
) -> CleanAncillaToffoliLadder[int]: ...


@overload
def clean_ancilla_toffoli_ladder(
    num_controls: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]: ...


def clean_ancilla_toffoli_ladder(
    num_controls: int | sp.Expr,
) -> CleanAncillaToffoliLadder[Any]:
    """Return the arithmetic for the named clean-ancilla recipe.

    The caller owns recipe selection and must use the result on a branch where
    at least two controls are active. Keeping validation outside this helper
    lets symbolic resource expressions construct inactive Piecewise branches
    without prematurely rejecting them.

    Args:
        num_controls (int | sp.Expr): Concrete or symbolic control count.

    Returns:
        CleanAncillaToffoliLadder[Any]: Compute, total-Toffoli, and clean-
            ancilla requirements in the same numeric domain as the input.
    """
    return _clean_ancilla_toffoli_ladder_from_compute(num_controls - 1)


@overload
def clean_ancilla_toffoli_ladder_or_empty(
    num_controls: int,
) -> CleanAncillaToffoliLadder[int]: ...


@overload
def clean_ancilla_toffoli_ladder_or_empty(
    num_controls: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]: ...


def clean_ancilla_toffoli_ladder_or_empty(
    num_controls: int | sp.Expr,
) -> CleanAncillaToffoliLadder[Any]:
    """Return the named ladder arithmetic including its empty cases.

    The caller guarantees that ``num_controls`` is a nonnegative integer in
    its concrete or symbolic domain. Zero and one control need no conjunction
    ladder; two or more controls use the same named clean-ancilla recipe as
    :func:`clean_ancilla_toffoli_ladder`.

    Args:
        num_controls (int | sp.Expr): Concrete or symbolic nonnegative control
            count.

    Returns:
        CleanAncillaToffoliLadder[Any]: Nonnegative compute, total-Toffoli,
            and clean-ancilla requirements.
    """
    active = clean_ancilla_toffoli_ladder(num_controls)
    compute_toffolis = active.compute_toffolis
    if isinstance(compute_toffolis, int):
        compute_toffolis = max(0, compute_toffolis)
    elif compute_toffolis.is_nonnegative is not True:
        compute_toffolis = sp.Max(sp.Integer(0), compute_toffolis)
    return _clean_ancilla_toffoli_ladder_from_compute(compute_toffolis)


@overload
def _clean_ancilla_toffoli_ladder_from_compute(
    compute_toffolis: int,
) -> CleanAncillaToffoliLadder[int]: ...


@overload
def _clean_ancilla_toffoli_ladder_from_compute(
    compute_toffolis: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]: ...


def _clean_ancilla_toffoli_ladder_from_compute(
    compute_toffolis: int | sp.Expr,
) -> CleanAncillaToffoliLadder[Any]:
    """Build one ladder summary from its compute-half Toffoli count.

    Args:
        compute_toffolis (int | sp.Expr): Toffoli count in the compute half.

    Returns:
        CleanAncillaToffoliLadder[Any]: Complete named-recipe arithmetic.
    """
    return CleanAncillaToffoliLadder(
        compute_toffolis=compute_toffolis,
        total_toffolis=2 * compute_toffolis,
        clean_ancillas=compute_toffolis,
    )


__all__ = [
    "CleanAncillaToffoliLadder",
    "clean_ancilla_toffoli_ladder",
    "clean_ancilla_toffoli_ladder_or_empty",
]
