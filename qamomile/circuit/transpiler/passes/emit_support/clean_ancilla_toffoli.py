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
) -> CleanAncillaToffoliLadder[int]:
    """Return active ladder arithmetic for a concrete control count.

    Args:
        num_controls (int): Concrete control count on the active recipe branch.

    Returns:
        CleanAncillaToffoliLadder[int]: Concrete ladder requirements.
    """
    ...


@overload
def clean_ancilla_toffoli_ladder(
    num_controls: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]:
    """Return active ladder arithmetic for a symbolic control count.

    Args:
        num_controls (sp.Expr): Symbolic control count on the active recipe
            branch.

    Returns:
        CleanAncillaToffoliLadder[sp.Expr]: Symbolic ladder requirements.
    """
    ...


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
) -> CleanAncillaToffoliLadder[int]:
    """Return nonnegative ladder arithmetic for a concrete control count.

    Args:
        num_controls (int): Concrete nonnegative control count.

    Returns:
        CleanAncillaToffoliLadder[int]: Concrete ladder requirements, including
            the empty cases.
    """
    ...


@overload
def clean_ancilla_toffoli_ladder_or_empty(
    num_controls: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]:
    """Return nonnegative ladder arithmetic for a symbolic control count.

    Args:
        num_controls (sp.Expr): Symbolic nonnegative control count.

    Returns:
        CleanAncillaToffoliLadder[sp.Expr]: Symbolic ladder requirements,
            including the empty cases.
    """
    ...


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
) -> CleanAncillaToffoliLadder[int]:
    """Build concrete ladder arithmetic from its compute-half count.

    Args:
        compute_toffolis (int): Concrete compute-half Toffoli count.

    Returns:
        CleanAncillaToffoliLadder[int]: Complete concrete ladder requirements.
    """
    ...


@overload
def _clean_ancilla_toffoli_ladder_from_compute(
    compute_toffolis: sp.Expr,
) -> CleanAncillaToffoliLadder[sp.Expr]:
    """Build symbolic ladder arithmetic from its compute-half count.

    Args:
        compute_toffolis (sp.Expr): Symbolic compute-half Toffoli count.

    Returns:
        CleanAncillaToffoliLadder[sp.Expr]: Complete symbolic ladder
            requirements.
    """
    ...


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
