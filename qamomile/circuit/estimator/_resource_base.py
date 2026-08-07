"""Define resource scalar aliases, model enums, and validation primitives."""

from __future__ import annotations

import enum
from typing import cast

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

ResourceExpr = sp.Expr
_SUM_EAGER_EVALUATION_LIMIT = 1024
_SYMPY_SIMPLIFICATION_ERRORS = (
    ArithmeticError,
    AttributeError,
    BasePolynomialError,
    NotImplementedError,
    RecursionError,
    TypeError,
    ValueError,
)


def _symbol_display_name(symbol: sp.Basic) -> str:
    """Return a stable user-facing name for a SymPy symbol.

    ``Dummy`` symbols print with a leading underscore even though their
    declared ``name`` is unchanged. Resource input/substitution APIs are keyed
    by the declared Qamomile name, so identity-distinct dummies must retain the
    same external spelling as ordinary symbols.

    Args:
        symbol (sp.Basic): Symbol or identity-distinct ``Dummy`` to name.

    Returns:
        str: The symbol's declared name without SymPy's dummy-printing prefix.
    """
    name = getattr(symbol, "name", None)
    return name if isinstance(name, str) else str(symbol)


def _is_concrete_integer(value: sp.Expr) -> bool:
    """Return whether a concrete SymPy number is integer-valued.

    SymPy leaves ``Float(1.0).is_integer`` undecided even though accepting an
    integer-valued NumPy or Python float for a UInt input is useful and
    unambiguous.

    Args:
        value (sp.Expr): Concrete numeric expression.

    Returns:
        bool: Whether the value is mathematically integral.
    """
    if value.is_integer is True:
        return True
    if value.is_number and value.is_real is True:
        return sp.Eq(value, sp.floor(value)) is sp.true
    return False


def _combine_derivation(
    left: EstimateDerivation,
    right: EstimateDerivation,
) -> EstimateDerivation:
    """Return whether either estimate depends on a resource model.

    Args:
        left (EstimateDerivation): Left derivation classification.
        right (EstimateDerivation): Right derivation classification.

    Returns:
        EstimateDerivation: ``MODELED`` if either input is modeled.
    """
    if EstimateDerivation.MODELED in {left, right}:
        return EstimateDerivation.MODELED
    return EstimateDerivation.STRUCTURAL


def _combine_quality(
    left: EstimateQuality,
    right: EstimateQuality,
) -> EstimateQuality:
    """Return the weakest resource-count quality of two estimates.

    Args:
        left (EstimateQuality): Left count quality.
        right (EstimateQuality): Right count quality.

    Returns:
        EstimateQuality: Combined count quality.
    """
    rank = {
        EstimateQuality.EXACT: 0,
        EstimateQuality.CONSERVATIVE: 1,
        EstimateQuality.UNKNOWN: 2,
    }
    return left if rank[left] >= rank[right] else right


def _combine_approximation(
    left: ApproximationStatus,
    right: ApproximationStatus,
) -> ApproximationStatus:
    """Return whether either composed estimate uses an approximation.

    Args:
        left (ApproximationStatus): Left approximation status.
        right (ApproximationStatus): Right approximation status.

    Returns:
        ApproximationStatus: ``APPROXIMATE`` if either input is approximate.
    """
    if ApproximationStatus.APPROXIMATE in {left, right}:
        return ApproximationStatus.APPROXIMATE
    return ApproximationStatus.EXACT


def _validate_event_count(value: ResourceExpr, *, label: str) -> None:
    """Validate one concrete per-qubit event count.

    Args:
        value (ResourceExpr): Concrete or symbolic event-count expression.
        label (str): User-facing resource label for diagnostics.

    Raises:
        ValueError: If a concrete value is Boolean, negative, non-finite, or
            non-integral.
    """
    sympified = sp.sympify(value)
    if sympified is sp.true or sympified is sp.false:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}.")
    expression = cast(sp.Expr, sympified)
    if expression.is_number and (
        expression.is_finite is not True
        or expression.is_negative is True
        or not _is_concrete_integer(expression)
    ):
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}.")


class ControlDecomposition(enum.StrEnum):
    """Select how coherent controls are represented in resource estimates.

    Values:
        ABSTRACT: Keep each controlled primitive as one abstract logical
            operation, independently of its control arity.
        CLEAN_ANCILLA_TOFFOLI: Use the fixed clean-ancilla Toffoli-ladder
            resource model, including its body-wide sharing rule. This
            algorithmic model is independent of any engine's native or
            fallback emission policy.
    """

    ABSTRACT = "abstract"
    CLEAN_ANCILLA_TOFFOLI = "clean_ancilla_toffoli"


class EstimateDerivation(enum.StrEnum):
    """Describe how the estimator obtained the reported resource counts.

    Values:
        STRUCTURAL: Derive counts recursively from visible IR and the selected
            estimator decomposition rules.
        MODELED: Use a declared, callback-provided, or fallback resource model
            for at least one part of the estimate.
    """

    STRUCTURAL = "structural"
    MODELED = "modeled"


class EstimateQuality(enum.StrEnum):
    """Describe the directional quality of reported resource counts.

    This axis is independent of both count derivation and mathematical
    approximation. A modeled estimate can therefore be exact, conservative,
    or unknown with respect to the selected resource model.

    Values:
        EXACT: Reported counts exactly follow the selected circuit model.
        CONSERVATIVE: Reported counts may overestimate but do not
            underestimate the selected circuit model.
        UNKNOWN: No exact or conservative relation is available.
    """

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    UNKNOWN = "unknown"


class ApproximationStatus(enum.StrEnum):
    """Describe whether the selected circuit approximates an ideal operation.

    Values:
        EXACT: No mathematical approximation is known to the estimator.
        APPROXIMATE: At least one selected circuit construction approximates
            its ideal mathematical operation.
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"
