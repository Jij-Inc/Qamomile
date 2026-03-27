import numpy as np
from .expr import BinaryExpr


def normalize_by_factor(
    model: BinaryExpr, factor: float, replace: bool = False
) -> BinaryExpr:
    """Normalize the BinaryExpr by a given factor.

    Args:
        model (BinaryExpr): The binary expression to be normalized.
        factor (float): The normalization factor.

    Returns:
        BinaryExpr: The normalized binary expression.
    """
    if replace:
        normalized_expr = model
    else:
        normalized_expr = model.copy()

    if factor == 0:
        return normalized_expr

    normalized_expr.constant /= factor
    for inds in normalized_expr.coefficients:
        normalized_expr.coefficients[inds] /= factor
    return normalized_expr


def normalize_by_abs_max(model: BinaryExpr, replace: bool = False) -> BinaryExpr:
    """Normalize the BinaryExpr by its absolute maximum coefficient.

    Args:
        model (BinaryExpr): The binary expression to be normalized.

    Returns:
        BinaryExpr: The normalized binary expression.
    """
    max_coeff = max(abs(coeff) for coeff in model.coefficients.values())
    return normalize_by_factor(model, max_coeff, replace=replace)


def normalize_by_rms(expr: BinaryExpr, replace: bool = False) -> BinaryExpr:
    r"""Normalize coefficients by the root mean square.

    The coefficients for normalized is defined as:

    .. math::
        W = \sqrt{\frac{1}{\lvert E_k \rvert} \sum_{\{u_1, \dots, u_k\}} (w_{u_1,...,u_k}^{(k)})^2 + \cdots + \frac{1}{\lvert E_1 \rvert} \sum_u (w_u^{(1)})^2}

    where w are coefficients and their subscriptions imply a term to be applied.
    E_i are the number of i-th order terms.
    We normalize the Ising Hamiltonian as

    .. math::
        \tilde{H} = \frac{1}{W} \left( C + \sum_i w_i Z_i + \cdots + \sum_{i_0, \dots, i_k} w_{i_0, \dots, i_k} Z_{i_0}\dots Z_{i_k} \right)
    This method is proposed in :cite:`Sureshbabu2024parametersettingin`

    .. bibliography::
        :filter: docname in docnames

    """

    # Get square sum and count for each kind of term.
    counts = {}  # key: term order, value: (sum of squares, count)
    for indices, coeff in expr.coefficients.items():
        order = len(indices)
        if order not in counts:
            counts[order] = [0.0, 0]

        # Add the square of the coefficient to the sum of squares.
        counts[order][0] += coeff**2
        # Increment the count of terms.
        counts[order][1] += 1

    # Compute the mean square for each kind of term.
    rms_components = 0.0
    for order, (sum_squares, count) in counts.items():
        if count > 0:  # This check is redundant but safe.
            mean_square = sum_squares / count
            rms_components += mean_square

    rms = np.sqrt(rms_components)
    return normalize_by_factor(expr, rms, replace)
