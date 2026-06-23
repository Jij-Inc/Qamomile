"""Summarize Hamiltonians for resource-estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

_SympyLike = sp.Expr | int | float
_CoefficientLike = _SympyLike | complex


@dataclass(frozen=True)
class PauliHamiltonianResource:
    """Summarize a Pauli Hamiltonian for resource estimates.

    Attributes:
        n_qubits (sp.Expr): Number of qubits in the encoded Hamiltonian.
        n_pauli_terms (sp.Expr): Number of non-identity Pauli terms.
        lambda_norm (sp.Expr): Sum of absolute non-identity Pauli
            coefficients used as the LCU normalization proxy.
        max_locality (sp.Expr): Maximum number of non-identity Pauli factors
            in any Hamiltonian term.
        constant (sp.Expr): Constant shift stored on the Hamiltonian.
        constant_included (bool): Whether ``constant`` was included in
            ``lambda_norm``.
        source (str): Human-readable source label.

    Raises:
        ValueError: If a positive-valued quantity is non-positive or if a
            nonnegative-valued quantity is negative.

    Example:
        >>> import qamomile.observable as qm_o
        >>> hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1)
        >>> summary = summarize_pauli_hamiltonian(hamiltonian)
        >>> summary.n_pauli_terms
        2
    """

    n_qubits: sp.Expr
    n_pauli_terms: sp.Expr
    lambda_norm: sp.Expr
    max_locality: sp.Expr
    constant: sp.Expr = sp.Integer(0)
    constant_included: bool = False
    source: str = "pauli_lcu"

    def __post_init__(self) -> None:
        """Validate summary fields after dataclass construction.

        Raises:
            ValueError: If a positive-valued quantity is non-positive or if a
                nonnegative-valued quantity is negative.
        """
        n_qubits = _as_expr(self.n_qubits, "n_qubits")
        n_pauli_terms = _as_expr(self.n_pauli_terms, "n_pauli_terms")
        lambda_norm = _as_expr(self.lambda_norm, "lambda_norm")
        max_locality = _as_expr(self.max_locality, "max_locality")
        constant = _coefficient_as_expr(self.constant, "constant")
        _validate_positive(n_qubits, "n_qubits")
        _validate_nonnegative(n_pauli_terms, "n_pauli_terms")
        _validate_nonnegative(lambda_norm, "lambda_norm")
        _validate_nonnegative(max_locality, "max_locality")
        _validate_nonnegative(sp.Abs(constant), "constant")
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "n_pauli_terms", n_pauli_terms)
        object.__setattr__(self, "lambda_norm", lambda_norm)
        object.__setattr__(self, "max_locality", max_locality)
        object.__setattr__(self, "constant", constant)

    def with_lambda_scale(
        self,
        scale: _SympyLike,
        *,
        source: str | None = None,
    ) -> PauliHamiltonianResource:
        """Return a copy with the Hamiltonian normalization rescaled.

        Args:
            scale (sp.Expr | int | float): Multiplicative scale applied to
                ``lambda_norm``. Values below one model transformations that
                reduce the effective Hamiltonian weight.
            source (str | None): Optional replacement source label. Defaults
                to preserving ``self.source``.

        Returns:
            PauliHamiltonianResource: New summary with rescaled
                ``lambda_norm``.

        Raises:
            ValueError: If ``scale`` is negative.
        """
        scale_expr = _as_expr(scale, "scale")
        _validate_nonnegative(scale_expr, "scale")
        return PauliHamiltonianResource(
            n_qubits=self.n_qubits,
            n_pauli_terms=self.n_pauli_terms,
            lambda_norm=sp.simplify(self.lambda_norm * scale_expr),
            max_locality=self.max_locality,
            constant=self.constant,
            constant_included=self.constant_included,
            source=self.source if source is None else source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Hamiltonian summary to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: String-valued resource summary.
        """
        return {
            "n_qubits": str(self.n_qubits),
            "n_pauli_terms": str(self.n_pauli_terms),
            "lambda_norm": str(self.lambda_norm),
            "max_locality": str(self.max_locality),
            "constant": str(self.constant),
            "constant_included": self.constant_included,
            "source": self.source,
        }


def summarize_pauli_hamiltonian(
    hamiltonian: Any,
    *,
    n_qubits: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "pauli_lcu",
) -> PauliHamiltonianResource:
    """Summarize a Qamomile Pauli Hamiltonian for resource estimates.

    Args:
        hamiltonian (Any): ``qamomile.observable.Hamiltonian`` instance to
            summarize.
        n_qubits (sp.Expr | int | float | None): Override for the encoded
            qubit count. Defaults to ``hamiltonian.num_qubits``.
        include_constant (bool): Whether to include the Hamiltonian constant
            term in ``lambda_norm``. Defaults to False, modeling the constant
            as a classical shift.
        source (str): Human-readable source label for the summary.

    Returns:
        PauliHamiltonianResource: Hamiltonian summary containing term count,
            lambda norm, constant, and max locality.

    Raises:
        TypeError: If ``hamiltonian`` is not a Qamomile Hamiltonian.
        ValueError: If the qubit count or derived norm is invalid.
    """
    import qamomile.observable as qm_o

    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise TypeError(
            "hamiltonian must be a qamomile.observable.Hamiltonian instance."
        )

    n_expr = (
        _as_expr(hamiltonian.num_qubits, "n_qubits")
        if n_qubits is None
        else _as_expr(n_qubits, "n_qubits")
    )
    lambda_norm = sp.Integer(0)
    max_locality = 0
    for operators, coeff in hamiltonian:
        lambda_norm += _abs_as_expr(coeff)
        max_locality = max(max_locality, len(operators))
    constant = _coefficient_as_expr(hamiltonian.constant, "constant")
    if include_constant:
        lambda_norm += _abs_as_expr(hamiltonian.constant)

    return PauliHamiltonianResource(
        n_qubits=n_expr,
        n_pauli_terms=sp.Integer(len(hamiltonian)),
        lambda_norm=sp.simplify(lambda_norm),
        max_locality=sp.Integer(max_locality),
        constant=constant,
        constant_included=include_constant,
        source=source,
    )


def _as_expr(value: _SympyLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    Args:
        value (sp.Expr | int | float): Value to convert.
        name (str): Field name used in error messages.

    Returns:
        sp.Expr: Converted SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be sympified.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


def _coefficient_as_expr(value: _CoefficientLike, name: str) -> sp.Expr:
    """Convert a Hamiltonian coefficient to a SymPy expression.

    Args:
        value (sp.Expr | int | float | complex): Coefficient to convert.
        name (str): Field name used in error messages.

    Returns:
        sp.Expr: Converted SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be sympified.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


def _abs_as_expr(value: _CoefficientLike) -> sp.Expr:
    """Return the symbolic absolute value of a numeric coefficient.

    Args:
        value (sp.Expr | int | float | complex): Coefficient to convert.

    Returns:
        sp.Expr: Absolute value represented as a SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be converted into a SymPy expression.
    """
    return sp.Abs(_coefficient_as_expr(value, "coefficient"))


def _validate_positive(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is positive when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is not positive.
    """
    if expr.is_positive is False:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is nonnegative when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is negative.
    """
    if expr.is_nonnegative is False:
        raise ValueError(f"{name} must be nonnegative.")
