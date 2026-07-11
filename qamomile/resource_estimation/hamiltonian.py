"""Summarize Hamiltonians for resource-estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from qamomile.resource_estimation._common import (
    _as_expr,
    _CoefficientLike,
    _SympyLike,
    _validate_nonnegative,
    _validate_positive,
)

_OPENFERMION_PAULI_LABELS = frozenset({"X", "Y", "Z"})


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
        constant = _as_expr(self.constant, "constant")
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

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the summary.

        Returns:
            dict[str, sp.Expr]: Problem-level Hamiltonian values keyed by
            canonical resource quantity names.
        """
        return {
            "n_qubits": self.n_qubits,
            "n_pauli_terms": self.n_pauli_terms,
            "lambda_norm": self.lambda_norm,
            "max_locality": self.max_locality,
        }


def summarize_pauli_hamiltonian(
    hamiltonian: Any,
    *,
    n_qubits: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "pauli_lcu",
) -> PauliHamiltonianResource:
    """Summarize a Qamomile Pauli Hamiltonian for resource estimates.

    Terms whose coefficients cancelled to exactly zero are excluded from
    every summary quantity, so ``n_pauli_terms``, ``lambda_norm``, and
    ``max_locality`` all describe the same effective operator. Numeric
    coefficients are accumulated in plain Python arithmetic before a single
    SymPy conversion, keeping the summary linear-time with a small constant
    even for chemistry-scale Hamiltonians.

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
    numeric_norm: int | float = 0
    symbolic_norm_terms: list[sp.Expr] = []
    n_pauli_terms = 0
    max_locality = 0
    for operators, coeff in hamiltonian:
        if coeff == 0:
            continue
        n_pauli_terms += 1
        max_locality = max(max_locality, len(operators))
        if isinstance(coeff, (int, float, complex)):
            numeric_norm += abs(coeff)
        else:
            symbolic_norm_terms.append(_abs_as_expr(coeff))
    constant = _as_expr(hamiltonian.constant, "constant")
    if include_constant:
        numeric_norm += abs(hamiltonian.constant)
    lambda_norm = sp.sympify(numeric_norm)
    if symbolic_norm_terms:
        lambda_norm = sp.simplify(sp.Add(lambda_norm, *symbolic_norm_terms))

    return PauliHamiltonianResource(
        n_qubits=n_expr,
        n_pauli_terms=sp.Integer(n_pauli_terms),
        lambda_norm=lambda_norm,
        max_locality=sp.Integer(max_locality),
        constant=constant,
        constant_included=include_constant,
        source=source,
    )


def hamiltonian_from_openfermion_qubit_operator(
    openfermion_operator: Any,
    *,
    num_qubits: int | None = None,
) -> Any:
    """Convert an OpenFermion qubit operator into a Qamomile Hamiltonian.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like object
            exposing a ``terms`` mapping from Pauli-string tuples to numeric
            coefficients.
        num_qubits (int | None): Optional encoded register size to store on
            the returned Hamiltonian. Defaults to inferring the size from the
            largest Pauli index.

    Returns:
        qamomile.observable.Hamiltonian: Equivalent Qamomile Hamiltonian with
            identity terms stored as ``constant``.

    Raises:
        TypeError: If ``openfermion_operator`` does not expose a terms mapping
            or a term is malformed.
        ValueError: If a Pauli label is not one of ``X``, ``Y``, or ``Z``.
    """
    import qamomile.observable as qm_o

    terms = getattr(openfermion_operator, "terms", None)
    if not isinstance(terms, dict):
        raise TypeError(
            "openfermion_operator must expose an OpenFermion-style terms mapping."
        )

    paulis = {
        "X": qm_o.Pauli.X,
        "Y": qm_o.Pauli.Y,
        "Z": qm_o.Pauli.Z,
    }
    hamiltonian = qm_o.Hamiltonian(num_qubits=num_qubits)
    for term, coefficient in terms.items():
        if term == ():
            hamiltonian.constant += coefficient
            continue
        if not isinstance(term, tuple):
            raise TypeError("OpenFermion Pauli terms must be tuples.")

        operators = []
        for factor in term:
            if (
                not isinstance(factor, tuple)
                or len(factor) != 2
                or not isinstance(factor[0], int)
                or not isinstance(factor[1], str)
            ):
                raise TypeError("OpenFermion Pauli factors must be (int, str) tuples.")
            qubit, label = factor
            if label not in _OPENFERMION_PAULI_LABELS:
                valid = ", ".join(sorted(_OPENFERMION_PAULI_LABELS))
                raise ValueError(
                    f"Unsupported OpenFermion Pauli label {label!r}; "
                    f"valid labels: {valid}."
                )
            operators.append(qm_o.PauliOperator(paulis[label], qubit))
        hamiltonian.add_term(tuple(operators), coefficient)
    return hamiltonian


def summarize_openfermion_qubit_operator(
    openfermion_operator: Any,
    *,
    n_qubits: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "openfermion_qubit_operator",
) -> PauliHamiltonianResource:
    """Summarize an OpenFermion qubit operator for resource estimates.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like object
            exposing a ``terms`` mapping.
        n_qubits (sp.Expr | int | float | None): Override for the encoded
            qubit count. Defaults to the inferred Qamomile Hamiltonian width.
        include_constant (bool): Whether to include the constant term in
            ``lambda_norm``. Defaults to False.
        source (str): Human-readable source label for the summary.

    Returns:
        PauliHamiltonianResource: Hamiltonian summary containing term count,
            lambda norm, constant, and max locality.

    Raises:
        TypeError: If the OpenFermion object is malformed.
        ValueError: If a Pauli label or derived resource quantity is invalid.
    """
    hamiltonian = hamiltonian_from_openfermion_qubit_operator(
        openfermion_operator,
    )
    return summarize_pauli_hamiltonian(
        hamiltonian,
        n_qubits=n_qubits,
        include_constant=include_constant,
        source=source,
    )


def _abs_as_expr(value: _CoefficientLike) -> sp.Expr:
    """Return the symbolic absolute value of a numeric coefficient.

    Args:
        value (sp.Expr | int | float | complex): Coefficient to convert.

    Returns:
        sp.Expr: Absolute value represented as a SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be converted into a SymPy expression.
    """
    return sp.Abs(_as_expr(value, "coefficient"))
