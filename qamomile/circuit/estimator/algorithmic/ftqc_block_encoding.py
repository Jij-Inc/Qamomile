"""Symbolic FTQC resource models for block-encoding based algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from qamomile.circuit.estimator.algorithmic.ftqc_chemistry import (
    FTQCCostModel,
    FTQCReference,
    FTQCResourceEstimate,
)
from qamomile.circuit.estimator.algorithmic.ftqc_resources import (
    FTQCResourceQuantity,
)

_SympyLike = sp.Expr | int | float


_QUBITIZATION_REFERENCE = FTQCReference(
    key="arXiv:1610.06546",
    title="Hamiltonian Simulation by Qubitization",
    url="https://arxiv.org/abs/1610.06546",
    note=(
        "Introduces qubitized quantum walks built from a block-encoding, "
        "including PREPARE, SELECT, and reflection-style primitives."
    ),
)


@dataclass(frozen=True)
class BlockEncodingResource:
    """Describe a block-encoding implementation for FTQC estimates.

    The model separates the logical problem size, block-encoding
    normalization, and the Toffoli cost of the reusable qubitization
    subroutines. It does not lower those subroutines into IR operations;
    backend-specific circuit realization remains a later concern.

    Attributes:
        system_qubits (sp.Expr | int | float): Logical system-register qubits.
        normalization (sp.Expr | int | float): Block-encoding normalization
            alpha such that ``H / alpha`` is encoded.
        select_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            SELECT or oracle application.
        prepare_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            PREPARE or PREPARE inverse call. Defaults to zero.
        reflection_cost_toffoli (sp.Expr | int | float): Toffoli cost for
            the reflection used by one qubitized walk. Defaults to zero.
        ancilla_qubits (sp.Expr | int | float): Ancilla/workspace qubits
            required by the block-encoding, excluding optional QPE readout
            qubits. Defaults to zero.
        name (str): Reader-facing label for this block-encoding model.
        references (tuple[FTQCReference, ...]): Research sources or internal
            notes that justify the supplied subroutine costs.

    Raises:
        ValueError: If a positive-valued quantity is non-positive or if a
            nonnegative-valued quantity is negative.

    Example:
        >>> block = BlockEncodingResource(
        ...     system_qubits=10,
        ...     normalization=100,
        ...     select_cost_toffoli=50,
        ...     prepare_cost_toffoli=20,
        ... )
        >>> block.walk_cost_toffoli
        90
    """

    system_qubits: _SympyLike
    normalization: _SympyLike
    select_cost_toffoli: _SympyLike
    prepare_cost_toffoli: _SympyLike = 0
    reflection_cost_toffoli: _SympyLike = 0
    ancilla_qubits: _SympyLike = 0
    name: str = "block_encoding"
    references: tuple[FTQCReference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate block-encoding fields after dataclass construction.

        Raises:
            ValueError: If a positive-valued quantity is non-positive or if a
                nonnegative-valued quantity is negative.
        """
        _validate_positive(self._system_qubits, "system_qubits")
        _validate_positive(self._normalization, "normalization")
        for name, expr in [
            ("select_cost_toffoli", self._select_cost_toffoli),
            ("prepare_cost_toffoli", self._prepare_cost_toffoli),
            ("reflection_cost_toffoli", self._reflection_cost_toffoli),
            ("ancilla_qubits", self._ancilla_qubits),
        ]:
            _validate_nonnegative(expr, name)

    @property
    def logical_qubits(self) -> sp.Expr:
        """Return block-encoding logical qubits before QPE readout.

        Returns:
            sp.Expr: ``system_qubits + ancilla_qubits``.
        """
        return sp.simplify(self._system_qubits + self._ancilla_qubits)

    @property
    def walk_cost_toffoli(self) -> sp.Expr:
        """Return the Toffoli cost of one qubitized walk operator.

        Returns:
            sp.Expr: ``2 * prepare_cost_toffoli + select_cost_toffoli +
            reflection_cost_toffoli``.
        """
        return sp.simplify(
            2 * self._prepare_cost_toffoli
            + self._select_cost_toffoli
            + self._reflection_cost_toffoli
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block-encoding model.

        Returns:
            dict[str, Any]: JSON-friendly block-encoding metadata.
        """
        return {
            "name": self.name,
            "system_qubits": str(self._system_qubits),
            "normalization": str(self._normalization),
            "select_cost_toffoli": str(self._select_cost_toffoli),
            "prepare_cost_toffoli": str(self._prepare_cost_toffoli),
            "reflection_cost_toffoli": str(self._reflection_cost_toffoli),
            "ancilla_qubits": str(self._ancilla_qubits),
            "logical_qubits": str(self.logical_qubits),
            "walk_cost_toffoli": str(self.walk_cost_toffoli),
            "references": [reference.to_dict() for reference in self.references],
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return block-encoding values keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Problem and algorithm
            quantities that feed qubitized QPE estimates.
        """
        return {
            FTQCResourceQuantity.LOGICAL_QUBITS: self.logical_qubits,
            FTQCResourceQuantity.LAMBDA_NORM: self._normalization,
            FTQCResourceQuantity.WALK_COST_TOFFOLI: self.walk_cost_toffoli,
        }

    @property
    def _system_qubits(self) -> sp.Expr:
        """Return system qubits as a SymPy expression.

        Returns:
            sp.Expr: Converted system-qubit count.
        """
        return _as_expr(self.system_qubits, "system_qubits")

    @property
    def _normalization(self) -> sp.Expr:
        """Return normalization as a SymPy expression.

        Returns:
            sp.Expr: Converted block-encoding normalization.
        """
        return _as_expr(self.normalization, "normalization")

    @property
    def _select_cost_toffoli(self) -> sp.Expr:
        """Return SELECT cost as a SymPy expression.

        Returns:
            sp.Expr: Converted SELECT Toffoli cost.
        """
        return _as_expr(self.select_cost_toffoli, "select_cost_toffoli")

    @property
    def _prepare_cost_toffoli(self) -> sp.Expr:
        """Return PREPARE cost as a SymPy expression.

        Returns:
            sp.Expr: Converted PREPARE Toffoli cost.
        """
        return _as_expr(self.prepare_cost_toffoli, "prepare_cost_toffoli")

    @property
    def _reflection_cost_toffoli(self) -> sp.Expr:
        """Return reflection cost as a SymPy expression.

        Returns:
            sp.Expr: Converted reflection Toffoli cost.
        """
        return _as_expr(self.reflection_cost_toffoli, "reflection_cost_toffoli")

    @property
    def _ancilla_qubits(self) -> sp.Expr:
        """Return ancilla qubits as a SymPy expression.

        Returns:
            sp.Expr: Converted ancilla-qubit count.
        """
        return _as_expr(self.ancilla_qubits, "ancilla_qubits")


def estimate_qubitized_qpe_from_block_encoding(
    block_encoding: BlockEncodingResource,
    precision: _SympyLike,
    *,
    qpe_register_qubits: _SympyLike = 0,
    cost_model: FTQCCostModel | None = None,
    references: tuple[FTQCReference, ...] = (),
) -> FTQCResourceEstimate:
    """Estimate qubitized QPE resources from a block-encoding model.

    Args:
        block_encoding (BlockEncodingResource): Block-encoding subroutine
            metadata, including normalization and walk-operator cost.
        precision (sp.Expr | int | float): Target energy precision in the
            same units as ``block_encoding.normalization``.
        qpe_register_qubits (sp.Expr | int | float): Optional readout qubits
            used by an explicit QPE circuit. Defaults to zero so callers can
            separately decide whether phase-readout qubits are included in
            the block-encoding workspace.
        cost_model (FTQCCostModel | None): Architecture model used to lift
            logical estimates to physical qubits and runtime. Defaults to a
            symbolic architecture model.
        references (tuple[FTQCReference, ...]): Additional research sources
            to attach to the estimate.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If ``precision`` is non-positive or
            ``qpe_register_qubits`` is negative.
    """
    precision_expr = _as_expr(precision, "precision")
    qpe_qubits = _as_expr(qpe_register_qubits, "qpe_register_qubits")
    _validate_positive(precision_expr, "precision")
    _validate_nonnegative(qpe_qubits, "qpe_register_qubits")

    qpe_iterations = sp.simplify(block_encoding._normalization / precision_expr)
    toffoli_gates = sp.simplify(qpe_iterations * block_encoding.walk_cost_toffoli)
    logical_depth = toffoli_gates
    logical_qubits = sp.simplify(block_encoding.logical_qubits + qpe_qubits)
    model = cost_model or FTQCCostModel()
    runtime_seconds = model.runtime_seconds_for(logical_depth, toffoli_gates)
    physical_qubits = model.physical_qubits_for(logical_qubits)
    assumptions = {
        "block_encoding": block_encoding.name,
        "walk_cost_toffoli": (
            "Uses 2 * prepare_cost_toffoli + select_cost_toffoli + "
            "reflection_cost_toffoli for one qubitized walk."
        ),
        "qpe_iterations": (
            "Uses block_encoding.normalization / precision as the walk-call proxy."
        ),
        "qpe_register_qubits": str(qpe_qubits),
    }
    references_for_estimate = _combine_references(
        (_QUBITIZATION_REFERENCE,),
        block_encoding.references,
        references,
    )
    return FTQCResourceEstimate(
        algorithm=f"qubitized_qpe:block_encoding:{block_encoding.name}",
        logical_qubits=sp.simplify(logical_qubits),
        physical_qubits=sp.simplify(physical_qubits),
        toffoli_gates=sp.simplify(toffoli_gates),
        t_gates=sp.Integer(0),
        clifford_gates=sp.Integer(0),
        qpe_iterations=sp.simplify(qpe_iterations),
        logical_depth=sp.simplify(logical_depth),
        runtime_seconds=sp.simplify(runtime_seconds),
        parameters=_collect_parameters(
            logical_qubits,
            physical_qubits,
            toffoli_gates,
            qpe_iterations,
            logical_depth,
            runtime_seconds,
        ),
        assumptions=assumptions,
        references=references_for_estimate,
    )


def _collect_parameters(*expressions: sp.Expr) -> dict[str, sp.Symbol]:
    """Collect free symbols from symbolic resource expressions.

    Args:
        *expressions (sp.Expr): Expressions to inspect.

    Returns:
        dict[str, sp.Symbol]: Symbols keyed by their display names.
    """
    symbols: set[sp.Symbol] = set()
    for expression in expressions:
        for symbol in expression.free_symbols:
            if isinstance(symbol, sp.Symbol):
                symbols.add(symbol)
    return {str(symbol): symbol for symbol in sorted(symbols, key=str)}


def _combine_references(
    *groups: tuple[FTQCReference, ...],
) -> tuple[FTQCReference, ...]:
    """Merge reference groups while preserving first-seen order.

    Args:
        *groups (tuple[FTQCReference, ...]): Reference groups to merge.

    Returns:
        tuple[FTQCReference, ...]: Deduplicated references keyed by
            ``FTQCReference.key``.
    """
    references: list[FTQCReference] = []
    seen: set[str] = set()
    for group in groups:
        for reference in group:
            if reference.key in seen:
                continue
            references.append(reference)
            seen.add(reference.key)
    return tuple(references)


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
