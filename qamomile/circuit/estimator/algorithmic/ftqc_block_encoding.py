"""Symbolic FTQC resource models for block-encoding based algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from qamomile.circuit.estimator.algorithmic.ftqc_chemistry import (
    ChemistryQPEModel,
    FTQCCostModel,
    FTQCReference,
    FTQCResourceEstimate,
    SurfaceCodeCostModel,
    references_for_chemistry_qpe_method,
)
from qamomile.circuit.estimator.algorithmic.ftqc_resources import (
    FTQCResourceFormula,
    FTQCResourcePlan,
    FTQCResourcePlanStep,
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


def _positive_symbol(name: str) -> sp.Symbol:
    """Create a positive SymPy symbol for formula metadata.

    Args:
        name (str): Symbol display name.

    Returns:
        sp.Symbol: Positive symbol with the requested name.
    """
    return sp.Symbol(name, positive=True)


def _nonnegative_symbol(name: str) -> sp.Symbol:
    """Create a nonnegative SymPy symbol for formula metadata.

    Args:
        name (str): Symbol display name.

    Returns:
        sp.Symbol: Nonnegative symbol with the requested name.
    """
    return sp.Symbol(name, nonnegative=True)


def _block_encoding_qpe_formulas() -> tuple[FTQCResourceFormula, ...]:
    """Return derivation formulas for block-encoding QPE estimates.

    Returns:
        tuple[FTQCResourceFormula, ...]: QPE iteration, walk-cost, Toffoli,
            logical-depth, and architecture-lift formulas.
    """
    normalization = _positive_symbol(FTQCResourceQuantity.LAMBDA_NORM.value)
    target_precision = _positive_symbol(FTQCResourceQuantity.TARGET_PRECISION.value)
    prepare_cost = _nonnegative_symbol(FTQCResourceQuantity.PREPARE_COST_TOFFOLI.value)
    select_cost = _nonnegative_symbol(FTQCResourceQuantity.SELECT_COST_TOFFOLI.value)
    reflection_cost = _nonnegative_symbol(
        FTQCResourceQuantity.REFLECTION_COST_TOFFOLI.value
    )
    qpe_iterations = _nonnegative_symbol(FTQCResourceQuantity.QPE_ITERATIONS.value)
    walk_cost = _nonnegative_symbol(FTQCResourceQuantity.WALK_COST_TOFFOLI.value)
    toffoli_gates = _nonnegative_symbol(FTQCResourceQuantity.TOFFOLI_GATES.value)
    logical_qubits = _nonnegative_symbol(FTQCResourceQuantity.LOGICAL_QUBITS.value)
    logical_depth = _nonnegative_symbol(FTQCResourceQuantity.LOGICAL_DEPTH.value)
    physical_qubits = _nonnegative_symbol(FTQCResourceQuantity.PHYSICAL_QUBITS.value)
    runtime_seconds = _nonnegative_symbol(FTQCResourceQuantity.RUNTIME_SECONDS.value)
    physical_qubits_per_logical = _positive_symbol(
        FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL.value
    )
    factory_qubits = _nonnegative_symbol(FTQCResourceQuantity.FACTORY_QUBITS.value)
    logical_cycle_time = _positive_symbol(
        FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS.value
    )
    throughput = _positive_symbol(
        FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND.value
    )
    return (
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.WALK_COST_TOFFOLI,
            expression=2 * prepare_cost + select_cost + reflection_cost,
            depends_on=(
                FTQCResourceQuantity.PREPARE_COST_TOFFOLI,
                FTQCResourceQuantity.SELECT_COST_TOFFOLI,
                FTQCResourceQuantity.REFLECTION_COST_TOFFOLI,
            ),
            description=(
                "Compose one qubitized walk from PREPARE, inverse PREPARE, "
                "SELECT, and reflection costs."
            ),
            reference_keys=(_QUBITIZATION_REFERENCE.key,),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.QPE_ITERATIONS,
            expression=normalization / target_precision,
            depends_on=(
                FTQCResourceQuantity.LAMBDA_NORM,
                FTQCResourceQuantity.TARGET_PRECISION,
            ),
            description="Use block-encoding normalization divided by QPE precision.",
            reference_keys=(_QUBITIZATION_REFERENCE.key,),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.TOFFOLI_GATES,
            expression=qpe_iterations * walk_cost,
            depends_on=(
                FTQCResourceQuantity.QPE_ITERATIONS,
                FTQCResourceQuantity.WALK_COST_TOFFOLI,
            ),
            description="Multiply walk calls by the Toffoli cost of one walk.",
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.LOGICAL_DEPTH,
            expression=toffoli_gates,
            depends_on=(FTQCResourceQuantity.TOFFOLI_GATES,),
            description="Use Toffoli count as the logical-depth proxy.",
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.PHYSICAL_QUBITS,
            expression=logical_qubits * physical_qubits_per_logical + factory_qubits,
            depends_on=(
                FTQCResourceQuantity.LOGICAL_QUBITS,
                FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL,
                FTQCResourceQuantity.FACTORY_QUBITS,
            ),
            description=(
                "Lift logical qubits with architecture overhead and add factory qubits."
            ),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.RUNTIME_SECONDS,
            expression=sp.Max(
                logical_depth * logical_cycle_time,
                toffoli_gates / throughput,
            ),
            depends_on=(
                FTQCResourceQuantity.LOGICAL_DEPTH,
                FTQCResourceQuantity.TOFFOLI_GATES,
                FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS,
                FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND,
            ),
            description=(
                "Use the slower of logical-cycle execution and factory "
                "throughput as the runtime proxy."
            ),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
            expression=logical_qubits * logical_depth,
            depends_on=(
                FTQCResourceQuantity.LOGICAL_QUBITS,
                FTQCResourceQuantity.LOGICAL_DEPTH,
            ),
            description="Multiply logical qubits by logical-depth proxy.",
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
            expression=physical_qubits * runtime_seconds,
            depends_on=(
                FTQCResourceQuantity.PHYSICAL_QUBITS,
                FTQCResourceQuantity.RUNTIME_SECONDS,
            ),
            description=(
                "Multiply physical qubits by runtime as a hardware "
                "space-time cost proxy."
            ),
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
            FTQCResourceQuantity.SYSTEM_QUBITS: self._system_qubits,
            FTQCResourceQuantity.LAMBDA_NORM: self._normalization,
            FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS: self._ancilla_qubits,
            FTQCResourceQuantity.PREPARE_COST_TOFFOLI: self._prepare_cost_toffoli,
            FTQCResourceQuantity.SELECT_COST_TOFFOLI: self._select_cost_toffoli,
            FTQCResourceQuantity.REFLECTION_COST_TOFFOLI: self._reflection_cost_toffoli,
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


def plan_qubitized_qpe_from_block_encoding(
    block_encoding: BlockEncodingResource,
    precision: _SympyLike,
    *,
    qpe_register_qubits: _SympyLike = 0,
    title: str | None = None,
) -> FTQCResourcePlan:
    """Build an abstract resource plan for block-encoding QPE.

    The plan exposes the reusable block-encoding contract separately from the
    repeated qubitized-walk step. This gives design reviews a subroutine-level
    view of PREPARE, SELECT, reflection, walk cost, QPE iterations, and logical
    footprint before any loader is lowered into concrete Qamomile IR.

    Args:
        block_encoding (BlockEncodingResource): Block-encoding subroutine
            metadata, including normalization and per-walk costs.
        precision (sp.Expr | int | float): Target energy precision in the
            same units as ``block_encoding.normalization``.
        qpe_register_qubits (sp.Expr | int | float): Optional readout qubits
            used by an explicit QPE circuit. Defaults to zero.
        title (str | None): Optional reader-facing plan title. Defaults to a
            label derived from ``block_encoding.name``.

    Returns:
        FTQCResourcePlan: Abstract resource plan whose aggregate logical
            quantities match the corresponding block-encoding QPE estimate
            before architecture lifting.

    Raises:
        TypeError: If ``precision`` or ``qpe_register_qubits`` cannot be
            converted to SymPy expressions.
        ValueError: If ``precision`` is non-positive or
            ``qpe_register_qubits`` is negative.

    Example:
        >>> block = BlockEncodingResource(
        ...     system_qubits=4,
        ...     normalization=100,
        ...     select_cost_toffoli=20,
        ...     prepare_cost_toffoli=5,
        ... )
        >>> plan = plan_qubitized_qpe_from_block_encoding(block, precision=2)
        >>> plan.resource_values()[FTQCResourceQuantity.QPE_ITERATIONS]
        50
    """
    precision_expr = _as_expr(precision, "precision")
    qpe_qubits = _as_expr(qpe_register_qubits, "qpe_register_qubits")
    _validate_positive(precision_expr, "precision")
    _validate_nonnegative(qpe_qubits, "qpe_register_qubits")

    qpe_iterations = sp.simplify(block_encoding._normalization / precision_expr)
    logical_qubits = sp.simplify(block_encoding.logical_qubits + qpe_qubits)
    walk_spacetime = sp.simplify(logical_qubits * block_encoding.walk_cost_toffoli)

    return FTQCResourcePlan(
        (
            FTQCResourcePlanStep(
                "block_encoding_contract",
                {
                    FTQCResourceQuantity.SYSTEM_QUBITS: block_encoding._system_qubits,
                    FTQCResourceQuantity.LAMBDA_NORM: block_encoding._normalization,
                    FTQCResourceQuantity.TARGET_PRECISION: precision_expr,
                    FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS: (
                        block_encoding._ancilla_qubits
                    ),
                    FTQCResourceQuantity.QPE_REGISTER_QUBITS: qpe_qubits,
                    FTQCResourceQuantity.PREPARE_COST_TOFFOLI: (
                        block_encoding._prepare_cost_toffoli
                    ),
                    FTQCResourceQuantity.SELECT_COST_TOFFOLI: (
                        block_encoding._select_cost_toffoli
                    ),
                    FTQCResourceQuantity.REFLECTION_COST_TOFFOLI: (
                        block_encoding._reflection_cost_toffoli
                    ),
                    FTQCResourceQuantity.WALK_COST_TOFFOLI: (
                        block_encoding.walk_cost_toffoli
                    ),
                    FTQCResourceQuantity.LOGICAL_QUBITS: logical_qubits,
                },
                label="Block-encoding contract",
            ),
            FTQCResourcePlanStep(
                "qubitized_walk_qpe",
                {
                    FTQCResourceQuantity.QPE_ITERATIONS: 1,
                    FTQCResourceQuantity.TOFFOLI_GATES: (
                        block_encoding.walk_cost_toffoli
                    ),
                    FTQCResourceQuantity.LOGICAL_DEPTH: (
                        block_encoding.walk_cost_toffoli
                    ),
                    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME: walk_spacetime,
                    FTQCResourceQuantity.LOGICAL_QUBITS: logical_qubits,
                },
                repetitions=qpe_iterations,
                label="Repeated qubitized walk",
            ),
        ),
        title=title or f"Qubitized QPE plan: {block_encoding.name}",
    )


def estimate_qubitized_qpe_from_block_encoding(
    block_encoding: BlockEncodingResource,
    precision: _SympyLike,
    *,
    qpe_register_qubits: _SympyLike = 0,
    cost_model: FTQCCostModel | SurfaceCodeCostModel | None = None,
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
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model used to lift logical estimates to physical
            qubits and runtime. Defaults to a symbolic architecture model.
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
    algorithm_values: dict[FTQCResourceQuantity, sp.Expr] = {
        quantity: value
        for quantity, value in block_encoding.resource_values().items()
        if quantity != FTQCResourceQuantity.LOGICAL_QUBITS
    }
    algorithm_values[FTQCResourceQuantity.QPE_REGISTER_QUBITS] = qpe_qubits
    model, architecture_values = _normalize_cost_model(cost_model)
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
        target_precision=sp.simplify(precision_expr),
        logical_depth=sp.simplify(logical_depth),
        runtime_seconds=sp.simplify(runtime_seconds),
        parameters=_collect_parameters(
            logical_qubits,
            physical_qubits,
            toffoli_gates,
            precision_expr,
            qpe_iterations,
            logical_depth,
            runtime_seconds,
            *algorithm_values.values(),
            *architecture_values.values(),
        ),
        assumptions=assumptions,
        references=references_for_estimate,
        formulas=_block_encoding_qpe_formulas(),
        algorithm_values=algorithm_values,
        architecture_values=architecture_values,
    )


def block_encoding_from_chemistry_model(
    model: ChemistryQPEModel,
    *,
    prepare_cost_toffoli: _SympyLike = 0,
    select_cost_toffoli: _SympyLike | None = None,
    reflection_cost_toffoli: _SympyLike = 0,
    ancilla_qubits: _SympyLike | None = None,
    name: str | None = None,
    references: tuple[FTQCReference, ...] = (),
) -> BlockEncodingResource:
    """Build a block-encoding resource from a chemistry QPE model.

    Args:
        model (ChemistryQPEModel): Chemistry QPE model to translate into a
            block-encoding contract.
        prepare_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            PREPARE call. Defaults to zero, preserving the model's existing
            aggregate walk cost as SELECT cost.
        select_cost_toffoli (sp.Expr | int | float | None): Toffoli cost for
            SELECT. Defaults to ``model.walk_cost_toffoli``.
        reflection_cost_toffoli (sp.Expr | int | float): Toffoli cost for the
            qubitization reflection. Defaults to zero.
        ancilla_qubits (sp.Expr | int | float | None): Ancilla qubits used by
            the block encoding. Defaults to ``model.logical_qubit_count -
            model.hamiltonian.n_spin_orbitals``.
        name (str | None): Optional block-encoding label. Defaults to
            ``model.description`` when set, otherwise the method value.
        references (tuple[FTQCReference, ...]): Additional references to carry
            on the block-encoding model.

    Returns:
        BlockEncodingResource: Block-encoding contract derived from the
            chemistry model.

    Raises:
        TypeError: If ``model`` is not a ``ChemistryQPEModel``.
        ValueError: If any derived or supplied resource quantity is invalid.
    """
    if not isinstance(model, ChemistryQPEModel):
        raise TypeError("model must be a ChemistryQPEModel instance.")

    select_cost = (
        _as_expr(model.walk_cost_toffoli, "walk_cost_toffoli")
        if select_cost_toffoli is None
        else _as_expr(select_cost_toffoli, "select_cost_toffoli")
    )
    ancilla = (
        sp.simplify(model.logical_qubit_count - model.hamiltonian.n_spin_orbitals)
        if ancilla_qubits is None
        else _as_expr(ancilla_qubits, "ancilla_qubits")
    )
    label = name or model.description or model.normalized_method.value
    return BlockEncodingResource(
        system_qubits=model.hamiltonian.n_spin_orbitals,
        normalization=model.hamiltonian.lambda_norm,
        prepare_cost_toffoli=prepare_cost_toffoli,
        select_cost_toffoli=select_cost,
        reflection_cost_toffoli=reflection_cost_toffoli,
        ancilla_qubits=ancilla,
        name=label,
        references=_combine_references(
            references_for_chemistry_qpe_method(model.normalized_method),
            model.references,
            references,
        ),
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


def _normalize_cost_model(
    cost_model: FTQCCostModel | SurfaceCodeCostModel | None,
) -> tuple[FTQCCostModel, dict[FTQCResourceQuantity, sp.Expr]]:
    """Normalize an architecture model for block-encoding estimates.

    Args:
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model supplied by the caller. ``None`` creates a
            symbolic ``FTQCCostModel``.

    Returns:
        tuple[FTQCCostModel, dict[FTQCResourceQuantity, sp.Expr]]: The
            cost-model interface used for lifting and the canonical
            architecture quantities retained on the estimate.

    Raises:
        TypeError: If ``cost_model`` is not a supported architecture model.
    """
    if cost_model is None:
        model = FTQCCostModel()
        return model, model.resource_values()
    if isinstance(cost_model, SurfaceCodeCostModel):
        return cost_model.to_cost_model(), cost_model.resource_values()
    if isinstance(cost_model, FTQCCostModel):
        return cost_model, cost_model.resource_values()
    raise TypeError("cost_model must be an FTQCCostModel or SurfaceCodeCostModel.")


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
