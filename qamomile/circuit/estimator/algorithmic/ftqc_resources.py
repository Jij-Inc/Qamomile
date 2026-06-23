"""Define canonical FTQC resource quantities for symbolic estimators."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

import sympy as sp

_CoverageReportDict = dict[
    str,
    str
    | list[dict[str, str | bool]]
    | list[dict[str, str | bool | list[str]]]
    | dict[str, int],
]


class FTQCResourceCategory(enum.StrEnum):
    """Group FTQC resource quantities by modeling layer.

    Attributes:
        PROBLEM: Input problem and Hamiltonian representation quantities.
        ALGORITHM: Algorithm-control quantities such as QPE iteration counts.
        LOGICAL: Logical-circuit quantities before hardware lifting.
        PHYSICAL: Physical-device quantities after architecture assumptions.
        ARCHITECTURE: Hardware-model knobs that map logical work to physical
            resources.

    Example:
        >>> FTQCResourceCategory("logical")
        <FTQCResourceCategory.LOGICAL: 'logical'>
    """

    PROBLEM = "problem"
    ALGORITHM = "algorithm"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    ARCHITECTURE = "architecture"


class FTQCResourceQuantity(enum.StrEnum):
    """Name canonical FTQC resource quantities.

    Attributes:
        N_SPIN_ORBITALS: Encoded active-space size or qubit-register size.
        N_PAULI_TERMS: Number of non-identity Pauli terms in an LCU model.
        LAMBDA_NORM: Hamiltonian normalization driving QPE walk calls.
        MAX_LOCALITY: Maximum Pauli-string locality.
        TARGET_PRECISION: Target algorithmic energy or phase precision.
        TRUNCATION_ERROR: Representation-level approximation error budget.
        SYSTEM_QUBITS: Logical system-register qubits encoded by a
            block-encoding model.
        BLOCK_ENCODING_ANCILLA_QUBITS: Ancilla and workspace qubits required
            by a block encoding before QPE readout qubits are added.
        QPE_REGISTER_QUBITS: Phase-readout qubits used by an explicit QPE
            circuit.
        STATE_PREPARATION_SUCCESS_PROBABILITY: Success probability or squared
            overlap of the prepared state with the target eigenstate subspace.
        QPE_REPETITIONS: Expected number of QPE runs needed to obtain one
            successful sample.
        STATE_PREPARATION_TOFFOLI: Toffoli overhead of one state-preparation
            or symmetry-filtering attempt.
        STATE_PREPARATION_T_GATES: T-gate overhead of one state-preparation or
            symmetry-filtering attempt.
        STATE_PREPARATION_LOGICAL_DEPTH: Logical-depth overhead of one
            state-preparation or symmetry-filtering attempt.
        PREPARE_COST_TOFFOLI: Toffoli cost of one PREPARE or inverse PREPARE
            subroutine call.
        SELECT_COST_TOFFOLI: Toffoli cost of one SELECT or oracle subroutine
            call.
        REFLECTION_COST_TOFFOLI: Toffoli cost of the reflection subroutine used
            by one qubitized walk.
        WALK_COST_TOFFOLI: Toffoli cost of one qubitized walk.
        QPE_ITERATIONS: Number of phase-estimation walk or time-evolution
            calls.
        LOGICAL_QUBITS: Logical qubits required by an algorithm.
        LOGICAL_DEPTH: Logical-depth proxy.
        LOGICAL_SPACETIME_VOLUME: Logical qubit-layer volume proxy.
        TOFFOLI_GATES: Toffoli or Toffoli-equivalent non-Clifford count.
        T_GATES: T-gate or T-equivalent count.
        CLIFFORD_GATES: Clifford gate count.
        PHYSICAL_QUBITS: Physical qubits under an architecture model.
        RUNTIME_SECONDS: Runtime proxy in seconds.
        PHYSICAL_QUBIT_SECONDS: Physical qubit-seconds space-time cost.
        LOGICAL_ERROR_RATE: Logical error-rate proxy per operation or cycle.
        TARGET_LOGICAL_FAILURE_PROBABILITY: Total allowed logical failure
            probability for an architecture budget.
        LOGICAL_OPERATION_BUDGET: Number of logical operations or cycles
            sharing the failure budget.
        PHYSICAL_ERROR_RATE: Physical error rate used by an architecture
            sizing model.
        THRESHOLD_ERROR_RATE: Threshold error rate used by an architecture
            sizing model.
        PHYSICAL_QUBITS_PER_LOGICAL: Physical overhead per logical qubit.
        LOGICAL_CYCLE_TIME_SECONDS: Logical layer or cycle time.
        FACTORY_QUBITS: Physical qubits reserved for factories.
        TOFFOLI_THROUGHPUT_PER_SECOND: Sustainable non-Clifford throughput.
        CODE_DISTANCE: Surface-code distance.
        PHYSICAL_CYCLE_TIME_SECONDS: Physical error-correction cycle time.
        PHYSICAL_QUBITS_PER_LOGICAL_FACTOR: Constant factor multiplying
            distance squared for one logical patch.
        LOGICAL_CYCLE_FACTOR: Constant factor multiplying code distance for
            one logical cycle.
        FACTORY_COUNT: Number of non-Clifford factories.
        PHYSICAL_QUBITS_PER_FACTORY: Physical qubits used by one factory.
        FACTORY_CYCLES_PER_TOFFOLI: Logical cycles needed per factory output.

    Example:
        >>> FTQCResourceQuantity("lambda_norm")
        <FTQCResourceQuantity.LAMBDA_NORM: 'lambda_norm'>
    """

    N_SPIN_ORBITALS = "n_spin_orbitals"
    N_PAULI_TERMS = "n_pauli_terms"
    LAMBDA_NORM = "lambda_norm"
    MAX_LOCALITY = "max_locality"
    TARGET_PRECISION = "target_precision"
    TRUNCATION_ERROR = "truncation_error"
    SYSTEM_QUBITS = "system_qubits"
    BLOCK_ENCODING_ANCILLA_QUBITS = "block_encoding_ancilla_qubits"
    QPE_REGISTER_QUBITS = "qpe_register_qubits"
    STATE_PREPARATION_SUCCESS_PROBABILITY = "state_preparation_success_probability"
    QPE_REPETITIONS = "qpe_repetitions"
    STATE_PREPARATION_TOFFOLI = "state_preparation_toffoli"
    STATE_PREPARATION_T_GATES = "state_preparation_t_gates"
    STATE_PREPARATION_LOGICAL_DEPTH = "state_preparation_logical_depth"
    PREPARE_COST_TOFFOLI = "prepare_cost_toffoli"
    SELECT_COST_TOFFOLI = "select_cost_toffoli"
    REFLECTION_COST_TOFFOLI = "reflection_cost_toffoli"
    WALK_COST_TOFFOLI = "walk_cost_toffoli"
    QPE_ITERATIONS = "qpe_iterations"
    LOGICAL_QUBITS = "logical_qubits"
    LOGICAL_DEPTH = "logical_depth"
    LOGICAL_SPACETIME_VOLUME = "logical_spacetime_volume"
    TOFFOLI_GATES = "toffoli_gates"
    T_GATES = "t_gates"
    CLIFFORD_GATES = "clifford_gates"
    PHYSICAL_QUBITS = "physical_qubits"
    RUNTIME_SECONDS = "runtime_seconds"
    PHYSICAL_QUBIT_SECONDS = "physical_qubit_seconds"
    LOGICAL_ERROR_RATE = "logical_error_rate"
    TARGET_LOGICAL_FAILURE_PROBABILITY = "target_logical_failure_probability"
    LOGICAL_OPERATION_BUDGET = "logical_operation_budget"
    PHYSICAL_ERROR_RATE = "physical_error_rate"
    THRESHOLD_ERROR_RATE = "threshold_error_rate"
    PHYSICAL_QUBITS_PER_LOGICAL = "physical_qubits_per_logical"
    LOGICAL_CYCLE_TIME_SECONDS = "logical_cycle_time_seconds"
    FACTORY_QUBITS = "factory_qubits"
    TOFFOLI_THROUGHPUT_PER_SECOND = "toffoli_throughput_per_second"
    CODE_DISTANCE = "code_distance"
    PHYSICAL_CYCLE_TIME_SECONDS = "physical_cycle_time_seconds"
    PHYSICAL_QUBITS_PER_LOGICAL_FACTOR = "physical_qubits_per_logical_factor"
    LOGICAL_CYCLE_FACTOR = "logical_cycle_factor"
    FACTORY_COUNT = "factory_count"
    PHYSICAL_QUBITS_PER_FACTORY = "physical_qubits_per_factory"
    FACTORY_CYCLES_PER_TOFFOLI = "factory_cycles_per_toffoli"


class FTQCResourceChangeDirection(enum.StrEnum):
    """Classify how a candidate FTQC quantity changed versus baseline.

    Attributes:
        SMALLER: Candidate value is provably smaller than the baseline.
        LARGER: Candidate value is provably larger than the baseline.
        UNCHANGED: Candidate value is provably equal to the baseline.
        SYMBOLIC: The sign of the symbolic change is not decidable from the
            available assumptions.

    Example:
        >>> FTQCResourceChangeDirection("smaller")
        <FTQCResourceChangeDirection.SMALLER: 'smaller'>
    """

    SMALLER = "smaller"
    LARGER = "larger"
    UNCHANGED = "unchanged"
    SYMBOLIC = "symbolic"


class FTQCResourceConstraintSense(enum.StrEnum):
    """Name the comparison sense for an FTQC resource constraint.

    Attributes:
        AT_MOST: The resource value must be less than or equal to the limit.
        AT_LEAST: The resource value must be greater than or equal to the
            limit.

    Example:
        >>> FTQCResourceConstraintSense("at_most")
        <FTQCResourceConstraintSense.AT_MOST: 'at_most'>
    """

    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class FTQCResourceConstraintStatus(enum.StrEnum):
    """Classify whether an FTQC resource constraint is met.

    Attributes:
        SATISFIED: The constraint is provably satisfied.
        VIOLATED: The constraint is provably violated.
        SYMBOLIC: The constraint cannot be decided from the symbolic
            expression under the current assumptions.

    Example:
        >>> FTQCResourceConstraintStatus("satisfied")
        <FTQCResourceConstraintStatus.SATISFIED: 'satisfied'>
    """

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    SYMBOLIC = "symbolic"


class FTQCResourceAggregationRule(enum.StrEnum):
    """Name how a quantity composes across FTQC subroutine steps.

    Attributes:
        ADD: Add the quantity across repeated or sequential subroutines.
        PEAK: Keep the maximum value across repeated or sequential
            subroutines.
        CONSISTENT: Require all contributing steps to provide the same value.

    Example:
        >>> FTQCResourceAggregationRule("add")
        <FTQCResourceAggregationRule.ADD: 'add'>
    """

    ADD = "add"
    PEAK = "peak"
    CONSISTENT = "consistent"


class FTQCResourceProfile(enum.StrEnum):
    """Name standard FTQC resource review profiles.

    Attributes:
        CHEMISTRY_QPE: End-to-end chemistry QPE comparison quantities.
        BLOCK_ENCODING: Block-encoding subroutine quantities.
        SPACETIME: Logical and physical space-time quantities.
        ERROR_BUDGET: Surface-code error-budget and distance quantities.
        ARCHITECTURE: Architecture knobs and derived architecture quantities.

    Example:
        >>> FTQCResourceProfile("spacetime")
        <FTQCResourceProfile.SPACETIME: 'spacetime'>
    """

    CHEMISTRY_QPE = "chemistry_qpe"
    BLOCK_ENCODING = "block_encoding"
    SPACETIME = "spacetime"
    ERROR_BUDGET = "error_budget"
    ARCHITECTURE = "architecture"


class FTQCResourceReportKind(enum.StrEnum):
    """Name standardized FTQC resource report snapshot kinds.

    Attributes:
        RESEARCH_SIGNAL_COVERAGE: Research-signal quantity coverage report.
        BUDGET: Resource-budget constraint report.
        COMPARISON: Pairwise resource-comparison report.
        DRIVER: Formula-driver resource-comparison report.
        PARETO: Multi-candidate Pareto-frontier report.
        SCENARIO: Symbolic scenario-sensitivity report.

    Example:
        >>> FTQCResourceReportKind("scenario")
        <FTQCResourceReportKind.SCENARIO: 'scenario'>
    """

    RESEARCH_SIGNAL_COVERAGE = "research_signal_coverage"
    BUDGET = "budget"
    COMPARISON = "comparison"
    DRIVER = "driver"
    PARETO = "pareto"
    SCENARIO = "scenario"


class SupportsFTQCResourceValues(Protocol):
    """Represent objects that expose canonical FTQC resource values.

    Example:
        >>> hasattr(object(), "resource_values")
        False
    """

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return resource values keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Resource values.
        """
        ...


@dataclass(frozen=True)
class FTQCResourceQuantitySpec:
    """Describe one canonical FTQC resource quantity.

    Attributes:
        quantity (FTQCResourceQuantity): Machine-readable quantity key.
        label (str): Reader-facing label.
        unit (str): Unit or dimension of the quantity.
        category (FTQCResourceCategory): Modeling layer that owns the
            quantity.
        description (str): Short description of what the quantity measures.

    Example:
        >>> spec = describe_ftqc_resource_quantity("logical_qubits")
        >>> spec.unit
        'logical qubits'
    """

    quantity: FTQCResourceQuantity
    label: str
    unit: str
    category: FTQCResourceCategory
    description: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the quantity specification.

        Returns:
            dict[str, str]: JSON-friendly quantity metadata.
        """
        return {
            "quantity": self.quantity.value,
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class FTQCResourceProfileSpec:
    """Describe a standard FTQC resource review profile.

    Attributes:
        profile (FTQCResourceProfile): Machine-readable profile key.
        label (str): Reader-facing profile label.
        description (str): Short explanation of the review question covered
            by the profile.
        quantities (tuple[FTQCResourceQuantity, ...]): Ordered canonical
            resource quantities included in the profile.

    Example:
        >>> spec = iter_ftqc_resource_profile_specs()[0]
        >>> spec.to_dict()["profile"]
        'chemistry_qpe'
    """

    profile: FTQCResourceProfile
    label: str
    description: str
    quantities: tuple[FTQCResourceQuantity, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize the profile specification.

        Returns:
            dict[str, str | list[str]]: JSON-friendly profile metadata.
        """
        return {
            "profile": self.profile.value,
            "label": self.label,
            "description": self.description,
            "quantities": [quantity.value for quantity in self.quantities],
        }


@dataclass(frozen=True)
class FTQCResearchSignal:
    """Map one research direction to Qamomile resource quantities.

    Attributes:
        reference_key (str): Stable citation key, usually an arXiv identifier.
        title (str): Reader-facing title of the research source or direction.
        url (str): Persistent URL for the source.
        cost_driver (str): Short explanation of the resource driver that the
            source changes or highlights.
        quantities (tuple[FTQCResourceQuantity, ...]): Canonical Qamomile
            quantities that should be inspected when modeling this direction.
        design_note (str): Guidance for how Qamomile should represent the
            signal without prematurely lowering it into backend circuits.
        profiles (tuple[FTQCResourceProfile, ...]): Standard review profiles
            that cover this research signal. Defaults to an empty tuple.

    Example:
        >>> signal = iter_ftqc_research_signals()[0]
        >>> signal.to_dict()["reference_key"].startswith("arXiv:")
        True
    """

    reference_key: str
    title: str
    url: str
    cost_driver: str
    quantities: tuple[FTQCResourceQuantity, ...]
    design_note: str
    profiles: tuple[FTQCResourceProfile, ...] = ()

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize the research signal.

        Returns:
            dict[str, str | list[str]]: JSON-friendly research signal metadata.
        """
        return {
            "reference_key": self.reference_key,
            "title": self.title,
            "url": self.url,
            "cost_driver": self.cost_driver,
            "quantities": [quantity.value for quantity in self.quantities],
            "design_note": self.design_note,
            "profiles": [profile.value for profile in self.profiles],
        }


@dataclass(frozen=True)
class FTQCResearchSignalCoverage:
    """Describe how one estimate covers a research-signal contract.

    Attributes:
        reference_key (str): Research-signal key being audited.
        title (str): Reader-facing title for the signal.
        available (tuple[FTQCResourceQuantity, ...]): Signal quantities
            exposed by the estimate.
        missing (tuple[FTQCResourceQuantity, ...]): Signal quantities not
            exposed by the estimate.
        total (int): Number of quantities in the research signal.

    Example:
        >>> coverage = FTQCResearchSignalCoverage(
        ...     reference_key="arXiv:example",
        ...     title="Toy signal",
        ...     available=(FTQCResourceQuantity.LAMBDA_NORM,),
        ...     missing=(FTQCResourceQuantity.QPE_ITERATIONS,),
        ...     total=2,
        ... )
        >>> coverage.is_complete
        False
    """

    reference_key: str
    title: str
    available: tuple[FTQCResourceQuantity, ...]
    missing: tuple[FTQCResourceQuantity, ...]
    total: int

    def __post_init__(self) -> None:
        """Validate coverage counts after dataclass construction.

        Raises:
            ValueError: If ``total`` is not positive or does not match the
                number of available plus missing quantities.
        """
        if self.total <= 0:
            raise ValueError("total must be positive.")
        observed_total = len(self.available) + len(self.missing)
        if observed_total != self.total:
            raise ValueError(
                "total must equal the number of available and missing quantities."
            )

    @property
    def is_complete(self) -> bool:
        """Return whether every signal quantity is exposed.

        Returns:
            bool: True when no research-signal quantities are missing.
        """
        return not self.missing

    @property
    def coverage_fraction(self) -> sp.Rational:
        """Return the covered fraction of the signal contract.

        Returns:
            sp.Rational: ``len(available) / total`` as an exact rational.
        """
        return sp.Rational(len(self.available), self.total)

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        """Serialize the coverage audit.

        Returns:
            dict[str, str | bool | list[str]]: JSON-friendly audit metadata,
                quantity lists, and completion status.
        """
        return {
            "reference_key": self.reference_key,
            "title": self.title,
            "available": [quantity.value for quantity in self.available],
            "missing": [quantity.value for quantity in self.missing],
            "coverage_fraction": str(self.coverage_fraction),
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class FTQCResearchSignalCoverageReport:
    """Group research-signal coverage audits for one estimate.

    Attributes:
        title (str): Reader-facing report title.
        estimate_label (str): Label for the audited estimate.
        coverages (tuple[FTQCResearchSignalCoverage, ...]): Coverage audits
            in research-signal order.

    Example:
        >>> coverage = FTQCResearchSignalCoverage(
        ...     reference_key="arXiv:example",
        ...     title="Toy signal",
        ...     available=(FTQCResourceQuantity.LAMBDA_NORM,),
        ...     missing=(),
        ...     total=1,
        ... )
        >>> report = FTQCResearchSignalCoverageReport(
        ...     "Toy coverage",
        ...     "estimate",
        ...     (coverage,),
        ... )
        >>> report.to_dict()["counts"]["complete"]
        1
    """

    title: str
    estimate_label: str
    coverages: tuple[FTQCResearchSignalCoverage, ...]

    def __post_init__(self) -> None:
        """Validate report rows after dataclass construction.

        Raises:
            TypeError: If ``coverages`` contains non-coverage items.
            ValueError: If ``coverages`` is empty.
        """
        if not self.coverages:
            raise ValueError("coverages must not be empty.")
        if not all(
            isinstance(coverage, FTQCResearchSignalCoverage)
            for coverage in self.coverages
        ):
            raise TypeError(
                "coverages must contain only FTQCResearchSignalCoverage instances."
            )
        object.__setattr__(self, "coverages", tuple(self.coverages))

    @property
    def complete(self) -> tuple[FTQCResearchSignalCoverage, ...]:
        """Return research signals with complete quantity coverage.

        Returns:
            tuple[FTQCResearchSignalCoverage, ...]: Complete coverage rows.
        """
        return tuple(coverage for coverage in self.coverages if coverage.is_complete)

    @property
    def incomplete(self) -> tuple[FTQCResearchSignalCoverage, ...]:
        """Return research signals with missing quantities.

        Returns:
            tuple[FTQCResearchSignalCoverage, ...]: Incomplete coverage rows.
        """
        return tuple(
            coverage for coverage in self.coverages if not coverage.is_complete
        )

    def to_row_table(self) -> list[dict[str, str | bool]]:
        """Return coverage audits as table rows.

        Returns:
            list[dict[str, str | bool]]: Rows with estimate label, signal key,
                coverage fraction, completion status, and quantity lists.
        """
        rows = []
        for coverage in self.coverages:
            rows.append(
                {
                    "estimate_label": self.estimate_label,
                    "reference_key": coverage.reference_key,
                    "title": coverage.title,
                    "coverage_fraction": str(coverage.coverage_fraction),
                    "is_complete": coverage.is_complete,
                    "available": ", ".join(
                        quantity.value for quantity in coverage.available
                    ),
                    "missing": ", ".join(
                        quantity.value for quantity in coverage.missing
                    ),
                }
            )
        return rows

    def to_dict(self) -> _CoverageReportDict:
        """Serialize grouped coverage audits.

        Returns:
            dict[str, str | list[dict[str, str | bool]] |
                list[dict[str, str | bool | list[str]]] | dict[str, int]]:
                JSON-friendly report metadata, rows, grouped coverage audits,
                and group counts.
        """
        return {
            "title": self.title,
            "estimate_label": self.estimate_label,
            "rows": self.to_row_table(),
            "complete": [coverage.to_dict() for coverage in self.complete],
            "incomplete": [coverage.to_dict() for coverage in self.incomplete],
            "counts": {
                "complete": len(self.complete),
                "incomplete": len(self.incomplete),
            },
        }


@dataclass(frozen=True)
class FTQCResourceFormula:
    """Describe how an FTQC resource quantity is derived.

    Args:
        quantity (str | FTQCResourceQuantity): Quantity produced by the
            formula.
        expression (sp.Expr | int | float): Symbolic expression for the
            derivation. Symbols should use stable resource or model knob
            names so reports can show the formula independently from a
            concrete estimate value.
        depends_on (tuple[str | FTQCResourceQuantity, ...]): Canonical
            resource quantities referenced by the formula. Defaults to an
            empty tuple for formulas that use only model-specific symbols.
        description (str): Reader-facing explanation of the formula's role.
            Defaults to an empty string.
        reference_keys (tuple[str, ...]): Research reference keys that justify
            the formula. Defaults to an empty tuple.

    Raises:
        TypeError: If ``expression`` cannot be converted to a SymPy
            expression.
        ValueError: If ``quantity`` or a dependency is not a known FTQC
            resource quantity.

    Example:
        >>> formula = FTQCResourceFormula(
        ...     quantity="qpe_iterations",
        ...     expression=sp.Symbol("lambda_norm") / sp.Symbol("target_precision"),
        ...     depends_on=("lambda_norm", "target_precision"),
        ... )
        >>> formula.to_dict()["quantity"]
        'qpe_iterations'
    """

    quantity: str | FTQCResourceQuantity
    expression: sp.Expr | int | float
    depends_on: tuple[str | FTQCResourceQuantity, ...] = ()
    description: str = ""
    reference_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize formula fields after dataclass construction.

        Raises:
            TypeError: If ``expression`` cannot be sympified.
            ValueError: If a quantity key is unknown.
        """
        object.__setattr__(
            self,
            "quantity",
            _normalize_resource_quantity(self.quantity),
        )
        try:
            expression = sp.sympify(self.expression)
        except (TypeError, sp.SympifyError) as exc:
            raise TypeError(
                "expression must be a numeric or SymPy expression."
            ) from exc
        object.__setattr__(self, "expression", expression)
        object.__setattr__(
            self,
            "depends_on",
            tuple(
                _normalize_resource_quantity(quantity) for quantity in self.depends_on
            ),
        )

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize the formula.

        Returns:
            dict[str, str | list[str]]: JSON-friendly formula metadata.
        """
        quantity = cast(FTQCResourceQuantity, self.quantity)
        depends_on = cast(tuple[FTQCResourceQuantity, ...], self.depends_on)
        return {
            "quantity": quantity.value,
            "expression": str(self.expression),
            "depends_on": [quantity.value for quantity in depends_on],
            "description": self.description,
            "reference_keys": list(self.reference_keys),
        }


@dataclass(frozen=True)
class FTQCResourceComparisonRow:
    """Compare one FTQC resource quantity between two estimates.

    Attributes:
        quantity (FTQCResourceQuantity): Compared resource quantity.
        baseline (sp.Expr): Baseline estimate value.
        candidate (sp.Expr): Candidate estimate value.
        ratio (sp.Expr): Candidate divided by baseline.
        reduction (sp.Expr): Fractional reduction, equal to
            ``1 - candidate / baseline``.
        label (str): Reader-facing quantity label.
        unit (str): Resource unit.
        category (FTQCResourceCategory): Modeling layer for the quantity.

    Example:
        >>> row = FTQCResourceComparisonRow(
        ...     quantity=FTQCResourceQuantity.TOFFOLI_GATES,
        ...     baseline=10,
        ...     candidate=4,
        ...     ratio=sp.Rational(2, 5),
        ...     reduction=sp.Rational(3, 5),
        ...     label="Toffoli gates",
        ...     unit="Toffoli gates",
        ...     category=FTQCResourceCategory.LOGICAL,
        ... )
        >>> row.to_dict()["ratio"]
        '2/5'
    """

    quantity: FTQCResourceQuantity
    baseline: sp.Expr
    candidate: sp.Expr
    ratio: sp.Expr
    reduction: sp.Expr
    label: str
    unit: str
    category: FTQCResourceCategory

    def to_dict(self) -> dict[str, str]:
        """Serialize the comparison row.

        Returns:
            dict[str, str]: JSON-friendly comparison metadata and values.
        """
        return {
            "quantity": self.quantity.value,
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
            "baseline": str(self.baseline),
            "candidate": str(self.candidate),
            "ratio": str(self.ratio),
            "reduction": str(self.reduction),
        }


@dataclass(frozen=True)
class FTQCResourceComparisonSummary:
    """Group FTQC comparison rows by the sign of their resource change.

    Attributes:
        rows (tuple[FTQCResourceComparisonRow, ...]): All comparison rows in
            the order returned by ``compare_ftqc_resource_estimates``.
        smaller (tuple[FTQCResourceComparisonRow, ...]): Rows where the
            candidate is provably smaller than the baseline. Rows are sorted by
            descending fractional reduction when the reduction is numeric.
        larger (tuple[FTQCResourceComparisonRow, ...]): Rows where the
            candidate is provably larger than the baseline. Rows are sorted by
            descending fractional increase when the increase is numeric.
        unchanged (tuple[FTQCResourceComparisonRow, ...]): Rows where the
            candidate and baseline are provably equal.
        symbolic (tuple[FTQCResourceComparisonRow, ...]): Rows whose change
            sign cannot be proven from the symbolic expression alone.

    Example:
        >>> row = FTQCResourceComparisonRow(
        ...     quantity=FTQCResourceQuantity.TOFFOLI_GATES,
        ...     baseline=10,
        ...     candidate=4,
        ...     ratio=sp.Rational(2, 5),
        ...     reduction=sp.Rational(3, 5),
        ...     label="Toffoli gates",
        ...     unit="Toffoli gates",
        ...     category=FTQCResourceCategory.LOGICAL,
        ... )
        >>> FTQCResourceComparisonSummary.from_rows((row,)).smaller[0].quantity
        <FTQCResourceQuantity.TOFFOLI_GATES: 'toffoli_gates'>
    """

    rows: tuple[FTQCResourceComparisonRow, ...]
    smaller: tuple[FTQCResourceComparisonRow, ...]
    larger: tuple[FTQCResourceComparisonRow, ...]
    unchanged: tuple[FTQCResourceComparisonRow, ...]
    symbolic: tuple[FTQCResourceComparisonRow, ...]

    @classmethod
    def from_rows(
        cls,
        rows: tuple[FTQCResourceComparisonRow, ...],
    ) -> FTQCResourceComparisonSummary:
        """Build a grouped summary from comparison rows.

        Args:
            rows (tuple[FTQCResourceComparisonRow, ...]): Comparison rows to
                classify by the sign of ``reduction``.

        Returns:
            FTQCResourceComparisonSummary: Grouped comparison summary.
        """
        grouped: dict[FTQCResourceChangeDirection, list[FTQCResourceComparisonRow]] = {
            FTQCResourceChangeDirection.SMALLER: [],
            FTQCResourceChangeDirection.LARGER: [],
            FTQCResourceChangeDirection.UNCHANGED: [],
            FTQCResourceChangeDirection.SYMBOLIC: [],
        }
        for row in rows:
            grouped[_classify_change(row.reduction)].append(row)

        return cls(
            rows=rows,
            smaller=_sort_rows_by_change(grouped[FTQCResourceChangeDirection.SMALLER]),
            larger=_sort_rows_by_change(grouped[FTQCResourceChangeDirection.LARGER]),
            unchanged=tuple(grouped[FTQCResourceChangeDirection.UNCHANGED]),
            symbolic=tuple(grouped[FTQCResourceChangeDirection.SYMBOLIC]),
        )

    def to_dict(self) -> dict[str, list[dict[str, str]] | dict[str, int]]:
        """Serialize grouped comparison rows.

        Returns:
            dict[str, list[dict[str, str]] | dict[str, int]]: JSON-friendly
                summary containing all rows, grouped rows, and group counts.
        """
        return {
            "rows": [row.to_dict() for row in self.rows],
            "smaller": [row.to_dict() for row in self.smaller],
            "larger": [row.to_dict() for row in self.larger],
            "unchanged": [row.to_dict() for row in self.unchanged],
            "symbolic": [row.to_dict() for row in self.symbolic],
            "counts": {
                "smaller": len(self.smaller),
                "larger": len(self.larger),
                "unchanged": len(self.unchanged),
                "symbolic": len(self.symbolic),
            },
        }


@dataclass(frozen=True)
class FTQCResourceReviewFinding:
    """Describe one review finding from an FTQC resource comparison.

    Attributes:
        direction (FTQCResourceChangeDirection): Change class for the finding.
        quantity (FTQCResourceQuantity): Compared resource quantity.
        label (str): Reader-facing quantity label.
        unit (str): Resource unit.
        category (FTQCResourceCategory): Modeling layer for the quantity.
        baseline (sp.Expr): Baseline estimate value.
        candidate (sp.Expr): Candidate estimate value.
        ratio (sp.Expr): Candidate divided by baseline.
        reduction (sp.Expr): Fractional reduction, equal to
            ``1 - candidate / baseline``.
        headline (str): Short reader-facing review headline.
        detail (str): Compact value-level explanation for reports.

    Example:
        >>> row = FTQCResourceComparisonRow(
        ...     quantity=FTQCResourceQuantity.TOFFOLI_GATES,
        ...     baseline=10,
        ...     candidate=4,
        ...     ratio=sp.Rational(2, 5),
        ...     reduction=sp.Rational(3, 5),
        ...     label="Toffoli gates",
        ...     unit="Toffoli gates",
        ...     category=FTQCResourceCategory.LOGICAL,
        ... )
        >>> finding = build_ftqc_resource_review_findings(
        ...     FTQCResourceComparisonSummary.from_rows((row,)),
        ... )[0]
        >>> finding.direction
        <FTQCResourceChangeDirection.SMALLER: 'smaller'>
    """

    direction: FTQCResourceChangeDirection
    quantity: FTQCResourceQuantity
    label: str
    unit: str
    category: FTQCResourceCategory
    baseline: sp.Expr
    candidate: sp.Expr
    ratio: sp.Expr
    reduction: sp.Expr
    headline: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the review finding.

        Returns:
            dict[str, str]: JSON-friendly finding metadata and values.
        """
        return {
            "direction": self.direction.value,
            "quantity": self.quantity.value,
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
            "baseline": str(self.baseline),
            "candidate": str(self.candidate),
            "ratio": str(self.ratio),
            "reduction": str(self.reduction),
            "headline": self.headline,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FTQCResourcePlanStep:
    """Describe one abstract FTQC algorithm subroutine resource step.

    A step records resource quantities before the compiler commits to a
    concrete circuit implementation. Repetition scales additive quantities
    such as non-Clifford counts and logical depth, while peak quantities such
    as logical qubits remain peak values unless callers override the rule.

    Args:
        name (str): Stable subroutine name, such as ``"PREPARE"`` or
            ``"filtered_qpe"``.
        resources (dict[str | FTQCResourceQuantity, sp.Expr | int | float]):
            Per-step resource values keyed by canonical FTQC quantity.
        repetitions (sp.Expr | int | float): Number of times this step is
            expected to run. Defaults to one.
        aggregation (dict[str | FTQCResourceQuantity, str |
            FTQCResourceAggregationRule]): Optional per-quantity aggregation
            override for this step. Defaults to canonical quantity rules.
        label (str): Optional reader-facing subroutine label. Defaults to
            ``name`` when serialized.
        formulas (tuple[FTQCResourceFormula, ...]): Symbolic formulas that
            justify this step's resources. Defaults to an empty tuple.
        reference_keys (tuple[str, ...]): Research or design-reference keys
            supporting this step. Defaults to an empty tuple.

    Raises:
        TypeError: If ``resources`` or ``repetitions`` cannot be converted to
            SymPy expressions, if ``formulas`` contains non-formula items, or
            if ``reference_keys`` contains non-string items.
        ValueError: If a quantity or aggregation rule is unknown, or if
            ``repetitions`` is negative when decidable.

    Example:
        >>> step = FTQCResourcePlanStep(
        ...     "walk",
        ...     {"toffoli_gates": 10, "logical_qubits": 4},
        ...     repetitions=3,
        ... )
        >>> step.resource_values()[FTQCResourceQuantity.TOFFOLI_GATES]
        30
    """

    name: str
    resources: dict[str | FTQCResourceQuantity, sp.Expr | int | float]
    repetitions: sp.Expr | int | float = 1
    aggregation: (
        dict[
            str | FTQCResourceQuantity,
            str | FTQCResourceAggregationRule,
        ]
        | None
    ) = None
    label: str = ""
    formulas: tuple[FTQCResourceFormula, ...] = ()
    reference_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize step fields after dataclass construction.

        Raises:
            TypeError: If resource values cannot be sympified, formulas are
                not ``FTQCResourceFormula`` instances, or reference keys are
                not strings.
            ValueError: If a closed-set key is unknown or repetitions are
                negative when decidable.
        """
        if not all(
            isinstance(formula, FTQCResourceFormula) for formula in self.formulas
        ):
            raise TypeError("formulas must contain only FTQCResourceFormula instances.")
        if not all(isinstance(key, str) for key in self.reference_keys):
            raise TypeError("reference_keys must contain only strings.")
        normalized_resources = {
            _normalize_resource_quantity(quantity): _sympify_resource_expr(
                value,
                str(quantity),
            )
            for quantity, value in self.resources.items()
        }
        repetitions = _sympify_resource_expr(self.repetitions, "repetitions")
        if repetitions.is_negative:
            raise ValueError("repetitions must be non-negative.")
        aggregation = self.aggregation or {}
        normalized_aggregation = {
            _normalize_resource_quantity(quantity): _normalize_aggregation_rule(rule)
            for quantity, rule in aggregation.items()
        }
        object.__setattr__(self, "resources", normalized_resources)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "aggregation", normalized_aggregation)
        object.__setattr__(self, "formulas", tuple(self.formulas))
        object.__setattr__(self, "reference_keys", tuple(self.reference_keys))

    def aggregation_rule_for(
        self,
        quantity: str | FTQCResourceQuantity,
    ) -> FTQCResourceAggregationRule:
        """Return the aggregation rule for one step quantity.

        Args:
            quantity (str | FTQCResourceQuantity): Quantity key to inspect.

        Returns:
            FTQCResourceAggregationRule: Step-specific or default aggregation
                rule.

        Raises:
            ValueError: If ``quantity`` is not a known FTQC resource quantity.
        """
        normalized = _normalize_resource_quantity(quantity)
        aggregation = cast(
            dict[FTQCResourceQuantity, FTQCResourceAggregationRule],
            self.aggregation,
        )
        return aggregation.get(
            normalized, default_ftqc_resource_aggregation_rule(normalized)
        )

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return repetition-adjusted step resources.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Canonical resource values for
                this step. Additive quantities are multiplied by
                ``repetitions``; peak and consistent quantities are left as
                per-step values.
        """
        values = {}
        resources = cast(dict[FTQCResourceQuantity, sp.Expr], self.resources)
        repetitions = cast(sp.Expr, self.repetitions)
        for quantity, value in resources.items():
            if self.aggregation_rule_for(quantity) == FTQCResourceAggregationRule.ADD:
                values[quantity] = sp.simplify(value * repetitions)
            else:
                values[quantity] = value
        return values

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan step.

        Returns:
            dict[str, Any]: JSON-friendly subroutine metadata, resource
                values, formulas, and references.
        """
        rows = []
        resources = cast(dict[FTQCResourceQuantity, sp.Expr], self.resources)
        for quantity, raw_value in resources.items():
            adjusted_value = self.resource_values()[quantity]
            spec = describe_ftqc_resource_quantity(quantity)
            rows.append(
                {
                    "quantity": quantity.value,
                    "label": spec.label,
                    "unit": spec.unit,
                    "category": spec.category.value,
                    "value": str(adjusted_value),
                    "per_step_value": str(raw_value),
                    "aggregation": self.aggregation_rule_for(quantity).value,
                }
            )
        return {
            "name": self.name,
            "label": self.label or self.name,
            "repetitions": str(self.repetitions),
            "resources": rows,
            "formulas": [formula.to_dict() for formula in self.formulas],
            "reference_keys": list(self.reference_keys),
        }


@dataclass(frozen=True)
class FTQCResourcePlan:
    """Compose abstract FTQC subroutine steps into one resource model.

    The plan is intentionally an estimator-level abstraction. It lets papers
    and tutorials model algorithms built from PREPARE, SELECT, filtering,
    QPE, or architecture-lift steps without introducing low-level IR gates
    before an implementation strategy is selected.

    Args:
        steps (tuple[FTQCResourcePlanStep, ...]): Ordered subroutine steps.
        title (str): Reader-facing plan title. Defaults to
            ``"FTQC resource plan"``.
        aggregation (dict[str | FTQCResourceQuantity, str |
            FTQCResourceAggregationRule] | None): Optional plan-level
            aggregation override used when combining steps. Defaults to
            canonical quantity rules.

    Raises:
        TypeError: If any item in ``steps`` is not an
            ``FTQCResourcePlanStep``.
        ValueError: If an aggregation key is unknown or if a consistent
            quantity has conflicting values across steps.

    Example:
        >>> step = FTQCResourcePlanStep("walk", {"toffoli_gates": 10}, 2)
        >>> plan = FTQCResourcePlan((step,))
        >>> plan.resource_values()[FTQCResourceQuantity.TOFFOLI_GATES]
        20
    """

    steps: tuple[FTQCResourcePlanStep, ...]
    title: str = "FTQC resource plan"
    aggregation: (
        dict[
            str | FTQCResourceQuantity,
            str | FTQCResourceAggregationRule,
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        """Normalize plan fields after dataclass construction.

        Raises:
            TypeError: If any step is not an ``FTQCResourcePlanStep``.
            ValueError: If an aggregation key or rule is unknown.
        """
        if not all(isinstance(step, FTQCResourcePlanStep) for step in self.steps):
            raise TypeError("steps must contain only FTQCResourcePlanStep instances.")
        aggregation = self.aggregation or {}
        normalized_aggregation = {
            _normalize_resource_quantity(quantity): _normalize_aggregation_rule(rule)
            for quantity, rule in aggregation.items()
        }
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "aggregation", normalized_aggregation)

    def aggregation_rule_for(
        self,
        quantity: str | FTQCResourceQuantity,
        step: FTQCResourcePlanStep | None = None,
    ) -> FTQCResourceAggregationRule:
        """Return the plan-level aggregation rule for one quantity.

        Args:
            quantity (str | FTQCResourceQuantity): Quantity key to inspect.
            step (FTQCResourcePlanStep | None): Optional step whose rule should
                be used when the plan has no explicit override. Defaults to
                None.

        Returns:
            FTQCResourceAggregationRule: Plan, step, or default aggregation
                rule.

        Raises:
            ValueError: If ``quantity`` is not a known FTQC resource quantity.
        """
        normalized = _normalize_resource_quantity(quantity)
        aggregation = cast(
            dict[FTQCResourceQuantity, FTQCResourceAggregationRule],
            self.aggregation,
        )
        if normalized in aggregation:
            return aggregation[normalized]
        if step is not None:
            return step.aggregation_rule_for(normalized)
        return default_ftqc_resource_aggregation_rule(normalized)

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return aggregate resources for the plan.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Canonical resource values
                aggregated across all steps.

        Raises:
            ValueError: If a consistent quantity has conflicting values across
                steps.
        """
        values: dict[FTQCResourceQuantity, sp.Expr] = {}
        for step in self.steps:
            for quantity, value in step.resource_values().items():
                rule = self.aggregation_rule_for(quantity, step)
                if quantity not in values:
                    values[quantity] = value
                    continue
                values[quantity] = _combine_resource_values(
                    quantity,
                    values[quantity],
                    value,
                    rule,
                )
        return values

    def formulas(self) -> tuple[FTQCResourceFormula, ...]:
        """Return step formulas with duplicate formulas removed.

        Formulas are deduplicated by their serialized content while preserving
        the first occurrence. This keeps report metadata compact when multiple
        steps cite the same derivation.

        Returns:
            tuple[FTQCResourceFormula, ...]: Formulas referenced by plan steps.
        """
        formulas: list[FTQCResourceFormula] = []
        seen: set[
            tuple[str, str, tuple[FTQCResourceQuantity, ...], str, tuple[str, ...]]
        ] = set()
        for step in self.steps:
            for formula in step.formulas:
                quantity = cast(FTQCResourceQuantity, formula.quantity)
                depends_on = cast(tuple[FTQCResourceQuantity, ...], formula.depends_on)
                key = (
                    quantity.value,
                    str(formula.expression),
                    depends_on,
                    formula.description,
                    formula.reference_keys,
                )
                if key in seen:
                    continue
                formulas.append(formula)
                seen.add(key)
        return tuple(formulas)

    def reference_keys(self) -> tuple[str, ...]:
        """Return reference keys used by plan steps and formulas.

        Returns:
            tuple[str, ...]: Deduplicated reference keys preserving first-seen
                order.
        """
        references: list[str] = []
        seen: set[str] = set()
        for step in self.steps:
            for key in (*step.reference_keys, *self._formula_reference_keys(step)):
                if key in seen:
                    continue
                references.append(key)
                seen.add(key)
        return tuple(references)

    def _formula_reference_keys(
        self,
        step: FTQCResourcePlanStep,
    ) -> tuple[str, ...]:
        """Return reference keys cited by one plan step's formulas.

        Args:
            step (FTQCResourcePlanStep): Step whose formulas should be
                inspected.

        Returns:
            tuple[str, ...]: Reference keys cited by the step's formulas.
        """
        return tuple(key for formula in step.formulas for key in formula.reference_keys)

    def to_quantity_table(self) -> list[dict[str, str]]:
        """Return aggregate plan resources as table rows.

        Returns:
            list[dict[str, str]]: Rows containing quantity metadata,
                aggregate values, and aggregation rules.

        Raises:
            ValueError: If a consistent quantity has conflicting values across
                steps.
        """
        values = self.resource_values()
        rows = []
        for spec in FTQC_RESOURCE_QUANTITY_SPECS:
            if spec.quantity not in values:
                continue
            rows.append(
                {
                    "quantity": spec.quantity.value,
                    "label": spec.label,
                    "unit": spec.unit,
                    "category": spec.category.value,
                    "value": str(values[spec.quantity]),
                    "aggregation": self.aggregation_rule_for(spec.quantity).value,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resource plan.

        Returns:
            dict[str, Any]: JSON-friendly plan metadata, steps, aggregate
                resource rows, formulas, and references.

        Raises:
            ValueError: If a consistent quantity has conflicting values across
                steps.
        """
        return {
            "title": self.title,
            "steps": [step.to_dict() for step in self.steps],
            "resources": self.to_quantity_table(),
            "formulas": [formula.to_dict() for formula in self.formulas()],
            "reference_keys": list(self.reference_keys()),
        }


@dataclass(frozen=True)
class FTQCResourceConstraint:
    """Declare a symbolic budget constraint for one FTQC resource quantity.

    Args:
        quantity (str | FTQCResourceQuantity): Canonical resource quantity to
            constrain.
        limit (sp.Expr | int | float): Symbolic or numeric constraint limit.
        sense (str | FTQCResourceConstraintSense): Constraint direction.
            ``AT_MOST`` means the resource value may not exceed ``limit``;
            ``AT_LEAST`` means it must meet or exceed ``limit``. Defaults to
            ``AT_MOST``.
        label (str): Optional reader-facing label for this constraint.
            Defaults to an empty string, meaning the quantity label is used.

    Raises:
        TypeError: If ``limit`` cannot be converted to a SymPy expression.
        ValueError: If ``quantity`` or ``sense`` is unknown.

    Example:
        >>> constraint = FTQCResourceConstraint("physical_qubits", 100_000)
        >>> constraint.quantity
        <FTQCResourceQuantity.PHYSICAL_QUBITS: 'physical_qubits'>
    """

    quantity: str | FTQCResourceQuantity
    limit: sp.Expr | int | float
    sense: str | FTQCResourceConstraintSense = FTQCResourceConstraintSense.AT_MOST
    label: str = ""

    def __post_init__(self) -> None:
        """Normalize constraint fields after dataclass construction.

        Raises:
            TypeError: If ``limit`` cannot be sympified.
            ValueError: If a closed-set key is unknown.
        """
        object.__setattr__(
            self,
            "quantity",
            _normalize_resource_quantity(self.quantity),
        )
        object.__setattr__(
            self,
            "sense",
            _normalize_constraint_sense(self.sense),
        )
        object.__setattr__(
            self,
            "limit",
            _sympify_resource_expr(self.limit, "limit"),
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize the resource constraint.

        Returns:
            dict[str, str]: JSON-friendly constraint metadata.
        """
        quantity = cast(FTQCResourceQuantity, self.quantity)
        sense = cast(FTQCResourceConstraintSense, self.sense)
        return {
            "quantity": quantity.value,
            "limit": str(self.limit),
            "sense": sense.value,
            "label": self.label,
        }


@dataclass(frozen=True)
class FTQCResourceConstraintResult:
    """Describe the result of evaluating one FTQC resource constraint.

    Attributes:
        quantity (FTQCResourceQuantity): Constrained resource quantity.
        status (FTQCResourceConstraintStatus): Whether the constraint is met,
            violated, or symbolic.
        sense (FTQCResourceConstraintSense): Constraint comparison direction.
        value (sp.Expr): Resource value from the evaluated estimate.
        limit (sp.Expr): Constraint limit.
        margin (sp.Expr): Signed headroom. Positive or zero means the
            constraint is satisfied; negative means it is violated.
        label (str): Reader-facing constraint label.
        unit (str): Resource unit.
        category (FTQCResourceCategory): Modeling layer for the quantity.

    Example:
        >>> result = FTQCResourceConstraintResult(
        ...     quantity=FTQCResourceQuantity.PHYSICAL_QUBITS,
        ...     status=FTQCResourceConstraintStatus.SATISFIED,
        ...     sense=FTQCResourceConstraintSense.AT_MOST,
        ...     value=10,
        ...     limit=100,
        ...     margin=90,
        ...     label="Physical qubits",
        ...     unit="physical qubits",
        ...     category=FTQCResourceCategory.PHYSICAL,
        ... )
        >>> result.to_dict()["status"]
        'satisfied'
    """

    quantity: FTQCResourceQuantity
    status: FTQCResourceConstraintStatus
    sense: FTQCResourceConstraintSense
    value: sp.Expr
    limit: sp.Expr
    margin: sp.Expr
    label: str
    unit: str
    category: FTQCResourceCategory

    def to_dict(self) -> dict[str, str]:
        """Serialize the constraint result.

        Returns:
            dict[str, str]: JSON-friendly result metadata and values.
        """
        return {
            "quantity": self.quantity.value,
            "status": self.status.value,
            "sense": self.sense.value,
            "value": str(self.value),
            "limit": str(self.limit),
            "margin": str(self.margin),
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
        }


@dataclass(frozen=True)
class FTQCResourceBudgetReport:
    """Group FTQC resource constraint results for a budget review.

    Attributes:
        title (str): Reader-facing report title.
        results (tuple[FTQCResourceConstraintResult, ...]): Evaluated
            constraints in input order.
        satisfied (tuple[FTQCResourceConstraintResult, ...]): Constraints that
            are provably satisfied.
        violated (tuple[FTQCResourceConstraintResult, ...]): Constraints that
            are provably violated.
        symbolic (tuple[FTQCResourceConstraintResult, ...]): Constraints whose
            status cannot be decided from the symbolic expression.

    Example:
        >>> constraint = FTQCResourceConstraint("physical_qubits", 100)
        >>> report = FTQCResourceBudgetReport.from_results(
        ...     "Toy budget",
        ...     (
        ...         FTQCResourceConstraintResult(
        ...             quantity=FTQCResourceQuantity.PHYSICAL_QUBITS,
        ...             status=FTQCResourceConstraintStatus.SATISFIED,
        ...             sense=FTQCResourceConstraintSense.AT_MOST,
        ...             value=10,
        ...             limit=constraint.limit,
        ...             margin=90,
        ...             label="Physical qubits",
        ...             unit="physical qubits",
        ...             category=FTQCResourceCategory.PHYSICAL,
        ...         ),
        ...     ),
        ... )
        >>> report.to_dict()["counts"]["satisfied"]
        1
    """

    title: str
    results: tuple[FTQCResourceConstraintResult, ...]
    satisfied: tuple[FTQCResourceConstraintResult, ...]
    violated: tuple[FTQCResourceConstraintResult, ...]
    symbolic: tuple[FTQCResourceConstraintResult, ...]

    @classmethod
    def from_results(
        cls,
        title: str,
        results: tuple[FTQCResourceConstraintResult, ...],
    ) -> FTQCResourceBudgetReport:
        """Build a grouped budget report from constraint results.

        Args:
            title (str): Reader-facing report title.
            results (tuple[FTQCResourceConstraintResult, ...]): Evaluated
                constraints to group.

        Returns:
            FTQCResourceBudgetReport: Grouped budget report.
        """
        grouped: dict[
            FTQCResourceConstraintStatus,
            list[FTQCResourceConstraintResult],
        ] = {
            FTQCResourceConstraintStatus.SATISFIED: [],
            FTQCResourceConstraintStatus.VIOLATED: [],
            FTQCResourceConstraintStatus.SYMBOLIC: [],
        }
        for result in results:
            grouped[result.status].append(result)

        return cls(
            title=title,
            results=results,
            satisfied=tuple(grouped[FTQCResourceConstraintStatus.SATISFIED]),
            violated=tuple(grouped[FTQCResourceConstraintStatus.VIOLATED]),
            symbolic=tuple(grouped[FTQCResourceConstraintStatus.SYMBOLIC]),
        )

    def to_dict(self) -> dict[str, str | list[dict[str, str]] | dict[str, int]]:
        """Serialize grouped budget results.

        Returns:
            dict[str, str | list[dict[str, str]] | dict[str, int]]: JSON-friendly
                report metadata, rows, grouped rows, and counts.
        """
        return {
            "title": self.title,
            "results": [result.to_dict() for result in self.results],
            "satisfied": [result.to_dict() for result in self.satisfied],
            "violated": [result.to_dict() for result in self.violated],
            "symbolic": [result.to_dict() for result in self.symbolic],
            "counts": {
                "satisfied": len(self.satisfied),
                "violated": len(self.violated),
                "symbolic": len(self.symbolic),
            },
        }


@dataclass(frozen=True)
class FTQCResourceComparisonReport:
    """Package an FTQC comparison summary with review metadata.

    Attributes:
        title (str): Reader-facing report title.
        baseline_label (str): Label for the baseline estimate.
        candidate_label (str): Label for the candidate estimate.
        profile (FTQCResourceProfile | None): Standard review profile used to
            select comparison quantities, or None when no profile was used.
        summary (FTQCResourceComparisonSummary): Grouped comparison rows.

    Example:
        >>> row = FTQCResourceComparisonRow(
        ...     quantity=FTQCResourceQuantity.TOFFOLI_GATES,
        ...     baseline=10,
        ...     candidate=4,
        ...     ratio=sp.Rational(2, 5),
        ...     reduction=sp.Rational(3, 5),
        ...     label="Toffoli gates",
        ...     unit="Toffoli gates",
        ...     category=FTQCResourceCategory.LOGICAL,
        ... )
        >>> report = FTQCResourceComparisonReport(
        ...     title="Toy report",
        ...     baseline_label="baseline",
        ...     candidate_label="candidate",
        ...     profile=None,
        ...     summary=FTQCResourceComparisonSummary.from_rows((row,)),
        ... )
        >>> report.to_dict()["counts"]["smaller"]
        1
    """

    title: str
    baseline_label: str
    candidate_label: str
    profile: FTQCResourceProfile | None
    summary: FTQCResourceComparisonSummary

    def to_dict(
        self,
    ) -> dict[
        str,
        str | None | list[str] | list[dict[str, str]] | dict[str, int],
    ]:
        """Serialize report metadata and comparison rows.

        Returns:
            dict[str, str | None | list[str] | list[dict[str, str]] | dict[str, int]]:
                JSON-friendly report metadata, selected quantities, rows,
                prioritized findings, and grouped-count summary.
        """
        summary = self.summary.to_dict()
        return {
            "title": self.title,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "profile": None if self.profile is None else self.profile.value,
            "quantities": [row.quantity.value for row in self.summary.rows],
            "rows": summary["rows"],
            "smaller": summary["smaller"],
            "larger": summary["larger"],
            "unchanged": summary["unchanged"],
            "symbolic": summary["symbolic"],
            "findings": [finding.to_dict() for finding in self.to_review_findings()],
            "counts": summary["counts"],
        }

    def to_row_table(self) -> list[dict[str, str]]:
        """Return comparison rows with report labels attached.

        Returns:
            list[dict[str, str]]: Rows containing the serialized comparison
                plus ``baseline_label`` and ``candidate_label`` columns.
        """
        rows = []
        for row in self.summary.rows:
            serialized = row.to_dict()
            serialized["baseline_label"] = self.baseline_label
            serialized["candidate_label"] = self.candidate_label
            rows.append(serialized)
        return rows

    def to_review_findings(
        self,
        *,
        max_improvements: int = 3,
        max_tradeoffs: int = 3,
        include_symbolic: bool = True,
        include_unchanged: bool = False,
    ) -> tuple[FTQCResourceReviewFinding, ...]:
        """Return prioritized review findings for this report.

        Args:
            max_improvements (int): Maximum number of reducing quantities to
                include. Defaults to 3.
            max_tradeoffs (int): Maximum number of increasing quantities to
                include. Defaults to 3.
            include_symbolic (bool): Whether to include undecidable symbolic
                changes after numeric findings. Defaults to True.
            include_unchanged (bool): Whether to include unchanged quantities
                after symbolic findings. Defaults to False.

        Returns:
            tuple[FTQCResourceReviewFinding, ...]: Prioritized findings for
                report review.

        Raises:
            ValueError: If either maximum is negative.
        """
        return build_ftqc_resource_review_findings(
            self.summary,
            max_improvements=max_improvements,
            max_tradeoffs=max_tradeoffs,
            include_symbolic=include_symbolic,
            include_unchanged=include_unchanged,
        )


@dataclass(frozen=True)
class FTQCResourceDriverReport:
    """Package a formula-scoped FTQC resource driver comparison.

    Attributes:
        title (str): Reader-facing report title.
        baseline_label (str): Label for the baseline estimate.
        candidate_label (str): Label for the candidate estimate.
        targets (tuple[FTQCResourceQuantity, ...]): Target output quantities
            whose formula dependencies scoped the report.
        summary (FTQCResourceComparisonSummary): Grouped comparison rows for
            available target and driver quantities.

    Example:
        >>> row = FTQCResourceComparisonRow(
        ...     quantity=FTQCResourceQuantity.T_GATES,
        ...     baseline=10,
        ...     candidate=2,
        ...     ratio=sp.Rational(1, 5),
        ...     reduction=sp.Rational(4, 5),
        ...     label="T gates",
        ...     unit="T gates",
        ...     category=FTQCResourceCategory.LOGICAL,
        ... )
        >>> report = FTQCResourceDriverReport(
        ...     title="Toy driver report",
        ...     baseline_label="baseline",
        ...     candidate_label="candidate",
        ...     targets=(FTQCResourceQuantity.T_GATES,),
        ...     summary=FTQCResourceComparisonSummary.from_rows((row,)),
        ... )
        >>> report.to_row_table()[0]["is_target"]
        True
    """

    title: str
    baseline_label: str
    candidate_label: str
    targets: tuple[FTQCResourceQuantity, ...]
    summary: FTQCResourceComparisonSummary

    def __post_init__(self) -> None:
        """Validate driver-report targets after dataclass construction.

        Raises:
            ValueError: If ``targets`` is empty.
        """
        if not self.targets:
            raise ValueError("targets must not be empty.")
        object.__setattr__(self, "targets", tuple(self.targets))

    def to_row_table(self) -> list[dict[str, str | bool]]:
        """Return driver rows with target markers and estimate labels.

        Returns:
            list[dict[str, str | bool]]: Rows containing serialized comparison
                values plus baseline, candidate, and target-marker columns.
        """
        target_set = set(self.targets)
        rows = []
        for row in self.summary.rows:
            serialized: dict[str, str | bool] = dict(row.to_dict())
            serialized["baseline_label"] = self.baseline_label
            serialized["candidate_label"] = self.candidate_label
            serialized["is_target"] = row.quantity in target_set
            rows.append(serialized)
        return rows

    def to_dict(
        self,
    ) -> dict[
        str,
        str
        | list[str]
        | list[dict[str, str]]
        | list[dict[str, str | bool]]
        | dict[str, int],
    ]:
        """Serialize report metadata and driver rows.

        Returns:
            dict[str, str | list[str] | list[dict[str, str]] |
                list[dict[str, str | bool]] | dict[str, int]]: JSON-friendly
                report metadata, target quantities, row table, prioritized
                findings, and grouped counts.
        """
        counts = cast(dict[str, int], self.summary.to_dict()["counts"])
        return {
            "title": self.title,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "targets": [quantity.value for quantity in self.targets],
            "quantities": [row.quantity.value for row in self.summary.rows],
            "rows": self.to_row_table(),
            "findings": [
                finding.to_dict()
                for finding in build_ftqc_resource_review_findings(self.summary)
            ],
            "counts": counts,
        }


@dataclass(frozen=True)
class FTQCResourceParetoRow:
    """Describe one candidate in an FTQC Pareto review.

    Args:
        label (str): Reader-facing candidate label.
        values (dict[str | FTQCResourceQuantity, sp.Expr | int | float]):
            Candidate resource values keyed by canonical quantities. Smaller
            values are treated as better for Pareto dominance.
        dominated_by (tuple[str, ...]): Labels of candidates that provably
            dominate this row. Defaults to an empty tuple.

    Raises:
        TypeError: If any value cannot be converted to a SymPy expression.
        ValueError: If ``label`` is empty or a resource key is unknown.

    Example:
        >>> row = FTQCResourceParetoRow(
        ...     "compressed",
        ...     {"physical_qubits": 100, "runtime_seconds": 2},
        ... )
        >>> row.is_frontier
        True
    """

    label: str
    values: dict[str | FTQCResourceQuantity, sp.Expr | int | float]
    dominated_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize Pareto-row fields after dataclass construction.

        Raises:
            TypeError: If any value cannot be converted to a SymPy expression.
            ValueError: If ``label`` is empty or a resource key is unknown.
        """
        if not self.label:
            raise ValueError("label must not be empty.")
        normalized_values = {
            _normalize_resource_quantity(quantity): _sympify_resource_expr(
                value,
                str(quantity),
            )
            for quantity, value in self.values.items()
        }
        object.__setattr__(self, "values", normalized_values)
        object.__setattr__(self, "dominated_by", tuple(self.dominated_by))

    @property
    def is_frontier(self) -> bool:
        """Return whether no other candidate dominates this row.

        Returns:
            bool: True when ``dominated_by`` is empty.
        """
        return not self.dominated_by

    def to_dict(self) -> dict[str, str | bool | list[str] | dict[str, str]]:
        """Serialize the Pareto row.

        Returns:
            dict[str, str | bool | list[str] | dict[str, str]]: JSON-friendly
                candidate label, values, frontier marker, and dominators.
        """
        values = cast(dict[FTQCResourceQuantity, sp.Expr], self.values)
        return {
            "label": self.label,
            "values": {
                quantity.value: str(value) for quantity, value in values.items()
            },
            "is_frontier": self.is_frontier,
            "dominated_by": list(self.dominated_by),
        }


@dataclass(frozen=True)
class FTQCResourceParetoReport:
    """Package a multi-candidate FTQC Pareto-frontier review.

    Attributes:
        title (str): Reader-facing report title.
        quantities (tuple[FTQCResourceQuantity, ...]): Quantities used for
            dominance checks. Smaller values are treated as better.
        rows (tuple[FTQCResourceParetoRow, ...]): Candidate rows in input
            order, annotated with dominance status.

    Example:
        >>> report = FTQCResourceParetoReport(
        ...     title="Toy frontier",
        ...     quantities=(FTQCResourceQuantity.RUNTIME_SECONDS,),
        ...     rows=(
        ...         FTQCResourceParetoRow("fast", {"runtime_seconds": 1}),
        ...         FTQCResourceParetoRow("slow", {"runtime_seconds": 2}, ("fast",)),
        ...     ),
        ... )
        >>> report.frontier[0].label
        'fast'
    """

    title: str
    quantities: tuple[FTQCResourceQuantity, ...]
    rows: tuple[FTQCResourceParetoRow, ...]

    def __post_init__(self) -> None:
        """Validate Pareto-report rows after dataclass construction.

        Raises:
            TypeError: If ``rows`` contains non-row items.
            ValueError: If ``quantities`` or ``rows`` is empty.
        """
        if not self.quantities:
            raise ValueError("quantities must not be empty.")
        if not self.rows:
            raise ValueError("rows must not be empty.")
        if not all(isinstance(row, FTQCResourceParetoRow) for row in self.rows):
            raise TypeError("rows must contain only FTQCResourceParetoRow instances.")
        object.__setattr__(self, "quantities", tuple(self.quantities))
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def frontier(self) -> tuple[FTQCResourceParetoRow, ...]:
        """Return non-dominated candidate rows.

        Returns:
            tuple[FTQCResourceParetoRow, ...]: Rows with no provable
                dominators.
        """
        return tuple(row for row in self.rows if row.is_frontier)

    @property
    def dominated(self) -> tuple[FTQCResourceParetoRow, ...]:
        """Return dominated candidate rows.

        Returns:
            tuple[FTQCResourceParetoRow, ...]: Rows with at least one
                dominator.
        """
        return tuple(row for row in self.rows if not row.is_frontier)

    def to_row_table(self) -> list[dict[str, str | bool]]:
        """Return candidates as flat table rows.

        Returns:
            list[dict[str, str | bool]]: Rows with one column per selected
                quantity plus frontier and dominator metadata.
        """
        rows = []
        for row in self.rows:
            values = cast(dict[FTQCResourceQuantity, sp.Expr], row.values)
            serialized: dict[str, str | bool] = {
                "label": row.label,
                "is_frontier": row.is_frontier,
                "dominated_by": ", ".join(row.dominated_by),
            }
            for quantity in self.quantities:
                serialized[quantity.value] = str(values[quantity])
            rows.append(serialized)
        return rows

    def to_dict(
        self,
    ) -> dict[
        str,
        str
        | list[str]
        | list[dict[str, str | bool]]
        | list[dict[str, str | bool | list[str] | dict[str, str]]]
        | dict[str, int],
    ]:
        """Serialize report metadata and candidate rows.

        Returns:
            dict[str, str | list[str] | list[dict[str, str | bool]] |
                list[dict[str, str | bool | list[str] | dict[str, str]]] |
                dict[str, int]]: JSON-friendly report metadata, selected
                quantities, rows, frontier rows, dominated rows, and counts.
        """
        return {
            "title": self.title,
            "quantities": [quantity.value for quantity in self.quantities],
            "rows": self.to_row_table(),
            "frontier": [row.to_dict() for row in self.frontier],
            "dominated": [row.to_dict() for row in self.dominated],
            "counts": {
                "frontier": len(self.frontier),
                "dominated": len(self.dominated),
            },
        }


@dataclass(frozen=True)
class FTQCResourceScenario:
    """Describe one symbolic FTQC resource-evaluation scenario.

    Args:
        label (str): Reader-facing scenario label.
        substitutions (dict[str | sp.Symbol, sp.Expr | int | float]):
            Symbol substitutions applied to selected resource expressions.
            String keys are converted to SymPy symbols.

    Raises:
        TypeError: If a substitution value cannot be converted to a SymPy
            expression.
        ValueError: If ``label`` is empty.

    Example:
        >>> scenario = FTQCResourceScenario(
        ...     "fast factory",
        ...     {"toffoli_throughput": 1_000_000},
        ... )
        >>> scenario.to_dict()["substitutions"]["toffoli_throughput"]
        '1000000'
    """

    label: str
    substitutions: dict[str | sp.Symbol, sp.Expr | int | float]

    def __post_init__(self) -> None:
        """Normalize scenario fields after dataclass construction.

        Raises:
            TypeError: If a substitution value cannot be sympified.
            ValueError: If ``label`` is empty.
        """
        if not self.label:
            raise ValueError("label must not be empty.")
        normalized = {
            _normalize_substitution_symbol(symbol): _sympify_resource_expr(
                value,
                str(symbol),
            )
            for symbol, value in self.substitutions.items()
        }
        object.__setattr__(self, "substitutions", normalized)

    def to_dict(self) -> dict[str, str | dict[str, str]]:
        """Serialize the scenario.

        Returns:
            dict[str, str | dict[str, str]]: JSON-friendly label and
                substitution mapping.
        """
        substitutions = cast(dict[sp.Symbol, sp.Expr], self.substitutions)
        return {
            "label": self.label,
            "substitutions": {
                str(symbol): str(value) for symbol, value in substitutions.items()
            },
        }


@dataclass(frozen=True)
class FTQCResourceScenarioRow:
    """Describe selected resource values under one scenario.

    Args:
        label (str): Reader-facing scenario label.
        values (dict[str | FTQCResourceQuantity, sp.Expr | int | float]):
            Scenario-evaluated resource values keyed by canonical quantity.
        unresolved_symbols (tuple[str, ...]): Remaining free-symbol names after
            substitutions are applied. Defaults to an empty tuple.

    Raises:
        TypeError: If any value cannot be converted to a SymPy expression.
        ValueError: If ``label`` is empty or a resource key is unknown.

    Example:
        >>> row = FTQCResourceScenarioRow(
        ...     "toy",
        ...     {"runtime_seconds": 10},
        ... )
        >>> row.unresolved_symbols
        ()
    """

    label: str
    values: dict[str | FTQCResourceQuantity, sp.Expr | int | float]
    unresolved_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize scenario-row fields after dataclass construction.

        Raises:
            TypeError: If any value cannot be converted to a SymPy expression.
            ValueError: If ``label`` is empty or a resource key is unknown.
        """
        if not self.label:
            raise ValueError("label must not be empty.")
        normalized_values = {
            _normalize_resource_quantity(quantity): _sympify_resource_expr(
                value,
                str(quantity),
            )
            for quantity, value in self.values.items()
        }
        object.__setattr__(self, "values", normalized_values)
        object.__setattr__(
            self,
            "unresolved_symbols",
            tuple(self.unresolved_symbols),
        )

    @property
    def is_fully_resolved(self) -> bool:
        """Return whether all selected values are concrete under the scenario.

        Returns:
            bool: True when no free symbols remain.
        """
        return not self.unresolved_symbols

    def to_dict(self) -> dict[str, str | bool | list[str] | dict[str, str]]:
        """Serialize the scenario row.

        Returns:
            dict[str, str | bool | list[str] | dict[str, str]]: JSON-friendly
                scenario values and unresolved-symbol metadata.
        """
        values = cast(dict[FTQCResourceQuantity, sp.Expr], self.values)
        return {
            "label": self.label,
            "values": {
                quantity.value: str(value) for quantity, value in values.items()
            },
            "is_fully_resolved": self.is_fully_resolved,
            "unresolved_symbols": list(self.unresolved_symbols),
        }


@dataclass(frozen=True)
class FTQCResourceScenarioReport:
    """Package scenario-evaluated FTQC resource values.

    Args:
        title (str): Reader-facing report title.
        quantities (tuple[str | FTQCResourceQuantity, ...]): Quantities
            evaluated in each scenario.
        scenarios (tuple[FTQCResourceScenario, ...]): Scenario definitions in
            input order.
        rows (tuple[FTQCResourceScenarioRow, ...]): Evaluated rows in scenario
            order.

    Raises:
        TypeError: If scenarios or rows contain incorrect item types.
        ValueError: If quantities, scenarios, or rows are empty, a quantity key
            is unknown, a row omits a selected quantity, or row count does not
            match scenario count.

    Example:
        >>> scenario = FTQCResourceScenario("toy", {"x": 2})
        >>> report = FTQCResourceScenarioReport(
        ...     title="Toy scenarios",
        ...     quantities=(FTQCResourceQuantity.RUNTIME_SECONDS,),
        ...     scenarios=(scenario,),
        ...     rows=(FTQCResourceScenarioRow("toy", {"runtime_seconds": 2}),),
        ... )
        >>> report.rows[0].is_fully_resolved
        True
    """

    title: str
    quantities: tuple[str | FTQCResourceQuantity, ...]
    scenarios: tuple[FTQCResourceScenario, ...]
    rows: tuple[FTQCResourceScenarioRow, ...]

    def __post_init__(self) -> None:
        """Validate scenario-report fields after dataclass construction.

        Raises:
            TypeError: If scenarios or rows contain incorrect item types.
            ValueError: If quantities, scenarios, or rows are empty, a
                quantity key is unknown, a row omits a selected quantity, or
                row count does not match scenario count.
        """
        if not self.quantities:
            raise ValueError("quantities must not be empty.")
        if not self.scenarios:
            raise ValueError("scenarios must not be empty.")
        if not self.rows:
            raise ValueError("rows must not be empty.")
        if len(self.rows) != len(self.scenarios):
            raise ValueError("rows must match scenarios one-to-one.")
        if not all(
            isinstance(scenario, FTQCResourceScenario) for scenario in self.scenarios
        ):
            raise TypeError("scenarios must contain only FTQCResourceScenario items.")
        if not all(isinstance(row, FTQCResourceScenarioRow) for row in self.rows):
            raise TypeError("rows must contain only FTQCResourceScenarioRow items.")
        normalized_quantities = tuple(
            _normalize_resource_quantity(quantity) for quantity in self.quantities
        )
        for row in self.rows:
            values = cast(dict[FTQCResourceQuantity, sp.Expr], row.values)
            missing = [
                quantity.value
                for quantity in normalized_quantities
                if quantity not in values
            ]
            if missing:
                raise ValueError(
                    f"row {row.label!r} is missing selected scenario quantities: "
                    + ", ".join(missing)
                    + "."
                )
        object.__setattr__(self, "quantities", normalized_quantities)
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "rows", tuple(self.rows))

    def to_row_table(self) -> list[dict[str, str | bool]]:
        """Return scenario rows as a flat table.

        Returns:
            list[dict[str, str | bool]]: Rows with one column per selected
                quantity plus resolution metadata.
        """
        rows = []
        quantities = cast(tuple[FTQCResourceQuantity, ...], self.quantities)
        for row in self.rows:
            values = cast(dict[FTQCResourceQuantity, sp.Expr], row.values)
            serialized: dict[str, str | bool] = {
                "label": row.label,
                "is_fully_resolved": row.is_fully_resolved,
                "unresolved_symbols": ", ".join(row.unresolved_symbols),
            }
            for quantity in quantities:
                serialized[quantity.value] = str(values[quantity])
            rows.append(serialized)
        return rows

    def to_dict(
        self,
    ) -> dict[
        str,
        str
        | list[str]
        | list[dict[str, str | dict[str, str]]]
        | list[dict[str, str | bool]]
        | list[dict[str, str | bool | list[str] | dict[str, str]]]
        | dict[str, int],
    ]:
        """Serialize report metadata and evaluated scenario rows.

        Returns:
            dict[str, str | list[str] | list[dict[str, str | dict[str, str]]] |
                list[dict[str, str | bool]] |
                list[dict[str, str | bool | list[str] | dict[str, str]]] |
                dict[str, int]]: JSON-friendly scenario report.
        """
        unresolved_rows = [
            row.to_dict() for row in self.rows if not row.is_fully_resolved
        ]
        quantities = cast(tuple[FTQCResourceQuantity, ...], self.quantities)
        return {
            "title": self.title,
            "quantities": [quantity.value for quantity in quantities],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "rows": self.to_row_table(),
            "unresolved": unresolved_rows,
            "counts": {
                "resolved": len(self.rows) - len(unresolved_rows),
                "unresolved": len(unresolved_rows),
            },
        }


FTQCResourceReportLike: TypeAlias = (
    FTQCResearchSignalCoverageReport
    | FTQCResourceBudgetReport
    | FTQCResourceComparisonReport
    | FTQCResourceDriverReport
    | FTQCResourceParetoReport
    | FTQCResourceScenarioReport
)


@dataclass(frozen=True)
class FTQCResourceReportSnapshot:
    """Package one FTQC report in a stable machine-readable envelope.

    Args:
        kind (str | FTQCResourceReportKind): Standard report kind.
        title (str): Reader-facing report title.
        payload (dict[str, Any]): JSON-friendly report payload, usually from
            the report's ``to_dict()`` method.
        row_count (int): Number of primary rows represented by the report.
        counts (dict[str, int]): Grouped count metadata extracted from the
            payload. Defaults to an empty mapping.

    Raises:
        ValueError: If ``kind`` is unknown, ``title`` is empty, ``row_count``
            is negative, or any count value is negative.

    Example:
        >>> snapshot = FTQCResourceReportSnapshot(
        ...     kind="scenario",
        ...     title="Toy",
        ...     payload={"title": "Toy", "rows": []},
        ...     row_count=0,
        ... )
        >>> snapshot.to_dict()["kind"]
        'scenario'
    """

    kind: str | FTQCResourceReportKind
    title: str
    payload: dict[str, Any]
    row_count: int
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate snapshot fields after construction.

        Raises:
            ValueError: If ``kind`` is unknown, ``title`` is empty,
                ``row_count`` is negative, or any count value is negative.
        """
        normalized_kind = FTQCResourceReportKind(self.kind)
        if not self.title:
            raise ValueError("title must not be empty.")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative.")
        counts = {} if self.counts is None else dict(self.counts)
        negative_counts = [name for name, count in counts.items() if count < 0]
        if negative_counts:
            raise ValueError(
                "counts must be non-negative: " + ", ".join(negative_counts) + "."
            )
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "counts", counts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report snapshot.

        Returns:
            dict[str, Any]: JSON-friendly snapshot envelope containing kind,
                title, row count, grouped counts, and payload.
        """
        kind = cast(FTQCResourceReportKind, self.kind)
        return {
            "kind": kind.value,
            "title": self.title,
            "row_count": self.row_count,
            "counts": dict(self.counts or {}),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class FTQCResourceReportBundle:
    """Group FTQC report snapshots for one review artifact.

    Args:
        title (str): Reader-facing bundle title.
        snapshots (tuple[FTQCResourceReportSnapshot, ...]): Snapshots in the
            order they should be reviewed.

    Raises:
        TypeError: If ``snapshots`` contains non-snapshot items.
        ValueError: If ``title`` or ``snapshots`` is empty.

    Example:
        >>> snapshot = FTQCResourceReportSnapshot(
        ...     "comparison",
        ...     "Toy",
        ...     {"title": "Toy", "rows": [{"quantity": "runtime_seconds"}]},
        ...     1,
        ... )
        >>> bundle = FTQCResourceReportBundle("Review", (snapshot,))
        >>> bundle.to_dict()["counts"]["rows"]
        1
        >>> bundle.to_row_table()[0]["report_kind"]
        'comparison'
        >>> bundle.to_manifest()["counts_by_kind"]
        {'comparison': 1}
    """

    title: str
    snapshots: tuple[FTQCResourceReportSnapshot, ...]

    def __post_init__(self) -> None:
        """Validate bundle fields after dataclass construction.

        Raises:
            TypeError: If ``snapshots`` contains non-snapshot items.
            ValueError: If ``title`` or ``snapshots`` is empty.
        """
        if not self.title:
            raise ValueError("title must not be empty.")
        if not self.snapshots:
            raise ValueError("snapshots must not be empty.")
        if not all(
            isinstance(snapshot, FTQCResourceReportSnapshot)
            for snapshot in self.snapshots
        ):
            raise TypeError(
                "snapshots must contain only FTQCResourceReportSnapshot items."
            )
        object.__setattr__(self, "snapshots", tuple(self.snapshots))

    def counts_by_kind(self) -> dict[str, int]:
        """Return snapshot counts grouped by report kind.

        Returns:
            dict[str, int]: Number of snapshots per report kind.
        """
        counts: dict[str, int] = {}
        for snapshot in self.snapshots:
            kind = cast(FTQCResourceReportKind, snapshot.kind)
            counts[kind.value] = counts.get(kind.value, 0) + 1
        return counts

    def to_row_table(self) -> list[dict[str, Any]]:
        """Return all snapshot rows as one review table.

        Returns:
            list[dict[str, Any]]: Flattened rows. Each row preserves the
                original report row fields and adds ``snapshot_index``,
                ``report_kind``, ``report_title``, and ``row_index`` columns.
        """
        rows: list[dict[str, Any]] = []
        for snapshot_index, snapshot in enumerate(self.snapshots):
            kind = cast(FTQCResourceReportKind, snapshot.kind)
            payload_rows = _extract_report_rows(snapshot.payload)
            for row_index, payload_row in enumerate(payload_rows):
                rows.append(
                    {
                        **payload_row,
                        "snapshot_index": snapshot_index,
                        "report_kind": kind.value,
                        "report_title": snapshot.title,
                        "row_index": row_index,
                    }
                )
        return rows

    def to_manifest(self) -> dict[str, Any]:
        """Return stable bundle metadata for review tooling.

        Returns:
            dict[str, Any]: JSON-friendly bundle manifest containing title,
                aggregate counts, counts grouped by report kind, and compact
                snapshot metadata without the full report payloads.
        """
        return {
            "title": self.title,
            "counts": {
                "snapshots": len(self.snapshots),
                "rows": sum(snapshot.row_count for snapshot in self.snapshots),
            },
            "counts_by_kind": self.counts_by_kind(),
            "snapshots": [
                {
                    "index": index,
                    "kind": cast(FTQCResourceReportKind, snapshot.kind).value,
                    "title": snapshot.title,
                    "row_count": snapshot.row_count,
                    "counts": dict(snapshot.counts or {}),
                }
                for index, snapshot in enumerate(self.snapshots)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report bundle.

        Returns:
            dict[str, Any]: JSON-friendly bundle containing snapshot payloads,
                flattened rows, total counts, and counts grouped by report
                kind.
        """
        manifest = self.to_manifest()
        return {
            "title": self.title,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "rows": self.to_row_table(),
            "manifest": manifest,
            "counts": manifest["counts"],
            "counts_by_kind": manifest["counts_by_kind"],
        }


FTQC_RESOURCE_QUANTITY_SPECS: tuple[FTQCResourceQuantitySpec, ...] = (
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.N_SPIN_ORBITALS,
        "Spin orbitals",
        "spin orbitals",
        FTQCResourceCategory.PROBLEM,
        "Encoded active-space size, often equal to the qubit-register size.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.N_PAULI_TERMS,
        "Pauli terms",
        "terms",
        FTQCResourceCategory.PROBLEM,
        "Number of non-identity Pauli strings in the Hamiltonian model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LAMBDA_NORM,
        "Hamiltonian normalization",
        "energy",
        FTQCResourceCategory.PROBLEM,
        "LCU normalization that controls QPE walk or evolution calls.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.MAX_LOCALITY,
        "Maximum locality",
        "Pauli factors",
        FTQCResourceCategory.PROBLEM,
        "Maximum number of non-identity Pauli factors in one term.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.TARGET_PRECISION,
        "Target precision",
        "energy",
        FTQCResourceCategory.ALGORITHM,
        "Requested QPE energy or phase precision that controls iteration count.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.TRUNCATION_ERROR,
        "Truncation error",
        "energy",
        FTQCResourceCategory.PROBLEM,
        "Representation-level approximation error budget before QPE sampling.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.SYSTEM_QUBITS,
        "System qubits",
        "qubits",
        FTQCResourceCategory.PROBLEM,
        "Logical system-register qubits encoded by a block-encoding model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS,
        "Block-encoding ancilla qubits",
        "qubits",
        FTQCResourceCategory.ALGORITHM,
        "Ancilla and workspace qubits required by a block encoding before QPE readout.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.QPE_REGISTER_QUBITS,
        "QPE register qubits",
        "qubits",
        FTQCResourceCategory.ALGORITHM,
        "Phase-readout qubits used by an explicit QPE circuit.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY,
        "State-preparation success probability",
        "probability",
        FTQCResourceCategory.ALGORITHM,
        "Probability that one prepared state lands in the target QPE subspace.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.QPE_REPETITIONS,
        "QPE repetitions",
        "runs",
        FTQCResourceCategory.ALGORITHM,
        "Expected repeated QPE runs needed for one successful sample.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.STATE_PREPARATION_TOFFOLI,
        "State-preparation Toffoli overhead",
        "Toffoli gates / run",
        FTQCResourceCategory.ALGORITHM,
        "Toffoli overhead of one state-preparation or filtering attempt.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.STATE_PREPARATION_T_GATES,
        "State-preparation T overhead",
        "T gates / run",
        FTQCResourceCategory.ALGORITHM,
        "T-gate overhead of one state-preparation or filtering attempt.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.STATE_PREPARATION_LOGICAL_DEPTH,
        "State-preparation depth overhead",
        "logical layers / run",
        FTQCResourceCategory.ALGORITHM,
        "Logical-depth overhead of one state-preparation or filtering attempt.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PREPARE_COST_TOFFOLI,
        "PREPARE cost",
        "Toffoli gates / call",
        FTQCResourceCategory.ALGORITHM,
        "Toffoli cost of one PREPARE or inverse PREPARE subroutine call.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.SELECT_COST_TOFFOLI,
        "SELECT cost",
        "Toffoli gates / call",
        FTQCResourceCategory.ALGORITHM,
        "Toffoli cost of one SELECT or oracle subroutine call.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.REFLECTION_COST_TOFFOLI,
        "Reflection cost",
        "Toffoli gates / call",
        FTQCResourceCategory.ALGORITHM,
        "Toffoli cost of the reflection subroutine used by one qubitized walk.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.WALK_COST_TOFFOLI,
        "Walk cost",
        "Toffoli gates per walk",
        FTQCResourceCategory.ALGORITHM,
        "Toffoli cost of one qubitized walk operator call.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.QPE_ITERATIONS,
        "QPE iterations",
        "iterations",
        FTQCResourceCategory.ALGORITHM,
        "Number of QPE walk or time-evolution calls.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_QUBITS,
        "Logical qubits",
        "logical qubits",
        FTQCResourceCategory.LOGICAL,
        "Logical data, ancilla, and algorithm workspace qubits.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_DEPTH,
        "Logical depth",
        "logical layers",
        FTQCResourceCategory.LOGICAL,
        "Logical circuit-depth proxy after algorithmic repetition factors.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
        "Logical space-time volume",
        "logical qubit-layers",
        FTQCResourceCategory.LOGICAL,
        "Product of logical qubits and logical-depth proxy.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.TOFFOLI_GATES,
        "Toffoli gates",
        "Toffoli gates",
        FTQCResourceCategory.LOGICAL,
        "Toffoli or Toffoli-equivalent non-Clifford gate count.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.T_GATES,
        "T gates",
        "T gates",
        FTQCResourceCategory.LOGICAL,
        "T-gate or T-equivalent count when it differs from Toffoli count.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.CLIFFORD_GATES,
        "Clifford gates",
        "Clifford gates",
        FTQCResourceCategory.LOGICAL,
        "Clifford gate count when an estimator has a reliable model for it.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_QUBITS,
        "Physical qubits",
        "physical qubits",
        FTQCResourceCategory.PHYSICAL,
        "Physical qubits after logical-qubit overhead and factory allocation.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.RUNTIME_SECONDS,
        "Runtime",
        "seconds",
        FTQCResourceCategory.PHYSICAL,
        "Wall-clock runtime proxy under the selected architecture model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        "Physical qubit-seconds",
        "physical qubit-seconds",
        FTQCResourceCategory.PHYSICAL,
        "Product of physical qubits and wall-clock runtime proxy.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_ERROR_RATE,
        "Logical error rate",
        "probability",
        FTQCResourceCategory.PHYSICAL,
        "Logical error-rate proxy under an architecture sizing model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.TARGET_LOGICAL_FAILURE_PROBABILITY,
        "Target logical failure probability",
        "probability",
        FTQCResourceCategory.ARCHITECTURE,
        "Total logical failure budget allocated to an FTQC estimate.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_OPERATION_BUDGET,
        "Logical operation budget",
        "operations",
        FTQCResourceCategory.ARCHITECTURE,
        "Number of logical operations or cycles sharing the failure budget.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_ERROR_RATE,
        "Physical error rate",
        "probability",
        FTQCResourceCategory.ARCHITECTURE,
        "Physical error rate used by an architecture sizing model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.THRESHOLD_ERROR_RATE,
        "Threshold error rate",
        "probability",
        FTQCResourceCategory.ARCHITECTURE,
        "Threshold error rate used by an architecture sizing model.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL,
        "Physical qubits per logical qubit",
        "physical qubits / logical qubit",
        FTQCResourceCategory.ARCHITECTURE,
        "Physical overhead used to lift logical qubits to hardware qubits.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS,
        "Logical cycle time",
        "seconds",
        FTQCResourceCategory.ARCHITECTURE,
        "Duration of one logical layer or logical error-correction cycle.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.FACTORY_QUBITS,
        "Factory qubits",
        "physical qubits",
        FTQCResourceCategory.ARCHITECTURE,
        "Physical qubits reserved for magic-state factories or equivalents.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND,
        "Toffoli throughput",
        "Toffoli gates / second",
        FTQCResourceCategory.ARCHITECTURE,
        "Sustainable non-Clifford throughput from factories or hardware.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.CODE_DISTANCE,
        "Code distance",
        "distance",
        FTQCResourceCategory.ARCHITECTURE,
        "Surface-code distance used to lift logical resources to hardware.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
        "Physical cycle time",
        "seconds",
        FTQCResourceCategory.ARCHITECTURE,
        "Duration of one physical error-correction cycle.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL_FACTOR,
        "Patch qubit factor",
        "physical qubits / distance^2",
        FTQCResourceCategory.ARCHITECTURE,
        "Constant multiplying code_distance^2 for one logical qubit patch.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.LOGICAL_CYCLE_FACTOR,
        "Logical cycle factor",
        "physical cycles / distance",
        FTQCResourceCategory.ARCHITECTURE,
        "Constant multiplying code distance to model one logical cycle.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.FACTORY_COUNT,
        "Factory count",
        "factories",
        FTQCResourceCategory.ARCHITECTURE,
        "Number of parallel non-Clifford factories.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.PHYSICAL_QUBITS_PER_FACTORY,
        "Factory size",
        "physical qubits / factory",
        FTQCResourceCategory.ARCHITECTURE,
        "Physical qubits reserved for one non-Clifford factory.",
    ),
    FTQCResourceQuantitySpec(
        FTQCResourceQuantity.FACTORY_CYCLES_PER_TOFFOLI,
        "Factory cycles per Toffoli",
        "logical cycles / Toffoli",
        FTQCResourceCategory.ARCHITECTURE,
        "Logical cycles required by one factory to produce a Toffoli resource.",
    ),
)

_SPECS_BY_QUANTITY = {spec.quantity: spec for spec in FTQC_RESOURCE_QUANTITY_SPECS}


_FTQC_ADDITIVE_RESOURCE_QUANTITIES = {
    FTQCResourceQuantity.QPE_REPETITIONS,
    FTQCResourceQuantity.STATE_PREPARATION_TOFFOLI,
    FTQCResourceQuantity.STATE_PREPARATION_T_GATES,
    FTQCResourceQuantity.STATE_PREPARATION_LOGICAL_DEPTH,
    FTQCResourceQuantity.QPE_ITERATIONS,
    FTQCResourceQuantity.LOGICAL_DEPTH,
    FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
    FTQCResourceQuantity.TOFFOLI_GATES,
    FTQCResourceQuantity.T_GATES,
    FTQCResourceQuantity.CLIFFORD_GATES,
    FTQCResourceQuantity.RUNTIME_SECONDS,
    FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
}


_FTQC_PEAK_RESOURCE_QUANTITIES = {
    FTQCResourceQuantity.SYSTEM_QUBITS,
    FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS,
    FTQCResourceQuantity.QPE_REGISTER_QUBITS,
    FTQCResourceQuantity.LOGICAL_QUBITS,
    FTQCResourceQuantity.PHYSICAL_QUBITS,
    FTQCResourceQuantity.FACTORY_QUBITS,
}


FTQC_RESOURCE_PROFILE_SPECS: tuple[FTQCResourceProfileSpec, ...] = (
    FTQCResourceProfileSpec(
        profile=FTQCResourceProfile.CHEMISTRY_QPE,
        label="Chemistry QPE review",
        description=(
            "Compare Hamiltonian normalization, phase-estimation work, "
            "non-Clifford counts, logical footprint, and physical space-time "
            "costs for chemistry QPE variants."
        ),
        quantities=(
            FTQCResourceQuantity.LAMBDA_NORM,
            FTQCResourceQuantity.TARGET_PRECISION,
            FTQCResourceQuantity.TRUNCATION_ERROR,
            FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY,
            FTQCResourceQuantity.QPE_REPETITIONS,
            FTQCResourceQuantity.STATE_PREPARATION_TOFFOLI,
            FTQCResourceQuantity.STATE_PREPARATION_T_GATES,
            FTQCResourceQuantity.STATE_PREPARATION_LOGICAL_DEPTH,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
            FTQCResourceQuantity.T_GATES,
            FTQCResourceQuantity.LOGICAL_QUBITS,
            FTQCResourceQuantity.LOGICAL_DEPTH,
            FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
    ),
    FTQCResourceProfileSpec(
        profile=FTQCResourceProfile.BLOCK_ENCODING,
        label="Block-encoding subroutines",
        description=(
            "Inspect system, ancilla, PREPARE, SELECT, reflection, walk, and "
            "QPE readout quantities before lowering a loader to concrete IR."
        ),
        quantities=(
            FTQCResourceQuantity.SYSTEM_QUBITS,
            FTQCResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS,
            FTQCResourceQuantity.QPE_REGISTER_QUBITS,
            FTQCResourceQuantity.PREPARE_COST_TOFFOLI,
            FTQCResourceQuantity.SELECT_COST_TOFFOLI,
            FTQCResourceQuantity.REFLECTION_COST_TOFFOLI,
            FTQCResourceQuantity.WALK_COST_TOFFOLI,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
            FTQCResourceQuantity.LOGICAL_QUBITS,
        ),
    ),
    FTQCResourceProfileSpec(
        profile=FTQCResourceProfile.SPACETIME,
        label="Space-time footprint",
        description=(
            "Review logical qubit-layers and physical qubit-seconds alongside "
            "their qubit, depth, and runtime factors."
        ),
        quantities=(
            FTQCResourceQuantity.LOGICAL_QUBITS,
            FTQCResourceQuantity.LOGICAL_DEPTH,
            FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
    ),
    FTQCResourceProfileSpec(
        profile=FTQCResourceProfile.ERROR_BUDGET,
        label="Surface-code error budget",
        description=(
            "Audit physical error assumptions, target logical failure "
            "probability, operation budget, selected code distance, and "
            "resulting logical error rate."
        ),
        quantities=(
            FTQCResourceQuantity.PHYSICAL_ERROR_RATE,
            FTQCResourceQuantity.THRESHOLD_ERROR_RATE,
            FTQCResourceQuantity.TARGET_LOGICAL_FAILURE_PROBABILITY,
            FTQCResourceQuantity.LOGICAL_OPERATION_BUDGET,
            FTQCResourceQuantity.CODE_DISTANCE,
            FTQCResourceQuantity.LOGICAL_ERROR_RATE,
        ),
    ),
    FTQCResourceProfileSpec(
        profile=FTQCResourceProfile.ARCHITECTURE,
        label="Architecture lift",
        description=(
            "Audit the patch overhead, cycle time, and factory assumptions "
            "used to lift logical work after a code distance is selected."
        ),
        quantities=(
            FTQCResourceQuantity.CODE_DISTANCE,
            FTQCResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL_FACTOR,
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL,
            FTQCResourceQuantity.LOGICAL_CYCLE_FACTOR,
            FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS,
            FTQCResourceQuantity.FACTORY_COUNT,
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_FACTORY,
            FTQCResourceQuantity.FACTORY_QUBITS,
            FTQCResourceQuantity.FACTORY_CYCLES_PER_TOFFOLI,
            FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND,
        ),
    ),
)

_PROFILE_SPECS_BY_PROFILE = {spec.profile: spec for spec in FTQC_RESOURCE_PROFILE_SPECS}


FTQC_RESEARCH_SIGNALS = (
    FTQCResearchSignal(
        reference_key="arXiv:1610.06546",
        title="Hamiltonian simulation by qubitization",
        url="https://arxiv.org/abs/1610.06546",
        cost_driver=(
            "Qubitized walks compose block-encoding PREPARE, SELECT, and "
            "reflection primitives before QPE repeats the walk."
        ),
        quantities=(
            FTQCResourceQuantity.LAMBDA_NORM,
            FTQCResourceQuantity.PREPARE_COST_TOFFOLI,
            FTQCResourceQuantity.SELECT_COST_TOFFOLI,
            FTQCResourceQuantity.REFLECTION_COST_TOFFOLI,
            FTQCResourceQuantity.WALK_COST_TOFFOLI,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
        ),
        design_note=(
            "Represent the block encoding as algorithmic metadata until a "
            "loader implementation is ready to emit concrete circuits."
        ),
        profiles=(
            FTQCResourceProfile.BLOCK_ENCODING,
            FTQCResourceProfile.CHEMISTRY_QPE,
        ),
    ),
    FTQCResearchSignal(
        reference_key="arXiv:2403.03502",
        title=(
            "Symmetry-compressed double factorization for fault-tolerant "
            "chemistry simulation"
        ),
        url="https://arxiv.org/abs/2403.03502",
        cost_driver=(
            "Symmetry-compressed factorization reduces Hamiltonian normalization "
            "and Toffoli-dominated qubitized QPE cost."
        ),
        quantities=(
            FTQCResourceQuantity.LAMBDA_NORM,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.WALK_COST_TOFFOLI,
            FTQCResourceQuantity.TOFFOLI_GATES,
            FTQCResourceQuantity.LOGICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
        design_note=(
            "Keep factorization ranks, normalization, and walk cost on the "
            "chemistry model rather than baking them into IR gates."
        ),
        profiles=(
            FTQCResourceProfile.BLOCK_ENCODING,
            FTQCResourceProfile.CHEMISTRY_QPE,
            FTQCResourceProfile.SPACETIME,
        ),
    ),
    FTQCResearchSignal(
        reference_key="arXiv:2412.01338",
        title=(
            "Symmetry shifts with tensor factorizations for cost-efficient "
            "electronic Hamiltonians"
        ),
        url="https://arxiv.org/abs/2412.01338",
        cost_driver=(
            "Joint symmetry-shift and tensor-factorization optimization reduces "
            "the block-encoding scaling constant."
        ),
        quantities=(
            FTQCResourceQuantity.LAMBDA_NORM,
            FTQCResourceQuantity.TRUNCATION_ERROR,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.TOFFOLI_GATES,
        ),
        design_note=(
            "Model symmetry-shift effects as representation metadata with an "
            "explicit accuracy budget."
        ),
        profiles=(FTQCResourceProfile.CHEMISTRY_QPE,),
    ),
    FTQCResearchSignal(
        reference_key="arXiv:2601.08533",
        title="Symmetry-adapted state preparation for FTQC chemistry",
        url="https://arxiv.org/abs/2601.08533",
        cost_driver=(
            "Symmetry filtering can increase QPE success probability while "
            "adding a smaller state-preparation overhead."
        ),
        quantities=(
            FTQCResourceQuantity.STATE_PREPARATION_SUCCESS_PROBABILITY,
            FTQCResourceQuantity.QPE_REPETITIONS,
            FTQCResourceQuantity.STATE_PREPARATION_T_GATES,
            FTQCResourceQuantity.STATE_PREPARATION_LOGICAL_DEPTH,
            FTQCResourceQuantity.T_GATES,
            FTQCResourceQuantity.RUNTIME_SECONDS,
        ),
        design_note=(
            "Represent overlap and filtering success as an algorithmic budget "
            "that scales expected QPE work."
        ),
        profiles=(
            FTQCResourceProfile.CHEMISTRY_QPE,
            FTQCResourceProfile.SPACETIME,
        ),
    ),
    FTQCResearchSignal(
        reference_key="arXiv:2603.22778",
        title="Chemically accurate QPE in the early fault-tolerant regime",
        url="https://arxiv.org/abs/2603.22778",
        cost_driver=(
            "Single-ancilla Trotter QPE and unitary-weight concentration trade "
            "block-encoding style costs for T gates, depth, and physical-qubit "
            "constraints."
        ),
        quantities=(
            FTQCResourceQuantity.LAMBDA_NORM,
            FTQCResourceQuantity.QPE_ITERATIONS,
            FTQCResourceQuantity.T_GATES,
            FTQCResourceQuantity.LOGICAL_DEPTH,
            FTQCResourceQuantity.LOGICAL_SPACETIME_VOLUME,
            FTQCResourceQuantity.PHYSICAL_QUBITS,
            FTQCResourceQuantity.RUNTIME_SECONDS,
            FTQCResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
        design_note=(
            "Track T gates, logical depth, and architecture knobs alongside "
            "Toffoli-native qubitization estimates."
        ),
        profiles=(
            FTQCResourceProfile.CHEMISTRY_QPE,
            FTQCResourceProfile.SPACETIME,
            FTQCResourceProfile.ARCHITECTURE,
        ),
    ),
)


def iter_ftqc_research_signals() -> tuple[FTQCResearchSignal, ...]:
    """Return research signals that motivate Qamomile FTQC quantities.

    Returns:
        tuple[FTQCResearchSignal, ...]: Survey entries mapping research
            directions to canonical FTQC quantity keys.
    """
    return FTQC_RESEARCH_SIGNALS


def describe_ftqc_research_signal(reference_key: str) -> FTQCResearchSignal:
    """Return metadata for one FTQC research signal.

    Args:
        reference_key (str): Stable research-signal key, such as an arXiv
            identifier.

    Returns:
        FTQCResearchSignal: Research signal metadata, including canonical
            quantities and recommended review profiles.

    Raises:
        ValueError: If ``reference_key`` is not known.
    """
    for signal in FTQC_RESEARCH_SIGNALS:
        if signal.reference_key == reference_key:
            return signal
    available = ", ".join(signal.reference_key for signal in FTQC_RESEARCH_SIGNALS)
    raise ValueError(
        f"Unknown FTQC research signal {reference_key!r}. Available keys: {available}."
    )


def audit_ftqc_research_signal_coverage(
    reference_key: str,
    estimate: SupportsFTQCResourceValues,
) -> FTQCResearchSignalCoverage:
    """Audit which research-signal quantities an estimate exposes.

    This helper is a design-time check: it answers whether a symbolic estimate
    has the canonical quantities needed to review a paper's resource claim
    before a comparison report is built.

    Args:
        reference_key (str): Research-signal key used to select quantities.
        estimate (SupportsFTQCResourceValues): Estimate, model, or summary
            exposing ``resource_values()``.

    Returns:
        FTQCResearchSignalCoverage: Available and missing quantities for the
            requested research signal.

    Raises:
        ValueError: If ``reference_key`` is unknown.
    """
    signal = describe_ftqc_research_signal(reference_key)
    values = estimate.resource_values()
    available = tuple(quantity for quantity in signal.quantities if quantity in values)
    missing = tuple(
        quantity for quantity in signal.quantities if quantity not in values
    )
    return FTQCResearchSignalCoverage(
        reference_key=signal.reference_key,
        title=signal.title,
        available=available,
        missing=missing,
        total=len(signal.quantities),
    )


def audit_ftqc_research_signal_catalog(
    estimate: SupportsFTQCResourceValues,
    *,
    reference_keys: tuple[str, ...] | None = None,
    title: str = "FTQC research signal coverage",
    estimate_label: str = "estimate",
) -> FTQCResearchSignalCoverageReport:
    """Audit one estimate against a catalog of FTQC research signals.

    Args:
        estimate (SupportsFTQCResourceValues): Estimate, model, or summary
            exposing ``resource_values()``.
        reference_keys (tuple[str, ...] | None): Optional subset of research
            signal keys to audit. Defaults to None, auditing the full catalog.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC research signal coverage"``.
        estimate_label (str): Label for the audited estimate. Defaults to
            ``"estimate"``.

    Returns:
        FTQCResearchSignalCoverageReport: Coverage report for the requested
            research signals.

    Raises:
        ValueError: If any requested research-signal key is unknown or if the
            requested key list is empty.
    """
    if reference_keys is None:
        keys = tuple(signal.reference_key for signal in FTQC_RESEARCH_SIGNALS)
    elif not reference_keys:
        raise ValueError("reference_keys must not be empty.")
    else:
        keys = reference_keys

    return FTQCResearchSignalCoverageReport(
        title=title,
        estimate_label=estimate_label,
        coverages=tuple(
            audit_ftqc_research_signal_coverage(reference_key, estimate)
            for reference_key in keys
        ),
    )


def iter_ftqc_resource_quantity_specs() -> tuple[FTQCResourceQuantitySpec, ...]:
    """Return the canonical FTQC resource quantity specifications.

    Returns:
        tuple[FTQCResourceQuantitySpec, ...]: Quantity specifications in a
            reader-friendly order from problem inputs to physical outputs.
    """
    return FTQC_RESOURCE_QUANTITY_SPECS


def iter_ftqc_resource_profile_specs() -> tuple[FTQCResourceProfileSpec, ...]:
    """Return the standard FTQC resource review profile specifications.

    Returns:
        tuple[FTQCResourceProfileSpec, ...]: Profile specifications in a
            reader-friendly order from end-to-end review to focused audits.
    """
    return FTQC_RESOURCE_PROFILE_SPECS


def ftqc_resource_profile_quantities(
    profile: str | FTQCResourceProfile,
) -> tuple[FTQCResourceQuantity, ...]:
    """Return canonical quantities for one FTQC review profile.

    Args:
        profile (str | FTQCResourceProfile): Profile key or enum value.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Ordered canonical quantities for the
            requested profile.

    Raises:
        ValueError: If ``profile`` is not a known FTQC resource profile.

    Example:
        >>> ftqc_resource_profile_quantities("spacetime")[0]
        <FTQCResourceQuantity.LOGICAL_QUBITS: 'logical_qubits'>
    """
    normalized = _normalize_resource_profile(profile)
    return _PROFILE_SPECS_BY_PROFILE[normalized].quantities


def describe_ftqc_resource_quantity(
    quantity: str | FTQCResourceQuantity,
) -> FTQCResourceQuantitySpec:
    """Return metadata for one FTQC resource quantity.

    Args:
        quantity (str | FTQCResourceQuantity): Quantity key or enum value.

    Returns:
        FTQCResourceQuantitySpec: Metadata describing the quantity.

    Raises:
        ValueError: If ``quantity`` is not a known FTQC resource quantity.
    """
    normalized = _normalize_resource_quantity(quantity)
    return _SPECS_BY_QUANTITY[normalized]


def default_ftqc_resource_aggregation_rule(
    quantity: str | FTQCResourceQuantity,
) -> FTQCResourceAggregationRule:
    """Return the default subroutine-composition rule for one quantity.

    Args:
        quantity (str | FTQCResourceQuantity): Quantity key to inspect.

    Returns:
        FTQCResourceAggregationRule: Default aggregation rule. Counts, depth,
            runtime, and space-time volume add across sequential steps; qubit
            footprints use peak aggregation; problem and architecture
            metadata must remain consistent across steps.

    Raises:
        ValueError: If ``quantity`` is not a known FTQC resource quantity.

    Example:
        >>> default_ftqc_resource_aggregation_rule("logical_qubits")
        <FTQCResourceAggregationRule.PEAK: 'peak'>
    """
    normalized = _normalize_resource_quantity(quantity)
    if normalized in _FTQC_ADDITIVE_RESOURCE_QUANTITIES:
        return FTQCResourceAggregationRule.ADD
    if normalized in _FTQC_PEAK_RESOURCE_QUANTITIES:
        return FTQCResourceAggregationRule.PEAK
    return FTQCResourceAggregationRule.CONSISTENT


def compare_ftqc_resource_estimates(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    quantities: tuple[str | FTQCResourceQuantity, ...] | None = None,
    profile: str | FTQCResourceProfile | None = None,
) -> tuple[FTQCResourceComparisonRow, ...]:
    """Compare canonical FTQC quantities between two estimates.

    Args:
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            summary exposing ``resource_values()``.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            summary exposing ``resource_values()``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to compare before any profile quantities. Defaults to the
            intersection of quantities exposed by both inputs when ``profile``
            is None.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile whose quantities are appended after ``quantities`` with
            duplicates removed. Defaults to None.

    Returns:
        tuple[FTQCResourceComparisonRow, ...]: Comparison rows containing
            baseline values, candidate values, ratios, and fractional
            reductions.

    Raises:
        ValueError: If a requested quantity is missing from either input or if
            a baseline value is exactly zero.
    """
    baseline_values = baseline.resource_values()
    candidate_values = candidate.resource_values()
    selected = _select_comparison_quantities(
        baseline_values,
        candidate_values,
        quantities,
        profile,
    )

    rows = []
    for quantity in selected:
        baseline_value = sp.sympify(baseline_values[quantity])
        candidate_value = sp.sympify(candidate_values[quantity])
        if baseline_value.equals(0):
            raise ValueError(
                f"Cannot compare {quantity.value!r} against a zero baseline."
            )
        ratio = sp.simplify(candidate_value / baseline_value)
        reduction = sp.simplify(1 - ratio)
        spec = describe_ftqc_resource_quantity(quantity)
        rows.append(
            FTQCResourceComparisonRow(
                quantity=quantity,
                baseline=baseline_value,
                candidate=candidate_value,
                ratio=ratio,
                reduction=reduction,
                label=spec.label,
                unit=spec.unit,
                category=spec.category,
            )
        )
    return tuple(rows)


def summarize_ftqc_resource_comparison(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    quantities: tuple[str | FTQCResourceQuantity, ...] | None = None,
    profile: str | FTQCResourceProfile | None = None,
) -> FTQCResourceComparisonSummary:
    """Summarize FTQC resource changes between two estimates.

    This is a convenience wrapper over ``compare_ftqc_resource_estimates`` for
    reviews and reports that need to distinguish reductions, regressions,
    unchanged quantities, and symbolic quantities whose sign is undecidable.

    Args:
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            summary exposing ``resource_values()``.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            summary exposing ``resource_values()``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to compare before any profile quantities. Defaults to the
            intersection of quantities exposed by both inputs when ``profile``
            is None.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile whose quantities are appended after ``quantities`` with
            duplicates removed. Defaults to None.

    Returns:
        FTQCResourceComparisonSummary: Grouped comparison rows.

    Raises:
        ValueError: If a requested quantity is missing from either input or if
            a baseline value is exactly zero.
    """
    return FTQCResourceComparisonSummary.from_rows(
        compare_ftqc_resource_estimates(
            baseline,
            candidate,
            quantities=quantities,
            profile=profile,
        )
    )


def build_ftqc_resource_comparison_report(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    title: str = "FTQC resource comparison",
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    quantities: tuple[str | FTQCResourceQuantity, ...] | None = None,
    profile: str | FTQCResourceProfile | None = None,
) -> FTQCResourceComparisonReport:
    """Build a labeled FTQC resource comparison report.

    Args:
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            summary exposing ``resource_values()``.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            summary exposing ``resource_values()``.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC resource comparison"``.
        baseline_label (str): Label for the baseline estimate. Defaults to
            ``"baseline"``.
        candidate_label (str): Label for the candidate estimate. Defaults to
            ``"candidate"``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to compare before any profile quantities. Defaults to the
            intersection of quantities exposed by both inputs when ``profile``
            is None.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile whose quantities are appended after ``quantities`` with
            duplicates removed. Defaults to None.

    Returns:
        FTQCResourceComparisonReport: Report metadata and grouped comparison
            summary.

    Raises:
        ValueError: If a requested quantity is missing from either input, if a
            baseline value is exactly zero, or if ``profile`` is unknown.
    """
    summary = summarize_ftqc_resource_comparison(
        baseline,
        candidate,
        quantities=quantities,
        profile=profile,
    )
    normalized_profile = (
        None if profile is None else _normalize_resource_profile(profile)
    )
    return FTQCResourceComparisonReport(
        title=title,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        profile=normalized_profile,
        summary=summary,
    )


def build_ftqc_research_signal_report(
    reference_key: str,
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    title: str = "",
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    require_all_quantities: bool = False,
) -> FTQCResourceComparisonReport:
    """Build a comparison report scoped to one FTQC research signal.

    Research signals may list both problem-level drivers, such as
    ``lambda_norm``, and output resources, such as ``logical_depth``. Concrete
    estimates do not always expose every problem driver. By default this helper
    compares the signal quantities exposed by both inputs. Set
    ``require_all_quantities`` to true when auditing whether a model exposes
    the full signal contract.

    Args:
        reference_key (str): Research-signal key used to select quantities.
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            summary exposing ``resource_values()``.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            summary exposing ``resource_values()``.
        title (str): Reader-facing report title. Defaults to an empty string,
            which derives a title from the research signal.
        baseline_label (str): Label for the baseline estimate. Defaults to
            ``"baseline"``.
        candidate_label (str): Label for the candidate estimate. Defaults to
            ``"candidate"``.
        require_all_quantities (bool): Whether every research-signal quantity
            must be present on both inputs. Defaults to False.

    Returns:
        FTQCResourceComparisonReport: Report over the selected research-signal
            quantities.

    Raises:
        ValueError: If ``reference_key`` is unknown, no research-signal
            quantities are exposed by both inputs, a required quantity is
            missing, or a selected baseline value is zero.
    """
    signal = describe_ftqc_research_signal(reference_key)
    quantities = _select_research_signal_quantities(
        signal,
        baseline.resource_values(),
        candidate.resource_values(),
        require_all_quantities=require_all_quantities,
    )
    report_title = title or f"{signal.reference_key} resource signal review"
    return build_ftqc_resource_comparison_report(
        baseline,
        candidate,
        title=report_title,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        quantities=quantities,
    )


def build_ftqc_resource_driver_report(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    targets: tuple[str | FTQCResourceQuantity, ...],
    title: str = "FTQC resource driver report",
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    include_targets: bool = True,
) -> FTQCResourceDriverReport:
    """Build a comparison report scoped by formula dependencies.

    This helper starts from target output quantities, follows any
    ``FTQCResourceFormula.depends_on`` links exposed by either estimate, and
    compares the available dependency closure. It is useful for explaining
    which symbolic cost drivers changed when a recent FTQC paper reports a
    better physical-qubit or runtime estimate.

    Args:
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            plan exposing ``resource_values()`` and optionally formulas.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            plan exposing ``resource_values()`` and optionally formulas.
        targets (tuple[str | FTQCResourceQuantity, ...]): Output quantities to
            explain through formula dependencies.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC resource driver report"``.
        baseline_label (str): Label for the baseline estimate. Defaults to
            ``"baseline"``.
        candidate_label (str): Label for the candidate estimate. Defaults to
            ``"candidate"``.
        include_targets (bool): Whether target quantities themselves should
            appear in the comparison rows. Defaults to True.

    Returns:
        FTQCResourceDriverReport: Formula-scoped comparison report.

    Raises:
        ValueError: If ``targets`` is empty, a target quantity is unknown, no
            comparable driver quantities remain, or a selected baseline value
            is zero.
    """
    normalized_targets = _normalize_driver_targets(targets)
    quantities = _select_driver_report_quantities(
        baseline,
        candidate,
        targets=normalized_targets,
        include_targets=include_targets,
    )
    summary = summarize_ftqc_resource_comparison(
        baseline,
        candidate,
        quantities=quantities,
    )
    return FTQCResourceDriverReport(
        title=title,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        targets=normalized_targets,
        summary=summary,
    )


def build_ftqc_resource_pareto_report(
    candidates: tuple[tuple[str, SupportsFTQCResourceValues], ...],
    *,
    title: str = "FTQC resource Pareto frontier",
    quantities: tuple[str | FTQCResourceQuantity, ...] | None = None,
    profile: str | FTQCResourceProfile | None = None,
) -> FTQCResourceParetoReport:
    """Build a Pareto-frontier report across FTQC candidate estimates.

    Every selected quantity is treated as a resource cost where smaller is
    better. A candidate dominates another candidate only when every selected
    quantity is provably no larger and at least one quantity is provably
    smaller. Symbolic comparisons that cannot be decided keep both candidates
    on the frontier for later review.

    Args:
        candidates (tuple[tuple[str, SupportsFTQCResourceValues], ...]):
            Candidate labels paired with estimates, models, or plans exposing
            ``resource_values()``.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC resource Pareto frontier"``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to inspect before optional profile quantities. Defaults to the
            canonical intersection exposed by all candidates when ``profile``
            is None.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile whose quantities are appended after ``quantities`` with
            duplicates removed. Defaults to None.

    Returns:
        FTQCResourceParetoReport: Multi-candidate frontier report.

    Raises:
        ValueError: If fewer than two candidates are provided, labels repeat,
            selected quantities are missing, or a profile key is unknown.
    """
    if len(candidates) < 2:
        raise ValueError("candidates must contain at least two entries.")
    labels = tuple(label for label, _ in candidates)
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique.")

    value_maps = tuple(estimate.resource_values() for _, estimate in candidates)
    selected = _select_pareto_quantities(value_maps, quantities, profile)
    rows = []
    for index, (label, _) in enumerate(candidates):
        values: dict[str | FTQCResourceQuantity, sp.Expr | int | float] = {
            quantity: _sympify_resource_expr(
                value_maps[index][quantity], quantity.value
            )
            for quantity in selected
        }
        dominated_by = tuple(
            other_label
            for other_index, (other_label, _) in enumerate(candidates)
            if other_index != index
            and _pareto_dominates(value_maps[other_index], value_maps[index], selected)
        )
        rows.append(
            FTQCResourceParetoRow(
                label=label,
                values=values,
                dominated_by=dominated_by,
            )
        )

    return FTQCResourceParetoReport(
        title=title,
        quantities=selected,
        rows=tuple(rows),
    )


def build_ftqc_resource_scenario_report(
    estimate: SupportsFTQCResourceValues,
    scenarios: tuple[FTQCResourceScenario, ...],
    *,
    title: str = "FTQC resource scenario report",
    quantities: tuple[str | FTQCResourceQuantity, ...] | None = None,
    profile: str | FTQCResourceProfile | None = None,
) -> FTQCResourceScenarioReport:
    """Evaluate a symbolic FTQC estimate under multiple scenarios.

    This helper substitutes scenario values into selected resource expressions
    without changing the underlying estimate. It is useful for architecture
    sensitivity reviews, where physical qubit counts, runtimes, and
    qubit-seconds should be compared under several hardware assumptions.

    Args:
        estimate (SupportsFTQCResourceValues): Estimate, model, or plan
            exposing ``resource_values()``.
        scenarios (tuple[FTQCResourceScenario, ...]): Symbol-substitution
            scenarios to evaluate in order.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC resource scenario report"``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to inspect before optional profile quantities. Defaults to all
            quantities exposed by ``estimate`` when ``profile`` is None.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile whose quantities are appended after ``quantities`` with
            duplicates removed. Defaults to None.

    Returns:
        FTQCResourceScenarioReport: Scenario-evaluated resource report.

    Raises:
        TypeError: If ``scenarios`` contains non-scenario items.
        ValueError: If ``scenarios`` is empty, a selected quantity is missing,
            or a profile key is unknown.
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty.")
    if not all(isinstance(scenario, FTQCResourceScenario) for scenario in scenarios):
        raise TypeError("scenarios must contain only FTQCResourceScenario items.")

    values = estimate.resource_values()
    selected = _select_scenario_quantities(values, quantities, profile)
    rows = []
    for scenario in scenarios:
        substitutions = cast(dict[sp.Symbol, sp.Expr], scenario.substitutions)
        evaluated_values: dict[str | FTQCResourceQuantity, sp.Expr | int | float] = {
            quantity: _substitute_resource_expression(
                _sympify_resource_expr(values[quantity], quantity.value),
                substitutions,
            )
            for quantity in selected
        }
        evaluated_exprs = cast(dict[FTQCResourceQuantity, sp.Expr], evaluated_values)
        rows.append(
            FTQCResourceScenarioRow(
                label=scenario.label,
                values=evaluated_values,
                unresolved_symbols=_free_symbol_names(tuple(evaluated_exprs.values())),
            )
        )

    return FTQCResourceScenarioReport(
        title=title,
        quantities=selected,
        scenarios=scenarios,
        rows=tuple(rows),
    )


def build_ftqc_resource_report_snapshot(
    report: FTQCResourceReportLike,
    *,
    kind: str | FTQCResourceReportKind | None = None,
    title: str | None = None,
) -> FTQCResourceReportSnapshot:
    """Build a stable snapshot envelope for one FTQC report.

    Args:
        report (FTQCResourceReportLike): Report object exposing ``title`` and
            ``to_dict()``.
        kind (str | FTQCResourceReportKind | None): Optional explicit report
            kind. Defaults to inferring the kind from the report class.
        title (str | None): Optional title override. Defaults to
            ``report.title``.

    Returns:
        FTQCResourceReportSnapshot: Snapshot envelope containing the report
            kind, title, row count, grouped counts, and payload.

    Raises:
        TypeError: If ``report`` is not a supported FTQC report type.
        ValueError: If ``kind`` is unknown, the resolved title is empty, or
            extracted count metadata is malformed.
    """
    normalized_kind = (
        _infer_resource_report_kind(report)
        if kind is None
        else FTQCResourceReportKind(kind)
    )
    payload = cast(dict[str, Any], report.to_dict())
    return FTQCResourceReportSnapshot(
        kind=normalized_kind,
        title=report.title if title is None else title,
        payload=payload,
        row_count=_extract_report_row_count(payload),
        counts=_extract_report_counts(payload),
    )


def build_ftqc_resource_report_bundle(
    title: str,
    reports: tuple[FTQCResourceReportLike, ...],
) -> FTQCResourceReportBundle:
    """Build a review bundle from several FTQC reports.

    Args:
        title (str): Reader-facing bundle title.
        reports (tuple[FTQCResourceReportLike, ...]): Reports to snapshot in
            review order.

    Returns:
        FTQCResourceReportBundle: Bundle containing one snapshot per report.

    Raises:
        TypeError: If any report is not a supported FTQC report type.
        ValueError: If ``title`` or ``reports`` is empty, or if any snapshot
            cannot be built.
    """
    if not reports:
        raise ValueError("reports must not be empty.")
    return FTQCResourceReportBundle(
        title=title,
        snapshots=tuple(
            build_ftqc_resource_report_snapshot(report) for report in reports
        ),
    )


def build_ftqc_resource_review_findings(
    summary: FTQCResourceComparisonSummary,
    *,
    max_improvements: int = 3,
    max_tradeoffs: int = 3,
    include_symbolic: bool = True,
    include_unchanged: bool = False,
) -> tuple[FTQCResourceReviewFinding, ...]:
    """Build prioritized findings from an FTQC comparison summary.

    The returned tuple is ordered for design review: largest reductions first,
    largest regressions or tradeoffs second, then symbolic and unchanged rows
    when requested. This keeps numeric progress visible while preserving
    undecidable symbolic quantities for follow-up modeling.

    Args:
        summary (FTQCResourceComparisonSummary): Grouped comparison rows to
            turn into review findings.
        max_improvements (int): Maximum number of reducing quantities to
            include. Defaults to 3.
        max_tradeoffs (int): Maximum number of increasing quantities to
            include. Defaults to 3.
        include_symbolic (bool): Whether to include undecidable symbolic
            changes after numeric findings. Defaults to True.
        include_unchanged (bool): Whether to include unchanged quantities
            after symbolic findings. Defaults to False.

    Returns:
        tuple[FTQCResourceReviewFinding, ...]: Prioritized review findings.

    Raises:
        ValueError: If either maximum is negative.
    """
    if max_improvements < 0:
        raise ValueError("max_improvements must be non-negative.")
    if max_tradeoffs < 0:
        raise ValueError("max_tradeoffs must be non-negative.")

    rows: list[tuple[FTQCResourceChangeDirection, FTQCResourceComparisonRow]] = [
        (FTQCResourceChangeDirection.SMALLER, row)
        for row in summary.smaller[:max_improvements]
    ]
    rows.extend(
        (FTQCResourceChangeDirection.LARGER, row)
        for row in summary.larger[:max_tradeoffs]
    )
    if include_symbolic:
        rows.extend(
            (FTQCResourceChangeDirection.SYMBOLIC, row) for row in summary.symbolic
        )
    if include_unchanged:
        rows.extend(
            (FTQCResourceChangeDirection.UNCHANGED, row) for row in summary.unchanged
        )

    return tuple(_review_finding_from_row(row, direction) for direction, row in rows)


def evaluate_ftqc_resource_constraints(
    estimate: SupportsFTQCResourceValues,
    constraints: tuple[FTQCResourceConstraint, ...],
    *,
    title: str = "FTQC resource budget",
) -> FTQCResourceBudgetReport:
    """Evaluate FTQC resource values against symbolic budget constraints.

    Args:
        estimate (SupportsFTQCResourceValues): Estimate, model, or summary
            exposing ``resource_values()``.
        constraints (tuple[FTQCResourceConstraint, ...]): Constraints to
            evaluate in order.
        title (str): Reader-facing report title. Defaults to
            ``"FTQC resource budget"``.

    Returns:
        FTQCResourceBudgetReport: Grouped constraint results.

    Raises:
        ValueError: If a constraint quantity is missing from ``estimate``.
    """
    values = estimate.resource_values()
    results = []
    for constraint in constraints:
        quantity = cast(FTQCResourceQuantity, constraint.quantity)
        if quantity not in values:
            raise ValueError(
                "Requested FTQC resource constraint is missing from the input: "
                f"{quantity.value}."
            )
        value = _sympify_resource_expr(values[quantity], quantity.value)
        limit = cast(sp.Expr, constraint.limit)
        sense = cast(FTQCResourceConstraintSense, constraint.sense)
        margin = _constraint_margin(value, limit, sense)
        spec = describe_ftqc_resource_quantity(quantity)
        results.append(
            FTQCResourceConstraintResult(
                quantity=quantity,
                status=_classify_constraint_margin(margin),
                sense=sense,
                value=value,
                limit=limit,
                margin=margin,
                label=constraint.label or spec.label,
                unit=spec.unit,
                category=spec.category,
            )
        )

    return FTQCResourceBudgetReport.from_results(title, tuple(results))


def _normalize_driver_targets(
    targets: tuple[str | FTQCResourceQuantity, ...],
) -> tuple[FTQCResourceQuantity, ...]:
    """Normalize target quantities for a driver report.

    Args:
        targets (tuple[str | FTQCResourceQuantity, ...]): Requested driver
            report target quantities.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Deduplicated normalized targets.

    Raises:
        ValueError: If ``targets`` is empty or contains an unknown quantity.
    """
    if not targets:
        raise ValueError("targets must not be empty.")
    return _dedupe_resource_quantities(
        tuple(_normalize_resource_quantity(target) for target in targets)
    )


def _select_driver_report_quantities(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    *,
    targets: tuple[FTQCResourceQuantity, ...],
    include_targets: bool,
) -> tuple[FTQCResourceQuantity, ...]:
    """Select comparable formula-driver quantities for a report.

    Args:
        baseline (SupportsFTQCResourceValues): Baseline resource object.
        candidate (SupportsFTQCResourceValues): Candidate resource object.
        targets (tuple[FTQCResourceQuantity, ...]): Normalized report targets.
        include_targets (bool): Whether target quantities stay in the output
            selection.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Comparable driver quantities.

    Raises:
        ValueError: If no comparable driver quantity remains.
    """
    baseline_values = baseline.resource_values()
    candidate_values = candidate.resource_values()
    closure = _driver_dependency_closure(baseline, candidate, targets)
    selected = tuple(
        quantity
        for quantity in closure
        if (include_targets or quantity not in targets)
        and quantity in baseline_values
        and quantity in candidate_values
    )
    if not selected:
        target_labels = ", ".join(quantity.value for quantity in targets)
        raise ValueError(
            "No comparable FTQC driver quantities are available for targets: "
            f"{target_labels}."
        )
    return selected


def _driver_dependency_closure(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
    targets: tuple[FTQCResourceQuantity, ...],
) -> tuple[FTQCResourceQuantity, ...]:
    """Return target formula dependencies in dependency-first order.

    Args:
        baseline (SupportsFTQCResourceValues): Baseline resource object.
        candidate (SupportsFTQCResourceValues): Candidate resource object.
        targets (tuple[FTQCResourceQuantity, ...]): Target quantities to
            explain.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Deduplicated dependency closure.
    """
    dependencies = _formula_dependency_map(baseline, candidate)
    ordered: list[FTQCResourceQuantity] = []
    visiting: set[FTQCResourceQuantity] = set()
    seen: set[FTQCResourceQuantity] = set()

    def visit(quantity: FTQCResourceQuantity) -> None:
        """Append dependencies before the requested quantity.

        Args:
            quantity (FTQCResourceQuantity): Quantity to visit.
        """
        if quantity in seen:
            return
        if quantity in visiting:
            ordered.append(quantity)
            seen.add(quantity)
            return
        visiting.add(quantity)
        for dependency in dependencies.get(quantity, ()):
            visit(dependency)
        visiting.remove(quantity)
        if quantity not in seen:
            ordered.append(quantity)
            seen.add(quantity)

    for target in targets:
        visit(target)
    return tuple(ordered)


def _formula_dependency_map(
    baseline: SupportsFTQCResourceValues,
    candidate: SupportsFTQCResourceValues,
) -> dict[FTQCResourceQuantity, tuple[FTQCResourceQuantity, ...]]:
    """Collect formula dependencies from two resource objects.

    Args:
        baseline (SupportsFTQCResourceValues): Baseline resource object.
        candidate (SupportsFTQCResourceValues): Candidate resource object.

    Returns:
        dict[FTQCResourceQuantity, tuple[FTQCResourceQuantity, ...]]:
            Formula dependencies keyed by produced quantity.
    """
    dependencies: dict[FTQCResourceQuantity, tuple[FTQCResourceQuantity, ...]] = {}
    for formula in (*_resource_formulas(baseline), *_resource_formulas(candidate)):
        quantity = cast(FTQCResourceQuantity, formula.quantity)
        depends_on = cast(tuple[FTQCResourceQuantity, ...], formula.depends_on)
        if quantity in dependencies:
            dependencies[quantity] = _dedupe_resource_quantities(
                (*dependencies[quantity], *depends_on)
            )
        else:
            dependencies[quantity] = depends_on
    return dependencies


def _resource_formulas(
    estimate: SupportsFTQCResourceValues,
) -> tuple[FTQCResourceFormula, ...]:
    """Return formula metadata exposed by a resource object.

    Args:
        estimate (SupportsFTQCResourceValues): Resource object that may expose
            a ``formulas`` tuple or ``formulas()`` method.

    Returns:
        tuple[FTQCResourceFormula, ...]: Formula metadata, or an empty tuple
            when the object has no formula metadata.

    Raises:
        TypeError: If the exposed formula metadata contains non-formula items.
    """
    raw_formulas = getattr(estimate, "formulas", ())
    if raw_formulas is None:
        return ()
    if callable(raw_formulas):
        raw_formulas = raw_formulas()
    formulas = tuple(raw_formulas)
    if not all(isinstance(formula, FTQCResourceFormula) for formula in formulas):
        raise TypeError("formulas must contain only FTQCResourceFormula instances.")
    return formulas


def _select_comparison_quantities(
    baseline_values: dict[FTQCResourceQuantity, sp.Expr],
    candidate_values: dict[FTQCResourceQuantity, sp.Expr],
    quantities: tuple[str | FTQCResourceQuantity, ...] | None,
    profile: str | FTQCResourceProfile | None,
) -> tuple[FTQCResourceQuantity, ...]:
    """Select quantities from explicit requests and optional review profile.

    Args:
        baseline_values (dict[FTQCResourceQuantity, sp.Expr]): Baseline
            resource values.
        candidate_values (dict[FTQCResourceQuantity, sp.Expr]): Candidate
            resource values.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Explicit
            requested quantities.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile to append after explicit quantities.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Normalized quantities.

    Raises:
        ValueError: If a requested quantity is absent from either value map, or
            if ``profile`` is not a known FTQC resource profile.
    """
    if profile is None:
        return _normalize_comparison_quantities(
            baseline_values,
            candidate_values,
            quantities,
        )

    profile_quantities = ftqc_resource_profile_quantities(profile)
    if quantities is None:
        selected = profile_quantities
    else:
        selected = (*quantities, *profile_quantities)

    return _normalize_comparison_quantities(
        baseline_values,
        candidate_values,
        selected,
    )


def _normalize_comparison_quantities(
    baseline_values: dict[FTQCResourceQuantity, sp.Expr],
    candidate_values: dict[FTQCResourceQuantity, sp.Expr],
    quantities: tuple[str | FTQCResourceQuantity, ...] | None,
) -> tuple[FTQCResourceQuantity, ...]:
    """Normalize comparison quantity selection.

    Args:
        baseline_values (dict[FTQCResourceQuantity, sp.Expr]): Baseline
            resource values.
        candidate_values (dict[FTQCResourceQuantity, sp.Expr]): Candidate
            resource values.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Requested
            quantities or None for the canonical intersection.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Normalized quantities.

    Raises:
        ValueError: If a requested quantity is absent from either value map.
    """
    if quantities is None:
        common = set(baseline_values) & set(candidate_values)
        return tuple(
            spec.quantity
            for spec in FTQC_RESOURCE_QUANTITY_SPECS
            if spec.quantity in common
        )

    normalized = _dedupe_resource_quantities(
        tuple(_normalize_resource_quantity(quantity) for quantity in quantities)
    )
    missing = [
        quantity.value
        for quantity in normalized
        if quantity not in baseline_values or quantity not in candidate_values
    ]
    if missing:
        raise ValueError(
            "Requested FTQC resource quantities are missing from the inputs: "
            + ", ".join(missing)
            + "."
        )
    return normalized


def _select_pareto_quantities(
    value_maps: tuple[dict[FTQCResourceQuantity, sp.Expr], ...],
    quantities: tuple[str | FTQCResourceQuantity, ...] | None,
    profile: str | FTQCResourceProfile | None,
) -> tuple[FTQCResourceQuantity, ...]:
    """Select quantities shared by all Pareto candidates.

    Args:
        value_maps (tuple[dict[FTQCResourceQuantity, sp.Expr], ...]): Resource
            values for each candidate.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Explicit
            requested quantities.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Normalized selected quantities.

    Raises:
        ValueError: If a requested quantity is absent from any candidate or if
            the selected common quantity set is empty.
    """
    if profile is None:
        selected = quantities
    elif quantities is None:
        selected = ftqc_resource_profile_quantities(profile)
    else:
        selected = (*quantities, *ftqc_resource_profile_quantities(profile))

    if selected is None:
        common = set(value_maps[0])
        for values in value_maps[1:]:
            common &= set(values)
        normalized = tuple(
            spec.quantity
            for spec in FTQC_RESOURCE_QUANTITY_SPECS
            if spec.quantity in common
        )
    else:
        normalized = _dedupe_resource_quantities(
            tuple(_normalize_resource_quantity(quantity) for quantity in selected)
        )
    if not normalized:
        raise ValueError("No common FTQC resource quantities are available.")

    missing = [
        quantity.value
        for quantity in normalized
        if any(quantity not in values for values in value_maps)
    ]
    if missing:
        raise ValueError(
            "Requested FTQC Pareto quantities are missing from at least one "
            "candidate: " + ", ".join(missing) + "."
        )
    return normalized


def _select_scenario_quantities(
    values: dict[FTQCResourceQuantity, sp.Expr],
    quantities: tuple[str | FTQCResourceQuantity, ...] | None,
    profile: str | FTQCResourceProfile | None,
) -> tuple[FTQCResourceQuantity, ...]:
    """Select quantities available on one scenario-evaluated estimate.

    Args:
        values (dict[FTQCResourceQuantity, sp.Expr]): Resource values exposed
            by the estimate.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Explicit
            requested quantities.
        profile (str | FTQCResourceProfile | None): Optional standard review
            profile.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Normalized selected quantities.

    Raises:
        ValueError: If a requested quantity is absent from ``values`` or if no
            selected quantity remains.
    """
    if profile is None:
        selected = quantities
    elif quantities is None:
        selected = ftqc_resource_profile_quantities(profile)
    else:
        selected = (*quantities, *ftqc_resource_profile_quantities(profile))

    if selected is None:
        normalized = tuple(
            spec.quantity
            for spec in FTQC_RESOURCE_QUANTITY_SPECS
            if spec.quantity in values
        )
    else:
        normalized = _dedupe_resource_quantities(
            tuple(_normalize_resource_quantity(quantity) for quantity in selected)
        )
    if not normalized:
        raise ValueError("No FTQC resource quantities are available.")

    missing = [quantity.value for quantity in normalized if quantity not in values]
    if missing:
        raise ValueError(
            "Requested FTQC scenario quantities are missing from the input: "
            + ", ".join(missing)
            + "."
        )
    return normalized


def _infer_resource_report_kind(
    report: FTQCResourceReportLike,
) -> FTQCResourceReportKind:
    """Infer the standard snapshot kind for one report object.

    Args:
        report (FTQCResourceReportLike): FTQC report object.

    Returns:
        FTQCResourceReportKind: Inferred report kind.

    Raises:
        TypeError: If ``report`` is not a supported FTQC report type.
    """
    if isinstance(report, FTQCResearchSignalCoverageReport):
        return FTQCResourceReportKind.RESEARCH_SIGNAL_COVERAGE
    if isinstance(report, FTQCResourceBudgetReport):
        return FTQCResourceReportKind.BUDGET
    if isinstance(report, FTQCResourceDriverReport):
        return FTQCResourceReportKind.DRIVER
    if isinstance(report, FTQCResourceParetoReport):
        return FTQCResourceReportKind.PARETO
    if isinstance(report, FTQCResourceScenarioReport):
        return FTQCResourceReportKind.SCENARIO
    if isinstance(report, FTQCResourceComparisonReport):
        return FTQCResourceReportKind.COMPARISON
    raise TypeError("Unsupported FTQC resource report type.")


def _extract_report_row_count(payload: dict[str, Any]) -> int:
    """Extract the primary row count from a report payload.

    Args:
        payload (dict[str, Any]): Serialized report payload.

    Returns:
        int: Number of primary report rows, or zero when no row-like field is
            present.
    """
    rows = payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    results = payload.get("results")
    if isinstance(results, list):
        return len(results)
    return 0


def _extract_report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract primary rows from a report payload.

    Args:
        payload (dict[str, Any]): Serialized report payload.

    Returns:
        list[dict[str, Any]]: Shallow copies of row dictionaries from
            ``payload["rows"]`` or, for budget reports, ``payload["results"]``.

    Raises:
        ValueError: If a row-like payload field exists but does not contain
            dictionaries.
    """
    raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("results", [])
    if not isinstance(raw_rows, list):
        raise ValueError("report payload rows must be a list.")
    rows = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("report payload rows must contain dictionaries.")
        rows.append(dict(row))
    return rows


def _extract_report_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Extract grouped integer counts from a report payload.

    Args:
        payload (dict[str, Any]): Serialized report payload.

    Returns:
        dict[str, int]: Count metadata copied from ``payload["counts"]`` when
            present, otherwise an empty mapping.

    Raises:
        ValueError: If ``payload["counts"]`` is present but is not a mapping
            from strings to integers.
    """
    raw_counts = payload.get("counts", {})
    if not isinstance(raw_counts, dict):
        raise ValueError("report payload counts must be a dictionary.")
    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("report payload counts must map strings to integers.")
        counts[key] = value
    return counts


def _normalize_substitution_symbol(symbol: str | sp.Symbol) -> sp.Symbol:
    """Normalize a scenario substitution key to a SymPy symbol.

    Args:
        symbol (str | sp.Symbol): Symbol name or Symbol object.

    Returns:
        sp.Symbol: Normalized symbol.

    Raises:
        TypeError: If ``symbol`` is neither a string nor a SymPy symbol.
        ValueError: If ``symbol`` is an empty string.
    """
    if isinstance(symbol, sp.Symbol):
        return symbol
    if isinstance(symbol, str):
        if not symbol:
            raise ValueError("substitution symbol names must not be empty.")
        return sp.Symbol(symbol)
    raise TypeError("substitution keys must be strings or SymPy symbols.")


def _substitute_resource_expression(
    expression: sp.Expr,
    substitutions: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """Apply exact and name-matched symbol substitutions to an expression.

    Args:
        expression (sp.Expr): Resource expression to substitute.
        substitutions (dict[sp.Symbol, sp.Expr]): Scenario substitutions
            keyed by normalized symbols.

    Returns:
        sp.Expr: Simplified expression after applying substitutions.
    """
    substitutions_by_name = {
        str(symbol): value for symbol, value in substitutions.items()
    }
    effective_substitutions = {
        symbol: substitutions_by_name[str(symbol)]
        for symbol in expression.free_symbols
        if str(symbol) in substitutions_by_name
    }
    return sp.simplify(
        expression.subs(
            cast(dict[sp.Basic | complex, sp.Expr | complex], effective_substitutions)
        )
    )


def _free_symbol_names(expressions: tuple[sp.Expr, ...]) -> tuple[str, ...]:
    """Return sorted free-symbol names from expressions.

    Args:
        expressions (tuple[sp.Expr, ...]): Expressions to inspect.

    Returns:
        tuple[str, ...]: Sorted names of remaining free symbols.
    """
    names = {
        str(symbol) for expression in expressions for symbol in expression.free_symbols
    }
    return tuple(sorted(names))


def _dedupe_resource_quantities(
    quantities: tuple[FTQCResourceQuantity, ...],
) -> tuple[FTQCResourceQuantity, ...]:
    """Remove duplicate resource quantities while preserving input order.

    Args:
        quantities (tuple[FTQCResourceQuantity, ...]): Normalized resource
            quantities.

    Returns:
        tuple[FTQCResourceQuantity, ...]: First occurrence of each quantity.
    """
    seen: set[FTQCResourceQuantity] = set()
    deduped = []
    for quantity in quantities:
        if quantity in seen:
            continue
        seen.add(quantity)
        deduped.append(quantity)
    return tuple(deduped)


def _normalize_resource_quantity(
    quantity: str | FTQCResourceQuantity,
) -> FTQCResourceQuantity:
    """Normalize one resource quantity key.

    Args:
        quantity (str | FTQCResourceQuantity): Resource quantity key.

    Returns:
        FTQCResourceQuantity: Normalized quantity enum.

    Raises:
        ValueError: If ``quantity`` is not a known FTQC resource quantity.
    """
    try:
        return FTQCResourceQuantity(quantity)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FTQCResourceQuantity)
        raise ValueError(
            f"Unknown FTQC resource quantity {quantity!r}; valid: {valid}."
        ) from exc


def _normalize_resource_profile(
    profile: str | FTQCResourceProfile,
) -> FTQCResourceProfile:
    """Normalize one resource review profile key.

    Args:
        profile (str | FTQCResourceProfile): Resource profile key.

    Returns:
        FTQCResourceProfile: Normalized profile enum.

    Raises:
        ValueError: If ``profile`` is not a known FTQC resource profile.
    """
    try:
        return FTQCResourceProfile(profile)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FTQCResourceProfile)
        raise ValueError(
            f"Unknown FTQC resource profile {profile!r}; valid: {valid}."
        ) from exc


def _normalize_constraint_sense(
    sense: str | FTQCResourceConstraintSense,
) -> FTQCResourceConstraintSense:
    """Normalize one FTQC resource constraint sense.

    Args:
        sense (str | FTQCResourceConstraintSense): Constraint sense key.

    Returns:
        FTQCResourceConstraintSense: Normalized sense enum.

    Raises:
        ValueError: If ``sense`` is not a known constraint sense.
    """
    try:
        return FTQCResourceConstraintSense(sense)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FTQCResourceConstraintSense)
        raise ValueError(
            f"Unknown FTQC resource constraint sense {sense!r}; valid: {valid}."
        ) from exc


def _normalize_aggregation_rule(
    rule: str | FTQCResourceAggregationRule,
) -> FTQCResourceAggregationRule:
    """Normalize one FTQC resource aggregation rule.

    Args:
        rule (str | FTQCResourceAggregationRule): Aggregation rule key.

    Returns:
        FTQCResourceAggregationRule: Normalized aggregation rule enum.

    Raises:
        ValueError: If ``rule`` is not a known aggregation rule.
    """
    try:
        return FTQCResourceAggregationRule(rule)
    except ValueError as exc:
        valid = ", ".join(item.value for item in FTQCResourceAggregationRule)
        raise ValueError(
            f"Unknown FTQC resource aggregation rule {rule!r}; valid: {valid}."
        ) from exc


def _sympify_resource_expr(value: sp.Expr | int | float, name: str) -> sp.Expr:
    """Convert a resource expression to SymPy.

    Args:
        value (sp.Expr | int | float): Resource value to convert.
        name (str): Field name used in diagnostics.

    Returns:
        sp.Expr: SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be converted to a SymPy expression.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


def _combine_resource_values(
    quantity: FTQCResourceQuantity,
    current: sp.Expr,
    incoming: sp.Expr,
    rule: FTQCResourceAggregationRule,
) -> sp.Expr:
    """Combine two resource values with an aggregation rule.

    Args:
        quantity (FTQCResourceQuantity): Quantity being combined.
        current (sp.Expr): Existing aggregate value.
        incoming (sp.Expr): New step value.
        rule (FTQCResourceAggregationRule): Aggregation rule to apply.

    Returns:
        sp.Expr: Combined resource value.

    Raises:
        ValueError: If ``rule`` is ``CONSISTENT`` and the two values are not
            provably equal.
    """
    if rule == FTQCResourceAggregationRule.ADD:
        return sp.simplify(current + incoming)
    if rule == FTQCResourceAggregationRule.PEAK:
        return sp.simplify(sp.Max(current, incoming))
    if rule == FTQCResourceAggregationRule.CONSISTENT:
        difference = sp.simplify(current - incoming)
        if difference.equals(0):
            return current
        raise ValueError(
            "Conflicting FTQC resource values for consistent quantity "
            f"{quantity.value!r}: {current} vs {incoming}."
        )
    assert False, f"Unhandled FTQC resource aggregation rule: {rule!r}."


def _select_research_signal_quantities(
    signal: FTQCResearchSignal,
    baseline_values: dict[FTQCResourceQuantity, sp.Expr],
    candidate_values: dict[FTQCResourceQuantity, sp.Expr],
    *,
    require_all_quantities: bool,
) -> tuple[FTQCResourceQuantity, ...]:
    """Select comparable quantities for one research signal.

    Args:
        signal (FTQCResearchSignal): Research signal whose quantities should
            scope the comparison.
        baseline_values (dict[FTQCResourceQuantity, sp.Expr]): Resource values
            exposed by the baseline input.
        candidate_values (dict[FTQCResourceQuantity, sp.Expr]): Resource
            values exposed by the candidate input.
        require_all_quantities (bool): Whether to reject any missing
            research-signal quantity.

    Returns:
        tuple[FTQCResourceQuantity, ...]: Research-signal quantities available
            on both inputs, preserving signal order.

    Raises:
        ValueError: If a required quantity is missing or no comparable
            quantities remain.
    """
    baseline_missing = [
        quantity for quantity in signal.quantities if quantity not in baseline_values
    ]
    candidate_missing = [
        quantity for quantity in signal.quantities if quantity not in candidate_values
    ]
    if require_all_quantities and (baseline_missing or candidate_missing):
        missing = [
            f"baseline={','.join(quantity.value for quantity in baseline_missing)}",
            f"candidate={','.join(quantity.value for quantity in candidate_missing)}",
        ]
        raise ValueError(
            "Missing FTQC research-signal quantities for "
            f"{signal.reference_key!r}: {'; '.join(missing)}."
        )

    quantities = tuple(
        quantity
        for quantity in signal.quantities
        if quantity in baseline_values and quantity in candidate_values
    )
    if not quantities:
        raise ValueError(
            "No comparable FTQC research-signal quantities are exposed by both "
            f"inputs for {signal.reference_key!r}."
        )
    return quantities


def _classify_change(reduction: sp.Expr) -> FTQCResourceChangeDirection:
    """Classify a fractional reduction expression by sign.

    Args:
        reduction (sp.Expr): Fractional reduction, where positive means the
            candidate is smaller than the baseline.

    Returns:
        FTQCResourceChangeDirection: Sign classification for the change.
    """
    simplified = sp.simplify(reduction)
    if simplified.equals(0):
        return FTQCResourceChangeDirection.UNCHANGED
    if simplified.is_positive:
        return FTQCResourceChangeDirection.SMALLER
    if simplified.is_negative:
        return FTQCResourceChangeDirection.LARGER
    return FTQCResourceChangeDirection.SYMBOLIC


def _pareto_dominates(
    challenger_values: dict[FTQCResourceQuantity, sp.Expr],
    incumbent_values: dict[FTQCResourceQuantity, sp.Expr],
    quantities: tuple[FTQCResourceQuantity, ...],
) -> bool:
    """Return whether one candidate provably dominates another.

    Args:
        challenger_values (dict[FTQCResourceQuantity, sp.Expr]): Candidate
            values for the possible dominator.
        incumbent_values (dict[FTQCResourceQuantity, sp.Expr]): Candidate
            values for the possible dominated row.
        quantities (tuple[FTQCResourceQuantity, ...]): Quantities used for
            dominance checks, where smaller is better.

    Returns:
        bool: True only when the challenger is no larger for every quantity
            and strictly smaller for at least one quantity.
    """
    strictly_better = False
    for quantity in quantities:
        challenger = _sympify_resource_expr(
            challenger_values[quantity],
            quantity.value,
        )
        incumbent = _sympify_resource_expr(
            incumbent_values[quantity],
            quantity.value,
        )
        difference = sp.simplify(incumbent - challenger)
        if difference.equals(0):
            continue
        if difference.is_positive:
            strictly_better = True
            continue
        return False
    return strictly_better


def _constraint_margin(
    value: sp.Expr,
    limit: sp.Expr,
    sense: FTQCResourceConstraintSense,
) -> sp.Expr:
    """Return signed constraint headroom.

    Args:
        value (sp.Expr): Evaluated resource value.
        limit (sp.Expr): Constraint limit.
        sense (FTQCResourceConstraintSense): Constraint direction.

    Returns:
        sp.Expr: Signed headroom where nonnegative means the constraint is met.
    """
    if sense == FTQCResourceConstraintSense.AT_MOST:
        return sp.simplify(limit - value)
    if sense == FTQCResourceConstraintSense.AT_LEAST:
        return sp.simplify(value - limit)
    assert False, f"Unhandled FTQC resource constraint sense: {sense!r}."


def _classify_constraint_margin(
    margin: sp.Expr,
) -> FTQCResourceConstraintStatus:
    """Classify a resource constraint margin by sign.

    Args:
        margin (sp.Expr): Signed headroom where nonnegative means satisfied.

    Returns:
        FTQCResourceConstraintStatus: Constraint status.
    """
    simplified = sp.simplify(margin)
    if simplified.equals(0) or simplified.is_nonnegative:
        return FTQCResourceConstraintStatus.SATISFIED
    if simplified.is_negative:
        return FTQCResourceConstraintStatus.VIOLATED
    return FTQCResourceConstraintStatus.SYMBOLIC


def _sort_rows_by_change(
    rows: list[FTQCResourceComparisonRow],
) -> tuple[FTQCResourceComparisonRow, ...]:
    """Sort rows by descending numeric fractional change when available.

    Args:
        rows (list[FTQCResourceComparisonRow]): Rows to sort.

    Returns:
        tuple[FTQCResourceComparisonRow, ...]: Rows with numerically comparable
            changes first and symbolic ties left in their input order.
    """
    indexed_rows = list(enumerate(rows))

    def key(item: tuple[int, FTQCResourceComparisonRow]) -> tuple[int, float, int]:
        index, row = item
        magnitude = sp.Abs(sp.simplify(row.reduction))
        if magnitude.is_number:
            return (0, -float(sp.N(magnitude)), index)
        return (1, 0.0, index)

    return tuple(row for _, row in sorted(indexed_rows, key=key))


def _review_finding_from_row(
    row: FTQCResourceComparisonRow,
    direction: FTQCResourceChangeDirection,
) -> FTQCResourceReviewFinding:
    """Build a review finding for one comparison row.

    Args:
        row (FTQCResourceComparisonRow): Comparison row to describe.
        direction (FTQCResourceChangeDirection): Change class already assigned
            by the summary.

    Returns:
        FTQCResourceReviewFinding: Reader-facing finding for reports.
    """
    headline = _review_finding_headline(row, direction)
    detail = (
        f"{row.quantity.value}: baseline={row.baseline}, "
        f"candidate={row.candidate}, ratio={row.ratio}, "
        f"reduction={row.reduction}."
    )
    return FTQCResourceReviewFinding(
        direction=direction,
        quantity=row.quantity,
        label=row.label,
        unit=row.unit,
        category=row.category,
        baseline=row.baseline,
        candidate=row.candidate,
        ratio=row.ratio,
        reduction=row.reduction,
        headline=headline,
        detail=detail,
    )


def _review_finding_headline(
    row: FTQCResourceComparisonRow,
    direction: FTQCResourceChangeDirection,
) -> str:
    """Return a short headline for a review finding.

    Args:
        row (FTQCResourceComparisonRow): Comparison row to describe.
        direction (FTQCResourceChangeDirection): Change class for the row.

    Returns:
        str: Reader-facing headline.
    """
    if direction == FTQCResourceChangeDirection.SMALLER:
        return f"Candidate reduces {row.label}."
    if direction == FTQCResourceChangeDirection.LARGER:
        return f"Candidate increases {row.label}."
    if direction == FTQCResourceChangeDirection.UNCHANGED:
        return f"Candidate leaves {row.label} unchanged."
    return f"Candidate change for {row.label} remains symbolic."
