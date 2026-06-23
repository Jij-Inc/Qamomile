"""Define canonical FTQC resource quantities for symbolic estimators."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, cast

import sympy as sp


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
                JSON-friendly report metadata, selected quantities, rows, and
                grouped-count summary.
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
            FTQCResourceQuantity.QPE_REPETITIONS,
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
    ),
)


def iter_ftqc_research_signals() -> tuple[FTQCResearchSignal, ...]:
    """Return research signals that motivate Qamomile FTQC quantities.

    Returns:
        tuple[FTQCResearchSignal, ...]: Survey entries mapping research
            directions to canonical FTQC quantity keys.
    """
    return FTQC_RESEARCH_SIGNALS


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
