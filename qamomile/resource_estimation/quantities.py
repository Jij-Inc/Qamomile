"""Define canonical resource quantities for reports and comparisons."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import sympy as sp

from qamomile.circuit.estimator import ResourceEstimate

_SympyLike = sp.Expr | int | float


class ResourceCategory(enum.StrEnum):
    """Group resource quantities by modeling layer.

    Attributes:
        PROBLEM: Input problem and Hamiltonian representation quantities.
        ALGORITHM: Algorithm-control quantities such as QPE iteration counts.
        LOGICAL: Logical-circuit quantities before hardware lifting.
        PHYSICAL: Physical-device quantities after architecture assumptions.
        ARCHITECTURE: Hardware-model knobs that map logical work to physical
            resources.

    Example:
        >>> ResourceCategory("logical")
        <ResourceCategory.LOGICAL: 'logical'>
    """

    PROBLEM = "problem"
    ALGORITHM = "algorithm"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    ARCHITECTURE = "architecture"


class ResourceQuantity(enum.StrEnum):
    """Name canonical resource quantities.

    Attributes:
        N_QUBITS: Encoded problem size or qubit-register width.
        N_PAULI_TERMS: Number of non-identity Pauli terms.
        LAMBDA_NORM: Hamiltonian normalization driving QPE calls.
        MAX_LOCALITY: Maximum Pauli-string locality.
        SYSTEM_QUBITS: Logical system-register qubits in a block encoding.
        BLOCK_ENCODING_ANCILLA_QUBITS: Ancilla and workspace qubits used by a
            block encoding before QPE readout qubits are added.
        PREPARE_COST_TOFFOLI: Toffoli cost of one PREPARE call.
        SELECT_COST_TOFFOLI: Toffoli cost of one SELECT or oracle call.
        REFLECTION_COST_TOFFOLI: Toffoli cost of the reflection in one
            qubitized walk.
        QPE_REGISTER_QUBITS: Optional QPE readout-register qubits.
        WALK_COST_TOFFOLI: Toffoli cost of one qubitized walk.
        TARGET_PRECISION: Total target energy precision budget for an
            algorithm estimate.
        REPRESENTATION_ERROR: Error budget consumed by Hamiltonian
            representation or compression before phase estimation.
        ALGORITHMIC_PRECISION: Precision budget left for QPE after
            representation error is removed.
        EFFECTIVE_LAMBDA_NORM: Hamiltonian normalization after algorithmic
            weight reduction.
        TROTTER_STEPS_PER_SAMPLE: Product-formula steps per sampled time.
        TROTTER_SAMPLES: Number of sampled times or signal-processing shots.
        UNITARY_WEIGHT_FACTOR: Multiplicative reduction applied to Hamiltonian
            weight.
        RANDOMIZED_COMPILATION_FACTOR: Multiplicative cost factor from
            randomized time evolution or compilation.
        ROTATION_SYNTHESIS_T_GATES: T gates used to synthesize one Pauli
            rotation.
        QPE_ITERATIONS: Number of phase-estimation walk or time-evolution
            calls.
        PAULI_ROTATIONS: Pauli rotations used by product-formula evolution.
        LOGICAL_QUBITS: Logical qubits required by an algorithm.
        LOGICAL_DEPTH: Logical-depth proxy.
        LOGICAL_SPACETIME_VOLUME: Logical qubit-layer volume proxy.
        NON_CLIFFORD_COUNT: T, Toffoli, or equivalent non-Clifford count.
        T_GATES: T-gate or T-equivalent count.
        MULTI_QUBIT_GATES: Multi-qubit gate-count proxy.
        PHYSICAL_QUBITS: Physical qubits under an architecture model.
        RUNTIME_SECONDS: Runtime proxy in seconds.
        PHYSICAL_QUBIT_SECONDS: Physical qubit-second space-time proxy.
        PHYSICAL_QUBITS_PER_LOGICAL: Physical overhead per logical qubit.
        LOGICAL_CYCLE_TIME_SECONDS: Logical layer or cycle time.
        FACTORY_QUBITS: Physical qubits reserved for factories.
        NON_CLIFFORD_THROUGHPUT_PER_SECOND: Sustainable non-Clifford
            throughput.
        CODE_DISTANCE: Surface-code distance.
        PHYSICAL_CYCLE_TIME_SECONDS: Physical error-correction cycle time.
        PHYSICAL_QUBITS_PER_LOGICAL_FACTOR: Constant factor multiplying
            distance squared for one logical patch.
        LOGICAL_CYCLE_FACTOR: Constant factor multiplying code distance for
            one logical cycle.
        FACTORY_COUNT: Number of non-Clifford factories.
        PHYSICAL_QUBITS_PER_FACTORY: Physical qubits used by one factory.
        FACTORY_CYCLES_PER_NON_CLIFFORD: Logical cycles needed per factory
            output.

    Example:
        >>> ResourceQuantity("lambda_norm")
        <ResourceQuantity.LAMBDA_NORM: 'lambda_norm'>
    """

    N_QUBITS = "n_qubits"
    N_PAULI_TERMS = "n_pauli_terms"
    LAMBDA_NORM = "lambda_norm"
    MAX_LOCALITY = "max_locality"
    SYSTEM_QUBITS = "system_qubits"
    BLOCK_ENCODING_ANCILLA_QUBITS = "block_encoding_ancilla_qubits"
    PREPARE_COST_TOFFOLI = "prepare_cost_toffoli"
    SELECT_COST_TOFFOLI = "select_cost_toffoli"
    REFLECTION_COST_TOFFOLI = "reflection_cost_toffoli"
    QPE_REGISTER_QUBITS = "qpe_register_qubits"
    WALK_COST_TOFFOLI = "walk_cost_toffoli"
    TARGET_PRECISION = "target_precision"
    REPRESENTATION_ERROR = "representation_error"
    ALGORITHMIC_PRECISION = "algorithmic_precision"
    EFFECTIVE_LAMBDA_NORM = "effective_lambda_norm"
    TROTTER_STEPS_PER_SAMPLE = "trotter_steps_per_sample"
    TROTTER_SAMPLES = "trotter_samples"
    UNITARY_WEIGHT_FACTOR = "unitary_weight_factor"
    RANDOMIZED_COMPILATION_FACTOR = "randomized_compilation_factor"
    ROTATION_SYNTHESIS_T_GATES = "rotation_synthesis_t_gates"
    QPE_ITERATIONS = "qpe_iterations"
    PAULI_ROTATIONS = "pauli_rotations"
    LOGICAL_QUBITS = "logical_qubits"
    LOGICAL_DEPTH = "logical_depth"
    LOGICAL_SPACETIME_VOLUME = "logical_spacetime_volume"
    NON_CLIFFORD_COUNT = "non_clifford_count"
    T_GATES = "t_gates"
    MULTI_QUBIT_GATES = "multi_qubit_gates"
    PHYSICAL_QUBITS = "physical_qubits"
    RUNTIME_SECONDS = "runtime_seconds"
    PHYSICAL_QUBIT_SECONDS = "physical_qubit_seconds"
    PHYSICAL_QUBITS_PER_LOGICAL = "physical_qubits_per_logical"
    LOGICAL_CYCLE_TIME_SECONDS = "logical_cycle_time_seconds"
    FACTORY_QUBITS = "factory_qubits"
    NON_CLIFFORD_THROUGHPUT_PER_SECOND = "non_clifford_throughput_per_second"
    CODE_DISTANCE = "code_distance"
    PHYSICAL_CYCLE_TIME_SECONDS = "physical_cycle_time_seconds"
    PHYSICAL_QUBITS_PER_LOGICAL_FACTOR = "physical_qubits_per_logical_factor"
    LOGICAL_CYCLE_FACTOR = "logical_cycle_factor"
    FACTORY_COUNT = "factory_count"
    PHYSICAL_QUBITS_PER_FACTORY = "physical_qubits_per_factory"
    FACTORY_CYCLES_PER_NON_CLIFFORD = "factory_cycles_per_non_clifford"


class ResourceReviewProfile(enum.StrEnum):
    """Select a recommended quantity set for resource reviews.

    Attributes:
        HAMILTONIAN_QPE_WORKLOAD: Problem and algorithm inputs that drive a
            Hamiltonian QPE workload.
        TROTTER_QPE_WORKLOAD: Product-formula and weight-reduction inputs
            that drive a Trotter QPE workload.
        FTQC_LOGICAL_OUTCOMES: Logical algorithm outcomes before architecture
            lifting.
        FTQC_PHYSICAL_OUTCOMES: Physical proxy outcomes after architecture
            lifting.
        SURFACE_CODE_ARCHITECTURE: Surface-code knobs that should be recorded
            beside a physical proxy estimate.

    Example:
        >>> ResourceReviewProfile("ftqc_logical_outcomes")
        <ResourceReviewProfile.FTQC_LOGICAL_OUTCOMES: 'ftqc_logical_outcomes'>
    """

    HAMILTONIAN_QPE_WORKLOAD = "hamiltonian_qpe_workload"
    TROTTER_QPE_WORKLOAD = "trotter_qpe_workload"
    FTQC_LOGICAL_OUTCOMES = "ftqc_logical_outcomes"
    FTQC_PHYSICAL_OUTCOMES = "ftqc_physical_outcomes"
    SURFACE_CODE_ARCHITECTURE = "surface_code_architecture"


class SupportsResourceValues(Protocol):
    """Represent objects that expose canonical resource values.

    Example:
        >>> hasattr(object(), "resource_values")
        False
    """

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return resource values keyed by canonical names.

        Returns:
            dict[str, sp.Expr]: Resource values.
        """
        ...


_ResourceValuesInput = (
    SupportsResourceValues
    | Mapping[str | ResourceQuantity, _SympyLike]
    | ResourceEstimate
)
_ResourceCandidatesInput = (
    Mapping[str, _ResourceValuesInput] | tuple[tuple[str, _ResourceValuesInput], ...]
)
_ScenarioSubstitutions = Mapping[str | sp.Symbol, _SympyLike]


@dataclass(frozen=True)
class ResourceQuantitySpec:
    """Describe one canonical resource quantity.

    Attributes:
        quantity (ResourceQuantity): Machine-readable quantity key.
        label (str): Reader-facing label.
        unit (str): Unit or dimension of the quantity.
        category (ResourceCategory): Modeling layer that owns the quantity.
        description (str): Short description of what the quantity measures.

    Example:
        >>> spec = describe_resource_quantity("logical_qubits")
        >>> spec.unit
        'logical qubits'
    """

    quantity: ResourceQuantity
    label: str
    unit: str
    category: ResourceCategory
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
class ResourceQuantityProfile:
    """Describe a recommended set of resource quantities.

    Attributes:
        profile (ResourceReviewProfile): Machine-readable profile key.
        label (str): Reader-facing profile label.
        description (str): Short description of the review purpose.
        quantities (tuple[ResourceQuantity, ...]): Canonical quantities to
            inspect for this profile.

    Example:
        >>> profile = describe_resource_review_profile(
        ...     ResourceReviewProfile.FTQC_PHYSICAL_OUTCOMES
        ... )
        >>> ResourceQuantity.RUNTIME_SECONDS in profile.quantities
        True
    """

    profile: ResourceReviewProfile
    label: str
    description: str
    quantities: tuple[ResourceQuantity, ...]

    def specs(self) -> tuple[ResourceQuantitySpec, ...]:
        """Return quantity specifications for this profile.

        Returns:
            tuple[ResourceQuantitySpec, ...]: Quantity metadata in profile
                order.
        """
        return tuple(
            describe_resource_quantity(quantity) for quantity in self.quantities
        )

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialize the resource quantity profile.

        Returns:
            dict[str, str | list[str]]: JSON-friendly profile metadata and
                quantity keys.
        """
        return {
            "profile": self.profile.value,
            "label": self.label,
            "description": self.description,
            "quantities": [quantity.value for quantity in self.quantities],
        }


@dataclass(frozen=True)
class ResourceComparisonRow:
    """Compare one resource quantity between two value providers.

    Attributes:
        quantity (ResourceQuantity): Compared resource quantity.
        baseline (sp.Expr): Baseline value.
        candidate (sp.Expr): Candidate value.
        ratio (sp.Expr): Candidate divided by baseline.
        reduction (sp.Expr): Fractional reduction, equal to
            ``1 - candidate / baseline``.
        label (str): Reader-facing quantity label.
        unit (str): Resource unit.
        category (ResourceCategory): Modeling layer for the quantity.

    Example:
        >>> row = ResourceComparisonRow(
        ...     quantity=ResourceQuantity.NON_CLIFFORD_COUNT,
        ...     baseline=10,
        ...     candidate=4,
        ...     ratio=sp.Rational(2, 5),
        ...     reduction=sp.Rational(3, 5),
        ...     label="Non-Clifford count",
        ...     unit="gates",
        ...     category=ResourceCategory.LOGICAL,
        ... )
        >>> row.to_dict()["ratio"]
        '2/5'
    """

    quantity: ResourceQuantity
    baseline: sp.Expr
    candidate: sp.Expr
    ratio: sp.Expr
    reduction: sp.Expr
    label: str
    unit: str
    category: ResourceCategory

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
class ResourceParetoRow:
    """Describe one candidate in a Pareto-frontier resource review.

    Attributes:
        label (str): Reader-facing candidate label.
        values (dict[ResourceQuantity, sp.Expr]): Selected resource values for
            the candidate. Smaller values are treated as better.
        dominated_by (tuple[str, ...]): Labels of candidates that provably
            dominate this row.

    Raises:
        TypeError: If any value cannot be converted to a SymPy expression.
        ValueError: If ``label`` is empty or a resource key is unknown.

    Example:
        >>> row = ResourceParetoRow(
        ...     label="compressed",
        ...     values={ResourceQuantity.RUNTIME_SECONDS: 2},
        ... )
        >>> row.is_frontier
        True
    """

    label: str
    values: dict[ResourceQuantity, sp.Expr]
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
            _normalize_resource_quantity(quantity): _as_expr(value, str(quantity))
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
                candidate label, selected values, frontier marker, and
                dominator labels.
        """
        return {
            "label": self.label,
            "values": {
                quantity.value: str(value) for quantity, value in self.values.items()
            },
            "is_frontier": self.is_frontier,
            "dominated_by": list(self.dominated_by),
        }


@dataclass(frozen=True)
class ResourceSymbolDependencyRow:
    """Describe unresolved symbolic dependencies for one resource quantity.

    Attributes:
        quantity (ResourceQuantity): Audited resource quantity.
        value (sp.Expr): Resource value expression.
        symbols (tuple[str, ...]): Sorted free-symbol names in ``value``.
        label (str): Reader-facing quantity label.
        unit (str): Resource unit.
        category (ResourceCategory): Modeling layer for the quantity.

    Example:
        >>> row = ResourceSymbolDependencyRow(
        ...     quantity=ResourceQuantity.RUNTIME_SECONDS,
        ...     value=sp.Symbol("runtime"),
        ...     symbols=("runtime",),
        ...     label="Runtime",
        ...     unit="seconds",
        ...     category=ResourceCategory.PHYSICAL,
        ... )
        >>> row.is_symbolic
        True
    """

    quantity: ResourceQuantity
    value: sp.Expr
    symbols: tuple[str, ...]
    label: str
    unit: str
    category: ResourceCategory

    @property
    def is_symbolic(self) -> bool:
        """Return whether the resource value still has free symbols.

        Returns:
            bool: True when ``value`` has unresolved SymPy symbols.
        """
        return bool(self.symbols)

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        """Serialize the symbol-dependency row.

        Returns:
            dict[str, str | bool | list[str]]: JSON-friendly row metadata,
                value expression, symbolic-state flag, and symbol names.
        """
        return {
            "quantity": self.quantity.value,
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
            "value": str(self.value),
            "is_symbolic": self.is_symbolic,
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True)
class ResourceSymbolDriverRow:
    """Describe which resource quantities one free symbol drives.

    Attributes:
        symbol (str): Free-symbol name.
        quantities (tuple[ResourceQuantity, ...]): Canonical quantities whose
            expressions contain ``symbol``.
        labels (tuple[str, ...]): Reader-facing labels for ``quantities``.
        categories (tuple[ResourceCategory, ...]): Distinct modeling layers
            touched by ``symbol`` in first-seen order.

    Example:
        >>> row = ResourceSymbolDriverRow(
        ...     symbol="lambda",
        ...     quantities=(ResourceQuantity.QPE_ITERATIONS,),
        ...     labels=("QPE iterations",),
        ...     categories=(ResourceCategory.ALGORITHM,),
        ... )
        >>> row.quantity_count
        1
    """

    symbol: str
    quantities: tuple[ResourceQuantity, ...]
    labels: tuple[str, ...]
    categories: tuple[ResourceCategory, ...]

    @property
    def quantity_count(self) -> int:
        """Return the number of quantities driven by the symbol.

        Returns:
            int: Count of impacted canonical quantities.
        """
        return len(self.quantities)

    def to_dict(self) -> dict[str, str | int | list[str]]:
        """Serialize the symbol-driver row.

        Returns:
            dict[str, str | int | list[str]]: JSON-friendly symbol name,
                impacted quantities, labels, categories, and count.
        """
        return {
            "symbol": self.symbol,
            "quantity_count": self.quantity_count,
            "quantities": [quantity.value for quantity in self.quantities],
            "labels": list(self.labels),
            "categories": [category.value for category in self.categories],
        }


@dataclass(frozen=True)
class ResourceScenarioValueRow:
    """Describe one resource quantity evaluated under a scenario.

    Attributes:
        scenario (str): Scenario label.
        quantity (ResourceQuantity): Evaluated resource quantity.
        expression (sp.Expr): Original symbolic expression before applying
            scenario substitutions.
        value (sp.Expr): Expression after scenario substitutions.
        symbols (tuple[str, ...]): Remaining free-symbol names in ``value``.
        label (str): Reader-facing quantity label.
        unit (str): Resource unit.
        category (ResourceCategory): Modeling layer for the quantity.

    Example:
        >>> row = ResourceScenarioValueRow(
        ...     scenario="baseline",
        ...     quantity=ResourceQuantity.LOGICAL_QUBITS,
        ...     expression=sp.Symbol("n") + 2,
        ...     value=10,
        ...     symbols=(),
        ...     label="Logical qubits",
        ...     unit="logical qubits",
        ...     category=ResourceCategory.LOGICAL,
        ... )
        >>> row.is_resolved
        True
    """

    scenario: str
    quantity: ResourceQuantity
    expression: sp.Expr
    value: sp.Expr
    symbols: tuple[str, ...]
    label: str
    unit: str
    category: ResourceCategory

    @property
    def is_resolved(self) -> bool:
        """Return whether the scenario resolved every free symbol.

        Returns:
            bool: True when ``value`` has no remaining SymPy symbols.
        """
        return not self.symbols

    def to_dict(self) -> dict[str, str | bool | list[str]]:
        """Serialize the scenario value row.

        Returns:
            dict[str, str | bool | list[str]]: JSON-friendly scenario label,
                quantity metadata, evaluated value, original expression, and
                remaining symbols.
        """
        return {
            "scenario": self.scenario,
            "quantity": self.quantity.value,
            "label": self.label,
            "unit": self.unit,
            "category": self.category.value,
            "expression": str(self.expression),
            "value": str(self.value),
            "is_resolved": self.is_resolved,
            "symbols": list(self.symbols),
        }


RESOURCE_QUANTITY_SPECS: tuple[ResourceQuantitySpec, ...] = (
    ResourceQuantitySpec(
        ResourceQuantity.N_QUBITS,
        "Qubits",
        "qubits",
        ResourceCategory.PROBLEM,
        "Encoded problem size or qubit-register width.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.N_PAULI_TERMS,
        "Pauli terms",
        "terms",
        ResourceCategory.PROBLEM,
        "Number of non-identity Pauli strings in the Hamiltonian model.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LAMBDA_NORM,
        "Hamiltonian normalization",
        "energy",
        ResourceCategory.PROBLEM,
        "LCU normalization that controls QPE walk or evolution calls.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.MAX_LOCALITY,
        "Maximum locality",
        "Pauli factors",
        ResourceCategory.PROBLEM,
        "Maximum number of non-identity Pauli factors in one term.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.SYSTEM_QUBITS,
        "System qubits",
        "logical qubits",
        ResourceCategory.PROBLEM,
        "Logical system-register qubits in a block-encoding model.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.BLOCK_ENCODING_ANCILLA_QUBITS,
        "Block-encoding ancilla qubits",
        "logical qubits",
        ResourceCategory.ALGORITHM,
        "Ancilla and workspace qubits used by a block encoding.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PREPARE_COST_TOFFOLI,
        "PREPARE cost",
        "Toffoli gates per call",
        ResourceCategory.ALGORITHM,
        "Toffoli cost of one PREPARE or inverse-PREPARE call.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.SELECT_COST_TOFFOLI,
        "SELECT cost",
        "Toffoli gates per call",
        ResourceCategory.ALGORITHM,
        "Toffoli cost of one SELECT or oracle call.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.REFLECTION_COST_TOFFOLI,
        "Reflection cost",
        "Toffoli gates per walk",
        ResourceCategory.ALGORITHM,
        "Toffoli cost of the reflection used by one qubitized walk.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.QPE_REGISTER_QUBITS,
        "QPE register qubits",
        "logical qubits",
        ResourceCategory.ALGORITHM,
        "Optional phase-readout qubits used by an explicit QPE routine.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.WALK_COST_TOFFOLI,
        "Walk cost",
        "Toffoli gates per walk",
        ResourceCategory.ALGORITHM,
        "Toffoli cost of one qubitized walk operator call.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.TARGET_PRECISION,
        "Target precision",
        "energy",
        ResourceCategory.ALGORITHM,
        "Total energy precision budget requested for an algorithm estimate.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.REPRESENTATION_ERROR,
        "Representation error",
        "energy",
        ResourceCategory.ALGORITHM,
        "Energy error budget consumed before phase-estimation sampling.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.ALGORITHMIC_PRECISION,
        "Algorithmic precision",
        "energy",
        ResourceCategory.ALGORITHM,
        "Energy precision budget left for QPE after representation error.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.EFFECTIVE_LAMBDA_NORM,
        "Effective Hamiltonian normalization",
        "energy",
        ResourceCategory.ALGORITHM,
        "Hamiltonian normalization after algorithmic weight reduction.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.TROTTER_STEPS_PER_SAMPLE,
        "Trotter steps per sample",
        "steps / sample",
        ResourceCategory.ALGORITHM,
        "Product-formula steps used for one sampled time-evolution segment.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.TROTTER_SAMPLES,
        "Trotter samples",
        "samples",
        ResourceCategory.ALGORITHM,
        "Number of sampled time points or signal-processing shots.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.UNITARY_WEIGHT_FACTOR,
        "Unitary weight factor",
        "multiplier",
        ResourceCategory.ALGORITHM,
        "Multiplicative Hamiltonian-weight reduction applied before QPE.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.RANDOMIZED_COMPILATION_FACTOR,
        "Randomized compilation factor",
        "multiplier",
        ResourceCategory.ALGORITHM,
        "Multiplicative product-formula cost factor from randomized evolution.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.ROTATION_SYNTHESIS_T_GATES,
        "Rotation synthesis cost",
        "T gates / rotation",
        ResourceCategory.ALGORITHM,
        "T gates used to synthesize one Pauli rotation.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.QPE_ITERATIONS,
        "QPE iterations",
        "iterations",
        ResourceCategory.ALGORITHM,
        "Number of QPE walk or time-evolution calls.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PAULI_ROTATIONS,
        "Pauli rotations",
        "rotations",
        ResourceCategory.LOGICAL,
        "Pauli rotations used by product-formula time evolution.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LOGICAL_QUBITS,
        "Logical qubits",
        "logical qubits",
        ResourceCategory.LOGICAL,
        "Logical data, ancilla, and algorithm workspace qubits.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LOGICAL_DEPTH,
        "Logical depth",
        "logical layers",
        ResourceCategory.LOGICAL,
        "Logical circuit-depth proxy after algorithmic repetition factors.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LOGICAL_SPACETIME_VOLUME,
        "Logical space-time volume",
        "logical qubit-layers",
        ResourceCategory.LOGICAL,
        "Logical qubits multiplied by logical-depth proxy.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.NON_CLIFFORD_COUNT,
        "Non-Clifford count",
        "gates",
        ResourceCategory.LOGICAL,
        "T, Toffoli, or equivalent non-Clifford workload.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.T_GATES,
        "T gates",
        "T gates",
        ResourceCategory.LOGICAL,
        "T-gate or T-equivalent count when it differs from the generic workload.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.MULTI_QUBIT_GATES,
        "Multi-qubit gates",
        "gates",
        ResourceCategory.LOGICAL,
        "Multi-qubit gate-count proxy.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_QUBITS,
        "Physical qubits",
        "physical qubits",
        ResourceCategory.PHYSICAL,
        "Physical qubits after logical-qubit overhead and factory allocation.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.RUNTIME_SECONDS,
        "Runtime",
        "seconds",
        ResourceCategory.PHYSICAL,
        "Wall-clock runtime proxy under the selected architecture model.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        "Physical qubit-seconds",
        "physical qubit-seconds",
        ResourceCategory.PHYSICAL,
        "Physical qubits multiplied by runtime proxy.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL,
        "Physical qubits per logical qubit",
        "physical qubits / logical qubit",
        ResourceCategory.ARCHITECTURE,
        "Physical overhead used to lift logical qubits to hardware qubits.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS,
        "Logical cycle time",
        "seconds",
        ResourceCategory.ARCHITECTURE,
        "Duration of one logical layer or logical error-correction cycle.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.FACTORY_QUBITS,
        "Factory qubits",
        "physical qubits",
        ResourceCategory.ARCHITECTURE,
        "Physical qubits reserved for magic-state factories or equivalents.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.NON_CLIFFORD_THROUGHPUT_PER_SECOND,
        "Non-Clifford throughput",
        "gates / second",
        ResourceCategory.ARCHITECTURE,
        "Sustainable non-Clifford throughput from factories or hardware.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.CODE_DISTANCE,
        "Code distance",
        "distance",
        ResourceCategory.ARCHITECTURE,
        "Surface-code distance used to lift logical resources to hardware.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
        "Physical cycle time",
        "seconds",
        ResourceCategory.ARCHITECTURE,
        "Duration of one physical error-correction cycle.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL_FACTOR,
        "Patch qubit factor",
        "physical qubits / distance^2",
        ResourceCategory.ARCHITECTURE,
        "Constant multiplying code_distance^2 for one logical qubit patch.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.LOGICAL_CYCLE_FACTOR,
        "Logical cycle factor",
        "physical cycles / distance",
        ResourceCategory.ARCHITECTURE,
        "Constant multiplying code distance to model one logical cycle.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.FACTORY_COUNT,
        "Factory count",
        "factories",
        ResourceCategory.ARCHITECTURE,
        "Number of parallel non-Clifford factories.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.PHYSICAL_QUBITS_PER_FACTORY,
        "Factory size",
        "physical qubits / factory",
        ResourceCategory.ARCHITECTURE,
        "Physical qubits reserved for one non-Clifford factory.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.FACTORY_CYCLES_PER_NON_CLIFFORD,
        "Factory cycles per non-Clifford",
        "logical cycles / output",
        ResourceCategory.ARCHITECTURE,
        "Logical cycles required by one factory to produce a non-Clifford resource.",
    ),
)

_SPECS_BY_QUANTITY = {spec.quantity: spec for spec in RESOURCE_QUANTITY_SPECS}

RESOURCE_REVIEW_PROFILES: tuple[ResourceQuantityProfile, ...] = (
    ResourceQuantityProfile(
        ResourceReviewProfile.HAMILTONIAN_QPE_WORKLOAD,
        "Hamiltonian QPE workload",
        "Problem and algorithm quantities that drive Hamiltonian phase estimation.",
        (
            ResourceQuantity.N_QUBITS,
            ResourceQuantity.N_PAULI_TERMS,
            ResourceQuantity.LAMBDA_NORM,
            ResourceQuantity.WALK_COST_TOFFOLI,
            ResourceQuantity.QPE_REGISTER_QUBITS,
            ResourceQuantity.REPRESENTATION_ERROR,
            ResourceQuantity.ALGORITHMIC_PRECISION,
        ),
    ),
    ResourceQuantityProfile(
        ResourceReviewProfile.TROTTER_QPE_WORKLOAD,
        "Trotter QPE workload",
        "Product-formula and weight-reduction quantities that drive Trotter QPE.",
        (
            ResourceQuantity.N_QUBITS,
            ResourceQuantity.N_PAULI_TERMS,
            ResourceQuantity.LAMBDA_NORM,
            ResourceQuantity.EFFECTIVE_LAMBDA_NORM,
            ResourceQuantity.TROTTER_STEPS_PER_SAMPLE,
            ResourceQuantity.TROTTER_SAMPLES,
            ResourceQuantity.UNITARY_WEIGHT_FACTOR,
            ResourceQuantity.RANDOMIZED_COMPILATION_FACTOR,
            ResourceQuantity.ROTATION_SYNTHESIS_T_GATES,
            ResourceQuantity.TARGET_PRECISION,
            ResourceQuantity.ALGORITHMIC_PRECISION,
        ),
    ),
    ResourceQuantityProfile(
        ResourceReviewProfile.FTQC_LOGICAL_OUTCOMES,
        "FTQC logical outcomes",
        "Architecture-independent logical quantities used to compare FTQC algorithms.",
        (
            ResourceQuantity.QPE_ITERATIONS,
            ResourceQuantity.LOGICAL_QUBITS,
            ResourceQuantity.LOGICAL_DEPTH,
            ResourceQuantity.LOGICAL_SPACETIME_VOLUME,
            ResourceQuantity.NON_CLIFFORD_COUNT,
        ),
    ),
    ResourceQuantityProfile(
        ResourceReviewProfile.FTQC_PHYSICAL_OUTCOMES,
        "FTQC physical outcomes",
        "Physical proxy quantities used to compare candidates under one architecture model.",
        (
            ResourceQuantity.LOGICAL_QUBITS,
            ResourceQuantity.NON_CLIFFORD_COUNT,
            ResourceQuantity.PHYSICAL_QUBITS,
            ResourceQuantity.RUNTIME_SECONDS,
            ResourceQuantity.PHYSICAL_QUBIT_SECONDS,
        ),
    ),
    ResourceQuantityProfile(
        ResourceReviewProfile.SURFACE_CODE_ARCHITECTURE,
        "Surface-code architecture",
        "Surface-code assumptions that should accompany a physical resource proxy.",
        (
            ResourceQuantity.CODE_DISTANCE,
            ResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS,
            ResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL_FACTOR,
            ResourceQuantity.LOGICAL_CYCLE_FACTOR,
            ResourceQuantity.FACTORY_COUNT,
            ResourceQuantity.PHYSICAL_QUBITS_PER_FACTORY,
            ResourceQuantity.FACTORY_CYCLES_PER_NON_CLIFFORD,
        ),
    ),
)

_PROFILES_BY_KEY = {profile.profile: profile for profile in RESOURCE_REVIEW_PROFILES}


def iter_resource_quantity_specs() -> tuple[ResourceQuantitySpec, ...]:
    """Return the canonical resource quantity specifications.

    Returns:
        tuple[ResourceQuantitySpec, ...]: Quantity specifications in a
            reader-friendly order from problem inputs to physical outputs.
    """
    return RESOURCE_QUANTITY_SPECS


def iter_resource_review_profiles() -> tuple[ResourceQuantityProfile, ...]:
    """Return the recommended resource review profiles.

    Returns:
        tuple[ResourceQuantityProfile, ...]: Quantity profiles ordered from
            algorithm inputs to physical and architecture review surfaces.
    """
    return RESOURCE_REVIEW_PROFILES


def describe_resource_quantity(
    quantity: str | ResourceQuantity,
) -> ResourceQuantitySpec:
    """Return metadata for one resource quantity.

    Args:
        quantity (str | ResourceQuantity): Quantity key or enum value.

    Returns:
        ResourceQuantitySpec: Metadata describing the quantity.

    Raises:
        ValueError: If ``quantity`` is not a known resource quantity.
    """
    normalized = _normalize_resource_quantity(quantity)
    return _SPECS_BY_QUANTITY[normalized]


def describe_resource_review_profile(
    profile: str | ResourceReviewProfile,
) -> ResourceQuantityProfile:
    """Return a recommended resource review profile.

    Args:
        profile (str | ResourceReviewProfile): Profile key or enum value.

    Returns:
        ResourceQuantityProfile: Recommended profile metadata and quantity set.

    Raises:
        ValueError: If ``profile`` is not a known resource review profile.
    """
    normalized = _normalize_resource_review_profile(profile)
    return _PROFILES_BY_KEY[normalized]


def compare_resource_values(
    baseline: _ResourceValuesInput,
    candidate: _ResourceValuesInput,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
) -> tuple[ResourceComparisonRow, ...]:
    """Compare canonical quantities between two value providers.

    Args:
        baseline (_ResourceValuesInput): Reference resource values. Accepts an
            object exposing ``resource_values()``, a mapping keyed by canonical
            quantities, or a logical Qamomile ``ResourceEstimate``.
        candidate (_ResourceValuesInput): Candidate resource values. Accepts
            the same shapes as ``baseline``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            compare. Defaults to nonzero-baseline quantities exposed by both
            inputs, ordered by the canonical quantity catalog.

    Returns:
        tuple[ResourceComparisonRow, ...]: Comparison rows containing baseline
            values, candidate values, ratios, and fractional reductions.

    Raises:
        ValueError: If a requested quantity is missing from either input or if
            a baseline value is exactly zero.
        TypeError: If either input cannot be interpreted as resource values.
    """
    baseline_values = _coerce_resource_values(baseline)
    candidate_values = _coerce_resource_values(candidate)
    selected = _normalize_comparison_quantities(
        baseline_values,
        candidate_values,
        quantities,
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
        spec = describe_resource_quantity(quantity)
        rows.append(
            ResourceComparisonRow(
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


def pareto_resource_values(
    candidates: _ResourceCandidatesInput,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
) -> tuple[ResourceParetoRow, ...]:
    """Mark Pareto-frontier candidates across resource values.

    Every selected quantity is treated as a resource cost where smaller is
    better. A candidate dominates another candidate only when every selected
    quantity is provably no larger and at least one quantity is provably
    smaller. Symbolic comparisons that cannot be decided keep both candidates
    on the frontier for later review.

    Args:
        candidates (_ResourceCandidatesInput): Candidate labels paired with
            resource values. Accepts an ordered mapping or a tuple of
            ``(label, provider)`` pairs. Providers follow the same contract as
            ``compare_resource_values``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            use for dominance checks. Defaults to the canonical intersection
            exposed by all candidates.

    Returns:
        tuple[ResourceParetoRow, ...]: Candidate rows in input order, annotated
            with frontier and dominator metadata.

    Raises:
        ValueError: If fewer than two candidates are supplied, labels repeat,
            selected quantities are missing, or no common quantity exists.
        TypeError: If a candidate provider cannot be interpreted as resource
            values.
    """
    normalized_candidates = _normalize_pareto_candidates(candidates)
    value_maps = tuple(
        _coerce_resource_values(provider) for _, provider in normalized_candidates
    )
    selected = _normalize_pareto_quantities(value_maps, quantities)
    rows: list[ResourceParetoRow] = []
    for index, (label, _) in enumerate(normalized_candidates):
        values = {quantity: value_maps[index][quantity] for quantity in selected}
        dominated_by = tuple(
            other_label
            for other_index, (other_label, _) in enumerate(normalized_candidates)
            if other_index != index
            and _pareto_dominates(value_maps[other_index], value_maps[index], selected)
        )
        rows.append(
            ResourceParetoRow(
                label=label,
                values=values,
                dominated_by=dominated_by,
            )
        )
    return tuple(rows)


def audit_resource_value_symbols(
    provider: _ResourceValuesInput,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
    include_resolved: bool = True,
) -> tuple[ResourceSymbolDependencyRow, ...]:
    """Audit free symbols in canonical resource values.

    Args:
        provider (_ResourceValuesInput): Resource values to inspect. Accepts
            an object exposing ``resource_values()``, a mapping keyed by
            canonical quantities, or a logical Qamomile ``ResourceEstimate``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            audit. Defaults to all canonical quantities exposed by
            ``provider`` in catalog order.
        include_resolved (bool): Whether rows with no free symbols should be
            returned. Defaults to True. Set to False to return only unresolved
            quantities.

    Returns:
        tuple[ResourceSymbolDependencyRow, ...]: Symbol-dependency rows with
            value expressions and sorted free-symbol names.

    Raises:
        ValueError: If a requested quantity is missing from ``provider``.
        TypeError: If ``provider`` cannot be interpreted as resource values.
    """
    values = _coerce_resource_values(provider)
    selected = _normalize_audit_quantities(values, quantities)

    rows = []
    for quantity in selected:
        value = sp.sympify(values[quantity])
        symbols = _sorted_free_symbol_names(value)
        if not include_resolved and not symbols:
            continue
        spec = describe_resource_quantity(quantity)
        rows.append(
            ResourceSymbolDependencyRow(
                quantity=quantity,
                value=value,
                symbols=symbols,
                label=spec.label,
                unit=spec.unit,
                category=spec.category,
            )
        )
    return tuple(rows)


def audit_resource_value_drivers(
    provider: _ResourceValuesInput,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
) -> tuple[ResourceSymbolDriverRow, ...]:
    """Audit which canonical resource quantities each symbol drives.

    This is the inverse view of ``audit_resource_value_symbols``. It is useful
    when a resource estimate remains symbolic and the caller wants to explain
    which problem, algorithm, or architecture assumptions affect each reported
    quantity.

    Args:
        provider (_ResourceValuesInput): Resource values to inspect. Accepts
            an object exposing ``resource_values()``, a mapping keyed by
            canonical quantities, or a logical Qamomile ``ResourceEstimate``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            audit. Defaults to all canonical quantities exposed by
            ``provider`` in catalog order.

    Returns:
        tuple[ResourceSymbolDriverRow, ...]: Symbol-driver rows sorted by
            symbol name. Each row lists impacted quantities in canonical
            catalog order.

    Raises:
        ValueError: If a requested quantity is missing from ``provider``.
        TypeError: If ``provider`` cannot be interpreted as resource values.
    """
    values = _coerce_resource_values(provider)
    selected = _normalize_audit_quantities(values, quantities)
    impacted_by_symbol: dict[str, list[ResourceQuantity]] = {}
    for quantity in selected:
        for symbol in values[quantity].free_symbols:
            impacted_by_symbol.setdefault(str(symbol), []).append(quantity)

    rows = []
    for symbol in sorted(impacted_by_symbol):
        impacted = tuple(impacted_by_symbol[symbol])
        labels = tuple(
            describe_resource_quantity(quantity).label for quantity in impacted
        )
        categories = _distinct_categories(impacted)
        rows.append(
            ResourceSymbolDriverRow(
                symbol=symbol,
                quantities=impacted,
                labels=labels,
                categories=categories,
            )
        )
    return tuple(rows)


def evaluate_resource_values(
    provider: _ResourceValuesInput,
    substitutions: _ScenarioSubstitutions | None = None,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
    scenario: str = "scenario",
    require_resolved: bool = True,
) -> tuple[ResourceScenarioValueRow, ...]:
    """Evaluate canonical resource values under one scenario.

    Args:
        provider (_ResourceValuesInput): Resource values to evaluate. Accepts
            an object exposing ``resource_values()``, a mapping keyed by
            canonical quantities, or a logical Qamomile ``ResourceEstimate``.
        substitutions (_ScenarioSubstitutions | None): Scenario substitutions
            keyed by symbol name or ``sp.Symbol``. Defaults to None, meaning
            no substitutions are applied.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            evaluate. Defaults to all canonical quantities exposed by
            ``provider`` in catalog order.
        scenario (str): Scenario label included in every returned row.
            Defaults to ``"scenario"``.
        require_resolved (bool): Whether unresolved symbols after
            substitution should raise an error. Defaults to True.

    Returns:
        tuple[ResourceScenarioValueRow, ...]: Evaluated resource rows.

    Raises:
        ValueError: If a requested quantity is missing from ``provider`` or if
            ``require_resolved`` is True and a row remains symbolic.
        TypeError: If ``provider`` or a substitution value cannot be
            interpreted as a SymPy expression.
    """
    values = _coerce_resource_values(provider)
    selected = _normalize_audit_quantities(values, quantities)
    return _evaluate_resource_values_for_scenario(
        values,
        selected,
        scenario,
        substitutions or {},
        require_resolved=require_resolved,
    )


def evaluate_resource_value_scenarios(
    provider: _ResourceValuesInput,
    scenarios: Mapping[str, _ScenarioSubstitutions],
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
    require_resolved: bool = True,
) -> tuple[ResourceScenarioValueRow, ...]:
    """Evaluate canonical resource values under multiple scenarios.

    Args:
        provider (_ResourceValuesInput): Resource values to evaluate. Accepts
            an object exposing ``resource_values()``, a mapping keyed by
            canonical quantities, or a logical Qamomile ``ResourceEstimate``.
        scenarios (Mapping[str, _ScenarioSubstitutions]): Ordered mapping
            from scenario label to substitutions keyed by symbol name or
            ``sp.Symbol``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            evaluate. Defaults to all canonical quantities exposed by
            ``provider`` in catalog order.
        require_resolved (bool): Whether unresolved symbols after
            substitution should raise an error. Defaults to True.

    Returns:
        tuple[ResourceScenarioValueRow, ...]: Scenario rows in input scenario
            order and canonical quantity order.

    Raises:
        ValueError: If no scenarios are supplied, a requested quantity is
            missing from ``provider``, or ``require_resolved`` is True and a
            row remains symbolic.
        TypeError: If ``provider`` or a substitution value cannot be
            interpreted as a SymPy expression.
    """
    if not scenarios:
        raise ValueError("scenarios must contain at least one scenario.")

    values = _coerce_resource_values(provider)
    selected = _normalize_audit_quantities(values, quantities)
    rows: list[ResourceScenarioValueRow] = []
    for scenario, substitutions in scenarios.items():
        rows.extend(
            _evaluate_resource_values_for_scenario(
                values,
                selected,
                scenario,
                substitutions,
                require_resolved=require_resolved,
            )
        )
    return tuple(rows)


def resource_values_from_estimate(
    estimate: ResourceEstimate,
    *,
    logical_depth: _SympyLike | None = None,
    non_clifford_count: _SympyLike | None = None,
) -> dict[str, sp.Expr]:
    """Convert a logical resource estimate to canonical resource values.

    Args:
        estimate (ResourceEstimate): Logical Qamomile resource estimate.
        logical_depth (sp.Expr | int | float | None): Optional logical-depth
            proxy. Defaults to ``estimate.gates.total``.
        non_clifford_count (sp.Expr | int | float | None): Optional
            non-Clifford workload. Defaults to ``estimate.gates.t_gates +
            estimate.gates.multi_qubit``.

    Returns:
        dict[str, sp.Expr]: Canonical logical resource values suitable for
            ``compare_resource_values``.

    Raises:
        TypeError: If ``estimate`` is not a ``ResourceEstimate`` or an
            override cannot be converted to a SymPy expression.
        ValueError: If an override is provably negative.
    """
    if not isinstance(estimate, ResourceEstimate):
        raise TypeError("estimate must be a ResourceEstimate instance.")

    depth_expr = (
        estimate.gates.total
        if logical_depth is None
        else _as_expr(logical_depth, "logical_depth")
    )
    non_clifford_expr = (
        sp.simplify(estimate.gates.t_gates + estimate.gates.multi_qubit)
        if non_clifford_count is None
        else _as_expr(non_clifford_count, "non_clifford_count")
    )
    _validate_nonnegative(depth_expr, "logical_depth")
    _validate_nonnegative(non_clifford_expr, "non_clifford_count")

    values = {
        "logical_qubits": estimate.qubits,
        "logical_depth": sp.simplify(depth_expr),
        "logical_spacetime_volume": sp.simplify(estimate.qubits * depth_expr),
        "non_clifford_count": sp.simplify(non_clifford_expr),
        "t_gates": estimate.gates.t_gates,
        "multi_qubit_gates": estimate.gates.multi_qubit,
        "pauli_rotations": estimate.gates.rotation_gates,
    }
    if "qpe_iterations" in estimate.gates.oracle_calls:
        values["qpe_iterations"] = estimate.gates.oracle_calls["qpe_iterations"]
    return values


def _coerce_resource_values(
    provider: _ResourceValuesInput,
) -> dict[ResourceQuantity, sp.Expr]:
    """Return normalized resource values from any supported provider.

    Args:
        provider (_ResourceValuesInput): Provider to normalize.

    Returns:
        dict[ResourceQuantity, sp.Expr]: Resource values keyed by enum.

    Raises:
        TypeError: If ``provider`` has no supported resource-value shape.
        ValueError: If any resource key is not known.
    """
    if isinstance(provider, ResourceEstimate):
        return _normalize_resource_values(resource_values_from_estimate(provider))
    if isinstance(provider, Mapping):
        return _normalize_resource_values(provider)
    resource_values = getattr(provider, "resource_values", None)
    if callable(resource_values):
        return _normalize_resource_values(resource_values())
    raise TypeError(
        "resource value providers must expose resource_values(), be a mapping, "
        "or be a ResourceEstimate."
    )


def _normalize_resource_values(
    values: Mapping[str | ResourceQuantity, _SympyLike],
) -> dict[ResourceQuantity, sp.Expr]:
    """Normalize resource-value dictionary keys.

    Args:
        values (Mapping[str | ResourceQuantity, _SympyLike]): Resource values
            keyed by canonical strings or enum values.

    Returns:
        dict[ResourceQuantity, sp.Expr]: Resource values keyed by enum.

    Raises:
        ValueError: If any key is not a known resource quantity.
    """
    normalized: dict[ResourceQuantity, sp.Expr] = {}
    for quantity, value in values.items():
        normalized_quantity = _normalize_resource_quantity(quantity)
        normalized[normalized_quantity] = _as_expr(value, normalized_quantity.value)
    return normalized


def _normalize_comparison_quantities(
    baseline_values: dict[ResourceQuantity, sp.Expr],
    candidate_values: dict[ResourceQuantity, sp.Expr],
    quantities: tuple[str | ResourceQuantity, ...] | None,
) -> tuple[ResourceQuantity, ...]:
    """Normalize comparison quantity selection.

    Args:
        baseline_values (dict[ResourceQuantity, sp.Expr]): Baseline values.
        candidate_values (dict[ResourceQuantity, sp.Expr]): Candidate values.
        quantities (tuple[str | ResourceQuantity, ...] | None): Requested
            quantities or None for the nonzero-baseline canonical
            intersection.

    Returns:
        tuple[ResourceQuantity, ...]: Normalized quantities.

    Raises:
        ValueError: If a requested quantity is absent from either value map.
    """
    if quantities is None:
        common = set(baseline_values) & set(candidate_values)
        return tuple(
            spec.quantity
            for spec in RESOURCE_QUANTITY_SPECS
            if spec.quantity in common
            and not baseline_values[spec.quantity].equals(sp.Integer(0))
        )

    normalized = tuple(
        _normalize_resource_quantity(quantity) for quantity in quantities
    )
    missing = [
        quantity.value
        for quantity in normalized
        if quantity not in baseline_values or quantity not in candidate_values
    ]
    if missing:
        raise ValueError(
            "Requested resource quantities are missing from the inputs: "
            + ", ".join(missing)
            + "."
        )
    return normalized


def _normalize_audit_quantities(
    values: dict[ResourceQuantity, sp.Expr],
    quantities: tuple[str | ResourceQuantity, ...] | None,
) -> tuple[ResourceQuantity, ...]:
    """Normalize symbol-audit quantity selection.

    Args:
        values (dict[ResourceQuantity, sp.Expr]): Resource values keyed by
            normalized quantity.
        quantities (tuple[str | ResourceQuantity, ...] | None): Requested
            quantities, or None for every exposed canonical quantity.

    Returns:
        tuple[ResourceQuantity, ...]: Normalized quantities in catalog order
            for default selection, or request order for explicit selection.

    Raises:
        ValueError: If a requested quantity is absent from ``values``.
    """
    if quantities is None:
        return tuple(
            spec.quantity for spec in RESOURCE_QUANTITY_SPECS if spec.quantity in values
        )

    normalized = tuple(
        _normalize_resource_quantity(quantity) for quantity in quantities
    )
    missing = [quantity.value for quantity in normalized if quantity not in values]
    if missing:
        raise ValueError(
            "Requested resource quantities are missing from the inputs: "
            + ", ".join(missing)
            + "."
        )
    return normalized


def _normalize_pareto_candidates(
    candidates: _ResourceCandidatesInput,
) -> tuple[tuple[str, _ResourceValuesInput], ...]:
    """Normalize Pareto candidate input to ordered label/provider pairs.

    Args:
        candidates (_ResourceCandidatesInput): Ordered mapping or tuple of
            ``(label, provider)`` pairs.

    Returns:
        tuple[tuple[str, _ResourceValuesInput], ...]: Normalized candidates.

    Raises:
        ValueError: If fewer than two candidates are supplied, labels repeat,
            or a label is empty.
    """
    if isinstance(candidates, Mapping):
        items = tuple(candidates.items())
    else:
        items = tuple(candidates)
    if len(items) < 2:
        raise ValueError("candidates must contain at least two entries.")

    normalized = tuple((str(label), provider) for label, provider in items)
    labels = tuple(label for label, _ in normalized)
    if any(not label for label in labels):
        raise ValueError("candidate labels must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique.")
    return normalized


def _normalize_pareto_quantities(
    value_maps: tuple[dict[ResourceQuantity, sp.Expr], ...],
    quantities: tuple[str | ResourceQuantity, ...] | None,
) -> tuple[ResourceQuantity, ...]:
    """Normalize Pareto quantity selection.

    Args:
        value_maps (tuple[dict[ResourceQuantity, sp.Expr], ...]): Resource
            values for each candidate.
        quantities (tuple[str | ResourceQuantity, ...] | None): Requested
            quantities, or None for the canonical intersection.

    Returns:
        tuple[ResourceQuantity, ...]: Normalized quantities.

    Raises:
        ValueError: If no common quantity exists or a requested quantity is
            missing from any candidate.
    """
    if quantities is None:
        common = set(value_maps[0])
        for values in value_maps[1:]:
            common &= set(values)
        normalized = tuple(
            spec.quantity for spec in RESOURCE_QUANTITY_SPECS if spec.quantity in common
        )
        if not normalized:
            raise ValueError("No common resource quantities are available.")
        return normalized

    normalized = tuple(
        _normalize_resource_quantity(quantity) for quantity in quantities
    )
    missing = [
        quantity.value
        for quantity in normalized
        if any(quantity not in values for values in value_maps)
    ]
    if missing:
        raise ValueError(
            "Requested Pareto quantities are missing from at least one "
            "candidate: " + ", ".join(missing) + "."
        )
    return normalized


def _evaluate_resource_values_for_scenario(
    values: dict[ResourceQuantity, sp.Expr],
    quantities: tuple[ResourceQuantity, ...],
    scenario: str,
    substitutions: _ScenarioSubstitutions,
    *,
    require_resolved: bool,
) -> tuple[ResourceScenarioValueRow, ...]:
    """Evaluate normalized resource values for one scenario.

    Args:
        values (dict[ResourceQuantity, sp.Expr]): Resource values keyed by
            normalized quantity.
        quantities (tuple[ResourceQuantity, ...]): Quantities to evaluate.
        scenario (str): Scenario label.
        substitutions (_ScenarioSubstitutions): Scenario substitutions keyed
            by symbol name or ``sp.Symbol``.
        require_resolved (bool): Whether remaining free symbols should raise.

    Returns:
        tuple[ResourceScenarioValueRow, ...]: Evaluated rows.

    Raises:
        ValueError: If ``require_resolved`` is True and any row remains
            symbolic.
        TypeError: If any substitution value cannot be converted to a SymPy
            expression.
    """
    normalized_substitutions = _normalize_substitutions(
        substitutions,
        tuple(values[quantity] for quantity in quantities),
    )
    rows = []
    unresolved = []
    for quantity in quantities:
        expression = values[quantity]
        value = sp.simplify(expression.subs(normalized_substitutions).doit())
        symbols = _sorted_free_symbol_names(value)
        spec = describe_resource_quantity(quantity)
        row = ResourceScenarioValueRow(
            scenario=str(scenario),
            quantity=quantity,
            expression=expression,
            value=value,
            symbols=symbols,
            label=spec.label,
            unit=spec.unit,
            category=spec.category,
        )
        rows.append(row)
        if symbols:
            unresolved.append(row)

    if require_resolved and unresolved:
        raise ValueError(_format_unresolved_scenario_values(unresolved))
    return tuple(rows)


def _pareto_dominates(
    challenger_values: dict[ResourceQuantity, sp.Expr],
    incumbent_values: dict[ResourceQuantity, sp.Expr],
    quantities: tuple[ResourceQuantity, ...],
) -> bool:
    """Return whether one candidate provably dominates another.

    Args:
        challenger_values (dict[ResourceQuantity, sp.Expr]): Candidate values
            for the possible dominator.
        incumbent_values (dict[ResourceQuantity, sp.Expr]): Candidate values
            for the possible dominated row.
        quantities (tuple[ResourceQuantity, ...]): Quantities used for
            dominance checks, where smaller is better.

    Returns:
        bool: True only when the challenger is no larger for every quantity
            and strictly smaller for at least one quantity.
    """
    strictly_better = False
    for quantity in quantities:
        difference = sp.simplify(
            incumbent_values[quantity] - challenger_values[quantity]
        )
        if difference.equals(0):
            continue
        if difference.is_positive is True:
            strictly_better = True
            continue
        return False
    return strictly_better


def _normalize_substitutions(
    substitutions: _ScenarioSubstitutions,
    expressions: tuple[sp.Expr, ...],
) -> dict[sp.Symbol, sp.Expr]:
    """Normalize scenario substitutions to SymPy keys and values.

    Args:
        substitutions (_ScenarioSubstitutions): Substitutions keyed by symbol
            name or ``sp.Symbol``.
        expressions (tuple[sp.Expr, ...]): Expressions whose free symbols
            should be matched when substitutions are keyed by string name.

    Returns:
        dict[sp.Symbol, sp.Expr]: Normalized SymPy substitutions.

    Raises:
        TypeError: If any substitution value cannot be converted to a SymPy
            expression.
    """
    symbols_by_name: dict[str, list[sp.Symbol]] = {}
    for expression in expressions:
        for symbol in expression.free_symbols:
            if not isinstance(symbol, sp.Symbol):
                continue
            symbols_by_name.setdefault(str(symbol), []).append(symbol)

    normalized: dict[sp.Symbol, sp.Expr] = {}
    for key, value in substitutions.items():
        value_expr = _as_expr(value, str(key))
        if isinstance(key, sp.Symbol):
            normalized[key] = value_expr
            continue
        matched_symbols = symbols_by_name.get(str(key), [sp.Symbol(str(key))])
        for symbol in matched_symbols:
            normalized[symbol] = value_expr
    return normalized


def _sorted_free_symbol_names(value: sp.Expr) -> tuple[str, ...]:
    """Return sorted free-symbol names for an expression.

    Args:
        value (sp.Expr): Expression to inspect.

    Returns:
        tuple[str, ...]: Free-symbol names sorted lexicographically.
    """
    return tuple(sorted(str(symbol) for symbol in value.free_symbols))


def _distinct_categories(
    quantities: tuple[ResourceQuantity, ...],
) -> tuple[ResourceCategory, ...]:
    """Return distinct categories for quantities in first-seen order.

    Args:
        quantities (tuple[ResourceQuantity, ...]): Quantities to inspect.

    Returns:
        tuple[ResourceCategory, ...]: Distinct categories in first-seen order.
    """
    categories: list[ResourceCategory] = []
    for quantity in quantities:
        category = describe_resource_quantity(quantity).category
        if category not in categories:
            categories.append(category)
    return tuple(categories)


def _format_unresolved_scenario_values(
    rows: list[ResourceScenarioValueRow],
) -> str:
    """Format unresolved scenario rows for an exception message.

    Args:
        rows (list[ResourceScenarioValueRow]): Rows that still have free
            symbols.

    Returns:
        str: Human-readable error message.
    """
    details = "; ".join(
        f"{row.scenario}:{row.quantity.value}({', '.join(row.symbols)})" for row in rows
    )
    return (
        "Scenario substitutions did not resolve every requested resource "
        f"quantity: {details}."
    )


def _normalize_resource_quantity(
    quantity: str | ResourceQuantity,
) -> ResourceQuantity:
    """Normalize one resource quantity key.

    Args:
        quantity (str | ResourceQuantity): Resource quantity key.

    Returns:
        ResourceQuantity: Normalized quantity enum.

    Raises:
        ValueError: If ``quantity`` is not a known resource quantity.
    """
    try:
        return ResourceQuantity(quantity)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ResourceQuantity)
        raise ValueError(
            f"Unknown resource quantity {quantity!r}; valid: {valid}."
        ) from exc


def _normalize_resource_review_profile(
    profile: str | ResourceReviewProfile,
) -> ResourceReviewProfile:
    """Normalize one resource review profile key.

    Args:
        profile (str | ResourceReviewProfile): Resource review profile key.

    Returns:
        ResourceReviewProfile: Normalized profile enum.

    Raises:
        ValueError: If ``profile`` is not a known resource review profile.
    """
    try:
        return ResourceReviewProfile(profile)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ResourceReviewProfile)
        raise ValueError(
            f"Unknown resource review profile {profile!r}; valid: {valid}."
        ) from exc


def _as_expr(value: _SympyLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    Args:
        value (sp.Expr | int | float): Value to convert.
        name (str): Field name used in error messages.

    Returns:
        sp.Expr: Converted SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be converted by SymPy.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


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
