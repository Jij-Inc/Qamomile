"""Define canonical resource quantities for reports and comparisons."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol

import sympy as sp


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
        WALK_COST_TOFFOLI: Toffoli cost of one qubitized walk.
        QPE_ITERATIONS: Number of phase-estimation walk or time-evolution
            calls.
        LOGICAL_QUBITS: Logical qubits required by an algorithm.
        LOGICAL_DEPTH: Logical-depth proxy.
        NON_CLIFFORD_COUNT: T, Toffoli, or equivalent non-Clifford count.
        T_GATES: T-gate or T-equivalent count.
        MULTI_QUBIT_GATES: Multi-qubit gate-count proxy.
        PHYSICAL_QUBITS: Physical qubits under an architecture model.
        RUNTIME_SECONDS: Runtime proxy in seconds.
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
    WALK_COST_TOFFOLI = "walk_cost_toffoli"
    QPE_ITERATIONS = "qpe_iterations"
    LOGICAL_QUBITS = "logical_qubits"
    LOGICAL_DEPTH = "logical_depth"
    NON_CLIFFORD_COUNT = "non_clifford_count"
    T_GATES = "t_gates"
    MULTI_QUBIT_GATES = "multi_qubit_gates"
    PHYSICAL_QUBITS = "physical_qubits"
    RUNTIME_SECONDS = "runtime_seconds"
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
        ResourceQuantity.WALK_COST_TOFFOLI,
        "Walk cost",
        "Toffoli gates per walk",
        ResourceCategory.ALGORITHM,
        "Toffoli cost of one qubitized walk operator call.",
    ),
    ResourceQuantitySpec(
        ResourceQuantity.QPE_ITERATIONS,
        "QPE iterations",
        "iterations",
        ResourceCategory.ALGORITHM,
        "Number of QPE walk or time-evolution calls.",
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


def iter_resource_quantity_specs() -> tuple[ResourceQuantitySpec, ...]:
    """Return the canonical resource quantity specifications.

    Returns:
        tuple[ResourceQuantitySpec, ...]: Quantity specifications in a
            reader-friendly order from problem inputs to physical outputs.
    """
    return RESOURCE_QUANTITY_SPECS


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


def compare_resource_values(
    baseline: SupportsResourceValues,
    candidate: SupportsResourceValues,
    *,
    quantities: tuple[str | ResourceQuantity, ...] | None = None,
) -> tuple[ResourceComparisonRow, ...]:
    """Compare canonical quantities between two value providers.

    Args:
        baseline (SupportsResourceValues): Reference object exposing
            ``resource_values()``.
        candidate (SupportsResourceValues): Candidate object exposing
            ``resource_values()``.
        quantities (tuple[str | ResourceQuantity, ...] | None): Quantities to
            compare. Defaults to the intersection of quantities exposed by both
            inputs, ordered by the canonical quantity catalog.

    Returns:
        tuple[ResourceComparisonRow, ...]: Comparison rows containing baseline
            values, candidate values, ratios, and fractional reductions.

    Raises:
        ValueError: If a requested quantity is missing from either input or if
            a baseline value is exactly zero.
    """
    baseline_values = _normalize_resource_values(baseline.resource_values())
    candidate_values = _normalize_resource_values(candidate.resource_values())
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


def _normalize_resource_values(
    values: dict[str | ResourceQuantity, sp.Expr],
) -> dict[ResourceQuantity, sp.Expr]:
    """Normalize resource-value dictionary keys.

    Args:
        values (dict[str | ResourceQuantity, sp.Expr]): Resource values keyed
            by canonical strings or enum values.

    Returns:
        dict[ResourceQuantity, sp.Expr]: Resource values keyed by enum.

    Raises:
        ValueError: If any key is not a known resource quantity.
    """
    return {
        _normalize_resource_quantity(quantity): value
        for quantity, value in values.items()
    }


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
            quantities or None for the canonical intersection.

    Returns:
        tuple[ResourceQuantity, ...]: Normalized quantities.

    Raises:
        ValueError: If a requested quantity is absent from either value map.
    """
    if quantities is None:
        common = set(baseline_values) & set(candidate_values)
        return tuple(
            spec.quantity for spec in RESOURCE_QUANTITY_SPECS if spec.quantity in common
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
