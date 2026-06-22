"""Define canonical FTQC resource quantities for symbolic estimators."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol

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
        WALK_COST_TOFFOLI: Toffoli cost of one qubitized walk.
        QPE_ITERATIONS: Number of phase-estimation walk or time-evolution
            calls.
        LOGICAL_QUBITS: Logical qubits required by an algorithm.
        LOGICAL_DEPTH: Logical-depth proxy.
        TOFFOLI_GATES: Toffoli or Toffoli-equivalent non-Clifford count.
        T_GATES: T-gate or T-equivalent count.
        CLIFFORD_GATES: Clifford gate count.
        PHYSICAL_QUBITS: Physical qubits under an architecture model.
        RUNTIME_SECONDS: Runtime proxy in seconds.
        PHYSICAL_QUBITS_PER_LOGICAL: Physical overhead per logical qubit.
        LOGICAL_CYCLE_TIME_SECONDS: Logical layer or cycle time.
        FACTORY_QUBITS: Physical qubits reserved for factories.
        TOFFOLI_THROUGHPUT_PER_SECOND: Sustainable non-Clifford throughput.

    Example:
        >>> FTQCResourceQuantity("lambda_norm")
        <FTQCResourceQuantity.LAMBDA_NORM: 'lambda_norm'>
    """

    N_SPIN_ORBITALS = "n_spin_orbitals"
    N_PAULI_TERMS = "n_pauli_terms"
    LAMBDA_NORM = "lambda_norm"
    MAX_LOCALITY = "max_locality"
    WALK_COST_TOFFOLI = "walk_cost_toffoli"
    QPE_ITERATIONS = "qpe_iterations"
    LOGICAL_QUBITS = "logical_qubits"
    LOGICAL_DEPTH = "logical_depth"
    TOFFOLI_GATES = "toffoli_gates"
    T_GATES = "t_gates"
    CLIFFORD_GATES = "clifford_gates"
    PHYSICAL_QUBITS = "physical_qubits"
    RUNTIME_SECONDS = "runtime_seconds"
    PHYSICAL_QUBITS_PER_LOGICAL = "physical_qubits_per_logical"
    LOGICAL_CYCLE_TIME_SECONDS = "logical_cycle_time_seconds"
    FACTORY_QUBITS = "factory_qubits"
    TOFFOLI_THROUGHPUT_PER_SECOND = "toffoli_throughput_per_second"


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
)

_SPECS_BY_QUANTITY = {spec.quantity: spec for spec in FTQC_RESOURCE_QUANTITY_SPECS}


def iter_ftqc_resource_quantity_specs() -> tuple[FTQCResourceQuantitySpec, ...]:
    """Return the canonical FTQC resource quantity specifications.

    Returns:
        tuple[FTQCResourceQuantitySpec, ...]: Quantity specifications in a
            reader-friendly order from problem inputs to physical outputs.
    """
    return FTQC_RESOURCE_QUANTITY_SPECS


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
) -> tuple[FTQCResourceComparisonRow, ...]:
    """Compare canonical FTQC quantities between two estimates.

    Args:
        baseline (SupportsFTQCResourceValues): Reference estimate, model, or
            summary exposing ``resource_values()``.
        candidate (SupportsFTQCResourceValues): Candidate estimate, model, or
            summary exposing ``resource_values()``.
        quantities (tuple[str | FTQCResourceQuantity, ...] | None): Quantities
            to compare. Defaults to the intersection of quantities exposed by
            both inputs, ordered by the canonical quantity catalog.

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
            "Requested FTQC resource quantities are missing from the inputs: "
            + ", ".join(missing)
            + "."
        )
    return normalized


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
