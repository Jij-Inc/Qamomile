"""Fault-tolerant resource estimates for quantum chemistry algorithms."""

from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from qamomile.circuit.estimator.algorithmic.ftqc_resources import (
    FTQCResourceFormula,
    FTQCResourceQuantity,
    describe_ftqc_resource_quantity,
)
from qamomile.circuit.estimator.gate_counter import GateCount
from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

if TYPE_CHECKING:
    import qamomile.observable as qm_o

_SympyLike = sp.Expr | int | float
_CoefficientLike = _SympyLike | complex
_OPENFERMION_PAULI_LABELS = {"X": "X", "Y": "Y", "Z": "Z"}


class ChemistryQPEMethod(enum.StrEnum):
    """Select a chemistry Hamiltonian representation for QPE estimates.

    Attributes:
        SPARSE: Sparse Pauli-LCU representation.
        SINGLE_FACTORIZATION: Single-factorized two-electron tensor.
        DOUBLE_FACTORIZATION: Double-factorized two-electron tensor.
        TENSOR_HYPERCONTRACTION: Tensor-hypercontracted representation.
        SYMMETRY_COMPRESSED_DF: Symmetry-compressed double factorization.
        UNITARY_WEIGHT_CONCENTRATION: Early-FTQC unitary-weight concentration.

    Example:
        >>> ChemistryQPEMethod("tensor_hypercontraction")
        <ChemistryQPEMethod.TENSOR_HYPERCONTRACTION: 'tensor_hypercontraction'>
    """

    SPARSE = "sparse"
    SINGLE_FACTORIZATION = "single_factorization"
    DOUBLE_FACTORIZATION = "double_factorization"
    TENSOR_HYPERCONTRACTION = "tensor_hypercontraction"
    SYMMETRY_COMPRESSED_DF = "symmetry_compressed_df"
    UNITARY_WEIGHT_CONCENTRATION = "unitary_weight_concentration"


@dataclass(frozen=True)
class FTQCReference:
    """Describe a research source behind an FTQC estimate.

    Attributes:
        key (str): Stable short key for the source, usually an arXiv
            identifier or paper nickname.
        title (str): Full title of the source.
        url (str): Persistent source URL.
        note (str): Short note explaining which model assumption the source
            supports.

    Example:
        >>> reference = FTQCReference(
        ...     key="arXiv:2403.03502",
        ...     title="Reducing the runtime of fault-tolerant quantum simulations...",
        ...     url="https://arxiv.org/abs/2403.03502",
        ...     note="Introduces symmetry-compressed double factorization.",
        ... )
        >>> reference.to_dict()["key"]
        'arXiv:2403.03502'
    """

    key: str
    title: str
    url: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize the reference to string metadata.

        Returns:
            dict[str, str]: JSON-friendly reference metadata.
        """
        return {
            "key": self.key,
            "title": self.title,
            "url": self.url,
            "note": self.note,
        }


_LOW_RANK_QUBITIZATION_REFERENCE = FTQCReference(
    key="arXiv:1902.02134",
    title=(
        "Qubitization of Arbitrary Basis Quantum Chemistry Leveraging "
        "Sparsity and Low Rank Factorization"
    ),
    url="https://arxiv.org/abs/1902.02134",
    note=(
        "Provides sparse and low-rank qubitized chemistry cost models that "
        "motivate representation-specific logical-resource scaling."
    ),
)
_SCDF_REFERENCE = FTQCReference(
    key="arXiv:2403.03502",
    title=(
        "Reducing the runtime of fault-tolerant quantum simulations in "
        "chemistry through symmetry-compressed double factorization"
    ),
    url="https://arxiv.org/abs/2403.03502",
    note=(
        "Introduces symmetry-compressed double factorization and reports "
        "Hamiltonian 1-norm and Toffoli-count reductions."
    ),
)
_SYMMETRY_SHIFT_TENSOR_REFERENCE = FTQCReference(
    key="arXiv:2412.01338",
    title=(
        "Simultaneously optimizing symmetry shifts and tensor factorizations "
        "for cost-efficient Fault-Tolerant Quantum Simulations of electronic "
        "Hamiltonians"
    ),
    url="https://arxiv.org/abs/2412.01338",
    note=(
        "Optimizes symmetry shifts together with tensor factorizations to "
        "reduce block-encoding normalization."
    ),
)
_UWC_REFERENCE = FTQCReference(
    key="arXiv:2603.22778",
    title=(
        "Enabling Chemically Accurate Quantum Phase Estimation in the Early "
        "Fault-Tolerant Regime"
    ),
    url="https://arxiv.org/abs/2603.22778",
    note=(
        "Models early-FTQC single-ancilla Trotter QPE with unitary weight "
        "concentration and constrained physical-qubit budgets."
    ),
)
_METHOD_REFERENCES: dict[ChemistryQPEMethod, tuple[FTQCReference, ...]] = {
    ChemistryQPEMethod.SPARSE: (_LOW_RANK_QUBITIZATION_REFERENCE,),
    ChemistryQPEMethod.SINGLE_FACTORIZATION: (_LOW_RANK_QUBITIZATION_REFERENCE,),
    ChemistryQPEMethod.DOUBLE_FACTORIZATION: (_LOW_RANK_QUBITIZATION_REFERENCE,),
    ChemistryQPEMethod.TENSOR_HYPERCONTRACTION: (
        _LOW_RANK_QUBITIZATION_REFERENCE,
        _SYMMETRY_SHIFT_TENSOR_REFERENCE,
    ),
    ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF: (
        _LOW_RANK_QUBITIZATION_REFERENCE,
        _SCDF_REFERENCE,
        _SYMMETRY_SHIFT_TENSOR_REFERENCE,
    ),
    ChemistryQPEMethod.UNITARY_WEIGHT_CONCENTRATION: (_UWC_REFERENCE,),
}


def references_for_chemistry_qpe_method(
    method: str | ChemistryQPEMethod,
) -> tuple[FTQCReference, ...]:
    """Return default research references for a chemistry QPE method.

    Args:
        method (str | ChemistryQPEMethod): Chemistry QPE method to inspect.

    Returns:
        tuple[FTQCReference, ...]: Default references associated with the
            method's symbolic resource model.

    Raises:
        ValueError: If ``method`` is not a known chemistry QPE method.
    """
    return _METHOD_REFERENCES[_normalize_method(method)]


@dataclass(frozen=True)
class FTQCCostModel:
    """Describe an architecture-level FTQC cost model.

    Attributes:
        physical_qubits_per_logical (sp.Expr | int | float): Physical qubit
            overhead for one data or ancilla logical qubit.
        logical_cycle_time_seconds (sp.Expr | int | float): Time for one
            logical layer or logical cycle, in seconds.
        factory_qubits (sp.Expr | int | float): Physical qubits reserved for
            magic-state factories or equivalent non-Clifford support.
        toffoli_throughput_per_second (sp.Expr | int | float): Sustainable
            Toffoli or non-Clifford-equivalent throughput. Runtime uses the
            larger of logical-depth time and factory-throughput time.

    Raises:
        ValueError: If any positive-valued field is non-positive or if
            ``factory_qubits`` is negative.

    Example:
        >>> model = FTQCCostModel(
        ...     physical_qubits_per_logical=1000,
        ...     logical_cycle_time_seconds=1e-6,
        ...     factory_qubits=20000,
        ...     toffoli_throughput_per_second=1e5,
        ... )
        >>> model.physical_qubits_per_logical
        1000
    """

    physical_qubits_per_logical: _SympyLike = field(
        default_factory=lambda: sp.Symbol(
            "physical_qubits_per_logical",
            positive=True,
        )
    )
    logical_cycle_time_seconds: _SympyLike = field(
        default_factory=lambda: sp.Symbol("logical_cycle_time", positive=True)
    )
    factory_qubits: _SympyLike = field(
        default_factory=lambda: sp.Symbol("factory_qubits", nonnegative=True)
    )
    toffoli_throughput_per_second: _SympyLike = field(
        default_factory=lambda: sp.Symbol("toffoli_throughput", positive=True)
    )

    def __post_init__(self) -> None:
        """Validate cost-model fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued field is non-positive or if
                ``factory_qubits`` is negative.
        """
        _validate_positive(
            _as_expr(
                self.physical_qubits_per_logical,
                "physical_qubits_per_logical",
            ),
            "physical_qubits_per_logical",
        )
        _validate_positive(
            _as_expr(
                self.logical_cycle_time_seconds,
                "logical_cycle_time_seconds",
            ),
            "logical_cycle_time_seconds",
        )
        _validate_nonnegative(
            _as_expr(self.factory_qubits, "factory_qubits"),
            "factory_qubits",
        )
        _validate_positive(
            _as_expr(
                self.toffoli_throughput_per_second,
                "toffoli_throughput_per_second",
            ),
            "toffoli_throughput_per_second",
        )

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return architecture knobs keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Architecture-model values.
        """
        return {
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL: _as_expr(
                self.physical_qubits_per_logical,
                "physical_qubits_per_logical",
            ),
            FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS: _as_expr(
                self.logical_cycle_time_seconds,
                "logical_cycle_time_seconds",
            ),
            FTQCResourceQuantity.FACTORY_QUBITS: _as_expr(
                self.factory_qubits,
                "factory_qubits",
            ),
            FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND: _as_expr(
                self.toffoli_throughput_per_second,
                "toffoli_throughput_per_second",
            ),
        }

    def physical_qubits_for(self, logical_qubits: _SympyLike) -> sp.Expr:
        """Lift logical-qubit count to physical qubits.

        Args:
            logical_qubits (sp.Expr | int | float): Logical qubit count to
                lift with this architecture model.

        Returns:
            sp.Expr: Physical qubits including data patches and factories.

        Raises:
            ValueError: If ``logical_qubits`` is negative.
        """
        logical_expr = _as_expr(logical_qubits, "logical_qubits")
        _validate_nonnegative(logical_expr, "logical_qubits")
        return sp.simplify(
            logical_expr
            * _as_expr(
                self.physical_qubits_per_logical,
                "physical_qubits_per_logical",
            )
            + _as_expr(self.factory_qubits, "factory_qubits")
        )

    def runtime_seconds_for(
        self,
        logical_depth: _SympyLike,
        non_clifford_gates: _SympyLike,
    ) -> sp.Expr:
        """Lift logical work to a wall-clock runtime proxy.

        Args:
            logical_depth (sp.Expr | int | float): Logical-depth proxy.
            non_clifford_gates (sp.Expr | int | float): Toffoli-equivalent or
                T-equivalent non-Clifford gate count served by factories.

        Returns:
            sp.Expr: Runtime proxy in seconds, as the maximum of logical-cycle
                time and factory-throughput time.

        Raises:
            ValueError: If either input is negative.
        """
        depth_expr = _as_expr(logical_depth, "logical_depth")
        non_clifford_expr = _as_expr(non_clifford_gates, "non_clifford_gates")
        _validate_nonnegative(depth_expr, "logical_depth")
        _validate_nonnegative(non_clifford_expr, "non_clifford_gates")
        return sp.simplify(
            sp.Max(
                depth_expr
                * _as_expr(
                    self.logical_cycle_time_seconds,
                    "logical_cycle_time_seconds",
                ),
                non_clifford_expr
                / _as_expr(
                    self.toffoli_throughput_per_second,
                    "toffoli_throughput_per_second",
                ),
            )
        )

    def lift_estimate(
        self,
        estimate: FTQCResourceEstimate,
    ) -> FTQCResourceEstimate:
        """Recompute physical resources for an existing logical estimate.

        Args:
            estimate (FTQCResourceEstimate): Estimate whose logical qubits,
                logical depth, and non-Clifford counts should be preserved
                while replacing physical architecture assumptions.

        Returns:
            FTQCResourceEstimate: Copy of ``estimate`` with new
                ``physical_qubits`` and ``runtime_seconds`` fields.
        """
        return estimate.with_cost_model(self)


@dataclass(frozen=True)
class SurfaceCodeDistanceBudget:
    """Choose a surface-code distance from a logical failure budget.

    This helper uses the explicit phenomenological proxy
    ``prefactor * (physical_error_rate / threshold_error_rate)**((d + 1) / 2)``
    for a distance-``d`` logical operation or cycle. It is intentionally a
    sizing model rather than a decoder-specific claim.

    Attributes:
        physical_error_rate (sp.Expr | int | float): Physical error rate
            below threshold.
        threshold_error_rate (sp.Expr | int | float): Error threshold used by
            the proxy model.
        target_logical_failure_probability (sp.Expr | int | float): Total
            logical failure budget.
        logical_operation_budget (sp.Expr | int | float): Number of logical
            operations or cycles sharing the total failure budget.
        prefactor (sp.Expr | int | float): Leading constant in the logical
            error proxy. Defaults to 0.1.

    Raises:
        ValueError: If any probability or operation count is non-positive, if
            ``physical_error_rate`` is not below ``threshold_error_rate``, or
            if ``prefactor`` is non-positive.

    Example:
        >>> budget = SurfaceCodeDistanceBudget(
        ...     physical_error_rate=1e-3,
        ...     threshold_error_rate=1e-2,
        ...     target_logical_failure_probability=1e-9,
        ...     logical_operation_budget=1000,
        ... )
        >>> budget.code_distance
        21
    """

    physical_error_rate: _SympyLike
    threshold_error_rate: _SympyLike
    target_logical_failure_probability: _SympyLike
    logical_operation_budget: _SympyLike
    prefactor: _SympyLike = sp.Float("0.1")

    def __post_init__(self) -> None:
        """Validate distance-budget fields after dataclass construction.

        Raises:
            ValueError: If any field violates the probability-budget model.
        """
        for name, expr in [
            ("physical_error_rate", self._physical_error_rate),
            ("threshold_error_rate", self._threshold_error_rate),
            (
                "target_logical_failure_probability",
                self._target_logical_failure_probability,
            ),
            ("logical_operation_budget", self._logical_operation_budget),
            ("prefactor", self._prefactor),
        ]:
            _validate_positive(expr, name)

        if (
            self._physical_error_rate / self._threshold_error_rate
        ).is_nonnegative is False:
            raise ValueError("physical_error_rate / threshold_error_rate is invalid.")

        ratio = _as_float(
            self._physical_error_rate / self._threshold_error_rate,
            "physical_error_rate / threshold_error_rate",
        )
        if ratio >= 1:
            raise ValueError("physical_error_rate must be below threshold_error_rate.")

    @property
    def logical_failure_probability_per_operation(self) -> sp.Expr:
        """Return the allocated logical failure budget per operation.

        Returns:
            sp.Expr: ``target_logical_failure_probability /
            logical_operation_budget``.
        """
        return sp.simplify(
            self._target_logical_failure_probability / self._logical_operation_budget
        )

    @property
    def code_distance(self) -> int:
        """Return the smallest odd distance satisfying the budget.

        Returns:
            int: Smallest odd positive surface-code distance whose proxy
                logical error rate does not exceed the allocated budget.
        """
        ratio = _as_float(
            self._physical_error_rate / self._threshold_error_rate,
            "physical_error_rate / threshold_error_rate",
        )
        per_operation = _as_float(
            self.logical_failure_probability_per_operation,
            "logical_failure_probability_per_operation",
        )
        prefactor = _as_float(self._prefactor, "prefactor")
        if per_operation >= prefactor:
            return 1
        raw_exponent = math.log(per_operation / prefactor) / math.log(ratio)
        exponent = max(1, math.ceil(raw_exponent - 1e-12))
        distance = 2 * exponent - 1
        while _as_float(
            self.logical_error_rate_for_distance(distance),
            "logical_error_rate",
        ) > per_operation * (1 + 1e-12):
            distance += 2
        return distance

    def logical_error_rate_for_distance(self, distance: _SympyLike) -> sp.Expr:
        """Estimate the logical error rate for a code distance.

        Args:
            distance (sp.Expr | int | float): Surface-code distance.

        Returns:
            sp.Expr: Logical error-rate proxy for one operation or cycle.

        Raises:
            ValueError: If ``distance`` is non-positive.
        """
        distance_expr = _as_expr(distance, "distance")
        _validate_positive(distance_expr, "distance")
        return sp.simplify(
            self._prefactor
            * (self._physical_error_rate / self._threshold_error_rate)
            ** ((distance_expr + 1) / 2)
        )

    def to_surface_code_cost_model(
        self,
        *,
        physical_cycle_time_seconds: _SympyLike,
        physical_qubits_per_logical_factor: _SympyLike = 2,
        logical_cycle_factor: _SympyLike = 1,
        factory_count: _SympyLike,
        physical_qubits_per_factory: _SympyLike,
        factory_cycles_per_toffoli: _SympyLike,
    ) -> SurfaceCodeCostModel:
        """Build a surface-code cost model using the selected distance.

        Args:
            physical_cycle_time_seconds (sp.Expr | int | float): Duration of
                one physical error-correction cycle.
            physical_qubits_per_logical_factor (sp.Expr | int | float):
                Constant multiplying ``code_distance**2`` for one logical
                patch. Defaults to 2.
            logical_cycle_factor (sp.Expr | int | float): Constant
                multiplying code distance and physical cycle time. Defaults to
                1.
            factory_count (sp.Expr | int | float): Number of parallel
                non-Clifford factories.
            physical_qubits_per_factory (sp.Expr | int | float): Physical
                qubits reserved for one factory.
            factory_cycles_per_toffoli (sp.Expr | int | float): Logical cycles
                needed by one factory to produce a Toffoli resource.

        Returns:
            SurfaceCodeCostModel: Cost model configured with ``code_distance``.
        """
        return SurfaceCodeCostModel(
            code_distance=self.code_distance,
            physical_cycle_time_seconds=physical_cycle_time_seconds,
            physical_qubits_per_logical_factor=physical_qubits_per_logical_factor,
            logical_cycle_factor=logical_cycle_factor,
            factory_count=factory_count,
            physical_qubits_per_factory=physical_qubits_per_factory,
            factory_cycles_per_toffoli=factory_cycles_per_toffoli,
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize the distance-budget model.

        Returns:
            dict[str, str]: JSON-friendly surface-code distance budget.
        """
        return {
            "physical_error_rate": str(self._physical_error_rate),
            "threshold_error_rate": str(self._threshold_error_rate),
            "target_logical_failure_probability": str(
                self._target_logical_failure_probability
            ),
            "logical_operation_budget": str(self._logical_operation_budget),
            "prefactor": str(self._prefactor),
            "logical_failure_probability_per_operation": str(
                self.logical_failure_probability_per_operation
            ),
            "code_distance": str(self.code_distance),
            "logical_error_rate": str(
                self.logical_error_rate_for_distance(self.code_distance)
            ),
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return distance-budget values keyed by canonical quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Error-budget and selected
                distance quantities.
        """
        return {
            FTQCResourceQuantity.PHYSICAL_ERROR_RATE: self._physical_error_rate,
            FTQCResourceQuantity.THRESHOLD_ERROR_RATE: self._threshold_error_rate,
            FTQCResourceQuantity.TARGET_LOGICAL_FAILURE_PROBABILITY: (
                self._target_logical_failure_probability
            ),
            FTQCResourceQuantity.LOGICAL_OPERATION_BUDGET: (
                self._logical_operation_budget
            ),
            FTQCResourceQuantity.CODE_DISTANCE: sp.Integer(self.code_distance),
            FTQCResourceQuantity.LOGICAL_ERROR_RATE: (
                self.logical_error_rate_for_distance(self.code_distance)
            ),
        }

    @property
    def _physical_error_rate(self) -> sp.Expr:
        """Return ``physical_error_rate`` as a SymPy expression.

        Returns:
            sp.Expr: Converted physical error rate.
        """
        return _as_expr(self.physical_error_rate, "physical_error_rate")

    @property
    def _threshold_error_rate(self) -> sp.Expr:
        """Return ``threshold_error_rate`` as a SymPy expression.

        Returns:
            sp.Expr: Converted threshold error rate.
        """
        return _as_expr(self.threshold_error_rate, "threshold_error_rate")

    @property
    def _target_logical_failure_probability(self) -> sp.Expr:
        """Return ``target_logical_failure_probability`` as an expression.

        Returns:
            sp.Expr: Converted target logical failure probability.
        """
        return _as_expr(
            self.target_logical_failure_probability,
            "target_logical_failure_probability",
        )

    @property
    def _logical_operation_budget(self) -> sp.Expr:
        """Return ``logical_operation_budget`` as a SymPy expression.

        Returns:
            sp.Expr: Converted logical operation budget.
        """
        return _as_expr(self.logical_operation_budget, "logical_operation_budget")

    @property
    def _prefactor(self) -> sp.Expr:
        """Return ``prefactor`` as a SymPy expression.

        Returns:
            sp.Expr: Converted logical-error prefactor.
        """
        return _as_expr(self.prefactor, "prefactor")


@dataclass(frozen=True)
class SurfaceCodeCostModel:
    """Lift surface-code architecture knobs into an FTQC cost model.

    This model keeps surface-code assumptions explicit while still producing
    the lower-level ``FTQCCostModel`` consumed by chemistry estimators.

    Attributes:
        code_distance (sp.Expr | int | float): Surface-code distance.
        physical_cycle_time_seconds (sp.Expr | int | float): Duration of one
            physical error-correction cycle, in seconds.
        physical_qubits_per_logical_factor (sp.Expr | int | float): Constant
            multiplying ``code_distance**2`` for one logical patch.
        logical_cycle_factor (sp.Expr | int | float): Constant multiplying
            code distance and physical cycle time for one logical cycle.
        factory_count (sp.Expr | int | float): Number of parallel
            non-Clifford factories.
        physical_qubits_per_factory (sp.Expr | int | float): Physical qubits
            reserved for one factory.
        factory_cycles_per_toffoli (sp.Expr | int | float): Logical cycles
            required for one factory to produce a Toffoli resource.

    Raises:
        ValueError: If any positive-valued field is non-positive or if
            ``physical_qubits_per_factory`` is negative.

    Example:
        >>> model = SurfaceCodeCostModel(
        ...     code_distance=15,
        ...     physical_cycle_time_seconds=1e-6,
        ...     factory_count=4,
        ...     physical_qubits_per_factory=10000,
        ...     factory_cycles_per_toffoli=5,
        ... )
        >>> model.to_cost_model().physical_qubits_per_logical
        450
    """

    code_distance: _SympyLike = field(
        default_factory=lambda: sp.Symbol("code_distance", positive=True)
    )
    physical_cycle_time_seconds: _SympyLike = field(
        default_factory=lambda: sp.Symbol(
            "physical_cycle_time_seconds",
            positive=True,
        )
    )
    physical_qubits_per_logical_factor: _SympyLike = 2
    logical_cycle_factor: _SympyLike = 1
    factory_count: _SympyLike = field(
        default_factory=lambda: sp.Symbol("factory_count", positive=True)
    )
    physical_qubits_per_factory: _SympyLike = field(
        default_factory=lambda: sp.Symbol(
            "physical_qubits_per_factory",
            nonnegative=True,
        )
    )
    factory_cycles_per_toffoli: _SympyLike = field(
        default_factory=lambda: sp.Symbol(
            "factory_cycles_per_toffoli",
            positive=True,
        )
    )

    def __post_init__(self) -> None:
        """Validate surface-code fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued field is non-positive or if
                ``physical_qubits_per_factory`` is negative.
        """
        for name, expr in [
            ("code_distance", self._code_distance),
            ("physical_cycle_time_seconds", self._physical_cycle_time_seconds),
            (
                "physical_qubits_per_logical_factor",
                self._physical_qubits_per_logical_factor,
            ),
            ("logical_cycle_factor", self._logical_cycle_factor),
            ("factory_count", self._factory_count),
            ("factory_cycles_per_toffoli", self._factory_cycles_per_toffoli),
        ]:
            _validate_positive(expr, name)
        _validate_nonnegative(
            self._physical_qubits_per_factory,
            "physical_qubits_per_factory",
        )

    @property
    def physical_qubits_per_logical(self) -> sp.Expr:
        """Return physical-qubit overhead for one logical patch.

        Returns:
            sp.Expr: ``physical_qubits_per_logical_factor * code_distance**2``.
        """
        return sp.simplify(
            self._physical_qubits_per_logical_factor * self._code_distance**2
        )

    @property
    def logical_cycle_time_seconds(self) -> sp.Expr:
        """Return logical-cycle time implied by the surface-code model.

        Returns:
            sp.Expr: ``logical_cycle_factor * code_distance *
                physical_cycle_time_seconds``.
        """
        return sp.simplify(
            self._logical_cycle_factor
            * self._code_distance
            * self._physical_cycle_time_seconds
        )

    @property
    def factory_qubits(self) -> sp.Expr:
        """Return total physical qubits reserved for factories.

        Returns:
            sp.Expr: ``factory_count * physical_qubits_per_factory``.
        """
        return sp.simplify(self._factory_count * self._physical_qubits_per_factory)

    @property
    def toffoli_throughput_per_second(self) -> sp.Expr:
        """Return sustained Toffoli-resource throughput.

        Returns:
            sp.Expr: Factory throughput in Toffoli resources per second.
        """
        return sp.simplify(
            self._factory_count
            / (self._factory_cycles_per_toffoli * self.logical_cycle_time_seconds)
        )

    def to_cost_model(self) -> FTQCCostModel:
        """Convert surface-code knobs into the estimator cost model.

        Returns:
            FTQCCostModel: Derived model accepted by FTQC chemistry
                estimators.
        """
        return FTQCCostModel(
            physical_qubits_per_logical=self.physical_qubits_per_logical,
            logical_cycle_time_seconds=self.logical_cycle_time_seconds,
            factory_qubits=self.factory_qubits,
            toffoli_throughput_per_second=self.toffoli_throughput_per_second,
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize raw and derived architecture quantities.

        Returns:
            dict[str, str]: JSON-friendly surface-code architecture model.
        """
        return {
            "code_distance": str(self._code_distance),
            "physical_cycle_time_seconds": str(self._physical_cycle_time_seconds),
            "physical_qubits_per_logical_factor": str(
                self._physical_qubits_per_logical_factor
            ),
            "logical_cycle_factor": str(self._logical_cycle_factor),
            "factory_count": str(self._factory_count),
            "physical_qubits_per_factory": str(self._physical_qubits_per_factory),
            "factory_cycles_per_toffoli": str(self._factory_cycles_per_toffoli),
            "physical_qubits_per_logical": str(self.physical_qubits_per_logical),
            "logical_cycle_time_seconds": str(self.logical_cycle_time_seconds),
            "factory_qubits": str(self.factory_qubits),
            "toffoli_throughput_per_second": str(self.toffoli_throughput_per_second),
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return raw and derived architecture quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Surface-code inputs and
                derived cost-model quantities keyed by the canonical catalog.
        """
        values = {
            FTQCResourceQuantity.CODE_DISTANCE: self._code_distance,
            FTQCResourceQuantity.PHYSICAL_CYCLE_TIME_SECONDS: (
                self._physical_cycle_time_seconds
            ),
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL_FACTOR: (
                self._physical_qubits_per_logical_factor
            ),
            FTQCResourceQuantity.LOGICAL_CYCLE_FACTOR: self._logical_cycle_factor,
            FTQCResourceQuantity.FACTORY_COUNT: self._factory_count,
            FTQCResourceQuantity.PHYSICAL_QUBITS_PER_FACTORY: (
                self._physical_qubits_per_factory
            ),
            FTQCResourceQuantity.FACTORY_CYCLES_PER_TOFFOLI: (
                self._factory_cycles_per_toffoli
            ),
        }
        values.update(self.to_cost_model().resource_values())
        return values

    @property
    def _code_distance(self) -> sp.Expr:
        """Return ``code_distance`` as a SymPy expression.

        Returns:
            sp.Expr: Converted code distance.
        """
        return _as_expr(self.code_distance, "code_distance")

    @property
    def _physical_cycle_time_seconds(self) -> sp.Expr:
        """Return ``physical_cycle_time_seconds`` as a SymPy expression.

        Returns:
            sp.Expr: Converted physical cycle time.
        """
        return _as_expr(
            self.physical_cycle_time_seconds,
            "physical_cycle_time_seconds",
        )

    @property
    def _physical_qubits_per_logical_factor(self) -> sp.Expr:
        """Return logical-patch factor as a SymPy expression.

        Returns:
            sp.Expr: Converted physical-qubit factor.
        """
        return _as_expr(
            self.physical_qubits_per_logical_factor,
            "physical_qubits_per_logical_factor",
        )

    @property
    def _logical_cycle_factor(self) -> sp.Expr:
        """Return logical-cycle factor as a SymPy expression.

        Returns:
            sp.Expr: Converted logical-cycle factor.
        """
        return _as_expr(self.logical_cycle_factor, "logical_cycle_factor")

    @property
    def _factory_count(self) -> sp.Expr:
        """Return ``factory_count`` as a SymPy expression.

        Returns:
            sp.Expr: Converted factory count.
        """
        return _as_expr(self.factory_count, "factory_count")

    @property
    def _physical_qubits_per_factory(self) -> sp.Expr:
        """Return ``physical_qubits_per_factory`` as a SymPy expression.

        Returns:
            sp.Expr: Converted factory size.
        """
        return _as_expr(
            self.physical_qubits_per_factory,
            "physical_qubits_per_factory",
        )

    @property
    def _factory_cycles_per_toffoli(self) -> sp.Expr:
        """Return ``factory_cycles_per_toffoli`` as a SymPy expression.

        Returns:
            sp.Expr: Converted factory cycle count.
        """
        return _as_expr(
            self.factory_cycles_per_toffoli,
            "factory_cycles_per_toffoli",
        )


_FTQCCostModelLike = FTQCCostModel | SurfaceCodeCostModel


@dataclass(frozen=True)
class FTQCResourceEstimate:
    """Represent algorithm-level FTQC resource estimates.

    Attributes:
        algorithm (str): Human-readable algorithm or representation name.
        logical_qubits (sp.Expr): Logical qubits required by the algorithm.
        physical_qubits (sp.Expr): Physical qubits under the selected
            architecture model.
        toffoli_gates (sp.Expr): Toffoli gate count or Toffoli-equivalent
            non-Clifford count.
        t_gates (sp.Expr): T gate count when it is distinct from Toffoli
            count. Defaults to zero for Toffoli-native estimates.
        clifford_gates (sp.Expr): Clifford gate estimate when available.
        qpe_iterations (sp.Expr): Number of phase-estimation walk or
            time-evolution calls.
        target_precision (sp.Expr): Requested energy or phase precision that
            controls the QPE iteration count.
        logical_depth (sp.Expr): Logical circuit depth proxy.
        runtime_seconds (sp.Expr): Runtime estimate in seconds.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in the
            estimate, keyed by display name.
        assumptions (dict[str, str]): Reader-facing notes about model choices.
        references (tuple[FTQCReference, ...]): Research sources that motivate
            the estimate's scaling model or resource assumptions.
        formulas (tuple[FTQCResourceFormula, ...]): Structured derivation
            formulas for key resource quantities. Defaults to an empty tuple.
        architecture_values (dict[FTQCResourceQuantity, sp.Expr]): Architecture
            quantities used to lift logical resources to physical resources.
            These are keyed by the canonical FTQC quantity catalog.

    Example:
        >>> n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
        >>> est = estimate_qubitized_chemistry_qpe(n, lam, eps, walk)
        >>> est.qpe_iterations
        lambda/eps
    """

    algorithm: str
    logical_qubits: sp.Expr
    physical_qubits: sp.Expr
    toffoli_gates: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr
    qpe_iterations: sp.Expr
    target_precision: sp.Expr
    logical_depth: sp.Expr
    runtime_seconds: sp.Expr
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)
    assumptions: dict[str, str] = field(default_factory=dict)
    references: tuple[FTQCReference, ...] = ()
    formulas: tuple[FTQCResourceFormula, ...] = ()
    architecture_values: dict[FTQCResourceQuantity, sp.Expr] = field(
        default_factory=dict
    )

    def substitute(self, **values: int | float) -> FTQCResourceEstimate:
        """Substitute concrete values into all symbolic fields.

        Args:
            **values (int | float): Mapping from symbol name to concrete
                value. Unknown names are accepted and converted into SymPy
                symbols so callers can substitute ad hoc expressions.

        Returns:
            FTQCResourceEstimate: New estimate with substitutions applied.
        """
        substitutions: dict[Any, Any] = {}
        for name, value in values.items():
            substitutions[self.parameters.get(name, sp.Symbol(name))] = value

        return self._map_exprs(lambda expr: expr.subs(substitutions).doit())

    def simplify(self) -> FTQCResourceEstimate:
        """Simplify every symbolic field.

        Returns:
            FTQCResourceEstimate: New estimate with simplified expressions.
        """
        return self._map_exprs(sp.simplify)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate to string-valued dictionaries.

        Returns:
            dict[str, Any]: JSON-friendly dictionary containing all resource
                fields, parameters, and assumptions.
        """
        return {
            "algorithm": self.algorithm,
            "logical_qubits": str(self.logical_qubits),
            "physical_qubits": str(self.physical_qubits),
            "toffoli_gates": str(self.toffoli_gates),
            "t_gates": str(self.t_gates),
            "clifford_gates": str(self.clifford_gates),
            "qpe_iterations": str(self.qpe_iterations),
            "target_precision": str(self.target_precision),
            "logical_depth": str(self.logical_depth),
            "runtime_seconds": str(self.runtime_seconds),
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
            "assumptions": dict(self.assumptions),
            "references": [reference.to_dict() for reference in self.references],
            "formulas": [formula.to_dict() for formula in self.formulas],
            "architecture_values": {
                quantity.value: str(value)
                for quantity, value in self.architecture_values.items()
            },
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return estimate values keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Resource values for logical
                and physical output quantities.
        """
        values = {
            FTQCResourceQuantity.LOGICAL_QUBITS: self.logical_qubits,
            FTQCResourceQuantity.PHYSICAL_QUBITS: self.physical_qubits,
            FTQCResourceQuantity.TOFFOLI_GATES: self.toffoli_gates,
            FTQCResourceQuantity.T_GATES: self.t_gates,
            FTQCResourceQuantity.CLIFFORD_GATES: self.clifford_gates,
            FTQCResourceQuantity.QPE_ITERATIONS: self.qpe_iterations,
            FTQCResourceQuantity.TARGET_PRECISION: self.target_precision,
            FTQCResourceQuantity.LOGICAL_DEPTH: self.logical_depth,
            FTQCResourceQuantity.RUNTIME_SECONDS: self.runtime_seconds,
        }
        values.update(self.architecture_values)
        return values

    def to_quantity_table(self) -> list[dict[str, str]]:
        """Serialize estimate values with quantity metadata.

        Returns:
            list[dict[str, str]]: Rows containing ``quantity``, ``label``,
                ``category``, ``unit``, ``value``, and ``description``.
        """
        rows = []
        for quantity, value in self.resource_values().items():
            spec = describe_ftqc_resource_quantity(quantity)
            row = spec.to_dict()
            row["value"] = str(value)
            rows.append(row)
        return rows

    def to_formula_table(self) -> list[dict[str, str | list[str]]]:
        """Serialize derivation formulas with quantity metadata.

        Returns:
            list[dict[str, str | list[str]]]: Rows containing quantity
                metadata plus ``expression``, ``depends_on``, ``description``,
                and ``reference_keys`` fields.
        """
        rows = []
        for formula in self.formulas:
            spec = describe_ftqc_resource_quantity(formula.quantity)
            row: dict[str, str | list[str]] = {**spec.to_dict()}
            row.update(formula.to_dict())
            rows.append(row)
        return rows

    def to_logical_resource_estimate(self) -> ResourceEstimate:
        """Convert logical FTQC work into the common resource-estimate shape.

        The returned estimate intentionally contains only logical-circuit-like
        quantities. Physical qubits, runtime, architecture assumptions, and
        research references remain on ``FTQCResourceEstimate`` so callers do
        not accidentally treat architecture-lifted resources as circuit gate
        counts.

        Returns:
            ResourceEstimate: Existing circuit-resource container populated
                with logical qubits, non-Clifford counts, Clifford counts, and
                QPE iteration count as an oracle-call proxy.
        """
        total_gates = sp.simplify(
            self.toffoli_gates + self.t_gates + self.clifford_gates
        )
        gates = GateCount(
            total=total_gates,
            single_qubit=sp.simplify(self.t_gates + self.clifford_gates),
            two_qubit=sp.Integer(0),
            multi_qubit=self.toffoli_gates,
            t_gates=self.t_gates,
            clifford_gates=self.clifford_gates,
            rotation_gates=sp.Integer(0),
            oracle_calls={"qpe_iterations": self.qpe_iterations},
            oracle_queries={},
        )
        parameters = _collect_parameters(
            (
                self.logical_qubits,
                gates.total,
                gates.single_qubit,
                gates.two_qubit,
                gates.multi_qubit,
                gates.t_gates,
                gates.clifford_gates,
                gates.rotation_gates,
                *gates.oracle_calls.values(),
                *gates.oracle_queries.values(),
            )
        )
        return ResourceEstimate(
            qubits=self.logical_qubits,
            gates=gates,
            parameters=parameters,
        )

    def with_cost_model(
        self,
        cost_model: _FTQCCostModelLike,
    ) -> FTQCResourceEstimate:
        """Relift physical resources with a different architecture model.

        The logical algorithm quantities are preserved. Physical qubits and
        runtime are recomputed from ``cost_model`` using ``logical_qubits``,
        ``logical_depth``, and the sum of Toffoli and T counts as the
        non-Clifford throughput demand.

        Args:
            cost_model (FTQCCostModel | SurfaceCodeCostModel): Architecture
                model used for the new physical-qubit and runtime estimates.

        Returns:
            FTQCResourceEstimate: Estimate with identical logical resources
                and updated physical resources.
        """
        normalized_cost_model, architecture_values = _normalize_cost_model(cost_model)
        assumptions = dict(self.assumptions)
        assumptions["architecture_relift"] = (
            "physical_qubits and runtime_seconds were recomputed from an "
            "existing logical estimate with a replacement architecture model."
        )
        non_clifford_gates = sp.simplify(self.toffoli_gates + self.t_gates)
        return _build_estimate(
            algorithm=self.algorithm,
            logical_qubits=self.logical_qubits,
            physical_qubits=normalized_cost_model.physical_qubits_for(
                self.logical_qubits
            ),
            toffoli_gates=self.toffoli_gates,
            t_gates=self.t_gates,
            clifford_gates=self.clifford_gates,
            qpe_iterations=self.qpe_iterations,
            target_precision=self.target_precision,
            logical_depth=self.logical_depth,
            runtime_seconds=normalized_cost_model.runtime_seconds_for(
                self.logical_depth,
                non_clifford_gates,
            ),
            assumptions=assumptions,
            references=self.references,
            formulas=self.formulas,
            architecture_values=architecture_values,
        )

    def _map_exprs(self, fn: Callable[[sp.Expr], sp.Expr]) -> FTQCResourceEstimate:
        """Apply a function to each symbolic resource field.

        Args:
            fn (Callable[[sp.Expr], sp.Expr]): Callable accepting and
                returning a SymPy expression.

        Returns:
            FTQCResourceEstimate: New estimate with mapped expressions.
        """
        return FTQCResourceEstimate(
            algorithm=self.algorithm,
            logical_qubits=fn(self.logical_qubits),
            physical_qubits=fn(self.physical_qubits),
            toffoli_gates=fn(self.toffoli_gates),
            t_gates=fn(self.t_gates),
            clifford_gates=fn(self.clifford_gates),
            qpe_iterations=fn(self.qpe_iterations),
            target_precision=fn(self.target_precision),
            logical_depth=fn(self.logical_depth),
            runtime_seconds=fn(self.runtime_seconds),
            parameters=self.parameters,
            assumptions=self.assumptions,
            references=self.references,
            formulas=tuple(
                FTQCResourceFormula(
                    quantity=formula.quantity,
                    expression=fn(cast(sp.Expr, formula.expression)),
                    depends_on=formula.depends_on,
                    description=formula.description,
                    reference_keys=formula.reference_keys,
                )
                for formula in self.formulas
            ),
            architecture_values={
                quantity: fn(value)
                for quantity, value in self.architecture_values.items()
            },
        )


@dataclass(frozen=True)
class PauliHamiltonianResource:
    """Summarize a Pauli-LCU Hamiltonian for FTQC resource estimates.

    Attributes:
        n_spin_orbitals (sp.Expr): Number of spin orbitals or qubits in the
            encoded active-space Hamiltonian.
        n_pauli_terms (sp.Expr): Number of non-identity Pauli terms in the
            Hamiltonian representation.
        lambda_norm (sp.Expr): Sum of absolute non-identity Pauli
            coefficients used as the LCU normalization proxy.
        max_locality (sp.Expr): Maximum number of non-identity Pauli factors
            in any term.
        constant (sp.Expr): Constant energy shift stored on the Hamiltonian.
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

    n_spin_orbitals: sp.Expr
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
        _validate_positive(self.n_spin_orbitals, "n_spin_orbitals")
        _validate_nonnegative(self.n_pauli_terms, "n_pauli_terms")
        _validate_nonnegative(self.lambda_norm, "lambda_norm")
        _validate_nonnegative(self.max_locality, "max_locality")
        _validate_nonnegative(sp.Abs(self.constant), "constant")

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
            n_spin_orbitals=self.n_spin_orbitals,
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
            "n_spin_orbitals": str(self.n_spin_orbitals),
            "n_pauli_terms": str(self.n_pauli_terms),
            "lambda_norm": str(self.lambda_norm),
            "max_locality": str(self.max_locality),
            "constant": str(self.constant),
            "constant_included": self.constant_included,
            "source": self.source,
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return Hamiltonian metadata keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Problem-level Hamiltonian
                quantities that feed algorithmic FTQC estimates.
        """
        return {
            FTQCResourceQuantity.N_SPIN_ORBITALS: self.n_spin_orbitals,
            FTQCResourceQuantity.N_PAULI_TERMS: self.n_pauli_terms,
            FTQCResourceQuantity.LAMBDA_NORM: self.lambda_norm,
            FTQCResourceQuantity.MAX_LOCALITY: self.max_locality,
        }


@dataclass(frozen=True)
class FTQCAccuracyBudget:
    """Allocate a total FTQC energy-error budget across model components.

    The budget separates representation error, such as tensor-factorization
    truncation, from the precision left for phase estimation. This keeps
    chemistry estimates comparable: two models can share the same total target
    precision while spending different amounts on representation error.

    Args:
        target_precision (sp.Expr | int | float): Total energy-error budget.
        truncation_error (sp.Expr | int | float): Error already spent by the
            Hamiltonian representation before QPE. Defaults to zero.
        safety_margin (sp.Expr | int | float): Optional unallocated margin
            reserved for synthesis, fitting, or other modeled error sources.
            Defaults to zero.

    Raises:
        ValueError: If the total target precision is non-positive, if either
            non-QPE component is negative, or if the remaining QPE precision is
            provably non-positive.

    Example:
        >>> budget = FTQCAccuracyBudget(
        ...     target_precision=sp.Float("1.6e-3"),
        ...     truncation_error=sp.Float("1e-4"),
        ... )
        >>> budget.qpe_precision
        0.00150000000000000
    """

    target_precision: _SympyLike
    truncation_error: _SympyLike = 0
    safety_margin: _SympyLike = 0

    def __post_init__(self) -> None:
        """Validate the accuracy-budget fields after construction.

        Raises:
            ValueError: If any budget component violates the allocation
                contract.
        """
        _validate_positive(self._target_precision, "target_precision")
        _validate_nonnegative(self._truncation_error, "truncation_error")
        _validate_nonnegative(self._safety_margin, "safety_margin")
        _validate_positive(self.qpe_precision, "qpe_precision")

    @property
    def qpe_precision(self) -> sp.Expr:
        """Return the precision remaining for QPE iterations.

        Returns:
            sp.Expr: ``target_precision - truncation_error - safety_margin``.
        """
        return sp.simplify(
            self._target_precision - self._truncation_error - self._safety_margin
        )

    def with_model(self, model: ChemistryQPEModel) -> ChemistryQPEModel:
        """Return a chemistry model carrying this budget's truncation error.

        Args:
            model (ChemistryQPEModel): Chemistry model whose representation
                error should be aligned with this accuracy budget.

        Returns:
            ChemistryQPEModel: Copy of ``model`` with ``truncation_error`` set
                to this budget's truncation error.

        Raises:
            TypeError: If ``model`` is not a ``ChemistryQPEModel``.
        """
        if not isinstance(model, ChemistryQPEModel):
            raise TypeError("model must be a ChemistryQPEModel instance.")
        return replace(model, truncation_error=self._truncation_error)

    def to_dict(self) -> dict[str, str]:
        """Serialize the accuracy budget.

        Returns:
            dict[str, str]: JSON-friendly accuracy-budget values.
        """
        return {
            "target_precision": str(self._target_precision),
            "truncation_error": str(self._truncation_error),
            "safety_margin": str(self._safety_margin),
            "qpe_precision": str(self.qpe_precision),
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return accuracy-budget values keyed by canonical quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Target precision and
                truncation-error quantities.
        """
        return {
            FTQCResourceQuantity.TARGET_PRECISION: self._target_precision,
            FTQCResourceQuantity.TRUNCATION_ERROR: self._truncation_error,
        }

    @property
    def _target_precision(self) -> sp.Expr:
        """Return ``target_precision`` as a SymPy expression.

        Returns:
            sp.Expr: Converted total target precision.
        """
        return _as_expr(self.target_precision, "target_precision")

    @property
    def _truncation_error(self) -> sp.Expr:
        """Return ``truncation_error`` as a SymPy expression.

        Returns:
            sp.Expr: Converted truncation error.
        """
        return _as_expr(self.truncation_error, "truncation_error")

    @property
    def _safety_margin(self) -> sp.Expr:
        """Return ``safety_margin`` as a SymPy expression.

        Returns:
            sp.Expr: Converted safety margin.
        """
        return _as_expr(self.safety_margin, "safety_margin")


@dataclass(frozen=True)
class ChemistryQPEModel:
    """Describe a concrete chemistry representation for QPE estimates.

    Attributes:
        hamiltonian (PauliHamiltonianResource): Pauli-LCU Hamiltonian summary.
        method (ChemistryQPEMethod): QPE representation or optimization
            strategy used to choose default logical-qubit scaling.
        walk_cost_toffoli (sp.Expr): Toffoli cost for one qubitized walk.
        sparsity (sp.Expr | None): Sparse-method nonzero term count. Defaults
            to ``hamiltonian.n_pauli_terms`` when omitted.
        second_factor_rank (sp.Expr | None): Average second factorization
            rank for double-factorized methods.
        logical_qubits (sp.Expr | None): Explicit logical-qubit count.
        truncation_error (sp.Expr): Hamiltonian representation error budget.
        description (str): Reader-facing model label.
        references (tuple[FTQCReference, ...]): Additional model-specific
            sources, such as molecule-specific resource-estimation papers.

    Raises:
        ValueError: If any positive-valued quantity is non-positive or if
            ``truncation_error`` is negative.

    Example:
        >>> summary = PauliHamiltonianResource(
        ...     n_spin_orbitals=4,
        ...     n_pauli_terms=10,
        ...     lambda_norm=20,
        ...     max_locality=2,
        ... )
        >>> model = ChemistryQPEModel(
        ...     summary,
        ...     walk_cost_toffoli=100,
        ...     method=ChemistryQPEMethod.SPARSE,
        ... )
        >>> model.effective_sparsity
        10
    """

    hamiltonian: PauliHamiltonianResource
    walk_cost_toffoli: _SympyLike
    method: str | ChemistryQPEMethod = ChemistryQPEMethod.DOUBLE_FACTORIZATION
    sparsity: _SympyLike | None = None
    second_factor_rank: _SympyLike | None = None
    logical_qubits: _SympyLike | None = None
    truncation_error: _SympyLike = 0
    description: str = ""
    references: tuple[FTQCReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate model fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued quantity is non-positive or if
                ``truncation_error`` is negative.
        """
        _validate_positive(
            _as_expr(self.walk_cost_toffoli, "walk_cost_toffoli"),
            "walk_cost_toffoli",
        )
        if self.sparsity is not None:
            _validate_positive(_as_expr(self.sparsity, "sparsity"), "sparsity")
        if self.second_factor_rank is not None:
            _validate_positive(
                _as_expr(self.second_factor_rank, "second_factor_rank"),
                "second_factor_rank",
            )
        if self.logical_qubits is not None:
            _validate_positive(
                _as_expr(self.logical_qubits, "logical_qubits"),
                "logical_qubits",
            )
        _validate_nonnegative(
            _as_expr(self.truncation_error, "truncation_error"),
            "truncation_error",
        )
        _normalize_method(self.method)

    @property
    def normalized_method(self) -> ChemistryQPEMethod:
        """Return the normalized QPE method.

        Returns:
            ChemistryQPEMethod: Normalized finite-set method.
        """
        return _normalize_method(self.method)

    @property
    def effective_sparsity(self) -> sp.Expr | None:
        """Return sparse-method term count with a Hamiltonian fallback.

        Returns:
            sp.Expr | None: Explicit sparsity, Hamiltonian term count for the
                sparse method, or None for non-sparse methods.
        """
        if self.sparsity is not None:
            return _as_expr(self.sparsity, "sparsity")
        if self.normalized_method == ChemistryQPEMethod.SPARSE:
            return self.hamiltonian.n_pauli_terms
        return None

    @property
    def logical_qubit_count(self) -> sp.Expr:
        """Return the logical-qubit model used by chemistry QPE estimates.

        Returns:
            sp.Expr: Explicit logical-qubit count when supplied, otherwise
            the representation-specific default.
        """
        if self.logical_qubits is not None:
            return _as_expr(self.logical_qubits, "logical_qubits")
        return _default_logical_qubits(
            self.normalized_method,
            self.hamiltonian.n_spin_orbitals,
            sparsity=self.effective_sparsity,
            second_factor_rank=(
                None
                if self.second_factor_rank is None
                else _as_expr(self.second_factor_rank, "second_factor_rank")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: String-valued model metadata.
        """
        return {
            "hamiltonian": self.hamiltonian.to_dict(),
            "method": self.normalized_method.value,
            "walk_cost_toffoli": str(
                _as_expr(self.walk_cost_toffoli, "walk_cost_toffoli")
            ),
            "sparsity": (
                None
                if self.effective_sparsity is None
                else str(self.effective_sparsity)
            ),
            "second_factor_rank": (
                None
                if self.second_factor_rank is None
                else str(_as_expr(self.second_factor_rank, "second_factor_rank"))
            ),
            "logical_qubits": (
                None
                if self.logical_qubits is None
                else str(_as_expr(self.logical_qubits, "logical_qubits"))
            ),
            "logical_qubit_count": str(self.logical_qubit_count),
            "truncation_error": str(
                _as_expr(self.truncation_error, "truncation_error")
            ),
            "description": self.description,
            "references": [reference.to_dict() for reference in self.references],
        }

    def resource_values(self) -> dict[FTQCResourceQuantity, sp.Expr]:
        """Return model inputs keyed by canonical FTQC quantities.

        Returns:
            dict[FTQCResourceQuantity, sp.Expr]: Hamiltonian metadata plus
                the representation-level walk cost.
        """
        values = self.hamiltonian.resource_values()
        values[FTQCResourceQuantity.WALK_COST_TOFFOLI] = _as_expr(
            self.walk_cost_toffoli,
            "walk_cost_toffoli",
        )
        values[FTQCResourceQuantity.TRUNCATION_ERROR] = _as_expr(
            self.truncation_error,
            "truncation_error",
        )
        return values


def summarize_pauli_hamiltonian(
    hamiltonian: Any,
    *,
    n_spin_orbitals: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "pauli_lcu",
) -> PauliHamiltonianResource:
    """Summarize a Qamomile Pauli Hamiltonian for FTQC estimates.

    Args:
        hamiltonian (Any): ``qamomile.observable.Hamiltonian`` instance to
            summarize.
        n_spin_orbitals (sp.Expr | int | float | None): Override for the
            encoded active-space size. Defaults to ``hamiltonian.num_qubits``.
        include_constant (bool): Whether to include the Hamiltonian constant
            term in ``lambda_norm``. Defaults to False, modeling the constant
            as a classical energy shift.
        source (str): Human-readable source label for the summary.

    Returns:
        PauliHamiltonianResource: Hamiltonian summary containing term count,
            lambda norm, constant, and max locality.

    Raises:
        TypeError: If ``hamiltonian`` is not a Qamomile Hamiltonian.
        ValueError: If the orbital count or derived norm is invalid.
    """
    import qamomile.observable as qm_o

    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise TypeError(
            "hamiltonian must be a qamomile.observable.Hamiltonian instance."
        )

    n_expr = (
        _as_expr(hamiltonian.num_qubits, "n_spin_orbitals")
        if n_spin_orbitals is None
        else _as_expr(n_spin_orbitals, "n_spin_orbitals")
    )
    lambda_norm = sp.Integer(0)
    max_locality = 0
    for operators, coeff in hamiltonian:
        lambda_norm += _abs_as_expr(coeff)
        max_locality = max(max_locality, len(operators))
    constant = _as_expr(hamiltonian.constant, "constant")
    if include_constant:
        lambda_norm += _abs_as_expr(hamiltonian.constant)

    return PauliHamiltonianResource(
        n_spin_orbitals=n_expr,
        n_pauli_terms=sp.Integer(len(hamiltonian)),
        lambda_norm=sp.simplify(lambda_norm),
        max_locality=sp.Integer(max_locality),
        constant=constant,
        constant_included=include_constant,
        source=source,
    )


def hamiltonian_from_openfermion_qubit_operator(
    openfermion_operator: Any,
    *,
    num_qubits: int | None = None,
) -> qm_o.Hamiltonian:
    """Convert an OpenFermion qubit operator into a Qamomile Hamiltonian.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like
            object exposing a ``terms`` mapping from Pauli-string tuples to
            numeric coefficients.
        num_qubits (int | None): Optional encoded register size to store on
            the returned Hamiltonian. Defaults to inferring the size from the
            largest Pauli index.

    Returns:
        qamomile.observable.Hamiltonian: Equivalent Qamomile Hamiltonian with
            identity terms stored as ``constant``.

    Raises:
        TypeError: If ``openfermion_operator`` does not expose a terms
            mapping or a term is malformed.
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
    n_spin_orbitals: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "openfermion_qubit_operator",
) -> PauliHamiltonianResource:
    """Summarize an OpenFermion qubit operator for FTQC estimates.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like
            object exposing a ``terms`` mapping.
        n_spin_orbitals (sp.Expr | int | float | None): Override for the
            encoded active-space size. Defaults to the inferred Qamomile
            Hamiltonian qubit count.
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
        n_spin_orbitals=n_spin_orbitals,
        include_constant=include_constant,
        source=source,
    )


def estimate_qubitized_chemistry_qpe(
    n_spin_orbitals: sp.Expr | int,
    lambda_norm: _SympyLike,
    precision: _SympyLike,
    walk_cost_toffoli: sp.Expr | int,
    *,
    method: str | ChemistryQPEMethod = ChemistryQPEMethod.DOUBLE_FACTORIZATION,
    sparsity: sp.Expr | int | None = None,
    second_factor_rank: sp.Expr | int | None = None,
    logical_qubits: sp.Expr | int | None = None,
    cost_model: _FTQCCostModelLike | None = None,
    references: tuple[FTQCReference, ...] = (),
) -> FTQCResourceEstimate:
    """Estimate qubitized QPE resources for molecular Hamiltonians.

    Args:
        n_spin_orbitals (sp.Expr | int): Number of spin orbitals in the
            active-space Hamiltonian.
        lambda_norm (sp.Expr | int | float): LCU block-encoding
            normalization, often the representation-dependent Hamiltonian
            1-norm.
        precision (sp.Expr | int | float): Target phase-estimation energy
            precision in Hartree or another consistent energy unit.
        walk_cost_toffoli (sp.Expr | int): Toffoli cost for one qubitized
            walk operator call.
        method (str | ChemistryQPEMethod): Hamiltonian representation used
            to choose a default logical-qubit model. Defaults to double
            factorization.
        sparsity (sp.Expr | int | None): Number of nonzero Pauli or LCU
            terms for the sparse method. Required only when using the sparse
            default logical-qubit model.
        second_factor_rank (sp.Expr | int | None): Average second
            factorization rank for double-factorized methods. Defaults to a
            symbolic ``Xi``.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            When omitted, a representation-level scaling model is used.
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model used to lift logical estimates to physical
            qubits and runtime. Defaults to a symbolic model.
        references (tuple[FTQCReference, ...]): Additional research sources
            to attach to the estimate. Method-level default references are
            attached automatically.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If a finite-set method is unknown, a required sparse
            parameter is missing, or a positive-valued input is non-positive.
    """
    method_enum = _normalize_method(method)
    n_expr = _as_expr(n_spin_orbitals, "n_spin_orbitals")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    walk_expr = _as_expr(walk_cost_toffoli, "walk_cost_toffoli")

    _validate_positive(n_expr, "n_spin_orbitals")
    _validate_positive(lambda_expr, "lambda_norm")
    _validate_positive(precision_expr, "precision")
    _validate_positive(walk_expr, "walk_cost_toffoli")

    if logical_qubits is None:
        logical_expr = _default_logical_qubits(
            method_enum,
            n_expr,
            sparsity=sparsity,
            second_factor_rank=second_factor_rank,
        )
    else:
        logical_expr = _as_expr(logical_qubits, "logical_qubits")
        _validate_positive(logical_expr, "logical_qubits")

    model, architecture_values = _normalize_cost_model(cost_model)
    qpe_iterations = sp.simplify(lambda_expr / precision_expr)
    toffoli_gates = sp.simplify(qpe_iterations * walk_expr)
    logical_depth = toffoli_gates
    physical_qubits = model.physical_qubits_for(logical_expr)
    runtime_seconds = model.runtime_seconds_for(logical_depth, toffoli_gates)
    assumptions = {
        "qpe_iterations": "Uses lambda_norm / precision as the walk-call proxy.",
        "method": method_enum.value,
        "walk_cost_toffoli": (
            "Caller supplies the per-walk Toffoli model so Qamomile does not "
            "bake in one chemistry factorization implementation."
        ),
    }
    return _build_estimate(
        algorithm=f"qubitized_qpe:{method_enum.value}",
        logical_qubits=logical_expr,
        physical_qubits=physical_qubits,
        toffoli_gates=toffoli_gates,
        t_gates=sp.Integer(0),
        clifford_gates=sp.Integer(0),
        qpe_iterations=qpe_iterations,
        target_precision=precision_expr,
        logical_depth=logical_depth,
        runtime_seconds=runtime_seconds,
        assumptions=assumptions,
        references=_combine_references(
            _METHOD_REFERENCES[method_enum],
            references,
        ),
        formulas=_qubitized_qpe_formulas(),
        architecture_values=architecture_values,
    )


def estimate_qubitized_chemistry_qpe_from_model(
    model: ChemistryQPEModel,
    precision: _SympyLike,
    *,
    cost_model: _FTQCCostModelLike | None = None,
) -> FTQCResourceEstimate:
    """Estimate qubitized QPE resources from a chemistry model object.

    Args:
        model (ChemistryQPEModel): Hamiltonian representation model carrying
            lambda norm, sparsity/rank metadata, and walk cost.
        precision (sp.Expr | int | float): Target phase-estimation energy
            precision.
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model used to lift logical estimates to physical
            qubits and runtime. Defaults to a symbolic model.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If the model or precision fields are invalid.
    """
    estimate = estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=model.hamiltonian.n_spin_orbitals,
        lambda_norm=model.hamiltonian.lambda_norm,
        precision=precision,
        walk_cost_toffoli=_as_expr(model.walk_cost_toffoli, "walk_cost_toffoli"),
        method=model.normalized_method,
        sparsity=model.effective_sparsity,
        second_factor_rank=(
            None
            if model.second_factor_rank is None
            else _as_expr(model.second_factor_rank, "second_factor_rank")
        ),
        logical_qubits=(
            None
            if model.logical_qubits is None
            else _as_expr(model.logical_qubits, "logical_qubits")
        ),
        cost_model=cost_model,
    )
    assumptions = dict(estimate.assumptions)
    assumptions.update(
        {
            "hamiltonian_source": model.hamiltonian.source,
            "truncation_error": str(
                _as_expr(model.truncation_error, "truncation_error")
            ),
        }
    )
    if model.description:
        assumptions["description"] = model.description
    return _build_estimate(
        algorithm=estimate.algorithm,
        logical_qubits=estimate.logical_qubits,
        physical_qubits=estimate.physical_qubits,
        toffoli_gates=estimate.toffoli_gates,
        t_gates=estimate.t_gates,
        clifford_gates=estimate.clifford_gates,
        qpe_iterations=estimate.qpe_iterations,
        target_precision=estimate.target_precision,
        logical_depth=estimate.logical_depth,
        runtime_seconds=estimate.runtime_seconds,
        assumptions=assumptions,
        references=_combine_references(estimate.references, model.references),
        formulas=estimate.formulas,
        architecture_values=estimate.architecture_values,
    )


def estimate_single_ancilla_trotter_qpe(
    n_spin_orbitals: sp.Expr | int,
    n_pauli_terms: sp.Expr | int,
    lambda_norm: _SympyLike,
    precision: _SympyLike,
    *,
    trotter_steps_per_sample: sp.Expr | int,
    samples: sp.Expr | int,
    unitary_weight_factor: _SympyLike = 1,
    randomized_compilation_factor: _SympyLike = 1,
    rotation_synthesis_t_gates: sp.Expr | int = 1,
    logical_qubits: sp.Expr | int | None = None,
    cost_model: _FTQCCostModelLike | None = None,
    references: tuple[FTQCReference, ...] = (),
) -> FTQCResourceEstimate:
    """Estimate early-FTQC single-ancilla Trotter QPE resources.

    This estimator models the style of early-FTQC chemistry proposals that
    combine single-ancilla QPE, partially randomized product formulas, and
    Hamiltonian-weight reduction such as unitary weight concentration.

    Args:
        n_spin_orbitals (sp.Expr | int): Number of spin orbitals.
        n_pauli_terms (sp.Expr | int): Number of Pauli LCU terms.
        lambda_norm (sp.Expr | int | float): Original Hamiltonian 1-norm.
        precision (sp.Expr | int | float): Target energy precision.
        trotter_steps_per_sample (sp.Expr | int): Product-formula steps per
            Hadamard-test sample.
        samples (sp.Expr | int): Number of sampled time points or shots in
            the signal-processing routine.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            reduction in Hamiltonian weight after spectrally invariant
            transformations. Values below one model cost reduction. Defaults
            to one.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            cost factor for partially randomized compilation. Defaults to one.
        rotation_synthesis_t_gates (sp.Expr | int): T-gate cost per small
            Pauli rotation. Defaults to one symbolic T-equivalent unit.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            Defaults to ``n_spin_orbitals + 1`` for the data register plus
            the Hadamard-test ancilla.
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model used to lift logical estimates to physical
            qubits and runtime.
        references (tuple[FTQCReference, ...]): Additional research sources
            to attach to the estimate. The early-FTQC unitary-weight
            concentration source is attached automatically.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If any positive-valued input is non-positive or either
            multiplicative reduction factor is negative.
    """
    n_expr = _as_expr(n_spin_orbitals, "n_spin_orbitals")
    terms_expr = _as_expr(n_pauli_terms, "n_pauli_terms")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    steps_expr = _as_expr(trotter_steps_per_sample, "trotter_steps_per_sample")
    samples_expr = _as_expr(samples, "samples")
    weight_factor = _as_expr(unitary_weight_factor, "unitary_weight_factor")
    randomized_factor = _as_expr(
        randomized_compilation_factor,
        "randomized_compilation_factor",
    )
    rotation_t = _as_expr(rotation_synthesis_t_gates, "rotation_synthesis_t_gates")

    for name, expr in [
        ("n_spin_orbitals", n_expr),
        ("n_pauli_terms", terms_expr),
        ("lambda_norm", lambda_expr),
        ("precision", precision_expr),
        ("trotter_steps_per_sample", steps_expr),
        ("samples", samples_expr),
        ("rotation_synthesis_t_gates", rotation_t),
    ]:
        _validate_positive(expr, name)
    _validate_nonnegative(weight_factor, "unitary_weight_factor")
    _validate_nonnegative(randomized_factor, "randomized_compilation_factor")

    logical_expr = (
        n_expr + 1
        if logical_qubits is None
        else _as_expr(logical_qubits, "logical_qubits")
    )
    _validate_positive(logical_expr, "logical_qubits")

    effective_lambda = sp.simplify(lambda_expr * weight_factor)
    qpe_iterations = sp.simplify(effective_lambda / precision_expr)
    pauli_rotations = sp.simplify(
        samples_expr * steps_expr * terms_expr * randomized_factor
    )
    logical_depth = sp.simplify(qpe_iterations * pauli_rotations)
    t_gates = sp.simplify(logical_depth * rotation_t)
    toffoli_gates = sp.Integer(0)
    model, architecture_values = _normalize_cost_model(cost_model)
    physical_qubits = model.physical_qubits_for(logical_expr)
    runtime_seconds = model.runtime_seconds_for(logical_depth, t_gates)
    assumptions = {
        "effective_lambda": "lambda_norm * unitary_weight_factor.",
        "qpe_style": "Single-ancilla Hadamard-test QPE with product-formula evolution.",
        "randomization": "randomized_compilation_factor rescales Pauli-rotation work.",
    }
    return _build_estimate(
        algorithm="single_ancilla_trotter_qpe:unitary_weight_concentration",
        logical_qubits=logical_expr,
        physical_qubits=physical_qubits,
        toffoli_gates=toffoli_gates,
        t_gates=t_gates,
        clifford_gates=sp.Integer(0),
        qpe_iterations=qpe_iterations,
        target_precision=precision_expr,
        logical_depth=logical_depth,
        runtime_seconds=runtime_seconds,
        assumptions=assumptions,
        references=_combine_references((_UWC_REFERENCE,), references),
        formulas=_single_ancilla_trotter_formulas(),
        architecture_values=architecture_values,
    )


def estimate_single_ancilla_trotter_qpe_from_hamiltonian(
    hamiltonian: PauliHamiltonianResource,
    precision: _SympyLike,
    *,
    trotter_steps_per_sample: sp.Expr | int,
    samples: sp.Expr | int,
    unitary_weight_factor: _SympyLike = 1,
    randomized_compilation_factor: _SympyLike = 1,
    rotation_synthesis_t_gates: sp.Expr | int = 1,
    logical_qubits: sp.Expr | int | None = None,
    cost_model: _FTQCCostModelLike | None = None,
    references: tuple[FTQCReference, ...] = (),
) -> FTQCResourceEstimate:
    """Estimate single-ancilla Trotter QPE from a Hamiltonian summary.

    Args:
        hamiltonian (PauliHamiltonianResource): Pauli-LCU Hamiltonian summary.
        precision (sp.Expr | int | float): Target energy precision.
        trotter_steps_per_sample (sp.Expr | int): Product-formula steps per
            Hadamard-test sample.
        samples (sp.Expr | int): Number of sampled time points or shots.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            reduction in Hamiltonian weight. Defaults to one.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            cost factor for randomized compilation. Defaults to one.
        rotation_synthesis_t_gates (sp.Expr | int): T-gate cost per Pauli
            rotation. Defaults to one.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            Defaults to ``hamiltonian.n_spin_orbitals + 1``.
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model used to lift logical estimates to physical
            qubits and runtime.
        references (tuple[FTQCReference, ...]): Additional research sources
            to attach to the estimate.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.
    """
    return estimate_single_ancilla_trotter_qpe(
        n_spin_orbitals=hamiltonian.n_spin_orbitals,
        n_pauli_terms=hamiltonian.n_pauli_terms,
        lambda_norm=hamiltonian.lambda_norm,
        precision=precision,
        trotter_steps_per_sample=trotter_steps_per_sample,
        samples=samples,
        unitary_weight_factor=unitary_weight_factor,
        randomized_compilation_factor=randomized_compilation_factor,
        rotation_synthesis_t_gates=rotation_synthesis_t_gates,
        logical_qubits=logical_qubits,
        cost_model=cost_model,
        references=references,
    )


def _default_logical_qubits(
    method: ChemistryQPEMethod,
    n_spin_orbitals: sp.Expr,
    *,
    sparsity: sp.Expr | int | None,
    second_factor_rank: sp.Expr | int | None,
) -> sp.Expr:
    """Return representation-level logical-qubit scaling.

    Args:
        method (ChemistryQPEMethod): Hamiltonian representation.
        n_spin_orbitals (sp.Expr): Number of spin orbitals.
        sparsity (sp.Expr | int | None): Sparse-method nonzero term count.
        second_factor_rank (sp.Expr | int | None): Average rank for
            double-factorized methods.

    Returns:
        sp.Expr: Symbolic logical-qubit estimate.

    Raises:
        ValueError: If the sparse method lacks ``sparsity``.
    """
    n = n_spin_orbitals
    match method:
        case ChemistryQPEMethod.SPARSE:
            if sparsity is None:
                raise ValueError("sparsity is required for sparse QPE estimates.")
            sparsity_expr = _as_expr(sparsity, "sparsity")
            _validate_positive(sparsity_expr, "sparsity")
            return sp.simplify(n + sp.sqrt(sparsity_expr))
        case ChemistryQPEMethod.SINGLE_FACTORIZATION:
            return sp.simplify(n ** sp.Rational(3, 2))
        case (
            ChemistryQPEMethod.DOUBLE_FACTORIZATION
            | ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF
        ):
            rank_expr = (
                sp.Symbol("Xi", positive=True)
                if second_factor_rank is None
                else _as_expr(second_factor_rank, "second_factor_rank")
            )
            _validate_positive(rank_expr, "second_factor_rank")
            return sp.simplify(n * sp.sqrt(rank_expr))
        case ChemistryQPEMethod.TENSOR_HYPERCONTRACTION:
            return n
        case ChemistryQPEMethod.UNITARY_WEIGHT_CONCENTRATION:
            return n + 1
        case _:
            raise ValueError(f"Unhandled chemistry QPE method: {method}")


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


def _architecture_formulas(
    non_clifford_quantity: FTQCResourceQuantity,
) -> tuple[FTQCResourceFormula, ...]:
    """Return formulas for lifting logical work to physical resources.

    Args:
        non_clifford_quantity (FTQCResourceQuantity): Non-Clifford resource
            quantity served by the factory-throughput term.

    Returns:
        tuple[FTQCResourceFormula, ...]: Physical-qubit and runtime formulas.
    """
    logical_qubits = _nonnegative_symbol(FTQCResourceQuantity.LOGICAL_QUBITS.value)
    logical_depth = _nonnegative_symbol(FTQCResourceQuantity.LOGICAL_DEPTH.value)
    non_clifford = _nonnegative_symbol(non_clifford_quantity.value)
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
            quantity=FTQCResourceQuantity.PHYSICAL_QUBITS,
            expression=logical_qubits * physical_qubits_per_logical + factory_qubits,
            depends_on=(
                FTQCResourceQuantity.LOGICAL_QUBITS,
                FTQCResourceQuantity.PHYSICAL_QUBITS_PER_LOGICAL,
                FTQCResourceQuantity.FACTORY_QUBITS,
            ),
            description=(
                "Lift logical qubits with the architecture overhead and add "
                "factory qubits."
            ),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.RUNTIME_SECONDS,
            expression=sp.Max(
                logical_depth * logical_cycle_time,
                non_clifford / throughput,
            ),
            depends_on=(
                FTQCResourceQuantity.LOGICAL_DEPTH,
                non_clifford_quantity,
                FTQCResourceQuantity.LOGICAL_CYCLE_TIME_SECONDS,
                FTQCResourceQuantity.TOFFOLI_THROUGHPUT_PER_SECOND,
            ),
            description=(
                "Use the slower of logical-cycle execution and factory "
                "throughput as the runtime proxy."
            ),
        ),
    )


def _qubitized_qpe_formulas() -> tuple[FTQCResourceFormula, ...]:
    """Return derivation formulas for qubitized QPE estimates.

    Returns:
        tuple[FTQCResourceFormula, ...]: QPE iteration, Toffoli, logical-depth,
            and architecture-lift formulas.
    """
    lambda_norm = _positive_symbol(FTQCResourceQuantity.LAMBDA_NORM.value)
    target_precision = _positive_symbol(FTQCResourceQuantity.TARGET_PRECISION.value)
    qpe_iterations = _nonnegative_symbol(FTQCResourceQuantity.QPE_ITERATIONS.value)
    walk_cost = _positive_symbol(FTQCResourceQuantity.WALK_COST_TOFFOLI.value)
    toffoli_gates = _nonnegative_symbol(FTQCResourceQuantity.TOFFOLI_GATES.value)
    return (
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.QPE_ITERATIONS,
            expression=lambda_norm / target_precision,
            depends_on=(
                FTQCResourceQuantity.LAMBDA_NORM,
                FTQCResourceQuantity.TARGET_PRECISION,
            ),
            description="Use Hamiltonian normalization divided by QPE precision.",
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
        *_architecture_formulas(FTQCResourceQuantity.TOFFOLI_GATES),
    )


def _single_ancilla_trotter_formulas() -> tuple[FTQCResourceFormula, ...]:
    """Return derivation formulas for single-ancilla Trotter QPE estimates.

    Returns:
        tuple[FTQCResourceFormula, ...]: QPE iteration, logical-depth, T-count,
            and architecture-lift formulas.
    """
    lambda_norm = _positive_symbol(FTQCResourceQuantity.LAMBDA_NORM.value)
    target_precision = _positive_symbol(FTQCResourceQuantity.TARGET_PRECISION.value)
    unitary_weight_factor = _nonnegative_symbol("unitary_weight_factor")
    qpe_iterations = _nonnegative_symbol(FTQCResourceQuantity.QPE_ITERATIONS.value)
    samples = _positive_symbol("samples")
    steps = _positive_symbol("trotter_steps_per_sample")
    terms = _positive_symbol(FTQCResourceQuantity.N_PAULI_TERMS.value)
    randomized_factor = _nonnegative_symbol("randomized_compilation_factor")
    logical_depth = _nonnegative_symbol(FTQCResourceQuantity.LOGICAL_DEPTH.value)
    rotation_t = _positive_symbol("rotation_synthesis_t_gates")
    return (
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.QPE_ITERATIONS,
            expression=lambda_norm * unitary_weight_factor / target_precision,
            depends_on=(
                FTQCResourceQuantity.LAMBDA_NORM,
                FTQCResourceQuantity.TARGET_PRECISION,
            ),
            description=(
                "Scale the Hamiltonian normalization by the unitary-weight "
                "factor before dividing by QPE precision."
            ),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.LOGICAL_DEPTH,
            expression=qpe_iterations * samples * steps * terms * randomized_factor,
            depends_on=(
                FTQCResourceQuantity.QPE_ITERATIONS,
                FTQCResourceQuantity.N_PAULI_TERMS,
            ),
            description=(
                "Count sampled product-formula rotations across QPE calls, "
                "samples, Trotter steps, terms, and randomization overhead."
            ),
        ),
        FTQCResourceFormula(
            quantity=FTQCResourceQuantity.T_GATES,
            expression=logical_depth * rotation_t,
            depends_on=(FTQCResourceQuantity.LOGICAL_DEPTH,),
            description="Multiply logical rotations by the synthesis T cost.",
        ),
        *_architecture_formulas(FTQCResourceQuantity.T_GATES),
    )


def _build_estimate(
    *,
    algorithm: str,
    logical_qubits: sp.Expr,
    physical_qubits: sp.Expr,
    toffoli_gates: sp.Expr,
    t_gates: sp.Expr,
    clifford_gates: sp.Expr,
    qpe_iterations: sp.Expr,
    target_precision: sp.Expr,
    logical_depth: sp.Expr,
    runtime_seconds: sp.Expr,
    assumptions: dict[str, str],
    references: tuple[FTQCReference, ...] = (),
    formulas: tuple[FTQCResourceFormula, ...] = (),
    architecture_values: dict[FTQCResourceQuantity, sp.Expr] | None = None,
) -> FTQCResourceEstimate:
    """Create an estimate and collect its free symbolic parameters.

    Args:
        algorithm (str): Algorithm name.
        logical_qubits (sp.Expr): Logical qubit count.
        physical_qubits (sp.Expr): Physical qubit count.
        toffoli_gates (sp.Expr): Toffoli count.
        t_gates (sp.Expr): T count.
        clifford_gates (sp.Expr): Clifford count.
        qpe_iterations (sp.Expr): QPE iteration count.
        target_precision (sp.Expr): Requested energy or phase precision.
        logical_depth (sp.Expr): Logical depth proxy.
        runtime_seconds (sp.Expr): Runtime estimate in seconds.
        assumptions (dict[str, str]): Notes about model assumptions.
        references (tuple[FTQCReference, ...]): Research sources for the
            scaling model.
        formulas (tuple[FTQCResourceFormula, ...]): Structured derivation
            formulas to attach to the estimate. Defaults to an empty tuple.
        architecture_values (dict[FTQCResourceQuantity, sp.Expr] | None):
            Architecture quantities used to lift logical resources to physical
            resources. Defaults to None.

    Returns:
        FTQCResourceEstimate: Estimate with collected parameters.
    """
    architecture = dict(architecture_values or {})
    expressions = [
        logical_qubits,
        physical_qubits,
        toffoli_gates,
        t_gates,
        clifford_gates,
        qpe_iterations,
        target_precision,
        logical_depth,
        runtime_seconds,
        *architecture.values(),
    ]
    return FTQCResourceEstimate(
        algorithm=algorithm,
        logical_qubits=sp.simplify(logical_qubits),
        physical_qubits=sp.simplify(physical_qubits),
        toffoli_gates=sp.simplify(toffoli_gates),
        t_gates=sp.simplify(t_gates),
        clifford_gates=sp.simplify(clifford_gates),
        qpe_iterations=sp.simplify(qpe_iterations),
        target_precision=sp.simplify(target_precision),
        logical_depth=sp.simplify(logical_depth),
        runtime_seconds=sp.simplify(runtime_seconds),
        parameters=_collect_parameters(expressions),
        assumptions=assumptions,
        references=_combine_references(references),
        formulas=formulas,
        architecture_values=architecture,
    )


def _normalize_cost_model(
    cost_model: _FTQCCostModelLike | None,
) -> tuple[FTQCCostModel, dict[FTQCResourceQuantity, sp.Expr]]:
    """Normalize an architecture model for estimator use.

    Args:
        cost_model (FTQCCostModel | SurfaceCodeCostModel | None):
            Architecture model supplied by the caller. ``None`` creates a
            symbolic ``FTQCCostModel``.

    Returns:
        tuple[FTQCCostModel, dict[FTQCResourceQuantity, sp.Expr]]: The
            low-level cost model used for lifting and the canonical
            architecture quantities to keep on the estimate.

    Raises:
        TypeError: If ``cost_model`` is not a supported FTQC architecture
            model.
    """
    if cost_model is None:
        model = FTQCCostModel()
        return model, model.resource_values()
    if isinstance(cost_model, SurfaceCodeCostModel):
        return cost_model.to_cost_model(), cost_model.resource_values()
    if isinstance(cost_model, FTQCCostModel):
        return cost_model, cost_model.resource_values()
    raise TypeError("cost_model must be an FTQCCostModel or SurfaceCodeCostModel.")


def _collect_parameters(
    expressions: tuple[sp.Expr, ...] | list[sp.Expr],
) -> dict[str, sp.Symbol]:
    """Collect free SymPy symbols from resource expressions.

    Args:
        expressions (tuple[sp.Expr, ...] | list[sp.Expr]): Resource
            expressions whose free symbols should become substitution
            parameters.

    Returns:
        dict[str, sp.Symbol]: Free symbols keyed by their display name.
    """
    symbols: set[sp.Symbol] = set()
    for expr in expressions:
        for symbol in expr.free_symbols:
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


def _normalize_method(method: str | ChemistryQPEMethod) -> ChemistryQPEMethod:
    """Normalize a public method value to ``ChemistryQPEMethod``.

    Args:
        method (str | ChemistryQPEMethod): User-provided method.

    Returns:
        ChemistryQPEMethod: Normalized enum value.

    Raises:
        ValueError: If ``method`` is not a known chemistry QPE method.
    """
    try:
        return ChemistryQPEMethod(method)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ChemistryQPEMethod)
        raise ValueError(
            f"Unknown chemistry QPE method {method!r}; valid: {valid}."
        ) from exc


def _as_expr(value: _CoefficientLike, name: str) -> sp.Expr:
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


def _as_float(value: _CoefficientLike, name: str) -> float:
    """Convert a numeric expression to a finite float.

    Args:
        value (sp.Expr | int | float | complex): Value to convert.
        name (str): Field name used in error messages.

    Returns:
        float: Finite floating-point value.

    Raises:
        TypeError: If ``value`` cannot be sympified.
        ValueError: If ``value`` is symbolic, complex, NaN, or infinite.
    """
    expr = _as_expr(value, name)
    if expr.free_symbols:
        raise ValueError(f"{name} must be numeric for distance selection.")
    numeric = complex(sp.N(expr))
    if abs(numeric.imag) > 0:
        raise ValueError(f"{name} must be real.")
    result = float(numeric.real)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _abs_as_expr(value: _CoefficientLike) -> sp.Expr:
    """Return the symbolic absolute value of a numeric expression.

    Args:
        value (sp.Expr | int | float | complex): Coefficient or constant to
            convert.

    Returns:
        sp.Expr: Nonnegative SymPy absolute value.

    Raises:
        TypeError: If ``value`` cannot be sympified.
    """
    return sp.Abs(_as_expr(value, "value"))


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
