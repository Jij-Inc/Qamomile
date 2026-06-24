"""Lift logical Qamomile resource estimates to rough FTQC resource proxies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from qamomile.circuit.estimator import ResourceEstimate

_SympyLike = sp.Expr | int | float


@dataclass(frozen=True)
class FTQCCostModel:
    """Map logical resource estimates to rough physical resource proxies.

    This compact model is architecture-generic and does not choose an error
    correction code, code distance, error budget, or factory layout. It
    consumes the existing Qamomile ``ResourceEstimate`` output and converts
    logical qubits plus non-Clifford work into coarse physical-qubit and
    runtime proxies.

    Attributes:
        physical_qubits_per_logical (sp.Expr | int | float): Physical qubit
            overhead for one logical data or ancilla qubit.
        logical_cycle_time_seconds (sp.Expr | int | float): Time for one
            logical layer or cycle, in seconds.
        factory_qubits (sp.Expr | int | float): Physical qubits reserved for
            magic-state factories or equivalent non-Clifford support.
        non_clifford_throughput_per_second (sp.Expr | int | float): Sustainable
            T, Toffoli, or non-Clifford-equivalent throughput.

    Raises:
        ValueError: If a positive-valued field is non-positive or if
            ``factory_qubits`` is negative.

    Example:
        >>> model = FTQCCostModel(
        ...     physical_qubits_per_logical=1000,
        ...     logical_cycle_time_seconds=1e-6,
        ...     factory_qubits=20000,
        ...     non_clifford_throughput_per_second=1e5,
        ... )
        >>> model.physical_qubits_for(10)
        30000
    """

    physical_qubits_per_logical: _SympyLike
    logical_cycle_time_seconds: _SympyLike
    factory_qubits: _SympyLike = 0
    non_clifford_throughput_per_second: _SympyLike = 1

    def __post_init__(self) -> None:
        """Validate cost-model fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued field is non-positive or if
                ``factory_qubits`` is negative.
        """
        _validate_positive(
            self._physical_qubits_per_logical,
            "physical_qubits_per_logical",
        )
        _validate_positive(
            self._logical_cycle_time_seconds,
            "logical_cycle_time_seconds",
        )
        _validate_nonnegative(self._factory_qubits, "factory_qubits")
        _validate_positive(
            self._non_clifford_throughput_per_second,
            "non_clifford_throughput_per_second",
        )

    def physical_qubits_for(self, logical_qubits: _SympyLike) -> sp.Expr:
        """Compute physical qubits for a logical footprint.

        Args:
            logical_qubits (sp.Expr | int | float): Logical qubits required by
                the logical resource estimate.

        Returns:
            sp.Expr: Physical qubit proxy including factory qubits.

        Raises:
            ValueError: If ``logical_qubits`` is provably non-positive.
        """
        logical_expr = _as_expr(logical_qubits, "logical_qubits")
        _validate_positive(logical_expr, "logical_qubits")
        return sp.simplify(
            logical_expr * self._physical_qubits_per_logical + self._factory_qubits
        )

    def runtime_seconds_for(
        self,
        logical_depth: _SympyLike,
        non_clifford_count: _SympyLike,
    ) -> sp.Expr:
        """Compute runtime for logical depth and non-Clifford work.

        Args:
            logical_depth (sp.Expr | int | float): Logical depth proxy.
            non_clifford_count (sp.Expr | int | float): Toffoli/T-equivalent
                count that consumes factory throughput.

        Returns:
            sp.Expr: Runtime proxy as the maximum of depth-limited and
                factory-throughput-limited runtime.

        Raises:
            ValueError: If either quantity is negative.
        """
        return sp.simplify(
            sp.Max(
                self.depth_limited_runtime_seconds_for(logical_depth),
                self.non_clifford_limited_runtime_seconds_for(non_clifford_count),
            )
        )

    def depth_limited_runtime_seconds_for(
        self,
        logical_depth: _SympyLike,
    ) -> sp.Expr:
        """Compute the depth-limited runtime component.

        Args:
            logical_depth (sp.Expr | int | float): Logical depth proxy.

        Returns:
            sp.Expr: Runtime in seconds implied by executing logical layers
            at the architecture's logical cycle time.

        Raises:
            ValueError: If ``logical_depth`` is negative.
        """
        depth_expr = _as_expr(logical_depth, "logical_depth")
        _validate_nonnegative(depth_expr, "logical_depth")
        return sp.simplify(depth_expr * self._logical_cycle_time_seconds)

    def non_clifford_limited_runtime_seconds_for(
        self,
        non_clifford_count: _SympyLike,
    ) -> sp.Expr:
        """Compute the non-Clifford-throughput runtime component.

        Args:
            non_clifford_count (sp.Expr | int | float): Toffoli/T-equivalent
                count that consumes factory or hardware throughput.

        Returns:
            sp.Expr: Runtime in seconds implied by non-Clifford throughput.

        Raises:
            ValueError: If ``non_clifford_count`` is negative.
        """
        non_clifford_expr = _as_expr(non_clifford_count, "non_clifford_count")
        _validate_nonnegative(non_clifford_expr, "non_clifford_count")
        return sp.simplify(non_clifford_expr / self._non_clifford_throughput_per_second)

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return architecture inputs as named symbolic values.

        Returns:
            dict[str, sp.Expr]: Architecture values keyed by stable names.
        """
        return {
            "physical_qubits_per_logical": self._physical_qubits_per_logical,
            "logical_cycle_time_seconds": self._logical_cycle_time_seconds,
            "factory_qubits": self._factory_qubits,
            "non_clifford_throughput_per_second": (
                self._non_clifford_throughput_per_second
            ),
        }

    @property
    def _physical_qubits_per_logical(self) -> sp.Expr:
        """Return physical overhead as a SymPy expression.

        Returns:
            sp.Expr: Converted physical qubit overhead.
        """
        return _as_expr(
            self.physical_qubits_per_logical,
            "physical_qubits_per_logical",
        )

    @property
    def _logical_cycle_time_seconds(self) -> sp.Expr:
        """Return logical cycle time as a SymPy expression.

        Returns:
            sp.Expr: Converted logical cycle time.
        """
        return _as_expr(self.logical_cycle_time_seconds, "logical_cycle_time_seconds")

    @property
    def _factory_qubits(self) -> sp.Expr:
        """Return factory qubits as a SymPy expression.

        Returns:
            sp.Expr: Converted factory qubit count.
        """
        return _as_expr(self.factory_qubits, "factory_qubits")

    @property
    def _non_clifford_throughput_per_second(self) -> sp.Expr:
        """Return non-Clifford throughput as a SymPy expression.

        Returns:
            sp.Expr: Converted throughput.
        """
        return _as_expr(
            self.non_clifford_throughput_per_second,
            "non_clifford_throughput_per_second",
        )


@dataclass(frozen=True)
class SurfaceCodeCostModel:
    """Derive rough FTQC proxy knobs from surface-code assumptions.

    This model is still intentionally compact. It does not estimate logical
    error rates, choose a distance from a target failure probability, or model
    detailed factory layouts. It turns explicit surface-code and factory
    assumptions into the generic ``FTQCCostModel`` consumed by
    ``estimate_physical_resources``.

    Attributes:
        code_distance (sp.Expr | int | float): Surface-code distance.
        physical_cycle_time_seconds (sp.Expr | int | float): Duration of one
            physical error-correction cycle, in seconds.
        physical_qubits_per_logical_factor (sp.Expr | int | float): Constant
            factor multiplying ``code_distance ** 2`` for one logical patch.
        logical_cycle_factor (sp.Expr | int | float): Constant factor
            multiplying ``code_distance`` to form one logical cycle.
        factory_count (sp.Expr | int | float): Number of parallel factories.
        physical_qubits_per_factory (sp.Expr | int | float): Physical qubits
            reserved for one factory.
        factory_cycles_per_non_clifford (sp.Expr | int | float): Logical cycles
            required by one factory output.

    Raises:
        ValueError: If a positive-valued quantity is non-positive or if
            ``physical_qubits_per_factory`` is negative.

    Example:
        >>> model = SurfaceCodeCostModel(
        ...     code_distance=7,
        ...     physical_cycle_time_seconds=1e-6,
        ...     physical_qubits_per_logical_factor=2,
        ...     logical_cycle_factor=3,
        ...     factory_count=4,
        ...     physical_qubits_per_factory=1000,
        ...     factory_cycles_per_non_clifford=2,
        ... )
        >>> model.physical_qubits_per_logical
        98
    """

    code_distance: _SympyLike
    physical_cycle_time_seconds: _SympyLike
    physical_qubits_per_logical_factor: _SympyLike = 2
    logical_cycle_factor: _SympyLike = 1
    factory_count: _SympyLike = 1
    physical_qubits_per_factory: _SympyLike = 0
    factory_cycles_per_non_clifford: _SympyLike = 1

    def __post_init__(self) -> None:
        """Validate surface-code fields after dataclass construction.

        Raises:
            ValueError: If a positive-valued quantity is non-positive or if
                ``physical_qubits_per_factory`` is negative.
        """
        for name, value in {
            key: value
            for key, value in self.resource_inputs().items()
            if key != "physical_qubits_per_factory"
        }.items():
            _validate_positive(value, name)
        _validate_nonnegative(
            self._physical_qubits_per_factory,
            "physical_qubits_per_factory",
        )

    @property
    def physical_qubits_per_logical(self) -> sp.Expr:
        """Return physical qubits used by one logical patch.

        Returns:
            sp.Expr: ``physical_qubits_per_logical_factor * code_distance**2``.
        """
        return sp.simplify(
            self._physical_qubits_per_logical_factor * self._code_distance**2
        )

    @property
    def logical_cycle_time_seconds(self) -> sp.Expr:
        """Return logical cycle time in seconds.

        Returns:
            sp.Expr: Logical cycle time derived from physical cycles.
        """
        return sp.simplify(
            self._logical_cycle_factor
            * self._code_distance
            * self._physical_cycle_time_seconds
        )

    @property
    def factory_qubits(self) -> sp.Expr:
        """Return physical qubits reserved for factories.

        Returns:
            sp.Expr: Total factory qubit count.
        """
        return sp.simplify(self._factory_count * self._physical_qubits_per_factory)

    @property
    def non_clifford_throughput_per_second(self) -> sp.Expr:
        """Return sustainable non-Clifford throughput.

        Returns:
            sp.Expr: Factory outputs per second under the compact factory
            model.
        """
        return sp.simplify(
            self._factory_count
            / (self._factory_cycles_per_non_clifford * self.logical_cycle_time_seconds)
        )

    def to_cost_model(self) -> FTQCCostModel:
        """Return the generic FTQC cost model represented by this architecture.

        Returns:
            FTQCCostModel: Generic physical-lift model derived from
            surface-code assumptions.
        """
        return FTQCCostModel(
            physical_qubits_per_logical=self.physical_qubits_per_logical,
            logical_cycle_time_seconds=self.logical_cycle_time_seconds,
            factory_qubits=self.factory_qubits,
            non_clifford_throughput_per_second=(
                self.non_clifford_throughput_per_second
            ),
        )

    def resource_inputs(self) -> dict[str, sp.Expr]:
        """Return raw surface-code inputs as named symbolic values.

        Returns:
            dict[str, sp.Expr]: Surface-code inputs keyed by stable names.
        """
        return {
            "code_distance": self._code_distance,
            "physical_cycle_time_seconds": self._physical_cycle_time_seconds,
            "physical_qubits_per_logical_factor": (
                self._physical_qubits_per_logical_factor
            ),
            "logical_cycle_factor": self._logical_cycle_factor,
            "factory_count": self._factory_count,
            "physical_qubits_per_factory": self._physical_qubits_per_factory,
            "factory_cycles_per_non_clifford": (self._factory_cycles_per_non_clifford),
        }

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return raw and derived architecture values.

        Returns:
            dict[str, sp.Expr]: Surface-code inputs plus generic FTQC cost
            values.
        """
        values = self.resource_inputs()
        values.update(self.to_cost_model().resource_values())
        return values

    @property
    def _code_distance(self) -> sp.Expr:
        """Return code distance as a SymPy expression.

        Returns:
            sp.Expr: Converted code distance.
        """
        return _as_expr(self.code_distance, "code_distance")

    @property
    def _physical_cycle_time_seconds(self) -> sp.Expr:
        """Return physical cycle time as a SymPy expression.

        Returns:
            sp.Expr: Converted physical cycle time.
        """
        return _as_expr(
            self.physical_cycle_time_seconds,
            "physical_cycle_time_seconds",
        )

    @property
    def _physical_qubits_per_logical_factor(self) -> sp.Expr:
        """Return logical patch qubit factor as a SymPy expression.

        Returns:
            sp.Expr: Converted patch qubit factor.
        """
        return _as_expr(
            self.physical_qubits_per_logical_factor,
            "physical_qubits_per_logical_factor",
        )

    @property
    def _logical_cycle_factor(self) -> sp.Expr:
        """Return logical cycle factor as a SymPy expression.

        Returns:
            sp.Expr: Converted logical cycle factor.
        """
        return _as_expr(self.logical_cycle_factor, "logical_cycle_factor")

    @property
    def _factory_count(self) -> sp.Expr:
        """Return factory count as a SymPy expression.

        Returns:
            sp.Expr: Converted factory count.
        """
        return _as_expr(self.factory_count, "factory_count")

    @property
    def _physical_qubits_per_factory(self) -> sp.Expr:
        """Return factory size as a SymPy expression.

        Returns:
            sp.Expr: Converted factory size.
        """
        return _as_expr(
            self.physical_qubits_per_factory,
            "physical_qubits_per_factory",
        )

    @property
    def _factory_cycles_per_non_clifford(self) -> sp.Expr:
        """Return factory cycles per non-Clifford output.

        Returns:
            sp.Expr: Converted factory latency.
        """
        return _as_expr(
            self.factory_cycles_per_non_clifford,
            "factory_cycles_per_non_clifford",
        )


@dataclass(frozen=True)
class ActiveVolumeCostModel:
    """Map logical estimates to active-volume resource proxies.

    Active-volume models price only the resources that are active during an
    operation, rather than multiplying the full physical footprint by the
    logical depth. The model is intentionally compact: it records explicit
    per-operation assumptions and derives a runtime from an active-volume
    throughput.

    Attributes:
        active_volume_per_logical_gate (sp.Expr | int | float): Active-volume
            units assigned to one logical gate or operation.
        active_volume_per_non_clifford (sp.Expr | int | float): Additional
            active-volume units assigned to one non-Clifford operation.
        active_volume_throughput_per_second (sp.Expr | int | float):
            Sustainable active-volume throughput.

    Raises:
        ValueError: If a positive-valued quantity is non-positive or if
            ``active_volume_per_non_clifford`` is negative.

    Example:
        >>> model = ActiveVolumeCostModel(
        ...     active_volume_per_logical_gate=3,
        ...     active_volume_per_non_clifford=2,
        ...     active_volume_throughput_per_second=10,
        ... )
        >>> model.active_volume_for(5, 2)
        19
    """

    active_volume_per_logical_gate: _SympyLike
    active_volume_per_non_clifford: _SympyLike = 0
    active_volume_throughput_per_second: _SympyLike = 1

    def __post_init__(self) -> None:
        """Validate active-volume fields after dataclass construction.

        Raises:
            ValueError: If a positive-valued quantity is non-positive or if
                ``active_volume_per_non_clifford`` is negative.
        """
        _validate_positive(
            self._active_volume_per_logical_gate,
            "active_volume_per_logical_gate",
        )
        _validate_nonnegative(
            self._active_volume_per_non_clifford,
            "active_volume_per_non_clifford",
        )
        _validate_positive(
            self._active_volume_throughput_per_second,
            "active_volume_throughput_per_second",
        )

    def active_volume_for(
        self,
        logical_gate_count: _SympyLike,
        non_clifford_count: _SympyLike,
    ) -> sp.Expr:
        """Compute active-volume units for logical work.

        Args:
            logical_gate_count (sp.Expr | int | float): Logical operation or
                gate count to price.
            non_clifford_count (sp.Expr | int | float): Non-Clifford
                operation count that receives the additional cost.

        Returns:
            sp.Expr: Active-volume units under this model.

        Raises:
            ValueError: If either input is provably negative.
        """
        logical_expr = _as_expr(logical_gate_count, "logical_gate_count")
        non_clifford_expr = _as_expr(non_clifford_count, "non_clifford_count")
        _validate_nonnegative(logical_expr, "logical_gate_count")
        _validate_nonnegative(non_clifford_expr, "non_clifford_count")
        return sp.simplify(
            logical_expr * self._active_volume_per_logical_gate
            + non_clifford_expr * self._active_volume_per_non_clifford
        )

    def runtime_seconds_for(self, active_volume: _SympyLike) -> sp.Expr:
        """Compute runtime from active volume and throughput.

        Args:
            active_volume (sp.Expr | int | float): Active-volume units to
                execute.

        Returns:
            sp.Expr: Runtime in seconds.

        Raises:
            ValueError: If ``active_volume`` is provably negative.
        """
        active_volume_expr = _as_expr(active_volume, "active_volume")
        _validate_nonnegative(active_volume_expr, "active_volume")
        return sp.simplify(
            active_volume_expr / self._active_volume_throughput_per_second
        )

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return architecture inputs as named symbolic values.

        Returns:
            dict[str, sp.Expr]: Active-volume model values keyed by stable
            names.
        """
        return {
            "active_volume_per_logical_gate": self._active_volume_per_logical_gate,
            "active_volume_per_non_clifford": self._active_volume_per_non_clifford,
            "active_volume_throughput_per_second": (
                self._active_volume_throughput_per_second
            ),
        }

    @property
    def _active_volume_per_logical_gate(self) -> sp.Expr:
        """Return per-logical-gate active volume as a SymPy expression.

        Returns:
            sp.Expr: Converted active-volume cost.
        """
        return _as_expr(
            self.active_volume_per_logical_gate,
            "active_volume_per_logical_gate",
        )

    @property
    def _active_volume_per_non_clifford(self) -> sp.Expr:
        """Return per-non-Clifford active volume as a SymPy expression.

        Returns:
            sp.Expr: Converted active-volume cost.
        """
        return _as_expr(
            self.active_volume_per_non_clifford,
            "active_volume_per_non_clifford",
        )

    @property
    def _active_volume_throughput_per_second(self) -> sp.Expr:
        """Return active-volume throughput as a SymPy expression.

        Returns:
            sp.Expr: Converted active-volume throughput.
        """
        return _as_expr(
            self.active_volume_throughput_per_second,
            "active_volume_throughput_per_second",
        )


@dataclass(frozen=True)
class FTQCPhysicalResourceEstimate:
    """Represent architecture-lifted FTQC resource proxies.

    Attributes:
        logical (ResourceEstimate): Existing Qamomile logical resource
            estimate used as the proxy-estimation input.
        logical_depth (sp.Expr): Logical depth proxy used for runtime.
        non_clifford_count (sp.Expr): Toffoli/T-equivalent count used for
            factory-throughput runtime.
        physical_qubits (sp.Expr): Physical qubit proxy.
        runtime_seconds (sp.Expr): Runtime proxy in seconds.
        architecture_values (dict[str, sp.Expr]): Architecture parameters
            used for the physical lift.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in logical
            and physical quantities.

    Example:
        >>> from qamomile.resource_estimation import GateCount, ResourceEstimate
        >>> logical = ResourceEstimate(
        ...     qubits=10,
        ...     gates=GateCount.zero().with_total(100),
        ... )
        >>> physical = estimate_physical_resources(logical, FTQCCostModel(2, 1, 3, 4))
        >>> physical.physical_qubits
        23
    """

    logical: ResourceEstimate
    logical_depth: sp.Expr
    non_clifford_count: sp.Expr
    physical_qubits: sp.Expr
    runtime_seconds: sp.Expr
    architecture_values: dict[str, sp.Expr] = field(default_factory=dict)
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)

    def substitute(self, **values: int | float) -> FTQCPhysicalResourceEstimate:
        """Substitute concrete values into logical and physical quantities.

        Args:
            **values (int | float): Mapping from symbol name to concrete value.

        Returns:
            FTQCPhysicalResourceEstimate: Estimate with substituted symbolic
                fields and refreshed free-symbol metadata.
        """
        substitutions: dict[Any, Any] = {}
        for name, value in values.items():
            substitutions[self.parameters.get(name, sp.Symbol(name))] = value

        logical = self.logical.substitute(**values)
        logical_depth = self.logical_depth.subs(substitutions).doit()
        non_clifford_count = self.non_clifford_count.subs(substitutions).doit()
        physical_qubits = self.physical_qubits.subs(substitutions).doit()
        runtime_seconds = self.runtime_seconds.subs(substitutions).doit()
        architecture_values = {
            name: expr.subs(substitutions).doit()
            for name, expr in self.architecture_values.items()
        }
        return FTQCPhysicalResourceEstimate(
            logical=logical,
            logical_depth=logical_depth,
            non_clifford_count=non_clifford_count,
            physical_qubits=physical_qubits,
            runtime_seconds=runtime_seconds,
            architecture_values=architecture_values,
            parameters=_collect_parameters(
                *resource_estimate_expressions(logical),
                logical_depth,
                non_clifford_count,
                physical_qubits,
                runtime_seconds,
                *architecture_values.values(),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the physical estimate to string-valued dictionaries.

        Returns:
            dict[str, Any]: JSON-friendly physical estimate.
        """
        return {
            "logical": self.logical.to_dict(),
            "logical_depth": str(self.logical_depth),
            "non_clifford_count": str(self.non_clifford_count),
            "depth_limited_runtime_seconds": str(self.depth_limited_runtime_seconds),
            "non_clifford_limited_runtime_seconds": str(
                self.non_clifford_limited_runtime_seconds
            ),
            "physical_qubits": str(self.physical_qubits),
            "runtime_seconds": str(self.runtime_seconds),
            "architecture_values": {
                name: str(value) for name, value in self.architecture_values.items()
            },
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
        }

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the physical estimate.

        Returns:
            dict[str, sp.Expr]: Logical and physical values keyed by canonical
            resource quantity names.
        """
        values = {
            "logical_qubits": self.logical.qubits,
            "logical_depth": self.logical_depth,
            "logical_spacetime_volume": sp.simplify(
                self.logical.qubits * self.logical_depth
            ),
            "non_clifford_count": self.non_clifford_count,
            "t_gates": self.logical.gates.t_gates,
            "multi_qubit_gates": self.logical.gates.multi_qubit,
            "depth_limited_runtime_seconds": self.depth_limited_runtime_seconds,
            "non_clifford_limited_runtime_seconds": (
                self.non_clifford_limited_runtime_seconds
            ),
            "physical_qubits": self.physical_qubits,
            "runtime_seconds": self.runtime_seconds,
            "physical_qubit_seconds": sp.simplify(
                self.physical_qubits * self.runtime_seconds
            ),
            **self.architecture_values,
        }
        if "qpe_iterations" in self.logical.gates.oracle_calls:
            values["qpe_iterations"] = self.logical.gates.oracle_calls["qpe_iterations"]
        return values

    @property
    def depth_limited_runtime_seconds(self) -> sp.Expr:
        """Return the logical-depth-limited runtime component.

        Returns:
            sp.Expr: ``logical_depth * logical_cycle_time_seconds``.
        """
        return sp.simplify(
            self.logical_depth * self.architecture_values["logical_cycle_time_seconds"]
        )

    @property
    def non_clifford_limited_runtime_seconds(self) -> sp.Expr:
        """Return the non-Clifford-throughput-limited runtime component.

        Returns:
            sp.Expr: ``non_clifford_count /
            non_clifford_throughput_per_second``.
        """
        return sp.simplify(
            self.non_clifford_count
            / self.architecture_values["non_clifford_throughput_per_second"]
        )


@dataclass(frozen=True)
class FTQCActiveVolumeResourceEstimate:
    """Represent active-volume-lifted FTQC resource proxies.

    Attributes:
        logical (ResourceEstimate): Existing Qamomile logical resource
            estimate used as the proxy-estimation input.
        logical_gate_count (sp.Expr): Logical operation or gate-count proxy
            used to compute active volume.
        non_clifford_count (sp.Expr): Toffoli/T-equivalent count receiving
            additional active-volume cost.
        active_volume (sp.Expr): Active-volume operation cost proxy.
        runtime_seconds (sp.Expr): Runtime proxy in seconds.
        architecture_values (dict[str, sp.Expr]): Architecture parameters
            used for the active-volume lift.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in logical
            and active-volume quantities.

    Example:
        >>> from qamomile.resource_estimation import GateCount, ResourceEstimate
        >>> logical = ResourceEstimate(
        ...     qubits=5,
        ...     gates=GateCount.zero().with_total(10),
        ... )
        >>> estimate = estimate_active_volume_resources(
        ...     logical,
        ...     ActiveVolumeCostModel(2, 1, 5),
        ... )
        >>> estimate.active_volume
        20
    """

    logical: ResourceEstimate
    logical_gate_count: sp.Expr
    non_clifford_count: sp.Expr
    active_volume: sp.Expr
    runtime_seconds: sp.Expr
    architecture_values: dict[str, sp.Expr] = field(default_factory=dict)
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)

    def substitute(self, **values: int | float) -> FTQCActiveVolumeResourceEstimate:
        """Substitute concrete values into active-volume quantities.

        Args:
            **values (int | float): Mapping from symbol name to concrete value.

        Returns:
            FTQCActiveVolumeResourceEstimate: Estimate with substituted fields
            and refreshed free-symbol metadata.
        """
        substitutions: dict[Any, Any] = {}
        for name, value in values.items():
            substitutions[self.parameters.get(name, sp.Symbol(name))] = value

        logical = self.logical.substitute(**values)
        logical_gate_count = self.logical_gate_count.subs(substitutions).doit()
        non_clifford_count = self.non_clifford_count.subs(substitutions).doit()
        active_volume = self.active_volume.subs(substitutions).doit()
        runtime_seconds = self.runtime_seconds.subs(substitutions).doit()
        architecture_values = {
            name: expr.subs(substitutions).doit()
            for name, expr in self.architecture_values.items()
        }
        return FTQCActiveVolumeResourceEstimate(
            logical=logical,
            logical_gate_count=logical_gate_count,
            non_clifford_count=non_clifford_count,
            active_volume=active_volume,
            runtime_seconds=runtime_seconds,
            architecture_values=architecture_values,
            parameters=_collect_parameters(
                *resource_estimate_expressions(logical),
                logical_gate_count,
                non_clifford_count,
                active_volume,
                runtime_seconds,
                *architecture_values.values(),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the active-volume estimate to string-valued dictionaries.

        Returns:
            dict[str, Any]: JSON-friendly active-volume estimate.
        """
        return {
            "logical": self.logical.to_dict(),
            "logical_gate_count": str(self.logical_gate_count),
            "non_clifford_count": str(self.non_clifford_count),
            "active_volume": str(self.active_volume),
            "active_volume_runtime_seconds": str(self.active_volume_runtime_seconds),
            "runtime_seconds": str(self.runtime_seconds),
            "architecture_values": {
                name: str(value) for name, value in self.architecture_values.items()
            },
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
        }

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the estimate.

        Returns:
            dict[str, sp.Expr]: Logical and active-volume values keyed by
            canonical resource quantity names.
        """
        values = {
            "logical_qubits": self.logical.qubits,
            "logical_operations": self.logical_gate_count,
            "non_clifford_count": self.non_clifford_count,
            "t_gates": self.logical.gates.t_gates,
            "multi_qubit_gates": self.logical.gates.multi_qubit,
            "active_volume": self.active_volume,
            "active_volume_runtime_seconds": self.active_volume_runtime_seconds,
            "runtime_seconds": self.runtime_seconds,
            **self.architecture_values,
        }
        if "qpe_iterations" in self.logical.gates.oracle_calls:
            values["qpe_iterations"] = self.logical.gates.oracle_calls["qpe_iterations"]
        return values

    @property
    def active_volume_runtime_seconds(self) -> sp.Expr:
        """Return the active-volume-throughput-limited runtime component.

        Returns:
            sp.Expr: ``active_volume / active_volume_throughput_per_second``.
        """
        return sp.simplify(
            self.active_volume
            / self.architecture_values["active_volume_throughput_per_second"]
        )


def estimate_physical_resources(
    logical: ResourceEstimate,
    cost_model: FTQCCostModel | SurfaceCodeCostModel,
    *,
    logical_depth: _SympyLike | None = None,
    non_clifford_count: _SympyLike | None = None,
) -> FTQCPhysicalResourceEstimate:
    """Lift a logical Qamomile estimate to rough physical resource proxies.

    Args:
        logical (ResourceEstimate): Existing logical circuit or algorithmic
            resource estimate.
        cost_model (FTQCCostModel | SurfaceCodeCostModel): Architecture model
            used to compute physical qubits and runtime. ``FTQCCostModel`` is
            a compact direct proxy; ``SurfaceCodeCostModel`` derives that proxy
            from explicit surface-code knobs.
        logical_depth (sp.Expr | int | float | None): Optional logical-depth
            proxy. Defaults to ``logical.gates.total`` because the current
            generic ``ResourceEstimate`` does not track depth separately.
        non_clifford_count (sp.Expr | int | float | None): Optional factory
            workload. Defaults to ``logical.gates.t_gates +
            logical.gates.multi_qubit``.

    Returns:
        FTQCPhysicalResourceEstimate: Physical resource proxy linked to
            ``logical``.

    Raises:
        TypeError: If ``logical`` is not a ``ResourceEstimate`` or if
            ``cost_model`` is not a supported architecture model.
    """
    if not isinstance(logical, ResourceEstimate):
        raise TypeError("logical must be a ResourceEstimate instance.")
    if isinstance(cost_model, SurfaceCodeCostModel):
        architecture_values = cost_model.resource_values()
        cost_model = cost_model.to_cost_model()
    elif isinstance(cost_model, FTQCCostModel):
        architecture_values = cost_model.resource_values()
    else:
        raise TypeError(
            "cost_model must be an FTQCCostModel or SurfaceCodeCostModel instance."
        )

    depth_expr = (
        logical.gates.total
        if logical_depth is None
        else _as_expr(logical_depth, "logical_depth")
    )
    non_clifford_expr = (
        sp.simplify(logical.gates.t_gates + logical.gates.multi_qubit)
        if non_clifford_count is None
        else _as_expr(non_clifford_count, "non_clifford_count")
    )
    physical_qubits = cost_model.physical_qubits_for(logical.qubits)
    runtime_seconds = cost_model.runtime_seconds_for(depth_expr, non_clifford_expr)
    return FTQCPhysicalResourceEstimate(
        logical=logical,
        logical_depth=sp.simplify(depth_expr),
        non_clifford_count=sp.simplify(non_clifford_expr),
        physical_qubits=sp.simplify(physical_qubits),
        runtime_seconds=sp.simplify(runtime_seconds),
        architecture_values=architecture_values,
        parameters=_collect_parameters(
            *resource_estimate_expressions(logical),
            depth_expr,
            non_clifford_expr,
            physical_qubits,
            runtime_seconds,
            *architecture_values.values(),
        ),
    )


def estimate_active_volume_resources(
    logical: ResourceEstimate,
    cost_model: ActiveVolumeCostModel,
    *,
    logical_gate_count: _SympyLike | None = None,
    non_clifford_count: _SympyLike | None = None,
) -> FTQCActiveVolumeResourceEstimate:
    """Lift a logical estimate to active-volume resource proxies.

    Args:
        logical (ResourceEstimate): Existing logical circuit or algorithmic
            resource estimate.
        cost_model (ActiveVolumeCostModel): Architecture model used to compute
            active volume and throughput-limited runtime.
        logical_gate_count (sp.Expr | int | float | None): Optional logical
            operation-count proxy. Defaults to ``logical.gates.total``.
        non_clifford_count (sp.Expr | int | float | None): Optional
            non-Clifford workload. Defaults to ``logical.gates.t_gates +
            logical.gates.multi_qubit``.

    Returns:
        FTQCActiveVolumeResourceEstimate: Active-volume resource proxy linked
        to ``logical``.

    Raises:
        TypeError: If ``logical`` is not a ``ResourceEstimate`` or
            ``cost_model`` is not an ``ActiveVolumeCostModel``.
    """
    if not isinstance(logical, ResourceEstimate):
        raise TypeError("logical must be a ResourceEstimate instance.")
    if not isinstance(cost_model, ActiveVolumeCostModel):
        raise TypeError("cost_model must be an ActiveVolumeCostModel instance.")

    logical_gate_expr = (
        logical.gates.total
        if logical_gate_count is None
        else _as_expr(logical_gate_count, "logical_gate_count")
    )
    non_clifford_expr = (
        sp.simplify(logical.gates.t_gates + logical.gates.multi_qubit)
        if non_clifford_count is None
        else _as_expr(non_clifford_count, "non_clifford_count")
    )
    active_volume = cost_model.active_volume_for(
        logical_gate_expr,
        non_clifford_expr,
    )
    runtime_seconds = cost_model.runtime_seconds_for(active_volume)
    architecture_values = cost_model.resource_values()
    return FTQCActiveVolumeResourceEstimate(
        logical=logical,
        logical_gate_count=sp.simplify(logical_gate_expr),
        non_clifford_count=sp.simplify(non_clifford_expr),
        active_volume=sp.simplify(active_volume),
        runtime_seconds=sp.simplify(runtime_seconds),
        architecture_values=architecture_values,
        parameters=_collect_parameters(
            *resource_estimate_expressions(logical),
            logical_gate_expr,
            non_clifford_expr,
            active_volume,
            runtime_seconds,
            *architecture_values.values(),
        ),
    )


def resource_estimate_expressions(logical: ResourceEstimate) -> tuple[sp.Expr, ...]:
    """Return all symbolic expressions carried by a resource estimate.

    Args:
        logical (ResourceEstimate): Logical estimate to inspect.

    Returns:
        tuple[sp.Expr, ...]: Logical resource expressions.
    """
    return (
        logical.qubits,
        logical.gates.total,
        logical.gates.single_qubit,
        logical.gates.two_qubit,
        logical.gates.multi_qubit,
        logical.gates.t_gates,
        logical.gates.clifford_gates,
        logical.gates.rotation_gates,
        *logical.gates.oracle_calls.values(),
        *logical.gates.oracle_queries.values(),
    )


def _collect_parameters(*expressions: sp.Expr) -> dict[str, sp.Symbol]:
    """Collect free SymPy symbols from resource expressions.

    Args:
        *expressions (sp.Expr): Resource expressions to scan.

    Returns:
        dict[str, sp.Symbol]: Free symbols keyed by their display names.
    """
    symbols: set[sp.Symbol] = set()
    for expr in expressions:
        for symbol in expr.free_symbols:
            if isinstance(symbol, sp.Symbol):
                symbols.add(symbol)
    return {str(symbol): symbol for symbol in sorted(symbols, key=str)}


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
