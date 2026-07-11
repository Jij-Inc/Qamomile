"""Lift logical Qamomile resource estimates to rough FTQC resource proxies."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import sympy as sp

from qamomile.circuit.estimator import ResourceEstimate
from qamomile.resource_estimation._common import (
    _as_expr,
    _convert_fields,
    _SympyLike,
    _validate_nonnegative,
    _validate_positive,
)


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

    physical_qubits_per_logical: sp.Expr
    logical_cycle_time_seconds: sp.Expr
    factory_qubits: sp.Expr = sp.Integer(0)
    non_clifford_throughput_per_second: sp.Expr = sp.Integer(1)

    def __post_init__(self) -> None:
        """Convert and validate cost-model fields after construction.

        Numeric fields are sympified once here, so later accesses see
        ``sp.Expr`` values without per-access conversion.

        Raises:
            TypeError: If a field cannot be converted to a SymPy expression.
            ValueError: If any positive-valued field is non-positive or if
                ``factory_qubits`` is negative.
        """
        _convert_fields(
            self,
            (
                "physical_qubits_per_logical",
                "logical_cycle_time_seconds",
                "factory_qubits",
                "non_clifford_throughput_per_second",
            ),
        )
        _validate_positive(
            self.physical_qubits_per_logical,
            "physical_qubits_per_logical",
        )
        _validate_positive(
            self.logical_cycle_time_seconds,
            "logical_cycle_time_seconds",
        )
        _validate_nonnegative(self.factory_qubits, "factory_qubits")
        _validate_positive(
            self.non_clifford_throughput_per_second,
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
            logical_expr * self.physical_qubits_per_logical + self.factory_qubits
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
        return sp.simplify(depth_expr * self.logical_cycle_time_seconds)

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
        return sp.simplify(non_clifford_expr / self.non_clifford_throughput_per_second)

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return architecture inputs as named symbolic values.

        Returns:
            dict[str, sp.Expr]: Architecture values keyed by stable names.
        """
        return {
            "physical_qubits_per_logical": self.physical_qubits_per_logical,
            "logical_cycle_time_seconds": self.logical_cycle_time_seconds,
            "factory_qubits": self.factory_qubits,
            "non_clifford_throughput_per_second": (
                self.non_clifford_throughput_per_second
            ),
        }


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

    code_distance: sp.Expr
    physical_cycle_time_seconds: sp.Expr
    physical_qubits_per_logical_factor: sp.Expr = sp.Integer(2)
    logical_cycle_factor: sp.Expr = sp.Integer(1)
    factory_count: sp.Expr = sp.Integer(1)
    physical_qubits_per_factory: sp.Expr = sp.Integer(0)
    factory_cycles_per_non_clifford: sp.Expr = sp.Integer(1)

    def __post_init__(self) -> None:
        """Convert and validate surface-code fields after construction.

        Numeric fields are sympified once here, so later accesses see
        ``sp.Expr`` values without per-access conversion.

        Raises:
            TypeError: If a field cannot be converted to a SymPy expression.
            ValueError: If a positive-valued quantity is non-positive or if
                ``physical_qubits_per_factory`` is negative.
        """
        _convert_fields(
            self,
            (
                "code_distance",
                "physical_cycle_time_seconds",
                "physical_qubits_per_logical_factor",
                "logical_cycle_factor",
                "factory_count",
                "physical_qubits_per_factory",
                "factory_cycles_per_non_clifford",
            ),
        )
        for name, value in {
            key: value
            for key, value in self.resource_inputs().items()
            if key != "physical_qubits_per_factory"
        }.items():
            _validate_positive(value, name)
        _validate_nonnegative(
            self.physical_qubits_per_factory,
            "physical_qubits_per_factory",
        )

    @cached_property
    def physical_qubits_per_logical(self) -> sp.Expr:
        """Return physical qubits used by one logical patch.

        Returns:
            sp.Expr: ``physical_qubits_per_logical_factor * code_distance**2``.
        """
        return sp.simplify(
            self.physical_qubits_per_logical_factor * self.code_distance**2
        )

    @cached_property
    def logical_cycle_time_seconds(self) -> sp.Expr:
        """Return logical cycle time in seconds.

        Returns:
            sp.Expr: Logical cycle time derived from physical cycles.
        """
        return sp.simplify(
            self.logical_cycle_factor
            * self.code_distance
            * self.physical_cycle_time_seconds
        )

    @cached_property
    def factory_qubits(self) -> sp.Expr:
        """Return physical qubits reserved for factories.

        Returns:
            sp.Expr: Total factory qubit count.
        """
        return sp.simplify(self.factory_count * self.physical_qubits_per_factory)

    @cached_property
    def non_clifford_throughput_per_second(self) -> sp.Expr:
        """Return sustainable non-Clifford throughput.

        Returns:
            sp.Expr: Factory outputs per second under the compact factory
            model.
        """
        return sp.simplify(
            self.factory_count
            / (self.factory_cycles_per_non_clifford * self.logical_cycle_time_seconds)
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
            "code_distance": self.code_distance,
            "physical_cycle_time_seconds": self.physical_cycle_time_seconds,
            "physical_qubits_per_logical_factor": (
                self.physical_qubits_per_logical_factor
            ),
            "logical_cycle_factor": self.logical_cycle_factor,
            "factory_count": self.factory_count,
            "physical_qubits_per_factory": self.physical_qubits_per_factory,
            "factory_cycles_per_non_clifford": (self.factory_cycles_per_non_clifford),
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

    active_volume_per_logical_gate: sp.Expr
    active_volume_per_non_clifford: sp.Expr = sp.Integer(0)
    active_volume_throughput_per_second: sp.Expr = sp.Integer(1)

    def __post_init__(self) -> None:
        """Convert and validate active-volume fields after construction.

        Numeric fields are sympified once here, so later accesses see
        ``sp.Expr`` values without per-access conversion.

        Raises:
            TypeError: If a field cannot be converted to a SymPy expression.
            ValueError: If a positive-valued quantity is non-positive or if
                ``active_volume_per_non_clifford`` is negative.
        """
        _convert_fields(
            self,
            (
                "active_volume_per_logical_gate",
                "active_volume_per_non_clifford",
                "active_volume_throughput_per_second",
            ),
        )
        _validate_positive(
            self.active_volume_per_logical_gate,
            "active_volume_per_logical_gate",
        )
        _validate_nonnegative(
            self.active_volume_per_non_clifford,
            "active_volume_per_non_clifford",
        )
        _validate_positive(
            self.active_volume_throughput_per_second,
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
            logical_expr * self.active_volume_per_logical_gate
            + non_clifford_expr * self.active_volume_per_non_clifford
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
            active_volume_expr / self.active_volume_throughput_per_second
        )

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return architecture inputs as named symbolic values.

        Returns:
            dict[str, sp.Expr]: Active-volume model values keyed by stable
            names.
        """
        return {
            "active_volume_per_logical_gate": self.active_volume_per_logical_gate,
            "active_volume_per_non_clifford": self.active_volume_per_non_clifford,
            "active_volume_throughput_per_second": (
                self.active_volume_throughput_per_second
            ),
        }


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
            used for the physical lift. Must contain
            ``logical_cycle_time_seconds`` and
            ``non_clifford_throughput_per_second``.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in logical
            and physical quantities.

    Raises:
        ValueError: If ``architecture_values`` is missing an architecture
            key required by the runtime properties.

    Example:
        >>> import sympy as sp
        >>> from dataclasses import replace
        >>> from qamomile.resource_estimation import GateCount, ResourceEstimate
        >>> logical = ResourceEstimate(
        ...     qubits=sp.Integer(10),
        ...     gates=replace(GateCount.zero(), total=sp.Integer(100)),
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
    architecture_values: dict[str, sp.Expr]
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)

    _REQUIRED_ARCHITECTURE_KEYS = (
        "logical_cycle_time_seconds",
        "non_clifford_throughput_per_second",
    )

    def __post_init__(self) -> None:
        """Validate that runtime-critical architecture values are present.

        Raises:
            ValueError: If ``architecture_values`` is missing an architecture
                key required by the runtime properties.
        """
        _validate_architecture_keys(
            self.architecture_values,
            self._REQUIRED_ARCHITECTURE_KEYS,
        )

    def substitute(self, **values: int | float) -> FTQCPhysicalResourceEstimate:
        """Substitute concrete values into logical and physical quantities.

        Args:
            **values (int | float): Mapping from symbol name to concrete value.

        Returns:
            FTQCPhysicalResourceEstimate: Estimate with substituted symbolic
                fields and refreshed free-symbol metadata.
        """
        logical = self.logical.substitute(**values)
        subbed = _substitute_expressions(
            self.parameters,
            values,
            {
                "logical_depth": self.logical_depth,
                "non_clifford_count": self.non_clifford_count,
                "physical_qubits": self.physical_qubits,
                "runtime_seconds": self.runtime_seconds,
            },
        )
        architecture_values = _substitute_expressions(
            self.parameters,
            values,
            self.architecture_values,
        )
        return FTQCPhysicalResourceEstimate(
            logical=logical,
            architecture_values=architecture_values,
            parameters=_collect_parameters(
                *resource_estimate_expressions(logical),
                *subbed.values(),
                *architecture_values.values(),
            ),
            **subbed,
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
            resource quantity names, plus every oracle-call counter from the
            logical estimate under its own name (canonical keys win on
            collision).
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
        for name, count in self.logical.gates.oracle_calls.items():
            values.setdefault(name, count)
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
            used for the active-volume lift. Must contain
            ``active_volume_throughput_per_second``.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in logical
            and active-volume quantities.

    Raises:
        ValueError: If ``architecture_values`` is missing an architecture
            key required by the runtime properties.

    Example:
        >>> import sympy as sp
        >>> from dataclasses import replace
        >>> from qamomile.resource_estimation import GateCount, ResourceEstimate
        >>> logical = ResourceEstimate(
        ...     qubits=sp.Integer(5),
        ...     gates=replace(GateCount.zero(), total=sp.Integer(10)),
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
    architecture_values: dict[str, sp.Expr]
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)

    _REQUIRED_ARCHITECTURE_KEYS = ("active_volume_throughput_per_second",)

    def __post_init__(self) -> None:
        """Validate that runtime-critical architecture values are present.

        Raises:
            ValueError: If ``architecture_values`` is missing an architecture
                key required by the runtime properties.
        """
        _validate_architecture_keys(
            self.architecture_values,
            self._REQUIRED_ARCHITECTURE_KEYS,
        )

    def substitute(self, **values: int | float) -> FTQCActiveVolumeResourceEstimate:
        """Substitute concrete values into active-volume quantities.

        Args:
            **values (int | float): Mapping from symbol name to concrete value.

        Returns:
            FTQCActiveVolumeResourceEstimate: Estimate with substituted fields
            and refreshed free-symbol metadata.
        """
        logical = self.logical.substitute(**values)
        subbed = _substitute_expressions(
            self.parameters,
            values,
            {
                "logical_gate_count": self.logical_gate_count,
                "non_clifford_count": self.non_clifford_count,
                "active_volume": self.active_volume,
                "runtime_seconds": self.runtime_seconds,
            },
        )
        architecture_values = _substitute_expressions(
            self.parameters,
            values,
            self.architecture_values,
        )
        return FTQCActiveVolumeResourceEstimate(
            logical=logical,
            architecture_values=architecture_values,
            parameters=_collect_parameters(
                *resource_estimate_expressions(logical),
                *subbed.values(),
                *architecture_values.values(),
            ),
            **subbed,
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
            canonical resource quantity names, plus every oracle-call counter
            from the logical estimate under its own name (canonical keys win
            on collision).
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
        for name, count in self.logical.gates.oracle_calls.items():
            values.setdefault(name, count)
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
            ``cost_model`` is not a supported architecture model. Also raised
            if ``logical_depth`` or ``non_clifford_count`` cannot be converted
            to a SymPy expression.
        ValueError: If the cost model rejects the derived or overridden
            quantities, such as non-positive logical qubits or negative
            logical-depth or non-Clifford counts.
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
            ``cost_model`` is not an ``ActiveVolumeCostModel``. Also raised if
            ``logical_gate_count`` or ``non_clifford_count`` cannot be
            converted to a SymPy expression.
        ValueError: If the active-volume model rejects the derived or
            overridden quantities, such as negative logical-gate or
            non-Clifford counts.
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

    The gate-count fields are enumerated by ``GateCount.expressions()`` so
    that this helper stays in sync with the field list owned by
    ``qamomile.circuit.estimator``.

    Args:
        logical (ResourceEstimate): Logical estimate to inspect.

    Returns:
        tuple[sp.Expr, ...]: Logical resource expressions.
    """
    return (logical.qubits, *logical.gates.expressions())


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


def _substitute_expressions(
    parameters: dict[str, sp.Symbol],
    values: dict[str, int | float],
    expressions: dict[str, sp.Expr],
) -> dict[str, sp.Expr]:
    """Apply named numeric substitutions to a dictionary of expressions.

    Substitution matches symbols by display name over each expression's
    own free symbols, so symbols carrying assumptions (``positive=True``
    and similar) substitute correctly even when they are missing from
    ``parameters``.

    Args:
        parameters (dict[str, sp.Symbol]): Known free symbols keyed by their
            display names. Retained for API symmetry; matching is by name
            over each expression's free symbols.
        values (dict[str, int | float]): Substitutions keyed by symbol name.
        expressions (dict[str, sp.Expr]): Expressions to substitute into.

    Returns:
        dict[str, sp.Expr]: Substituted expressions under the same keys.
    """
    del parameters  # Matching is by name over each expression's symbols.
    result: dict[str, sp.Expr] = {}
    for name, expr in expressions.items():
        substitutions: dict[Any, Any] = {
            symbol: values[str(symbol)]
            for symbol in expr.free_symbols
            if str(symbol) in values
        }
        result[name] = expr.subs(substitutions).doit() if substitutions else expr
    return result


def _validate_architecture_keys(
    architecture_values: dict[str, sp.Expr],
    required_keys: tuple[str, ...],
) -> None:
    """Validate that an estimate carries its required architecture values.

    Args:
        architecture_values (dict[str, sp.Expr]): Architecture parameters
            attached to a physical or active-volume estimate.
        required_keys (tuple[str, ...]): Keys the estimate's runtime
            properties index unconditionally.

    Raises:
        ValueError: If any required key is missing, naming the missing keys.
    """
    missing = [key for key in required_keys if key not in architecture_values]
    if missing:
        raise ValueError(
            "architecture_values is missing required keys: "
            + ", ".join(missing)
            + ". Build estimates via estimate_physical_resources / "
            "estimate_active_volume_resources, or provide these keys directly."
        )
