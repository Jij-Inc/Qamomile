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
        depth_expr = _as_expr(logical_depth, "logical_depth")
        non_clifford_expr = _as_expr(non_clifford_count, "non_clifford_count")
        _validate_nonnegative(depth_expr, "logical_depth")
        _validate_nonnegative(non_clifford_expr, "non_clifford_count")
        return sp.simplify(
            sp.Max(
                depth_expr * self._logical_cycle_time_seconds,
                non_clifford_expr / self._non_clifford_throughput_per_second,
            )
        )

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
            "physical_qubits": str(self.physical_qubits),
            "runtime_seconds": str(self.runtime_seconds),
            "architecture_values": {
                name: str(value) for name, value in self.architecture_values.items()
            },
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
        }


def estimate_physical_resources(
    logical: ResourceEstimate,
    cost_model: FTQCCostModel,
    *,
    logical_depth: _SympyLike | None = None,
    non_clifford_count: _SympyLike | None = None,
) -> FTQCPhysicalResourceEstimate:
    """Lift a logical Qamomile estimate to rough physical resource proxies.

    Args:
        logical (ResourceEstimate): Existing logical circuit or algorithmic
            resource estimate.
        cost_model (FTQCCostModel): Coarse architecture proxy used to compute
            physical qubits and runtime. It does not encode a concrete QEC
            architecture.
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
            ``cost_model`` is not an ``FTQCCostModel``.
    """
    if not isinstance(logical, ResourceEstimate):
        raise TypeError("logical must be a ResourceEstimate instance.")
    if not isinstance(cost_model, FTQCCostModel):
        raise TypeError("cost_model must be an FTQCCostModel instance.")

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
    architecture_values = cost_model.resource_values()
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
