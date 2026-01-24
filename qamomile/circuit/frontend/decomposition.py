"""Decomposition strategy framework for composite gates.

This module provides the infrastructure for defining multiple decomposition
patterns for composite gates, enabling flexible gate synthesis strategies.

Example:
    class StandardQFTStrategy:
        @property
        def name(self) -> str:
            return "standard"

        def decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
            # Full precision QFT implementation
            ...

        def resources(self, num_qubits: int) -> ResourceMetadata:
            return ResourceMetadata(...)

    class ApproximateQFTStrategy:
        def __init__(self, truncation_depth: int = 3):
            self._k = truncation_depth

        @property
        def name(self) -> str:
            return f"approximate_k{self._k}"

        def decompose(self, qubits: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
            # Truncated rotations QFT
            ...

    # Register strategies
    QFT.register_strategy("standard", StandardQFTStrategy())
    QFT.register_strategy("approximate", ApproximateQFTStrategy(truncation_depth=3))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qamomile.circuit.frontend.handle.primitives import Qubit
    from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


@runtime_checkable
class DecompositionStrategy(Protocol):
    """Protocol for defining decomposition strategies.

    A decomposition strategy provides:
    1. A unique name for identification
    2. A decompose method that performs the actual decomposition
    3. Resource estimation for the decomposition

    Strategies allow the same composite gate to have multiple implementations
    with different trade-offs (e.g., precision vs. gate count).
    """

    @property
    def name(self) -> str:
        """Unique identifier for this strategy.

        Examples: "standard", "approximate", "approximate_k3"
        """
        ...

    def decompose(self, qubits: tuple["Qubit", ...]) -> tuple["Qubit", ...]:
        """Perform the decomposition.

        Args:
            qubits: Input qubits to decompose

        Returns:
            Output qubits after decomposition
        """
        ...

    def resources(self, num_qubits: int) -> "ResourceMetadata":
        """Return resource estimates for this decomposition.

        Args:
            num_qubits: Number of qubits the gate operates on

        Returns:
            ResourceMetadata with gate counts, depth estimates, etc.
        """
        ...


@dataclass
class DecompositionConfig:
    """Configuration for decomposition strategy selection.

    This configuration is passed to the transpiler to control which
    decomposition strategies are used for composite gates.

    Attributes:
        strategy_overrides: Map of gate name to strategy name.
            Use this to override the default strategy for specific gates.
            Example: {"qft": "approximate", "iqft": "approximate_k2"}

        strategy_params: Map of strategy name to parameters.
            Use this to configure strategy-specific parameters.
            Example: {"approximate": {"truncation_depth": 3}}

        default_strategy: Default strategy name if no override is specified.
            Use "standard" for full precision, "approximate" for reduced gates.
    """

    strategy_overrides: dict[str, str] = field(default_factory=dict)
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_strategy: str = "standard"

    def get_strategy_for_gate(self, gate_name: str) -> str:
        """Get the strategy name for a specific gate.

        Args:
            gate_name: The gate name (e.g., "qft", "iqft")

        Returns:
            Strategy name to use
        """
        return self.strategy_overrides.get(gate_name, self.default_strategy)

    def get_strategy_params(self, strategy_name: str) -> dict[str, Any]:
        """Get parameters for a specific strategy.

        Args:
            strategy_name: The strategy name

        Returns:
            Dictionary of parameters
        """
        return self.strategy_params.get(strategy_name, {})


class StrategyRegistry:
    """Registry for managing decomposition strategies.

    This class provides a centralized registry for strategies, allowing
    them to be looked up by name across the transpiler pipeline.

    Example:
        registry = StrategyRegistry()
        registry.register("qft", "standard", StandardQFTStrategy())
        registry.register("qft", "approximate", ApproximateQFTStrategy())

        strategy = registry.get("qft", "standard")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._strategies: dict[str, dict[str, DecompositionStrategy]] = {}

    def register(
        self,
        gate_name: str,
        strategy_name: str,
        strategy: DecompositionStrategy,
    ) -> None:
        """Register a strategy for a gate.

        Args:
            gate_name: The gate name (e.g., "qft", "iqft")
            strategy_name: The strategy name (e.g., "standard", "approximate")
            strategy: The strategy instance
        """
        if gate_name not in self._strategies:
            self._strategies[gate_name] = {}
        self._strategies[gate_name][strategy_name] = strategy

    def get(
        self,
        gate_name: str,
        strategy_name: str | None = None,
    ) -> DecompositionStrategy | None:
        """Get a strategy for a gate.

        Args:
            gate_name: The gate name
            strategy_name: The strategy name (uses "standard" if None)

        Returns:
            The strategy instance, or None if not found
        """
        if gate_name not in self._strategies:
            return None

        strategy_name = strategy_name or "standard"
        return self._strategies[gate_name].get(strategy_name)

    def list_strategies(self, gate_name: str) -> list[str]:
        """List available strategies for a gate.

        Args:
            gate_name: The gate name

        Returns:
            List of strategy names
        """
        if gate_name not in self._strategies:
            return []
        return list(self._strategies[gate_name].keys())

    def list_gates(self) -> list[str]:
        """List all gates with registered strategies.

        Returns:
            List of gate names
        """
        return list(self._strategies.keys())


# Global registry instance
_global_registry = StrategyRegistry()


def get_global_registry() -> StrategyRegistry:
    """Get the global strategy registry.

    Returns:
        The global StrategyRegistry instance
    """
    return _global_registry


def register_strategy(
    gate_name: str,
    strategy_name: str,
    strategy: DecompositionStrategy,
) -> None:
    """Register a strategy in the global registry.

    Args:
        gate_name: The gate name (e.g., "qft", "iqft")
        strategy_name: The strategy name (e.g., "standard", "approximate")
        strategy: The strategy instance
    """
    _global_registry.register(gate_name, strategy_name, strategy)


def get_strategy(
    gate_name: str,
    strategy_name: str | None = None,
) -> DecompositionStrategy | None:
    """Get a strategy from the global registry.

    Args:
        gate_name: The gate name
        strategy_name: The strategy name (uses "standard" if None)

    Returns:
        The strategy instance, or None if not found
    """
    return _global_registry.get(gate_name, strategy_name)
