"""Configuration for compiler implementation-strategy selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecompositionConfig:
    """Configure named implementation strategies for callable lowering.

    Implementations live on each callable definition. This object only records
    user selection; it is intentionally not a second global strategy registry.

    Args:
        strategy_overrides (dict[str, str]): Callable-name to strategy-name
            overrides.
        strategy_params (dict[str, dict[str, Any]]): Optional strategy
            parameters keyed by strategy name.
        default_strategy (str): Fallback strategy name. Defaults to
            ``"standard"``.
    """

    strategy_overrides: dict[str, str] = field(default_factory=dict)
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_strategy: str = "standard"

    def get_strategy_for_gate(self, gate_name: str) -> str:
        """Return the selected strategy name for a callable.

        Args:
            gate_name (str): Callable name.

        Returns:
            str: Explicit override or the configured default.
        """
        return self.strategy_overrides.get(gate_name, self.default_strategy)

    def get_strategy_params(self, strategy_name: str) -> dict[str, Any]:
        """Return parameters for one strategy.

        Args:
            strategy_name (str): Strategy name.

        Returns:
            dict[str, Any]: Copy of the configured parameter mapping.
        """
        return dict(self.strategy_params.get(strategy_name, {}))
