"""Configuration shared by semantic preparation and target compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionRule,
)


@dataclass
class CompilerConfig:
    """Configure semantic preparation and target-independent rewrites.

    Args:
        decomposition (DecompositionConfig): Composite-gate decomposition
            choices. Defaults to the standard decomposition configuration.
        substitutions (SubstitutionConfig): Callable substitution rules.
            Defaults to no substitutions.
    """

    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    substitutions: SubstitutionConfig = field(default_factory=SubstitutionConfig)

    @classmethod
    def with_strategies(
        cls,
        strategy_overrides: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> "CompilerConfig":
        """Create configuration with named decomposition strategies.

        Args:
            strategy_overrides (dict[str, str] | None): Gate-name to strategy
                mapping. Defaults to an empty mapping.
            **kwargs (Any): Additional :class:`CompilerConfig` constructor
                arguments.

        Returns:
            CompilerConfig: Configuration containing matching decomposition
                and substitution rules.
        """
        overrides = strategy_overrides or {}
        decomposition = DecompositionConfig(strategy_overrides=overrides)
        rules = [
            SubstitutionRule(source_name=name, strategy=strategy)
            for name, strategy in overrides.items()
        ]
        return cls(
            decomposition=decomposition,
            substitutions=SubstitutionConfig(rules=rules),
            **kwargs,
        )


TranspilerConfig = CompilerConfig
"""Backward name for :class:`CompilerConfig` during backend migration."""
