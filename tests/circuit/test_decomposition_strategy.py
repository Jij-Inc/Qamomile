"""Tests for decomposition strategy framework."""

import math
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.decomposition import (
    DecompositionConfig,
    DecompositionStrategy,
    StrategyRegistry,
    get_global_registry,
    register_strategy,
    get_strategy,
)
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)
from qamomile.circuit.stdlib.qft import QFT, IQFT
from qamomile.circuit.stdlib.qft_strategies import (
    StandardQFTStrategy,
    ApproximateQFTStrategy,
    StandardIQFTStrategy,
    ApproximateIQFTStrategy,
)


class TestDecompositionConfig:
    """Tests for DecompositionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DecompositionConfig()
        assert config.strategy_overrides == {}
        assert config.strategy_params == {}
        assert config.default_strategy == "standard"

    def test_get_strategy_for_gate_with_override(self):
        """Test getting strategy with override."""
        config = DecompositionConfig(
            strategy_overrides={"qft": "approximate"},
        )
        assert config.get_strategy_for_gate("qft") == "approximate"
        assert config.get_strategy_for_gate("iqft") == "standard"

    def test_get_strategy_params(self):
        """Test getting strategy parameters."""
        config = DecompositionConfig(
            strategy_params={
                "approximate": {"truncation_depth": 3},
            },
        )
        assert config.get_strategy_params("approximate") == {"truncation_depth": 3}
        assert config.get_strategy_params("standard") == {}


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving strategies."""
        registry = StrategyRegistry()
        strategy = StandardQFTStrategy()

        registry.register("qft", "standard", strategy)
        retrieved = registry.get("qft", "standard")

        assert retrieved is strategy

    def test_get_nonexistent(self):
        """Test getting nonexistent strategy."""
        registry = StrategyRegistry()
        assert registry.get("qft", "nonexistent") is None
        assert registry.get("nonexistent", "standard") is None

    def test_list_strategies(self):
        """Test listing strategies for a gate."""
        registry = StrategyRegistry()
        registry.register("qft", "standard", StandardQFTStrategy())
        registry.register("qft", "approximate", ApproximateQFTStrategy())

        strategies = registry.list_strategies("qft")
        assert "standard" in strategies
        assert "approximate" in strategies

    def test_list_gates(self):
        """Test listing gates with strategies."""
        registry = StrategyRegistry()
        registry.register("qft", "standard", StandardQFTStrategy())
        registry.register("iqft", "standard", StandardIQFTStrategy())

        gates = registry.list_gates()
        assert "qft" in gates
        assert "iqft" in gates


class TestQFTStrategies:
    """Tests for QFT strategies."""

    def test_qft_has_strategies(self):
        """Test that QFT class has strategies registered."""
        strategies = QFT.list_strategies()
        assert "standard" in strategies
        assert "approximate" in strategies

    def test_iqft_has_strategies(self):
        """Test that IQFT class has strategies registered."""
        strategies = IQFT.list_strategies()
        assert "standard" in strategies
        assert "approximate" in strategies

    def test_get_standard_strategy(self):
        """Test getting standard strategy."""
        strategy = QFT.get_strategy("standard")
        assert strategy is not None
        assert strategy.name == "standard"

    def test_get_approximate_strategy(self):
        """Test getting approximate strategy."""
        strategy = QFT.get_strategy("approximate")
        assert strategy is not None
        assert "approximate" in strategy.name

    def test_standard_qft_resources(self):
        """Test standard QFT resource estimation."""
        strategy = StandardQFTStrategy()
        resources = strategy.resources(4)

        assert resources.t_gate_count == 0
        assert resources.custom_metadata["num_h_gates"] == 4
        assert resources.custom_metadata["num_cp_gates"] == 6  # 4*3/2
        assert resources.custom_metadata["precision"] == "full"

    def test_approximate_qft_resources(self):
        """Test approximate QFT resource estimation."""
        strategy = ApproximateQFTStrategy(truncation_depth=2)
        resources = strategy.resources(4)

        assert resources.t_gate_count == 0
        # With k=2, should have fewer CP gates than full QFT
        assert resources.custom_metadata["num_cp_gates"] < 6
        assert resources.custom_metadata["truncation_depth"] == 2

    def test_qft_get_resources_for_strategy(self):
        """Test getting resources for specific strategy."""
        qft = QFT(5)

        standard_resources = qft.get_resources_for_strategy("standard")
        approx_resources = qft.get_resources_for_strategy("approximate")

        assert standard_resources is not None
        assert approx_resources is not None

        # Approximate should have fewer gates
        standard_gates = standard_resources.custom_metadata["total_gates"]
        approx_gates = approx_resources.custom_metadata["total_gates"]
        assert approx_gates <= standard_gates


class TestCompositeGateStrategyRegistry:
    """Tests for CompositeGate strategy registry."""

    def test_register_strategy(self):
        """Test registering a custom strategy."""

        class CustomGate(CompositeGate):
            _strategies = {}  # Fresh registry

            def __init__(self, n: int):
                self._n = n

            @property
            def num_target_qubits(self) -> int:
                return self._n

        # Register custom strategy
        custom_strategy = StandardQFTStrategy()
        CustomGate.register_strategy("my_strategy", custom_strategy)

        assert "my_strategy" in CustomGate.list_strategies()
        assert CustomGate.get_strategy("my_strategy") is custom_strategy

    def test_set_default_strategy(self):
        """Test setting default strategy."""

        class AnotherGate(CompositeGate):
            _strategies = {"a": StandardQFTStrategy(), "b": ApproximateQFTStrategy()}
            _default_strategy = "a"

            @property
            def num_target_qubits(self) -> int:
                return 2

        AnotherGate.set_default_strategy("b")
        assert AnotherGate._default_strategy == "b"

    def test_set_invalid_default_strategy(self):
        """Test setting invalid default strategy raises error."""

        class YetAnotherGate(CompositeGate):
            _strategies = {"a": StandardQFTStrategy()}
            _default_strategy = "a"

            @property
            def num_target_qubits(self) -> int:
                return 2

        with pytest.raises(ValueError):
            YetAnotherGate.set_default_strategy("nonexistent")
