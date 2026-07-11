"""Tests for compiler-side callable implementation selection."""

from qamomile.circuit.frontend.decomposition import DecompositionConfig


def test_decomposition_config_selects_override_or_default() -> None:
    """Selection is configuration only, without a global strategy registry."""
    config = DecompositionConfig(
        strategy_overrides={"qft": "native"},
        default_strategy="body",
    )

    assert config.get_strategy_for_gate("qft") == "native"
    assert config.get_strategy_for_gate("custom") == "body"


def test_decomposition_config_returns_parameter_copy() -> None:
    """Callers cannot mutate the stored strategy parameters accidentally."""
    config = DecompositionConfig(strategy_params={"approximate": {"degree": 3}})

    params = config.get_strategy_params("approximate")
    params["degree"] = 7

    assert config.get_strategy_params("approximate") == {"degree": 3}
