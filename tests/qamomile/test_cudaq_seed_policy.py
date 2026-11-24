"""CUDA-Q random-state policy without requiring the optional SDK."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qamomile.cudaq._integration import _selected_runtime_target


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_cudaq_sets_random_seed_only_when_explicit(
    monkeypatch, seed: int | None
) -> None:
    """An unspecified seed leaves SDK state alone; explicit zero reaches the SDK."""
    current = SimpleNamespace(name="original")

    def set_target(target):
        nonlocal current
        current = SimpleNamespace(name=target) if isinstance(target, str) else target

    setter = Mock()
    sdk = SimpleNamespace(
        get_target=lambda: current,
        set_target=set_target,
        set_random_seed=setter,
    )
    monkeypatch.setitem(sys.modules, "cudaq", sdk)
    with _selected_runtime_target("qpp-cpu", seed):
        assert current.name == "qpp-cpu"
    assert current.name == "original"
    if seed is None:
        setter.assert_not_called()
    else:
        setter.assert_called_once_with(seed)
