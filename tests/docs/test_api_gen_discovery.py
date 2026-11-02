"""Test package-membership discovery for the API reference generator."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.api_gen.discovery import module_belongs_to_package
from docs.api_gen.realizations import generate_realizations_table

# The docs marker keeps this file in the documentation CI job, which is the
# only job guaranteed to run when a pull request touches docs/api_gen alone.
pytestmark = pytest.mark.docs


def test_realization_reference_is_generated_from_native_definitions() -> None:
    from qamomile.lowering import list_realizations

    content = generate_realizations_table()
    for info in list_realizations():
        assert f"| `{info.name}` | `{info.family}` |" in content
        assert info.description in content
        for capability in info.required_capabilities:
            assert f"`{capability}`" in content


def test_explicit_public_modules_are_discovered() -> None:
    from docs.api_gen.config import ApiGenConfig
    from docs.api_gen.discovery import discover_subpackages

    assert "lowering" in discover_subpackages(ApiGenConfig())


def test_module_belongs_to_package_returns_true_for_nested_path(tmp_path: Path) -> None:
    package_dir = tmp_path / "qamomile"
    package_dir.mkdir()
    module = SimpleNamespace(filepath=package_dir / "submodule.py")

    assert module_belongs_to_package(module, package_dir) is True


def test_module_belongs_to_package_rejects_prefix_match_sibling(tmp_path: Path) -> None:
    package_dir = tmp_path / "qamomile"
    package_dir.mkdir()
    sibling_dir = tmp_path / "qamomile_extra"
    sibling_dir.mkdir()
    module = SimpleNamespace(filepath=sibling_dir / "submodule.py")

    assert module_belongs_to_package(module, package_dir) is False


def test_module_belongs_to_package_returns_false_without_filepath(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "qamomile"
    package_dir.mkdir()
    module = SimpleNamespace(filepath=None)

    assert module_belongs_to_package(module, package_dir) is False
