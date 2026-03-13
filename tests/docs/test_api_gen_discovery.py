from pathlib import Path
from types import SimpleNamespace

from docs.api_gen.discovery import module_belongs_to_package


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


def test_module_belongs_to_package_returns_false_without_filepath(tmp_path: Path) -> None:
    package_dir = tmp_path / "qamomile"
    package_dir.mkdir()
    module = SimpleNamespace(filepath=None)

    assert module_belongs_to_package(module, package_dir) is False
