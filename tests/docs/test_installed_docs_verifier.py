"""Unit tests for installed-wheel documentation verification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.docs
SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "verify-installed-docs.py"


def _load_verifier() -> ModuleType:
    """Load the release verifier without adding its directory to imports.

    Returns:
        Loaded verifier module.
    """

    spec = importlib.util.spec_from_file_location("verify_installed_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_docs_reject_repository_import_origin(tmp_path: Path) -> None:
    """An import satisfied by the checkout fails the release boundary."""

    verifier = _load_verifier()
    origin = tmp_path / "qamomile" / "__init__.py"

    with pytest.raises(SystemExit, match="resolved from the repository"):
        verifier.validate_module_origin("qamomile", str(origin), tmp_path)


def test_bilingual_ast_ignores_only_translated_docstrings(tmp_path: Path) -> None:
    """Translated docstrings compare equal while executable drift remains visible."""

    verifier = _load_verifier()
    english = tmp_path / "english.py"
    japanese = tmp_path / "japanese.py"
    english.write_text(
        '"""English."""\n\ndef value():\n    """Return one."""\n    return 1\n',
        encoding="utf-8",
    )
    japanese.write_text(
        '"""Japanese."""\n\ndef value():\n    """One in Japanese."""\n    return 1\n',
        encoding="utf-8",
    )

    assert verifier.executable_ast(english) == verifier.executable_ast(japanese)

    japanese.write_text("def value():\n    return 2\n", encoding="utf-8")
    assert verifier.executable_ast(english) != verifier.executable_ast(japanese)
