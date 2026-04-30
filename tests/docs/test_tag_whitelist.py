"""Verify every documentation article only uses tags in the whitelist.

The whitelist lives in ``docs/scripts/build_doc_tags.py`` as
``ALLOWED_TAGS``. The same script enforces it at build / pre-commit
time by raising ``UnknownTagError`` from ``_load_article``. This test
re-runs that loader against every ``.py`` source under the tagged
sections so a stray tag is caught in CI even when a contributor has
not installed the pre-commit hook locally.

Runs in the default unit-test step (no ``docs`` marker) — it only
parses frontmatter, not executes notebooks, so it stays cheap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "docs" / "scripts" / "build_doc_tags.py"


def _load_build_doc_tags():
    """Import ``build_doc_tags.py`` as a module without making it a package."""
    spec = importlib.util.spec_from_file_location("build_doc_tags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_doc_tags", module)
    spec.loader.exec_module(module)
    return module


_btags = _load_build_doc_tags()


def _all_article_paths() -> list[tuple[Path, str]]:
    """Collect every (py_path, section) pair under tagged sections."""
    pairs: list[tuple[Path, str]] = []
    for section in _btags.SECTIONS:
        for lang in ("en", "ja"):
            section_dir = _btags.DOCS_ROOT / lang / section
            if not section_dir.is_dir():
                continue
            for py in sorted(section_dir.glob("*.py")):
                pairs.append((py, section))
    return pairs


_ARTICLE_PATHS = _all_article_paths()


@pytest.mark.parametrize(
    "py_path,section",
    _ARTICLE_PATHS,
    ids=[str(p.relative_to(PROJECT_ROOT)) for p, _ in _ARTICLE_PATHS],
)
def test_article_tags_are_in_whitelist(py_path: Path, section: str) -> None:
    """Each article's frontmatter tags must be a subset of ALLOWED_TAGS.

    ``_load_article`` raises ``UnknownTagError`` for any out-of-whitelist
    tag, so simply calling it (and letting it return) is the assertion.
    Untagged articles return ``None`` and that is accepted — empty tag
    lists do not pollute the taxonomy.
    """
    _btags._load_article(py_path, section)


def test_allowed_tags_is_nonempty() -> None:
    """Sanity: at least one tag should be allowed (else nothing renders)."""
    assert _btags.ALLOWED_TAGS, "ALLOWED_TAGS is empty"
