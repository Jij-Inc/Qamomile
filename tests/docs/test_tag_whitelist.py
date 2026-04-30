"""Verify every documentation article only uses tags in the whitelist.

The whitelist lives in ``docs/scripts/build_doc_tags.py`` as
``ALLOWED_TAGS``. This test is the *only* enforcement point: the
script itself no longer validates (so a stray tag during local
development does not crash the build), and there is no pre-commit
hook. CI catches the tag drift before merge, which is sufficient.

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

    Untagged articles (``_load_article`` returns ``None``) are accepted —
    an empty tag list does not pollute the taxonomy.
    """
    article = _btags._load_article(py_path, section)
    if article is None:
        return
    unknown = sorted(set(article.tags) - _btags.ALLOWED_TAGS)
    assert not unknown, (
        f"{py_path.relative_to(PROJECT_ROOT)} uses unknown tag(s) {unknown}. "
        f"Allowed: {sorted(_btags.ALLOWED_TAGS)}. "
        "Adding a new tag to ALLOWED_TAGS is a deliberate maintainer "
        "decision — please confirm with the project owner before extending "
        "the whitelist."
    )


def test_allowed_tags_is_nonempty() -> None:
    """Sanity: at least one tag should be allowed (else nothing renders)."""
    assert _btags.ALLOWED_TAGS, "ALLOWED_TAGS is empty"
