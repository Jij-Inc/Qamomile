"""Verify every documentation article carries the right tag invariants.

Three invariants are enforced per article (en + ja, every section):

1. The article declares ``tags:`` in its first markdown cell's MyST
   frontmatter — empty / missing tags fail.
2. The tag list contains the **section tag** matching the article's
   containing directory (``tutorial`` / ``algorithm`` / ``usage`` /
   ``integration``). Section tags are the foundation of the
   discovery UX — every article is reachable via its section's
   ``tags/<section>.md`` page.
3. Every tag in the list is in ``ALLOWED_TAGS`` (the whitelist in
   ``docs/scripts/build_doc_tags.py``).

This test is the *only* enforcement point: ``build_doc_tags.py``
itself does no validation (so a stray tag during local development
does not crash the build), and there is no pre-commit hook. CI
catches every violation before merge.

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
def test_article_tags_satisfy_invariants(py_path: Path, section: str) -> None:
    """Each article must (1) declare tags, (2) include its section tag,
    and (3) keep every tag inside ALLOWED_TAGS.
    """
    article = _btags._load_article(py_path, section)

    # Invariant 1: tags: frontmatter is mandatory.
    # ``_load_article`` returns None when the .py has no frontmatter or
    # the frontmatter has an empty / missing ``tags:`` list. Both cases
    # are violations now that section tags are required.
    assert article is not None, (
        f"{py_path.relative_to(PROJECT_ROOT)} is missing the `tags:` "
        f"frontmatter (or it is empty). Every article must declare at "
        f"least its section tag ({section!r})."
    )

    # Invariant 2: the section tag must be present.
    # We surface the article's containing directory through the chip
    # cloud / per-tag page UX, so the article must carry the matching
    # tag — otherwise it would not appear under tags/<section>.md.
    assert section in article.tags, (
        f"{py_path.relative_to(PROJECT_ROOT)} does not carry its section "
        f"tag {section!r}. Articles under docs/<lang>/{section}/ must "
        f"include {section!r} in their `tags:` list (alongside any "
        f"topical tags)."
    )

    # Invariant 3: every tag is in the whitelist.
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
