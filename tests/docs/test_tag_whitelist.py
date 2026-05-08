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

Carries the ``docs`` pytest marker for category alignment with the
rest of ``tests/docs/`` — even though this file is cheap (frontmatter
parse only, no notebook execution), the marker name reflects the
test's *subject matter* (documentation articles), which keeps the
``tests/docs/`` directory and the ``docs`` marker in lock-step. The
existing CI workflow's ``Run docs tests`` step (``pytest -m docs``)
picks it up alongside ``test_tutorials.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "docs" / "scripts" / "build_doc_tags.py"

# Apply the ``docs`` marker to every test in this module so it routes
# through the workflow's ``Run docs tests`` step alongside the rest of
# tests/docs/. The marker reflects subject matter, not cost — these
# checks are cheap (frontmatter parse), but they live in the docs/
# test bucket for taxonomy.
pytestmark = pytest.mark.docs


def _load_build_doc_tags():
    """Import ``build_doc_tags.py`` as a module without making it a package.

    ``build_doc_tags.py`` reads ``DOCS_ROOT_OVERRIDE`` at import time
    to decide which docs tree to scan. CI must always validate the
    in-repo ``docs/`` tree regardless of the caller's environment, so
    we strip ``DOCS_ROOT_OVERRIDE`` before executing the module — a
    leftover value (from a previous ``./build.sh`` invocation in the
    same shell, or from a CI job that sets it for an unrelated step)
    would otherwise make the tests pass/fail nondeterministically by
    silently scanning the wrong directory.
    """
    spec = importlib.util.spec_from_file_location("build_doc_tags", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``setdefault`` would leave a stale ``build_doc_tags`` entry from
    # an earlier import in place, so anything else that imports the
    # module later in the same process would see a different object
    # than the one we just executed and returned. Overwrite
    # unconditionally so the entry in ``sys.modules`` and our return
    # value always refer to the same module instance.
    sys.modules["build_doc_tags"] = module
    saved_override = os.environ.pop("DOCS_ROOT_OVERRIDE", None)
    try:
        spec.loader.exec_module(module)
    finally:
        if saved_override is not None:
            os.environ["DOCS_ROOT_OVERRIDE"] = saved_override
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


def test_article_discovery_is_nonempty() -> None:
    """Sanity: article-path discovery must turn up at least one article.

    Without this guard, an unexpected ``DOCS_ROOT`` / path issue (e.g. a
    typo'd ``SECTIONS`` entry or a refactor that moves the docs tree)
    would leave ``_ARTICLE_PATHS`` empty. The parametrized
    ``test_article_tags_satisfy_invariants`` would then expand to zero
    cases and silently pass — defeating the entire enforcement point.
    Assert that discovery actually walked something.
    """
    assert _ARTICLE_PATHS, (
        "Article discovery returned no (path, section) pairs. Check that "
        f"DOCS_ROOT={_btags.DOCS_ROOT!s} points at the docs tree and that "
        f"SECTIONS={_btags.SECTIONS!r} matches the on-disk directory names."
    )
