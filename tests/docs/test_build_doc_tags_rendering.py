"""Verify generated documentation tag cards keep navigation clickable."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "docs" / "scripts" / "build_doc_tags.py"

pytestmark = pytest.mark.docs


def _load_build_doc_tags():
    """Import ``build_doc_tags.py`` as an isolated module for rendering tests."""
    spec = importlib.util.spec_from_file_location("build_doc_tags_rendering", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["build_doc_tags_rendering"] = module
    spec.loader.exec_module(module)
    return module


def _article(
    mod,
    title: str = "QAOA for MaxCut",
    card_body: str | None = "Build a QAOA circuit with `qaoa_state`.",
):
    """Build a minimal tagged article fixture."""
    return mod.Article(
        section="algorithm",
        slug="qaoa_maxcut",
        title=title,
        tags=("algorithm", "optimization"),
        thumbnail=None,
        py_path=PROJECT_ROOT / "docs" / "en" / "algorithm" / "qaoa_maxcut.py",
        card_body=card_body,
    )


def test_section_card_tags_are_clickable() -> None:
    """Section cards should link their title and keep tag chips clickable."""
    mod = _load_build_doc_tags()
    article = _article(mod)
    card = """:::{card}
:header: **QAOA for MaxCut**
:link: qaoa_maxcut
Build a QAOA circuit with `qaoa_state`.
:::
"""

    rendered = mod.CARD_DIRECTIVE_RE.sub(
        lambda match: mod._enhance_card_block(match, {article.slug: article}),
        card,
    )

    assert ":link:" not in rendered
    assert ":header: [**QAOA for MaxCut**](qaoa_maxcut)" in rendered
    assert '<a class="tag-chip" href="../tags/algorithm.md">algorithm</a>' in rendered
    assert (
        '<a class="tag-chip" href="../tags/optimization.md">optimization</a>'
        in rendered
    )
    assert "tag-chip-static" not in rendered
    assert "<code>qaoa_state</code>" in rendered


def test_tag_page_results_are_cards() -> None:
    """Per-tag result pages should render article results as full cards."""
    mod = _load_build_doc_tags()
    article = _article(
        mod,
        title="Steane [7,1,3] Code",
        card_body="CSS construction with <code>qaoa_state</code> as inline code.",
    )

    rendered = mod._render_tag_page(
        "algorithm",
        [article],
        mod.STRINGS["en"],
        ["algorithm", "optimization"],
    )

    assert "::::{grid} 1 1 1 1" in rendered
    assert ":::{card}" in rendered
    assert (
        ":header: [**Steane \\[7,1,3\\] Code**](../algorithm/qaoa_maxcut.ipynb)"
        in rendered
    )
    assert "CSS construction with <code>qaoa_state</code> as inline code." in rendered
    assert '<a class="tag-chip" href="./algorithm.md">algorithm</a>' in rendered
    assert '<a class="tag-chip" href="./optimization.md">optimization</a>' in rendered
    assert "### [" not in rendered
    assert "tag-chip-static" not in rendered


def test_section_card_body_extraction_reuses_handwritten_summary(tmp_path) -> None:
    """Section card body extraction should map card summaries by article slug."""
    mod = _load_build_doc_tags()
    index = tmp_path / "index.md"
    index.write_text(
        """::::{grid} 1 1 1 1

:::{card}
:header: **QAOA for MaxCut**
:link: qaoa_maxcut
Build a QAOA circuit with `qaoa_state`.
:::

::::
""",
        encoding="utf-8",
    )

    bodies = mod._extract_section_card_bodies(index)

    assert bodies == {
        "qaoa_maxcut": "Build a QAOA circuit with <code>qaoa_state</code>."
    }
