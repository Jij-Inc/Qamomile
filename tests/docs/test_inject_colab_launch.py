"""Unit tests for ``docs/scripts/inject_colab_launch.py``.

Covers the post-build HTML passes we own, focusing on
``sanitize_cite_ids`` — the citation-id sanitization pass added in
PR #385 to stop React hydration mismatches from cascading on
bibliography-heavy pages (mottonen / qsci). The other passes
(``inject_script_tag``, ``inline_theme_css``, etc.) have load-bearing
side effects and are exercised by the full ``./build.sh build`` run
on CI / RTD, which is enough for them; this file zooms in on the
sanitizer because its behavior is logic-heavy and easy to regress
silently (wrong scope, drift between id and href, collision aliasing).

Carries the ``docs`` pytest marker for category alignment with the
rest of ``tests/docs/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "docs" / "scripts" / "inject_colab_launch.py"

pytestmark = pytest.mark.docs


def _load_module():
    """Import ``docs/scripts/inject_colab_launch.py`` as a module.

    The file lives outside any package, so we import by path. We
    re-import on each call so test mutations of module globals (if
    any future test adds them) can't leak between tests.

    Returns:
        ModuleType: A freshly imported copy of the script module.
    """
    spec = importlib.util.spec_from_file_location("inject_colab_launch", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, SCRIPT_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules["inject_colab_launch"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    """Yield a freshly imported ``inject_colab_launch`` module."""
    return _load_module()


def _write_html(tmp_path: Path, name: str, body: str) -> Path:
    """Write ``body`` into ``tmp_path/name`` and return the path.

    Args:
        tmp_path (Path): pytest's per-test temp dir.
        name (str): Filename inside ``tmp_path``.
        body (str): Raw HTML body to write.

    Returns:
        Path: Absolute path to the written file.
    """
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ----------------------------------------------------------------- #
# _sanitize_cite_id (the low-level character collapse)              #
# ----------------------------------------------------------------- #


def test_sanitize_cite_id_collapses_url_chars(mod):
    """Slashes, colons, and dots in a DOI URL collapse to single dashes."""
    raw = "https://doi.org/10.48550/arxiv.quant-ph/0407010"
    assert mod._sanitize_cite_id(raw) == "https-doi-org-10-48550-arxiv-quant-ph-0407010"


def test_sanitize_cite_id_preserves_safe_chars(mod):
    """Already-safe labels round-trip exactly."""
    raw = "mottonen2004_state_prep"
    assert mod._sanitize_cite_id(raw) == raw


def test_sanitize_cite_id_strips_edges(mod):
    """Leading / trailing dashes are stripped after collapsing."""
    assert mod._sanitize_cite_id("/abc/") == "abc"
    assert mod._sanitize_cite_id("...abc...") == "abc"


def test_sanitize_cite_id_empty_fallback(mod):
    """All-unsafe input that collapses to empty gets a stable fallback."""
    # `///` -> '-' after sub, then stripped to '', then fallback to 'id'.
    assert mod._sanitize_cite_id("/" * 5) == "id"


# ----------------------------------------------------------------- #
# sanitize_cite_ids — scope, idempotency, href sync                 #
# ----------------------------------------------------------------- #


_DOI_BIB_HTML = """
<html><body>
<section id="references" class="myst-bibliography article-grid">
  <ol>
    <li class="myst-bibliography-item" id="cite-https://doi.org/10.48550/arxiv.quant-ph/0407010">
      Author. <i>Title</i>.
      <a href="https://doi.org/10.48550/arxiv.quant-ph/0407010">DOI</a>
    </li>
  </ol>
</section>
<a href="#cite-https://doi.org/10.48550/arxiv.quant-ph/0407010">In-text ref</a>
</body></html>
"""


def test_sanitize_rewrites_doi_id_and_matching_fragment(tmp_path, mod):
    """DOI-form id is collapsed; matching #cite fragment follows the same map."""
    p = _write_html(tmp_path, "page.html", _DOI_BIB_HTML)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'id="cite-https-doi-org-10-48550-arxiv-quant-ph-0407010"' in out
    assert 'href="#cite-https-doi-org-10-48550-arxiv-quant-ph-0407010"' in out
    # The outbound DOI URL is NOT a #cite fragment and must not be touched.
    assert 'href="https://doi.org/10.48550/arxiv.quant-ph/0407010"' in out


def test_sanitize_is_idempotent(tmp_path, mod):
    """A second run on the same file is a no-op."""
    p = _write_html(tmp_path, "page.html", _DOI_BIB_HTML)
    assert mod.sanitize_cite_ids(p) is True
    snapshot = p.read_text(encoding="utf-8")
    assert mod.sanitize_cite_ids(p) is False
    assert p.read_text(encoding="utf-8") == snapshot


def test_sanitize_already_safe_ids_are_noop(tmp_path, mod):
    """Bibliography that already uses safe-character ids triggers no rewrite."""
    safe_html = """
<html><body>
<section class="myst-bibliography">
  <ol><li id="cite-mottonen2004">x</li></ol>
</section>
<a href="#cite-mottonen2004">ref</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", safe_html)
    assert mod.sanitize_cite_ids(p) is False
    assert p.read_text(encoding="utf-8") == safe_html


def test_sanitize_skips_pages_without_bibliography(tmp_path, mod):
    """A page without ``<section class="myst-bibliography">`` is untouched.

    Even if it contains an ``id="cite-…"`` somewhere (e.g. a hand-rolled
    HTML cell or a code example), we must not rewrite it — only the
    SSR'd bibliography is in scope.
    """
    no_bib_html = """
<html><body>
<p>Code example: <span id="cite-https://example.com/x">tag</span></p>
<a href="#cite-https://example.com/x">ref</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", no_bib_html)
    assert mod.sanitize_cite_ids(p) is False
    assert p.read_text(encoding="utf-8") == no_bib_html


def test_sanitize_leaves_ids_outside_bibliography_alone(tmp_path, mod):
    """Unsafe ``id="cite-…"`` outside the bibliography section stays put."""
    mixed_html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-https://doi.org/safe">in scope</li>
</section>
<p>Out-of-scope: <span id="cite-https://other.example/x">should NOT be rewritten</span></p>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", mixed_html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    # In-scope id was sanitized.
    assert 'id="cite-https-doi-org-safe"' in out
    # Out-of-scope id was NOT.
    assert 'id="cite-https://other.example/x"' in out


def test_sanitize_leaves_unrelated_href_fragments_alone(tmp_path, mod):
    """``href="#cite-foo"`` whose label isn't in the bibliography stays put.

    Unknown fragments may point at a non-bibliography anchor (e.g.
    something hand-rolled). Rewriting them would break the link.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-https://doi.org/known">x</li>
</section>
<a href="#cite-https://doi.org/known">known</a>
<a href="#cite-https://doi.org/unknown">unknown — leave alone</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'href="#cite-https-doi-org-known"' in out
    # Unknown fragment must not have been rewritten.
    assert 'href="#cite-https://doi.org/unknown"' in out


def test_sanitize_decodes_percent_encoded_href(tmp_path, mod):
    """URL-encoded ``href="#cite-…"`` resolves to the same map entry as the id.

    Some renderers emit the fragment in percent-encoded form
    (``#cite-https%3A%2F%2F…``). We decode before lookup so the
    sanitized form still matches the unencoded ``<li id="cite-…">``.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-https://doi.org/abc">x</li>
</section>
<a href="#cite-https%3A%2F%2Fdoi.org%2Fabc">encoded ref</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'href="#cite-https-doi-org-abc"' in out


def test_sanitize_raises_on_collisions(tmp_path, mod):
    """Two distinct raw labels that collapse to the same sanitized id fail.

    ``cite-a/b`` and ``cite-a:b`` both sanitize to ``cite-a-b``;
    shipping that would produce duplicate ids in the rendered page.
    Build must fail loudly rather than silently emit broken HTML.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-a/b">first</li>
  <li id="cite-a:b">second</li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    with pytest.raises(RuntimeError, match="duplicate id attributes"):
        mod.sanitize_cite_ids(p)


def test_sanitize_raises_on_repeated_decoded_occurrences(tmp_path, mod):
    """Same decoded label appearing in multiple SSR ``id`` attributes fails.

    ``cite-a&amp;b`` and ``cite-a&b`` are two ``<li>`` items in the
    source whose ``id`` values DECODE to the same string (``a&b``),
    so both would rewrite to the same sanitized form ``cite-a-b``.
    The SSR is already broken in this case — duplicate ids at the
    DOM level — but our sanitizer would silently fan the two encoded
    forms into the canonical one and hide the upstream problem. Fail
    loud instead so the build surfaces the issue.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-a&amp;b">first</li>
  <li id="cite-a&b">second</li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    with pytest.raises(RuntimeError, match="duplicate id attributes"):
        mod.sanitize_cite_ids(p)


def test_sanitize_does_not_touch_outbound_doi_href(tmp_path, mod):
    """``href="https://doi.org/…"`` (no ``#cite-`` prefix) is left alone."""
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-https://doi.org/x">
    <a href="https://doi.org/x">canonical DOI</a>
  </li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    mod.sanitize_cite_ids(p)
    out = p.read_text(encoding="utf-8")
    # Internal id sanitized.
    assert 'id="cite-https-doi-org-x"' in out
    # External link still points to the real DOI URL.
    assert 'href="https://doi.org/x"' in out


# ----------------------------------------------------------------- #
# Class-token correctness — ``not-myst-bibliography`` must NOT match #
# ----------------------------------------------------------------- #


def test_sanitize_does_not_match_substring_class_token(tmp_path, mod):
    """A ``<section>`` whose class contains ``not-myst-bibliography``
    (or any other ``…myst-bibliography`` superstring) is NOT treated
    as a bibliography section.

    A naive ``\\bmyst-bibliography\\b`` regex finds a match inside
    ``not-myst-bibliography`` because the hyphen counts as a word
    boundary, which would silently sanitize ids in unrelated
    sections. We split the class attribute on whitespace and look for
    ``myst-bibliography`` as an exact token instead.
    """
    html = """
<html><body>
<section class="not-myst-bibliography prose">
  <li id="cite-https://doi.org/x">should NOT be rewritten</li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is False
    out = p.read_text(encoding="utf-8")
    assert 'id="cite-https://doi.org/x"' in out


def test_sanitize_matches_class_token_with_neighbors(tmp_path, mod):
    """``myst-bibliography`` is found when it shares the ``class``
    attribute with other unrelated tokens (the realistic mystmd
    case)."""
    html = """
<html><body>
<section id="references" class="prose myst-bibliography subgrid-gap">
  <li id="cite-https://doi.org/x">x</li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'id="cite-https-doi-org-x"' in out


# ----------------------------------------------------------------- #
# HTML entity decoding — id raws and href fragments                  #
# ----------------------------------------------------------------- #


def test_sanitize_decodes_html_entity_in_id(tmp_path, mod):
    """``id="cite-a&amp;b"`` is sanitized as if the literal label is
    ``a&b``.

    React computes the bibliography id from the citation's decoded
    label (the JSON ``label`` field, post-entity-decode), not from
    the raw SSR encoding. We must therefore decode entities BEFORE
    sanitizing so the SSR and client agree on the result. Without
    decoding, ``a&amp;b`` would collapse to ``a-amp-b`` while React
    would produce ``a-b``, reintroducing the very mismatch this pass
    is supposed to eliminate.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-a&amp;b">x</li>
</section>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'id="cite-a-b"' in out


def test_sanitize_tolerates_multi_space_and_newline_id_attribute(tmp_path, mod):
    """``<li ... \\n  id="cite-…">`` and similar wrap-formatted attrs match.

    Pretty-printers and hand-edited HTML can put a newline + indent
    between attributes, or multiple spaces around ``=``. A tight
    ``\\sid="`` regex would silently skip such ids and leave them
    unsanitized. The regex tolerates ``\\s+`` before ``id``, ``\\s*``
    around ``=``, and either quote style.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li class="myst-bibliography-item"
      id  =  "cite-https://doi.org/a">
    a
  </li>
</section>
<a href = "#cite-https://doi.org/a">ref</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert 'id  =  "cite-https-doi-org-a"' in out
    assert 'href = "#cite-https-doi-org-a"' in out


def test_sanitize_tolerates_single_quoted_id_attribute(tmp_path, mod):
    """``<li id='cite-…'>`` (single-quoted) is also handled.

    HTML5 lets attribute values be wrapped in either ``"`` or ``'``.
    The regex captures the quote char and reuses it as the closing
    quote so single-quoted citations round-trip with single quotes
    instead of getting silently rewritten with double quotes.
    """
    html = (
        "<html><body>"
        "<section class='myst-bibliography'>"
        "<li id='cite-https://doi.org/a'>x</li>"
        "</section>"
        "<a href='#cite-https://doi.org/a'>ref</a>"
        "</body></html>"
    )
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    assert "id='cite-https-doi-org-a'" in out
    assert "href='#cite-https-doi-org-a'" in out


def test_sanitize_resolves_both_percent_and_entity_href_forms(tmp_path, mod):
    """Percent-encoded and HTML-entity-encoded ``href="#cite-…"`` both
    decode to the same map key and rewrite consistently.

    Some renderers emit fragment identifiers with percent-encoding
    (``#cite-a%26b``); others emit HTML entities (``#cite-a&amp;b``);
    both must resolve to the same ``<li>`` and pick up the same
    sanitized form.
    """
    html = """
<html><body>
<section class="myst-bibliography">
  <li id="cite-a&amp;b">x</li>
</section>
<a href="#cite-a%26b">percent-encoded ref</a>
<a href="#cite-a&amp;b">entity ref</a>
</body></html>
"""
    p = _write_html(tmp_path, "page.html", html)
    assert mod.sanitize_cite_ids(p) is True
    out = p.read_text(encoding="utf-8")
    # Both encoded forms point at the same sanitized id.
    assert out.count('href="#cite-a-b"') == 2
