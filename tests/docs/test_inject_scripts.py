"""Unit tests for the docs build-output script-injection helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from docs.scripts.inject_scripts import (
    INJECTIONS,
    ScriptInjection,
    inject_script_tag,
    patch_language_build,
)

# ---------- inject_script_tag ----------


def test_inject_script_tag_adds_tag_before_closing_head(tmp_path: Path) -> None:
    """Tag is inserted immediately before </head> when not yet present."""
    html = tmp_path / "page.html"
    html.write_text("<html><head><title>x</title></head><body></body></html>")

    modified = inject_script_tag(html, "build/foo.js", "qamomile-foo")

    assert modified is True
    content = html.read_text()
    assert (
        '<script defer src="build/foo.js" id="qamomile-foo"></script></head>'
        in content
    )
    # Sanity: tag is BEFORE </head>, not after.
    assert content.index("qamomile-foo") < content.index("</head>")


def test_inject_script_tag_is_idempotent(tmp_path: Path) -> None:
    """Re-running on an already-injected file leaves the file untouched."""
    html = tmp_path / "page.html"
    original = (
        '<html><head><script defer src="build/foo.js" id="qamomile-foo">'
        "</script></head><body></body></html>"
    )
    html.write_text(original)

    modified = inject_script_tag(html, "build/foo.js", "qamomile-foo")

    assert modified is False
    assert html.read_text() == original


def test_inject_script_tag_returns_false_when_no_head(tmp_path: Path) -> None:
    """Files with no </head> anchor are skipped without raising."""
    html = tmp_path / "page.html"
    html.write_text("<html><body>no head element</body></html>")

    modified = inject_script_tag(html, "build/foo.js", "qamomile-foo")

    assert modified is False
    assert html.read_text() == "<html><body>no head element</body></html>"


def test_inject_script_tag_injects_only_first_head(tmp_path: Path) -> None:
    """Only the first </head> occurrence is anchored even if duplicates exist."""
    html = tmp_path / "page.html"
    html.write_text("<head></head><head></head>")

    inject_script_tag(html, "build/foo.js", "qamomile-foo")

    content = html.read_text()
    # The first </head> got the script, the second is still bare.
    assert content.count('id="qamomile-foo"') == 1


# ---------- patch_language_build ----------


@pytest.fixture
def fake_docs_root(tmp_path: Path) -> Path:
    """Build a minimal docs/ tree fake suitable for patch_language_build."""
    docs_root = tmp_path / "docs"
    (docs_root / "assets").mkdir(parents=True)
    (docs_root / "assets" / "rtd-search.js").write_text("// fake script")

    en_html = docs_root / "en" / "_build" / "html"
    en_html.mkdir(parents=True)
    (en_html / "index.html").write_text("<head></head><body>en</body>")
    sub = en_html / "tutorial"
    sub.mkdir()
    (sub / "page.html").write_text("<head></head><body>tut</body>")
    return docs_root


_RTD_INJECTION = ScriptInjection(
    tag_id="qamomile-rtd-search-script",
    script_file_name="rtd-search.js",
    description="ReadTheDocs search integration",
)


def test_patch_language_build_copies_script_and_injects_all_html(
    fake_docs_root: Path,
) -> None:
    """The script is copied to build/ and every HTML file is patched once."""
    injected, total = patch_language_build(fake_docs_root, "en", _RTD_INJECTION)

    assert (injected, total) == (2, 2)
    copied = fake_docs_root / "en" / "_build" / "html" / "build" / "rtd-search.js"
    assert copied.exists()
    assert copied.read_text() == "// fake script"

    for html in (fake_docs_root / "en" / "_build" / "html").rglob("*.html"):
        assert 'id="qamomile-rtd-search-script"' in html.read_text()


def test_patch_language_build_uses_relative_script_src(fake_docs_root: Path) -> None:
    """Nested HTML files reference the script via the correct relative path."""
    patch_language_build(fake_docs_root, "en", _RTD_INJECTION)

    # Top-level page should reference build/rtd-search.js
    top = (fake_docs_root / "en" / "_build" / "html" / "index.html").read_text()
    assert 'src="build/rtd-search.js"' in top

    # Nested page (one level deeper) should walk back up via ../
    nested = (
        fake_docs_root / "en" / "_build" / "html" / "tutorial" / "page.html"
    ).read_text()
    assert 'src="../build/rtd-search.js"' in nested


def test_patch_language_build_is_idempotent(fake_docs_root: Path) -> None:
    """Running the patcher twice modifies zero files on the second run."""
    patch_language_build(fake_docs_root, "en", _RTD_INJECTION)
    second_injected, second_total = patch_language_build(
        fake_docs_root, "en", _RTD_INJECTION
    )

    assert (second_injected, second_total) == (0, 2)


def test_patch_language_build_raises_when_build_dir_missing(tmp_path: Path) -> None:
    """A missing language build directory raises RuntimeError."""
    docs_root = tmp_path / "docs"
    (docs_root / "assets").mkdir(parents=True)
    (docs_root / "assets" / "rtd-search.js").write_text("// fake")

    with pytest.raises(RuntimeError, match="build directory not found"):
        patch_language_build(docs_root, "en", _RTD_INJECTION)


def test_patch_language_build_raises_when_source_script_missing(
    tmp_path: Path,
) -> None:
    """A missing asset file in docs/assets raises RuntimeError."""
    docs_root = tmp_path / "docs"
    (docs_root / "assets").mkdir(parents=True)
    (docs_root / "en" / "_build" / "html").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Missing source script"):
        patch_language_build(docs_root, "en", _RTD_INJECTION)


# ---------- INJECTIONS declarative table ----------


def test_injections_have_unique_tag_ids() -> None:
    """Every entry in INJECTIONS uses a distinct HTML id (idempotency safety)."""
    tag_ids = [inj.tag_id for inj in INJECTIONS]
    assert len(tag_ids) == len(set(tag_ids))


def test_injections_have_unique_script_file_names() -> None:
    """Every entry in INJECTIONS uses a distinct asset file name."""
    names = [inj.script_file_name for inj in INJECTIONS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("injection", INJECTIONS)
def test_injections_reference_existing_asset(injection: ScriptInjection) -> None:
    """Each declared injection points to a real file under docs/assets/."""
    repo_root = Path(__file__).resolve().parents[2]
    asset = repo_root / "docs" / "assets" / injection.script_file_name
    assert asset.exists(), f"Missing asset for {injection.tag_id}: {asset}"
