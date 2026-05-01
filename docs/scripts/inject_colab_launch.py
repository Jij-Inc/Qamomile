#!/usr/bin/env python3
"""Post-build HTML patches for the Qamomile docs.

Originally just a colab-launch script-tag injector; the script has
since absorbed two more passes that all run against the same set of
``docs/<lang>/_build/html/*.html`` (and ``*.json``) files, so they live
together for a single rglob walk.

The three passes:

1. Rewrite build-dir paths in GitHub URLs (HTML + JSON). ``./build.sh
   build`` runs mystmd from ``docs/_build_src/<lang>/`` so the
   committed source tree stays free of auto-managed injections;
   mystmd then derives the project-relative source path as
   ``docs/_build_src/<lang>/<section>/<slug>.ipynb`` and bakes that
   into the "Edit on GitHub" / "View source" anchors in **both** the
   HTML page header and the per-page JSON data layer the SPA hydrates
   from. Those URLs would 404 on ``main`` (``_build_src/`` is
   gitignored) and would also cause ``colab-launch.js`` to fail its
   ``^docs/(en|ja)/`` path check, dropping the Colab button after the
   first SPA navigation. We rewrite them to ``docs/<lang>/...``.

2. Inline the tag-chip CSS in ``<head>``. mystmd bundles
   ``docs/assets/custom-theme.css`` into the LAST
   ``<link rel="stylesheet" href="/myst-theme.css">`` of the page;
   that file arrives later than the 160 KB ``app-*.css``, so on first
   paint the browser renders each chip as a default ``<a>`` link
   (blue, underlined) and snaps into pill shape only once
   ``myst-theme.css`` finishes downloading. Re-emitting just the chip
   subsection of ``custom-theme.css`` as an inline ``<style>`` early
   in ``<head>`` removes that flash because the rules are parsed
   before any external stylesheet has arrived.

3. Inject the colab-launch ``<script>`` tag right before ``</head>``
   on every HTML. The script then renders the "Open in Colab" button
   client-side, gated to ``.ipynb``-derived pages by the path check
   that pass 1 protects.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


SCRIPT_TAG_ID = "qamomile-colab-launch-script"
SCRIPT_FILE_NAME = "colab-launch.js"

# Inline-CSS pass: avoids FOUC on the tag chip styling. The chip CSS
# lives in docs/assets/custom-theme.css, which mystmd bundles into the
# last <link rel="stylesheet" href="/myst-theme.css"> in <head>. The
# bundled myst-theme.css arrives later than the 160 KB app-*.css, so on
# first paint the browser displays each chip as a default <a> link
# (blue, underlined) and re-renders into pill shape only once
# myst-theme.css finishes downloading. Re-emitting the chip CSS as an
# inline <style> block early in <head> means the browser parses the
# rules during HTML parse, before any external stylesheet has arrived,
# so the chips render with their pill styling on first paint.
CHIP_CSS_STYLE_ID = "qamomile-chip-css-inline"
CHIP_CSS_BEGIN_MARKER = "/* === Tag chips ==="
# We stop slicing right before the next subsystem's marker so the
# inlined CSS stays scoped to chip styling — pulling the colab-button
# rules in too is harmless for FOUC but bloats every HTML page for no
# user-visible win (the colab button is JS-injected after page load,
# styling arrives in the same paint as the element).
CHIP_CSS_END_MARKER = "/* Google Colab launch button"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject Colab launcher script into docs build output."
    )
    parser.add_argument(
        "languages",
        nargs="+",
        choices=("en", "ja"),
        help="Language build directories to patch.",
    )
    return parser.parse_args()


def _read_chip_css(docs_root: Path) -> str:
    """Return the tag-chip section of ``docs/assets/custom-theme.css``.

    Slices between :data:`CHIP_CSS_BEGIN_MARKER` and
    :data:`CHIP_CSS_END_MARKER` so we re-emit only the rules that drive
    the chip's visual identity (pill shape, warm-amber palette, ``#``
    prefix, dark-mode variant) rather than the whole stylesheet.

    Raises:
        RuntimeError: when the markers cannot be located. Fail loud
            rather than silently inlining nothing — a missing marker
            usually means somebody refactored ``custom-theme.css`` and
            this script needs to be updated alongside it.
    """
    css_path = docs_root / "assets" / "custom-theme.css"
    if not css_path.exists():
        raise RuntimeError(f"Missing source CSS: {css_path}")
    text = css_path.read_text(encoding="utf-8")
    start = text.find(CHIP_CSS_BEGIN_MARKER)
    end = text.find(CHIP_CSS_END_MARKER)
    if start < 0 or end < 0 or start >= end:
        raise RuntimeError(
            f"Could not locate chip-CSS section in {css_path} (markers "
            f"{CHIP_CSS_BEGIN_MARKER!r} ... {CHIP_CSS_END_MARKER!r})"
        )
    return text[start:end].strip()


def inline_chip_css(html_path: Path, css_text: str) -> bool:
    """Inline the tag-chip CSS as a ``<style>`` block in ``<head>``.

    Inserted right after the opening ``<head>`` tag — earlier than the
    external ``<link rel="stylesheet">`` entries — so the rules apply
    on first paint and the chip never flashes as an unstyled ``<a>``.
    The block carries a stable ``id`` so the function is idempotent
    across re-runs of the patcher.

    Args:
        html_path: Path to the HTML file under
            ``docs/<lang>/_build/html/``.
        css_text: The chip-CSS payload (from :func:`_read_chip_css`).

    Returns:
        True when the file was patched. False if the page already
        carries the inline ``<style>`` (idempotent re-run) or has no
        ``<head>`` to anchor against (e.g. a 404 stub).
    """
    content = html_path.read_text(encoding="utf-8")
    if f'id="{CHIP_CSS_STYLE_ID}"' in content:
        return False
    head_open = "<head>"
    if head_open not in content:
        return False
    style_block = f'<style id="{CHIP_CSS_STYLE_ID}">{css_text}</style>'
    new = content.replace(head_open, head_open + style_block, 1)
    html_path.write_text(new, encoding="utf-8")
    return True


def inject_script_tag(html_path: Path, script_src: str) -> bool:
    """Insert the colab-launch ``<script>`` tag right before ``</head>``.

    Returns True when a tag was added. Returns False if the page already
    carries the tag (idempotency) or if the page has no ``</head>`` to
    anchor against (e.g. a 404 stub).
    """
    content = html_path.read_text(encoding="utf-8")
    if SCRIPT_TAG_ID in content:
        return False

    closing_head = "</head>"
    if closing_head not in content:
        return False

    script_tag = (
        f'<script defer src="{script_src}" id="{SCRIPT_TAG_ID}"></script>{closing_head}'
    )
    updated = content.replace(closing_head, script_tag, 1)
    html_path.write_text(updated, encoding="utf-8")
    return True


def rewrite_build_src_paths(target_path: Path, language: str) -> bool:
    """Rewrite ``docs/_build_src/<lang>/`` → ``docs/<lang>/`` in ``target_path``.

    mystmd builds from ``docs/_build_src/<lang>/`` and bakes the build-dir
    path into the GitHub edit / blob URLs both in the page-header HTML and
    in the per-page JSON data layer that the SPA hydrates from on
    client-side navigation. The canonical source lives at ``docs/<lang>/``
    on the branch; ``_build_src/`` is gitignored. Strip the ``_build_src/``
    segment so the "Edit on GitHub" / "View source" links resolve, and so
    ``colab-launch.js``'s ``^docs/(en|ja)/`` path check accepts the
    embedded edit URL and renders the Colab button.

    Patching only the HTML pages would leave the JSON data layer stale: a
    full page load would pick up the rewritten HTML, but as soon as the
    SPA navigated to another page it would re-hydrate the page header from
    the JSON and re-introduce the ``_build_src/`` URL — so the Colab
    button would silently disappear after the first navigation. The
    caller therefore runs this on the union of ``*.html`` and ``*.json``.

    Returns True when the file was rewritten.
    """
    needle = f"docs/_build_src/{language}/"
    replacement = f"docs/{language}/"
    content = target_path.read_text(encoding="utf-8")
    if needle not in content:
        return False
    target_path.write_text(content.replace(needle, replacement), encoding="utf-8")
    return True


def patch_language_build(
    docs_root: Path, language: str
) -> tuple[int, int, int, int, int]:
    """Run all post-build patches for one language's build output.

    Patches applied:

    1. Rewrite build-dir paths in GitHub URLs across both ``*.html`` and
       the SPA's ``*.json`` data layer (see
       :func:`rewrite_build_src_paths`).
    2. Inject the colab-launch ``<script>`` tag into every ``*.html``
       (see :func:`inject_script_tag`).
    3. Inline the tag-chip CSS into every ``*.html``'s ``<head>`` to
       avoid FOUC on chip styling (see :func:`inline_chip_css`).

    Returns:
        ``(injected_count, rewritten_count, css_inlined_count,
        total_html, total_json)``.
    """
    html_root = docs_root / language / "_build" / "html"
    if not html_root.exists():
        raise RuntimeError(f"{language}: build directory not found: {html_root}")

    source_script = docs_root / "assets" / SCRIPT_FILE_NAME
    if not source_script.exists():
        raise RuntimeError(f"Missing source script: {source_script}")

    output_script = html_root / "build" / SCRIPT_FILE_NAME
    output_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_script, output_script)

    chip_css = _read_chip_css(docs_root)

    html_files = sorted(html_root.rglob("*.html"))
    json_files = sorted(html_root.rglob("*.json"))
    injected_count = 0
    rewritten_count = 0
    css_inlined_count = 0

    # Rewrite build-dir paths across HTML + JSON. JSON is the SPA's data
    # layer; without rewriting it, post-navigation re-hydration puts the
    # _build_src/ URL back into the DOM and colab-launch.js drops the
    # button (see rewrite_build_src_paths' docstring).
    for target in (*html_files, *json_files):
        if rewrite_build_src_paths(target, language):
            rewritten_count += 1

    # Inline chip CSS + inject the script tag into HTML only. JSON pages
    # are not navigated to directly; the SPA loads them via the host
    # HTML, which already carries both patches.
    for html_file in html_files:
        if inline_chip_css(html_file, chip_css):
            css_inlined_count += 1
        relative_script = os.path.relpath(output_script, html_file.parent)
        relative_script = Path(relative_script).as_posix()
        if inject_script_tag(html_file, relative_script):
            injected_count += 1

    return (
        injected_count,
        rewritten_count,
        css_inlined_count,
        len(html_files),
        len(json_files),
    )


def main() -> int:
    """Patch every requested language's build output."""
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        (
            injected_count,
            rewritten_count,
            css_inlined_count,
            total_html,
            total_json,
        ) = patch_language_build(docs_root, language)
        print(
            f"{language}: injected script tag into {injected_count}/{total_html} HTML, "
            f"inlined chip CSS into {css_inlined_count}/{total_html} HTML, "
            f"rewrote build-dir paths in {rewritten_count}/{total_html + total_json} "
            f"({total_html} HTML + {total_json} JSON)"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
