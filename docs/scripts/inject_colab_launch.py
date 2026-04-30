#!/usr/bin/env python3
"""Inject Google Colab launcher script into built documentation HTML.

Also rewrites the build-dir scratch path out of any GitHub edit / blob
URLs that mystmd embedded in the page header. Because ``./build.sh
build`` runs mystmd from ``docs/_build_src/<lang>/`` (so the committed
source tree never receives auto-managed injections), mystmd derives
the project-relative source path as
``docs/_build_src/<lang>/<section>/<slug>.ipynb`` and bakes that into
the "Edit on GitHub" / "View source" anchors. Those URLs would 404 on
``main`` because ``_build_src/`` is gitignored, and they would also
cause ``colab-launch.js`` to fail its ``^docs/(en|ja)/`` path check
and silently skip rendering the Colab button. We rewrite the URLs
back to their canonical ``docs/<lang>/...`` form here, after the HTML
has been copied into ``docs/<lang>/_build/html/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


SCRIPT_TAG_ID = "qamomile-colab-launch-script"
SCRIPT_FILE_NAME = "colab-launch.js"


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


def rewrite_build_src_paths(html_path: Path, language: str) -> bool:
    """Rewrite ``docs/_build_src/<lang>/`` → ``docs/<lang>/`` in the HTML.

    mystmd builds from ``docs/_build_src/<lang>/`` and bakes the build-dir
    path into the GitHub edit / blob URLs in the page header. The
    canonical source lives at ``docs/<lang>/`` on the branch; ``_build_src/``
    is gitignored. Strip the ``_build_src/`` segment so the "Edit on
    GitHub" / "View source" links resolve, and so ``colab-launch.js``'s
    ``^docs/(en|ja)/`` path check accepts the embedded edit URL and
    renders the Colab button.

    Returns True when the file was rewritten.
    """
    needle = f"docs/_build_src/{language}/"
    replacement = f"docs/{language}/"
    content = html_path.read_text(encoding="utf-8")
    if needle not in content:
        return False
    html_path.write_text(content.replace(needle, replacement), encoding="utf-8")
    return True


def patch_language_build(docs_root: Path, language: str) -> tuple[int, int, int]:
    """Run all post-build patches for one language's HTML output.

    Patches applied per HTML page:

    1. Rewrite build-dir paths in GitHub URLs (see
       :func:`rewrite_build_src_paths`).
    2. Inject the colab-launch ``<script>`` tag (see
       :func:`inject_script_tag`).

    Returns ``(injected_count, rewritten_count, total_html)``.
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

    html_files = sorted(html_root.rglob("*.html"))
    injected_count = 0
    rewritten_count = 0
    for html_file in html_files:
        if rewrite_build_src_paths(html_file, language):
            rewritten_count += 1
        relative_script = os.path.relpath(output_script, html_file.parent)
        relative_script = Path(relative_script).as_posix()
        if inject_script_tag(html_file, relative_script):
            injected_count += 1

    return injected_count, rewritten_count, len(html_files)


def main() -> int:
    """Patch every requested language's build output."""
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        injected_count, rewritten_count, total_html = patch_language_build(
            docs_root, language
        )
        print(
            f"{language}: injected script tag into {injected_count}/{total_html}, "
            f"rewrote build-dir paths in {rewritten_count}/{total_html} HTML files"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
