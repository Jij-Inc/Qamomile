#!/usr/bin/env python3
"""Inject ReadTheDocs search integration script into built documentation HTML.

The MyST book-theme search bar does not work on static HTML hosts.
This script injects a small JS shim that opens the ReadTheDocs server-side
search modal instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


SCRIPT_TAG_ID = "qamomile-rtd-search-script"
SCRIPT_FILE_NAME = "rtd-search.js"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the injection script.

    Returns:
        Parsed namespace with a ``languages`` attribute containing the
        list of language subdirectories (``"en"`` and/or ``"ja"``) to patch.
    """
    parser = argparse.ArgumentParser(
        description="Inject RTD search script into docs build output."
    )
    parser.add_argument(
        "languages",
        nargs="+",
        choices=("en", "ja"),
        help="Language build directories to patch.",
    )
    return parser.parse_args()


def inject_script_tag(html_path: Path, script_src: str) -> bool:
    """Inject a ``<script defer>`` tag just before ``</head>`` in an HTML file.

    The injection is idempotent: if the file already contains a tag with
    ``SCRIPT_TAG_ID``, the file is left untouched. If the file has no
    ``</head>`` element, no modification occurs.

    Args:
        html_path: Absolute path to the HTML file to patch.
        script_src: Relative ``src`` URL for the injected script tag,
            resolved from the HTML file's own directory.

    Returns:
        True if the file was modified, False if it was already patched
        or has no ``</head>`` element to anchor on.
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


def patch_language_build(docs_root: Path, language: str) -> tuple[int, int]:
    """Copy the RTD search script and inject tags into one language build.

    Copies ``docs/assets/rtd-search.js`` into the language's
    ``_build/html/build/`` directory, then walks every ``.html`` file under
    the build output and injects a ``<script defer>`` tag referencing the
    copied asset via a relative path.

    Args:
        docs_root: Absolute path to the ``docs`` directory.
        language: Language subdirectory name (``"en"`` or ``"ja"``).

    Returns:
        A tuple ``(injected_count, total_html_count)`` where the first
        element is the number of HTML files modified and the second is
        the total number of HTML files walked.

    Raises:
        RuntimeError: If the language's build directory does not exist
            or the source script is missing from ``docs/assets``.
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
    for html_file in html_files:
        relative_script = os.path.relpath(output_script, html_file.parent)
        relative_script = Path(relative_script).as_posix()
        if inject_script_tag(html_file, relative_script):
            injected_count += 1

    return injected_count, len(html_files)


def main() -> int:
    """Entry point for the injection script.

    Parses command-line arguments and patches each requested language
    build, printing a one-line summary per language.

    Returns:
        Process exit code (always 0 on success; non-zero paths raise).
    """
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        injected_count, total_html = patch_language_build(docs_root, language)
        print(
            f"{language}: injected RTD search script into {injected_count}/{total_html} HTML files"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
