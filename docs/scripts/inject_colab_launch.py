#!/usr/bin/env python3
"""Inject Google Colab launcher script into built documentation HTML."""

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
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        injected_count, total_html = patch_language_build(docs_root, language)
        print(
            f"{language}: injected script tag into {injected_count}/{total_html} HTML files"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
