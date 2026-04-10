#!/usr/bin/env python3
"""Inject custom client-side scripts into built documentation HTML.

This script copies a set of JS assets from ``docs/assets/`` into each
language's build output and injects ``<script defer>`` tags pointing at
them just before ``</head>`` in every HTML file.

The current set of injections is declared in ``INJECTIONS`` below. To add
a new client-side enhancement:

1. Place the JS file in ``docs/assets/``.
2. Append a ``ScriptInjection`` entry to ``INJECTIONS`` with a unique
   ``tag_id`` (used as the HTML ``id`` attribute) and the file name.
3. Re-run the build.

Each injection is idempotent — re-running the script will not duplicate
tags that are already present.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScriptInjection:
    """Declarative description of one client-side script to inject.

    Attributes:
        tag_id: HTML ``id`` attribute applied to the injected
            ``<script>`` element. Must be unique across all injections;
            also used as the idempotency sentinel when re-running the
            patcher.
        script_file_name: File name (without directory) of the source
            JS asset under ``docs/assets/``. The same file name is used
            for the copy in the build output's ``build/`` directory.
        description: Short human-readable label printed in build output.
    """

    tag_id: str
    script_file_name: str
    description: str


INJECTIONS: tuple[ScriptInjection, ...] = (
    ScriptInjection(
        tag_id="qamomile-colab-launch-script",
        script_file_name="colab-launch.js",
        description="Colab launch button",
    ),
    ScriptInjection(
        tag_id="qamomile-rtd-search-script",
        script_file_name="rtd-search.js",
        description="ReadTheDocs search integration",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the injection script.

    Returns:
        Parsed namespace with a ``languages`` attribute containing the
        list of language subdirectories (``"en"`` and/or ``"ja"``) to patch.
    """
    parser = argparse.ArgumentParser(
        description="Inject client-side scripts into docs build output."
    )
    parser.add_argument(
        "languages",
        nargs="+",
        choices=("en", "ja"),
        help="Language build directories to patch.",
    )
    return parser.parse_args()


def inject_script_tag(html_path: Path, script_src: str, tag_id: str) -> bool:
    """Inject a ``<script defer>`` tag just before ``</head>`` in an HTML file.

    The injection is idempotent: if the file already contains a tag with
    ``tag_id`` as an ``id`` attribute, the file is left untouched. If the
    file has no ``</head>`` element, no modification occurs.

    Args:
        html_path: Absolute path to the HTML file to patch.
        script_src: Relative ``src`` URL for the injected script tag,
            resolved from the HTML file's own directory.
        tag_id: Value for the ``id`` attribute of the injected tag.
            Used both for HTML targeting and as the idempotency sentinel.

    Returns:
        True if the file was modified, False if it was already patched
        or has no ``</head>`` element to anchor on.
    """
    content = html_path.read_text(encoding="utf-8")

    # Look for the specific `id="<tag_id>"` attribute pattern rather than a
    # raw substring match, so a stray mention of ``tag_id`` inside page body
    # content (e.g. a code snippet in a tutorial) does not falsely block
    # injection.
    id_attribute = f'id="{tag_id}"'
    if id_attribute in content:
        return False

    closing_head = "</head>"
    if closing_head not in content:
        return False

    script_tag = (
        f'<script defer src="{script_src}" {id_attribute}></script>{closing_head}'
    )
    updated = content.replace(closing_head, script_tag, 1)
    html_path.write_text(updated, encoding="utf-8")
    return True


def patch_language_build(
    docs_root: Path,
    language: str,
    injection: ScriptInjection,
) -> tuple[int, int]:
    """Copy one script asset and inject its tag across one language build.

    Copies ``docs/assets/<script_file_name>`` into the language's
    ``_build/html/build/`` directory, then walks every ``.html`` file
    under the build output and injects a ``<script defer>`` tag
    referencing the copied asset via a relative path.

    Args:
        docs_root: Absolute path to the ``docs`` directory.
        language: Language subdirectory name (``"en"`` or ``"ja"``).
        injection: Declarative description of the script to inject.

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

    source_script = docs_root / "assets" / injection.script_file_name
    if not source_script.exists():
        raise RuntimeError(f"Missing source script: {source_script}")

    output_script = html_root / "build" / injection.script_file_name
    output_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_script, output_script)

    html_files = sorted(html_root.rglob("*.html"))
    injected_count = 0
    for html_file in html_files:
        relative_script = os.path.relpath(output_script, html_file.parent)
        relative_script = Path(relative_script).as_posix()
        if inject_script_tag(html_file, relative_script, injection.tag_id):
            injected_count += 1

    return injected_count, len(html_files)


def main() -> int:
    """Entry point for the consolidated injection script.

    Parses command-line arguments and applies every injection in
    ``INJECTIONS`` to each requested language build, printing a one-line
    summary per (language, script) pair.

    Returns:
        Process exit code (always 0 on success; non-zero paths raise).
    """
    args = parse_args()
    docs_root = Path(__file__).resolve().parents[1]

    for language in args.languages:
        for injection in INJECTIONS:
            injected, total = patch_language_build(docs_root, language, injection)
            print(
                f"{language}: injected {injection.description} "
                f"({injection.script_file_name}) into {injected}/{total} HTML files"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
