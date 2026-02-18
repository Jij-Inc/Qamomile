"""TOC entry generation and myst.yml injection."""

from __future__ import annotations

from pathlib import Path

BEGIN_MARKER = "# --- API Reference (auto-generated) ---"
END_MARKER = "# --- End API Reference ---"


def build_toc_entries(
    subpackages: list[str],
    split_packages: set[str],
    split_data: dict[str, list[str]],
) -> list[str]:
    """Build myst.yml TOC YAML snippet lines.

    Generates extensionless file references (MyST standard).
    """
    lines = [
        BEGIN_MARKER,
        "    - title: API Reference",
        "      file: api/index",
        "      children:",
    ]
    for name in subpackages:
        if name in split_packages:
            lines.append(f"        - file: api/{name}/index")
            for sub_file in split_data.get(name, []):
                lines.append(f"        - file: api/{name}/{sub_file}")
        else:
            lines.append(f"        - file: api/{name}")
    lines.append(END_MARKER)
    return lines


def inject_toc(myst_yml_path: Path, toc_lines: list[str]) -> None:
    """Inject API Reference TOC entries into a myst.yml file.

    Uses marker comments for reliable replacement.
    """
    content = myst_yml_path.read_text()
    toc_block = "\n".join(toc_lines)

    # Check for existing markers
    if BEGIN_MARKER in content and END_MARKER in content:
        begin_idx = content.index(BEGIN_MARKER)
        end_idx = content.index(END_MARKER) + len(END_MARKER)
        # Include trailing newline if present
        if end_idx < len(content) and content[end_idx] == "\n":
            end_idx += 1
        content = content[:begin_idx] + toc_block + "\n" + content[end_idx:]
    else:
        # First time: remove any existing API Reference section (legacy format)
        # and insert before the "site:" line
        import re

        content = re.sub(
            r" *- title: API Reference\n"
            r"(?:      file: [^\n]+\n)?"
            r"(?:      children:\n)?"
            r"(?:        - file: [^\n]+\n)*",
            "",
            content,
        )
        # Remove excessive blank lines before site:
        content = re.sub(r"\n{3,}(site:)", r"\n\n\1", content)
        content = content.replace("\nsite:\n", f"\n{toc_block}\n\nsite:\n")

    myst_yml_path.write_text(content)
    print(f"  Injected API TOC into {myst_yml_path}")
