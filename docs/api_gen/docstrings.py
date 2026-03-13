"""Docstring parsing and rendering to MyST-compatible Markdown."""

from __future__ import annotations

import griffe

from .config import ApiGenConfig
from .rst_converter import rst_to_myst


def _escape_table_cell(text: str) -> str:
    """Escape pipe characters and newlines for markdown table cells."""
    return text.replace("|", "\\|").replace("\n", " ")


def get_first_line(docstring: griffe.Docstring | None) -> str:
    """Extract the first line of a docstring for summary tables."""
    if not docstring or not docstring.value:
        return ""
    converted = rst_to_myst(docstring.value.strip())
    return converted.splitlines()[0]


def write_docstring(
    lines: list[str],
    docstring: griffe.Docstring | None,
    config: ApiGenConfig,
) -> None:
    """Write docstring content as structured markdown using griffe's parser.

    Pre-processes the docstring with RST-to-MyST conversion before parsing.
    """
    if not docstring or not docstring.value:
        return

    # Pre-process RST markup before griffe parses it
    original_value = docstring.value
    docstring.value = rst_to_myst(original_value)

    try:
        sections = docstring.parse(config.docstring_style)
    except Exception:
        for line in docstring.value.strip().splitlines():
            lines.append(line)
        lines.append("")
        docstring.value = original_value
        return

    for section in sections:
        kind = section.kind

        if kind == griffe.DocstringSectionKind.text:
            for line in section.value.strip().splitlines():
                lines.append(line)
            lines.append("")

        elif kind == griffe.DocstringSectionKind.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for param in section.value:
                ann = _escape_table_cell(str(param.annotation)) if param.annotation else ""
                desc = (
                    _escape_table_cell(param.description)
                    if param.description
                    else ""
                )
                lines.append(f"| `{param.name}` | `{ann}` | {desc} |")
            lines.append("")

        elif kind == griffe.DocstringSectionKind.returns:
            lines.append("**Returns:**")
            lines.append("")
            for ret in section.value:
                ann = str(ret.annotation) if ret.annotation else ""
                desc = ret.description or ""
                if ann:
                    lines.append(f"`{ann}` — {desc}")
                else:
                    lines.append(desc)
            lines.append("")

        elif kind == griffe.DocstringSectionKind.raises:
            lines.append("**Raises:**")
            lines.append("")
            for exc in section.value:
                ann = str(exc.annotation) if exc.annotation else ""
                desc = exc.description or ""
                lines.append(f"- `{ann}` — {desc}")
            lines.append("")

        elif kind == griffe.DocstringSectionKind.admonition:
            title = getattr(section, "title", "Note")
            adm = section.value
            desc = adm.description if hasattr(adm, "description") else str(adm)
            if title and title.lower() == "example":
                lines.append("**Example:**")
                lines.append("")
                if "```" in desc:
                    lines.append(desc)
                else:
                    lines.append("```python")
                    lines.append(desc.strip())
                    lines.append("```")
                lines.append("")
            elif title and title.lower() == "note":
                lines.append(":::{note}")
                lines.append(desc)
                lines.append(":::")
                lines.append("")
            else:
                lines.append(f"**{title}:**")
                lines.append("")
                for line in desc.strip().splitlines():
                    lines.append(line)
                lines.append("")

        elif kind == griffe.DocstringSectionKind.examples:
            lines.append("**Examples:**")
            lines.append("")
            for kind_str, text in section.value:
                if kind_str == "text":
                    lines.append(text)
                else:
                    lines.append("```python")
                    lines.append(text.strip())
                    lines.append("```")
            lines.append("")

        elif kind == griffe.DocstringSectionKind.yields:
            lines.append("**Yields:**")
            lines.append("")
            for y in section.value:
                ann = str(y.annotation) if y.annotation else ""
                desc = y.description or ""
                if ann:
                    lines.append(f"`{ann}` — {desc}")
                else:
                    lines.append(desc)
            lines.append("")

        else:
            val = section.value
            if isinstance(val, str):
                for line in val.strip().splitlines():
                    lines.append(line)
                lines.append("")

    # Restore original value
    docstring.value = original_value
