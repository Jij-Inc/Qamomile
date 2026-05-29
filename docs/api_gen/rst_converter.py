"""Convert RST-specific markup in docstrings to MyST-compatible Markdown."""

from __future__ import annotations

import re


def rst_to_myst(text: str) -> str:
    """Convert RST-specific markup to MyST-compatible Markdown.

    Handles inline roles and block directives found in qamomile docstrings.
    """
    text = _convert_inline_roles(text)
    text = _convert_block_directives(text)
    return text


def _convert_inline_roles(text: str) -> str:
    """Convert RST inline roles to MyST equivalents."""
    # :math:`expr` -> $expr$
    text = re.sub(r":math:`([^`]+)`", r"$\1$", text)
    # :cite:`key` -> [key]
    text = re.sub(r":cite:`([^`]+)`", r"[\1]", text)
    return text


def _convert_block_directives(text: str) -> str:
    """Convert RST block directives to MyST equivalents."""
    lines = text.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # .. bibliography:: block — remove entirely
        if stripped.startswith(".. bibliography::"):
            i += 1
            # Skip options lines (indented or blank)
            while i < len(lines):
                next_stripped = lines[i].strip()
                if next_stripped == "":
                    i += 1
                    continue
                if next_stripped.startswith(":") or lines[i][0] in (" ", "\t"):
                    i += 1
                    continue
                break
            continue

        # .. math:: block -> $$ block
        if stripped.startswith(".. math::"):
            indent = _get_indent(line)
            i += 1
            # Skip blank line after directive
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            # Collect indented content
            math_lines = _collect_indented_block(lines, i, indent)
            i += len(math_lines)
            content = "\n".join(l.strip() for l in math_lines)
            if content:
                result.append(f"{' ' * indent}$$")
                result.append(content)
                result.append(f"{' ' * indent}$$")
                result.append("")
            continue

        # .. code:: or .. code-block:: -> ```python block
        if stripped.startswith(".. code::") or stripped.startswith(".. code-block::"):
            indent = _get_indent(line)
            i += 1
            # Skip blank line after directive
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            # Collect indented content
            code_lines = _collect_indented_block(lines, i, indent)
            i += len(code_lines)
            # Dedent code content
            if code_lines:
                min_indent = min(
                    (len(l) - len(l.lstrip()) for l in code_lines if l.strip()),
                    default=0,
                )
                dedented = [l[min_indent:] for l in code_lines]
                prefix = " " * indent
                result.append(f"{prefix}```python")
                result.extend(f"{prefix}{l}" for l in dedented)
                result.append(f"{prefix}```")
                result.append("")
            continue

        # .. note:: -> :::{note}
        if stripped.startswith(".. note::"):
            indent = _get_indent(line)
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            note_lines = _collect_indented_block(lines, i, indent)
            i += len(note_lines)
            content = "\n".join(l.strip() for l in note_lines)
            prefix = " " * indent
            result.append(f"{prefix}:::{{note}}")
            result.append(content)
            result.append(f"{prefix}:::")
            result.append("")
            continue

        # .. warning:: -> :::{warning}
        if stripped.startswith(".. warning::"):
            indent = _get_indent(line)
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            warn_lines = _collect_indented_block(lines, i, indent)
            i += len(warn_lines)
            content = "\n".join(l.strip() for l in warn_lines)
            prefix = " " * indent
            result.append(f"{prefix}:::{{warning}}")
            result.append(content)
            result.append(f"{prefix}:::")
            result.append("")
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


def _get_indent(line: str) -> int:
    """Get the indentation level of a line."""
    return len(line) - len(line.lstrip())


def _collect_indented_block(
    lines: list[str], start: int, directive_indent: int
) -> list[str]:
    """Collect lines that are indented more than the directive."""
    block: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            # Blank lines are part of the block if followed by indented content
            lookahead = i + 1
            while lookahead < len(lines) and lines[lookahead].strip() == "":
                lookahead += 1
            if (
                lookahead < len(lines)
                and _get_indent(lines[lookahead]) > directive_indent
            ):
                block.append(line)
                i += 1
                continue
            break
        if _get_indent(line) > directive_indent:
            block.append(line)
            i += 1
        else:
            break
    return block
