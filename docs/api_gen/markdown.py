"""Markdown generation for functions, classes, and modules."""

from __future__ import annotations

from pathlib import Path

import griffe

from .annotations import (
    format_annotation,
    format_class_signature,
    format_signature,
)
from .config import ApiGenConfig
from .crossref import CrossRefRegistry
from .discovery import get_public_non_module_members, is_public, resolve_member
from .docstrings import get_first_line, write_docstring, _escape_table_cell


def _source_link(obj: griffe.Object, config: ApiGenConfig) -> str:
    """Generate a GitHub source link for an object."""
    if obj.filepath is None or obj.lineno is None:
        return ""
    try:
        rel = Path(obj.filepath).resolve().relative_to(config.repo_root)
    except ValueError:
        return ""
    return f" [[source]({config.github_base_url}/{rel}#L{obj.lineno})]"


def _make_anchor_target(name: str) -> str:
    """Generate a MyST anchor target for stable heading links."""
    return f"({name})=\n"


def generate_function_doc(
    func: griffe.Function,
    config: ApiGenConfig,
    page_path: str = "",
    registry: CrossRefRegistry | None = None,
) -> list[str]:
    """Generate markdown for a function."""
    lines = []
    sig = format_signature(func, config)
    decorators = ""
    if hasattr(func, "labels"):
        if "staticmethod" in func.labels:
            decorators = "@staticmethod\n"
        elif "classmethod" in func.labels:
            decorators = "@classmethod\n"

    lines.append("```python")
    if decorators:
        lines.append(decorators.strip())
    lines.append(f"def {func.name}{sig}")
    lines.append("```")
    lines.append("")
    write_docstring(lines, func.docstring, config)
    return lines


def generate_class_doc(
    cls: griffe.Class,
    heading_level: int,
    config: ApiGenConfig,
    page_path: str = "",
    registry: CrossRefRegistry | None = None,
) -> list[str]:
    """Generate markdown for a class."""
    lines = []
    h = "#" * heading_level
    src = _source_link(cls, config)
    lines.append(_make_anchor_target(cls.name))
    lines.append(f"{h} `{cls.name}`{src}")
    lines.append("")

    sig = format_class_signature(cls)
    lines.append("```python")
    lines.append(sig)
    lines.append("```")
    lines.append("")
    write_docstring(lines, cls.docstring, config)

    # Constructor
    if "__init__" in cls.members:
        init = resolve_member(cls.members["__init__"])
        if init and isinstance(init, griffe.Function):
            init_sig = format_signature(init, config)
            lines.append(f"{h}# Constructor")
            lines.append("")
            lines.append("```python")
            lines.append(f"def __init__{init_sig}")
            lines.append("```")
            lines.append("")
            write_docstring(lines, init.docstring, config)

    methods: list[tuple[str, griffe.Function]] = []
    attributes: list[tuple[str, griffe.Attribute]] = []

    for name, member in cls.members.items():
        if not is_public(name):
            continue
        resolved = resolve_member(member)
        if resolved is None:
            continue
        if isinstance(resolved, griffe.Function):
            methods.append((name, resolved))
        elif isinstance(resolved, griffe.Attribute):
            attributes.append((name, resolved))

    if attributes:
        lines.append(f"{h}# Attributes")
        lines.append("")
        for attr_name, attr in sorted(attributes):
            ann = format_annotation(attr.annotation)
            ann_str = f": {ann}" if ann else ""
            lines.append(f"- **`{attr_name}`**{ann_str}")
            if attr.docstring and attr.docstring.value:
                first_line = attr.docstring.value.strip().splitlines()[0]
                lines.append(f"  {first_line}")
            lines.append("")

    if methods:
        lines.append(f"{h}# Methods")
        lines.append("")
        for method_name, method in sorted(methods):
            lines.append(_make_anchor_target(f"{cls.name}.{method_name}"))
            lines.append(f"{h}## `{method_name}`")
            lines.append("")
            lines.extend(
                generate_function_doc(method, config, page_path, registry)
            )

    return lines


def generate_module_content(
    module: griffe.Module,
    heading_level: int,
    config: ApiGenConfig,
    page_path: str = "",
    registry: CrossRefRegistry | None = None,
) -> list[str]:
    """Generate markdown content for a module's non-module public members."""
    lines = []

    public = get_public_non_module_members(module)

    functions: list[tuple[str, griffe.Function]] = []
    classes: list[tuple[str, griffe.Class]] = []
    attributes: list[tuple[str, griffe.Attribute]] = []

    for name, member in public.items():
        if isinstance(member, griffe.Function):
            functions.append((name, member))
        elif isinstance(member, griffe.Class):
            classes.append((name, member))
        elif isinstance(member, griffe.Attribute):
            attributes.append((name, member))

    h = "#" * heading_level

    if module.docstring and module.docstring.value:
        write_docstring(lines, module.docstring, config)

    # Summary table
    has_summary = bool(functions) or bool(classes)
    if has_summary:
        lines.append(f"{h} Overview")
        lines.append("")
        if functions:
            lines.append("| Function | Description |")
            lines.append("|----------|-------------|")
            for func_name, func in sorted(functions):
                desc = _escape_table_cell(get_first_line(func.docstring))
                lines.append(f"| [`{func_name}`](#{func_name}) | {desc} |")
            lines.append("")
        if classes:
            lines.append("| Class | Description |")
            lines.append("|-------|-------------|")
            for cls_name, cls in sorted(classes):
                desc = _escape_table_cell(get_first_line(cls.docstring))
                lines.append(f"| [`{cls_name}`](#{cls_name}) | {desc} |")
            lines.append("")

    if attributes:
        # Filter out module aliases (attributes that are just module references)
        real_attrs = [
            (n, a)
            for n, a in sorted(attributes)
            if a.annotation is not None
            or (a.docstring and a.docstring.value)
        ]
        if real_attrs:
            lines.append(f"{h} Constants")
            lines.append("")
            for attr_name, attr in real_attrs:
                ann = format_annotation(attr.annotation)
                ann_str = f": `{ann}`" if ann else ""
                val_str = ""
                if attr.value is not None:
                    val = str(attr.value)
                    if len(val) < 60:
                        val_str = f" = `{val}`"
                lines.append(f"- **`{attr_name}`**{ann_str}{val_str}")
                if attr.docstring and attr.docstring.value:
                    first_line = attr.docstring.value.strip().splitlines()[0]
                    lines.append(f"  {first_line}")
                lines.append("")

    if functions:
        lines.append(f"{h} Functions")
        lines.append("")
        for i, (func_name, func) in enumerate(sorted(functions)):
            src = _source_link(func, config)
            lines.append(_make_anchor_target(func_name))
            lines.append(f"{h}# `{func_name}`{src}")
            lines.append("")
            lines.extend(
                generate_function_doc(func, config, page_path, registry)
            )
            if i < len(functions) - 1:
                lines.append("---")
                lines.append("")

    if classes:
        lines.append(f"{h} Classes")
        lines.append("")
        for i, (_, cls) in enumerate(sorted(classes)):
            lines.extend(
                generate_class_doc(
                    cls,
                    heading_level=heading_level + 1,
                    config=config,
                    page_path=page_path,
                    registry=registry,
                )
            )
            if i < len(classes) - 1:
                lines.append("---")
                lines.append("")

    return lines
