"""Type annotation formatting and function/class signature generation."""

from __future__ import annotations

import re

import griffe

from .config import ApiGenConfig


def format_annotation(annotation: str | griffe.Expression | None) -> str:
    """Format a type annotation for display, cleaning up module prefixes."""
    if annotation is None:
        return ""
    raw = str(annotation)
    return _clean_annotation_str(raw)


def _clean_annotation_str(raw: str) -> str:
    """Clean up a raw annotation string for display."""
    cleaned = re.sub(r"\btyp\.", "", raw)
    cleaned = re.sub(r"\btyping\.", "", cleaned)
    cleaned = re.sub(r"\bcollections\.abc\.", "", cleaned)
    return cleaned


def _format_param(param: griffe.Parameter) -> str:
    """Format a single parameter for signature display."""
    if param.name in ("self", "cls"):
        return param.name
    p = param.name
    ann = format_annotation(param.annotation)
    if ann:
        p += f": {ann}"
    if param.default is not None:
        p += f" = {param.default}"
    if param.kind == griffe.ParameterKind.var_positional:
        p = f"*{p}"
    elif param.kind == griffe.ParameterKind.var_keyword:
        p = f"**{p}"
    return p


def format_signature(
    func: griffe.Function,
    config: ApiGenConfig | None = None,
) -> str:
    """Format a function signature, using multiple lines for long signatures."""
    width = config.multiline_sig_width if config else 80

    parts: list[str] = []
    for param in func.parameters:
        if param.kind == griffe.ParameterKind.keyword_only:
            if parts and not any(pp.startswith("*") for pp in parts):
                parts.append("*")
        parts.append(_format_param(param))

    ret = format_annotation(func.returns)
    ret_str = f" -> {ret}" if ret else ""

    one_line = f"({', '.join(parts)}){ret_str}"

    if len(one_line) > width:
        indent = "    "
        param_lines = [f"{indent}{p}," for p in parts]
        return "(\n" + "\n".join(param_lines) + f"\n){ret_str}"

    return one_line


def format_class_signature(cls: griffe.Class) -> str:
    """Format a class definition line."""
    bases = [_clean_annotation_str(str(base)) for base in cls.bases]
    if bases:
        return f"class {cls.name}({', '.join(bases)})"
    return f"class {cls.name}"
