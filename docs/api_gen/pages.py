"""Page-level generation for API documentation (flat, split, index)."""

from __future__ import annotations

from pathlib import Path

import griffe

from .config import ApiGenConfig
from .crossref import CrossRefRegistry
from .discovery import (
    collect_submodules_recursive,
    get_immediate_submodules,
)
from .markdown import generate_module_content


def generate_flat_page(
    pkg: griffe.Module,
    full_name: str,
    package_dir: Path,
    config: ApiGenConfig,
    page_path: str = "",
    registry: CrossRefRegistry | None = None,
) -> str:
    """Generate a single markdown page for a small subpackage."""
    lines = [f"# {full_name}", ""]

    top_level = generate_module_content(
        pkg, heading_level=2, config=config, page_path=page_path, registry=registry
    )
    lines.extend(top_level)

    submodules = collect_submodules_recursive(pkg, full_name, package_dir)
    for sub_path, sub_module in submodules:
        lines.append("---")
        lines.append("")
        lines.append(f"## {sub_path}")
        lines.append("")
        sub_content = generate_module_content(
            sub_module,
            heading_level=3,
            config=config,
            page_path=page_path,
            registry=registry,
        )
        lines.extend(sub_content)

    return "\n".join(lines) + "\n"


def generate_split_pages(
    pkg: griffe.Module,
    full_name: str,
    subpackage_name: str,
    package_dir: Path,
    config: ApiGenConfig,
    registry: CrossRefRegistry | None = None,
) -> dict[str, str]:
    """Generate multiple markdown pages for a large subpackage.

    Returns a dict of {relative_path: content}.
    """
    pages: dict[str, str] = {}

    immediate_subs = get_immediate_submodules(pkg, package_dir)
    index_page_path = f"api/{subpackage_name}/index.md"
    index_lines = [f"# {full_name}", ""]
    top_level = generate_module_content(
        pkg,
        heading_level=2,
        config=config,
        page_path=index_page_path,
        registry=registry,
    )
    index_lines.extend(top_level)

    if immediate_subs:
        index_lines.append("## Submodules")
        index_lines.append("")
        for sub_name, _ in immediate_subs:
            index_lines.append(f"- [{full_name}.{sub_name}]({sub_name}.md)")
        index_lines.append("")

    pages[f"{subpackage_name}/index.md"] = "\n".join(index_lines) + "\n"

    for sub_name, sub_module in immediate_subs:
        sub_full = f"{full_name}.{sub_name}"
        sub_dir = package_dir / sub_name
        sub_page_path = f"api/{subpackage_name}/{sub_name}.md"
        sub_lines = [f"# {sub_full}", ""]

        sub_content = generate_module_content(
            sub_module,
            heading_level=2,
            config=config,
            page_path=sub_page_path,
            registry=registry,
        )
        sub_lines.extend(sub_content)

        deeper = collect_submodules_recursive(sub_module, sub_full, sub_dir)
        for deep_path, deep_module in deeper:
            sub_lines.append("---")
            sub_lines.append("")
            sub_lines.append(f"## {deep_path}")
            sub_lines.append("")
            deep_content = generate_module_content(
                deep_module,
                heading_level=3,
                config=config,
                page_path=sub_page_path,
                registry=registry,
            )
            sub_lines.extend(deep_content)

        pages[f"{subpackage_name}/{sub_name}.md"] = "\n".join(sub_lines) + "\n"

    return pages


def generate_index_page(
    subpackages: list[str],
    split_packages: set[str],
    package_name: str,
) -> str:
    """Generate the API reference index page."""
    lines = [
        "# API Reference",
        "",
        "Complete API reference for the Qamomile package.",
        "",
        "## Packages",
        "",
    ]
    for name in subpackages:
        full_name = f"{package_name}.{name}"
        if name in split_packages:
            lines.append(f"- [{full_name}]({name}/index.md)")
        else:
            lines.append(f"- [{full_name}]({name}.md)")
    lines.append("")
    return "\n".join(lines) + "\n"
