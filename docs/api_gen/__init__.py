"""API documentation generator for Qamomile."""

from __future__ import annotations

import shutil
from pathlib import Path

import griffe

from .config import ApiGenConfig
from .crossref import CrossRefRegistry
from .discovery import (
    discover_subpackages,
    get_immediate_submodules,
    get_public_non_module_members,
    resolve_member,
    is_public,
    collect_submodules_recursive,
)
from .pages import generate_flat_page, generate_index_page, generate_split_pages
from .toc import build_toc_entries, inject_toc


def _build_registry(
    subpackages: list[str],
    config: ApiGenConfig,
    split_packages: set[str],
) -> CrossRefRegistry:
    """Pre-build cross-reference registry from all loaded packages."""
    registry = CrossRefRegistry()

    for name in subpackages:
        full_name = f"{config.package_name}.{name}"
        package_dir = config.package_dir / name
        try:
            pkg = griffe.load(full_name)
        except Exception:
            continue

        if name in split_packages:
            _register_module_symbols(
                registry, pkg, full_name, f"api/{name}/index.md"
            )
            immediate_subs = get_immediate_submodules(pkg, package_dir)
            for sub_name, sub_module in immediate_subs:
                sub_full = f"{full_name}.{sub_name}"
                page = f"api/{name}/{sub_name}.md"
                _register_module_symbols(registry, sub_module, sub_full, page)
                sub_dir = package_dir / sub_name
                deeper = collect_submodules_recursive(
                    sub_module, sub_full, sub_dir
                )
                for deep_path, deep_module in deeper:
                    _register_module_symbols(
                        registry, deep_module, deep_path, page
                    )
        else:
            page = f"api/{name}.md"
            _register_module_symbols(registry, pkg, full_name, page)
            deeper = collect_submodules_recursive(
                pkg, full_name, package_dir
            )
            for deep_path, deep_module in deeper:
                _register_module_symbols(
                    registry, deep_module, deep_path, page
                )

    return registry


def _register_module_symbols(
    registry: CrossRefRegistry,
    module: griffe.Module,
    module_path: str,
    page_path: str,
) -> None:
    """Register all public symbols from a module into the registry."""
    public = get_public_non_module_members(module)
    for sym_name, member in public.items():
        canonical = f"{module_path}.{sym_name}"
        registry.register(canonical, page_path, sym_name)
        if isinstance(member, griffe.Class):
            for method_name, method in member.members.items():
                if is_public(method_name):
                    resolved = resolve_member(method)
                    if resolved and isinstance(resolved, griffe.Function):
                        registry.register(
                            f"{canonical}.{method_name}",
                            page_path,
                            f"{sym_name}.{method_name}",
                        )


def main(config: ApiGenConfig | None = None) -> None:
    """Generate API documentation."""
    if config is None:
        config = ApiGenConfig()

    subpackages = discover_subpackages(config)
    print(f"Discovered subpackages: {subpackages}")

    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True)

    # First pass: determine which packages get split
    split_packages: set[str] = set()
    for name in subpackages:
        full_name = f"{config.package_name}.{name}"
        package_dir = config.package_dir / name
        try:
            pkg = griffe.load(full_name)
        except Exception:
            continue
        immediate_subs = get_immediate_submodules(pkg, package_dir)
        if len(immediate_subs) > config.split_threshold:
            split_packages.add(name)

    # Build cross-reference registry
    registry = _build_registry(subpackages, config, split_packages)

    # Second pass: generate pages
    split_data: dict[str, list[str]] = {}

    for name in subpackages:
        full_name = f"{config.package_name}.{name}"
        package_dir = config.package_dir / name
        try:
            pkg = griffe.load(full_name)
        except Exception as e:
            print(f"  ERROR loading {full_name}: {e}")
            continue

        if name in split_packages:
            sub_dir = config.output_dir / name
            sub_dir.mkdir(parents=True)

            pages = generate_split_pages(
                pkg, full_name, name, package_dir, config, registry
            )
            sub_files = []
            for rel_path, content in sorted(pages.items()):
                out_file = config.output_dir / rel_path
                out_file.write_text(content)
                line_count = content.count("\n")
                print(f"  Generated api/{rel_path} ({line_count} lines)")
                stem = Path(rel_path).stem
                if stem != "index":
                    sub_files.append(stem)
            split_data[name] = sub_files
        else:
            page_path = f"api/{name}.md"
            content = generate_flat_page(
                pkg, full_name, package_dir, config, page_path, registry
            )
            out_file = config.output_dir / f"{name}.md"
            out_file.write_text(content)
            line_count = content.count("\n")
            print(f"  Generated api/{name}.md ({line_count} lines)")

    # Generate index page
    index_content = generate_index_page(
        subpackages, split_packages, config.package_name
    )
    (config.output_dir / "index.md").write_text(index_content)
    print("  Generated api/index.md")

    # Inject TOC into myst.yml files
    toc_lines = build_toc_entries(subpackages, split_packages, split_data)
    for lang in ("en", "ja"):
        myst_yml = config.docs_dir / lang / "myst.yml"
        if myst_yml.exists():
            inject_toc(myst_yml, toc_lines)

    print(f"\nAPI docs generated in {config.output_dir}")
