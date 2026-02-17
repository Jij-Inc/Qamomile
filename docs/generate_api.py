"""Generate API reference markdown from qamomile package docstrings.

Uses griffe to introspect the qamomile package and generates
MyST-compatible markdown files in docs/api/.

Usage:
    python generate_api.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import griffe


PACKAGE_NAME = "qamomile"
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / PACKAGE_NAME
OUTPUT_DIR = Path(__file__).resolve().parent / "api"

# Packages with more immediate submodules than this threshold
# will be split into a subdirectory with one file per submodule.
SPLIT_THRESHOLD = 3


def discover_subpackages() -> list[str]:
    """Find all top-level subpackages under qamomile/."""
    subpackages = []
    for entry in sorted(PACKAGE_DIR.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            if not entry.name.startswith("_"):
                subpackages.append(entry.name)
    return subpackages


def format_annotation(annotation: str | griffe.Expression | None) -> str:
    """Format a type annotation for display."""
    if annotation is None:
        return ""
    return str(annotation)


def format_signature(func: griffe.Function) -> str:
    """Format a function signature."""
    parts = []
    for param in func.parameters:
        if param.name == "self" or param.name == "cls":
            parts.append(param.name)
            continue
        p = param.name
        ann = format_annotation(param.annotation)
        if ann:
            p += f": {ann}"
        if param.default is not None:
            default_str = str(param.default)
            p += f" = {default_str}"
        if param.kind == griffe.ParameterKind.var_positional:
            p = f"*{p}"
        elif param.kind == griffe.ParameterKind.var_keyword:
            p = f"**{p}"
        elif param.kind == griffe.ParameterKind.keyword_only:
            if parts and not any(pp.startswith("*") for pp in parts):
                parts.append("*")
        parts.append(p)

    params_str = ", ".join(parts)
    ret = format_annotation(func.returns)
    ret_str = f" -> {ret}" if ret else ""
    return f"({params_str}){ret_str}"


def format_class_signature(cls: griffe.Class) -> str:
    """Format a class definition line."""
    bases = [str(base) for base in cls.bases]
    if bases:
        return f"class {cls.name}({', '.join(bases)})"
    return f"class {cls.name}"


def write_docstring(lines: list[str], docstring: griffe.Docstring | None) -> None:
    """Write docstring content to lines."""
    if not docstring or not docstring.value:
        return
    for line in docstring.value.strip().splitlines():
        lines.append(line)
    lines.append("")


def resolve_member(member: griffe.Alias | griffe.Object) -> griffe.Object | None:
    """Resolve an alias to its target, handling errors gracefully."""
    if isinstance(member, griffe.Alias):
        try:
            return member.target  # type: ignore[return-value]
        except Exception:
            return None
    return member


def is_public(name: str) -> bool:
    """Check if a name is public (not starting with _)."""
    return not name.startswith("_")


def module_belongs_to_package(module: griffe.Module, package_dir: Path) -> bool:
    """Check if a module's source file is within the given package directory."""
    if module.filepath is None:
        return False
    try:
        module_path = Path(module.filepath).resolve()
        return str(module_path).startswith(str(package_dir.resolve()))
    except Exception:
        return False


def get_public_non_module_members(
    obj: griffe.Module | griffe.Class,
) -> dict[str, griffe.Object]:
    """Get all public members of a module or class, excluding submodules."""
    result: dict[str, griffe.Object] = {}
    for name, member in obj.members.items():
        if not is_public(name):
            continue
        resolved = resolve_member(member)
        if resolved is None:
            continue
        if isinstance(resolved, griffe.Module):
            continue
        result[name] = resolved
    return result


def get_immediate_submodules(
    module: griffe.Module, package_dir: Path
) -> list[tuple[str, griffe.Module]]:
    """Get immediate child submodules that belong to the package."""
    result = []
    for name, member in sorted(module.members.items()):
        if name.startswith("_"):
            continue
        resolved = resolve_member(member)
        if resolved is None:
            continue
        if isinstance(resolved, griffe.Module):
            if module_belongs_to_package(resolved, package_dir):
                result.append((name, resolved))
    return result


def collect_submodules_recursive(
    module: griffe.Module,
    prefix: str,
    package_dir: Path,
    _visited: set[int] | None = None,
) -> list[tuple[str, griffe.Module]]:
    """Recursively collect all submodules within the package directory."""
    if _visited is None:
        _visited = set()

    module_id = id(module)
    if module_id in _visited:
        return []
    _visited.add(module_id)

    result: list[tuple[str, griffe.Module]] = []
    for name, member in sorted(module.members.items()):
        if name.startswith("_"):
            continue
        resolved = resolve_member(member)
        if resolved is None:
            continue
        if isinstance(resolved, griffe.Module):
            if not module_belongs_to_package(resolved, package_dir):
                continue
            full_path = f"{prefix}.{name}"
            if id(resolved) not in _visited:
                result.append((full_path, resolved))
                result.extend(
                    collect_submodules_recursive(
                        resolved, full_path, package_dir, _visited
                    )
                )
    return result


def generate_function_doc(func: griffe.Function) -> list[str]:
    """Generate markdown for a function."""
    lines = []
    sig = format_signature(func)
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
    write_docstring(lines, func.docstring)
    return lines


def generate_class_doc(cls: griffe.Class, heading_level: int = 3) -> list[str]:
    """Generate markdown for a class."""
    lines = []
    h = "#" * heading_level
    lines.append(f"{h} `{cls.name}`")
    lines.append("")

    sig = format_class_signature(cls)
    lines.append("```python")
    lines.append(sig)
    lines.append("```")
    lines.append("")
    write_docstring(lines, cls.docstring)

    # Constructor
    if "__init__" in cls.members:
        init = resolve_member(cls.members["__init__"])
        if init and isinstance(init, griffe.Function):
            init_sig = format_signature(init)
            lines.append(f"{h}# Constructor")
            lines.append("")
            lines.append("```python")
            lines.append(f"def __init__{init_sig}")
            lines.append("```")
            lines.append("")
            write_docstring(lines, init.docstring)

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
            lines.append(f"{h}## `{method_name}`")
            lines.append("")
            lines.extend(generate_function_doc(method))

    return lines


def generate_module_content(
    module: griffe.Module,
    heading_level: int = 2,
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
        write_docstring(lines, module.docstring)

    if attributes:
        lines.append(f"{h} Constants")
        lines.append("")
        for attr_name, attr in sorted(attributes):
            ann = format_annotation(attr.annotation)
            ann_str = f": {ann}" if ann else ""
            lines.append(f"- **`{attr_name}`**{ann_str}")
            if attr.docstring and attr.docstring.value:
                first_line = attr.docstring.value.strip().splitlines()[0]
                lines.append(f"  {first_line}")
            lines.append("")

    if functions:
        lines.append(f"{h} Functions")
        lines.append("")
        for func_name, func in sorted(functions):
            lines.append(f"{h}# `{func_name}`")
            lines.append("")
            lines.extend(generate_function_doc(func))

    if classes:
        lines.append(f"{h} Classes")
        lines.append("")
        for _, cls in sorted(classes):
            lines.extend(generate_class_doc(cls, heading_level=heading_level + 1))

    return lines


def generate_flat_page(
    pkg: griffe.Module,
    full_name: str,
    package_dir: Path,
) -> str:
    """Generate a single markdown page for a small subpackage."""
    lines = [f"# {full_name}", ""]

    top_level = generate_module_content(pkg, heading_level=2)
    lines.extend(top_level)

    submodules = collect_submodules_recursive(pkg, full_name, package_dir)
    for sub_path, sub_module in submodules:
        lines.append("---")
        lines.append("")
        lines.append(f"## {sub_path}")
        lines.append("")
        sub_content = generate_module_content(sub_module, heading_level=3)
        lines.extend(sub_content)

    return "\n".join(lines) + "\n"


def generate_split_pages(
    pkg: griffe.Module,
    full_name: str,
    subpackage_name: str,
    package_dir: Path,
) -> dict[str, str]:
    """Generate multiple markdown pages for a large subpackage.

    Returns a dict of {relative_path: content}.
    """
    pages: dict[str, str] = {}

    # Index page: top-level exports + links to submodules
    immediate_subs = get_immediate_submodules(pkg, package_dir)
    index_lines = [f"# {full_name}", ""]
    top_level = generate_module_content(pkg, heading_level=2)
    index_lines.extend(top_level)

    if immediate_subs:
        index_lines.append("## Submodules")
        index_lines.append("")
        for sub_name, _ in immediate_subs:
            index_lines.append(f"- [{full_name}.{sub_name}]({sub_name}.md)")
        index_lines.append("")

    pages[f"{subpackage_name}/index.md"] = "\n".join(index_lines) + "\n"

    # One page per immediate submodule (with its recursive children)
    for sub_name, sub_module in immediate_subs:
        sub_full = f"{full_name}.{sub_name}"
        sub_dir = package_dir / sub_name
        sub_lines = [f"# {sub_full}", ""]

        sub_content = generate_module_content(sub_module, heading_level=2)
        sub_lines.extend(sub_content)

        # Recursively collect deeper submodules
        deeper = collect_submodules_recursive(sub_module, sub_full, sub_dir)
        for deep_path, deep_module in deeper:
            sub_lines.append("---")
            sub_lines.append("")
            sub_lines.append(f"## {deep_path}")
            sub_lines.append("")
            deep_content = generate_module_content(deep_module, heading_level=3)
            sub_lines.extend(deep_content)

        pages[f"{subpackage_name}/{sub_name}.md"] = "\n".join(sub_lines) + "\n"

    return pages


def generate_index_page(
    subpackages: list[str],
    split_packages: set[str],
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
        full_name = f"{PACKAGE_NAME}.{name}"
        if name in split_packages:
            lines.append(f"- [{full_name}]({name}/index.md)")
        else:
            lines.append(f"- [{full_name}]({name}.md)")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── TOC generation helpers ────────────────────────────────────────────


def build_toc_entries(
    subpackages: list[str],
    split_packages: set[str],
    split_data: dict[str, list[str]],
) -> list[str]:
    """Build myst.yml TOC YAML snippet lines."""
    lines = ["    - title: API Reference", "      children:"]
    for name in subpackages:
        if name in split_packages:
            lines.append(f"        - file: api/{name}/index.md")
            for sub_file in split_data.get(name, []):
                lines.append(f"        - file: api/{name}/{sub_file}.md")
        else:
            lines.append(f"        - file: api/{name}.md")
    return lines


def main() -> None:
    subpackages = discover_subpackages()
    print(f"Discovered subpackages: {subpackages}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    split_packages: set[str] = set()
    split_data: dict[str, list[str]] = {}
    all_files: list[str] = ["api/index.md"]

    for name in subpackages:
        full_name = f"{PACKAGE_NAME}.{name}"
        package_dir = PACKAGE_DIR / name
        try:
            pkg = griffe.load(full_name)
        except Exception as e:
            print(f"  ERROR loading {full_name}: {e}")
            continue

        # Decide whether to split
        immediate_subs = get_immediate_submodules(pkg, package_dir)

        if len(immediate_subs) > SPLIT_THRESHOLD:
            # Split into subdirectory
            split_packages.add(name)
            sub_dir = OUTPUT_DIR / name
            sub_dir.mkdir(parents=True)

            pages = generate_split_pages(pkg, full_name, name, package_dir)
            sub_files = []
            for rel_path, content in sorted(pages.items()):
                out_file = OUTPUT_DIR / rel_path
                out_file.write_text(content)
                line_count = content.count("\n")
                print(f"  Generated api/{rel_path} ({line_count} lines)")
                all_files.append(f"api/{rel_path}")
                # Track sub-file names (excluding index)
                stem = Path(rel_path).stem
                if stem != "index":
                    sub_files.append(stem)
            split_data[name] = sub_files
        else:
            # Single flat page
            content = generate_flat_page(pkg, full_name, package_dir)
            out_file = OUTPUT_DIR / f"{name}.md"
            out_file.write_text(content)
            line_count = content.count("\n")
            print(f"  Generated api/{name}.md ({line_count} lines)")
            all_files.append(f"api/{name}.md")

    # Generate index page
    index_content = generate_index_page(subpackages, split_packages)
    (OUTPUT_DIR / "index.md").write_text(index_content)
    print("  Generated api/index.md")

    # Print TOC entries
    toc_lines = build_toc_entries(subpackages, split_packages, split_data)
    print(f"\nAPI docs generated in {OUTPUT_DIR}")
    print(f"\nmyst.yml TOC entries needed:")
    for line in toc_lines:
        print(line)


if __name__ == "__main__":
    main()
