"""Package and module discovery via griffe."""

from __future__ import annotations

from pathlib import Path

import griffe

from .config import ApiGenConfig


def discover_subpackages(config: ApiGenConfig) -> list[str]:
    """Find all top-level subpackages under the package directory."""
    subpackages = []
    for entry in sorted(config.package_dir.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            if not entry.name.startswith("_"):
                subpackages.append(entry.name)
    return subpackages


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
    filepath = module.filepath
    # griffe's stub declares ``filepath`` as ``Path | list[Path]`` but at
    # runtime modules loaded without an on-disk source produce ``None``,
    # hence the defensive check (``type: ignore[unreachable]`` quiets the
    # stub-driven analysis). Namespace packages produce ``list[Path]``;
    # pick the first source for the source-tree containment check.
    if filepath is None:  # type: ignore[unreachable]
        return False  # type: ignore[unreachable]
    if isinstance(filepath, list):
        if not filepath:
            return False
        filepath = filepath[0]
    try:
        module_path = Path(filepath).resolve()
        package_root = package_dir.resolve()
        return module_path.is_relative_to(package_root)
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
