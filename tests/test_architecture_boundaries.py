"""Source-level regression tests for Qamomile package boundaries."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import qamomile.circuit.transpiler as transpiler_api

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
QAMOMILE_ROOT = REPOSITORY_ROOT / "qamomile"
BACKEND_PACKAGES = (
    "cudaq",
    "hugr",
    "qbraid",
    "qiskit",
    "quration",
    "quri_parts",
)
FRONTEND_MODULE = "qamomile.circuit.frontend"
TRANSPILER_MODULE = "qamomile.circuit.transpiler"


@dataclass(frozen=True)
class _ImportReference:
    """Describe one source-level import dependency.

    Args:
        path (Path): Source file containing the import.
        lineno (int): One-based source line number.
        module (str): Absolute imported module name.
        imported_names (tuple[str, ...] | None): Names selected by a
            ``from`` import, or ``None`` for a plain ``import`` statement.
    """

    path: Path
    lineno: int
    module: str
    imported_names: tuple[str, ...] | None


def _module_name_for_path(path: Path) -> str:
    """Return the dotted module name represented by a source path.

    Args:
        path (Path): Python source path below the repository root.

    Returns:
        str: Absolute dotted module name, with ``.__init__`` removed.
    """
    parts = list(path.relative_to(REPOSITORY_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` node to an absolute module name.

    Args:
        path (Path): Source file containing the import.
        node (ast.ImportFrom): Import node to resolve.

    Returns:
        str: Absolute module named by the import.

    Raises:
        ImportError: If a relative import escapes its top-level package.
    """
    if node.level == 0:
        return node.module or ""

    source_module = _module_name_for_path(path)
    package = (
        source_module if path.stem == "__init__" else source_module.rpartition(".")[0]
    )
    relative_name = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative_name, package)


def _iter_source_files(root: Path) -> Iterator[Path]:
    """Yield Python implementation and stub files below a package root.

    Args:
        root (Path): Package directory to scan recursively.

    Yields:
        Path: A ``.py`` or ``.pyi`` source path in stable order.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".pyi"}:
            yield path


def _iter_import_references(root: Path) -> Iterator[_ImportReference]:
    """Yield every static import found below a package root.

    Function-local imports and imports guarded by ``TYPE_CHECKING`` still
    express an architectural dependency, so the complete AST is traversed.

    Args:
        root (Path): Package directory to scan recursively.

    Yields:
        _ImportReference: An absolute source-level import reference.
    """
    for path in _iter_source_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield _ImportReference(path, node.lineno, alias.name, None)
            elif isinstance(node, ast.ImportFrom):
                yield _ImportReference(
                    path,
                    node.lineno,
                    _resolve_import_from(path, node),
                    tuple(alias.name for alias in node.names),
                )


def _is_module_or_child(module: str, prefix: str) -> bool:
    """Return whether a module is a package prefix or one of its children.

    Args:
        module (str): Absolute module or imported-name candidate.
        prefix (str): Package prefix whose boundary must be respected.

    Returns:
        bool: Whether ``module`` equals ``prefix`` or starts below it.
    """
    return module == prefix or module.startswith(f"{prefix}.")


def _reference_targets(reference: _ImportReference) -> Iterator[str]:
    """Yield modules that one import statement can reference.

    Checking selected names catches forms such as
    ``from qamomile.circuit import frontend`` in addition to imports whose
    base module already names ``qamomile.circuit.frontend``.

    Args:
        reference (_ImportReference): Parsed import dependency.

    Yields:
        str: Base module followed by each explicitly selected child name.
    """
    yield reference.module
    if reference.imported_names is None:
        return
    for name in reference.imported_names:
        if name != "*":
            yield f"{reference.module}.{name}"


def _display_import(reference: _ImportReference) -> str:
    """Render a compact import spelling for a failure message.

    Args:
        reference (_ImportReference): Parsed import dependency.

    Returns:
        str: Human-readable import statement without aliases.
    """
    if reference.imported_names is None:
        return f"import {reference.module}"
    names = ", ".join(reference.imported_names)
    return f"from {reference.module} import {names}"


def _format_violation(reference: _ImportReference, reason: str) -> str:
    """Format one architecture violation with its source location.

    Args:
        reference (_ImportReference): Parsed import dependency.
        reason (str): Concise explanation of the violated rule.

    Returns:
        str: Repository-relative ``path:line`` diagnostic.
    """
    relative_path = reference.path.relative_to(REPOSITORY_ROOT).as_posix()
    return f"{relative_path}:{reference.lineno}: {reason}: {_display_import(reference)}"


def test_backends_do_not_import_circuit_frontend() -> None:
    """Backend packages depend on public circuit APIs, never frontend internals."""
    violations: list[str] = []
    for backend in BACKEND_PACKAGES:
        for reference in _iter_import_references(QAMOMILE_ROOT / backend):
            if any(
                _is_module_or_child(target, FRONTEND_MODULE)
                for target in _reference_targets(reference)
            ):
                violations.append(
                    _format_violation(
                        reference,
                        "backend imports qamomile.circuit.frontend",
                    )
                )

    assert not violations, (
        "Backend packages must not depend on circuit.frontend internals. "
        "Expose the required contract through a public circuit API instead:\n"
        + "\n".join(sorted(violations))
    )


def test_hugr_uses_only_public_transpiler_facade_imports() -> None:
    """HUGR imports named transpiler symbols only from its public facade."""
    public_names = frozenset(transpiler_api.__all__)
    violations: list[str] = []

    for reference in _iter_import_references(QAMOMILE_ROOT / "hugr"):
        if not any(
            _is_module_or_child(target, TRANSPILER_MODULE)
            for target in _reference_targets(reference)
        ):
            continue

        if reference.module != TRANSPILER_MODULE:
            violations.append(
                _format_violation(
                    reference,
                    "HUGR bypasses the public transpiler facade",
                )
            )
            continue
        if reference.imported_names is None:
            violations.append(
                _format_violation(
                    reference,
                    "HUGR must use a named from-import from the transpiler facade",
                )
            )
            continue
        if "*" in reference.imported_names:
            violations.append(
                _format_violation(
                    reference,
                    "HUGR must not wildcard-import the transpiler facade",
                )
            )
            continue

        non_public = sorted(set(reference.imported_names) - public_names)
        if non_public:
            violations.append(
                _format_violation(
                    reference,
                    "HUGR imports names missing from transpiler.__all__ "
                    f"({', '.join(non_public)})",
                )
            )

    assert not violations, (
        "HUGR must consume transpiler contracts through named imports from "
        "qamomile.circuit.transpiler, and every imported name must be listed "
        "in that facade's __all__:\n" + "\n".join(sorted(violations))
    )
