#!/usr/bin/env python3
"""Execute canonical documentation pages against installed release wheels."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

_CHILD_RUNNER = """
import runpy
import sys

import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None
runpy.run_path(sys.argv[1], run_name="__main__")
"""


class _StripDocstrings(ast.NodeTransformer):
    """Remove translatable docstrings while retaining executable structure."""

    def _visit_body(self, node: ast.AST) -> ast.AST:
        """Visit one body-bearing node and remove its leading docstring.

        Args:
            node: Module, function, or class node with a statement body.

        Returns:
            The visited node without its leading string expression.
        """

        self.generic_visit(node)
        body = getattr(node, "body")
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def fail(message: str) -> None:
    """Stop installed-documentation verification.

    Args:
        message: Explanation of the violated release contract.

    Raises:
        SystemExit: Always, with a nonzero status.
    """

    raise SystemExit(f"installed documentation verification failed: {message}")


def _within(path: Path, directory: Path) -> bool:
    """Return whether a resolved path is inside a resolved directory.

    Args:
        path: Candidate file path.
        directory: Directory that must not contain the candidate.

    Returns:
        Whether ``path`` is within ``directory``.
    """

    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def validate_module_origin(
    module_name: str,
    origin: str | None,
    repository_root: Path,
) -> None:
    """Reject a missing package or an import resolved from the checkout.

    Args:
        module_name: Public package being checked.
        origin: Import-spec origin for that package.
        repository_root: Checkout that must not satisfy release imports.

    Raises:
        SystemExit: If the package is missing, has no file origin, or resolves
            inside the repository.
    """

    if origin is None or origin in {"built-in", "frozen"}:
        fail(f"{module_name!r} has no installed package origin")
    if _within(Path(origin), repository_root):
        fail(f"{module_name!r} resolved from the repository: {origin}")


def installed_public_packages(distribution: metadata.Distribution) -> tuple[str, ...]:
    """Derive top-level Qamomile packages from an installed wheel record.

    Args:
        distribution: Installed root Qamomile distribution.

    Returns:
        Sorted top-level package names beginning with ``qamomile``.

    Raises:
        SystemExit: If the installed distribution has no file record or no
            Qamomile package.
    """

    files = distribution.files
    if files is None:
        fail("installed Qamomile distribution has no file record")
    packages = tuple(
        sorted(
            {
                file.parts[0]
                for file in files
                if len(file.parts) == 2
                and file.parts[0].startswith("qamomile")
                and file.parts[1] == "__init__.py"
            }
        )
    )
    if not packages:
        fail("installed distribution contains no Qamomile package")
    return packages


def verify_import_boundary(repository_root: Path, expected_version: str) -> None:
    """Verify that every Qamomile package comes from installed artifacts.

    Args:
        repository_root: Checkout that must not appear in package origins.
        expected_version: Version required for the installed root distribution.

    Raises:
        SystemExit: If package identity, version, or origin is invalid.
    """

    try:
        distribution = metadata.distribution("qamomile")
    except metadata.PackageNotFoundError as error:
        fail(f"required release distribution is not installed: {error.name}")
    if distribution.version != expected_version:
        fail(
            "installed Qamomile version differs from the release artifact: "
            f"expected {expected_version}, found {distribution.version}"
        )
    native_name = "qamomile._qamomile_native"
    native_spec = importlib.util.find_spec(native_name)
    if native_spec is None:
        fail("installed distribution is missing its native compiler")
    validate_module_origin(native_name, native_spec.origin, repository_root)
    package_root = Path(distribution.locate_file("qamomile")).resolve()
    if Path(native_spec.origin).resolve().parent != package_root:
        fail("native compiler is outside the installed Qamomile package")
    try:
        metadata.distribution("qamomile-native")
    except metadata.PackageNotFoundError:
        pass
    else:
        fail("obsolete qamomile-native distribution is installed")
    for module_name in installed_public_packages(distribution):
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            fail(f"installed distribution is missing {module_name!r}")
        validate_module_origin(module_name, spec.origin, repository_root)


def executable_ast(path: Path) -> str:
    """Return a stable executable AST with translated docstrings removed.

    Args:
        path: Jupytext Python source to inspect.

    Returns:
        Stable AST dump suitable for bilingual equality checks.

    Raises:
        SyntaxError: If the source is not valid Python.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stripped = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(stripped)
    return ast.dump(stripped, include_attributes=False)


def discover_pages(docs_root: Path) -> tuple[Path, ...]:
    """Discover paired canonical English documentation sources.

    Args:
        docs_root: Bilingual documentation root containing ``en`` and ``ja``.

    Returns:
        English Jupytext sources in deterministic section/name order.

    Raises:
        SystemExit: If a source lacks a paired notebook or Japanese source.
    """

    pages = tuple(sorted((docs_root / "en").glob("*/*.py")))
    if not pages:
        fail(f"no executable documentation found under {docs_root}")
    for english in pages:
        japanese = docs_root / "ja" / english.parent.name / english.name
        notebook = english.with_suffix(".ipynb")
        japanese_notebook = japanese.with_suffix(".ipynb")
        missing = [
            path
            for path in (japanese, notebook, japanese_notebook)
            if not path.is_file()
        ]
        if missing:
            fail(
                f"{english.relative_to(docs_root)} has missing pairs: "
                + ", ".join(str(path.relative_to(docs_root)) for path in missing)
            )
    return pages


def verify_bilingual_programs(docs_root: Path, pages: tuple[Path, ...]) -> None:
    """Require English and Japanese pages to share one executable program.

    Args:
        docs_root: Bilingual documentation root.
        pages: Canonical English source pages to compare.

    Raises:
        SystemExit: If translated executable code differs from English.
    """

    for english in pages:
        japanese = docs_root / "ja" / english.parent.name / english.name
        if executable_ast(english) != executable_ast(japanese):
            fail(
                f"English/Japanese executable drift in {english.relative_to(docs_root)}"
            )


def select_pages(
    docs_root: Path,
    pages: tuple[Path, ...],
    requested: tuple[str, ...],
) -> tuple[Path, ...]:
    """Select an explicit focused subset or every discovered page.

    Args:
        docs_root: Bilingual documentation root.
        pages: Every discovered canonical English page.
        requested: Optional POSIX paths relative to ``docs_root``.

    Returns:
        Selected pages in caller order, or every page when not requested.

    Raises:
        SystemExit: If a requested path is not a canonical English page.
    """

    if not requested:
        return pages
    available = {path.relative_to(docs_root).as_posix(): path for path in pages}
    selected: list[Path] = []
    for relative in requested:
        page = available.get(relative)
        if page is None:
            fail(
                f"requested page is not canonical executable documentation: {relative}"
            )
        selected.append(page)
    return tuple(selected)


def execute_pages(docs_root: Path, pages: tuple[Path, ...]) -> None:
    """Execute each page in a fresh isolated interpreter.

    Args:
        docs_root: Documentation root used for concise diagnostics.
        pages: Canonical English pages to execute sequentially.

    Raises:
        SystemExit: If a page times out or exits unsuccessfully.
    """

    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    environment["QAMOMILE_DOCS_TEST"] = "1"
    with tempfile.TemporaryDirectory(prefix="qamomile-installed-docs-") as workdir:
        matplotlib_config = Path(workdir) / "matplotlib"
        matplotlib_config.mkdir()
        environment["MPLCONFIGDIR"] = str(matplotlib_config)
        for page in pages:
            relative = page.relative_to(docs_root)
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", _CHILD_RUNNER, str(page.resolve())],
                    cwd=workdir,
                    env=environment,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as error:
                fail(f"{relative} exceeded the 600-second page limit: {error}")
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-4000:]
                fail(f"{relative} exited with {completed.returncode}: {detail}")
            print(f"verified installed docs page: {relative}", flush=True)


def main() -> None:
    """Parse arguments and verify installed-wheel documentation execution."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    parser.add_argument("--expected-version", required=True)
    parser.add_argument(
        "--page",
        action="append",
        default=[],
        help="focused English .py path relative to docs root; repeat as needed",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    docs_root = args.docs_root.resolve()
    repository_root = docs_root.parent
    verify_import_boundary(repository_root, args.expected_version)
    pages = discover_pages(docs_root)
    verify_bilingual_programs(docs_root, pages)
    selected = select_pages(docs_root, pages, tuple(args.page))
    if not args.check_only:
        execute_pages(docs_root, selected)
    print(
        "verified installed Qamomile documentation: "
        f"{len(selected)} executable pages, {len(pages)} bilingual pairs"
    )


if __name__ == "__main__":
    main()
