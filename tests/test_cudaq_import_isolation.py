"""Guard the CUDA-Q collection-isolation policy.

Verifies that every test module importing ``cudaq`` at module scope is
registered in ``CUDAQ_MODULE_LEVEL_IMPORTERS`` (so default ``-m "not
cudaq"`` runs never load cudaq during collection) and carries the
``cudaq`` pytestmark (so skipping collection never hides tests a default
run would have executed), that no conftest imports cudaq at module scope
(conftests are always imported), and that ``markexpr_can_select_cudaq``
decides the marker expressions the suite relies on. Background:
tests/_cudaq_isolation.py — mixing the cudaq/torch OpenMP runtimes with
qiskit-aer in one process segfaults.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._cudaq_isolation import (
    CUDAQ_MODULE_LEVEL_IMPORTERS,
    markexpr_can_select_cudaq,
)

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def _iter_module_level_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield AST nodes that execute when the module is imported.

    Descends into every construct except function bodies (class bodies do
    execute at import time, so they are included).
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _is_cudaq_import(node: ast.AST) -> bool:
    """Return True when the node imports cudaq (directly or via skip)."""
    if isinstance(node, ast.Import):
        return any(
            alias.name == "cudaq" or alias.name.startswith("cudaq.")
            for alias in node.names
        )
    if isinstance(node, ast.ImportFrom):
        # ``from qamomile.cudaq import X`` is fine: the package resolves
        # its public symbols through a lazy ``__getattr__`` that does not
        # load the cudaq runtime (pinned by test_cudaq_import_ux.py).
        module = node.module or ""
        return module == "cudaq" or module.startswith("cudaq.")
    if isinstance(node, ast.Call):
        func = node.func
        is_importorskip = (
            isinstance(func, ast.Attribute) and func.attr == "importorskip"
        ) or (isinstance(func, ast.Name) and func.id == "importorskip")
        return (
            is_importorskip
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "cudaq"
        )
    return False


def _imports_cudaq_at_module_level(path: Path) -> bool:
    """Return True when the file loads cudaq as a side effect of import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(_is_cudaq_import(node) for node in _iter_module_level_nodes(tree))


def test_module_level_cudaq_importers_are_registered():
    """Module-level cudaq importers stay in sync with the ignore table."""
    found = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(TESTS_DIR.rglob("test_*.py"))
        if _imports_cudaq_at_module_level(path)
    }
    assert found == set(CUDAQ_MODULE_LEVEL_IMPORTERS), (
        "Test modules importing cudaq at module scope must match "
        "tests._cudaq_isolation.CUDAQ_MODULE_LEVEL_IMPORTERS, so that "
        "default (-m 'not cudaq') runs never load cudaq during collection. "
        f"Found in sources: {sorted(found)}; registered: "
        f"{sorted(CUDAQ_MODULE_LEVEL_IMPORTERS)}. Either register the new "
        "module (and give it `pytestmark = pytest.mark.cudaq`) or move the "
        "cudaq import into the tests."
    )


def test_registered_modules_exist_and_carry_cudaq_pytestmark():
    """Every registered module exists and is cudaq-marked at module level."""
    for relative in sorted(CUDAQ_MODULE_LEVEL_IMPORTERS):
        path = REPO_ROOT / relative
        assert path.is_file(), f"{relative} is registered but does not exist"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        has_cudaq_pytestmark = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            and "cudaq" in ast.dump(node.value)
            for node in tree.body
        )
        assert has_cudaq_pytestmark, (
            f"{relative} must set `pytestmark = pytest.mark.cudaq`: the "
            "collection-ignore in tests/conftest.py assumes every test in "
            "the module is deselected by `-m 'not cudaq'` anyway."
        )


def test_conftests_do_not_import_cudaq_at_module_level():
    """Conftests must not import cudaq: they load in every session."""
    for path in sorted(TESTS_DIR.rglob("conftest.py")):
        assert not _imports_cudaq_at_module_level(path), (
            f"{path.relative_to(REPO_ROOT)} imports cudaq at module scope; "
            "conftest modules are imported unconditionally, which would "
            "defeat the cudaq isolation."
        )


@pytest.mark.parametrize(
    ("markexpr", "expected"),
    [
        ("not docs and not quri_parts and not cudaq", False),
        ("not cudaq", False),
        ("", True),
        ("   ", True),
        ("cudaq", True),
        ("cudaq and two_qubit", True),
        ("not docs", True),
    ],
)
def test_markexpr_can_select_cudaq(markexpr: str, expected: bool):
    """Pins the selection decisions the collection-ignore relies on."""
    assert markexpr_can_select_cudaq(markexpr) is expected
