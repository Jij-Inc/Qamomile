"""Keep the generated private ABI complete and aligned with runtime exports."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from qamomile import _qamomile_native as native


def _definitions() -> dict[str, ast.AST]:
    stub = Path(__file__).resolve().parents[3] / "qamomile/_qamomile_native.pyi"
    tree = ast.parse(stub.read_text())
    return {
        node.name
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        else node.target.id: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        or isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }


def test_stub_covers_all_runtime_exports_and_members() -> None:
    definitions = _definitions()
    exports = {name for name in dir(native) if not name.startswith("_")}
    assert definitions.keys() == exports
    for name, node in definitions.items():
        runtime = getattr(native, name)
        if isinstance(node, ast.ClassDef):
            members = {
                member.name
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and not member.name.startswith("_")
            }
            assert members == {
                attr for attr in dir(runtime) if not attr.startswith("_")
            }
        elif isinstance(node, ast.FunctionDef):
            signature = inspect.signature(runtime)
            expected = [arg.arg for arg in node.args.posonlyargs + node.args.args]
            expected += [arg.arg for arg in node.args.kwonlyargs]
            assert list(signature.parameters) == expected, name


def test_stub_preserves_bytes_at_the_pyo3_boundary() -> None:
    definitions = _definitions()
    function = definitions["inspect_qcore_envelope"]
    assert ast.unparse(function.args.args[0].annotation) == "bytes"
    assert ast.unparse(function.returns) == "tuple[int, int, str, bytes]"
    function = definitions["sample_static_circuit_at"]
    assert ast.unparse(function.returns) == "list[tuple[bytes, int]]"
    projection = definitions["CircuitFamilyProjection"]
    getter = next(
        node
        for node in projection.body
        if isinstance(node, ast.FunctionDef) and node.name == "selection_plan"
    )
    assert ast.unparse(getter.returns).removeprefix("builtins.") == "bytes"


def test_native_distribution_contains_current_stub_and_typing_marker() -> None:
    package_root = Path(native.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
    installed = package_root / "_qamomile_native.pyi"
    expected = Path(__file__).resolve().parents[3] / "qamomile/_qamomile_native.pyi"
    assert installed.read_bytes() == expected.read_bytes()
