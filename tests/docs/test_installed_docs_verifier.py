"""Unit tests for installed-wheel documentation verification."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.docs
SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "verify-installed-docs.py"


def _load_verifier() -> ModuleType:
    """Load the release verifier without adding its directory to imports.

    Returns:
        Loaded verifier module.
    """

    spec = importlib.util.spec_from_file_location("verify_installed_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_docs_reject_repository_import_origin(tmp_path: Path) -> None:
    """An import satisfied by the checkout fails the release boundary."""

    verifier = _load_verifier()
    origin = tmp_path / "qamomile" / "__init__.py"

    with pytest.raises(SystemExit, match="resolved from the repository"):
        verifier.validate_module_origin("qamomile", str(origin), tmp_path)


def test_bilingual_ast_ignores_only_translated_docstrings(tmp_path: Path) -> None:
    """Translated docstrings compare equal while executable drift remains visible."""

    verifier = _load_verifier()
    english = tmp_path / "english.py"
    japanese = tmp_path / "japanese.py"
    english.write_text(
        '"""English."""\n\ndef value():\n    """Return one."""\n    return 1\n',
        encoding="utf-8",
    )
    japanese.write_text(
        '"""Japanese."""\n\ndef value():\n    """One in Japanese."""\n    return 1\n',
        encoding="utf-8",
    )

    assert verifier.executable_ast(english) == verifier.executable_ast(japanese)

    japanese.write_text("def value():\n    return 2\n", encoding="utf-8")
    assert verifier.executable_ast(english) != verifier.executable_ast(japanese)


def _localized_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ModuleType, Path, Path, Path, Path]:
    """Create a reviewed pair with translated output and a moved definition."""
    verifier = _load_verifier()
    docs_root = tmp_path / "docs"
    english = docs_root / "en/algorithm/example.py"
    japanese = docs_root / "ja/algorithm/example.py"
    english.parent.mkdir(parents=True)
    japanese.parent.mkdir(parents=True)
    english.write_text(
        'value = 1\ndef output(value):\n    print("Value:", value)\noutput(value)\n'
    )
    japanese.write_text(
        'def output(value):\n    print("値:", value)\nvalue = 1\noutput(value)\n'
    )
    localization_root = tmp_path / "docs-localization"
    monkeypatch.setattr(verifier, "_LOCALIZATION_ROOT", localization_root)
    record_path = localization_root / "algorithm/example.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "source_pr": "Jij-Inc/Qamomile#375",
                "source_sha": "a" * 40,
                "english_ast_sha256": hashlib.sha256(
                    verifier.executable_ast(english).encode()
                ).hexdigest(),
                "japanese_ast_sha256": hashlib.sha256(
                    verifier.executable_ast(japanese).encode()
                ).hexdigest(),
            }
        )
    )
    return verifier, docs_root, english, japanese, record_path


def test_reviewed_localized_pair_requires_both_exact_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known translation and definition placement pass without normalizing code."""
    verifier, root, english, japanese, _ = _localized_pair(tmp_path, monkeypatch)
    assert verifier.executable_ast(english) != verifier.executable_ast(japanese)
    verifier.verify_bilingual_programs(root, (english,))
    japanese.write_text(
        japanese.read_text() + "# Formatting does not change the AST.\n"
    )
    verifier.verify_bilingual_programs(root, (english,))


@pytest.mark.parametrize("language", ["en", "ja"])
@pytest.mark.parametrize(
    "original,changed",
    [
        ("value = 1", "value = 2"),
        ('print("', 'print("unknown '),
        ("output(value)\n", "output(-value)\n"),
        ("value = 1\n", "pass\nvalue = 1\n"),
    ],
)
def test_reviewed_pair_rejects_unknown_program_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    original: str,
    changed: str,
) -> None:
    """Numeric values, display text, arguments and statements stay checked."""
    verifier, root, english, japanese, _ = _localized_pair(tmp_path, monkeypatch)
    target = english if language == "en" else japanese
    target.write_text(target.read_text().replace(original, changed))
    with pytest.raises(SystemExit, match="localized executable drift"):
        verifier.verify_bilingual_programs(root, (english,))


def test_registered_pair_does_not_fall_back_to_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical edits in both languages cannot bypass an existing record."""
    verifier, root, english, japanese, _ = _localized_pair(tmp_path, monkeypatch)
    for page in (english, japanese):
        page.write_text("value = 2\n")
    with pytest.raises(SystemExit, match="localized executable drift"):
        verifier.verify_bilingual_programs(root, (english,))


def test_unregistered_pair_keeps_strict_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reviewed record for one page cannot authorize another page."""
    verifier, root, english, japanese, _ = _localized_pair(tmp_path, monkeypatch)
    other = english.with_name("other.py")
    english.rename(other)
    japanese.rename(japanese.with_name("other.py"))
    with pytest.raises(SystemExit, match="English/Japanese executable drift"):
        verifier.verify_bilingual_programs(root, (other,))


@pytest.mark.parametrize("invalid", ["{", "[]", "{}", '{"english_ast_sha256": 1}'])
def test_invalid_localization_record_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    """Present but malformed records are errors, including otherwise equal code."""
    verifier, root, english, japanese, record = _localized_pair(tmp_path, monkeypatch)
    japanese.write_text(english.read_text())
    record.write_text(invalid)
    with pytest.raises(SystemExit, match="invalid localization record"):
        verifier.verify_bilingual_programs(root, (english,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_pr", "375"),
        ("source_sha", "a" * 39),
        ("english_ast_sha256", "g" * 64),
        ("japanese_ast_sha256", None),
    ],
)
def test_localization_record_validates_provenance_and_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    """A record has complete provenance and well-formed language-specific hashes."""
    verifier, root, english, _, record = _localized_pair(tmp_path, monkeypatch)
    payload = json.loads(record.read_text())
    payload[field] = value
    record.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="invalid localization record"):
        verifier.verify_bilingual_programs(root, (english,))


def test_swapped_language_digests_are_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record distinguishes the English and Japanese executable programs."""
    verifier, root, english, _, record = _localized_pair(tmp_path, monkeypatch)
    payload = json.loads(record.read_text())
    payload["english_ast_sha256"], payload["japanese_ast_sha256"] = (
        payload["japanese_ast_sha256"],
        payload["english_ast_sha256"],
    )
    record.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="localized executable drift"):
        verifier.verify_bilingual_programs(root, (english,))


@pytest.mark.parametrize("japanese_status", [0, 1])
def test_registered_pair_executes_both_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, japanese_status: int
) -> None:
    """A selected English page also runs Japanese, whose failure is fatal."""
    verifier, root, english, japanese, _ = _localized_pair(tmp_path, monkeypatch)
    executed = []

    def run(command, **kwargs):
        page = Path(command[-1])
        executed.append(page)
        assert "-I" in command
        assert Path(kwargs["cwd"]) != root
        assert kwargs["env"]["QAMOMILE_DOCS_TEST"] == "1"
        return SimpleNamespace(
            returncode=japanese_status if page == japanese else 0,
            stderr="Japanese execution failed",
            stdout="",
        )

    monkeypatch.setattr(verifier.subprocess, "run", run)
    if japanese_status:
        with pytest.raises(SystemExit, match="ja/algorithm/example.py exited with 1"):
            verifier.execute_pages(root, (english,))
    else:
        verifier.execute_pages(root, (english,))
    assert executed == [english, japanese]


def test_only_empty_optional_type_parameters_are_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.12's empty field stays compatible while real parameters remain."""
    verifier = _load_verifier()
    source = tmp_path / "example.py"
    source.write_text("def value():\n    return 1\n")
    expected = verifier.executable_ast(source)
    original_parse = ast.parse
    parameters = []

    def parse(*args, **kwargs):
        tree = original_parse(*args, **kwargs)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "type_params" not in node._fields:
                    node._fields = (*node._fields, "type_params")
                node.type_params = parameters
        return tree

    monkeypatch.setattr(verifier.ast, "parse", parse)
    assert verifier.executable_ast(source) == expected
    parameters.append(ast.Name(id="T", ctx=ast.Load()))
    assert verifier.executable_ast(source) != expected
