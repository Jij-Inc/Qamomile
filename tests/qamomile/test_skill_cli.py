"""Regression coverage for the packaged skill and project-local CLI."""

from __future__ import annotations

import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import qamomile as qm
from qamomile import _skills
from qamomile._cli import main


def _command(project: Path, action: str, agent: str = "codex") -> int:
    """Run a skill management command against an isolated project."""
    return main(["skill", action, "--agent", agent, "--project", str(project)])


def _target(project: Path, agent: str = "codex") -> Path:
    """Locate the isolated project's skill directory."""
    return project / _skills.AGENT_PATHS[agent] / _skills.NAME


def _outdated(target: Path) -> None:
    """Mark a verified installation as belonging to an older distribution."""
    record = target / _skills.RECORD
    data = json.loads(record.read_bytes())
    data["sdk_version"] = "0.0.0"
    record.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("agent", ["codex", "claude-code"])
def test_skill_lifecycle(
    tmp_path: Path, agent: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both agents receive complete copies, no-op updates, and preserved backups."""
    assert _command(tmp_path, "status", agent) == 1
    assert not (tmp_path / _skills.AGENT_PATHS[agent]).exists()
    assert _command(tmp_path, "install", agent) == 0
    target = _target(tmp_path, agent)
    source = _skills.bundled_root() / _skills.NAME
    for path in source.rglob("*"):
        if path.is_file():
            assert (target / path.relative_to(source)).read_bytes() == path.read_bytes()
    assert _command(tmp_path, "status", agent) == 0
    record = target / _skills.RECORD
    before = record.read_bytes(), record.stat().st_mtime_ns
    assert _command(tmp_path, "update", agent) == 0
    assert (record.read_bytes(), record.stat().st_mtime_ns) == before
    assert not list(target.parent.parent.glob(".qamomile-backup-*"))
    _outdated(target)
    assert _command(tmp_path, "status", agent) == 1
    assert _command(tmp_path, "update", agent) == 0
    assert _command(tmp_path, "status", agent) == 0
    assert _command(tmp_path, "remove", agent) == 0
    assert not target.exists()
    backups = list(target.parent.parent.glob(".qamomile-backup-*"))
    assert len(backups) == 2
    assert all((backup / "SKILL.md").is_file() for backup in backups)
    output = capsys.readouterr().out
    assert all(str(backup) in output for backup in backups)
    assert _command(tmp_path, "status", agent) == 1


@pytest.mark.parametrize("action", ["install", "update", "remove"])
@pytest.mark.parametrize(
    "change", ["edit", "delete", "add", "directory", "record", "link"]
)
def test_mutations_protect_local_changes(
    tmp_path: Path, action: str, change: str
) -> None:
    """No managed mutation overwrites edited, missing, or additional entries."""
    assert _command(tmp_path, "install") == 0
    target = _target(tmp_path)
    reference = target / "references/authoring.md"
    if change == "edit":
        reference.write_text("User notes", encoding="utf-8")
    elif change == "delete":
        reference.unlink()
    elif change == "add":
        (target / "notes.txt").write_text("User notes", encoding="utf-8")
    elif change == "directory":
        (target / "personal").mkdir()
    elif change == "record":
        (target / _skills.RECORD).write_bytes(b"\xff")
    else:
        reference.unlink()
        reference.symlink_to(tmp_path / "missing.md")
    before = {p: p.lstat().st_mtime_ns for p in target.rglob("*")}
    assert _command(tmp_path, "status") == 1
    with pytest.raises(SystemExit) as error:
        _command(tmp_path, action)
    assert error.value.code == 2
    assert {p: p.lstat().st_mtime_ns for p in target.rglob("*")} == before


@pytest.mark.parametrize("action", ["install", "update", "remove"])
def test_unmanaged_skill_is_preserved(tmp_path: Path, action: str) -> None:
    """User-created skills are never adopted or overwritten automatically."""
    target = _target(tmp_path)
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("Personal skill", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _command(tmp_path, action)
    assert error.value.code == 2
    assert (target / "SKILL.md").read_text() == "Personal skill"


def test_default_project_and_repeated_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The working directory is used, and reinstalling preserves managed files."""
    monkeypatch.chdir(tmp_path)
    assert main(["skill", "install", "--agent", "codex"]) == 0
    record = _target(tmp_path) / _skills.RECORD
    before = record.read_bytes(), record.stat().st_mtime_ns
    with pytest.raises(SystemExit) as error:
        main(["skill", "install", "--agent", "codex"])
    assert error.value.code == 2
    assert (record.read_bytes(), record.stat().st_mtime_ns) == before


@pytest.mark.parametrize(
    "relative", [".agents", ".agents/skills", ".agents/skills/qamomile"]
)
@pytest.mark.parametrize("action", ["install", "update", "remove"])
def test_linked_destinations_are_rejected(
    tmp_path: Path, relative: str, action: str
) -> None:
    """Mutations cannot redirect writes through symlinked skill destinations."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SystemExit) as error:
        _command(tmp_path, action)
    assert error.value.code == 2
    assert list(outside.iterdir()) == []
    assert link.is_symlink()


def test_markdown_checkout_line_endings_are_portable(tmp_path: Path) -> None:
    """CRLF conversion does not falsely mark skill instructions as edited."""
    assert _command(tmp_path, "install") == 0
    for path in _target(tmp_path).rglob("*.md"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert _command(tmp_path, "status") == 0


def test_update_tracks_content_changes_within_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content checks detect revised references even without a version bump."""
    source = tmp_path / "bundle"
    shutil.copytree(_skills.bundled_root(), source)
    monkeypatch.setattr(_skills, "bundled_root", lambda: source)
    assert _command(tmp_path, "install") == 0
    references = source / "qamomile/references"
    (references / "authoring.md").unlink()
    (references / "new.md").write_text("New guidance", encoding="utf-8")
    assert _command(tmp_path, "status") == 1
    assert _command(tmp_path, "update") == 0
    assert _command(tmp_path, "status") == 0
    target = _target(tmp_path)
    assert not (target / "references/authoring.md").exists()
    assert (target / "references/new.md").read_text() == "New guidance"
    backup = next(target.parent.parent.glob(".qamomile-backup-*"))
    assert (backup / "references/authoring.md").is_file()


@pytest.mark.parametrize("action", ["update", "remove"])
def test_late_edits_are_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Edits between inspection and mutation are detected and restored."""
    assert _command(tmp_path, "install") == 0
    target = _target(tmp_path)
    _outdated(target)
    inspect = _skills._inspect

    def edit_after_inspection(path: Path) -> tuple[str, _skills._Installation | None]:
        """Simulate an editor saving after the original destination was checked."""
        result = inspect(path)
        if path == target:
            (target / "SKILL.md").write_text("Late edit", encoding="utf-8")
        return result

    monkeypatch.setattr(_skills, "_inspect", edit_after_inspection)
    with pytest.raises(SystemExit) as error:
        _command(tmp_path, action)
    assert error.value.code == 2
    assert (target / "SKILL.md").read_text() == "Late edit"


def test_failed_update_restores_previous_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication failure restores the previous managed installation."""
    assert _command(tmp_path, "install") == 0
    target = _target(tmp_path)
    _outdated(target)
    before = (target / _skills.RECORD).read_bytes()
    rename = Path.rename

    def fail_staging(path: Path, destination: Path) -> Path:
        """Fail only the staged replacement's final rename."""
        if path.parent.name.startswith(".qamomile-stage-"):
            raise OSError("Simulated rename failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_staging)
    with pytest.raises(SystemExit) as error:
        _command(tmp_path, "update")
    assert error.value.code == 2
    assert (target / _skills.RECORD).read_bytes() == before
    assert (target / "SKILL.md").is_file()
    assert not list(target.parent.parent.glob(".qamomile-stage-*"))


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows locks open files against renames"
)
@pytest.mark.parametrize("action", ["update", "remove"])
def test_open_editor_keeps_old_file_in_backup(tmp_path: Path, action: str) -> None:
    """Later saves through an open file handle cannot corrupt a new copy."""
    assert _command(tmp_path, "install") == 0
    target = _target(tmp_path)
    _outdated(target)
    with (target / "SKILL.md").open("a", encoding="utf-8") as editor:
        assert _command(tmp_path, action) == 0
        editor.write("\nLater save\n")
    backup = next(target.parent.parent.glob(".qamomile-backup-*"))
    assert (backup / "SKILL.md").read_text().endswith("Later save\n")
    assert target.exists() == (action == "update")


@pytest.mark.parametrize(
    "arguments",
    [[], ["skill"], ["skill", "install"], ["skill", "install", "--agent", "unknown"]],
)
def test_invalid_arguments(arguments: list[str]) -> None:
    """Argparse rejects incomplete commands and unsupported agents."""
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_project(tmp_path: Path, kind: str) -> None:
    """The CLI requires an existing project directory."""
    project = tmp_path / "project"
    if kind == "file":
        project.write_text("Not a directory", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _command(project, "install")
    assert error.value.code == 2


def test_read_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read-only CLI commands print bundle contents without creating a copy."""
    monkeypatch.chdir(tmp_path)
    assert main(["skill", "path"]) == 0
    root = Path(capsys.readouterr().out.strip())
    assert root.is_absolute()
    assert (root / "qamomile/SKILL.md").is_file()
    for reference in ["SKILL.md", "references/authoring.md"]:
        args = ["skill", "show"] + ([] if reference == "SKILL.md" else [reference])
        assert main(args) == 0
        assert capsys.readouterr().out == (root / "qamomile" / reference).read_text()
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == _skills.sdk_version()
    assert list(tmp_path.iterdir()) == []


def test_plugin_and_marketplace_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both agent catalogs resolve to one plugin and the direct skill source."""
    monkeypatch.chdir(tmp_path)
    assert main(["marketplace", "path"]) == 0
    marketplace = Path(capsys.readouterr().out.strip())
    assert main(["plugin", "path"]) == 0
    plugin = Path(capsys.readouterr().out.strip())
    assert marketplace.is_absolute() and plugin.is_absolute()
    assert plugin == marketplace / "plugins/qamomile"
    assert _skills.bundled_root() == plugin / "skills"
    portable = json.loads((plugin / "plugin.json").read_bytes())
    claude = json.loads((plugin / ".claude-plugin/plugin.json").read_bytes())
    assert portable == claude
    assert portable["name"] == "qamomile"
    codex_catalog = json.loads(
        (marketplace / ".agents/plugins/marketplace.json").read_bytes()
    )
    claude_catalog = json.loads(
        (marketplace / ".claude-plugin/marketplace.json").read_bytes()
    )
    assert codex_catalog["name"] == claude_catalog["name"] == "qamomile"
    codex_source = codex_catalog["plugins"][0]["source"]
    assert codex_source["source"] == "local"
    assert (marketplace / codex_source["path"]).resolve() == plugin
    assert (marketplace / claude_catalog["plugins"][0]["source"]).resolve() == plugin
    assert portable["version"] == claude_catalog["metadata"]["version"]
    assert portable["version"] == claude_catalog["plugins"][0]["version"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "relative",
    [
        ".agents/plugins/marketplace.json",
        ".claude-plugin/marketplace.json",
        "plugins/qamomile/plugin.json",
        "plugins/qamomile/.claude-plugin/plugin.json",
    ],
)
def test_plugin_commands_reject_incomplete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    """Missing package resources yield actionable CLI errors."""
    source = tmp_path / "package"
    shutil.copytree(_skills.marketplace_root(), source / "agent_plugins")
    (source / "agent_plugins" / relative).unlink()
    monkeypatch.setattr(_skills, "files", lambda name: source)
    with pytest.raises(SystemExit) as error:
        main(["plugin", "path"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "reference",
    ["../README.md", "/etc/passwd", "references/missing.md", "agents/openai.yaml"],
)
def test_show_rejects_non_reference_paths(reference: str) -> None:
    """Show can only read exact bundled Markdown paths."""
    with pytest.raises(SystemExit) as error:
        main(["skill", "show", reference])
    assert error.value.code == 2


def test_module_entrypoint_matches_console_script() -> None:
    """The supported module and console entrypoints print the same skill."""
    for command in (
        [sys.executable, "-m", "qamomile"],
        [str(Path(sys.executable).parent / "qamomile")],
    ):
        result = subprocess.run(
            [*command, "skill", "show"], check=True, capture_output=True, text=True
        )
        assert result.stdout == _skills.read_reference("SKILL.md")


def test_skill_metadata_and_relative_links() -> None:
    """All routed resources exist and metadata names the discoverable skill."""
    root = _skills.bundled_root() / "qamomile"
    entry = (root / "SKILL.md").read_text()
    metadata = yaml.safe_load(entry.split("---", 2)[1])
    assert metadata["name"] == root.name
    assert metadata["description"]
    ui = yaml.safe_load((root / "agents/openai.yaml").read_text())
    assert "$qamomile" in ui["interface"]["default_prompt"]
    assert ui.get("policy", {}).get("allow_implicit_invocation", True)
    for path in root.rglob("*.md"):
        for reference in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            assert (path.parent / reference).is_file(), (path, reference)


@pytest.mark.parametrize(
    "reference, entry, bindings, parameters, expected",
    [
        ("SKILL.md", "bell", {}, {}, {(0, 0), (1, 1)}),
        (
            "references/quantum/ownership.md",
            "flip_window",
            {},
            {},
            {(0, 1, 1, 1, 0, 0)},
        ),
        ("references/quantum/control.md", "selected", {}, {"at": 0}, {(1, 0, 1)}),
        ("references/quantum/reuse.md", "sample_repeat", {"rounds": 3}, {}, {(1,)}),
        ("references/quantum/reuse.md", "sample_pair", {}, {}, {(1, 1)}),
        ("references/quantum/reuse.md", "sample_recursive", {"rounds": 3}, {}, {(1,)}),
        ("references/quantum/arithmetic.md", "sum_sample", {}, {}, {(1, 1, 0, 1)}),
        ("references/quantum/flow.md", "sqrt_rotation", {}, {}, {(1,)}),
        pytest.param(
            "references/quantum/flow.md",
            "integer_rotation",
            {},
            {"offset": 0},
            {(1,)},
            id="integer-rotation-offset-0",
        ),
        pytest.param(
            "references/quantum/flow.md",
            "integer_rotation",
            {},
            {"offset": 1},
            {(0,)},
            id="integer-rotation-offset-1",
        ),
        ("references/quantum/flow.md", "correlated_pair", {}, {}, {(0, 0), (1, 1)}),
    ],
)
def test_skill_examples_execute_from_packaged_sources(
    tmp_path: Path,
    reference: str,
    entry: str,
    bindings: dict[str, object],
    parameters: dict[str, object],
    expected: set[tuple[int, ...]],
) -> None:
    """Shipped quantum examples execute their documented mathematical operation."""
    blocks = re.findall(
        r"```python\n(.*?)```", _skills.read_reference(reference), re.DOTALL
    )
    body = "\n\n".join(blocks).replace("from __future__ import annotations\n", "")
    source = "from __future__ import annotations\nimport qamomile as qm\n" + body
    candidate = tmp_path / "example.py"
    candidate.write_text(source, encoding="utf-8")
    namespace = runpy.run_path(str(candidate))
    program = qm.specialize(namespace[entry], **bindings)
    record = qm.run(
        program, on=qm.Reference(), parameters=parameters, shots=128, seed=7
    )
    assert set(record.result.counts) == expected


def test_skill_flow_and_library_examples(tmp_path: Path) -> None:
    """Per-shot flow and product-formula examples retain their typed semantics."""
    for reference in ["references/quantum/flow.md", "references/libraries.md"]:
        body = "\n\n".join(
            re.findall(
                r"```python\n(.*?)```", _skills.read_reference(reference), re.DOTALL
            )
        )
        path = tmp_path / (Path(reference).stem + ".py")
        path.write_text("import qamomile as qm\n" + body, encoding="utf-8")
        namespace = runpy.run_path(str(path))
        if "score" in namespace:
            record = qm.run(namespace["score"], on=qm.Reference(), shots=128, seed=7)
            assert set(record.result.values) == {(-1.0, 1.0), (1.0, 1.0)}
            record = qm.run(
                namespace["repeat_once"], on=qm.Reference(), shots=8, seed=7
            )
            assert dict(record.result.values) == {True: 8}
            record = qm.run(
                qm.specialize(
                    namespace["sliced_rotation"],
                    angles=(0.0, 3.141592653589793),
                    start=1,
                ),
                on=qm.Reference(),
                shots=8,
                seed=7,
            )
            assert record.result.counts == {(1,): 8}
            record = qm.run(
                namespace["duplicate_measured_bit"], on=qm.Reference(), shots=8, seed=7
            )
            assert record.result.values == {((1, 1), (1, 0)): 8}
            record = qm.run(
                namespace["table_program"], on=qm.Reference(), shots=8, seed=7
            )
            assert record.result.counts == {(1,): 8}

        else:
            program = qm.specialize(namespace["state"], terms=(qm.Z(0), qm.X(0)))
            assert program.to_bytes()
            filter_result = namespace["filter_result"]
            assert filter_result.success_probability > 0.0
            assert filter_result.samples.lowest.energy == 0.0
            record = qm.run(
                namespace["approximate_roundtrip"], on=qm.Reference(), shots=8, seed=7
            )
            assert record.result.counts == {(1, 0, 1): 8}
            assert namespace["initial_run"].samples.total_occurrences == 8
            assert namespace["initial_run"].invocation.values() == {}


def test_skill_selected_qubits_and_chain_depth_examples(tmp_path: Path) -> None:
    """Execute the packaged owned-selection and symbolic-depth examples."""
    for reference in ("references/libraries.md", "references/resources.md"):
        blocks = re.findall(
            r"```python\n(.*?)```", _skills.read_reference(reference), re.DOTALL
        )
        selected = next(
            block
            for block in blocks
            if "noncontiguous_sample" in block or "ghz_depth_sample" in block
        )
        path = tmp_path / "selected_example.py"
        path.write_text(
            "from __future__ import annotations\nimport qamomile as qm\n" + selected
        )
        namespace = runpy.run_path(str(path))
        if "noncontiguous_sample" in namespace:
            assert qm.run(
                namespace["noncontiguous_sample"], on=qm.Reference(), shots=8, seed=7
            ).result.counts == {(1, 0, 1, 0): 8}


def test_skill_compiler_optimization_example(tmp_path: Path) -> None:
    """The actual packaged extension example composes and executes through Rust."""
    blocks = re.findall(
        r"```python\n(.*?)```",
        _skills.read_reference("references/compiler-optimization.md"),
        re.DOTALL,
    )
    assert len(blocks) == 3
    path = tmp_path / "compiler_optimization.py"
    path.write_text("\n\n".join(blocks))
    namespace = runpy.run_path(str(path))
    assert namespace["result"].counts == {(1,): 4}
    assert namespace["multilevel_result"].counts == {(1,): 4}
    assert namespace["aware"].seen == (None, "sha256:maxcut-3")
    assert namespace["profile"].topology.edges[0].symmetric is True
    assert (
        namespace["artifact"].report.stages.count(
            "optimization:example.hadamard_conjugation@1"
        )
        == 1
    )


def test_skill_keyed_parameters_and_exact_gas_examples(tmp_path: Path) -> None:
    """Execute the actual shipped dictionary input and fractional GAS examples."""
    for reference, marker in (
        ("references/execution.md", "keyed_angles"),
        ("references/libraries.md", "exact_gas"),
    ):
        blocks = re.findall(
            r"```python\n(.*?)```", _skills.read_reference(reference), re.DOTALL
        )
        body = next(block for block in blocks if marker in block)
        path = tmp_path / "keyed_gas_example.py"
        path.write_text("import qamomile as qm\n" + body)
        namespace = runpy.run_path(str(path))
        if marker == "keyed_angles":
            assert namespace["keyed_result"].counts == {(1,): 8}
        else:
            assert namespace["exact_gas_run"].samples.lowest.assignment == (
                (4, 0),
                (9, 0),
            )
            assert namespace["exact_gas_run"].samples.lowest.occurrences == 8


@pytest.mark.braket
def test_skill_braket_restore_example(monkeypatch) -> None:
    """The shipped restoration example runs against an existing SDK stand-in task."""
    from .test_braket_restore import create_saved_task

    saved = create_saved_task(monkeypatch)
    blocks = re.findall(
        r"```python\n(.*?)```",
        _skills.read_reference("references/execution.md"),
        re.DOTALL,
    )
    namespace = {
        "qm": qm,
        "saved_snapshot": saved.snapshot.to_bytes(),
        "endpoint": saved.endpoint,
        "artifact": saved.artifact,
        "original_parameters": {"flip": False},
    }
    exec(blocks[-1], namespace)
    assert namespace["result"].values == {((1, 0, 0), True): 4}
    assert len(saved.launches) == 1


@pytest.mark.qir
def test_skill_qir_export_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packaged export example writes valid standalone LLVM text and bitcode."""
    pyqir = pytest.importorskip("pyqir")
    runner = pytest.importorskip("qirrunner")
    blocks = re.findall(
        r"```python\n(.*?)```", _skills.read_reference("references/qir.md"), re.DOTALL
    )
    path = tmp_path / "qir_example.py"
    path.write_text(blocks[0])
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path))
    artifact = namespace["artifact"]
    module = pyqir.Module.from_ir(pyqir.Context(), (tmp_path / "bell.ll").read_text())
    assert module.verify() is None
    assert (tmp_path / "bell.bc").read_bytes() == qm.qir.load_bitcode(artifact)
    handler = runner.OutputHandler()
    runner.run_bytes(
        (tmp_path / "bell.bc").read_bytes(),
        shots=8,
        rng_seed=7,
        output_fn=handler.handle,
    )
    bits = [
        line.split("\t")[2]
        for line in handler.get_output().splitlines()
        if line.startswith("OUTPUT\tRESULT\t")
    ]
    assert len(bits) == 16
    assert set(zip(bits[::2], bits[1::2])) <= {("0", "0"), ("1", "1")}


def test_skill_visualization_example(tmp_path: Path, monkeypatch) -> None:
    """The shipped drawing workflow exports register and native qubit diagrams."""
    from xml.etree import ElementTree

    reference = _skills.read_reference("references/visualization.md")
    body = "\n\n".join(re.findall(r"```python\n(.*?)```", reference, re.DOTALL)[:2])
    path = tmp_path / "visualization_example.py"
    path.write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path))
    assert "<svg " in namespace["diagram"]._repr_html_()
    assert namespace["circuit"].wires == "qubits"
    for name in ["algorithm.svg", "circuit.svg"]:
        assert ElementTree.parse(tmp_path / name).getroot().tag == (
            "{http://www.w3.org/2000/svg}svg"
        )
    assert "Expand iterations" in (tmp_path / "algorithm.html").read_text()


def test_skill_visualization_static_exports(tmp_path, monkeypatch, static_export):
    """The optional export example produces actual PNG and PDF documents."""
    reference = _skills.read_reference("references/visualization.md")
    body = "\n\n".join(re.findall(r"```python\n(.*?)```", reference, re.DOTALL))
    path = tmp_path / "static_example.py"
    path.write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(path))
    assert (tmp_path / "algorithm.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "circuit.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "algorithm.pdf").read_bytes().startswith(b"%PDF-")


def test_packaged_recursive_resource_example(tmp_path: Path) -> None:
    """The shipped recurrence example evaluates and persists native equations."""
    blocks = re.findall(
        r"```python\n(.*?)```",
        _skills.read_reference("references/resources.md"),
        re.DOTALL,
    )
    block = next(block for block in blocks if "def recursive_resource_step(" in block)
    candidate = tmp_path / "recursive_resources.py"
    candidate.write_text("import qamomile as qm\n" + block, encoding="utf-8")
    runpy.run_path(str(candidate))


def test_packaged_while_resource_example(tmp_path: Path) -> None:
    """The shipped while example binds assumptions and executes its real condition."""
    blocks = re.findall(
        r"```python\n(.*?)```",
        _skills.read_reference("references/resources.md"),
        re.DOTALL,
    )
    block = next(block for block in blocks if "def while_resource_step(" in block)
    candidate = tmp_path / "while_resources.py"
    candidate.write_text("import qamomile as qm\n" + block, encoding="utf-8")
    runpy.run_path(str(candidate))
