"""Bounded regression coverage for the release archive verifier."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_verifier() -> ModuleType:
    """Load the repository-owned distribution verifier as a Python module."""

    path = Path(__file__).parents[2] / ".github/scripts/verify-distributions.py"
    spec = importlib.util.spec_from_file_location("verify_distributions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "target",
    [
        "qamomile/hugr/_target.py",
        "qamomile/_cli.py",
        "qamomile/qir/_emission.py",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/SKILL.md",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/references/authoring.md",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/references/qir.md",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/references/quantum/arithmetic.md",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/references/quantum/control.md",
        "qamomile/agent_plugins/plugins/qamomile/skills/qamomile/agents/openai.yaml",
        "qamomile/agent_plugins/plugins/qamomile/plugin.json",
        "qamomile/agent_plugins/plugins/qamomile/.claude-plugin/plugin.json",
        "qamomile/agent_plugins/.agents/plugins/marketplace.json",
        "qamomile/agent_plugins/.claude-plugin/marketplace.json",
    ],
)
def test_distribution_verifier_rejects_a_missing_package_member(target: str) -> None:
    """Archive closure includes private modules and skill resources."""

    verifier = _load_verifier()
    source_root = Path(__file__).parents[2]
    required = verifier.required_package_members(source_root)

    assert target in required
    with pytest.raises(SystemExit, match=target):
        verifier.verify_python_members(
            required - {target}, Path("fixture.whl"), required
        )


@pytest.mark.parametrize(
    "target", ["qamomile/README.md", "qamomile/quration/README.md"]
)
def test_distribution_verifier_rejects_removed_package_readmes(target: str) -> None:
    """Old package documentation must not leak from stale build directories."""
    verifier = _load_verifier()
    required = verifier.required_package_members(Path(__file__).parents[2])

    assert target not in required
    with pytest.raises(SystemExit, match="stale runtime files"):
        verifier.verify_python_members(
            required | {target}, Path("fixture.whl"), required
        )


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "metadata_file", [None, "scm_file_list.json", "scm_version.json"]
)
def test_root_archives_exclude_repository_metadata(
    tmp_path: Path, kind: str, metadata_file: str | None
) -> None:
    """Public archives reject repository paths and branch metadata."""
    verifier = _load_verifier()
    version = "0.16.0rc1"
    members = {
        "qamomile/__init__.py": "",
        "qamomile/py.typed": "",
        "qamomile/proto/qcore.proto": "",
        "qamomile/_qamomile_native.pyi": "",
    }
    required = set(members)
    metadata = f"Name: qamomile\nVersion: {version}\n"
    if kind == "wheel":
        metadata_dir = f"qamomile-{version}.dist-info"
        members[f"{metadata_dir}/METADATA"] = metadata
        members["qamomile/_qamomile_native.abi3.so"] = ""
        artifact = tmp_path / f"qamomile-{version}-cp311-abi3-test.whl"
        verify = verifier.verify_root_wheel
    else:
        metadata_dir = "qamomile.egg-info"
        members["PKG-INFO"] = metadata
        members.update(
            dict.fromkeys(
                (
                    "Cargo.toml",
                    "Cargo.lock",
                    "rust-toolchain.toml",
                    "setup.cfg",
                    "rust/qamomile-compiler/Cargo.toml",
                    "rust/qamomile-compiler/src/lib.rs",
                    "rust/qamomile-python/Cargo.toml",
                    "rust/qamomile-python/src/lib.rs",
                ),
                "",
            )
        )
        artifact = tmp_path / f"qamomile-{version}.tar.gz"
        verify = verifier.verify_root_sdist
    if metadata_file is not None:
        members[f"{metadata_dir}/{metadata_file}"] = (
            '{"files": ["design/private-plan.md"]}'
            if metadata_file == "scm_file_list.json"
            else '{"branch": "private-work"}'
        )

    if kind == "wheel":
        with zipfile.ZipFile(artifact, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
    else:
        with tarfile.open(artifact, "w:gz") as source_archive:
            for name, content in members.items():
                payload = content.encode()
                info = tarfile.TarInfo(f"qamomile-{version}/{name}")
                info.size = len(payload)
                source_archive.addfile(info, io.BytesIO(payload))

    if metadata_file is None:
        verify(artifact, version, required)
    else:
        with pytest.raises(SystemExit, match="source-control metadata"):
            verify(artifact, version, required)


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "qamomile/_qamomile_native.pyi",
        "qamomile/py.typed",
        "qamomile/_qamomile_native.abi3.so",
    ],
)
def test_combined_wheel_requires_native_code_and_type_information(
    missing: str | None,
) -> None:
    """A single install includes the compiler and its adjacent type contract."""
    verifier = _load_verifier()
    wheel = Path("qamomile-1.2.3-cp311-abi3-test.whl")
    members = {
        "qamomile/_qamomile_native.abi3.so",
        "qamomile/_qamomile_native.pyi",
        "qamomile/py.typed",
    } - {missing}
    if missing is None:
        verifier.verify_native_members(wheel, members)
    else:
        with pytest.raises(SystemExit):
            verifier.verify_native_members(wheel, members)


@pytest.mark.parametrize("tag", ["py3-none-any", "cp312-cp312-linux_x86_64"])
def test_combined_wheel_rejects_non_stable_abi_tags(tag: str) -> None:
    """Pure Python and interpreter-specific binaries violate the release contract."""
    verifier = _load_verifier()
    with pytest.raises(SystemExit, match="ABI tag"):
        verifier.verify_native_members(Path(f"qamomile-1.2.3-{tag}.whl"), set())


def test_combined_metadata_rejects_a_separate_native_dependency() -> None:
    """One distribution cannot silently fall back to the obsolete package."""
    verifier = _load_verifier()
    artifact = Path("fixture.whl")
    metadata = verifier.parse_metadata(
        b"Name: qamomile\nVersion: 1.2.3\nRequires-Dist: qamomile-native>=0.1\n",
        artifact,
    )
    with pytest.raises(SystemExit, match="without a qamomile-native dependency"):
        verifier.verify_root_metadata(metadata, "1.2.3", artifact)
