#!/usr/bin/env python3
"""Verify that release archives contain one coherent Qamomile product."""

from __future__ import annotations

import argparse
import os
import re
import tarfile
import zipfile
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import NoReturn

PUBLIC_PACKAGES = {"qamomile"}
FORBIDDEN_RUNTIME_DEPENDENCIES = {"jijmodeling", "sympy"}
FORBIDDEN_SCM_METADATA = {"scm_file_list.json", "scm_version.json"}
RUNTIME_PACKAGE_NAMES = {"README.md", "py.typed"}
RUNTIME_PACKAGE_SUFFIXES = {".proto", ".py", ".pyi"}


def is_package_member(member: str) -> bool:
    """Identify runtime sources and bundled authoring skill resources.

    Args:
        member: Archive-relative package file path.

    Returns:
        Whether the release verifier must check this file for closure.
    """
    path = Path(member)
    return (
        path.name in RUNTIME_PACKAGE_NAMES
        or path.suffix in RUNTIME_PACKAGE_SUFFIXES
        or (
            member.startswith("qamomile/agent_plugins/")
            and path.suffix in {".md", ".yaml", ".json"}
        )
    )


def fail(message: str) -> NoReturn:
    """Stop distribution verification with a stable diagnostic.

    Args:
        message: Explanation of the violated release contract.

    Raises:
        SystemExit: Always, with a nonzero status.
    """

    raise SystemExit(f"distribution verification failed: {message}")


def one_match(directory: Path, pattern: str, label: str) -> Path:
    """Return the only artifact matching a release pattern.

    Args:
        directory: Directory containing release artifacts.
        pattern: Glob pattern used to select an artifact.
        label: Human-readable artifact kind for diagnostics.

    Returns:
        The unique matching path.

    Raises:
        SystemExit: If the directory does not contain exactly one match.
    """

    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        fail(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def root_artifacts(directory: Path) -> tuple[Path, Path, str]:
    """Resolve the root wheel, source archive, and shared version.

    Args:
        directory: Directory containing root Python distributions.

    Returns:
        Root wheel, source distribution, and normalized release version.

    Raises:
        SystemExit: If names or versions do not satisfy the release contract.
    """

    wheel = one_match(directory, "qamomile-*.whl", "qamomile wheel")
    sdist = one_match(directory, "qamomile-*.tar.gz", "qamomile source archive")
    match = re.fullmatch(r"qamomile-(.+)\.tar\.gz", sdist.name)
    if match is None:
        fail(f"cannot read a version from {sdist.name}")
    version = match.group(1)
    if not wheel.name.startswith(f"qamomile-{version}-"):
        fail("root wheel and source archive versions differ")
    return wheel, sdist, version


def parse_metadata(payload: bytes, artifact: Path) -> Message:
    """Parse core metadata from a wheel or source archive.

    Args:
        payload: Raw RFC 822 metadata bytes.
        artifact: Artifact used to locate parse failures.

    Returns:
        Parsed package metadata.

    Raises:
        SystemExit: If required metadata fields are missing.
    """

    metadata = BytesParser().parsebytes(payload)
    if metadata.get("Name") is None or metadata.get("Version") is None:
        fail(f"{artifact.name} has incomplete package metadata")
    return metadata


def dependency_name(requirement: str) -> str:
    """Extract a normalized distribution name from one requirement.

    Args:
        requirement: One ``Requires-Dist`` metadata value.

    Returns:
        Lowercase distribution name with underscores normalized to hyphens.
    """

    name = re.split(r"[ (;<=>!~]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def verify_root_metadata(metadata: Message, version: str, artifact: Path) -> None:
    """Verify identity and runtime dependencies of the root distribution.

    Args:
        metadata: Parsed root distribution metadata.
        version: Version derived from artifact names.
        artifact: Artifact being checked.

    Raises:
        SystemExit: If identity or dependency metadata is inconsistent.
    """

    if metadata["Name"].lower() != "qamomile":
        fail(f"{artifact.name} declares package {metadata['Name']!r}")
    if metadata["Version"] != version:
        fail(f"{artifact.name} metadata version differs from its filename")

    requirements = metadata.get_all("Requires-Dist", [])
    if any(dependency_name(value) == "qamomile-native" for value in requirements):
        fail("Qamomile must bundle its compiler without a qamomile-native dependency")

    dependency_names = {dependency_name(value) for value in requirements}
    legacy = sorted(dependency_names & FORBIDDEN_RUNTIME_DEPENDENCIES)
    if legacy:
        fail(f"legacy runtime dependencies remain: {', '.join(legacy)}")


def required_package_members(source_root: Path) -> set[str]:
    """Collect checked-in files required by the runtime package tree.

    Args:
        source_root: Repository root containing every public package.

    Returns:
        Archive-relative paths for Python modules, typing markers, schemas, and
        packaged authoring skill resources.

    Raises:
        SystemExit: If a declared public package is absent from the checkout.
    """

    required: set[str] = set()
    for package in sorted(PUBLIC_PACKAGES):
        package_root = source_root / package
        if not package_root.is_dir():
            fail(f"source checkout is missing package {package}")
        for path in package_root.rglob("*"):
            if path.is_file() and is_package_member(
                path.relative_to(source_root).as_posix()
            ):
                required.add(path.relative_to(source_root).as_posix())
    return required


def verify_python_members(
    members: set[str],
    artifact: Path,
    required_members: set[str],
) -> None:
    """Verify public sources and reject repository-only build metadata.

    Args:
        members: Archive member paths relative to the archive root.
        artifact: Artifact being checked.
        required_members: Runtime package files discovered from the checkout.

    Raises:
        SystemExit: If a package is absent or non-public files leak in.
    """
    repository_metadata = sorted(
        member for member in members if Path(member).name in FORBIDDEN_SCM_METADATA
    )
    if repository_metadata:
        fail(
            f"{artifact.name} contains source-control metadata: "
            + ", ".join(repository_metadata)
        )
    missing = sorted(
        package
        for package in PUBLIC_PACKAGES
        if f"{package}/__init__.py" not in members
    )
    if missing:
        fail(f"{artifact.name} is missing packages: {', '.join(missing)}")
    if "qamomile/py.typed" not in members:
        fail(f"{artifact.name} is missing the typed-package marker")
    if "qamomile/proto/qcore.proto" not in members:
        fail(f"{artifact.name} is missing its canonical QCore schema")
    missing_members = sorted(required_members - members)
    if missing_members:
        preview = ", ".join(missing_members[:10])
        remainder = len(missing_members) - 10
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        fail(f"{artifact.name} is missing runtime package files: {preview}{suffix}")
    unexpected_members = sorted(
        member
        for member in members
        if member.split("/", maxsplit=1)[0].startswith("qamomile")
        and is_package_member(member)
        and member not in required_members
    )
    if unexpected_members:
        preview = ", ".join(unexpected_members[:10])
        remainder = len(unexpected_members) - 10
        suffix = f" (and {remainder} more)" if remainder > 0 else ""
        fail(f"{artifact.name} contains stale runtime files: {preview}{suffix}")


def verify_root_wheel(
    wheel: Path,
    version: str,
    required_members: set[str],
) -> None:
    """Verify root wheel contents and metadata.

    Args:
        wheel: Root wheel path.
        version: Expected release version.
        required_members: Runtime package files discovered from the checkout.

    Raises:
        SystemExit: If the wheel violates the product boundary.
    """

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_paths = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            fail(f"{wheel.name} must contain exactly one METADATA file")
        metadata = parse_metadata(archive.read(metadata_paths[0]), wheel)
    verify_root_metadata(metadata, version, wheel)
    verify_python_members(members, wheel, required_members)
    verify_native_members(wheel, members)


def verify_root_sdist(
    sdist: Path,
    version: str,
    required_members: set[str],
) -> None:
    """Verify root source archive contents and metadata.

    Args:
        sdist: Root source archive path.
        version: Expected release version.
        required_members: Runtime package files discovered from the checkout.

    Raises:
        SystemExit: If the archive violates the product boundary.
    """

    with tarfile.open(sdist, "r:gz") as archive:
        raw_members = archive.getnames()
        prefixes = {name.split("/", maxsplit=1)[0] for name in raw_members}
        if len(prefixes) != 1:
            fail(f"{sdist.name} must have exactly one archive root")
        prefix = next(iter(prefixes))
        members = {
            name[len(prefix) + 1 :]
            for name in raw_members
            if name.startswith(f"{prefix}/")
        }
        pkg_info = archive.extractfile(f"{prefix}/PKG-INFO")
        if pkg_info is None:
            fail(f"{sdist.name} is missing PKG-INFO")
        metadata = parse_metadata(pkg_info.read(), sdist)
    verify_root_metadata(metadata, version, sdist)
    verify_python_members(members, sdist, required_members)
    required_rust = {
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "setup.cfg",
        "rust/qamomile-compiler/Cargo.toml",
        "rust/qamomile-compiler/src/lib.rs",
        "rust/qamomile-python/Cargo.toml",
        "rust/qamomile-python/src/lib.rs",
    }
    missing_rust = sorted(required_rust - members)
    if missing_rust:
        fail(f"source archive is missing Rust build inputs: {', '.join(missing_rust)}")
    forbidden_roots = {"design", "docs", "tests", ".claude", ".codex", ".github"}
    leaked = sorted(
        root
        for root in forbidden_roots
        if any(member == root or member.startswith(f"{root}/") for member in members)
    )
    if leaked:
        fail(f"source archive contains repository-only roots: {', '.join(leaked)}")


def verify_native_members(wheel: Path, members: set[str]) -> None:
    """Require the bundled stable-ABI extension and its package-local stub.

    Args:
        wheel: Combined Qamomile wheel path.
        members: Archive-relative wheel file paths.

    Raises:
        SystemExit: If native code or type information is absent or misplaced.
    """
    if not re.fullmatch(r"qamomile-.+-cp311-abi3-(?!any).+\.whl", wheel.name):
        fail(f"wheel has an unexpected name or ABI tag: {wheel.name}")
    extensions = {
        name
        for name in members
        if re.fullmatch(r"qamomile/_qamomile_native(?:\.[^/]+)?\.(?:so|pyd)", name)
    }
    if len(extensions) != 1:
        fail(f"{wheel.name} must contain exactly one package-local native extension")
    for member in ("qamomile/_qamomile_native.pyi", "qamomile/py.typed"):
        if member not in members:
            fail(f"{wheel.name} is missing native type information: {member}")
    if any(name.startswith("_qamomile_native") for name in members):
        fail(f"{wheel.name} contains the obsolete top-level native package")


def main() -> None:
    """Parse command-line arguments and verify all requested artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("distribution_dir", type=Path)
    parser.add_argument("--sdist-only", action="store_true")
    parser.add_argument(
        "--expected-version", default=os.environ.get("EXPECTED_VERSION")
    )
    args = parser.parse_args()

    if args.sdist_only:
        sdist = one_match(
            args.distribution_dir, "qamomile-*.tar.gz", "qamomile source archive"
        )
        version = sdist.name.removeprefix("qamomile-").removesuffix(".tar.gz")
        wheel = None
    else:
        wheel, sdist, version = root_artifacts(args.distribution_dir)
    if args.expected_version is not None and version != args.expected_version:
        fail(f"expected version {args.expected_version}, found {version}")
    source_root = Path(__file__).resolve().parents[2]
    required_members = required_package_members(source_root)
    if wheel is not None:
        verify_root_wheel(wheel, version, required_members)
    verify_root_sdist(sdist, version, required_members)
    print(f"verified Qamomile {version} distributions")


if __name__ == "__main__":
    main()
