"""Distribute authoring instructions without changing compiler semantics.

This module manages only bundled files and project-local copies. It neither
executes skill code nor contacts an agent or a remote skill catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import cast
from uuid import uuid4

NAME = "qamomile"
RECORD = ".qamomile-install.json"
AGENT_PATHS = {"codex": Path(".agents/skills"), "claude-code": Path(".claude/skills")}


def sdk_version() -> str:
    """Return the installed Qamomile distribution version.

    Returns:
        str: Version recorded in installed package metadata.
    """
    return version(NAME)


def _linked(path: Path) -> bool:
    """Detect symlinks and Windows reparse points, including Python 3.11.

    Args:
        path (Path): Entry to inspect without following a link.

    Returns:
        bool: Whether the entry redirects filesystem traversal.
    """
    if os.name != "nt":
        return path.is_symlink()
    try:
        return bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except FileNotFoundError:
        return False


def marketplace_root() -> Path:
    """Locate the packaged local marketplace for Codex and Claude Code.

    Returns:
        Path: Absolute marketplace root containing both agent catalogs.

    Raises:
        ValueError: If either bundled marketplace catalog is unavailable.
    """
    root = files(NAME).joinpath("agent_plugins")
    if (
        not isinstance(root, Path)
        or not (root / ".agents/plugins/marketplace.json").is_file()
        or not (root / ".claude-plugin/marketplace.json").is_file()
    ):
        raise ValueError(
            "Bundled Qamomile marketplace is unavailable on disk; reinstall Qamomile"
        )
    return root.resolve(strict=True)


def plugin_root() -> Path:
    """Locate the bundled authoring plugin shared by local skill managers.

    Returns:
        Path: Absolute directory containing plugin manifests and skills.

    Raises:
        ValueError: If a required manifest or the authoring skill is unavailable.
    """
    root = marketplace_root() / "plugins" / NAME
    if not all(
        (root / relative).is_file()
        for relative in (
            "plugin.json",
            ".claude-plugin/plugin.json",
            f"skills/{NAME}/SKILL.md",
        )
    ):
        raise ValueError(
            "Bundled Qamomile plugin is unavailable on disk; reinstall Qamomile"
        )
    return root.resolve(strict=True)


def bundled_root() -> Path:
    """Locate the directory containing the packaged authoring skill.

    Returns:
        Path: Absolute on-disk parent of ``qamomile/SKILL.md``.

    Raises:
        ValueError: If the bundled plugin or marketplace is unavailable.
    """
    return plugin_root() / "skills"


def _walk_error(error: OSError) -> None:
    """Propagate traversal failures instead of accepting an incomplete tree.

    Args:
        error (OSError): Filesystem failure reported by ``os.walk``.

    Raises:
        OSError: Always, preserving the filesystem diagnostic.
    """
    raise error


def _read_tree(root: Path) -> dict[str, bytes]:
    """Read regular files and record directories without following links.

    Args:
        root (Path): Existing skill directory.

    Returns:
        dict[str, bytes]: Relative file contents and empty directory markers.

    Raises:
        ValueError: If a link, junction, or non-regular entry is found.
        OSError: If any entry cannot be inspected or read.
    """
    if _linked(root) or not root.is_dir():
        raise ValueError(f"Skill must be an unlinked directory: {root}")
    content: dict[str, bytes] = {}
    for parent, directories, filenames in os.walk(root, onerror=_walk_error):
        for name in directories + filenames:
            entry = Path(parent) / name
            if _linked(entry):
                raise ValueError(f"Skill contains a symbolic link or junction: {entry}")
        for name in filenames:
            entry = Path(parent) / name
            if not entry.is_file():
                raise ValueError(f"Skill contains a non-regular file: {entry}")
            content[entry.relative_to(root).as_posix()] = entry.read_bytes()
        for name in directories:
            relative = (Path(parent) / name).relative_to(root).as_posix()
            content[f"{relative}/"] = b""
    return content


def read_reference(reference: str) -> str:
    """Read a Markdown file selected from the bundled skill tree.

    Args:
        reference (str): Exact relative path, such as ``SKILL.md`` or
            ``references/authoring.md``.

    Returns:
        str: UTF-8 instruction text.

    Raises:
        ValueError: If the requested path is not a bundled Markdown file.
    """
    content = _read_tree(bundled_root() / NAME)
    if reference not in content or not reference.endswith(".md"):
        choices = ", ".join(sorted(name for name in content if name.endswith(".md")))
        raise ValueError(
            f"Unknown skill reference {reference!r}; choose from: {choices}"
        )
    return content[reference].decode("utf-8")


def _digests(content: dict[str, bytes]) -> dict[str, str]:
    """Hash tree contents while tolerating Markdown checkout line endings.

    Args:
        content (dict[str, bytes]): Skill files and directory markers.

    Returns:
        dict[str, str]: SHA-256 hex digests keyed by relative entry path.
    """
    return {
        name: hashlib.sha256(
            data.replace(b"\r\n", b"\n") if name.endswith(".md") else data
        ).hexdigest()
        for name, data in content.items()
    }


@dataclass(frozen=True)
class _Installation:
    """Record the distribution and tree last copied by this CLI.

    Attributes:
        sdk_version (str): Installed distribution version at copying time.
        digests (dict[str, str]): Digests for every file and directory marker.
    """

    sdk_version: str
    digests: dict[str, str]

    def encode(self) -> str:
        """Serialize the installation record deterministically.

        Returns:
            str: JSON record with a trailing newline.
        """
        return (
            json.dumps(
                {"sdk_version": self.sdk_version, "files": self.digests}, sort_keys=True
            )
            + "\n"
        )


def _decode(raw: bytes) -> _Installation:
    """Validate an installation record before treating files as managed.

    Args:
        raw (bytes): JSON installation record.

    Returns:
        _Installation: Validated version and file digests.

    Raises:
        ValueError: If JSON, encoding, or required fields are invalid.
    """
    record: object = json.loads(raw)
    if not isinstance(record, dict) or set(record) != {"sdk_version", "files"}:
        raise ValueError("Invalid installation record")
    sdk, digests = record["sdk_version"], record["files"]
    if (
        not isinstance(sdk, str)
        or not isinstance(digests, dict)
        or "SKILL.md" not in digests
        or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in digests.items()
        )
    ):
        raise ValueError("Invalid installation record")
    return _Installation(sdk, cast(dict[str, str], digests))


def _inspect(target: Path) -> tuple[str, _Installation | None]:
    """Classify a destination and verify all managed contents.

    Args:
        target (Path): Skill destination or backup directory.

    Returns:
        tuple[str, _Installation | None]: Absent, unmanaged, modified, or
            managed state and the verified record when managed.
    """
    if _linked(target):
        return "unmanaged", None
    if not target.exists():
        return "absent", None
    if (
        not target.is_dir()
        or _linked(target / RECORD)
        or not (target / RECORD).is_file()
    ):
        return "unmanaged", None
    try:
        content = _read_tree(target)
        installed = _decode(content.pop(RECORD))
        if _digests(content) != installed.digests:
            return "modified", None
    except (OSError, ValueError, KeyError):
        return "modified", None
    return "managed", installed


def _destination(project: Path, agent: str) -> Path:
    """Resolve an existing project without traversing linked skill directories.

    Args:
        project (Path): Existing project root, with optional home expansion.
        agent (str): Agent key from ``AGENT_PATHS``.

    Returns:
        Path: Absolute project-local skill destination.

    Raises:
        ValueError: If the project is not a directory or the destination is linked.
        OSError: If the project does not exist or cannot be resolved.
    """
    root = project.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Project must be a directory: {root}")
    target = root
    for part in (*AGENT_PATHS[agent].parts, NAME):
        target /= part
        if _linked(target):
            raise ValueError(
                f"Skill destination contains a symbolic link or junction: {target}"
            )
    return target


def _backup(target: Path, expected: _Installation) -> Path:
    """Preserve a verified copy outside skill discovery, then recheck late edits.

    Args:
        target (Path): Managed directory to move.
        expected (_Installation): Previously verified record.

    Returns:
        Path: Preserved previous directory.

    Raises:
        ValueError: If the directory changed since inspection; it is restored.
        OSError: If moving or restoring the directory fails.
    """
    backup = target.parent.parent / f".qamomile-backup-{uuid4().hex}"
    target.rename(backup)
    if _inspect(backup) != ("managed", expected):
        backup.rename(target)
        raise ValueError(f"Skill changed during operation; restored {target}")
    return backup


def _write(
    target: Path,
    content: dict[str, bytes],
    current: _Installation,
    previous: _Installation | None,
) -> None:
    """Stage a complete copy before replacing a verified installation.

    Args:
        target (Path): Skill destination.
        content (dict[str, bytes]): Bundled files and directory markers.
        current (_Installation): Record for the new copy.
        previous (_Installation | None): Verified record, or None for installation.

    Raises:
        ValueError: If the destination appears or changes during the operation.
        OSError: If copying or renaming fails; the old copy is restored when possible.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".qamomile-stage-", dir=target.parent.parent
    ) as tmp:
        staged = Path(tmp) / NAME
        staged.mkdir()
        for relative, data in content.items():
            entry = staged / relative
            if relative.endswith("/"):
                entry.mkdir(parents=True, exist_ok=True)
            else:
                entry.parent.mkdir(parents=True, exist_ok=True)
                entry.write_bytes(data)
        (staged / RECORD).write_text(current.encode(), encoding="utf-8")
        backup = _backup(target, previous) if previous is not None else None
        try:
            if target.exists() or _linked(target):
                raise ValueError(
                    f"Skill destination appeared during operation: {target}"
                )
            staged.rename(target)
        except (OSError, ValueError):
            if backup is not None:
                if not target.exists() and not _linked(target):
                    backup.rename(target)
                else:
                    print(f"Previous skill preserved at {backup}")
            raise
        if backup is not None:
            print(f"Previous skill preserved at {backup}")


def _bundle() -> tuple[dict[str, bytes], _Installation]:
    """Read the packaged skill and construct its installation record.

    Returns:
        tuple[dict[str, bytes], _Installation]: Content and matching record.

    Raises:
        ValueError: If SKILL.md is absent or an install record is bundled.
    """
    content = _read_tree(bundled_root() / NAME)
    if "SKILL.md" not in content or RECORD in content:
        raise ValueError(
            "Bundled skill must contain SKILL.md and no installation record"
        )
    return content, _Installation(sdk_version(), _digests(content))


def manage(action: str, agent: str, project: Path) -> int:
    """Manage a project copy without overwriting local customizations.

    Args:
        action (str): Install, status, update, or remove.
        agent (str): Agent key from ``AGENT_PATHS``.
        project (Path): Existing project root.

    Returns:
        int: Zero on success; one when status is not current.

    Raises:
        ValueError: If a mutation would overwrite an unmanaged or edited copy.
        OSError: If the destination or package data cannot be accessed.
    """
    target = _destination(project, agent)
    state, installed = _inspect(target)
    if action == "status":
        if installed is None:
            print(f"{target}: {state}")
            return 1
        _, current = _bundle()
        condition = "current" if installed == current else "out of sync"
        print(
            f"{target}: {condition} (skill SDK {installed.sdk_version}; installed SDK {current.sdk_version})"
        )
        return 0 if installed == current else 1
    if action == "install":
        if state != "absent":
            raise ValueError(
                f"Skill destination is {state}; refusing to overwrite {target}. Use status or update."
            )
        content, current = _bundle()
        _write(target, content, current, None)
        print(f"Installed {target} from SDK {current.sdk_version}")
        return 0
    if installed is None:
        raise ValueError(
            f"Skill destination is {state}; refusing to {action}: {target}"
        )
    if action == "remove":
        backup = _backup(target, installed)
        print(f"Removed {target}; previous skill preserved at {backup}")
        return 0
    content, current = _bundle()
    if installed == current:
        print(f"{target}: already current (SDK {current.sdk_version})")
    else:
        _write(target, content, current, installed)
        print(
            f"Updated {target} from SDK {installed.sdk_version} to {current.sdk_version}"
        )
    return 0
