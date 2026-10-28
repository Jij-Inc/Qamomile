"""Exercise CI routing through real Git diffs and the production classifier."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = PROJECT_ROOT / ".github/scripts/classify-changes.sh"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required.")


def _git(repository: Path, *arguments: str) -> str:
    """Run Git without relying on user-specific commit configuration.

    Args:
        repository (Path): Temporary repository working directory.
        *arguments (str): Git command and arguments.

    Returns:
        str: Stripped command output.
    """
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=CI routing test",
            "-c",
            "user.email=ci-routing@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *arguments,
        ],
        cwd=repository,
        text=True,
    ).strip()


def _write(repository: Path, relative_path: str, content: str) -> None:
    """Write a test file and create its parent directories.

    Args:
        repository (Path): Temporary repository root.
        relative_path (str): Repository-relative destination.
        content (str): File contents.
    """
    destination = repository / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)


def _commit(repository: Path) -> str:
    """Commit all test changes and return the resulting revision.

    Args:
        repository (Path): Temporary repository root.

    Returns:
        str: Full commit identifier.
    """
    _git(repository, "add", "--all")
    _git(repository, "commit", "-qm", "Update routing fixture")
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an isolated Git repository with the current skip fragments.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        Path: Initialized repository containing the classifier's skip inputs.
    """
    _git(tmp_path, "init", "-q")
    for fragment in (PROJECT_ROOT / ".github/ci-skip-paths").glob("*.txt"):
        _write(
            tmp_path,
            f".github/ci-skip-paths/{fragment.name}",
            fragment.read_text(),
        )
    return tmp_path


def _classify(repository: Path, base: str, head: str) -> dict[str, str]:
    """Run the real classifier and read every workflow output.

    Args:
        repository (Path): Temporary repository containing the compared commits.
        base (str): Trusted base revision.
        head (str): Revision under review.

    Returns:
        dict[str, str]: Classification outputs written for GitHub Actions.
    """
    output = repository / "classification-output"
    subprocess.run(
        ["bash", str(CLASSIFIER), base, head, "merge-base"],
        cwd=repository,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize("operation", ["add", "modify", "delete"])
@pytest.mark.parametrize(
    ("path", "full", "docs"),
    [
        ("README.md", "false", "false"),
        ("LICENSE.txt", "false", "false"),
        (".gitignore", "false", "false"),
        (".readthedocs.yaml", "false", "true"),
        ("qamomile/example.py", "true", "false"),
        ("tests/test_example.py", "true", "false"),
        ("docs/en/tutorial/example.py", "false", "true"),
        ("unknown-input.txt", "true", "false"),
        ("LIMITATIONS.md", "false", "false"),
    ],
)
def test_path_classification(repository, operation, path, full, docs):
    """Skip repository metadata and preserve routing for checked path families."""
    if operation != "add":
        _write(repository, path, "original\n")
    base = _commit(repository)
    if operation == "delete":
        (repository / path).unlink()
    else:
        _write(repository, path, "changed\n")
    head = _commit(repository)

    assert _classify(repository, base, head) == {
        "full_ci_required": full,
        "docs_changed": docs,
        "classification_complete": "true",
    }


@pytest.mark.parametrize(
    ("old_path", "new_path", "full", "docs"),
    [
        ("README.md", "docs/en/package-readme.md", "false", "true"),
        ("docs/en/example.md", "unknown-output.md", "true", "true"),
        ("qamomile/example.py", "README.md", "true", "false"),
        ("LICENSE.txt", "unknown-output.txt", "true", "false"),
    ],
)
def test_renames_check_both_path_families(repository, old_path, new_path, full, docs):
    """A move must preserve checks required by both the old and new paths."""
    _write(repository, old_path, "unchanged contents\n")
    base = _commit(repository)
    (repository / new_path).parent.mkdir(parents=True, exist_ok=True)
    _git(repository, "mv", old_path, new_path)
    head = _commit(repository)

    assert _classify(repository, base, head) == {
        "full_ci_required": full,
        "docs_changed": docs,
        "classification_complete": "true",
    }


@pytest.mark.parametrize(
    ("path", "full", "docs"),
    [
        ("README.md", "false", "false"),
        ("LICENSE.txt", "false", "false"),
        (".gitignore", "false", "false"),
        (".readthedocs.yaml", "false", "true"),
        (".github/ci-skip-paths/test-policy.txt", "true", "false"),
    ],
)
def test_required_paths_take_precedence_over_skip_patterns(
    repository, path, full, docs
):
    """Even a broad trusted skip cannot suppress required checks or its edits."""
    _write(repository, ".github/ci-skip-paths/test-policy.txt", "*\n")
    base = _commit(repository)
    _write(repository, path, "changed\n")
    head = _commit(repository)

    assert _classify(repository, base, head) == {
        "full_ci_required": full,
        "docs_changed": docs,
        "classification_complete": "true",
    }


def test_skip_policy_is_loaded_from_the_trusted_base(repository):
    """Uncommitted skip rules cannot suppress an unknown committed change."""
    base = _commit(repository)
    _write(repository, "unknown-input.txt", "changed\n")
    head = _commit(repository)
    _write(repository, ".github/ci-skip-paths/untrusted.txt", "*\n")

    assert _classify(repository, base, head) == {
        "full_ci_required": "true",
        "docs_changed": "false",
        "classification_complete": "true",
    }


@pytest.mark.parametrize("metadata_path", ["README.md", "LICENSE.txt", ".gitignore"])
@pytest.mark.parametrize(
    ("checked_path", "full", "docs"),
    [
        ("qamomile/example.py", "true", "false"),
        ("tests/test_example.py", "true", "false"),
        ("docs/en/example.md", "false", "true"),
        (".github/ci-skip-paths/test-policy.txt", "true", "false"),
        ("unknown-input.txt", "true", "false"),
    ],
)
def test_metadata_changes_do_not_suppress_other_checks(
    repository, metadata_path, checked_path, full, docs
):
    """Keep required checks when skipped metadata and checked paths change together."""
    base = _commit(repository)
    _write(repository, metadata_path, "changed\n")
    _write(repository, checked_path, "changed\n")
    head = _commit(repository)

    assert _classify(repository, base, head) == {
        "full_ci_required": full,
        "docs_changed": docs,
        "classification_complete": "true",
    }
