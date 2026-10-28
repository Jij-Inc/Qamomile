"""Exercise CI routing against actual commit diffs and trusted base policies."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/classify-changes.sh"


def _git(repo: Path, *args: str) -> str:
    """Run Git in the isolated repository and return its standard output.

    Args:
        repo (Path): Isolated Git repository.
        *args (str): Git subcommand and arguments.

    Returns:
        str: Standard output with surrounding whitespace removed.

    Raises:
        subprocess.CalledProcessError: If the Git command fails.
    """
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, path: str, content: str) -> None:
    """Write one tracked fixture file, creating its parent directories.

    Args:
        repo (Path): Isolated Git repository.
        path (str): Repository-relative fixture path.
        content (str): Text to write to the fixture.
    """
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def _commit(repo: Path) -> str:
    """Commit fixture changes and return their full object ID.

    Args:
        repo (Path): Isolated Git repository containing fixture changes.

    Returns:
        str: Full commit object ID.
    """
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Update fixture")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create an isolated repository whose initial policy skips all paths.

    Args:
        tmp_path (Path): Temporary directory supplied by pytest.

    Returns:
        Path: Initialized repository with a committed skip policy.
    """
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "CI Test")
    _git(repo, "config", "user.email", "ci-test@example.invalid")
    _write(repo, ".github/ci-skip-paths/fixture.txt", "*\n")
    _commit(repo)
    return repo


def _classify(repo: Path, base: str, head: str, tmp_path: Path) -> dict[str, str]:
    """Run the real classifier and read its GitHub Actions outputs.

    Args:
        repo (Path): Isolated Git repository whose changes are classified.
        base (str): Trusted base commit object ID.
        head (str): Candidate head commit object ID.
        tmp_path (Path): Temporary directory for the Actions output file.

    Returns:
        dict[str, str]: Classification flags and completion marker.

    Raises:
        subprocess.CalledProcessError: If classification fails.
    """
    output = tmp_path / "github-output"
    subprocess.run(
        ["bash", str(SCRIPT), base, head, "merge-base"],
        cwd=repo,
        check=True,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize(
    ("path", "full", "docs"),
    [
        ("README.md", "false", "false"),
        ("LICENSE.txt", "false", "false"),
        (".gitignore", "false", "false"),
        (".readthedocs.yaml", "false", "true"),
        ("qamomile/example.ipynb", "true", "false"),
        ("docs/example.md", "false", "true"),
        (".github/ci-skip-paths/other.txt", "true", "false"),
        ("unconsumed.txt", "false", "false"),
    ],
)
def test_explicit_routing_precedes_skip_policy(
    repository: Path, tmp_path: Path, path: str, full: str, docs: str
) -> None:
    """Skip metadata while preserving required checks despite a broad skip rule."""
    base = _git(repository, "rev-parse", "HEAD")
    _write(repository, path, "changed\n")
    head = _commit(repository)

    assert _classify(repository, base, head, tmp_path) == {
        "full_ci_required": full,
        "docs_changed": docs,
        "classification_complete": "true",
    }


def test_skip_policy_comes_from_trusted_base(repository: Path, tmp_path: Path) -> None:
    """Use base policy even when the pull-request branch retains a broader one."""
    ancestor = _git(repository, "rev-parse", "HEAD")
    _write(repository, ".github/ci-skip-paths/fixture.txt", "# No skipped paths\n")
    base = _commit(repository)
    _git(repository, "checkout", "--detach", ancestor)
    _write(repository, "unclassified.txt", "changed\n")
    head = _commit(repository)

    assert _classify(repository, base, head, tmp_path) == {
        "full_ci_required": "true",
        "docs_changed": "false",
        "classification_complete": "true",
    }
