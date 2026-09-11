"""Verify ownership of Selene build artifacts without native compilation."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from unittest.mock import Mock

import pytest

from qamomile.hugr import HugrExecutor, SeleneExecutionOptions

pytestmark = pytest.mark.hugr


@pytest.mark.parametrize("explicit_root", [False, True])
@pytest.mark.parametrize("build_fails", [False, True])
def test_selene_build_directory_ownership(
    tmp_path, monkeypatch, explicit_root, build_fails
) -> None:
    """Only implicit caches are removed on release, including failed builds."""
    selene_sim = pytest.importorskip("selene_sim")
    build_paths = []
    runner = Mock()
    runner.run_shots.return_value = [[("value", 1)]]

    def build(package: bytes, *, name: str, build_dir: Path) -> Mock:
        """Create a representative build artifact before returning or failing.

        Args:
            package (bytes): Serialized package accepted by the build stub.
            name (str): Name assigned to the compiled program.
            build_dir (Path): Directory owned by the executor or user.

        Returns:
            Mock: Runner returning a completed scalar shot.

        Raises:
            RuntimeError: If compilation failure is requested by the test.
        """
        build_paths.append(build_dir)
        build_dir.mkdir(parents=True)
        (build_dir / "artifact").write_bytes(package)
        if build_fails:
            raise RuntimeError("Compilation failed")
        return runner

    monkeypatch.setattr(selene_sim, "build", build)
    user_root = tmp_path / "builds"
    user_root.mkdir()
    sentinel = user_root / "keep.txt"
    sentinel.write_text("User-owned content")
    executor = HugrExecutor(
        options=SeleneExecutionOptions(
            n_qubits=1, build_dir=user_root if explicit_root else None
        )
    )
    reference = weakref.ref(executor)
    package = Mock()
    package.to_bytes.return_value = b"representative package"
    if build_fails:
        with pytest.raises(RuntimeError, match="Compilation failed"):
            executor.submit(package, shots=1)
    else:
        result = executor.submit(package, shots=1)
        assert result.result() == [{"value": 1}]
        assert executor.submit(package, shots=1).result() == [{"value": 1}]
    assert len(build_paths) == 1
    namespace = build_paths[0].parent
    assert namespace.exists()
    assert (build_paths[0] / "artifact").read_bytes() == package.to_bytes()

    del executor
    gc.collect()

    assert reference() is None
    assert namespace.exists() is explicit_root
    assert sentinel.read_text() == "User-owned content"
    if explicit_root:
        assert namespace.parent == user_root
        assert (build_paths[0] / "artifact").read_bytes() == package.to_bytes()
    elif not build_fails:
        assert result.result() == [{"value": 1}]
