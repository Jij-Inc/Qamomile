"""Retain HUGR packages and lifetime-managed local Selene build products."""

from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class HugrExecutable(ExecutablePayload):
    """Own one package and lazily build at most one local Selene runner.

    Args:
        package (Any): Validated HUGR package.
    """

    def __init__(self, package: Any) -> None:
        """Retain the package and initialize synchronized build ownership.

        Args:
            package (Any): Validated HUGR package retained for execution.
        """
        self.package = package
        self.lock = RLock()
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._runner: Any = None

    def runner(self, selene: Any) -> Any:
        """Build once and keep the executable directory alive with the runner.

        Args:
            selene (Any): Loaded Selene SDK module.

        Returns:
            Any: Runner reused by launches, each creating fresh quantum state.
        """
        with self.lock:
            if self._runner is None:
                directory = tempfile.TemporaryDirectory(prefix="qamomile-selene-")
                try:
                    runner = selene.build(
                        self.package,
                        name="qamomile_program",
                        build_dir=Path(directory.name),
                    )
                except BaseException:
                    directory.cleanup()
                    raise
                self._directory = directory
                self._runner = runner
            return self._runner

    def to_bytes(self) -> bytes:
        """Serialize the package for explicit persistence or external transfer.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        return self.package.to_bytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> HugrExecutable:
        """Restore a saved package without building a local runner yet.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            HugrExecutable: Restored executable with its SDK lifetime owners.
        """
        from hugr.package import Package

        try:
            return cls(Package.from_bytes(payload))
        except Exception as error:
            raise ExecutionError("Q8605: invalid serialized HUGR package") from error
