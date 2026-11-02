"""Retain CUDA-Q kernel definitions and their module-local helper owners."""

from __future__ import annotations

from typing import Any

from qamomile._execution_payload import ExecutablePayload
from qamomile.errors import ExecutionError


class CudaqExecutable(ExecutablePayload):
    """Own source for export and one kernel for the currently selected target.

    Args:
        source (str): Generated module containing the entry and helper kernels.
    """

    def __init__(self, source: str) -> None:
        """Define the module once under the adapter's SDK lock.

        Args:
            source (str): Verified circuit or generated source retained by this backend.
        """
        import cudaq

        from ._integration import _CUDAQ_EXECUTION_LOCK, _load_source

        self.source = source
        with _CUDAQ_EXECUTION_LOCK:
            self._target = str(cudaq.get_target())
            self._kernel = _load_source(source)

    def kernel(self) -> Any:
        """Reuse the kernel, replacing it only when the SDK target changes.

        Returns:
            Any: Kernel owning the namespace of all helper definitions.
        """
        import cudaq

        from ._integration import _CUDAQ_EXECUTION_LOCK, _load_source

        with _CUDAQ_EXECUTION_LOCK:
            target = str(cudaq.get_target())
            if target != self._target:
                self._kernel = _load_source(self.source)
                self._target = target
            return self._kernel

    def to_bytes(self) -> bytes:
        """Export the generated source without serializing SDK decorators.

        Returns:
            bytes: Portable backend data for explicit persistence.
        """
        return self.source.encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> CudaqExecutable:
        """Restore one kernel definition from explicitly saved source.

        Args:
            payload (bytes): Portable backend data from an explicit artifact export.

        Returns:
            CudaqExecutable: Restored executable with its SDK lifetime owners.
        """
        try:
            return cls(payload.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ExecutionError("Q8504: invalid CUDA-Q source encoding") from error
