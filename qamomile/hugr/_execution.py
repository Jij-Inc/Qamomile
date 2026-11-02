"""Load sealed HUGR package bytes for supported execution endpoints."""

from __future__ import annotations

from typing import Any

from qamomile.compilation import CompiledProgram
from qamomile.errors import ExecutionError

from ._profile import FORMAT


def load_package(artifact: CompiledProgram) -> Any:
    """Materialize a fresh HUGR package from an inspected artifact.

    Args:
        artifact (CompiledProgram): Inspected target artifact carrying HUGR bytes.

    Returns:
        Any: Newly decoded ``hugr.package.Package``.

    Raises:
        ExecutionError: If the format or serialized package is invalid.
    """

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8604: expected {FORMAT!r}, got {artifact.format!r}")
    from copy import deepcopy

    from ._payload import HugrExecutable

    return deepcopy(artifact.execution_payload(HugrExecutable).package)
