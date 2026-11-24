"""Load HUGR packages and normalize source-ordered named result records."""

from __future__ import annotations

from collections.abc import Iterable
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


def _decode_result_row(
    records: Iterable[tuple[str, object]],
    *,
    width: int,
) -> tuple[int, ...]:
    """Normalize one SDK result row through its exact source-position tags.

    Args:
        records (Iterable[tuple[str, object]]): SDK-tagged Boolean result values.
        width (int): Logical width from the native-derived result ABI.

    Returns:
        tuple[int, ...]: Bit row in source result order.

    Raises:
        ValueError: If a logical tag is missing, repeated, unexpected, or non-bit.
    """

    positions = {f"qamomile_{index}": index for index in range(width)}
    bits: list[int | None] = [None] * width
    for tag, value in records:
        if not isinstance(tag, str):
            raise ValueError("logical result tag must be a string")
        if tag not in positions:
            if tag.startswith("qamomile"):
                raise ValueError("unexpected logical result tag")
            continue
        index = positions[tag]
        if bits[index] is not None:
            raise ValueError("repeated logical result tag")
        if not isinstance(value, (bool, int)) or int(value) not in {0, 1}:
            raise ValueError("logical result contains a non-bit value")
        bits[index] = int(value)
    if any(bit is None for bit in bits):
        raise ValueError("logical result width disagrees with ResultABI")
    return tuple(bit for bit in bits if bit is not None)
