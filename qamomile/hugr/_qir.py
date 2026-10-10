"""Convert closed HUGR execution wrappers into validated H2 QIR bitcode."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from qamomile.circuit.transpiler import ExecutionError


def to_h2_qir(package: Any) -> tuple[bytes, dict[str, tuple[str, int]]]:
    """Convert an instrumented package without changing its public output tags.

    This adapter records Boolean results. Floating-point calculations inside
    the program remain available. Integer output decoding requires a separate
    verified provider contract before it can preserve unsigned values safely.

    Args:
        package (Any): HUGR package with a closed ``main`` and tagged outputs.

    Returns:
        tuple[bytes, dict[str, tuple[str, int]]]: QIR bitcode and each output
            label's scalar kind and recorded bit width.

    Raises:
        ImportError: If Python is unsupported or the converter is unavailable.
        ExecutionError: If output types or HUGR operations cannot target H2,
            or conversion does not produce validated bitcode.
    """
    runtime_python = sys.version_info[:2]
    if runtime_python >= (3, 13):
        raise ImportError(
            "H2 execution currently requires Python 3.11 or 3.12 and "
            "qamomile[hugr,hugr-qir]"
        )
    outputs: dict[str, tuple[str, int]] = {}
    for graph in package.modules:
        for _, data in graph.nodes():
            name = getattr(data.op, "name", None)
            operation = name().partition("<")[0] if callable(name) else ""
            if not operation.startswith("tket.result."):
                continue
            if operation == "tket.result.result_f64":
                raise ExecutionError(
                    "H2 does not support Float outputs; return Bit values"
                )
            if operation == "tket.result.result_uint":
                raise ExecutionError(
                    "H2 UInt output decoding is not supported yet; return Bit values"
                )
            kinds = {
                "tket.result.result_bool": ("bool", 1),
            }
            if operation not in kinds:
                raise ExecutionError(
                    f"H2 does not support output operation {operation}"
                )
            args: Any = getattr(data.op, "args", ())
            tag = getattr(args[0], "value", None) if args else None
            if not isinstance(tag, str) or not tag or tag in outputs:
                raise ExecutionError("H2 requires unique, non-empty output labels")
            outputs[tag] = kinds[operation]
    try:
        converter = importlib.import_module("hugr_qir.hugr_to_qir")
        output = importlib.import_module("hugr_qir.output")
    except ImportError as exc:
        raise ImportError(
            "H2 execution requires hugr-qir; install qamomile[hugr,hugr-qir]"
        ) from exc
    try:
        bitcode = converter.hugr_to_qir(
            package,
            output_format=output.OutputFormat.BITCODE,
            validate_qir=True,
            validate_hugr=True,
        )
    except Exception as exc:
        raise ExecutionError(f"H2 QIR conversion failed: {exc}") from exc
    if not isinstance(bitcode, bytes) or not bitcode.startswith(b"BC\xc0\xde"):
        raise ExecutionError("H2 QIR conversion did not produce LLVM bitcode")
    return bitcode, outputs
