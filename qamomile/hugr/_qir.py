"""Convert closed HUGR execution wrappers into validated H2 QIR bitcode."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

from qamomile.circuit.transpiler import ExecutionError

if TYPE_CHECKING:
    from hugr.package import Package


class _H2AbortError(ExecutionError):
    """Identify an abort call rejected by the H2 converter's validator."""


def to_h2_qir(package: Package) -> tuple[bytes, dict[str, tuple[str, int]]]:
    """Convert an instrumented package without changing its public output tags.

    This adapter records Boolean results. H2 permits constant floating-point
    gate arguments but only integer runtime computations. Integer output
    decoding requires a separate verified provider contract before it can
    preserve unsigned values safely.

    Args:
        package (Package): HUGR package with a closed ``main`` and tagged outputs.

    Returns:
        tuple[bytes, dict[str, tuple[str, int]]]: QIR bitcode and each output
            label's scalar kind and logical bit width, independently of the
            provider's register storage width.

    Raises:
        ImportError: If Python is unsupported or the converter is unavailable.
        ExecutionError: If output types or HUGR operations cannot target H2,
            or conversion does not produce validated bitcode.
    """
    runtime_python = sys.version_info[:2]
    if runtime_python >= (3, 13):
        raise ImportError(
            "H2 execution currently requires Python 3.11 or 3.12 and qamomile[hugr]"
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
            "H2 execution requires hugr-qir; install qamomile[hugr]"
        ) from exc
    try:
        bitcode = converter.hugr_to_qir(
            package,
            output_format=output.OutputFormat.BITCODE,
            validate_qir=True,
            validate_hugr=True,
        )
    except Exception as exc:
        if "@abort()" in str(exc):
            raise _H2AbortError(
                f"H2 QIR conversion failed: {exc}"
                " The installed hugr-qir converter lowers runtime bounds failure "
                "to abort(), which H2 QIR validation rejects. For fixed UInt or Bit "
                "tables indexed by get_current_shot(), use the H2 executor's "
                "automatic shot-range preparation before submission."
            ) from exc
        raise ExecutionError(f"H2 QIR conversion failed: {exc}") from exc
    if not isinstance(bitcode, bytes) or not bitcode.startswith(b"BC\xc0\xde"):
        raise ExecutionError("H2 QIR conversion did not produce LLVM bitcode")
    _validate_h2_classical_instructions(bitcode)
    return bitcode, outputs


def _validate_h2_classical_instructions(bitcode: bytes) -> None:
    """Reject operations accepted by the shared validator but unavailable on H2.

    The pinned validator covers both platform families and accepts floating
    computations and integer division/remainder that require Helios. Inspect
    optimized QIR so constant binding and constant angle folding remain valid.

    Args:
        bitcode (bytes): LLVM bitcode already accepted by the shared validator.

    Raises:
        ExecutionError: If QIR contains runtime Float values, integer division
            or remainder, or invalid bitcode.
    """
    import pyqir

    try:
        module = pyqir.Module.from_bitcode(pyqir.Context(), bitcode)
    except Exception as exc:
        raise ExecutionError(f"H2 QIR bitcode could not be inspected: {exc}") from exc
    unsupported = {
        pyqir.Opcode.UDIV,
        pyqir.Opcode.SDIV,
        pyqir.Opcode.UREM,
        pyqir.Opcode.SREM,
    }
    for function in module.functions:
        for block in function.basic_blocks:
            for instruction in block.instructions:
                if instruction.opcode in unsupported:
                    raise ExecutionError(
                        "H2 does not support runtime integer division or remainder"
                    )
                if instruction.type.is_double or any(
                    operand.type.is_double and not isinstance(operand, pyqir.Constant)
                    for operand in instruction.operands
                ):
                    raise ExecutionError(
                        "H2 does not support runtime Float computations or selected "
                        "Float gate arguments; use integer predicates to select "
                        "gates with constant angles, or target Helios"
                    )
