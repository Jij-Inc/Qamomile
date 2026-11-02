"""Export compiler-projected circuits as QIR 1.0 Base Profile programs.

Rust owns specialization, circuit lowering, resource facts and result ABIs.
This optional PyQIR adapter emits LLVM bitcode and text for static execution
entries with terminal measurements. Outputs are flat bit arrays in the
compiler's logical source order; the artifact retains their typed result ABI.
Execution belongs to the external QIR consumer.
"""

from ._integration import (
    FORMAT as FORMAT,
    QIR as QIR,
    load_bitcode as load_bitcode,
    load_ir as load_ir,
)

__all__ = ["FORMAT", "QIR", "load_bitcode", "load_ir"]
