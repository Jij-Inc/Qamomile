# Export a QIR program

Install `qamomile[qir]` to compile a static execution entry to QIR 1.0 Base
Profile. Save this example in a `.py` file:

```python
from pathlib import Path

import qamomile as qm
from qamomile.qir import QIR, load_bitcode, load_ir


@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


artifact = qm.compile(bell, to=QIR())
Path("bell.ll").write_text(load_ir(artifact), encoding="utf-8")
Path("bell.bc").write_bytes(load_bitcode(artifact))
```

The `.ll` file is human-readable LLVM IR, and `.bc` is standalone LLVM bitcode
for an external QIR consumer. Both contain the entry point `main`, the Base
Profile attributes, static qubit/result counts, and output recording calls.
The external consumer must support the QIS functions declared by the module.
Qamomile provides an export target; execute the file in the selected QIR runtime.

Specialize `Const` inputs with `qm.compile(kernel, to=QIR(), width=3)` or
`qm.specialize` before export. Runtime `Param` inputs, reset, discard,
measurement-dependent control, loops retained at runtime, gates following
measurement, and computed classical results are outside this profile and fail
at compilation. Results must be direct measurement bits or `Unit`. Keep all
state preparation and gates before any measurements.

Supported primitives include H, Pauli, S/T and adjoints, RX/RY/RZ, phase, CNOT,
CZ, Toffoli, controlled phase, SWAP, RZZ, and terminal Z measurements. Unary
gates accept one coherent control; Pauli gates accept up to two. Control on
global phase becomes a relative phase on the control and is retained. Overall
global phase is omitted for these classical-output entries.

Output is one flat bit array in the compiler's logical source order, including
reordered or repeated returned bits. Private measurements are not recorded.
Nested or numeric measurement results use the same native result ABI retained
in `artifact.result_abi`; standalone consumers receive the flat bit array.
Module construction is bounded to one million emitted LLVM instructions,
including measurement and output calls.

Use `qamomile.persistence.serialize` and `deserialize` to save and restore the
complete Qamomile artifact, including its typed ABI and resource study. Use
`load_bitcode` or `qamomile.persistence.export_payload` for the standalone module.
