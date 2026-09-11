---
slug: integration
---

# Integration

Notes on using Qamomile together with external libraries and quantum platforms.

::::{grid} 1 1 1 1

:::{card}
:header: **Amazon Braket Support**
:link: braket_support
Transpile native Braket circuits and run local or AWS-backed execution.
:::

:::{card}
:header: **CUDA-Q Support**
:link: cudaq_support
Transpile MaxCut QAOA to CUDA-Q and run sampling and expectation-value workflows.
:::

:::{card}
:header: **Using OMMX Quantum Benchmarks: Implementing and Benchmarking Quantum Algorithms with Qamomile**
:link: ommx_quantum_benchmarks_qaoa
Drive QAOA on a LABS instance loaded from the OMMX Quantum Benchmarks dataset and compare against SCIP.
:::

:::{card}
:header: **qBraid Support**
:link: qbraid_executor
Run Qiskit circuits on qBraid-supported devices.
:::

:::{card}
:header: **Qiskit Support**
:link: qiskit_support
Transpile to Qiskit, run local simulators, and inspect native Qiskit circuit features.
:::

:::{card}
:header: **QURI Parts Support**
:link: quri_parts_support
Transpile to QURI Parts and run on a Qulacs state-vector simulator.
:::

::::

## HUGR: measuring quantum integers

With the `hugr` extra installed, `HugrTranspiler` supports `qmc.measure(qmc.cast(register, qmc.QInt))` for registers with a concrete width from 0 through 64 bits. QInt registers can also be arguments and results of direct calls between quantum kernels. Use `HugrTranspiler().transpile(kernel, bindings=...)` to obtain a `HugrExecutable` with `run()` and `sample()`, then execute locally with `HugrExecutor(target="selene")`. Use `compile()` when you need the HUGR graph and its input/output description as a `CompiledProgram`.

Measurement consumes the register and returns `UInt`. Bit 0 is the least significant bit, so the decoded value is the sum of `bit[i] * 2**i`; for a slice, bit 0 is the first qubit in the slice. `run()` exposes the value as a Python `int`, and `sample()` counts the decoded integers. Supported tuple and dictionary returns retain that integer type. An empty register yields `0`. Decoding uses integer operations inside the HUGR graph and preserves all 64 bits, including bit 63 and `2**64 - 1`, without conversion through `Float`.

Supply compile-time `bindings` for arguments that determine register widths. Programs may be serialized before supplying those bindings: symbolic widths remain connected through direct calls and are resolved when the restored program is compiled.

:::{note}
Both `compile()` and `transpile()` raise `EmitError` for unresolved QInt widths or widths above 64 bits because the current HUGR `UInt` representation is 64 bits. An unresolved width is distinct from a known empty register. This limit belongs to HUGR; general QInt decoding and other engines do not acquire it. Compiling a 64-qubit graph does not imply that a local state-vector simulator can execute it within available memory.

Merging QInt registers across runtime conditional branches is currently unsupported and raises `EmitError`. Compile-time branch selection resolved through `bindings` remains supported.

Passing a QInt target to `control()` or `select()` is unsupported and raises `TypeError` before consuming the input handles. Use a direct quantum-kernel call for QInt arguments.
:::
