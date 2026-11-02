---
slug: integration
---

# Integration

Notes on using Qamomile together with external libraries and quantum platforms.

::::{grid} 1 1 1 1

:::{card}
:header: **Amazon Braket Support**
:link: braket_support
Compile typed measurement feedback to portable OpenQASM 3, run it locally, and keep managed devices at the endpoint boundary.
:::

:::{card}
:header: **CUDA-Q Support through Serialized Kernel Artifacts**
:link: cudaq_support
Compile one typed program to serialized CUDA-Q kernel data, inspect it, and execute it in an isolated `qpp-cpu` worker.
:::

:::{card}
:header: **HUGR Support**
:link: hugr_support
Compile a Qamomile program to HUGR and run it locally with Selene.
:::

:::{card}
:header: **OMMX Quantum Benchmarks with Qamomile**
:link: ommx_quantum_benchmarks_qaoa
Normalize a nine-qubit LABS instance immutably, run typed QAOA once, and compare it with the bundled reference solution.
:::

:::{card}
:header: **qBraid Provider Execution**
:link: qbraid_executor
Compile typed Qamomile programs to deterministic QPY and submit them through a live qBraid endpoint.
:::

:::{card}
:header: **Qiskit Support**
:link: qiskit_support
Compile one typed program to portable QPY, reuse it on Aer, and route the same source to a live BackendV2.
:::

:::{card}
:header: **QURI Parts Support**
:link: quri_parts_support
Compile one typed program to a portable logical circuit, inspect it, and reuse it for Qulacs sampling and expectation evaluation.
:::

::::
