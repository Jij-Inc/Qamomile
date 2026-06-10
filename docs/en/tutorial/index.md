---
slug: tutorial
---

# Tutorials

Step-by-step guides to learn Qamomile.

::::{grid} 1 2 2 2

:::{card}
:header: **Your First Quantum Kernel**
:link: 01_your_first_quantum_kernel
Define, visualize, and execute a kernel; the affine rule.
:::

:::{card}
:header: **Parameterized Kernels**
:link: 02_parameterized_kernels
Structure vs runtime parameters, bind/sweep pattern.
:::

:::{card}
:header: **Vector Slicing**
:link: 03_vector_slicing
`VectorView`, slice assignment, nested slices, and passing views to helper kernels.
:::

:::{card}
:header: **Controlled Gates**
:link: 04_controlled_gates
`qmc.control` for built-in gates and sub-kernels, concrete vs symbolic control counts, error catalogue.
:::

:::{card}
:header: **Resource Estimation**
:link: 05_resource_estimation
Symbolic cost analysis, gate breakdown, scaling.
:::

:::{card}
:header: **Execution Models**
:link: 06_execution_models
`sample()` vs `run()`, observables, bit ordering.
:::

:::{card}
:header: **Classical Flow Patterns**
:link: 07_classical_flow_patterns
Loops, sparse data, conditional branching.
:::

:::{card}
:header: **Reuse Patterns**
:link: 08_reuse_patterns
Helper kernels, composite gates, stubs.
:::

:::{card}
:header: **Hermitian Matrix Decomposition**
:link: 09_hermitian_decomposition
From a dense Hermitian matrix to a Pauli sum and a time-evolution circuit.
:::

:::{card}
:header: **Compilation and Transpilation**
:link: 10_compilation_and_transpilation
The 10-stage pipeline, IR walkthrough, backend emission.
:::

::::

Ready for concrete algorithms? Head over to [Algorithms](../algorithm/index.md) — browse by tag to jump to QAOA, Hamiltonian simulation, and more.
