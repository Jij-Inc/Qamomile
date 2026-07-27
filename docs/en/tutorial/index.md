---
slug: tutorial
---

# Tutorials

Step-by-step guides to learn Qamomile.

::::{grid} 1 1 1 1

:::{card}
:header: **1. Your First Quantum Kernel**
:link: 01_your_first_quantum_kernel
Define, visualize, and execute a kernel; the affine rule.
:::

:::{card}
:header: **2. Parameterized Kernels**
:link: 02_parameterized_kernels
Structure vs runtime parameters, bind/sweep pattern.
:::

:::{card}
:header: **3. Vector Slicing**
:link: 03_vector_slicing
`VectorView`, slice assignment, nested slices, and passing views to helper kernels.
:::

:::{card}
:header: **4. Controlled Gates**
:link: 04_controlled_gates
`qmc.control` for built-in gates and sub-kernels, concrete vs symbolic control counts, error catalogue.
:::

:::{card}
:header: **5. Resource Estimation**
:link: 05_resource_estimation
Symbolic cost analysis, gate breakdown, scaling.
:::

:::{card}
:header: **6. Execution Models**
:link: 06_execution_models
`sample()` vs `run()`, observables, bit ordering.
:::

:::{card}
:header: **7. Classical Flow Patterns**
:link: 07_classical_flow_patterns
Loops, sparse data, conditional branching.
:::

:::{card}
:header: **8. Reuse Patterns**
:link: 08_reuse_patterns
Helper kernels, composite gates, stubs.
:::

:::{card}
:header: **9. Hermitian Matrix Decomposition**
:link: 09_hermitian_decomposition
From a dense Hermitian matrix to a Pauli sum and a time-evolution circuit.
:::

:::{card}
:header: **10. Compilation and Transpilation**
:link: 10_compilation_and_transpilation
The 10-stage pipeline, IR walkthrough, backend emission.
:::

::::

Ready for concrete algorithms? Head over to [Algorithms](../algorithm/index.md) — browse by tag to jump to QAOA, Hamiltonian simulation, and more.
