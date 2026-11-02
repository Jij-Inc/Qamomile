---
slug: algorithm
---

# Algorithms

Quantum algorithms implemented with Qamomile, with checks for their results.

::::{grid} 1 1 1 1

:::{card}
:header: **Grover Adaptive Search**
:link: grover_adaptive_search
Minimize a binary portfolio objective with `BinaryModel`, a Grover oracle, and a classical threshold-update loop.
:::

:::{card}
:header: **Hamiltonian Simulation**
:link: hamiltonian_simulation
Simulate Rabi oscillations with a Suzuki product formula and compare the result with the analytic solution.
:::

:::{card}
:header: **Hybrid Quantum Neural Network**
:link: hybrid_qnn
Combine a quantum circuit with PyTorch and train it using parameter-shift gradients.
:::

:::{card}
:header: **Möttönen Amplitude Encoding**
:link: mottonen_amplitude_encoding
Prepare a quantum state from classical amplitudes and verify its measurement probabilities.
:::

:::{card}
:header: **Option Pricing with Quantum Amplitude Estimation**
:link: option_pricing
Encode an asset distribution and payoff, estimate the expected payoff, and calculate a discounted option price.
:::

:::{card}
:header: **Multidimensional Quantum Fourier Transform**
:link: multidimensional_qft
Apply a reusable QFT to each logical axis of a quantum register.
:::

:::{card}
:header: **Pauli Correlation Encoding**
:link: pce_maxcut
Encode a four-variable MaxCut problem with two qubits and recover a partition from estimated Pauli correlations.
:::

:::{card}
:header: **QAOA for Constrained Graph Partitioning**
:link: qaoa_graph_partition
Formulate a constrained problem with JijModeling and OMMX, run QAOA, and check feasibility and objective values.
:::

:::{card}
:header: **QAOA for MaxCut**
:link: qaoa_maxcut
Build QAOA from reusable functions and decode a cut of a four-vertex cycle.
:::

:::{card}
:header: **QSVT Eigenstate Filtering for Graph Partitioning**
:link: qsvt_eigenstate_filter_graph_partitioning
Build a real-polynomial eigenstate filter, estimate its success probability, and recover a balanced graph partition.
:::

:::{card}
:header: **Quantum-Enhanced Markov Chain Monte Carlo**
:link: qe_mcmc
Use a quantum time-evolution circuit to propose classical samples and verify detailed balance.
:::

:::{card}
:header: **Quantum Selected Configuration Interaction**
:link: qsci
Sample trial states, construct their subspace Hamiltonian, and solve for a variational energy.
:::

:::{card}
:header: **Quantum Singular Value Transformation (QSVT)**
:link: qsvt
Block encode a non-Hermitian matrix, apply a degree-two singular-value polynomial, verify the projected matrix, and estimate logical resources.
:::

:::{card}
:header: **Introduction to Quantum Error Correction**
:link: quantum_error_correction
Encode a logical state, detect a single bit flip, and correct it while preserving the logical amplitudes.
:::

:::{card}
:header: **Quantum Phase Estimation (QPE)**
:link: qpe
Implement quantum phase estimation from individual operations and with `phase_estimation` from `qamomile.stdlib`, then compare precision and logical gate counts.
:::

:::{card}
:header: **Quantum Kernel Classification**
:link: quantum_kernel_classification
Estimate a quantum overlap kernel and use it with a support-vector classifier.
:::

:::{card}
:header: **Stabilizers and the Steane Code**
:link: steane_code
Construct the Steane code and use its syndrome to correct a Pauli error.
:::

:::{card}
:header: **VQE for the Hydrogen Molecule**
:link: vqe_for_hydrogen
Build a molecular Hamiltonian and optimize a two-qubit trial state to estimate its ground-state energy.
:::

::::
