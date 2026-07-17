---
slug: algorithm
---

# Algorithms

Concrete quantum algorithm examples built with Qamomile.

::::{grid} 1 1 1 1

:::{card}
:header: **Hamiltonian Simulation with Suzuki–Trotter (Rabi oscillation)**
:link: hamiltonian_simulation
Trotter–Suzuki product formulas on the Rabi model with empirical convergence orders.
:::

:::{card}
:header: **Introduction to Grover search**
:link: grover_search
Use a five-qubit phase-oracle example to compare search-state probabilities before and after amplitude amplification.
:::

:::{card}
:header: **Hybrid Quantum Neural Network (HQNN)**
:link: hybrid_qnn
End-to-end CNN + quantum variational circuit on Fashion-MNIST with the parameter shift rule.
:::

:::{card}
:header: **Möttönen Amplitude Encoding**
:link: mottonen_amplitude_encoding
Prepare an arbitrary real or complex amplitude vector via Gray-code Ry/Rz multiplexers, with three input modes (concrete, bound `Vector[Float]`, runtime-parametric angles).
:::

:::{card}
:header: **Multidimensional Quantum Fourier Transform**
:link: multidimensional_qft
Implement multidimensional QFT and classical preprocessing for inputs whose grid sizes are not powers of two.
:::

:::{card}
:header: **Pauli Correlation Encoding (PCE)**
:link: pce_maxcut
Solve a 20-variable MaxCut on just 3 qubits with Pauli Correlation Encoding, training a hardware-efficient ansatz against a tanh-relaxed surrogate and decoding via sign rounding.
:::

:::{card}
:header: **QAOA for Graph Partitioning**
:link: qaoa_graph_partition
End-to-end optimization example using OMMX, JijModeling, and `QAOAConverter`.
:::

:::{card}
:header: **QAOA for MaxCut: Building the Circuit from Scratch**
:link: qaoa_maxcut
Build a QAOA circuit from scratch to solve MaxCut, then compare with the built-in `qaoa_state`.
:::

:::{card}
:header: **Quantum-enhanced Markov chain Monte Carlo**
:link: qe_mcmc
Implement Quantum-enhanced MCMC using Trotter-decomposed time evolution.
:::

:::{card}
:header: **Quantum Selected Configuration Interaction (QSCI)**
:link: qsci
Sample bitstrings from a quantum state, build an effective Hamiltonian, and diagonalize it classically with a strict variational guarantee.
:::

:::{card}
:header: **Introduction to Quantum Error Correction**
:link: quantum_error_correction
3-qubit bit-flip/phase-flip codes, Shor's 9-qubit code, and stabilizers.
:::

:::{card}
:header: **Quantum Kernel Classification**
:link: quantum_kernel_classification
Use quantum feature maps and kernel methods for classification on the make_circles dataset.
:::

:::{card}
:header: **Stabilizer Formalism and the Steane Code**
:link: steane_code
CSS construction, syndrome decoding, and transversal Hadamard.
:::

:::{card}
:header: **Variational Quantum Eigensolver (VQE) for the Hydrogen Molecule**
:link: vqe_for_hydrogen
Build a molecular Hamiltonian with OpenFermion and find the ground state energy via VQE.
:::

::::
