---
slug: algorithm
---

# Algorithms

Concrete quantum algorithm examples built with Qamomile.

## All articles

- [Introduction to Quantum Fourier transform (QFT)](qft) — DFT background, QFT circuit steps, stdlib implementation, execution, and resource estimation
- [QAOA for MaxCut](qaoa_maxcut) — Build a QAOA circuit from scratch to solve MaxCut, then compare with the built-in `qaoa_state`
- [QAOA for Graph Partitioning](qaoa_graph_partition) — End-to-end optimization example using OMMX, JijModeling, and `QAOAConverter`
- [VQE for the Hydrogen Molecule](vqe_for_hydrogen) — Build a molecular Hamiltonian with OpenFermion and find the ground state energy via VQE
- [Hamiltonian Simulation with Suzuki–Trotter](hamiltonian_simulation) — Trotter–Suzuki product formulas on the Rabi model with empirical convergence orders
- [Introduction to Quantum Error Correction](quantum_error_correction) — 3-qubit bit-flip/phase-flip codes, Shor's 9-qubit code, and stabilizers
- [Steane [[7,1,3]] Code](steane_code) — CSS construction, syndrome decoding, and transversal Hadamard
- [Quantum Selected Configuration Interaction (QSCI)](qsci) — Sample bitstrings from a quantum state, build an effective Hamiltonian, and diagonalize it classically with a strict variational guarantee
- [Hybrid Quantum Neural Network (HQNN)](hybrid_qnn) — End-to-end CNN + quantum variational circuit on Fashion-MNIST with the parameter shift rule
- [Quantum Kernel Methods](quantum_kernel_classification) — Use quantum feature maps and kernel methods for classification on the make_circles dataset
- [Implementing Quantum-enhanced MCMC](qe_mcmc) — Implement Quantum-enhanced MCMC using Trotter-decomposed time evolution
- [Möttönen Amplitude Encoding](mottonen_amplitude_encoding) — Prepare an arbitrary real or complex amplitude vector via Gray-code Ry/Rz multiplexers, with three input modes (concrete, bound `Vector[Float]`, runtime-parametric angles)
- [Multidimensional Quantum Fourier Transform](multidimensional_qft) — Implement multidimensional QFT and classical preprocessing for inputs whose grid sizes are not powers of two
- [PCE for MaxCut](pce_maxcut) — Solve a 20-variable MaxCut on just 3 qubits with Pauli Correlation Encoding, training a hardware-efficient ansatz against a tanh-relaxed surrogate and decoding via sign rounding
