---
slug: algorithm
title: Algorithms
---

# Algorithms

Concrete quantum algorithm examples built with Qamomile. Click a tag below to filter, or browse all algorithms.

## Browse by tag

[`advanced`](tags/advanced.md) (1) · [`built-in`](tags/built-in.md) (1) · [`chemistry`](tags/chemistry.md) (1) · [`decomposition`](tags/decomposition.md) (1) · [`from-scratch`](tags/from-scratch.md) (1) · [`graph`](tags/graph.md) (2) · [`graph-partition`](tags/graph-partition.md) (1) · [`ground-state`](tags/ground-state.md) (1) · [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) (2) · [`intermediate`](tags/intermediate.md) (4) · [`jijmodeling`](tags/jijmodeling.md) (1) · [`linalg`](tags/linalg.md) (1) · [`maxcut`](tags/maxcut.md) (1) · [`openfermion`](tags/openfermion.md) (1) · [`optimization`](tags/optimization.md) (2) · [`pauli`](tags/pauli.md) (2) · [`qaoa`](tags/qaoa.md) (2) · [`simulation`](tags/simulation.md) (1) · [`suzuki`](tags/suzuki.md) (1) · [`time-evolution`](tags/time-evolution.md) (2) · [`trotter`](tags/trotter.md) (1) · [`variational`](tags/variational.md) (3) · [`vqe`](tags/vqe.md) (1)

## All algorithms

### [From a Hermitian Matrix to a Quantum Circuit](hermitian_decomposition.ipynb)

**Tags:** [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) [`pauli`](tags/pauli.md) [`decomposition`](tags/decomposition.md) [`time-evolution`](tags/time-evolution.md) [`linalg`](tags/linalg.md) [`intermediate`](tags/intermediate.md)

In many quantum algorithms you start from a **Hermitian matrix** — a Hamiltonian given as a dense $2^n \times 2^n$ numpy array — and you want to simulate its time evolution $e^{-iHt}$ on a quantum computer. The standard path is:

### [Hamiltonian Simulation with Suzuki–Trotter](hamiltonian_simulation.ipynb)

**Tags:** [`hamiltonian-simulation`](tags/hamiltonian-simulation.md) [`trotter`](tags/trotter.md) [`suzuki`](tags/suzuki.md) [`simulation`](tags/simulation.md) [`time-evolution`](tags/time-evolution.md) [`pauli`](tags/pauli.md) [`advanced`](tags/advanced.md)

Simulating the time evolution $e^{-iHt}$ of a quantum system is one of the canonical applications of a quantum computer. When the Hamiltonian splits into non-commuting pieces $H = A + B$ with $[A, B] \neq 0$, the naive factorisation $e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$ is wrong; the standard fix is **Trotter–Suzuki product formulas**, which interleave short evolutions of each piece. The error decreases as we take smaller steps (Lie–Trotter, first order), symmetrise the step (Strang, second order), or nest the symmetric step recursively via Suzuki's construction (any even order).

### [QAOA for Graph Partitioning](qaoa_graph_partition.ipynb)

**Tags:** [`qaoa`](tags/qaoa.md) [`optimization`](tags/optimization.md) [`variational`](tags/variational.md) [`graph`](tags/graph.md) [`graph-partition`](tags/graph-partition.md) [`jijmodeling`](tags/jijmodeling.md) [`built-in`](tags/built-in.md) [`intermediate`](tags/intermediate.md)

This tutorial demonstrates how to solve the **graph partitioning problem** using the Quantum Approximate Optimization Algorithm (QAOA) with Qamomile.

### [QAOA for MaxCut](qaoa_maxcut.ipynb)

**Tags:** [`qaoa`](tags/qaoa.md) [`optimization`](tags/optimization.md) [`variational`](tags/variational.md) [`graph`](tags/graph.md) [`maxcut`](tags/maxcut.md) [`from-scratch`](tags/from-scratch.md) [`intermediate`](tags/intermediate.md)

This tutorial walks through the Quantum Approximate Optimization Algorithm (QAOA) pipeline step by step, using Qamomile's low-level circuit primitives. Rather than using the high-level `QAOAConverter`, we will:

### [VQE for the Hydrogen Molecule](vqe_for_hydrogen.ipynb)

**Tags:** [`vqe`](tags/vqe.md) [`variational`](tags/variational.md) [`chemistry`](tags/chemistry.md) [`ground-state`](tags/ground-state.md) [`openfermion`](tags/openfermion.md) [`intermediate`](tags/intermediate.md)

This tutorial demonstrates how to implement the Variational Quantum Eigensolver (VQE) algorithm to find the ground state energy of the hydrogen molecule (H₂). We use [OpenFermion](https://quantumai.google/openfermion) for generating molecular Hamiltonians.
