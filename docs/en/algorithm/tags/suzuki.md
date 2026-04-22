---
title: "Tag: suzuki"
tags: [suzuki]
---

# `suzuki`

Algorithm examples tagged **`suzuki`** (1).

[← Back to all algorithms](../index.md)

---

**Browse by tag:** 
[`advanced`](./advanced.md) · [`built-in`](./built-in.md) · [`decomposition`](./decomposition.md) · [`from-scratch`](./from-scratch.md) · [`graph`](./graph.md) · [`graph-partition`](./graph-partition.md) · [`hamiltonian-simulation`](./hamiltonian-simulation.md) · [`intermediate`](./intermediate.md) · [`jijmodeling`](./jijmodeling.md) · [`linalg`](./linalg.md) · [`maxcut`](./maxcut.md) · [`optimization`](./optimization.md) · [`pauli`](./pauli.md) · [`qaoa`](./qaoa.md) · [`simulation`](./simulation.md) · [`time-evolution`](./time-evolution.md) · [`trotter`](./trotter.md) · [`variational`](./variational.md)

---

### [Hamiltonian Simulation with Suzuki–Trotter](../hamiltonian_simulation.ipynb)

**Tags:** [`hamiltonian-simulation`](./hamiltonian-simulation.md) [`trotter`](./trotter.md) [`simulation`](./simulation.md) [`time-evolution`](./time-evolution.md) [`pauli`](./pauli.md) [`advanced`](./advanced.md)

Simulating the time evolution $e^{-iHt}$ of a quantum system is one of the canonical applications of a quantum computer. When the Hamiltonian splits into non-commuting pieces $H = A + B$ with $[A, B] \neq 0$, the naive factorisation $e^{-i(A+B)t} = e^{-iAt}\,e^{-iBt}$ is wrong; the standard fix is **Trotter–Suzuki product formulas**, which interleave short evolutions of each piece. The error decreases as we take smaller steps (Lie–Trotter, first order), symmetrise the step (Strang, second order), or nest the symmetric step recursively via Suzuki's construction (any even order).
