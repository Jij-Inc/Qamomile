---
slug: release-notes
---

# Release Notes

- [v0.12.5](v0_12_5) — `PCEConverter` / `PCEEncoder` for Pauli Correlation Encoding, seeded QURI Parts sampling, measured `Vector[Bit]` condition fixes, and `qmc.expval` fixes for `Vector` elements
- [v0.12.4](v0_12_4) — `qmc.controlled` renamed to `qmc.control` with a more expressive symbolic mode, higher-order Ising model construction via `BinaryModel.from_higher_ising`, unary `-` on `Float` handles
- [v0.12.3](v0_12_3) — Python-style `Vector` slicing, `commutator(a, b)` for Pauli-Hamiltonians, `computational_basis_state` algorithm helper
- [v0.12.2](v0_12_2) — Möttönen amplitude encoding, sample-based subspace diagonalization (QSCI), `qmc.controlled` accepts built-in gates, `LocalSearch` on `BinaryModel`, docs restructured into `tutorial/` / `algorithm/` / `usage/` / `integration/`
- [v0.12.1](v0_12_1) — single-qubit gate broadcast over `Vector[Qubit]`, scalar literal promotion at sub-`@qkernel` call sites, QURI Parts symbolic-parameter arithmetic fix
- [v0.12.0](v0_12_0) — Suzuki–Trotter time evolution, `qamomile.linalg`, self-recursive `@qkernel`, OMMX `SampleSet` output from optimization converters
- [v0.11.1](v0_11_1) — Python 3.11 support
- [v0.11.0](v0_11_0) — Parametric Vector QAOA hardening, `cx_entangling_layer`, compiler-core cleanup
- [v0.10.0](v0_10_0) — Ground-up rebuild of the circuit programming layer
