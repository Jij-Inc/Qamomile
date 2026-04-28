# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# ---
# title: From a Hermitian Matrix to a Quantum Circuit
# tags: [hamiltonian-simulation, pauli-decomposition, algorithm]
# ---
#
# # From a Hermitian Matrix to a Quantum Circuit
#
# <!-- BEGIN auto-tags -->
# **Tags:** [`hamiltonian-simulation`](../tags/hamiltonian-simulation.md) · [`pauli-decomposition`](../tags/pauli-decomposition.md) · [`algorithm`](../tags/algorithm.md)
# <!-- END auto-tags -->
#
# In many quantum algorithms you start from a **Hermitian matrix** — a Hamiltonian given as a dense $2^n \times 2^n$ numpy array — and you want to simulate its time evolution $e^{-iHt}$ on a quantum computer. The standard path is:
#
# 1. Decompose $H$ into a weighted sum of Pauli strings.
# 2. Feed that sum into `pauli_evolve`, which emits the corresponding quantum circuit.
#
# Qamomile exposes this as a tiny, type-safe pipeline:
#
# ```
# np.ndarray  →  HermitianMatrix  →  Hamiltonian  →  pauli_evolve  →  circuit
# ```
#
# In this chapter we will:
#
# - Build a small Hermitian matrix with numpy,
# - Wrap it in `qamomile.linalg.HermitianMatrix` and call `to_hamiltonian()`,
# - Use the resulting `Hamiltonian` inside a `@qkernel` with `pauli_evolve`,
# - Verify the final statevector against an exact matrix exponential.
#
# The decomposition uses the **Fast Walsh-Hadamard Transform** internally and runs in $O(n \cdot 4^n)$ time with only NumPy — no Qiskit dependency.

# %%
import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.linalg import HermitianMatrix
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## A Small Hermitian Matrix
#
# We will use the two-site transverse-field Ising model:
#
# $$
# M \;=\; -Z_0 Z_1 \;-\; h \, (X_0 + X_1),
# $$
#
# with transverse field $h = 0.7$. Nothing here is Qamomile-specific — we just build an ordinary $4 \times 4$ numpy array with `np.kron`.

# %%
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

h_field = 0.7
M = -np.kron(Z, Z) - h_field * (np.kron(I2, X) + np.kron(X, I2))
print("shape:", M.shape)
print("Hermitian:", np.allclose(M, M.conj().T))

# %% [markdown]
# ## Wrap and Decompose
#
# `HermitianMatrix` is a thin wrapper that validates the array on construction (2D, square, power-of-two dimension, Hermitian) and exposes `to_hamiltonian()`. The resulting `Hamiltonian` lives in `qamomile.observable` — the same type consumed by `pauli_evolve`, `expval`, and the optimization helpers.
#
# In Qamomile's decomposition convention, **qubit 0 corresponds to the least-significant bit** of the matrix's computational-basis indices, which matches Qiskit's internal ordering.

# %%
H_mat = HermitianMatrix(M)
print("num_qubits:", H_mat.num_qubits)

H_op = H_mat.to_hamiltonian()
print("constant:", H_op.constant)
for ops, coeff in H_op.terms.items():
    print(f"  {ops}: {coeff:+.3f}")

# %% [markdown]
# For a two-site transverse-field Ising model we expect exactly three non-zero terms: one $Z_0 Z_1$ with coefficient $-1$ and two single-qubit $X$ terms with coefficient $-h$.

# %%
expected_zz = (
    qmo.PauliOperator(qmo.Pauli.Z, 0),
    qmo.PauliOperator(qmo.Pauli.Z, 1),
)
expected_x0 = (qmo.PauliOperator(qmo.Pauli.X, 0),)
expected_x1 = (qmo.PauliOperator(qmo.Pauli.X, 1),)

assert set(H_op.terms.keys()) == {expected_zz, expected_x0, expected_x1}
assert abs(H_op.terms[expected_zz] - (-1.0)) < 1e-12
assert abs(H_op.terms[expected_x0] - (-h_field)) < 1e-12
assert abs(H_op.terms[expected_x1] - (-h_field)) < 1e-12
assert H_op.constant == 0.0

# %% [markdown]
# ## Time Evolution in a `@qkernel`
#
# `pauli_evolve(q, H, t)` applies $e^{-iHt}$ to a qubit register. `H` is declared with the `qmc.Observable` handle type and supplied via bindings at transpile time; `t` can be bound or left as a sweepable parameter.
#
# We start from $\lvert 00\rangle$, put qubit 0 into a superposition with a Hadamard, apply the time-evolution step, and measure so the top-level kernel has a classical output (a requirement for entry points passed to `transpile()` / `to_circuit()`).


# %%
@qmc.qkernel
def time_evolution(
    n: qmc.UInt,
    hamiltonian: qmc.Observable,
    t: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    q = qmc.pauli_evolve(q, hamiltonian, t)
    return qmc.measure(q)


# %% [markdown]
# Transpile with all three bindings fixed. The Qiskit circuit comes back as a plain `QuantumCircuit` via `transpiler.to_circuit()`.

# %%
t_value = 0.8

qiskit_circuit = transpiler.to_circuit(
    time_evolution,
    bindings={
        "n": H_mat.num_qubits,
        "hamiltonian": H_op,
        "t": t_value,
    },
)
print(qiskit_circuit)

# %% [markdown]
# ## Verify Against the Exact Exponential
#
# For small Hermitian matrices we can build $U = e^{-iMt}$ directly from an eigendecomposition with numpy, apply it to the same initial state, and check that the Qamomile-generated circuit produces the same final state up to a global phase.

# %%
import warnings

from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

# Qiskit's `PauliEvolutionGate.to_matrix()` — called by `Statevector` to turn
# the gate emitted by `pauli_evolve` into an exact unitary — uses
# `scipy.sparse.linalg.expm` on a non-CSC Hamiltonian and emits noisy
# `SparseEfficiencyWarning`s. We suppress them locally so the executed
# notebook stays path-free. (Decomposing the circuit first would avoid the
# warning but replace the exact evolution with its Trotter approximation,
# which would break the fidelity check below.)
qiskit_unitary_circuit = qiskit_circuit.remove_final_measurements(inplace=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    psi_qm = np.array(Statevector.from_instruction(qiskit_unitary_circuit).data)

psi0 = np.zeros(4, dtype=complex)
psi0[0] = 1.0 / np.sqrt(2.0)
psi0[1] = 1.0 / np.sqrt(2.0)

eigvals, eigvecs = np.linalg.eigh(M)
U = eigvecs @ np.diag(np.exp(-1j * t_value * eigvals)) @ eigvecs.conj().T
psi_exact = U @ psi0

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert abs(fidelity - 1.0) < 1e-8

# %% [markdown]
# The fidelity is numerically indistinguishable from $1$: the circuit Qamomile built from the Pauli decomposition realises the same unitary as the direct matrix exponential.
#
# ## Recap
#
# - `HermitianMatrix` validates a dense Hermitian numpy array and owns the conversion to a Pauli sum via an FWHT-based decomposition.
# - `to_hamiltonian()` returns a `qamomile.observable.Hamiltonian`, which is the same type consumed by `pauli_evolve`, `expval`, and the rest of the quantum algorithms toolbox.
# - Together this gives a direct `np.ndarray → Hamiltonian → circuit` path for algorithms that start from a Hermitian matrix, such as Hamiltonian simulation, VQE, and the Hermitian side of block-encoding / LCU workflows.
#
# **Support for non-self-adjoint operators** (e.g. advection stencils that appear in numerical fluid simulations) is a natural follow-up and will be revisited once the right integration point for non-Hermitian matrices has been decided.
