# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [tutorial, primitive, encoding, simulation]
# ---
#
# # From a Hermitian Matrix to a Quantum Circuit
#
# Quantum algorithms use Hamiltonians to represent physical systems and optimization problems.
# A Hamiltonian is represented by a Hermitian matrix ($H^\dagger = H$) and is used,
# for example, to describe the time evolution of a quantum state through $e^{-iHt}$.
# Qamomile can convert a matrix created with NumPy into a `Hamiltonian`
# that acts as an operator on qubits.
#
# ```text
# Create a Hermitian matrix with NumPy
#     ↓
# Decompose it into a sum of Pauli operators
#     ↓
# Use it in a qkernel
# ```
#
# In this chapter, we will:
#
# 1. Create a two-qubit Hermitian matrix with NumPy.
# 2. Convert it into a sum of Pauli operators with `HermitianMatrix.to_hamiltonian()`.
# 3. Use the Hamiltonian for time evolution in a function decorated with `@qm.qkernel`.
# 4. Inspect measurement results from a simulator.
#
# Finally, as an optional extension, we will check the accuracy of the time-evolution approximation.

# %%
import numpy as np
from qiskit.quantum_info import Statevector

import qamomile as qm
from qamomile.qiskit import to_circuit
from qamomile.stdlib import suzuki_trotter

# %% [markdown]
# ## 1. Create a Hermitian Matrix with NumPy
#
# As an example, consider the two-site transverse-field Ising model.
# Its Hamiltonian is
#
# $$
# \label{tutorial08-ising-hamiltonian}
# H = -Z_0 Z_1 - h(X_0 + X_1), \qquad h = 0.7
# $$
#
# The two-site Hamiltonian is represented by a $4 \times 4$ matrix.
# We construct it with `np.kron(A, B)`, which computes the Kronecker product—the matrix representation of the tensor product $A \otimes B$.\
# Qamomile uses the computational-basis order $\lvert q_1 q_0 \rangle: \lvert 00 \rangle, \lvert 01 \rangle, \lvert 10 \rangle, \lvert 11 \rangle$, where $q_i$ is the bit associated with qubit $i$.
# Thus, `np.kron(identity, pauli_x)` applies X to qubit 0, while `np.kron(pauli_x, identity)` applies it to qubit 1.

# %%
pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
identity = np.eye(2, dtype=np.complex128)

field_strength = 0.7
matrix = -np.kron(pauli_z, pauli_z) - field_strength * (
    np.kron(identity, pauli_x) + np.kron(pauli_x, identity)
)
operator = qm.linalg.HermitianMatrix(matrix)

print("Matrix shape:", matrix.shape)
print("Is Hermitian:", np.allclose(matrix, matrix.conj().T))
print(matrix)

# %% [markdown]
# ## 2. Convert to a Qamomile `Hamiltonian`
#
# Pass the matrix to `qm.linalg.HermitianMatrix(...)`, which checks its shape and Hermiticity.
# An $n$-qubit register has $2^n$ computational-basis states, so its Hamiltonian is represented by a $2^n \times 2^n$ matrix.\
# Then call `to_hamiltonian(...)` to express the matrix as a sum of Pauli operators.
# Here, the decomposition contains three terms: $Z_0 Z_1, X_0$, and $X_1$.
# The `tol` parameter sets the threshold for omitting small Pauli coefficients: coefficients with absolute value at most `tol` are omitted.
# Setting `tol=0.0` keeps all computed nonzero coefficients.
# This threshold is separate from the time-evolution approximation introduced below.

# %%
operator = qm.linalg.HermitianMatrix(matrix)
hamiltonian = operator.to_hamiltonian(tol=0.0)

print("Number of qubits:", operator.num_qubits)
print("Number of Pauli terms:", len(hamiltonian.terms))
print("Pauli decomposition:", hamiltonian)

expected = -qm.Z(0) * qm.Z(1) - field_strength * (qm.X(0) + qm.X(1))

assert operator.num_qubits == 2
assert hamiltonian == expected

# %% [markdown]
# The coefficients of the three terms are
#
# * $X_0$: -0.7
# * $Z_0 Z_1$: -1.0
# * $X_1$: -0.7
#
# matching the coefficients in [](#tutorial08-ising-hamiltonian).
#
# ## 3. Time Evolution with a `Hamiltonian`
#
# We now use the `hamiltonian` obtained above to construct a time-evolution circuit.
#
# Write $H = A+B$, where $A = -Z_0 Z_1$ is the interaction and $B = -h (X_0 + X_1)$ is the transverse field.
# Because $A$ and $B$ do not commute, applying their time evolutions one after the other generally does not reproduce $e^{-iHt}$ exactly.
# The Suzuki–Trotter formula approximates this evolution by combining shorter evolutions under the two parts.\
# Within the transverse-field part $B$, $X_0$ and $X_1$ commute because they act on different qubits. Its time evolution therefore factors exactly as
#
# $$
# e^{-iB\Delta t}
# = e^{ihX_0 \Delta t} e^{ihX_1 \Delta t}.
# $$
#
# The interaction part $A$ contains one Pauli term, so its evolution can also be implemented exactly.
# In this example, the approximation arises when combining the evolutions of $A$ and $B$ to reproduce the evolution under the full Hamiltonian $H$.
# Define the interaction term, then subtract it from `hamiltonian` to obtain the transverse-field part.

# %%
interaction = -qm.Z(0) * qm.Z(1)
transverse_field = hamiltonian - interaction
evolve = suzuki_trotter(term_count=2, order=2, steps=2)


# %% [markdown]
# `suzuki_trotter(...)` returns a quantum kernel, stored here as evolve, which accepts a quantum state, the Hamiltonian groups, and the evolution time.
# We use two groups (`term_count=2`), a second-order formula (`order=2`), and two equal time intervals (`steps=2`).
#
# :::{note}
# Each Hamiltonian group passed to `evolve(...)` must consist of mutually commuting Pauli terms. Different groups need not commute.
# :::
#
# The `ising_state(...)` kernel prepares the two-qubit state and calls `evolve(...)`.
# The `simulate_ising(...)` kernel adds the final measurement.
# Keeping measurement separate lets us compare quantum states later.\
# The annotation `qm.Const[qm.Vector[qm.Observable, 2]]` specifies two operator values fixed during specialization.
# After defining the kernels, we supply (`interaction`, `transverse_field`) through `qm.specialize(...)` to create program.
# The evolution time remains a `Param` value supplied for each run.


# %%
@qm.qkernel
def ising_state(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = evolve(state, hamiltonians, time)
    return state


@qm.qkernel
def simulate_ising(
    hamiltonians: qm.Const[qm.Vector[qm.Observable, 2]],
    time: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(ising_state(hamiltonians, time))


program = qm.specialize(
    simulate_ising,
    hamiltonians=(interaction, transverse_field),
)

# %% [markdown]
# ## 4. Inspect the Measurement Results
#
# Run `program` with `qm.run(...)`, setting the evolution time to 0.8 in the `parameters` dictionary.\
# `on=qm.Reference()` selects Qamomile’s built-in simulator for local execution.
# With `shots=128`, state preparation, time evolution, and measurement are repeated 128 times.
# A fixed `seed` makes the sampling reproducible.

# %%
result = qm.run(
    program,
    on=qm.Reference(),
    parameters={"time": 0.8},
    shots=128,
    seed=7,
).result

print("Number of shots:", result.shots)
print("Measurement counts:", result.counts)

assert result.shots == 128
assert sum(result.counts.values()) == 128

# %% [markdown]
# `result.counts` stores each observed bit string and the number of times it appeared.
# We have now obtained measurement results from a quantum circuit using the Hamiltonian represented by our NumPy matrix.
#
# ## 5. Check the Accuracy of the Time Evolution
#
# We assess the approximation by comparing two final states obtained from the same initial state: one calculated from $e^{-iHt}$, and the other produced by the Suzuki–Trotter circuit.\
# To calculate their fidelity, we need both statevectors.
# The measurement counts collected in the previous section do not determine these statevectors.\
# We use Qiskit to calculate the approximate circuit’s final statevector: `to_circuit(...)` converts `ising_state` into a Qiskit circuit, and `Statevector.from_instruction(...)` simulates it, including the initial-state preparation.
# For this small example, NumPy calculates the reference state directly from the Hamiltonian, without the Suzuki–Trotter approximation.\
# We compare the two states using the following state fidelity:
#
# $$
# F = |\langle\psi_{\mathrm{exact}}\mid\psi_{\mathrm{approx}}\rangle|^2
# $$
#
# The closer $F$ is to 1, the more closely the two states agree.

# %%
unitary_circuit = to_circuit(
    ising_state,
    hamiltonians=(interaction, transverse_field),
    parameters={"time": 0.8},
)
approximate_state = np.asarray(Statevector.from_instruction(unitary_circuit).data)

# In |q1 q0> order, the initial state is |0+> = (|00> + |01>) / sqrt(2).
initial_state = np.zeros(4, dtype=np.complex128)
initial_state[0] = 1.0 / np.sqrt(2.0)
initial_state[1] = 1.0 / np.sqrt(2.0)

eigenvalues, eigenvectors = np.linalg.eigh(matrix)
exact_unitary = (
    eigenvectors @ np.diag(np.exp(-1j * 0.8 * eigenvalues)) @ eigenvectors.conj().T
)
exact_state = exact_unitary @ initial_state

fidelity = float(abs(np.vdot(exact_state, approximate_state)) ** 2)
print(f"State fidelity: {fidelity:.6f}")
assert fidelity > 0.99

# %% [markdown]
# ## 6. Summary
#
# In this tutorial, you learned how to:
#
# * Load a Hermitian matrix prepared with NumPy using `qm.linalg.HermitianMatrix()`.
# * Decompose it into a sum of Pauli operators with `to_hamiltonian()`.
# * Use the Hamiltonian for time evolution inside a function decorated with `@qm.qkernel`.
# * Obtain measurement results with `qm.specialize()` and `qm.run()`.
