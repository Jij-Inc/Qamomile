# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# Install the latest Qamomile through pip!
# (Google Colab) Pick the line that matches your chosen Transpiler tab
# below and remove the leading "# " from it to run.
# # !pip install qamomile                  # Qiskit (default)
# # !pip install "qamomile[quri_parts]"    # QURI Parts
# # !pip install "qamomile[cudaq-cu12]"    # CUDA-Q on a CUDA 12.x toolchain (use qamomile[cudaq-cu13] on CUDA 13.x). Linux / macOS-arm64 / WSL2 only.

# %% [markdown]
# This article uses Qiskit by default. Qamomile transpiles the same
# `@qkernel` to multiple quantum SDKs, so you can follow it with another
# SDK by swapping the import shown below — the rest of the article code
# is identical regardless of the SDK you pick. On Colab, uncomment the
# matching `pip install` line in the cell above first.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from qamomile.quri_parts import QuriPartsTranspiler
#
# transpiler = QuriPartsTranspiler()
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# Use `qamomile[cudaq-cu12]` for a CUDA 12.x toolchain or
# `qamomile[cudaq-cu13]` for a CUDA 13.x toolchain — pick the one that
# matches your installed CUDA Toolkit. CUDA-Q is supported on Linux,
# macOS arm64, and Windows-via-WSL2 only.
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# ```
# :::
# ::::

# %%
# Transpiler — by default this article uses Qiskit. If you picked a
# different tab above (QURI Parts / CUDA-Q), copy the two lines from
# that tab into this cell in place of the two below, and make sure the
# matching pip install line further up has been uncommented.
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %%
import numpy as np

import qamomile.circuit as qmc
import qamomile.observable as qmo
from qamomile.linalg import HermitianMatrix

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
# Transpile with all three bindings fixed. `transpiler.to_circuit()` returns the SDK-native circuit object (e.g. a Qiskit `QuantumCircuit`); the exact type depends on the Transpiler tab you picked at the top.

# %%
t_value = 0.8

circuit = transpiler.to_circuit(
    time_evolution,
    bindings={
        "n": H_mat.num_qubits,
        "hamiltonian": H_op,
        "t": t_value,
    },
)
print(type(circuit).__name__)

# %% [markdown]
# ## Verify Against the Exact Exponential
#
# For small Hermitian matrices we can build $U = e^{-iMt}$ directly from an eigendecomposition with numpy, apply it to the same initial state, and check that the Qamomile-generated circuit produces the same final state up to a global phase.

# %%
# Analytic ground truth — pure numpy, SDK-independent.
psi0 = np.zeros(4, dtype=complex)
psi0[0] = 1.0 / np.sqrt(2.0)
psi0[1] = 1.0 / np.sqrt(2.0)

eigvals, eigvecs = np.linalg.eigh(M)
U = eigvecs @ np.diag(np.exp(-1j * t_value * eigvals)) @ eigvecs.conj().T
psi_exact = U @ psi0

# %% [markdown]
# Reading the statevector out of the transpiled circuit (and how
# tightly we can assert against `psi_exact`) is SDK-specific. Qamomile's
# Qiskit emit pass routes `pauli_evolve` through Qiskit's native
# `PauliEvolutionGate` (analytic matrix exponential) so the realised
# unitary is exactly $e^{-iHt}$ and the fidelity rounds to 1. The QURI
# Parts and CUDA-Q emit passes don't yet have a corresponding native
# path and currently fall back to first-order Trotter, which gives an
# approximate unitary (~0.9 fidelity for this 2-qubit example at
# $t=0.8$). The Qiskit tab therefore asserts strict equality; the
# other two test a fidelity floor.
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: qiskit
#
# ```python
# import warnings
#
# from qiskit.quantum_info import Statevector
# from scipy.sparse import SparseEfficiencyWarning
#
# # Qiskit's PauliEvolutionGate.to_matrix() — called by Statevector to turn
# # the gate emitted by pauli_evolve into an exact unitary — uses
# # scipy.sparse.linalg.expm on a non-CSC Hamiltonian and emits noisy
# # SparseEfficiencyWarnings. We suppress them locally so the executed
# # notebook stays path-free.
# unitary_circuit = circuit.remove_final_measurements(inplace=False)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
#     psi_qm = np.array(Statevector.from_instruction(unitary_circuit).data)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # Qiskit emit uses analytic PauliEvolutionGate -> exact unitary.
# assert abs(fidelity - 1.0) < 1e-8
# ```
# :::
#
# :::{tab-item} QURI Parts
# :sync: quri_parts
#
# ```python
# from quri_parts.core.state import GeneralCircuitQuantumState
# from quri_parts.qulacs.simulator import evaluate_state_to_vector
#
# state = GeneralCircuitQuantumState(circuit.qubit_count, circuit)
# psi_qm = np.asarray(evaluate_state_to_vector(state).vector)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # Qamomile's QURI Parts emit currently lowers `pauli_evolve` via the
# # default phase-gadget decomposition (effectively first-order Trotter),
# # so the realised unitary is approximate (~0.9 fidelity for this
# # 2-qubit example at t=0.8). We test a floor instead of strict
# # equality. Once the QURI Parts emit pass grows a native
# # exponentiation path (e.g. via UnitaryMatrix-gate injection of
# # `expm(-i H t)`), this can be tightened back to the Qiskit-style
# # `abs(fidelity - 1.0) < 1e-8`.
# assert fidelity > 0.85
# ```
# :::
#
# :::{tab-item} CUDA-Q
# :sync: cudaq
#
# ```python
# import cudaq
#
# state = cudaq.get_state(circuit.kernel_func)
# psi_qm = np.asarray(state)
#
# fidelity = abs(np.vdot(psi_exact, psi_qm))
# print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
# # Same situation as QURI Parts: Qamomile's CUDA-Q emit currently uses
# # the default phase-gadget decomposition (first-order Trotter), so
# # the realised unitary is approximate (~0.9 fidelity for this 2-qubit
# # example at t=0.8). The CUDA-Q kernel surface doesn't expose a
# # public unitary-matrix-injection primitive equivalent to QURI Parts'
# # UnitaryMatrix gate, so the analytic-exact path needs more design
# # work; we test a floor in the meantime.
# assert fidelity > 0.85
# ```
# :::
# ::::

# %%
# Statevector extraction + fidelity check (Qiskit by default).
# If you picked a different tab above, copy that tab's full snippet
# over the lines below — including the assertion, since the
# tolerance differs per SDK. (And make sure the matching pip install
# line at the top of this article is uncommented.)
import warnings

from qiskit.quantum_info import Statevector
from scipy.sparse import SparseEfficiencyWarning

unitary_circuit = circuit.remove_final_measurements(inplace=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
    psi_qm = np.array(Statevector.from_instruction(unitary_circuit).data)

fidelity = abs(np.vdot(psi_exact, psi_qm))
print(f"fidelity (|<exact|qamomile>|): {fidelity:.12f}")
assert abs(fidelity - 1.0) < 1e-8

# %% [markdown]
# On Qiskit the assertion succeeds at the strict $1\\mathrm{e}{-8}$ tolerance: Qamomile's Qiskit emit pass routes `pauli_evolve` through `PauliEvolutionGate`, so the realised unitary is exactly $e^{-iHt}$. On QURI Parts and CUDA-Q the same decomposition currently emits as first-order Trotter — fidelity stays around $0.9$ for this example, until the QURI Parts and CUDA-Q emit passes grow a native analytic path.
#
# ## Recap
#
# - `HermitianMatrix` validates a dense Hermitian numpy array and owns the conversion to a Pauli sum via an FWHT-based decomposition.
# - `to_hamiltonian()` returns a `qamomile.observable.Hamiltonian`, which is the same type consumed by `pauli_evolve`, `expval`, and the rest of the quantum algorithms toolbox.
# - Together this gives a direct `np.ndarray → Hamiltonian → circuit` path for algorithms that start from a Hermitian matrix, such as Hamiltonian simulation, VQE, and the Hermitian side of block-encoding / LCU workflows.
#
# **Support for non-self-adjoint operators** (e.g. advection stencils that appear in numerical fluid simulations) is a natural follow-up and will be revisited once the right integration point for non-Hermitian matrices has been decided.
