# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [algorithm, sample-based]
# ---
#
# # Quantum Selected Configuration Interaction (QSCI)
#
# QSCI selects computational basis states from bitstrings obtained by measuring a quantum state, then classically diagonalizes the Hamiltonian restricted to their span. The quantum computation gathers information about which basis states to use. The classical computation diagonalizes the Hamiltonian represented in the selected basis to obtain the lowest energy and corresponding state within that subspace.
#
# Here, we use a four-qubit transverse-field Ising model rather than a molecule. A configuration in this example is a bitstring representing a computational basis state of the four spins. We prepare the input state with a short VQE optimization and compare the energies as we enlarge the QSCI subspace.

# %%
# Install Qamomile with the QURI Parts extra and the numerical and plotting dependencies.
# # !pip install "qamomile[quri_parts]" numpy scipy matplotlib

# %%
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.quri_parts import Qulacs, QuriParts

# %% [markdown]
# ## Problem Settings
#
# Consider a four-qubit transverse-field Ising chain with open boundaries. Its Hamiltonian is
#
# $$
# H=-J\sum_{i=0}^{n-2} Z_iZ_{i+1}-h\sum_{i=0}^{n-1}X_i,
# \qquad n=4,\quad J=1,\quad h=0.7
# $$
#
# where $Z_i$ and $X_i$ are Pauli operators acting on qubit $i$. The first term describes interactions between neighboring spins, and the second describes the transverse field.
#
# Four qubits have $2^4=16$ computational basis states. For comparison, we construct the Hamiltonian matrix using the full basis and find its lowest eigenvalue with NumPy. The resulting `EXACT_ENERGY` is the reference value for this finite model. We construct the full-space matrix only to check this small example.

# %%
N_QUBITS = 4
J = 1.0
FIELD = 0.7
HAMILTONIAN = qm.Hamiltonian.zero(num_qubits=N_QUBITS)
for index in range(N_QUBITS - 1):
    HAMILTONIAN = HAMILTONIAN - J * qm.Z(index) * qm.Z(index + 1)
for index in range(N_QUBITS):
    HAMILTONIAN = HAMILTONIAN - FIELD * qm.X(index)

ALL_CONFIGURATIONS = tuple(itertools.product((0, 1), repeat=N_QUBITS))
exact_matrix = qm.linalg.subspace_hamiltonian(ALL_CONFIGURATIONS, HAMILTONIAN)
EXACT_ENERGY = float(np.linalg.eigvalsh(exact_matrix)[0])

assert HAMILTONIAN.num_qubits == N_QUBITS
assert HAMILTONIAN.is_hermitian
assert exact_matrix.shape == (2**N_QUBITS, 2**N_QUBITS)
print(f"Exact ground-state energy: {EXACT_ENERGY:+.6f}")

# %% [markdown]
# ## Algorithm
#
# **Quantum-Selected Configuration Interaction (QSCI)** is a hybrid quantum-classical algorithm that estimates the ground-state energy by measuring a quantum state in the computational basis, selecting the most frequent distinct bitstrings to form a subspace, and classically diagonalizing the Hamiltonian within that subspace.
#
# The algorithm proposed by Kanno *et al.*{cite:p}`10.48550/arXiv.2302.11320` proceeds as follows:
#
# 1. Prepare an input state $\lvert \psi_{\mathrm{in}} \rangle$. For example, use VQE to adjust circuit parameters in search of a lower-energy state.
# 2. Measure $\lvert \psi_{\mathrm{in}} \rangle$ in the computational (Z) basis many times.
# 3. Select the $K$ most frequently observed distinct bitstrings and use their corresponding basis states $\{|s_i\rangle\}_{i=1}^{K}$ to span the subspace.
# 4. Construct the effective Hamiltonian $H^{\mathrm{sub}}_{ij}=\langle s_i|H|s_j\rangle$, diagonalize it classically, and take its lowest eigenvalue as the QSCI energy.
#
# A feature of QSCI is its **variational guarantee**: the computed energy is no lower than the exact ground-state energy, provided that the matrix elements of the original Hamiltonian are correctly constructed in the selected basis. Noise in state preparation or measurement can change which basis states are selected, but the resulting subspace energy still satisfies
#
# $$
# E_{\mathrm{QSCI}}(K)\geq E_{\mathrm{exact}}.
# $$
#
# Here, $E_{\mathrm{exact}}$ is the ground-state energy of the same Hamiltonian in the full space. This does not guarantee that the computed energy is close to the exact value.
#
# When we increase $K$ by adding basis states while retaining those already selected, the QSCI energy cannot increase. It agrees with the exact ground-state energy when the subspace includes the full computational basis. Finite sampling may leave some basis states unobserved, so increasing $K$ within the observed configurations does not always reach the exact energy.
#
# QSCI does not require the input-state parameters to be fully optimized. The input state may use random parameters or parameters obtained from a short VQE optimization. Whether a small subspace gives good accuracy depends on selecting bitstrings that are important for representing the true ground state. A short VQE optimization aims to make these bitstrings easier to observe, but its effect depends on the input state and the sampling results.

# %% [markdown]
# ## Implementation
#
# ### A Circuit for Preparing the Input State
#
# A parameterized circuit for preparing trial states is called an ansatz. Here, we repeat a layer of RY gates on all qubits followed by CX gates between neighboring qubits twice, then apply one final layer of RY gates. The circuit contains $3\times4=12$ RY gates and $2\times3=6$ CX gates. We therefore pass the 12 RY rotation angles as `thetas`.
#
# Each quantum operation returns an updated `state`, which we pass to the next operation. The `qsci_state(...)` kernel returns the quantum state before measurement and provides the state preparation for the VQE `Estimator`. The `qsci_samples(...)` kernel calls the same state preparation, measures it, and returns four bits for one shot. Sharing the circuit between these two functions lets VQE and QSCI sampling use the same rotation angles.
#
# The kernels in this example are fixed to four qubits and 12 parameters. Changing `N_QUBITS` alone does not extend them to an arbitrary number of qubits.

# %%
@qm.qkernel
def qsci_state(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare one runtime-rebindable four-qubit QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared trial state.
    """

    state = qm.qubits(4)
    for repetition in range(2):
        for target in range(4):
            state = qm.ry(state, thetas[repetition * 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[8 + target], target=target)
    return state


@qm.qkernel
def qsci_samples(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Shot[qm.Bits[4]]: Computational-basis configuration per shot.
    """

    return qm.measure(qsci_state(thetas))


# %% [markdown]
# ### Compiling the QSCI Measurement Circuit for QURI Parts
#
# We use `qm.specialize()` to prepare `qsci_samples()` as a program and compile it for `QuriParts()`. The resulting `artifact` is the circuit we will later sample using `Qulacs()`. Because `thetas` is declared as `Param`, we supply the specific rotation angles at execution time.
#
# The kernel explicitly compiled in this cell is `qsci_samples(...)`, which performs measurements. For VQE energy evaluations, the next section creates an `Estimator` that takes `qsci_state(...)`.

# %%
program = qm.specialize(qsci_samples)
sample_endpoint = Qulacs()
artifact = qm.compile(program, to=QuriParts())

# %% [markdown]
# ### Step 1: Choosing the Input State with a Short VQE Optimization
#
# We first run a short VQE optimization to obtain a candidate input state. By searching for a state with lower energy, we aim to make bitstrings useful for representing the ground state more likely to be observed. However, a limited optimization budget does not guarantee a useful distribution or improved QSCI accuracy for the same $K$. QSCI input states can also be prepared by methods other than VQE.
#
# Here, we use `Estimator(qsci_state, HAMILTONIAN, grouping="qwc")`. It groups Pauli terms that share a local measurement basis and estimates the energy on `Qulacs`. These VQE evaluations are separate from the Z-basis measurements for QSCI in the next section.
#
# We use a total of 128 shots per energy evaluation and a maximum of 500 objective evaluations for COBYLA. In SciPy's COBYLA implementation, `maxiter` specifies the maximum number of objective evaluations.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
expectation_endpoint = Qulacs()
vqe_shots = 2 if docs_test_mode else 128
energy_estimator = Estimator(qsci_state, HAMILTONIAN, grouping="qwc")
cost_history = []


def cost_function(parameters):
    """Estimate the trial-state energy at one variational point."""

    estimate = energy_estimator.estimate(
        on=expectation_endpoint,
        parameters={"thetas": tuple(float(value) for value in parameters)},
        shots=vqe_shots,
        seed=10 + len(cost_history),
    )
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(0)
initial_parameters = rng.uniform(0.0, 2.0 * np.pi, 12)
max_evaluations = 14 if docs_test_mode else 500
optimizer_options = {"maxiter": max_evaluations, "rhobeg": 0.5}
if docs_test_mode:
    optimizer_options["f_target"] = float("inf")

optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options=optimizer_options,
)
if docs_test_mode:
    assert optimization.nfev == 1
optimized_parameters = tuple(float(value) for value in optimization.x)

print(
    f"Optimizer objective estimate = {optimization.fun:+.6f} "
    f"(gap to E_exact: {optimization.fun - EXACT_ENERGY:+.4e})"
)

print("Optimizer success:", optimization.success)
print("Optimizer message:", optimization.message)
print("Objective evaluations:", optimization.nfev)

assert np.isfinite(optimization.fun)
assert len(optimized_parameters) == 12
assert len(cost_history) == optimization.nfev

# %% [markdown]
# #### Distinguishing the VQE Estimate from the Variational Principle
#
# The displayed `optimization.fun` is the finite-shot objective value reported by the optimizer. It is neither an independent measurement at the returned parameters nor the state's exact expectation value. Statistical fluctuations can cause a finite-shot estimate to fall below the ground-state energy.
#
# In contrast, the true expectation value of a normalized state prepared by VQE is no lower than the ground-state energy of the same Hamiltonian. This also holds for a normalized state after noise.
#
# QSCI uses the basis states selected by measurement and classically calculates the subspace matrix elements from the known $H$. Measurement noise and finite shots affect which basis states are selected. A different selection can change the accuracy, but the variational upper bound provided by the lowest eigenvalue of a correctly constructed subspace Hamiltonian is preserved. We will examine this distinction in the diagonalization section.

# %% [markdown]
# ### Step 2: Sampling Bitstrings in the Z Basis
#
# Using the `optimized_parameters` obtained above, we repeatedly measure the input state in the computational basis. We execute the measurement circuit containing the same `qsci_state(...)` used for VQE on QURI Parts / Qulacs with 4,000 shots.
#
# Each measurement result is a tuple `(b_0, ..., b_{n-1})`. Element `b_q` is the measured bit for qubit $q$: 0 corresponds to the Z eigenvalue $+1$, and 1 to $-1$. For example, `(0, 1, 0, 1)` means that qubits 0 through 3 were measured as `0, 1, 0, 1`, respectively.
#
# The `counts` dictionary maps distinct bitstrings to their observed counts. Observing the same bitstring repeatedly does not increase the subspace dimension. We sort by decreasing count and break ties using the lexicographic order of the tuples.

# %%
SHOTS = 1 if docs_test_mode else 4_000

record = qm.run(
    artifact,
    on=sample_endpoint,
    parameters={"thetas": optimized_parameters},
    shots=SHOTS,
    seed=2,
)
counts = record.result.counts
configurations = tuple(
    bits
    for bits, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if count > 0
)

if not configurations:
    raise ValueError("No observed configurations are available for QSCI.")


print("sample counts:", counts)
print("Number of distinct observed configurations:", len(configurations))

assert sum(counts.values()) == SHOTS
assert len(configurations) <= 2**N_QUBITS
assert all(len(bits) == N_QUBITS for bits in configurations)

# %% [markdown]
# ### Steps 3 + 4: Constructing and Diagonalizing the Subspace
#
# Denote the $K$ most frequently observed distinct bitstrings by $s_1, \ldots, s_K$. The corresponding basis states span
#
# $$
# \mathcal{S}_K=\operatorname{span}\{|s_1\rangle,\ldots,|s_K\rangle\}
# $$
#
# The Hamiltonian in this subspace is the $K\times K$ matrix
#
# $$
# H^{\mathrm{sub}}_{ij}=\langle s_i|H|s_j\rangle
# $$
#
# We solve the subspace eigenvalue problem using `qm.linalg.solve_subspace(configurations[:K], HAMILTONIAN)`, take the eigenvalues from the first element of its return value, and use the lowest one as the QSCI energy.
#
# This is not an average of energies weighted by observed counts. The counts select the basis states; the coefficients of the selected basis states come from the eigenvalue equation
#
# $$
# H^{\mathrm{sub}}c=Ec
# $$
#
# The resulting state is $|\psi_{\mathrm{QSCI}}\rangle=\sum_{i=1}^{K}c_i|s_i\rangle$. This diagonalization uses both diagonal and off-diagonal matrix elements.
#
# #### Preserving the Variational Upper Bound
#
# Distinct computational basis states are orthonormal, so when $\sum_i|c_i|^2=1$, the state $|\psi_{\mathrm{QSCI}}\rangle$ is also normalized. The lowest eigenvalue in the subspace therefore satisfies
#
# $$
# E_{\mathrm{QSCI}}(K)
# =
# \min_{\substack{|\phi\rangle\in\mathcal{S}_K\\ \langle\phi|\phi\rangle=1}}
# \langle\phi|H|\phi\rangle
# \geq E_{\mathrm{exact}}
# $$
#
# Here, $E_{\mathrm{exact}}$ is the lowest eigenvalue of the same Hamiltonian in the full space.
#
# This argument assumes that we correctly construct the matrix elements of the original Hermitian $H$ for the selected basis states. It does not apply directly to methods that approximate matrix elements using separate noisy measurements or to comparisons against a reference with different symmetry conditions. Floating-point calculations also introduce rounding errors, so the assertions below allow a small tolerance.

# %% [markdown]
# ## Result

# %%
number_observed = len(configurations)
SUBSPACE_SIZES = tuple(
    sorted(
        {
            size
            for size in (1, 2, 4, 8, min(14, number_observed))
            if 1 <= size <= number_observed
        }
    )
)
energies = tuple(
    float(qm.linalg.solve_subspace(configurations[:size], HAMILTONIAN)[0][0])
    for size in SUBSPACE_SIZES
)
for size, energy in zip(SUBSPACE_SIZES, energies):
    print(
        f"K={size}: E_QSCI={energy:+.12f}, "
        f"gap={energy - EXACT_ENERGY:+.6e}"
    )
print(f"E_exact={EXACT_ENERGY:+.12f}")

assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
assert all(energy >= EXACT_ENERGY - 1e-12 for energy in energies)

# %% [markdown]
# ### Subspace Size and Interpreting the Plot
#
# Here, we sort the same `counts` once and add basis states from the beginning of that ordered list. Thus, if $K<K'$, then $\mathcal{S}_K\subseteq\mathcal{S}_{K'}$, giving
#
# $$
# E_{\mathrm{exact}}\leq E_{\mathrm{QSCI}}(K')\leq E_{\mathrm{QSCI}}(K)
# $$
#
# This holds because enlarging the subspace provides more states over which to minimize. Monotonicity is not guaranteed for non-nested subspaces selected from separate sampling runs.
#
# In this example, we limit the subspace size to 14. If fewer than 14 distinct bitstrings are observed, their number is the largest $K$ used. Unobserved basis states are not automatically added to the calculation.
#
# We compare these subspace energies against the exact energy calculated from all 16 configurations. Whether a small $K$ gives sufficient accuracy depends on the input state and the selection. The QSCI energy is not guaranteed to be lower than the energy of the input VQE state.

# %%
plt.figure(figsize=(6, 4))
plt.plot(SUBSPACE_SIZES, energies, "-o", label=r"$E_{\mathrm{QSCI}}$")
plt.axhline(EXACT_ENERGY, linestyle="--", label=r"$E_{\mathrm{exact}}$")
plt.xlabel("Subspace size $K$")
plt.ylabel("Energy")
plt.title("QSCI energy in nested subspaces (4-qubit TFIM)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Additional Notes: Removing Duplicates and the Classical Computational Cost
#
# Using the keys of `counts` prevents us from including the same bitstring more than once as a subspace basis state. Distinct computational basis states are orthonormal, with overlap matrix
#
# $$
# S_{ij}=\langle s_i|s_j\rangle=\delta_{ij}
# $$
#
# We therefore need only solve an ordinary Hermitian eigenvalue problem, rather than the generalized eigenvalue problem required for a nonorthogonal basis. This does not imply that $H^{\mathrm{sub}}$ itself is necessarily well-conditioned.
#
# The subspace matrix has size $K\times K$. Increasing $K$ can improve accuracy, but storing a dense matrix requires space proportional to $K^2$, and the cost of diagonalization also grows. The number of selected basis states therefore matters as well as the shot count.
#
# In this example, `EXACT_ENERGY` is obtained by using Qamomile's `subspace_hamiltonian(...)` to construct the matrix in the full basis. Both the reference and QSCI calculations use the same code to construct their matrices, so this comparison alone does not independently validate that code.

# %% [markdown]
# ## Summary
#
# - We measure the input state in the computational basis and select basis states from the most frequently observed bitstrings.
# - For the selected basis states, we classically calculate matrix elements of the known Hamiltonian and diagonalize it in the subspace.
# - The lowest eigenvalue of a correct projection of the same Hamiltonian provides a variational upper bound on the full-space ground-state energy. This does not guarantee good accuracy.
# - In Qamomile, we can share a state-preparation kernel between VQE and sampling, then pass the selected bitstrings and Hamiltonian to `solve_subspace(...)`.
#
# We have demonstrated the QSCI workflow on a small model for which a full 16-dimensional reference calculation is feasible. This run alone does not establish quantum advantage or accuracy under physical-device noise.
