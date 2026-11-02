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
# ---
# tags: [algorithm, sample-based]
# ---
#
# # Quantum Selected Configuration Interaction (QSCI)
#
# **QSCI** is a hybrid quantum–classical algorithm that uses bitstrings
# sampled from a quantum state to build a small effective Hamiltonian
# and then diagonalizes it exactly on a classical computer. The
# advantage over plain VQE is that the result inherits a strict
# **variational guarantee**: even on noisy hardware,
#
# $$
# E_{\mathrm{QSCI}} \;\geq\; E_{\mathrm{exact}}.
# $$
#
# The four-step recipe introduced by Kanno *et al.* {cite:p}`10.48550/arXiv.2302.11320` is:
#
# 1. Prepare an input state $|\psi_{\mathrm{in}}\rangle$ on the quantum
#    computer (typically a roughly optimised VQE state).
# 2. Measure $|\psi_{\mathrm{in}}\rangle$ in the computational basis many
#    times.
# 3. Pick the top-$K$ most-frequent bitstrings as a discrete subspace
#    $\{|s_i\rangle\}_{i=1}^{K}$.
# 4. Build the effective Hamiltonian
#    $H^{\mathrm{sub}}_{ij} = \langle s_i | H | s_j \rangle$ and
#    diagonalize it classically.
#
# This tutorial walks through the full flow on a four-qubit transverse-
# field Ising model. The quantum state preparation and sampling run on
# the **QURI Parts** integration (Qulacs simulator); the subspace
# construction and diagonalization use
# `qamomile.linalg.solve_subspace`, which internally calls the
# vectorised Z-basis routine `subspace_hamiltonian` (XOR / parity, no
# matrix products).

# %% [markdown]
# ## Installation
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

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
# ## A small test Hamiltonian
#
# We use the four-qubit transverse-field Ising model on an open chain:
#
# $$
# H \;=\; -J \sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h \sum_{i=0}^{n-1} X_i,
# \quad J = 1,\; h = 0.7.
# $$
#
# The 16-dimensional Hilbert space is small enough that we can compute
# the exact ground-state energy directly with NumPy and use it as the
# QSCI reference.

# %%
N_QUBITS = 4
FIELD = 0.7
HAMILTONIAN = qm.Hamiltonian.zero(num_qubits=N_QUBITS)
for index in range(N_QUBITS - 1):
    HAMILTONIAN = HAMILTONIAN - qm.Z(index) * qm.Z(index + 1)
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
# ## Hardware-efficient Ansatz
#
# A simple alternating-layer ansatz applies RY to every quantum bit, followed
# by a linear CX chain, twice, and finishes with one more RY layer. The three
# layers use twelve runtime parameters in total. We define a state-returning
# program for VQE and a measurement program for QSCI sampling.


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
# ## Compile Both Programs with QURI Parts
#
# The specialized program contains ordinary calls and no QSCI-specific
# operation. Rust derives twelve rotations and six CX gates from the visible
# body. The QURI Parts artifact retains only `thetas` in its runtime ABI.

# %%
program = qm.specialize(qsci_samples)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.call" in inspection
assert "quantum.qsci" not in inspection
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (18, 18)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

sample_endpoint = Qulacs()
artifact = qm.compile(program, to=QuriParts())
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("thetas",)
assert artifact.format == "quri-parts.logical-circuit.v2"

# %% [markdown]
# ## Step 1: Prepare $|\psi_{\mathrm{in}}\rangle$ via a quick VQE
#
# QSCI is robust to a poorly optimised input state — even random
# parameters give a meaningful subspace — but a short VQE run
# concentrates the sampling distribution on the bitstrings that
# dominate the true ground state, making the subspace much more
# informative for a given $K$. We run only a handful of COBYLA
# iterations.

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
expectation_endpoint = qm.Reference()
vqe_shots = 4 if docs_test_mode else 128
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
max_iterations = 14 if docs_test_mode else 80
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)
optimized_parameters = tuple(float(value) for value in optimization.x)

print(
    f"VQE energy = {optimization.fun:+.6f} "
    f"(gap to E_exact: {optimization.fun - EXACT_ENERGY:+.4e})"
)

assert np.isfinite(optimization.fun)
assert len(optimized_parameters) == 12
assert len(cost_history) == optimization.nfev

# %% [markdown]
# Finite-shot energy estimates can fall below the exact ground energy due to sampling noise. The variational bound applies to the true expectation, not every estimate or the minimum selected during optimization. The exact subspace diagonalization below has a different error source.

# %% [markdown]
# ## Step 2: Sample bitstrings in the Z basis
#
# Each sample is a tuple `(b_0, ..., b_{n-1})` whose $q$-th entry is
# the Z-eigenvalue index of qubit $q$ — exactly the format
# `subspace_hamiltonian` expects.

# %%
SHOTS = 128 if docs_test_mode else 4_000

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
    for bits, _ in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )
)

print("sample counts:", counts)

assert record.artifact is artifact
assert sum(counts.values()) == SHOTS
assert len(configurations) <= 2**N_QUBITS
assert all(len(bits) == N_QUBITS for bits in configurations)

# %% [markdown]
# ## Step 3 + 4: Build the QSCI subspace and diagonalize
#
# For each subspace size $K$ we feed the $K$ most-frequent bitstrings
# to `solve_subspace`, which builds
# $H^{\mathrm{sub}}_{ij} = \langle s_i|H|s_j\rangle$ via a vectorised
# XOR / parity routine and runs `numpy.linalg.eigh`. The lowest
# returned eigenvalue is the QSCI energy estimate, and the variational
# principle guarantees $E_{\mathrm{QSCI}}(K) \geq E_{\mathrm{exact}}$
# for every $K$.

# %%
SUBSPACE_SIZES = tuple(
    size
    for size in (1, 2, 4, 8, 16, len(configurations))
    if size <= len(configurations)
)
energies = tuple(
    float(qm.linalg.solve_subspace(configurations[:size], HAMILTONIAN)[0][0])
    for size in SUBSPACE_SIZES
)
for size, energy in zip(SUBSPACE_SIZES, energies):
    print(f"K={size}: E_QSCI={energy:+.12f}")
print(f"E_exact={EXACT_ENERGY:+.12f}")

assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
assert all(energy >= EXACT_ENERGY - 1e-12 for energy in energies)

# %%
plt.figure(figsize=(6, 4))
plt.plot(SUBSPACE_SIZES, energies, "-o", label=r"$E_{\mathrm{QSCI}}$")
plt.axhline(EXACT_ENERGY, color="black", linestyle="--", label=r"$E_{\mathrm{exact}}$")
plt.xlabel("Subspace size $K$")
plt.ylabel("Energy")
plt.title("QSCI convergence — 4-qubit TFIM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - `Q[Bits[4]]`, `Param[Vector[Real, 12]]`, and `Shot[Bits[4]]` make ownership,
#   staging, and result width explicit.
# - One source defines inspection, exact logical resources, compilation, and
#   sampling; QSCI adds no compiler operation or backend branch.
# - Rust projects the same immutable Hamiltonian onto the sampled states before
#   the host eigensolve.
# - A continuous COBYLA VQE selects the state; QURI Parts/Qulacs sampling builds
#   a subspace no larger than the full 16-state Hilbert space.

# %% [markdown]
# ## Notes
#
# - The Z-basis fast path (`subspace_hamiltonian`) used inside
#   `solve_subspace` requires no matrix multiplication: each Pauli
#   term contributes a single XOR mask and parity sign, vectorised
#   across all $K^2$ sample pairs.
# - Duplicate sampled bitstrings drop out of the unique-bitstring
#   list above; the resulting subspace is well-conditioned and
#   `solve_subspace` returns an ordinary Hermitian eigendecomposition.
