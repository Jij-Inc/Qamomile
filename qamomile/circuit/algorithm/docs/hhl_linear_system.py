# ---
# jupyter:
#   jupytext:
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
# # HHL Algorithm: Solving Linear Systems on a Quantum Computer
#
# This tutorial walks through the Harrow-Hassidim-Lloyd (HHL) algorithm
# step by step, using Qamomile's low-level circuit primitives.
# Rather than using the high-level `hhl()` function directly, we will:
#
# 1. Review the theory behind the HHL algorithm.
# 2. Set up a concrete diagonal linear system.
# 3. Build every component of the HHL circuit by hand — forward QPE,
#    reciprocal rotation, and inverse QPE.
# 4. Simulate with a statevector backend and verify the result against the
#    classical solution.
# 5. Show that `qamomile.circuit.algorithm.hhl` provides the same circuit
#    in a single function call.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %% [markdown]
# ## The HHL Algorithm
#
# Given a Hermitian matrix $A$ and a vector $|b\rangle$, the HHL algorithm
# prepares a quantum state proportional to $A^{-1}|b\rangle$.
# The algorithm proceeds in four stages:
#
# ### 1. Quantum Phase Estimation (QPE)
#
# Expand $|b\rangle$ in the eigenbasis of $A$:
# $|b\rangle = \sum_j \beta_j |u_j\rangle$ where $A|u_j\rangle = \lambda_j |u_j\rangle$.
#
# QPE encodes the eigenvalues into a clock register:
#
# $$
# |0\rangle_c |b\rangle_s |0\rangle_a
# \;\xrightarrow{\text{QPE}}\;
# \sum_j \beta_j |\tilde\lambda_j\rangle_c |u_j\rangle_s |0\rangle_a
# $$
#
# ### 2. Reciprocal Rotation
#
# A controlled rotation embeds $C / \tilde\lambda_j$ into the ancilla amplitude:
#
# $$
# \xrightarrow{\text{Reciprocal}}\;
# \sum_j \beta_j |\tilde\lambda_j\rangle_c |u_j\rangle_s
# \left(
#   \sqrt{1 - \frac{C^2}{\tilde\lambda_j^2}}\,|0\rangle
#   + \frac{C}{\tilde\lambda_j}\,|1\rangle
# \right)_a
# $$
#
# ### 3. Inverse QPE
#
# Restores the clock register back to $|0\rangle_c$.
#
# ### 4. Post-selection
#
# Measuring the ancilla in $|1\rangle$ projects the system register onto:
#
# $$
# C \sum_j \frac{\beta_j}{\tilde\lambda_j} |u_j\rangle_s
# \;\propto\; A^{-1}|b\rangle
# $$

# %% [markdown]
# ## Problem Setup: A Diagonal Linear System
#
# We use the single-qubit gate $U = R_z(\pi)$ as our Hamiltonian simulation
# unitary.  $R_z(\alpha)$ has eigenstates $|0\rangle$ and $|1\rangle$ with
# eigenvalues $e^{-i\alpha/2}$ and $e^{+i\alpha/2}$ respectively:
#
# $$
# R_z(\pi) = e^{-i\frac{\pi}{2}Z}
# = \begin{pmatrix} e^{-i\pi/2} & 0 \\ 0 & e^{i\pi/2} \end{pmatrix}
# = \begin{pmatrix} -i & 0 \\ 0 & i \end{pmatrix}
# $$
#
# With 2 clock qubits and `phase_scale` $= 2\pi$ (unsigned mode), QPE maps
# these eigenvalues to clock register bins:
#
# | Eigenstate | Eigenvalue | Eigenphase $\varphi$ | Raw bin | $\hat\lambda = 2\pi \cdot \text{raw}/4$ |
# |:---:|:---:|:---:|:---:|:---:|
# | $\|0\rangle$ | $e^{-i\pi/2}$ | $-\tfrac{1}{4} \bmod 1 = \tfrac{3}{4}$ | 3 | $\tfrac{3\pi}{2}$ |
# | $\|1\rangle$ | $e^{+i\pi/2}$ | $\tfrac{1}{4}$ | 1 | $\tfrac{\pi}{2}$ |
#
# Under the unsigned phase decoding convention used here
# (`phase_scale` $= 2\pi$, mapping raw bins to $[0, 2\pi)$),
# the effective diagonal matrix that HHL inverts is
# $A = \mathrm{diag}(3\pi/2,\; \pi/2)$.
# Note that $3\pi/2$ arises because the negative eigenphase
# $-1/4$ wraps to $3/4$ modulo 1.
#
# We choose $|b\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, so the
# exact solution is:
#
# $$
# A^{-1}|b\rangle \;\propto\;
# \frac{1}{3\pi/2}|0\rangle + \frac{1}{\pi/2}|1\rangle
# \;\propto\; \tfrac{1}{3}|0\rangle + |1\rangle
# $$

# %%
import math

import numpy as np

# Problem parameters
alpha_val = math.pi
C = 0.4
phase_scale = 2.0 * math.pi

# Input vector |b> (will be normalized by amplitude_encoding)
b_amplitudes = [1.0, 1.0]

# Eigenvalues decoded by QPE
lambda_0 = 3 * math.pi / 2  # |0> : raw bin 3
lambda_1 = math.pi / 2  # |1> : raw bin 1

# Classical exact solution
b_norm = np.array(b_amplitudes) / np.linalg.norm(b_amplitudes)
x_exact = np.array([b_norm[0] / lambda_0, b_norm[1] / lambda_1])
x_exact_normalized = x_exact / np.linalg.norm(x_exact)

print(f"Eigenvalues: lambda_0 = {lambda_0:.4f}, lambda_1 = {lambda_1:.4f}")
print(f"|b> (normalized): {b_norm}")
print(f"A^{{-1}}|b> (unnormalized): [{b_norm[0]/lambda_0:.4f}, {b_norm[1]/lambda_1:.4f}]")
print(f"A^{{-1}}|b> (normalized):   {x_exact_normalized}")

# %% [markdown]
# ## Building HHL from Scratch
#
# Before using the built-in `hhl()` function, let's construct each
# component manually to understand the algorithm's structure.

# %% [markdown]
# ### Step 1: Define the Unitary Kernels
#
# HHL requires the unitary $U$ and its adjoint $U^\dagger$ as `@qkernel`
# functions.  The `qmc.controlled()` wrapper with `power`$=2^k$ creates
# the controlled-$U^{2^k}$ operations needed by QPE.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.algorithm.mottonen_amplitude_encoding import amplitude_encoding
from qamomile.circuit.stdlib.qft import iqft, qft


# fmt: off
@qmc.qkernel
def rz_unitary(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """U = Rz(alpha)."""
    return qmc.rz(q, alpha)


@qmc.qkernel
def rz_unitary_inv(q: qmc.Qubit, alpha: qmc.Float) -> qmc.Qubit:
    """U-dagger = Rz(-alpha)."""
    return qmc.rz(q, -1.0 * alpha)


@qmc.qkernel
def ry_gate(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Single RY gate (wrapper for creating multi-controlled version)."""
    return qmc.ry(q, angle)
# fmt: on

# %% [markdown]
# ### Step 2: Forward QPE
#
# Forward QPE encodes eigenvalues into the clock register:
#
# 1. Apply Hadamard to all clock qubits (create equal superposition).
# 2. Apply controlled-$U^{2^k}$: clock qubit $k$ controls $U^{2^k}$.
# 3. Apply the inverse QFT to convert phase information into
#    computational basis states.
#
# With 2 clock qubits and little-endian convention (`clock[0]` = LSB):
# - `clock[0]` controls $U^1$
# - `clock[1]` controls $U^2$

# %% [markdown]
# ### Step 3: Reciprocal Rotation
#
# For each eigenvalue bin, we compute the rotation angle
# $\theta = 2 \arcsin(C / \hat\lambda)$ and apply a multi-controlled
# $R_Y(\theta)$ to the ancilla, conditioned on the clock register being
# in the corresponding basis state.
#
# For $R_z(\pi)$ with 2 clock qubits, only bins 1 and 3 are populated:
#
# - **Bin 1** (clock = $|01\rangle$): $\hat\lambda_1 = \pi/2$,
#   $\theta_1 = 2\arcsin(0.4 / (\pi/2)) \approx 0.519$
# - **Bin 3** (clock = $|11\rangle$): $\hat\lambda_3 = 3\pi/2$,
#   $\theta_3 = 2\arcsin(0.4 / (3\pi/2)) \approx 0.170$
#
# To select a specific basis state, we flip qubits that should be $|0\rangle$
# using X gates, so that all multi-controlled RY controls see $|1\rangle$.

# %% [markdown]
# ### Step 4: Inverse QPE
#
# Mirror the forward QPE in reverse order:
# QFT → controlled-$U^{\dagger 2^k}$ (reverse qubit order) → Hadamard.

# %% [markdown]
# ### Complete Naive HHL Circuit
#
# Composing all three steps into a single `@qkernel`:

# %%
# Precompute rotation angles for reciprocal rotation
theta_bin1 = 2.0 * math.asin(C / lambda_1)  # bin 1: lambda = pi/2
theta_bin3 = 2.0 * math.asin(C / lambda_0)  # bin 3: lambda = 3*pi/2

print(f"Reciprocal rotation angles:")
print(f"  Bin 1 (lambda={lambda_1:.4f}): theta = {theta_bin1:.6f}")
print(f"  Bin 3 (lambda={lambda_0:.4f}): theta = {theta_bin3:.6f}")


# %%
@qmc.qkernel
def hhl_naive(alpha: qmc.Float) -> qmc.Bit:
    # --- Allocate registers ---
    sys = qmc.qubit_array(1, name="sys")
    sys = amplitude_encoding(sys, b_amplitudes)
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")

    # === Step 1: Forward QPE ===
    # 1a. Hadamard on all clock qubits
    clock[0] = qmc.h(clock[0])
    clock[1] = qmc.h(clock[1])

    # 1b. Controlled-U^(2^k) operations
    # clock[0] (LSB) controls U^1, clock[1] controls U^2
    controlled_u = qmc.controlled(rz_unitary)
    clock[0], sys[0] = controlled_u(clock[0], sys[0], power=1, alpha=alpha)
    clock[1], sys[0] = controlled_u(clock[1], sys[0], power=2, alpha=alpha)

    # 1c. Inverse QFT on clock register
    clock = iqft(clock)

    # === Step 2: Reciprocal Rotation ===
    mc_ry = qmc.controlled(ry_gate, num_controls=2)

    # Bin 1 (clock = |01>): clock[0]=1, clock[1]=0
    # Flip clock[1] so both controls see |1>
    clock[1] = qmc.x(clock[1])
    clock[0], clock[1], anc = mc_ry(
        clock[0], clock[1], anc, angle=theta_bin1
    )
    clock[1] = qmc.x(clock[1])  # undo flip

    # Bin 3 (clock = |11>): clock[0]=1, clock[1]=1
    # No flips needed — both controls are already |1>
    clock[0], clock[1], anc = mc_ry(
        clock[0], clock[1], anc, angle=theta_bin3
    )

    # === Step 3: Inverse QPE ===
    # 3a. QFT on clock (inverse of IQFT)
    clock = qft(clock)

    # 3b. Controlled-U-dagger in reverse qubit order
    controlled_u_inv = qmc.controlled(rz_unitary_inv)
    clock[1], sys[0] = controlled_u_inv(
        clock[1], sys[0], power=2, alpha=alpha
    )
    clock[0], sys[0] = controlled_u_inv(
        clock[0], sys[0], power=1, alpha=alpha
    )

    # 3c. Hadamard on all clock qubits
    clock[0] = qmc.h(clock[0])
    clock[1] = qmc.h(clock[1])

    return qmc.measure(anc)


# %% [markdown]
# ## Simulate the Naive HHL Circuit

# %%
import matplotlib.pyplot as plt
from qiskit import transpile as qk_transpile
from qiskit_aer import AerSimulator

from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()
simulator = AerSimulator(method="statevector")


def simulate_hhl(kernel, alpha):
    """Transpile, simulate, and extract post-selected system state."""
    exe = transpiler.transpile(kernel, bindings={"alpha": alpha})
    qc = exe.compiled_quantum[0].circuit.copy()
    qc.remove_final_measurements()
    qc_compiled = qk_transpile(qc, simulator)
    qc_compiled.save_statevector()
    result = simulator.run(qc_compiled).result()
    sv = np.array(result.get_statevector())
    return sv


def extract_postselected_system(sv, n_system, n_clock):
    """Extract system amplitudes conditioned on ancilla=|1> and clock=|00...0>.

    Qubit allocation order: sys[0..n_sys-1], clock[0..n_clk-1], anc.
    Qiskit little-endian: statevector index = q0 + q1*2 + q2*4 + ...
    """
    n_total = n_system + n_clock + 1
    anc_pos = n_system + n_clock
    system_amps = np.zeros(2**n_system, dtype=complex)

    for idx in range(len(sv)):
        if not ((idx >> anc_pos) & 1):
            continue
        clock_zero = all(
            not ((idx >> (n_system + c)) & 1) for c in range(n_clock)
        )
        if not clock_zero:
            continue
        sys_val = idx & ((1 << n_system) - 1)
        system_amps[sys_val] = sv[idx]

    return system_amps


# %%
sv_naive = simulate_hhl(hhl_naive, alpha_val)
sys_naive = extract_postselected_system(sv_naive, n_system=1, n_clock=2)
norm_naive = np.linalg.norm(sys_naive)

print("=== Naive HHL Results ===")
print(f"Post-selection probability: {norm_naive**2:.6f}")
print(f"Post-selected system state: {sys_naive / norm_naive}")

fid_naive = float(
    np.abs(np.vdot(x_exact_normalized, sys_naive / norm_naive)) ** 2
)
print(f"Fidelity with exact A^{{-1}}|b>: {fid_naive:.6f}")

# %% [markdown]
# ## Using the Built-in `hhl()`
#
# Everything we implemented above — forward QPE, reciprocal rotation,
# and inverse QPE — is already provided by
# `qamomile.circuit.algorithm.hhl`.  It accepts the same unitary kernels
# and handles the eigenvalue decoding, bin selection, and multi-controlled
# rotations automatically.
#
# Let's build the same circuit using the built-in function to confirm
# that it produces identical results.

# %%
from qamomile.circuit.algorithm.hhl import hhl


@qmc.qkernel
def hhl_builtin(alpha: qmc.Float) -> qmc.Bit:
    sys = qmc.qubit_array(1, name="sys")
    sys = amplitude_encoding(sys, b_amplitudes)
    clock = qmc.qubit_array(2, name="clock")
    anc = qmc.qubit("anc")

    sys, clock, anc = hhl(
        sys,
        clock,
        anc,
        unitary=rz_unitary,
        inv_unitary=rz_unitary_inv,
        scaling=C,
        phase_scale=phase_scale,
        supported_raw_bins=(1, 3),
        strict=True,
        alpha=alpha,
    )

    return qmc.measure(anc)


# %%
sv_builtin = simulate_hhl(hhl_builtin, alpha_val)
sys_builtin = extract_postselected_system(sv_builtin, n_system=1, n_clock=2)
norm_builtin = np.linalg.norm(sys_builtin)

print("=== Built-in hhl() Results ===")
print(f"Post-selection probability: {norm_builtin**2:.6f}")
print(f"Post-selected system state: {sys_builtin / norm_builtin}")

fid_builtin = float(
    np.abs(np.vdot(x_exact_normalized, sys_builtin / norm_builtin)) ** 2
)
print(f"Fidelity with exact A^{{-1}}|b>: {fid_builtin:.6f}")

# %% [markdown]
# Both circuits should produce identical fidelity because the built-in
# `hhl()` implements the same algorithm we built by hand.

# %%
# Verify post-selection probability against theory
p_expected = (
    abs(b_norm[0]) ** 2 * (C / lambda_0) ** 2
    + abs(b_norm[1]) ** 2 * (C / lambda_1) ** 2
)
print(f"\nTheoretical post-selection probability: {p_expected:.6f}")
print(f"Naive   post-selection probability:     {norm_naive**2:.6f}")
print(f"Built-in post-selection probability:    {norm_builtin**2:.6f}")

# %% [markdown]
# ## Visualization
#
# The bar chart below compares the probability distribution of the HHL
# output state with the exact classical solution.

# %%
quantum_probs = np.abs(sys_builtin / norm_builtin) ** 2
classical_probs = np.abs(x_exact_normalized) ** 2

labels = ["|0>", "|1>"]
x_pos = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(
    x_pos - width / 2,
    quantum_probs,
    width,
    label="HHL (quantum)",
    color="#2696EB",
)
bars2 = ax.bar(
    x_pos + width / 2,
    classical_probs,
    width,
    label=r"Exact $A^{-1}|b\rangle$",
    color="#FF6B6B",
)

ax.set_xlabel("Basis state")
ax.set_ylabel("Probability")
ax.set_title("HHL Result vs Exact Solution")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.0)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Different Input Vectors
#
# The HHL circuit works for any input $|b\rangle$.  Below we verify the
# fidelity for several choices using the built-in `hhl()`.

# %%
test_cases = [
    ([1.0, 1.0], "uniform"),
    ([1.0, 2.0], "asymmetric"),
    ([1.0, 0.0], "basis |0>"),
    ([0.0, 1.0], "basis |1>"),
]

for amps, label in test_cases:

    @qmc.qkernel
    def _circuit(alpha: qmc.Float) -> qmc.Bit:
        sys = qmc.qubit_array(1, name="sys")
        sys = amplitude_encoding(sys, amps)
        clock = qmc.qubit_array(2, name="clock")
        anc = qmc.qubit("anc")
        sys, clock, anc = hhl(
            sys,
            clock,
            anc,
            unitary=rz_unitary,
            inv_unitary=rz_unitary_inv,
            scaling=C,
            phase_scale=phase_scale,
            supported_raw_bins=(1, 3),
            strict=True,
            alpha=alpha,
        )
        return qmc.measure(anc)

    sv_i = simulate_hhl(_circuit, alpha_val)
    sys_i = extract_postselected_system(sv_i, 1, 2)

    b_n = np.array(amps) / np.linalg.norm(amps)
    x_ex = np.array([b_n[0] / lambda_0, b_n[1] / lambda_1])
    x_ex_norm = np.linalg.norm(x_ex)

    if x_ex_norm > 1e-15 and np.linalg.norm(sys_i) > 1e-15:
        f_i = float(
            np.abs(
                np.vdot(x_ex / x_ex_norm, sys_i / np.linalg.norm(sys_i))
            )
            ** 2
        )
    else:
        f_i = (
            1.0
            if x_ex_norm < 1e-15 and np.linalg.norm(sys_i) < 1e-15
            else 0.0
        )

    print(
        f"|b> = {str(amps):>12s}  ({label:>12s}):  fidelity = {f_i:.6f}"
    )
