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
# tags: [algorithm, primitive, resource-estimation]
# ---
#
# # Quantum Phase Estimation (QPE)
#
# Quantum Phase Estimation (QPE) estimates the eigenphase $\phi$ of a unitary
# matrix $U$ from an eigenstate $|\psi\rangle$ satisfying
# $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$. It is a central primitive in
# Shor's algorithm and other algorithms that use phases encoded by the
# eigenvalues of unitary matrices
# {cite:p}`10.48550/arXiv.quant-ph/9511026,10.1098/rspa.1998.0164`.
#
# This notebook implements the QPE procedure as a Qamomile qkernel and compares
# it with the built-in `phase_estimation` function. It then explores the relationship
# between the number of counting qubits, phase-estimation precision, and gate
# count.

# %%
# Install the latest Qamomile and the extras used by this notebook.
# # !pip install "qamomile[qiskit,visualization]"

# %%
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

import qamomile as qm
from qamomile.qiskit import Aer
from qamomile.stdlib import inverse_qft, phase_estimation
from qamomile.visualization import draw

endpoint = Aer()

# %% [markdown]
# ## Background: Phase Kickback and Quantum Fourier Transform
#
# QPE combines two ideas: phase kickback from controlled unitaries
# {cite:p}`10.1098/rspa.1998.0164` and the Quantum Fourier Transform
# {cite:p}`10.48550/arXiv.quant-ph/0201067`.
#
# ### Phase Kickback
#
# Phase kickback is the mechanism by which, when a controlled-$U$ gate is
# applied to an eigenstate $|\psi\rangle$ of a unitary matrix $U$, the
# eigenphase of $U$ appears as a relative phase on the control qubit. Suppose
#
# $$
# U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle.
# $$
#
# Prepare one control qubit in a superposition and the target quantum state in
# $|\psi\rangle$:
#
# $$
# \frac{|0\rangle + |1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# A controlled-$U$ gate applies $U$ only to the component in which the control
# qubit is $|1\rangle$. Because $|\psi\rangle$ is an eigenstate of $U$, the
# eigenphase appears on that component:
#
# $$
# \frac{|0\rangle|\psi\rangle + |1\rangle U|\psi\rangle}{\sqrt{2}}
# =
# \frac{|0\rangle + e^{2\pi i\phi}|1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# This transformation leaves the target quantum state in $|\psi\rangle$ and
# encodes the eigenphase in the relative phase of the control qubit. QPE
# deliberately uses this mechanism with multiple control qubits. The Algorithm
# section explains the encoding in detail.
#
# ### Quantum Fourier Transform
#
# The Quantum Fourier Transform (QFT) maps a computational-basis state to a
# superposition whose relative phases encode that state's value. On $M=2^m$
# basis states, the QFT is defined as
#
# $$
# \mathrm{QFT}_M|x\rangle
# = \frac{1}{\sqrt{M}}
#   \sum_{y=0}^{M-1} e^{2\pi i xy/M}|y\rangle.
# $$
#
# It maps an integer $x$ to a regular phase pattern across the computational
# basis. We refer to a state that stores a value in these relative phases as a
# Fourier-encoded state. For an application in Qamomile, see the
# [multidimensional QFT example](multidimensional_qft.ipynb). The inverse QFT
# $\mathrm{QFT}_M^{-1}$ performs the reverse operation. If $\phi=a/M$ is
# exactly representable with $m$ bits, then the
# controlled powers $U^{2^0}, U^{2^1}, \ldots, U^{2^{m-1}}$ prepare the
# counting register as
#
# $$
# \frac{1}{\sqrt{M}}\sum_{k=0}^{M-1} e^{2\pi i a k/M}|k\rangle
# = \mathrm{QFT}_M|a\rangle.
# $$
#
# Applying $\mathrm{QFT}_M^{-1}$ then returns $|a\rangle$. The important point
# is that the inverse QFT turns phase information into a computational-basis bit
# string.
#
# $$
# \underbrace{
# \frac{1}{\sqrt{M}}
# \begin{pmatrix}
# 1 \\
# e^{2\pi i a/M} \\
# \vdots \\
# e^{2\pi i a(M-1)/M}
# \end{pmatrix}
# }_{\text{vector of phase values}}
# \xrightarrow{\mathrm{QFT}_M^{-1}}
# \underbrace{|a\rangle = |a_{m-1}\cdots a_0\rangle}_{\text{computational-basis bit string}}.
# $$

# %% [markdown]
# ## Algorithm
#
# QPE uses two groups of qubits: $m$ counting qubits that read out the phase and
# target qubits that hold an eigenstate $|\psi\rangle$ of $U$. The counting
# qubits are auxiliary qubits used to represent the phase as an $m$-bit binary
# fraction. Suppose $U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle$. The counting
# qubits start in $|0\rangle^{\otimes m}$, and we write $M=2^m$.
#
# :::{note} Superpositions of Eigenstates
# More generally, the target-register input does not have to be a single
# eigenstate. If it is a superposition of eigenstates, QPE measures one of the
# corresponding eigenphases with probability determined by that eigenstate's
# weight in the input state. This tutorial uses one known eigenstate so the
# phase-kickback algebra and sampled output are easy to read.
# :::
#
# ### Step 1: Put the counting qubits in superposition
#
# Apply Hadamard gates to the counting qubits. This creates a uniform
# superposition over the $M$ possible values.
#
# $$
# |\Psi_1\rangle =
# H^{\otimes m}|0\rangle^{\otimes m}|\psi\rangle
# =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}|r\rangle|\psi\rangle.
# $$
#
# ### Step 2: Apply controlled-$U^{2^k}$ gates
#
# Use each counting qubit $k$ as the control qubit for a controlled-$U^{2^k}$
# gate. If $r=\sum_{k=0}^{m-1} r_k2^k$, the target eigenstate picks up the phase
# $e^{2\pi i\phi r}$:
#
# $$
# |\Psi_2\rangle =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}
# e^{2\pi i \phi r}|r\rangle|\psi\rangle.
# $$
#
# ### Step 3: Convert phase information into a bit string with inverse QFT
#
# If $\phi=a/M$ is exactly representable, the counting register is
# $\mathrm{QFT}_M|a\rangle$. Applying the inverse QFT returns $|a\rangle$.
#
# $$
# |\Psi_2\rangle =
# \left(
#   \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1} e^{2\pi i ar/M}|r\rangle
# \right)|\psi\rangle
# =
# \mathrm{QFT}_M|a\rangle|\psi\rangle,
# $$
#
# $$
# |\Psi_3\rangle =
# \mathrm{QFT}_M^{-1}\mathrm{QFT}_M|a\rangle|\psi\rangle
# =
# |a\rangle|\psi\rangle.
# $$
#
# ### Step 4: Measure the bit string and decode the phase
#
# Measure the counting qubits. In the exact case, the measurement returns $a$,
# and decoding this bit string gives the phase estimate
#
# $$
# \tilde{\phi} = \frac{a}{M}.
# $$
#
# If $\phi$ is not exactly representable with $m$ bits, the distribution
# concentrates around the closest $m$-bit approximations. Increasing $m$
# reduces the spacing between representable phases to $1/2^m$, allowing the
# estimate to approach the true eigenphase.
#
# :::{note} Binary Fractions and Precision
# The number of fractional bits determines the phase resolution available to
# QPE. For example, $\phi=0.6$ is not represented exactly on a binary grid. With
# two fractional bits, the closest binary fraction is
#
# $$
# 0.10_2 = \frac{1}{2^1} + \frac{0}{2^2} = 0.5
# $$
#
# With three fractional bits, the grid is finer and gives a closer value:
#
# $$
# 0.101_2
# = \frac{1}{2^1} + \frac{0}{2^2} + \frac{1}{2^3}
# = 0.625
# $$
#
# Increasing the number of counting qubits therefore increases the number of
# binary-fraction bits used to report the phase estimate.
# :::
#
# ```{figure} assets/qpe_circuit.png
# :alt: Quantum phase estimation circuit with a counting register, controlled powers of U, inverse QFT, and measurement.
# :width: 720px
#
# Schematic QPE circuit. The counting register controls the powers of $U$, and
# the inverse QFT converts the accumulated phase pattern into measured bits.
# ```

# %% [markdown]
# ## Implementation with Qamomile
#
# We use a diagonal 4x4 unitary matrix:
#
# $$
# U =
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & e^{i\theta_{01}} & 0 & 0 \\
# 0 & 0 & e^{i\theta_{10}} & 0 \\
# 0 & 0 & 0 & e^{i\theta_{11}}
# \end{pmatrix}.
# $$
#
# Every computational-basis state is an eigenstate of this matrix. We prepare
# the target state $|01\rangle$ and set the phase to estimate,
# $\theta_{01} / 2\pi$, to $0.6$. Because $0.6$ is not exactly representable on
# a finite binary fraction, increasing the number of counting qubits visibly
# improves the approximation.

# %%
# Set the sampling settings and target eigenstate.
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
SHOTS = 512 if docs_test_mode else 4096
SAMPLER_SEED = 42

# Set the phases of the diagonal unitary matrix and the target phase.
TARGET_PHASE_FRACTION = 0.6
phase_fractions = np.array([0.0, TARGET_PHASE_FRACTION, 0.23, 0.81])
phase_angles = 2 * math.pi * phase_fractions

# Convert the phase fractions to a unitary matrix and check unitarity.
unitary = np.diag(np.exp(1j * phase_angles))
assert np.allclose(unitary.conj().T @ unitary, np.eye(4))

# Store the concrete phase parameters used by the qkernel.
PHI_01 = float(phase_angles[1])
PHI_10 = float(phase_angles[2])
PHI_11 = float(phase_angles[3])

# Print the problem instance for reference.
print("phase fractions:", np.round(phase_fractions, 6))
print("target phase fraction:", f"{TARGET_PHASE_FRACTION:.8f}")
print("U =")
print(np.round(unitary, 3))
assert 0.0 <= TARGET_PHASE_FRACTION < 1.0

# %% [markdown]
# ### From-Scratch Implementation
#
# First, we define the 4x4 unitary matrix whose phase we want to estimate. The
# `diagonal_4x4` qkernel implements this matrix directly. The phase gate
# $P(\theta)$ multiplies the $|1\rangle$ component of a qubit by $e^{i\theta}$;
# Qamomile expresses this gate as `qm.p(q, theta, target=index)`. Applying
# `qm.p(q, PHI_10, target=0)` therefore gives every basis
# state whose first target bit is 1 the phase $e^{i\theta_{10}}$, and
# `qm.p(q, PHI_01, target=1)` gives every basis state whose second target bit is 1
# the phase $e^{i\theta_{01}}$. At that point $|11\rangle$ has accumulated
# $e^{i(\theta_{10}+\theta_{01})}$, so the controlled phase gate adds only the
# correction
#
# $$
# \theta_{11} - \theta_{10} - \theta_{01}
# $$
#
# on $|11\rangle$. The resulting diagonal is exactly
# $\operatorname{diag}(1, e^{i\theta_{01}}, e^{i\theta_{10}},
# e^{i\theta_{11}})$ in the ordered basis
# $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.


# %%
@qm.qkernel
def diagonal_4x4(q: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Apply the diagonal unitary with the phase constants defined above."""
    q = qm.p(q, PHI_10, target=0)
    q = qm.p(q, PHI_01, target=1)
    q = qm.cp(q, PHI_11 - PHI_10 - PHI_01, control=0, target=1)
    return q


display(HTML(draw(qm.specialize(diagonal_4x4), loops="expanded").to_html()))

# %% [markdown]
# We now implement the four QPE steps using Qamomile's basic operations.
# `qm.control(diagonal_4x4, counting, target, at=k)` applies controlled-$U$
# using counting qubit $k$. Repeating it `2**k` times applies
# controlled-$U^{2^k}$. After the inverse QFT, we reverse the little-endian
# Fourier register into most-significant-first fixed-point digits and use
# `qm.as_fixed` to view it as `UFixed[0, counting_bits]`, with no integer bits.
# `qm.measure` measures those qubits and decodes the resulting bit string into
# a numeric phase estimate. We also measure the retained target register to
# consume every quantum value; its result is not returned.


# %%
@qm.qkernel
def qpe_from_scratch(
    counting_bits: qm.Const[qm.Int],
) -> qm.Shot[qm.UFixed[0, counting_bits]]:
    """Read the eigenphase using explicit phase kickback and inverse QFT."""
    counting = qm.qubits(counting_bits)
    target = qm.prepare((0, 1))
    for k in range(counting_bits):
        counting = qm.h(counting, target=k)

    system = (counting, target)
    for k in range(counting_bits):
        for _ in range(2**k):
            counting, target = system
            counting, target = qm.control(diagonal_4x4, counting, target, at=k)
            system = (counting, target)

    counting, target = system
    counting = inverse_qft(counting, counting_bits)
    for left in range(counting_bits // 2):
        counting = qm.swap(counting, left=left, right=counting_bits - left - 1)
    phase = qm.as_fixed(counting, integer_bits=0)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


display(
    HTML(
        draw(
            qm.specialize(qpe_from_scratch, counting_bits=3), loops="expanded"
        ).to_html()
    )
)

# %% [markdown]
# ### Built-in Function: `phase_estimation`
#
# The `phase_estimation` function combines the Hadamard gates,
# controlled-$U^{2^k}$ gates, inverse QFT, and fixed-point view used in the
# from-scratch implementation. It returns both the phase and the retained
# target register. Passing the returned phase to `qm.measure` measures the
# qubits and decodes the fixed-point bit string into a numeric phase estimate.


# %%
@qm.qkernel
def qpe_with_stdlib(
    counting_bits: qm.Const[qm.Int],
) -> qm.Shot[qm.UFixed[0, counting_bits]]:
    """Read the same eigenphase with the standard-library QPE component."""
    target = qm.prepare((0, 1))
    phase, target = phase_estimation(diagonal_4x4, target, counting_bits)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


display(
    HTML(
        draw(
            qm.specialize(qpe_with_stdlib, counting_bits=3), loops="expanded"
        ).to_html()
    )
)

# %% [markdown]
# ## Execution Result
#
# The target phase $0.6$ cannot be represented exactly with only a few bits, so
# QPE returns a distribution over nearby $m$-bit approximations. We vary the
# number of counting qubits from 3 to 9 and compare the estimate produced by the
# built-in `phase_estimation` function with the exact phase.


# %%
def phase_distance(a: float, b: float) -> float:
    """Compute cyclic distance between two phase fractions."""
    raw_distance = abs(a - b)
    return min(raw_distance, 1.0 - raw_distance)


def run_qpe_experiment(qpe_kernel, counting_bits: int) -> float:
    """Specialize, compile, and sample one QPE register width."""
    program = qm.specialize(qpe_kernel, counting_bits=counting_bits)
    artifact = qm.compile(program, to=endpoint.target)
    sample_result = qm.run(
        artifact,
        on=endpoint,
        shots=SHOTS,
        seed=SAMPLER_SEED + counting_bits,
    ).result
    assert sample_result.shots == SHOTS
    assert sum(sample_result.values.values()) == SHOTS
    most_observed_result = max(
        sample_result.values.items(), key=lambda item: (item[1], -item[0])
    )
    print(most_observed_result)
    qpe_output, most_observed_shots = most_observed_result
    assert 0 < most_observed_shots <= SHOTS
    return float(qpe_output)


# Run QPE for all requested counting-qubit counts.
bits = list(range(3, 6) if docs_test_mode else range(3, 10))
estimated_phases = [
    run_qpe_experiment(qpe_with_stdlib, counting_bits) for counting_bits in bits
]
# Check that the from-scratch and built-in implementations return the same estimate.
scratch_phase = run_qpe_experiment(qpe_from_scratch, bits[0])
assert np.isclose(scratch_phase, estimated_phases[0])
# Compare estimates with the exact phase and compute wraparound errors.
exact_phases = [TARGET_PHASE_FRACTION for _ in bits]
phase_errors = [
    phase_distance(estimated_phase, TARGET_PHASE_FRACTION)
    for estimated_phase in estimated_phases
]

# Plot the phase estimates against the exact phase.
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(bits, estimated_phases, marker="o", color="#2696EB", label="QPE estimate")
ax.plot(bits, exact_phases, linestyle="--", color="#DB4D3F", label="exact phase")
ax.set_xlabel("counting qubits")
ax.set_ylabel("phase fraction")
ax.set_xticks(bits)
phase_margin = max(0.02, max(phase_errors) + 0.01)
ax.set_ylim(
    max(0.0, TARGET_PHASE_FRACTION - phase_margin),
    min(1.0, TARGET_PHASE_FRACTION + phase_margin),
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

# Check that additional counting qubits improve the estimate in this example.
assert phase_errors[-1] < phase_errors[0]
for counting_bits, phase_error in zip(bits, phase_errors):
    assert phase_error <= 1 / 2**counting_bits

# %% [markdown]
# ## Resource Estimation
#
# The previous section showed that more counting qubits improve precision.
# We apply `qm.resources.logical` to the same specialized QPE kernels used by
# `run_qpe_experiment()` above. This study includes the Hadamards, controlled
# powers from `phase_estimation`, inverse QFT, and measurements.
# The plot uses the lower endpoint of `LOGICAL_GATES`. The native study retains
# an unresolved upper bound for the controlled and adjoint transforms in this
# program, so this plot shows a guaranteed lower bound rather than an exact
# gate count. The number of controlled-$U$ calls is exact and checked separately
# as $2^m-1$. Measurements are reported separately by `LOGICAL_MEASUREMENTS`.
# Engine compilation and optimization can change the emitted gate count. See the
# [resource-estimation tutorial](../tutorial/04_resource_estimation.ipynb)
# for details of resource studies and their bounds.

# %%
resource_gate_counts: list[int] = []
for counting_bits in bits:
    program = qm.specialize(qpe_with_stdlib, counting_bits=counting_bits)
    logical_study = qm.resources.logical(program)
    controlled_queries = logical_study.metric(
        qm.resources.logical_operation_count("qcore.control_call")
    ).numeric_enclosure()
    assert controlled_queries == (2**counting_bits - 1, 2**counting_bits - 1)
    gate_bounds = logical_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
    assert gate_bounds[0] is not None
    assert logical_study.metric(
        qm.resources.LOGICAL_MEASUREMENTS
    ).numeric_enclosure() == (counting_bits + 2, counting_bits + 2)
    resource_gate_counts.append(int(gate_bounds[0]))

# Anchor a 2^m reference curve at the first resource-estimate point.
scaling_reference = [
    resource_gate_counts[0] * 2 ** (counting_bits - bits[0]) for counting_bits in bits
]

# Plot the resource estimate with its expected exponential-in-m trend.
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(
    bits,
    resource_gate_counts,
    marker="o",
    color="#2696EB",
    label="QPE logical gate lower bound",
)
ax.plot(
    bits,
    scaling_reference,
    linestyle="--",
    color="#DB4D3F",
    label=r"$O(2^m)$",
)
ax.set_xlabel(r"counting qubits $m$")
ax.set_ylabel("logical gate lower bound")
ax.set_yscale("log")
ax.set_xticks(bits)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

# Check that the direct estimates increase over this range.
assert all(
    later > earlier
    for earlier, later in zip(resource_gate_counts, resource_gate_counts[1:])
)

# %% [markdown]
# The plot compares the number of counting qubits $m$ and the resulting lower bound on total
# gate count with a reference line proportional to $2^m$. If the target
# additive error is $\epsilon$, the spacing between representable phases should
# satisfy roughly
#
# $$
# 2^{-m} \lesssim \epsilon,
# $$
#
# so the required number of counting qubits is
#
# $$
# m = O\!\left(\log\frac{1}{\epsilon}\right).
# $$
#
# In this implementation, each controlled-$U^{2^k}$ gate is constructed by
# repeating $U$. Therefore the number of applications of $U$ is
#
# $$
# \sum_{k=0}^{m-1} 2^k = 2^m - 1
# =
# O\!\left(\frac{1}{\epsilon}\right).
# $$
#
# Thus, the number of counting qubits grows logarithmically in $1/\epsilon$,
# while the number of applications of $U$ grows as $O(1/\epsilon)$
# {cite:p}`10.1017/CBO9780511976667`.
#
# In this case, the gate count is $O(1/\epsilon)$ as shown above, making this
# approach inefficient for high-precision QPE. More generally, let $G(V)$ be
# the number of gates needed to implement a unitary matrix $V$. The gate count
# for the QPE body can then be written as
#
# $$
# G_{\mathrm{QPE}}(m)
# =
# \sum_{k=0}^{m-1} G\!\left(\mathrm{controlled}\text{-}U^{2^k}\right)
# + O(m^2),
# $$
#
# :::{note} Inverse QFT Gate Count
# The inverse QFT on $m$ qubits can be decomposed using $O(m^2)$ gates.
# This is the contribution represented by the $O(m^2)$ term above.
# :::
#
# In practice, however, this estimate can change significantly. The expression
# above counts the gates in the QPE body and does not include initial-state
# preparation. The gate count of each controlled-$U^{2^k}$ also depends on its
# implementation. We consider these two factors separately below.
#
# 1. **Controlled-$U$ implementation cost.**
#
#    QPE is used for order finding in Shor's algorithm and for estimating
#    energies from eigenphases of time-evolution operators in quantum-system
#    simulation. In Shor's algorithm, modular-multiplication powers can be
#    constructed from the problem structure, so they do not require
#    exponentially many repetitions of $U$. In Hamiltonian simulation,
#    controlled time evolution may also be implementable with polynomial
#    resources, depending on the Hamiltonian and simulation method. A useful
#    QPE resource estimate should therefore state whether controlled-$U^{2^k}$
#    gates are repeated applications of $U$, directly synthesized circuits, or
#    problem-specific arithmetic or simulation circuits.
#
# 2. **Initial-state preparation cost.**
#
#    The probability of measuring a particular eigenphase is the squared
#    overlap between the prepared state and the corresponding eigenstate.
#    Preparing an approximate state with sufficient overlap can require a
#    nontrivial quantum circuit, so an end-to-end resource estimate must count
#    those gates separately.
#
# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - QPE uses phase kickback from controlled-$U^{2^k}$ gates and the inverse QFT
#   to read an eigenphase of a unitary matrix as a binary value.
# - Qamomile can construct QPE from `qm.control`, `inverse_qft`, and `qm.as_fixed`,
#   while the built-in `phase_estimation` expresses the same procedure concisely.
# - The phase resolution from $m$ counting qubits is $O(2^{-m})$, while the
#   required gate count depends on the implementation of controlled-$U^{2^k}$
#   gates and the initial-state preparation method.
