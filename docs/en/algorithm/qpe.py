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
# # Introduction to Quantum Phase Estimation
#
# Quantum Phase Estimation (QPE) estimates the eigenphase $\phi$ of a unitary
# $U$ from an eigenstate $|\psi\rangle$ satisfying
# $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$. It is a central primitive in
# order finding and related algorithms that use phases encoded by unitary
# eigenvalues {cite:p}`10.48550/arXiv.quant-ph/9511026,10.1098/rspa.1998.0164`.
#
# This notebook applies the built-in `qpe` helper to a 4x4 unitary. You will
# prepare a known eigenstate, run the circuit on a local Qiskit simulator,
# compare the decoded phase with the target phase $0.6$, see how additional
# counting qubits improve precision, and use a symbolic resource estimate to
# see how the cost grows with the requested precision.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
# Import numerical, plotting, simulator, and Qamomile utilities.
import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

# Decompose composite gates so the example exposes the QPE circuit structure.
transpiler = QiskitTranspiler(use_native_composite=False)

# %% [markdown]
# ## Background: Phase Kickback and Quantum Fourier Transform
#
# QPE combines two ideas: phase kickback from controlled unitaries
# {cite:p}`10.1098/rspa.1998.0164` and the Quantum Fourier Transform
# {cite:p}`10.48550/arXiv.quant-ph/0201067`.
#
# ### Phase Kickback
#
# Phase kickback moves an eigenphase of the target register onto a control
# register. Suppose
#
# $$
# U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle.
# $$
#
# Put one control qubit in superposition and keep the target in $|\psi\rangle$:
#
# $$
# \frac{|0\rangle + |1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# A controlled-$U$ applies $U$ only on the $|1\rangle$ branch of the control.
# Because the target is an eigenstate, that branch picks up the eigenphase:
#
# $$
# \frac{|0\rangle|\psi\rangle + |1\rangle U|\psi\rangle}{\sqrt{2}}
# =
# \frac{|0\rangle + e^{2\pi i\phi}|1\rangle}{\sqrt{2}}|\psi\rangle.
# $$
#
# The target returns to the same eigenstate, while the control carries the
# phase. QPE repeats this idea with controlled powers. A controlled-$U^{2^k}$
# gate kicks back the phase $e^{2\pi i 2^k\phi}$, so different control qubits
# receive different binary weights of the same eigenphase. QPE keeps these
# kicked-back phases coherent instead of measuring each control qubit
# separately.
#
# ### Quantum Fourier Transform
#
# The QFT on $M=2^m$ basis states is
#
# $$
# \mathrm{QFT}_M|x\rangle
# = \frac{1}{\sqrt{M}}
#   \sum_{y=0}^{M-1} e^{2\pi i xy/M}|y\rangle.
# $$
#
# It maps an integer $x$ to a regular phase pattern across the computational
# basis. See the QFT tutorial for details. The inverse QFT performs the reverse
# operation. If $\phi=a/M$ is exactly representable with $m$ bits, then the
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
# QPE turns the single-control phase-kickback idea into a register-wide
# procedure. The counting register first stores all binary weights in
# superposition. Controlled powers of $U$ then kick the corresponding weighted
# eigenphases onto that register. The inverse QFT reads this Fourier-encoded
# phase pattern as an ordinary binary integer, which gives an estimate of
# $\phi$.
#
# Suppose the target-register input is an eigenstate $|\psi\rangle$ of $U$,
# so that $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$, and we want $m$ bits
# of precision. The counting register starts in $|0\rangle^{\otimes m}$.
# Let $M=2^m$.
#
# ### Step 1: Put the counting register in superposition
#
# Apply Hadamard gates to the counting register. This creates a uniform
# superposition over the $M$ possible counting-register values.
#
# $$
# |\Psi_1\rangle =
# H^{\otimes m}|0\rangle^{\otimes m}|\psi\rangle
# =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}|r\rangle|\psi\rangle.
# $$
#
# ### Step 2: Apply controlled powers
#
# For each counting qubit $k$, apply controlled-$U^{2^k}$. If
# $r=\sum_{k=0}^{m-1} r_k2^k$, the target eigenstate picks up the phase
# $e^{2\pi i\phi r}$:
#
# $$
# |\Psi_2\rangle =
# \frac{1}{\sqrt{M}}\sum_{r=0}^{M-1}
# e^{2\pi i \phi r}|r\rangle|\psi\rangle.
# $$
#
# ### Step 3: Decode with inverse QFT
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
# ### Step 4: Measure the phase
#
# Measure the counting register. In the exact case, the measurement returns
# $a$ and the phase estimate is
#
# $$
# \tilde{\phi} = \frac{a}{M}.
# $$
#
# If $\phi$ is not exactly representable with $m$ bits, the distribution
# concentrates around the closest $m$-bit approximations. Increasing $m$ makes
# the grid spacing $1/2^m$ smaller, so the nearest representable phase can get
# closer to the true eigenphase.
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
# ## Implementation: `qpe` function
#
# The `qmc.qpe` function applies the full QPE pattern: Hadamard gates,
# controlled powers of the supplied unitary qkernel, inverse QFT, and
# fixed-point phase decoding.
#
# ### Problem Example
#
# We use a **diagonal** 4x4 unitary:
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
# a binary grid, increasing the number of counting qubits visibly improves the
# approximation.

# %%
# Set the counting-register sizes, sampling settings, and target eigenstate.
COUNTING_BIT_OPTIONS = tuple(range(3, 10))
DRAW_COUNTING_BITS = 3
EXAMPLE_COUNTING_BITS = 3
SHOTS = 4096
SAMPLER_SEED = 321
TARGET_BASIS = 1  # |01>

# Set the diagonal-unitary phases and the target phase.
TARGET_PHASE_FRACTION = 0.6
phase_fractions = np.array([0.0, TARGET_PHASE_FRACTION, 0.23, 0.81])
phase_fractions[TARGET_BASIS] = TARGET_PHASE_FRACTION
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
# ### Quantum Kernel with `qpe`
#
# First, we define the 4x4 unitary whose phase we want to estimate. The
# `diagonal_4x4` qkernel implements this matrix directly. In Qamomile,
# `qmc.p(q, theta)` multiplies the $|1\rangle$ component of `q` by
# $e^{i\theta}$. Applying `qmc.p(q[0], phi10)` therefore gives every basis
# state whose first target bit is 1 the phase $e^{i\theta_{10}}$, and
# `qmc.p(q[1], phi01)` gives every basis state whose second target bit is 1
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
# Implement the diagonal 4x4 unitary with phase gates.
@qmc.qkernel
def diagonal_4x4(
    q: qmc.Vector[qmc.Qubit],
    phi01: qmc.Float,
    phi10: qmc.Float,
    phi11: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    # Add phases controlled by each target bit.
    q[0] = qmc.p(q[0], phi10)
    q[1] = qmc.p(q[1], phi01)
    # Correct the |11> entry so the full diagonal matches the matrix.
    q[0], q[1] = qmc.cp(q[0], q[1], phi11 - phi10 - phi01)
    return q


# Draw the target unitary with the concrete phase parameters.
diagonal_4x4.draw(
    q=2,
    phi01=PHI_01,
    phi10=PHI_10,
    phi11=PHI_11,
    fold_loops=False,
)

# %% [markdown]
# The `qmc.qpe` function returns a `QFixed` phase register. `QFixed` is
# Qamomile's fixed-point number type backed by a quantum register. Here it
# represents the phase fraction in $[0,1)$, and measuring it returns a
# floating-point phase estimate directly. The qkernel below prepares the
# eigenstate $|01\rangle$, applies QPE to `diagonal_4x4`, and measures the
# decoded phase. We build it with a small factory so the number of counting
# qubits can vary while the problem unitary stays the same.


# %%
# Build QPE kernels parametrized by the counting-register size.
def make_qpe_kernel(counting_bits: int):
    @qmc.qkernel
    def qpe_with_stdlib(
        phi01: qmc.Float,
        phi10: qmc.Float,
        phi11: qmc.Float,
    ) -> qmc.Float:
        # Allocate the counting register and the two-qubit target register.
        counting = qmc.qubit_array(counting_bits, name="counting")
        target = qmc.qubit_array(2, name="target")
        # Prepare |01> as the target eigenstate.
        target[1] = qmc.x(target[1])

        # Apply stdlib QPE and measure the decoded fixed-point phase.
        phase = qmc.qpe(
            target,
            counting,
            diagonal_4x4,
            phi01=phi01,
            phi10=phi10,
            phi11=phi11,
        )
        return qmc.measure(phase)

    return qpe_with_stdlib


# Draw a small QPE instance used in the circuit illustration.
qpe_to_draw = make_qpe_kernel(DRAW_COUNTING_BITS)
qpe_to_draw.draw(
    phi01=PHI_01,
    phi10=PHI_10,
    phi11=PHI_11,
    fold_loops=False,
)

# %% [markdown]
# ## Execution Result
#
# ### Circuit Transpilation and Execution
#
# The measured `QFixed` value is returned as a floating-point phase estimate, so
# no manual bit-string decoding is needed. For the first run, we use the same
# three counting qubits as in the circuit drawing. The experiment prints the
# `sample_result.results` entry with the largest shot count and returns only the
# phase estimate used by later cells.

# %%
# Bind the diagonal-unitary phases at transpile time.
bindings = {"phi01": PHI_01, "phi10": PHI_10, "phi11": PHI_11}


# Compute cyclic distance between two phase fractions.
def phase_distance(a: float, b: float) -> float:
    raw_distance = abs(a - b)
    return min(raw_distance, 1.0 - raw_distance)


# Transpile and sample one QPE circuit for a chosen counting-register size.
def run_qpe_experiment(counting_bits: int) -> float:
    # Build and transpile the QPE kernel with fixed unitary phases.
    qpe_kernel = make_qpe_kernel(counting_bits)
    executable = transpiler.transpile(qpe_kernel, bindings=bindings)
    # Seed the simulator deterministically for reproducible documentation output.
    executor = transpiler.executor(
        backend=AerSimulator(
            seed_simulator=SAMPLER_SEED + counting_bits,
            max_parallel_threads=1,
        )
    )
    # Sample the measured QFixed phase estimate.
    sample_result = executable.sample(
        executor,
        shots=SHOTS,
        bindings={},
    ).result()

    # Keep the most frequently observed decoded phase.
    most_observed_result = max(sample_result.results, key=lambda item: item[1])
    print(most_observed_result)
    qpe_output, most_observed_shots = most_observed_result

    assert 0 < most_observed_shots <= SHOTS
    return qpe_output


# Run the three-counting-qubit example and verify the expected grid accuracy.
example_phase = run_qpe_experiment(EXAMPLE_COUNTING_BITS)
assert phase_distance(example_phase, TARGET_PHASE_FRACTION) <= (
    1 / 2**EXAMPLE_COUNTING_BITS
)

# %% [markdown]
# ### Counting Register Size and Precision
#
# Because the target phase $0.6$ is not exactly representable with only a few
# bits, QPE returns a distribution over nearby grid points. We now vary the
# number of counting qubits from 3 to 9 and compare the final `qpe` output with
# the exact phase fraction. The printed rows are the most observed
# `sample_result.results` entry for each counting-register size.

# %%
# Run QPE for all requested counting-register sizes.
bits = list(COUNTING_BIT_OPTIONS)
estimated_phases = [run_qpe_experiment(counting_bits) for counting_bits in bits]
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
# The previous subsection showed that more counting qubits improve precision.
# The following symbolic resource-estimation kernel keeps the number of counting
# qubits as `m` and uses a one-qubit phase unitary, so the cell first exposes the
# QPE scaling that comes from the counting register itself.

# %%
# Use a minimal one-qubit phase unitary to isolate QPE scaling.
@qmc.qkernel
def phase_unitary_for_resources(q: qmc.Qubit) -> qmc.Qubit:
    q = qmc.p(q, 1.0)
    return q


# Build a symbolic QPE-like resource kernel with m counting qubits.
@qmc.qkernel
def symbolic_qpe_resource_kernel(m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    # Allocate the symbolic counting register and a target eigenstate.
    counting = qmc.qubit_array(m, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    # Put every counting qubit into superposition.
    for i in qmc.range(m):
        counting[i] = qmc.h(counting[i])

    # Apply controlled U, U^2, ..., U^{2^{m-1}} by repetition.
    controlled_unitary = qmc.control(phase_unitary_for_resources)
    for i in qmc.range(m):
        repetitions = 2**i
        for _ in qmc.range(repetitions):
            counting[i], target = controlled_unitary(counting[i], target)

    return counting


# Estimate resources symbolically before substituting a concrete m.
symbolic_resource_estimate = symbolic_qpe_resource_kernel.estimate_resources().simplify()

# Substitute concrete counting-qubit counts and collect total gate counts.
resource_gate_counts: list[int] = []
for counting_bits in bits:
    concrete_estimate = symbolic_resource_estimate.substitute(
        m=counting_bits
    ).simplify()
    resource_gate_counts.append(int(concrete_estimate.gates.total))

# Anchor a 2^m reference curve at the first resource-estimate point.
scaling_reference = [
    resource_gate_counts[0] * 2 ** (counting_bits - bits[0])
    for counting_bits in bits
]

# Plot the resource estimate with its expected exponential-in-m trend.
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(
    bits,
    resource_gate_counts,
    marker="o",
    color="#6A5ACD",
    label="resource estimate",
)
ax.plot(
    bits,
    scaling_reference,
    linestyle="--",
    color="#DB4D3F",
    label=r"$\propto 2^m$",
)
ax.set_xlabel("counting qubits")
ax.set_ylabel("Qamomile gate count")
ax.set_xticks(bits)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

# Check that the symbolic estimate depends on m and increases over this range.
assert "m" in str(symbolic_resource_estimate.gates.total)
assert all(
    later > earlier
    for earlier, later in zip(resource_gate_counts, resource_gate_counts[1:])
)

# %% [markdown]
# The plot shows the cost of refining the phase grid together with a reference
# line proportional to $2^m$. If the target additive error is $\epsilon$, the
# grid spacing should satisfy roughly
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
# In the repetition model used by this resource-estimation kernel, QPE applies
# controlled unitaries with weights $1,2,\ldots,2^{m-1}$. Therefore the number
# of controlled-unitary applications is
#
# $$
# \sum_{k=0}^{m-1} 2^k = 2^m - 1
# =
# O\!\left(\frac{1}{\epsilon}\right).
# $$
#
# This is the main reading of the plot: logarithmically many counting qubits
# in $1/\epsilon$ are enough, but the repeated controlled-unitary work scales as
# $O(1/\epsilon)$ {cite:p}`10.1017/CBO9780511976667`.
#
# The last point is implementation-dependent. In general,
#
# $$
# C_{\mathrm{QPE}}(m)
# =
# \sum_{k=0}^{m-1} C\!\left(\mathrm{controlled}\text{-}U^{2^k}\right)
# + O(m^2),
# $$
#
# :::{note} Inverse QFT Gate Count
# The inverse QFT on $m$ qubits can be decomposed using $O(m^2)$ gates. Refer
# to the QFT tutorial for details.
# :::
#
# The $O(1/\epsilon)$ estimate above applies when $U^{2^k}$ is realized by
# repeated uses of $U$. If the problem structure lets you implement
# $\mathrm{controlled}\text{-}U^{2^k}$ more directly, the resource estimate
# should use that implementation cost instead.

# %% [markdown]
# ## Summary
#
# In this notebook, we implemented QPE on a diagonal 4x4 unitary, sampled the
# decoded phase, and checked how the precision and resources change as the
# counting register grows.
#
# - `qmc.qpe` applies Hadamard gates, controlled powers, inverse QFT, and
#   fixed-point phase decoding to a supplied unitary qkernel.
# - The example prepares the eigenstate $|01\rangle$ and estimates the target
#   phase $0.6$ directly as a floating-point `QFixed` measurement result.
# - Adding counting qubits makes the binary phase grid finer, so the sampled
#   estimate moves closer to the target phase.
# - A symbolic `estimate_resources()` call shows that $m=O(\log(1/\epsilon))$
#   counting qubits lead to $O(1/\epsilon)$ repeated controlled-unitary
#   applications in the simple repetition model.
# - In real applications, the main cost driver is the implementation of
#   $\mathrm{controlled}\text{-}U^{2^k}$, so resource estimates should state how
#   those powers are realized.
