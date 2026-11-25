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
# tags: [tutorial]
# ---
#
# # QPE and Resource Estimation
#
# Quantum Phase Estimation (QPE) is an algorithm that estimates the eigenphase $\phi$
# associated with an eigenstate $|\psi\rangle$ of a unitary $U$:
#
# $$
# U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle
# $$
#
# In this notebook, we first build the QPE circuit by hand and then compare it with
# `phase_estimation` from the Qamomile stdlib.
#
# ## 1. How QPE works
#
# QPE uses two quantum registers with different roles.
#
# - The **target register** holds the eigenstate $|\psi\rangle$ of $U$.
# - The **counting register** consists of $n$ qubits and holds an $n$-bit
#   approximation of the eigenphase. Its initial state is $|0\rangle^{\otimes n}$.
#
# The mechanism that transfers the eigenphase from the target to the counting
# register is phase kickback. Applying a controlled-$U$ with a single control in
# $|+\rangle$ and the target in $|\psi\rangle$ gives
#
# $$
# \frac{|0\rangle|\psi\rangle+|1\rangle U|\psi\rangle}{\sqrt2}
# =
# \frac{|0\rangle+e^{2\pi i\phi}|1\rangle}{\sqrt2}|\psi\rangle
# $$
#
# The target returns to the same eigenstate, and only the eigenphase appears as a
# relative phase on the control. QPE extends this operation to $n$ qubits and reads
# out the phase in the following three stages.
#
# ### 1. Prepare a uniform superposition
#
# Let $N=2^n$. We prepare $|\psi\rangle$ in the target and apply Hadamards to the
# counting register. This gives a uniform superposition of the $N$ computational
# basis states,
#
# $$
# |\Psi_1\rangle
# =
# \frac{1}{\sqrt N}
# \sum_{y=0}^{N-1}|y\rangle|\psi\rangle.
# $$
#
# ### 2. Accumulate the phase with controlled powers
#
# Write the basis state as $y=\sum_{k=0}^{n-1}y_k2^k$. We apply $U^{2^k}$ to the
# target, controlled by the $k$-th counting qubit. As a result,
#
# $$
# |\Psi_2\rangle
# =
# \frac{1}{\sqrt N}
# \sum_{y=0}^{N-1}
# e^{2\pi i\phi y}|y\rangle|\psi\rangle
# $$
#
# The information about $\phi$ is now recorded in the amplitudes.
#
# ### 3. Decode with the inverse QFT and measure
#
# If $\phi=a/N$ can be represented with $n$ bits, the counting register is now in
# the state $\operatorname{QFT}_N|a\rangle$. Applying the inverse QFT therefore gives
#
# $$
# |\Psi_3\rangle=|a\rangle|\psi\rangle
# $$
#
# Measuring the counting register yields the integer $a$, and the phase estimate is
#
# $$
# \widetilde{\phi}=\frac{a}{2^n}
# $$
#
# If $\phi$ cannot be represented exactly with $n$ bits, the measurement outcomes are
# distributed around the closest values. Adding one more bit halves the phase
# resolution $2^{-n}$, but it also doubles the largest power of the controlled
# unitary.
#
# ### The example used in this notebook
#
# We take the phase gate with $\theta=3\pi/4$, $P(\theta)|1\rangle=e^{i\theta}|1\rangle$,
# as the unitary whose phase we want to estimate. The phase to be found is then
# $\phi=\theta/(2\pi)=3/8=0.011_2$. Since this phase can be represented exactly with
# 3 qubits, QPE should return the computational basis state $|011\rangle$.
#
# :::{note}
# A general input $|\chi\rangle=\sum_jc_j|\psi_j\rangle$ contains several eigenstates.
# When the eigenphases are distinct and exactly representable with the counting bits,
# measuring the counting register gives $\phi_j$ with probability $|c_j|^2$ and projects
# the target onto $|\psi_j\rangle$.
# To keep the explanation simple, this notebook uses the known eigenstate $|1\rangle$.
# :::

# %%
# Install Qamomile and the packages used in this notebook.
# # !pip install "qamomile[qiskit]" matplotlib

# %%
from __future__ import annotations

import math

import matplotlib.pyplot as plt

import qamomile as qm
from qamomile.stdlib import inverse_qft, phase_estimation

SHOTS = 512
THETA = 3.0 * math.pi / 4.0
EXPECTED_PHASE = 3.0 / 8.0


@qm.qkernel(kind="unitary")
def phase_gate(q: qm.Q[qm.Bits[1]], theta: qm.Real) -> qm.Q[qm.Bits[1]]:
    """Apply a phase of theta to the eigenstate |1⟩."""
    return qm.p(q, theta)


# %% [markdown]
# ## 2. Build QPE by hand
#
# `phase_gate` applies a phase of `theta` to the $|1\rangle$ state. It implements
#
# $$
# P(\theta)|0\rangle=|0\rangle,\qquad
# P(\theta)|1\rangle=e^{i\theta}|1\rangle
# $$
#
# If we prepare the target in $|1\rangle$, the eigenvalue is
# $e^{i\theta}=e^{2\pi i\phi}$, so the phase returned by QPE is
# $\phi=\theta/(2\pi)$.
#
# In `manual_qpe` below, the QPE steps we saw above are mapped directly onto code.
#
# 1. Allocate the counting register with `qm.qubits(n)` and the target register with `qm.qubits(1)`
# 2. Prepare the target register in the eigenstate $|1\rangle$ with `qm.x`, and put the counting register into a superposition with Hadamards
# 3. Use `qm.control(phase_gate, counting, target, theta, at=k)` to apply the unitary from each counting qubit, repeating it `2**k` times
# 4. Decode the phase with `inverse_qft(counting, n)`, reverse the register order, view it as `UFixed`, and then measure it
#
# `UFixed` is the Qamomile type for treating a quantum register as an unsigned fixed-point
# number. With `integer_bits=0`, all $n$ qubits are interpreted as the fractional part, so
# the integer $a$ after the inverse QFT is read out as the phase $a/2^n$.


# %%
@qm.qkernel
def manual_qpe(
    n: qm.Const[qm.Int], theta: qm.Const[qm.Real]
) -> qm.Shot[qm.UFixed[0, n]]:  # noqa: F821
    """Build QPE from Hadamards, controlled powers, and an inverse QFT."""
    counting = qm.qubits(n)
    target = qm.x(qm.qubits(1))
    for k in range(n):
        counting = qm.h(counting, target=k)

    system = (counting, target)
    for k in range(n):
        for _ in range(2**k):
            counting, target = system
            counting, target = qm.control(phase_gate, counting, target, theta, at=k)
            system = (counting, target)

    counting, target = system
    counting = inverse_qft(counting, n)
    for left in range(n // 2):
        counting = qm.swap(counting, left=left, right=n - left - 1)
    phase = qm.as_fixed(counting, integer_bits=0)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


# %% [markdown]
# `qm.control(...)` applies an existing qkernel as a controlled unitary.
# This means there is no need to rewrite the unitary itself for QPE.
#
# Repeating it `2**k` times realizes the operation "controlled-$U^{2^k}$".
#
# The final `inverse_qft(counting, n)` converts the phase pattern back into the integer $a$.
# The swaps put the most significant bit first, as required by `UFixed`.
# The following `qm.as_fixed(counting, integer_bits=0)` interprets the same
# quantum register as the fixed-point number $a/2^n$. For example, for $|011\rangle$,
#
# $$
# \frac{1\cdot2^0+1\cdot2^1+0\cdot2^2}{2^3}
# =\frac38
# $$
#
# `as_fixed` reinterprets the register as a `UFixed` without adding gates. Measuring it
# returns a numeric phase. We also measure the retained target to consume it.

# %%
endpoint = qm.qiskit.Aer()
manual_program = qm.specialize(manual_qpe, n=3, theta=THETA)
manual_executable = qm.compile(manual_program, to=endpoint.target)
manual_result = qm.run(manual_executable, on=endpoint, shots=SHOTS, seed=42).result
print("Manual QPE:", dict(manual_result.values))

assert len(manual_result.values) == 1
((manual_phase, manual_count),) = manual_result.values.items()
assert manual_count == SHOTS
assert math.isclose(manual_phase, EXPECTED_PHASE, abs_tol=1e-12)

# %% [markdown]
# Because $\phi=3/8$ can be represented exactly with 3 bits, every shot gives the same
# phase value on a noiseless simulator. For a general phase, several nearby values
# appear, so `result.values` must be read as a distribution of measured values
# and their counts.

# %% [markdown]
# ## 3. The built-in `phase_estimation`
#
# Qamomile provides the same procedure as `phase_estimation`. Building QPE by hand is well
# suited to learning and understanding the circuit structure. When assembling more
# complex algorithms, however, using the algorithms included in the stdlib makes even
# complex algorithms easy to implement.
#
# The arguments of `phase_estimation` are:
#
# - `unitary`: the unitary qkernel whose phase is to be estimated, here `example_phase_gate`
# - `target`: the target register prepared in an eigenstate
# - `precision`: the number of qubits in the counting register, here `n`
#
# `phase_estimation` expects a gate with one quantum argument. We fix `theta` in `example_phase_gate` before passing it in.
# The function prepares the counting register and returns the phase and the retained target.
#
# In the hand-built version we wrote `qm.as_fixed(..., integer_bits=0)`
# explicitly, but `phase_estimation` already includes the same conversion. Therefore, in both
# implementations, `measure(phase)` gives a numeric value decoded to $a/2^n$.


# %%
@qm.qkernel(kind="unitary")
def example_phase_gate(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return phase_gate(q, THETA)


@qm.qkernel
def measure_phase(
    unitary: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    n: qm.Const[qm.Int],
) -> qm.Shot[qm.UFixed[0, n]]:  # noqa: F821
    """Measure the phase of a single-qubit unitary with the standard library."""
    target = qm.x(qm.qubits(1))
    phase, target = phase_estimation(unitary, target, n)
    phase_value = qm.measure(phase)
    _target_value = qm.measure(target)
    return phase_value


@qm.qkernel
def builtin_qpe(n: qm.Const[qm.Int]) -> qm.Shot[qm.UFixed[0, n]]:  # noqa: F821
    return measure_phase(example_phase_gate, n)


# %%
builtin_program = qm.specialize(builtin_qpe, n=3)
builtin_executable = qm.compile(builtin_program, to=endpoint.target)
builtin_result = qm.run(builtin_executable, on=endpoint, shots=SHOTS, seed=42).result
print("Built-in QPE:", dict(builtin_result.values))

assert len(builtin_result.values) == 1
((builtin_phase, builtin_count),) = builtin_result.values.items()
assert math.isclose(builtin_phase, EXPECTED_PHASE, abs_tol=1e-12)
assert builtin_count == SHOTS

# %% [markdown]
# ## 4. Count logical resources from a concrete circuit
#
# `qm.resources.logical()` analyzes the Qamomile IR without executing a qkernel.
# We compare the hand-built and built-in versions on the same 3-bit problem.
# Each estimate gives a pair of bounds. For gate counts, `None` indicates that
# the analysis does not supply an upper bound for the controlled phase calls.
# Both versions give the same bounds, although their compiled circuits can differ.

# %%
manual_resources = qm.resources.logical(manual_program)
builtin_resources = qm.resources.logical(builtin_program)

metrics = {
    "qubits": qm.resources.LOGICAL_PEAK_QUBITS,
    "gates": qm.resources.LOGICAL_GATES,
    "two-qubit": qm.resources.LOGICAL_TWO_QUBIT_GATES,
}
manual_bounds = {
    name: manual_resources.metric(metric).numeric_enclosure()
    for name, metric in metrics.items()
}
builtin_bounds = {
    name: builtin_resources.metric(metric).numeric_enclosure()
    for name, metric in metrics.items()
}
print("Manual bounds:", manual_bounds)
print("Built-in bounds:", builtin_bounds)

assert manual_bounds["qubits"] == builtin_bounds["qubits"] == (4, 4)
assert manual_bounds == builtin_bounds

# %% [markdown]
# ## 5. Plug a variable-width Oracle into the built-in QPE
#
# So far we have executed a concrete phase gate and counted resources from its
# circuit. Sometimes, however, you want to study the scaling of the whole QPE before
# a subroutine such as Hamiltonian simulation has been implemented. An opaque
# unitary qkernel represents such an unimplemented unitary by its name and signature alone.
#
# In this section, we clearly separate the roles of the symbols.
#
# - $n$: the number of qubits in the target register on which the unitary $U$ acts
# - $m$: the number of qubits in the counting register, which estimates the phase with $m$ bits


# %%
@qm.qkernel(kind="unitary")
def u(target: qm.Q[qm.Bits[n]]) -> qm.Q[qm.Bits[n]]: ...  # noqa: F821


# %% [markdown]
# The signature uses a symbolic register width `n`. This lets us reuse the
# same oracle for different numbers of qubits $n$.


# %%
@qm.qkernel
def qpe_model(
    n: qm.Const[qm.Int],
    m: qm.Const[qm.Int],
) -> tuple[qm.Q[qm.UFixed[0, m]], qm.Q[qm.Bits[n]]]:  # noqa: F821
    """Build a symbolic resource model of QPE with a variable-width oracle."""
    target = qm.qubits(n)

    # In an actual QPE, the target would be prepared in an eigenstate of U here.
    return phase_estimation(u, target, m)


# %% [markdown]
# Because we give no concrete inputs to `qm.resources.logical()`, the result is an
# algebraic expression in $n$ and $m$. In this example, the number of qubits is $m+n$.
#
# In addition, since `phase_estimation` calls controlled-$U^{2^k}$ for counting qubit $k$, in a
# model that counts $U^{2^k}$ as $2^k$ queries to the base Oracle $U$, the number of
# queries is
#
# $$
# 1+2+\cdots+2^{m-1}=2^m-1
# $$
#
# Qamomile records oracle calls and queries by name even when the gate cost is unknown.
#
# The overall gate count is incomplete without the oracle cost, so we report the
# qubit and query estimates here. Compilation for execution would require an oracle body.

# %%
symbolic_resources = qm.resources.logical(qpe_model)
qubit_metric = qm.resources.LOGICAL_PEAK_QUBITS
query_metric = qm.resources.algorithmic_query_count(u.symbol)

print("Symbolic Oracle queries:", symbolic_resources.metric(query_metric))

# %% [markdown]
# Finally, we can substitute values into the symbolic qubit and query estimates to obtain
# concrete numbers. This is a substitution into a formula, not a circuit construction,
# so it can be computed quickly.

# %%
for n_value, m_value in [(2, 3), (4, 3), (4, 5)]:
    concrete = symbolic_resources.evaluate(n=n_value, m=m_value)
    qubits = concrete.metric(qubit_metric).numeric_enclosure()
    queries = concrete.metric(query_metric).numeric_enclosure()
    print(f"n={n_value}, m={m_value}: qubits={qubits[0]}, queries={queries[0]}")
    assert qubits == (n_value + m_value, n_value + m_value)
    assert queries == (2**m_value - 1, 2**m_value - 1)

# %% [markdown]
# ## Exercise: A phase that cannot be represented with 3 bits
#
# For this exercise, use a new angle `2*pi/3`, corresponding to $\phi=1/3$. Because this phase cannot be
# represented exactly with 3 bits, several measurement outcomes appear. Increasing the
# number of counting bits makes the phase resolution $2^{-n}$ finer, so confirm that
# the peak of the distribution approaches $1/3$.
#
# Specifically, investigate the following two points.
#
# 1. Compare the phase distributions for 3 and 5 counting bits
# 2. Increase the number of counting bits from 2 to 6 and confirm that the error between the most frequent value and $1/3$ decreases

# %%
# TODO: For theta=2*pi/3, examine the phase distribution and the error versus the number of counting bits.

# %% [markdown]
# ### Example Solution

# %%
exercise_phase = 1.0 / 3.0
exercise_theta = 2.0 * math.pi * exercise_phase
counting_widths = [2, 3, 4, 5, 6]
phase_distributions = {}
peak_errors = {}


@qm.qkernel(kind="unitary")
def exercise_phase_gate(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return phase_gate(q, exercise_theta)


@qm.qkernel
def exercise_qpe(n: qm.Const[qm.Int]) -> qm.Shot[qm.UFixed[0, n]]:  # noqa: F821
    return measure_phase(exercise_phase_gate, n)


for counting_bits in counting_widths:
    exercise_program = qm.specialize(exercise_qpe, n=counting_bits)
    exercise_executable = qm.compile(exercise_program, to=endpoint.target)
    exercise_result = qm.run(
        exercise_executable, on=endpoint, shots=SHOTS, seed=42
    ).result
    phase_distributions[counting_bits] = sorted(exercise_result.values.items())

    peak_phase, _ = max(exercise_result.values.items(), key=lambda item: item[1])
    peak_errors[counting_bits] = abs(peak_phase - exercise_phase)
    print(
        f"counting bits={counting_bits}: "
        f"peak={peak_phase:.5f}, error={peak_errors[counting_bits]:.5f}"
    )

assert all(
    peak_errors[right] < peak_errors[left]
    for left, right in zip(counting_widths, counting_widths[1:], strict=False)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))

for counting_bits in [3, 5]:
    distribution = phase_distributions[counting_bits]
    phases = [phase for phase, _ in distribution]
    probabilities = [count / SHOTS for _, count in distribution]
    axes[0].plot(
        phases,
        probabilities,
        marker="o",
        linewidth=1,
        label=f"{counting_bits} counting bits",
    )

axes[0].axvline(exercise_phase, color="black", linestyle="--", label="exact 1/3")
axes[0].set(
    xlabel="Estimated phase",
    ylabel="Probability",
    title="QPE output distribution",
)
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].plot(
    counting_widths,
    [peak_errors[width] for width in counting_widths],
    marker="o",
)
axes[1].set(
    xlabel="Number of counting bits",
    ylabel="Peak error",
    title="Discretization error",
    yscale="log",
)
axes[1].set_xticks(counting_widths)
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - In QPE, the target is prepared in an eigenstate (or a state with a large overlap with an eigenstate), and the counting qubits receive Hadamards, control the powers of $U$ on the target, and then undergo the inverse QFT.
# - The Qamomile stdlib provides several algorithms, including QPE.
# - The same qkernel can be used both to build an executable circuit and to estimate resources.
# - Unimplemented subroutines can also be represented as an opaque unitary qkernel, and the number of Oracle queries can be estimated symbolically without assuming an implementation.
