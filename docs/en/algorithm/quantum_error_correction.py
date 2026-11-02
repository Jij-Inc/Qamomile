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
# tags: [algorithm, error-correction]
# ---
#
# # Introduction to Quantum Error Correction
#
# Qubits are easily disturbed by noise. **Quantum error correction (QEC)** protects information by spreading one logical qubit across multiple physical qubits.
#
# This article is the first part of a QEC introduction. We implement the following three codes with Qamomile's `@qkernel`.
#
# 1. **The 3-qubit bit-flip code** — corrects a single bit-flip ($X$) error.
# 2. **The 3-qubit phase-flip code** — corrects a single phase-flip ($Z$) error.
# 3. **Shor's 9-qubit code** — corrects any single-qubit error ($X$, $Y$, $Z$).
#
# The second part, [Stabilizer Formalism and the Steane Code](steane_code.ipynb), presents the framework that unifies these codes.
#
# **Prerequisites**: `@qkernel`, the CNOT gate, and measurement. If you are new to these, start with [Your First Quantum Kernel](../tutorial/01_your_first_quantum_kernel.ipynb).

# %% [markdown]
# ## 1. Why Quantum Error Correction Is Hard
#
# Before getting into actual quantum error correction, we explain what makes the quantum case harder than the classical one.


# %% [markdown]
# ### 1.1 The Classical Repetition Code
#
# On a classical computer, the simplest way to protect a bit $b$ from noise is the **repetition code**.
#
# ```text
# encode:   0 → 000     1 → 111
# compute:  process while still encoded
# recover:  read all 3 bits at the end, take a majority vote
# ```
#
# Even if one bit flips along the way (`000 → 010`, say), the majority vote recovers the original value.


# %% [markdown]
# ### 1.2 Encoding Works Fine on Quantum States Too
#
# We carry the same idea over to quantum. Suppose we want to spread the logical state
#
# $$\alpha\lvert0\rangle + \beta\lvert1\rangle$$
#
# across three qubits. Two CNOT gates produce the following state:
#
# $$\alpha\lvert000\rangle + \beta\lvert111\rangle$$
#
# This is not three copies of the state ($(\alpha\lvert0\rangle+\beta\lvert1\rangle)^{\otimes 3}$); it is an **entanglement** spread across the three qubits. Encoding is built without trouble this way.


# %% [markdown]
# ### 1.3 The Real Obstacle Is Correction
#
# What breaks down is not encoding but **correction**.
#
# The classical repetition code **reads** the three bits during correction and takes a majority vote. Classical bits are not destroyed by reading.
#
# In the quantum case, "reading" is **measurement**, and measurement destroys superposition. If you measure $\alpha\lvert000\rangle+\beta\lvert111\rangle$ directly, it collapses to either $\lvert000\rangle$ or $\lvert111\rangle$, and $\alpha$ and $\beta$ are lost.
#
# For the final readout at the end of a computation, "measure everything and take a majority vote" is fine, just as in the classical case. The problem is correcting *during* a computation — keeping the state intact for the rest of the computation while fixing only the error.
#
# So quantum error correction does not measure the data itself. It measures only **where and what kind of error occurred**. This information about the location and type of the error is called the **syndrome**. "Syndrome" is originally a medical term for a set of symptoms; just as a doctor diagnoses a disease from its symptoms without observing it directly, error correction pins down the error from its "symptoms" alone, without measuring the error itself.
#
# The syndrome is extracted through ancilla qubits, so the logical state ($\alpha,\beta$) is left untouched. This operation is **syndrome measurement**.


# %% [markdown]
# ### 1.4 Another Difference: Phase Errors
#
# There is one more difference between the classical and quantum cases.
#
# Classical errors are only bit flips ($0 \leftrightarrow 1$). Qubits also have **phase errors**. The error that flips only the sign of $\lvert1\rangle$,
#
# $$\alpha\lvert0\rangle + \beta\lvert1\rangle \;\longrightarrow\; \alpha\lvert0\rangle - \beta\lvert1\rangle$$
#
# does not change any probability when measured in the computational basis, so a majority vote cannot detect it at all.
#
# Quantum errors are also **continuous** — there are infinitely many intermediate errors, such as a qubit rotating slightly. Correcting each of these individually looks impossible, but in fact it is enough to consider only discrete errors.
#
# The reasoning has two steps. First, any error on a single qubit can be written, as a matrix, as a linear combination of $I$ (do nothing), $X$, $Y$, $Z$. A small rotation, for instance, is a slight mixture of $I$ and $X$. Applying such an error to an encoded state turns it into a superposition of "no error", "$X$ error", "$Z$ error", and so on.
#
# Second, measuring the syndrome of this state **collapses** it onto one of the superposed cases. What is left after the measurement is a single discrete error, such as "an $X$ error on qubit $i$". A continuous error is turned into a discrete **Pauli error** by syndrome measurement (we will see this collapse in action in the following sections).
#
# So the only errors we need to correct are the full $X$, $Y$, $Z$. And since $Y=iXZ$, correcting $X$ and $Z$ automatically covers $Y$.


# %% [markdown]
# ### 1.5 The Quantum Error Correction Flow
#
# Quantum error correction proceeds as follows.
#
# ```text
# encode  →  error  →  syndrome measurement  →  correction
# ```
#
# - **Encode**: spread one logical qubit across multiple physical qubits using entanglement.
# - **Syndrome measurement**: extract only the location and type of the error into ancilla qubits, without measuring the logical state.
# - **Correction**: apply a Pauli gate according to the syndrome to cancel the error.
#
# From the next section on, we implement this flow with the simplest code — the 3-qubit bit-flip code.


# %% [markdown]
# Before getting into the implementation, we load Qamomile and the Qiskit integration and define two helper functions. `_first_bit_distribution` and `_sample_first_bit` are just utilities that compile and run a kernel and return the 0/1 counts of the first bit. They are not central to QEC, so feel free to skip them.

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
import math

import qamomile as qm

# %% [markdown]
# ## 2. The 3-Qubit Bit-Flip Code
#
# The first code we build is the **bit-flip code**. It is the simplest quantum error correction code, correcting just a single bit-flip ($X$) error. We implement the whole flow from 1.5 — encode → error → syndrome measurement → correction — with this code.


# %% [markdown]
# ### 2.1 The Target Error: Bit Flip
#
# An $X$ error flips a qubit, $\lvert0\rangle\leftrightarrow\lvert1\rangle$. It is the error corresponding to a classical bit flip. The bit-flip code aims to correct this single $X$ error.


# %% [markdown]
# ### 2.2 The Code Space
#
# We encode the logical state $\alpha\lvert0\rangle+\beta\lvert1\rangle$ into the 3-qubit state
#
# $$\alpha\lvert000\rangle + \beta\lvert111\rangle.$$
#
# The logical $\lvert0\rangle$ corresponds to $\lvert000\rangle$ and the logical $\lvert1\rangle$ to $\lvert111\rangle$. The space spanned by these two is called the **code space**, and the only valid codewords are $\lvert000\rangle$ and $\lvert111\rangle$.


# %% [markdown]
# ### 2.3 The Encoding Circuit
#
# Encoding is just one CNOT each from the data qubit $q_0$ to $q_1$ and $q_2$.


# %% [markdown]
# If $q_0$ is $\lvert1\rangle$, the two CNOTs flip $q_1$ and $q_2$ as well, giving $\lvert111\rangle$. If $q_0$ is $\lvert0\rangle$, nothing happens and it stays $\lvert000\rangle$. If $q_0$ is a superposition $\alpha\lvert0\rangle+\beta\lvert1\rangle$, the result is $\alpha\lvert000\rangle+\beta\lvert111\rangle$.


# %% [markdown]
# ### 2.4 Syndrome Measurement: $Z$ Parity
#
# To find the location of the error, we measure two **parities** (whether two qubits hold the same value).
#
# - $S_0 = Z_0Z_1$ — whether $q_0$ and $q_1$ are equal
# - $S_1 = Z_0Z_2$ — whether $q_0$ and $q_2$ are equal
#
# In the code space ($\lvert000\rangle$ and $\lvert111\rangle$) all three qubits hold the same value, so both parities report "equal". When a single $X$ error occurs, the parity pattern changes according to its location.
#
# | Error | Syndrome $(s_0, s_1)$ | Correction |
# | --- | --- | --- |
# | none | $(0, 0)$ | none |
# | $X_0$ | $(1, 1)$ | $X_0$ |
# | $X_1$ | $(1, 0)$ | $X_1$ |
# | $X_2$ | $(0, 1)$ | $X_2$ |
#
# The three $X$ errors each show a distinct syndrome, so the syndrome uniquely determines the error location.
#
# To measure $Z_iZ_j$, prepare an ancilla qubit, apply `CX(data[i], anc)` and `CX(data[j], anc)`, then measure `anc`. Only the ancilla is measured; the data qubits are left untouched.
#
# Let us follow a concrete example. Suppose an $X$ error hits the second qubit $q_1$ of the encoded state $a\lvert000\rangle+b\lvert111\rangle$. Since $X$ flips $q_1$, the state becomes
#
# $$a\lvert010\rangle + b\lvert101\rangle.$$
#
# We add two ancilla qubits initialized to $\lvert00\rangle$ and perform syndrome measurement. `CX` adds the control qubit's value into the ancilla by XOR, so
#
# - $S_0=Z_0Z_1$: ancilla 1 receives $q_0\oplus q_1$. For $\lvert010\rangle$ it is $0\oplus1=1$; for $\lvert101\rangle$ it is $1\oplus0=1$.
# - $S_1=Z_0Z_2$: ancilla 2 receives $q_0\oplus q_2$. For $\lvert010\rangle$ it is $0\oplus0=0$; for $\lvert101\rangle$ it is $1\oplus1=0$.
#
# Both terms give the same ancilla values $(1,0)$, and the state evolves as follows.
#
# $$(a\lvert010\rangle + b\lvert101\rangle)\lvert00\rangle \;\longrightarrow\; (a\lvert010\rangle + b\lvert101\rangle)\lvert10\rangle$$
#
# The key point is that the two terms of the superposition give the **same** ancilla values. Because of this, measuring the ancilla does not destroy the superposition of $a$ and $b$.
# All the measurement yields is the syndrome $(s_0,s_1)=(1,0)$, which the table identifies as an $X$ error on $q_1$.


# %% [markdown]
# ### 2.5 Correction
#
# Once the syndrome is known, correction is just applying the same $X$ as the detected error, at the same location, once more. Since $X$ returns to the identity when applied twice ($X^2=I$), the error's $X$ and the correction's $X$ cancel out.
#
# Continuing the previous example: the syndrome $(1,0)$ indicates an $X$ error on $q_1$, so applying $X$ to $q_1$ gives
#
# $$a\lvert010\rangle + b\lvert101\rangle \;\longrightarrow\; a\lvert000\rangle + b\lvert111\rangle,$$
#
# restoring the encoded state exactly. The "Correction" column of the table above gives the correction for each syndrome.


# %% [markdown]
# ### 2.6 Implementation and Run: Logical $\lvert1\rangle$
#
# We collect everything into a single `@qkernel`. `error_pos` specifies where to inject the error, and the kernel performs encoding, error injection, syndrome measurement, and correction.


# %% [markdown]
# `error_pos` is a compile-time parameter. The values `0`, `1`, `2` inject an $X$ error at that location. The value `3` matches no branch, so it means "no error".
#
# First we prepare the logical $\lvert1\rangle$ (an `ry` gate with `theta` $=\pi$). If correction works, `data[0]` should always be 1.


# %%
@qm.qkernel
def bit_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one bit-flip error.

    Args:
        state (qm.Q[qm.Bits[5]]): Three data and two clean syndrome qubits.
        theta (qm.Real): Logical-state Y rotation angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Q[qm.Bits[5]]: Decoded logical qubit, clean data, and syndrome bits.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.cx(state, control=0, target=2)
    state = qm.x(state, target=error_position)

    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=0, target=4)
    state = qm.cx(state, control=2, target=4)

    state = qm.ccx(state, control1=3, control2=4, target=0)
    state = qm.x(state, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=1)
    state = qm.x(state, target=4)
    state = qm.x(state, target=3)
    state = qm.ccx(state, control1=3, control2=4, target=2)
    state = qm.x(state, target=3)

    state = qm.cx(state, control=0, target=2)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def bit_flip_code_samples(
    theta: qm.Param[qm.Real],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the runtime-parametric three-qubit repetition code.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    return qm.measure(bit_flip_code_state(qm.qubits(5), theta, error_position))


# %% [markdown]
# `error_position` is `Const` because it changes circuit topology. `theta` is a
# `Param`, so one compiled artifact can accept different logical amplitudes. The
# two names cannot accidentally overlap as compile-time and launch-time values.

# %% [markdown]
# ## Inspect and estimate the corrected body
#
# We specialize one representative error on data qubit one. No QEC-specific
# operation is introduced: Rust follows the ordinary call and derives the exact
# gate families and five-qubit peak width from its body.

# %%
ERROR_POSITION = 1
program = bit_flip_code_samples.specialize(
    error_position=ERROR_POSITION,
)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.call" in inspection
assert "quantum.error_correction" not in inspection
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (5, 5)
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (17, 17)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (8, 8)
assert study.metric(
    qm.resources.logical_operation_count("quantum.ccx")
).numeric_enclosure() == (3, 3)
assert study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure() == (1, 1)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## Compile once and verify amplitude preservation
#
# For $\theta=\pi/3$, the original logical state has
# $P(1)=\sin^2(\theta/2)=1/4$. A 64-shot sample is sufficient for a bounded
# six-standard-error witness. The two clean data bits must always be zero after
# decoding, and the retained syndrome must always identify data qubit one.

# %%
THETA = math.pi / 3.0
SHOTS = 64
EXPECTED_SYNDROME = (1, 0)

endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(
    artifact,
    on=endpoint,
    parameters={"theta": THETA},
    shots=SHOTS,
    seed=2,
)
result = record.result
expected_one_probability = math.sin(THETA / 2.0) ** 2
observed_one_probability = (
    sum(count for bits, count in result.counts.items() if bits[0] == 1) / result.shots
)
sampling_error = 6.0 * math.sqrt(
    expected_one_probability * (1.0 - expected_one_probability) / SHOTS
)

print("corrected counts:", result.counts)
print("P(logical=1):", observed_one_probability)

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("theta",)
assert record.artifact is artifact
assert all(bits[1:3] == (0, 0) for bits in result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in result.counts)
assert abs(observed_one_probability - expected_one_probability) <= sampling_error

# %% [markdown]
# ### 2.7 Superposition Input
#
# The measured probability confirms that correction preserves the amplitudes of
# $\cos(\theta/2)|0\rangle+\sin(\theta/2)|1\rangle$, not just the classical
# values zero and one.
#
# ### 2.8 Limitation: Powerless Against Phase Errors
#
# The $Z_0Z_1$ and $Z_0Z_2$ parities cannot see a sign change. To correct $Z$,
# move the same repetition code into the $X$ basis.

# %% [markdown]
# ## 3. The 3-Qubit Phase-Flip Code
#
# The bit-flip code could correct only $X$ errors. Next we build a code that corrects the phase error $Z$. Rather than designing one from scratch, we reuse the bit-flip code "in a different basis".


# %% [markdown]
# ### 3.1 The Target Error: Phase Flip
#
# A $Z$ error flips only the sign of $\lvert1\rangle$ ($\lvert0\rangle\to\lvert0\rangle$, $\lvert1\rangle\to-\lvert1\rangle$). As seen in 1.4, it is an error invisible when measured in the computational basis. The phase-flip code aims to correct this single $Z$ error.


# %% [markdown]
# ### 3.2 The Key Identity: $H$ Swaps $X$ and $Z$
#
# The Hadamard gate $H$ satisfies the following relations.
#
# $$HZH = X, \qquad HXH = Z$$
#
# That is, conjugating by $H$ swaps $X$ errors and $Z$ errors. The bit-flip code could correct $X$ errors. If we change the basis of each qubit with $H$, that code becomes a code that corrects $Z$ errors. This is the phase-flip code.


# %% [markdown]
# ### 3.3 The Code Space
#
# Since $H\lvert0\rangle=\lvert+\rangle$ and $H\lvert1\rangle=\lvert-\rangle$, transforming the bit-flip codewords $\lvert000\rangle$ and $\lvert111\rangle$ with $H$ on all three qubits gives the logical states of the phase-flip code.
#
# - $\lvert0_L\rangle = \lvert+++\rangle$
# - $\lvert1_L\rangle = \lvert---\rangle$
#
# Just as an $X$ error swaps $\lvert0\rangle\leftrightarrow\lvert1\rangle$ in the bit-flip code, a $Z$ error swaps $\lvert+\rangle\leftrightarrow\lvert-\rangle$ in the phase-flip code ($Z\lvert+\rangle=\lvert-\rangle$).


# %% [markdown]
# ### 3.4 The Encoding Circuit
#
# Encoding is just the bit-flip encoding followed by an $H$ on all three qubits.


# %% [markdown]
# ### 3.5 Syndrome Measurement: $X$ Parity
#
# For the bit-flip code we measured the $Z$ parity $Z_iZ_j$. For the phase-flip code, with $X$ and $Z$ swapped, we measure the $X$ parity $X_iX_j$.
#
# To measure $X_iX_j$, prepare an ancilla in $\lvert+\rangle$ (apply $H$), use it as the control for CNOTs into the two data qubits, then apply $H$ again before measuring. The syndrome-to-error correspondence has the same shape as for the bit-flip code; only the correction changes from $X$ to $Z$.
#
# | Error | Syndrome $(s_0, s_1)$ | Correction |
# | --- | --- | --- |
# | none | $(0, 0)$ | none |
# | $Z_0$ | $(1, 1)$ | $Z_0$ |
# | $Z_1$ | $(1, 0)$ | $Z_1$ |
# | $Z_2$ | $(0, 1)$ | $Z_2$ |


# %% [markdown]
# ### 3.6 Implementation and Run
#
# We collect everything into a single `@qkernel`. This time we prepare the logical $\lvert0_L\rangle=\lvert+++\rangle$.


# %% [markdown]
# After correction, `data[0]` is back in $\lvert+\rangle$. Since $\lvert+\rangle$ measured directly gives 0 and 1 half the time, we apply one final $H$ to `data[0]` to turn it into $\lvert0\rangle$ before measuring. If correction works, `data[0]` should always be 0.


# %% [markdown]
# ### 3.7 Limitation: Only One Error Type
#
# The phase-flip code can correct $Z$ errors, but now it cannot correct $X$ errors. Since it is a basis transformation of the bit-flip code, the errors it can correct were simply swapped.
#
# The bit-flip code handles only $X$, the phase-flip code only $Z$ — each corrects just one error type. But real noise causes both $X$ and $Z$, and also $Y$, where both occur at once. In the next section we combine the two codes into **Shor's 9-qubit code**, which corrects any single-qubit error.


# %%
@qm.qkernel
def phase_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one phase-flip error.

    Args:
        state (qm.Q[qm.Bits[5]]): Three data and two clean syndrome qubits.
        theta (qm.Real): Logical-state Y rotation angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Q[qm.Bits[5]]: Decoded logical qubit, clean data, and syndrome bits.
    """

    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.cx(state, control=0, target=2)
    for index in range(3):
        state = qm.h(state, target=index)
    state = qm.z(state, target=error_position)
    for index in range(3):
        state = qm.h(state, target=index)

    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=0, target=4)
    state = qm.cx(state, control=2, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=0)
    state = qm.x(state, target=4)
    state = qm.ccx(state, control1=3, control2=4, target=1)
    state = qm.x(state, target=4)
    state = qm.x(state, target=3)
    state = qm.ccx(state, control1=3, control2=4, target=2)
    state = qm.x(state, target=3)
    state = qm.cx(state, control=0, target=2)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def phase_flip_code_samples(
    theta: qm.Param[qm.Real],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[5]]:  # type: ignore[valid-type]
    """Sample the runtime-parametric phase-flip repetition code.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[5]]: Logical, clean-data, and syndrome bits.
    """

    return qm.measure(phase_flip_code_state(qm.qubits(5), theta, error_position))


phase_result = qm.run(
    phase_flip_code_samples,
    on=endpoint,
    error_position=1,
    parameters={"theta": THETA},
    shots=SHOTS,
    seed=3,
).result
phase_one_probability = (
    sum(count for bits, count in phase_result.counts.items() if bits[0] == 1)
    / phase_result.shots
)
assert all(bits[1:3] == (0, 0) for bits in phase_result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in phase_result.counts)
assert abs(phase_one_probability - expected_one_probability) <= sampling_error

# %% [markdown]
# ## 4. Shor's 9-Qubit Code
#
# Combining the bit-flip code and the phase-flip code should let us correct both $X$ and $Z$. **Shor's 9-qubit code** is what realizes this.


# %% [markdown]
# ### 4.1 The Idea: Nest Two Codes
#
# Shor's code encodes in two stages.
#
# 1. Encode one qubit into three with the phase-flip code.
# 2. Encode each of those three qubits into three more with the bit-flip code.
#
# This spreads 1 → 3 → 9 qubits. This construction of "putting one more code inside a code" is called a **concatenated code**. The outer phase-flip layer handles $Z$ errors, and the inner bit-flip layer handles $X$ errors.


# %% [markdown]
# ### 4.2 Viewing 9 Qubits as 3 Blocks
#
# We view the 9 qubits as three **blocks**.
#
# ```text
# (q0, q1, q2)   (q3, q4, q5)   (q6, q7, q8)
# ```
#
# Each block is the inner bit-flip code. The representatives of the three blocks, $q_0, q_3, q_6$, form the outer phase-flip code.


# %% [markdown]
# ### 4.3 The Encoding Circuit
#
# Encoding is just applying the outer phase-flip encoding to $q_0, q_3, q_6$, then bit-flip encoding each block.


# %% [markdown]
# ### 4.4 Syndrome Measurement
#
# Shor's syndrome has 8 bits, split into two kinds.
#
# - **Intra-block $Z$ parities** (`anc[0]`–`anc[5]`, two per block): these are exactly the bit-flip code's syndrome measurement, locating an $X$ error within each block.
# - **Inter-block $X$ parities** (`anc[6]`, `anc[7]`): the two parities $X_0X_1X_2X_3X_4X_5$ and $X_3X_4X_5X_6X_7X_8$. These correspond to the phase-flip code's syndrome measurement, identifying the block containing a $Z$ error.


# %% [markdown]
# ### 4.5 Why $Y$ Errors Are Corrected
#
# Shor's code corrects not only $X$ and $Z$ but also $Y$ errors. Since $Y=iXZ$, a $Y$ error contains both an $X$ component and a $Z$ component.
#
# The intra-block $Z$ parities detect the $X$ component and the inter-block $X$ parities detect the $Z$ component, independently. With both the $X$ correction and the $Z$ correction applied, the $Y$ error is cancelled.

# %% [markdown]
# ### 4.6 Implementation and Run
#
# `error_type` uses `1=X`, `2=Y`, and `3=Z`. Both the error type and location
# are `Const` values because they select circuit structure in this bounded
# witness. `theta` remains a runtime parameter, exactly as in the smaller codes.


# %%
@qm.qkernel
def flip_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Flip one component of a Shor data block.

    Args:
        state (qm.Q[qm.Bits[9]]): Nine encoded data qubits.
        position (qm.Int): Component to flip.

    Returns:
        qm.Q[qm.Bits[9]]: Updated data block.
    """

    return qm.x(state, target=position)


@qm.qkernel
def phase_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Apply one phase correction to a Shor data block.

    Args:
        state (qm.Q[qm.Bits[9]]): Nine encoded data qubits.
        position (qm.Int): Component receiving the phase correction.

    Returns:
        qm.Q[qm.Bits[9]]: Updated data block.
    """

    return qm.z(state, target=position)


@qm.qkernel
def shor_code_samples(
    theta: qm.Param[qm.Real],
    error_type: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[17]]:  # type: ignore[valid-type]
    """Encode, correct, decode, and sample one Shor-code experiment.

    Args:
        theta (qm.Param[qm.Real]): Runtime logical-state angle in radians.
        error_type (qm.Const[qm.Int]): Pauli error selector: 1, 2, or 3.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[17]]: Decoded data followed by measured syndromes.
    """

    state = qm.qubits(17)
    state = qm.ry(state, theta, target=0)

    # Outer phase-flip encoding on representatives 0, 3, and 6.
    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=0, target=6)
    for target in range(0, 9, 3):
        state = qm.h(state, target=target)

    # Inner bit-flip encoding in each three-qubit block.
    for block in range(3):
        representative = 3 * block
        state = qm.cx(state, control=representative, target=representative + 1)
        state = qm.cx(state, control=representative, target=representative + 2)

    if error_type == 1:
        state = qm.x(state, target=error_position)
    elif error_type == 2:
        state = qm.y(state, target=error_position)
    elif error_type == 3:
        state = qm.z(state, target=error_position)

    # Six intra-block Z-parity syndromes occupy components 9 through 14.
    for block in range(3):
        representative = 3 * block
        first_syndrome = 9 + 2 * block
        second_syndrome = first_syndrome + 1
        state = qm.cx(state, control=representative, target=first_syndrome)
        state = qm.cx(state, control=representative + 1, target=first_syndrome)
        state = qm.cx(state, control=representative, target=second_syndrome)
        state = qm.cx(state, control=representative + 2, target=second_syndrome)

    # Inter-block X parities occupy components 15 and 16.
    for parity in range(2):
        syndrome = 15 + parity
        state = qm.h(state, target=syndrome)
        for target in range(3 * parity, 3 * parity + 6):
            state = qm.cx(state, control=syndrome, target=target)
        state = qm.h(state, target=syndrome)

    # Keep syndrome extraction coherent and realize each lookup table with a
    # two-control call. Control component zero is the least-significant bit.
    data, syndromes = qm.split_at(state, 9)
    x0, syndromes = qm.split_at(syndromes, 2)
    x1, syndromes = qm.split_at(syndromes, 2)
    x2, phase = qm.split_at(syndromes, 2)

    x0, data = qm.control(flip_shor_data, x0, data, 0, control_state=3)
    x0, data = qm.control(flip_shor_data, x0, data, 1, control_state=1)
    x0, data = qm.control(flip_shor_data, x0, data, 2, control_state=2)
    x1, data = qm.control(flip_shor_data, x1, data, 3, control_state=3)
    x1, data = qm.control(flip_shor_data, x1, data, 4, control_state=1)
    x1, data = qm.control(flip_shor_data, x1, data, 5, control_state=2)
    x2, data = qm.control(flip_shor_data, x2, data, 6, control_state=3)
    x2, data = qm.control(flip_shor_data, x2, data, 7, control_state=1)
    x2, data = qm.control(flip_shor_data, x2, data, 8, control_state=2)

    phase, data = qm.control(phase_shor_data, phase, data, 0, control_state=1)
    phase, data = qm.control(phase_shor_data, phase, data, 3, control_state=3)
    phase, data = qm.control(phase_shor_data, phase, data, 6, control_state=2)

    # Invert the inner and outer encoders to expose the logical qubit at q0.
    for block in range(3):
        representative = 3 * block
        data = qm.cx(data, control=representative, target=representative + 2)
        data = qm.cx(data, control=representative, target=representative + 1)
        data = qm.h(data, target=representative)
    data = qm.cx(data, control=0, target=6)
    data = qm.cx(data, control=0, target=3)
    syndromes = qm.concat(x0, x1, x2, phase)
    return qm.measure(qm.concat(data, syndromes))


# %%
shor_cases = (("X", 1, 0), ("Y", 2, 4), ("Z", 3, 8))
shor_results = tuple(
    qm.run(
        shor_code_samples,
        on=endpoint,
        error_type=error_type,
        error_position=error_position,
        parameters={"theta": math.pi},
        shots=1,
        seed=10 + error_position,
    ).result
    for _, error_type, error_position in shor_cases
)

for (name, _, position), shor_result in zip(
    shor_cases,
    shor_results,
    strict=True,
):
    print(f"Shor code: {name} error on q[{position}] -> {shor_result.counts}")
    assert all(bits[0] == 1 for bits in shor_result.counts)
#
# ### 4.7 Why It Corrects “Any Single Error”
#
# Syndrome extraction projects an arbitrary one-qubit error onto one of the
# Pauli components $I,X,Y,Z$. Shor's code corrects each component, so by
# linearity it corrects any single-qubit error.

# %% [markdown]
# ## 5. The Common Pattern
#
# We have built three codes — bit-flip, phase-flip, and Shor. In fact all of them follow the same four-step template.
#
# 1. **Encode into a code space**: spread one logical qubit across multiple physical qubits using entanglement.
# 2. **Parity measurement**: instead of measuring the data directly, extract a parity of several qubits ($Z_iZ_j$ or $X_iX_j$) into ancilla qubits.
# 3. **Syndrome**: read the location and type of the error from the parity measurements.
# 4. **Feedback correction**: apply a Pauli gate according to the syndrome to cancel the error.
#
# This skeleton does not change as the code changes. What differs is only "which parity to measure".


# %% [markdown]
# ### 5.1 Naming the Parity Operators: Stabilizers
#
# The parity operators we have been measuring — $Z_0Z_1$, $X_0X_1$, and so on — have a name: **stabilizers**.
#
# A stabilizer is a Pauli operator that leaves every state of the code space unchanged. Applied to a valid codeword, the state is unchanged (eigenvalue $+1$). When an error occurs, the measured value of a stabilizer that anticommutes with that error flips to $-1$, and that becomes a bit of the syndrome.
#
# The $Z_0Z_1$ of the bit-flip code, the $X_0X_1$ of the phase-flip code, and the eight parities of Shor's code are all stabilizers. The three codes were different manifestations of one idea: "measure stabilizers to obtain a syndrome." This view is called the **stabilizer formalism**, and the second part treats it in earnest.


# %% [markdown]
# ### 5.2 Summary of the Three Codes
#
# We summarize the three codes together with their stabilizers.
#
# | Code | $[[n,k,d]]$ | Stabilizer generators | Corrects |
# | --- | --- | --- | --- |
# | 3-qubit bit-flip | $[[3,1,1]]$ | $Z_0Z_1,\ Z_0Z_2$ | a single $X$ |
# | 3-qubit phase-flip | $[[3,1,1]]$ | $X_0X_1,\ X_0X_2$ | a single $Z$ |
# | Shor 9-qubit | $[[9,1,3]]$ | $Z_0Z_1,\ Z_0Z_2,\ Z_3Z_4,\ Z_3Z_5,\ Z_6Z_7,\ Z_6Z_8,$ $X_0X_1X_2X_3X_4X_5,\ X_3X_4X_5X_6X_7X_8$ | a single $X,\ Y,\ Z$ |
#
# $[[n,k,d]]$ is a notation for a code: $n$ is the number of physical qubits, $k$ is the number of logical qubits protected, and $d$ is the **code distance**. The distance $d$ is an indicator with the meaning "$d\ge3$ is required to correct any single-qubit error". The 3-qubit codes have $d=1$ — they can correct only one specific type of error ($X$ or $Z$). Shor's code has $d=3$ and corrects any single error. The precise definition of distance is covered in the second part.

# %% [markdown]
# ## 6. Try It Yourself
#
# To test your understanding, modify the code a little and run it.
#
# - **Change the error location**: set `error_pos` of each `*_syndrome_run` to various values and confirm that correction works.
# - **Make it fail on purpose**: below, we experience the limitation of the bit-flip code.


# %% [markdown]
# ### Make It Fail on Purpose
#
# The bit-flip code can correct only up to a single $X$ error. What happens when $X$ errors occur at two locations?
#
# The following kernel injects $X$ errors at two locations, `data[0]` and `data[1]`, into the logical $\lvert1\rangle$ ($\lvert111\rangle$), and performs syndrome measurement and correction as usual.


# %% [markdown]
# `data[0]` becomes 0, not 1. Far from fixing the state, the correction turned the logical $\lvert1\rangle$ into the logical $\lvert0\rangle$.
#
# With $X$ at two locations, $\lvert001\rangle$ looks "different in only one place" from $\lvert111\rangle$, so the correction applies an $X$ to the remaining location as well, ending at $\lvert000\rangle$. This is the limitation "up to a single error" — the actual meaning of the bit-flip code having $d=1$ in the table of 5.2.

# %% [markdown]
# ## From bit flips to general Pauli errors
#
# A three-qubit phase-flip code is the same repetition construction conjugated
# by Hadamards: $HZH=X$ turns one $Z$ error into the bit-flip problem above.
# Correcting an arbitrary single-qubit error requires independent bit and phase
# syndromes because $Y=iXZ$. The next article,
# [Stabilizer Formalism and the Steane Code](steane_code.ipynb), implements that
# complete composition with one bounded ten-qubit Qamomile program. Keeping
# its execution witness there avoids repeating another deep statevector test in
# this introductory page.

# %% [markdown]
# ## 7. Summary
#
# In this article we implemented three quantum error correction codes.
#
# - **The 3-qubit bit-flip code** — corrects a single $X$ error with $Z$ parity checks.
# - **The 3-qubit phase-flip code** — corrects a single $Z$ error with $X$ parity checks.
# - **Shor's 9-qubit code** — concatenates the two and corrects any single-qubit error ($X,\ Y,\ Z$).
#
# The common skeleton was "encode into a code space → parity (stabilizer) measurement → syndrome → feedback correction".
#
# ### Next
#
# The second part, [Stabilizer Formalism and the Steane Code](steane_code.ipynb), treats the stabilizers — only named here — formally. Its main subjects are the **CSS construction**, which systematically builds quantum codes from classical codes, and the **Steane code**, which achieves $d=3$ with seven qubits, fewer than Shor's nine.
