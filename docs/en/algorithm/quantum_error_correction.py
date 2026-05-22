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

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile
# # or
# # !uv add qamomile

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
# Before getting into the implementation, we load Qamomile and the Qiskit backend and define two helper functions. `_first_bit_distribution` and `_sample_first_bit` are just utilities that compile and run a kernel and return the 0/1 counts of the first bit. They are not central to QEC, so feel free to skip them.

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create a seeded backend for reproducible documentation output.
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _first_bit_distribution(result: SampleResult) -> dict[int, int]:
    """Return the 0/1 counts for the first measured bit."""
    counts = {0: 0, 1: 0}
    for outcome, count in result.results:
        bit = outcome[0] if isinstance(outcome, (list, tuple)) else outcome & 1
        counts[bit] += count
    return counts


def _sample_first_bit(
    kernel,
    *,
    bindings: dict[str, object] | None = None,
    parameters: list[str] | None = None,
    runtime_bindings: dict[str, object] | None = None,
    shots: int = 256,
) -> dict[int, int]:
    """Compile and run a kernel, returning the 0/1 counts of the first bit."""
    executable = transpiler.transpile(
        kernel,
        bindings=bindings or {},
        parameters=parameters or [],
    )
    job = executable.sample(
        _seeded_executor,
        shots=shots,
        bindings=runtime_bindings or {},
    )
    return _first_bit_distribution(job.result())


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


# %%
@qmc.qkernel
def encode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    return q0, q1, q2


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


# %%
@qmc.qkernel
def bitflip_syndrome_run(
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    # Allocate 3 data qubits and 2 ancilla qubits for syndrome measurement.
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    # Prepare the logical state with ry(theta), then encode it into 3 qubits.
    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    # Inject an X error at error_pos (error_pos=3 means no error).
    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.x(data[i])

    # Syndrome measurement 1: extract the Z0 Z1 parity into anc[0].
    data[0], anc[0] = qmc.cx(data[0], anc[0])
    data[1], anc[0] = qmc.cx(data[1], anc[0])
    s0 = qmc.measure(anc[0])

    # Syndrome measurement 2: extract the Z0 Z2 parity into anc[1].
    data[0], anc[1] = qmc.cx(data[0], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    s1 = qmc.measure(anc[1])

    # Identify the error location from the syndrome (s0, s1) and correct with X.
    if s0 & s1:  # (1, 1) -> data[0]
        data[0] = qmc.x(data[0])
    if s0 & ~s1:  # (1, 0) -> data[1]
        data[1] = qmc.x(data[1])
    if ~s0 & s1:  # (0, 1) -> data[2]
        data[2] = qmc.x(data[2])

    return qmc.measure(data)


# %% [markdown]
# `error_pos` is a compile-time parameter. The values `0`, `1`, `2` inject an $X$ error at that location. The value `3` matches no branch, so it means "no error".
#
# First we prepare the logical $\lvert1\rangle$ (an `ry` gate with `theta` $=\pi$). If correction works, `data[0]` should always be 1.

# %%
bitflip_cases = [
    ("no error", 3),
    ("X on data[0]", 0),
    ("X on data[1]", 1),
    ("X on data[2]", 2),
]

print("3-qubit bit-flip code: logical |1>")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi},
    )
    print(f"  {label:14s}: data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")

# %% [markdown]
# ### 2.7 Superposition Input
#
# The code also preserves amplitudes. The state prepared with `theta` $=\pi/3$ has probability
#
# $$P(\text{data}[0]=1)=\sin^2(\pi/6)=0.25.$$
#
# This probability should be preserved regardless of the injected error.

# %%
print("3-qubit bit-flip code: superposition input")
for label, error_pos in bitflip_cases:
    counts = _sample_first_bit(
        bitflip_syndrome_run,
        bindings={"error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi / 3},
        shots=2000,
    )
    total = counts[0] + counts[1]
    print(f"  {label:14s}: P(data[0]=1) = {counts[1] / total:.3f}")

# %% [markdown]
# ### 2.8 Limitation: Powerless Against Phase Errors
#
# The bit-flip code can correct only $X$ errors. It is powerless against the phase error $Z$.
#
# The reason lies in how the $Z$ parity is measured. A $Z$ error only changes the sign of $\lvert000\rangle$ or $\lvert111\rangle$; it does not change the bit values. The $Z_iZ_j$ parity only looks at whether bits are equal, so even when a $Z$ error occurs the syndrome stays $(0,0)$ — the error cannot be detected.
#
# Correcting $Z$ errors requires a different code. In the next section we build the **phase-flip code**, which uses Hadamard gates to move the bit-flip code into "the world of phases".

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


# %%
@qmc.qkernel
def encode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    # After bit-flip encoding, move all 3 qubits to the X basis with H.
    q0, q1, q2 = encode_3qubit_bitflip(q0, q1, q2)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    return q0, q1, q2


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


# %%
@qmc.qkernel
def phaseflip_syndrome_run(error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    # Allocate 3 data qubits and 2 ancilla qubits for syndrome measurement.
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    # Encode into the logical |0_L> = |+++>.
    data[0], data[1], data[2] = encode_3qubit_phaseflip(data[0], data[1], data[2])

    # Inject a Z error at error_pos (error_pos=3 means no error).
    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.z(data[i])

    # Syndrome measurement 1: extract the X0 X1 parity into anc[0] (H gives the X basis).
    anc[0] = qmc.h(anc[0])
    anc[0], data[0] = qmc.cx(anc[0], data[0])
    anc[0], data[1] = qmc.cx(anc[0], data[1])
    anc[0] = qmc.h(anc[0])
    s0 = qmc.measure(anc[0])

    # Syndrome measurement 2: extract the X0 X2 parity into anc[1].
    anc[1] = qmc.h(anc[1])
    anc[1], data[0] = qmc.cx(anc[1], data[0])
    anc[1], data[2] = qmc.cx(anc[1], data[2])
    anc[1] = qmc.h(anc[1])
    s1 = qmc.measure(anc[1])

    # Identify the error location from the syndrome (s0, s1) and correct with Z.
    if s0 & s1:  # (1, 1) -> data[0]
        data[0] = qmc.z(data[0])
    if s0 & ~s1:  # (1, 0) -> data[1]
        data[1] = qmc.z(data[1])
    if ~s0 & s1:  # (0, 1) -> data[2]
        data[2] = qmc.z(data[2])

    # data[0] is back in |+>, so apply H to turn it into |0> before measuring.
    data[0] = qmc.h(data[0])
    return qmc.measure(data)


# %% [markdown]
# After correction, `data[0]` is back in $\lvert+\rangle$. Since $\lvert+\rangle$ measured directly gives 0 and 1 half the time, we apply one final $H$ to `data[0]` to turn it into $\lvert0\rangle$ before measuring. If correction works, `data[0]` should always be 0.

# %%
phaseflip_cases = [
    ("no error", 3),
    ("Z on data[0]", 0),
    ("Z on data[1]", 1),
    ("Z on data[2]", 2),
]

print("3-qubit phase-flip code: logical |0_L> = |+++>")
for label, error_pos in phaseflip_cases:
    counts = _sample_first_bit(
        phaseflip_syndrome_run,
        bindings={"error_pos": error_pos},
    )
    print(f"  {label:14s}: data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")

# %% [markdown]
# ### 3.7 Limitation: Only One Error Type
#
# The phase-flip code can correct $Z$ errors, but now it cannot correct $X$ errors. Since it is a basis transformation of the bit-flip code, the errors it can correct were simply swapped.
#
# The bit-flip code handles only $X$, the phase-flip code only $Z$ — each corrects just one error type. But real noise causes both $X$ and $Z$, and also $Y$, where both occur at once. In the next section we combine the two codes into **Shor's 9-qubit code**, which corrects any single-qubit error.

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


# %%
@qmc.qkernel
def encode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # Outer layer: encode q[0], q[3], q[6] with the phase-flip code.
    q[0], q[3], q[6] = encode_3qubit_phaseflip(q[0], q[3], q[6])

    # Inner layer: encode each of the 3 blocks with the bit-flip code.
    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = encode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = encode_3qubit_bitflip(q[6], q[7], q[8])
    return q


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
# We collect everything into a single `@qkernel`. `error_type` is `1=X`, `2=Y`, `3=Z`, and `error_pos` is the index of the qubit to inject the error into.
#
# Even after correction, the logical state is still encoded across nine qubits. For this demonstration we apply the inverse encoding circuit at the end so that the logical bit can be read directly from `q[0]`. This inverse encoding is a step for checking the result; the correction itself is already complete with syndrome measurement and feedback.


# %%
@qmc.qkernel
def shor_syndrome_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    # Allocate 9 data qubits and 8 ancilla qubits for syndrome measurement.
    q = qmc.qubit_array(9, name="q")
    anc = qmc.qubit_array(8, name="anc")

    # Prepare the logical state with ry(theta), then encode it into 9 qubits.
    q[0] = qmc.ry(q[0], theta)
    q = encode_shor(q)

    # Inject the X / Y / Z error specified by error_type / error_pos.
    for i in qmc.range(9):
        if (error_type == 1) & (error_pos == i):  # 1: X error
            q[i] = qmc.x(q[i])
        if (error_type == 2) & (error_pos == i):  # 2: Y error
            q[i] = qmc.y(q[i])
        if (error_type == 3) & (error_pos == i):  # 3: Z error
            q[i] = qmc.z(q[i])

    # Intra-block Z parity: for block b, anc[2b]=Z(3b,3b+1), anc[2b+1]=Z(3b,3b+2).
    for b in qmc.range(3):
        q[3 * b], anc[2 * b] = qmc.cx(q[3 * b], anc[2 * b])
        q[3 * b + 1], anc[2 * b] = qmc.cx(q[3 * b + 1], anc[2 * b])
        q[3 * b], anc[2 * b + 1] = qmc.cx(q[3 * b], anc[2 * b + 1])
        q[3 * b + 2], anc[2 * b + 1] = qmc.cx(q[3 * b + 2], anc[2 * b + 1])

    # Inter-block X parity: anc[6] spans q[0..5], anc[7] spans q[3..8].
    for p in qmc.range(2):
        anc[6 + p] = qmc.h(anc[6 + p])
        for i in qmc.range(6):
            anc[6 + p], q[3 * p + i] = qmc.cx(anc[6 + p], q[3 * p + i])
        anc[6 + p] = qmc.h(anc[6 + p])

    # X-component correction: per block, measure the syndrome, locate the X error, and correct.
    for b in qmc.range(3):
        s0 = qmc.measure(anc[2 * b])
        s1 = qmc.measure(anc[2 * b + 1])
        if s0 & s1:  # (1, 1) -> qubit 0 of the block
            q[3 * b] = qmc.x(q[3 * b])
        if s0 & ~s1:  # (1, 0) -> qubit 1 of the block
            q[3 * b + 1] = qmc.x(q[3 * b + 1])
        if ~s0 & s1:  # (0, 1) -> qubit 2 of the block
            q[3 * b + 2] = qmc.x(q[3 * b + 2])

    # Z-component correction: find the block holding the Z error, apply Z to its representative.
    phase_s0 = qmc.measure(anc[6])
    phase_s1 = qmc.measure(anc[7])
    if phase_s0 & ~phase_s1:
        q[0] = qmc.z(q[0])
    if phase_s0 & phase_s1:
        q[3] = qmc.z(q[3])
    if ~phase_s0 & phase_s1:
        q[6] = qmc.z(q[6])

    # Verification: apply the inverse encoder to collect the logical bit into q[0].
    for b in qmc.range(3):
        q[3 * b], q[3 * b + 1] = qmc.cx(q[3 * b], q[3 * b + 1])
        q[3 * b], q[3 * b + 2] = qmc.cx(q[3 * b], q[3 * b + 2])
        q[3 * b] = qmc.h(q[3 * b])
    q[0], q[3] = qmc.cx(q[0], q[3])
    q[0], q[6] = qmc.cx(q[0], q[6])

    return qmc.measure(q)


# %% [markdown]
# We try one representative error per block ($X$ on block 0, $Y$ on block 1, $Z$ on block 2). If the logical $\lvert1\rangle$ is preserved, `q[0]` should always be 1.

# %%
shor_cases = [
    ("X", 1, 0),
    ("Y", 2, 4),
    ("Z", 3, 8),
]

print("Shor 9-qubit code: logical |1>")
print(f"  {'error':6s} | {'pos':5s} | P(q[0]=1)")
print(f"  {'-' * 6}-+-{'-' * 5}-+-{'-' * 9}")
for name, error_type, error_pos in shor_cases:
    counts = _sample_first_bit(
        shor_syndrome_run,
        bindings={"error_type": error_type, "error_pos": error_pos},
        parameters=["theta"],
        runtime_bindings={"theta": math.pi},
    )
    total = counts[0] + counts[1]
    print(f"  {name:6s} | q[{error_pos}]  | {counts[1] / total:.3f}")

# %% [markdown]
# ### 4.7 Why It Corrects "Any Single Error"
#
# Shor's code corrects all three of $X$, $Y$, $Z$. This directly means it "corrects any single-qubit error".
#
# As seen in 1.4, any error can be written, as a matrix, as a linear combination of $I, X, Y, Z$, and syndrome measurement collapses that superposition into a single discrete Pauli error. What remains after the collapse is just one of $X$, $Y$, $Z$ (or no error), all of which Shor's code can correct. This is why it handles any single-qubit error, including continuous ones.
#
# The formal treatment is organized in the second part, [Stabilizer Formalism and the Steane Code](steane_code.ipynb).

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

# %%
@qmc.qkernel
def bitflip_two_errors(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    # Inject X errors at two locations (beyond the single-error correction capability).
    data[0] = qmc.x(data[0])
    data[1] = qmc.x(data[1])

    # Syndrome measurement and correction are the same as in bitflip_syndrome_run.
    data[0], anc[0] = qmc.cx(data[0], anc[0])
    data[1], anc[0] = qmc.cx(data[1], anc[0])
    s0 = qmc.measure(anc[0])
    data[0], anc[1] = qmc.cx(data[0], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    s1 = qmc.measure(anc[1])

    if s0 & s1:
        data[0] = qmc.x(data[0])
    if s0 & ~s1:
        data[1] = qmc.x(data[1])
    if ~s0 & s1:
        data[2] = qmc.x(data[2])

    return qmc.measure(data)


# %%
print("bit-flip code with TWO X errors (logical |1>)")
counts = _sample_first_bit(
    bitflip_two_errors,
    parameters=["theta"],
    runtime_bindings={"theta": math.pi},
)
print(f"  data[0]=0 -> {counts[0]:3d}, data[0]=1 -> {counts[1]:3d}")

# %% [markdown]
# `data[0]` becomes 0, not 1. Far from fixing the state, the correction turned the logical $\lvert1\rangle$ into the logical $\lvert0\rangle$.
#
# With $X$ at two locations, $\lvert001\rangle$ looks "different in only one place" from $\lvert111\rangle$, so the correction applies an $X$ to the remaining location as well, ending at $\lvert000\rangle$. This is the limitation "up to a single error" — the actual meaning of the bit-flip code having $d=1$ in the table of 5.2.

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
