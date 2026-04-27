# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Introduction to Quantum Error Correction
#
# Quantum error correction (QEC) protects fragile quantum states by spreading one logical state across multiple physical qubits. The key point is that we do not measure the logical state directly. Instead, we measure only enough information to identify the error.
#
# In this tutorial, we implement three codes with Qamomile's `@qkernel`.
#
# 1. Correct a single $X$ error with the 3-qubit bit-flip code.
# 2. Turn that idea into a phase-flip code with Hadamard gates, and correct a single $Z$ error.
# 3. Combine both ideas into Shor's 9-qubit code, which corrects a single $X$, $Y$, or $Z$ error.
# 4. Reinterpret the circuits with the stabilizer formalism.
#
# The central tool throughout is **syndrome measurement**: use ancilla qubits to learn where the error happened without reading out the logical state itself.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile
# # or
# # !uv add qamomile

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()


def _first_bit_distribution(result) -> dict[int, int]:
    """Return counts for the first measured bit."""
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
    """Compile a tutorial kernel and return the first-bit count distribution."""
    executable = transpiler.transpile(
        kernel,
        bindings=bindings or {},
        parameters=parameters or [],
    )
    job = executable.sample(
        transpiler.executor(),
        shots=shots,
        bindings=runtime_bindings or {},
    )
    return _first_bit_distribution(job.result())


# %% [markdown]
# ## 1. What We Need To Correct
#
# In a classical repetition code, we can send a bit $b$ as `bbb` and recover from one flipped bit by majority vote. The same idea cannot be copied directly into quantum computing.
#
# - An unknown quantum state $\lvert\psi\rangle$ cannot be cloned.
# - Measuring the logical state directly destroys superposition.
#
# Quantum error correction works around both issues. It uses entanglement instead of copying, and it measures **syndromes** instead of measuring the logical information.
#
# The standard flow is:
#
# ```text
# encode -> error -> syndrome measurement -> correction -> encoded state
# ```

# %% [markdown]
# ## 2. The 3-Qubit Bit-Flip Code
#
# We start with the simplest code: a code that corrects only bit-flip errors, represented by $X$.
#
# A logical state
#
# $$\alpha\lvert 0\rangle + \beta\lvert 1\rangle$$
#
# is encoded into three physical qubits as
#
# $$\alpha\lvert 000\rangle + \beta\lvert 111\rangle.$$
#
# This is not cloning. The two CNOT gates create correlations with the data qubit.


# %%
@qmc.qkernel
def encode_3qubit_bitflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    q0, q2 = qmc.cx(q0, q2)
    return q0, q1, q2


# %% [markdown]
# ### Syndrome Measurement
#
# The bit-flip code measures two $Z$ parities:
#
# - $S_0 = Z_0Z_1$
# - $S_1 = Z_0Z_2$
#
# In the code space $\{\lvert000\rangle,\lvert111\rangle\}$, both parities say that the two compared qubits are equal. A single $X$ error flips a unique pattern of parities, so the syndrome identifies the error location.
#
# | Error | $(s_0, s_1)$ | Correction |
# | --- | --- | --- |
# | none | $(0, 0)$ | none |
# | $X_0$ | $(1, 1)$ | $X_0$ |
# | $X_1$ | $(1, 0)$ | $X_1$ |
# | $X_2$ | $(0, 1)$ | $X_2$ |
#
# To measure $Z_iZ_j$, prepare an ancilla and apply `CX(data[i], anc); CX(data[j], anc); measure(anc)`. Only the ancilla is measured.


# %%
@qmc.qkernel
def bitflip_syndrome_run(
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0] = qmc.ry(data[0], theta)
    data[0], data[1], data[2] = encode_3qubit_bitflip(data[0], data[1], data[2])

    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.x(data[i])

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


# %% [markdown]
# `error_pos` is a compile-time parameter. Values `0`, `1`, and `2` inject an $X$ error at that location. The value `3` does not match any branch, so it means no error.
#
# First prepare logical $\lvert1\rangle$. After correction, `data[0]` should always be 1.

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
# The code also preserves amplitudes. With $\theta=\pi/3$, the prepared state has
#
# $$P(data[0]=1)=\sin^2(\pi/6)=0.25.$$
#
# That probability should be unchanged by the injected error.

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
# ## 3. The 3-Qubit Phase-Flip Code
#
# Phase-flip errors are represented by $Z$. Hadamard gates swap $X$ and $Z$:
#
# $$H Z H = X,\qquad H X H = Z.$$
#
# So a phase-flip code is just the bit-flip code in the Hadamard basis. Its logical basis states are:
#
# - $\lvert0_L\rangle=\lvert+++\rangle$
# - $\lvert1_L\rangle=\lvert---\rangle$
#
# The stabilizers are $X$ parities instead of $Z$ parities.


# %%
@qmc.qkernel
def encode_3qubit_phaseflip(
    q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    q0, q1, q2 = encode_3qubit_bitflip(q0, q1, q2)
    q0 = qmc.h(q0)
    q1 = qmc.h(q1)
    q2 = qmc.h(q2)
    return q0, q1, q2


# %% [markdown]
# To measure $X_iX_j$, prepare the ancilla in $\lvert+\rangle$, use it as the control for CNOTs into the data qubits, then apply another $H$ and measure the ancilla.
#
# | Error | $(s_0, s_1)$ | Correction |
# | --- | --- | --- |
# | none | $(0, 0)$ | none |
# | $Z_0$ | $(1, 1)$ | $Z_0$ |
# | $Z_1$ | $(1, 0)$ | $Z_1$ |
# | $Z_2$ | $(0, 1)$ | $Z_2$ |


# %%
@qmc.qkernel
def phaseflip_syndrome_run(error_pos: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(3, name="data")
    anc = qmc.qubit_array(2, name="anc")

    data[0], data[1], data[2] = encode_3qubit_phaseflip(data[0], data[1], data[2])

    for i in qmc.range(3):
        if error_pos == i:
            data[i] = qmc.z(data[i])

    anc[0] = qmc.h(anc[0])
    anc[0], data[0] = qmc.cx(anc[0], data[0])
    anc[0], data[1] = qmc.cx(anc[0], data[1])
    anc[0] = qmc.h(anc[0])
    s0 = qmc.measure(anc[0])

    anc[1] = qmc.h(anc[1])
    anc[1], data[0] = qmc.cx(anc[1], data[0])
    anc[1], data[2] = qmc.cx(anc[1], data[2])
    anc[1] = qmc.h(anc[1])
    s1 = qmc.measure(anc[1])

    if s0 & s1:
        data[0] = qmc.z(data[0])
    if s0 & ~s1:
        data[1] = qmc.z(data[1])
    if ~s0 & s1:
        data[2] = qmc.z(data[2])

    data[0] = qmc.h(data[0])
    return qmc.measure(data)


# %% [markdown]
# We prepare logical $\lvert0_L\rangle=\lvert+++\rangle$. After correction, `data[0]` is back in $\lvert+\rangle$, so applying one final $H$ makes it measure as 0.

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
# ## 4. Shor's 9-Qubit Code
#
# The bit-flip code corrects only $X$ errors, and the phase-flip code corrects only $Z$ errors. To correct any single-qubit Pauli error, we combine both ideas.
#
# Shor's code is a concatenated code:
#
# 1. Encode one qubit with the 3-qubit phase-flip code.
# 2. Encode each of those three qubits with the 3-qubit bit-flip code.
#
# We view the 9 qubits as three blocks:
#
# ```text
# (q0, q1, q2), (q3, q4, q5), (q6, q7, q8)
# ```
#
# The within-block $Z$ parities detect the $X$ component. The across-block $X$ parities detect the $Z$ component. Since $Y=iXZ$, a $Y$ error triggers both parts and is corrected by applying both corrections.


# %%
@qmc.qkernel
def encode_shor(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    q[0], q[3], q[6] = encode_3qubit_phaseflip(q[0], q[3], q[6])

    q[0], q[1], q[2] = encode_3qubit_bitflip(q[0], q[1], q[2])
    q[3], q[4], q[5] = encode_3qubit_bitflip(q[3], q[4], q[5])
    q[6], q[7], q[8] = encode_3qubit_bitflip(q[6], q[7], q[8])
    return q


# %% [markdown]
# Shor's syndrome has eight bits.
#
# - `anc[0..5]`: within-block $Z$ parities, used to locate the $X$ component.
# - `anc[6..7]`: across-block $X$ parities, used to locate the block containing the $Z$ component.
#
# After correction, the logical state is still encoded across nine qubits. For this demonstration, we apply the inverse encoder at the end so that the logical bit is readable from `q[0]`. This final inverse is only a verification step; the correction itself is done by syndrome measurement and feedback.


# %%
@qmc.qkernel
def shor_syndrome_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(9, name="q")
    anc = qmc.qubit_array(8, name="anc")

    q[0] = qmc.ry(q[0], theta)
    q = encode_shor(q)

    for i in qmc.range(9):
        if (error_type == 1) & (error_pos == i):  # 1 means X error.
            q[i] = qmc.x(q[i])
        if (error_type == 2) & (error_pos == i):  # 2 means Y error.
            q[i] = qmc.y(q[i])
        if (error_type == 3) & (error_pos == i):  # 3 means Z error.
            q[i] = qmc.z(q[i])

    # Block 0: Z0Z1, Z0Z2
    q[0], anc[0] = qmc.cx(q[0], anc[0])
    q[1], anc[0] = qmc.cx(q[1], anc[0])
    b0_s0 = qmc.measure(anc[0])
    q[0], anc[1] = qmc.cx(q[0], anc[1])
    q[2], anc[1] = qmc.cx(q[2], anc[1])
    b0_s1 = qmc.measure(anc[1])

    # Block 1: Z3Z4, Z3Z5
    q[3], anc[2] = qmc.cx(q[3], anc[2])
    q[4], anc[2] = qmc.cx(q[4], anc[2])
    b1_s0 = qmc.measure(anc[2])
    q[3], anc[3] = qmc.cx(q[3], anc[3])
    q[5], anc[3] = qmc.cx(q[5], anc[3])
    b1_s1 = qmc.measure(anc[3])

    # Block 2: Z6Z7, Z6Z8
    q[6], anc[4] = qmc.cx(q[6], anc[4])
    q[7], anc[4] = qmc.cx(q[7], anc[4])
    b2_s0 = qmc.measure(anc[4])
    q[6], anc[5] = qmc.cx(q[6], anc[5])
    q[8], anc[5] = qmc.cx(q[8], anc[5])
    b2_s1 = qmc.measure(anc[5])

    # X0X1X2X3X4X5
    anc[6] = qmc.h(anc[6])
    anc[6], q[0] = qmc.cx(anc[6], q[0])
    anc[6], q[1] = qmc.cx(anc[6], q[1])
    anc[6], q[2] = qmc.cx(anc[6], q[2])
    anc[6], q[3] = qmc.cx(anc[6], q[3])
    anc[6], q[4] = qmc.cx(anc[6], q[4])
    anc[6], q[5] = qmc.cx(anc[6], q[5])
    anc[6] = qmc.h(anc[6])
    phase_s0 = qmc.measure(anc[6])

    # X3X4X5X6X7X8
    anc[7] = qmc.h(anc[7])
    anc[7], q[3] = qmc.cx(anc[7], q[3])
    anc[7], q[4] = qmc.cx(anc[7], q[4])
    anc[7], q[5] = qmc.cx(anc[7], q[5])
    anc[7], q[6] = qmc.cx(anc[7], q[6])
    anc[7], q[7] = qmc.cx(anc[7], q[7])
    anc[7], q[8] = qmc.cx(anc[7], q[8])
    anc[7] = qmc.h(anc[7])
    phase_s1 = qmc.measure(anc[7])

    if b0_s0 & b0_s1:
        q[0] = qmc.x(q[0])
    if b0_s0 & ~b0_s1:
        q[1] = qmc.x(q[1])
    if ~b0_s0 & b0_s1:
        q[2] = qmc.x(q[2])

    if b1_s0 & b1_s1:
        q[3] = qmc.x(q[3])
    if b1_s0 & ~b1_s1:
        q[4] = qmc.x(q[4])
    if ~b1_s0 & b1_s1:
        q[5] = qmc.x(q[5])

    if b2_s0 & b2_s1:
        q[6] = qmc.x(q[6])
    if b2_s0 & ~b2_s1:
        q[7] = qmc.x(q[7])
    if ~b2_s0 & b2_s1:
        q[8] = qmc.x(q[8])

    if phase_s0 & ~phase_s1:
        q[0] = qmc.z(q[0])
    if phase_s0 & phase_s1:
        q[3] = qmc.z(q[3])
    if ~phase_s0 & phase_s1:
        q[6] = qmc.z(q[6])

    q[0], q[1] = qmc.cx(q[0], q[1])
    q[0], q[2] = qmc.cx(q[0], q[2])
    q[3], q[4] = qmc.cx(q[3], q[4])
    q[3], q[5] = qmc.cx(q[3], q[5])
    q[6], q[7] = qmc.cx(q[6], q[7])
    q[6], q[8] = qmc.cx(q[6], q[8])

    q[0] = qmc.h(q[0])
    q[3] = qmc.h(q[3])
    q[6] = qmc.h(q[6])
    q[0], q[3] = qmc.cx(q[0], q[3])
    q[0], q[6] = qmc.cx(q[0], q[6])

    return qmc.measure(q)


# %% [markdown]
# We test one representative qubit from each block. If the logical $\lvert1\rangle$ is preserved, `q[0]` is always 1.

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
# Shor's code corrects any single-qubit Pauli error. Since small general noise can be expanded in the Pauli basis, correcting Pauli errors is the starting point for quantum error correction.

# %% [markdown]
# ## 5. Stabilizer View
#
# All three codes can be described with stabilizers. A stabilizer is a Pauli operator that leaves every valid code state unchanged. A $+1$ measurement means the state is still in the corresponding parity sector; a $-1$ measurement reveals part of the error syndrome.
#
# | Code | Stabilizer generators | Corrects |
# | --- | --- | --- |
# | 3-qubit bit-flip | $Z_0Z_1$, $Z_0Z_2$ | single $X$ |
# | 3-qubit phase-flip | $X_0X_1$, $X_0X_2$ | single $Z$ |
# | Shor 9-qubit | $Z_0Z_1$, $Z_0Z_2$, $Z_3Z_4$, $Z_3Z_5$, $Z_6Z_7$, $Z_6Z_8$, $X_0X_1X_2X_3X_4X_5$, $X_3X_4X_5X_6X_7X_8$ | single $X$, $Y$, or $Z$ |
#
# Larger codes such as the Steane code and the surface code use the same pattern: prepare one ancilla per stabilizer, measure the parity, then apply a correction based on the syndrome.

# %% [markdown]
# ## 6. Summary
#
# In this tutorial, we implemented:
#
# - The 3-qubit bit-flip code, using $Z$ parity checks to correct a single $X$ error.
# - The 3-qubit phase-flip code, using $X$ parity checks to correct a single $Z$ error.
# - Shor's 9-qubit code, which separately detects the $X$ and $Z$ components and corrects a single $X$, $Y$, or $Z$ error.
# - A stabilizer interpretation of the same circuits.
#
# A natural next step is [Steane [[7,1,3]] Code and CSS Construction](11_steane_code.ipynb), where $X$-type and $Z$-type stabilizers are organized more systematically.
