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
# # Stabilizer Formalism and the Steane Code
#
# In [Introduction to Quantum Error Correction](quantum_error_correction.ipynb) (the first part), we built the 3-qubit bit-flip and phase-flip codes and Shor's 9-qubit code, and at the end gave the name "stabilizer" to the parity operators.
#
# This second part treats stabilizers formally. We then implement the **CSS construction**, which systematically builds quantum codes from classical codes, and its flagship example, the **Steane code**. The Steane code achieves the same $d=3$ as Shor's code with seven qubits instead of nine.
#
# What this article covers:
#
# 1. **Stabilizer formalism** — formalize stabilizers, generators, and the syndrome.
# 2. **The CSS construction** — build the Steane code's stabilizers from a classical Hamming code.
# 3. Implement encoding, syndrome measurement, and correction for the Steane code.
# 4. Confirm that seven physical Hadamard gates act as a logical Hadamard.
#
# **Prerequisites**: the content of [the first part](quantum_error_correction.ipynb) (syndrome measurement, Pauli errors, the term "stabilizer").

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile
# # or
# # !uv add qamomile

# %% [markdown]
# ## 1. Stabilizer Formalism
#
# In the first part, we called parity operators such as $Z_0Z_1$ and $X_0X_1$ **stabilizers**. We first formalize this term.

# %% [markdown]
# ### 1.1 What Is a Stabilizer
#
# A **stabilizer** is a Pauli operator $S$ that leaves every state of the code space unchanged. For any codeword $\lvert\psi\rangle$,
#
# $$S\lvert\psi\rangle = \lvert\psi\rangle$$
#
# holds. Equivalently, the code space is the $+1$ eigenspace of the stabilizer.
#
# For example, the codewords of the bit-flip code are $\lvert000\rangle$ and $\lvert111\rangle$. Applying $Z_0Z_1$ gives $Z_0Z_1\lvert000\rangle=\lvert000\rangle$ and $Z_0Z_1\lvert111\rangle=(-1)(-1)\lvert111\rangle=\lvert111\rangle$ — neither changes. So $Z_0Z_1$ is a stabilizer of the bit-flip code.

# %% [markdown]
# ### 1.2 Generators and the Syndrome
#
# A code has many stabilizers, but we do not need to list them all. Choosing a few **generators** is enough; the rest are obtained as their products. For the bit-flip code, $Z_0Z_1$ and $Z_0Z_2$ are the two generators.
#
# Measuring a stabilizer generator yields the eigenvalue $+1$ or $-1$.
#
# - With no error, the state is in the code space and every generator returns $+1$.
# - When an error $E$ occurs, the measured value of any generator that **anticommutes** with $E$ flips to $-1$.
#
# This pattern of $\pm1$ (written as $0/1$ in bits) is the **syndrome**. The values we extracted into ancilla qubits in the first part were exactly the measurement results of stabilizer generators. Seeing which generators turned $-1$ tells us the location and type of the error.

# %% [markdown]
# ### 1.3 The First Part's Codes as Stabilizers
#
# The three codes from the first part can all be described with stabilizer generators.
#
# | Code | Stabilizer generators |
# | --- | --- |
# | 3-qubit bit-flip | $Z_0Z_1,\ Z_0Z_2$ |
# | 3-qubit phase-flip | $X_0X_1,\ X_0X_2$ |
# | Shor 9-qubit | $Z_0Z_1,\ Z_0Z_2,\ Z_3Z_4,\ Z_3Z_5,\ Z_6Z_7,\ Z_6Z_8,\ X_0X_1X_2X_3X_4X_5,\ X_3X_4X_5X_6X_7X_8$ |
#
# The bit-flip code has only $Z$-type generators, and the phase-flip code only $X$-type generators. Shor's code has both, with the $Z$-type detecting $X$ errors and the $X$-type detecting $Z$ errors. This structure of "holding $X$-type and $Z$-type generators separately" leads to the CSS construction in the next section.

# %% [markdown]
# ### 1.4 $[[n,k,d]]$ and the Code Distance
#
# A stabilizer code is characterized by three numbers, $[[n,k,d]]$.
#
# - $n$: the number of physical qubits.
# - $k$: the number of logical qubits protected. Each generator removes one degree of freedom, so $k = n - (\text{number of generators})$.
# - $d$: the **code distance**. It is the minimum weight (the number of qubits acted on) among the Pauli operators that commute with every stabilizer but are not themselves stabilizers — these are called **logical operators**.
#
# A code of distance $d$ can correct up to $\lfloor(d-1)/2\rfloor$ errors. Correcting any single-qubit error requires $d\ge3$. The 3-qubit codes from the first part had $d=1$ and Shor's code had $d=3$. The Steane code we are about to build is $[[7,1,3]]$.

# %% [markdown]
# ## 2. From Classical Hamming Codes to CSS Codes
#
# A **CSS code (Calderbank-Shor-Steane code)** is a way to build quantum codes systematically from classical error-correcting codes. The Steane code is its flagship example, based on the classical Hamming code.

# %% [markdown]
# ### 2.1 The Classical Hamming [7,4,3] Code
#
# The classical Hamming [7,4,3] code protects 4 bits of information with 7 bits and can correct a single-bit error. It is defined by a **parity-check matrix** $H$.
#
# $$
# H =
# \begin{pmatrix}
# 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
# 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
# 1 & 0 & 1 & 0 & 1 & 0 & 1
# \end{pmatrix}
# $$
#
# A 7-bit word $c$ is a codeword if and only if $Hc=0$ (its parity with every row is even).

# %% [markdown]
# ### 2.2 How the Columns Point to the Error Location
#
# Look closely at the columns of $H$: column $j$ is the binary representation of the number $j+1$ (column 0 is $001$, column 1 is $010$, …, column 6 is $111$).
#
# When a single-bit error occurs at position $j$, $Hc$ equals exactly column $j$. So reading the three bits of $Hc$ as a binary number gives the error location directly. This is the mechanism by which the classical Hamming code pinpoints the error location in one shot.

# %% [markdown]
# ### 2.3 The CSS Construction: $X$-Type and $Z$-Type Stabilizers
#
# The CSS construction builds two kinds of stabilizers from this parity-check matrix $H$.
#
# - **$Z$-type stabilizers**: read each row of $H$ as an operator placing $Z$ where the $1$s are. Detects $X$ errors.
# - **$X$-type stabilizers**: read the same rows as operators placing $X$. Detects $Z$ errors.
#
# An $X$ error anticommutes with the $Z$-type stabilizers, and a $Z$ error anticommutes with the $X$-type stabilizers. So the $Z$-type locate the $X$ errors and the $X$-type locate the $Z$ errors, each in exactly the same way as the classical Hamming code. The "correct $X$ and $Z$ independently" seen in Shor's code in the first part comes out automatically from the classical code here.

# %% [markdown]
# ### 2.4 The Six Generators of the Steane Code
#
# From the three rows of the Hamming matrix $H$ we obtain three $Z$-type and three $X$-type generators — six stabilizer generators in total.
#
# | Type | Stabilizer | Detects |
# | --- | --- | --- |
# | $X$ type | $X_3X_4X_5X_6$ | $Z$ |
# | $X$ type | $X_1X_2X_5X_6$ | $Z$ |
# | $X$ type | $X_0X_2X_4X_6$ | $Z$ |
# | $Z$ type | $Z_3Z_4Z_5Z_6$ | $X$ |
# | $Z$ type | $Z_1Z_2Z_5Z_6$ | $X$ |
# | $Z$ type | $Z_0Z_2Z_4Z_6$ | $X$ |
#
# With 7 physical qubits and 6 generators, the number of logical qubits protected is $7-6=1$. This is the Steane $[[7,1,3]]$ code.

# %% [markdown]
# Before getting into the implementation, we load Qamomile and the Qiskit backend and define helper functions. `_bits7`, `_passes_hamming_checks`, and `_is_steane_zero_word` are utilities that decide whether a measurement outcome is a Hamming codeword or a $\lvert0_L\rangle$ codeword. They are not central to QEC, so feel free to skip them.

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create a seeded backend for reproducible documentation output.
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _bits7(outcome) -> list[int]:
    """Return the outcome as a list of 7 bits in qubit-index order."""
    if isinstance(outcome, (list, tuple)):
        return list(outcome)
    return [(outcome >> i) & 1 for i in range(7)]


def _passes_hamming_checks(outcome) -> bool:
    """Return whether the outcome is a Hamming [7,4,3] codeword (passes all 3 parity checks)."""
    bits = _bits7(outcome)
    h_checks = [
        bits[3] ^ bits[4] ^ bits[5] ^ bits[6],
        bits[1] ^ bits[2] ^ bits[5] ^ bits[6],
        bits[0] ^ bits[2] ^ bits[4] ^ bits[6],
    ]
    return all(check == 0 for check in h_checks)


def _is_steane_zero_word(outcome) -> bool:
    """Return whether the outcome is a |0_L> codeword (an even-weight Hamming codeword)."""
    return _passes_hamming_checks(outcome) and sum(_bits7(outcome)) % 2 == 0


# %% [markdown]
# ## 3. Encoding the Logical $\lvert0_L\rangle$
#
# ### 3.1 The Logical Zero Is a Superposition of Even-Weight Hamming Codewords
#
# The logical $\lvert0_L\rangle$ of the Steane code is the superposition of all even-weight Hamming codewords.
#
# $$\lvert0_L\rangle = \frac{1}{2\sqrt2}\sum_{c\in C,\ w(c)\,\text{even}} \lvert c\rangle$$
#
# Here $C$ is the set of Hamming codewords and $w(c)$ is the weight. There are 8 even-weight Hamming codewords, so this is an 8-term superposition.

# %% [markdown]
# ### 3.2 The Encoding Circuit
#
# The following circuit builds $\lvert0_L\rangle$ from $\lvert0\rangle^{\otimes7}$. It writes in the three $X$-type stabilizer patterns ($X_3X_4X_5X_6$, $X_1X_2X_5X_6$, $X_0X_2X_4X_6$) one by one with Hadamard and CNOT gates.


# %%
@qmc.qkernel
def encode_steane_zero(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    # Write in the X3 X4 X5 X6 pattern.
    data[3] = qmc.h(data[3])
    data[3], data[4] = qmc.cx(data[3], data[4])
    data[3], data[5] = qmc.cx(data[3], data[5])
    data[3], data[6] = qmc.cx(data[3], data[6])

    # Write in the X1 X2 X5 X6 pattern.
    data[1] = qmc.h(data[1])
    data[1], data[2] = qmc.cx(data[1], data[2])
    data[1], data[5] = qmc.cx(data[1], data[5])
    data[1], data[6] = qmc.cx(data[1], data[6])

    # Write in the X0 X2 X4 X6 pattern.
    data[0] = qmc.h(data[0])
    data[0], data[2] = qmc.cx(data[0], data[2])
    data[0], data[4] = qmc.cx(data[0], data[4])
    data[0], data[6] = qmc.cx(data[0], data[6])
    return data


# %%
@qmc.qkernel
def encode_zero_and_measure() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)
    return qmc.measure(data)


# %% [markdown]
# ### 3.3 Checking the Encoding
#
# We measure the encoder's output and check that every observed bitstring is an even-weight Hamming codeword (a $\lvert0_L\rangle$ codeword).

# %%
print("Encode and measure |0_L>")
exe = transpiler.transpile(encode_zero_and_measure)
result = exe.sample(_seeded_executor, shots=1024).result()
total = sum(count for _, count in result.results)
valid = sum(count for outcome, count in result.results if _is_steane_zero_word(outcome))
print(f"  |0_L> codeword ratio: {valid / total:.3f}")
print(f"  distinct codewords observed: {len(result.results)}")
# The Steane |0_L> is an equal superposition of exactly 8 even-weight
# Hamming codewords, so every shot reads a valid |0_L> codeword and the
# distribution covers at most those 8 outcomes.
assert total == 1024
assert valid == total
assert len(result.results) <= 8

# %% [markdown]
# ## 4. Syndrome Measurement and Correction
#
# ### 4.1 Decoding $X$ and $Z$ Independently
#
# The advantage of a CSS code is that $X$ errors and $Z$ errors can be handled completely independently.
#
# - Measuring the $Z$-type stabilizers gives the syndrome for $X$ errors.
# - Measuring the $X$-type stabilizers gives the syndrome for $Z$ errors.
#
# Each is a 3-bit syndrome, and the column mechanism of the Hamming matrix seen in 2.2 applies directly.

# %% [markdown]
# ### 4.2 The Syndrome Table
#
# The 3-bit syndrome $(s_2,s_1,s_0)$ represents the error location directly in binary.
#
# | Error location | Syndrome $(s_2,s_1,s_0)$ |
# | --- | --- |
# | none | $(0,0,0)$ |
# | $q_0$ | $(0,0,1)$ |
# | $q_1$ | $(0,1,0)$ |
# | $q_2$ | $(0,1,1)$ |
# | $q_3$ | $(1,0,0)$ |
# | $q_4$ | $(1,0,1)$ |
# | $q_5$ | $(1,1,0)$ |
# | $q_6$ | $(1,1,1)$ |
#
# The $(s_2,s_1,s_0)$ obtained from the $Z$-type stabilizers points to the $X$ error location, and the one from the $X$-type points to the $Z$ error location.

# %% [markdown]
# ### 4.3 Implementation
#
# `error_type` is `1=X`, `2=Y`, `3=Z`, and `error_pos` is the qubit index `0..6` to inject the error into.


# %%
@qmc.qkernel
def steane_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    # Allocate 7 data qubits and 6 ancilla qubits for syndrome measurement.
    data = qmc.qubit_array(7, name="data")
    anc = qmc.qubit_array(6, name="anc")

    # Encode into |0_L>.
    data = encode_steane_zero(data)

    # Inject the X / Y / Z error specified by error_type / error_pos.
    for i in qmc.range(7):
        if (error_type == 1) & (error_pos == i):  # 1: X error
            data[i] = qmc.x(data[i])
        if (error_type == 2) & (error_pos == i):  # 2: Y error
            data[i] = qmc.y(data[i])
        if (error_type == 3) & (error_pos == i):  # 3: Z error
            data[i] = qmc.z(data[i])

    # Z-type stabilizers: measure the X-error syndrome (sx_2, sx_1, sx_0).
    data[3], anc[0] = qmc.cx(data[3], anc[0])
    data[4], anc[0] = qmc.cx(data[4], anc[0])
    data[5], anc[0] = qmc.cx(data[5], anc[0])
    data[6], anc[0] = qmc.cx(data[6], anc[0])
    sx_2 = qmc.measure(anc[0])

    data[1], anc[1] = qmc.cx(data[1], anc[1])
    data[2], anc[1] = qmc.cx(data[2], anc[1])
    data[5], anc[1] = qmc.cx(data[5], anc[1])
    data[6], anc[1] = qmc.cx(data[6], anc[1])
    sx_1 = qmc.measure(anc[1])

    data[0], anc[2] = qmc.cx(data[0], anc[2])
    data[2], anc[2] = qmc.cx(data[2], anc[2])
    data[4], anc[2] = qmc.cx(data[4], anc[2])
    data[6], anc[2] = qmc.cx(data[6], anc[2])
    sx_0 = qmc.measure(anc[2])

    # X-type stabilizers: measure the Z-error syndrome (sz_2, sz_1, sz_0).
    anc[3] = qmc.h(anc[3])
    anc[3], data[3] = qmc.cx(anc[3], data[3])
    anc[3], data[4] = qmc.cx(anc[3], data[4])
    anc[3], data[5] = qmc.cx(anc[3], data[5])
    anc[3], data[6] = qmc.cx(anc[3], data[6])
    anc[3] = qmc.h(anc[3])
    sz_2 = qmc.measure(anc[3])

    anc[4] = qmc.h(anc[4])
    anc[4], data[1] = qmc.cx(anc[4], data[1])
    anc[4], data[2] = qmc.cx(anc[4], data[2])
    anc[4], data[5] = qmc.cx(anc[4], data[5])
    anc[4], data[6] = qmc.cx(anc[4], data[6])
    anc[4] = qmc.h(anc[4])
    sz_1 = qmc.measure(anc[4])

    anc[5] = qmc.h(anc[5])
    anc[5], data[0] = qmc.cx(anc[5], data[0])
    anc[5], data[2] = qmc.cx(anc[5], data[2])
    anc[5], data[4] = qmc.cx(anc[5], data[4])
    anc[5], data[6] = qmc.cx(anc[5], data[6])
    anc[5] = qmc.h(anc[5])
    sz_0 = qmc.measure(anc[5])

    # X-component correction: apply X at the location pointed to by (sx_2, sx_1, sx_0).
    if (~sx_2) & (~sx_1) & sx_0:
        data[0] = qmc.x(data[0])
    if (~sx_2) & sx_1 & (~sx_0):
        data[1] = qmc.x(data[1])
    if (~sx_2) & sx_1 & sx_0:
        data[2] = qmc.x(data[2])
    if sx_2 & (~sx_1) & (~sx_0):
        data[3] = qmc.x(data[3])
    if sx_2 & (~sx_1) & sx_0:
        data[4] = qmc.x(data[4])
    if sx_2 & sx_1 & (~sx_0):
        data[5] = qmc.x(data[5])
    if sx_2 & sx_1 & sx_0:
        data[6] = qmc.x(data[6])

    # Z-component correction: apply Z at the location pointed to by (sz_2, sz_1, sz_0).
    if (~sz_2) & (~sz_1) & sz_0:
        data[0] = qmc.z(data[0])
    if (~sz_2) & sz_1 & (~sz_0):
        data[1] = qmc.z(data[1])
    if (~sz_2) & sz_1 & sz_0:
        data[2] = qmc.z(data[2])
    if sz_2 & (~sz_1) & (~sz_0):
        data[3] = qmc.z(data[3])
    if sz_2 & (~sz_1) & sz_0:
        data[4] = qmc.z(data[4])
    if sz_2 & sz_1 & (~sz_0):
        data[5] = qmc.z(data[5])
    if sz_2 & sz_1 & sz_0:
        data[6] = qmc.z(data[6])

    return qmc.measure(data)


# %% [markdown]
# ### 4.4 Verification: All 21 Single Errors
#
# We try all 21 single errors — $X$, $Y$, $Z$ on each of the seven qubits. After correction, every measured bitstring should return to a $\lvert0_L\rangle$ codeword.

# %%
print("Steane code: correct X/Y/Z on all 7 locations")
print(f"  {'err':4s} | {'pos':5s} | |0_L> codeword")
print(f"  {'-' * 4}-+-{'-' * 5}-+-{'-' * 14}")
for name, error_type in [("X", 1), ("Y", 2), ("Z", 3)]:
    for pos in range(7):
        exe = transpiler.transpile(
            steane_run,
            bindings={"error_type": error_type, "error_pos": pos},
        )
        result = exe.sample(_seeded_executor, shots=128).result()
        total = sum(count for _, count in result.results)
        valid = sum(
            count for outcome, count in result.results if _is_steane_zero_word(outcome)
        )
        print(f"  {name:4s} | q[{pos}]  | {valid / total:.3f}")
        # Steane corrects every single Pauli error on any of the 7 qubits,
        # so the post-correction state lies entirely in the |0_L> code
        # space — every shot reads a valid codeword.
        assert total == 128
        assert valid == total

# %% [markdown]
# A ratio of 1.000 means the state returned to the $\lvert0_L\rangle$ code space for that single Pauli error. A $Y=iXZ$ error triggers both the $X$-component and the $Z$-component correction, but in a CSS code these two are independent, so it is corrected as is.

# %% [markdown]
# ## 5. The Transversal Hadamard Gate
#
# ### 5.1 What Is a Transversal Gate
#
# When applying a logical gate to a logical qubit, if it suffices to apply a physical gate **independently** to each physical qubit, that gate is called **transversal**.
#
# Transversal gates are important in fault-tolerant quantum computation. Because the physical gates are independent of one another, a failure in one physical gate does not spread the error across multiple qubits within a block.
#
# A major feature of the Steane code is that the logical Hadamard $\bar H$ is transversal. Applying $H$ to each of the seven physical qubits is itself the logical Hadamard.
#
# $$\bar H = H^{\otimes 7}$$
#
# This follows from the $X$-type and $Z$-type stabilizers of the Steane code having the same Hamming pattern — from the CSS construction using the same matrix $H$.

# %% [markdown]
# ### 5.2 Verifying the Logical Hadamard
#
# We verify $\bar H = H^{\otimes7}$ through two properties.
#
# **Property 1: $\bar H$ maps $\lvert0_L\rangle$ to $\lvert+_L\rangle$.**
# $\lvert+_L\rangle$ is a superposition of $\lvert0_L\rangle$ and $\lvert1_L\rangle$. While $\lvert0_L\rangle$ consists of even-weight Hamming codewords, $\lvert1_L\rangle$ consists of odd-weight Hamming codewords. So measuring $\lvert+_L\rangle$ produces Hamming codewords of both even and odd weight (whereas $\lvert0_L\rangle$ produces only even-weight ones).


# %%
@qmc.qkernel
def transversal_h_to_plus() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)
    # Apply the transversal Hadamard once: |0_L> -> |+_L>.
    data = qmc.h(data)
    return qmc.measure(data)


# %%
print("Transversal H: |0_L> -> |+_L>")
exe = transpiler.transpile(transversal_h_to_plus)
result = exe.sample(_seeded_executor, shots=1024).result()
total = sum(count for _, count in result.results)
hamming = sum(
    count for outcome, count in result.results if _passes_hamming_checks(outcome)
)
odd = sum(count for outcome, count in result.results if sum(_bits7(outcome)) % 2 == 1)
print(f"  Hamming codeword ratio: {hamming / total:.3f}")
print(f"  odd-weight (|1_L>) fraction: {odd / total:.3f}")
# |+_L> is an equal superposition of all 16 Hamming codewords, so every
# shot reads some Hamming codeword and the odd-weight half (|1_L>) shows
# up roughly half the time — 5% window > 1024-shot standard error of < 0.02.
assert total == 1024
assert hamming == total
assert len(result.results) <= 16
assert abs(odd / total - 0.5) < 0.05

# %% [markdown]
# The Hamming codeword ratio is 1.000, and odd-weight codewords appear about half the time. Unlike $\lvert0_L\rangle$, which produces only even-weight codewords, this confirms the state has moved to $\lvert+_L\rangle$.
#
# **Property 2: applying $\bar H$ twice returns to the identity ($\bar H^2 = I$).**
# Applying $H^{\otimes7}$ twice to $\lvert0_L\rangle$ should return to $\lvert0_L\rangle$.


# %%
@qmc.qkernel
def transversal_h_round_trip() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)
    # Apply the transversal Hadamard twice: H^2 = I, so it returns to |0_L>.
    data = qmc.h(data)
    data = qmc.h(data)
    return qmc.measure(data)


# %%
print("Transversal H round trip: |0_L> -> H -> H -> |0_L>")
exe = transpiler.transpile(transversal_h_round_trip)
result = exe.sample(_seeded_executor, shots=1024).result()
total = sum(count for _, count in result.results)
valid = sum(count for outcome, count in result.results if _is_steane_zero_word(outcome))
print(f"  |0_L> codeword ratio: {valid / total:.3f}")
# H^2 = I, so the round trip restores |0_L> exactly — every shot reads a
# |0_L> codeword.
assert total == 1024
assert valid == total

# %% [markdown]
# The ratio is 1.000 — two transversal Hadamards return to $\lvert0_L\rangle$. The seven physical $H$ gates indeed act as the logical $\bar H$.

# %% [markdown]
# ## 6. Summary
#
# In this article we covered the stabilizer formalism and the Steane $[[7,1,3]]$ code.
#
# - **Stabilizer formalism** — we viewed the code space as the $+1$ eigenspace of Pauli operators (stabilizers) and defined the syndrome as the measurement values of the generators.
# - **The CSS construction** — we built $X$-type and $Z$-type stabilizers from the parity-check matrix of the classical Hamming [7,4,3] code.
# - **The Steane code** — we implemented the encoding of $\lvert0_L\rangle$, syndrome measurement with the six stabilizers, and correction of all 21 single Pauli errors.
# - **The transversal Hadamard** — we confirmed that seven physical $H$ gates act as the logical $\bar H$.
#
# The Steane code achieves the same $d=3$ as Shor's code with seven qubits instead of nine, and is a clean example of the CSS construction and transversal Clifford gates.
#
# ### Beyond
#
# - **The surface code** — a code whose stabilizers are local operators on a 2D lattice. It is the leading approach to error correction on today's superconducting quantum computers, with repeated syndrome measurement as a new ingredient.
# - **Fault-tolerant quantum computation** — a framework for running logical gates while staying encoded and advancing the computation without amplifying errors. Transversal gates are its starting point.
