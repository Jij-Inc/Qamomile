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
# ---
# title: Steane [[7,1,3]] Code
# tags: [qec, tutorial]
# ---
#
# # Quantum Error Correction (2): Steane [[7,1,3]] Code
#
# <!-- BEGIN auto-tags -->
# **Tags:** [`qec`](../tags/qec.md) · [`tutorial`](../tags/tutorial.md)
# <!-- END auto-tags -->
#
# In the [previous tutorial](10_quantum_error_correction.ipynb), we implemented the 3-qubit repetition codes and Shor's 9-qubit code. Here we move to the **Steane [[7,1,3]] code**, a cleaner and more structured code.
#
# The Steane code is a CSS code built from the classical Hamming [7,4,3] code. It protects one logical qubit with seven physical qubits and corrects any single-qubit Pauli error: $X$, $Y$, or $Z$.
#
# We will implement three things:
#
# 1. Build Steane stabilizers from the Hamming code structure.
# 2. Measure six stabilizers and correct a single error from the syndrome.
# 3. Verify that seven physical Hadamard gates implement the logical Hadamard $\bar{H}$.

# %%
# Install the latest Qamomile from pip.
# # !pip install qamomile
# # or
# # !uv add qamomile

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# Create a seeded backend for reproducible documentation output
from qiskit_aer import AerSimulator

_seeded_backend = AerSimulator(seed_simulator=42, max_parallel_threads=1)
_seeded_executor = transpiler.executor(backend=_seeded_backend)


def _bits7(outcome) -> list[int]:
    """Return seven measured bits in qubit-index order."""
    if isinstance(outcome, (list, tuple)):
        return list(outcome)
    return [(outcome >> i) & 1 for i in range(7)]


def _is_steane_zero_word(outcome) -> bool:
    """Return True when the outcome is an even Hamming codeword."""
    bits = _bits7(outcome)
    h_checks = [
        bits[3] ^ bits[4] ^ bits[5] ^ bits[6],
        bits[1] ^ bits[2] ^ bits[5] ^ bits[6],
        bits[0] ^ bits[2] ^ bits[4] ^ bits[6],
    ]
    return all(check == 0 for check in h_checks) and sum(bits) % 2 == 0


# %% [markdown]
# ## 1. From Hamming Codes to CSS Codes
#
# Use the following parity-check matrix for the classical Hamming [7,4,3] code:
#
# $$
# H =
# \begin{pmatrix}
# 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
# 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
# 1 & 0 & 1 & 0 & 1 & 0 & 1
# \end{pmatrix}.
# $$
#
# Column $j$ is the binary representation of $j+1$. Therefore, a 3-bit syndrome identifies the error location directly.
#
# A CSS code creates two kinds of stabilizers from this matrix:
#
# - $Z$-type stabilizers detect $X$ errors.
# - $X$-type stabilizers detect $Z$ errors.
#
# The Steane code has six generators:
#
# | Type | Stabilizer | Detects |
# | --- | --- | --- |
# | $X$ type | $X_3X_4X_5X_6$ | $Z$ |
# | $X$ type | $X_1X_2X_5X_6$ | $Z$ |
# | $X$ type | $X_0X_2X_4X_6$ | $Z$ |
# | $Z$ type | $Z_3Z_4Z_5Z_6$ | $X$ |
# | $Z$ type | $Z_1Z_2Z_5Z_6$ | $X$ |
# | $Z$ type | $Z_0Z_2Z_4Z_6$ | $X$ |

# %% [markdown]
# ## 2. Encoding $\lvert0_L\rangle$
#
# The Steane logical $\lvert0_L\rangle$ is the superposition of the eight even-weight Hamming codewords:
#
# $$
# \lvert0_L\rangle =
# \frac{1}{2\sqrt{2}}
# \sum_{c \in C,\; w(c)\ {\rm even}} \lvert c\rangle.
# $$
#
# The following circuit prepares $\lvert0_L\rangle$ from $\lvert0\rangle^{\otimes 7}$ by building the three $X$-type stabilizer patterns.


# %%
@qmc.qkernel
def encode_steane_zero(data: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    data[3] = qmc.h(data[3])
    data[3], data[4] = qmc.cx(data[3], data[4])
    data[3], data[5] = qmc.cx(data[3], data[5])
    data[3], data[6] = qmc.cx(data[3], data[6])

    data[1] = qmc.h(data[1])
    data[1], data[2] = qmc.cx(data[1], data[2])
    data[1], data[5] = qmc.cx(data[1], data[5])
    data[1], data[6] = qmc.cx(data[1], data[6])

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
# Measure the encoder output and check that all observed bitstrings are even Hamming codewords.

# %%
print("Encode and measure |0_L>")
exe = transpiler.transpile(encode_zero_and_measure)
result = exe.sample(_seeded_executor, shots=1024).result()
valid = sum(count for outcome, count in result.results if _is_steane_zero_word(outcome))
total = sum(count for _, count in result.results)
print(f"  even Hamming codeword ratio: {valid / total:.3f}")
print(f"  observed codewords: {len(result.results)}")

# %% [markdown]
# ## 3. Syndrome Measurement and Correction
#
# The Steane code decodes $X$ and $Z$ errors independently.
#
# - Measuring the $Z$-type stabilizers gives the syndrome for the $X$ component.
# - Measuring the $X$-type stabilizers gives the syndrome for the $Z$ component.
#
# The syndrome table is the Hamming matrix column table:
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
# In the implementation, pass `error_type` as `1=X`, `2=Y`, or `3=Z`. The `error_pos` parameter is the physical qubit index `0..6`.


# %%
@qmc.qkernel
def steane_run(
    error_type: qmc.UInt,
    error_pos: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    anc = qmc.qubit_array(6, name="anc")

    data = encode_steane_zero(data)

    for i in qmc.range(7):
        if (error_type == 1) & (error_pos == i):  # 1 means X error.
            data[i] = qmc.x(data[i])
        if (error_type == 2) & (error_pos == i):  # 2 means Y error.
            data[i] = qmc.y(data[i])
        if (error_type == 3) & (error_pos == i):  # 3 means Z error.
            data[i] = qmc.z(data[i])

    # Z-type stabilizers: detect the X component.
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

    # X-type stabilizers: detect the Z component.
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
# Run all 21 single errors: $X$, $Y$, and $Z$ on each of the seven physical qubits. After correction, every measured bitstring should still be a $\lvert0_L\rangle$ codeword.

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
        valid = sum(
            count for outcome, count in result.results if _is_steane_zero_word(outcome)
        )
        total = sum(count for _, count in result.results)
        print(f"  {name:4s} | q[{pos}]  | {valid / total:.3f}")

# %% [markdown]
# A ratio of 1.000 means the state returns to the $\lvert0_L\rangle$ code space for every single-qubit Pauli error. A $Y=iXZ$ error triggers both the $X$-component and $Z$-component corrections.

# %% [markdown]
# ## 4. Transversal Hadamard
#
# A key feature of the Steane code is that the logical Hadamard $\bar{H}$ is implemented by applying physical Hadamard gates to all seven qubits:
#
# $$
# \bar{H} = H^{\otimes 7}.
# $$
#
# This works because the $X$-type and $Z$-type stabilizers have the same Hamming pattern. Transversal gates are important in fault-tolerant computation because one physical gate failure does not spread across many qubits.


# %%
@qmc.qkernel
def transversal_hadamard_to_plus_l() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    return qmc.measure(data)


@qmc.qkernel
def transversal_hadamard_round_trip() -> qmc.Vector[qmc.Bit]:
    data = qmc.qubit_array(7, name="data")
    data = encode_steane_zero(data)

    for i in qmc.range(7):
        data[i] = qmc.h(data[i])
    for i in qmc.range(7):
        data[i] = qmc.h(data[i])

    return qmc.measure(data)


def _logical_x_parity(outcome) -> int:
    """Return the measured parity for logical X = X0 X1 X2."""
    bits = _bits7(outcome)
    return (bits[0] + bits[1] + bits[2]) % 2


# %% [markdown]
# First check that $\bar{H}\lvert0_L\rangle=\lvert+_L\rangle$. The state $\lvert+_L\rangle$ is a +1 eigenstate of logical $X$, so measuring in the $X$ basis should give $q_0 \oplus q_1 \oplus q_2 = 0$.

# %%
print("Transversal H: |0_L> -> |+_L>")
exe_plus = transpiler.transpile(transversal_hadamard_to_plus_l)
result_plus = exe_plus.sample(_seeded_executor, shots=1024).result()
parity_zero = sum(
    count for outcome, count in result_plus.results if _logical_x_parity(outcome) == 0
)
total_plus = sum(count for _, count in result_plus.results)
print(f"  q[0] xor q[1] xor q[2] = 0 ratio: {parity_zero / total_plus:.3f}")

# %% [markdown]
# Next check the round trip. Applying transversal $H$ twice should return to $\lvert0_L\rangle$ because $\bar{H}^2=I$.

# %%
print("Transversal H round trip: |0_L> -> H -> H -> |0_L>")
exe_round_trip = transpiler.transpile(transversal_hadamard_round_trip)
result_round_trip = exe_round_trip.sample(_seeded_executor, shots=1024).result()
valid = sum(
    count
    for outcome, count in result_round_trip.results
    if _is_steane_zero_word(outcome)
)
total = sum(count for _, count in result_round_trip.results)
print(f"  |0_L> codeword ratio: {valid / total:.3f}")

# %% [markdown]
# ## 5. Summary
#
# In this tutorial, we implemented the Steane [[7,1,3]] code.
#
# - Built three $X$-type and three $Z$-type stabilizers from the Hamming [7,4,3] code.
# - Used $Z$-type stabilizers to detect the $X$ component and $X$-type stabilizers to detect the $Z$ component.
# - Verified correction for all 21 single Pauli errors: $X/Y/Z \times 7$ locations.
# - Verified that seven physical Hadamard gates act as the logical Hadamard $\bar{H}$.
#
# Compared with Shor's code, the Steane code uses fewer physical qubits for the same distance $d=3$ and gives a clean example of CSS structure and transversal Clifford gates.
#
# ### Next
#
# - [Quantum Error Correction (1)](10_quantum_error_correction.ipynb) — 3-qubit bit-flip / phase-flip / Shor codes
# - Surface codes — local stabilizers on a 2D lattice and repeated syndrome measurement
