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
# ## Installation
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import qamomile as qm

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
# Before getting into the implementation, we load Qamomile and the Qiskit integration and define helper functions. `_bits7`, `_passes_hamming_checks`, and `_is_steane_zero_word` are utilities that decide whether a measurement outcome is a Hamming codeword or a $\lvert0_L\rangle$ codeword. They are not central to QEC, so feel free to skip them.

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


# %% [markdown]
# ### 3.3 Checking the Encoding
#
# We measure the encoder's output and check that every observed bitstring is an even-weight Hamming codeword (a $\lvert0_L\rangle$ codeword).


# %%
@qm.qkernel
def steane_encode_zero(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Encode the final seven components as the Steane logical zero.

    Args:
        state (qm.Q[qm.Bits[10]]): Three syndrome and seven data qubits.

    Returns:
        qm.Q[qm.Bits[10]]: State with components 3 through 9 encoded as logical zero.
    """

    state = qm.h(state, target=6)
    for target in (7, 8, 9):
        state = qm.cx(state, control=6, target=target)
    state = qm.h(state, target=4)
    for target in (5, 8, 9):
        state = qm.cx(state, control=4, target=target)
    state = qm.h(state, target=3)
    for target in (5, 7, 9):
        state = qm.cx(state, control=3, target=target)
    return state


@qm.qkernel
def steane_extract_bit_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane bit-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean syndrome qubits.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=data + 3, target=0)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=data + 3, target=1)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=data + 3, target=2)
    return state


@qm.qkernel
def steane_extract_phase_syndrome(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Extract one Steane phase-error syndrome coherently.

    Args:
        state (qm.Q[qm.Bits[10]]): Encoded data and three clean syndrome qubits.

    Returns:
        qm.Q[qm.Bits[10]]: Data with the syndrome in components 0 through 2.
    """

    state = qm.h(state, target=0)
    for data in (3, 4, 5, 6):
        state = qm.cx(state, control=0, target=data + 3)
    state = qm.h(state, target=0)
    state = qm.h(state, target=1)
    for data in (1, 2, 5, 6):
        state = qm.cx(state, control=1, target=data + 3)
    state = qm.h(state, target=1)
    state = qm.h(state, target=2)
    for data in (0, 2, 4, 6):
        state = qm.cx(state, control=2, target=data + 3)
    state = qm.h(state, target=2)
    return state


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
def _steane_syndrome_correction(
    target: int,
    *,
    pauli: int,
) -> qm.QKernel:
    """Construct one coherent three-bit syndrome correction.

    Args:
        target: Data component corrected by this syndrome value.
        pauli: Pauli selector: one X, two Y, three Z.

    Returns:
        Unary correction with its syndrome pattern closed in.
    """

    syndrome_value = target + 1
    zero_controls = tuple(
        offset for offset, mask in enumerate((4, 2, 1)) if syndrome_value & mask == 0
    )

    if pauli == 1:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.x(data, target=target)

    elif pauli == 2:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.y(data, target=target)

    else:

        @qm.qkernel
        def correct_data(
            data: qm.Q[qm.Bits[7]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[7]]:  # type: ignore[valid-type]
            return qm.z(data, target=target)

    if not zero_controls:

        @qm.qkernel
        def active_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            syndrome, data = qm.split_at(state, 3)
            syndrome, data = qm.control(correct_data, syndrome, data)
            return qm.concat(syndrome, data)

        return active_correction

    @qm.qkernel
    def correct(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        syndrome, data = qm.split_at(state, 3)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        syndrome, data = qm.control(correct_data, syndrome, data)
        for control in zero_controls:
            syndrome = qm.x(syndrome, target=control)
        return qm.concat(syndrome, data)

    return correct


def _steane_correction(pauli: int) -> qm.QKernel:
    """Compose all seven nonzero syndrome corrections.

    Args:
        pauli: Pauli selector: one X, two Y, three Z.

    Returns:
        Total coherent correction for one Pauli component.
    """

    @qm.qkernel
    def identity(
        state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
        return state

    current = identity
    for target in range(7):
        previous = current
        correction = _steane_syndrome_correction(target, pauli=pauli)

        @qm.qkernel
        def append_correction(
            state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
            return correction(previous(state))

        current = append_correction
    return current


_STEANE_X_CORRECTION = _steane_correction(1)
_STEANE_Y_CORRECTION = _steane_correction(2)
_STEANE_Z_CORRECTION = _steane_correction(3)

# %% [markdown]
# ## Compose one statically selected error experiment
#
# A $Y=iXZ$ error contains both bit and phase components. The representative
# case below extracts the bit syndrome and applies the matching coherent $Y$
# correction. The other static branches use the corresponding $X$ or $Z$
# extraction and the same generated correction structure.


# %%
@qm.qkernel
def steane_code_samples(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[10]]:  # type: ignore[valid-type]
    """Correct one statically selected Pauli error in the Steane code.

    Args:
        error_kind (qm.Const[qm.Int]): Pauli selector: one X, two Y, three Z.
        error_position (qm.Const[qm.Int]): Erroneous data-qubit index.

    Returns:
        qm.Shot[qm.Bits[10]]: Corrected codeword and retained syndrome.
    """

    state = steane_encode_zero(qm.qubits(10))
    if error_kind == 1:
        state = qm.x(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_X_CORRECTION(state)
    if error_kind == 2:
        state = qm.y(state, target=error_position + 3)
        state = steane_extract_bit_syndrome(state)
        state = _STEANE_Y_CORRECTION(state)
    if error_kind == 3:
        state = qm.z(state, target=error_position + 3)
        state = steane_extract_phase_syndrome(state)
        state = _STEANE_Z_CORRECTION(state)
    return qm.measure(state)


@qm.qkernel
def steane_transversal_h(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Apply the logical Hadamard transversally to the Steane data.

    Args:
        state (qm.Q[qm.Bits[10]]): Three syndrome and seven encoded data qubits.

    Returns:
        qm.Q[qm.Bits[10]]: State after one Hadamard on every data component.
    """

    for target in range(3, 10):
        state = qm.h(state, target=target)
    return state


@qm.qkernel
def steane_logical_plus_state() -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Prepare logical plus by a transversal Hadamard on logical zero.

    Returns:
        qm.Q[qm.Bits[10]]: Three clean syndrome qubits and logical plus data.
    """

    return steane_transversal_h(steane_encode_zero(qm.qubits(10)))


# %% [markdown]
# ## Inspect, estimate, compile, and execute
#
# We select a $Y$ error on data position three. This one case exercises both
# Pauli components while staying within four deterministic shots. Rust proves a
# peak width of ten, 21 `CX` operations, and seven coherent calls from the same
# specialized body. The logical Hadamard remains a separate seven-gate resource
# witness and starts no second simulator.

# %%
ERROR_KIND = 2
ERROR_POSITION = 3
SHOTS = 4

program = steane_code_samples.specialize(
    **{
        "error_kind": ERROR_KIND,
        "error_position": ERROR_POSITION,
    }
)
inspection = program.inspect()
study = qm.resources.logical(program)
transversal_study = qm.resources.logical(steane_transversal_h)

assert "quantum.steane" not in inspection
assert "quantum.error_correction" not in inspection
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (10, 10)
assert study.metric(
    qm.resources.logical_operation_count("quantum.cx")
).numeric_enclosure() == (21, 21)
assert study.metric(
    qm.resources.logical_operation_count("qcore.control_call")
).numeric_enclosure() == (7, 7)
assert transversal_study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
    7,
    7,
)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(artifact, on=endpoint, shots=SHOTS, seed=2)

logical_x = qm.X(3) * qm.X(4) * qm.X(5) * qm.X(6) * qm.X(7) * qm.X(8) * qm.X(9)
logical_plus_x = qm.expectation(
    steane_logical_plus_state,
    logical_x,
    on=endpoint,
    shots=4,
    seed=3,
)
assert logical_plus_x.value == 1.0

# %% [markdown]
# The retained syndrome is the big-endian bit pattern of `error_position + 1`.
# Every data outcome must remain an even-weight Hamming codeword, which is the
# computational support of the Steane logical zero.


# %%
def is_steane_zero_word(bits: tuple[int, ...]) -> bool:
    """Return whether seven bits form a Steane logical-zero codeword.

    Args:
        bits: Seven data bits in source order.

    Returns:
        True for an even-weight word satisfying all Hamming checks.
    """

    return (
        len(bits) == 7
        and (bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) == 0
        and (bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) == 0
        and (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) == 0
        and sum(bits) % 2 == 0
    )


syndrome_value = ERROR_POSITION + 1
expected_syndrome = tuple(int(bool(syndrome_value & mask)) for mask in (4, 2, 1))

print("corrected counts:", record.result.counts)
print("retained syndrome:", expected_syndrome)

assert artifact.parameter_abi.slots == ()
assert record.artifact is artifact
assert record.result.shots == SHOTS
assert all(bits[:3] == expected_syndrome for bits in record.result.counts)
assert all(is_steane_zero_word(bits[3:10]) for bits in record.result.counts)

# %% [markdown]
# ### 4.4 Verification: All 21 Single Errors
#
# We try all 21 single errors — $X$, $Y$, $Z$ on each of the seven qubits. After correction, every measured bitstring should return to a $\lvert0_L\rangle$ codeword.


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


# %% [markdown]
# The Hamming codeword ratio is 1.000, and odd-weight codewords appear about half the time. Unlike $\lvert0_L\rangle$, which produces only even-weight codewords, this confirms the state has moved to $\lvert+_L\rangle$.
#
# **Property 2: applying $\bar H$ twice returns to the identity ($\bar H^2 = I$).**
# Applying $H^{\otimes7}$ twice to $\lvert0_L\rangle$ should return to $\lvert0_L\rangle$.


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
