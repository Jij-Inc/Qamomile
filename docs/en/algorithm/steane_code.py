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
# Express the syndromes from [Introduction to Quantum Error Correction](quantum_error_correction.ipynb) through stabilizers, and implement Steane encoding, correction, and logical Hadamard with Qamomile. The recovery circuit receives neither the error kind nor its position: it uses both bit-flip and phase-flip syndromes to correct a single-qubit error. We use seven data qubits and three ancillary qubits that are measured, reset, and reused, for one correction cycle with ideal gates.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## Stabilizers and the code space
#
# Stabilizer generators are mutually commuting Pauli operators satisfying $S|\psi_L\rangle=|\psi_L\rangle$ for states $|\psi_L\rangle$ in the code space. If an error $E$ anticommutes with $S$, then $SE|\psi_L\rangle=-E|\psi_L\rangle$: the check's eigenvalue flips. We read this change as a syndrome.
#
# The Steane code is a $[[7,1,3]]$ code. It encodes one logical qubit into seven physical qubits and, with distance three, corrects an arbitrary single-qubit error. See {cite:t}`10.7907/rzr7-dt72` for details of the stabilizer formalism.

# %% [markdown]
# ## Construct checks from the Hamming matrix
#
# The Calderbank–Shor–Steane (CSS) construction uses classical parity checks to define separate $X$-type and $Z$-type checks. For the Steane code, label the data qubits $q_0,\ldots,q_6$ and use the following Hamming parity-check matrix for both sets of checks
#
# $$
# H_{\mathrm{cl}}=
# \begin{pmatrix}
# 0&0&0&1&1&1&1\\
# 0&1&1&0&0&1&1\\
# 1&0&1&0&1&0&1
# \end{pmatrix}.
# $$
#
# For each row, place $Z$ at its nonzero positions to obtain three generators, and similarly place $X$ to obtain three more. Each row has weight four, and distinct rows overlap at two positions, so all six checks commute. The $Z$ checks detect the bit-flip component, and the $X$ checks detect the phase-flip component.
#
# Column $j$ is the binary representation of $j+1$, so three syndrome bits locate a single error. The bits have weights 4, 2, and 1, in that order. The full ten-qubit register is ordered as three ancillas followed by seven data qubits, with $q_j$ at position `3+j`. We measure the two syndromes sequentially using the same three ancillas and store the results in classical bits.

# %% [markdown]
# ## Circuits for logical zero and syndrome extraction
#
# Logical zero $|0_L\rangle$ is the equal-amplitude superposition of the eight even-weight seven-bit strings satisfying $H_{\mathrm{cl}}\mathbf{x}=0\pmod 2$. Below we define the encoder and the bit-flip and phase-flip syndrome extractors. A Z check records parity with data-to-ancilla CX gates; an X check uses ancilla-to-data CX gates between Hadamards on the ancilla. Both record eigenvalues $+1,-1$ as bits 0,1, using ancillas initialized to $|0\rangle$.


# %%
@qm.qkernel
def steane_encode_zero(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Encode the final seven components as the Steane logical zero."""

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
    """Extract one Steane bit-error syndrome coherently."""

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
    """Extract one Steane phase-error syndrome coherently."""

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
# ## Correct using the measured syndrome
#
# When the three measured bits match the binary representation of $j+1$, apply a correction to data qubit $q_j$. The following `if` statements check weights 4, 2, and 1 in order and do nothing for 000. The conditions use measurement outcomes obtained during circuit execution.


# %%
@qm.qkernel
def steane_correct_x(
    data: qm.Q[qm.Bits[7]], syndrome: qm.Shot[qm.Bits[3]]
) -> qm.Q[qm.Bits[7]]:
    """Apply X at the position selected by a measured syndrome."""

    if not syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=0)
    if not syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=1)
    if not syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=2)
    if syndrome[0]:
        if not syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=3)
    if syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=4)
    if syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.x(data, target=5)
    if syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.x(data, target=6)
    return data


@qm.qkernel
def steane_correct_z(
    data: qm.Q[qm.Bits[7]], syndrome: qm.Shot[qm.Bits[3]]
) -> qm.Q[qm.Bits[7]]:
    """Apply Z at the position selected by a measured syndrome."""

    if not syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=0)
    if not syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=1)
    if not syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=2)
    if syndrome[0]:
        if not syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=3)
    if syndrome[0]:
        if not syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=4)
    if syndrome[0]:
        if syndrome[1]:
            if not syndrome[2]:
                data = qm.z(data, target=5)
    if syndrome[0]:
        if syndrome[1]:
            if syndrome[2]:
                data = qm.z(data, target=6)
    return data


# %% [markdown]
# ## Recover from both syndromes
#
# Let $h_j$ denote column $j$ of the Hamming matrix. A single Pauli error gives the following pair of syndromes:
#
# | Error | Bit-flip syndrome | Phase-flip syndrome |
# | --- | --- | --- |
# | None | $0$ | $0$ |
# | $X_j$ | $h_j$ | $0$ |
# | $Z_j$ | $0$ | $h_j$ |
# | $Y_j$ | $h_j$ | $h_j$ |
#
# `steane_recover` receives only the quantum state. After extracting the phase-flip syndrome, `qm.measure_reset` stores the outcome in classical bits and returns the three ancillas to $|000\rangle$. The same ancillas extract the bit-flip syndrome and are measured and reset again. Finally, apply X for the bit-flip component and Z for the phase-flip component. Both act for a Y error, while the seven data qubits are retained.
#
# For the same correctable Pauli error, logical zero and logical one produce the same syndrome, so an ideal syndrome measurement cannot tell them apart and preserves their superposition. `qm.measure_reset` then resets only the measured ancillas, preserving the logical information in the data while the saved syndrome bits determine the corrections.
#
# :::{note}
# This circuit requires mid-circuit measurement, reset, and branching on measurement outcomes. The example below runs on Reference. Qiskit Aer, Selene, Braket Local, and CUDA-Q have also been checked, but Qamomile's current Cirq and QURI Parts/Qulacs integrations do not support this construction.
# :::


# %%
@qm.qkernel
def steane_recover(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Q[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Measure both syndromes with three reusable ancillas and correct the data."""

    state = steane_extract_phase_syndrome(state)
    ancilla, data = qm.split_at(state, 3)
    phase, ancilla = qm.measure_reset(ancilla)
    state = steane_extract_bit_syndrome(qm.concat(ancilla, data))
    ancilla, data = qm.split_at(state, 3)
    bit, ancilla = qm.measure_reset(ancilla)
    data = steane_correct_x(data, bit)
    data = steane_correct_z(data, phase)
    return bit, phase, qm.concat(ancilla, data)


# %% [markdown]
# ## Inject a test error and run recovery
#
# `error_kind` and `error_position` select the fault injected in the experiment. Kind 0 means no error; 1, 2, and 3 mean X, Y, and Z. These values are not passed to the recovery function. `logical_plus=False` prepares logical zero; the later phase check uses logical plus.


# %%
@qm.qkernel
def steane_transversal_h(
    state: qm.Q[qm.Bits[10]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[10]]:  # type: ignore[valid-type]
    """Apply the logical Hadamard transversally to the Steane data."""

    for target in range(3, 10):
        state = qm.h(state, target=target)
    return state


@qm.qkernel
def steane_code_state(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
    logical_plus: qm.Const[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Q[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Prepare a logical state, inject a test error, and recover without its label."""

    state = steane_encode_zero(qm.qubits(10))
    if logical_plus:
        state = steane_transversal_h(state)
    if error_kind == 1:
        state = qm.x(state, target=error_position + 3)
    if error_kind == 2:
        state = qm.y(state, target=error_position + 3)
    if error_kind == 3:
        state = qm.z(state, target=error_position + 3)
    return steane_recover(state)


@qm.qkernel
def steane_code_samples(
    error_kind: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
    logical_plus: qm.Const[qm.Bool],
) -> tuple[qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[3]], qm.Shot[qm.Bits[10]]]:  # type: ignore[valid-type]
    """Return both classical syndromes and data read in the preparation basis."""

    bit, phase, state = steane_code_state(error_kind, error_position, logical_plus)
    if logical_plus:
        state = steane_transversal_h(state)
    return bit, phase, qm.measure(state)


# %%
ERROR_KIND = 2
ERROR_POSITION = 3
SHOTS = 1 if docs_test_mode else 4

program = steane_code_samples.specialize(
    error_kind=ERROR_KIND,
    error_position=ERROR_POSITION,
    logical_plus=False,
)
endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
record = qm.run(artifact, on=endpoint, shots=SHOTS, seed=2)


# %% [markdown]
# The keys of `record.result.values` are `(bit, phase, bits)`. `bit` and `phase` are the two saved three-bit syndromes. `bits` is the final ten-qubit measurement: `bits[:3]` contains the reset ancillas, and `bits[3:]` contains the data. The following host-side function constructs expected values for checks; it does not provide inputs to the recovery circuit.


# %%
def expected_syndromes(error_kind: int, position: int) -> tuple[int, ...]:
    """Return the two expected syndromes for checking an injected test error."""

    location = tuple(int(bool((position + 1) & mask)) for mask in (4, 2, 1))
    bit = location if error_kind in (1, 2) else (0, 0, 0)
    phase = location if error_kind in (2, 3) else (0, 0, 0)
    return bit + phase


def is_steane_zero_word(bits: tuple[int, ...]) -> bool:
    """Return whether seven bits form a Steane logical-zero codeword."""

    return (
        len(bits) == 7
        and (bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) == 0
        and (bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) == 0
        and (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) == 0
        and sum(bits) % 2 == 0
    )


expected = expected_syndromes(ERROR_KIND, ERROR_POSITION)
print("corrected values:", record.result.values)
print("bit syndromes:", {bit for bit, phase, bits in record.result.values})
print("phase syndromes:", {phase for bit, phase, bits in record.result.values})

assert record.result.shots == SHOTS
assert all(bit + phase == expected for bit, phase, bits in record.result.values)
assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in record.result.values)
assert all(is_steane_zero_word(bits[3:]) for bit, phase, bits in record.result.values)

# %% [markdown]
# ## Test single errors at every position
#
# Normal execution checks no error and all 21 combinations of X, Y, and Z at seven positions. Every case uses the same recovery function. Computational-basis measurements check the syndrome and codeword bit strings. These alone cannot verify recovery of relative phase, so we next check in another basis.

# %%
cases = (
    ((ERROR_KIND, ERROR_POSITION),)
    if docs_test_mode
    else ((0, 0),)
    + tuple((kind, position) for kind in (1, 2, 3) for position in range(7))
)
for kind, position in cases:
    result = qm.run(
        steane_code_samples,
        on=endpoint,
        error_kind=kind,
        error_position=position,
        logical_plus=False,
        shots=SHOTS,
        seed=10 + position,
    ).result
    assert all(
        bit + phase == expected_syndromes(kind, position)
        for bit, phase, bits in result.values
    )
    assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in result.values)
    assert all(is_steane_zero_word(bits[3:]) for bit, phase, bits in result.values)
print(f"Checked {len(cases)} patterns (no error and single-qubit errors)")

# %% [markdown]
# ## Logical Hadamard and a phase-sensitive check
#
# Applying H to each of the seven data qubits is called a transversal Hadamard. Since $HXH=Z$ and $HZH=X$, it exchanges the Steane code's X and Z generators. It also exchanges the logical operators $\overline Z=Z_0\cdots Z_6$ and $\overline X=X_0\cdots X_6$, taking the logical-Z $+1$ state $|0_L\rangle$ to the logical-X $+1$ state $|+_L\rangle$.
#
# Here we inject a Z error into logical plus, correct it, and apply H to each data qubit before measuring. For measured data string $\mathbf b$, the mean of $(-1)^{\sum_j b_j}$ is the expectation value of the logical operator $\overline X=X_0\cdots X_6$. We check that it returns to $+1$. This quantity is sensitive to relative phase: leaving the physical Z error uncorrected reverses its sign, revealing an effect that computational-basis bit strings cannot show. This is one check, not complete verification for arbitrary states or fault tolerance.

# %%
plus_result = qm.run(
    steane_code_samples,
    on=endpoint,
    error_kind=3,
    error_position=ERROR_POSITION,
    logical_plus=True,
    shots=SHOTS,
    seed=3,
).result
assert all(
    bit + phase == expected_syndromes(3, ERROR_POSITION)
    for bit, phase, bits in plus_result.values
)
assert all(bits[:3] == (0, 0, 0) for bit, phase, bits in plus_result.values)
corrected_plus_x = (
    sum(
        (-1) ** sum(bits[3:]) * count
        for (bit, phase, bits), count in plus_result.values.items()
    )
    / plus_result.shots
)
assert corrected_plus_x == 1.0
print("Logical X after phase-error correction:", corrected_plus_x)

# %% [markdown]
# ## Summary
#
# We extracted two syndromes from the Steane code's six generators and used only their values to control X and Z corrections. Error kind and position select only the test fault. Measuring, resetting, and reusing three ancillas while storing both syndromes classically keeps the circuit at ten qubits.
#
# In the full run, all 22 cases produced the expected syndromes, and every bit string measured after correction was a logical-zero codeword. After correcting a $Z$ error on logical plus, we measured a logical $X$ expectation of $+1$. Faults in gates, measurements, or resets, and fault tolerance with repeated correction, are outside this example.
