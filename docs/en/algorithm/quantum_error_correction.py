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
# Implement the three-qubit repetition code and the nine-qubit Shor code using Qamomile gates and controlled operations. We encode one logical qubit, inject an error, extract a syndrome, correct the error, and decode the state. Familiarity with Pauli gates and controlled gates is assumed.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import math
import os

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"


# %% [markdown]
# ## The three-qubit bit-flip code
#
# Let the input state be $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$. Adding two data qubits initialized to $|00\rangle$ and encoding it as $\alpha|000\rangle+\beta|111\rangle$ allows correction of an $X$ error on any one data qubit.
#
# Label the data qubits $q_0,q_1,q_2$ and record the eigenvalues of $Z_0Z_1$ and $Z_0Z_2$ in two ancillary qubits. These Pauli checks are stabilizers: every encoded state has eigenvalue $+1$ for each. Mapping eigenvalue $+1$ to 0 and $-1$ to 1 gives the syndrome $(s_0,s_1)$. It distinguishes error positions without measuring the logical amplitudes $\alpha,\beta$.
#
# | Error position | $(s_0,s_1)$ | Correction |
# | --- | --- | --- |
# | None | $(0,0)$ | None |
# | $q_0$ | $(1,1)$ | $X_0$ |
# | $q_1$ | $(1,0)$ | $X_1$ |
# | $q_2$ | $(0,1)$ | $X_2$ |
#
# Here `qm.qubits` initializes all data and ancillary qubits to $|0\rangle$. Below, RY prepares the input, and an $X$ error is applied at `error_position`. The ancillary qubits control the correction gates without an intermediate syndrome measurement. The syndrome remains after decoding.


# %%
@qm.qkernel
def bit_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one bit-flip error."""

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
    """Sample the runtime-parametric three-qubit repetition code."""

    return qm.measure(bit_flip_code_state(qm.qubits(5), theta, error_position))


# %% [markdown]
# ## Correct and decode
#
# Fix the error position with `specialize` and run on `Reference`, Qamomile’s built-in simulator. Supply the angle `theta` at execution time. The measurement result contains the logical qubit, two data qubits returned to zero, and the two syndrome bits, in that order.

# %%
ERROR_POSITION = 1
program = bit_flip_code_samples.specialize(
    error_position=ERROR_POSITION,
)

# %%
THETA = math.pi / 3.0
SHOTS = 1 if docs_test_mode else 256
EXPECTED_SYNDROME = ((1, 1), (1, 0), (0, 1))[ERROR_POSITION]

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
sampling_error = math.sqrt(math.log(4.0e6) / (2 * SHOTS))

print("corrected counts:", result.counts)
print("P(logical=1):", observed_one_probability)

assert result.shots == SHOTS
assert sum(result.counts.values()) == SHOTS
assert all(bits[1:3] == (0, 0) for bits in result.counts)
assert all(bits[3:5] == EXPECTED_SYNDROME for bits in result.counts)
if not docs_test_mode:
    assert abs(observed_one_probability - expected_one_probability) <= sampling_error


# %% [markdown]
# Since the input has $\alpha=\cos(\theta/2)$ and $\beta=\sin(\theta/2)$, compare the decoded $P(1)$ with $\sin^2(\theta/2)$. `sampling_error` is a Hoeffding tolerance with a two-sided failure probability of at most $5\times10^{-7}$ per comparison, assuming independent ideal shots {cite:p}`10.1080/01621459.1963.10500830`. Computational-basis measurements alone cannot determine whether relative phase was preserved. Checking states with relative phase requires measurements in another basis or comparison of the full state.

# %% [markdown]
# ## The phase-flip code
#
# Apply H to the three encoded data qubits, inject a $Z$ error, and apply H again. Since $HZH=X$, the same syndrome table and correction circuit can then be used. This code corrects one $Z$ error. When passing the kernel directly to `qm.run`, supply the `Const` error position as a keyword argument and the `Param` angle through `parameters`.


# %%
@qm.qkernel
def phase_flip_code_state(
    state: qm.Q[qm.Bits[5]],  # type: ignore[valid-type]
    theta: qm.Real,
    error_position: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[5]]:  # type: ignore[valid-type]
    """Encode, diagnose, and coherently correct one phase-flip error."""

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
    """Sample the runtime-parametric phase-flip repetition code."""

    return qm.measure(phase_flip_code_state(qm.qubits(5), theta, error_position))


phase_result = qm.run(
    phase_flip_code_samples,
    on=endpoint,
    error_position=ERROR_POSITION,
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
assert phase_result.shots == SHOTS
assert sum(phase_result.counts.values()) == SHOTS
if not docs_test_mode:
    assert abs(phase_one_probability - expected_one_probability) <= sampling_error


# %% [markdown]
# ## Correct both components with the Shor code
#
# A single-qubit error operator is a linear combination of $I,X,Y,Z$, which is why we use Pauli errors as the error basis. Correcting this basis throughout the code space with one recovery operation also covers its linear combinations; the finite-shot examples below do not establish that general result on their own.
#
# To handle single-qubit Pauli errors, including $Y=iXZ$, the Shor code combines phase-flip and bit-flip encoding. Divide nine data qubits into three blocks of three. Extract six $Z$ parities within the blocks and two $X$ parities between blocks. This example adds eight ancillary qubits, giving a 17-qubit circuit. The two X checks are $X_0X_1X_2X_3X_4X_5$ and $X_3X_4X_5X_6X_7X_8$. A phase error in block 0, 1, or 2 gives syndrome $(1,0)$, $(1,1)$, or $(0,1)$, corresponding to `control_state` 1, 3, or 2. Within an encoded block, single-qubit Z errors act identically, so phase correction can use representatives $q_0,q_3,q_6$.
#
# `error_type` selects $X,Y,Z$ as 1, 2, and 3, respectively. Correction uses the extracted syndrome to control `qm.control`. Finally, reverse the encoding and return the logical state to $q_0$. See Gottesman's introduction {cite:p}`10.48550/arXiv.quant-ph/9705052` for the coding theory.


# %%
@qm.qkernel
def flip_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Flip one component of a Shor data block."""

    return qm.x(state, target=position)


@qm.qkernel
def phase_shor_data(state: qm.Q[qm.Bits[9]], position: qm.Int) -> qm.Q[qm.Bits[9]]:
    """Apply one phase correction to a Shor data block."""

    return qm.z(state, target=position)


@qm.qkernel
def shor_code_samples(
    theta: qm.Param[qm.Real],
    error_type: qm.Const[qm.Int],
    error_position: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[17]]:  # type: ignore[valid-type]
    """Encode, correct, decode, and sample one Shor-code experiment."""

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


# %% [markdown]
# ## Check single Pauli errors
#
# Normal execution checks $X,Y,Z$ at each of the nine positions, for 27 cases. Here the input is fixed to $|1\rangle$, and we check the decoded logical bit and the remaining data bits. This is a minimal execution example of the correction circuit, not a complete verification for arbitrary states.

# %%
shor_cases = (
    (("Y", 2, 4),)
    if docs_test_mode
    else tuple(
        (name, kind, position)
        for kind, name in enumerate(("X", "Y", "Z"), 1)
        for position in range(9)
    )
)
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
    assert all(bits[0] == 1 and bits[1:9] == (0,) * 8 for bits in shor_result.counts)

# %% [markdown]
# ## Summary and scope
#
# The repetition code used the syndrome to correct an $X$ error. Changing basis with $H$ let the same circuit correct a $Z$ error. In the Shor example, the decoded logical bit was 1 for all 27 combinations of error type and position tested with input $|1\rangle$.
#
# The three-qubit repetition code corrects one error of the assumed kind. The Shor code can correct an arbitrary single-qubit error, but correction of errors on two or more qubits is not generally guaranteed. For example, flipping both $q_0$ and $q_1$ in the bit-flip code gives the same syndrome as flipping only $q_2$, causing an incorrect logical bit flip.
#
# The encoding, syndrome-extraction, and correction gates used here are ideal. This does not demonstrate fault tolerance with circuit faults or repeated correction. The next page, [Stabilizer Formalism and the Steane Code](steane_code.ipynb), describes syndromes through Pauli operators.
