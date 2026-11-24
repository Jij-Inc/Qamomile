# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# tags: [algorithm, encoding, primitive]
# ---
#
# # Möttönen Amplitude Encoding
#
# Prepare two-qubit states from positive real, signed real, and complex amplitude vectors using the Möttönen method. Compare two ways to prepare the same inputs: passing precomputed angles or passing the amplitudes directly. We compare the measurement probabilities and gate counts of the two circuits.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import os

import qamomile as qm
from qamomile.stdlib import amplitude_angles, amplitude_encoding, amplitude_encoding_from_angles

SAMPLE_SHOTS = 1 if os.environ.get("QAMOMILE_DOCS_TEST") == "1" else 128

# %% [markdown]
# ## Background
#
# Amplitude encoding normalizes a nonzero vector $a\in\mathbb{C}^{2^n}$ of length $2^n$ and maps it to an $n$-qubit quantum state. Writing $\tilde a_i=a_i/\|a\|_2$, we have
#
# $$
# |0\rangle^{\otimes n}\longmapsto
# |\psi(a)\rangle=\sum_{i=0}^{2^n-1}\tilde a_i|i\rangle,
# \qquad \sum_i|\tilde a_i|^2=1.
# $$
#
# The original paper uses uniformly controlled rotations to transform an arbitrary initial state into a chosen target state {cite:p}`10.48550/arXiv.quant-ph/0407010`. Here we compute the rotation angles for a register that starts in the all-zero state. Applying the same circuit to a different initial state generally produces a different final state.
#
# ## Problem Settings
#
# We use two qubits (four amplitudes) to prepare states from three types of unnormalized input vector:
#
# - Positive real vector: $(1,2,3,4)$, with norm $\sqrt{30}$.
# - Signed real vector: $(1,-1,1,-1)$, with norm $2$.
# - Complex vector: $(1,1+i,1-i,2i)$, with norm $3$.
#
# In each case, the amplitudes of the prepared state are the input values divided by the vector's norm.
#
# We take qubit $0$ to be the least significant bit and write the state corresponding to index $i=b_0+2b_1$ as $|b_1b_0\rangle$. Qamomile's measurement bitstrings are ordered as $(b_0,b_1)$. For example, $(1,0)$ corresponds to index 1, or $|01\rangle$.

# %%
input_vectors = {
    "positive": (1.0, 2.0, 3.0, 4.0),
    "signed": (1.0, -1.0, 1.0, -1.0),
    "complex": (1.0, 1.0 + 1.0j, 1.0 - 1.0j, 2.0j),
}

# %% [markdown]
# ## Algorithm
#
# Möttönen state preparation consists of two types of rotation:
#
# 1. Uniformly controlled $R_y$ rotations distribute the required amplitude magnitudes across computational-basis states.
# 2. Uniformly controlled $R_z$ rotations introduce relative phases, including the $\pi$ phase difference between positive and negative real amplitudes. Qamomile's fixed-input `amplitude_encoding` omits this phase-rotation sequence when all computed $R_z$ angles are zero.
#
# Uniformly controlled rotations are decomposed into a gate sequence based on Gray codes. For a fixed-input state-preparation circuit constructed with `amplitude_encoding`, the logical gate counts depend on the computed $R_z$ angles:
#
# | Circuit type | Total $R_y$ / $R_z$ rotations | `CX` gates |
# |---|---:|---:|
# | All $R_z$ angles are zero ($R_y$ only) | $2^n-1$ | $2^n-2$ |
# | At least one $R_z$ angle is nonzero ($R_y$ and $R_z$) | $2(2^n-1)$ | $2(2^n-2)$ |
#
# A phase shared by all nonzero amplitudes is retained separately as a global phase. A fixed-input circuit can omit the $R_z$ sequence even for negative amplitudes if all computed $R_z$ angles are zero.
#
# In the two-qubit $R_y$ sequence, the first stage rotates qubit 1 once. The next stage rotates qubit 0 through an angle selected by the value of qubit 1. This stage uses two rotations and two `CX` gates, giving three rotations and two `CX` gates in total.
#
# In general, a stage with $k\ge1$ control qubits has $2^k$ possible control bitstrings. This decomposition uses $2^k$ rotations and $2^k$ `CX` gates at that stage. Summing over the $n$ stages, including the first uncontrolled rotation, gives $2^n-1$ rotations and $2^n-2$ `CX` gates.

# %% [markdown]
# ## Implementation
#
# ### Pass Angles to `amplitude_encoding_from_angles`
#
# This circuit starts two qubits in the all-zero state. At run time it takes three $R_y$ angles, three $R_z$ angles, and a global phase. Compile the circuit once. When running it, use `amplitude_angles` to compute the angles for the current input.

# %%
encode_angles = amplitude_encoding_from_angles(2)


@qm.qkernel
def samples_from_angles(
    ry_angles: qm.Param[qm.Vector[qm.Real, 3]],
    rz_angles: qm.Param[qm.Vector[qm.Real, 3]],
    phase: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:
    """Prepare and measure two qubits using the supplied rotation angles."""
    state = encode_angles(qm.qubits(2), ry_angles, rz_angles, phase)
    return qm.measure(state)


reference = qm.Reference()
angle_program = qm.specialize(samples_from_angles)
angle_artifact = qm.compile(angle_program, to=reference)

# %% [markdown]
# ### Pass Amplitudes Directly to `amplitude_encoding`
#
# Given an amplitude vector, `amplitude_encoding` computes the Möttönen rotation angles. Define one measurement circuit for positive real amplitudes, one for real amplitudes of both signs, and one for complex amplitudes.

# %%
prepare_positive = amplitude_encoding(input_vectors["positive"])


@qm.qkernel
def sample_positive() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_positive(qubits)
    return qm.measure(state)


# %%
prepare_signed = amplitude_encoding(input_vectors["signed"])


@qm.qkernel
def sample_signed() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_signed(qubits)
    return qm.measure(state)


# %%
prepare_complex = amplitude_encoding(input_vectors["complex"])


@qm.qkernel
def sample_complex() -> qm.Shot[qm.Bits[2]]:
    qubits = qm.qubits(2)
    state = prepare_complex(qubits)
    return qm.measure(state)


# %%
fixed_programs = {
    "positive": qm.specialize(sample_positive),
    "signed": qm.specialize(sample_signed),
    "complex": qm.specialize(sample_complex),
}

# %% [markdown]
# ## Results
#
# Measure each input with both circuits and compare the probability of each computational basis state $|b_1b_0\rangle$. In the table, `ideal` is the theoretical probability $|a_i|^2 / \sum_j |a_j|^2$. The `from_angles` and `fixed` columns are the observed frequencies for the circuit with runtime angles and the circuit with fixed amplitudes, respectively.

# %%
bitstrings = ((0, 0), (1, 0), (0, 1), (1, 1))


def ideal_probabilities(amplitudes):
    """Return the normalized squared magnitudes in basis-index order."""
    scale = max(max(abs(value.real), abs(value.imag)) for value in amplitudes)
    weights = [abs(value / scale) ** 2 for value in amplitudes]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def sample_probabilities(program, *, seed, parameters=None):
    """Run the sampling circuit and return probabilities in the same order."""
    result = qm.run(
        program, on=reference, parameters=parameters, shots=SAMPLE_SHOTS, seed=seed
    ).result
    assert sum(result.counts.values()) == SAMPLE_SHOTS
    return tuple(result.probability(bits) for bits in bitstrings)


for name, amplitudes in input_vectors.items():
    ideal = ideal_probabilities(amplitudes)
    from_angles = sample_probabilities(
        angle_artifact, parameters=amplitude_angles(amplitudes).bindings(), seed=2
    )
    fixed = sample_probabilities(fixed_programs[name], seed=3)

    print(f"\n{name}")
    print("basis   ideal   from_angles   fixed")
    for i in range(len(ideal)):
        print(f"|{i:02b}>   {ideal[i]:.3f}   {from_angles[i]:11.3f}   {fixed[i]:.3f}")

# %% [markdown]
# The ideal probabilities are $(1,4,9,16)/30$ for the positive vector, $(1,1,1,1)/4$ for the signed vector, and $(1,2,2,4)/9$ for the complex vector. Both methods predict the same probabilities for each input. The observed frequencies fluctuate with finite sampling. This measurement checks the magnitudes of the amplitudes.
#
# Now compare the rotation and `CX` gate counts for the same three inputs.

# %%
def gate_counts(program):
    """Return the rotation and CX counts for this two-qubit circuit."""
    study = qm.resources.logical(program)
    rotations = study.metric(qm.resources.LOGICAL_ROTATION_GATES).numeric_enclosure()
    cx_metric = qm.resources.logical_operation_count("quantum.cx")
    cx = study.metric(cx_metric).numeric_enclosure()
    assert rotations[0] == rotations[1] and cx[0] == cx[1]
    return rotations[0], cx[0]


reusable_counts = gate_counts(angle_program)
print("input      from_angles (rot, CX)   fixed (rot, CX)")
for name, program in fixed_programs.items():
    print(f"{name:10} {str(reusable_counts):23} {gate_counts(program)}")

# %% [markdown]
# The circuit with runtime angles uses six rotations and four `CX` gates for every input. Among these examples, only the positive vector lets the fixed-input circuit omit the $R_z$ rotations, reducing the counts to three rotations and two `CX` gates.
#
# ## Summary
#
# To reuse one circuit across inputs, pass the angles from `amplitude_angles` to the circuit built with `amplitude_encoding_from_angles`. This quantum kernel takes the rotation angles as runtime parameters, so the circuit includes the $R_z$ gates even when all their angles are zero.
#
# For a fixed input, you can pass the amplitudes directly to `amplitude_encoding`. If all the computed $R_z$ angles are zero, the circuit omits those rotations and uses fewer gates.
