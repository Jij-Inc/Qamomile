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
# tags: [integration]
# ---
#
# # HUGR Support
#
# Compile a two-qubit circuit to [HUGR](https://github.com/Quantinuum/hugr), an intermediate representation for quantum programs {cite:p}`hugr-en`. Run it locally with Selene and see how to submit the same circuit to Quantinuum’s Helios-1 hardware.
#
# :::{note}
# In this guide, “supported” means you can use the feature through Qamomile. HUGR and individual devices may offer different capabilities.
# :::

# %% [markdown]
# ## Setup
#
# Install Qamomile with HUGR support. The local examples use the Selene simulator.
#
# ```bash
# pip install "qamomile[hugr]"
# ```

# %%
# # %pip install "qamomile[hugr]"

# %%
import math
import os

import qnexus as qnx  # noqa: F401 -- Used in the hardware example.

import qamomile as qm
from qamomile.hugr import Helios, HeliosOptions, Selene  # noqa: F401

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
shots = 1 if docs_test_mode else 32
local = Selene()

# %% [markdown]
# ## What you can do with HUGR
#
# ✅ means supported and ✗ means unsupported.
#
# | Feature | `Selene` | `Helios` (Helios-1) |
# | --- | :---: | :---: |
# | Supply a rotation angle at run time | ✅ | ✅ |
# | Discard a qubit without measuring it (`qm.discard`) | ✅ | ✅ |
# | Measure a qubit mid-circuit (`qm.measure_z`, `qm.measure_x`, `qm.measure_y`) | ✅ | ✅ |
# | Branch within the same circuit on a mid-circuit measurement result (`if`) | ✅ | ✅ |
# | Use a mid-circuit measurement result to set a `while` condition or a `for` loop's range in the same circuit | ✗ | ✗ |
# | Reset and reuse a qubit mid-circuit (`qm.reset`) | ✅ | ✅ |
# | Select a classical array element using an index derived from a mid-circuit measurement | ✅ | ✅ |
# | Read the current shot number inside the circuit (`qm.get_current_shot`) | ✅ | ✗ |
#
# Selene and Helios-1 support measurements in the X, Y, and Z bases. Both require loop conditions and iteration counts to be known before the circuit runs.

# %% [markdown]
# ## Local simulation
#
# ### Supply a rotation angle at run time
#
# Rotate the first qubit by `theta` radians. Apply a CNOT gate. The only measurements are at the end, on both qubits. Declaring `theta` as `qm.Param` lets you supply its value at each run.


# %%
@qm.qkernel
def parameterized_bell(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Prepare and measure a parameterized Bell-like state.

    Args:
        theta (qm.Param[qm.Real]): Rotation angle in radians.

    Returns:
        qm.Shot[qm.Bits[2]]: Two measured output bits.
    """

    state = qm.qubits(2)
    state = qm.ry(state, theta, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# Compile the circuit once for the local simulator. Run it with two values of `theta`, using `shots` runs per angle.

# %%
bell_program = qm.specialize(parameterized_bell)
bell_artifact = qm.compile(bell_program, to=local.target)

for theta, expected_bits in [(0.0, (0, 0)), (math.pi, (1, 1))]:
    result = qm.run(
        bell_artifact, on=local, parameters={"theta": theta}, shots=shots
    ).result
    counts = result.counts
    print(f"theta = {theta:.3f}", counts)
    assert counts == {expected_bits: shots}

# %% [markdown]
# `counts` maps each pair of measured bits to its frequency. Both bits are 0 at zero radians and 1 at $\pi$ radians on every shot.

# %% [markdown]
# ### Discard a qubit without measuring it (`qm.discard`)
#
# Suppose you only need to know whether two bits differ. A CNOT gate writes their parity into the first qubit. It holds 0 if the bits match and 1 if they differ. Use `qm.split_at()` to separate the qubits, release the second with `qm.discard()`, and measure the first. Here the input is `(1, 1)`, so the result is 0.


# %%
@qm.qkernel
def parity_only() -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    """Return the parity of two bits and discard the unused qubit."""
    state = qm.prepare((1, 1))
    state = qm.cx(state, control=1, target=0)
    parity, unused = qm.split_at(state, 1)
    qm.discard(unused)
    return qm.measure(parity)


parity_counts = qm.run(parity_only, on=local, shots=shots).result.counts
print(parity_counts)
assert parity_counts == {(0,): shots}

# %% [markdown]
# ### Measure a qubit mid-circuit (`qm.measure_z`, `qm.measure_x`, `qm.measure_y`)
#
# Prepare a qubit in state 1 and measure it with `qm.measure_z(state, target=0)`. The function returns both the measured bit and the qubit. Apply an X gate to that qubit and measure it again. The two measurements give `(1, 0)` on every shot.


# %%
@qm.qkernel
def measure_then_flip() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    """Measure the same qubit before and after a flip."""
    state = qm.prepare((1,))
    before, state = qm.measure_z(state, target=0)
    state = qm.x(state, target=0)
    return bool(before), bool(qm.measure(state)[0])


measurement_counts = qm.run(measure_then_flip, on=local, shots=shots).result.counts
print(measurement_counts)
assert measurement_counts == {(1, 0): shots}

# %% [markdown]
# ### Branch within the same circuit on a mid-circuit measurement result (`if`)
#
# Prepare the first qubit so that measuring it can give either 0 or 1. `measure_z` returns that bit (`flag`) and the state after measurement. If `flag` is 1, flip the second qubit. It started at 0, so the two bits will agree at the end.


# %%
@qm.qkernel
def correlated_feedback() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Correlate two output bits through measurement feedback.

    Returns:
        qm.Shot[qm.Bits[2]]: Two equal bits in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, target=0)
    if flag:
        state = qm.x(state, target=1)
    return qm.measure(state)


counts = qm.run(correlated_feedback, on=local, shots=shots).result.counts
print(counts)
assert set(counts) <= {(0, 0), (1, 1)}
assert sum(counts.values()) == shots

# %% [markdown]
# The count for each outcome can vary between runs.

# %% [markdown]
# ### Reset and reuse a qubit mid-circuit (`qm.reset`)
#
# The first three qubits hold `010`. Use a fourth qubit to check whether each neighboring pair differs. Two CNOT gates flip this auxiliary qubit once for each 1 in the pair. Starting from 0, it therefore reads 1 for `01`. Reset it to 0 and repeat the check for `10`. Both checks give 1.


# %%
@qm.qkernel
def check_neighbors() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    """Use one auxiliary qubit to compare two neighboring pairs."""
    state = qm.prepare((0, 1, 0, 0))
    state = qm.cx(state, control=0, target=3)
    state = qm.cx(state, control=1, target=3)
    left, state = qm.measure_z(state, target=3)

    data, probe = qm.split_at(state, 3)
    probe = qm.reset(probe)
    state = qm.concat(data, probe)

    state = qm.cx(state, control=1, target=3)
    state = qm.cx(state, control=2, target=3)
    right, state = qm.measure_z(state, target=3)
    qm.discard(state)
    return bool(left), bool(right)


checks = qm.run(check_neighbors, on=local, shots=shots).result.counts
print(checks)
assert checks == {(1, 1): shots}

# %% [markdown]
# ### Select a classical array element using an index derived from a mid-circuit measurement
#
# Use the first measured bit as an index into a pair of rotation angles. Measuring 1 selects $\pi$ and flips the second qubit, so every shot gives `(1, 1)`.


# %%
@qm.qkernel
def select_rotation() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Use a measured index to select the rotation on the second qubit."""
    state = qm.prepare((1, 0))
    flag, state = qm.measure_z(state, target=0)
    index = 1 if flag else 0
    angles = [0.0, qm.pi]
    state = qm.ry(state, angles[index], target=1)
    return qm.measure(state)


lookup_counts = qm.run(select_rotation, on=local, shots=shots).result.counts
print(lookup_counts)
assert lookup_counts == {(1, 1): shots}

# %% [markdown]
# ### Read the current shot number inside the circuit (`qm.get_current_shot`)
#
# Return whether the current shot number is 0. The first shot returns 1 and each remaining shot returns 0. With 32 shots, the counts are 1 and 31 respectively.


# %%
@qm.qkernel
def first_shot() -> qm.Shot[qm.Bool]:
    """Return whether this is the first shot."""
    return qm.get_current_shot() == 0


shot_counts = qm.run(first_shot, on=local, shots=shots).result.counts
print(shot_counts)
assert shot_counts[(1,)] == 1
assert shot_counts.get((0,), 0) == shots - 1

# %% [markdown]
# ## Hardware execution
#
# To submit the same `bell_program` to Helios-1, sign in to Quantinuum Nexus with an account that has device access. Replace `"your-project"` with the name of your Nexus project. Use `job.job_id` to identify the submission and `job.result()` to wait for the counts.
#
# ```python
# qnx.login()
# project = qnx.projects.get(name="your-project")
# endpoint = Helios(HeliosOptions(project=project, system_name="Helios-1"))
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
