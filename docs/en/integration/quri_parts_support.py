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
# # QURI Parts Support
#
# Run a two-qubit circuit on the Qulacs simulator and see how to submit it to IQM Emerald through [QURI Parts](https://quri-sdk.qunasys.com/docs/tutorials/quri-parts/) {cite:p}`quri-parts-en`.
#
# :::{note}
# In this guide, “supported” means you can use the feature through Qamomile. QURI Parts and individual devices may offer different capabilities.
# :::

# %% [markdown]
# ## Setup
#
# Install Qamomile with its Qulacs simulator support.
#
# ```bash
# pip install "qamomile[quri_parts]"
# ```

# %%
# # %pip install "qamomile[quri_parts]"

# %%
import math
import os

import qamomile as qm
from qamomile.quri_parts import Qulacs

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
shots = 1 if docs_test_mode else 32
local = Qulacs()

# %% [markdown]
# For the hardware example, configure AWS credentials with access to Amazon Braket, then install and import the Braket sampling backend.
#
# ```bash
# pip install quri-parts-braket
# ```
#
# ```python
# from braket.aws import AwsDevice
# from braket.devices import Devices
# from quri_parts.braket.backend import BraketSamplingBackend
# from quri_parts.core.sampling import create_sampler_from_sampling_backend
#
# from qamomile.quri_parts import QuriPartsSampler
# ```

# %% [markdown]
# ## What you can do with QURI Parts
#
# `Qulacs` runs the circuit on a local simulator. `QuriPartsSampler` is a Qamomile class that passes the circuit and shot count to a QURI Parts sampler. The sampler you supply determines which simulator or quantum processor runs the circuit.
#
# ✅ means supported and ✗ means unsupported.
#
# | Feature | `Qulacs` | `QuriPartsSampler` |
# | --- | :---: | :---: |
# | Supply a rotation angle at run time | ✅ | ✅ |
# | Discard a qubit without measuring it (`qm.discard`) | ✅ | ✅ |
# | Measure a qubit mid-circuit (`qm.measure_z`, `qm.measure_x`, `qm.measure_y`) | ✗ | ✗ |
# | Branch within the same circuit on a mid-circuit measurement result (`if`) | ✗ | ✗ |
# | Use a mid-circuit measurement result to set a `while` condition or a `for` loop's range in the same circuit | ✗ | ✗ |
# | Reset and reuse a qubit mid-circuit (`qm.reset`) | ✗ | ✗ |
# | Select a classical array element using an index derived from a mid-circuit measurement | ✗ | ✗ |
# | Read the current shot number inside the circuit (`qm.get_current_shot`) | ✗ | ✗ |
#
# Both endpoints accept a rotation angle for each run and measure the circuit at the end.

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
# ## Hardware execution
#
# Submit the same `bell_program` to IQM Emerald through Amazon Braket. This example uses your AWS account and incurs charges. The sampler waits for the measurement counts during `qm.submit()`. The returned `job.job_id` identifies the completed Qamomile job.
#
# ```python
# device = AwsDevice(Devices.IQM.Emerald)
# sampler = create_sampler_from_sampling_backend(BraketSamplingBackend(device))
# endpoint = QuriPartsSampler(sampler, label="iqm-emerald")
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
