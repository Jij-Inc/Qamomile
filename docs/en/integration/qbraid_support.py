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
# # qBraid Support
#
# Prepare a two-qubit circuit and see how to submit it through [qBraid](https://docs.qbraid.com/) {cite:p}`qbraid-en`. This example uses IQM Emerald, a quantum processor accessible through the cloud.
#
# :::{note}
# In this guide, “supported” means you can use the feature through Qamomile. qBraid and individual devices may offer different capabilities.
# :::

# %% [markdown]
# ## Setup
#
# Install Qamomile with qBraid support. You can prepare and inspect the circuit before signing in to qBraid.
#
# ```bash
# pip install "qamomile[qbraid]"
# ```

# %%
# # %pip install "qamomile[qbraid]"

# %%
# ruff: noqa: F401 -- The hardware example uses math and QBraidDevice.
import math

import qamomile as qm
from qamomile.qbraid import QBraidDevice
from qamomile.qiskit import Qiskit, load_circuit

# %% [markdown]
# Rotate the first qubit by `theta` radians, apply a CNOT gate, and measure both qubits. Declaring `theta` as `qm.Param` lets you choose the angle when you submit the circuit.


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


bell_program = qm.specialize(parameterized_bell)

# %% [markdown]
# qBraid accepts a Qiskit circuit. Compile `bell_program` and inspect its two qubits and one unbound angle locally.

# %%
artifact = qm.compile(bell_program, to=Qiskit())
circuit = load_circuit(artifact)
print(circuit.draw(output="text"))
assert circuit.num_qubits == 2
assert circuit.num_parameters == 1

# %% [markdown]
# ## What you can do with qBraid
#
# This table shows the features Qamomile supports through `QBraidDevice` across devices.
#
# ✅ means supported and ✗ means unsupported.
#
# | Feature | `QBraidDevice` |
# | --- | :---: |
# | Supply a rotation angle at run time | ✅ |
# | Discard a qubit without measuring it (`qm.discard`) | ✅ |
# | Measure a qubit mid-circuit (`qm.measure_z`, `qm.measure_x`, `qm.measure_y`) | ✗ |
# | Branch within the same circuit on a mid-circuit measurement result (`if`) | ✗ |
# | Use a mid-circuit measurement result to set a `while` condition or a `for` loop's range in the same circuit | ✗ |
# | Reset and reuse a qubit mid-circuit (`qm.reset`) | ✗ |
# | Select a classical array element using an index derived from a mid-circuit measurement | ✗ |
# | Read the current shot number inside the circuit (`qm.get_current_shot`) | ✗ |
#
# Qamomile binds the angle before sending the circuit. `QBraidDevice` supports measurements at the end of the circuit.

# %% [markdown]
# ## Hardware execution
#
# Configure your qBraid API key and check that your account has access to IQM Emerald. Submitting a job uses qBraid credits. The example sends the same `bell_program` with a rotation of $\pi$ radians. Use `job.job_id` to identify the submission and `job.result()` to wait for the counts.
#
# ```python
# endpoint = QBraidDevice.from_device_id("aws:iqm:qpu:emerald")
# artifact = qm.compile(bell_program, to=endpoint.target)
# job = qm.submit(artifact, on=endpoint, parameters={"theta": math.pi}, shots=100)
# print(job.job_id)
# print(job.result().counts)
# ```
