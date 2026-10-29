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
# tags: [integration]
# ---
#
# # HUGR Support
#
# Transpile a Qamomile Bell-state circuit to HUGR, an intermediate representation for quantum programs, and check it locally with Selene. This page also shows how to submit the same program to Quantinuum H2 hardware through Nexus.
#
# Use Python 3.11 or 3.12 and install the `hugr` extra. Hardware execution requires a Nexus account with access to H2-1.
#
# :::{note}
# Execution on Helios has not been verified.
# :::

# %%
# Install Qamomile with HUGR support.
# # !pip install "qamomile[hugr]"

# %%
import qamomile.circuit as qmc
from qamomile.hugr import HugrExecutor, HugrTranspiler

# %% [markdown]
# ## Build a circuit with Qamomile
#
# Prepare two entangled qubits and measure them. The return value contains the two measurement bits.


# %%
@qmc.qkernel
def bell() -> tuple[qmc.Bit, qmc.Bit]:
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


program = HugrTranspiler().transpile(bell)

# %% [markdown]
# ## Check the circuit locally
#
# Run 100 shots with Selene. In this noiseless simulation, the two measured bits always agree. `False` represents 0, and `True` represents 1.

# %%
shots = 100
local_result = program.sample(HugrExecutor("selene"), shots=shots).result()
counts = dict(local_result.results)

assert sum(counts.values()) == shots
assert set(counts) <= {(False, False), (True, True)}
print(counts)

# %% [markdown]
# ## Run on H2 hardware
#
# Set `system_name="H2-1"` to select H2 hardware. `project` is the Nexus project name; Qamomile creates it if it does not exist. Authentication starts on the first submission, with a browser login if needed.
#
# :::{note}
# The following code submits a hardware job and consumes device credits. Charges may apply. Copy it into a code cell when you are ready to submit.
# :::
#
# ```python
# from qamomile.hugr import NexusExecutionOptions
#
# executor = HugrExecutor(
#     "nexus",
#     options=NexusExecutionOptions(
#         system_name="H2-1",
#         project="qamomile-demo",
#     ),
# )
# job = program.sample(executor, shots=100)
# print(job.status())
# ```
#
# Qamomile converts and validates the program before submission. To use the H2 emulator through Nexus, change the device to `system_name="H2-1E"`.
#
# Call `job.result()` to wait for completion and retrieve the measurement counts. Hardware noise can produce outcomes where the two bits differ.
#
# ```python
# result = job.result()
# print(dict(result.results))
# ```

# %% [markdown]
# ## Use get_current_shot
#
# A shot is one repetition of a quantum program. Call `qmc.get_current_shot()` inside a function defined with `@qmc.qkernel` to read the current shot number. Repeated calls within the same shot return the same number.
#
# When running native programs on Quantinuum H2 through Nexus, device shot numbers are 1-origin: the first shot is 1. Selene uses 0-origin: the first shot is 0. Qamomile absorbs this difference, so `qmc.get_current_shot()` uses 0-origin on both Selene and H2. You do not need to subtract 1 yourself when switching to H2.
#
# The following example applies `qmc.x`, which flips 0 and 1, only on shot 0. With four shots on noiseless Selene, the first measurement is `True` and the remaining three are `False`. The counts are `{True: 1, False: 3}`.


# %%
@qmc.qkernel
def first_shot_only() -> qmc.Bit:
    q = qmc.qubit("q")
    if qmc.get_current_shot() == 0:
        q = qmc.x(q)
    return qmc.measure(q)


shot_program = HugrTranspiler().transpile(first_shot_only)
shot_result = shot_program.sample(HugrExecutor("selene"), shots=4).result()
shot_counts = dict(shot_result.results)

assert shot_counts == {True: 1, False: 3}
print(shot_counts)

# %% [markdown]
# To run this example on H2, reuse the H2 `executor` from the previous section. The following code submits another hardware job and may incur cost. The ideal counts are the same, although hardware noise can change the observed counts.
#
# ```python
# shot_job = shot_program.sample(executor, shots=4)
# print(dict(shot_job.result().results))
# ```

# %% [markdown]
# ## Summary
#
# - Check the circuit locally with `HugrExecutor("selene")`.
# - Submit the same program to H2 with `HugrExecutor("nexus")` and retrieve its measurement counts with `job.result()`.
# - `qmc.get_current_shot()` uses 0-origin on both Selene and H2.
