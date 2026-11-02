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
# Compile a Qamomile Bell-state circuit to HUGR, an intermediate representation for quantum programs, and check it locally with Selene.
#
# Install Qamomile with the `hugr` extra.

# %%
# Install Qamomile with HUGR support.
# # !pip install "qamomile[hugr]"

# %%
import qamomile as qm
from qamomile.hugr import Hugr, Selene

# %% [markdown]
# ## Build a circuit with Qamomile
#
# Prepare two entangled qubits and measure them. The return value contains the two measurement bits.


# %%
@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


program = qm.specialize(bell)
artifact = qm.compile(program, to=Hugr())

# %% [markdown]
# `qm.specialize(bell)` validates the program before choosing a target. `qm.compile(program, to=Hugr())` creates the compiled HUGR program used in the next step.
#
# ## Check the circuit locally
#
# Run 100 shots with Selene. A shot is one repetition of the quantum program. In this noiseless simulation, the two measured bits always agree. The counts use tuples of `0` and `1` for the measurement results.

# %%
shots = 100
record = qm.run(artifact, on=Selene(), shots=shots, seed=7)
local_result = record.result
counts = dict(local_result.counts)

assert local_result.shots == shots
assert sum(counts.values()) == shots
assert set(counts) <= {(0, 0), (1, 1)}
print(counts)

# %% [markdown]
# ## Summary
#
# - Compile the circuit to HUGR with `qm.compile(program, to=Hugr())`.
# - Check the circuit locally with `qm.run(artifact, on=Selene(), shots=100)` and read the measurement counts from `record.result.counts`.
