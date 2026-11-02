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
# # Amazon Braket Support
#
# Qamomile keeps Amazon Braket at the realization boundary. Users write one
# typed Qamomile program; the `Braket` target emits portable OpenQASM 3, and
# `BraketLocal` executes it in an isolated local-simulator worker. A
# `BraketDevice` endpoint can submit compatible artifacts to a managed device
# without placing AWS sessions or device objects inside QCore.
#
# This page uses measurement feedback because it distinguishes Braket's dynamic
# OpenQASM route from the static logical-circuit integrations. The source-level
# meaning, linear ownership, resources, and result shape still belong to Rust.

# %% [markdown]
# ## Installation
#
# ```bash
# pip install "qamomile[braket]"
# ```

# %%
# Install the latest Qamomile and its Amazon Braket integration through pip!
# # !pip install "qamomile[braket]"

# %%
import math

import qamomile as qm

from qamomile.persistence import deserialize, serialize
from qamomile.braket import Braket, BraketLocal, load_openqasm

# %% [markdown]
# ## Build with Qamomile
#
# Runtime parameters remain symbolic in the emitted Braket program. The
# endpoint supplies their values at invocation time, so provider-side
# compilation and parameter handling remain available while one immutable
# artifact is reused.

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
# ## Transpile to Braket
#
# The `Braket` target emits portable OpenQASM 3. Terminal measurements stay in
# Qamomile's result ABI, allowing the endpoint to normalize Braket's measured
# qubit order consistently.

# %%
bell_program = qm.specialize(parameterized_bell)
bell_artifact = qm.compile(bell_program, to=Braket())
bell_source = load_openqasm(bell_artifact)

assert bell_artifact.format == "braket.openqasm3.v1"
assert tuple(slot.name for slot in bell_artifact.parameter_abi.slots) == ("theta",)
print(bell_source)
assert "input float[64] p0;" in bell_source

# %% [markdown]
# ## Run locally
#
# `BraketLocal` creates Braket's local simulator in an isolated worker. The
# same artifact can be invoked repeatedly with different `theta` values.

# %%
bell_result = qm.run(
    bell_artifact,
    on=BraketLocal(),
    parameters={"theta": math.pi},
    shots=32,
    seed=0,
).result

assert bell_result.counts == {(1, 1): 32}
print(bell_result.counts)

# %% [markdown]
# ## Author one typed feedback program
#
# `measure_z` returns a shot-time Boolean and the same linear quantum state.
# Ordinary Python `if` syntax then selects an operation on the second qubit.
# Both branches return exactly one owned state, and the final measurement has a
# public two-bit result even though execution performs three measurement events.

# %%
@qm.qkernel
def correlated_feedback() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Correlate two output bits through measurement feedback.

    Returns:
        qm.Shot[qm.Bits[2]]: Two equal bits in Qamomile source order.
    """

    state = qm.qubits(2)
    state = qm.h(state, target=0)
    flag, state = qm.measure_z(state, 0)
    if flag:
        state = qm.x(state, target=1)
    else:
        state = qm.z(state, target=1)
    return qm.measure(state)


program = correlated_feedback.specialize()
study = qm.resources.logical(program)

assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (2, 2)
assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (2, 2)

# %% [markdown]
# ## Compile once and retain the executable
#
# `qm.compile` returns a process-local `CompiledProgram` containing the Braket OpenQASM Program,
# its SDK lifetime owners, and compiler-derived ABI and resource facts. Ordinary
# execution does not export or restore it. The optional `serialize`/`deserialize`
# calls below demonstrate saving through `qamomile.persistence`; loading creates
# a new executable owner. Compare compilation identities, not Python objects.

# %%
target = Braket()
artifact = qm.compile(program, to=target)
restored = deserialize(serialize(artifact))

assert artifact.format == "braket.openqasm3.v1"
assert restored.execution_identity == artifact.execution_identity
assert artifact.execution_identity == qm.compile(program, to=target).execution_identity
assert artifact.result_abi.width == 2

# %% [markdown]
# `load_openqasm` is an explicit inspection helper, not an alternative compiler
# or execution path. The private feedback measurement occupies `c[0]`; the two
# final measurements occupy the remaining native bits. The result ABI retains
# only the declared two-bit value and owns its source ordering.

# %%
source = load_openqasm(restored)

print(source)

assert source.startswith("OPENQASM 3.0;")
assert "bit[3] c;" in source
assert "if (c[0])" in source
assert "c[1] = measure" in source
assert "c[2] = measure" in source

# %% [markdown]
# ## Execute through an isolated local simulator
#
# `BraketLocal` constructs the SDK simulator only inside a worker process. The
# endpoint executes the already compiled OpenQASM, normalizes Braket's native
# q0-first result once, and returns the common typed `SampleResult`. This local
# profile does not promise reproducible simulator seeding, so the portable seed
# is explicitly left at zero.

# %%
record = qm.run(
    restored,
    on=BraketLocal(),
    shots=32,
    seed=0,
)

print(record.result.counts)

assert record.artifact is restored
assert sum(record.result.counts.values()) == 32
assert set(record.result.counts) <= {(0, 0), (1, 1)}
assert all(left == right for left, right in record.result.counts)

# %% [markdown]
# ## Expectation values on the same local endpoint
#
# Sampling is not a special Braket-only API. A state-returning source can be
# evaluated through Qamomile's common expectation contract, which lowers the
# observable measurement work to the same OpenQASM target. For a plus state,
# the exact expectation of $X$ is one.

# %%
@qm.qkernel
def plus_state() -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Prepare a one-qubit plus state.

    Returns:
        qm.Q[qm.Bits[1]]: Owned plus state.
    """

    return qm.h(qm.qubits(1), target=0)


plus_x = qm.expectation(
    plus_state,
    qm.X(0),
    on=BraketLocal(),
    shots=8,
    seed=0,
)
print("<X> =", plus_x.value)
assert plus_x.value == 1.0

# %% [markdown]
# ## Route compatible programs to a managed device
#
# A live device remains endpoint-local. `BraketDevice.target` seals an immutable
# device capability and calibration snapshot into artifact fingerprinting, but
# the SDK object, credentials, and task never enter persistent program data.
# Managed devices do not promise the local dynamic-control capability or a
# portable seed contract. Qamomile therefore checks capabilities before
# submission and requires `seed=0` rather than silently weakening the request.
#
# ```python
# from braket.aws import AwsDevice
# from qamomile.braket import BraketDevice
#
# device = AwsDevice("your-device-arn")
# endpoint = BraketDevice(
#     device,
#     poll_timeout_seconds=600,
#     poll_interval_seconds=2,
# )
# job = qm.submit(your_static_program, on=endpoint, shots=100, seed=0)
# print(job.job_id, job.status)
# result = job.result()
# print(result.counts)
# job.cancel()  # best effort; the task reference remains available
# ```
#
# Remote submission needs configured AWS credentials and can incur cost, so it
# is intentionally not executed by this page. The same `run` and typed result
# contracts apply to local and managed endpoints.

# %% [markdown]
# ## Limitations
#
# The local OpenQASM simulator accepts the dynamic feedback demonstrated above.
# Managed devices advertise their own immutable capability snapshot; the
# default `BraketDevice` contract excludes dynamic branches and reset, so those
# programs fail capability checks before submission. Remote tasks also require
# AWS credentials, may incur cost, and do not promise a portable random seed.
# Keep device selection, polling limits, cancellation, and retry policy at the
# endpoint boundary rather than encoding them into the quantum program.

# %% [markdown]
# ## Summary
#
# - QCore remains the sole persistent meaning of the feedback program.
# - `Braket` emits deterministic OpenQASM 3 without storing SDK objects.
# - Shot-time control is lowered only at the Braket emission boundary.
# - `BraketLocal` isolates local simulation and normalizes native bit order.
# - `BraketDevice` keeps live devices, credentials, tasks, and capability
#   snapshots at the endpoint boundary.
# - The executable page uses at most two qubits, with 32 sampling shots and one
#   bounded eight-shot expectation evaluation.
