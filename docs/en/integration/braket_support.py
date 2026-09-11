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
# # Amazon Braket Support
#
# This page transpiles Qamomile quantum kernels to native Amazon Braket
# circuits, then runs sampling and expectation-value evaluation with the local
# simulator. The same executor boundary can accept an AWS device when remote
# execution is needed.

# %%
# Install the latest Qamomile with Amazon Braket support.
# # !pip install "qamomile[braket]"

# %%
import math

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.braket import BraketExecutionOptions, BraketTranspiler

# %% [markdown]
# ## Build with Qamomile
#
# Runtime parameters remain symbolic in the emitted Braket circuit. The
# executor sends their values through Braket's native `inputs` API, preserving
# provider-side compilation and parameter handling.

# %%
@qmc.qkernel
def parameterized_bell(theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare and measure a parameterized Bell-like state.

    Args:
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Bit]: Two measured output bits.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.ry(q[0], theta)
    q[0], q[1] = qmc.cx(q[0], q[1])
    return qmc.measure(q)


@qmc.qkernel
def plus_expectation(observable: qmc.Observable) -> qmc.Float:
    """Prepare a plus state and evaluate an observable.

    Args:
        observable (qmc.Observable): Observable to evaluate.

    Returns:
        qmc.Float: Observable expectation value.
    """
    q = qmc.qubit_array(1, "q")
    q[0] = qmc.h(q[0])
    return qmc.expval(q, observable)

# %% [markdown]
# ## Transpile to Braket
#
# `BraketTranspiler` emits a native `braket.circuits.Circuit`. Terminal
# measurements stay in Qamomile's result mapping, allowing the executor to
# normalize Braket's measured-qubit order consistently.

# %%
transpiler = BraketTranspiler()
executable = transpiler.transpile(parameterized_bell, parameters=["theta"])
braket_circuit = executable.quantum_circuit

assert type(braket_circuit).__module__.startswith("braket.")
assert {str(parameter) for parameter in braket_circuit.parameters} == {"theta"}
print(braket_circuit)

# %% [markdown]
# ## Run locally
#
# With no device argument, the executor creates Braket's `LocalSimulator`.

# %%
executor = transpiler.executor()
sample_job = executable.sample(
    executor,
    shots=128,
    bindings={"theta": math.pi},
)
sample = sample_job.result()

assert sample.results == [((1, 1), 128)]
print(sample.results)

# %%
energy_program = transpiler.transpile(
    plus_expectation,
    bindings={"observable": qm_o.X(0)},
)
energy = energy_program.run(executor).result()

assert abs(energy - 1.0) < 1e-10
print("<X> =", energy)

# %% [markdown]
# ## Use an AWS device
#
# Pass an `AwsDevice` to `transpiler.executor(device)` to use the same job API
# remotely. Submission returns before result retrieval, and the public job
# exposes normalized status, cancellation, native task access, and task ARN
# references:
#
# ```python
# from braket.aws import AwsDevice
#
# device = AwsDevice("your-device-arn")
# options = BraketExecutionOptions(
#     s3_destination_folder=("your-bucket", "qamomile-results"),
#     poll_timeout_seconds=600,
#     poll_interval_seconds=2,
#     batch_max_retries=0,
# )
# aws_executor = transpiler.executor(device, options=options)
# job = executable.sample(aws_executor, shots=1_000, bindings={"theta": 0.4})
# print(job.status(), job.raw_status())
# references = job.references()
#
# # In a later process, recreate the executor and restore raw task results.
# restored = aws_executor.restore(references[0])
# counts = restored.result()
#
# # Cancellation is best effort and does not discard the task reference.
# job.cancel()
# ```
#
# Batch resubmission is disabled by default because retries can create
# additional billed tasks. Set `batch_max_retries` only when that behavior is
# intentional. `poll_timeout_seconds` sets the default local result wait as well
# as the provider polling limit. An explicit `result(timeout=...)` overrides
# that default. A timeout does not cancel a remote task;
# `await job.result_async()` provides an async waiting path.
#
# Remote tasks require AWS credentials, an S3 destination, and may incur cost,
# so this page executes only the local simulator path.

# %% [markdown]
# ## Limitations
#
# The Braket engine currently targets static gate-model circuits. It does not
# support measurement-dependent `if` or `while` control flow, mid-circuit
# reset, or algorithms that require those operations. This includes the
# reset-dependent paths in modular arithmetic and Shor workflows. Refactor
# such programs into static circuits or choose an engine with the required
# dynamic-circuit primitives.

# %% [markdown]
# ## Summary
#
# - `BraketTranspiler` emits native Braket circuits and free parameters.
# - `BraketExecutor` submits sampling and Hamiltonian expectation workflows as
#   native tasks without forcing immediate result retrieval.
# - Jobs expose status, cancellation, native tasks, and restorable AWS task
#   references; batch retries are explicit and bounded.
# - Injecting an `AwsDevice` switches execution targets without recompiling the
#   Qamomile kernel.
