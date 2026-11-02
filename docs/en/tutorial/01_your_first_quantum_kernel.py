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
# tags: [tutorial]
# ---
#
# # Your First Quantum Kernel
#
# This chapter introduces the basic workflow for defining and running your first quantum kernel in Qamomile. It does not go into the fundamentals of quantum computing or the details of quantum algorithms.
#
# ## What is Qamomile?
#
# Qamomile is a source-compiled functional quantum programming language embedded in Python.
#
# You define a quantum kernel by decorating a Python function with `@qm.qkernel`. Type annotations are required for a quantum kernel’s arguments and return value. Qamomile uses those annotations to check type consistency before execution.
#
# Qamomile reads the kernel’s source code, validates it, and converts it into an intermediate representation that can be analyzed, saved, and compiled for supported quantum SDKs such as Qiskit, QURI Parts, and CUDA-Q.
#
# Qamomile can also handle expressions symbolically, allowing them to contain parameters whose values are not yet specified.
#
# The basic workflow for the examples in this chapter is:
#
# - **Define:** Write a type-annotated Python function decorated with `@qm.qkernel`.
# - **Run:** Use `record = qm.run(...)` to compile and execute the kernel and receive a `RunRecord` containing the execution result.
# - **Read the result:** Use `record.result` to retrieve the result of the run.
#
# The same quantum kernel can be used both to run a computation and to estimate requirements such as qubit counts and logical gate counts before execution. Qamomile also provides explicit compilation and tools for examining a program before compilation. We will explore these after completing the basic workflow.

# %% [markdown]
# ## Installation
#
# This tutorial uses Qiskit Aer to simulate quantum computations. No quantum hardware is needed.
#
# Remove the leading `#` from the following line and run the cell to install Qamomile together with Qiskit and Aer:

# %%
# # !pip install "qamomile[qiskit]"

# %%
import math
import os

import qamomile as qm
from qamomile.qiskit import Aer, load_circuit

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
sample_shots = 1 if docs_test_mode else 256

# %% [markdown]
# ## Basic Workflow
#
# ### First QKernel: The Biased Coin
#
# A **QKernel** in Qamomile is a Python function decorated with `@qm.qkernel`. It describes a quantum computation using typed values and gate operations.
#
# Let's start with a simple example: rotate a single qubit by an angle `theta`, then measure it. By changing `theta`, we can change the probabilities of measuring `0` or `1`, much like adjusting the bias of a coin.


# %%
@qm.qkernel
def biased_coin(
    theta: qm.Param[qm.Real],
) -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    # Create a single owned quantum state.
    state = qm.qubits(1)

    # Apply an RY rotation. This biases the measured result.
    state = qm.ry(state, theta)

    # Measure and return one classical bit.
    return qm.measure(state)


# %% [markdown]
# :::{note}
# Some Python type checkers do not recognize `qm.Shot[qm.Bits[1]]` as a valid type annotation. The `# type: ignore[valid-type]` comment suppresses that message on this line without changing how the kernel runs or disabling Qamomile's own type checking.
# :::
#
# **Key things to notice:**
#
# - **Type annotations are required.** In `theta: qm.Param[qm.Real]`, `qm.Real` specifies the value’s type (a floating-point number), while `qm.Param[...]` indicates that its value is supplied at execution time. This lets you choose a rotation angle `theta` for each run. The return type `qm.Shot[qm.Bits[1]]` indicates that this kernel returns a one-bit value from each execution (shot). In this example, that value is the measurement result returned by `qm.measure(state)`.
# - `qm.qubits(1)` creates a one-qubit quantum state.
# - `state = qm.ry(state, theta)` applies an RY rotation and receives the updated state under the same variable name, `state`. Subsequent operations use this updated value; the old value passed to the gate cannot be reused.
# - `qm.measure(state)` measures the qubit and returns a classical bit.

# %% [markdown]
# ### Run the Kernel
#
# In this example, `parameters` specifies the rotation angle `theta`, and `shots` specifies how many times to repeat the computation.
#
# We can pass `biased_coin` directly to `qm.run(...)`. Qamomile handles the necessary validation and compilation internally.
#
# Here, `Aer()` creates an endpoint for the Qiskit Aer simulator. An endpoint specifies where the computation will run.

# %% [markdown]
# :::{note}
# A kernel passed to `qm.run(...)` must have classical inputs and outputs, including any nested values. The caller receives classical data, such as measurement results, as the computation’s output. Within a kernel, calls to other quantum kernels can pass quantum states as arguments and return values.
#
# Classical computation and branching on measurement results within a single execution (shot) can be written inside a quantum kernel. Results from multiple shots are aggregated outside the kernel. [Tutorial 05](05_execution_models.ipynb) introduces `statistics` and `Estimator` for statistical analysis and expectation-value estimation.
# :::

# %%
# Choose an execution endpoint.
endpoint = Aer()

# Run the circuit sample_shots times.
record = qm.run(
    biased_coin,
    on=endpoint,
    parameters={"theta": math.pi / 4},
    shots=sample_shots,
    seed=7,
)

# Retrieve the SampleResult containing the measurements.
result = record.result

print("sample counts:", result.counts)
assert result.shots == sample_shots
assert sum(result.counts.values()) == sample_shots

# %% [markdown]
# Let's look at the key parts:
#
# - **`parameters={"theta": math.pi / 4}`** sets the rotation angle to π/4 radians for this run.
#
# - **`shots=sample_shots`** specifies how many times to repeat the computation. With `sample_shots` set to 256, this example produces 256 measurement results, each containing one classical bit.
#
# - **`seed=7`** sets the simulator's random seed, making repeated runs with the same settings reproducible.
#
# - **`record.result`** provides the execution result. `qm.run(...)` waits for execution to finish before returning a `RunRecord`. In this example, its result is a `SampleResult` containing measurement results in Qamomile’s data format.
#
# In this example, you can specify `theta` for each run. [Tutorial 02](02_parameterized_kernels.ipynb) explains the parameter types and when their values are supplied.

# %% [markdown]
# ### Reading `SampleResult`
#
# In the previous section, we assigned `record.result` to `result`. For this kernel, `result` is an instance of `SampleResult`, which holds measurement results in Qamomile’s data format. Its properties and methods let us read measurement counts, find the most common outcome, and estimate probabilities.
#
# Let's start with `result.counts`, a read-only mapping where:
#
# - Each key is a measured outcome, represented as a tuple of bits. Here, the one-bit outcomes are `(0,)` and `(1,)`.
# - Each value is the number of shots that produced that outcome.
#
# For example, `{(0,): 214, (1,): 42}` means that `0` was measured 214 times and `1` was measured 42 times, for a total of 256 shots.

# %%
for value, count in result.counts.items():
    print(f"  outcome={value}, count={count}")

# %% [markdown]
# You can also use `SampleResult` to summarize and interpret the results:
#
# - **`result.most_likely`** returns the most frequently observed outcome.
# - **`result.probability((1,))`** estimates the probability of measuring `1` by dividing its count by the total number of shots.
# - **`result.values`** provides a read-only mapping from decoded outcomes to their counts, using the kernel's declared return type. For this kernel, it has the same contents as `result.counts`.

# %%
# Most common outcome, with deterministic tie breaking.
print("most likely:", result.most_likely)

# Estimated probability of measuring 1.
print("P(1):", result.probability((1,)))

# Counts decoded according to the kernel's return type.
print("typed values:", result.values)

# %% [markdown]
# ## Additional Features
#
# The basic workflow is now complete. The following sections use the same kernel to show how to compile it separately, examine it before compilation, and inspect the resulting circuit.
#
# ### Compile Explicitly
#
# `qm.compile(...)` accepts a quantum kernel and returns a compiled program that can be reused for multiple runs. In the example below, we pass `biased_coin` and store the returned `CompiledProgram` instance in `artifact`. This `artifact` can be passed to `qm.run(...)`.
#
# `endpoint.target` specifies the format and settings required by the endpoint. Here, it selects compilation for Qiskit Aer.

# %%
artifact = qm.compile(biased_coin, to=endpoint.target)
compiled_record = qm.run(
    artifact,
    on=endpoint,
    parameters={"theta": math.pi / 4},
    shots=sample_shots,
    seed=7,
)
assert compiled_record.result.counts == result.counts

# %% [markdown]
# The compiled `artifact` retains `theta` as a parameter. You can reuse it with a different rotation angle by changing `parameters` in `qm.run(...)`.
#
# If another endpoint requires a different compilation target, compile the kernel for that target before running it.

# %% [markdown]
# ### Before You Compile
#
# Before compiling a kernel, you can validate it, look at the program it describes, and estimate its resource requirements. You can do this without choosing a backend or running the quantum computation.
#
# #### Validate the Kernel
#
# `qm.specialize(...)` applies the supplied `Const` inputs and creates a verified `Program`, checking types and quantum-state usage. This is useful when you want to examine the program before compilation.

# %%
program = qm.specialize(biased_coin)

# %% [markdown]
# Here, `biased_coin` has no `Const` inputs, so no additional values are needed. A separate call to `qm.specialize(...)` is optional: when a kernel is passed directly to `qm.compile(...)`, specialization and validation happen internally. You can also compile the resulting `Program` by writing `qm.compile(program, to=endpoint.target)`.
#
# #### Look at the Program
#
# `program.inspect()` shows the program’s operations and types as text. `qm.resources.logical(program)` estimates requirements such as qubit counts and logical gate counts. We will focus on resource estimation in the example below.

# %% [markdown]
# #### Estimate Resources
#
# `qm.resources.logical(program)` estimates the program’s logical resource requirements. Here, we check the maximum number of qubits in use at once and the number of logical gates:

# %%
study = qm.resources.logical(program)
peak_qubits = study.metric(qm.resources.LOGICAL_PEAK_QUBITS)
logical_gates = study.metric(qm.resources.LOGICAL_GATES)

print("peak logical qubits:", peak_qubits)
assert peak_qubits.numeric_enclosure() == (1, 1)
print("logical gates:", logical_gates)
assert logical_gates.numeric_enclosure() == (1, 1)

# %% [markdown]
# This kernel uses one qubit and one logical gate: the RY rotation. These are logical resource estimates; they do not include the physical resources needed to run on a particular quantum computer. See [Resource Estimation](04_resource_estimation.ipynb) for more detail.

# %% [markdown]
# ### Inspecting the Compiled Circuit
#
# You can also look at the circuit produced by compilation. `load_circuit(artifact)`, imported from `qamomile.qiskit`, returns a copy of the Qiskit circuit stored in the compiled artifact.
#
# In the code below, `qiskit_circuit` is an instance of Qiskit's `QuantumCircuit`. Printing it shows the circuit's gates and measurements, which is useful for checking its structure and debugging.

# %%
qiskit_circuit = load_circuit(artifact)
print(qiskit_circuit)
assert qiskit_circuit.num_qubits == 1
assert qiskit_circuit.num_clbits == 1

# %% [markdown]
# In the printed circuit, `p0` represents the kernel's `theta` parameter. The compiled circuit keeps this parameter symbolic even after a run; `qm.run(...)` supplies its value for each execution.

# %% [markdown]
# ### Multi-Qubit Example
#
# Let's extend the example to two qubits. We will apply a Hadamard gate to the first qubit, followed by a CNOT gate, then measure both qubits.


# %%
@qm.qkernel
def two_qubit_demo() -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


# %% [markdown]
# Here, `qm.qubits(2)` creates a state containing two qubits. Qubit indices start at zero:
#
# - `target=0` selects the first qubit for the Hadamard gate.
# - `control=0, target=1` selects the first qubit as the control and the second as the target of the CNOT gate.
#
# `qm.measure(state)` measures both qubits. The return type, `qm.Shot[qm.Bits[2]]`, indicates that each shot produces two classical bits.
#
# For this kernel, we can pass `two_qubit_demo` directly to `qm.run(...)`. It handles validation, compilation, and execution using the same Aer endpoint as before.

# %%
demo_record = qm.run(
    two_qubit_demo,
    on=endpoint,
    shots=sample_shots,
    seed=11,
)
demo_result = demo_record.result

for outcome, count in demo_result.counts.items():
    print(f"  outcome={outcome}, count={count}")
assert demo_result.shots == sample_shots
assert sum(demo_result.counts.values()) == sample_shots
# Bell state |Phi+>: only (0,0) and (1,1) outcomes appear.
assert set(demo_result.counts) <= {(0, 0), (1, 1)}

# %% [markdown]
# The Bell state prepared by this circuit ideally produces only `(0, 0)` and `(1, 1)` when measured, each with probability 1/2. The noiseless simulator used here samples this probability distribution.
#
# With a finite number of shots, the two outcomes need not occur equally often.
#
# This example uses a fixed number of qubits. In Tutorial 02, we will make that number configurable with a `Const[Int]` input.
#
# Notice that both the Hadamard gate and the CNOT gate return an updated state, which we receive under the same name, `state`. The next section explains why subsequent operations must use the updated state.

# %% [markdown]
# ### The Linear Type System
#
# Qamomile uses a **linear type system** to track quantum states. A gate consumes the state value it receives and returns an updated state value. Once consumed, the old value cannot be used again.
#
# In our examples, we receive the updated state under the same variable name:
#
# - Single-qubit gate: `state = qm.h(state)`
# - Two-qubit gate: `state = qm.cx(state, control=0, target=1)`
#
# You can also assign the returned state to a different variable. What matters is that subsequent operations use the updated value.
#
# #### Why linear?
#
# Qamomile’s linear type system tracks how quantum state values are passed between operations and detects reuse of consumed values before execution. This prevents code from treating the old and updated states as independently usable values. It also checks that quantum states that are no longer used are handled appropriately. In the examples above, `qm.measure(state)` consumes the final quantum state and returns classical data.
#
# Temporary qubits need care: they may remain entangled with the rest of the computation, affecting interference and final measurement probabilities. **Uncomputation** reverses a temporary computation to remove unwanted entanglement. You can explicitly apply the inverse of a unitary helper with `qm.adjoint(...)`.
#
# Qamomile also supports **automatic uncomputation**. If a state has been assigned to a variable but is no longer used, the compiler attempts to uncompute and release it. If it cannot verify the necessary cleanup, it reports an error.
#
# If you intentionally want to discard a state, use `qm.discard(state)`. This explicitly consumes the state without uncomputing it: discarding a qubit does not undo the operations that entangled it with other qubits.
#
# For example, the following kernel ignores the state returned by the Hadamard gate:


# %%
@qm.qkernel
def bad_rebind() -> qm.Shot[qm.Bits[1]]:  # type: ignore[valid-type]
    state = qm.qubits(1)
    qm.h(state)  # Oops: the returned state is left unused.
    state = qm.x(state)  # This still refers to the old state.
    return qm.measure(state)


# %% [markdown]
# We can check this kernel with `qm.specialize(...)` without running it:

# %%
try:
    qm.specialize(bad_rebind)
except qm.FrontendError as error:
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")
    assert "unused expression has non-Unit type" in str(error)

# %% [markdown]
# The `unused expression` error refers to `qm.h(state)`, whose return value is ignored. Change this line to `state = qm.h(state)` so that the following `qm.x(state)` call uses the updated state.

# %% [markdown]
# ## Supported Quantum SDKs
#
# Qamomile supports several quantum SDKs and compilation targets. The available options are:
#
# | SDK / target | Status | Notes |
# |---|---|---|
# | **Qiskit** | Supported | Qiskit circuits, Aer simulation, and execution on compatible Qiskit backends |
# | **QURI Parts** | Supported | QURI Parts circuits, Qulacs simulation, and execution through QURI Parts samplers |
# | **CUDA-Q** | Supported | CUDA-Q kernels and execution on available CUDA-Q simulation targets |
# | **Cirq** | Supported | Cirq circuits, local simulation, and execution through compatible samplers and Google Quantum Engine |
# | **Amazon Braket** | Supported | Local simulation and execution on compatible Amazon Braket devices |
# | **Quration (PyQret)** | Supported | Local simulation and resource estimation for fault-tolerant quantum computing |
# | **HUGR** | Supported | HUGR compilation, local Selene simulation, and Helios execution through Quantinuum Nexus |
# | **qBraid** | Supported via Qiskit | Execution of compiled Qiskit circuits on compatible qBraid devices |
# | **QIR** | Export only | QIR 1.0 Base Profile output for execution by an external QIR-compatible system |
#
# :::{note}
# Available features depend on the chosen backend. For example, using a measurement result to decide which gate to apply next requires support for measurement-dependent control flow. Conditions determined by `Const` inputs are resolved at compile time.
# :::
#
# ### CUDA-Q Platform Support
#
# CUDA-Q supports the following environments:
#
# | Environment | Status | Notes |
# |---|---|---|
# | **Linux** | Supported | CPU simulation; GPU acceleration requires a supported NVIDIA GPU and compatible CUDA environment |
# | **macOS ARM64 (Apple silicon)** | Supported | CPU-only local simulation |
# | **Windows via WSL2** | Supported | Install and run inside the WSL2 Linux environment |
# | **Native Windows** | Unsupported | Use WSL2 |
# | **macOS x86_64 (Intel)** | Unsupported | macOS support requires Apple silicon |
#
# Platform requirements follow the [CUDA-Q installation documentation](https://nvidia.github.io/cuda-quantum/0.14.0/using/install/local_installation.html).
#
# ## Summary
#
# The basic workflow is to define a quantum kernel with `@qm.qkernel` and type annotations, execute it with `qm.run(...)`, and read its result through `record.result`. For the sampling examples here, `SampleResult` provides counts and estimated probabilities.
#
# You also learned how to:
#
# - Compile a kernel separately and reuse the compiled program with different parameter values.
# - Use optional specialization, inspection, and logical resource estimation before compilation.
# - Examine the compiled Qiskit circuit with `load_circuit(...)`.
# - Work with multiple qubits and use updated quantum states correctly.
# - Distinguish uncomputation from explicit discard.
#
# ## Next Chapters
#
# The following chapters build on the workflow introduced here:
#
# 2. [Parameterized Quantum Kernels](02_parameterized_kernels.ipynb) — Distinguish circuit structure from runtime parameters, and reuse a compiled program with different parameter values.
# 3. [Coherent Control with qm.control](03_controlled_gates.ipynb) — Apply controlled versions of unitary kernels using one or more control qubits.
# 4. [Resource Estimation](04_resource_estimation.ipynb) — Estimate logical resources such as qubit counts, gate counts, and depth, and understand symbolic estimates and their limitations.
# 5. [Execution, Statistics, and Observable Estimation](05_execution_models.ipynb) — Work with individual shots and repeated samples, summarize measurement results, and estimate observable expectation values.
# 6. [Classical Control Flow Patterns](06_classical_flow_patterns.ipynb) — Use loops, sparse interaction data, and conditional branches, including branches based on measurement results.
# 7. [Reuse Patterns: QKernel Composition and Composite Gates](07_reuse_patterns.ipynb) — Compose helper kernels and composite gates, pass kernels as arguments, and declare functions without implementation bodies for resource estimation.
#
# For operations on part of a qubit array, see [Working with Qubit Arrays](../usage/qubit_arrays.ipynb).
