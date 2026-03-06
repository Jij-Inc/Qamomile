# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---

# %% [markdown]
# # Your First Quantum Kernel
#
# This chapter takes you from zero to a running quantum program.
# By the end, you will understand every line and every concept needed
# to write, visualize, and execute a Qamomile kernel.
#
# ## What is Qamomile?
#
# Qamomile is a quantum circuit SDK that lets you write quantum programs
# in Python, then run them on any supported backend (Qiskit, QuriParts, and more in plan).
# It uses a **typed, symbolic** approach: you write a Python function
# decorated with `@qkernel`, and Qamomile traces it into an intermediate
# representation that can be analyzed, visualized, and compiled.
#
# The core workflow is a simple pipeline:
#
# ```
# @qkernel define  →  draw() / estimate_resources()  →  transpile()  →  sample() / run()  →  .result()
# ```
#
# - **Define**: Write a kernel function with type annotations.
# - **Inspect**: Visualize the circuit with `draw()`, or estimate costs with `estimate_resources()`.
# - **Transpile**: Compile the kernel into a backend-specific executable.
# - **Execute**: Run it with `sample()` (for measured bits) or `run()` (for expectation values).
# - **Read results**: Call `.result()` to get the output.
#
# You do not need every step every time — use what fits your task.

# %% [markdown]
# ## Installation
#
# For normal use:
#
# ```bash
# pip install qamomile
# ```
#
# For development in the repository:
#
# ```bash
# uv sync
# ```
#
# In this tutorial we use Qiskit as the concrete backend.
# QuriParts is also supported, and backend options will continue to grow.

# %%
import math

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## First Kernel: The Biased Coin
#
# A **kernel** is a Python function decorated with `@qmc.qkernel`.
# It describes a quantum circuit using typed handles and gate operations.
#
# Let's build the simplest possible example: a single qubit rotated by
# an angle `theta`, then measured. Depending on `theta`, the qubit is
# biased toward `0` or `1` — like a biased coin.


# %%
@qmc.qkernel
def biased_coin(theta: qmc.Float) -> qmc.Bit:
    # Create a single qubit handle named "q"
    q = qmc.qubit(name="q")

    # Apply an RY rotation — this biases the qubit
    q = qmc.ry(q, theta)

    # Measure and return the result as a classical Bit
    return qmc.measure(q)


# %% [markdown]
# Key things to notice:
#
# - **Type annotations are required**: `theta: qmc.Float` says theta is a
#   floating-point parameter. The return type `qmc.Bit` says this kernel
#   produces one classical bit.
# - **`qmc.qubit(name="q")`** creates a qubit handle. The `name` appears
#   in circuit diagrams.
# - **`q = qmc.ry(q, theta)`** applies the RY gate and **reassigns `q`**.
#   This reassignment is important — we will explain why shortly.
# - **`qmc.measure(q)`** measures the qubit state and returns a `Bit`.

# %% [markdown]
# ## Inspect Before Running
#
# Before executing, you can inspect your kernel. `draw()` shows the circuit diagram:
#
# > **Note**: `draw()` visualizes the circuit at Qamomile's IR level.
# > When transpiling to a backend (e.g., Qiskit), the backend may decompose
# > or optimize gates, so the actual executed circuit can differ from what
# > `draw()` shows. Use `to_circuit()` to see the backend-native circuit.

# %%
biased_coin.draw(theta=0.6)

# %% [markdown]
# You can also check the cost of a kernel before running it.
# `estimate_resources()` reports qubit count and gate counts:

# %%
est = biased_coin.estimate_resources()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %% [markdown]
# For this simple kernel the numbers are concrete, but for parameterized
# kernels they become symbolic SymPy expressions — we will explore this
# in detail in [Tutorial 02](02_parameterized_kernels.ipynb).

# %% [markdown]
# ## The Execution Pipeline
#
# Now let's actually run this kernel. The process has three steps:
#
# 1. **Transpile**: Compile the kernel into a backend-executable form.
# 2. **Execute**: Call `sample()` to run it with specific parameter values.
# 3. **Read results**: Call `.result()` on the returned Job.
#
# Here is the code, then we explain each part:

# %%
# Step 1: Transpile
# parameters=["theta"] tells the transpiler: "theta will be provided later,
# keep it as a sweepable parameter in the compiled circuit."
exe = transpiler.transpile(biased_coin, parameters=["theta"])

# Step 2: Execute
# bindings={"theta": ...} provides the concrete value for theta.
# shots=256 means we run the circuit 256 times.
job = exe.sample(
    transpiler.executor(),
    shots=256,
    bindings={"theta": math.pi / 4},
)

# Step 3: Read results
# .result() blocks until the job completes and returns a SampleResult.
# The default executor uses a local simulator, but you can plug in
# your own custom executor (e.g., for real hardware or cloud services).
result = job.result()

print("sample results:", result.results)

# %% [markdown]
# Let's unpack the three concepts:
#
# - **`parameters=["theta"]`** at transpile time declares which kernel inputs
#   remain as tunable knobs in the compiled program. Inputs *not* listed here
#   must be provided via `bindings` at transpile time (we will see this in Tutorial 02).
#
# - **`bindings={"theta": math.pi / 4}`** at execution time fills in the
#   concrete value for the parameter.
#
# - **`.result()`**: `sample()` returns a **Job** object, not the result
#   directly. Calling `.result()` waits for the job to finish and returns a
#   `SampleResult`. The default executor uses a local simulator, but the
#   Job pattern lets you swap in a custom executor (e.g., real hardware or
#   a cloud service) without changing your kernel code.

# %% [markdown]
# ## Reading `SampleResult`
#
# `result.results` is a `list[tuple[T, int]]` where:
#
# - `T` is the measured output type (here, `int` — `0` or `1` for a `Bit`)
# - `int` is the count: how many times that outcome appeared
#
# For example, `[(0, 150), (1, 106)]` means outcome `0` appeared 150 times
# and outcome `1` appeared 106 times out of 256 shots.

# %%
for value, count in result.results:
    print(f"  outcome={value}, count={count}")

# %% [markdown]
# `SampleResult` also provides convenience methods:

# %%
# Most common outcome
print("most common:", result.most_common(1))

# Probability distribution
print("probabilities:", result.probabilities())

# %% [markdown]
# ## Inspecting the Backend Circuit
#
# `to_circuit()` compiles a kernel with **all** parameters bound and returns
# the backend-native circuit (e.g., a Qiskit `QuantumCircuit`).
# This is useful for debugging — you can see exactly what the backend receives.

# %%
qiskit_circuit = transpiler.to_circuit(
    biased_coin,
    bindings={"theta": math.pi / 4},
)
print(qiskit_circuit)

# %% [markdown]
# ## Multi-Qubit Example
#
# Let's use more than one qubit. This example introduces two new things:
#
# - **`qubit_array(n)`** to allocate multiple qubits at once
# - **`cx()`** (the CNOT gate), a two-qubit gate that returns **both** handles


# %%
@qmc.qkernel
def two_qubit_demo() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(2, name="q")

    q[0] = qmc.h(q[0])  # Apply H to q[0]
    q[0], q[1] = qmc.cx(q[0], q[1])  # Apply CNOT on q[0], q[1]

    return qmc.measure(q)


# %%
two_qubit_demo.draw()

# %%
demo_result = (
    transpiler.transpile(two_qubit_demo)
    .sample(
        transpiler.executor(),
        shots=256,
    )
    .result()
)

for outcome, count in demo_result.results:
    print(f"  outcome={outcome}, count={count}")

# %% [markdown]
# Notice two patterns here:
#
# - **`qubit_array(2)`** creates multiple qubits in one call. You access
#   them by index: `q[0]`, `q[1]`. In Tutorial 02 we will make the size
#   symbolic with `qubit_array(n)`.
# - **Two-qubit gates return both handles**: `q[0], q[1] = qmc.cx(q[0], q[1])`.
#   Both sides must be reassigned.
#
# This brings us to an important rule.

# %% [markdown]
# ## The Affine Rule
#
# In Qamomile, quantum handles are **affine**: once a gate consumes a handle,
# you **must** use the returned handle for all subsequent operations.
# Think of it as a "move" — the old handle is invalidated.
#
# - Single-qubit gate: `q = qmc.h(q)` — reassign the same variable.
# - Two-qubit gate: `q0, q1 = qmc.cx(q0, q1)` — reassign both variables.
#
# ### Why affine, not linear?
#
# In quantum computing, if you use a qubit for a temporary computation and
# leave it entangled with the rest of the system without cleaning it up,
# subsequent operations on other qubits can be affected in unexpected ways.
# Strictly speaking, a **linear** type system (where every handle must be
# used exactly once) would be the safest model — it would force you to
# always "uncompute" (reverse) temporary qubits before discarding them.
#
# However, enforcing linear types in Python would make simple programs
# awkward to write. Qamomile chooses **affine** types instead: a handle
# must be used **at most** once, but you are allowed to drop it. This
# keeps the API Pythonic — you can write natural code without ceremony.
#
# > **Trade-off**: if you allocate a temporary qubit, entangle it with your
# > main register, and then forget about it, the physics still applies —
# > that leftover qubit will pollute your results. The compiler won't
# > catch this for you. So remember: **if you entangle a temporary qubit,
# > uncompute it before you stop using it.**
#
# If you forget to reassign, you get an error. Here is what happens:

# %%
try:

    @qmc.qkernel
    def bad_rebind() -> qmc.Bit:
        q = qmc.qubit(name="q")
        qmc.h(q)  # Oops: we consumed q but didn't capture the result
        q = qmc.x(q)  # This uses the stale (already consumed) handle
        return qmc.measure(q)

    bad_rebind.draw()
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# %% [markdown]
# The fix is simple: always write `q = qmc.h(q)`, not just `qmc.h(q)`.

# %% [markdown]
# ## Summary
#
# You now know how to:
#
# - Define a kernel with `@qmc.qkernel`
# - Create qubits, apply gates, and measure
# - Visualize with `draw()`
# - Execute with `transpile()` → `sample()` → `.result()`
# - Read `SampleResult` outcomes
# - Inspect the backend circuit with `to_circuit()`
# - Follow the affine rule (`q = qmc.gate(q)`)
# - Estimate resources with `estimate_resources()`
#
# ## Supported Backends
#
# Qamomile compiles the same `@qkernel` to different quantum frameworks.
# Current backend support:
#
# | Backend | Status | Notes |
# |---------|--------|-------|
# | **Qiskit** | Supported | Full gate set, control flow, observables |
# | **QuriParts** | Supported | Full gate set, observables |
# | **CUDA-Q** | Coming soon | GPU-accelerated simulation |
#
# > **Important**: Not every kernel feature is available on every backend.
# > For example, `if` branching inside a kernel is supported by Qiskit but
# > may not yet be supported by other backends. If a feature is not available
# > for your chosen backend, you will get a clear error at transpile time.
# > When in doubt, start with Qiskit for development and switch backends
# > when you are ready to deploy.
#
# ## Next Chapters
#
# 1. [Parameterized Kernels](02_parameterized_kernels.ipynb) — structure vs runtime parameters, the bind/sweep pattern
# 2. [Resource Estimation](03_resource_estimation.ipynb) — symbolic cost analysis, gate breakdowns, comparing designs
# 3. [Execution Models](04_execution_models.ipynb) — `sample()` vs `run()`, observables, bit ordering
# 4. [Classical Flow Patterns](05_classical_flow_patterns.ipynb) — loops, sparse data, branching
# 5. [Reuse Patterns](06_reuse_patterns.ipynb) — helper kernels, composite gates, stubs
# 6. [Debugging and Backend](07_debugging_and_backend.ipynb) — debug checklist, error messages, quick reference
