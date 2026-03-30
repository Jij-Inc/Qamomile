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
# # Qamomile v0.10.0
#
# Qamomile v0.10.0 is a ground-up rebuild of the circuit programming layer. The amount of changes from v0.9.0 is massive, so rather than listing what changed, this page describes **what v0.10.0 can do**.
#
# [Write what Qamomile v0.10.0 can do: write circuit with classical control flow, symbolical resource estimation with/without oracle, multi-quantum SDK transpilation.]
#
# See the [tutorials](../tutorial) for hands-on guides.
# ```
# pip install qamomile===0.10
# ````

# %% [markdown]
# ## Circuit Frontend: `@qkernel`
#
# The central API is the `@qkernel` decorator. You write a normal Python function with **type-annotated** arguments and return values, and Qamomile traces it into an intermediate representation (IR) that can be analyzed, visualized, and transpiled to any supported quantum SDK such as Qiskit. Although this IR is a key of Qamomile's architecture, you don't need to interact with it directly — just write Python code with `@qkernel` and let the transpiler handle the rest.

# %% [markdown]
# ### Rules of writing a `@qkernel`:
#
# - **Type hints are required.** Arguments and return types use Qamomile's symbolic types: `Qubit`, `Bit`, `Float`, `UInt`, `Vector[T]`, `Dict[K, V]`, `Tuple[...]`.
# - **Affine typing for qubits.** A `Qubit` handle must be reassigned after every gate (`q = qmc.h(q)`), ensuring the compiler can track qubit lifetimes.
# - **Return type determines execution mode.** Returning `Bit` / `Vector[Bit]` means `sample()` (shot-based); returning `Float` (via `expval`) means `run()` (expectation value).
#
# Available gates include H, X, Y, Z, S, T, RX, RY, RZ, RZZ, CX, CZ, CCX, CP, SWAP, and more. Measurement is done with `qmc.measure()`.
#
# See [your-first-quantum-kernel](../tutorial/Your First Quantum Kernel), [parametrized-quantum-kernels](../tutorial/Parametrized Quantum Kernels), [Execution Models: sample() vs run()](../tutorial/execution-models) and [Reuse Patterns: QKernel Composition and Composite Gates](../tutorial/reuse-patterns).

# %% [markdown]
# ### Classical Control Flow
#
# Qamomile supports classical control flow **inside** qkernels:
#
# - **`qmc.range(start, stop, step)`** (or **`qmc.range(stop)`**) — parameterized `for` loops over qubits.
# - **`qmc.items(dict)`** — iterate over sparse data such as graph edges or interaction maps, useful for QAOA-style circuits.
# - **`if` / `else`** on `Bit` — mid-circuit measurement-based branching.
# - **`while`** on `Bit` — runtime loops conditioned on measurement outcomes.
#
# These are not plain Python loops: the compiler traces them into IR nodes that each backend can handle (unrolling, native control flow, etc.).
#
# See [Classical Control Flow Patterns](../tutorial/classical-flow-patterns).

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def ghz_state(
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:  # Write type hint for the argument and return type
    q = qmc.qubit_array(
        n, name="q"
    )  # Allocate an array of n-qubits with the given size and name

    q[0] = qmc.h(q[0])
    for i in qmc.range(1, n):  # Iterate over qubits 1 to n-1 using qmc.range
        q[0], q[i] = qmc.cx(q[0], q[i])

    return qmc.measure(q)


# %%
ghz_state.draw(n=4, fold_loops=False)

# %% [markdown]
# ## Resource Estimation
#
# Qamomile provides symbolic resource estimation. Call `estimate_resources()` on any qkernel to get qubit counts and gate breakdowns **without executing** the circuit. Those estimates are symbolic expressions in terms of the input parameters, so you can analyze how resources scale with the input parameters. You can also substitute specific values to get concrete estimates for a given input size.
#
# Furthermore, the estimation can be done with black-box oracles, defined by `@composite_gate` with `stub=True`. This allows you to define your qkernel without actual implementation of the oracle, and still get resource estimates that include the oracle as black box with specified resource costs and the number of queries to it.
#
# See [Resource Estimation](../tutorial/resource-estimation) and [Reuse Patterns: QKernel Composition and Composite Gates](../tutorial/reuse-patterns) for more details.

# %%
est = ghz_state.estimate_resources()
print("qubits:", est.qubits)
print("total two-qubit gates:", est.gates.two_qubit)

# %%
# Evaluate at a specific size
print("two-qubit gates at n=100:", est.substitute(n=100).gates.two_qubit)

# %% [markdown]
# ## Multi-Quantum SDK Transpilation
#
# Write your circuit once with `@qkernel`, then transpile to any supported quantum SDK. Qamomile v0.10.0 has a preset executor for each supported quantum SDK. Just call `execute()` on the transpiled circuit with the preset executor. Of course, you can also write your own custom executors for controlling the detailed or use actual quantum hardware.
#
# | Backend | Module | Transpiler | Default Execution |
# |---------|--------|-----------|-----------|
# | Qiskit | `qamomile.qiskit` | `QiskitTranspiler` | Local simulator; qBraid for cloud devices |
# | QURI Parts | `qamomile.quri_parts` | `QuriPartsTranspiler` | Local simulator |
# | CUDA-Q | `qamomile.cudaq` | `CudaqTranspiler` | Local simulation |
#
# Furthermore, for Qiskit, [qBraid](https://docs.qbraid.com/) integration is available via `qamomile.qbraid.QBraidExecutor` to execute on cloud quantum devices.
#
# [Mention that Qiskit is a default, which means it is installed by pip install qamomile. Write how to install other optional dependencies.]
#
# See [Execution Models: sample() vs run()](../tutorial/execution-models).

# %%
from qamomile.qiskit import QiskitTranspiler

qiskit_transpiler = QiskitTranspiler()
qiskit_executable = qiskit_transpiler.transpile(ghz_state, bindings={"n": 4})
samples = qiskit_executable.sample(
    qiskit_transpiler.executor(),  # Specify the executor to run on
    shots=1024,
).result()

for outcome, count in samples.results:
    print(f"  outcome={outcome}, count={count}")

# %% [markdown]
# ## Standard Library
#
# In order to help users to write their own algorithms easily, Qamomile v0.10.0 provides multiple algorithms as `qmc.stdlib` and `qmc.algorithm` modules, including Quantum Fourier Transform (QFT), Quantum Phase Estimation (QPE) and more. `qmc.stdlib` is for broadly used algorithms that are commonly used as building blocks, while `qmc.algorithm` is for more specific algorithms that are not as widely used but still useful to have as ready-to-use implementations; however, those categorizations are not strict and may change in the future.
#
# We will keep on adding more algorithms to those modules, so stay tuned!

# %%
from qamomile.circuit.stdlib import qft


@qmc.qkernel
def qft_example() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    q[0] = qmc.x(q[0])
    q = qft(q)
    return qmc.measure(q)


qft_example.draw(expand_composite=True)

# %% [markdown]
# ## Optimization Converters
#
# The converters from v0.9.0 are available under `qamomile.optimization`, rewritten to use the new `@qkernel`-based circuit layer.
# They take a mathematical model (via [JijModeling](https://www.documentation.jijzept.com/docs/jijmodeling/) or OMMX) and produce a ready-to-run circuit.

# %% [markdown]
# ## Learn More
# - [Tutorials](../tutorial) to introduce you to Qamomile with hands-on examples.
# - [GitHub Repository](https://github.com/Jij-Inc/Qamomile)
# - [Full Changelog: v0.9.0 → v0.10.0](https://github.com/Jij-Inc/Qamomile/compare/v0.9.0...v0.10.0)
