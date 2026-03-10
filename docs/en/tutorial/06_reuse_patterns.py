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
# # Reuse Patterns: QKernel Composition and Composite Gates
#
# As circuits grow, you want to avoid copy-pasting gate sequences.
# Qamomile offers two complementary reuse mechanisms:
#
# 1. **Helper QKernel** — call one `@qkernel` from another, like normal
#    function composition.
# 2. **`@composite_gate`** — promote a qkernel to a **named gate**
#    with customizable settings that appears as a single box in diagrams.
#
# There is also a third pattern for top-down design:
#
# 3. **Stub composite gate** — a gate with no implementation body, used for
#    resource estimation.
#    For example, if you are designing a Grover search algorithm,
#    you know the oracle will use ~40 T-gates, but you haven't implemented it yet.
#    A stub composite gate lets you estimate the total cost of the algorithm
#    without the full oracle implementation.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Pattern 1: Helper QKernel
#
# Any `@qkernel` function can be called from another `@qkernel`.
# The transpiler inlines the call — the result is a flat circuit.


# %%
@qmc.qkernel
def entangle_once(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def ghz_with_helper(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_once(q[i], q[i + 1])

    return qmc.measure(q)


# %%
ghz_with_helper.draw(n=4, fold_loops=False)

# %%
result = (
    transpiler.transpile(ghz_with_helper, bindings={"n": 4})
    .sample(
        transpiler.executor(),
        shots=128,
    )
    .result()
)
print("GHZ result:", result.results)

# %% [markdown]
# The helper `entangle_once` keeps the call site readable. In the transpiled
# circuit, it is inlined — you see individual CX gates, not a sub-block.

# %%
qc = transpiler.to_circuit(ghz_with_helper, bindings={"n": 4})
print(qc.draw())

# %% [markdown]
# ## Pattern 2: `@composite_gate`
#
# When you want a reusable block to appear as a **named box** in circuit
# diagrams, promote it with `@composite_gate`.
# Also, as a more advanced use case,
# making it a composite gate allows you to give it custom settings such as
# giving it multiple ways of implementation.
#
# Stack `@composite_gate(name="...")` on top of `@qkernel`:


# %%
@qmc.composite_gate(name="entangle")
@qmc.qkernel
def entangle_link(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def ghz_with_composite(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])

    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_link(q[i], q[i + 1])

    return qmc.measure(q)


# %%
ghz_with_composite.draw(n=4, fold_loops=False)

# %% [markdown]
# ### When to use which?
#
# | Pattern | Appears in `draw()` | Use when |
# |---------|---------------------|------|
# | Helper `@qkernel` | Inlined (flat) | Code organization |
# | `@composite_gate` | Named box | Domain-level abstraction/advanced settings |

# %% [markdown]
# ## Pattern 3: Stub Composite Gate for Top-Down Design
#
# Sometimes you want to design an algorithm's structure before implementing
# every sub-component. A **stub composite gate** has no implementation body — just
# a name, qubit count, and optional resource metadata.
#
# This lets you estimate the cost of the overall algorithm while the
# oracle or sub-routine is still under development.
#
# To use a stub composite gate, specify `stub=True` in the `@composite_gate` decorator.
# At the same time, you can also give it resource information as `ResourceMetadata`.


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle",
    num_qubits=3,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        total_gates=40,
        t_gates=40,
    ),
)
def oracle_box():
    pass


@qmc.qkernel
def algorithm_skeleton() -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])

    q[0], q[1], q[2] = oracle_box(q[0], q[1], q[2])
    return q


# %%
algorithm_skeleton.draw(fold_loops=False)

# %% [markdown]
# ### Resource Estimation with Stub Composite Gates
#
# `estimate_resources()` picks up the stub's metadata automatically.
# You can also query the metadata directly.

# %%
est = algorithm_skeleton.estimate_resources().simplify()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)

# %%
meta = oracle_box.get_resource_metadata()
print("oracle query complexity:", meta.query_complexity)
print("oracle T-gate count:", meta.t_gates)

# %% [markdown]
# This top-down approach lets you reason about algorithm-level costs
# (such as qubit count, oracle queries) before committing to
# a full decomposition.

# %% [markdown]
# ## Summary
#
# - **Helper `@qkernel`**: call one qkernel from another for code reuse.
#   The transpiler inlines the call into a flat circuit.
# - **`@composite_gate`**: gives a qkernel a named identity visible in
#   diagrams. Stack `@composite_gate` on top of `@qkernel`.
# - **Stub composite gate**: `stub=True` with `ResourceMetadata` for top-down
#   design and resource estimation without a full implementation.
