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
# 2. **`@composite_gate`** — promote a kernel to a **named gate** that
#    appears as a single box in diagrams and can have backend-specific handling.
#
# There is also a third pattern for top-down design:
#
# 3. **Stub composite** — a gate with no implementation body, used for
#    resource estimation before the decomposition is finalized.

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## Pattern 1: Helper QKernel
#
# Any `@qkernel` function can be called from another `@qkernel`.
# The compiler inlines the call — the result is a flat circuit.


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
ghz_with_helper.draw(n=4)

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
# The helper `entangle_once` keeps the call site readable. In the compiled
# circuit, it is inlined — you see individual CX gates, not a sub-block.

# %% [markdown]
# ## Pattern 2: `@composite_gate`
#
# When you want a reusable block to appear as a **named box** in circuit
# diagrams (and potentially have backend-specific native implementations),
# promote it with `@composite_gate`.
#
# Stack `@composite_gate(name="...")` on top of `@qkernel`:


# %%
@qmc.composite_gate(name="entangle_link")
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
ghz_with_composite.draw(n=4)

# %% [markdown]
# ### When to use which?
#
# | Pattern | Appears in `draw()` | Backend-specific handling | Use when |
# |---------|---------------------|--------------------------|------|
# | Helper `@qkernel` | Inlined (flat) | No | Code organization |
# | `@composite_gate` | Named box | Yes (emitters can provide native versions) | Domain-level abstraction |
#
# Use a plain helper when you just want to avoid repetition.
# Use `@composite_gate` when the block has a meaningful name that
# should be visible in diagrams and may benefit from native backend support
# (like QFT, which Qiskit can implement natively).

# %% [markdown]
# ## Pattern 3: Stub Composite for Top-Down Design
#
# Sometimes you want to design an algorithm's structure before implementing
# every sub-component. A **stub composite** has no implementation body — just
# a name, qubit count, and optional resource metadata.
#
# This lets you estimate the cost of the overall algorithm while the
# oracle or sub-routine is still under development.


# %%
@qmc.composite_gate(
    stub=True,
    name="oracle_box",
    num_qubits=3,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
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
algorithm_skeleton.draw()

# %% [markdown]
# ### Resource Estimation with Stubs
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
# (qubit count, oracle queries, T-gate budget) before committing to
# a full decomposition.

# %% [markdown]
# ## Summary
#
# - **Helper `@qkernel`**: call one kernel from another for code reuse.
#   The compiler inlines the call into a flat circuit.
# - **`@composite_gate`**: gives a kernel a named identity visible in
#   diagrams and backends. Stack it on top of `@qkernel`.
# - **Stub composite**: `stub=True` with `ResourceMetadata` for top-down
#   design and resource estimation without a full implementation.
