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
# # Reuse Patterns: QKernel, Composite Gate, and Opaque Calls
#
# Qamomile has three reuse patterns.
#
# A helper `qkernel` is ordinary program structure.
#
# `composite_gate()` creates a named quantum box with a Qamomile body.
#
# `opaque()` creates a bodyless callable for top-down algorithm design and resource estimation.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()


# %% [markdown]
# ## Pattern 1: helper QKernel
#
# Use a helper qkernel when you want ordinary code reuse.
#
# During compilation, helper calls are inlined into the caller.


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
assert result.shots == 128
assert sum(count for _, count in result.results) == 128
assert all(outcome in {(0, 0, 0, 0), (1, 1, 1, 1)} for outcome, _ in result.results)

# %% [markdown]
# ### Passing scalar literals to helpers
#
# A helper qkernel can declare scalar handles such as `UInt`, `Float`, or `Bit`.
#
# At the call site, raw Python literals are promoted to the expected handle type.


# %%
@qmc.qkernel
def rotate_first(
    q: qmc.Vector[qmc.Qubit],
    idx: qmc.UInt,
    angle: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    q[idx] = qmc.ry(q[idx], angle)
    return q


@qmc.qkernel
def helper_with_literals(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = rotate_first(q, 0, 0.5)  # type: ignore[arg-type]
    return qmc.measure(q)


helper_with_literals.draw(n=3, fold_loops=False, inline=True)

# %% [markdown]
# ## Pattern 2: composite gate
#
# Use a composite gate callable when the reusable unit should remain visible as a
# named quantum operation.
#
# The public user-facing form is `@qmc.composite_gate` on a typed Python function.
#
# Qamomile turns the function into a qkernel body internally, while calls to it
# remain visible as a named box.


# %%
@qmc.composite_gate(
    name="h_layer",
    resource_model=qmc.FixedResourceModel(
        gates=qmc.GateResources(total=3, single_qubit=3),
    ),
)
def h_layer(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    for i in qmc.range(q.shape[0]):
        q[i] = qmc.h(q[i])
    return q


@qmc.qkernel
def custom_composite_layer() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")
    q = h_layer(q)
    return qmc.measure(q)


block = custom_composite_layer.build()
assert block.name == "custom_composite_layer"

composite_est = custom_composite_layer.estimate_resources().simplify()
assert composite_est.gates.total == 3

custom_composite_layer.draw(fold_loops=False)

# %% [markdown]
# QFT is a built-in example.
#
# It has a Qamomile body, but a backend may emit it natively.


# %%
@qmc.qkernel
def qft_round_trip() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(3, name="q")
    q = qmc.h(q)
    q = qmc.qft(q)
    q = qmc.iqft(q)
    return qmc.measure(q)


block = qft_round_trip.build()
assert block.name == "qft_round_trip"

qft_est = qft_round_trip.estimate_resources().simplify()
print("QFT round-trip gates:", qft_est.gates.total)
assert qft_est.gates.total == 17

qft_round_trip.draw(fold_loops=False)

# %%
qft_result = (
    transpiler.transpile(qft_round_trip)
    .sample(
        transpiler.executor(),
        shots=64,
    )
    .result()
)
print("QFT round-trip result:", qft_result.results)
assert sum(count for _, count in qft_result.results) == 64

# %% [markdown]
# ## Pattern 3: Opaque callable
#
# Use an opaque callable when the algorithm needs a named operation before its implementation is available.
#
# An opaque callable has no body.
#
# It can still carry a resource model, so resource estimation can count it.


# %%
marked_state_oracle = qmc.opaque(
    "marked_state_oracle",
    signature=qmc.CallableSignature(
        inputs=[qmc.Vector[qmc.Qubit]],
        outputs=[qmc.Vector[qmc.Qubit]],
    ),
    resource_model=qmc.FixedResourceModel(
        gates=qmc.GateResources(t=40),
        calls=qmc.CallResources(
            calls_by_name={"marked_state_oracle": 1},
            queries_by_name={"marked_state_oracle": 1},
        ),
    ),
)


@qmc.qkernel
def grover_skeleton(rounds: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(3, name="q")
    q = qmc.h(q)

    for _ in qmc.range(rounds):
        q = marked_state_oracle(q)
        q = qmc.h(q)

    return q


# %%
grover_skeleton.draw(rounds=2, fold_loops=False)

# %% [markdown]
# The callable is opaque, so this qkernel is not executable yet.
#
# It is still useful for resource estimation.

# %%
est = grover_skeleton.estimate_resources().simplify()
print("qubits:", est.qubits)
print("total gates:", est.gates.total)
print("T gates:", est.gates.t_gates)
print("oracle calls:", est.calls.oracle_calls)
print("oracle queries:", est.calls.oracle_queries)
assert est.qubits == 3
assert str(est.gates.total) == "3*rounds + 3"
assert str(est.gates.t_gates) == "40*rounds"
assert {k: str(v) for k, v in est.calls.oracle_calls.items()} == {
    "marked_state_oracle": "rounds"
}
assert {k: str(v) for k, v in est.calls.oracle_queries.items()} == {
    "marked_state_oracle": "rounds"
}

# %% [markdown]
# Substitute a concrete loop count when you want numeric estimates.

# %%
est_4 = est.substitute(rounds=4)
print("T gates for 4 rounds:", est_4.gates.t_gates)
assert est_4.gates.t_gates == 160
assert est_4.calls.oracle_calls == {"marked_state_oracle": 4}

# %% [markdown]
# ## Summary
#
# Use a helper `qkernel` for ordinary code structure.
#
# Use `composite_gate()` when the call has a body but should remain visible as a named operation.
#
# Use `opaque()` when the call intentionally has no body yet, but should participate in diagrams and resource estimation.
#
# For controlled gates (`qmc.control`), see [Tutorial 04 — Controlled Gates](04_controlled_gates.ipynb).
