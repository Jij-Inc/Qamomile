# Estimate the algorithm body

```python
study = qm.resources.logical(kernel)
gates = study.metric(qm.resources.LOGICAL_GATES)
print(gates.numeric_enclosure())
```

A `Const`-parameterized body can be studied before specialization. Evaluate a
symbolic metric with `gates.evaluate({"width": 4})`. Measures and typed quantum
components can each be studied; their measurement costs differ.

`ResourceValue` retains units, a relation, and evidence. Report bounds and
unresolved symbols as such. `LOGICAL_PEAK_QUBITS` and `LOGICAL_ALLOCATED_QUBITS`
answer different questions. Physical projections require a named resource model
and its assumptions. Inspect `resources.project`/`surface_code` when needed.

Verify a small executable instance separately from a large symbolic study.
Structured loops preserve the source's symbolic extent. Opaque-call rules price
explicit declarations; a body-derived estimate remains tied to that body.

An adjacent dependence chain can have exact depth before specialization. A
single gate on lanes `(i, i + 1)` in `range(n - 1)`, seeded by lane zero or a
whole-register barrier, admits a compact native proof rather than loop expansion:

```python
@qm.qkernel
def ghz_depth_sample(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.h(qm.qubits(n), target=0)
    for i in range(n - 1):
        state = qm.cx(state, control=i, target=i + 1)
    return qm.measure(state)

ghz_depth = qm.resources.logical(ghz_depth_sample).metric(qm.resources.LOGICAL_GATE_DEPTH)
assert ghz_depth.evaluate({"n": 4}).numeric_enclosure() == (4, 4)
```

The proof retains individual lane completion times, so a following gate on a
finished head lane can overlap the remaining chain. Other selected prefixes,
partial/nonadjacent chains, nested repeats, and path-dependent exits may retain
conservative enclosures. Always inspect the reported relation and evidence.

Natural additive costs of supported Const integer recursion are retained as scoped,
simultaneous equations, including base-case conditions. Concrete inputs evaluate
with bounded memoization in Rust, rather than expanding a circuit or invoking
SymPy:

```python
@qm.qkernel
def recursive_resource_step(
    state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]
) -> qm.Q[qm.Bits[1]]:
    if rounds == 0:
        return state
    else:
        return recursive_resource_step(qm.x(state), rounds - 1)

recursive_study = qm.resources.logical(recursive_resource_step)
recursive_gates = recursive_study.metric(qm.resources.LOGICAL_GATES)
assert recursive_gates.evaluate({"rounds": 3}).numeric_enclosure() == (3, 3)
restored_recursive_study = qm.resources.ResourceStudy.from_bytes(
    recursive_study.to_bytes()
)
assert restored_recursive_study.metric(qm.resources.LOGICAL_GATES).evaluate(
    {"rounds": 4}
).numeric_enclosure() == (4, 4)
```

The equation above is `G(0) = 0`, `G(n) = 1 + G(n - 1)` otherwise. Its finite
solution applies to `n >= 0`; a signed Const integer may also be negative, so
`G(n) = n` is not an unconditional result. No automatic closed-form solution or
termination proof is claimed. A concrete call that cycles, exceeds the resource
expression work budget, or has undefined arithmetic remains unresolved.
Definitions and their lexical parameter bindings survive expression arithmetic,
substitution, and canonical study persistence.

Portable circuit projection of recursive bodies, recursive scheduling depth,
and allocation/peak liveness retain named unknown
contributors. Param/Shot-dependent recursive costs and probabilistic expected
costs remain unsupported in this phase. Inspect each metric separately: an
unknown peak estimate does not erase a supported additive gate count. Strict
analysis rejects named unknowns; ordinary compilation can still proceed.


A `while` loop's body-execution count becomes a scoped natural symbol. Inspect
`study.iteration_symbols` to identify its source and invocation context. Bind
symbols explicitly rather than ambiguous short names when several loops exist:

```python
@qm.qkernel
def while_resource_step(
    state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]
) -> qm.Q[qm.Bits[1]]:
    index = 0
    while index < rounds:
        state = qm.x(state)
        index += 1
    return state

while_study = qm.resources.logical(while_resource_step)
(iterations,) = while_study.iteration_symbols
scenario = while_study.assume_iterations({iterations: 3})
assert scenario.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (3, 3)
bounded = while_study.bound_iterations({iterations: 5})
assert bounded.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (None, 5)
assert bounded.metric(qm.resources.LOGICAL_GATES).assumptions

@qm.qkernel
def while_resource_sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(while_resource_step(qm.qubits(1), 3))

assert qm.run(while_resource_sample, on=qm.Reference(), shots=8, seed=7).result.counts == {(1,): 8}
```

A scenario is conditional on the supplied counts. An upper limit is a user
assertion, with monotonicity checked in Rust; neither constitutes a compiler
termination proof or changes execution. The original symbolic study stays
immutable, and assumptions survive serialization. Additive fixed-body costs can
be exact functions of the count. State-dependent costs, varying nested counts,
and exit-dependent allocation/liveness can remain unknown. Peak qubits and depth
follow their own composition rules rather than blindly multiplying by the count.
