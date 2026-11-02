# Reusable quantum components

Annotate every quantum helper's input and return, and compose it inside the
containing qkernel. Use `qm.Q[qm.Bits[n]]` for variable-width owned states.
The caller transfers the successor through its loop or branch.

A callable parameter includes its complete logical input/output signature:

```python
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def repeat(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
    rounds: qm.Const[qm.Int],
) -> qm.Q[qm.Bits[1]]:
    for _ in range(rounds):
        state = operation(state)
    return state

@qm.qkernel
def sample_repeat(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(repeat(flip, qm.qubits(1), rounds))
```

Library factories are called on the host once to create reusable components;
invoke the resulting component from a kernel. `composite_gate` is for a reusable
gate abstraction. Opaque declarations describe intentional external interfaces;
resource rules price their declared contract rather than inventing a body.

Use `typing.NamedTuple` to name the fields of an owned quantum aggregate.
Destructure the entire record before using its fields, then construct its
successor explicitly:

```python
from typing import NamedTuple

class Pair(NamedTuple):
    first: qm.Q[qm.Bool]
    second: qm.Q[qm.Bool]

@qm.qkernel
def flip_first(pair: Pair) -> Pair:
    first, second = pair
    first = qm.x(first)
    return Pair(first, second)

@qm.qkernel
def sample_pair() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    pair = Pair(qm.prepare(False), qm.prepare(True))
    first, second = flip_first(pair)
    return qm.measure(first), qm.measure(second)
```

Construction and complete destructuring use the same linear ownership rules
as tuples. After `first, second = pair`, the old `pair` is consumed; transfer
each quantum field exactly once. Quantum-containing records require complete
destructuring instead of `pair.first` or `pair[0]`. Quantum fields cannot have
defaults. A record containing `Q` fields is an aggregate of owners; a classical
record is not a supported payload for `Q[Record]`.

Self-recursive and mutually recursive qkernels are allowed. Keep full input and
return annotations on every function. Each call transfers its quantum inputs
to that invocation and returns successor owners under the ordinary call rules:

```python
@qm.qkernel
def recursive_flip(
    state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]
) -> qm.Q[qm.Bits[1]]:
    if rounds == 0:
        return state
    else:
        return recursive_flip(qm.x(state), rounds - 1)

@qm.qkernel
def sample_recursive(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(recursive_flip(qm.qubits(1), rounds))
```

Recursion preserves `Const`, `Param`, and `Shot` availability. Pure classical
recursion with Param arguments runs through the native Reference evaluator.
Quantum calls must resolve to a finite circuit at compilation or launch
materialization; a Param value controlling recursion is materialized at launch
on targets supporting that route. Recursive helpers can receive closed QKernel
arguments and support `adjoint` and `control` when their transitive effects and
ownership satisfy the usual coherent contracts.

Building a recursive module does not prove termination. Circuit expansion is
bounded by 64 active call frames and 10 million operation visits; the native
classical Reference evaluator retains its 256-frame and work budgets. Exceeding
a budget fails explicitly. Shot-dependent quantum recursion currently has no
runtime call-stack emission route on circuit targets.

Additive symbolic resources retain equations for recursive calls whose classical
parameters are supported `Const[Int]` expressions. Concrete evaluation is bounded
and memoized in Rust, without circuit expansion or a SymPy dependency. Equations
are partial: they apply to terminating invocations satisfying their parameter
conditions. Divergence, undefined arithmetic, or an exhausted evaluation budget
leaves the expression unresolved, rather than inventing a finite cost. See
[resource estimation](../resources.md) for evaluation and persistence examples.

Recursive depth and peak/allocation liveness require different recurrences and
currently retain named unknown contributors. Param/Shot-dependent recursive
costs, including expected costs requiring probabilities, also remain unsupported.
`resources.logical(..., strict=True)` rejects named unknowns. Circuit compilation
and execution remain independent of whether these estimates are available.
