# Use the logical shot index

Use `qm.get_current_shot()` inside a qkernel to read its invocation's logical
shot index. An invocation with `shots=N` assigns the indices `0..N-1` once
each. Repeated reads, helper calls, and loop iterations in one trajectory see
the same index. Each new invocation starts at zero, including when you reuse
a compiled artifact.

The result is `Shot[Int]`. Use it in predicates, integer calculations, or
classical element selection inside the kernel. The read does not measure a
quantum state or draw randomness. A measurement-free Boolean result still
produces one sample for every requested logical shot.

```python
import qamomile as qm


@qm.qkernel
def is_first() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() == 0


@qm.qkernel
def alternating() -> qm.Shot[qm.Bit]:
    state = qm.prepare(0)
    if qm.get_current_shot() % 2 == 0:
        state = qm.x(state)
    return qm.measure(state)


first_artifact = qm.compile(qm.specialize(is_first), to=qm.Reference())
first_result = qm.execute(first_artifact, on=qm.Reference(), shots=5)
assert first_result.values == {True: 1, False: 4}

alternating_program = qm.specialize(alternating)
alternating_result = qm.run(alternating_program, on=qm.Reference(), shots=5).result
assert alternating_result.values == {1: 3, 0: 2}
```

Keep the result at `Shot` availability. It cannot supply a `Const` shape, a
`Param` helper argument, or a host Python loop counter. Calling the intrinsic
from host Python raises `RuntimeError`. Returning its unbounded `Int` directly
from `run` or `execute` has no sampled result encoding. Return a Boolean predicate or another
supported finite sampled value instead.

Compilation fixes the target and its checked logical-index realization.
Execution supplies `shots=N`; it does not turn the source value into a public
`Param` input. The Reference endpoint evaluates the index in Rust. A Qiskit
realization retains one parameterized circuit when its structure is fixed.
Rust evaluates hidden numeric bindings for each logical index at execution.
Runtime Sampler sends those rows in one PUB with one physical shot per row.
Aer sends one prepared circuit with a native parameter table; if preparation
removes every numeric slot, it requests ordered shot memory from that circuit.
Ordinary backends bind checked copies and send one batch. When the index
changes circuit structure, a checked target-bound recipe generates the
variants. Both paths preserve the source `Shot` meaning and keep the logical
coordinate until native result evaluation is complete. Host-indexed launches
accept at most 10,000,000 logical shots, subject to native work and allocation
budgets and endpoint batch limits. This cap applies to current-shot execution,
not to ordinary invocation batches or the certified native HUGR counter route.
The numeric binding table retains a separate 4,000,000-cell limit: one hidden
numeric slot therefore cannot prepare 10,000,000 rows. Ordinary Reference
sampling also has a 10,000,000-shot limit; ordinary Qiskit execution follows its
SDK and endpoint limits rather than this host current-shot count ceiling.
Oversized requests fail before submission; Qamomile does not automatically
partition them into additional SDK jobs. Numeric launch preparation and retained-result
reconstruction share an aggregate native work limit checked before submission.
Result reconstruction checks its work limit again when results arrive.

Reference shares its existing simulator-work budget across every logical index
in one invocation. Statevector width and expanded loops count toward that limit;
the shot-count ceiling alone does not establish that a circuit fits. For a
current-shot program, `qm.evaluate(program, seed=k)` and Reference sampling with
`shots=1, seed=k` derive the index-zero random stream from the same seed.

With the `qamomile[qiskit]` extra, execute the same
program on local Aer:

```python
from qamomile.qiskit import Aer, Qiskit

qiskit_artifact = qm.compile(alternating_program, to=Qiskit())
qiskit_result = qm.execute(qiskit_artifact, on=Aer(), shots=5)
assert qiskit_result.values == {1: 3, 0: 2}
```

Support belongs to the selected execution route. A native provider counter
needs a checked origin convention that the target normalizes to zero. A HUGR
format alone does not certify an endpoint's numbering. The default `Hugr()`
profile and `Selene` endpoint qualify zero-origin execution for Selene 0.2.17
and 0.3.2 with explicit shot offsets and increments. The `hugr` extra pins 0.2.17.
That route checks signed 32-bit integer arithmetic before using HUGR's wider
integer representation; conversion of these integers to `Real` is exact.
Its current-shot count domain is at most `2**31` shots, with zero-based indices
through `2**31 - 1`; this is not a provider capacity or performance guarantee.
Internal Helios has no certified shot-origin convention and rejects this feature.
Unsupported endpoints and segmented plans fail before provider submission. Do not
substitute a host counter or add a dummy measurement to obtain sampled execution.

Resource estimation describes one trial using the shared nonnegative symbol
`shot`. Repeated reads, helpers and loop bodies share that symbol. A loop over
`range(qm.get_current_shot())` retains its shot-dependent count, and supported
complementary conditions retain their aggregate relationship. See
[per-shot costs](../resources.md) for a complete example.
Binding `shot` in a resource expression is analysis substitution, not an
execution parameter. Estimates do not automatically sum every requested shot.
