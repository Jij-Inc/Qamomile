# Measured control and typed classical data

A scalar prepared Boolean produces a scalar measured Boolean. A measured
register produces `Bits[n]`; select `bool(bits[index])` for a branch or score.

```python
@qm.qkernel
def score() -> tuple[qm.Shot[qm.Real], qm.Shot[qm.Real]]:
    bits = qm.measure(qm.h(qm.qubits(1)))
    e = -1.0 if bool(bits[0]) else 1.0
    return e, e * e
```

Per-shot expressions stay in the kernel. Compute means and expectations on
host results afterward. `if` uses a scalar `Bool`/`Bit` condition.

Use `qm.sqrt(value)` for a source-language square root. It accepts Int or Real
and returns Real at the same Const, Param, or Shot stage. The value must be
finite and nonnegative when evaluated. Int uses the ordinary finite binary64
conversion, so an integer too large for Real is rejected. Backend support for
Shot arithmetic still depends on the target's existing capabilities.
Python library calls such as `math.sqrt` remain outside the source language;
host calculations can be passed through Const or Param inputs.

```python
@qm.qkernel
def sqrt_rotation() -> qm.Shot[qm.Bool]:
    angle = qm.sqrt(9.0) * 3.141592653589793 / 3.0
    return qm.measure(qm.ry(qm.prepare(False), angle))
```

This program returns `True` on every shot. Const square roots are evaluated at
specialization. Param square roots can be reused with new launch values without
rebuilding the source program.

Use `float(value)` to convert an Int or Real scalar to Real at the same
availability stage. For Param inputs, the native compiler evaluates integer
arithmetic and each conversion at launch before SDK emission. Integer values
outside the finite binary64 range are rejected. Other Param rotation inputs
can remain symbolic, and one compiled artifact can be reused with new values.

```python
@qm.qkernel
def integer_rotation(offset: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    angle = 3.141592653589793 * float(offset + 1)
    return qm.measure(qm.rx(qm.qubits(1), angle))
```

This program measures `1` for `offset=0` and `0` for `offset=1`. The endpoint
must support runtime parameters; Shot conversions still depend on its existing
per-shot arithmetic capabilities. For Qiskit circuit inspection, supply launch
values with `to_circuit(program, parameters={"offset": 0})`.

`measure_reset` and nondestructive `measure_z` return `(bit, successor_state)`.
For a scalar Boolean state, `bit` is scalar; for a whole register reset it is
a bit register, so select a lane. Retain the successor for subsequent work.

```python
@qm.qkernel
def repeat_once() -> qm.Shot[qm.Bool]:
    bit, state = qm.measure_reset(qm.prepare(True))
    marker = qm.prepare(False)
    for _ in range(2):
        if not bit:
            break
        marker = qm.x(marker)
        bit, state = qm.measure_reset(state)
    qm.discard(state)
    return qm.measure(marker)
```

Use bounded `for` loops for measurement-driven repetition. `break` exits the
nearest loop and skips its `else`; `continue` skips the remaining body for that
iteration. Rebind every quantum successor on the paths that continue. Resource
estimates retain enclosures when measurement results determine how much work runs.
Keep `return` after the loop; measurement-dependent loop exits combined with
early returns remain unsupported.

For fixed sparse data, annotate maps with their key/value types and size:
`terms: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Real, n_terms]]`, where
`n_terms: qm.Const[qm.Int]`. Bind an ordinary Python dict at specialization.
Iterate with `for edge, weight in terms.items():`, index `scales[edge]`, and
apply `qm.rzz(state, gamma * weight * scales[edge], left=edge[0], right=edge[1])`.
The source remains a structured loop over typed data.

Const maps can contain more than 1024 entries. They use the shared native
package limits (64 MiB encoded data, one million structural nodes, and bounded
nesting) rather than a separate entry-count cap. Keys remain unique and match
the declared key type. A single lookup does not expand into one gate per entry.

```python
@qm.qkernel
def table_rotation(angles: qm.Const[qm.Map[qm.Int, qm.Real, 1025]]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), angles[1024]))

angles = dict.fromkeys(range(1025), 0.0)
angles[1024] = 3.141592653589793
table_program = qm.specialize(table_rotation, angles=angles)
```

This program applies one RX gate with the selected angle and produces `(1,)`.

Use integer literals or direct `Const[Int]` parameters for classical array
slice boundaries. Start, stop, and stride are resolved at specialization using
Python's negative-index and clipping rules. The complete slice retains its
array type and can be passed to a helper or iterated. `Param`/`Shot` boundaries
would change the runtime shape and remain unsupported. Empty slices have no
nonempty array type; zero strides and more than 1024 selected elements also fail.

```python
@qm.qkernel
def sliced_rotation(
    angles: qm.Const[qm.Vector[qm.Real, 2]],
    start: qm.Const[qm.Int],
) -> qm.Shot[qm.Bits[1]]:
    window = angles[start:]
    return qm.measure(qm.rx(qm.qubits(1), window[0]))
```

A single element of a local classical array or `Bits`/`UInt` word can be assigned
with `values[index] = replacement`. This creates a successor value and rebinds
that local name; aliases still refer to the old value. Measurement-derived data
retains its `Shot` stage. Quantum values, maps, slice assignment, and nested
assignment targets are excluded. Aggregate updates retain at most 1024 leading
elements. For quantum circuit execution, use an index
known during circuit projection, such as a literal, Const, or static loop index.

```python
@qm.qkernel
def duplicate_measured_bit() -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[2]]]:
    bits = qm.measure(qm.x(qm.qubits(2), target=0))
    original = bits
    bits[1] = bits[0]
    return bits, original
```


A measured value can control a fresh circuit after the previous quantum state
has been fully consumed. Keep the dependent computation inside the qkernel:

```python
@qm.qkernel
def correlated_pair() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    first = qm.measure(qm.h(qm.prepare(False)))
    second = qm.prepare(False)
    if first:
        second = qm.x(second)
    return first, qm.measure(second)
```

This returns only `(False, False)` and `(True, True)`. Circuit compilation can
split the closed quantum regions and batch the second circuit by the first
measurement's observed populations. Terminal classical-only `if` uses native
lazy postprocessing. A surviving quantum state, including one involved in a
later join, prevents a host cut; the target must support the remaining dynamic
circuit. A finite choice between constant/Param rotation angles can become a
device branch; arbitrary measured-angle arithmetic needs dynamic-rotation
support or a legal closed cut.

`Shot[T]` remains a per-trajectory value, not an array of shots. Reusing a
measurement preserves correlation; different quantum preparations are sampled
freshly. Current source classical operations are deterministic. There is no
classical random API.


`while` uses a scalar classical Bool/Bit condition, re-evaluated before each
iteration. It supports `break`, `continue`, nested loops, early `return`, and
Python's loop `else` (normal exit only). Carry types, quantum widths, and linear
owners must agree on every continuation and exit; the condition cannot consume
an owned quantum carry. Changed classical carries join the condition's stage.

Const conditions can expand at circuit emission within the compiler's work
budget. Param conditions materialize at launch; pure-classical Param loops also
execute through native `qm.evaluate`. Shot conditions retain the DynamicLoop
requirement: finite circuit targets currently reject unbounded measured loops.
Acceptance does not prove termination. A resource iteration assumption never
limits actual execution. See `references/resources.md` for symbolic while counts.
