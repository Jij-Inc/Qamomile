# Owned registers and components

Select lanes through the whole owner's gate selectors. Apply a unary gate to
all lanes with `state = qm.h(state)`, or select lanes in an index loop with
`state = qm.h(state, target=index)`.

Transfer a window to a helper by splitting and then reassembling all owners:

```python
@qm.qkernel
def flip_window() -> qm.Shot[qm.Bits[6]]:
    state = qm.qubits(6)
    prefix, remainder = qm.split_at(state, 1)
    window, suffix = qm.split_at(remainder, 3)
    window = qm.x(window)
    return qm.measure(qm.concat(prefix, window, suffix))
```

`split_at` returns two owners. `concat` consumes its owners in the supplied
order. Returning or terminally measuring each successor closes ownership;
`discard` expresses intentional physical discarding.

Annotate a variable-width helper's quantum input and output explicitly:

```python
from __future__ import annotations

@qm.qkernel
def link(state: qm.Q[qm.Bits[n]], control: qm.Int, target: qm.Int) -> qm.Q[qm.Bits[n]]:
    return qm.cx(state, control=control, target=target)
```

Here `n` is a dependent dimension bound by the input shape. Allocation widths
and structural shapes need compile-time information. Keep quantum states
inside kernel composition; executable host entries return classical data.

Widths may be polynomials over `Const[Int]` parameters, including floor
division by an integer such as `n // 2`. A `split_at` boundary can be a literal
or such an expression over Const Int parameters that the kernel never rebinds,
and `concat` adds the widths of its owners:

```python
@qm.qkernel
def suffix_flip(n: qm.Const[qm.Int], k: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n + 1]]:
    prefix, rest = qm.split_at(qm.qubits(n), k)
    rest = qm.x(rest)
    return qm.measure(qm.concat(prefix, rest, qm.qubits(1)))
```

`prefix` is `Bits[k]` and `rest` is `Bits[n-k]`; `qm.qubits(2 * n + 1)` is
likewise `Bits[2*n+1]`, and `qm.split_at(state, n // 2)` gives `Bits[(n//2)]`
and `Bits[-(n//2)+n]`. Widths compare as polynomials, so concatenating the
parts in their original order restores `Bits[n]` and the result can be a loop
carry. A reordered concatenation such as a rotation `concat(tail, head)` is a
new owner; inside a loop, rotate through a helper qkernel that takes and
returns the register. Specialization
rejects bindings that make any width nonpositive (`Q3007`). A loop index is not
a dimension binder: select a lane with a gate selector or `qm.control(..., at=i)`.

