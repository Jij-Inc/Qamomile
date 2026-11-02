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
