# Coherent callable control

Wrap a primitive in an annotated local quantum helper, then control that
helper. Destructure the returned control and target successors:

```python
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def selected(at: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target, at=at)
    return qm.measure(qm.concat(controls, target))
```

The control callable is a local qkernel or a typed `QKernel` parameter.
`qm.control` applies its unitary body; all control/target owners are retained.

- `at=index` selects which control lane to test. It can be a launch parameter.
- `control_state=pattern` sets the activating basis integer of the entire
  selected control register. For controls `(1,0)`, use `control_state=1`.
- Omitting a pattern activates on all ones. Integer patterns use lane zero as
  the least significant bit.

For an RX helper, annotate the target as `qm.Q[qm.Bits[1]]` and the angle as
`qm.Real` or the appropriate staged real; return `qm.rx(state, theta)`.
Pass the angle after the target in the `control` call. Use `qm.multiplex` for
an indexed table of unitary callables after inspecting its current contract.
