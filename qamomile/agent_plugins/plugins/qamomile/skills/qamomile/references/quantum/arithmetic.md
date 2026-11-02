# Coherent unsigned-register arithmetic

Prepare a little-endian bit tuple, then reinterpret it with `qm.as_uint`.
For example, `(0,1)` represents two. Use `qm.as_bits` before concatenating
measured output lanes. Quantum operands are returned in the operation's order.

```python
@qm.qkernel
def sum_sample() -> qm.Shot[qm.Bits[4]]:
    target = qm.as_uint(qm.prepare((1, 0)))
    addend = qm.as_uint(qm.prepare((0, 1)))
    target, addend = target + addend
    return qm.measure(qm.concat(qm.as_bits(target), qm.as_bits(addend)))
```

| Operation | Successors to retain |
| --- | --- |
| `target + addend` / `target - addend` | `target, addend` |
| `target + constant` / `target - constant` | `target` |
| `qm.mod_add(target, addend, modulus)` with quantum operands | `target, addend, modulus` |
| `qm.mod_add(target, constant, modulus_constant)` | `target` |
| `qm.mul_add(accumulator, left, right)` | `accumulator, left, right` |
| `qm.less_than_xor(flag, left, right)` | `flag, left, right` |
| `qm.mod_mul(target, factor=..., modulus=...)` | `target` |

Prepare a numeric comparison flag with `qm.as_uint(qm.prepare((0,)))` when it
will pass through `as_bits`. A scalar Boolean flag can be concatenated directly.

Ordinary addition/subtraction wraps at `2**width`. For `mul_add`, the accumulator
has twice the factors' width. A comparison flag has width one. Modular operations
check their mathematical domain; reversible multiplication needs a valid factor.

Use these source-level operations for arithmetic contracts. More specialized
stdlib factories are reusable kernels, with their own typed signatures; create
the factory on the host and inspect its actual input/output owners before use.
