---
name: qamomile
description: Write, debug, execute, and estimate quantum algorithms with Qamomile. Use for code importing qamomile or requests to implement a Qamomile quantum program.
---

# Qamomile quantum programs

Write annotated `@qm.qkernel` functions in a `.py` file. Compose gates, helpers,
and supported control flow inside those functions; keep data preparation,
classical optimization, SDK configuration, and plotting in host Python.
Every gate consumes a quantum value and returns its successor: rebind it.

```python
import qamomile as qm


@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)
```

Match annotations to the value's shape, including width one:

| Preparation | Owned type | Terminal measurement type |
| --- | --- | --- |
| `qm.qubits(n)` or `qm.prepare((0, 1))` | `qm.Q[qm.Bits[n]]` | `qm.Shot[qm.Bits[n]]` |
| `qm.prepare(False)` or `qm.prepare(True)` | `qm.Q[qm.Bool]` | `qm.Shot[qm.Bool]` |

For a register result, select a bit with `bool(bits[0])` before a Boolean branch.
A quantum helper has both input and return annotations such as
`qm.Q[qm.Bits[1]]`; the width belongs inside `Bits`. Use `Const` for dimensions
and static data, `Param` for launch values, and `Shot` for measurement data.
Use `from __future__ import annotations` for dependent shapes such as `Bits[n]`.

Read the relevant branch before implementing its feature:

- [Quantum authoring](references/authoring.md): ownership, coherent control,
  measured control flow, typed callable reuse, and integer arithmetic.
- [Execution](references/execution.md): specialize, compile, launch, and results.
- [Libraries and resources](references/libraries-resources.md): algorithm
  composition, observables, body-derived estimates, and algorithm visualization.
- [Installation](references/skill-cli.md): direct skill and marketplace methods.

Use the installed public signatures/docstrings to confirm advanced options.
Verify a small instance through the same Qamomile program and selected endpoint.
