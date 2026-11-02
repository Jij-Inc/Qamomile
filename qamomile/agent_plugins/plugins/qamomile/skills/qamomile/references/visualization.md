# Inspect an algorithm without executing it

Use `qamomile.visualization.draw` on a specialized `Program`, symbolic
`ProgramFamily`, or source `QKernel`. Source declarations are compiled without
Python tracing or backend execution. The returned immutable `Diagram` displays
as HTML when it is the last expression in a Jupyter cell, or via
`IPython.display.display(diagram)`.

```python
import qamomile as qm
from qamomile.visualization import draw


@qm.qkernel
def layered(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    for i in range(rounds):
        state = qm.cx(state, control=0, target=1)
    return qm.measure(state)


program = qm.specialize(layered, rounds=2)
diagram = draw(program, loops="body", classical="expressions")
diagram
```

Time runs left to right, including nested helpers and loop iterations. Solid
buses retain quantum types and widths; dashed wires carry classical values.
`vN` identifies one dataflow value within a view, not a physical qubit number.
Branches show exclusive alternatives and explicit result bindings.

Choose `loops="folded"` for the interface only, `"body"` for one symbolic body,
or `"expanded"` for known normally yielding iterations. Expansion rejects
unknown ranges and early exits rather than assuming an execution path.
Use `classical="expanded"` to expose native classical operations, and
`inline_depth=0` to keep helper calls boxed. `Q[UInt[n]]` and fixed-point
registers retain their numeric interpretations on typed buses.

```python
diagram.save("algorithm.html")
diagram.save("algorithm.svg")
circuit = draw(program, loops="expanded", wires="qubits")
circuit.save("circuit.svg")
```

HTML exports include loop/classical controls, horizontal scrolling, Fit, and
100% display. `to_svg()` returns the selected static view. Concrete qubit lanes
require a specialized `Program`, expanded loops, and support in the existing
native circuit projection; they do not provide a fallback for unsupported
lowering. Large displays have bounded traversal, canvas, and output sizes;
fold loops or reduce call expansion when a diagram exceeds those bounds.

Individual qubit views support more than 64 lanes. `max_nodes` (default 2048,
maximum 16384) bounds both the native projection and the sum of concrete lanes
and instructions. Increase it for a larger view; routing and canvas bounds
still apply. Drawing does not allocate a simulator statevector.

For PNG and vector PDF export, install `qamomile[visualization]` and system
Cairo (for example, `brew install cairo` on macOS or your Linux distribution's
`libcairo2` package). SVG and HTML need no extra dependency. Static export uses
the selected view and retains its gates, branches, colors, and labels.
If Python cannot locate Homebrew's Cairo library on macOS, start Python/Jupyter
with `DYLD_FALLBACK_LIBRARY_PATH` set to the `lib` directory under your Homebrew
prefix (usually `/opt/homebrew/lib` on Apple Silicon).

```python
diagram.save("algorithm.png")
diagram.save("algorithm.pdf")
circuit.save("circuit.png", scale=2.0)
```

`scale` changes PNG resolution or PDF page size and must be positive and finite.
PNG allocations are limited to 32 million pixels and 32768 pixels on either
axis. Reduce `scale` or save SVG/PDF when a raster view exceeds those limits.
