# Compose library algorithms

Inspect `qamomile.stdlib` for QFT/QPE, Grover, state preparation, block encoding,
QSVT, lookup, reversible arithmetic, and variational building blocks. Factories
produce ordinary typed components that are called inside qkernels.

Use `on_qubits(process, width=..., indices=(...))` to apply a closed unary
unitary helper to noncontiguous positions. Helper coordinate zero corresponds
to the first supplied index, including reverse-order selections. The lifted
kernel consumes and returns the whole register in its original order. Owned
splits and concatenations perform the selection without additional SWAP gates.

```python
from qamomile.stdlib import on_qubits

@qm.qkernel
def flip_pair(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.x(state)

flip_evens = on_qubits(flip_pair, width=4, indices=(0, 2))

@qm.qkernel
def noncontiguous_sample() -> qm.Shot[qm.Bits[4]]:
    return qm.measure(flip_evens(qm.qubits(4)))
```

This example produces `(1, 0, 1, 0)`. Use a nonempty tuple of unique in-range
indices and an ambient width from 1 through 128. Wrap additional fixed helper
arguments in a closed unary qkernel. The generated component supports ordinary
adjoint and control transformations; runtime-selected positions and borrowed
quantum slice assignment remain outside this API.

Use `approximate_qft(register, size, max_distance)` to omit controlled rotations
whose control-target distance exceeds a Const cutoff. The cutoff changes the
operator; values at least `size - 1` recover exact QFT, and nonpositive values
leave only Hadamards and reversal swaps. `qm.adjoint` gives the matching inverse:

```python
from qamomile.stdlib import approximate_qft

@qm.qkernel
def approximate_roundtrip() -> qm.Shot[qm.Bits[3]]:
    register = qm.prepare((1, 0, 1))
    register = approximate_qft(register, 3, 1)
    register = qm.adjoint(approximate_qft, register, 3, 1)
    return qm.measure(register)
```

For two-term second-order Suzuki evolution:

```python
from qamomile.stdlib import suzuki_trotter

evolve = suzuki_trotter(term_count=2, order=2, steps=4)

@qm.qkernel
def state(terms: qm.Const[qm.Vector[qm.Observable, 2]], time: qm.Param[qm.Real]) -> qm.Q[qm.Bits[2]]:
    register = qm.h(qm.qubits(2), target=0)
    return evolve(register, terms, time)
```

Construct Hamiltonians/Pauli terms and dense decompositions on the host with
`qm.X`, `qm.Z`, or `qm.linalg.HermitianMatrix`. Library contracts check commuting
terms, normalization, particle sectors, and polynomial certificates where needed.

For optimization, inspect `QAOAExperiment`, `FQAOAExperiment`, `GASExperiment`,
`BinaryModel`, and `OMMXProblem` in `qamomile.optimization`. Host optimizers and
model extraction surround the same compiled quantum program. Use model adapters
for source-variable decoding and objective evaluation.

For threshold filtering, use `QSVTEigenstateFilterExperiment(model, mu, ...)`.
The positive sign response retains energies below `mu`. The complete filter
includes real-response extraction and projector signal qubits; its run reports
both the full execution and successful decoded assignments. `samples` is `None`
when no shot succeeds. Reuse a `synthesize_sign_filter(...)` certificate through
`polynomial=` for several thresholds. Synthesis uses Qamomile's required
`pyqsp` dependency, and Rust independently verifies the external phase candidate.

This one-variable example has normalized shifted eigenvalues `-1` and `+1`,
so the exact linear QSP response already separates them without synthesis:

```python
from qamomile.optimization import BinaryModel, QSVTEigenstateFilterExperiment
from qamomile.stdlib import QSVTPhases

filter_model = BinaryModel.from_hubo({(0,): 1.0})
filter_phases = QSVTPhases.from_qsp_wx((0.0, 0.0), odd=True)
filter_experiment = QSVTEigenstateFilterExperiment(
    filter_model, mu=0.5, phases=filter_phases
)
filter_endpoint = qm.Reference()
filter_compiled = filter_experiment.compile(to=filter_endpoint.target)
filter_result = filter_compiled.run(on=filter_endpoint, shots=32, seed=7)
assert filter_result.samples is not None
assert filter_result.samples.lowest.energy == 0.0
```

For an existing Hermitian `BlockEncoding`, compose
`qsvt_filter_projector(encoding, polynomial=...)` or sample the uniform input
with `qsvt_filter_probe(encoding, polynomial=...)`. Explicit `phases=` are in
projector convention; use `QSVTPhases.from_qsp_wx` to convert `Wx` coordinates.
Raw phases carry no sign-approximation certificate. For OMMX models, filter
`OMMXProblem.model`, then evaluate the successful assignments separately with
the original problem adapter.
`GASExperiment(..., approximate_real_coefficients=False)` preserves the original
binary64 coefficients, including small nonzero values. Native exact arithmetic
chooses a common power-of-two multiplier, derives the strict integer threshold,
and computes a safe two's-complement register width. Inspect `effective_model`,
`quantization_scale`, and `resolved_output_bits`; this path does not round
coefficients or emit a quantization warning. Encodings above 54 objective bits
or with nonfinite scaling fail before circuit construction. Decimal literals
are interpreted as their binary64 values. The default approximation path and
its explicit `quantization_parameter` remain available.

```python
from qamomile.optimization import BinaryModel, GASExperiment

exact_gas = GASExperiment(
    BinaryModel.from_hubo({(4,): 0.25, (9,): 0.5}),
    threshold=0.2,
    approximate_real_coefficients=False,
)
exact_gas_run = exact_gas.run(on=qm.Reference(), shots=8, seed=7)
```

Only assignment `((4, 0), (9, 0))` is below the threshold in this example;
one Grover iteration returns it deterministically, with scale `4.0`.

`QAOAExperiment(layers=0)` samples the uniform initial state. Supply empty angle
sequences; its compiled program has no angle parameters. Negative layer counts
are rejected, while positive layers keep one gamma and beta per layer.

```python
from qamomile.optimization import BinaryModel, QAOAExperiment

initial = QAOAExperiment(BinaryModel.from_hubo({(4,): -1.0}), layers=0)
initial_run = initial.run(
    on=qm.Reference(), gammas=(), betas=(), shots=8, seed=7
)
```
