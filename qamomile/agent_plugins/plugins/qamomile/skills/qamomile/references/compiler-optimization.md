# Extend compiler optimization

Use `optimizer=` on `qm.compile` or `qm.run` to select an explicit immutable
multi-level optimization pipeline. Import compiler extension APIs from
`qamomile.compilation`; `qamomile.optimization` contains mathematical optimization
workflows and models.

```python
import qamomile as qm
from qamomile.compilation import (
    OptimizationContext,
    OptimizationPipeline,
    OptimizationProgram,
    StandardSimplification,
)

class HadamardConjugation:
    name = "example.hadamard_conjugation"
    version = "1"
    configuration = b"HZH=X:v1"

    def run(
        self, program: OptimizationProgram, context: OptimizationContext
    ) -> OptimizationProgram:
        return program.rewrite(
            ("quantum.h", "quantum.z", "quantum.h"),
            ("quantum.x",),
            context=context,
        )

@qm.qkernel
def conjugation_sample() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = qm.h(state)
    state = qm.z(state)
    state = qm.h(state)
    return qm.measure(state)

base = OptimizationPipeline.standard(seed=7)
optimizer = base.append(HadamardConjugation())
artifact = qm.compile(conjugation_sample, to=qm.Reference(), optimizer=optimizer)
result = qm.execute(artifact, on=qm.Reference(), shots=4, seed=7)
assert result.counts == {(1,): 4}

configured = base.replace(
    "qcore.simplify", StandardSimplification(inverses=False)
)
```

The standard public pass is `qcore.simplify`. `append`, `insert_before`,
`insert_after`, and `replace` return a new pipeline; the base remains unchanged.
An explicit pipeline runs only its listed passes. `OptimizationPipeline()` skips
optional optimization while retaining mandatory native verification. Omitting
`optimizer` retains ordinary built-in target preparation. Names beginning with
`qcore.`, `family.`, `realization.`, and `circuit.` are reserved for built-in implementations; custom passes use their own
namespaced identifiers. Invalid or duplicate IDs fail before invocation.

A pass implements `run(program, context)` and exposes a stable `name`, `version`,
and deterministic `configuration: bytes`. Change its version or configuration
when its behavior changes. The context supplies the target fingerprint, supported
requirements, deterministic optimization seed, and native rewrite work limit.
Treat these as immutable inputs. `max_work` bounds each native rewrite/simplify
invocation; extension code is trusted Python and is responsible for its own
additional computation and network limits.

Reuse `program.simplify(context=context, options=StandardSimplification(...))`
inside an extension instead of copying the standard algorithm. Built-in options
control identities, inverses, structural round trips, vacuous roots, and dead
constants. `program.rewrite` handles native traversal, SSA relinking, resealing,
and full verification. Its initial rule language supports contracting nonempty
sequences of at most sixteen consecutive angle-free unary gates on identical
selectors. It traverses structured blocks but does not move gates across
constants or effectful operations. Expanding synthesis and general IR editing
are outside this initial author API.

An independently distributed native module or hosted service can consume
`program.to_bytes()` and return `OptimizationProgram.from_bytes(result_bytes)`.
Rust verifies the graph and cleanup claims and rejects changed public entry ABIs.
The pass author remains responsible for equivalence, including phase under
coherent control, result distributions, and invalid-input errors. Graph
verification alone does not prove equivalence. This API exposes exact passes;
approximate strategies require a separately defined error-budget contract.

Compilation retains the original source and logical resource study. Receipts
identify the ordered passes, versions, configuration, target policy, seed, and
prepared result. Artifact identities distinguish different selected pipelines.
Artifacts persist the native preparation without callbacks or credentials;
execution, launch materialization, and artifact restoration do not invoke the
extension again. A pass failure propagates; compilation never silently substitutes
the standard optimizer. The core provides no automatic cloud submission or job
service in this initial implementation.

`optimizer` is now a reserved compile/run keyword. Specialize a kernel's Const
argument named `optimizer` explicitly before calling `compile`. An existing
compiled artifact passed to `run` cannot select another optimizer.

## Optimize before and after specialization

The multi-level factory composes symbolic common-expression elimination (F),
common-expression elimination plus standard QCore simplification (H), and
phase-exact local circuit simplification (C). Bounded selector-count addition
selection (R) is opt-in with `multilevel(realization=RealizationSelection())`.
The target owns device synthesis/routing (T); Qiskit optimization level 3
is an ordinary OSS target option. Keep these levels distinct from Const/Param/Shot.

```python
import qamomile as qm
from qamomile.compilation import OptimizationPipeline

@qm.qkernel
def symbolic_sample(n: qm.Const[qm.Int], theta: qm.Param[qm.Real]) -> "qm.Shot[qm.Bits[n]]":
    state = qm.qubits(n)
    for i in range(n):
        a = theta * 2.0
        b = theta * 2.0
        state = qm.rx(state, a, target=i)
        state = qm.rz(state, b, target=i)
    state = qm.h(state, target=0)
    state = qm.z(state, target=0)
    state = qm.h(state, target=0)
    return qm.measure(state)

multilevel = OptimizationPipeline.multilevel(seed=7)
symbolic = multilevel.optimize_family(symbolic_sample)
assert symbolic.module.inspect().count("classical.mul") == 1
multilevel_artifact = qm.compile(
    symbolic_sample, n=1, to=qm.Reference(), optimizer=multilevel
)
multilevel_result = qm.execute(
    multilevel_artifact, on=qm.Reference(), parameters={"theta": 0.0}, shots=4
)
assert multilevel_result.counts == {(1,): 4}
```

`optimize_family` requires no Const binding and does not unroll symbolic loops.
Family authors implement `FamilyOptimizationPass.run(OptimizationFamily, context)`.
Use `append_family` or `replace_family("family.common-expressions", pass)` to
change that stage while retaining other defaults. The native
`eliminate_common_expressions(context=context)` helper is available on both
`OptimizationFamily` and `OptimizationProgram`. Family providers can also return
`OptimizationFamily.from_bytes(payload, entrypoint)`; the native driver verifies
all callable declarations. Equivalence for every admitted binding remains the
provider's obligation.

Use `multilevel(family=False)` for an already specialized Program. A requested
family pass on a Program or ResourceStudy is rejected. Structured HUGR emission
supports `multilevel(realization=None, circuit=None)`; requesting the current
late circuit policies on that adapter fails before any pass runs. Circuit targets,
Reference, and QIR consume the portable late policies. QIR does not thereby gain
finite-family or shot-execution support.

Without a policy or explicit choice, portable `add_into` picks the candidate
with the fewest estimated CNOTs for its width and surrounding coherent controls:
`add.ripple` (ancilla-free ripple carry, 16n-13 CNOTs; only its 2n-1 target
CNOTs inherit controls), `add.qft` for three or four bits, and `add.increments`
for at most two. A coherently controlled `add_into` also replaces a default
native choice with this portable choice; explicit `add.native` is kept.
`RealizationSelection` instead compares schedules by exact generated selector
count, with step count as a tie-breaker. This is a compiler/emitter work metric,
not a fidelity/latency prediction, and it can select a much larger circuit
(selectors count a multi-controlled X linearly). Target-native choices and
explicit `RealizationOptions` take precedence.

`CircuitSimplification` (version 2) moves each gate back past earlier gates
that provably commute with it: on every shared wire both act diagonally
(phases, controls) or both lie in the Pauli-X algebra (X, RX, CX/CCX targets).
It then cancels inverse pairs, fuses exact phase gates (T T = S, Z S = Sdg),
merges same-axis rotation angles (folding literal sums and building symbolic
sums), and removes zero rotations. Top-level uncontrolled SWAPs become wire
relabeling of later gates and measurements; the net permutation is restored
with SWAPs before regions, cuts, numeric checks, and live result wires. It
preserves global phase, controls, measurements, classically conditioned gates,
structured regions, numeric-check positions, and segment boundaries. It
optimizes each shared retained body once and keeps arithmetic recipes lazy.
Its work bound includes shared retained bodies, counted once per projection.
Artifacts recorded with version 1 keep its adjacent-instruction rules.

R/C currently use versioned native strategies rather than arbitrary Python late
callbacks. Include or omit them with pipeline constructor arguments; extending
the strategy catalog requires a native implementation. Stored artifacts retain
these policies for deferred family/launch/shot projection without executing an
extension again. Reports list a late stage only when the circuit projection
actually ran it; deferred projections may execute after compilation.

All presets use quantum inverse equality on the defined execution domain:
`U` followed by its exact adjoint can cancel without proving totality or
termination. Before rewriting, native diagnostics check the original selected
source with known arguments and reject proven domain failures (for example a
known out-of-range selector or division by zero). Unknown arguments or paths,
unsupported diagnostic constructs, and analysis-budget exhaustion do not block
cancellation. Therefore erased calls need not retain runtime errors or divergence
for inputs that were unknown during compilation. An empty pipeline retains calls
and their runtime checks, while still diagnosing known source-domain errors.
Diagnostics have a separate bound of 10,000 visits and 64 nested frames.

HUGR target lowering retains typed inverse helpers with shared and recursive
calls, including approximate QFT and terminal owned conditionals. Their ABI
preserves classical parameter stages and dependent quantum types. Structured
inversion currently requires one owned carry and unchanged classical carries;
unsupported forms fail closed.
`max_work` is a per-native-invocation bound, not a whole-compilation time limit.
Generic cross-stage candidate search, a global search budget, approximate passes,
and a built-in paid/cloud transport are not supplied by this API.

## Read device facts and program intent

`context.profile` is an immutable `TargetProfile` when the target supplies one
and `None` otherwise. Rust validates and canonicalizes it: the `topology` facet
holds the qubit count, coupling edges, and native operation names; the optional
`calibration` facet holds finite per-qubit and per-edge rows with units; the
optional `logical` facet holds code family, distance, cycle time, and magic-state
cost. `qm.qiskit.Qiskit(backend=...)` and `QiskitBackend(backend)` derive it with
`qm.qiskit.profile_from_backend` when the backend target has a coupling map, and
an explicit `profile=` overrides the derived one. The profile digest enters the
target fingerprint; `CapabilitySnapshot.require` ignores it, so an artifact
compiled against one calibration still executes on another. Reference, HUGR,
QIR, and basis-gates-only Qiskit targets expose `None`.

`Function.intent` and `Operation.intent` are untrusted provenance rows that never
change meaning: the semantic digest erases them, and no verifier, equation, or
lowering reads them. The frontend fills `origin` with the kernel that declared
each linked function (for example `qamomile.stdlib.variational.qaoa_layers`) and
`doc` with its docstring, cut to 4000 bytes; `qm.qkernel(retain_docstrings=False)`
or `build_source(..., retain_docstrings=False)` omits `doc`. `Program.with_intent`
merges rows such as `problem`, `variable_map`, `tolerance.objective`, and
`tolerance.error` onto the entry; Rust re-seals the program and bounds intent to
16 rows of lowercase dotted names and 4096 encoded bytes each (`Q1060`). Intent
enters persisted bytes and artifacts, and `inspect()` prints it as `intent{...}`.
A pass that replaces an operation leaves the replacement without intent.

```python
from qamomile.compilation import (
    CouplingEdge,
    OptimizationContext,
    OptimizationPipeline,
    OptimizationProgram,
    TargetProfile,
    TargetTopology,
)

profile = TargetProfile(
    "qiskit.qpy.v1",
    TargetTopology(
        3, (CouplingEdge(0, 1, True), CouplingEdge(1, 2)), ("rz", "sx", "x"), ("cx",)
    ),
    backend_name="fake_line",
)
assert TargetProfile.from_bytes(profile.to_bytes()) == profile

class ProfileAware:
    name = "example.profile_aware"
    version = "1"
    configuration = b"v1"

    def run(
        self, program: OptimizationProgram, context: OptimizationContext
    ) -> OptimizationProgram:
        entry = qm.Program.from_bytes(program.to_bytes()).entry
        self.seen = (context.profile, dict(entry.intent).get("problem"))
        return program

plain = conjugation_sample.specialize()
tagged = plain.with_intent(problem="sha256:maxcut-3", **{"tolerance.objective": 0.05})
assert tagged.semantic_digest == plain.semantic_digest
aware = ProfileAware()
tagged_artifact = qm.compile(
    tagged, to=qm.Reference(), optimizer=OptimizationPipeline((aware,))
)
assert aware.seen == (None, "sha256:maxcut-3")
```
