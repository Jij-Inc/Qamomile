# Execute and reuse a program

For the Bell kernel saved in `bell.py` from SKILL.md:

```python
import qamomile as qm

module = qm.build("bell.py")
program = qm.specialize(module["bell"])
artifact = qm.compile(program, to=qm.Reference())
result = qm.execute(artifact, on=qm.Reference(), shots=128, seed=7)
print(result.counts)
```

Bind named `Const` inputs with `qm.specialize(kernel, width=3)`. Pass named
`Param` values at launch with `parameters={"theta": 0.25}`. An artifact's ABI
checks names, types, and shapes. `qm.run` combines the lifecycle and returns a
`RunRecord`; its result is `record.result`. `qm.submit` returns a `Job`.

For fixed dictionary keys with values that change at launch, use a host
`ParameterMap`. Bind its `indices` as a Const Map and pass `pack(values)` as a
Param Vector. The declared key order is stable even when dictionary insertion
order changes; missing and extra keys fail before execution. The native ABI
still checks value types. This does not require runtime Map support.

```python
from qamomile.parameters import ParameterMap

keyed_angles = ParameterMap(((8, 2), (3, 7)))

@qm.qkernel
def keyed_rotation(
    keys: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Int, 2]],
    angles: qm.Param[qm.Vector[qm.Real, 2]],
) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), angles[keys[(3, 7)]]))

keyed_program = qm.specialize(keyed_rotation, keys=keyed_angles.indices)
keyed_artifact = qm.compile(keyed_program, to=qm.Reference())
keyed_result = qm.execute(
    keyed_artifact, on=qm.Reference(), shots=8, seed=7,
    parameters={"angles": keyed_angles.pack({(3, 7): 3.141592653589793, (8, 2): 0.0})},
)
```

This produces `(1,)`. Reuse `keyed_artifact` with newly packed values for later
launches. The selected endpoint must support the ordinary Param Vector ABI.

Sampled bit registers have canonical tuple-keyed `counts`; scalar and aggregate
logical values are available through `values`. SDK adapters normalize native
bit order. A quantum component returns owned state to another kernel; an
executable entry measures it and returns classical results to the host.

Choose compilation target and execution endpoint separately. With the
`qamomile[qiskit]` extra, use `Qiskit()` as target and `Aer()` as endpoint from
`qamomile.qiskit`. Other SDKs have their own namespaces and capability contracts.
Keep a small local Reference witness useful for checking the algorithm.

`artifact.report.stages` records the steps performed by compilation and the
selected target. Built-in targets record standard QCore optimization when they
prepare their executable; a custom target supplies its own performed stages.
A retained launch-materialization recipe records planning first, and its child
artifact records the lowering performed once launch parameters are known.

For explicit pass selection and independently distributed compiler extensions,
read [Compiler optimization](compiler-optimization.md). `optimizer=` selects
optional QCore passes; target emission and execution remain in the ordinary
lifecycle.

For LLVM text or bitcode export, use `QIR()` from `qamomile.qir` with the
`qamomile[qir]` extra. Read [QIR export](qir.md) for the Base Profile restrictions,
output ordering, and a complete example for an external QIR consumer.

For a quantum component, `qm.qiskit.to_circuit(program)` retains symbolic angles.
Pass `parameters={"theta": 0.25}` to obtain a numeric circuit. Specialize `Const`
arguments before extraction, including those forwarded to controlled helpers;
they do not belong in the runtime parameter dictionary. Include any `Param`
values that select branches or circuit topology in that dictionary.

Use canonical `Program.to_bytes`/`Program.from_bytes` for program persistence;
inspect `qamomile.persistence` for artifact export and supported target formats.
A missing native compiler is resolved by the compatible package dependency.
Remote execution uses the user's selected provider and execution authorization.

For a managed Braket task, retain the original artifact and runtime arguments
alongside `job.snapshot().to_bytes()`. With `qamomile[braket]`, restore through
the same configured `BraketDevice` endpoint:

```python
snapshot = qm.ExecutionSnapshot.from_bytes(saved_snapshot)
restored_job = endpoint.restore(
    snapshot,
    artifact=artifact,
    parameters=original_parameters,
)
result = restored_job.result()
```

Restoration attaches to the existing AWS task and checks the device, artifact
identity, and original parameters before contacting the SDK. Newly captured
references include decoding identities and shots; older task-ID-only references
lack this context and are rejected. Keep AWS credentials and sessions in the
configured device, outside saved artifacts and snapshots. Restoring a remote
task requires the user's authorization for provider access.


Circuit targets may compile a qkernel to `qamomile.shot-execution.v1` when
source-order frontiers have no live quantum state. Use ordinary `compile`,
`submit`, `execute`, and `run` with the matching endpoint. Retrieve a submitted
job's result to drive its dependent children. For 1000 logical shots, upstream
counts of 700/300 cause downstream submissions of 700/300. The final joint
result still contains exactly 1000 shots; an inactive quantum branch consumes
no downstream shots. Inactive measurement slots are normalized by native
postprocessing, so different branches keep the same joint source-row layout.
A branch with quantum work but no readout remains fused when supported.

`result.execution_statistics` records `logical_shots`,
`physical_segment_shots`, and `submissions`; `RunRecord.resources` includes the
observed physical demand. Source symbolic costs continue to describe one
logical trajectory. Reusing an explicit seed repeats the same plan's simulator
streams; different child occurrences receive different streams. The default seed also
uses distinct repeatable child streams on seeded local endpoints; managed
endpoints retain their provider's seed contract.

Export/restore the reusable artifact through `qamomile.persistence` as usual.
A segmented artifact contains a source recipe rather than one SDK circuit, so
SDK `load_circuit(artifact)` inspection is unavailable for it. Use the original
program's `to_circuit` inspection for its quantum operations. Completed logical
results have portable snapshots; in-flight segmented snapshots currently raise
`Q4470` explicitly. Retain the original Job across retrieval timeouts rather
than submitting it again. Runtime loop/recursion continuations, preparation
motion, automatic provider shot chunking, early-return continuations, and
arbitrary ordered classical checks within live quantum state are outside the initial segmented route.
