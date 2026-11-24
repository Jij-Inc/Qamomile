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

Execution and measurement default to `seed=None`, which leaves randomness to
the endpoint. An integer from `0` through `2**64 - 1` requests a reproducible
simulator stream; zero is an explicit seed too. Endpoints without seed control,
including BraketLocal and managed devices, accept `None` and reject every
integer instead of silently ignoring it. For repeatable local runs, pass an
explicit seed. `execute_batch` advances an integer seed for each row; `None`
stays unspecified for all rows. Measurement groups likewise retain `None` or
derive distinct streams from an explicit integer seed.
`QurationLocal` currently requires an explicit integer: its simulator has no
unseeded execution mode, so it rejects `None` rather than substituting zero.

Bind named `Const` inputs with `qm.specialize(kernel, width=3)`. Pass named
`Param` values at launch with `parameters={"theta": 0.25}`. An artifact's ABI
checks names, types, and shapes. `qm.run` combines the lifecycle and returns a
`RunRecord`; its result is `record.result`. `qm.submit` returns a `SubmittedJob`
with status, cancellation, saved snapshots, and provider metadata. Use
`job.result()` to wait for its result. Sampling endpoints return a `SampleResult`,
and value endpoints return a `ValueResult`.

Rotation and phase angles must be finite. Qamomile checks the final angle
expressions before passing concrete circuit bindings to an SDK, including
`Estimator` execution and numeric circuit extraction. A nonfinite final angle
is rejected whether supplied directly or produced by arithmetic. General `Real` values
may still contain `NaN` or infinity when they are not consumed as angles;
for example, an infinite input can produce the valid angle `1.0 / inf == 0.0`.
Older saved circuit artifacts containing rotations require recompilation from
source so they carry the angle checks.

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
    keyed_artifact, on=qm.Reference(), shots=8, seed=0,
    parameters={"angles": keyed_angles.pack({(3, 7): qm.pi, (8, 2): 0.0})},
)
unseeded_result = qm.execute(
    keyed_artifact, on=qm.Reference(), shots=8, seed=None,
    parameters={"angles": keyed_angles.pack({(3, 7): qm.pi, (8, 2): 0.0})},
)
```

Both calls produce `(1,)` because this rotation is deterministic. The second
call leaves the random stream unspecified. Reuse `keyed_artifact` with newly
packed values for later launches. The selected endpoint must support the ordinary Param Vector ABI.

Sampled bit registers have canonical tuple-keyed `counts`; scalar and aggregate
logical values are available through `values`. SDK adapters normalize native
bit order. A quantum component returns owned state to another kernel; an
executable entry measures it and returns classical results to the host.

Choose compilation target and execution endpoint separately. With the
`qamomile[qiskit]` extra, use `Qiskit()` as target and `Aer()` as endpoint from
`qamomile.qiskit`. Other SDKs have their own namespaces and capability contracts.
Keep a small local Reference witness useful for checking the algorithm.

For IBM cloud execution, use `IBMRuntime` from `qamomile.qiskit`. Its `backend`
argument is the backend object returned by `service.backend(name)`, not the
device-name string. The `qamomile[qiskit]` extra requires
`qiskit-ibm-runtime>=0.51`, which supports server-side dry runs. With an
authorized cloud backend and the measured `program` prepared above:

```python
from qamomile.qiskit import IBMRuntime

preview = IBMRuntime(backend=backend, dry_run=True)
checked = qm.run(program, on=preview, shots=128)
report = checked.result
details = report.metadata()
print(report.status)
print(details["usage"])
```

Dry runs validate the submission in IBM's cloud without consuming QPU time.
They require an eligible IBM plan and support job or Batch mode, not Session
mode or a local simulator. IBM returns randomized mock data, not simulated
measurements. Qamomile returns a `ValidationResult` containing `dry_run`,
`job_id`, and `usage` metadata rather than sampled counts. Usage describes the
estimated consumption of the corresponding real job, with seconds in
`estimated_quantum_seconds` and resource units in `estimated_resource_units`.
Unavailable or unfinished estimates remain `None`; inspect the provider's
usage status in `details["metrics"]` instead of treating them as zero.

Use `qm.execute(artifact, on=preview, shots=128)` for an existing matching
artifact, or `qm.submit(...)` to obtain a job before completion. For real
execution, construct `IBMRuntime(backend=backend)` with the same compilation
settings and reuse `checked.artifact` while its target still matches. Dry-run
snapshots must be restored through a dry-run endpoint. Expectation estimation,
segmented execution that uses measurement results to schedule later circuits,
and zero-qubit paths that complete locally are unsupported for dry runs and
fail before submission. An unsupported dry run never falls back to real
execution.

`artifact.report.stages` records the steps performed by compilation and the
selected target. Built-in targets record standard QCore optimization when they
prepare their executable; a custom target supplies its own performed stages.
For repeated runs, compile once and pass the artifact to `qm.execute` with
each new parameter set. For example, when the width of `row` is fixed and
`row[len(row) - 1]` supplies a rotation angle, a circuit target can emit one
parameterized circuit and bind the angle at each execution.
A retained launch-materialization recipe records planning first, and its child
artifact records the lowering performed once launch parameters are known.

For explicit pass selection and independently distributed compiler extensions,
read [Compiler optimization](compiler-optimization.md). `optimizer=` selects
optional QCore passes; target emission and execution remain in the ordinary
lifecycle.

For LLVM text or bitcode export, use `QIR()` from `qamomile.qir` with the
`qamomile[qir]` extra. Read [QIR export](qir.md) for the Base Profile restrictions,
output ordering, and a complete example for an external QIR consumer.

For direct HUGR execution, choose `Selene()` or `Helios()` from `qamomile.hugr`.
Compile with `endpoint.target` and execute on that endpoint. HUGR records returned
measurements and computed Boolean values directly in source order, including
repeated values and branches with different measurement counts. Selene and Helios
share this output path without allocating qubits solely to report results.

For a quantum component, `qm.qiskit.to_circuit(program)` retains symbolic angles.
Pass `parameters={"theta": 0.25}` to obtain a numeric circuit. Specialize `Const`
arguments before extraction, including those forwarded to controlled helpers;
they do not belong in the runtime parameter dictionary. Include any `Param`
values that select branches or circuit topology in that dictionary.

Use canonical `Program.to_bytes`/`Program.from_bytes` for program persistence;
inspect `qamomile.persistence` for artifact export and supported target formats.
Regenerate saved Programs whose measurements return `Shot[Bool]` from source
with `Shot[Bit]` annotations. Previously saved execution results retain their
stored interpretation, including Boolean results.
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
streams; different child occurrences receive different streams. With `seed=None`,
children leave their random streams to the endpoint and repetition is not promised.

Export/restore the reusable artifact through `qamomile.persistence` as usual.
A segmented artifact contains a source recipe rather than one SDK circuit, so
SDK `load_circuit(artifact)` inspection is unavailable for it. Use the original
program's `to_circuit` inspection for its quantum operations. Completed logical
results have portable snapshots; in-flight segmented snapshots currently raise
`Q4470` explicitly. Retain the original Job across retrieval timeouts rather
than submitting it again. Runtime loop/recursion continuations, preparation
motion, automatic provider shot chunking, early-return continuations, and
arbitrary ordered classical checks within live quantum state are outside the initial segmented route.
