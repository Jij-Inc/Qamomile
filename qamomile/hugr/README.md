# HUGR execution through Nexus

Install `qamomile[hugr]` on Python 3.11 or 3.12 to enable H2 conversion. The `hugr`
extra automatically includes the 0.1 series of `hugr-qir`, compatible with this
Python implementation's HUGR 0.16 series. The converter is omitted on Python
3.13 and later because its QIR validator currently requires Python below 3.13.
Qamomile authenticates through Nexus when you first submit or restore a job.
Saved credentials are reused; an initial login may require browser approval.
There is no need to import or call `qnexus` in normal execution code.

Use the same executor class for Helios and H2:

```python
import qamomile.circuit as qmc
from qamomile.hugr import HugrExecutor, HugrTranspiler, NexusExecutionOptions


@qmc.qkernel
def bell() -> tuple[qmc.Bit, qmc.Bit]:
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


program = HugrTranspiler().transpile(bell)
executor = HugrExecutor(
    "nexus",
    options=NexusExecutionOptions(system_name="H2-1E", project="qamomile-demo"),
)
job = program.sample(executor, shots=100)
print(job.result().results)
```

`project` accepts a project name or an existing native Nexus project reference.
A name is looked up or created on the first submission. If omitted, Qamomile
uses the active Nexus project or looks up or creates `qamomile` when none is
active. The SDK's active project is left unchanged. Restoring an existing job
authenticates without creating a project or submitting another job.

Change `system_name` to `Helios-1E` for Helios emulation or `H2-1` for H2
hardware. Helios receives HUGR directly. H2 receives QIR bitcode converted and
validated locally before upload. The legacy executor destination `"helios"`
also continues to work as a Nexus alias, including with H2 options.

For native provider settings, pass `qnexus.HeliosConfig(system_name=...)` or
`qnexus.QuantinuumConfig(device_name=...)` as `backend_config`. The device in
that config takes precedence over `NexusExecutionOptions.system_name`.
Supported H2 names are numbered hardware, emulator, and syntax-check targets
such as `H2-1`, `H2-1E`, and `H2-1SC`, plus `H2-Emulator`. Unsupported device
families are rejected before upload.

Select an H2 syntax checker with the same `sample`, `run`, or executor `submit`
interface used for hardware and emulators. Shot-dependent programs retain the
same shot-number normalization, runtime range guard, and private success output
in their submitted QIR. Local conversion and range validation still happen
before upload.

H2 syntax-check jobs provide compiler status and cost instead of measurement
results. Refresh their provider state with `job.status()` or `job.raw_status()`;
`job.metadata()` then includes `cost` when the provider supplies it. Cancellation
and `job.snapshot()` work through the existing job interface. Expectation jobs
group each measurement job's metadata under `children`. For all H2
syntax-check jobs, `job.result()` raises `ExecutionError` without downloading
synthetic measurements or changing a completed job to failed. This behavior
also applies after restoration, using the target saved in the snapshot.
Hardware and emulator results continue to require the private success output
for every guarded shot.

Runtime bindings, typed sampling, expectation estimation, cancellation, and
job restoration use the existing HUGR execution interface. H2 restoration
references retain output labels so a later process can recover the same typed
values without converting or submitting the program again. Existing Helios
references remain compatible.

## Inspect and save the latest H2 QIR

Ordinary `program.sample(...)` and `program.run(...)` retain the exact H2 QIR
sent to Nexus. `program.artifact` remains the original HUGR package.
`program.qir` and `program.qir_info` are read-only properties, initially `None`.
For one payload, `qir` is `bytes`. For a multi-term expectation it is an ordered
`tuple[bytes, ...]` containing every payload in that logical operation.

Preparation binds runtime arguments, normalizes H2 shot numbers, and applies
range guards and the existing table-proof fallback when needed. All payloads,
full generated names, targets, and typed restoration context are ready before
authentication, project lookup, upload, or execution starts. A complete
operation replaces one immutable latest record. Local preparation or validation
failure preserves the previous record. A network failure retains the new one.
Helios, Selene, and constant-only expectations leave the last H2 record unchanged.
This record describes the last successfully prepared H2 operation, even if no
remote job was accepted.

Read the properties after the call returns or after catching its exception.
Qamomile does not save files or pause execution for caller-managed persistence.
For the single-payload sample above, save the bytes and detached metadata with
ordinary Python file operations:

```python
import json
from pathlib import Path

Path("program.bc").write_bytes(program.qir)
Path("submission.json").write_text(json.dumps(program.qir_info), encoding="utf-8")
```

`qir_info` returns an independent JSON-compatible dictionary with this shape:

```python
{
    "version": 1,
    "operation": "sample",  # "run" for ordinary runs and expectations
    "shots": 100,  # None for run; each term's count is in its context
    "submissions": [
        {
            "provider": "quantinuum-nexus",
            "name": "qamomile-hugr-<generated-suffix>",
            "target": "H2-1E",
            "qir_sha256": "<digest of the exact uploaded bytes>",
            "context": {
                "shots": "100",
                "package_sha256": "<original bound package digest>",
                "program_format": "qir",
                "qir_outputs": "<JSON output labels, types, and widths>",
                "abi_sha256": "<public input and output contract digest>",
            },
        },
    ],
}
```

Guarded payloads also include `shot_success_tag`, `shot_origin`, and
`h2_package_sha256` in their context. Expectation terms include
`expectation_sha256`, which identifies the reconstruction recipe. All context
keys and values are strings accepted by the existing `ExecutionReference`.
The submissions and QIR tuple use the same term order. The record stores no
runtime argument values or job IDs. Previously read bytes and dictionaries
remain unchanged when another call replaces the latest record. Editing a
returned dictionary cannot change retained data or future requests.

## Reuse QIR from a syntax check

Consecutive compatible H2 calls on the same live program reuse the latest
payloads without QIR conversion. This requires the same operation, bound
instrumented packages, shot counts, term order, and H2 shot-origin convention.
Changing runtime bindings or shot counts generates a new record. Only the
immediately retained operation is available for reuse, with no history or
cache indexed by older arguments.

```python
sc = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1SC"))
checked = program.sample(sc, shots=100)
# Inspect checked.status() and checked.metadata() before continuing.
hardware = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1"))
job = program.sample(hardware, shots=100)
```

The second call reuses the exact QIR but creates a new upload and execute
request. Target, name, cost limit, and provider settings belong to each call.
The recorded name includes the normal generated suffix. Names do not provide
submission deduplication. An SC success confirms syntax checking, not runtime
success of every shot's guard.

## Restore after a lost submission response

After a submission exception, save `program.qir_info` if preparation completed.
Find the existing execute-job IDs through your own Nexus lookup, using the
saved full names and actual targets. Then use the existing typed restoration
interface with the original program and runtime bindings:

```python
from qamomile.circuit.transpiler import ExecutionReference, JobKind, JobSnapshot

info = json.loads(Path("submission.json").read_text(encoding="utf-8"))
matched_job_ids = ["caller-found-execute-job-id"]
references = tuple(
    ExecutionReference(
        provider=item["provider"],
        job_ids=(job_id,),
        target=item["target"],
        context=item["context"],
    )
    for item, job_id in zip(info["submissions"], matched_job_ids, strict=True)
)
snapshot = JobSnapshot(
    kind=JobKind(info["operation"]),
    executions=references,
    shots=info["shots"],
)
restored = program.restore(executor, snapshot, bindings={})
print(restored.result())
```

Pass the original runtime bindings instead of `{}` for a parameterized program.
For an expectation, match every ID to its saved term position. Partial submission
can leave some terms without jobs, and the existing failure cleanup attempts
to cancel earlier accepted jobs. Complete restoration requires every term's
existing job and available results.

Restoration validates the package, public result contract, and expectation
recipe, then retrieves existing jobs without QIR conversion, upload, or execute
requests. It does not need the saved QIR bytes. Checksums identify matching
content but do not authenticate saved metadata. Qamomile does not find jobs
automatically. These properties also do not provide a new-submission interface
for arbitrary QIR loaded from files in a later process.

This H2 adapter supports Boolean public outputs, including fixed-size
containers of these values. Public floating-point outputs are rejected before
upload because H2 does not support their QIR output operation. Unsigned integer
public outputs are also rejected until their provider result representation
can be decoded reliably. H2 supports constant floating-point gate arguments
and runtime integer control flow. Runtime floating-point computations,
runtime-selected floating-point gate arguments, and integer division or
remainder are rejected before upload. Constant bindings can eliminate these
computations before conversion and validation. Use integer predicates to select
gates with constant angles, or target Helios for these operations.
Other operations must be supported by `hugr-qir` and its H2 QIR validator;
conversion errors stop submission without an automatic retry on another device.

The automated tests exercise real local HUGR-to-QIR conversion and the Nexus
SDK interface with simulated provider responses. They do not authenticate or
submit paid remote jobs. Successful offline validation does not establish that
every generated program is accepted by a live H2 device.
