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

Runtime bindings, typed sampling, expectation estimation, cancellation, and
job restoration use the existing HUGR execution interface. H2 restoration
references retain output labels so a later process can recover the same typed
values without converting or submitting the program again. Existing Helios
references remain compatible.

## Save exact H2 bytes before submission

Use `prepare_h2_submission(package, shots=1024)` with the same bound,
zero-argument HUGR `Package` with recorded outputs that `HugrExecutor.submit`
accepts. Preparation runs locally and preserves the original package. It
normalizes shot reads, applies the existing table-proof fallback when required,
and returns the final QIR bytes and complete tagged-result recovery context.
It requires no Nexus client, authentication, or project.

Save the binary bytes separately from the JSON metadata before submitting:

```python
import json
from pathlib import Path

from hugr.package import Package
from qamomile.hugr import (
    HugrExecutor,
    NexusExecutionOptions,
    PreparedH2Submission,
    prepare_h2_submission,
)

package = Package.from_bytes(Path("bound-instrumented.hugr").read_bytes())
prepared = prepare_h2_submission(package, shots=100)
Path("program.bc").write_bytes(prepared.qir)
Path("prepared.json").write_text(json.dumps(prepared.to_dict()))
request = {"name": "my-h2-run-001", "target": "H2-1E"}
Path("request.json").write_text(json.dumps(request))

executor = HugrExecutor(
    "nexus",
    options=NexusExecutionOptions(system_name=request["target"], project="qamomile-demo"),
)
handle = executor.submit_prepared(prepared, name=request["name"])
print(handle.result())
```

`submit_prepared` sends `prepared.qir` unchanged and uses the saved shot count.
Its required `name` is the complete program and job name, with no random suffix.
The executor supplies the actual device, project, credentials, cost limit, and
polling settings. A compatible H2 executor can submit the same bytes with a
separate name. Syntax-check targets continue to reject programs with required
shot-success guards before authentication or upload. Ordinary `submit` uses
the same preparation and send path while retaining its generated name suffix.

`PreparedH2Submission` is frozen, and its `context` is a detached read-only
mapping. `shots` and `qir_sha256` are derived properties. `to_dict()` returns
version `1`, `qir_sha256`, `context`, and `prepared_sha256`. The combined checksum
covers canonical JSON metadata containing the version, QIR digest, and complete
context. These checksums detect accidental corruption; they provide no proof
of authenticity. `from_dict` validates the schema, checksums, LLVM bitcode
magic, and recovery contract locally. Unknown string context fields survive
round trips; unsupported versions and extra top-level fields fail validation.

If the submission response is lost, find the existing execute-job ID through
your own Nexus lookup and attach it to the saved payload:

```python
loaded = PreparedH2Submission.from_dict(
    json.loads(Path("prepared.json").read_text()),
    qir=Path("program.bc").read_bytes(),
)
request = json.loads(Path("request.json").read_text())
reference = loaded.reference(job_id="caller-found-execute-job-id", target=request["target"])
restored = executor.retrieve(reference)
print(restored.result())
```

Use the device that actually received the job as `target`. Each submission
attempts a new upload and execute request. Qamomile does not deduplicate calls
by name. Resolve an uncertain response through lookup before making another
submission.
Restoration uses the saved output labels and guard metadata, checks every
required success marker, and hides the private marker from returned rows.
This low-level interface returns one tag-to-value dictionary per shot.
Typed `HugrExecutable.restore` still requires its existing ABI and job snapshot
metadata; preparation from a bare package does not create that metadata.

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
