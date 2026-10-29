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
