# HUGR execution through Nexus

Install `qamomile[hugr,hugr-qir]` on Python 3.11 or 3.12 to enable H2 conversion. The converter extra
uses the 0.1 series of `hugr-qir`, compatible with this Python implementation's
HUGR 0.16 series. Its QIR validator currently requires Python below 3.13.
Authenticate with `qnexus` and select an active Nexus project
before submitting a job, or pass a project reference in `NexusExecutionOptions`.

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
    "nexus", options=NexusExecutionOptions(system_name="H2-1E")
)
job = program.sample(executor, shots=100)
print(job.result().results)
```

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

This H2 adapter supports Boolean public outputs, including fixed-size
containers of these values. Public floating-point outputs are rejected before
upload because H2 does not support their QIR output operation. Unsigned integer
public outputs are also rejected until their provider result representation
can be decoded reliably. Floating-point and integer calculations used
internally, for example rotation angles and loop indices, remain available.
Other operations must be supported by `hugr-qir` and its H2 QIR validator;
conversion errors stop submission without an automatic retry on another device.

The automated tests exercise real local HUGR-to-QIR conversion and the Nexus
SDK interface with simulated provider responses. They do not authenticate or
submit paid remote jobs. Successful offline validation does not establish that
every generated program is accepted by a live H2 device.
