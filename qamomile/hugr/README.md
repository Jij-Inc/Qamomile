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

This H2 adapter supports Boolean public outputs, including fixed-size
containers of these values. Public floating-point outputs are rejected before
upload because H2 does not support their QIR output operation. Unsigned integer
public outputs are also rejected until their provider result representation
can be decoded reliably. H2 supports integer classical computations and constant
floating-point gate arguments. Runtime Float computations, integer division,
and remainder require Helios. Constant bindings may eliminate such computations
before the H2 capability check. The shared upstream QIR validator also accepts
Helios operations, so Qamomile checks these H2 restrictions after conversion.
Other operations must be supported by `hugr-qir` and its QIR validator;
conversion errors stop submission without an automatic retry on another device.

The automated tests exercise real local HUGR-to-QIR conversion and the Nexus
SDK interface with simulated provider responses. They do not authenticate or
submit paid remote jobs. Successful offline validation does not establish that
every generated program is accepted by a live H2 device.


## Device shot context

Inside a qkernel, `qmc.get_current_shot()` returns a device-supplied `UInt`.
It emits one abstract context read, lowered on HUGR to
`tket.qsystem.utils.GetCurrentShot`, and on H2 QIR to
`___get_current_shot`. It has no host counter or random-number semantics.
Repeated reads during one shot are stable, including across ordinary qkernel
calls. The value can drive arithmetic, a classical `if`, or selection from a
fixed-size classical vector. Those branches are runtime device control flow;
they are an exception to the circuit-family requirement that classical branch
predicates be compile-time bindings. Structural values such as the number of
qubits still belong in compile-time `bindings`.

An ordinary `program.sample(executor, shots=B, bindings=...)` submits one
zero-argument program with one set of bindings and requests B repetitions.
On local Selene, the indices are `0, ..., B-1`; each new sample or one-shot
`run` starts again at zero, including when the compiled runner is cached.
The table is a single argument whose contents are fixed for that call. This
is different from a parameter sweep over multiple programs, and the kernel
still describes one repetition, not B concatenated proposals.

For Nexus, Qamomile forwards the native device index without assigning offsets
or renumbering. Quantinuum lists shot-index support for both H2 and Helios in
its [platform specifications](https://docs.quantinuum.com/nexus/trainings/notebooks/basics/qir/index.html#platform-specifications)
and demonstrates table selection in its
[parameterized-shot guide](https://docs.quantinuum.com/systems/trainings/helios/getting_started/parameterized_shot.html).
The supplied sources do not establish how provider-internal splitting,
rescheduling, or retries affect the index. These cases have not been verified
on a remote device. Do not use this index as a globally unique shot identifier
or assume it persists across separately submitted jobs. Qamomile does not
split or automatically retry ordinary sampling submissions. Retrying a sample
call explicitly creates another execution; recovering a saved snapshot retrieves
the existing job and performs no new submission. Expectation estimation can
submit separate measurement programs, each with its own device context; there
is no global counter across terms.

### One table, one proposal per shot

This credential-free example generates masks in the application, passes one
flattened `Vector[UInt]`, and returns the decoded measured Bit vector. The
fixed proposal prepares `|10>`; replace that preparation with an application's
proposal circuit. Mask generation, RNG independence, X placement, and decoding
are application choices. Qamomile supplies generic context and classical
operations, with no twirling-specific API.

```python
import numpy as np
import qamomile.circuit as qmc
from qamomile.hugr import HugrExecutor, HugrTranspiler


@qmc.qkernel
def masked_proposal(
    n: qmc.UInt, masks: qmc.Vector[qmc.UInt]
) -> qmc.Vector[qmc.Bit]:
    shot = qmc.get_current_shot()
    qs = qmc.qubit_array(n, "qs")
    qs[0] = qmc.x(qs[0])  # Fixed proposal before readout masks.
    for i in qmc.range(n):
        index = shot * n + i
        if index < masks.shape[0]:
            mask = masks[index]
            if mask == 1:
                qs[i] = qmc.x(qs[i])
    bits = qmc.measure(qs)
    for i in qmc.range(n):
        index = shot * n + i
        mask = qmc.uint(0)
        if index < masks.shape[0]:
            mask = masks[index]
        bits[i] = bits[i] != mask
    return bits


B, N = 4, 2
rng = np.random.default_rng(17)
flattened_masks = rng.integers(0, 2, size=(B, N)).ravel().tolist()
assert len(flattened_masks) == B * N
program = HugrTranspiler().transpile(
    masked_proposal,
    bindings={"n": N},
    parameters=["masks"],
    parameter_shapes={"masks": (B * N,)},
)
executor = HugrExecutor()  # Local Selene; no authentication or remote job.
counts = program.sample(
    executor, shots=B, bindings={"masks": flattened_masks}
).result()
assert counts.results == [((True, False), B)]
```

The mask table belongs only in `parameters` during transpilation and only in
execution-time `bindings` during sampling. The structural `n` belongs only in
transpile-time `bindings`. No argument appears in both lists at compilation.
The returned `bits[i] != mask` is ordinary kernel classical computation;
counts aggregate the decoded Bit values, using the existing typed-result and
snapshot machinery. There is no raw-shot or dedicated decoding API requirement.

The example deliberately defines an out-of-table fallback: skip that readout X
and decode with mask zero. The application checks the intended table length.
This fallback also makes a provider-supplied unexpected index memory-safe, but
does not establish which remote execution or repetition produced that index.
Choose an application-specific fallback when this behavior is unsuitable.

### Supported indexing and H2 limitations

Runtime selection currently supports nonempty, fixed-size, unsliced classical
vectors, with UInt indices and Bit, UInt, or Float elements. The existing public
parameter rules allow runtime UInt and Float tables; a Bit vector can be bound
at compile time or computed inside a kernel. Runtime quantum array indexing,
dynamic classical stores, and multidimensional runtime indexing remain
unsupported and raise target diagnostics. Classical vector stores at static
indices and persistent array updates in statically unrolled loops are supported
so decoded Bit vectors can be returned. Selection is lowered to forward
branches; code size grows with table length and the number of accesses. Large
tables should be assessed for compilation time and device resource cost.

Without an explicit source-level guard, an out-of-range runtime lookup raises
a device panic on Selene/HUGR; it never wraps or silently returns another row.
The local execution reports failure instead of aggregating a successful prefix
as complete counts. Empty dynamic lookups fail at compilation. Binding shape
mismatches fail before submission. There is no general host-side proof that an
arbitrary index expression fits the table, so the caller must size the table
and avoid UInt overflow in expressions such as `shot * N + i`.

With `hugr-qir==0.1.2`, unguarded bounds failure lowers to an unsupported
`abort()` call, so **H2 conversion rejects that program even when the requested
shots would fit**. The explicit `index < masks.shape[0]` guards in the example
allow the converter to eliminate the unreachable bounds failure and pass both
HUGR and QIR validation. Qamomile does not disable validation or delete failing
paths. More complex guards may still fail to simplify and will fail locally
before upload with the converter's diagnostic.

H2 supports integer predicates, so UInt masks can directly select constant X
gates. There is no need to cast masks to Float rotation angles. Float table
selection can also succeed if conversion produces branches with constant gate
angles; it is accepted only when no runtime Float computation survives.
Runtime Float arithmetic or integer division/remainder is rejected for H2.
Helios HUGR and local Selene retain their broader classical capabilities.
Circuit-family targets (Qiskit, QuriParts, CUDA-Q, and Braket, including services
wrapping them) reject `get_current_shot()` with `TargetCapabilityError` before
emission. No backend silently substitutes zero.

### Offline compatibility evidence

Regression tests cover the generic shot-dependent gate, stable reads and
arithmetic, runtime table selection and rebinding, fixed table bindings, nested
qkernels, local mask application and decoded vector counts, static persistence,
real validated H2 QIR, and typed saved-job restoration with a mocked Nexus
transport. They make no cloud submissions. See
[`test_current_shot.py`](../../tests/hugr/test_current_shot.py).

In the Python 3.12 environment with Guppy 0.21.16, HUGR 0.16.0,
tket-exts 0.12.4, hugr-qir 0.1.2, and quantinuum-qircheck 0.7.0, a direct
Guppy function returning `measure(q)` reproduces rejection of an `insertvalue`
instruction for its aggregate return. This is distinct from shot-index support:
Qamomile's tagged, void execution wrapper eliminates that return representation,
and the equivalent `shot == 1` conditional passes validation. An investigation
using `% 2` additionally encounters `freeze`; changing the LLVM optimization
level does not repair the aggregate-return example. Removing validation is not
a supported workaround, and integer remainder is separately unavailable on H2.
The bounds-panic conversion limitation remains unresolved in the pinned upstream
converter. No remote H2 or Helios acceptance, internal retry behavior, or
hardware result distribution has been verified by these offline tests.
