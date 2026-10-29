# Qamomile repeated execution

This vocabulary distinguishes a program's requested repetitions from device numbering and describes failures during shot-dependent parameter selection.

## Language

**Shot**:
One repetition of a quantum program within an execution request.

**Shot position**:
The zero-based ordinal of a repetition within an execution request.
_Avoid_: Device shot number

**Logical shot number**:
A zero-based shot number obtained after accounting for the device's starting number. It is distinct from the raw device shot number and the position of a returned result row.

**Device shot number**:
The number supplied by an execution device for the current shot. It is distinct from the position of a returned result row.
_Avoid_: Result-row index

**Shot origin**:
The first device shot number for an execution. It is distinct from the zero-based position of a shot.

**Parameter table**:
A fixed-length ordered collection of classical values from which a quantum program selects parameters during a shot.
_Avoid_: Parameter sweep

**Shot failure**:
A failure that invalidates one repetition's result.

**Execution failure**:
A failure that invalidates the result of the execution request as a whole, including any partial results.
_Avoid_: Skipped shot

**Result decoding failure**:
A failure to interpret returned records as the outputs declared by a program. Provider completion alone does not establish a usable program result.

**Program output**:
A value explicitly produced by a program for its caller.

**Destructive measurement**:
A measurement that produces a classical result and consumes the quantum resource without returning its measured state.

**Projection**:
A measurement that returns both its classical result and the quantum resource in the corresponding eigenstate of the measured basis. It does not restore the state before measurement.

**Measurement with reset**:
A measurement that returns its original classical result and the quantum resource initialized to the zero state.

**Execution statistics**:
Information about running a program, such as operation counts and elapsed time. It is distinct from the program's output.

**Early termination**:
Ending a shot before the program's normal return. A service can consider the overall job completed even when a shot terminates early.
