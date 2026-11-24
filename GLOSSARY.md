# Qamomile language and execution

These terms distinguish binary quantum data, logical truth values, and requested trials from the physical work used to execute them.

## Language

**Bit**:
One binary digit with value zero or one.
_Avoid_: Boolean

**Bool**:
A logical truth value, either false or true.
_Avoid_: Measured bit

**Bit register**:
An ordered collection of binary digits with an explicit width. A register with width one is still a collection.
_Avoid_: Scalar bit

**Numeric quantum word**:
A coherent collection of binary digits interpreted as an unsigned integer or a fixed-point number.
_Avoid_: Bit register when the numeric interpretation matters

## Execution

**Syntax checker**:
A service that checks whether a quantum program can be accepted for execution and reports diagnostics without returning measurement samples.
_Avoid_: Emulator

**Logical shot**:
One requested trial of a Qamomile program within one invocation. Its zero-based index stays constant throughout that trial.
_Avoid_: Job number, completion rank

**Native shot number**:
The repetition number supplied by an execution endpoint under that endpoint's numbering convention.
_Avoid_: Logical shot index

**Invocation**:
One execution request for a compiled program with its parameters, shot count, and execution policy. Independent invocations have separate logical shot domains.
_Avoid_: PUB, physical submission

**Shot cohort**:
A population of logical shots sharing the classical information needed for their remaining computation. Equal measurement counts alone do not identify the individual logical shots.
_Avoid_: Independent marginal counts

**Prepared invocation**:
An execution request whose exact parameters, shot domain, and execution policy have been checked for its selected target. A different shot domain requires a new check when the prepared work depends on that domain.
_Avoid_: Specialized source, reusable proof

**Resource shot symbol**:
The nonnegative logical shot index named `shot` in a resource estimate for one trial. Repeated reads within that trial refer to the same symbol.
_Avoid_: Shot count, launch parameter

**Per-shot resource estimate**:
The work required by one trial, possibly expressed as a function of its logical shot index. It does not automatically sum the work of every requested trial.
_Avoid_: Total invocation resources
