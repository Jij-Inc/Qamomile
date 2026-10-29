---
status: accepted
---

# Preserve measurement ownership and state in native HUGR operations

Qamomile distinguishes destructive measurement, projection, and measurement with reset in its existing public API and semantic operations. Keep those contracts and return types. Select native target operations in the HUGR adapter without adding a shared operation, changing resource estimates, or rewriting final QIR.

| Public API | Quantum resource after the operation | Native HUGR |
| --- | --- | --- |
| `measure` | Consumed | `tket.quantum.MeasureFree`, then `tket.bool.read` |
| `measure_reset` | Returned in the zero state | A directly connected `Measure` and `Reset` become `tket.qsystem.MeasureReset`, then `tket.bool.read` |
| `project_z` | Returned in the measured Z eigenstate | `tket.quantum.Measure` |
| `project_x` | Returned in the measured X eigenstate | `H`, Z projection, `H` |
| `project_y` | Returned in the measured Y eigenstate | `Sdg`, `H`, Z projection, `H`, `S` |

In tket-exts 0.12.3, `MeasureFree` and `MeasureReset` return an opaque Boolean, while `Measure` returns native HUGR `Bool`. The explicit `read` operation preserves Qamomile's existing Bit representation. Apply destructive measurement to scalar, Vector, QInt, and QFixed carriers. Preserve alias retirement and the existing number-decoding paths. Register the required qsystem and futures extension definitions; do not update dependencies.

The existing frontend expresses `measure_reset` as projection followed by reset. A rewrite of the completed HUGR graph covers function bodies, branches, and loops without duplicating their emission logic. Fuse only when the measured quantum output directly feeds reset in the same parent region. Retain the measured bit, reroute all its consumers through `read`, and transfer the reset quantum output to the combined operation. Explicit state-order edges prevent fusion. An intervening quantum gate or a callable boundary also prevents this local match. The pass visits a fixed node snapshot once; individual link-removal costs depend on Boolean fanout in the installed graph SDK.

An adjacent-operation rewrite in the top-level block emitter would miss nested paths that lower operations individually. A new shared measure/reset operation would expand this change into the frontend, other backends, resource estimates, and persistence. The target-local graph rewrite avoids both expansions.

With hugr 0.16.0 and hugr-qir 0.1.2, the converter implements retaining `Measure` with measurement, reset, and a conditional X that recreates the measured Z eigenstate. Those extra instructions are absent from Qamomile's input HUGR. `MeasureFree` needs no reconstruction. `MeasureReset` emits one reset and retains the original measurement result without a reconstruction branch. `project_z` deliberately keeps retaining semantics; X and Y projections also keep their basis-restoring gates.

The [measurement tests](../../tests/hugr/test_measurement_semantics.py) validate native HUGR, final QIR, deterministic results zero and one, repeated projections, reset state, aggregate decoding, ownership, and nested control flow. The converter's emitted instructions are evidence about these installed versions. They do not establish every device's measurement behavior or require every target to reconstruct a projected state. No provider jobs are submitted for this verification.
