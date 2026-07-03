# Symbolic Resource Estimation — Design Notes

This document records the architecture of `qamomile.resource_estimation`,
the lessons learned from the first foundation iteration, and the extension
contract that new algorithm estimators must follow. It is the reference for
"where does my change go?" decisions in this package.

## Layering (stable)

Every estimate flows through the same five layers, and every value at every
layer is a SymPy expression so that estimates stay symbolic until the caller
substitutes concrete parameters:

1. **Problem summary** — a Hamiltonian (or OpenFermion-style input) reduced
   to problem-level quantities: `PauliHamiltonianResource`.
2. **Algorithm workload** — algorithm parameters plus derived algorithmic
   quantities: `HamiltonianQPEWorkload`, `TrotterQPEWorkload`,
   `BlockEncodingResource`.
3. **Logical estimate** — architecture-independent circuit-shaped counts:
   the existing `qamomile.circuit.estimator.ResourceEstimate`.
4. **Architecture lift** — physical proxies under an explicit hardware
   model: `FTQCCostModel`, `SurfaceCodeCostModel`, `ActiveVolumeCostModel`.
5. **Comparison / review inputs** — canonical quantity keys and
   `compare_resource_values` rows. Report formats stay out of this package
   until the modeling surface is stable.

Dependency direction is `resource_estimation → qamomile.circuit`, never the
reverse.

## Extension contract (the load-bearing decisions)

### Quantity keys are open

Resource values are exchanged as `dict[str, sp.Expr]` keyed by plain string
quantity names. Any producer may introduce a new key without touching this
package: `compare_resource_values` works on whatever keys both inputs share,
and unregistered keys simply have no display metadata (label falls back to
the key, unit is empty, category is None).

`ResourceQuantity` enumerates the *built-in* keys for discoverability and
autocomplete; it is not a whitelist. Presentation metadata lives in a
runtime registry seeded from `RESOURCE_QUANTITY_SPECS`; algorithm packages
attach metadata for their own keys at import time via
`register_resource_quantity(...)`.

Rationale: the first iteration validated every key against the enum and
raised on unknown keys, which meant every new algorithm required editing
this package in three places (producer key, enum member, spec entry). An
extension point that requires modifying the framework does not scale past a
handful of algorithms.

### Oracle-call counters pass through generically

`GateCount.oracle_calls` entries flow into `resource_values()` under their
own names (`values.setdefault(name, count)` — canonical keys win on
collision). No layer may special-case a specific oracle name: the first
iteration hard-coded `"qpe_iterations"` in three places, which silently
dropped any other algorithm's iteration counter.

### GateCount owns its field enumeration

Anything that needs "all expressions in a gate count" iterates
`GateCount.expressions()` (owned by `qamomile.circuit.estimator`) instead of
re-listing fields. A new `GateCount` field then propagates to parameter
collection and FTQC lifting automatically.

### Shared helpers live in `_common.py`

`_as_expr`, `_validate_positive`, `_validate_nonnegative` and the
`_SympyLike` aliases are defined once in `qamomile/resource_estimation/_common.py`.
Do not copy them into new modules.

## Known debt / roadmap

Recorded here so follow-up iterations have an explicit target. Ordered by
priority:

1. **Declarative workload core.** `HamiltonianQPEWorkload` (~315 lines) and
   `TrotterQPEWorkload` (~310 lines) hand-roll the same machinery: per-field
   sympify validation, `to_dict`, `resource_values`, precision-budget
   accounting, and `from_*` constructors — with byte-identical method pairs
   between them. Before a third workload is added, extract a base where an
   algorithm declares `(input fields + validators, derived expressions)` and
   inherits the rest; a new algorithm should cost ~30–50 lines of formulas,
   not ~300 lines of ceremony.
2. **One home for formula-based estimators.**
   `qamomile/circuit/estimator/algorithmic/` (estimate_trotter, estimate_qsvt,
   estimate_qdrift, estimate_qaoa, estimate_qpe) predates this package and
   overlaps it — Trotter QPE currently has two implementations with
   different formulas and no cross-reference. Formula-based estimators
   should consolidate here (`resource_estimation`), leaving
   `circuit/estimator` to circuit-derived counting; until then, new
   estimators go in this package, not in `circuit/estimator/algorithmic`.
3. **Pluggable representation models.** `_default_logical_qubits` matches on
   the closed `HamiltonianRepresentation` enum, so adding a Hamiltonian
   representation means editing this package. Replace with a protocol
   (representation → logical-qubit scaling and validation) with the enum as
   the built-in lookup.
4. **FTQC estimate deduplication.** `FTQCPhysicalResourceEstimate` and
   `FTQCActiveVolumeResourceEstimate` are near-clones (substitute/to_dict/
   resource_values differ only in field names), and their per-access
   `sp.simplify` properties recompute on every call. Share the substitution
   machinery and cache derived expressions.

## Review checklist for new estimators

- Values exposed via `resource_values()` returning `dict[str, sp.Expr]`;
  register presentation metadata for new keys.
- All inputs validated through `_common` helpers; everything symbolic.
- Logical output is a plain `ResourceEstimate` (layer 3), so FTQC lifting
  and comparisons work unchanged.
- Cross-backend execution tests are NOT required (this package builds no
  circuits), but estimator formulas need randomized + boundary-value tests
  and, where a circuit-derived counterpart exists, a consistency test
  against `estimate_resources` on a small instance.
