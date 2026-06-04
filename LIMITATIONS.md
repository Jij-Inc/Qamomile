# Known Limitations

This file collects known limitations of the Qamomile compiler — gaps deliberately left open by recent fixes and trade-offs the codebase carries on purpose. Each entry documents what the limitation is, when it bites, why the simpler fix was deferred, and the future fix path. Entries here cover the call-time specialization fix for issue #392, the eager qkernel rebind-detection change, and the slice/control-flow work tracked by recent controlled-view fixes.

## Re-trace cost is uncached

Every nested call site that resolves new compile-time information about the callee triggers a fresh re-trace of the callee body via `_build_specialized` in `qamomile/circuit/frontend/qkernel.py`. There is no per-signature cache; deeply nested or repeatedly-called sub-kernels with identical arguments are re-traced from scratch on every call site.

**When it bites**: tracing overhead grows linearly with the number of nested call sites that trigger specialization. For most applications this is invisible; for code that builds the same sub-kernel many times inside a parent tracing pass (e.g., a Python loop in the outer kernel body that calls a helper with `qmc.qubit_array(n)` each iteration) the overhead becomes visible.

**Workaround**: build the callee once with `helper.build(...)`, then operate on the resulting `Block` directly instead of relying on the call-time path. Or wait for a follow-up cache.

**Future fix**: per-signature cache keyed by the immutable `(parameters, bindings, qubit_sizes)` triple, with a stable hash for arbitrary `bindings` values. The stability problem (numpy arrays, dicts of arbitrary types) mirrors Qamomile's content-addressable hashing convention, so the cache key derivation can follow the same `repr()`-based canonicalization that `canonical.py` uses.

## Self-recursive shape-dependent stdlib loses gates in recursive layers

When a qkernel calls itself during its own specialized re-trace, the inner self-call observes `self._specializing == True` and falls back to the cached symbolic `self.block`. The transpiler's `inline ↔ partial_eval` loop then unrolls the recursion the same way as before this fix. The cached block has shape-dependent stdlib ops (qft / iqft / qpe) no-op'd because the recursion-passed register is symbolic in the cached trace, so the recursive layers do not emit the gate.

**When it bites**: a self-recursive qkernel that applies `qmc.qft` / `qmc.iqft` / `qmc.qpe` to a passed-in register and recurses. The outer specialized layer keeps the gate, but the recursive layers lose it. The fully-unrolled IR ends up with the gate applied exactly once regardless of recursion depth, instead of once per recursion layer.

**In practice**: this is virtually never written. Standard quantum algorithms apply QFT once on a register or in an explicit loop, not in a self-recursive shape-shrinking pattern. The Cooley-Tukey recursive FFT decomposition has no direct quantum analogue because the standard QFT decomposition emits its `O(n²)` gates iteratively. The limitation is pinned by `test_self_recursive_with_shape_dependent_stdlib_limitation` in `tests/circuit/test_qft.py`; if a future change relaxes the re-entry guard, that test will need to be updated to the expected per-depth count.

**Future fix**: codex's short-term recommendation is *signature canonicalization* — replace the boolean `_specializing` flag with a set of currently-being-specialized `(parameters, bindings, qubit_sizes)` signatures, where each signature is resolved using only pure-const expressions already on each `Value` (literals, parameters with const bindings, simple `BinOp` / unary expressions over const operands). Same signature blocks (true infinite loop), different signature re-specializes. Add a maximum specialization-depth guard to bound pathological recursion. The long-term answer is deferred-shape composite IR (see the qft / iqft standalone no-op entry below), which makes this whole class of re-trace bug disappear.

## `Matrix[Qubit]` / `Tensor[Qubit]` callee parameters skip specialization

Higher-rank quantum-array parameters are not modeled in the `qubit_sizes` bucket. The specialization extractor leaves them out of all three buckets and continues, matching how `Dict` / `Tuple` / non-parameterizable runtime classicals are handled. The cached symbolic block is used for that argument position, with inline-time substitution wiring the caller's actual register through.

**When it bites**: a callee that applies shape-dependent stdlib (`qmc.qft` / `qmc.iqft` / `qmc.qpe`) directly to a `Matrix[Qubit]` / `Tensor[Qubit]` parameter continues to silently no-op on those qubits — the same as the pre-#392 baseline for that specific case.

**Why this trade-off was chosen**: raising `NotImplementedError` for every nested call carrying a higher-rank quantum argument would forbid composing any kernel that simply *takes* such an argument, including helpers that never touch shape-dependent stdlib on it. The silent-no-op trade-off is consistent with how `Dict` / `Tuple` / unbound runtime classicals are already handled in this PR.

**Future fix**: change `_extract_calltime_specialization`'s `qubit_sizes: dict[str, int]` to a `qubit_shapes: dict[str, tuple[int, ...]]` carrying the full shape, then route each entry through `_create_traced_block` with the full tuple instead of `(qubit_sizes[name],)`. `create_dummy_input` itself already accepts any-rank concrete shapes via its `shape=` kwarg and validates `len(shape) == ndim`, so the change is confined to the specialization data model and the one `_create_traced_block` call site. Alternatively, the deferred-shape composite IR (next entry) makes this entry moot.

## `qmc.qft` / `qmc.iqft` standalone no-op fallback is preserved

The stdlib helpers `qft` / `iqft` silently return the input register unchanged when `get_size(qubits)` raises `ValueError`. `qpe` has the same concrete-shape hazard but a slightly different failure shape: it routes through `_get_concrete_size` in `qamomile/circuit/stdlib/qpe.py`, which falls back to the symbolic `UInt`'s `init_value` (default `0`) rather than raising. The result is that a symbolic-counting-size QPE emits a 0-width `CompositeGateOperation(IQFT)` and a 0-width `Cast`-to-`QFixed` while the surrounding QPE structure (controlled-U loop, etc.) is still emitted — partial / incorrect rather than the clean no-op of `qft` / `iqft`. The factory docstrings document the `get_size`-based fallback for `qft` / `iqft` as the way the outer composition layer is expected to re-trace once the shape is resolved.

**When it bites**: a kernel containing shape-dependent stdlib that is built directly via `my_kernel.build()` without bindings, or visualized via `my_kernel.draw()` without arguments, shows an empty / partial circuit body for the gate that no-op'd. The nested-call path is covered by this PR; the standalone trace path is not.

**Why we did not raise instead**: the no-op fallback is also reached during `QKernel.block` lazy build, which traces the kernel body with all parameters symbolic before any binding is applied. Raising at that point would break every kernel that uses shape-dependent stdlib on a parameter-driven register, including the ones the nested-call path now handles correctly. The standalone build path is a debug / inspection surface (`transpile()` itself rejects quantum-I/O entrypoints), so users who hit it see the documented no-op rather than a crash.

**Future fix**: deferred-shape composite IR — change `qmc.qft` / `qmc.iqft` to always emit a `CompositeGateOperation` even when the input shape is symbolic, with `num_target_qubits` typed as `int | Value`. Lower it at `resolve_parameter_shapes` once bindings are applied. `qpe` needs the parallel change in `_emit_iqft_and_cast_to_qfixed`: drop the `_get_concrete_size` fallback to `init_value`, emit the inner `CompositeGateOperation(IQFT)` and the `Cast`-to-`QFixed` with a symbolic size carried in metadata, and lower both at the same `resolve_parameter_shapes` stage. This is the codex-recommended medium-term refactor and obviates both the standalone no-op above and the recursive-layer limitation. The design surface is non-trivial: `CompositeGateOperation` itself, `CastOperation` (or `QFixedMetadata`), serialization, canonicalization, resource estimation, backend emitters, and the decomposition strategies under `qamomile/circuit/stdlib/` all need to handle the symbolic-shape case.

## Rebind analysis silently allows branch-internal silent discard

The qkernel rebind analyzer (`QuantumRebindAnalyzer` in `qamomile/circuit/frontend/ast_transform.py`) walks `if` / `for` / `while` bodies inside a snapshot-restore scope (`_visit_branch_scope`) that not only restores `quantum_vars` but also **truncates any violations recorded inside the branch back to the pre-branch length**. This is what lets compile-time-`if` dead-branch rebinds like `if flag: ... ; else: alt = qubit_array(...); q = alt` decorate successfully — the compile-time-if lowering pass selects one branch and discards the other, so flagging the dead-branch rebind at decoration time would reject legitimate code.

**When it bites**: a *runtime* conditional that silently discards a parameter via a fresh allocation:

```python
@qm.qkernel
def k(q: qm.Qubit, cond: qm.Bit) -> qm.Bit:
    if cond:
        q = qm.qubit("fresh")   # discards q only when cond is true
    return qm.measure(q)
```

The decoration-time analyzer suppresses this `FRESH_ALLOCATION` violation because the assignment is inside an `if` body. The IR-level safety net does NOT close the gap either: `AffineValidationPass` in `qamomile/circuit/transpiler/passes/affine_validate.py` only enforces "consumed at most once" and explicitly does **not** detect "never consumed" / silent-discard patterns, despite the docstring's "Quantum values are not silently discarded" line.

**Why this trade-off was chosen**: the AST analyzer is flow-insensitive and cannot tell compile-time-if from runtime-if. Distinguishing them would require either pulling the constant-folding logic into the frontend or moving the check to the IR layer after `CompileTimeIfLoweringPass`. Both are larger refactors than the eager-raise PR's scope. Top-level (non-branch-internal) bypasses continue to raise at decoration time, and silent-discard inside branches is at least no worse than it was before the eager-raise change.

**Future fix**: either (a) a dedicated IR-level silent-discard pass that runs after `inline` and `partial_eval` (where compile-time `if`s have been folded away and only genuine runtime branches remain), or (b) a flow-sensitive frontend extension that propagates a per-branch consume set and reports inputs that no branch consumed. (a) is more in line with Qamomile's "delegate concretization to IR / emit" principle and would also retroactively cover any other silent-discard pattern the AST analyzer cannot see.

## Rebind analyzer is blind to alias-imported `measure` / `expval` / quantum constructors

`QuantumRebindAnalyzer` recognizes `measure` / `expval` and `qubit` / `qubit_array` by **syntactic name only** — `_is_classical_returning_call` and `_is_quantum_constructor_call` check `call.func.id` (for bare names) or `call.func.attr` (for attribute access) against a frozen-set of known names. They do not resolve `func.__globals__` to identify the actual callable, so an import alias is not recognized as the same primitive.

**When it bites**: `from qamomile.circuit import measure as my_measure` (and the analogous renaming of constructors) creates three different behaviors compared to using `qm.measure(...)` / bare `measure(...)`:

- **`q = factory("s")`** (alias constructor used directly, single-line assignment) — caught as `UNKNOWN_CALL` because the call has no quantum arguments and the target was already quantum. The error wording is generic ("a value that does not thread the original quantum variable through the call") rather than the more specific `FRESH_ALLOCATION` wording, but the violation IS reported.
- **`tmp = factory("s"); q = tmp`** (alias constructor result threaded through an intermediate name) — slips through. Because the constructor isn't recognized, `tmp` is never registered as a fresh quantum origin in `quantum_vars`, so the subsequent `q = tmp` falls through Case 1 of `_check_single_assign` (`tmp not in self.quantum_vars`) and no violation is recorded.
- **`my_measure(q)` followed by `q = qm.qubit("fresh")`** — false positive. Because the alias measure isn't recognized as classical-returning, `q` stays in `quantum_vars` and the subsequent fresh allocation trips a `FRESH_ALLOCATION` violation even though `q` was genuinely consumed.

**Workaround**: refer to the primitives by their qualified path (`qm.measure(...)` / `qmc.qubit_array(...)`) or import the unaliased name. The four syntactic forms accepted by `_is_classical_returning_call` and `_is_quantum_constructor_call` (`Name` and `Attribute` for each of the two names) cover both bare and `<module>.<name>` styles, so unless the importer deliberately renames the symbol the analyzer sees it correctly.

**Future fix**: replace the syntactic-name check with an identity-based one. Each `QKernel` has access to the user function's `__globals__` (and surrounding closure cells), so the analyzer can resolve `call.func.id` to a Python object at decoration time and compare against the canonical `qamomile.circuit.measure` / `qamomile.circuit.expval` / `qamomile.circuit.qubit` / `qamomile.circuit.qubit_array` references. Closure-bound aliases would need a separate pass over the function's `__closure__`. Attribute-form aliases (`obj.measure(...)` where `obj` is not `qm` / `qmc`) remain ambiguous and would either require local type information or a structurally-conservative fall-back.

## Controlled QFT/IQFT over sub-kernel `UInt` slices

**When it bites**: a controlled sub-kernel receives a classical `UInt` argument, forms a prefix slice such as `q[:m]` inside the sub-kernel, and applies `qmc.qft` or `qmc.iqft` to that prefix.

**Why this trade-off was chosen**: a sub-kernel that applies QFT/IQFT to the whole target vector can be specialized from the concrete target size at the controlled call site. The narrower pattern above is different: the composite gate target is a parameterized slice created inside the controlled body. The controlled-U emitter does not yet lower that parameterized sliced composite block into a backend gate while also reflecting the resolved slice width into the target mapping.

**Future fix**: extend controlled-U composite lowering so it can safely resolve sub-kernel parameterized slice widths from bindings and reflect those resolved widths in the backend target mapping.

## Symbolic slice disjointness beyond direct unit-stride intervals

**When it bites**: two symbolic slices on the same root are actually disjoint, but their disjointness cannot be expressed as direct unit-stride root-space intervals.

**Why this trade-off was chosen**: `_symbolic_slices_definitely_disjoint()` only compares direct `slice_step == 1` root slices as half-open intervals, such as `q[:k]` and `q[k:n]`. Strided symbolic slices, nested symbolic affine maps, and more complex symbolic arithmetic are treated conservatively as potentially overlapping.

**Future fix**: add small proof helpers only for shapes that can be proven safely, such as parity partitions or limited affine intervals, without turning the borrow checker into a general symbolic inequality solver.
