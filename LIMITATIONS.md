# Known Limitations

This file collects known limitations that survive (or were introduced by) the call-time specialization fix for issue #392. Each entry documents what the limitation is, when it bites, why the simpler fix was deferred, and the future fix path.

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
