# Known Limitations

This file collects known limitations of the Qamomile compiler — gaps deliberately left open by recent fixes and trade-offs the codebase carries on purpose. Each entry documents what the limitation is, when it bites, why the simpler fix was deferred, and the future fix path. Entries here cover the call-time specialization fix for issue #392, the eager qkernel rebind-detection change, dict subscript lookup for container values, and the slice/control-flow work tracked by recent controlled-view fixes.

## Container-valued operands outgrow the current `Operation.operands` type annotation

Some IR operations legitimately carry container-valued operands even though the base `Operation.operands` field is still annotated as `list[Value]`. In particular, `ForItemsOperation` stores the iterated `DictValue` as its first operand, and `InverseBlockOperation` can keep `DictValue` / `TupleValue` parameters from the inverted qkernel as non-quantum parameter operands. The serialization decoder now restores those shapes explicitly, but it has to use a local `cast(list[Value], ...)` because the static base type has not been widened yet.

**When it bites**: this is primarily a maintainer-facing typing gap. Runtime behavior is supported for the known container-carrying operations, and the decoder keeps ordinary gate / measure / arithmetic operands strict. The rough edge appears when adding a new operation that wants to carry `DictValue` / `TupleValue` in `operands`, or when making type-checker-driven refactors around `Operation.operands`: the current annotation suggests every operand is a scalar / array `Value`, while two operation families intentionally carry broader `ValueBase` instances.

**Why this trade-off was chosen**: widening `Operation.operands` globally to `list[ValueBase]` or a `ValueLike` alias would touch many passes and emitters whose local logic genuinely assumes scalar / array `Value` operands. Splitting container operands into dedicated fields would be cleaner for those operations but requires an IR-contract migration and encoder / decoder schema cleanup. The current fix keeps the behavioral repair local to the block-I/O and container-carrying-operation decoder paths, without weakening `_materialize_as_value`, which still guards positions where containers are invalid.

**Future fix**: make the container operand contract explicit. Either widen `Operation.operands` to a shared `ValueLike` / `ValueBase` type and audit all passes for places that require scalar / array `Value`, or move container parameters onto operation-specific fields for `ForItemsOperation` and `InverseBlockOperation` so the base operand list stays strictly scalar / array. Once that contract is explicit, the decoder-side casts can be removed.

## Dict subscript lookup does not yet support container values

When `Dict[K, V]` uses a container value type such as `qmc.Tuple` or `qmc.Vector`, `Dict.__getitem__` raises `NotImplementedError`.

**When it bites**: a symbolic dictionary lookup such as `d[key]` currently represents the result as a single scalar `Value`. That is sufficient for scalar values, but it cannot rebuild frontend handles for structured lookup results and cannot represent multi-value results in `DictGetItemOperation`, serialization, emission, or the classical executor.

**Why this trade-off was chosen**: the existing dictionary lookup path is scalar-oriented end to end. Supporting structured values would require the frontend, IR operation result model, wire formats, emit passes, and classical executor to agree on how a single lookup produces multiple structured values. The current guard fails explicitly instead of tracing an incomplete container result that later stages cannot materialize correctly.

**Future fix**: allow `DictGetItemOperation` to produce `TupleValue` or `ArrayValue` results, then extend frontend handle reconstruction, serialization, emission, and classical executor support for those structured results.

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

## `qmc.inverse` requires trace-time-resolvable control flow

**When it bites**: `qmc.inverse(layer)(...)` is traced for a `layer` kernel whose quantum body contains control flow the eager inverse builder cannot resolve while constructing the fallback `implementation_block`:

- `qmc.range(...)` bounds that are still symbolic at trace time raise `NotImplementedError`. Because `transpile(bindings=...)` bakes bindings into the trace before the inverse wrapper runs, a loop bound supplied via `bindings` resolves fine — the rejection only fires for bounds kept as runtime parameters (`parameters=[...]`) or for a bindings-free `build()`.
- `if` / `while` / `for ... in items(...)` are rejected unconditionally — **even when `bindings` would resolve the condition at transpile time**. The fallback block is built eagerly at trace time, before `partial_eval` runs `CompileTimeIfLoweringPass`, so the traced block still contains the `IfOperation` / `WhileOperation` / `ForItemsOperation`. Loop bounds are the one control-flow input the inverter can resolve itself, because the bound constants ride directly on the operand `Value`s; a compile-time `if` would need the inverter to fold the branch on its own.

**Why this trade-off was chosen**: the PR keeps inverse fallback blocks explicit in Qamomile IR so every backend has a correct decomposition when its native inverse path is unavailable. Deferring fallback construction would require a later transpiler pass to build the inverse after parameter shape resolution and to keep serialization/canonicalization compatible with a lazy inverse representation.

**Future fix**: make inverse fallback construction lazy for blocks that need transpile-time shape, loop-bound, or branch resolution, or add a dedicated pass that lowers unresolved `InverseBlockOperation` implementations after `resolve_parameter_shapes` / `partial_eval`. That pass would also let bindings-resolved `if` branches invert by folding the selected branch.

## `qmc.control` / `qmc.inverse` of a self-recursive kernel is unsupported

**When it bites**: a self-recursive `@qkernel` — one whose body calls itself and stops via a base-case `if` on a compile-time-constant driver — is passed to `qmc.control` (or `qmc.inverse`). The same kernel transpiles fine when called directly (the `inline ↔ partial_eval` unroll loop folds the base case at the top level), but wrapping it in `qmc.control` makes compilation fail. `qmc.inverse` of such a kernel usually fails even earlier, at trace time, because the eager inverse builder rejects `if` / `while` / `for ... in items(...)` outright (see "`qmc.inverse` requires trace-time-resolvable control flow" above), so a recursive kernel with an `if` base case cannot reach the unroll loop through the inverse path at all.

**Why this trade-off was chosen**: `qmc.control` embeds the controlled kernel's body as a nested `Block` in `ControlledUOperation.block` (and `qmc.inverse` similarly in `InverseBlockOperation`'s nested blocks). `InlinePass` descends into that block and unrolls one self-recursion layer per pass, but `partial_eval` (`ConstantFoldingPass` / `CompileTimeIfLoweringPass`) does not: its recursion driver (`OperationTransformer` in `qamomile/circuit/transpiler/passes/control_flow_visitor.py`) only enters `HasNestedOps` bodies (`For` / `ForItems` / `While` / `If`), never operation-owned blocks. So the base-case `if` inside `ControlledUOperation.block` is never folded, the self-call never disappears, and `unroll_recursion` (`qamomile/circuit/transpiler/transpiler.py`) cannot reach a fixed point. Teaching `partial_eval` to fold inside an operation-owned block is non-trivial: that block is a self-contained unitary with its own formal `input_values`, so the controlled-U's classical param operands must first be resolved to constants and mapped onto those formals (the pre-emit equivalent of emit-time `bind_block_params`) before a scoped fold can run — a new "scoped sub-evaluation into operation-owned blocks" capability rather than a one-line recursion. Given how exotic controlling a recursive kernel is, that work was deferred.

Rather than spin the unroll loop to `MAX_UNROLL_DEPTH` and emit the generic "did not terminate after N iterations" error (which wrongly blames the bindings — the recursion does terminate and its driver is concrete), `unroll_recursion` now detects this shape and fails fast with a targeted error. The detection uses `count_unrollable_call_blocks` (`qamomile/circuit/transpiler/passes/inline.py`), which counts only the residual calls the loop can still resolve (top-level and inside `HasNestedOps` bodies — `For` / `ForItems` / `While` / `If` — but not inside `ControlledUOperation.block` / `InverseBlockOperation` blocks): when that count is zero while `count_call_blocks` is non-zero, every remaining call is trapped inside an operation-owned block, which is the signature of this limitation. The targeted message names `qmc.control` / `qmc.inverse` and points at the non-recursive-rewrite workaround. The genuinely non-terminating top-level recursion case still produces the original "did not terminate" message.

**Workaround**: rewrite the kernel non-recursively — manually unrolled to the required depth — and pass that flat kernel to `qmc.control` / `qmc.inverse`. (A pass-through wrapper that simply forwards to a leaf kernel is fine and fully supported; only genuine self/mutual recursion inside the controlled body is rejected.)

**Future fix**: add scoped partial evaluation into operation-owned blocks — resolve the `ControlledUOperation` / `InverseBlockOperation` classical param operands to constants, map them onto the nested block's formal inputs, and run `ConstantFoldingPass` + `CompileTimeIfLoweringPass` in that scope during each `unroll_recursion` iteration. Once the base-case `if` can be folded inside the block, the existing inline-unroll machinery converges and the targeted error becomes unreachable.

## QURI Parts multi-controlled gates use a bounded dense unitary

**When it bites**: a QURI Parts controlled-U fallback reduces to an irreducible multi-controlled single-qubit gate — three or more controls left after the shared structural reductions, e.g. the multi-controlled X gates `qamomile.circuit.modular_increment` / `modular_decrement` generate on larger registers. QURI Parts has no native multi-controlled gate object, so the gate is emitted as one dense `UnitaryMatrix` over the controls plus target. That path is capped at `_MAX_MC_MATRIX_QUBITS` (10) local qubits — a `2**(k+1)`-dimensional matrix — and raises `EmitError` beyond that. Multi-controlled rotation gates additionally require a compile-time-numeric angle, because the angle is baked into the dense matrix; a runtime-parametric angle raises `EmitError`.

**Why this trade-off was chosen**: the dense matrix is exact, simple, and reuses QURI Parts' `add_UnitaryMatrix_gate`. A gate-synthesis decomposition (ancilla-assisted multi-controlled-X ladders, or recursive V-gate constructions) would avoid the exponential matrix size and support runtime-parametric angles, but needs ancilla management or many more primitive gates. The cap keeps emission memory bounded while covering realistic register sizes.

**Future fix**: add an ancilla-based or recursive gate-synthesis decomposition for multi-controlled single-qubit gates in `qamomile/quri_parts/transpiler.py::QuriPartsEmitPass._emit_irreducible_multi_controlled_gate`, which would lift both the qubit-count cap and the compile-time-angle restriction.

## Controlled Pauli evolution requires a compile-time-numeric gamma on QURI Parts

**When it bites**: wrapping a `qmc.pauli_evolve` sub-kernel in `qmc.control` and transpiling for QURI Parts with `gamma` left as a runtime parameter (`parameters=["gamma"]`). The controlled lowering raises `EmitError` at transpile time; only a concrete `gamma` binding compiles. A concrete `gamma` works for any number of controls (one control via `CRZ`, two or more via the dense `UnitaryMatrix` path).

**Why this trade-off was chosen**: the controlled fallback (`qamomile/circuit/transpiler/passes/emit_support/controlled_emission.py::emit_controlled_pauli_evolve`) emits each Pauli term's basis change and CX ladder uncontrolled and controls only the central `RZ(2 * coeff * gamma)`. Forming that angle scales the resolved `gamma` by `2 * coeff`, but QURI Parts' Rust-backed runtime `Parameter` exposes no Python arithmetic, so the scaling cannot be expressed — the same pre-existing limitation that already prevents uncontrolled `pauli_evolve` from accepting a parametric `gamma` on QURI Parts. Independently, for two or more controls the central `RZ` lowers to a dense `UnitaryMatrix` (see "QURI Parts multi-controlled gates use a bounded dense unitary"), which bakes the angle into a matrix and therefore needs a compile-time-numeric value regardless. The controlled lowering converts the raw `TypeError` from the unsupported scaling into a clear `EmitError` that names the cause and points at binding `gamma`.

**Future fix**: thread the per-term angle through QURI Parts' linear-combination angle form (`LinearMappedUnboundParametricQuantumCircuit` plus the emitter's `combine_symbolic`) instead of scaling the `Parameter` in Python, which would let single-control controlled (and uncontrolled) `pauli_evolve` accept a parametric `gamma` via `CRZ`; the two-or-more-control dense path keeps the concrete-angle requirement until the multi-controlled gate gains a gate-synthesis decomposition.

## `qmc.pauli_evolve` is a single-step first-order Trotter approximation

`qmc.pauli_evolve(q, H, gamma)` lowers `exp(-i * gamma * H)` as a single-step first-order Trotter product: each Hamiltonian term `coeff * P` becomes one `exp(-i * gamma * coeff * P)` rotation, applied once in Hamiltonian iteration order. When the terms of `H` do not all commute this is an `O(gamma^2)` approximation of the exact propagator, not the exact evolution. CUDA-Q (native `exp_pauli`), QURI Parts, and Qiskit's manual fallback path (the shared `h` / `rz` + CX-ladder gadget) all emit this Trotter product as explicit gates, so the approximation is a property of the IR-level decomposition rather than of one backend. Qiskit's default path instead emits a single high-level `PauliEvolutionGate` (see *When it bites*), which synthesizes to the same first-order Trotter circuit when decomposed to basis gates, but evaluates to the exact exponential when read as a gate matrix.

**When it bites**: a non-commuting multi-term Hamiltonian (e.g. `X0*X1 + 1.1*Z0`) evolved in a single `pauli_evolve` call. A statevector cross-check against an *exact* reference then shows a discrepancy on the order of `gamma^2` times the term commutator (measured `1 - 0.9857 ≈ 0.014` for that Hamiltonian at `gamma = 0.4`). The common way to hit this is comparing the emitted circuit's statevector against Qiskit's `PauliEvolutionGate` evaluated through `qiskit.quantum_info.Statevector`: that gate's matrix representation is the **exact** `exp(-i * gamma * H)` (no Trotterization), so `Statevector(circuit_with_PauliEvolutionGate)` returns the exact evolution while the emitted gate-level circuit returns the first-order Trotter approximation. Commuting or single-term Hamiltonians have zero Trotter error and match to machine precision; controlled and uncontrolled evolution are affected identically. The term *order* is the same across backends — this is purely an approximate-vs-exact gap, not a term-ordering difference.

**Why this trade-off was chosen**: `pauli_evolve` is the low-level "one Trotter step" primitive; multi-step and higher-order Suzuki-Trotter accuracy are composed on top of it (the Suzuki recursion in `qamomile/circuit/algorithm/trotter.py` drives `pauli_evolve` with the per-step angles) rather than baked into the primitive. Making the primitive silently exact for small instances and approximate for large ones would hide the approximation from algorithms that depend on controlling the step count.

**Workaround**: drive `pauli_evolve` from a multi-step / higher-order Trotter loop (the Trotter algorithm module, or an explicit `qmc.range(steps)` loop applying `gamma / steps` per step) when a non-commuting Hamiltonian needs accuracy below the single-step `O(gamma^2)` error. When cross-checking against Qiskit specifically, decompose the `PauliEvolutionGate` to basis gates first (so Qiskit also realizes the first-order Trotter circuit) instead of evaluating the exact gate matrix via `Statevector`.

**Future fix**: add an optional step-count / order parameter to `qmc.pauli_evolve` (defaulting to the current single-step first-order behavior) so callers can request a tighter Trotter approximation without hand-rolling the loop, mirroring the per-step angle scaling the Suzuki recursion already performs.

## CUDA-Q observe cannot estimate controlled modular arithmetic yet

**When it bites**: running the controlled modular arithmetic expectation-value path on the CUDA-Q backend. Sampling uses a separate execution path, but `cudaq.observe()` does not yet support the runtime control flow used by these controlled modular arithmetic kernels.

**Why this trade-off was chosen**: the controlled modular primitives are tested through `qmc.control(modular_increment)` / `qmc.control(modular_decrement)` rather than through backend-specific rewrites. CUDA-Q sampling can execute that shape through the generated runnable/sample paths, while observe currently rejects the runtime-control-flow shape. The limitation is marked with an explicit `pytest.xfail` in `tests/circuit/algorithm/arithmetic/test_modular.py::_expval_if_supported`.

**Future fix**: remove the xfail once CUDA-Q observe supports this runtime-control-flow pattern, or add a CUDA-Q-specific expectation-value lowering that avoids the unsupported observe shape while preserving the same qkernel semantics.

## Negative element indices are rejected, not normalized

Python-style negative indexing on traced array handles — `vec[-1]`, `view[-1]`, `q[-1] = ...`, `mat[0, -1]` — raises `NotImplementedError` at trace time instead of selecting an element from the end. Symbolic indices that resolve to a negative value from bindings at compile or emit time stay unresolved and surface as the standard compile error (`Cannot resolve array size ...` / qubit resolution failure) instead of wrapping.

**When it bites**: code written with Python container habits, e.g. `qmc.qubit_array(sizes[-1])` or `q[k - 1]` where `k` folds to `0`. The latter is usually a genuine precondition violation (see `phase_gadget` with empty `indices`), which now fails at trace time with a negative-index message rather than surviving to emit.

**Why this trade-off was chosen**: negative-index semantics were never actually implemented. Root-container classical reads (`sizes[-1]`) only worked by accident of Python container indexing, while every slice-composed path silently mis-addressed: `sizes[1:3][-1]` read `sizes[0]` (the affine root composition `start + step * local` yields `1 + 1 * (-1) = 0`) and produced a wrong-sized circuit, a gate on `view[-1]` for `view = q[1:3]` was routed onto physical `q[0]` instead of `q[2]`, and `q[-1]` on a root qubit array tripped an internal allocator assertion claiming a transpiler bug. Rejecting negative constants in `ArrayBase._get_element` / `_return_element` — exactly where negative slice bounds were already rejected — eliminates the silent-miscompilation class without committing to wrap semantics. Defense-in-depth guards in the compile-time constant-folding resolver (`qamomile.circuit.transpiler.value_resolver.ValueResolver`), the emit resolver (`qamomile.circuit.transpiler.passes.emit_support.value_resolver.ValueResolver`), and the IR-level `resolve_root_qubit_address` / shared `resolve_root_array_index` refuse negative resolved indices and out-of-contract slice bounds (`start < 0`, `step <= 0`) so programmatically constructed IR and binding-resolved negatives cannot slip through either.

**Future fix**: support Python semantics by normalizing `idx < 0` to `dim + idx` in `_get_element` when the dimension is a compile-time constant (rejecting only when the dimension is symbolic), and extend the same normalization to binding-resolved indices using the resolved view length. Until then, compute the index explicitly (e.g. `vec[n - 1]` with a bound `n`).

## Symbolic slice disjointness beyond direct unit-stride intervals

**When it bites**: two symbolic slices on the same root are actually disjoint, but their disjointness cannot be expressed as direct unit-stride root-space intervals.

**Why this trade-off was chosen**: `_symbolic_slices_definitely_disjoint()` only compares direct `slice_step == 1` root slices as half-open intervals, such as `q[:k]` and `q[k:n]`. Strided symbolic slices, nested symbolic affine maps, and more complex symbolic arithmetic are treated conservatively as potentially overlapping.

**Future fix**: add small proof helpers only for shapes that can be proven safely, such as parity partitions or limited affine intervals, without turning the borrow checker into a general symbolic inequality solver.

## Tuple-form expval metadata cannot encode symbolic root indices

**When it bites**: an inlined helper packs a qubit into tuple-form expval metadata, and the caller-side replacement is an element of a symbolic slice, such as `q[j:j+1][0]` where `j` is a `qmc.range(...)` loop variable. Conceptually, the shape is a scalar helper called on an element whose root index is still symbolic:

```python
@qmc.qkernel
def helper(q: qmc.Qubit, obs: qmc.Observable) -> qmc.Float:
    return qmc.expval((q,), obs)

@qmc.qkernel
def caller(obs: qmc.Observable) -> qmc.Float:
    q = qmc.qubit_array(4, "q")
    for j in qmc.range(4):
        view = q[j : j + 1]
        # The desired metadata root address for view[0] is (q.uuid, j),
        # but the current metadata can only store integer indices.
        helper(view[0], obs)
```

The concrete-index cases (`q[1]`, `q[1::2]`, or a slice bound that has already been resolved from compile-time bindings) are not affected. The limitation is about values that still carry a symbolic affine index when `ValueSubstitutor._substitute_metadata()` runs. The regression pin is an IR-level `xfail` rather than a backend execution test because current qkernel control-flow / expval composition hits other frontend and execution constraints before this metadata representation gap can be isolated cleanly.

**Why this trade-off was chosen**: `ArrayRuntimeMetadata.element_parent_uuids` and `element_parent_indices` currently encode a root address as `(array_uuid: str, index: int)`. During inline substitution, `_substitute_metadata()` can promote a standalone sentinel `("", -1)` to a real root address only when `resolve_root_qubit_address()` can reduce the caller-side element to a constant root index. For `q[j:j+1][0]`, the desired root address is conceptually `(q.uuid, j)`, but `j` is a symbolic `Value`, not an `int`. Storing the slice view UUID plus local index, or leaving the standalone sentinel in place, is the only representable state today. The known gap is pinned by `tests/transpiler/test_value_mapping.py::TestArrayRuntimeMetadataSymbolicRootLimitations::test_scalar_inline_metadata_cannot_promote_symbolic_slice_parent`, which is marked `xfail(strict=True)` until metadata can represent symbolic affine root indices.

**Future fix**: extend array-runtime metadata to carry a symbolic affine root expression, for example `(root_uuid, offset_value, stride_value, local_index_value)` or an equivalent small expression tree, and teach emit-time qubit-map construction to resolve that expression with the same binding resolver used for runtime slice chains. Once that exists, the xfail test should be flipped to a normal passing test and this limitation entry can be removed.

## QFixed cast over a non-constant-bound slice view is rejected, not lowered

**When it bites**: a sub-kernel casts a `Vector[Qubit]` parameter to `QFixed` (and typically measures it), and the caller supplies a slice view whose slice bounds are not compile-time constants — for example `sub(q[lo:hi])` where `lo` / `hi` are runtime parameters, or a compile-time `if` that selects such a view. The direct `qmc.cast(q[lo:hi], qmc.QFixed, ...)` form is already rejected at trace time by the frontend (`ValueError`, message mentioning symbolic slice bounds). This entry is about the same shape reaching the compiler indirectly: the cast is traced against a non-view sub-kernel parameter and only becomes a view when the caller's argument is substituted in, so the trace-time guard does not catch it.

**Why this trade-off was chosen**: QFixed carrier qubits are recorded as composite keys `"<root_uuid>_<index>"` indexing into the root array's element space, and `QInitOperation` registers physical qubits under `QubitAddress(root_uuid, index)` in that same space. `resolve_root_array_index()` folds a view-local index through the `slice_of` chain (`start + step * i`) into root space, but only when every slice bound on the chain is a compile-time constant; a symbolic affine bound makes it return `None`. With no constant root index the carrier cannot be mapped to a physical qubit, and emitting a verbatim view-local key (`"<view_uuid>_<i>"`) would leave the carrier unregistered and silently drop the measurement at emit. Rather than fail silently, the inline value-substitution path raises `ValueError` (`ValueSubstitutor._resolve_mapped_carrier` in `qamomile/circuit/ir/value_mapping.py`, which lives in the IR layer and therefore cannot depend on the transpiler's `ValidationError`, so it mirrors the frontend's `ValueError`), and the compile-time-`if` lowering path raises `ValidationError` (`qamomile/circuit/transpiler/passes/compile_time_if_lowering.py`). This is the same `(array_uuid: str, index: int)` root-address representation gap described in "Tuple-form expval metadata cannot encode symbolic root indices" above, surfaced for QFixed carrier keys.

**Future fix**: carry a symbolic affine root expression for carrier keys — the same direction proposed for the tuple-form expval metadata limitation, e.g. `(root_uuid, offset_value, stride_value, local_index_value)` — and resolve it at emit with the same binding resolver used for runtime slice chains, or fold the view bounds from `bindings` before carrier substitution runs. Either lets a binding-resolvable view drive the cast instead of being rejected.
