---
name: generate_release_note
description: Generate Qamomile's EN/JA release notes (`docs/{en,ja}/release_notes/v<X_Y_Z>.md`) from the diff between the previous version tag and current main. Defines the section order, snippet verification, link conventions, and the related-file updates.
---

# Release Note Generation Skill

Create `docs/en/release_notes/v<X_Y_Z>.md` and `docs/ja/release_notes/v<X_Y_Z>.md`, and update the related index files. The skill takes the previous version tag and the new version as arguments (e.g. `/generate_release_note v0.11.1 v0.12.0`). If the new version is omitted, infer it from `pyproject.toml` etc. and confirm with the user.

## Workflow

### Phase 1: Gather changes

1. List existing tags with `git tag --list | sort -V` and identify the previous version tag.
2. Get a rough sense of the commit volume with `git log <prev_tag>..HEAD --oneline --no-merges | wc -l`.
3. Get the list of merged PRs with `git log <prev_tag>..HEAD --oneline --merges`.
4. Spot the files that changed substantially with `git diff <prev_tag>..HEAD --stat | tail -50`.
5. Extract newly added files with `git log <prev_tag>..HEAD --diff-filter=A --name-only --format= | sort -u`.
6. Read the most recent release note (`docs/en/release_notes/v<latest>.md`) to absorb the existing style (phrasing, one-line blurb, `pip install`, section layout).

### Phase 2: Classify changes

Classify each change from the **user's** point of view. **Do NOT rely on commit-message prefixes (`feat`/`fix` etc.)** — decide by whether the user-facing surface actually changes.

| Category | Criterion |
|---|---|
| **Breaking Changes** | Public import path removed/moved, signature changed, behaviour-compatibility broken. Including moves to private (e.g. `qamomile.optimization.utils` → `qamomile._utils`). |
| **New Features** | New public API or module the user calls directly. |
| **Internal Changes** | New compiler / frontend capabilities that are themselves a public-API extension (new parameter types, new passes, etc.). |
| **Bug Fixes** | Fixes that change user-visible behaviour (visualization rendering, performance, etc.). |
| **Documentation** | New tutorials, new VQAs, index-page changes. |
| **DX / Tooling** | CI, lint config, AGENTS.md, etc. **Often omitted** because the user doesn't see it. |

### Phase 3: Section order (mandatory)

The sections must appear in **this order**:

````markdown
# Qamomile vX.Y.Z

<one-paragraph overview — the "big story" of this release in 1–2 sentences>

```
pip install qamomile==X.Y.Z
```

## Breaking Changes        # 1. breaking changes (if any)
## New Features            # 2. new features (headline first)
## Internal Changes        # 3. internal changes
## Bug Fixes               # 4. bug fixes
## Documentation           # 5. documentation
## Learn More              # 6. links
````

If there are no breaking changes, drop the `## Breaking Changes` section entirely. Whether to surface `## DX / Tooling` is a judgement call — usually skip it.

### Phase 4: Per-section writing rules

#### The overview paragraph

**Write strictly from the user's point of view: what the user can now do.** Anything the user doesn't directly touch belongs in the body sections, not the overview.

- Name only the public APIs / features the user actually calls.
- **Avoid enumerations**: at most "one API name + one or two representative examples" per feature. Do not list feature catalogues like `q[1::2]`, `q[lo:hi]`, nested views, slice-bounded composite gates, ... in the overview.
- **No internal-implementation vocabulary**: keep IR primitive names, compiler-pass names, new exception-class hierarchies, "affine types", supported-SDK enumerations, etc. out of the overview.
- **Compress internal updates the user doesn't touch into a single sentence**: e.g., "There are also internal IR updates that lay the groundwork for treating a compiled `@qkernel` as a portable subgraph of an outer DSL's computation graph." Component names like canonical form, content hash, param_slots, JSON/msgpack serialization belong in the body (`## Internal Changes`), not the overview.
- The overview should land in **3–5 sentences / 5–8 lines**. If it grows beyond that, suspect enumeration or leaked implementation vocabulary.

**Why**: the overview is read by people who want to know what this release means for their code in three seconds. Anyone who wants details reads the body. Stuffing component names into the overview makes the release look like internal-only churn and buries the headline feature.

#### Shared phrasing rules

Rules that apply throughout the body.

- **For class relationships, only state the parent-subclass relationship**. Do not write same-level "sibling" relationships like "`Foo` is a sibling of `Bar`" or its Japanese equivalent "`Foo`は`Bar`の兄弟例外". Saying "subclass of `AffineTypeError`" / "`AffineTypeError`のサブクラス" is enough. "兄弟例外" reads unnaturally in Japanese anyway.
- **Avoid "deliberately" / "意図的に"** when stating that a feature is unsupported. Say "not supported" / "サポートしていません" without the modifier. There is almost no case where the meaning changes if you omit "deliberately".
- **No references to non-public internal discussions or design meetings**: do not write "(2026-05-16 IR-design discussion)" or "(internal RFC #42)" or any reference to artefacts outside the public repo. Release notes are written **strictly from the diff between the previous version tag and the latest main plus public PRs / issues / code**. When writing the `**Why**:` paragraph, ground the motivation in facts visible in the code (e.g., "this kernel needs this IR node").
- **Do not literally translate English metaphors like "sibling", "twin", or "umbrella"**. Describe the relationship in plain technical terms ("parent class", "the same kind of", "bundled", ...).

#### `## New Features`

Use `### <feature name>` for each feature heading and include:

1. **A one- or two-paragraph description** — what the user can now do and how it works, from the user's point of view. Cluster PR links at the end (always render as `[#NNN](https://github.com/Jij-Inc/Qamomile/pull/NNN)` — plain `(#NNN)` is forbidden, see Phase 6).
2. **A code example** — only snippets verified by Phase 5.
3. **Expected output** — if the snippet calls `print(...)`, include a ` ```text` block whose content matches the actual output **exactly**.
4. **A tutorial pointer** — if there is a corresponding tutorial, end with "See [Tutorial NN](...)".

#### `## Internal Changes`

Open the section with a one-paragraph statement of **what these internal changes are for**. Example:

> The Trotter feature above is built on three new compiler/frontend capabilities. They are independently usable, but their direct motivation is making `trotterized_time_evolution` expressible as natural Python.

Then use `### <change name>` for each change and include:

1. **A specification** — what changed, what is now possible.
2. **A code example** — verified per Phase 5.
3. **A `**Why**:` paragraph** — why the change was made.
   - If the change supports a headline feature, cite that feature in the Why.
   - **If the change is independent, say so explicitly** (don't force a connection to the headline — e.g. "**Unlike the items above, this is not part of the X feature**").
   - The Why can also be a **design-philosophy** statement (e.g. "strengthening `if` support naturally completed `bool` handling") — feature A doesn't always strictly "need" feature B.

#### `## Bug Fixes`

Bullet list. One or two sentences each. **Do not enumerate every individual visualizer fix** — bundle them under a single entry like "visualization polish:".

#### `## Documentation`

New tutorials, new VQAs, index updates. One line per entry, with the **tutorial name as the link text**. In the EN version, don't write things like "in EN and JA" (the EN doc has no business referencing JA).

### Phase 5: Snippet verification (mandatory)

**Every** code example in the release note must satisfy:

1. `QiskitTranspiler().transpile(kernel, bindings=...)` succeeds.
2. `executable.sample(transpiler.executor(), shots=...).result()` returns a value.

Verification procedure:

```bash
# Write the verification script under /tmp (do not commit it)
cat > /tmp/verify_release_snippets.py <<'EOF'
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler
# ... paste each snippet verbatim and run it
EOF
uv run python /tmp/verify_release_snippets.py
```

If a snippet errors, **fix the snippet** (unless the underlying feature is genuinely buggy). Then reflect the fix back into the md.

#### Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `EntrypointValidationError: ... quantum inputs/outputs` | A kernel with quantum I/O is passed directly to `transpile()`. | Wrap it in an outer entry-point kernel with classical I/O (`qmc.qubit_array(...)` for allocation, `qmc.measure()` for return). |
| `AffineTypeError: Cannot return a value to 'q[0]' that was not borrowed` | Array assignment like `q[0] = my_kernel(q[0], ...)`. The affine type system cannot trace borrows that flow through a user kernel. | For single qubits use `qmc.qubit(name="q")` + `q = my_kernel(q, ...)`. For arrays pass the whole register: `q = my_kernel(q, ...)`. |
| `MultipleQuantumSegmentsError: Found N quantum segments` | `parameters=[...]` splits the quantum value's flow. | Put every scalar into `bindings`. **Do not use runtime parameters here**. |
| `Line N: only the 'if' branch has a 'return' statement` | A `return` inside an `if` plus another `return` after the `if` in a `@qkernel`. | Put a `return` in every branch, or **keep exactly one `return` at the end** (refactor into `if/else`). |
| Output disagrees with the doc | The actual `repr` wasn't checked. For example `Hamiltonian.terms` renders as `(Z0, Z1)`, not `(Z(0), Z(1))`. | Paste the verifier script's actual output into the md. |

#### What goes in the md vs. what does not

- ✅ Include: kernel definition, the `transpile()` call.
- ❌ Exclude: execution lines like `executable.sample(...)` (verification-only, not displayed).

### Phase 5b: Cross-check technical claims against code (mandatory)

Snippet execution only catches "works / doesn't work". The **factual claims in the prose** — pass names, class names, counts, subclass relationships, raised exception types, supported arguments, preconditions — must be cross-checked against the code separately.

**Do not quote PR-description prose verbatim**:

- PR-description prose often contains pass / class names that were **renamed or deleted during review** (e.g., `SliceLinearityCheckPass` ended up as `SliceBorrowCheckPass`).
- Counts like "N raise sites" or "three new passes" may also be stale relative to the merged code.
- Treat PR prose as draft text. Before it lands in a release note, `grep` for the corresponding code (class definition / `__all__` / `__init__.py` / actual `raise` line) and confirm.

**What to cross-check against what**:

| Claim type | Cross-check against |
|---|---|
| Class / function / module names | `grep`/`rg` for the same-named definition under `qamomile/` |
| Public API path (`qamomile.foo.bar`) | `__all__` / re-export in `qamomile/foo/__init__.py` |
| Subclass relationship (`X is a subclass of Y`) | Read `class X(Y):` directly |
| Raised exception type | `grep` `raise <ExceptionName>` near the claim site |
| Counts ("N raise sites") | Measure with `grep -c` |
| Pipeline order / stages | Read the body of `Transpiler.transpile()` |
| Expected `print(...)` output in examples | Actually run it once and paste the output (overlaps with Phase 5) |

**Automated cross-check via the codex skill (recommended)**:

Hand each subsection of the finished md to codex and ask it to compare the claims with the code. Report format: `file:line + claim + what the code says + severity (P0/P1/P2/P3)`.

```bash
codex exec --skip-git-repo-check --sandbox read-only --color never \
  "Verify the technical claims in docs/en/release_notes/v<X_Y_Z>.md against the code..." \
  < /dev/null
```

**Iterate**: when codex reports findings, fix the prose and re-run with `codex exec resume --last --skip-git-repo-check "Applied the fixes; please re-verify." < /dev/null`. Loop **until codex returns zero findings** (P3 is a judgement call; P2 and above must be fixed).

> **Note**: `codex exec resume` does **not** accept the `--color` flag (only `codex exec` does). Omit `--color` on resume.

### Phase 6: Link conventions

| Kind | Format |
|---|---|
| PR reference | `[#NNN](https://github.com/Jij-Inc/Qamomile/pull/NNN)` — **always render as a link**. Plain `(#NNN)` is forbidden. |
| Tutorial / optimization / VQA notebook link | `https://github.com/Jij-Inc/Qamomile/blob/v<X.Y.Z>/docs/en/<section>/<file>.ipynb` — **GitHub blob URL with the release tag**. `<section>` is `tutorial` / `optimization` / `vqa` etc. Do not use the ReadTheDocs-hosted URLs or relative paths. |
| GitHub repository | `https://github.com/Jij-Inc/Qamomile` |

Do not put a link to the Tutorials top page in the `Learn More` / `さらに詳しく` section — it's always visible in the RTD-hosted sidebar TOC, so the link is redundant.

The `v<X.Y.Z>` tag will 404 momentarily before release; understand that it resolves when the release lands.

### Phase 7: Related-file updates (EN)

1. Add the new release note at the **top** of the `release_notes` section's `children` in `docs/en/myst.yml`:
   ```yaml
       - title: Release Notes
         file: release_notes/index.md
         children:
           - file: release_notes/v<X_Y_Z>.md   # newly added
           - file: release_notes/v<previous>.md
           ...
   ```
2. Prepend one line to `docs/en/release_notes/index.md`:
   ```markdown
   - [v<X.Y.Z>](v<X_Y_Z>) — <one-line summary: roughly three key features; backticks OK>
   ```

### Phase 8: Japanese version

Follow the rules in the `/translate` skill (`.claude/skills/translate/SKILL.md`). Release-note-specific additions:

#### Translation rules (release-note edition)

- **Headings**:
  - `Breaking Changes` → 破壊的変更
  - `New Features` → 新機能
  - `Internal Changes` → 内部的な変更
  - `Bug Fixes` → バグ修正
  - `Documentation` → ドキュメント
  - `Learn More` → さらに詳しく
- **`**Why**:`** stays as the English label (it's a structural marker).
- **Tutorial names** are translated to Japanese:
  - `Tutorial 07 — Hamiltonian Simulation` → `チュートリアル07 — ハミルトニアンシミュレーション`
- **Tutorial links** swap `docs/en/...` for `docs/ja/...`:
  - `.../blob/v0.12.0/docs/en/tutorial/07_xxx.ipynb` → `.../blob/v0.12.0/docs/ja/tutorial/07_xxx.ipynb`
- **In-code comments** are translated to Japanese. **The code itself does not change.**
- **No space between Japanese characters and digits / ASCII** (translate-skill rule 2).
- **Widely-known technical terms stay in English** (`@qkernel`, `Hamiltonian`, `Vector[Observable]`, `pauli_evolve`, `Suzuki–Trotter`, etc.).

#### JA file updates

1. Create `docs/ja/release_notes/v<X_Y_Z>.md`.
2. Add it to the top of `release_notes` `children` in `docs/ja/myst.yml`.
3. Prepend one line to `docs/ja/release_notes/index.md` (translate the summary too).

## Checklist

Final pass:

- [ ] Both `v<X_Y_Z>.md` (EN and JA) created.
- [ ] **Every code example** succeeds via `transpile()` + `sample()` (verified through `/tmp/verify_*.py`).
- [ ] The md shows up to `transpile()`; `sample()` is hidden.
- [ ] **Every** PR number is rendered as a link.
- [ ] Tutorial links use a GitHub blob URL with the correct release tag.
- [ ] No "in EN and JA" phrasing in the EN body.
- [ ] Each internal-change entry has a `**Why**:` paragraph.
- [ ] Independent internal changes carry a "not part of <headline feature>" note.
- [ ] EN/JA `myst.yml` toc updated.
- [ ] One-line entry prepended to EN/JA `release_notes/index.md`.
- [ ] JA has no spaces between Japanese and ASCII characters.
- [ ] JA tutorial names are translated to Japanese.
- [ ] JA tutorial links point at `docs/ja/...`.

## Reference example

For a recent example see [docs/en/release_notes/v0_12_0.md](../../../docs/en/release_notes/v0_12_0.md). It centers the Trotter feature and separates the internal changes that support it (self-recursive `@qkernel`, `Vector[Observable]`, `bool`) from independent changes (the MLIR pretty-printer).
