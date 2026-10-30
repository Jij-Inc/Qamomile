# Qamomile section guide

Detailed guidance for each of the six required sections. Read this in full before writing your first tutorial in a session.

The two canonical tutorials — [qe_mcmc.py](../../../../docs/en/algorithm/qe_mcmc.py) (hand-wired `@qkernel`, no converter) and [pce_maxcut.py](../../../../docs/en/algorithm/pce_maxcut.py) (converter-based) — show the voice, cell rhythm, and code patterns. Keep both open while drafting. `pce_maxcut.py` already follows this guide's structure exactly (the `### Step N:` prefix under `## Implementation`, the `## Result` / `## Summary` headings). `qe_mcmc.py` predates the formalized structure and uses older section titles (`## Example Run`, numbered `### 1.` steps), so match its *content and cell pacing* but use the H2 titles this guide prescribes.

---

## Cell and cross-reference conventions

The file-level shape is defined once in `references/tutorial_conventions.md` — read it before drafting; this guide does **not** restate it. That file owns: the strict H1–H4 heading hierarchy (including the five fixed H2 labels and the `Step N:` prefix required on every H3 under `## Implementation`), the `# `-prefix markdown rule, math rendering, the `{cite:p}` paper-citation form, the jupytext percent cell format, and the single up-front imports cell.

For quick orientation while reading the per-section rules below: the file is the **Abstract** (no H2 heading, body of the first cell under the H1 title), then the five fixed H2 sections in order — `## Background` → `## Algorithm` → `## Implementation` → `## Result` → `## Summary`.

Two rules are about cells and cross-references rather than any single section, so they live here rather than in `tutorial_conventions.md`:

- **One idea per cell.** Split a markdown cell into two before pushing an unrelated idea into it; same for code cells. A markdown cell introduces every non-trivial code cell — the reader should know *why* a cell exists before seeing it.
- **Cross-reference sections with a MyST link, not a bare heading string.** Give every heading you reference an explicit label by putting `(label)=` on its own line directly above the heading:

  ```markdown
  # (pce-result)=
  # ## Result
  ```

  Then refer to it with an *empty-text* cross-link `[](#label)`. mystmd renders it using the target heading's title, so the link text is the section name and stays in sync if the heading is renamed — e.g. "the brute-force baseline in `[](#pce-result)`". Do **not** write the bare heading string in inline code (`` `## Result` ``), and never invent shorthand like `§2`, `§5`, "above", or "earlier" — those are unstable across revisions and unclear to a reader landing mid-page. Label names must be unique within a file; use a short topic prefix (`pce-`, `qaoa-`) to keep them readable.

---

## 1. Abstract

**Goal:** In a short **prose** summary, tell the reader what the algorithm does, which Qamomile API they will end up calling, and how the tutorial will get them there. Typically 2–5 sentences across 1–2 short paragraphs.

**Format:** A single markdown cell containing:

1. The H1 tutorial title.
2. A 1–2-paragraph prose summary.

**No H2 heading inside this cell.** The tutorial title is the H1, and the abstract is the body text below it — there is no `## Abstract` line.

**No list of any kind inside the Abstract.** Earlier versions of this guide asked for a numbered step list mirroring the section structure; that pattern has been removed. Write the abstract as plain sentences. If the reader needs a roadmap of the steps, they get it from the H2 / H3 headings of the file itself — duplicating that roadmap as a numbered list in the abstract is noise.

Immediately after this markdown cell come two code cells: the install stub (`# # !pip install qamomile`), then a *separate* cell holding every `import` the tutorial uses. Their exact shape is specified in `references/tutorial_conventions.md` → "Opening cells".

**Shape to match:**

```markdown
# # QAOA for MaxCut: Building the Circuit from Scratch
#
# This tutorial walks through the Quantum Approximate Optimization
# Algorithm (QAOA) pipeline for MaxCut on a small NetworkX graph,
# using Qamomile's low-level circuit primitives rather than the
# high-level `QAOAConverter`. We will formulate the problem as a
# QUBO, lift it to an Ising model, hand-wire the QAOA cost and
# mixer layers as `@qkernel`s, optimize the variational angles with
# a classical optimizer, and decode the resulting samples into a
# discrete partition.
#
# At the end we show that `qamomile.circuit.algorithm.qaoa_state`
# produces the same circuit in a single function call, so the
# hand-wired build-up is purely pedagogical.
```

Name the **target Qamomile API** — `qaoa_state`, `QAOAConverter`, `PCEConverter`, `QRAC31Converter`, … — in the prose itself. Readers who land on the file via search should see which Qamomile symbol they are about to learn.

**Avoid:**

- Numbered or bulleted lists of any kind — the Abstract is prose-only.
- Hype words ("powerful", "state-of-the-art") without substance.
- Restating the algorithm name as its own definition.
- Adding a `## Abstract` H2 heading — the section has none.
- Leaving the "built-in equivalent" payoff unannounced. If the tutorial ends by calling `qaoa_state` / `QAOAConverter` after a from-scratch build-up, the abstract should hint at it in prose.

---

## 2. Background

**H2 heading (strict):** `## Background`

**Goal:** Equip the reader with the background of *the problem the algorithm solves*, not the algorithm itself. For a QAOA-for-MaxCut tutorial, this section is about the MaxCut problem — its definition and a concrete instance — not about QAOA.

**What belongs here:**

1. **Problem definition.** The objective function (in `$$...$$` math), constraints if any, and the domain of the decision variables ($x_i \in \{0, 1\}$ / $s_i \in \{\pm 1\}$).
2. **Problem instance.** The concrete data the rest of the tutorial will operate on — a small graph, a small QUBO dict, a hand-crafted matrix. For graph problems use `networkx` with a fixed layout seed (`nx.spring_layout(G, seed=42)`). Keep the instance small enough that a classical baseline (e.g. brute-force enumeration) remains tractable — typically ≤ 8 nodes / ≤ 15 binary variables — so that the Result section can compare against ground truth.
3. **(Converter-based tutorials only) JijModeling formulation.** A JijModeling `Problem` with `@problem.update` that `problem.eval(instance_data)` will later turn into an `ommx.v1.Instance`. Render the problem at the end of the cell by writing a bare `problem` on its own line.

**What does *not* belong here:**

- QUBO → Ising conversion. That is a Qamomile implementation step, not problem background — it lives in the Implementation section.
- The ansatz, the encoding, the cost Hamiltonian. Those are the algorithm, not the problem — they live in the Algorithm section.
- **Any classical baseline / brute-force solver.** Even when the instance is small enough to enumerate, the baseline implementation and the ground-truth value it produces both belong in the Result section, next to the quantum result they are being compared against. Background only states the problem and provides the instance.
- Qamomile API calls other than those strictly needed to build the instance (`BinaryModel.from_qubo`, `problem.eval`, …).

**Format:** H2 `## Background`, then H3 subsections. Typical breakdown:

- `### What is <Problem>?` — problem definition, math.
- `### Create the <instance>` — concrete data builder, optional `matplotlib` visualization.
- `### Define the Problem with JijModeling` — converter-based tutorials only.

**Length:** Usually 2–4 markdown paragraphs plus 1–3 code cells. Scale down for a very simple problem; scale up only if the problem has multiple constraints that each need their own line.

**Avoid:**

- Textbook prerequisites ("a graph is a pair $(V, E)$"). State the *problem*, not the prerequisites.
- Unseeded random instances — the committed `.ipynb` is re-run in CI and must reproduce.
- Pulling the classical baseline forward into this section. Pick an instance size that *can* be brute-forced when the Result section asks for it, but write the enumeration itself in Result.

---

## 3. Algorithm

**H2 heading (strict):** `## Algorithm`

**Goal:** The paper-style conceptual explanation of the algorithm. After reading this section, a reader should be able to describe the algorithm at a whiteboard even before seeing any code. This section is **purely theoretical** — it states the algorithm at the level of math, symbols, and pseudocode, exactly the way the source paper would. **No Qamomile API calls, no `@qkernel` definitions, no implementation snippets, and no Qamomile-specific call-outs appear in this section** — those all belong in `## Implementation`, including the rotation-gate factor-of-2 convention that earlier versions of this guide placed here.

**What belongs here:**

- The central equation — ansatz, encoding, cost function — in `$$...$$`.
- A bullet list mapping every symbol to its meaning. Connect each symbol to the Python variable name that will hold it in `## Implementation` (e.g. "$\gamma$ (the cost-layer angles) will become the `gammas` parameter when we wire the circuit in `## Implementation`").
- A numbered list of the conceptual steps, or a short fenced pseudocode block.
- A paragraph on *why* the method works — the insight, not just the recipe.

**Format:** H2 `## Algorithm`, followed by H3 subsections for sub-topics. Typical breakdown for a variational algorithm:

- `### <Algorithm name> Ansatz` — the trial-state formula.
- `### Cost Function` — what the classical optimizer minimizes.
- `### Parameters` — roles of the variational angles, depth $p$, etc.

**Length:** Proportional to the algorithm. A simple variational ansatz takes ~25 lines of markdown; a novel encoding might need 60. If you pass ~80 lines, check whether some content belongs in `## Implementation`.

**Critical — `## Algorithm` and `## Implementation` must agree.** Mismatches between these two sections are the #1 way Qamomile tutorials feel unprofessional:

- If `## Algorithm` says "$p$ layers until convergence", `## Implementation` must loop $p$ times with a convergence check, not a fixed `p = 3`.
- If `## Algorithm` defines $\gamma$ and $\beta$, `## Implementation` must name them `gammas` and `betas`.
- If `## Algorithm` promises a $k$-body correlator encoding with a compression rate $k$, `## Implementation` must expose $k$ as a kwarg of the Qamomile call.

**Avoid:**

- Any `@qmc.qkernel`, `transpiler.transpile`, or `executable.sample` calls — those belong in `## Implementation`.
- QUBO → Ising conversion steps — implementation detail, not theory.
- Reimplementing the algorithm in pure Python for pedagogical purposes. A tiny illustrative one-cell snippet for a single conceptual step is fine; full from-scratch code is not.
- Qamomile-specific notes such as the rotation-gate factor-of-2 convention, parameter packing, or how `bindings=` vs `parameters=` are routed — all of those are properties of how the algorithm is *implemented* in Qamomile, not of the algorithm itself, so they live in `## Implementation`.

---

## 4. Implementation

**H2 heading (strict):** `## Implementation`

**Goal:** Walk the reader through Qamomile's pipeline end to end, in order, applied to the problem from `## Background` with the algorithm from `## Algorithm`. By the end of this section the reader has a converged `ExecutableProgram` and a final sample set ready to analyse in `## Result`.

This is the runnable example of Qamomile's public API. **Import and call the real Qamomile surface**; do not reimplement the algorithm. The target function (`qaoa_state`, `QAOAConverter`, `PCEConverter`, …) must appear by name in a code cell — `grep` the finished file to confirm.

### Format: H3 steps in pipeline order

Structure the section as a sequence of H3-headed steps, each a (markdown cell, code cell) pair, matching the order in which a Qamomile user goes from problem to sample. **Every H3 in this section MUST start with `Step N:`** (numbered sequentially from `Step 1:`), so the reader can follow the pipeline at a glance. The example titles below show the convention; reuse them verbatim where the algorithm matches, or coin a similar name with the same `Step N:` prefix:

1. **`### Step 1: QUBO to Ising`** — convert the problem to the form the quantum layer expects. For hand-wired VQA: `BinaryModel.change_vartype(VarType.SPIN)` + optional `.normalize_by_abs_max()`. For converter-based: instantiate the converter (`QAOAConverter(instance)`, `PCEConverter(instance)`, …) and call `get_cost_hamiltonian()` / `encode_hamiltonian(k=...)`. Print the resulting Hamiltonian so the reader sees the coefficients.

   Include the $x_i = (1 - s_i) / 2$ map in prose (one line of math), matching how `## Algorithm` introduced the notation. Do not re-derive it — `## Algorithm` already did.

2. **`### Step 2: Circuit Definition`** — build the `@qkernel`. For hand-wired VQA, this is the largest subsection and is itself split by H4 sub-steps like `#### Step 1: Uniform Superposition`, `#### Step 2: Cost Layer`, `#### Step 3: Mixer Layer`, `#### Step 4: Full Ansatz`, one kernel per step. For converter-based tutorials, this step shrinks to a single code cell — `converter.transpile(transpiler, p=...)` builds the kernel under the hood — and is often merged with the next step.

   **Gate-convention call-out (Qamomile-specific).** Whenever this step hand-wires a rotation, the markdown cell that introduces it must remind the reader, in a `:::{note}` callout, that Qamomile's rotation gates carry the standard $1/2$ factor: $\text{RZ}(\theta) = e^{-i \theta Z / 2}$, $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$, $\text{RX}(\theta) = e^{-i \theta X / 2}$. This Qamomile-implementation note used to live in `## Algorithm` but is now placed here next to the code it affects — `## Algorithm` is theory-only. The canonical move (see `pce_maxcut.py` → `### Step 3: Define the Hardware-Efficient Ansatz`) is to absorb the factor into the variational parameter:

   ```markdown
   # :::{note}
   # **Gate convention.** To match $e^{-i \gamma H_C}$ exactly one would
   # pass $2 J_{ij} \gamma$ as the angle. However, since $\gamma$ is a
   # **variational parameter** that the classical optimizer tunes freely,
   # this constant factor is absorbed into the optimal $\gamma$ values.
   # :::
   ```

   Skip the note for converter-based tutorials whose `@qkernel` is built behind a `*Converter.transpile(...)` call — the converter handles the factor and the reader never sees the rotation literal.

3. **`### Step 3: Transpile to an Executable Circuit`** — call `transpiler.transpile(kernel, bindings=..., parameters=...)` using `qamomile.qiskit.QiskitTranspiler` as the default backend. **Always explain the `bindings=` vs `parameters=` distinction in prose**: `bindings` are fixed at compile time (problem structure — coefficients, `n`, `p`); `parameters` stay as runtime parameters the optimizer will vary each call. Readers unfamiliar with this split will otherwise guess wrong.

4. **`### Step 4: Optimize the Variational Parameters`** — wrap the circuit in a `scipy.optimize.minimize` cost function that samples, decodes, and returns `energy_mean()`. Gate shot count and `maxiter` behind `QAMOMILE_DOCS_TEST` using the exact snippet in `references/tutorial_conventions.md` → "Heavy-compute cells (`QAMOMILE_DOCS_TEST`)" — copy that pattern verbatim, do not invent a new environment variable name or ad-hoc `if/else`. Seed the initial parameters with `np.random.default_rng(<seed>)`. After the optimizer returns, plot `cost_history` with matplotlib; a flat or monotonically-decreasing curve is how the reader confirms the loop worked.

5. **`### Step 5: Sample with Optimized Parameters`** — one short cell. Reuses the optimized `gammas`, `betas` and calls `executable.sample(executor, shots=sample_shots, bindings={...}).result()`, then decodes once. Kept separate from step 4 because the optimizer's per-iteration sample sets are discarded — this is the one `## Result` will analyse.

6. **`### Step 6: Decode`** — state explicitly which decode path is in use, because the two behave differently:

   - Hand-wired VQA → `spin_model.decode_from_sampleresult(result)`. Works when you own the `BinaryModel` directly.
   - Converter-based → `converter.decode(result)`. Folds in SPIN → BINARY conversion automatically when the input problem was BINARY.

Skip any step that genuinely does not apply (e.g. algorithms with fixed circuits have no "Optimize" step). Do not invent extra top-level steps to pad the section.

### Other rules

- **Use real names.** `gammas`, `betas`, `theta`, `quad`, `linear`, `p`, `n` — not `a`, `b`, `x`. Tutorials are read side-by-side and inconsistent naming is noisy.
- **Copy Qamomile type spellings exactly.** `qmc.Vector[qmc.Qubit]`, `qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — grab them from the Qamomile source, do not paraphrase.
- **Connect back to `## Algorithm`.** When a step implements what `## Algorithm` called "the cost unitary $e^{-i \gamma H_C}$", say so in the markdown.
- **Target function by name in a code cell.** `QAOAConverter(...)`, `qaoa_state(...)`, `converter.encode_hamiltonian(k=2)` — not wrapped behind a helper. Before delivering, `grep` the finished file for the target symbol and confirm.
- **Keep kernels minimal.** One conceptual step per `@qkernel`; no error handling or validation inside `@qkernel` bodies.

---

## 5. Result

**H2 heading (strict):** `## Result`

**Goal:** Take the converged sample set from `## Implementation` and show the reader what it means — best solution, distribution, visualisation, and comparison against the classical baseline. This is where the numeric payoff lives.

This is *not* about proving Qamomile matches a reference — Qamomile's tests already do that. It is about showing: "here is the sampled distribution, here is the best configuration it found, and here is how it compares to the optimum".

### Format

H2 `## Result`, then H3 subsections for each analysis axis. Typical breakdown:

- `### Classical Baseline (Brute Force)` **(when applicable)** — the classical solver that establishes the ground-truth value. When the problem instance is small enough to enumerate (typically ≤ 20 binary variables), implement the brute-force here: e.g. `itertools.product([0, 1], repeat=n)` for a discrete enumeration, or a vectorised NumPy scan for slightly larger instances. Report the optimum and (when relevant) how many distinct optima exist. Earlier versions of this guide placed the brute-force inside `## Background`; the baseline now lives in `## Result` next to the quantum value it is being compared against. For problem instances too large to enumerate, replace the brute-force with whichever classical heuristic was used as a reference (e.g. a greedy MaxCut, an SDP relaxation), and clearly label it as a heuristic rather than the true optimum.

- `### Decode and Analyze Results` — top-level container, with H4 sub-steps under it:
  - `#### Feasibility Check` (for constrained problems only) — separate feasible from infeasible samples. Note in prose that QAOA's energy folds in penalty terms, so `energy_mean()` ≠ the true objective.
  - `#### Best Feasible Solution` / `#### Best Cut` — report the best bitstring and its true objective. Compare to the brute-force optimum produced in `### Classical Baseline (Brute Force)`.
  - `#### Objective Value Distribution` — histogram of objective values over all samples.
  - `#### Visualize the Best Solution` — graph layout with nodes coloured by the solution, when the domain supports it.

- `### Using the Built-in <helper>` **(optional finale, when applicable)** — see below.

For unconstrained problems (pure MaxCut, …) collapse the feasibility check. For algorithms without a natural spatial structure, skip the visualization. For algorithms whose ground-truth value cannot be computed even by a heuristic at the chosen instance size, drop `### Classical Baseline (Brute Force)` entirely and explain in prose that no classical reference is available — do not invent one.

### Visualization rules

- `matplotlib` and `networkx` are allowed (both in dev deps). Use `matplotlib.pyplot.figure(figsize=(...))` with explicit sizes — notebooks in CI render to fixed-width HTML.
- Fixed layout seed for reproducibility: `nx.spring_layout(G, seed=42)`.
- Meaningful colours. Qamomile's existing figures use `#2696EB` (blue) for bars and `#FF6B6B` / `#4ECDC4` for two-group partitions. Reuse this palette across tutorials for visual consistency.
- Plots must teach. A bar chart of one value, or a graph with all nodes the same colour, is noise — cut it.

### Built-in equivalent (optional finale)

When `## Implementation` built something by hand that Qamomile already provides as a one-call helper, this section ends with an H3 `### Using the Built-in <helper>` subsection. Qamomile's signature pedagogical move: it shows the reader that their hand-wired pipeline compresses to a single call, and verifies that the two agree.

Common pairings:

- Hand-wired QAOA ansatz (the Circuit Definition step in `## Implementation`) → `qamomile.circuit.algorithm.qaoa_state`.
- Hand-wired VQA pipeline → `QAOAConverter`, `QRAOConverter`, `PCEConverter`.
- Hand-wired QFT gate sequence → `qamomile.circuit.stdlib.QFT`.

Skip this subsection when the target Qamomile function *is* the top-level helper — a PCE tutorial that uses `PCEConverter` from line one has nothing further to reveal.

Contents of the subsection:

1. One prose paragraph explaining which helper replaces which hand-wired pieces (bullet list).
2. A new `@qmc.qkernel` that calls the helper.
3. Transpile, sample with the **same optimized parameters**, and print the mean energy side-by-side with the hand-wired result.

Reference: [assets/example_tutorial.py](../assets/example_tutorial.py) → `### Using the Built-in qaoa_state`.

### Don't

- **Don't reintroduce a from-scratch pure-Python implementation here** for comparison. Qamomile's tests already verify correctness; the reader is learning to use Qamomile, not to audit it.
- **Don't add plots as decoration.** If the figure doesn't teach something the numbers don't already convey, skip it.

---

## 6. Summary

**H2 heading (strict):** `## Summary`

**Goal:** Re-confirm an overview that matches the tutorial's stated purpose and let the reader look back on what they were able to learn. This section is *not* a blow-by-blow recap of every step (the H2 / H3 headings already provide that). Surface only the points worth remembering and where the reader can take them next. A summary that merely re-lists what each section did is the failure mode to avoid.

**Format:** One markdown cell with H2 `## Summary`, in three parts and this order:

1. **Gist (1–2 sentences of prose).** Open with one or two sentences summarising the whole tutorial, framed by its purpose: name the algorithm and the Qamomile workflow the reader just walked through, and what it ultimately produced.
2. **Take-home messages (bullet list).** List the few points genuinely worth remembering, as bullets, each led by a short bold label. A bullet is an *insight or result*, not a step that was performed: the key idea behind the method, the headline result (e.g. a resource-reduction figure, or a match against the brute-force optimum), and the specific Qamomile surfaces that did the work, named (the converter, transpiler, gates / layer helpers, decode path). Aim for 2–4 bullets. Do **not** mirror the section / step order ("we built the model, then the ansatz, then optimised, then decoded") — that ordered recap is exactly what this section must avoid.
3. **Application (1–2 sentences of prose).** Close by telling the reader what they can now do or apply: the class of problems this same workflow generalises to, or what they can build next with the API they just learned.

**No code, no images.** A short bold inline label at the start of each take-home bullet is encouraged (e.g. "**Sub-qubit resource use:** …").

**Length:** 8–16 lines total.

**Take-home bullets are not a step recap.** This is the distinction reviewers care about most. The bullets capture *what is worth remembering* (the insight, the headline number, the API surfaces), not *what we did and in what order*. If a bullet could be reworded as "Step N: we did X", it belongs to the headings, not here.

**Removed patterns (intentional, do not re-add):**

- **Limitations subsection.** Known failure modes belong inline in `## Algorithm` (when they shape the math) or `## Implementation` (when they shape the code), or are dropped. Do not add a `### Limitations` block here.
- **Next steps link list.** Do not hand-curate a list of links to sibling tutorials — discovery is via the auto-generated category and tag pages. The "Application" closing (part 3) is a prose statement about what the reader can apply, **not** a link list.

**Shape to match:**

```markdown
# ## Summary
#
# In this tutorial we encoded a 20-node MaxCut problem with Pauli
# Correlation Encoding (PCE) and ran the full Qamomile workflow, from
# building the correlator encoding through to decoding the optimised
# expectation values back into a spin assignment.
#
# - **Sub-qubit resource use:** PCE represented 20 spin variables with
#   only 3 qubits via order-2 Pauli correlators, roughly a 7x reduction
#   over the one-qubit-per-variable QAOA encoding.
# - **End-to-end Qamomile path:** `PCEConverter` built the encoding and
#   exposed the per-variable observables through `get_encoded_pauli_list`,
#   `QiskitTranspiler` produced one executable per observable, and
#   `converter.decode` rounded the optimised expectations into spins.
#
# The same `PCEConverter` workflow applies to any QUBO / Ising
# combinatorial problem where qubit count is the bottleneck: swap in your
# own `BinaryModel` and reuse the encode / transpile / decode steps shown
# above.
```

**Avoid:**

- A blow-by-blow recap that mirrors the steps ("First we built the model, then we wrote the ansatz, then we optimised …").
- Generic recaps that could apply to any algorithm ("In summary, this is a useful method.").
- Promotional tone. The tutorial is a doc page, not a pitch.
- Re-introducing a `### Limitations` subsection or a hand-curated "Next steps" link list.
- Code or images in this section.
