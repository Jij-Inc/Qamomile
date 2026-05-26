# Qamomile section guide

Detailed guidance for each of the six required sections. Read this in full before writing your first tutorial in a session.

The two canonical tutorials — [qaoa_maxcut.py](../../../../docs/en/algorithm/qaoa_maxcut.py) (hand-wired VQA) and [qaoa_graph_partition.py](../../../../docs/en/algorithm/qaoa_graph_partition.py) (converter-based) — show the voice, cell rhythm, and code patterns. Keep both open while drafting. Their section *titles* differ from this guide (they were written before this structure was formalized); match their *content and cell pacing*, but use the H2 titles this guide prescribes.

---

## Heading and cell conventions

The rendered tutorial has a **strict** heading hierarchy:

- **H1** (`# <title>`) — the tutorial title. Appears exactly once, inside the Abstract cell.
- **H2** (`## <section>`) — one of the five fixed pedagogical labels, in order, and nothing else:
  1. **`## Background`**
  2. **`## Algorithm`**
  3. **`## Implementation`**
  4. **`## Result`**
  5. **`## Summary`**

  The Abstract does **not** get its own H2 heading — it is the body of the first markdown cell under the H1 title.
- **H3** (`### <name>`) — natural topic names under each H2. This is where `### What is MaxCut?`, `### Create the Graph`, `### QAOA Ansatz`, `### Decode and Analyze Results`, etc. live. **Implementation exception:** every H3 directly under `## Implementation` MUST be prefixed with `Step N:` (numbered sequentially from `Step 1:`), e.g. `### Step 1: Build the BinaryModel and PCEConverter`, `### Step 2: Transpile to an Executable Circuit`. This `Step N:` prefix appears under `## Implementation` only — never under any other H2.
- **H4** (`#### <name>`) — finer subsections inside an H3 when needed (e.g. `#### Feasibility Check` under `### Decode and Analyze Results`).

Other conventions:

- One idea per cell. Split a markdown cell into two before pushing an unrelated idea into it; same for code cells.
- A markdown cell introduces every non-trivial code cell. The reader should know *why* a cell exists before seeing it.
- Every markdown line starts with `# ` (hash + space). A blank line inside a markdown cell ends the cell — use `#` on its own for vertical spacing.
- Math is rendered with `$...$` inline and `$$...$$` as a block. Do not code-fence Hamiltonians, objectives, or update rules — render them.
- **Paper citations use bare DOI URLs.** When a sentence cites an academic paper, paste the canonical DOI URL (`https://doi.org/<doi>`) as a bare link — no `[text](url)` wrapper, no hand-typed "(Author et al., year)", no BibTeX/`{cite}` role. MyST auto-renders bare DOI URLs as rich citation cards. Example: `QAOA was introduced by Farhi et al. (https://doi.org/10.48550/arXiv.1411.4028).` See SKILL.md → "Paper citation conventions (MyST)" for the full rule, including which sections expect citations and which don't.
- **Imports go in a single up-front cell.** Collect every `import` the tutorial uses (e.g. `numpy`, `matplotlib.pyplot`, `networkx`, `scipy.optimize.minimize`, the relevant Qamomile symbols — `qamomile.circuit as qmc`, `BinaryModel`, `PCEConverter`, `QiskitTranspiler`, …) into the same install-stub code cell that opens the tutorial, immediately after the Abstract markdown cell. Do **not** scatter `import` statements next to the code cells that first use them — readers should see the full dependency footprint of the article in one place, at the top, the way `docs/en/algorithm/qsci.py` opens. The only exceptions are heavy / conditionally-installed imports that belong to a specific step (e.g. an optional-backend integration tutorial), which can be deferred to that step's cell with an inline comment explaining why.
- **Cite sections by their actual heading name, not by symbol.** When prose in one section refers back to another section, write the literal heading text in inline code or quotes — e.g. "as introduced in `## Background`" or "see the `### Step 3: Transpile to an Executable Circuit` step in `## Implementation`". Do **not** invent shorthand like `§2`, `§5`, `§Background`, "above", or "earlier" to point at a section — those are unstable across draft revisions and unclear to readers landing in the middle of the page. The heading name is the canonical identifier.

---

## 1. Abstract

**Goal:** In a short **prose** summary, tell the reader what the algorithm does, which Qamomile API they will end up calling, and how the tutorial will get them there. Typically 2–5 sentences across 1–2 short paragraphs.

**Format:** A single markdown cell containing:

1. The H1 tutorial title.
2. A 1–2-paragraph prose summary.

**No H2 heading inside this cell.** The tutorial title is the H1, and the abstract is the body text below it — there is no `## Abstract` line.

**No list of any kind inside the Abstract.** Earlier versions of this guide asked for a numbered step list mirroring the section structure; that pattern has been removed. Write the abstract as plain sentences. If the reader needs a roadmap of the steps, they get it from the H2 / H3 headings of the file itself — duplicating that roadmap as a numbered list in the abstract is noise.

Immediately after this markdown cell, include the install-stub-and-imports code cell described in the "Imports go in a single up-front cell" rule near the top of this guide. At minimum it carries:

```python
# Install the latest Qamomile through pip!
# # !pip install qamomile
```

followed by every `import` the tutorial uses.

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

   **Gate-convention call-out (Qamomile-specific).** Whenever this step hand-wires a rotation, the markdown cell that introduces it must remind the reader that Qamomile's rotation gates carry the standard $1/2$ factor: $\text{RZ}(\theta) = e^{-i \theta Z / 2}$, $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$, $\text{RX}(\theta) = e^{-i \theta X / 2}$. This Qamomile-implementation note used to live in `## Algorithm` but is now placed here next to the code it affects — `## Algorithm` is theory-only. The canonical move from `qaoa_maxcut.py` is to absorb the factor into the variational parameter:

   ```markdown
   # To match $e^{-i \gamma H_C}$ exactly one would pass $2 J_{ij} \gamma$
   # as the angle. However, since $\gamma$ is a **variational parameter**
   # that the classical optimizer tunes freely, this constant factor is
   # simply absorbed into the optimal $\gamma$ values.
   ```

   Skip the note for converter-based tutorials whose `@qkernel` is built behind a `*Converter.transpile(...)` call — the converter handles the factor and the reader never sees the rotation literal.

3. **`### Step 3: Transpile to an Executable Circuit`** — call `transpiler.transpile(kernel, bindings=..., parameters=...)` using `qamomile.qiskit.QiskitTranspiler` as the default backend. **Always explain the `bindings=` vs `parameters=` distinction in prose**: `bindings` are fixed at compile time (problem structure — coefficients, `n`, `p`); `parameters` stay as runtime parameters the optimizer will vary each call. Readers unfamiliar with this split will otherwise guess wrong.

4. **`### Step 4: Optimize the Variational Parameters`** — wrap the circuit in a `scipy.optimize.minimize` cost function that samples, decodes, and returns `energy_mean()`. Gate shot count and `maxiter` behind `QAMOMILE_DOCS_TEST`:

   ```python
   import os
   docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
   sample_shots = 256 if docs_test_mode else 2048
   maxiter = 20 if docs_test_mode else 500
   ```

   Copy this pattern verbatim — do not invent a new environment variable name or ad-hoc `if/else`. Seed the initial parameters with `np.random.default_rng(<seed>)`. After the optimizer returns, plot `cost_history` with matplotlib; a flat or monotonically-decreasing curve is how the reader confirms the loop worked.

5. **`### Step 5: Sample with Optimized Parameters`** — one short cell. Reuses the optimized `gammas`, `betas` and calls `executable.sample(executor, shots=sample_shots, bindings={...}).result()`, then decodes once. Kept separate from step 4 because the optimizer's per-iteration sample sets are discarded — this is the one `## Result` will analyse.

6. **`### Step 6: Decode`** — state explicitly which decode path is in use, because the two behave differently:

   - Hand-wired VQA → `spin_model.decode_from_sampleresult(result)`. Works when you own the `BinaryModel` directly.
   - Converter-based → `converter.decode(result)`. Folds in SPIN → BINARY conversion automatically when the input problem was BINARY.

Skip any step that genuinely does not apply (e.g. algorithms with fixed circuits have no "Optimize" step). Do not invent extra top-level steps to pad the section.

### Import rules (Qamomile-specific)

Tutorials under `tutorial/`, `algorithm/`, and `usage/` may import:

- **Runtime deps** (from `pyproject.toml` `[project].dependencies`): `qamomile`, `jijmodeling`, `ommx`, `qiskit`, `qiskit_aer`, `sympy`, `numpy` (transitive via qiskit).
- **Doc-build dev deps** (from `[dependency-groups].dev`): `matplotlib`, `networkx`, `scipy`. These are in the environment `jupyter-book build .` runs in, so every existing tutorial uses them freely.

Tutorials under `integration/` may additionally import the optional backend their extra installs (`qamomile[cudaq-cu13]`, `qamomile[quri_parts]`, `qamomile[qbraid]`), and must say so in the Abstract cell. Do not import optional backends outside `integration/`.

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

Reference: [qaoa_maxcut.py](../../../../docs/en/algorithm/qaoa_maxcut.py) lines 419–481.

### Don't

- **Don't reintroduce a from-scratch pure-Python implementation here** for comparison. Qamomile's tests already verify correctness; the reader is learning to use Qamomile, not to audit it.
- **Don't import backends outside the allowed set** just to make the example prettier.
- **Don't add plots as decoration.** If the figure doesn't teach something the numbers don't already convey, skip it.

---

## 6. Summary

**H2 heading (strict):** `## Summary`

**Goal:** Close the loop in one short markdown cell, written as **prose paragraphs**. The summary should give the reader a single self-contained recap of what they just built: the algorithm they implemented and the specific Qamomile machinery that did the heavy lifting.

**What belongs here:**

1. **A short prose summary of the algorithm the tutorial implemented.** One or two sentences naming the method (e.g. "the QAOA variational ansatz for MaxCut", "the Pauli Correlation Encoding for sub-qubit MaxCut") and what it ultimately produced (a sampled distribution, an expectation value, a decoded assignment). This anchors what the reader just spent 20 minutes on.
2. **A short prose paragraph naming the Qamomile features that did the work.** Call out the *specific* surfaces used by name — the gates and layer helpers (`ry_layer`, `cz_entangling_layer`, `qmc.h`, …), the converter (`QAOAConverter`, `PCEConverter`, …), the transpiler (`QiskitTranspiler`, `QuriPartsTranspiler`, …), the executor flavour (sampler vs estimator), the decode path (`converter.decode`, `BinaryModel.decode_from_sampleresult`). The reader leaves knowing exactly which Qamomile symbols they used and what role each played.

**Format:** One markdown cell with H2 `## Summary` followed by 2–4 short prose paragraphs. **No numbered recap list, no bullet lists, no code, no images.** 6–15 lines total.

**Strict removals (intentional, do not re-add):** Earlier versions of this guide asked for a **Limitations** subsection and a **Next steps** link list inside `## Summary`. Both have been removed and should *not* be reintroduced.

- The "Limitations" subsection used to list known failure modes; that material now either lives inline in `## Algorithm` (when it shapes the math) or in `## Implementation` (when it shapes the code), or is dropped entirely.
- The "Next steps" link list used to point at sibling tutorials; tutorials are now discovered through the auto-generated category landing pages and tag pages instead, so per-article hand-curated link lists are noise.

**Shape to match:**

```markdown
# ## Summary
#
# In this tutorial we implemented the QAOA variational ansatz for MaxCut
# end-to-end, taking a small NetworkX graph through the full quantum
# pipeline and decoding the sampled bitstrings into a discrete partition
# that matched the brute-force optimum.
#
# The heavy lifting was done by a small set of Qamomile surfaces: the
# `BinaryModel.from_qubo` constructor folded the problem into an Ising
# model, the cost-layer and mixer-layer `@qkernel`s were stacked with
# the `ry_layer` / `cz_entangling_layer` building blocks, and the
# `QiskitTranspiler` produced the executable that `executable.sample`
# evaluated through the Qiskit estimator each iteration. The final
# decode step reused `spin_model.decode_from_sampleresult` to lift the
# raw bitstrings back to the problem's vartype.
```

**Avoid:**

- Generic recaps that could apply to any algorithm ("In summary, this is a useful method.").
- Promotional tone. The tutorial is a doc page, not a pitch.
- Re-introducing the removed "Limitations" / "Next steps" subsections.
- Numbered or bulleted lists in this section — the summary is prose-only.
