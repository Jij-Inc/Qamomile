# Qamomile section guide

Detailed guidance for each of the six required sections. Read this in full before writing your first tutorial in a session.

The two canonical tutorials — [qaoa_maxcut.py](../../../../docs/en/vqa/qaoa_maxcut.py) (hand-wired VQA) and [qaoa_graph_partition.py](../../../../docs/en/optimization/qaoa_graph_partition.py) (converter-based) — show the voice, cell rhythm, and code patterns. Keep both open while drafting. Their section *titles* differ from this guide (they were written before this structure was formalized); match their *content and cell pacing*, but use the H2 titles this guide prescribes.

---

## Heading and cell conventions

The rendered tutorial has a **strict** heading hierarchy:

- **H1** (`# <title>`) — the tutorial title. Appears exactly once, inside the Abstract cell.
- **H2** (`## <section>`) — one of the five fixed pedagogical labels, in order, and nothing else:
  1. **`## Backgrounds`**
  2. **`## Algorithm`**
  3. **`## Implementation`**
  4. **`## Run example`**
  5. **`## Conclusion`**

  The Abstract does **not** get its own H2 heading — it is the body of the first markdown cell under the H1 title.
- **H3** (`### <name>`) — natural topic names under each H2. This is where `### What is MaxCut?`, `### Create the Graph`, `### QAOA Ansatz`, `### Step 1: Uniform Superposition`, `### Transpile to an Executable Circuit`, `### Decode and Analyze Results`, etc. live.
- **H4** (`#### <name>`) — finer subsections inside an H3 when needed (e.g. `#### Feasibility Check` under `### Decode and Analyze Results`).

Other conventions:

- One idea per cell. Split a markdown cell into two before pushing an unrelated idea into it; same for code cells.
- A markdown cell introduces every non-trivial code cell. The reader should know *why* a cell exists before seeing it.
- Every markdown line starts with `# ` (hash + space). A blank line inside a markdown cell ends the cell — use `#` on its own for vertical spacing.
- Math is rendered with `$...$` inline and `$$...$$` as a block. Do not code-fence Hamiltonians, objectives, or update rules — render them.
- **Paper citations use bare DOI URLs.** When a sentence cites an academic paper, paste the canonical DOI URL (`https://doi.org/<doi>`) as a bare link — no `[text](url)` wrapper, no hand-typed "(Author et al., year)", no BibTeX/`{cite}` role. MyST auto-renders bare DOI URLs as rich citation cards. Example: `QAOA was introduced by Farhi et al. (https://doi.org/10.48550/arXiv.1411.4028).` See SKILL.md → "Paper citation conventions (MyST)" for the full rule, including which sections expect citations and which don't.

---

## 1. Abstract

**Goal:** In 2–4 sentences plus a short numbered list, tell the reader what the algorithm does, which Qamomile API they will end up calling, and what the tutorial's steps are.

**Format:** A single markdown cell containing:

1. The H1 tutorial title.
2. A 2–4-sentence paragraph.
3. A numbered bullet list matching the sections that follow.

**No H2 heading inside this cell.** The tutorial title is the H1, and the abstract is the body text below it — there is no `## Abstract` line.

Immediately after this markdown cell, include exactly one code cell with the install stub, verbatim:

```python
# Install the latest Qamomile through pip!
# # !pip install qamomile
```

Every existing Qamomile tutorial opens with this pair of cells.

**Shape to match:**

```markdown
# # QAOA for MaxCut: Building the Circuit from Scratch
#
# This tutorial walks through the Quantum Approximate Optimization Algorithm
# (QAOA) pipeline step by step, using Qamomile's low-level circuit primitives.
# Rather than using the high-level `QAOAConverter`, we will:
#
# 1. Define a MaxCut problem for a small graph.
# 2. Formulate it as a QUBO, then convert to an Ising model.
# 3. Write the QAOA circuit step by step using `@qkernel`.
# 4. Optimize variational parameters with a classical optimizer.
# 5. Decode and visualize the results.
#
# At the end, we show that `qamomile.circuit.algorithm.qaoa_state` provides
# the same circuit in a single function call.
```

Name the **target Qamomile API** — `qaoa_state`, `QAOAConverter`, `PCEConverter`, `QRAC31Converter`, … — in this cell. Readers who land on the file via search should see which Qamomile symbol they are about to learn.

**Avoid:**

- Hype words ("powerful", "state-of-the-art") without substance.
- Restating the algorithm name as its own definition.
- Adding a `## Abstract` H2 heading — the section has none.
- Leaving the "built-in equivalent" payoff unannounced. If the tutorial ends by calling `qaoa_state` / `QAOAConverter` after a from-scratch build-up, the abstract should hint at it.

---

## 2. Backgrounds

**H2 heading (strict):** `## Backgrounds`

**Goal:** Equip the reader with the background of *the problem the algorithm solves*, not the algorithm itself. For a QAOA-for-MaxCut tutorial, this section is about the MaxCut problem — its definition and a concrete instance — not about QAOA.

**What belongs here:**

1. **Problem definition.** The objective function (in `$$...$$` math), constraints if any, and the domain of the decision variables ($x_i \in \{0, 1\}$ / $s_i \in \{\pm 1\}$).
2. **Problem instance.** The concrete data the rest of the tutorial will operate on — a small graph, a small QUBO dict, a hand-crafted matrix. For graph problems use `networkx` with a fixed layout seed (`nx.spring_layout(G, seed=42)`). Keep the instance small enough to brute-force — typically ≤ 8 nodes / ≤ 15 binary variables.
3. **(Optional) Brute-force reference.** When the instance admits it, enumerate all $2^n$ assignments with `itertools.product([0, 1], repeat=n)` and report the true optimum. This is the ground truth the quantum result will be compared against in §5.
4. **(Converter-based tutorials only) JijModeling formulation.** A JijModeling `Problem` with `@problem.update` that `problem.eval(instance_data)` will later turn into an `ommx.v1.Instance`. Render the problem at the end of the cell by writing a bare `problem` on its own line.

**What does *not* belong here:**

- QUBO → Ising conversion. That is a Qamomile implementation step, not problem background — it lives in §4.
- The ansatz, the encoding, the cost Hamiltonian. Those are the algorithm, not the problem — they live in §3.
- Qamomile API calls other than those strictly needed to build the instance (`BinaryModel.from_qubo`, `problem.eval`, …).

**Format:** H2 `## Backgrounds`, then H3 subsections. Typical breakdown:

- `### What is <Problem>?` — problem definition, math.
- `### Create the <instance>` — concrete data builder, optional `matplotlib` visualization.
- `### Exact Solution (Brute Force)` — the ground-truth enumerator, when feasible.
- `### Define the Problem with JijModeling` — converter-based tutorials only.

**Length:** Usually 2–4 markdown paragraphs plus 1–3 code cells. Scale down for a very simple problem; scale up only if the problem has multiple constraints that each need their own line.

**Avoid:**

- Textbook prerequisites ("a graph is a pair $(V, E)$"). State the *problem*, not the prerequisites.
- Unseeded random instances — the committed `.ipynb` is re-run in CI and must reproduce.
- Brute-forcing an instance that does not fit in memory. If $n > 20$, say so in prose and skip the reference.

---

## 3. Algorithm

**H2 heading (strict):** `## Algorithm`

**Goal:** The paper-style conceptual explanation of the algorithm. After reading this section, a reader should be able to describe the algorithm at a whiteboard even before seeing any code. **No Qamomile API calls, no `@qkernel` definitions, no implementation snippets appear in this section** — those are §4's job.

**What belongs here:**

- The central equation — ansatz, encoding, cost function — in `$$...$$`.
- A bullet list mapping every symbol to its meaning. Connect each symbol to the Python variable name that will hold it in §4 (e.g. "$\gamma$ (the cost-layer angles) will become the `gammas` parameter in §4").
- A numbered list of the conceptual steps, or a short fenced pseudocode block.
- A paragraph on *why* the method works — the insight, not just the recipe.

**Format:** H2 `## Algorithm`, followed by H3 subsections for sub-topics. Typical breakdown for a variational algorithm:

- `### <Algorithm name> Ansatz` — the trial-state formula.
- `### Cost Function` — what the classical optimizer minimizes.
- `### Parameters` — roles of the variational angles, depth $p$, etc.
- `### Gate-Convention Note` — see below; Qamomile-specific.

**Gate-convention call-out (Qamomile-specific).** Qamomile's rotation gates carry the standard $1/2$ factor: $\text{RZ}(\theta) = e^{-i \theta Z / 2}$, $\text{RZZ}(\theta) = e^{-i \theta Z \otimes Z / 2}$, $\text{RX}(\theta) = e^{-i \theta X / 2}$. Any tutorial that hand-wires rotations in §4 must raise this here, while the reader is still in math mode. The canonical move from `qaoa_maxcut.py` is to absorb the factor into the variational parameter:

```markdown
# To match $e^{-i \gamma H_C}$ exactly one would pass $2 J_{ij} \gamma$
# as the angle. However, since $\gamma$ is a **variational parameter**
# that the classical optimizer tunes freely, this constant factor is
# simply absorbed into the optimal $\gamma$ values.
```

Stating this in §4 alongside the code is too late — readers skim §3 to build their mental model, then read §4 looking for confirmation.

**Length:** Proportional to the algorithm. A simple variational ansatz takes ~25 lines of markdown; a novel encoding might need 60. If you pass ~80 lines, check whether some content belongs in §4.

**Critical — §3 and §4 must agree.** Mismatches between these two sections are the #1 way Qamomile tutorials feel unprofessional:

- If §3 says "$p$ layers until convergence", §4 must loop $p$ times with a convergence check, not a fixed `p = 3`.
- If §3 defines $\gamma$ and $\beta$, §4 must name them `gammas` and `betas`.
- If §3 promises a $k$-body correlator encoding with a compression rate $k$, §4 must expose $k$ as a kwarg of the Qamomile call.

**Avoid:**

- Any `@qmc.qkernel`, `transpiler.transpile`, or `executable.sample` calls — those are §4.
- QUBO → Ising conversion steps — those are implementation, §4.
- Reimplementing the algorithm in pure Python for pedagogical purposes. A tiny illustrative one-cell snippet for a single conceptual step is fine; full from-scratch code is not.

---

## 4. Implementation

**H2 heading (strict):** `## Implementation`

**Goal:** Walk the reader through Qamomile's pipeline end to end, in order, applied to the problem from §2 with the algorithm from §3. By the end of this section the reader has a converged `ExecutableProgram` and a final sample set ready to analyse in §5.

This is the runnable example of Qamomile's public API. **Import and call the real Qamomile surface**; do not reimplement the algorithm. The target function (`qaoa_state`, `QAOAConverter`, `PCEConverter`, …) must appear by name in a code cell — `grep` the finished file to confirm.

### Format: H3 steps in pipeline order

Structure the section as a sequence of H3-headed steps, each a (markdown cell, code cell) pair, matching the order in which a Qamomile user goes from problem to sample:

1. **`### QUBO to Ising`** — convert the problem to the form the quantum layer expects. For hand-wired VQA: `BinaryModel.change_vartype(VarType.SPIN)` + optional `.normalize_by_abs_max()`. For converter-based: instantiate the converter (`QAOAConverter(instance)`, `PCEConverter(instance)`, …) and call `get_cost_hamiltonian()` / `encode_hamiltonian(k=...)`. Print the resulting Hamiltonian so the reader sees the coefficients.

   Include the $x_i = (1 - s_i) / 2$ map in prose (one line of math), matching how §3 introduced the notation. Do not re-derive it — §3 already did.

2. **`### Circuit Definition`** — build the `@qkernel`. For hand-wired VQA, this is the largest subsection and is itself split by H4 sub-steps like `#### Step 1: Uniform Superposition`, `#### Step 2: Cost Layer`, `#### Step 3: Mixer Layer`, `#### Step 4: Full Ansatz`, one kernel per step. For converter-based tutorials, this step shrinks to a single code cell — `converter.transpile(transpiler, p=...)` builds the kernel under the hood — and is often merged with the next step.

3. **`### Transpile to an Executable Circuit`** — call `transpiler.transpile(kernel, bindings=..., parameters=...)` using `qamomile.qiskit.QiskitTranspiler` as the default backend. **Always explain the `bindings=` vs `parameters=` distinction in prose**: `bindings` are fixed at compile time (problem structure — coefficients, `n`, `p`); `parameters` stay as runtime parameters the optimizer will vary each call. Readers unfamiliar with this split will otherwise guess wrong.

4. **`### Optimize the Variational Parameters`** — wrap the circuit in a `scipy.optimize.minimize` cost function that samples, decodes, and returns `energy_mean()`. Gate shot count and `maxiter` behind `QAMOMILE_DOCS_TEST`:

   ```python
   import os
   docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
   sample_shots = 256 if docs_test_mode else 2048
   maxiter = 20 if docs_test_mode else 500
   ```

   Copy this pattern verbatim — do not invent a new environment variable name or ad-hoc `if/else`. Seed the initial parameters with `np.random.default_rng(<seed>)`. After the optimizer returns, plot `cost_history` with matplotlib; a flat or monotonically-decreasing curve is how the reader confirms the loop worked.

5. **`### Sample with Optimized Parameters`** — one short cell. Reuses the optimized `gammas`, `betas` and calls `executable.sample(executor, shots=sample_shots, bindings={...}).result()`, then decodes once. Kept separate from step 4 because the optimizer's per-iteration sample sets are discarded — this is the one §5 will analyse.

6. **`### Decode`** — state explicitly which decode path is in use, because the two behave differently:

   - Hand-wired VQA → `spin_model.decode_from_sampleresult(result)`. Works when you own the `BinaryModel` directly.
   - Converter-based → `converter.decode(result)`. Folds in SPIN → BINARY conversion automatically when the input problem was BINARY.

Skip any step that genuinely does not apply (e.g. algorithms with fixed circuits have no "Optimize" step). Do not invent extra top-level steps to pad the section.

### Import rules (Qamomile-specific)

Tutorials under `tutorial/`, `vqa/`, and `optimization/` may import:

- **Runtime deps** (from `pyproject.toml` `[project].dependencies`): `qamomile`, `jijmodeling`, `ommx`, `qiskit`, `qiskit_aer`, `sympy`, `numpy` (transitive via qiskit).
- **Doc-build dev deps** (from `[dependency-groups].dev`): `matplotlib`, `networkx`, `scipy`. These are in the environment `jupyter-book build .` runs in, so every existing tutorial uses them freely.

Tutorials under `collaboration/` may additionally import the optional backend their extra installs (`qamomile[cudaq-cu13]`, `qamomile[quri_parts]`, `qamomile[qbraid]`), and must say so in the Abstract cell. Do not import optional backends outside `collaboration/`.

### Other rules

- **Use real names.** `gammas`, `betas`, `theta`, `quad`, `linear`, `p`, `n` — not `a`, `b`, `x`. Tutorials are read side-by-side and inconsistent naming is noisy.
- **Copy Qamomile type spellings exactly.** `qmc.Vector[qmc.Qubit]`, `qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float]` — grab them from the Qamomile source, do not paraphrase.
- **Connect back to §3.** When a step implements what §3 called "the cost unitary $e^{-i \gamma H_C}$", say so in the markdown.
- **Target function by name in a code cell.** `QAOAConverter(...)`, `qaoa_state(...)`, `converter.encode_hamiltonian(k=2)` — not wrapped behind a helper. Before delivering, `grep` the finished file for the target symbol and confirm.
- **Keep kernels minimal.** One conceptual step per `@qkernel`; no error handling or validation inside `@qkernel` bodies.

---

## 5. Run example

**H2 heading (strict):** `## Run example`

**Goal:** Take the converged sample set from §4 and show the reader what it means — best solution, distribution, visualisation, and comparison to the brute-force reference from §2. This is where the numeric payoff lives.

This is *not* about proving Qamomile matches a reference — Qamomile's tests already do that. It is about showing: "here is the sampled distribution, here is the best configuration it found, and here is how it compares to the optimum".

### Format

H2 `## Run example`, then H3 subsections for each analysis axis. Typical breakdown:

- `### Decode and Analyze Results` — top-level container, with H4 sub-steps under it:
  - `#### Feasibility Check` (for constrained problems only) — separate feasible from infeasible samples. Note in prose that QAOA's energy folds in penalty terms, so `energy_mean()` ≠ the true objective.
  - `#### Best Feasible Solution` / `#### Best Cut` — report the best bitstring and its true objective. Compare to the brute-force optimum from §2.
  - `#### Objective Value Distribution` — histogram of objective values over all samples.
  - `#### Visualize the Best Solution` — graph layout with nodes coloured by the solution, when the domain supports it.

- `### Using the Built-in <helper>` **(optional finale, when applicable)** — see below.

For unconstrained problems (pure MaxCut, …) collapse the feasibility check. For algorithms without a natural spatial structure, skip the visualization.

### Visualization rules

- `matplotlib` and `networkx` are allowed (both in dev deps). Use `matplotlib.pyplot.figure(figsize=(...))` with explicit sizes — notebooks in CI render to fixed-width HTML.
- Fixed layout seed for reproducibility: `nx.spring_layout(G, seed=42)`.
- Meaningful colours. Qamomile's existing figures use `#2696EB` (blue) for bars and `#FF6B6B` / `#4ECDC4` for two-group partitions. Reuse this palette across tutorials for visual consistency.
- Plots must teach. A bar chart of one value, or a graph with all nodes the same colour, is noise — cut it.

### Built-in equivalent (optional finale)

When §4 built something by hand that Qamomile already provides as a one-call helper, this section ends with an H3 `### Using the Built-in <helper>` subsection. Qamomile's signature pedagogical move: it shows the reader that their hand-wired pipeline compresses to a single call, and verifies that the two agree.

Common pairings:

- Hand-wired QAOA ansatz (§4's Circuit Definition) → `qamomile.circuit.algorithm.qaoa_state`.
- Hand-wired VQA pipeline → `QAOAConverter`, `QRAOConverter`, `PCEConverter`.
- Hand-wired QFT gate sequence → `qamomile.circuit.stdlib.QFT`.

Skip this subsection when the target Qamomile function *is* the top-level helper — a PCE tutorial that uses `PCEConverter` from line one has nothing further to reveal.

Contents of the subsection:

1. One prose paragraph explaining which helper replaces which hand-wired pieces (bullet list).
2. A new `@qmc.qkernel` that calls the helper.
3. Transpile, sample with the **same optimized parameters**, and print the mean energy side-by-side with the hand-wired result.

Reference: [qaoa_maxcut.py](../../../../docs/en/vqa/qaoa_maxcut.py) lines 419–481.

### Don't

- **Don't reintroduce a from-scratch pure-Python implementation here** for comparison. Qamomile's tests already verify correctness; the reader is learning to use Qamomile, not to audit it.
- **Don't import backends outside the allowed set** just to make the example prettier.
- **Don't add plots as decoration.** If the figure doesn't teach something the numbers don't already convey, skip it.

---

## 6. Conclusion

**H2 heading (strict):** `## Conclusion`

**Goal:** Close the loop in one short markdown cell. Three pieces:

1. **Recap.** A numbered list matching the Abstract's step list — one line per step, past tense ("Defined a MaxCut problem", "Ran a classical optimization loop"). This anchors what the reader just spent 20 minutes on.
2. **Limitations.** Name 1–2 known failure modes or assumptions of the algorithm *or* the Qamomile implementation. QAOA: shallow-depth results are provably weak; QRAO: requires a graph colouring that may not fit; converter-based paths: penalty weights need tuning.
3. **Where to next.** 1–2 follow-up references as relative links to other Qamomile tutorials or API pages. Format: `[QAOA for Graph Partitioning](../optimization/qaoa_graph_partition)`. If the user gave you multiple source papers, this is where the one you didn't use as the primary source can appear.

**Format:** One markdown cell with H2 `## Conclusion`, a numbered recap, then a **Next steps** bullet list. 8–20 lines total. No code.

**Shape to match (adapt from `qaoa_maxcut.py`, which uses `## Summary` as its H2 — new tutorials should use `## Conclusion`):**

```markdown
# ## Conclusion
#
# In this tutorial we:
#
# 1. Defined a MaxCut problem and built its QUBO formulation from a
#    NetworkX graph.
# 2. Converted the QUBO to an Ising model using `BinaryModel`.
# 3. Built every component of the QAOA circuit as a `@qkernel` —
#    superposition, cost layer, mixer layer, and the full ansatz.
# 4. Ran a classical optimization loop and decoded the results.
# 5. Verified that `qamomile.circuit.algorithm.qaoa_state` provides the
#    same circuit with a single function call.
#
# **Next steps:**
#
# - For **constrained optimization** problems (where penalty terms are
#   needed), see [QAOA for Graph Partitioning](../optimization/qaoa_graph_partition)
#   which uses the higher-level `QAOAConverter` together with JijModeling.
```

**Avoid:**

- Generic conclusions that could apply to any algorithm ("In summary, this is a useful method.").
- Listing every limitation imaginable. Two well-chosen ones are stronger than five.
- Promotional tone. The tutorial is a doc page, not a pitch.
- Inventing links. Only reference tutorials and API pages you confirmed exist under `docs/en/`.
