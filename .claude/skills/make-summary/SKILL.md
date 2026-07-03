---
name: make-summary
description: Make summary markdown file for the current branch to explain the main branch. This skill uses another subagent or another ai model to validate the summary. Keep your eye on the usage limit.
model: opus 4.8
---

# Branch Summary
- This skill creates a markdown file that summarizes the work done on the current branch, based on the diff against `main` and the context accumulated so far.
- Purpose: to help a reader understand the state of this branch. The summary is ultimately meant to be posted to the corresponding GitHub PR.
- This skill does not touch the codebase as a rule. It only produces the summary.

## I/O rules
- Place the output file at the root of the current working worktree. Follow any other instruction the user gives instead. As a rule, do not place it in `/tmp` or similar.
  - Why: files under `/tmp` may be gone by the time someone tries to look at them later.
- Name the file `<branch-name>-summary.md`.
- As a rule, do not commit the summary file.
  - Why: the summary only exists to understand the current state of the branch. It gets posted as a PR comment, but is not kept as a file in the repository.
- If the file already exists, update it to match the current state.

## Writing rules
- Write the summary so that someone with no knowledge of this branch or its context can understand it.
- Do not write the history or the flow of the conversation into the summary.
  - Why: the summary must be self-contained and understandable on its own.
- Assume the reader is a junior engineer, so someone unfamiliar with Qamomile can still follow it.
- Never use a technical term for the first time without explaining it. Always add an explanation.
- Do not decide on your own that something is "obvious" and skip it. Convey the reasons and background without omission.
- Reference code concretely as `file:line`, in the form `qamomile/circuit/frontend/qkernel.py:249-260`, so the reader can jump to the relevant place from the summary alone.

## Structure rules
Create the summary by filling in the section template below. Do not add any other section (verification results / diff stats / commit history / conversation exchanges / TODO lists, etc.).

```md
## 0. Glossary (this section may be omitted if no explanation is needed)
- Briefly explain the terms that will be needed later.

## 1. Problem overview
- For a bugfix: what was originally happening (reproduction code), and why it happens (where the root cause lives).
- For a feature addition: what becomes possible, and why it is needed.
- Clearly distinguish which is the "behavior on main" and which is the "behavior on the branch".

### 2. Changes at the frontend (the code level the user writes)
- What changed from the perspective of a Qamomile user: the contents of the `@qm.qkernel` the user writes, the error messages the user receives, the behavior of the APIs the user touches, etc.
- Always include at least one code example.
- Do not write backend implementation details here.

### 3. Changes at the backend (IR and other internals overall)
- Changes to the compiler / IR / passes / backends under `qamomile/circuit/{frontend,ir,transpiler}/...`.
- New IR ops / dataclasses / passes / internal helpers, the role each of them solves, and the related file:line. Qamomile terms (`Block`, `Operation`, `affine_validate`, `CompositeGateOperation`, etc.) appear here for the first time, so add a one-line annotation where needed.

### 4. Alternatives that were not adopted, and why this approach was chosen
- For anything that had multiple design-level options, write:
  - the trade-off of each,
  - which one was adopted and what was given up.
- This section alone may include exchanges from the flow of the conversation. Even so, write it as the final form of "what options existed and which was chosen", not as "the flow of the exchange". Not "it was A at first but was fixed to B", but "of A / B / C, B was adopted because ...".
  - Why: it can serve as the rationale for a design decision.
- If there is no relevant design branch, it is fine to write this section as a single line "None" and stop.

### 5. Known limitations
- Write the gaps that remain even after this branch is merged.
- Write the ones that can actually be hit in real code (room for false negatives / false positives, unsupported AST forms, behavior differing on another backend, etc.) in a When / Why / Future fix frame. For ones already registered in a separate document such as LIMITATIONS.md, cross-reference them here; for follow-ups that are not registered, list them with their reasons.
- If there are none, write a single line "None".
```

## Workflow
### Step 1. Grasp the context
- Determine the root of the working worktree (the summary will live here from now on).
- Get a feel for the size of the diff.

```bash
pwd
git status -sb
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
```

### Step 2. Read the diff
- Actually `Read` each changed file. The accuracy of the summary is determined by how deeply you read here.
- For files that changed a lot in `diff --stat`, read the whole file / the relevant range.
- Grasp newly added files, new classes / functions / dataclasses, and deleted symbols.
- Count added tests too, but do not write the test list itself into the summary (absorb it into sections 2/3 as "this behavior is expected").
- For a bugfix, running the minimal code that reproduces the original bug on the branch and confirming that the error actually appears makes section 1 easier to write.

### Step 3. Write the draft of the 5 sections
- `Write` `<branch-name>-summary.md` at the worktree root and fill in the draft following the structure rules and writing rules.
- Do not blur the boundary between section 2 and 3. Separate the user perspective (API / errors / kernel code) and the internal perspective (IR / passes / dataclasses) into different paragraphs.
- Write section 4 in "final form". Do not mix in the chat / review exchanges themselves.
- Confirm that every section has a short annotation on each Qamomile-specific term that appears for the first time.

### Step 4. Discrepancy check with a subagent / another AI
- Call a subagent to cross-check the summary against the real code.
  - However, if the user gives an instruction for a different AI, follow that.
- Include the following in the prompt:
  - the absolute path of the summary,
  - the absolute paths of the main implementation files and the key symbols / line ranges within them that the summary references,
  - "point out in a bulleted list any statement that does not match the real code, and answer 'no discrepancies' if there are none".
- Repeat the following automatically until the subagent's points reach zero.
  1. Re-confirm the point against the real code.
  2. Fix the summary side (use the `Edit` tool).
  3. If you fixed a line number, also confirm that your edit did not shift other line references (the "+N shift" pattern that Codex points out).
  4. Once done fixing, ask a different subagent to re-check.
- There is no need to ask the user for confirmation on every cycle.
  - Why: you are only fixing the points that were raised.

### Step 5. Completion report
- Once Step 4 returns "no discrepancies", tell the user the following and finish.
  - the **absolute path** of the summary,
  - an explicit note that it has not been committed (state that it is untracked in `git status`).

## Things you must not do
- Commit the summary. Leave the summary untracked inside the worktree.
- Place it in `/tmp` or similar. Always the worktree root.
- Add sections on your own.
- Insert manual soft line breaks within a paragraph.
  - Why: it looks bad when posted to GitHub.
- Reflect the conversation history (e.g., "the user pointed out that ...").
- Relay the state of the subagent-point-processing loop to the user every time.
