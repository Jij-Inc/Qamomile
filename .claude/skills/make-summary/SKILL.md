---
name: make-summary
description: Create a summary markdown file for the current branch that explains its work relative to `origin/main`. This skill uses a separate subagent or AI model to validate the summary. Keep an eye on the usage limit.
model: opus
---

# Branch Summary

Summarize the diff between `origin/main` and the current worktree branch after reading the code. The result must be clear enough for reviewers and junior engineers to understand the branch without seeing the conversation history.

## When To Use

- Use after a coherent unit of branch work is complete.
- Use before writing a PR or asking for review.
- Use when the user asks for a branch summary, summary file, `branch-summary`, or `make-summary`.

## Output Rules

- Write the output file at the current worktree root unless the user explicitly chooses another location.
- Name the file `<branch-name>-summary.md`. Sanitize the branch name if needed so it is safe as a filename.
- If the file already exists, overwrite it without asking.
- Do not commit the summary file. Leave it untracked.
- Do not choose `/tmp` or another ephemeral directory as the output location yourself. If the current worktree root itself is under a temporary directory, prefer the worktree root.
- Write the summary in the language explicitly requested by the user; otherwise use the language of the current conversation. Translate the required headings and limitation-frame labels consistently while preserving their meaning and order.
- On completion, report only the absolute path and the fact that the file was not committed. If an independent discrepancy check could not run, also mention that briefly.

## Writing Rules

- Do not make the summary bullet-list driven. Write each section as explanatory prose paragraphs.
- Do not insert manual soft line breaks within a paragraph. Use one physical line per paragraph, and separate paragraphs with blank lines.
- Do not write the conversation, review history, or work log. Limit the summary to the current branch diff.
- Explain every Qamomile-specific term AND every general compiler / CS jargon term at its FIRST use, in the same sentence or the one just before it. This includes words like `lexical`, `phi`, `SSA`, `probe`, `contextvars`, `taint`, `over-approximation`, `dead-store`, `dead-after`, `logical_id`. The target reader knows the overall Qamomile flow (what `@qkernel` / transpile / IR are, roughly) but is NOT an expert in the subsystem this branch touches. Using a term before it is explained is a defect. If a paragraph reads like a symbol index written for someone who already knows these internals, it is wrong — rewrite it.
- Prefer several short paragraphs, each built around a single idea, over one dense paragraph that concatenates every mechanism and every `file:line`. A wall-of-text paragraph that a newcomer cannot follow is a defect even if every fact in it is correct.
- Write as if explaining the branch orally to a junior engineer. Lead each unit of explanation with the goal (what is being achieved) and the reason it is needed, then give the mechanism. Do not skip why the problem matters or why the fix is needed.
- Reference code concretely as `path/to/file.py:123` or `path/to/file.py:123-145`.
- Do not create standalone sections for test lists, diff stats, commit history, validation logs, or work chronology. Absorb only the relevant facts into the appropriate section.

## Code Example Rules

- For a code-bearing change, include a fenced code example in each of Sections 1, 2, and 3. Give each example a distinct job instead of repeating the same snippet three times.
- In Section 1, show the relevant `origin/main` behavior before the branch. For a bugfix, prefer a minimal reproducer that exposes the wrong result or error. For a feature, show the previously unsupported attempt or the nearest code form that demonstrates the missing capability.
- In Section 2, show user-written code that demonstrates the fixed or newly supported behavior, API, or diagnostic. State the expected result next to the example. Keep compiler and IR implementation details out of this example.
- In Section 3, show at least one concise compiler-, IR-, transpiler-, or backend-level example of how the branch implements the behavior. Put examples under the purpose-based `###` sub-section they clarify; when several independent mechanisms materially change, illustrate each one that benefits from code.
- Make every example directly exercise the branch-specific problem, behavior, or mechanism. An example of a neighboring unchanged API does not count merely because it uses the same subsystem. When a bug is unreachable from ordinary user code, make the Section 2 example follow the closest supported frontend path through the changed IR or compiler logic and explain why that path remains safe.
- Ground every example in `origin/main`, the branch diff, or its tests. Keep snippets minimal, preserve the semantics of the real code, and do not invent APIs, diagnostics, output, or unsupported combinations merely to satisfy the format.
- Run each example when practical, especially before claiming an exact result or diagnostic. If the before-branch example cannot run on the current branch, verify it against the `origin/main` source and tests. If execution is impractical, trace the relevant code and avoid unverified exact-output claims.
- If a section has no truthful and useful code example, say why in that section instead of adding a contrived snippet. Documentation-only and configuration-only branches are common reasons, but brevity alone is not.

## Summary Structure

Write exactly the following five semantic top-level (`##`) sections, in this order. Translate the headings into the output language when needed, but do not change their meaning. Do not add or remove a top-level section — no `0. Glossary`, verification results, TODOs, or conversation history as its own `##` heading. Section 3 is the one section that MUST be broken into `###` sub-sections (see its guidance below); the other four are prose and use no sub-headings.

```md
## 1. Problem Overview

## 2. Frontend Changes (User-Written Code Level)

## 3. Backend Changes (IR And Internals)

## 4. Alternatives Not Adopted And Why This Approach Was Chosen

## 5. Known Limitations
```

In `1. Problem Overview`, for a bugfix, distinguish what happened on `origin/main`, why it happened, and how the branch changes the behavior. For a feature, explain what becomes possible and why it is needed. Follow that explanation with the before-branch example required by the Code Example Rules.

In `2. Frontend Changes (User-Written Code Level)`, describe the user-facing behavior: qkernels the user writes, errors the user receives, and API behavior the user touches. Include the user-facing example and expected result required by the Code Example Rules. Do not put backend implementation details here.

In `3. Backend Changes (IR And Internals)`, describe compiler, IR, transpiler, and backend-side changes. Structure this section as follows, and do NOT write it as one flat mechanism dump:

- Open Section 3 with a short lead paragraph (no sub-heading) stating what the backend changes are collectively trying to achieve — the single objective the sub-sections below serve. A reader should be able to stop after this paragraph and know the point of the whole section.
- Then divide the rest into `###` sub-sections, one per *purpose*, not one per file or per symbol. Name each sub-section after the goal it serves (e.g. `### 記録機構の増設`, `### 破棄の検出`), not after a function. Order them so an earlier sub-section's output is consumed by a later one, and say so explicitly ("the records built here are what the check in the next sub-section reads").
- Inside each sub-section, write in this order: (1) what this piece is trying to achieve, (2) why the naive or pre-existing approach cannot achieve it (the forcing reason — e.g. "the frontend cannot raise here because the dead branch's value is gone from the IR by the time the check runs, so a record must be created to carry it forward"), then (3) how the code achieves it, citing each new IR operation, dataclass, pass, or helper with `file:line`. State the role of each cited symbol; explain every term on first use per the Writing Rules.
- Add the internal code examples required by the Code Example Rules to the relevant sub-sections after explaining their goal and forcing reason. Use prose to connect each snippet to the cited implementation rather than presenting an unexplained source-code dump.
- Prefer three to six focused sub-sections over one long one. If the whole section fits in a single idea, a single sub-section is fine, but the goal-first framing still applies.

In `4. Alternatives Not Adopted And Why This Approach Was Chosen`, describe design-level alternatives, trade-offs, and adoption or rejection reasons. If none apply, write `None`. This section may include alternatives raised in external review or conversation, but only as final design rationale, not as a narrative of the work history.

In `5. Known Limitations`, describe gaps that remain after merge. Cover real cases a user can hit, such as false negatives, false positives, unsupported AST forms, or backend differences, using the `When:` / `Why:` / `Future fix:` frame or faithful translations of those labels. If there are multiple limitations, use separate prose paragraphs rather than bullets. If none apply, write `None` in the output language.

## Workflow

### Step 1. Grasp The Context

Determine the worktree root and get a rough sense of the diff size.

```bash
pwd
git status -sb
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
```

### Step 2. Read The Diff

Actually read the changed files. Understand heavily changed files, new files, new classes or functions, and deleted symbols. Read added tests too, but do not make a test list in the summary; absorb expected behavior into the relevant sections. Gather three complementary examples when the change is code-bearing: the `origin/main` problem or missing capability for Section 1, the user-facing branch behavior for Section 2, and the internal implementation mechanism for Section 3.

### Step 3. Write The Draft

Create or overwrite `<branch-name>-summary.md` at the worktree root. Do not blur sections 2 and 3. Write section 4 as final design rationale, not work chronology. Write section 5 as concrete remaining conditions and future fixes, not abstract TODOs.

### Step 4. Check Discrepancies With Another AI

By default, ask a memory-isolated subagent to adversarially check the summary against the real code. If a subagent is unavailable, use another available AI checker. If no independent checker is available or the check is blocked on approval to send local repository content externally, reread the changed files and the summary yourself, perform an adversarial self-check, and fix any mismatch you find. In that case, briefly state in the completion report that no independent discrepancy check ran, but still complete the summary.

Include the absolute path of the summary, the absolute paths of the primary implementation files, the key symbols or line ranges referenced by the summary, a request to point out any statement that does not match the real code in a bulleted list and answer `no discrepancies` if there are none, and constraints forbidding code edits, file creation or deletion, and command execution.

The checker verifies factual accuracy, including that every code example matches the stated branch or `origin/main` behavior and directly exercises the change it claims to illustrate, but factual accuracy is not enough — a summary can be entirely correct and still fail its purpose by being unreadable to the target reader. So after the factual check passes, do a readability pass yourself against the Writing Rules: read Section 3 as if you were the junior engineer it is written for, and flag any term used before it is explained, any wall-of-text paragraph that concatenates unrelated mechanisms, any unexplained or redundant code example, and any sub-section that states a mechanism without first stating the goal it serves and why the naive approach fails. Fix what you find. A factual `no discrepancies` does not exempt the summary from this pass.

If the checker reports discrepancies, re-confirm each point against the real code, fix the summary, and verify that line-number edits did not shift other references. Then ask for another check. Repeat until there are no points left. If a point requires reconsidering a design decision, do not merely edit the summary; ask the user. Do not relay every iteration of the discrepancy-check loop to the user.

### Step 5. Report Completion

After Step 4 returns `no discrepancies`, or after self-checking because no independent checker was available, report the summary's absolute path and that it was not committed. If self-checking replaced an independent check, briefly mention that no independent discrepancy check ran.

## Do Not

- Do not commit the summary.
- Do not place it in `/tmp` unless the current worktree root itself is there.
- Do not add or remove a top-level (`##`) section; Section 3 must still use `###` sub-sections.
- Do not use a term before explaining it on first use.
- Do not write Section 3 as one flat mechanism dump; it must be goal-first `###` sub-sections.
- Do not insert manual soft line breaks within a paragraph.
- Do not write conversation history or a story of the work.
- Do not discuss alternatives outside section 4.
- Do not fabricate examples or copy long implementation blocks when a focused excerpt conveys the behavior.
- Do not relay every discrepancy-check iteration to the user.
