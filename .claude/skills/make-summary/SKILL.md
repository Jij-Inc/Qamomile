---
name: make-summary
description: Create a summary markdown file for the current branch that explains its work relative to the main branch. This skill uses a separate subagent or AI model to validate the summary. Keep an eye on the usage limit.
model: opus
---

# Branch Summary

Summarize the diff between `main` and the current worktree branch after reading the code. The result must be clear enough for reviewers and junior engineers to understand the branch without seeing the conversation history.

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
- On completion, report only the absolute path and the fact that the file was not committed. If an independent discrepancy check could not run, also mention that briefly.

## Writing Rules

- Do not make the summary bullet-list driven. Write each section as explanatory prose paragraphs.
- Do not insert manual soft line breaks within a paragraph. Use one physical line per paragraph, and separate paragraphs with blank lines.
- Do not write the conversation, review history, or work log. Limit the summary to the current branch diff.
- Briefly explain each Qamomile-specific term or technical term before relying on it.
- Write as if explaining the branch orally to a junior engineer. Do not skip why the problem matters or why the fix is needed.
- Reference code concretely as `path/to/file.py:123` or `path/to/file.py:123-145`.
- Do not create standalone sections for test lists, diff stats, commit history, validation logs, or work chronology. Absorb only the relevant facts into the appropriate section.

## Summary Structure

Write exactly the following five sections, in this order. Do not add `0. Glossary`, verification results, TODOs, conversation history, or any other section.

```md
1. Problem Overview

2. Frontend Changes (User-Written Code Level)

3. Backend Changes (IR And Internals)

4. Alternatives Not Adopted And Why This Approach Was Chosen

5. Known Limitations
```

In `1. Problem Overview`, for a bugfix, distinguish what happened on `main`, why it happened, and how the branch changes the behavior. For a feature, explain what becomes possible and why it is needed.

In `2. Frontend Changes (User-Written Code Level)`, describe the user-facing behavior: qkernels the user writes, errors the user receives, and API behavior the user touches. Include at least one code example. Do not put backend implementation details here.

In `3. Backend Changes (IR And Internals)`, describe compiler, IR, transpiler, and backend-side changes. If there is a new IR operation, dataclass, pass, or helper, state its role and cite `file:line`. Briefly explain Qamomile terms the first time they appear.

In `4. Alternatives Not Adopted And Why This Approach Was Chosen`, describe design-level alternatives, trade-offs, and adoption or rejection reasons. If none apply, write `None`. This section may include alternatives raised in external review or conversation, but only as final design rationale, not as a narrative of the work history.

In `5. Known Limitations`, describe gaps that remain after merge. Cover real cases a user can hit, such as false negatives, false positives, unsupported AST forms, or backend differences, using the `When:` / `Why:` / `Future fix:` frame. If there are multiple limitations, use separate prose paragraphs rather than bullets. If none apply, write `None`.

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

Actually read the changed files. Understand heavily changed files, new files, new classes or functions, and deleted symbols. Read added tests too, but do not make a test list in the summary; absorb expected behavior into the relevant sections. For a bugfix, gather enough detail to explain the bug on `main` and the branch behavior with concrete examples when possible.

### Step 3. Write The Draft

Create or overwrite `<branch-name>-summary.md` at the worktree root. Do not blur sections 2 and 3. Write section 4 as final design rationale, not work chronology. Write section 5 as concrete remaining conditions and future fixes, not abstract TODOs.

### Step 4. Check Discrepancies With Another AI

By default, use the `claude` skill to adversarially check the summary against the real code. Watch usage. If `claude` is unavailable or blocked on approval to send local repository content externally, ask a memory-isolated subagent or another available AI to perform the same check. If neither is available, reread the changed files and the summary yourself, perform an adversarial self-check, and fix any mismatch you find. In that case, briefly state in the completion report that no independent discrepancy check ran, but still complete the summary.

Include the absolute path of the summary, the absolute paths of the main implementation files, the key symbols or line ranges referenced by the summary, a request to point out any statement that does not match the real code in a bulleted list and answer `no discrepancies` if there are none, and constraints forbidding code edits, file creation or deletion, and command execution.

If the checker reports discrepancies, re-confirm each point against the real code, fix the summary, and verify that line-number edits did not shift other references. Then ask for another check. Repeat until there are no points left. If a point requires reconsidering a design decision, do not merely edit the summary; ask the user. Do not relay every iteration of the discrepancy-check loop to the user.

### Step 5. Report Completion

After Step 4 returns `no discrepancies`, or after self-checking because no independent checker was available, report the summary's absolute path and that it was not committed. If self-checking replaced an independent check, briefly mention that no independent discrepancy check ran.

## Do Not

- Do not commit the summary.
- Do not place it in `/tmp` unless the current worktree root itself is there.
- Do not add sections beyond the five specified sections.
- Do not insert manual soft line breaks within a paragraph.
- Do not write conversation history or a story of the work.
- Do not discuss alternatives outside section 4.
- Do not relay every discrepancy-check iteration to the user.
