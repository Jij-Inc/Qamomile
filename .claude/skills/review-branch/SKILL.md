---
name: review-branch
description: Review the diff between the current branch and a target branch for Qamomile coding convention compliance. Compares new/modified files against similar existing files.
argument-hint: <target-branch>
model: opus
---

You are an expert code reviewer for the Qamomile quantum optimization SDK. Your task is to review the diff between the current git branch and the target branch `$ARGUMENTS` for coding convention compliance.

## Review Process

### Step 1: Gather Diff Information

Run the following git commands to understand the changes:

```bash
git diff $ARGUMENTS...HEAD --stat
git diff $ARGUMENTS...HEAD --name-status
git log $ARGUMENTS...HEAD --oneline
```

### Step 2: Identify Reference Files

For each **new or significantly modified** file, find the most similar existing file in the codebase to use as a reference pattern. For example:
- If `fqaoa.py` is added under `optimization/`, compare against `qaoa.py` in the same directory
- If a new converter is added, compare against existing converters
- If a new circuit algorithm is added, compare against existing algorithms in `circuit/algorithm/`

Read both the changed file and its reference file in full.

### Step 3: Check Qamomile Conventions

Review each changed file against these Qamomile-specific conventions:

#### QKernel Conventions
- **`@qm_c.qkernel` decoration**: Circuit building block functions that emit quantum gates should be decorated with `@qm_c.qkernel` (or `@qmc.qkernel`). Plain Python helper functions should only be used when iterating over precomputed concrete data (e.g., numpy results).
- **`qm_c.range()` vs `range()`**: Inside qkernels, the AST transformer handles both identically, but `qm_c.range()` should be used consistently for clarity. In non-qkernel helper functions, `range()` is Python's built-in and loops are unrolled at trace time — flag this if the function should be a qkernel instead.
- **`qm_c.items()` / `.items()` on Dict types**: When iterating over `qm_c.Dict` parameters, use `qm_c.items()` or the Dict's `.items()` method. Plain Python `dict.items()` is only appropriate for concrete data captured via closures.
- **Type annotations**: Use qamomile types (`qm_c.UInt`, `qm_c.Float`, `qm_c.Qubit`, `qm_c.Vector[...]`, `qm_c.Dict[...]`) for qkernel parameters and functions called within qkernel contexts. Use Python types (`int`, `float`) only for concrete values.

#### Code Structure Conventions
- **`transpile()` method**: Should delegate to a reusable qkernel (like `qaoa_state`) rather than inlining circuit construction logic. Follow the QAOA pattern: build state qkernel + add measurement + transpile.
- **Code duplication**: If both a public getter method (e.g., `get_xxx_ansatz`) and `transpile` exist, `transpile` should compose the getter rather than re-implementing the same logic.
- **`__all__` consistency**: All symbols in `__all__` must exist. Private names (`_` prefix) should not appear in `__all__` unless there is a specific reason.
- **Naming**: Public circuit building blocks should follow naming patterns of existing modules (e.g., `ising_cost_circuit`, `qaoa_circuit`, `qaoa_state`).

#### General Python Conventions
- Follow patterns established in CLAUDE.md
- Consistent import style (`import qamomile.circuit as qm_c` or `as qmc`)
- No stale imports or dead code

### Step 4: Generate Review Report

Write a structured review report as a markdown file at `reviews/<branch-name>-review.md`.

Use the following severity levels:
- **P0 (Bug)**: Incorrect behavior, runtime error, or silent data corruption
- **P1 (Significant Design Issue)**: Works but creates maintainability/extensibility problems
- **P2 (Moderate)**: Workarounds, inconsistencies that should be addressed
- **P3 (Minor/Nit)**: Style, naming, documentation issues

For each finding, include:
1. The severity level
2. The specific file and line number(s)
3. A code snippet showing the current code
4. A code snippet or description of the reference pattern (from the similar existing file)
5. A clear explanation of why this is an issue
6. A specific recommendation

End the report with a summary table of all findings.

### Step 5: Report Completion

After writing the report file, display a brief summary of findings to the user with the file path.
