# Jupytext percent-format reference

This is the exact syntax the tutorial file must follow so that `jupytext --to ipynb tutorial.py` produces a clean notebook.

## File header

Every tutorial file should start with a YAML front-matter block embedded as a comment, plus the imports cell. The header tells jupytext that this `.py` file is a notebook in percent format with a Python kernel:

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

The `jupytext_version` doesn't have to match what the user has installed — jupytext is forgiving about this. Use `1.16.4` as a reasonable default.

## Cell separators

Two cell types matter:

```python
# %% [markdown]
# This is a markdown cell.
# Every line is prefixed with `# ` (hash + space).
# Math works: $f(x) = \sum_i x_i^2$ inline, or block:
#
# $$
# \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(y_i, f_\theta(x_i))
# $$
#
# Blank lines inside a markdown cell are just `#` (or `# ` — both work).

# %%
# This is a code cell. Normal Python, no special prefix.
import numpy as np
x = np.arange(10)
print(x)
```

### Critical rules

1. **Every markdown line starts with `# `**. A single un-prefixed line breaks the cell. Empty markdown lines should be either `#` or `# ` — don't use a fully blank line *inside* a markdown cell, because that ends the cell.
2. **Blank lines between cells are encouraged** for readability — jupytext doesn't care, the human reader does.
3. **The first `# %%` (or `# %% [markdown]`) starts cell 1.** Anything before it is treated as the file header (the YAML block).
4. **Don't mix markdown content into code cells via `# comment` lines** thinking it'll render as markdown — those stay as Python comments inside the code cell. To get a markdown cell, you need the `# %% [markdown]` separator.

## Cell titles (optional but useful)

You can label a cell:

```python
# %% [markdown]
# # Section: Implementation
# 
# Now we walk through using the library's function step by step.

# %%
import library
result = library.my_function(x)
```

The `# # Section: ...` is a markdown H1 inside the cell. This is the recommended way to mark the six required sections — use H1 (`# # title`) for the section name and H2/H3 (`# ## subsection`) for finer structure inside.

## Skeleton template

Use this skeleton as the starting point for every tutorial:

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: <Algorithm Name>
#
# *Based on <Author et al., Year> and the implementation in <library.module.function>.*

# %% [markdown]
# # 1. Abstract
#
# <2-4 sentences>

# %% [markdown]
# # 2. Backgrounds
#
# <problem statement, prior work, notation>

# %%
# (optional) notation setup, e.g., a tiny helper or a sample input
import numpy as np
np.random.seed(0)

# %% [markdown]
# # 3. Algorithm
#
# <step-by-step walk-through, with math>

# %% [markdown]
# # 4. Implementation
#
# <intro: 1–2 sentences orienting the reader to the steps below>

# %% [markdown]
# ## Step 1: <action>
#
# <one paragraph>

# %%
# code for step 1

# %% [markdown]
# ## Step 2: <action>

# %%
# code for step 2

# %% [markdown]
# ## Step 3: Call `<library.function>`
#
# <describe the call: which arguments and why, and what comes back>

# %%
result = library.function(prepared_input, ...)

# %% [markdown]
# # 5. Run example
#
# <data choice rationale>

# %%
# generate or load data, call the function, inspect the output

# %% [markdown]
# # 6. Conclusion
#
# <recap, limitations, follow-up reading>
```

## Common pitfalls

- **Forgetting the space after `#` in markdown lines.** `#hello` is treated as Python comment syntax even inside a markdown cell on some jupytext versions. Always use `# hello`.
- **Putting `# %%` inside a string or docstring.** Jupytext is regex-based and may split on it. Keep cell separators only at top level.
- **Multi-line strings as markdown.** It's possible (`# %% [markdown]` followed by `"""..."""`) but the `# `-prefix style is more portable; stick to it.
- **Forgetting imports in later cells.** Each cell can assume names from earlier cells *if executed in order*, but missing imports at the top will break the run-all-cells path. Put all imports in one cell early.