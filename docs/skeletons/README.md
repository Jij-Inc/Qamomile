# Documentation Skeletons

These files are authoring templates for Qamomile documentation pages. They live
outside [docs/en/](../en/) and [docs/ja/](../ja/), so they are not rendered,
synced, or executed by the documentation build.

## How to use

1. Choose the template pair for your article from the table below.
2. Copy the English template into `docs/en/<section>/<slug>.py` and its Japanese companion into `docs/ja/<section>/<slug>.py`. Use the same article filename in both directories.
3. Replace the English placeholders with the article, then translate it using the Japanese headings and wording as a guide.
4. Keep the tags, section order, and executable examples aligned between languages.
5. Update the matching section `index.md`.

Use one notebook for one purpose: each page should teach one workflow, algorithm,
integration, or feature. If a topic needs multiple goals, split it into
multiple pages.

## Available skeletons

The skeleton files are the source of truth for page shape and section-specific
body outlines.

| English | Japanese | Use for |
| --- | --- | --- |
| [tutorial.py](tutorial.py) | [tutorial_ja.py](tutorial_ja.py) | Teaching a new user a Qamomile workflow without deep side topics. |
| [algorithm.py](algorithm.py) | [algorithm_ja.py](algorithm_ja.py) | Explaining an algorithm through Qamomile implementation and results. |
| [usage.py](usage.py) | [usage_ja.py](usage_ja.py) | Explaining one Qamomile feature, API pattern, or workflow detail. |
| [integration.py](integration.py) | [integration_ja.py](integration_ja.py) | Walking through a specific workflow that connects Qamomile to another tool or service. |
| [sdk_support.py](sdk_support.py) | [sdk_support_ja.py](sdk_support_ja.py) | Documenting which features Qamomile supports through a particular SDK or IR (intermediate representation), with examples for its execution targets. |

## Writing an SDK or IR support page

1. Replace `SDK_NAME` with the public name and `OFFICIAL_URL` with the official
   page for that language, if available. Replace the citation key with a
   matching bibliography entry, as in the [Braket guide](../en/integration/braket_support.py).
   Check that the official page appears in the generated References.
2. Replace `SDK_EXTRA` in both install commands with the target SDK's
   package extra. If the base package suffices, use `pip install qamomile`.
   Import the local endpoint and replace `local = ...` with its initialization
   before running the examples. Use the package name without a version pin.
   For an IR, name the runner that executes it, or explain what the reader
   can export or inspect.
3. Keep all eight table rows in order. Name the actual endpoints in the
   column headers and replace every `…` with ✅, ✗, or Device-dependent
   (機種依存). Start with the [Qamomile requirements](../../qamomile/effects.py)
   and the `capabilities` properties of the compiler target and execution
   endpoint. Check that the compiler and runner implement the feature for
   that endpoint. Explain model-specific or partial support
   beside the table, including differences between the measurement bases or
   the two loop forms. An export-only route should be labeled as such.
4. Keep the local subsections for the supported rows, with the same headings,
   order, examples, and prose as the template. Adapt an example if the target
   requires it and explain the difference. Execute each example on the named
   local endpoint, check its counts, and try a second input with a predicted
   result. Keep assertions specific to the inputs shown in the article.
   If Qamomile has no local runner, say so in Local simulation and show how
   to compile and inspect or export the Bell circuit if that route exists.
5. For hardware, replace `HARDWARE_NAME` and `HARDWARE_ENDPOINT` with a
   concrete device and its constructor. Put its imports in Setup. Show
   submission of the same `bell_program` in a Markdown code block. If the
   local angle subsection is omitted, move the Bell circuit definition and
   `bell_program` into Setup. For a target that needs the angle at compile
   time, declare `theta` as `qm.Const[qm.Real]`, specialize with
   `qm.specialize(parameterized_bell, theta=math.pi)`, and omit
   `parameters` at submission. If Qamomile has no hardware endpoint,
   state that in the hardware section, describe the available route, and
   adjust the introduction and table columns accordingly.

## Authoring conventions

- Use jupytext percent format. Markdown cells start with `# %% [markdown]`,
  and code cells start with `# %%`.
- Keep the jupytext and kernelspec metadata header from the skeleton.
- Put the tag frontmatter in the first markdown cell.
- Start with an H1 and a short scope paragraph that says what the notebook will cover.
- Include a commented Colab install cell and a single imports cell near the top.
- Prefer Qamomile-native code, diagrams, assertions, and result checks wherever possible.
- Use `:::{note}` for tips, notes, and remarks.
- Use MyST cross-references, citations, or external links instead of hand-writing a References section.
- SDK and IR support pages end with `## Hardware execution` (`## 実機での実行`).
  The other skeletons end with `## Summary` (`## まとめ`) and explicit take-home messages.
