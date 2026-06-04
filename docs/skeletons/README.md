# Documentation Skeletons

These files are authoring templates for Qamomile documentation pages. They live
outside [docs/en/](../en/) and [docs/ja/](../ja/), so they are not rendered,
synced, or executed by the documentation build.

## How to use

1. Copy the closest skeleton into `docs/en/<section>/<slug>.py`.
2. Create the mirrored Japanese page under `docs/ja/<section>/<slug>.py`.
3. Keep `tags:` identical between languages.
4. Replace the placeholder sections with one focused story.
5. Update the matching section `index.md`.

Use one notebook for one purpose: each page should teach one workflow, algorithm,
integration, or feature. If a topic needs multiple goals, split it into
multiple pages.

## Available skeletons

The skeleton files are the source of truth for page shape and section-specific
body outlines.

| File | Use for |
| --- | --- |
| [tutorial.py](tutorial.py) | Teaching a new user a Qamomile workflow without deep side topics. |
| [algorithm.py](algorithm.py) | Explaining an algorithm through Qamomile implementation and results. |
| [usage.py](usage.py) | Explaining one Qamomile feature, API pattern, or workflow detail. |
| [integration.py](integration.py) | Explaining how Qamomile works with another SDK, service, or package. |

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
- End with `## Summary` and explicit take-home messages.
