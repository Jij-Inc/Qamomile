---
name: write-tutorial
description: Write or revise Qamomile documentation tutorials and algorithm articles (docs/{en,ja}/**/*.py jupytext, percent format). Defines the textbook-style pedagogical structure, the outline-first workflow, and the prose conventions distilled from author feedback. Use when drafting a new doc article, restructuring an existing one, or reviewing tutorial prose.
---

# Tutorial Writing Skill

Write Qamomile documentation that reads like a **textbook chapter**, not a feature tour.
Docs live under `docs/en/` and `docs/ja/` as jupytext `.py` files in percent format
(`# %%` code cells, `# %% [markdown]` prose cells).

Before touching any file under `docs/`, read `docs/README.md` for the build pipeline,
and `CLAUDE.md` for the docs-tag whitelist and the cross-backend execution-test rule.
For EN→JA translation mechanics (spacing, terminology), defer to the `translate` skill.

## Workflow: outline first, then prose

Never start writing prose into the file. Always:

1. **Draft the section / subsection outline first** and get it approved. List every
   `##` section and `###` subsection with a one-line intent each. Do not write body
   text yet.
2. Discuss and revise the outline until the structure is agreed.
3. **Then write, section by section.** When restructuring an existing article, reset
   the file and rebuild from the approved outline rather than patching in place.
4. Write incrementally — one section per round — and let the author react before
   continuing.

## Structural principles (textbook, not tutorial)

A tutorial lists tasks ("here is how to do X"). A textbook builds a mental model.
Educational docs in Qamomile should be the latter.

- **One claim per section.** Each section establishes a single idea; code is the
  *evidence* for that idea, not the point itself.
- **Limitation drives the next section.** End each section with the limitation that
  the next section exists to overcome. This single spine is what turns a list of
  topics into a chapter. Example: bit-flip code corrects only `X` → motivates the
  phase-flip code → neither alone corrects both → motivates Shor's code.
- **A recurring template for parallel topics.** When several items are structurally
  similar (e.g. several codes), give them the same subsection skeleton so the reader
  can predict the layout. Predictability is a learning aid.
- **Examples before theory.** Introduce a concept concretely first; consolidate the
  formal framework afterwards (often in a later article). Do not open with formalism.
- **No afterthought sections.** A "by the way, here is the general view" section
  tacked onto the end is the weakest possible placement. Either weave the framing in
  from the start, or make it the explicit subject of its own article.

## Prose principles: be concise

The author repeatedly cut verbosity. Default to short, plain, declarative sentences.

- **Every sentence must add information.** If a sentence only re-assures or restates,
  delete it.
- **Cut literary flourishes and double emphasis.** Avoid "腑に落ちる", "〜まで分かる",
  repeated "つまり/要するに", and chains of em-dashes.
- **Prefer the simple statement.** Before/after from real feedback:
  - ❌ 「量子の誤り訂正に入る前に、『そもそも何が難しいのか』をはっきりさせておきます。
    古典のやり方を量子へ持ち込もうとすると、どこで詰まるのか ― それを見ておくと、
    後で出てくるシンドローム測定という仕組みが『なぜそうなっているのか』まで腑に落ちます。」
  - ✅ 「実際の量子誤り訂正に入る前に、量子の場合は古典と何が違って難しいのかを説明します。」

## Logical continuity: no jumps

The reader must be able to follow every step.

- When a claim would feel like a leap, **break it into explicit stages.** Example: the
  jump "errors are continuous → therefore correcting X and Z is enough" needed two
  intermediate steps spelled out — (1) any error is a linear combination of
  `I/X/Y/Z`, (2) syndrome measurement collapses that superposition into one discrete
  Pauli error.
- A forward reference ("we confirm this in the next section") is fine to keep a bridge
  short, but it does not replace the bridge.

## Introducing terms

- **Never use an unknown technical term abruptly.** Define a term the first time it
  appears, before relying on it.
- **Define the noun before using it as a noun.** Naming an operation ("syndrome
  measurement") does not define the underlying noun ("syndrome"). Define the noun
  first, then the operation built on it.
- **Give the translation and the word-choice rationale when it aids understanding.**
  For a JA doc, gloss the term and, where the etymology illuminates the concept,
  explain it briefly. Example: シンドローム = 「症候群」; just as a doctor diagnoses a
  disease from its symptoms without observing it directly, error correction reads the
  error's "symptoms" without measuring the error itself.

## Japanese wording

- **Use katakana where it is the natural Japanese.** Do not force a kanji translation
  when the katakana form is what readers actually use: ノイズ (not 雑音),
  シンプル (not 素朴). Established technical terms stay as-is.
- For all other JA conventions (Japanese/Latin spacing, です・ます tone, which terms
  stay in English), follow the `translate` skill.

## Content selection: write for the reader, not the conversation

- **Do not add a note just because a point came up while authoring.** A clarification
  that surfaced in discussion belongs in the text only if it earns its place there on
  its own. If it is not worth a standalone note for a fresh reader, cut it.
- The article is for the reader, not a transcript of how it was written. Review drafts
  for sentences that exist only because of the authoring discussion and remove them.

## Final checklist

Before considering a section done:

- [ ] The section establishes one clear claim.
- [ ] It ends with the limitation that motivates the next section (where applicable).
- [ ] No sentence is purely reassurance or restatement.
- [ ] No technical term is used before it is defined.
- [ ] No logical jump — every step is bridged.
- [ ] No sentence exists only because of the authoring conversation.
- [ ] Katakana is used where it is the natural Japanese form (JA docs).
- [ ] Tags in the frontmatter are within the `CLAUDE.md` whitelist.
- [ ] New/changed `algorithm/` or `stdlib/` kernels have cross-backend execution tests.
