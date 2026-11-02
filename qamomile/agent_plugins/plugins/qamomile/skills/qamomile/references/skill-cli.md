# Read or install the packaged skill

Use the environment where Qamomile is installed. In a uv project prefix
commands with `uv run`; `python -m qamomile` is the equivalent entrypoint.

```bash
qamomile skill show
qamomile skill show references/quantum/control.md
qamomile skill path
qamomile plugin path
qamomile marketplace path
qamomile --version
```

`show` reads an exact bundled Markdown path (default: SKILL.md). `skill path`
prints the directory containing `qamomile/SKILL.md`. The other paths expose
the bundled plugin and local marketplace catalogs. These path commands only
print local paths; they do not install, register, or enable integrations.

Inspect the resources and obtain the user's approval before installing skills,
registering marketplaces, or enabling plugins through either method below.

## Marketplace installation

```bash
codex plugin marketplace add "$(qamomile marketplace path)"
codex plugin add qamomile@qamomile

claude plugin marketplace add "$(qamomile marketplace path)"
claude plugin install qamomile@qamomile
```

Use `$(uv run qamomile marketplace path)` in a uv project. The agent's plugin
manager owns installed copies and their updates. Plugin metadata has its own
version; skill content comes from the installed Qamomile package.

## Project skill installation

```bash
qamomile skill install --agent codex
qamomile skill status --agent codex
qamomile skill update --agent codex
qamomile skill remove --agent codex
```

Use `--agent claude-code` for Claude Code and `--project PATH` for another
existing project; default is the current directory. Destinations are
`.agents/skills/qamomile/` and `.claude/skills/qamomile/`. Reload the agent if
needed. Choose one installation method per agent to avoid duplicate discovery.

The CLI manages its own copies, protects local edits, and records version/hashes.
Update/remove preserve a printed `.qamomile-backup-*` outside skill discovery.
`status` exits 0 for current, 1 for absent/unmanaged/modified/out of sync, or 2
for errors. Other commands exit 0 on success or 2 on errors. A current update
is a no-op. Upgrade Qamomile through your dependency manager before updating.
