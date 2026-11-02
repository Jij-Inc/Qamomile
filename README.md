# Qamomile

[![PyPI version](https://badge.fury.io/py/qamomile.svg)](https://badge.fury.io/py/qamomile)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

**Documentation:** [English](https://jij-inc-qamomile.readthedocs-hosted.com/latest/en/) | [日本語](https://jij-inc-qamomile.readthedocs-hosted.com/latest/ja/)

> [!WARNING]
> Qamomile is under active development. Its APIs may still change before the
> first stable release.

Qamomile is a source-compiled, type-safe functional quantum programming
language embedded in Python. One program can be inspected, estimated
symbolically, executed by the native reference runtime, or compiled for an SDK
backend without rewriting the algorithm.

Python is the familiar source language; Rust owns the compiler semantics,
verification, canonical QCore representation, resource analysis, target
planning, and reference execution. Quantum values are linear values rather
than mutable circuit handles, so ownership mistakes are rejected before a
backend artifact exists.

```text
Python source -> verified QCore -> symbolic resources
                              -> Reference execution
                              -> Qiskit / Cirq / QURI Parts / CUDA-Q / Braket / HUGR
```

## Installation

Qamomile requires Python 3.11 or newer. The distribution includes the native compiler in one
ABI3 wheel (a binary package compatible with Python 3.11 and newer):

```bash
uv add qamomile
```
or
```bash
pip install qamomile
```

Install an SDK integration only when it is needed:

```bash
pip install "qamomile[qiskit]"
pip install "qamomile[quri_parts]"
pip install "qamomile[braket]"
pip install "qamomile[cudaq-cu13]"  # Apple Silicon or CUDA 13 on Linux
pip install "qamomile[hugr]"
pip install "qamomile[quration]"
pip install "qamomile[qir]"
```

For repository development, install the prerequisites in [Development](#development)
first. `uv` builds Qamomile and its bundled Rust compiler as part of the locked environment:

```bash
git clone https://github.com/Jij-Inc/Qamomile.git
cd Qamomile
uv sync --frozen --group dev
```

Alternatively, install the source checkout with pip from the repository root.
The same build prerequisites apply. This command installs Qamomile and its bundled Rust compiler together:

```bash
pip install -e .
```

## Coding agent skill

The installed package includes a Qamomile authoring plugin, local marketplace
catalogs, and a CLI. The `skill path`, `plugin path`, and `marketplace path`
commands only print local paths; they do not install, register, or enable
integrations. Coding agents must inspect the resources and obtain the user's
approval before installing skills, registering marketplaces, or enabling
plugins. After approval, register the marketplace and install the plugin:

```bash
codex plugin marketplace add "$(uv run qamomile marketplace path)"
codex plugin add qamomile@qamomile
```

For Claude Code:

```bash
claude plugin marketplace add "$(uv run qamomile marketplace path)"
claude plugin install qamomile@qamomile
```

`uv run qamomile plugin path` prints the plugin directory. Both marketplaces
and direct skill installation use the same packaged skill. The agent's plugin
manager owns marketplace installations; the Qamomile CLI owns only copies made
with `skill install`. Choose one installation method per agent to avoid
duplicate discovery. Read or install the skill directly with:

```bash
uv run qamomile skill show
uv run qamomile skill show references/authoring.md
uv run qamomile skill install --agent codex
uv run qamomile skill status --agent codex
```

Use `--agent claude-code` for Claude Code and `--project PATH` to select another
existing project directory. Codex uses `.agents/skills/qamomile/`; Claude Code
uses `.claude/skills/qamomile/`. Reload the agent's project/session if needed
for discovery. Commands also work as `python -m qamomile`.

After upgrading Qamomile, use `skill update` with the same agent and project.
Local edits and unmanaged copies are protected. Update and remove preserve
the previous copy outside skill discovery. `skill path` prints the bundled
skills directory; `skill show references/skill-cli.md` explains every command
and exit status. The CLI operates on local files without a remote catalog.

## Quick start

Qamomile compiles function bodies from Python source without executing them.
Save the program in a source file such as `bell.py`:

```python
import qamomile as qm


@qm.qkernel
def bell() -> qm.Bits[2]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)
```

The same function can be estimated and executed directly:

```python
import qamomile as qm

from bell import bell


study = qm.resources.logical(bell)
print(study.metric(qm.resources.LOGICAL_GATES))

record = qm.run(bell, on=qm.Reference(), shots=128, seed=7)
print(record.result.counts)
```

There is no hidden mutable circuit. Each operation consumes an owned quantum
value and returns its successor. For example, writing `qm.h(state)` without
rebinding the result is rejected instead of silently discarding a state
transition.

Hamiltonian construction can remain algorithmic while instance coefficients
stay ordinary classical data:

```python
from __future__ import annotations

import qamomile as qm


@qm.qkernel
def transverse_field(
    n: qm.Const[qm.Int],
    coefficients: qm.Const[qm.Vector[qm.Real, n]],
) -> qm.Observable:
    hamiltonian = qm.zero_observable(n)
    for index in range(n):
        hamiltonian = hamiltonian + coefficients[index] * qm.X(index)
    return hamiltonian
```

The `for` remains a structured QCore loop in the reusable recipe. Binding
`n` and `coefficients` materializes one canonical `Hamiltonian` through the
Rust observable algebra; SDK backends receive the same result without owning
another expression evaluator. Observable recipes are `Const` in this first
surface: launch-time `Param` coefficients are rejected before an artifact is
created.

`n` in `Vector[Real, n]` is Qamomile compiler syntax for a dependent
dimension, rather than a standard Python type variable. Keep the future import
when this source file may be imported by Python so the interpreter does not
eagerly evaluate `n`; `qm.build("file.py")` parses the annotation directly.
Ruff and Zuban can check a Qamomile source module without cluttering each
dependent annotation by placing these module-wide directives at the top of
that module:

```python
# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.
# mypy: disable-error-code="valid-type"
```

These are module-wide checker accommodations, so reserve them for Qamomile
source modules. Qamomile still validates the dimension binding and its uses in
the Rust compiler.

## Explicit compiler lifecycle

Interactive code can use `qm.run(...)`. Tools that need durable boundaries can
use the complete lifecycle:

```text
source -> Module -> ProgramFamily -> Program -> Artifact -> RunRecord
```

```python
import qamomile as qm


module = qm.build("bell.py")
program = qm.specialize(module["bell"])
study = qm.resources.logical(program)
artifact = qm.compile(study, to=qm.Reference())
job = qm.submit(artifact, on=qm.Reference(), shots=128, seed=7)
result = job.result()
```

Module, program, resource-study, and artifact boundaries use deterministic,
validated representations. Submitting an existing artifact never reparses or
recompiles Python source. The lower-level `submit(...)` path returns a `Job`
when asynchronous control is needed; `run(...)` retains the completed
artifact, invocation, and result together as a `RunRecord`.

## Backend namespaces

- `qamomile`: language surface, compiler lifecycle, reference runtime,
  observables, standard library, optimization helpers, and symbolic resources
- `qamomile.qiskit`: Qiskit lowering and Aer execution
- `qamomile.quration`: concrete PyQret circuits, local simulation, and Quration FTQC compilation
- `qamomile.qir`: QIR 1.0 Base Profile export as LLVM text and bitcode
- `qamomile.cirq`: Cirq JSON, local/noisy simulation, and Google endpoints
- `qamomile.quri_parts`: QURI Parts lowering and Qulacs execution
- `qamomile.cudaq`: CUDA-Q lowering and local execution
- `qamomile.braket`: OpenQASM 3 lowering and Amazon Braket execution
- `qamomile.hugr`: structured QCore lowering to HUGR and Selene

Backend integrations exist only under the nested `qamomile.<backend>`
namespaces.

With `qamomile[qir]`, export a static measurement entry to QIR for an external
consumer. Using `bell` from the quick start:

```python
from pathlib import Path

import qamomile as qm
from bell import bell

from qamomile.qir import QIR, load_bitcode, load_ir

artifact = qm.compile(bell, to=QIR())
Path("bell.ll").write_text(load_ir(artifact), encoding="utf-8")
Path("bell.bc").write_bytes(load_bitcode(artifact))
```

Specialize `Const` inputs before export. The initial Base Profile supports
terminal measurements and records a flat bit array in logical source order.
Runtime parameters, measurement-dependent control, reset, discard, gates after
measurement, and computed classical outputs are rejected. The artifact retains
the typed result ABI and supports `qamomile.persistence.serialize`/`deserialize`;
the `.bc` file is the standalone module for the external QIR runtime.

The repository contains one Qamomile source implementation. Compiler semantics
and execution live in Rust; Python contains the source adapter, public values,
and SDK-specific integration boundaries.

## Development

Run these commands from the repository root. Source checkouts require:

- Git and `uv` for the locked Python environment.
- Python 3.11 or newer; Python 3.12 matches the documentation build environment.
- `rustup`, which selects Rust **1.98.0**, Clippy, and rustfmt from
  [`rust-toolchain.toml`](rust-toolchain.toml).
- A native linker/toolchain: Xcode Command Line Tools on macOS, C/C++ build
  tools on Linux, or the MSVC C++ build tools on Windows.

A separate Protobuf compiler is unnecessary: normal builds use checked-in wire
bindings, and schema tests use the vendored compiler.

```bash
rustc --version
uv sync --frozen --group dev
uv run --no-sync python -c "import qamomile, qamomile._qamomile_native"
uv run --no-sync pytest
uv run --no-sync ruff check qamomile tests
uv run --no-sync ruff format --check qamomile tests
uv run --no-sync zuban check qamomile tests/typecheck
uv run --no-sync cargo test --workspace --locked
```

`uv sync` installs Qamomile in editable mode and builds its bundled native
compiler. After changing Rust code, rebuild before testing Python:

```bash
uv sync --frozen --group dev --reinstall-package qamomile
uv run --no-sync pytest rust/qamomile-python/tests
```

The default pytest profile excludes docs, several optional SDKs, and slow
suites. A test filename alone does not override these exclusions. For example,
run QURI Parts coverage with:

```bash
uv sync --frozen --group dev --extra quri_parts
uv run --no-sync pytest -m "quri_parts and not backend_slow" tests/qamomile/test_quri_parts_backend.py
```

Keep the selected extras/groups when syncing again, or use `uv run --no-sync`
in the prepared environment. The HUGR and Guppy extras require separate syncs
or environments.

### Repository map

| Path | Responsibility |
| --- | --- |
| [`qamomile/`](qamomile/__init__.py) | Python source capture, immutable public facades, library composition, SDK adapters. |
| [`rust/qamomile-compiler/`](rust/qamomile-compiler/src/lib.rs) | Compiler semantics, verified QCore, lowering, resource analysis, reference execution. |
| [`rust/qamomile-python/`](rust/qamomile-python/src/lib.rs) | PyO3 conversion and projection boundary; no independent compiler semantics. |
| [`tests/`](tests/) | Python API, backend, architecture, and documentation checks. |
| [`docs/`](docs/README.md) | English/Japanese documentation sources, notebooks, and build tools. |

Learn the language and lifecycle in the [tutorials](docs/en/tutorial/index.md).
Package and crate documentation describe implementation boundaries alongside
the code.

### Writing documentation

Documentation authors also need Bash and Node.js **24** (the version selected
by [Read the Docs](.readthedocs.yaml)). From the repository root:

```bash
uv sync --frozen --group docs
UV_NO_SYNC=1 ./docs/build.sh build-en
UV_NO_SYNC=1 ./docs/build.sh serve-en
```

This first preview uses committed notebook outputs. For executable changes,
edit the Jupytext `.py` source, then sync and execute the selected page with
`page-build`. See [the documentation quick start](docs/README.md#quick-start)
for kernel setup, Japanese previews, and page validation.

## Contributing

Contributions, bug reports, and feedback are welcome through GitHub Issues and
pull requests.

## License

Qamomile is released under the [Apache 2.0 License](LICENSE.txt).

## QCE 2026 Tutorial

The Qamomile team delivered **Introduction to Qamomile: Estimating Symbolically & Executing Concretely** at **IEEE Quantum Week (QCE 2026)** on September 14, 2026. See the official program for [Part 1](https://qce.quantum.ieee.org/2026/qce26-schedule/tutorial-schedule/#84-1) and [Part 2](https://qce.quantum.ieee.org/2026/qce26-schedule/tutorial-schedule/#84-2).

## Citation

If you use Qamomile in research, please cite:

```bibtex
@INPROCEEDINGS{11249901,
  author={Huang, Wei-Hao and Matsuyama, Hiromichi and Tam, Wai-Hong and Sato, Keisuke and Yamashiro, Yu},
  booktitle={2025 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  title={Qamomile: A Cross-SDK Bridge for Quantum Optimization},
  year={2025},
  volume={02},
  pages={516-517},
  doi={10.1109/QCE65121.2025.10423}
}
```
