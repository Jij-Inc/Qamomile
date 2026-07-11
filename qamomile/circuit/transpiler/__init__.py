"""Compile prepared Qamomile semantics through explicit target pipelines.

Design center
-------------

The stable user workflow remains circuit-first: a backend ``Transpiler``
accepts a qkernel, preserves compile-time ``bindings`` and runtime
``parameters``, and returns an ``ExecutableProgram``. Internally, compilation
now separates frontend preparation from target lowering so circuit SDKs and
program-graph targets do not have to pretend to consume the same abstraction.

Shared preparation
------------------

``QamomileCompiler.prepare()`` performs the target-independent prefix:

::

    QKernelLike
       │  trace + validate entrypoint
       │  substitute configured callables
       │  resolve parameter-array shapes
       ▼
    PreparedModule
       ├─ hierarchical semantic entrypoint
       ├─ reachable callable definitions
       ├─ call graph
       └─ public classical ABI

``PreparedModule`` deliberately preserves structured control flow and callable
boundaries. It is the last representation shared by every target family.

Target families
---------------

Circuit-family SDKs such as Qiskit, QURI Parts, CUDA-Q, and PyQret retain the
existing ``Transpiler`` execution UX and take the host-orchestrated path:

::

    PreparedModule
       │  inline + recursion unroll + affine/borrow validation
       │  partial evaluation + classical lowering + shape validation
       │  segment into C → Q → C
       ▼
    ProgramPlan
       │  lower quantum segments once
       ▼
    CircuitProgram                (immutable backend-neutral codegen IR)
       │  verify ordered linear wires, regions, calls, and expressions
       │  legalize + verify target capability declarations
       │  materialize native SDK objects
       ▼
    ExecutableProgram[ArtifactT]  (sampling/expectation orchestration)

Program-graph targets such as HUGR compile the preserved program structure
directly instead of passing through circuit segmentation:

::

    PreparedModule
       │  CompilationTarget.plan()
       │  CompilationTarget.compile()
       │  CompilationTarget.validate()
       ▼
    CompiledProgram[ArtifactT]    (artifact + ABI + diagnostics + metadata)

The two results are intentionally different. ``ExecutableProgram`` represents
Qamomile's host-driven execution model; ``CompiledProgram`` packages a native
module or graph whose runtime model belongs to the target.

Design principles
-----------------

- **Keep semantic IR abstract and lower late.** Per-qubit encoding, native gate
  selection, transformed-call expansion, and runtime control-flow syntax are
  target concerns. Segmentation lowers semantics only when separating host and
  quantum execution requires it.
- **Use an immutable circuit boundary.** Circuit backends consume verified
  ``CircuitProgram`` values rather than walking mutable semantic IR or sharing
  backend objects through emit-context side channels. The current
  semantic-to-circuit implementation reuses the established walk internally,
  but that walker is not a backend extension API.
- **Make target pipelines explicit.** A ``CompilationTarget`` owns planning,
  lowering/materialization, and native validation. Circuit targets declare the
  complete input language accepted by their materializer; shared legalization
  fixes realization choices and target verification enforces that declaration
  before native object construction begins.
- **Preserve dependency direction.** This package and ``qamomile.circuit`` do
  not import SDK backends. Backend packages depend on the public compiler,
  circuit-IR, executable, and artifact contracts.
- **Keep ``bindings`` and ``parameters`` disjoint.** Bindings determine
  compile-time values and structure; parameters survive as runtime artifact
  inputs. Overlap is rejected before compilation, and structural decisions
  such as classical-value branches and range bounds must use bindings.
- **Keep the common user surface small.** Backend users normally interact with
  a backend ``Transpiler`` and executor. Materializers, source writers, and
  backend artifact wrappers are implementation details unless a target exposes
  a distinct native compilation product intentionally.
"""

from qamomile.circuit.transpiler.artifact import (
    CompilationDiagnostic,
    CompilationMetadata,
    CompiledProgram,
    DiagnosticSeverity,
)
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.config import CompilerConfig, TranspilerConfig
from qamomile.circuit.transpiler.prepared import PreparedModule, prepare_module
from qamomile.circuit.transpiler.target import CompilationTarget

__all__ = [
    "CompilationDiagnostic",
    "CompilationMetadata",
    "CompilationTarget",
    "CompiledProgram",
    "CompilerConfig",
    "DiagnosticSeverity",
    "PreparedModule",
    "QamomileCompiler",
    "TranspilerConfig",
    "prepare_module",
]
