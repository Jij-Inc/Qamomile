"""Compile prepared Qamomile semantics through explicit target pipelines.

Design center
-------------

The stable user workflow remains circuit-first: an engine ``Transpiler``
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
    CircuitProgram                (immutable engine-neutral codegen IR)
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

Program-graph executables reuse the shared jobs, execution handles,
capabilities, and public ABI exported by this facade. Targets retain control
of native execution while presenting the same result and lifecycle contracts.

Design principles
-----------------

- **Keep semantic IR abstract and lower late.** Per-qubit encoding, native gate
  selection, transformed-call expansion, and runtime control-flow syntax are
  target concerns. Segmentation lowers semantics only when separating host and
  quantum execution requires it.
- **Use an immutable circuit boundary.** Circuit engines consume verified
  ``CircuitProgram`` values rather than walking mutable semantic IR or sharing
  engine objects through emit-context side channels. The current
  semantic-to-circuit implementation reuses the established walk internally,
  but that walker is not an engine extension API.
- **Make target pipelines explicit.** A ``CompilationTarget`` owns planning,
  lowering/materialization, and native validation. Circuit targets declare the
  complete input language accepted by their materializer; shared legalization
  fixes realization choices and target verification enforces that declaration
  before native object construction begins.
- **Preserve dependency direction.** This package and ``qamomile.circuit`` do
  not import SDK engines. Engine packages depend on the public compiler,
  circuit-IR, executable, and artifact contracts.
- **Keep ``bindings`` and ``parameters`` disjoint.** Bindings determine
  compile-time values and structure; parameters survive as runtime artifact
  inputs. Overlap is rejected before compilation, and structural decisions
  such as classical-value branches and range bounds must use bindings.
- **Keep the common user surface small.** Engine users normally interact with
  an engine ``Transpiler`` and executor. Materializers, source writers, and
  engine artifact wrappers are implementation details unless a target exposes
  a distinct native compilation product intentionally.
"""

from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.transpiler.artifact import (
    CompilationDiagnostic,
    CompilationMetadata,
    CompiledProgram,
    DiagnosticSeverity,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.config import CompilerConfig, TranspilerConfig
from qamomile.circuit.transpiler.errors import (
    CallableDefinitionConflictError,
    EmitError,
    ExecutionError,
    TargetCapabilityError,
)
from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    ExecutionHandle,
    ExecutionReference,
    JobStatus,
    MappedExecutionHandle,
)
from qamomile.circuit.transpiler.execution_request import (
    EstimationAccuracy,
    Exact,
    ShotBased,
    TargetPrecision,
)
from qamomile.circuit.transpiler.execution_snapshot import (
    ExecutionSnapshot,
    ExecutionSnapshotKind,
)
from qamomile.circuit.transpiler.job import (
    ExpvalJob,
    Job,
    JobKind,
    JobSnapshot,
    RunJob,
    SampleResult,
    aggregate_typed_results,
)
from qamomile.circuit.transpiler.param_keys import dict_param_key
from qamomile.circuit.transpiler.parameter_binding import flatten_user_bindings
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    lower_compile_time_ifs_preserving_loop_conditions,
)
from qamomile.circuit.transpiler.prepared import PreparedModule, prepare_module
from qamomile.circuit.transpiler.program_graph import (
    inline_callables,
    validate_program_graph_semantics,
)
from qamomile.circuit.transpiler.segments import ClassicalSegment, ProgramABI
from qamomile.circuit.transpiler.target import CompilationTarget

__all__ = [
    "CallableDefinitionConflictError",
    "ClassicalExecutor",
    "ClassicalSegment",
    "CompilationDiagnostic",
    "CompilationMetadata",
    "CompilationTarget",
    "CompletedExecutionHandle",
    "CompiledProgram",
    "CompilerConfig",
    "CompositeExecutionHandle",
    "DiagnosticSeverity",
    "EmitError",
    "EstimationAccuracy",
    "Exact",
    "ExecutionCapabilities",
    "ExecutionContext",
    "ExecutionError",
    "ExecutionHandle",
    "ExecutionReference",
    "ExecutionSnapshot",
    "ExecutionSnapshotKind",
    "ExpvalJob",
    "Job",
    "JobKind",
    "JobSnapshot",
    "JobStatus",
    "MappedExecutionHandle",
    "PreparedModule",
    "ProgramABI",
    "QKernelLike",
    "QamomileCompiler",
    "RunJob",
    "SampleResult",
    "ShotBased",
    "TargetCapabilityError",
    "TargetPrecision",
    "TranspilerConfig",
    "aggregate_typed_results",
    "dict_param_key",
    "flatten_user_bindings",
    "inline_callables",
    "lower_compile_time_ifs_preserving_loop_conditions",
    "pair_block_operands",
    "prepare_module",
    "validate_program_graph_semantics",
]
