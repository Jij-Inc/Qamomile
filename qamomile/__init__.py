"""Qamomile: a source-compiled functional quantum programming language.

The top-level namespace is the ordinary programming lifecycle: source
qkernels and types, semantic transforms, observables, compilation, and
execution. Compiler IR, target-author protocols, proof facts, and resource
model internals live in their dedicated submodules instead of burdening normal
application code. Circuit compilation can retain a native plan of closed quantum
segments. Measurement rows remain correlated across those segments and determine
the next shot populations; supporting classical computation stays compiler-owned.
Compiler pass composition lives in ``qamomile.compilation``. An explicit
``optimizer=`` selects optional symbolic, QCore, realization, and circuit
optimizations while Qamomile retains verification, target emission, and execution
ownership. Passes receive native-validated device facts as an immutable
``TargetProfile`` when the target supplies one, and programs carry untrusted
``intent`` rows (declaring kernel, docstring, problem and tolerance tags) that
never change meaning or semantic identity.

Coding agents: load the bundled Qamomile skill before authoring quantum code.
Run ``uv run qamomile skill show`` to read SKILL.md, then read the references
linked for the task with ``uv run qamomile skill show references/authoring.md``
or another referenced path. Locate the optional bundled integrations with::

    uv run qamomile skill path
    uv run qamomile plugin path
    uv run qamomile marketplace path

These path commands only print local paths; they do not install, register, or
enable integrations. Inspect the resources and obtain the user's approval
before installing skills, registering marketplaces, or enabling plugins.

After approval, if the skill is not already available to the agent, install it
from the user's project directory (or specify ``--project PATH``)::

    uv run qamomile skill install --agent codex
    uv run qamomile skill status --agent codex

Use ``--agent claude-code`` for Claude Code. The project destinations are
``.agents/skills/qamomile/`` and ``.claude/skills/qamomile/`` respectively.
Reload the agent's project/session if needed, and use ``skill update`` with
the same options after upgrading Qamomile.

Alternatively, install through the agent's marketplace manager::

    codex plugin marketplace add "$(uv run qamomile marketplace path)"
    codex plugin add qamomile@qamomile

    claude plugin marketplace add "$(uv run qamomile marketplace path)"
    claude plugin install qamomile@qamomile

Choose one installation method per agent. Outside a uv project, use
``qamomile`` or ``python -m qamomile`` in the environment containing Qamomile.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        braket as braket,
        cirq as cirq,
        cudaq as cudaq,
        hugr as hugr,
        qbraid as qbraid,
        qir as qir,
        qiskit as qiskit,
        quration as quration,
        quri_parts as quri_parts,
        visualization as visualization,
    )

from . import linalg, resources, statistics
from .compilation import (
    CompiledProgram,
    ExecutionReference,
    ExecutionSnapshot,
    ExecutionSnapshotKind,
    Invocation,
    Job,
    JobStatus,
    Reference,
    RunRecord,
    SampleResult,
    ValueResult,
    compile,
    evaluate,
    execute,
    execute_batch,
    run,
    submit,
)
from .errors import (
    ArtifactError,
    CapabilityError,
    Diagnostic,
    ExecutionError,
    FrontendError,
    OwnershipError,
    QamomileError,
    SerializationError,
    SourceUnavailableError,
    SpecializationError,
    TypeCheckError,
    VerificationError,
)
from .expectation import (
    ExpectationResult,
    PauliEstimate,
    PauliMeasurementGroup,
    PauliReadout,
    expectation,
)
from .intrinsics import (
    add_into,
    adjoint,
    as_bits,
    as_fixed,
    as_uint,
    ccx,
    ceil,
    concat,
    control,
    cp,
    cx,
    cz,
    discard,
    global_phase,
    h,
    less_than_xor,
    log2,
    measure,
    measure_reset,
    measure_x,
    measure_y,
    measure_z,
    mod_add,
    mod_mul,
    mod_mul_add,
    mod_multi_pow,
    mod_pow,
    mul_add,
    multiplex,
    p,
    pauli_evolve,
    permute_by,
    phase_by,
    prepare,
    qubits,
    reset,
    rx,
    ry,
    rz,
    rzz,
    s,
    sdg,
    split_at,
    sqrt,
    swap,
    t,
    tdg,
    x,
    xor_by,
    y,
    z,
)
from .lowering import (
    RealizationInfo as RealizationInfo,
    RealizationOptions as RealizationOptions,
    list_realizations as list_realizations,
    realize as realize,
)
from .observable import (
    Hamiltonian,
    Pauli,
    PauliFactor,
    PauliWord,
    X,
    Y,
    Z,
    commutator,
    zero_observable,
)
from .source import (
    Module,
    Program,
    ProgramFamily,
    QKernel,
    build,
    build_source,
    composite_gate,
    qkernel,
    specialize,
)
from .types import (
    Bit,
    Bits,
    Bool,
    Complex,
    Const,
    Int,
    Map,
    Matrix,
    Observable,
    Param,
    Probability,
    Q,
    Real,
    Shot,
    Tensor,
    UFixed,
    UInt,
    Unit,
    Vector,
)

_LAZY_NAMESPACES = frozenset(
    {
        "braket",
        "cirq",
        "cudaq",
        "hugr",
        "qbraid",
        "qir",
        "qiskit",
        "quri_parts",
        "quration",
        "visualization",
    }
)


def __getattr__(name: str) -> ModuleType:
    """Load a backend or visualization namespace on first attribute access.

    Args:
        name (str): Attribute requested from the top-level package.

    Returns:
        ModuleType: Imported ``qamomile.<name>`` module.

    Raises:
        AttributeError: If name is not a supported lazy namespace.
        ImportError: If the requested facade cannot be imported.
    """

    if name not in _LAZY_NAMESPACES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """List ordinary exports together with lazy namespaces.

    Returns:
        list[str]: Sorted visible attribute names.
    """

    return sorted({*globals(), *_LAZY_NAMESPACES})


__all__ = [
    "add_into",
    "less_than_xor",
    "mod_add",
    "mod_mul",
    "mod_mul_add",
    "mod_multi_pow",
    "mod_pow",
    "mul_add",
    "adjoint",
    "as_bits",
    "as_fixed",
    "as_uint",
    "CompiledProgram",
    "ArtifactError",
    "Bit",
    "Bits",
    "Bool",
    "build",
    "build_source",
    "CapabilityError",
    "ccx",
    "ceil",
    "commutator",
    "compile",
    "composite_gate",
    "Complex",
    "concat",
    "Const",
    "control",
    "cp",
    "cx",
    "cz",
    "Diagnostic",
    "discard",
    "evaluate",
    "Program",
    "execute",
    "execute_batch",
    "ExecutionError",
    "ExecutionReference",
    "ExecutionSnapshot",
    "ExecutionSnapshotKind",
    "expectation",
    "ExpectationResult",
    "FrontendError",
    "QKernel",
    "global_phase",
    "h",
    "Hamiltonian",
    "Int",
    "Invocation",
    "Job",
    "JobStatus",
    "linalg",
    "log2",
    "Map",
    "Matrix",
    "measure",
    "measure_x",
    "measure_y",
    "measure_z",
    "measure_reset",
    "Module",
    "multiplex",
    "Observable",
    "OwnershipError",
    "p",
    "Param",
    "Pauli",
    "PauliEstimate",
    "PauliFactor",
    "PauliMeasurementGroup",
    "PauliReadout",
    "PauliWord",
    "pauli_evolve",
    "permute_by",
    "phase_by",
    "prepare",
    "Probability",
    "qkernel",
    "ProgramFamily",
    "Q",
    "QamomileError",
    "qubits",
    "Real",
    "RealizationInfo",
    "RealizationOptions",
    "Reference",
    "realize",
    "reset",
    "resources",
    "RunRecord",
    "run",
    "rx",
    "ry",
    "rz",
    "rzz",
    "s",
    "SampleResult",
    "sdg",
    "SerializationError",
    "Shot",
    "Tensor",
    "SourceUnavailableError",
    "specialize",
    "statistics",
    "visualization",
    "SpecializationError",
    "split_at",
    "sqrt",
    "submit",
    "swap",
    "t",
    "tdg",
    "TypeCheckError",
    "UInt",
    "UFixed",
    "Unit",
    "ValueResult",
    "Vector",
    "VerificationError",
    "X",
    "x",
    "xor_by",
    "Y",
    "y",
    "Z",
    "z",
    "zero_observable",
]
