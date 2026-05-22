"""First-class manifest of a kernel's classical parameter interface.

This module defines ``ParamSlot`` and ``ParamKind``, which together
describe every classical (non-quantum) argument of a ``@qkernel``
function so the kernel's parameter contract is recoverable from the
IR alone — without an external Python-side manifest.

Motivation:
    The project rule documented in ``CLAUDE.md`` keeps ``bindings`` and
    ``parameters`` strictly disjoint at the ``Transpiler.transpile()``
    API boundary, but the IR itself does not record which name was
    decided which way. After ``partial_eval`` folds a binding into a
    concrete constant, downstream readers cannot tell whether a
    constant value originated from a compile-time binding or was a
    literal in the kernel source. This is especially limiting for the
    "qamomile as subgraph of an outer DSL's computation graph" use
    case, where the receiver needs to know the kernel's full classical
    interface (name, type, default, runtime-or-bound) to rebind values
    in subsequent calls.

    Per-kernel-argument metadata also makes it natural to attach
    optional hints (currently just ``differentiable``) for outer DSL
    tooling such as parameter-shift gradient back-ends.

Scope:
    ``ParamSlot`` covers only classical (non-quantum) arguments. Qubit
    and ``Vector[Qubit]`` inputs are not part of the parameter slot
    manifest; they appear in ``Block.input_values`` instead.

    The companion ``Block.parameters: dict[str, Value]`` field is
    retained for callers (passes, emitters) that need a direct Value
    reference for runtime parameters; ``Block.param_slots`` is the
    canonical, fully-typed contract that survives the pipeline.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

if typing.TYPE_CHECKING:
    from qamomile.circuit.ir.types import ValueType


class ParamKind(enum.Enum):
    """Lifecycle classification for a classical kernel argument.

    Values:
        RUNTIME_PARAMETER: The argument is intended to be bound at
            execution time by the backend (or, more generally, by the
            outer caller in a hybrid loop). It survives the
            compilation pipeline as a symbolic parameter.
        COMPILE_TIME_BOUND: The argument was provided as a binding
            (or via a Python default) and is folded into the IR by
            ``resolve_parameter_shapes`` / ``partial_eval``. No
            symbolic counterpart remains in the emitted circuit.
    """

    RUNTIME_PARAMETER = "runtime_parameter"
    COMPILE_TIME_BOUND = "compile_time_bound"


@dataclasses.dataclass(frozen=True)
class ParamSlot:
    """Metadata for a single classical kernel argument.

    A ``ParamSlot`` describes one position in the kernel's classical
    parameter contract — its declared type, whether it is a runtime
    parameter or a compile-time-bound value, the Python default (if
    any), the actually-bound value (when ``kind`` is
    ``COMPILE_TIME_BOUND``), and any outer-DSL hints. Slots are
    immutable; pipeline passes that need to update a slot must clone
    via ``dataclasses.replace``.

    The slot is identified by ``name``, which matches the kernel's
    Python parameter name and the corresponding entry in
    ``Block.label_args``. A slot's ``name`` MUST never overlap between
    ``RUNTIME_PARAMETER`` and ``COMPILE_TIME_BOUND`` instances within
    one Block (this mirrors the project-level ``bindings`` /
    ``parameters`` disjointness rule).

    Attributes:
        name (str): The kernel's Python parameter name.
        type (ValueType): The declared IR **element** type of the
            parameter (``UIntType``, ``FloatType``, ``ObservableType``,
            ...). For array-typed parameters (``Vector[Float]``,
            ``Matrix[UInt]``, ``Tensor[Float]``) this is the element
            type and the array dimensionality is recorded separately
            in ``ndim``.
        kind (ParamKind): Whether this argument is a runtime parameter
            or was bound at compile time.
        ndim (int): Array dimensionality. ``0`` means a scalar
            parameter; ``1`` / ``2`` / ``3`` match
            ``Vector`` / ``Matrix`` / ``Tensor`` wrappers respectively.
            Defaults to ``0``.
        default (Any): The Python signature default for this argument
            (``inspect.Parameter.empty`` is normalized to ``None``).
            Carried through unchanged regardless of ``kind`` so the
            receiver can distinguish "no default provided" (``None``)
            from a default that happens to be ``0`` or ``0.0``.
        bound_value (Any): The actual value applied when ``kind`` is
            ``COMPILE_TIME_BOUND``. ``None`` for ``RUNTIME_PARAMETER``
            slots. The value's type matches ``type``; the field is
            typed as ``Any`` because the bound payload can be a
            scalar, an ``np.ndarray``, a Hamiltonian object, etc.
        differentiable (bool): Hint for outer-DSL autodiff tooling
            (e.g., parameter-shift gradient drivers). Defaults to
            ``False``; qamomile itself does not currently consume this
            flag.
    """

    name: str
    type: "ValueType"
    kind: ParamKind
    ndim: int = 0
    default: typing.Any = None
    bound_value: typing.Any = None
    differentiable: bool = False
