from __future__ import annotations

import abc
import copy
import dataclasses
import os
import sys
import typing
import uuid

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QubitConsumedError
from qamomile.circuit.transpiler.passes.eval_utils import evaluate_binop_values

if typing.TYPE_CHECKING:
    from types import CodeType, FrameType

    from .array import ArrayBase
    from .primitives import UInt


def _emit_binop(
    lhs: Value, rhs: Value, result: "Handle | Value", kind: BinOpKind
) -> None:
    """Emit a BinOp to the current tracer, or fold eagerly when both
    operands are compile-time constants.

    When both ``lhs`` and ``rhs`` carry a concrete constant
    (``Value.is_constant()``), the operation is evaluated immediately
    and the result is materialised as a constant ``Value`` rather than
    a BinOp in the trace.  This eliminates trivial residual ops like
    ``lo + 0`` (where ``lo`` is constant) that would otherwise leave
    symbolic shapes in downstream ``ArrayValue.shape`` and break
    stdlib helpers (e.g. :func:`qamomile.circuit.stdlib.qft._get_size`)
    which require concrete sizes at trace time.

    The eager-fold path requires a mutable :class:`Handle` for
    ``result`` so the constant Value can be installed in place; passing
    a raw :class:`Value` keeps the legacy emit-only behaviour for the
    rare callers that don't construct a handle.

    Args:
        lhs: Left operand Value.
        rhs: Right operand Value.
        result: Either the result Handle (whose ``value`` is mutated
            on fold) or a raw Value (emit-only legacy path).
        kind: The :class:`BinOpKind` to apply.
    """
    from qamomile.circuit.frontend.handle.primitives import Bit, Float, UInt

    if isinstance(result, Handle) and lhs.is_constant() and rhs.is_constant():
        lhs_v = lhs.get_const()
        rhs_v = rhs.get_const()
        if lhs_v is not None and rhs_v is not None:
            folded = evaluate_binop_values(kind, lhs_v, rhs_v)
            if folded is not None:
                result.value = result.value.with_const(folded)
                if isinstance(result, (UInt, Float, Bit)):
                    try:
                        result.init_value = type(result.init_value)(folded)  # type: ignore[assignment]
                    except (TypeError, ValueError):
                        result.init_value = folded  # type: ignore[assignment]
                return

    result_value = result.value if isinstance(result, Handle) else result
    binop = BinOp(
        operands=[lhs, rhs],
        results=[result_value],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(binop)


def _emit_compop(lhs: Value, rhs: Value, result: Value, kind: CompOpKind) -> None:
    """Emit a CompOp to the current tracer."""
    compop = CompOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(compop)


def _emit_condop(lhs: Value, rhs: Value, result: Value, kind: CondOpKind) -> None:
    """Emit a CondOp (logical AND/OR) to the current tracer.

    Used by ``Bit.__and__`` / ``__or__`` to lift Python's ``&`` / ``|``
    on Bit handles into the IR. Distinct from ``BinOp`` because the
    semantics is short-circuit-style boolean logic, not bitwise integer
    arithmetic.

    Args:
        lhs: Left operand Value.
        rhs: Right operand Value.
        result: Result Value (``BitType``) the op writes into.
        kind: ``CondOpKind.AND`` or ``CondOpKind.OR``.

    Returns:
        None.
    """
    condop = CondOp(
        operands=[lhs, rhs],
        results=[result],
        kind=kind,
    )
    tracer = get_current_tracer()
    tracer.add_operation(condop)


def _emit_notop(operand: Value, result: Value) -> None:
    """Emit a NotOp (logical negation) to the current tracer.

    Used by ``Bit.__invert__`` to lift Python's ``~`` on a Bit handle
    into the IR.

    Args:
        operand: Operand Value to negate.
        result: Result Value (``BitType``) the op writes into.

    Returns:
        None.
    """
    notop = NotOp(operands=[operand], results=[result])
    tracer = get_current_tracer()
    tracer.add_operation(notop)


# Root directory of the ``qamomile`` package (with a trailing separator so
# sibling packages like ``qamomile_extras`` are not misclassified), used to
# classify stack frames as library-internal vs. user code when locating the
# source line of a ``consume()`` call. Computed once at import time.
_QAMOMILE_PACKAGE_ROOT = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "",
)

# Opaque reference to a source position: ``(filename, code object, f_lasti)``.
# The line number is intentionally NOT resolved here — reading
# ``frame.f_lineno`` scans the code object's line table from the start
# (O(code size) in CPython 3.11), which turns tracing of large straight-line
# kernels quadratic. ``f_lasti`` is a plain field read; the bytecode-offset →
# line resolution is deferred to ``_format_frame_ref`` on the (cold) error
# path. Holding the code object (unlike the frame) does not pin locals.
_FrameRef = tuple[str, "CodeType", int]


def _user_code_frame_ref() -> _FrameRef | None:
    """Return an opaque source-position ref for the innermost user frame.

    Walks the current call stack outward and returns a ``_FrameRef`` for the
    first frame whose code does not live inside the ``qamomile`` package —
    for a traced ``@qm.qkernel`` body this is the user's kernel source line,
    because the AST transform re-anchors the traced code to the original
    file. Qamomile-synthesized frames (``<qamomile-...>`` pseudo-filenames
    registered by kernel synthesis) are also skipped so the caller of the
    synthesized wrapper is reported instead. Other angle-bracket
    pseudo-filenames are treated as user code: notebook cells appear as
    ``<ipython-input-N-...>`` in classic IPython, and skipping them would
    lose the location for every kernel defined in a notebook.

    Returns:
        _FrameRef | None: ``(filename, code, f_lasti)`` of the innermost
            user frame, or ``None`` when every frame is library-internal
            (e.g. a handle consumed from a qamomile-internal helper with no
            user code on the stack).
    """
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        filename = code.co_filename
        if not filename.startswith(_QAMOMILE_PACKAGE_ROOT) and not filename.startswith(
            "<qamomile"
        ):
            return (filename, code, frame.f_lasti)
        frame = frame.f_back
    return None


def _format_frame_ref(ref: _FrameRef | None) -> str | None:
    """Resolve a ``_FrameRef`` to a human-readable ``file:line`` string.

    Performs the deferred bytecode-offset → line-number lookup via
    ``code.co_lines()``. Only called on the error path, so the O(code size)
    scan is paid once per raised diagnostic, never per traced operation.

    Args:
        ref (_FrameRef | None): Position ref captured by
            ``_user_code_frame_ref``, or ``None``.

    Returns:
        str | None: ``"<abspath>:<lineno>"``, the bare filename when the
            offset has no line entry, or ``None`` for a ``None`` ref.
    """
    if ref is None:
        return None
    filename, code, lasti = ref
    for start, end, lineno in code.co_lines():
        if lineno is not None and start <= lasti < end:
            return f"{filename}:{lineno}"
    return filename


def _describe_consume_sites(
    handle: "Handle", operation_name: str
) -> tuple[str, str, str | None]:
    """Format first-use / reuse descriptions for a consumed-handle error.

    Shared by every ``QubitConsumedError`` raise site that reports reuse
    of an already-consumed handle (scalar ``Handle.consume``, the array /
    view guards in ``array.py``, and the qkernel call-boundary guard), so
    all such diagnostics carry the same ``'<op>' at <file>:<line>`` shape
    for both the first-consuming call and the offending reuse. Only called
    on the error path — the stack walk and deferred line resolution are
    never paid during normal tracing.

    Args:
        handle (Handle): The already-consumed handle being reused. Its
            ``_consumed_by`` / ``_consumed_at`` describe the first use.
        operation_name (str): Name of the operation attempting the reuse.

    Returns:
        tuple[str, str, str | None]: ``(first_use, reuse, consumed_at)``
            where ``first_use`` is ``"'<op>' at <file>:<line>"`` (location
            omitted when unknown), ``reuse`` is the same shape for the
            current call site, and ``consumed_at`` is the bare first-use
            ``file:line`` string (or ``None``) for
            ``QubitConsumedError.first_use_location``.
    """
    # ``getattr`` guards handles built via ``object.__new__`` (which
    # bypasses dataclass defaults and may predate this field).
    consumed_at = _format_frame_ref(getattr(handle, "_consumed_at", None))
    reuse_location = _format_frame_ref(_user_code_frame_ref())
    first_use = f"'{handle._consumed_by}'"
    if consumed_at:
        first_use += f" at {consumed_at}"
    reuse = f"'{operation_name}'"
    if reuse_location:
        reuse += f" at {reuse_location}"
    return first_use, reuse, consumed_at


@dataclasses.dataclass
class Handle(abc.ABC):
    value: Value
    parent: "ArrayBase | None" = None
    indices: tuple["UInt", ...] = ()
    name: str | None = None
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    _consumed: bool = False
    _consumed_by: str | None = None
    _consumed_at: "_FrameRef | None" = None
    # Set to True only on branch-tracing copies made by
    # ``_fresh_handle_copy_for_tracing`` when the source handle was already
    # consumed BEFORE the branch. Lets the phi-merge machinery distinguish
    # pre-branch consumption (slot provably untouched by the branch — no
    # elision block, no conditional-move marking) from in-branch consumption
    # (which triggers the conditional-move rule).
    _consumed_pre_branch: bool = False

    def __bool__(self) -> bool:
        """Reject implicit Python truth-value testing of symbolic handles.

        Python's ``not``, ``and``, ``or``, conditional expressions, and
        chained comparisons all invoke ``bool`` internally. Allowing that
        protocol to fall back to object truthiness would silently treat every
        symbolic handle as true and discard part of the quantum program.

        Returns:
            bool: This method never returns.

        Raises:
            TypeError: Always, because a symbolic handle has no trace-time
                Python truth value.
        """
        raise TypeError(
            f"{type(self).__name__} is a symbolic Qamomile handle and cannot "
            "be converted to a Python bool. Inside a qkernel body, use "
            "'if handle:' for control flow; outside qkernel tracing, symbolic "
            "handles cannot be used as Python conditions. For compound "
            "conditions, use '&', '|', and '~' instead of 'and', 'or', and "
            "'not'."
        )

    def _should_enforce_linear(self) -> bool:
        """Check if this handle type requires linear enforcement.

        Only quantum types (Qubit) require affine type enforcement.
        Classical values (Float, UInt, Bit) can be used multiple times.
        """
        return self.value.type.is_quantum()

    def validate_consumable(self, operation_name: str = "unknown") -> None:
        """Validate a consume without changing affine ownership state.

        Args:
            operation_name (str): Name of the prospective consuming operation,
                used in diagnostics. Defaults to ``"unknown"``.

        Raises:
            QubitConsumedError: If this quantum handle was already consumed.
        """
        if self._consumed and self._should_enforce_linear():
            display_name = self.name or f"qubit_{self.id[:8]}"
            first_use, reuse, consumed_at = _describe_consume_sites(
                self, operation_name
            )
            raise QubitConsumedError(
                f"Qubit '{display_name}' was already consumed by {first_use} "
                f"and cannot be used again in {reuse}.\n\n"
                f"Affine type rule: Each qubit handle can only be used once. "
                f"After a gate operation, reassign the result to use the new handle.\n\n"
                f"Fix:\n"
                f"  q = qm.h(q)  # Reassign to capture the new handle\n"
                f"  q = qm.x(q)  # Use the reassigned handle",
                handle_name=display_name,
                operation_name=operation_name,
                first_use_location=consumed_at or self._consumed_by,
            )

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Mark this handle as consumed and return a fresh handle.

        Records the user-code source location (``file:line``) of the
        consuming call so a later affine violation can point at both the
        first-use and the reuse site.

        Args:
            operation_name (str): Name of the operation consuming this handle,
                used for error messages. Defaults to ``"unknown"``.

        Returns:
            typing.Self: New handle pointing to the same underlying value.

        Raises:
            QubitConsumedError: If this quantum handle was already consumed.
        """
        self.validate_consumable(operation_name)
        self._consumed = True
        self._consumed_by = operation_name
        if self._should_enforce_linear():
            # Stack-walk cost is paid only for quantum handles, where the
            # location is what a later affine-violation report needs. Line
            # resolution is deferred to the error path (see ``_FrameRef``).
            self._consumed_at = _user_code_frame_ref()

        # Use type(self) to preserve the actual subclass type (Qubit, UInt, etc.)
        cls = type(self)
        new_handle = object.__new__(cls)
        new_handle.value = self.value
        new_handle.parent = self.parent
        new_handle.indices = copy.copy(self.indices)
        new_handle.name = self.name
        new_handle.id = self.id
        new_handle._consumed = False
        new_handle._consumed_by = None
        new_handle._consumed_at = None
        new_handle._consumed_pre_branch = False
        self._copy_subclass_state_to(new_handle)
        if self.parent is not None and self.indices and self._should_enforce_linear():
            self.parent._update_direct_borrow_after_consume(
                self,
                new_handle,
                operation_name,
            )
        return new_handle

    def _handoff_direct_borrow_to(self, successor: "Handle") -> None:
        """Move a direct array-element borrow to an actual operation result.

        ``consume`` creates an intermediate successor around the input IR
        value. Operations then create their real output handle around a fresh
        SSA value. This helper replaces the intermediate owner in the parent
        array's borrow table with that real output handle.

        Args:
            successor (Handle): Actual result handle that continues ownership.

        Returns:
            None.
        """
        if self.parent is not None and self.indices and self._should_enforce_linear():
            self.parent._handoff_direct_borrow_owner(self, successor)

    def _copy_subclass_state_to(self, new_handle: "Handle") -> None:
        """Hook for subclasses to copy additional state during consume()."""
        pass

    def _wrap_merge_result(self, value: Value, counterpart: Value) -> "Handle":
        """Wrap a merged IR value in this handle's frontend family.

        Called on the true-branch handle when an if-else merges branch
        values. Subclasses that support merging override this to wrap
        ``value`` in their own handle type — and, where wrapping must
        consider both branches, to validate / copy metadata using
        ``counterpart``.

        Args:
            value (Value): Fresh IR value produced for the merge output.
            counterpart (Value): The false-branch IR value, for subclasses
                whose wrapping depends on both branch values (e.g. QFixed
                carrier-metadata validation).

        Returns:
            Handle: A handle of this handle's family wrapping ``value``
                (possibly rebuilt with copied metadata).

        Raises:
            TypeError: Always, for handle families without explicit
                merge support.
        """
        raise TypeError(
            "Unsupported Handle type for if-else merge: "
            f"{type(self).__name__}. Add explicit handle wrapping support "
            "before merging this handle type."
        )


class ArithmeticMixin:
    """Mixin providing arithmetic operations for numeric Handle types.

    Requires:
        - value: Value attribute
        - _make_result(): Method to create result Handle of same type
        - _coerce(): Method to convert Python literals to Handle
    """

    value: Value  # Declare for type checking

    def _coerce(self, other) -> "Handle":
        """Convert int/float to Handle if needed (to be implemented in subclass)."""
        raise NotImplementedError("_coerce must be implemented in subclass.")

    def _make_result(self) -> "Handle":
        """Create a result Handle for an operation (to be implemented in subclass)."""
        raise NotImplementedError("_make_result must be implemented in subclass.")

    def _make_float_result(self) -> "Handle":
        """Create a Float result for division operations (to be implemented in subclass)."""
        raise NotImplementedError("_make_float_result must be implemented in subclass.")

    def __add__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.ADD)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.SUB)
        return result

    def __rsub__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(other.value, self.value, result, BinOpKind.SUB)
        return result

    def __mul__(self, other):
        other = self._coerce(other)
        result = self._make_result()
        _emit_binop(self.value, other.value, result, BinOpKind.MUL)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()  # Division always returns Float
        _emit_binop(self.value, other.value, result, BinOpKind.DIV)
        return result

    def __rtruediv__(self, other):
        other = self._coerce(other)
        result = self._make_float_result()
        _emit_binop(other.value, self.value, result, BinOpKind.DIV)
        return result
