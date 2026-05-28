"""Controlled gate operations."""

from __future__ import annotations

import dataclasses
import inspect
import keyword
import linecache
import threading
import types as _types
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Sequence,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float, Qubit, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel

# Type alias for parameter values
ParamValue = Union[float, int, Float, UInt]

# Counter for synthesized-wrapper filenames; ensures distinct
# ``linecache`` entries even when the same gate is wrapped multiple times.
# An ``RLock`` (rather than ``Lock``) is required because we eagerly
# build ``qkernel_inst.block`` *inside* the lock — that build executes
# the synthesized wrapper, which calls the user-supplied ``fn``, and
# ``fn`` may itself call ``control(...)`` (e.g. a helper that
# constructs another controlled gate during its body).  A non-reentrant
# lock would deadlock the same thread on this re-entry; ``RLock`` lets
# the recursive ``_qkernel_for_callable`` call proceed normally.
_synthesized_kernel_counter = 0
_synthesized_kernel_lock = threading.RLock()

# Cache from a built-in gate callable to its synthesized ``QKernel`` wrapper.
# A ``WeakKeyDictionary`` lets the entry vanish automatically when the
# original callable is garbage-collected, so we never grow the
# ``linecache`` unboundedly in long-running processes that repeatedly call
# ``control(qmc.rx)`` etc.
_synthesized_kernel_cache: "weakref.WeakKeyDictionary[Callable[..., Any], Any]" = (
    weakref.WeakKeyDictionary()
)

# Strong-reference fallback cache for callables that do not support weak
# references (a small set of C-implemented builtins).  Without this,
# repeated ``control(fn)`` calls on the same non-weakrefable callable
# would re-synthesize a wrapper each time and accumulate ``linecache``
# entries indefinitely.  Holding a strong reference is acceptable here
# because such callables are typically module-level / immortal — the
# entry's lifetime is bounded by the process, exactly like the
# ``linecache`` source it pairs with.  Hashable-but-not-weakrefable is
# the only case that lands here; non-hashable callables fall through to
# no caching at all.
_synthesized_kernel_cache_strong: dict[Callable[..., Any], Any] = {}


def _wrapper_namespace(target_ref: Any) -> dict[str, Any]:
    """Build the exec namespace used when compiling a synthesized wrapper.

    This is the **single source of truth** for the names injected into
    the wrapper's compile-time namespace (used in annotations, the
    return type, and the forwarding call).  ``_RESERVED_WRAPPER_NAMES``
    below is derived from the keys returned here, so adding a new
    injection (e.g. a future ``Bit`` annotation) requires only one edit
    and the name-collision guard updates automatically — there is no
    second list to keep in sync.

    Args:
        target_ref: The object bound to ``__qmc_target__`` in the
            namespace so the synthesized ``return __qmc_target__(...)``
            forwards to it.  In normal use this is either the original
            callable or a ``weakref.proxy`` of it; ``None`` is allowed
            when the namespace is built only to enumerate keys (see
            ``_RESERVED_WRAPPER_NAMES``).

    Returns:
        A fresh ``dict`` suitable for passing as the ``globals``
        argument of ``exec()``.
    """
    return {
        "Qubit": Qubit,
        "Float": Float,
        "UInt": UInt,
        "tuple": tuple,
        "__qmc_target__": target_ref,
    }


# Names that the synthesized wrapper's body references directly (via
# annotations and the forwarding call).  If the original callable's
# ``__name__`` collides with any of these, defining the wrapper with that
# same name would shadow the injected binding and break type-hint
# resolution at decoration time, so we fall back to a fresh internal
# identifier in that case.  The set is derived from
# ``_wrapper_namespace`` so that any future change to the injected names
# stays consistent with the collision guard automatically — there is no
# second list to keep in sync.
_RESERVED_WRAPPER_NAMES: frozenset[str] = frozenset(_wrapper_namespace(None).keys())


@dataclasses.dataclass
class _ControlEntry:
    """Bookkeeping for one positional control or sub-quantum handle.

    The new ``ControlledGate.__call__`` concrete path consumes handles
    in two stages: scalar ``Qubit`` and whole-``Vector`` arguments are
    consumed eagerly via :meth:`Handle.consume`, while ``VectorView``
    arguments defer the consume until ``_wrap_results_by_input_kind``
    can build the fresh result view and rebind the parent's bulk-borrow
    through :meth:`VectorView._transfer_borrow_to` (same pattern as
    ``QKernel.__call__``).

    Attributes:
        original (Any): The handle as it was passed in by the caller
            (``Qubit`` / ``Vector`` / ``VectorView``).  Preserved so
            ``_wrap_results_by_input_kind`` can rebuild an output handle
            of the same kind and, for ``VectorView``, perform the
            deferred borrow transfer.
        consumed (Any | None): The post-consume handle for ``Qubit`` and
            ``Vector`` (whose consume is eager).  ``None`` for
            ``VectorView``, signalling that the consume is deferred and
            ``original`` should still be used to read the current
            ``ArrayValue``.
    """

    original: Any
    consumed: Any | None = None

    @property
    def is_deferred_view(self) -> bool:
        """Whether this entry represents a ``VectorView`` whose consume is deferred."""
        return self.consumed is None


class ControlledGate:
    """Wrapper for controlled version of a QKernel.

    Created by calling `control(qkernel)`. The resulting object
    can be called like a gate function.

    Example:
        @qmc.qkernel
        def phase_gate(q: Qubit, theta: float) -> Qubit:
            return qmc.p(q, theta)

        controlled_phase = qmc.control(phase_gate)
        ctrl_out, tgt_out = controlled_phase(ctrl, target, theta=0.5)

        # Double-controlled
        cc_phase = qmc.control(phase_gate, num_controls=2)
        c0, c1, tgt = cc_phase(ctrl0, ctrl1, target, theta=0.5)
    """

    def __init__(self, qkernel: "QKernel", num_controls: int | UInt = 1) -> None:
        if isinstance(num_controls, int) and num_controls < 1:
            raise ValueError(f"num_controls must be >= 1, got {num_controls}.")
        # For UInt (symbolic), validation is deferred to emit time

        # Compose-time validation of the wrapped object's shape.
        # Downstream helpers (``_sub_positional_count_for_symbolic``,
        # ``_bind_to_sub_signature``, ``_params_to_operands``) all
        # require ``input_types`` to be a real ``dict`` and ``signature``
        # to be an ``inspect.Signature`` so the operand split and
        # signature-driven kwarg binding produce a sane result.  Before
        # this check, each of those helpers silently fell back to a
        # "legacy single-pool" / caller-order interpretation whenever
        # the attribute was missing or wrong-typed.  In production that
        # is a silent miscompile (multi-arg control prefix collapses to
        # one pool, kwargs land at the wrong operand index, etc.); fail
        # loudly here so the error points back at the ``qmc.control``
        # call site.  The builtin-callable path goes through
        # :func:`_qkernel_for_callable` which synthesizes a proper
        # ``QKernel`` first, so anything legitimately wrapped by
        # ``qmc.control`` arrives here with both attributes populated.
        input_types = getattr(qkernel, "input_types", None)
        if not isinstance(input_types, dict):
            raise TypeError(
                f"qmc.control(): wrapped object {qkernel!r} does not expose "
                f"a dict ``input_types`` attribute (got "
                f"{type(input_types).__name__}).  Pass a "
                f"``@qmc.qkernel``-decorated function or a built-in gate "
                f"callable; arbitrary objects are not supported."
            )
        signature = getattr(qkernel, "signature", None)
        if not isinstance(signature, inspect.Signature):
            raise TypeError(
                f"qmc.control(): wrapped object {qkernel!r} does not expose "
                f"an ``inspect.Signature`` ``signature`` attribute (got "
                f"{type(signature).__name__}).  Pass a "
                f"``@qmc.qkernel``-decorated function or a built-in gate "
                f"callable; arbitrary objects are not supported."
            )

        self._qkernel = qkernel
        self._num_controls = num_controls

    @staticmethod
    def _normalize_power(power: int | UInt) -> int | Value:
        """Normalize power to an IR-compatible type (``int`` or ``Value``).

        ``UInt`` handles are unwrapped to their underlying ``Value`` so
        that the IR never stores frontend types.  Concrete ``int`` values
        are validated for strict positivity.

        Args:
            power: The power value from the user API.

        Returns:
            ``int`` for concrete values, ``Value`` for symbolic expressions.

        Raises:
            TypeError: If *power* is ``bool``, ``float``, or another
                unsupported type.
            ValueError: If a concrete *power* is ``<= 0``.
        """
        if isinstance(power, bool):
            raise TypeError(
                f"power must be a positive integer, got bool ({power}). "
                f"Use an integer value like power=1 or power=2."
            )
        if isinstance(power, UInt):
            return power.value
        if isinstance(power, int):
            if power <= 0:
                raise ValueError(
                    f"power must be a strictly positive integer, got {power}."
                )
            return power
        raise TypeError(f"power must be int or UInt, got {type(power).__name__}.")

    def _params_to_operands(
        self,
        params: dict[str, ParamValue],
        operands: list[Any],
    ) -> None:
        """Append parameter values to the operands list.

        Handle types are unwrapped to their underlying ``Value``.  Raw
        scalar values are coerced to the IR type the wrapped kernel
        actually declares for that parameter — ``UIntType`` when the
        kernel's annotation is ``UInt`` / ``int``, ``FloatType``
        otherwise.  This prevents an ``int`` argument to a
        ``UInt``-annotated parameter from sneaking in as a
        ``FloatType`` constant and violating IR invariants downstream
        (e.g. ``ForOperation`` requires ``UIntType`` operands at
        expand-controlled-U time).

        Two extra invariants are enforced (the validate in
        :meth:`ControlledGate.__init__` guarantees the wrapped object
        exposes a dict ``input_types``, so these always apply):

        * **Operand order follows the wrapped kernel's signature**, not
          the caller's kwarg insertion order.  ``ValueResolver``
          binds controlled-U parameter operands to the inner block's
          classical inputs *by position*, so caller-order would silently
          rebind the wrong values when the user passes kwargs in a
          different order than the kernel's declaration.
        * **Unknown / typo'd parameter names raise ``TypeError``** —
          previously these were silently dropped by the same positional
          binding, producing a controlled gate with the default
          (unbound) parameter values and no warning.
        """
        kernel_input_types: dict[str, Any] = self._qkernel.input_types
        # Classical-parameter names in the wrapped kernel's signature
        # order.  Python ``dict`` preserves insertion order (since 3.7),
        # so iterating ``input_types`` matches the declared signature.
        classical_names = [
            name
            for name, decl in kernel_input_types.items()
            if decl is UInt or decl is int or decl is Float or decl is float
        ]
        classical_set = set(classical_names)

        # Reject unknown parameter names up-front.  The downstream
        # ``ValueResolver.bind_block_params`` would otherwise drop them
        # silently, masking caller bugs (typo'd kwargs, kwargs intended
        # for a different gate, etc.).
        extras = sorted(set(params) - classical_set)
        if extras:
            raise TypeError(
                f"control(): unknown parameter(s) {extras!r}. "
                f"The wrapped kernel's classical parameters are "
                f"{classical_names!r}."
            )

        # Append operands in the kernel's signature order, regardless of
        # how the caller spelled the kwargs.  Missing names are skipped
        # silently — this is the existing behaviour for kwargs that the
        # caller intends to supply at runtime through
        # ``.sample(bindings=...)``.
        for param_name in classical_names:
            if param_name not in params:
                continue
            param_value = params[param_name]
            if isinstance(param_value, Handle):
                operands.append(param_value.value)
                continue
            declared = kernel_input_types[param_name]
            if declared is UInt or declared is int:
                # ``bool`` is technically an ``int`` subclass but its
                # meaning differs; reject explicitly to match the
                # qkernel decorator's literal-promotion rules.  ``float``
                # would be silently truncated by ``int(...)``; reject too
                # so callers must opt in via an explicit ``int(value)``
                # if truncation is genuinely intended.
                if isinstance(param_value, bool) or not isinstance(param_value, int):
                    raise TypeError(
                        f"control(): parameter {param_name!r} is "
                        f"declared as UInt/int but received "
                        f"{type(param_value).__name__} ({param_value!r}). "
                        f"Pass a Python int (or a UInt handle) instead."
                    )
                param_val = Value(
                    type=UIntType(),
                    name=f"ctrl_param_{param_name}",
                ).with_const(int(param_value))
            else:
                # Float-declared param.  Accept Python int / float
                # (auto-promote ``int`` as the qkernel decorator does)
                # but reject ``bool`` so ``True`` doesn't surprise as
                # ``1.0``.
                if isinstance(param_value, bool):
                    raise TypeError(
                        f"control(): parameter {param_name!r} is "
                        f"declared as Float/float but received bool "
                        f"({param_value!r}). Pass a numeric value instead."
                    )
                param_val = Value(
                    type=FloatType(),
                    name=f"ctrl_param_{param_name}",
                ).with_const(float(param_value))
            operands.append(param_val)

    @staticmethod
    def _wrap_qubit_outputs(
        consumed_handles: list[Any],
        results: list[Value],
        offset: int = 0,
    ) -> list[Qubit]:
        """Create output Qubit handles from consumed handles and result Values.

        Each output Qubit preserves the parent/indices metadata from the
        corresponding consumed handle for array write-back support.
        """
        output = []
        for i, handle in enumerate(consumed_handles):
            output.append(
                Qubit(
                    value=results[offset + i],
                    parent=handle.parent,
                    indices=handle.indices,
                )
            )
        return output

    def _build_and_emit_op(
        self,
        operands: list[Any],
        results: list[Value],
        num_controls: int | Value,
        power: int | Value,
    ) -> ControlledUOperation:
        """Create the appropriate ControlledUOperation subclass and add to tracer.

        Used by :meth:`_call_concrete`; the symbolic path constructs
        its ``SymbolicControlledU`` inline because it needs to pass
        the ``control_indices`` field directly.
        """
        block = self._qkernel.block
        op: ControlledUOperation
        if isinstance(num_controls, Value):
            op = SymbolicControlledU(
                operands=operands,
                results=results,
                num_controls=num_controls,
                power=power,
                block=block,
            )
        else:
            op = ConcreteControlledU(
                operands=operands,
                results=results,
                num_controls=num_controls,
                power=power,
                block=block,
            )
        tracer = get_current_tracer()
        tracer.add_operation(op)
        return op

    # ------------------------------------------------------------------
    # Helpers for ``__call__``'s concrete and symbolic paths.
    #
    # Each helper has a narrow contract so ``_call_concrete`` and
    # ``_call_symbolic`` can read top-to-bottom as a small choreography
    # (split → partition → validate → consume → emit → wrap).
    # ------------------------------------------------------------------

    def _split_controls_by_count(
        self,
        args: tuple[Any, ...],
        num_controls: int,
    ) -> tuple[list[Any], list[Any]]:
        """Split *args* into the leading ``num_controls`` qubits and the rest.

        Counts qubits by **element count**: scalar ``Qubit`` contributes
        one, ``Vector``/``VectorView`` contributes its length.  Walks
        *args* left to right and stops as soon as the cumulative count
        reaches ``num_controls``.  The boundary between control and
        sub-kernel arguments **must fall on an argument boundary** —
        splitting an argument in the middle is rejected with
        :class:`ValueError` (see the table in §2.1 of the design doc).

        Args:
            args (tuple[Any, ...]): Positional arguments handed to
                :meth:`ControlledGate.__call__` in concrete-mode.
            num_controls (int): The concrete control qubit count
                ``N`` configured on this gate.

        Returns:
            tuple[list[Any], list[Any]]: A pair ``(controls,
                sub_call_args)``.  ``controls`` is the prefix of *args*
                whose total qubit count is exactly ``num_controls``;
                ``sub_call_args`` is everything that follows.

        Raises:
            ValueError: If *args* runs out of qubits before reaching
                ``num_controls``, or if the boundary falls inside a
                single ``Vector``/``VectorView`` argument (i.e. the
                running count would jump past ``num_controls``), or if
                a symbolic-length ``VectorView`` is mixed with other
                positional args in the control region (decision #16).
        """
        from qamomile.circuit.frontend.handle.array import (
            ArrayBase,
            _as_int_const,
        )

        controls: list[Any] = []
        running = 0
        for idx, arg in enumerate(args):
            if running == num_controls:
                return controls, list(args[idx:])
            if isinstance(arg, ArrayBase):
                length = arg._shape[0] if arg._shape else None
                length_int = _as_int_const(length) if length is not None else None
                if length_int is None:
                    # Symbolic-length view/vector in the control region:
                    # only acceptable when it stands alone as the
                    # entire control prefix (decision #16).
                    if controls or running != 0:
                        raise ValueError(
                            "concrete num_controls: a symbolic-length "
                            "Vector / VectorView can only appear as the "
                            "first positional argument when it represents "
                            "the entire control prefix; mixing it with "
                            "other positional control args is ambiguous "
                            "(see design decision #16)."
                        )
                    # Defer the count vs. num_controls check to emit time.
                    controls.append(arg)
                    return controls, list(args[idx + 1 :])
                next_running = running + length_int
            elif isinstance(arg, Qubit):
                next_running = running + 1
            else:
                raise ValueError(
                    f"concrete num_controls: positional argument #{idx} "
                    f"in the control region must be a Qubit, Vector[Qubit], "
                    f"or VectorView[Qubit]; got {type(arg).__name__}."
                )
            if next_running > num_controls:
                raise ValueError(
                    f"concrete num_controls={num_controls}: positional "
                    f"argument #{idx} would push the control qubit count "
                    f"from {running} to {next_running}, crossing the "
                    f"control / sub-kernel boundary mid-argument.  Split "
                    f"the argument so the boundary falls between args."
                )
            controls.append(arg)
            running = next_running

        if running < num_controls:
            raise ValueError(
                f"ControlledU requires at least {num_controls + 1} qubits "
                f"({num_controls} controls + at least 1 sub-kernel target); "
                f"got only {running} control qubit(s) and no sub-kernel "
                f"arguments after them."
            )
        return controls, []

    @staticmethod
    def _collect_sub_quantum_args(sub_args_resolved: dict[str, Any]) -> list[Any]:
        """Filter to only quantum handles (``Qubit`` / ``Vector`` / ``VectorView``).

        Args:
            sub_args_resolved (dict[str, Any]): The sub-kernel argument
                dict returned by :meth:`_bind_to_sub_signature`,
                already in signature order.

        Returns:
            list[Any]: The quantum handles from *sub_args_resolved* in
                the same iteration order.
        """
        from qamomile.circuit.frontend.handle.array import ArrayBase

        return [
            h for h in sub_args_resolved.values() if isinstance(h, (Qubit, ArrayBase))
        ]

    # ``_validate_no_alias_or_overlap`` used to live here as an entry-
    # point alias / overlap check, mirroring the
    # ``_check_qubit_alias`` helper in ``qubit_gates.py``.  In practice
    # every adversarial call shape (``cg(q, q)``, ``cg(qs[0:3], qs[2])``,
    # ``cg(qs[0:3], qs[0:3])``, ``cg(qs[0:3], qs[1:4])``) is already
    # rejected by the linear-type / borrow-tracking layer one step
    # earlier — by ``Handle.consume()`` (scalar duplicates →
    # ``QubitConsumedError``) or by ``ArrayBase._get_element`` /
    # ``Vector._make_slice_view`` 's borrow table (view-touching
    # overlaps → ``QubitBorrowConflictError``).  The bespoke check was
    # therefore pure duplication and was removed; the underlying
    # safety guarantees are unchanged, only the error class on the
    # ``cg(q, q)`` shape moved from ``QubitAliasError`` to
    # ``QubitConsumedError``.

    @staticmethod
    def _consume_with_borrow_transfer(
        handles: list[Any],
        operation_name: str,
    ) -> list[_ControlEntry]:
        """Consume *handles*, deferring the consume for ``VectorView`` inputs.

        Scalar ``Qubit`` and whole ``Vector`` handles take the
        straightforward :meth:`Handle.consume` path so the affine /
        consumed-slot bookkeeping fires immediately.  ``VectorView``
        handles have their consume **deferred** to
        :meth:`_wrap_results_by_input_kind`, mirroring
        ``QKernel.__call__``'s VectorView handling: the deferred
        consume lets the caller build a fresh result view wrapping the
        next-versioned slice ``ArrayValue`` and then re-route the
        parent's bulk-borrow record directly to that result view via
        :meth:`VectorView._transfer_borrow_to`.  Without the deferral
        ``VectorView.consume`` would hand the borrow back to a private
        new-view wrapper around the *current* ``ArrayValue``, leaving
        the operation result handle without ownership of the parent
        slots.

        Args:
            handles (list[Any]): Quantum handles to consume in order.
            operation_name (str): Operation name passed through to
                :meth:`Handle.consume` for the eager-consume branch
                (currently ``"ControlledU[control]"`` or
                ``"ControlledU[target]"``).

        Returns:
            list[_ControlEntry]: One entry per input handle, in order.
                Scalar ``Qubit`` / ``Vector`` entries have both
                ``original`` and ``consumed`` populated; ``VectorView``
                entries have ``consumed`` set to ``None`` (deferred).

        Raises:
            QubitConsumedError: Surfaced from :meth:`Handle.consume`
                when a non-view handle has already been consumed.
        """
        from qamomile.circuit.frontend.handle.array import VectorView

        entries: list[_ControlEntry] = []
        for handle in handles:
            if isinstance(handle, VectorView):
                entries.append(_ControlEntry(original=handle, consumed=None))
                continue
            consumed = (
                handle.consume(operation_name=operation_name)
                if handle._should_enforce_linear()
                else handle
            )
            entries.append(_ControlEntry(original=handle, consumed=consumed))
        return entries

    def _bind_to_sub_signature(
        self,
        sub_positional_args: Sequence[Any],
        sub_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind sub-kernel arguments to the wrapped kernel's signature.

        The wrapped object always carries a real ``inspect.Signature``
        (qkernel-decorated functions populate it directly; built-in
        gate callables go through :func:`_qkernel_for_callable`'s
        synthesized wrapper; the compose-time validate in
        :meth:`ControlledGate.__init__` rejects anything else).  The
        binding is therefore always delegated to
        :meth:`inspect.Signature.bind` so positional / keyword mixing
        and Python defaults work out of the box.  ``apply_defaults`` is
        then called so kernels with a default-valued classical
        parameter (``def sub(qa, theta=0.5)``) end up with that
        default baked into the resulting dict.

        Args:
            sub_positional_args (Sequence[Any]): Positional arguments
                that follow the control prefix.
            sub_kwargs (dict[str, Any]): Keyword arguments passed to
                ``cg(...)`` after stripping the reserved ``power`` and
                ``control_indices`` kwargs.

        Returns:
            dict[str, Any]: An ordered dict mapping parameter name to
                its bound argument, in the wrapped kernel's signature
                order.

        Raises:
            TypeError: If :meth:`inspect.Signature.bind` reports a
                conflict (unexpected kwarg, missing required arg).
                Unexpected-kwarg errors are re-raised with the
                "unknown parameter" wording the previous
                ``_params_to_operands`` enforced so existing callers'
                error-message expectations remain stable.
        """
        # ``ControlledGate.__init__`` validates that ``input_types`` is a
        # dict and ``signature`` is an ``inspect.Signature``, so both
        # accesses are safe to use directly here.
        input_types = self._qkernel.input_types
        signature = self._qkernel.signature

        # Check for unknown kwargs up-front so the legacy "unknown
        # parameter(s)" message wins over ``inspect.Signature.bind``'s
        # "missing a required argument" / "unexpected keyword argument"
        # variants.  Without this, a typo'd kwarg paired with a missing
        # real one (the most common case in practice) surfaces as the
        # less-informative "missing required argument" error.
        extras = sorted(set(sub_kwargs) - set(input_types))
        if extras:
            classical_names = [
                n
                for n, decl in input_types.items()
                if decl is UInt or decl is int or decl is Float or decl is float
            ]
            raise TypeError(
                f"control(): unknown parameter(s) {extras!r}. "
                f"The wrapped kernel's classical parameters are "
                f"{classical_names!r}."
            )

        bound_args = signature.bind(*sub_positional_args, **sub_kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def _build_operands(
        self,
        consumed_controls: list[_ControlEntry],
        consumed_sub_quantum: list[_ControlEntry],
        sub_classical_dict: dict[str, Any],
    ) -> list[Any]:
        """Lay out the IR operand list for a ``ConcreteControlledU``.

        Layout (matches the existing ``ControlledUOperation`` operand
        contract):

        - ``operands[:num_controls]`` — per-element scalar control
          ``Value`` s.  Scalar ``Qubit`` inputs contribute their
          ``.value`` directly; ``VectorView`` and whole-``Vector``
          inputs are expanded into one scalar ``Value`` per covered
          qubit so the IR-level ``control_operands`` /
          ``target_operands`` properties keep working with no change.
        - ``operands[num_controls:num_controls + sum(M)]`` —
          sub-kernel quantum operands.  Kept as ``ArrayValue`` or
          scalar ``Value`` unchanged; the per-element expansion for
          sub-kernel ``Vector[Qubit]`` arguments is performed at emit
          time (see design §12.1).
        - ``operands[…:]`` — classical parameter operands, appended by
          the existing :meth:`_params_to_operands` so type-coercion
          and unknown-kwarg rejection stay in one place.

        Args:
            consumed_controls (list[_ControlEntry]): Per-control entries
                from :meth:`_consume_with_borrow_transfer`.
            consumed_sub_quantum (list[_ControlEntry]): Per-sub-quantum
                entries from :meth:`_consume_with_borrow_transfer`.
            sub_classical_dict (dict[str, Any]): The classical-only
                slice of the bound sub-kernel arguments (param name →
                value, in signature order for real qkernels).

        Returns:
            list[Any]: The flat operand list ready to hand to
                ``ConcreteControlledU(operands=...)``.
        """
        operands: list[Any] = []
        for entry in consumed_controls:
            operands.extend(self._expand_control_to_scalars(entry))
        for entry in consumed_sub_quantum:
            operands.append(self._sub_quantum_operand_value(entry))
        self._params_to_operands(sub_classical_dict, operands)
        return operands

    def _build_results(
        self,
        consumed_controls: list[_ControlEntry],
        consumed_sub_quantum: list[_ControlEntry],
    ) -> list[Value]:
        """Build the IR result list paired one-to-one with operands.

        The control region produces one ``next_version`` scalar
        ``Value`` per covered qubit, matching the per-element control
        operands.  The sub-quantum region produces one
        ``next_version`` ``Value`` per entry, preserving the operand
        kind (``ArrayValue`` → ``ArrayValue``, scalar → scalar).

        Args:
            consumed_controls (list[_ControlEntry]): The same entries
                that were handed to :meth:`_build_operands` for the
                control region.
            consumed_sub_quantum (list[_ControlEntry]): The same entries
                that were handed to :meth:`_build_operands` for the
                sub-quantum region.

        Returns:
            list[Value]: Result ``Value`` s in operand order
                (controls first, then sub-quantum).  Classical params
                are not represented because controlled-U has no
                classical outputs.
        """
        results: list[Value] = []
        for entry in consumed_controls:
            operands_for_entry = self._expand_control_to_scalars(entry)
            results.extend(op.next_version() for op in operands_for_entry)
        for entry in consumed_sub_quantum:
            results.append(self._sub_quantum_operand_value(entry).next_version())
        return results

    @staticmethod
    def _expand_control_to_scalars(entry: _ControlEntry) -> list[Value]:
        """Expand a control entry into one scalar ``Value`` per qubit.

        Scalar ``Qubit`` entries pass through unchanged.  Whole
        ``Vector`` and ``VectorView`` entries are expanded into
        ``length`` scalar ``Value`` s of ``QubitType``, each pointing
        at the source ``ArrayValue`` via ``parent_array`` and
        recording its compile-time index via ``element_indices``.  No
        ``_borrowed_indices`` slot is created — these synthetic
        scalars are pure IR plumbing for the controlled-U emit pass,
        not user-visible borrows.

        Args:
            entry (_ControlEntry): One bookkeeping entry produced by
                :meth:`_consume_with_borrow_transfer`.

        Returns:
            list[Value]: A list whose length equals the qubit count
                contributed by *entry*.

        Raises:
            NotImplementedError: For symbolic-length ``Vector`` /
                ``VectorView`` controls — the per-element expansion
                needs a concrete length at compose time.  The fix is
                tracked under Step 2.b (emit-side handling of
                symbolic-length controls).
        """
        from qamomile.circuit.frontend.handle.array import (
            ArrayBase,
            _as_int_const,
        )

        source = entry.original if entry.is_deferred_view else entry.consumed
        if isinstance(source, Qubit):
            return [source.value]
        assert isinstance(source, ArrayBase)
        length = source._shape[0] if source._shape else None
        length_int = _as_int_const(length) if length is not None else None
        if length_int is None:
            raise NotImplementedError(
                "concrete num_controls with a symbolic-length Vector / "
                "VectorView control is not yet implemented in the frontend "
                "(tracked under Step 2.b of the controlled-API redesign)."
            )
        array_value = source.value
        scalars: list[Value] = []
        for i in range(length_int):
            idx_value = Value(
                type=UIntType(),
                name=f"ctrl_idx_{i}",
            ).with_const(i)
            element = Value(
                type=array_value.type,
                name=(f"{array_value.name}[{i}]" if array_value.name else f"ctrl[{i}]"),
                parent_array=array_value,
                element_indices=(idx_value,),
            )
            scalars.append(element)
        return scalars

    @staticmethod
    def _sub_quantum_operand_value(entry: _ControlEntry) -> Value | ArrayValue:
        """Pick the IR ``Value`` representing one sub-kernel quantum operand.

        Scalar ``Qubit`` inputs hand back their consumed ``.value``;
        ``Vector`` / ``VectorView`` inputs hand back their *current*
        ``ArrayValue`` (the slice ``ArrayValue`` for views).  The
        per-element expansion of sub-kernel array operands is the
        emit-time helper added by Step 2.b — at the IR level we keep
        the whole-array shape to preserve aliasing information.

        Args:
            entry (_ControlEntry): A sub-quantum bookkeeping entry.

        Returns:
            Value | ArrayValue: The ``ArrayValue`` (for ``Vector`` /
                ``VectorView``) or scalar ``Value`` (for ``Qubit``) to
                place in the operand list.
        """
        source: Any = entry.original if entry.is_deferred_view else entry.consumed
        assert source is not None
        return source.value

    def _wrap_results_by_input_kind(
        self,
        consumed_controls: list[_ControlEntry],
        consumed_sub_quantum: list[_ControlEntry],
        results: list[Value],
        operation_name: str = "ControlledU",
    ) -> tuple[Any, ...]:
        """Aggregate per-element results back into input-kind handles.

        Each input handle gets one output handle of the same kind:

        - Scalar ``Qubit`` → scalar ``Qubit`` wrapping its corresponding
          per-element result, with ``parent`` / ``indices`` carried
          over to preserve array write-back support.
        - Whole ``Vector`` → ``Vector`` wrapping the ``next_version``
          of the source ``ArrayValue``.
        - ``VectorView`` → fresh ``VectorView`` wrapping the
          ``next_version`` of the source slice ``ArrayValue``;
          ownership of the parent's bulk borrow is rebound through
          :meth:`VectorView._transfer_borrow_to` so the deferred
          consume completes here.

        Args:
            consumed_controls (list[_ControlEntry]): Bookkeeping entries
                for the control region, in operand order.
            consumed_sub_quantum (list[_ControlEntry]): Bookkeeping
                entries for the sub-quantum region, in operand order.
            results (list[Value]): The full IR result list — controls
                first (one scalar ``Value`` per covered qubit), then
                one ``Value`` per sub-quantum entry.
            operation_name (str): Name used as the
                ``_transfer_borrow_to`` operation tag for deferred
                ``VectorView`` consumes.  Defaults to ``"ControlledU"``.

        Returns:
            tuple[Any, ...]: One output handle per input handle, in the
                same order as the caller supplied them (controls
                followed by sub-kernel quantum args).
        """
        wrapped: list[Any] = []
        cursor = 0
        for entry in consumed_controls:
            count = self._entry_qubit_count(entry)
            entry_results = results[cursor : cursor + count]
            cursor += count
            wrapped.append(
                self._wrap_entry_output(
                    entry,
                    entry_results,
                    operation_name=f"{operation_name}[control]",
                )
            )
        for entry in consumed_sub_quantum:
            wrapped.append(
                self._wrap_entry_output(
                    entry,
                    [results[cursor]],
                    operation_name=f"{operation_name}[target]",
                )
            )
            cursor += 1
        assert cursor == len(results), (
            f"unexpected leftover results after wrapping ({len(results) - cursor})"
        )
        return tuple(wrapped)

    @staticmethod
    def _entry_qubit_count(entry: _ControlEntry) -> int:
        """How many per-qubit results a control entry consumes.

        Args:
            entry (_ControlEntry): The bookkeeping entry whose result
                count is being requested.

        Returns:
            int: ``1`` for a scalar ``Qubit`` entry; the array length
                for a concrete-length ``Vector`` / ``VectorView``
                entry.

        Raises:
            NotImplementedError: For a symbolic-length ``Vector`` /
                ``VectorView`` entry — the per-element count needs a
                concrete length at compose time, which is the same
                restriction the per-element expansion in
                :meth:`_expand_control_to_scalars` enforces.
        """
        from qamomile.circuit.frontend.handle.array import (
            ArrayBase,
            _as_int_const,
        )

        source = entry.original if entry.is_deferred_view else entry.consumed
        if isinstance(source, Qubit):
            return 1
        assert isinstance(source, ArrayBase)
        length = source._shape[0] if source._shape else None
        length_int = _as_int_const(length) if length is not None else None
        if length_int is None:
            raise NotImplementedError(
                "concrete num_controls with a symbolic-length Vector / "
                "VectorView control is not yet implemented in the frontend."
            )
        return length_int

    @staticmethod
    def _wrap_entry_output(
        entry: _ControlEntry,
        entry_results: list[Value],
        operation_name: str,
    ) -> Any:
        """Build a single output handle for one control or sub-quantum entry.

        Two ``ArrayBase`` shapes are handled:

        - Per-element scalar results (``ConcreteControlledU`` controls
          where a ``Vector`` / ``VectorView`` was expanded to scalars
          by :meth:`_expand_control_to_scalars`).  No single
          ``ArrayValue`` exists in ``op.results`` for the entry, so a
          fresh ``next_version`` of the source array is synthesised
          and used as the wrapper handle's value.
        - Single ``ArrayValue`` result (``SymbolicControlledU`` control
          pool, or ``Vector`` / ``VectorView`` sub-kernel argument).
          The IR-side ``next_version`` is already laid out for us; the
          wrapper handle re-uses it directly so downstream lookups
          that go via the result UUID resolve through the same value
          the rest of the IR sees.

        Args:
            entry (_ControlEntry): The bookkeeping entry whose output
                handle is being constructed.
            entry_results (list[Value]): The IR result ``Value`` s
                belonging to *entry* — one scalar per qubit for the
                per-element-expanded case, exactly one ``ArrayValue``
                for the whole-array case.
            operation_name (str): Operation tag forwarded to
                :meth:`VectorView._transfer_borrow_to` when the entry's
                consume was deferred.

        Returns:
            Any: A ``Qubit`` / ``Vector`` / ``VectorView`` whose
                runtime kind matches ``entry.original``.
        """
        from qamomile.circuit.frontend.handle.array import (
            ArrayBase,
            Vector,
            VectorView,
        )

        original = entry.original
        if isinstance(original, Qubit):
            (result_value,) = entry_results
            return Qubit(
                value=result_value,
                parent=original.parent,
                indices=original.indices,
            )

        assert isinstance(original, ArrayBase)
        # Discriminate the two shapes by inspecting the caller-supplied
        # result list: a single ``ArrayValue`` means the IR already
        # carries the next-version array we should hand back; anything
        # else is a per-element scalar list and we synthesise a fresh
        # ``next_version`` from the source array.
        if len(entry_results) == 1 and isinstance(entry_results[0], ArrayValue):
            new_av = entry_results[0]
        else:
            source_for_av: Any = original if entry.is_deferred_view else entry.consumed
            assert source_for_av is not None
            source_av_for_synth = cast(ArrayValue, source_for_av.value)
            new_av = source_av_for_synth.next_version()

        if isinstance(original, VectorView):
            view = cast(VectorView, original)
            new_view = VectorView._wrap_unregistered(
                parent=view._slice_parent,
                sliced_av=new_av,
                length=view._shape[0],
                start_uint=view._slice_start,
                step_uint=view._slice_step,
            )
            view._transfer_borrow_to(new_view, operation_name)
            return new_view

        # Whole Vector — recover ``_shape`` / display name from whichever
        # side of the consume is non-``None``.
        shape_source: Any = entry.consumed if entry.consumed is not None else original
        consumed_vector = cast(Vector, shape_source)
        return Vector._create_from_value(
            new_av,
            consumed_vector._shape,
            consumed_vector.value.name,
        )

    def _call_concrete(
        self,
        args: tuple[Any, ...],
        sub_kwargs: dict[str, Any],
        power: int | Value,
    ) -> tuple[Any, ...]:
        """Concrete-``num_controls`` path for :meth:`ControlledGate.__call__`.

        Chains the helpers (split → bind → validate → consume →
        operands/results → emit → wrap) so the body of ``__call__``
        stays a thin dispatcher.  The symbolic counterpart lives in
        :meth:`_call_symbolic`.

        Args:
            args (tuple[Any, ...]): Positional arguments to ``cg(...)``
                including the leading control qubits.
            sub_kwargs (dict[str, Any]): Caller kwargs after stripping
                the reserved ``power`` and ``control_indices`` keys.
            power (int | Value): Normalised power (output of
                :meth:`_normalize_power`).

        Returns:
            tuple[Any, ...]: One output handle per input handle, in the
                concatenation order ``(controls, sub_kernel_quantum)``.

        Raises:
            ValueError: From :meth:`_split_controls_by_count` when the
                control boundary can't be honoured by the args.
            QubitConsumedError / QubitBorrowConflictError: From the
                ``Handle.consume()`` / array borrow-tracker layer when
                an argument duplicates a slot that another argument
                also touches.
            TypeError: From :meth:`_bind_to_sub_signature` or
                :meth:`_params_to_operands` on unknown/typoed kwargs
                or unsupported classical parameter types.
        """
        num_controls = cast(int, self._num_controls)

        controls, sub_positional = self._split_controls_by_count(args, num_controls)
        sub_args_resolved = self._bind_to_sub_signature(sub_positional, sub_kwargs)
        sub_quantum_args = self._collect_sub_quantum_args(sub_args_resolved)
        if not sub_quantum_args:
            raise ValueError(
                f"ControlledU requires at least one quantum sub-kernel "
                f"argument (target).  Got {num_controls} control(s) and "
                f"no sub-kernel quantum arg (see design decision #9)."
            )
        # Anything left over is classical.  ``id``-based filtering is
        # used because two distinct Handle instances may compare equal
        # but should still be treated as separate operands here.
        quantum_ids = {id(h) for h in sub_quantum_args}
        sub_classical_dict = {
            name: value
            for name, value in sub_args_resolved.items()
            if id(value) not in quantum_ids
        }

        # Alias / overlap checking is delegated entirely to the
        # ``Handle.consume()`` / array borrow-tracker layer below:
        # scalar duplicates raise ``QubitConsumedError`` on the
        # second consume, and view-touching overlaps raise
        # ``QubitBorrowConflictError`` at element / slice access time.
        consumed_controls = self._consume_with_borrow_transfer(
            controls, "ControlledU[control]"
        )
        consumed_sub_quantum = self._consume_with_borrow_transfer(
            sub_quantum_args, "ControlledU[target]"
        )

        operands = self._build_operands(
            consumed_controls, consumed_sub_quantum, sub_classical_dict
        )
        results = self._build_results(consumed_controls, consumed_sub_quantum)

        self._build_and_emit_op(operands, results, num_controls, power)

        return self._wrap_results_by_input_kind(
            consumed_controls, consumed_sub_quantum, results
        )

    def __call__(
        self,
        *args: Any,
        power: int | UInt = 1,
        control_indices: Sequence[int | UInt] | None = None,
        **params: ParamValue,
    ) -> tuple[Any, ...]:
        """Apply the controlled gate.

        The call protocol depends on whether ``num_controls`` is
        concrete (``int``) or symbolic (``UInt``).

        Concrete mode:

        - Positional args are read left-to-right; the leading
          ``num_controls`` qubits act as controls and everything that
          follows is bound to the sub-kernel via Python's standard
          signature semantics (``inspect.Signature.bind +
          apply_defaults``).
        - Each control "slot" may be a scalar ``Qubit``, a whole
          ``Vector[Qubit]`` (consumed entirely), or a ``VectorView``
          slice — the qubit count of each control argument adds up to
          ``num_controls`` and the split must fall on an argument
          boundary.
        - ``control_indices`` is **not accepted** in this mode and
          raises :class:`ValueError` immediately.

        Symbolic mode (``num_controls=UInt``):

        - ``args[0]`` is the *control pool* — a ``Vector[Qubit]`` /
          ``VectorView`` whose length need not match ``num_controls``
          at compose time.
        - ``args[1:]`` are passed to the sub-kernel.
        - ``control_indices=(i0, i1, ...)`` selects exactly
          ``num_controls`` slots from the pool to act as controls;
          omitted slots pass through unchanged.  Each index entry is
          ``int`` or :class:`UInt` (mixing allowed).  ``int``-only
          duplicates and negatives are rejected at compose time;
          everything else (length match, range, ``UInt`` duplicates)
          is deferred to emit time.

        Args:
            *args (Any): Control and sub-kernel arguments per the
                mode-specific protocol described above.
            power (int | UInt): How many times to apply ``U``.  Must
                be a strictly positive integer (``UInt`` handles are
                accepted for symbolic powers, e.g. ``2 ** k`` in QPE).
                Defaults to ``1``.
            control_indices (Sequence[int | UInt] | None): Symbolic
                mode only — see above.  Defaults to ``None`` which
                means "use the entire control pool".  Passing a
                non-``None`` value in concrete mode raises
                :class:`ValueError`.
            **params (ParamValue): Sub-kernel classical parameters
                (``theta=...``, etc.).

        Returns:
            tuple[Any, ...]: One output handle per input handle, in
                the concatenation order ``(controls, sub_kernel_quantum)``.
                Each output handle's runtime kind matches the
                corresponding input kind (scalar ``Qubit`` →
                ``Qubit``, ``VectorView`` → ``VectorView``,
                ``Vector`` → ``Vector``).

        Raises:
            ValueError: ``control_indices`` is non-``None`` in
                concrete mode, or the qubit-count split in concrete
                mode falls inside an argument.
            TypeError: ``power`` is not a positive integer / ``UInt``,
                a ``control_indices`` entry is not ``int`` / ``UInt``,
                or a sub-kernel kwarg does not match the wrapped
                kernel's signature.
            QubitConsumedError / QubitBorrowConflictError: Duplicate
                physical qubits across the control + sub-kernel args
                (caught by the ``Handle.consume()`` / array
                borrow-tracker layer), or a quantum arg that was
                already consumed before the call.
        """
        normalized_power = self._normalize_power(power)
        num_controls = self._num_controls

        if isinstance(num_controls, UInt):
            return self._call_symbolic(args, normalized_power, params, control_indices)

        if control_indices is not None:
            raise ValueError(
                "control_indices is only valid in symbolic mode "
                "(num_controls=UInt).  Got concrete num_controls; "
                "concrete-mode controls are positional and have no "
                "selection step (see design §1.1)."
            )
        return self._call_concrete(args, params, normalized_power)

    def _normalize_control_indices(
        self,
        control_indices: Sequence[int | UInt],
    ) -> tuple[Value, ...]:
        """Lift a caller-supplied index sequence to a tuple of ``UInt`` Values.

        ``int`` literals are wrapped in a constant ``Value`` of
        ``UIntType`` so the IR carries a uniform ``tuple[Value, ...]``
        regardless of whether the caller spelled the index with a
        Python literal or a ``UInt`` handle.  Compose-time validation
        catches the cheap-to-detect mistakes (sequence type, element
        type, ``int``-only duplicates and negatives) — the
        ``UInt``-aware checks are deferred to emit time, where the
        bindings are available.

        Args:
            control_indices (Sequence[int | UInt]): Caller-supplied
                index sequence.  Lists, tuples, and any other
                ``Sequence`` are accepted; the input is normalised to
                a tuple of ``Value``\\ s.

        Returns:
            tuple[Value, ...]: One ``Value`` (``UIntType``) per index
                entry, in the same order.

        Raises:
            TypeError: The sequence is not iterable or an entry is
                not ``int`` / ``UInt``.  ``bool`` is rejected
                explicitly because it would silently coerce through
                ``int(...)``.
            ValueError: Two ``int`` entries are equal (literal
                duplicate) or any ``int`` entry is negative.  Mixed
                ``int`` / ``UInt`` collisions and ``UInt`` /
                ``UInt`` duplicates are deferred to emit time.
        """
        try:
            entries = list(control_indices)
        except TypeError as e:
            raise TypeError(
                f"control_indices must be a Sequence of int / UInt; "
                f"got {type(control_indices).__name__}."
            ) from e

        seen_ints: set[int] = set()
        normalized: list[Value] = []
        for idx in entries:
            if isinstance(idx, bool):
                raise TypeError(
                    f"control_indices: bool entry ({idx!r}) is not "
                    f"allowed; cast to int explicitly if intentional."
                )
            if isinstance(idx, int):
                if idx < 0:
                    raise ValueError(
                        f"control_indices: negative entry ({idx}) is not allowed."
                    )
                if idx in seen_ints:
                    raise ValueError(f"control_indices: duplicate int entry ({idx}).")
                seen_ints.add(idx)
                normalized.append(
                    Value(type=UIntType(), name=f"ctrl_idx_{idx}").with_const(idx)
                )
            elif isinstance(idx, UInt):
                normalized.append(idx.value)
            else:
                raise TypeError(
                    f"control_indices entries must be int or UInt; "
                    f"got {type(idx).__name__}."
                )
        return tuple(normalized)

    def _call_symbolic(
        self,
        args: tuple[Any, ...],
        power: int | Value,
        sub_kwargs: dict[str, Any],
        control_indices: Sequence[int | UInt] | None,
    ) -> tuple[Any, ...]:
        """Symbolic-``num_controls`` path for :meth:`ControlledGate.__call__`.

        Mirrors :meth:`_call_concrete`'s structure but expects a
        ``Vector[Qubit]`` / ``VectorView[Qubit]`` as ``args[0]`` —
        the control *pool* — and routes ``control_indices`` into
        the new ``SymbolicControlledU.control_indices`` field.

        Args:
            args (tuple[Any, ...]): Positional arguments to ``cg(...)``.
            power (int | Value): Normalised power (output of
                :meth:`_normalize_power`).
            sub_kwargs (dict[str, Any]): Caller kwargs after stripping
                the reserved ``power`` and ``control_indices`` keys.
            control_indices (Sequence[int | UInt] | None): The
                caller-supplied selection (or ``None`` to use the
                entire pool).

        Returns:
            tuple[Any, ...]: One output handle per input handle, in
                the concatenation order ``(c_qs_out, sub_kernel_quantum_out)``.

        Raises:
            ValueError: ``args[0]`` is not a ``Vector`` / ``VectorView``,
                or the sub-kernel has no quantum arguments.
            TypeError / QubitConsumedError / QubitBorrowConflictError:
                As documented on :meth:`__call__`.
        """
        from qamomile.circuit.frontend.handle.array import ArrayBase

        num_controls = self._num_controls
        assert isinstance(num_controls, UInt)

        if not args:
            raise ValueError(
                "When num_controls is symbolic (UInt), at least one "
                "positional control argument is required."
            )

        # Split args into (control prefix, sub-kernel positional).
        # The boundary is derived from the wrapped kernel's signature:
        # everything the sub-kernel still expects positionally (after
        # accounting for kwargs) is the trailing chunk; the rest is
        # the control prefix.  The legacy single-pool form (one
        # ``ArrayBase`` control arg) and the new multi-arg form
        # (scalar ``Qubit`` and/or ``ArrayBase`` in sequence, qubit-
        # count sum equals ``num_controls`` at transpile time) both
        # flow through this split.
        sub_positional_count = self._sub_positional_count_for_symbolic(sub_kwargs)
        if sub_positional_count > len(args):
            raise ValueError(
                f"ControlledU: not enough positional args.  The wrapped "
                f"sub-kernel expects {sub_positional_count} positional "
                f"arg(s) after kwargs, got {len(args)} total."
            )
        control_args = list(args[: len(args) - sub_positional_count])
        sub_positional = list(args[len(args) - sub_positional_count :])
        if not control_args:
            raise ValueError(
                "When num_controls is symbolic (UInt), at least one "
                "positional control argument is required."
            )

        is_legacy_pool_form = len(control_args) == 1 and isinstance(
            control_args[0], ArrayBase
        )
        if not is_legacy_pool_form and control_indices is not None:
            raise ValueError(
                "control_indices is only supported with a single "
                "Vector[Qubit] / VectorView[Qubit] control argument "
                "(the pool form).  Combining control_indices with "
                "multiple positional control args is not supported."
            )

        ci_values: tuple[Value, ...] | None = (
            self._normalize_control_indices(control_indices)
            if control_indices is not None
            else None
        )

        sub_args_resolved = self._bind_to_sub_signature(sub_positional, sub_kwargs)
        sub_quantum_args = self._collect_sub_quantum_args(sub_args_resolved)
        if not sub_quantum_args:
            raise ValueError(
                "ControlledU requires at least one quantum sub-kernel "
                "argument (target); got the control prefix and no "
                "sub-kernel quantum arg (see design decision #9)."
            )
        quantum_ids = {id(h) for h in sub_quantum_args}
        sub_classical_dict = {
            name: value
            for name, value in sub_args_resolved.items()
            if id(value) not in quantum_ids
        }

        # Alias / overlap checking is delegated to the
        # ``Handle.consume()`` / array borrow-tracker layer below
        # (same rationale as in ``_call_concrete``).
        consumed_controls = self._consume_with_borrow_transfer(
            control_args, "ControlledU[control]"
        )
        consumed_sub_quantum = self._consume_with_borrow_transfer(
            sub_quantum_args, "ControlledU[target]"
        )

        # Build per-control-arg operand + result.  For the legacy
        # single-pool form (one ``ArrayBase``) this lands the pool
        # ``ArrayValue`` at ``operands[0]``; for the multi-arg form
        # each control arg becomes its own operand (scalar ``Value``
        # for a ``Qubit`` handle, ``ArrayValue`` for a
        # ``Vector``/``VectorView``).  The downstream emit pass walks
        # ``operands[:num_control_args]`` to recover the per-physical
        # qubit control set.
        operands: list[Any] = []
        control_results: list[Value] = []
        for entry in consumed_controls:
            op_value = self._sub_quantum_operand_value(entry)
            operands.append(op_value)
            control_results.append(op_value.next_version())

        sub_quantum_results: list[Value] = []
        for entry in consumed_sub_quantum:
            op_value = self._sub_quantum_operand_value(entry)
            operands.append(op_value)
            sub_quantum_results.append(op_value.next_version())
        self._params_to_operands(sub_classical_dict, operands)

        results: list[Value] = control_results + sub_quantum_results

        op = SymbolicControlledU(
            operands=operands,
            results=results,
            num_controls=num_controls.value,
            control_indices=ci_values,
            power=power,
            block=self._qkernel.block,
            num_control_args=len(consumed_controls),
        )
        get_current_tracer().add_operation(op)

        wrapped: list[Any] = []
        for entry, result_value in zip(consumed_controls, control_results):
            wrapped.append(
                self._wrap_entry_output(
                    entry,
                    [result_value],
                    operation_name="ControlledU[control]",
                )
            )
        for entry, result_value in zip(consumed_sub_quantum, sub_quantum_results):
            wrapped.append(
                self._wrap_entry_output(
                    entry,
                    [result_value],
                    operation_name="ControlledU[target]",
                )
            )
        return tuple(wrapped)

    def _sub_positional_count_for_symbolic(self, sub_kwargs: dict[str, Any]) -> int:
        """How many trailing positional args belong to the sub-kernel.

        Used by :meth:`_call_symbolic` to split the call-site args
        into the control prefix and the sub-kernel positional region.
        The wrapped kernel's declared signature is the source of
        truth: every parameter that is not satisfied via ``sub_kwargs``
        must arrive positionally.

        Args:
            sub_kwargs (dict[str, Any]): Caller kwargs after stripping
                the reserved ``power`` and ``control_indices`` keys.

        Returns:
            int: Number of trailing positional args expected by the
                sub-kernel.
        """
        # ``ControlledGate.__init__`` validates that ``input_types`` is a
        # dict, so the access is safe to use directly here.
        input_types = self._qkernel.input_types
        positional_names = [n for n in input_types.keys() if n not in sub_kwargs]
        return len(positional_names)


def _classify_callable_param(annotation: Any) -> str:
    """Classify a callable's parameter annotation for ``control()`` wrapping.

    Built-in gate functions (``qmc.rx``, ``qmc.h``, etc.) typically annotate
    their qubit operand as ``Qubit | Vector[Qubit]`` because they support
    both scalar and broadcast invocation.  When wrapping for ``control``
    we always invoke them with a single ``Qubit``, so we treat any union
    that includes ``Qubit`` as a qubit parameter and likewise for
    ``Float``/``UInt``.

    Args:
        annotation: The parameter annotation, already resolved via
            ``get_type_hints`` so string forms have been turned into real
            type objects.

    Returns:
        One of ``"qubit"``, ``"float"``, ``"uint"``, or ``"unknown"``.
    """
    if annotation is Qubit:
        return "qubit"
    if annotation is float or annotation is Float:
        return "float"
    if annotation is int or annotation is UInt:
        return "uint"

    origin = get_origin(annotation)
    if origin in (Union, _types.UnionType):
        for arg in get_args(annotation):
            kind = _classify_callable_param(arg)
            if kind != "unknown":
                return kind
        return "unknown"

    return "unknown"


def _qkernel_for_callable(fn: Callable[..., Any]) -> QKernel:
    """Synthesize a ``@qkernel`` wrapper around a built-in gate callable.

    Inspects the callable's signature to classify each parameter, then
    generates a tiny wrapper function that simply forwards the call.  The
    wrapper is decorated with ``@qkernel`` so that the rest of the
    ``control`` machinery can consume it unchanged.

    The wrapper's source is registered with ``linecache`` so that
    ``inspect.getsource`` (used by ``transform_control_flow``) can retrieve
    it.  The synthesized wrapper has no control flow, so the AST transform
    is effectively a no-op.

    The synthesized wrapper declares its parameters in ``qubits-first``
    order (required by the downstream controlled-U emit pass, which
    assumes the wrapped block's inputs are ``[qubit..., param...]``) but
    forwards arguments to ``fn`` **by keyword** in ``fn``'s original
    order.  This means callables whose qubit and classical parameters are
    interleaved (e.g. ``def gate(c: Qubit, theta: float, t: Qubit)``)
    still receive each argument at the correct position.

    Successful syntheses are cached per callable in a
    ``WeakKeyDictionary`` keyed by ``fn``; the cached ``QKernel``'s
    block is constructed eagerly and the wrapper holds only a
    ``weakref.proxy`` to ``fn``, so once the caller releases ``fn`` the
    cache entry (and its ``linecache`` source) is freed automatically.
    Concurrent calls for the same ``fn`` are de-duplicated under a
    module lock so we never compile the same wrapper twice.

    Args:
        fn: The callable to wrap.  Each parameter must have a type
            annotation that resolves to ``Qubit``, ``Float``/``float``, or
            ``UInt``/``int`` — possibly inside a ``Union`` (e.g., the
            ``Union[Qubit, Vector[Qubit]]`` used by broadcast gate
            functions).

    Returns:
        A ``QKernel`` whose body forwards every argument by keyword to
        the original callable (``return __qmc_target__(name=name, ...)``
        in ``fn``'s original signature order, where ``__qmc_target__``
        is the wrapper-globals binding for ``fn``).  The wrapper itself
        declares its parameters in ``qubits-first`` order with concrete
        ``Qubit`` / ``Float`` / ``UInt`` annotations matching the
        downstream emit pass's ``[qubit..., param...]`` block layout —
        kwargs forwarding is what reconciles that with ``fn``'s
        possibly-interleaved signature.

    Raises:
        TypeError: If ``fn`` is not callable, uses ``*args``/``**kwargs``
            or positional-only parameters, lacks an annotation on any
            parameter, has an unsupported annotation, has no qubit
            parameters at all, or fails to compile as a wrapper for any
            other reason.
    """
    from qamomile.circuit.frontend.qkernel import QKernel, qkernel as _qkernel_decorator

    if isinstance(fn, QKernel):
        return fn

    # Duck-typed kernel-like objects (e.g. test mocks with a stubbed
    # ``.block``) are passed through unchanged so the existing
    # ``ControlledGate`` consumer continues to work without coupling the
    # synthesis path to a strict ``isinstance`` check.
    if hasattr(fn, "block"):
        return cast("QKernel", fn)

    if not callable(fn):
        raise TypeError(
            f"control(): expected a QKernel or a built-in gate function, "
            f"got {type(fn).__name__}."
        )

    fn_name = getattr(fn, "__name__", "anonymous_gate")

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"control(): cannot inspect signature of {fn_name!r}: {e}. "
            f"Wrap the function in @qmc.qkernel manually."
        ) from e

    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError, AttributeError):
        # ``get_type_hints`` raises across a few branches: ``NameError``
        # for forward refs that cannot resolve, ``TypeError`` for
        # malformed annotations, and ``AttributeError`` from internal
        # ``__annotations__`` / namespace probing on certain wrapper
        # objects.  Catching this set lets us fall back to the raw
        # ``param.annotation`` while still letting genuinely unrelated
        # errors (e.g. an import error inside the caller's module) bubble
        # up rather than being silently re-emitted as "no type
        # annotation".  ``qamomile/circuit/frontend/qkernel.py`` has the
        # same fall-back at its own ``get_type_hints`` site, where it
        # uses a broader ``Exception`` catch — keeping ours narrower
        # surfaces real bugs faster while still covering the cases that
        # actually arise in practice.
        type_hints = {}

    # The wrapper's parameter order must be ``qubits-first`` because the
    # downstream ``ControlledUOperation`` emit pass assumes the wrapped
    # ``Block``'s ``input_values`` are laid out as ``[qubit..., param...]``
    # (matching ``ControlledGate.__call__``'s operand convention).
    # Re-ordering only affects the wrapper's *declaration*; the actual
    # forwarding call below uses keyword arguments, so the original
    # callable receives each parameter at the right position even when
    # its own signature interleaves qubits and classical params (e.g.
    # ``def gate(c: Qubit, theta: float, t: Qubit)``).
    qubit_args: list[str] = []
    classical_args: list[tuple[str, str]] = []  # (name, "Float" | "UInt")
    original_call_order: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"control(): callable {fn_name!r} uses *args/**kwargs, "
                f"positional-only, or keyword-only parameters, which "
                f"cannot be auto-wrapped (the synthesized wrapper places "
                f"all params in the standard POSITIONAL_OR_KEYWORD slot). "
                f"Wrap it in @qmc.qkernel manually."
            )
        # Reject default values up-front: the synthesized wrapper does
        # not propagate them to the wrapper signature, so silently
        # dropping a default would change the caller's expected
        # contract (``cg(c, t)`` would unexpectedly require an
        # otherwise-defaulted kwarg).  Built-in Qamomile gates have no
        # defaults today, so this only fires for user-supplied helpers,
        # which are exactly the ones that should be wrapped with
        # ``@qmc.qkernel`` directly anyway.
        if param.default is not inspect.Parameter.empty:
            raise TypeError(
                f"control(): parameter {param_name!r} of {fn_name!r} "
                f"has a default value ({param.default!r}), which the "
                f"wrapper synthesizer does not propagate. Wrap the "
                f"function in @qmc.qkernel manually."
            )
        # Reject parameter names that collide with a reserved
        # wrapper-internal binding.  Most damaging is a parameter named
        # ``__qmc_target__``: the synthesized body is
        # ``return __qmc_target__(...)``, and a parameter of the same
        # name would shadow the injected forwarding target with the
        # caller-supplied scalar value, raising a confusing ``TypeError``
        # ("'float' object is not callable") at block-construction time.
        # Param names that match the injected types (``Qubit`` / ``Float``
        # / ``UInt`` / ``tuple``) are rejected for the same reason
        # defensively — annotation evaluation happens before the
        # parameter binding so it usually still resolves to the type, but
        # any future change to that ordering would silently corrupt
        # type-hint resolution.  Catching it up-front keeps the contract
        # symmetric with the ``__name__`` collision guard above.
        if param_name in _RESERVED_WRAPPER_NAMES:
            raise TypeError(
                f"control(): parameter {param_name!r} of {fn_name!r} "
                f"collides with a reserved wrapper-internal name "
                f"({sorted(_RESERVED_WRAPPER_NAMES)}). Wrap the function "
                f"in @qmc.qkernel manually or rename the parameter."
            )
        annotation = type_hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(
                f"control(): parameter {param_name!r} of {fn_name!r} has "
                f"no type annotation. Wrap the function in @qmc.qkernel "
                f"manually."
            )
        kind = _classify_callable_param(annotation)
        if kind == "qubit":
            qubit_args.append(param_name)
        elif kind == "float":
            classical_args.append((param_name, "Float"))
        elif kind == "uint":
            classical_args.append((param_name, "UInt"))
        else:
            raise TypeError(
                f"control(): parameter {param_name!r} of {fn_name!r} has "
                f"annotation {annotation!r}; only Qubit, Float, UInt (or "
                f"unions including those types, e.g. "
                f"Union[Qubit, Vector[Qubit]]) are supported. Wrap the "
                f"function in @qmc.qkernel manually."
            )
        original_call_order.append(param_name)

    if not qubit_args:
        raise TypeError(
            f"control(): callable {fn_name!r} has no Qubit parameters; "
            f"a controlled gate requires at least one target qubit."
        )

    n_qubits = len(qubit_args)
    param_decls = [f"{q}: Qubit" for q in qubit_args]
    for c_name, c_type in classical_args:
        param_decls.append(f"{c_name}: {c_type}")
    # Forward by keyword in the *original* order so callables whose qubit
    # and classical params are interleaved still receive each argument at
    # the correct position.  ``POSITIONAL_ONLY`` is rejected above, so all
    # surviving params are name-addressable.
    invocation = ", ".join(f"{name}={name}" for name in original_call_order)

    return_anno = (
        "Qubit" if n_qubits == 1 else f"tuple[{', '.join(['Qubit'] * n_qubits)}]"
    )

    # All cache reads and writes — and the synthesis itself — happen
    # under the shared lock.  ``WeakKeyDictionary`` is *not* safe for
    # concurrent read/write: an unlocked ``get()`` can race with a
    # background weakref cleanup or with another thread populating the
    # cache and produce hard-to-reproduce errors, so we deliberately
    # forgo a lock-free fast path.  The same lock also protects against
    # concurrent ``control(fn)`` calls racing to (a) re-use the same
    # filename-counter sequence, (b) double-mutate ``linecache.cache``,
    # or (c) populate the cache with two distinct wrappers for the same
    # callable.  Cache hits are still cheap because Python locks are
    # uncontended in the common single-threaded case.
    global _synthesized_kernel_counter
    with _synthesized_kernel_lock:
        # Look up both caches: ``_synthesized_kernel_cache`` for normal
        # weakrefable callables, ``_synthesized_kernel_cache_strong`` for
        # the C-implemented builtins that ``WeakKeyDictionary`` rejects.
        try:
            cached = _synthesized_kernel_cache.get(fn)
        except TypeError:
            cached = None
        if cached is None:
            try:
                cached = _synthesized_kernel_cache_strong.get(fn)
            except TypeError:
                # Non-hashable callables (extremely rare) cannot be cached
                # in either backend; we just synthesize a fresh wrapper.
                cached = None
        if cached is not None:
            return cast("QKernel", cached)

        _synthesized_kernel_counter += 1
        seq = _synthesized_kernel_counter

        # Choose the wrapper's source-level identifier.  Prefer the original
        # callable name (so QKernel display + ``transform_control_flow``'s
        # ``name_space[func.__name__]`` lookup show the gate name), but
        # fall back to a fresh ``_qmc_controlled_wrapper_<seq>`` identifier
        # when ``fn_name``:
        #   * is not a valid Python identifier (e.g. ``<lambda>``),
        #   * is a reserved Python keyword (``def class(...):`` is a
        #     ``SyntaxError`` even though ``"class".isidentifier()`` is
        #     ``True``),
        #   * or would collide with one of the names we inject into the
        #     exec namespace (``Qubit``, ``Float``, ``UInt``, ...);
        #     shadowing those bindings with a same-named function
        #     definition would break type-hint resolution at decoration
        #     time.
        if (
            fn_name.isidentifier()
            and not keyword.iskeyword(fn_name)
            and fn_name not in _RESERVED_WRAPPER_NAMES
        ):
            wrapper_internal_name = fn_name
        else:
            wrapper_internal_name = f"_qmc_controlled_wrapper_{seq}"

        src = (
            f"def {wrapper_internal_name}({', '.join(param_decls)}) -> "
            f"{return_anno}:\n"
            f"    return __qmc_target__({invocation})\n"
        )

        # Filenames in ``<...>`` form bypass on-disk lookup in
        # inspect/linecache but still pass through ``linecache.getlines``.
        # ``mtime=None`` makes ``linecache.checkcache`` skip the entry so
        # the cache survives across calls.
        filename = f"<qamomile-control-wrapper-{fn_name}-{seq}>"
        src_lines = src.splitlines(keepends=True)
        linecache.cache[filename] = (len(src), None, src_lines, filename)

        # Use ``weakref.proxy`` for the forwarded target so the cached
        # ``QKernel`` does not transitively keep ``fn`` alive: the wrapper
        # captures the proxy in its globals, not ``fn`` itself.  This is
        # what lets ``WeakKeyDictionary`` actually free its entry once
        # the original callable becomes unreachable.  The wrapper is only
        # *called* once — during ``func_to_block`` below, while ``fn`` is
        # still strongly referenced by our caller — so post-build use of
        # ``cg._qkernel.block`` never re-invokes the proxy.  Some
        # C-implemented callables don't support weak references; in that
        # case fall back to a strong ref (the cache then simply holds
        # those entries indefinitely, which matches the prior behavior).
        target_ref: Any
        try:
            target_ref = weakref.proxy(fn)
        except TypeError:
            target_ref = fn

        namespace: dict[str, Any] = _wrapper_namespace(target_ref)
        try:
            code = compile(src, filename, "exec")
            exec(code, namespace)
            wrapper_fn = namespace[wrapper_internal_name]
            qkernel_inst = _qkernel_decorator(wrapper_fn)
            # Force ``Block`` construction now, while ``fn`` is still
            # strongly referenced by our caller.  After this point the
            # wrapper / proxy is never invoked again — ``cg._qkernel.block``
            # only consumes the already-built IR — so it's safe for ``fn``
            # to be GC'd later, releasing this cache entry along with it.
            _ = qkernel_inst.block
        except Exception as e:
            # Drop the half-populated linecache entry on failure so a
            # retry with a corrected callable does not leave dead source
            # behind.
            linecache.cache.pop(filename, None)
            raise TypeError(
                f"control(): failed to synthesize a wrapper for "
                f"{fn_name!r}: {e}. Wrap the function in @qmc.qkernel "
                f"manually."
            ) from e

        # Populate the cache.  Try the weak-key cache first so the
        # entry can be released when ``fn`` is GC'd.  When ``fn`` cannot
        # be weakly referenced (a small set of C-implemented builtins),
        # fall back to a strong-reference cache so repeated
        # ``control(fn)`` calls still memoize the wrapper instead of
        # re-synthesizing forever and leaking ``linecache`` entries.
        # Non-hashable callables fall through silently (no cache).
        try:
            _synthesized_kernel_cache[fn] = qkernel_inst
        except TypeError:
            try:
                _synthesized_kernel_cache_strong[fn] = qkernel_inst
            except TypeError:
                pass  # non-hashable: cannot cache, will re-synthesize

        # When ``fn`` is GC'd the WeakKeyDictionary entry vanishes on
        # its own, but ``linecache.cache`` is a plain dict and would
        # otherwise leak the synthesized source.  ``weakref.finalize``
        # runs the cleanup callback exactly when ``fn`` becomes
        # unreachable; for non-weakrefable callables registration
        # silently no-ops, and the strong-ref cache pinning ``fn`` keeps
        # the ``linecache`` entry alive for the process lifetime — which
        # mirrors the lifetime of those C-implemented builtins anyway.
        try:
            weakref.finalize(fn, linecache.cache.pop, filename, None)
        except TypeError:
            pass

    return qkernel_inst


def control(
    qkernel: QKernel | Callable[..., Any],
    num_controls: int | UInt = 1,
) -> ControlledGate:
    """Create a controlled version of a quantum gate.

    Accepts either a ``@qmc.qkernel``-decorated function or a plain built-in
    gate callable (``qmc.rx``, ``qmc.h``, ``qmc.cp``, ...).  When given a
    plain callable, a thin ``@qkernel`` wrapper is synthesized automatically
    by inspecting the callable's signature, so users no longer need to
    write a one-line wrapper just to control a primitive gate.

    Args:
        qkernel: A ``QKernel`` defining the gate to control, or a built-in
            gate callable whose parameters are annotated with ``Qubit``,
            ``Float``/``float``, or ``UInt``/``int`` (possibly inside a
            ``Union`` such as ``Union[Qubit, Vector[Qubit]]``).
        num_controls: Number of control qubits (default: 1).  Can be ``int``
            (concrete) or ``UInt`` (symbolic).

    Returns:
        A ``ControlledGate`` that can be called with
        ``(*controls, *targets, **params)``.

    Raises:
        TypeError: If ``qkernel`` is a callable that cannot be auto-wrapped
            (missing annotations, unsupported types, or no qubit
            parameters).
        ValueError: If ``num_controls`` is a concrete ``int`` less than 1.

    Example:
        Built-in gates can be controlled directly, with no wrapper::

            crx = qmc.control(qmc.rx)
            ctrl_out, tgt_out = crx(ctrl, target, angle=0.5)

            cch = qmc.control(qmc.h, num_controls=2)
            c0, c1, tgt = cch(ctrl0, ctrl1, target)

        ``@qmc.qkernel`` arguments are still supported for cases that need
        custom logic::

            @qmc.qkernel
            def rx_then_h(q: Qubit, theta: float) -> Qubit:
                q = qmc.rx(q, theta)
                q = qmc.h(q)
                return q

            ctrl_out, tgt_out = qmc.control(rx_then_h)(ctrl, target, theta=0.5)
    """
    return ControlledGate(_qkernel_for_callable(qkernel), num_controls=num_controls)
