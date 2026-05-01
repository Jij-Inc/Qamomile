"""Controlled gate operations."""

from __future__ import annotations

import inspect
import linecache
import threading
import types as _types
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
    IndexSpecControlledU,
    SymbolicControlledU,
)
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import QubitAliasError

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel

# Type alias for parameter values
ParamValue = Union[float, int, Float, UInt]

# Counter for synthesized-wrapper filenames; ensures distinct
# ``linecache`` entries even when the same gate is wrapped multiple times.
_synthesized_kernel_counter = 0
_synthesized_kernel_lock = threading.Lock()

# Cache from a built-in gate callable to its synthesized ``QKernel`` wrapper.
# A ``WeakKeyDictionary`` lets the entry vanish automatically when the
# original callable is garbage-collected, so we never grow the
# ``linecache`` unboundedly in long-running processes that repeatedly call
# ``controlled(qmc.rx)`` etc.  Callables that don't support weak refs
# (a small set of C-implemented builtins) silently bypass the cache.
_synthesized_kernel_cache: "weakref.WeakKeyDictionary[Callable[..., Any], Any]" = (
    weakref.WeakKeyDictionary()
)

# Names that the synthesized wrapper's body references directly (via
# annotations and the forwarding call).  If the original callable's
# ``__name__`` collides with any of these, defining the wrapper with that
# same name would shadow the injected binding and break type-hint
# resolution at decoration time, so we fall back to a fresh internal
# identifier in that case.
_RESERVED_WRAPPER_NAMES = frozenset(
    {"Qubit", "Float", "UInt", "tuple", "__qmc_target__"}
)


class ControlledGate:
    """Wrapper for controlled version of a QKernel.

    Created by calling `controlled(qkernel)`. The resulting object
    can be called like a gate function.

    Example:
        @qmc.qkernel
        def phase_gate(q: Qubit, theta: float) -> Qubit:
            return qmc.p(q, theta)

        controlled_phase = qmc.controlled(phase_gate)
        ctrl_out, tgt_out = controlled_phase(ctrl, target, theta=0.5)

        # Double-controlled
        cc_phase = qmc.controlled(phase_gate, num_controls=2)
        c0, c1, tgt = cc_phase(ctrl0, ctrl1, target, theta=0.5)
    """

    def __init__(self, qkernel: "QKernel", num_controls: int | UInt = 1) -> None:
        if isinstance(num_controls, int) and num_controls < 1:
            raise ValueError(f"num_controls must be >= 1, got {num_controls}.")
        # For UInt (symbolic), validation is deferred to emit time
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

    @staticmethod
    def _params_to_operands(
        params: dict[str, ParamValue],
        operands: list[Any],
    ) -> None:
        """Append parameter values to the operands list.

        Handle types are unwrapped to their underlying Value.
        Raw float/int values are wrapped as constant FloatType Values.
        """
        for param_name, param_value in params.items():
            if isinstance(param_value, Handle):
                operands.append(param_value.value)
            else:
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
        *,
        target_indices: list[Value] | None = None,
        controlled_indices: list[Value] | None = None,
    ) -> ControlledUOperation:
        """Create the appropriate ControlledUOperation subclass and add to tracer."""
        block = self._qkernel.block
        op: ControlledUOperation
        if target_indices is not None or controlled_indices is not None:
            op = IndexSpecControlledU(
                operands=operands,
                results=results,
                num_controls=num_controls,
                power=power,
                target_indices=target_indices,
                controlled_indices=controlled_indices,
                block=block,
            )
        elif isinstance(num_controls, Value):
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

    def __call__(
        self,
        *args: Any,
        power: int | UInt = 1,
        target_indices: list[int | UInt] | None = None,
        controlled_indices: list[int | UInt] | None = None,
        **params: ParamValue,
    ) -> tuple[Any, ...] | Any:
        """Apply controlled gate.

        For concrete num_controls (int):
            *args: First num_controls qubits are controls, rest are targets.
        For symbolic num_controls (UInt):
            args[0]: Vector[Qubit] (controls), args[1:]: individual Qubit targets.
        With target_indices or controlled_indices:
            args[0]: Single Vector[Qubit], indices specify which elements are
            targets (or controls). Returns a single Vector (not a tuple).

        Args:
            *args: Control and target qubits.
            power: Number of times to apply U. Must be a strictly positive
                integer. Accepts UInt handles for symbolic expressions
                (e.g., 2**k in QPE).
            target_indices: Indices within the Vector that are targets.
                The remaining elements become controls.
            controlled_indices: Indices within the Vector that are controls.
                The remaining elements become targets.
            **params: Parameters for the underlying gate (e.g., theta=0.5)

        Returns:
            Tuple of output handles, or single Vector when using index spec.
        """
        normalized_power = self._normalize_power(power)

        if target_indices is not None and controlled_indices is not None:
            raise ValueError(
                "Cannot specify both target_indices and controlled_indices. "
                "Use one or the other."
            )

        if target_indices is not None or controlled_indices is not None:
            return self._call_with_index_spec(
                args, normalized_power, target_indices, controlled_indices, params
            )

        num_controls = self._num_controls

        if isinstance(num_controls, UInt):
            return self._call_symbolic(args, normalized_power, params)

        # --- Concrete path (existing logic, unchanged) ---

        # Split args into controls and targets
        if len(args) <= num_controls:
            raise ValueError(
                f"ControlledU requires at least {num_controls + 1} qubits "
                f"({num_controls} controls + at least 1 target), got {len(args)}."
            )
        controls = args[:num_controls]
        target_args = args[num_controls:]

        # Check for aliasing (same physical qubit used in multiple positions)
        seen_ids: set[str] = set()
        for q in args:
            lid = q.value.logical_id
            if lid in seen_ids:
                q_name = q.name or "unnamed"
                raise QubitAliasError(
                    f"Cannot use the same qubit in multiple positions of ControlledU.\n"
                    f"Qubit '{q_name}' appears more than once.\n\n"
                    f"Fix: Use distinct qubits for each control and target.",
                    handle_name=q_name,
                    operation_name="ControlledU",
                )
            seen_ids.add(lid)

        # Consume all qubit handles (enforces affine type)
        controls = tuple(
            c.consume(operation_name="ControlledU[control]") for c in controls
        )
        target_args = tuple(
            t.consume(operation_name="ControlledU[target]") for t in target_args
        )

        # Build operands: [control(s), target(s), param(s)]
        operands: list[Any] = [c.value for c in controls] + [
            t.value for t in target_args
        ]
        self._params_to_operands(params, operands)

        # Build results: [control_out(s), target_out(s)]
        results: list[Value] = [c.value.next_version() for c in controls] + [
            t.value.next_version() for t in target_args
        ]

        self._build_and_emit_op(
            operands,
            results,
            num_controls,
            normalized_power,
        )

        # Return output handles
        ctrl_outs = self._wrap_qubit_outputs(list(controls), results)
        tgt_outs = self._wrap_qubit_outputs(
            list(target_args), results, offset=num_controls
        )
        return tuple(ctrl_outs + tgt_outs)

    def _call_with_index_spec(
        self,
        args: tuple[Any, ...],
        power: int | Value,
        target_indices: list[int | UInt] | None,
        controlled_indices: list[int | UInt] | None,
        params: dict[str, ParamValue],
    ) -> Any:
        """Handle index-spec mode: single Vector with explicit target/control indices."""
        from qamomile.circuit.frontend.handle.array import ArrayBase, Vector

        # Validate: exactly one Vector argument
        if len(args) != 1 or not isinstance(args[0], ArrayBase):
            raise ValueError(
                "When target_indices or controlled_indices is specified, "
                "exactly one Vector[Qubit] must be provided."
            )

        indices = target_indices if target_indices is not None else controlled_indices
        assert indices is not None

        # Validate: non-empty
        if len(indices) == 0:
            raise ValueError(
                "At least one index must be specified in "
                "target_indices or controlled_indices."
            )

        # Validate: no duplicate concrete indices
        concrete = [i for i in indices if isinstance(i, int)]
        if len(set(concrete)) != len(concrete):
            raise ValueError(
                "Duplicate indices are not allowed in "
                "target_indices/controlled_indices."
            )

        # Convert indices to Value list
        def _to_value_list(idx_list: list[int | UInt]) -> list[Value]:
            result: list[Value] = []
            for idx in idx_list:
                if isinstance(idx, int):
                    val = Value(
                        type=UIntType(),
                        name=f"idx_{idx}",
                    ).with_const(idx)
                    result.append(val)
                elif isinstance(idx, UInt):
                    result.append(idx.value)
                else:
                    raise TypeError(f"Index must be int or UInt, got {type(idx)}")
            return result

        ti_values = (
            _to_value_list(target_indices) if target_indices is not None else None
        )
        ci_values = (
            _to_value_list(controlled_indices)
            if controlled_indices is not None
            else None
        )

        vector = args[0]
        # Consume the Vector (affine type)
        vector = vector.consume(operation_name="ControlledU[index_spec]")

        # operands: [ArrayValue, params...]
        operands: list[Any] = [vector.value]
        self._params_to_operands(params, operands)

        results: list[Value[Any]] = [vector.value.next_version()]

        nc = self._num_controls
        if isinstance(nc, UInt):
            nc = nc.value

        self._build_and_emit_op(
            operands,
            results,
            nc,
            power,
            target_indices=ti_values,
            controlled_indices=ci_values,
        )

        return Vector._create_from_value(
            cast(ArrayValue[Any], results[0]), vector.shape, vector.value.name
        )

    def _call_symbolic(
        self,
        args: tuple[Any, ...],
        power: int | Value,
        params: dict[str, ParamValue],
    ) -> tuple[Any, ...]:
        """Handle symbolic num_controls (UInt).

        Convention: args[0] is Vector[Qubit] (controls), args[1:] are targets.
        """
        from qamomile.circuit.frontend.handle.array import ArrayBase, Vector

        num_controls = self._num_controls
        assert isinstance(num_controls, UInt)

        if not args or not isinstance(args[0], ArrayBase):
            raise ValueError(
                "When num_controls is symbolic (UInt), the first argument "
                "must be a Vector[Qubit] for control qubits."
            )

        control_vector = args[0]
        target_args = args[1:]

        if not target_args:
            raise ValueError("ControlledU requires at least 1 target qubit.")

        # Consume control vector (affine type enforcement)
        control_vector = control_vector.consume(operation_name="ControlledU[controls]")

        # Consume target qubits
        consumed_targets = [
            tgt.consume(operation_name="ControlledU[target]") for tgt in target_args
        ]

        # Build operands: [control_vector, target(s), param(s)]
        operands: list[Any] = [control_vector.value] + [
            t.value for t in consumed_targets
        ]
        self._params_to_operands(params, operands)

        # Build results: [control_vector_out, target_out(s)]
        results: list[Value] = [control_vector.value.next_version()] + [
            t.value.next_version() for t in consumed_targets
        ]

        self._build_and_emit_op(
            operands,
            results,
            num_controls.value,
            power,
        )

        # Control vector output
        control_out: Vector[Any] = Vector._create_from_value(
            cast(ArrayValue[Any], results[0]),
            control_vector.shape,
            control_vector.value.name,
        )

        # Target outputs
        tgt_outs = self._wrap_qubit_outputs(consumed_targets, results, offset=1)
        return tuple([control_out] + tgt_outs)


def _classify_callable_param(annotation: Any) -> str:
    """Classify a callable's parameter annotation for ``controlled()`` wrapping.

    Built-in gate functions (``qmc.rx``, ``qmc.h``, etc.) typically annotate
    their qubit operand as ``Qubit | Vector[Qubit]`` because they support
    both scalar and broadcast invocation.  When wrapping for ``controlled``
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
    ``controlled`` machinery can consume it unchanged.

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
    still receive each argument at the correct position.  Successful
    syntheses are cached per callable in a ``WeakKeyDictionary`` so
    repeated calls reuse the same ``QKernel`` and the ``linecache`` does
    not grow without bound.

    Args:
        fn: The callable to wrap.  Each parameter must have a type
            annotation that resolves to ``Qubit``, ``Float``/``float``, or
            ``UInt``/``int`` — possibly inside a ``Union`` (e.g., the
            ``Union[Qubit, Vector[Qubit]]`` used by broadcast gate
            functions).

    Returns:
        A ``QKernel`` whose body is ``return fn(*args)`` with concrete
        ``Qubit``/``Float``/``UInt`` annotations.

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
            f"controlled(): expected a QKernel or a built-in gate function, "
            f"got {type(fn).__name__}."
        )

    # Cache lookup: a previous ``controlled(fn)`` call for the same callable
    # already produced a wrapper QKernel, so reuse it.  Some C-implemented
    # builtins (rare for gate functions) cannot be weak-referenced; in that
    # case the cache silently no-ops via the TypeError fallbacks below.
    try:
        cached = _synthesized_kernel_cache.get(fn)
    except TypeError:
        cached = None
    if cached is not None:
        return cast("QKernel", cached)

    fn_name = getattr(fn, "__name__", "anonymous_gate")

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"controlled(): cannot inspect signature of {fn_name!r}: {e}. "
            f"Wrap the function in @qmc.qkernel manually."
        ) from e

    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError):
        # Forward refs that cannot resolve at this point fall back to
        # raw annotations; classification then proceeds against
        # ``param.annotation`` directly.  We catch only the two
        # exception types ``get_type_hints`` actually raises so a real
        # bug (e.g. an import error in the caller's module) is not
        # silently swallowed and re-emitted as "no type annotation".
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
        ):
            raise TypeError(
                f"controlled(): callable {fn_name!r} uses *args/**kwargs or "
                f"positional-only parameters, which cannot be auto-wrapped. "
                f"Wrap it in @qmc.qkernel manually."
            )
        annotation = type_hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(
                f"controlled(): parameter {param_name!r} of {fn_name!r} has "
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
                f"controlled(): parameter {param_name!r} of {fn_name!r} has "
                f"annotation {annotation!r}; only Qubit, Float, UInt (or "
                f"unions including those types, e.g. "
                f"Union[Qubit, Vector[Qubit]]) are supported. Wrap the "
                f"function in @qmc.qkernel manually."
            )
        original_call_order.append(param_name)

    if not qubit_args:
        raise TypeError(
            f"controlled(): callable {fn_name!r} has no Qubit parameters; "
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

    global _synthesized_kernel_counter
    with _synthesized_kernel_lock:
        _synthesized_kernel_counter += 1
        seq = _synthesized_kernel_counter

    # Choose the wrapper's source-level identifier.  Prefer the original
    # callable name (so QKernel display + ``transform_control_flow``'s
    # ``name_space[func.__name__]`` lookup show the gate name), but fall
    # back to a fresh ``_qmc_controlled_wrapper_<seq>`` identifier when
    # ``fn_name`` is not a valid Python identifier (e.g. ``<lambda>``) or
    # would collide with names we inject into the exec namespace
    # (``Qubit`` / ``Float`` / ``UInt`` / ``tuple`` / ``__qmc_target__``).
    # The collision case matters because the wrapper's annotations refer
    # to those injected names directly; shadowing them with a same-named
    # function definition would break type-hint resolution.
    if fn_name.isidentifier() and fn_name not in _RESERVED_WRAPPER_NAMES:
        wrapper_internal_name = fn_name
    else:
        wrapper_internal_name = f"_qmc_controlled_wrapper_{seq}"

    src = (
        f"def {wrapper_internal_name}({', '.join(param_decls)}) -> "
        f"{return_anno}:\n"
        f"    return __qmc_target__({invocation})\n"
    )

    # Filenames in ``<...>`` form bypass on-disk lookup in inspect/linecache
    # but still pass through ``linecache.getlines``.  Using ``mtime=None``
    # makes ``linecache.checkcache`` skip the entry, so the cache survives.
    filename = f"<qamomile-controlled-wrapper-{fn_name}-{seq}>"
    src_lines = src.splitlines(keepends=True)
    linecache.cache[filename] = (len(src), None, src_lines, filename)

    namespace: dict[str, Any] = {
        "Qubit": Qubit,
        "Float": Float,
        "UInt": UInt,
        "tuple": tuple,
        "__qmc_target__": fn,
    }
    try:
        code = compile(src, filename, "exec")
        exec(code, namespace)
        wrapper_fn = namespace[wrapper_internal_name]
    except Exception as e:
        # Drop the half-populated linecache entry on failure so a retry
        # with a corrected callable does not leave dead source behind.
        linecache.cache.pop(filename, None)
        raise TypeError(
            f"controlled(): failed to synthesize a wrapper for {fn_name!r}: "
            f"{e}. Wrap the function in @qmc.qkernel manually."
        ) from e

    qkernel_inst = _qkernel_decorator(wrapper_fn)

    # Populate the cache; ignore weakref failures for non-weakrefable
    # callables (the wrapper is still returned, just not memoized).
    try:
        _synthesized_kernel_cache[fn] = qkernel_inst
    except TypeError:
        pass

    return qkernel_inst


def controlled(
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

            crx = qmc.controlled(qmc.rx)
            ctrl_out, tgt_out = crx(ctrl, target, angle=0.5)

            cch = qmc.controlled(qmc.h, num_controls=2)
            c0, c1, tgt = cch(ctrl0, ctrl1, target)

        ``@qmc.qkernel`` arguments are still supported for cases that need
        custom logic::

            @qmc.qkernel
            def rx_then_h(q: Qubit, theta: float) -> Qubit:
                q = qmc.rx(q, theta)
                q = qmc.h(q)
                return q

            ctrl_out, tgt_out = qmc.controlled(rx_then_h)(ctrl, target, theta=0.5)
    """
    return ControlledGate(_qkernel_for_callable(qkernel), num_controls=num_controls)
