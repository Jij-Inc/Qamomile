from __future__ import annotations

import ast
import inspect
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
    get_type_hints,
)

import numpy as np

from qamomile.circuit.frontend.ast_transform import (
    collect_quantum_rebind_violations,
    transform_control_flow,
)
from qamomile.circuit.frontend.constructors import bit, float_, qubit_array, uint
from qamomile.circuit.frontend.func_to_block import (
    build_param_slots,
    create_dummy_input,
    func_to_block,
    handle_type_map,
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import Observable, Qubit
from qamomile.circuit.frontend.handle.array import ArrayBase, Vector
from qamomile.circuit.frontend.handle.containers import Dict
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, UInt
from qamomile.circuit.frontend.handle.utils import get_size as _get_size
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.types import BitType, FloatType, ObservableType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value
from qamomile.circuit.transpiler.errors import FrontendTransformError

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

P = ParamSpec("P")
R = TypeVar("R")


def _get_array_element_type(pt: Any) -> type | None:
    """Extract element type from an array type annotation."""
    if hasattr(pt, "__args__") and pt.__args__:
        return pt.__args__[0]
    return getattr(pt, "element_type", None)


def _promote_literal_to_handle(value: Any, expected_type: Any) -> Any:
    """Promote a Python literal to a scalar Handle based on the callee's annotation.

    When a sub-`@qkernel` declares a scalar parameter (`UInt`, `Float`,
    `Bit`), allow the call site to pass a raw Python literal and wrap it
    transparently. This mirrors the literal-acceptance pattern in built-in
    gate primitives such as ``qmc.rx`` (which take ``float | Float``) and
    saves users from manually calling ``qmc.uint(0)`` / ``qmc.float_(0.5)``
    / ``qmc.bit(True)`` at every call site.

    Promotion rules (applied only when ``expected_type`` is exactly one
    of the three scalar Handle classes):

    - ``UInt`` accepts ``int`` (excluding ``bool``).
    - ``Float`` accepts ``int`` (excluding ``bool``) or ``float`` —
      ``int → Float`` mirrors Python's natural numeric coercion (e.g.,
      ``math.cos(0)`` returns ``1.0``).
    - ``Bit`` accepts ``bool`` only. ``int`` values like ``0`` / ``1``
      are NOT promoted — symmetric to ``bool`` being excluded from the
      int-numeric paths.

    Anything else (already a Handle, an array-typed parameter, an
    incompatible literal) is returned unchanged so downstream
    validation produces the existing clear error.

    Args:
        value: The argument bound for this parameter. Can be any object.
        expected_type: The callee's declared annotation for this parameter.
            Typically a Handle class (``UInt``, ``Float``, ``Bit``,
            ``Vector[Qubit]``, ...) or a parameterized generic.

    Returns:
        Either a freshly-constructed scalar Handle wrapping the literal,
        or ``value`` unchanged if no promotion rule applies.

    Note:
        ``bool`` is a subclass of ``int`` in Python, so the rules above
        explicitly exclude bools from int/float promotion paths to avoid
        ``True`` silently becoming ``UInt(1)`` or ``Float(1.0)``. A
        ``bool`` is only promoted when the declared type is ``Bit``.
    """
    if isinstance(value, Handle):
        return value
    is_bool = isinstance(value, bool)
    if expected_type is UInt:
        if isinstance(value, int) and not is_bool:
            return uint(value)
    elif expected_type is Float:
        if isinstance(value, float) or (isinstance(value, int) and not is_bool):
            return float_(float(value))
    elif expected_type is Bit:
        if is_bool:
            return bit(value)
    return value


def _handle_types_equal(a: Any, b: Any) -> bool:
    """Compare two Handle type annotations, including generic aliases."""
    a_cls = getattr(a, "__origin__", a)
    b_cls = getattr(b, "__origin__", b)
    if a_cls is not b_cls:
        return False
    return getattr(a, "__args__", ()) == getattr(b, "__args__", ())


def _match_output_to_input(
    out_type: Any,
    input_types_list: list[Any],
    claimed: list[bool],
) -> int | None:
    """Return the position of the first unclaimed input whose Handle type
    matches ``out_type``; mutating the caller's ``claimed`` list is the
    caller's responsibility."""
    for idx, in_type in enumerate(input_types_list):
        if claimed[idx]:
            continue
        if _handle_types_equal(in_type, out_type):
            return idx
    return None


def _get_quantum_param_names(input_types: dict[str, type]) -> set[str]:
    """Return parameter names whose types are quantum (Qubit, Vector[Qubit])."""
    quantum_names: set[str] = set()
    for name, ptype in input_types.items():
        if ptype is Qubit:
            quantum_names.add(name)
        elif is_array_type(ptype):
            elem = _get_array_element_type(ptype)
            if elem is Qubit:
                quantum_names.add(name)
    return quantum_names


class QKernel(Generic[P, R]):
    """Decorator class for Qamomile quantum kernels."""

    def __init__(self, func: Callable[P, R]) -> None:
        # Hold a function where AST transformation has replaced control flow (if/while) with builder function calls
        self.raw_func = func
        try:
            self.func = transform_control_flow(func)
        except SyntaxError:
            # Syntax violations (invalid loop targets, etc.) must not be
            # silenced — propagate so the user sees the error at decoration time.
            raise
        except NotImplementedError as e:
            raise FrontendTransformError(
                f"AST transformation failed for function '{func.__name__}': {e}"
            )

        # transform_control_flow's exec namespace binds `func.__name__` to
        # the raw AST-transformed DSL function.  If the user body contains
        # a self-reference (e.g. for a recursive kernel), letting that name
        # resolve to the DSL function bypasses __call__ entirely: argument
        # validation, affine-type consumption, and CallBlockOperation emission
        # are all skipped, and the call becomes a direct in-place trace that
        # re-enters the same body forever.  Rebinding to the QKernel so that
        # self-calls always go through __call__ — where __call__ accesses
        # self.block, which then detects in-flight construction — fixes this.
        self.func.__globals__[func.__name__] = self

        self.name = func.__name__
        self.signature = inspect.signature(func)

        # Resolve type hints to handle string annotations (from __future__ import annotations)
        try:
            func_globals = getattr(func, "__globals__", {})
            type_hints = get_type_hints(func, globalns=func_globals, localns=None)
        except Exception:
            # Fallback to raw annotations if get_type_hints fails
            type_hints = {}
            for param in self.signature.parameters.values():
                if param.annotation is not inspect.Parameter.empty:
                    type_hints[param.name] = param.annotation
            if self.signature.return_annotation is not inspect.Signature.empty:
                type_hints["return"] = self.signature.return_annotation

        # Check type annotations
        input_types: dict[str, type[Handle]] = {}
        for param in self.signature.parameters.values():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter '{param.name}' must have a type annotation")
            # Use resolved type hint instead of raw annotation
            input_types[param.name] = type_hints.get(param.name, param.annotation)

        if self.signature.return_annotation is inspect.Signature.empty:
            raise TypeError("Return type must have a type annotation")

        output_types: list[type[Handle]] = []
        # Use resolved return type hint instead of raw annotation
        return_type = type_hints.get("return", self.signature.return_annotation)
        # check return is tuple or single
        if getattr(return_type, "__origin__", None) is tuple:
            for ret_type in return_type.__args__:
                output_types.append(ret_type)
        else:
            output_types.append(return_type)

        self.input_types = input_types
        self.output_types = output_types

        # Lazy initialization for hierarchical Block
        self._block: Block | None = None
        self._block_building: bool = False
        # Reentry guard for :meth:`__call__`'s call-time specialization
        # path. While the specialized re-trace runs the kernel body,
        # any self-call must fall back to the cached ``self.block`` to
        # avoid unbounded re-tracing of self-recursive kernels.
        self._specializing: bool = False
        # CallBlockOperations emitted by self-recursive calls during the
        # build get their ``block`` reference back-patched to ``self._block``
        # once ``func_to_block`` returns.  See _finalize_pending_self_calls.
        self._pending_self_calls: list[CallBlockOperation] = []

        # AST-level quantum rebind analysis (deferred raise until build/block)
        quantum_params = _get_quantum_param_names(input_types)
        if quantum_params:
            self._rebind_violations = collect_quantum_rebind_violations(
                self.raw_func, quantum_params
            )
        else:
            self._rebind_violations = []

    def _check_rebind_violations(self) -> None:
        if not self._rebind_violations:
            return
        from qamomile.circuit.transpiler.errors import QubitRebindError

        v = self._rebind_violations[0]
        if v.func_name:
            pattern = f"{v.target_name} = {v.func_name}({v.source_name}, ...)"
            fix = (
                f"  - Use self-update: {v.target_name} = {v.func_name}({v.target_name}, ...)\n"
                f"  - Use a new variable: new_var = {v.func_name}({v.source_name}, ...)"
            )
        else:
            pattern = f"{v.target_name} = {v.source_name}"
            fix = (
                f"  - Use a new variable: new_var = {v.source_name}\n"
                f"  - Remove the assignment if '{v.target_name}' is no longer needed"
            )
        raise QubitRebindError(
            f"Forbidden quantum variable reassignment at line {v.lineno}: "
            f"'{pattern}' overwrites quantum variable '{v.target_name}' "
            f"with a value derived from different quantum variable "
            f"'{v.source_name}'.\n\nTo fix, either:\n{fix}",
            handle_name=v.target_name,
            operation_name="assignment_rebind",
        )

    @property
    def block(self) -> Block:
        """Compile the function to a hierarchical Block if not already compiled."""
        self._check_rebind_violations()
        if self._block is None:
            if self._block_building:
                # Re-entry from outside ``__call__``'s forward-ref branch.
                # The self-recursive path in ``__call__`` avoids touching
                # ``.block`` during its own build; hitting this branch
                # means someone bypassed that routing (e.g. direct
                # ``self._block.call`` from the body), which would cause
                # unbounded re-tracing.
                raise FrontendTransformError(
                    f"Self-recursive @qkernel '{self.name}' accessed "
                    f".block during its own build.  Self-calls in a "
                    f"@qkernel body must use the plain call syntax "
                    f"(`{self.name}(args)`); direct `.block` access "
                    f"from inside the body is not supported."
                )
            self._block_building = True
            try:
                # Use self.func (AST-transformed) so that qm.range() is
                # properly converted to for_loop() context manager
                self._block = func_to_block(self.func)
                self._finalize_pending_self_calls()
            finally:
                self._block_building = False
        return self._block

    def _emit_self_call_forward_ref(
        self,
        inputs_map: dict[str, Value],
    ) -> CallBlockOperation:
        """Emit a CallBlockOperation for a self-call during build.

        The enclosing ``Block`` does not yet exist, so ``op.block`` stays
        ``None`` until ``_finalize_pending_self_calls`` back-patches it
        after ``func_to_block`` returns.  Result Values are position-matched
        to input Values by Handle type so quantum ``logical_id`` continuity
        (affine typing) is preserved across the call.
        """
        label_args = list(self.signature.parameters)
        operands = [inputs_map[label] for label in label_args]
        input_types_list = [self.input_types[label] for label in label_args]

        claimed = [False] * len(operands)
        results: list[Value] = []
        for i, out_type in enumerate(self.output_types):
            matched_idx = _match_output_to_input(out_type, input_types_list, claimed)
            if matched_idx is not None:
                claimed[matched_idx] = True
                results.append(operands[matched_idx].next_version())
            else:
                ir_type = handle_type_map(out_type)
                if is_array_type(out_type):
                    raise FrontendTransformError(
                        f"Self-recursive @qkernel '{self.name}' has an "
                        f"array output at position {i} with no matching "
                        f"array input of the same type.  Forward-ref "
                        f"emission cannot synthesize a symbolic shape "
                        f"without a matching input; restructure the "
                        f"signature so the quantum register is both "
                        f"input and output, or remove the self-recursion."
                    )
                results.append(Value(type=ir_type, name=f"{self.name}_result_{i}"))

        op = CallBlockOperation(block=None, operands=operands, results=results)
        self._pending_self_calls.append(op)
        return op

    def _finalize_pending_self_calls(self) -> None:
        """Back-patch each forward-ref self-call's ``block`` reference.

        Operands and results stay as produced by
        ``_emit_self_call_forward_ref`` — the result Values there already
        use ``input.next_version()`` for position-matched quantum outputs,
        which preserves ``logical_id`` continuity through the chain of
        sequential self-calls.  Replacing them with values from
        ``self._block.call`` would collapse both calls' results onto the
        block's ``phi_output`` (which has a fresh ``logical_id``), breaking
        the qubit-map lookup during backend emit.
        """
        if not self._pending_self_calls:
            return
        assert self._block is not None

        for op in self._pending_self_calls:
            op.block = self._block

        self._pending_self_calls = []

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Executes the kernel.
        If called within a tracing context, it emits a CallBlockOperation.

        ``VectorView`` arguments flow through the same
        ``CallBlockOperation`` path as plain ``Vector`` arguments: the
        view's sliced ``ArrayValue`` (with its ``slice_of`` /
        ``slice_start`` / ``slice_step`` metadata) is a first-class IR
        Value, so it can be an operand of the call and be substituted
        into the callee's dummy input by ``InlinePass``.  The emit-time
        resolver walks the ``slice_of`` chain back to the root parent,
        so the callee transparently operates on the slice's qubit
        subset.  This also means self-recursive kernels with view
        arguments (butterfly / divide-and-conquer patterns) work out of
        the box — the forward-ref back-patching handles the recursion.
        """
        tracer = get_current_tracer()

        # Bind arguments to signature to handle positional/keyword mapping
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Auto-promote raw Python literals to scalar Handles (UInt/Float/Bit)
        # so that calls like ``ry_layer(q, thetas, 0)`` work the same as
        # ``ry_layer(q, thetas, qmc.uint(0))``. Mirrors the literal-acceptance
        # pattern used by built-in gate primitives.
        for arg_name, arg_value in list(bound_args.arguments.items()):
            expected_type = self.input_types.get(arg_name)
            if expected_type is not None:
                bound_args.arguments[arg_name] = _promote_literal_to_handle(
                    arg_value, expected_type
                )

        # Prepare inputs for the IR call (unwrap Handles to Values)
        inputs_map = {}
        # Track borrow provenance for input-derived quantum scalar handles.
        # After the call, return values with matching logical_id will have
        # their parent/indices restored so that borrow-return validation
        # in ArrayBase._return_element succeeds.
        provenance_map: dict[str, tuple[Any, tuple]] = {}
        # Collect scalar handles borrowed from arrays that are also arguments.
        # Their borrows must be released before the parent array is consumed,
        # since ArrayBase.consume() enforces that all borrows are returned.
        borrowed_scalars: list[tuple[Any, tuple]] = []
        for name, handle in bound_args.arguments.items():
            if not isinstance(handle, Handle):
                raise TypeError(
                    f"Argument '{name}' must be a Handle instance, got {type(handle)}"
                )
            # Record provenance before consume (which preserves parent)
            if (
                handle.parent is not None
                and not is_array_type(type(handle))
                and handle._should_enforce_linear()
            ):
                provenance_map[handle.value.logical_id] = (
                    handle.parent,
                    handle.indices,
                )
                borrowed_scalars.append((handle.parent, handle.indices))

        # Release borrows for scalar elements whose parent arrays are also
        # being passed to this call.  This allows the parent's consume()
        # (which validates all borrows returned) to succeed.
        array_args = {
            id(h) for h in bound_args.arguments.values() if is_array_type(type(h))
        }
        for parent, indices in borrowed_scalars:
            if id(parent) in array_args:
                key = parent._make_indices_key(indices)
                parent._borrowed_indices.pop(key, None)

        # Slice-view inputs need their bulk-borrow transferred onto the
        # call's result handles under strict-return.  Stash the
        # pre-consume metadata so the result-wrapping loop can pair
        # each sliced result ``ArrayValue`` with its originating input
        # view by ``(root_logical_id, slice_start_uuid, slice_step_uuid,
        # length_uuid)``.  A single root may carry multiple disjoint
        # views (e.g. ``q[0::2]`` and ``q[1::2]``), and even views that
        # share ``start`` / ``step`` may have different lengths
        # (``q[lo:hi]`` and ``q[lo:hi2]`` reuse the same ``lo`` /
        # ``step`` ``UInt`` Values but differ in their ``shape[0]``
        # Value), so the length uuid is needed to disambiguate.  The
        # block's ``call`` helper builds each pass-through result via
        # ``inputs[i].next_version()`` which preserves
        # ``slice_start`` / ``slice_step`` / ``shape`` as the same
        # underlying ``Value`` references — so the result-side uuids
        # match the input-side uuids exactly.
        from qamomile.circuit.frontend.handle.array import VectorView

        InputViewKey = tuple[str, str | None, str | None, str | None]
        input_view_metas: dict[InputViewKey, VectorView[Any]] = {}
        for name, handle in bound_args.arguments.items():
            if isinstance(handle, VectorView) and handle._should_enforce_linear():
                root_av = handle.value
                while root_av.slice_of is not None:
                    root_av = root_av.slice_of
                start_uuid = (
                    handle._slice_start.value.uuid if handle._slice_start else None
                )
                step_uuid = (
                    handle._slice_step.value.uuid if handle._slice_step else None
                )
                length_uuid = handle.value.shape[0].uuid if handle.value.shape else None
                input_view_metas[
                    (root_av.logical_id, start_uuid, step_uuid, length_uuid)
                ] = handle

        for name, handle in bound_args.arguments.items():
            if not isinstance(handle, Handle):
                continue
            # ``VectorView`` argument consumption is deferred to
            # ``_transfer_borrow_to`` after the call so the parent's
            # borrow record can be rebound straight onto the result
            # handle.  Everything else takes the regular ``consume``
            # path to enforce affine type.
            if isinstance(handle, VectorView) and handle._should_enforce_linear():
                inputs_map[name] = handle.value
                continue
            if handle._should_enforce_linear():
                handle = handle.consume(operation_name=f"QKernel[{self.name}]")
            inputs_map[name] = handle.value

        # Self-recursive call: the enclosing Block is under construction,
        # so emit a forward-ref CallBlockOperation that gets back-patched
        # once the build completes.  Skipping ``self.block`` here is what
        # breaks the otherwise-infinite re-trace loop.
        if self._block_building:
            call_op = self._emit_self_call_forward_ref(inputs_map)
        else:
            # When the call site has fully resolved either the classical
            # bindings or the ``Vector[Qubit]`` sizes, re-trace a
            # specialized sub-block instead of reusing the cached
            # symbolic ``self.block``. Shape-dependent stdlib helpers
            # (``qmc.qft`` / ``qmc.iqft`` / ``qmc.qpe``) silently no-op
            # when their input vector has a symbolic shape; specializing
            # here is what makes those helpers emit the correct gate
            # sequence when wrapped inside an outer kernel. The
            # ``_specializing`` guard breaks the re-trace loop for
            # self-recursive kernels (see ``_extract_calltime_specialization``).
            block_ir = None
            if not self._specializing:
                spec = self._extract_calltime_specialization(bound_args.arguments)
                if spec is not None:
                    sub_parameters, sub_bindings, sub_qubit_sizes = spec
                    self._specializing = True
                    try:
                        block_ir = self._build_specialized(
                            parameters=sub_parameters,
                            bindings=sub_bindings,
                            qubit_sizes=sub_qubit_sizes,
                        )
                    finally:
                        self._specializing = False
            if block_ir is None:
                block_ir = self.block

            # Create the Call operation
            call_op = block_ir.call(**inputs_map)

        # Add the operation to the current tracer
        tracer.add_operation(call_op)

        # Wrap the result Values back into Handles
        results = call_op.results
        if len(results) != len(self.output_types):
            raise RuntimeError(
                f"Mismatch in return values: expected {len(self.output_types)}, got {len(results)}"
            )

        wrapped_results: list[Any] = []
        for val, handle_type in zip(results, self.output_types):
            if is_array_type(handle_type):
                # Use _create_from_value to avoid __post_init__ side effects
                actual_class = cast(
                    type[ArrayBase[Any]],
                    getattr(handle_type, "__origin__", handle_type),
                )
                # Extract shape from ArrayValue if available
                assert isinstance(val, ArrayValue)
                if val.shape:
                    shape = tuple(UInt(value=dim_val) for dim_val in val.shape)
                else:
                    # Fallback: empty shape (will need runtime resolution)
                    shape = ()

                # If the result has slice metadata and we have a
                # matching input view, wrap as ``VectorView`` and
                # transfer the parent's bulk-borrow over.  Strict-
                # return then demands the caller eventually slice-
                # assign the wrapped result back to the parent.
                if val.slice_of is not None:
                    result_root_av = val
                    while result_root_av.slice_of is not None:
                        result_root_av = result_root_av.slice_of
                    result_start_uuid = (
                        val.slice_start.uuid if val.slice_start else None
                    )
                    result_step_uuid = val.slice_step.uuid if val.slice_step else None
                    result_length_uuid = val.shape[0].uuid if val.shape else None
                    meta_key = (
                        result_root_av.logical_id,
                        result_start_uuid,
                        result_step_uuid,
                        result_length_uuid,
                    )
                    in_view = input_view_metas.get(meta_key)
                    if in_view is not None and in_view._slice_parent is not None:
                        length = shape[0] if shape else val.shape[0]
                        new_view = VectorView._wrap_unregistered(
                            parent=in_view._slice_parent,
                            sliced_av=val,
                            length=length,
                            start_uint=in_view._slice_start,
                            step_uint=in_view._slice_step,
                        )
                        in_view._transfer_borrow_to(new_view, f"QKernel[{self.name}]")
                        wrapped_results.append(new_view)
                        continue

                wrapped_results.append(
                    actual_class._create_from_value(value=val, shape=shape)
                )
            else:
                # Instantiate the specific Handle type (Qubit, UInt, etc.)
                # Restore borrow provenance for input-derived quantum scalars
                if val.logical_id in provenance_map:
                    parent, indices = provenance_map[val.logical_id]
                    wrapped_results.append(
                        handle_type(value=val, parent=parent, indices=indices)
                    )
                else:
                    wrapped_results.append(handle_type(value=val))

        # Any ``VectorView`` input that did NOT get its borrow transferred
        # to a matching sliced result must still be consumed — the call
        # logically passed it to the callee, so leaving it live would
        # break the affine / strict-return contract (use-after-move).
        # Use the destructive "qkernel call (view dropped)" consume name
        # so the covered parent slots become consumed-slot markers; the
        # qubits are effectively spent inside the callee (e.g. measure /
        # expval kernels) and re-touching them in the caller is rejected
        # rather than silently working with a corrupted state.
        for in_view in input_view_metas.values():
            if not in_view._consumed:
                in_view.consume(operation_name="qkernel call (view dropped)")

        # Return tuple or single value to match Python function signature
        if len(wrapped_results) == 1:
            return cast(R, wrapped_results[0])
        else:
            return cast(R, tuple(wrapped_results))

    def _is_parameterizable_type(self, param_type: Any) -> bool:
        """Check if a type can be used as a symbolic parameter."""
        if param_type in (float, Float, int, UInt):
            return True
        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)
            return element_type in (float, Float, int, UInt)
        return False

    def _auto_detect_parameters(self, kwargs: dict[str, Any]) -> list[str]:
        """Auto-detect parameters: non-Qubit arguments without value or default."""
        detected: list[str] = []
        for name, param in self.signature.parameters.items():
            # Prefer the resolved type hint from ``self.input_types`` so that
            # ``from __future__ import annotations`` stringified annotations
            # (e.g. ``"qmc.Vector[qmc.Float]"``) are compared as real types.
            param_type = self.input_types.get(name, param.annotation)

            # Skip Qubit types
            if param_type is Qubit:
                continue
            if is_array_type(param_type):
                if hasattr(param_type, "__args__") and param_type.__args__:
                    elem = param_type.__args__[0]
                else:
                    elem = getattr(param_type, "element_type", None)
                if elem is Qubit:
                    continue

            # Skip if value provided in kwargs
            if name in kwargs:
                continue

            # Skip if has default value
            if param.default is not inspect.Parameter.empty:
                continue

            # Auto-detect if parameterizable type
            if self._is_parameterizable_type(param_type):
                detected.append(name)

        return detected

    def _validate_parameters(self, parameters: list[str]) -> None:
        """Validate that parameter names exist and have valid types."""
        for name in parameters:
            if name not in self.input_types:
                raise ValueError(f"Unknown parameter: '{name}'")

            param_type = self.input_types[name]

            if not self._is_parameterizable_type(param_type):
                raise TypeError(
                    f"Parameter '{name}' has type {param_type}, "
                    f"but only float, int, UInt, and their arrays can be parameters"
                )

    def _validate_kwargs(self, parameters: list[str], kwargs: dict[str, Any]) -> None:
        """Validate that all non-parameter, non-Qubit arguments are provided or have defaults."""

        # Check for unknown kwargs that don't match any declared parameter
        known_names = set(self.signature.parameters.keys())
        unknown = set(kwargs.keys()) - known_names
        if unknown:
            names = ", ".join(f"'{n}'" for n in sorted(unknown))
            raise ValueError(
                f"Unknown argument(s) {names} provided. "
                f"Known arguments are: {sorted(known_names)}"
            )

        for name, param in self.signature.parameters.items():
            if name in parameters:
                continue

            # Prefer the resolved type hint from ``self.input_types`` so that
            # ``from __future__ import annotations`` stringified annotations
            # are compared as real types.
            param_type = self.input_types.get(name, param.annotation)

            # Qubit types are created as dummy inputs
            if param_type is Qubit:
                continue
            if is_array_type(param_type):
                if hasattr(param_type, "__args__") and param_type.__args__:
                    element_type = param_type.__args__[0]
                else:
                    element_type = getattr(param_type, "element_type", None)
                if element_type is Qubit:
                    continue
                # Vector[Observable] may be left unbound (resolved at emit).
                if element_type is Observable:
                    continue

            # Observable types are provided via bindings
            if param_type is Observable:
                continue

            # Dict/Tuple types are created as dummy inputs (symbolic for visualization)
            if is_dict_type(param_type) or is_tuple_type(param_type):
                continue

            # Non-qubit, non-parameter, non-observable types must be in kwargs or have a default value
            if name not in kwargs:
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Argument '{name}' must be provided or have a default value "
                        f"(not a parameter, Qubit, or Observable type)"
                    )

    def _create_parameter_input(self, param_type: Any, name: str) -> Handle:
        """Create a Handle for a parameter (unbound value)."""
        if param_type in (float, Float):
            value = Value(type=FloatType(), name=name).with_parameter(name)
            return Float(value=value)

        if param_type is Observable:
            value = Value(type=ObservableType(), name=name).with_parameter(name)
            return Observable(value=value)

        if param_type in (int, UInt):
            value = Value(type=UIntType(), name=name).with_parameter(name)
            return UInt(value=value)

        if is_array_type(param_type):
            # Restrict parameter arrays to scalar element types (Float/UInt)
            # and Observable. Qubit/Bit arrays are handled through other paths
            # (qubit_sizes etc.).
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)
            if element_type not in (Float, float, UInt, int, Observable):
                raise TypeError(
                    f"Array parameter must have Float, UInt, or Observable "
                    f"element type, got {element_type}"
                )
            if element_type is Observable and (
                getattr(param_type, "__origin__", param_type) is not Vector
            ):
                raise TypeError(
                    f"Only Vector[Observable] is supported; got {param_type}"
                )

            # Delegate to create_dummy_input so that parameter arrays get the
            # same symbolic shape treatment as inner-kernel arrays. This is
            # required for `arr.shape[i]` to return a usable symbolic Value at
            # the top level. emit_init=False because Float/UInt arrays never
            # emit QInitOperation regardless.
            return create_dummy_input(param_type, name, emit_init=False)

        raise TypeError(f"Cannot create parameter for type {param_type}")

    def _extract_return_names(self) -> list[str] | None:
        """Extract return variable names from AST.

        Returns:
            List of variable names from return statement, or None if not found.
        """
        source = inspect.getsource(self.raw_func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        return_value = stmt.value

                        if isinstance(return_value, ast.Tuple):
                            names = []
                            for elt in return_value.elts:
                                if isinstance(elt, ast.Name):
                                    names.append(elt.id)
                                elif isinstance(elt, ast.Subscript):
                                    names.append(ast.unparse(elt))
                                else:
                                    names.append(ast.unparse(elt))
                            return names

                        elif isinstance(return_value, ast.Name):
                            return [return_value.id]

                        elif isinstance(return_value, ast.Subscript):
                            return [ast.unparse(return_value)]

                        else:
                            return [ast.unparse(return_value)]

        return None

    def _create_bound_input(self, param_type: Any, name: str, value: Any) -> Handle:
        """Create a Handle for a bound (concrete) value."""
        # Scalar float
        if param_type in (float, Float):
            return Float(
                value=Value(
                    type=FloatType(),
                    name=name,
                ).with_const(float(value)),
                init_value=float(value),
            )

        # Scalar bool/Bit. Placed before the int/UInt branch for explicit
        # intent; exact-type matching via ``param_type in (...)`` means bool
        # would not fall into the int branch regardless of ordering, but
        # keeping bool first stays future-proof if matching ever changes to
        # isinstance/issubclass semantics.
        if param_type in (bool, Bit):
            return Bit(
                value=Value(
                    type=BitType(),
                    name=name,
                ).with_const(bool(value)),
                init_value=bool(value),
            )

        # Scalar int/UInt
        if param_type in (int, UInt):
            return UInt(
                value=Value(
                    type=UIntType(),
                    name=name,
                ).with_const(int(value)),
                init_value=int(value),
            )

        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)

            # Determine IR type and extract shape/data. Observable is an
            # arbitrary Python object so we avoid numpy (dtype=object arrays
            # are fragile); Float/UInt go through numpy for multi-dim support.
            if element_type in (Float, float):
                ir_element_type = FloatType()
                arr = np.asarray(value)
                shape = arr.shape
                const_data: Any = arr.tolist()
            elif element_type in (UInt, int):
                ir_element_type = UIntType()
                arr = np.asarray(value)
                shape = arr.shape
                const_data = arr.tolist()
            elif element_type is Observable:
                if getattr(param_type, "__origin__", param_type) is not Vector:
                    raise TypeError(
                        f"Only Vector[Observable] bindings are supported; "
                        f"got {param_type}"
                    )
                if isinstance(value, np.ndarray) and value.ndim != 1:
                    raise TypeError(
                        f"Vector[Observable] binding '{name}' must be 1-D; "
                        f"got ndarray with shape {value.shape}."
                    )
                items = list(value)
                for i, item in enumerate(items):
                    if isinstance(item, (list, tuple, np.ndarray)):
                        raise TypeError(
                            f"Vector[Observable] binding '{name}' must be a "
                            f"flat sequence of Hamiltonians; element {i} is "
                            f"{type(item).__name__}."
                        )
                ir_element_type = ObservableType()
                shape = (len(items),)
                const_data = items
            else:
                raise TypeError(
                    f"Unsupported element type for array binding: {element_type}"
                )

            shape_values = tuple(
                Value(type=UIntType(), name=f"dim_{i}").with_const(dim)
                for i, dim in enumerate(shape)
            )

            array_value = ArrayValue(
                type=ir_element_type,
                name=name,
                shape=shape_values,
            ).with_array_runtime_metadata(const_array=const_data)

            # Create instance without calling __init__
            # For generic aliases, use __origin__ to get the actual class
            actual_class = getattr(param_type, "__origin__", param_type)
            instance = object.__new__(actual_class)
            instance.value = array_value
            instance._shape = shape
            instance._borrowed_indices = {}
            instance.parent = None
            instance.indices = ()
            instance.name = name
            instance.id = str(id(instance))
            instance._consumed = False
            instance.element_type = element_type
            return instance

        # Dict binding - store dict data for transpile-time unrolling
        if is_dict_type(param_type):
            # Create DictValue with parameter binding
            # The actual dict data is stored and used during emit pass unrolling
            dict_value = (
                DictValue(
                    name=name,
                    entries=(),  # Empty - data stored in metadata
                )
                .with_parameter(name)
                .with_dict_runtime_metadata(value)
            )
            dict_handle = Dict(value=dict_value, _entries=[])
            if hasattr(param_type, "__args__") and param_type.__args__:
                dict_handle._key_type = param_type.__args__[0]
            return dict_handle

        raise TypeError(f"Cannot create bound value for type {param_type}")

    def estimate_resources(
        self,
        *,
        bindings: dict[str, Any] | None = None,
    ) -> ResourceEstimate:
        """Estimate all resources for this kernel's circuit.

        Convenience method that delegates to the module-level
        ``estimate_resources`` function, eliminating the need to
        access ``.block`` directly.

        Args:
            bindings: Optional concrete parameter bindings (scalars and dicts).
                      Dict values trigger ``|key|`` cardinality substitution.

        Returns:
            ResourceEstimate with qubits, gates, and parameters.

        Example:
            >>> @qm.qkernel
            ... def bell() -> qm.Vector[qm.Qubit]:
            ...     q = qm.qubit_array(2)
            ...     q[0] = qm.h(q[0])
            ...     q[0], q[1] = qm.cx(q[0], q[1])
            ...     return q
            >>> est = bell.estimate_resources()
            >>> print(est.qubits)  # 2
        """
        from qamomile.circuit.estimator.resource_estimator import (
            estimate_resources,
        )

        return estimate_resources(self.block, bindings=bindings)

    def _extract_calltime_specialization(
        self,
        arguments: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any], dict[str, int]] | None:
        """Extract call-time specialization data for a sub-kernel call.

        Inspects each argument bound at the call site and decides whether
        the sub-kernel body can be productively re-traced with concrete
        information (constant classical values and / or known ``Vector
        [Qubit]`` sizes). The triple returned mirrors the input contract
        of :meth:`_create_traced_block` and is consumed by
        :meth:`_build_specialized`.

        The motivation is to make shape-dependent stdlib helpers
        (``qmc.qft`` / ``qmc.iqft`` / ``qmc.qpe``) emit the correct gate
        sequence when invoked through a sub-kernel call. Those helpers
        silently fall back to a no-op when ``get_size`` of their input
        register fails (because the shape is still symbolic in a
        cached, never-specialized ``self.block``), so the outer
        composition layer must re-trace the callee once the relevant
        shapes are known.

        Args:
            arguments (dict[str, Any]): Pre-consume mapping from
                parameter name to the caller's frontend
                :class:`Handle`, as produced by
                ``signature.bind(...).arguments`` in :meth:`__call__`.

        Returns:
            tuple[list[str], dict[str, Any], dict[str, int]] | None:
                On success, a triple
                ``(parameters, bindings, qubit_sizes)`` where
                ``parameters`` lists classical argument names that
                remain symbolic at the call site, ``bindings`` maps
                classical arguments to their compile-time-known Python
                values, and ``qubit_sizes`` maps ``Vector[Qubit]``
                argument names to their resolved first-axis sizes.
                Arguments that cannot enter any of the three buckets
                (scalar ``Qubit``, scalar ``Observable``, unbound
                ``Vector[Observable]``, unbound ``Dict`` parameters,
                ``Tuple`` parameters, and non-parameterizable
                classical types like ``Bit`` / ``bool`` /
                ``Vector[Bit]`` / ``Vector[bool]`` without a
                compile-time constant) are deliberately left out so
                that :meth:`_create_traced_block` falls back to its
                standard ``create_dummy_input`` path for those argument
                positions; specialization of the rest of the call
                proceeds normally and inline-time substitution
                supplies the caller's actual Value. Returns ``None``
                only when no specialization is beneficial — i.e. the
                call adds no new compile-time information over the
                cached symbolic block. In the ``None`` case
                ``__call__`` falls back to ``self.block``.

        Raises:
            NotImplementedError: If ``arguments`` contains a quantum
                array argument that is not a 1-D ``Vector[Qubit]``
                (e.g., ``Matrix[Qubit]`` / ``Tensor[Qubit]``).
                Higher-rank quantum-array specialization is not yet
                implemented, and silently falling back to the cached
                symbolic block would lose shape-dependent stdlib gates
                applied inside the callee — surfacing the limitation
                as an exception prevents that.
        """
        parameters: list[str] = []
        bindings: dict[str, Any] = {}
        qubit_sizes: dict[str, int] = {}

        for name, param in self.signature.parameters.items():
            param_type = self.input_types.get(name, param.annotation)
            handle = arguments.get(name)
            # ``QKernel.__call__`` rejects non-``Handle`` arguments before
            # this method is reached, so an offending ``handle`` here is
            # a programmer error in the upstream call site, not a
            # normal specialization-abort condition. Use ``assert``
            # (Section L of CLAUDE.md: ``assert`` for internal
            # invariants).
            assert isinstance(handle, Handle), (
                f"Internal invariant violated: argument {name!r} should "
                f"already be a Handle by the time "
                f"_extract_calltime_specialization runs (upstream check "
                f"at __call__)."
            )

            # Scalar Qubit: no specialization-relevant info to extract;
            # ``_create_traced_block`` will fall through to the regular
            # ``create_dummy_input`` path, which yields a standalone
            # symbolic Qubit dummy. The caller's Value is bound to that
            # dummy at inline time, so behavior matches the symbolic
            # block.
            if param_type is Qubit:
                continue

            # Scalar Observable and unbound ``Vector[Observable]``:
            # ``_create_traced_block`` already auto-tracks these as
            # symbolic parameters regardless of the ``parameters`` list,
            # so we let them flow through untouched. A *bound*
            # ``Vector[Observable]`` (the caller passed concrete
            # observables) falls into the classical-array branch below
            # and gets baked in via ``bindings``.
            if param_type is Observable:
                continue
            if (
                is_array_type(param_type)
                and _get_array_element_type(param_type) is Observable
            ):
                const_array = handle.value.get_const_array()
                if const_array is not None:
                    bindings[name] = const_array
                continue

            # Quantum array. ``Vector[Qubit]`` is the fully supported
            # case: we extract the first-axis size via ``get_size`` and
            # carry it in ``qubit_sizes`` so the callee re-traces with
            # a concrete shape. ``Matrix[Qubit]`` / ``Tensor[Qubit]``
            # are higher-rank registers; we do not yet have the
            # multi-dim shape extraction nor the ``create_dummy_input
            # (shape=...)`` plumbing for them, so passing one through a
            # nested call would mean shape-dependent stdlib stays
            # silently no-op'd on those qubits. Surface that as an
            # explicit ``NotImplementedError`` rather than letting the
            # call silently produce a wrong-unitary circuit.
            if (
                is_array_type(param_type)
                and _get_array_element_type(param_type) is Qubit
            ):
                if getattr(param_type, "__origin__", param_type) is not Vector:
                    raise NotImplementedError(
                        f"Nested @qkernel call of '{self.name}' received "
                        f"argument {name!r} of type {param_type!r}. "
                        f"Call-time specialization currently supports only "
                        f"``Vector[Qubit]`` for quantum-array inputs; "
                        f"``Matrix[Qubit]`` / ``Tensor[Qubit]`` are not "
                        f"yet supported and would silently lose "
                        f"shape-dependent stdlib gates (qft, iqft, qpe) "
                        f"applied inside the callee. Reshape the call "
                        f"site so the callee receives a 1-D "
                        f"``Vector[Qubit]`` view, or file an issue if "
                        f"you need higher-rank specialization."
                    )
                # ``_get_size`` is declared over ``Vector``; the two
                # checks above narrow ``handle`` to a ``Vector[Qubit]``,
                # but the type system cannot see that — cast for the
                # call.
                try:
                    size = _get_size(cast(Vector[Qubit], handle))
                except ValueError:
                    continue
                qubit_sizes[name] = size
                continue

            # Classical scalar. Bind the compile-time constant when
            # available. If not, list the name in ``parameters`` only
            # when the type is parameterizable
            # (``_validate_parameters`` admits ``UInt`` / ``Float`` /
            # ``int`` / ``float`` and their arrays). For
            # non-parameterizable types like ``Bit`` / ``bool``, neither
            # bucket fits — but we ``continue`` past the argument
            # rather than aborting the whole specialization:
            # ``_create_traced_block``'s fall-through
            # ``create_dummy_input`` path creates a symbolic dummy
            # (matching the cached ``self.block``) and inline-time
            # substitution supplies the caller's actual Value. The
            # rest of the call still gets specialized.
            if param_type in (int, UInt, float, Float, bool, Bit):
                const_value = handle.value.get_const()
                if const_value is not None:
                    bindings[name] = const_value
                elif self._is_parameterizable_type(param_type):
                    parameters.append(name)
                continue

            # Classical array. Same approach as scalars: bind the const
            # array when available, list as a runtime parameter when
            # parameterizable, or ``continue`` and let
            # ``_create_traced_block`` fall through for
            # ``Vector[Bit]`` / ``Vector[bool]``.
            if is_array_type(param_type):
                const_array = handle.value.get_const_array()
                if const_array is not None:
                    bindings[name] = const_array
                elif self._is_parameterizable_type(param_type):
                    parameters.append(name)
                continue

            # Dict. ``handle.value.parameter_name()`` is set both for
            # unbound dicts and for bound ones (see
            # ``_create_bound_input``), so it can't distinguish the two.
            # Use the presence of ``dict_runtime`` metadata instead — it
            # is set if and only if the caller bound a concrete dict
            # value, even if that dict is empty. When no metadata is
            # present, ``continue`` without entering any bucket so
            # ``_create_traced_block``'s fall-through ``create_dummy_input``
            # path produces a symbolic Dict dummy (the same shape the
            # cached ``self.block`` would carry). Specialization on
            # other arguments still proceeds — only this Dict stays
            # symbolic.
            if is_dict_type(param_type):
                if handle.value.metadata.dict_runtime is None:
                    continue
                bindings[name] = handle.value.get_bound_data()
                continue

            # ``Tuple``. We do not yet have a ``_create_bound_input``
            # branch for tuples and ``Tuple`` is not parameterizable,
            # so there is no bucket we can place a tuple argument into.
            # ``continue`` is still safe: ``_create_traced_block``'s
            # fall-through ``create_dummy_input`` path creates a
            # symbolic ``TupleValue`` dummy, matching the cached block.
            if is_tuple_type(param_type):
                continue

            # Anything else not modeled above: abort and let the cached
            # symbolic block path handle the call.
            return None

        # Skip specialization when it would not change the traced
        # body. Re-tracing with no new constants is wasted work and
        # would only obscure the cached ``self.block`` path.
        if not bindings and not qubit_sizes:
            return None

        return parameters, bindings, qubit_sizes

    def _build_specialized(
        self,
        *,
        parameters: list[str],
        bindings: dict[str, Any],
        qubit_sizes: dict[str, int],
    ) -> Block:
        """Trace a specialized sub-block for a call site.

        Wraps :meth:`_create_traced_block` with the validations that
        :meth:`build` performs (parameter-name validation, rebind-
        violation check, output-name extraction) and disables
        ``QInitOperation`` emission for ``qubit_sizes`` entries — the
        caller's outer ``CallBlockOperation`` supplies those qubits, so
        emitting an init here would double-allocate the register after
        inlining.

        Args:
            parameters (list[str]): Classical argument names that
                remain symbolic in the specialized block (typically
                because the call site itself sees them as runtime
                parameters of the outer kernel).
            bindings (dict[str, Any]): Concrete Python values for
                classical arguments, baked into the block at trace
                time.
            qubit_sizes (dict[str, int]): First-axis sizes for
                ``Vector[Qubit]`` arguments. Each entry is realized as
                a dummy input with a concrete shape (no
                ``QInitOperation``).

        Returns:
            Block: The specialized hierarchical block, ready to be the
                target of :meth:`Block.call` from the caller's tracer.
        """
        self._check_rebind_violations()
        self._validate_parameters(parameters)
        block = self._create_traced_block(
            parameters,
            bindings,
            qubit_sizes=qubit_sizes,
            emit_qubit_init=False,
            emit_return_op=True,
        )
        block.output_names = self._extract_return_names() or []
        return block

    def _create_traced_block(
        self,
        parameters: list[str],
        kwargs: dict[str, Any],
        qubit_sizes: dict[str, int] | None = None,
        *,
        emit_qubit_init: bool = True,
        emit_return_op: bool = False,
    ) -> Block:
        """Trace the kernel and return a Block.

        This is the shared implementation for :meth:`build`,
        :meth:`_build_graph_with_qubit_arrays`, and the call-time
        specialization path used by :meth:`__call__`.

        Args:
            parameters (list[str]): Argument names to keep as unbound
                parameters.
            kwargs (dict[str, Any]): Concrete values for non-parameter
                arguments.
            qubit_sizes (dict[str, int] | None): Optional mapping from
                ``Vector[Qubit]`` parameter names to integer sizes. When
                provided, the corresponding arguments are created with
                concrete shape so shape-dependent stdlib helpers
                (``qft`` / ``iqft`` / ``qpe``) emit the expected gate
                sequence.
            emit_qubit_init (bool): When True (default), entries in
                ``qubit_sizes`` are realized via :func:`qubit_array`,
                which emits a ``QInitOperation`` — appropriate for
                top-level visualization / direct ``build()`` use. When
                False, the dummy quantum array is created without a
                ``QInitOperation`` because the call site will supply
                the qubits through a ``CallBlockOperation``; emitting
                an init in that case would double-allocate the
                register after inlining.
            emit_return_op (bool): When True, append a
                ``ReturnOperation`` to the traced operations so the
                inline pass can locate the cloned return Values when
                this block is used as the target of a
                ``CallBlockOperation`` (the call-time specialization
                path). The default ``False`` preserves the historical
                behavior of :meth:`build` (top-level entrypoint blocks
                consumed by ``transpile()``), where downstream code
                reads ``output_values`` directly and an explicit
                ``ReturnOperation`` is unnecessary.

        Returns:
            Block: The traced block, with ``label_args`` populated so
                the result is directly usable as the target of
                :meth:`Block.call`.
        """
        if qubit_sizes is None:
            qubit_sizes = {}

        tracer = Tracer()
        tracked_parameters: dict[str, Value] = {}

        with trace(tracer):
            dummy_inputs: dict[str, Handle] = {}

            for name, param in self.signature.parameters.items():
                # Prefer the resolved type hint from ``self.input_types`` so
                # that ``from __future__ import annotations`` stringified
                # annotations are handled correctly.
                param_type = self.input_types.get(name, param.annotation)

                # Scalar Observable is always a parameter; unbound
                # Vector[Observable] is too so its shape stays symbolic
                # (bound Vector[Observable] falls through to the kwargs
                # branch below, which resolves the shape from the value).
                is_scalar_observable = param_type is Observable
                is_unbound_observable_array = (
                    is_array_type(param_type)
                    and _get_array_element_type(param_type) is Observable
                    and name not in kwargs
                )
                if is_scalar_observable or is_unbound_observable_array:
                    handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in parameters:
                    # For qubit-array visualization, array-type parameters
                    # use a dummy input so that qubit wires are visible.
                    if qubit_sizes and is_array_type(param_type):
                        handle = create_dummy_input(param_type, name)
                    else:
                        handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in qubit_sizes:
                    if emit_qubit_init:
                        handle = qubit_array(qubit_sizes[name], name)
                    else:
                        # Call-time specialization: the caller's
                        # CallBlockOperation supplies the qubits, so we
                        # must not emit a QInitOperation here. The
                        # concrete shape lets stdlib helpers resolve
                        # ``get_size`` to a real integer.
                        handle = create_dummy_input(
                            param_type,
                            name,
                            emit_init=False,
                            shape=(qubit_sizes[name],),
                        )
                elif name in kwargs:
                    # Create bound value from kwargs
                    handle = self._create_bound_input(param_type, name, kwargs[name])
                elif param.default is not inspect.Parameter.empty:
                    # Use default value as bound value
                    handle = self._create_bound_input(param_type, name, param.default)
                else:
                    # Quantum-typed parameters (scalar ``Qubit`` and any
                    # ``Vector[Qubit]`` that did not get a concrete
                    # size). When ``emit_qubit_init`` is False (call-time
                    # specialization) the caller supplies the qubits
                    # via ``CallBlockOperation``, so we must not emit a
                    # ``QInitOperation`` — doing so would double-
                    # allocate the register after inlining.
                    handle = create_dummy_input(
                        param_type, name, emit_init=emit_qubit_init
                    )

                dummy_inputs[name] = handle

            # Execute the AST-transformed function with dummy inputs
            result = self.func(**dummy_inputs)

            # Extract output Values from the trace result. When
            # ``emit_return_op`` is set we additionally emit a
            # ReturnOperation so the inline pass can find the cloned
            # return Values when this block is later used as a
            # sub-call target (call-time specialization in
            # ``QKernel.__call__``). Without an explicit
            # ReturnOperation, ``_inline_call`` falls back to
            # ``block.output_values`` whose Values still carry the
            # ORIGINAL UUIDs — substitution then misses the caller-side
            # references and downstream ops (e.g. ``qmc.measure``)
            # lose their operand mapping.
            output_values: list[Value] = []
            if result is not None:
                if isinstance(result, tuple):
                    for r in result:
                        if hasattr(r, "value"):
                            output_values.append(r.value)
                else:
                    if hasattr(result, "value"):
                        output_values.append(result.value)
            if emit_return_op:
                tracer.add_operation(
                    ReturnOperation(operands=output_values, results=[])
                )

        # Extract input values for the Block. Outputs are populated
        # inside the trace context above so the optional
        # ReturnOperation can capture them.
        input_values = [h.value for h in dummy_inputs.values()]

        param_slots = build_param_slots(
            signature=self.signature,
            input_types=self.input_types,
            parameters=parameters,
            kwargs=kwargs,
            qubit_sizes=qubit_sizes,
            bind_defaults=True,
        )

        return Block(
            operations=tracer.operations,
            label_args=list(dummy_inputs.keys()),
            input_values=input_values,
            output_values=output_values,
            name=self.name,
            parameters=tracked_parameters,
            kind=BlockKind.TRACED,
            param_slots=param_slots,
        )

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Block:
        """Build a traced Block by tracing this kernel.

        Args:
            parameters: List of argument names to keep as unbound parameters.
                       - None (default): Auto-detect parameters (non-Qubit args without value/default)
                       - []: No parameters (all non-Qubit args must have value/default)
                       - ["name"]: Explicit parameter list
                       Only float, int, UInt, and their arrays are allowed as parameters.
            **kwargs: Concrete values for non-parameter arguments.

        Returns:
            Block: The traced block ready for transpilation, estimation,
                or visualization.

        Note:
            ``build()`` is a tracing/composition API and allows kernels with
            quantum inputs and/or quantum outputs. In contrast, kernels passed
            directly to ``Transpiler.transpile()`` / ``to_circuit()`` are
            treated as executable entrypoints and must expose classical
            inputs/outputs only.

        Raises:
            TypeError: If a non-parameterizable type is specified as parameter.
            ValueError: If required arguments are missing.

        Example:
            ```python
            @qm.qkernel
            def circuit(q: Qubit, theta: float) -> Qubit:
                q = qm.rx(q, theta)
                return q

            # Auto-detect theta as parameter
            block = circuit.build()

            # Explicit parameter list
            block = circuit.build(parameters=["theta"])

            # theta bound to concrete value
            block = circuit.build(theta=0.5)

            # Transpile with binding
            transpiler = QiskitTranspiler()
            result = transpiler.emit(graph, binding={"theta": 0.5})
            ```
        """
        self._check_rebind_violations()

        if parameters is None:
            parameters = self._auto_detect_parameters(kwargs)

        # Validate parameters argument
        self._validate_parameters(parameters)

        # Validate kwargs covers all non-parameter arguments
        self._validate_kwargs(parameters, kwargs)

        block = self._create_traced_block(parameters, kwargs)
        block.output_names = self._extract_return_names() or []
        return block

    def _has_qubit_array_params(self) -> bool:
        """Check if kernel has any Qubit array parameters (Vector[Qubit], etc.)."""
        for name, param in self.signature.parameters.items():
            pt = self.input_types.get(name, param.annotation)
            if is_array_type(pt) and _get_array_element_type(pt) is Qubit:
                return True
        return False

    def _build_graph_for_visualization(self, **kwargs: Any) -> Block:
        """Build a traced Block suitable for visualization.

        Handles Vector[Qubit] parameters by accepting integer sizes.
        Sets output_names from AST extraction.

        Args:
            **kwargs: Concrete values for kernel arguments. For Vector[Qubit]
                     parameters, pass an integer size.

        Returns:
            Block with output_names set.
        """
        if self._has_qubit_array_params():
            graph = self._build_graph_with_qubit_arrays(kwargs)
        else:
            graph = self.build(parameters=None, **kwargs)
        graph.output_names = self._extract_return_names() or []
        return graph

    def _build_graph_with_qubit_arrays(self, kwargs: dict[str, Any]) -> Block:
        """Build traced block with Vector[Qubit] support for visualization.

        Separates integer-valued kwargs for Qubit array parameters (used as
        array sizes via ``qubit_array()``) from other kwargs, then traces the
        kernel to produce a Block.
        """
        qubit_sizes: dict[str, int] = {}
        build_kwargs: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in self.signature.parameters:
                pt = self.input_types.get(
                    key, self.signature.parameters[key].annotation
                )
                if is_array_type(pt):
                    elem = _get_array_element_type(pt)
                    if elem is Qubit and isinstance(val, int):
                        qubit_sizes[key] = val
                        continue
            build_kwargs[key] = val

        missing = []
        for name, param in self.signature.parameters.items():
            pt = self.input_types.get(name, param.annotation)
            if is_array_type(pt):
                elem = _get_array_element_type(pt)
                if elem is Qubit and name not in qubit_sizes:
                    missing.append(name)
        if missing:
            names = ", ".join(f"'{n}'" for n in missing)
            raise ValueError(
                f"Vector[Qubit] parameter(s) {names} require an integer size "
                f"for visualization. Example: draw({missing[0]}=3)"
            )

        parameters = self._auto_detect_parameters(build_kwargs)
        self._validate_parameters(parameters)

        return self._create_traced_block(
            parameters, build_kwargs, qubit_sizes=qubit_sizes
        )

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the circuit using Matplotlib.

        This method builds the computation graph and creates a static visualization.
        Parameters are auto-detected: non-Qubit arguments without concrete values
        are shown as symbolic parameters.

        Args:
            inline: If True, expand CallBlockOperation contents (inlining).
                   If False (default), show CallBlockOperation as boxes.
            fold_loops: If True (default), display ForOperation as blocks instead of unrolling.
                       If False, expand loops and show all iterations.
            expand_composite: If True, expand CompositeGateOperation (QFT, IQFT, etc.).
                            If False (default), show as boxes. Independent of inline.
            inline_depth: Maximum nesting depth for inline expansion. None means
                         unlimited (default). 0 means no inlining, 1 means top-level
                         only, etc. Only affects CallBlock/ControlledU, not CompositeGate.
            **kwargs: Concrete values for arguments. Arguments not provided here
                     (and without defaults) will be shown as symbolic parameters.

        Returns:
            matplotlib.figure.Figure object.

        Raises:
            ImportError: If matplotlib is not installed.

        Example:
            ```python
            import qamomile.circuit as qm

            @qm.qkernel
            def inner(q: qm.Qubit) -> qm.Qubit:
                return qm.x(q)

            @qm.qkernel
            def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
                q = inner(q)
                q = qm.h(q)
                q = qm.rx(q, theta)
                return q

            # Draw with auto-detected symbolic parameter (theta)
            fig = circuit.draw()

            # Draw with bound parameter
            fig = circuit.draw(theta=0.5)

            # Draw with blocks as boxes (default)
            fig = circuit.draw()

            # Draw with blocks expanded (inlined)
            fig = circuit.draw(inline=True)

            # Draw with loops folded (shown as blocks)
            fig = circuit.draw(fold_loops=True)

            # Draw with composite gates expanded
            fig = circuit.draw(expand_composite=True)
            ```
        """
        from qamomile.circuit.visualization import MatplotlibDrawer

        return MatplotlibDrawer.draw_kernel(
            self,
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
            **kwargs,
        )


def qkernel(func: Callable[P, R]) -> QKernel[P, R]:
    """Decorator to define a Qamomile quantum kernel.

    Args:
        func: The function to decorate.

    Returns:
        An instance of QKernel wrapping the function.
    """
    return QKernel(func)
