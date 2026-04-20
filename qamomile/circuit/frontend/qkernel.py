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
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.func_to_block import (
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
from qamomile.circuit.frontend.handle.primitives import Float, Handle, UInt
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.types import FloatType, ObservableType, UIntType
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
        """
        tracer = get_current_tracer()

        # Bind arguments to signature to handle positional/keyword mapping
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

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

        for name, handle in bound_args.arguments.items():
            if not isinstance(handle, Handle):
                continue
            # Consume quantum handles to enforce affine type
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
            # Ensure the block IR is compiled
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
            param_type = param.annotation

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

            param_type = param.annotation

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

    def _create_traced_block(
        self,
        parameters: list[str],
        kwargs: dict[str, Any],
        qubit_sizes: dict[str, int] | None = None,
    ) -> Block:
        """Trace the kernel and return a Block.

        This is the shared implementation for :meth:`build` and
        :meth:`_build_graph_with_qubit_arrays`.

        Args:
            parameters: Argument names to keep as unbound parameters.
            kwargs: Concrete values for non-parameter arguments.
            qubit_sizes: Optional mapping from Vector[Qubit] parameter names
                to integer sizes.  When provided, the corresponding arguments
                are created via ``qubit_array()`` instead of the normal dummy
                input path.

        Returns:
            Block: The traced block.
        """
        if qubit_sizes is None:
            qubit_sizes = {}

        tracer = Tracer()
        tracked_parameters: dict[str, Value] = {}

        with trace(tracer):
            dummy_inputs: dict[str, Handle] = {}

            for name, param in self.signature.parameters.items():
                param_type = param.annotation

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
                    handle = qubit_array(qubit_sizes[name], name)
                elif name in kwargs:
                    # Create bound value from kwargs
                    handle = self._create_bound_input(param_type, name, kwargs[name])
                elif param.default is not inspect.Parameter.empty:
                    # Use default value as bound value
                    handle = self._create_bound_input(param_type, name, param.default)
                else:
                    # Create dummy input (for Qubit types)
                    handle = create_dummy_input(param_type, name)

                dummy_inputs[name] = handle

            # Execute the AST-transformed function with dummy inputs
            result = self.func(**dummy_inputs)

        # Extract input/output values
        input_values = [h.value for h in dummy_inputs.values()]

        output_values: list[Value] = []
        if result is not None:
            if isinstance(result, tuple):
                for r in result:
                    if hasattr(r, "value"):
                        output_values.append(r.value)
            else:
                if hasattr(result, "value"):
                    output_values.append(result.value)

        return Block(
            operations=tracer.operations,
            input_values=input_values,
            output_values=output_values,
            name=self.name,
            parameters=tracked_parameters,
            kind=BlockKind.TRACED,
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
        for param in self.signature.parameters.values():
            pt = param.annotation
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
                pt = self.signature.parameters[key].annotation
                if is_array_type(pt):
                    elem = _get_array_element_type(pt)
                    if elem is Qubit and isinstance(val, int):
                        qubit_sizes[key] = val
                        continue
            build_kwargs[key] = val

        missing = []
        for name, param in self.signature.parameters.items():
            pt = param.annotation
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
