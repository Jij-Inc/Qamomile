from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
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
    is_array_type,
    is_dict_type,
    is_tuple_type,
)
from qamomile.circuit.frontend.handle import Observable, Qubit
from qamomile.circuit.frontend.handle.containers import Dict
from qamomile.circuit.frontend.handle.primitives import Float, Handle, UInt
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.graph import Graph
from qamomile.circuit.ir.types import FloatType, ObservableType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value

if TYPE_CHECKING:
    from qamomile.circuit.estimator.resource_estimator import ResourceEstimate

P = ParamSpec("P")
R = TypeVar("R")


def _get_array_element_type(pt: Any) -> type | None:
    """Extract element type from an array type annotation."""
    if hasattr(pt, "__args__") and pt.__args__:
        return pt.__args__[0]
    return getattr(pt, "element_type", None)


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
        # AST変換を行い、制御構文(if/while)をビルダ関数呼び出しに置換した関数を保持する
        self.raw_func = func
        try:
            self.func = transform_control_flow(func)
        except NotImplementedError as e:
            # If transform fails, warn user and use original function
            warnings.warn(
                f"AST transformation failed for function '{func.__name__}': {e}. "
                f"Using original function. Note: qm.range() may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
            self.func = func

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

        # Lazy initialization for BlockValue
        self._block_value: BlockValue | None = None

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
    def block(self) -> BlockValue:
        """Compiles the function to a BlockValue (IR) if not already compiled."""
        self._check_rebind_violations()
        if self._block_value is None:
            # Use self.func (AST-transformed) so that qm.range() is properly
            # converted to for_loop() context manager
            self._block_value = func_to_block(self.func)
        return self._block_value

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

        # Ensure the block IR is compiled
        block_ir = self.block

        # Create the Call operation
        # BlockValue.call expects keyword arguments matching label_args
        call_op = block_ir.call(**inputs_map)

        # Add the operation to the current tracer
        tracer.add_operation(call_op)

        # Wrap the result Values back into Handles
        results = call_op.results
        if len(results) != len(self.output_types):
            raise RuntimeError(
                f"Mismatch in return values: expected {len(self.output_types)}, got {len(results)}"
            )

        wrapped_results = []
        for val, handle_type in zip(results, self.output_types):
            if is_array_type(handle_type):
                # Use _create_from_value to avoid __post_init__ side effects
                actual_class = getattr(handle_type, "__origin__", handle_type)
                # Extract shape from ArrayValue if available
                if isinstance(val, ArrayValue) and val.shape:
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
            value = Value(
                type=FloatType(),
                name=name,
                params={"parameter": name},
            )
            return Float(value=value)

        if param_type is Observable:
            value = Value(
                type=ObservableType(),
                name=name,
                params={"parameter": name},
            )
            return Observable(value=value)

        if param_type in (int, UInt):
            value = Value(
                type=UIntType(),
                name=name,
                params={"parameter": name},
            )
            return UInt(value=value)

        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)

            # Determine IR type for the element
            if element_type in (Float, float):
                ir_element_type = FloatType()
            elif element_type in (UInt, int):
                ir_element_type = UIntType()
            else:
                raise TypeError(
                    f"Array parameter must have Float or UInt element type, got {element_type}"
                )

            # Create placeholder ArrayValue (shape determined at runtime)
            array_value = ArrayValue(
                type=ir_element_type,
                name=name,
                shape=(),  # Empty - will be set at runtime
                params={"parameter": name},
            )

            # Create instance without calling __init__
            # For generic aliases, use __origin__ to get the actual class
            actual_class = getattr(param_type, "__origin__", param_type)
            instance = object.__new__(actual_class)
            instance.value = array_value
            instance._shape = ()
            instance._borrowed_indices = {}
            instance.parent = None
            instance.indices = ()
            instance.name = name
            instance.id = str(id(instance))
            instance._consumed = False
            instance.element_type = element_type
            return instance

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
                    params={"const": float(value)},
                ),
                init_value=float(value),
            )

        # Scalar int/UInt
        if param_type in (int, UInt):
            return UInt(
                value=Value(
                    type=UIntType(),
                    name=name,
                    params={"const": int(value)},
                ),
                init_value=int(value),
            )

        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)

            # Determine IR type for the element
            if element_type in (Float, float):
                ir_element_type = FloatType()
            elif element_type in (UInt, int):
                ir_element_type = UIntType()
            else:
                raise TypeError(
                    f"Unsupported element type for array binding: {element_type}"
                )

            # Convert to numpy array to handle multi-dimensional arrays
            arr = np.asarray(value)

            # Infer shape from the array
            shape = arr.shape
            shape_values = tuple(
                Value(type=UIntType(), name=f"dim_{i}", params={"const": dim})
                for i, dim in enumerate(shape)
            )

            # Store the bound values in params
            array_value = ArrayValue(
                type=ir_element_type,
                name=name,
                shape=shape_values,
                params={"const_array": arr.tolist()},
            )

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
            dict_value = DictValue(
                name=name,
                entries=[],  # Empty - data stored in params
                params={"parameter": name, "bound_data": value},
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

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Graph:
        """Build the computation graph by tracing this kernel.

        Args:
            parameters: List of argument names to keep as unbound parameters.
                       - None (default): Auto-detect parameters (non-Qubit args without value/default)
                       - []: No parameters (all non-Qubit args must have value/default)
                       - ["name"]: Explicit parameter list
                       Only float, int, UInt, and their arrays are allowed as parameters.
            **kwargs: Concrete values for non-parameter arguments.

        Returns:
            Graph: The traced computation graph ready for transpilation.

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
            graph = circuit.build()

            # Explicit parameter list
            graph = circuit.build(parameters=["theta"])

            # theta bound to concrete value
            graph = circuit.build(theta=0.5)

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

        tracer = Tracer()
        tracked_parameters: dict[str, Value] = {}

        with trace(tracer):
            dummy_inputs: dict[str, Handle] = {}

            for name, param in self.signature.parameters.items():
                param_type = param.annotation

                # Observable types are always treated as parameters (resolved during emit)
                if param_type is Observable:
                    handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in parameters:
                    # Create parameter placeholder
                    handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
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

        output_values = []
        if result is not None:
            if isinstance(result, tuple):
                for r in result:
                    if hasattr(r, "value"):
                        output_values.append(r.value)
            else:
                if hasattr(result, "value"):
                    output_values.append(result.value)

        return Graph(
            operations=tracer.operations,
            input_values=input_values,
            output_values=output_values,
            output_names=self._extract_return_names() or [],
            name=self.name,
            parameters=tracked_parameters,
        )

    def _has_qubit_array_params(self) -> bool:
        """Check if kernel has any Qubit array parameters (Vector[Qubit], etc.)."""
        for param in self.signature.parameters.values():
            pt = param.annotation
            if is_array_type(pt) and _get_array_element_type(pt) is Qubit:
                return True
        return False

    def _build_graph_for_visualization(self, **kwargs: Any) -> Graph:
        """Build a computation graph suitable for visualization.

        Handles Vector[Qubit] parameters by accepting integer sizes.
        Sets output_names from AST extraction.

        Args:
            **kwargs: Concrete values for kernel arguments. For Vector[Qubit]
                     parameters, pass an integer size.

        Returns:
            Graph with output_names set.
        """
        if self._has_qubit_array_params():
            graph = self._build_graph_with_qubit_arrays(kwargs)
        else:
            graph = self.build(parameters=None, **kwargs)
        graph.output_names = self._extract_return_names() or []
        return graph

    def _build_graph_with_qubit_arrays(self, kwargs: dict[str, Any]) -> Graph:
        """Build computation graph with Vector[Qubit] support for visualization.

        Separates integer-valued kwargs for Qubit array parameters (used as
        array sizes via ``qubit_array()``) from other kwargs, then traces the
        kernel to produce a Graph.
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

        tracer = Tracer()
        tracked_parameters: dict[str, Value] = {}

        with trace(tracer):
            dummy_inputs: dict[str, Any] = {}
            for name, param in self.signature.parameters.items():
                param_type = param.annotation
                if param_type is Observable:
                    handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in parameters:
                    if is_array_type(param_type):
                        handle = create_dummy_input(param_type, name)
                    else:
                        handle = self._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in qubit_sizes:
                    handle = qubit_array(qubit_sizes[name], name)
                elif name in build_kwargs:
                    handle = self._create_bound_input(
                        param_type, name, build_kwargs[name]
                    )
                elif param.default is not inspect.Parameter.empty:
                    handle = self._create_bound_input(param_type, name, param.default)
                else:
                    handle = create_dummy_input(param_type, name)
                dummy_inputs[name] = handle
            result = self.func(**dummy_inputs)

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

        return Graph(
            operations=tracer.operations,
            input_values=input_values,
            output_values=output_values,
            name=self.name,
            parameters=tracked_parameters,
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
