import ast
import inspect
import textwrap
import warnings
from typing import Any, Callable, Generic, ParamSpec, TypeVar, cast, get_type_hints

import numpy as np

from qamomile.circuit.frontend.ast_transform import transform_control_flow
from qamomile.circuit.frontend.func_to_block import (
    create_dummy_input,
    func_to_block,
    is_array_type,
    is_dict_type,
)
from qamomile.circuit.frontend.handle import Observable, Qubit
from qamomile.circuit.frontend.handle.containers import Dict
from qamomile.circuit.frontend.handle.primitives import Float, Handle, UInt
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.graph import Graph
from qamomile.circuit.ir.types import FloatType, ObservableType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, Value

P = ParamSpec("P")
R = TypeVar("R")


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

    @property
    def block(self) -> BlockValue:
        """Compiles the function to a BlockValue (IR) if not already compiled."""
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
        for name, handle in bound_args.arguments.items():
            if not isinstance(handle, Handle):
                raise TypeError(
                    f"Argument '{name}' must be a Handle instance, got {type(handle)}"
                )
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

        for name, param in self.signature.parameters.items():
            if name in parameters:
                continue

            param_type = param.annotation

            # Qubit types are created as dummy inputs
            if param_type is Qubit:
                continue
            if is_array_type(param_type):
                element_type = getattr(param_type, "element_type", None)
                if element_type is Qubit:
                    continue

            # Observable types are provided via bindings
            if param_type is Observable:
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
            List of variable names from return statement, or None if extraction fails.
        """
        try:
            source = inspect.getsource(self.raw_func)
            # Dedent the source to handle nested functions
            source = textwrap.dedent(source)
            tree = ast.parse(source)

            # Find the function definition (look for the first FunctionDef in the module)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    # Look for Return statements in the function body
                    for stmt in node.body:
                        if isinstance(stmt, ast.Return) and stmt.value is not None:
                            return_value = stmt.value

                            # Handle tuple return: return q0, q1, q2
                            if isinstance(return_value, ast.Tuple):
                                names = []
                                for elt in return_value.elts:
                                    if isinstance(elt, ast.Name):
                                        names.append(elt.id)
                                    elif isinstance(elt, ast.Subscript):
                                        # Handle subscript like qs[0]
                                        names.append(ast.unparse(elt))
                                    else:
                                        # For complex expressions, use unparsed form
                                        names.append(ast.unparse(elt))
                                return names

                            # Handle single return: return q
                            elif isinstance(return_value, ast.Name):
                                return [return_value.id]

                            # Handle subscript: return qs[0]
                            elif isinstance(return_value, ast.Subscript):
                                return [ast.unparse(return_value)]

                            # For other expressions, use unparsed form
                            else:
                                return [ast.unparse(return_value)]

            # No return statement found
            return None

        except Exception:
            # If AST parsing fails, return None (no right labels)
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
            return Dict(value=dict_value, _entries=[])

        raise TypeError(f"Cannot create bound value for type {param_type}")

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
            expand_composite: If True, expand CompositeGateOperation (QFT, QPE, etc.).
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

        graph = self.build(parameters=None, **kwargs)
        # Extract return variable names from AST and set them in the graph
        graph.output_names = self._extract_return_names() or []
        drawer = MatplotlibDrawer(graph)
        return drawer.draw(
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
        )


def qkernel(func: Callable[P, R]) -> QKernel[P, R]:
    """Decorator to define a Qamomile quantum kernel.

    Args:
        func: The function to decorate.

    Returns:
        An instance of QKernel wrapping the function.
    """
    return QKernel(func)
