import inspect
import warnings
from typing import Any, Callable, Generic, ParamSpec, Type, TypeVar, cast

import numpy as np

from qamomile.circuit.frontend.ast_transform import transform_control_flow
from qamomile.circuit.frontend.func_to_block import (
    create_dummy_input,
    func_to_block,
    is_array_type,
)
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.handle.primitives import Float, Handle, UInt
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace
from qamomile.circuit.ir.block_value import BlockValue
from qamomile.circuit.ir.graph import Graph
from qamomile.circuit.ir.types import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value

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

        # Check type annotations
        input_types: dict[str, Type[Handle]] = {}
        for param in self.signature.parameters.values():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter '{param.name}' must have a type annotation")
            input_types[param.name] = param.annotation

        if self.signature.return_annotation is inspect.Signature.empty:
            raise TypeError("Return type must have a type annotation")

        output_types: list[Type[Handle]] = []
        # check return is tuple or single
        if getattr(self.signature.return_annotation, "__origin__", None) is tuple:
            for ret_type in self.signature.return_annotation.__args__:
                output_types.append(ret_type)
        else:
            output_types.append(self.signature.return_annotation)

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
                    shape = tuple(
                        UInt(value=dim_val)
                        if not dim_val.is_constant()
                        else dim_val.params.get("const", 0)
                        for dim_val in val.shape
                    )
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

    def _is_float_or_array_float(self, param_type: Any) -> bool:
        """Check if type is float, Float, or Array[Float, ...]."""
        if param_type in (float, Float):
            return True
        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)
            return element_type in (float, Float)
        return False

    def _validate_parameters(self, parameters: list[str]) -> None:
        """Validate that parameter names exist and have valid types."""
        for name in parameters:
            if name not in self.input_types:
                raise ValueError(f"Unknown parameter: '{name}'")

            param_type = self.input_types[name]

            if not self._is_float_or_array_float(param_type):
                raise TypeError(
                    f"Parameter '{name}' has type {param_type}, "
                    f"but only float and Array[Float] can be parameters"
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

            # Non-qubit, non-parameter types must be in kwargs or have a default value
            if name not in kwargs:
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Argument '{name}' must be provided or have a default value "
                        f"(not a parameter or Qubit type)"
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

        if is_array_type(param_type):
            # For generic aliases like Vector[Float], get element type from __args__
            if hasattr(param_type, "__args__") and param_type.__args__:
                element_type = param_type.__args__[0]
            else:
                element_type = getattr(param_type, "element_type", None)

            if element_type not in (Float, float):
                raise TypeError("Array parameter must have Float element type")

            # Create placeholder ArrayValue (shape determined at runtime)
            array_value = ArrayValue(
                type=FloatType(),
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

        raise TypeError(f"Cannot create bound value for type {param_type}")

    def build(
        self,
        parameters: list[str] | None = None,
        **kwargs: Any,
    ) -> Graph:
        """Build the computation graph by tracing this kernel.

        Args:
            parameters: List of argument names to keep as unbound parameters.
                       Only float and Array[Float] types are allowed.
            **kwargs: Concrete values for non-parameter arguments.

        Returns:
            Graph: The traced computation graph ready for transpilation.

        Raises:
            TypeError: If a non-float type is specified as parameter.
            ValueError: If required arguments are missing.

        Example:
            ```python
            @qm.qkernel
            def circuit(q: Qubit, theta: float) -> Qubit:
                q = qm.rx(q, theta)
                return q

            # theta as parameter
            graph = circuit.build(parameters=["theta"])

            # theta bound to concrete value
            graph = circuit.build(theta=0.5)

            # Transpile with binding
            transpiler = QiskitTranspiler()
            result = transpiler.emit(graph, binding={"theta": 0.5})
            ```
        """
        if parameters is None:
            parameters = []

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

                if name in parameters:
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


def qkernel(func: Callable[P, R]) -> QKernel[P, R]:
    """Decorator to define a Qamomile quantum kernel.

    Args:
        func: The function to decorate.

    Returns:
        An instance of QKernel wrapping the function.
    """
    return QKernel(func)
