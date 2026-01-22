import contextlib
import typing

from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.value import Value
from qamomile.circuit.frontend.handle.primitives import Bit, Float, Handle, Qubit, UInt
from qamomile.circuit.frontend.tracer import Tracer, get_current_tracer, trace


class WhileLoop:
    pass


@contextlib.contextmanager
def while_loop(cond: typing.Callable) -> typing.Generator[WhileLoop, None, None]:
    """Builder function to create a while loop in Qamomile frontend.

    Args:
        cond (typing.Callable): A callable (lambda) that returns the loop condition expression.

    Yields:
        WhileLoop: A marker object for the while loop context.

    Example:
        ```python
        from qamomile.frontend.handle import Bit, UInt
        from qamomile.frontend.qkernel import QKernel
        from qamomile.frontend.operation import while_loop

        @QKernel
        def my_kernel(cond: Bit, x: UInt) -> UInt:
            with while_loop(lambda: cond):
                # Loop body operations
                x = x + UInt(1)
            return x
        ```
    """
    # 1. Get the PARENT tracer (the one active before entering the while loop)
    parent_tracer = get_current_tracer()

    # 2. Evaluate the condition lambda to get the condition expression
    # The lambda returns a Handle (e.g., result of i < n comparison)
    condition_result = cond()

    # 3. Create a new tracer for capturing body operations
    body_tracer = Tracer()

    # 4. Yield inside the body tracer context so body operations get captured
    with trace(body_tracer):
        yield WhileLoop()

    # 5. After the with block exits, body_tracer.operations contains the body
    # 6. Create WhileOperation with captured body operations
    while_op = WhileOperation(operations=body_tracer.operations)
    # Add condition result (the Handle representing the condition expression)
    while_op.operands.append(condition_result)

    # 7. Add the WhileOperation to the PARENT tracer (not a local one)
    parent_tracer.add_operation(while_op)


@contextlib.contextmanager
def for_loop(
    start, stop, step=1, var_name: str = "_loop_idx"
) -> typing.Generator[UInt, None, None]:
    """Builder function to create a for loop in Qamomile frontend.

    Args:
        start: Loop start value (can be Handle or int)
        stop: Loop stop value (can be Handle or int)
        step: Loop step value (default=1)
        var_name: Name of the loop variable (default="_loop_idx")

    Yields:
        UInt: The loop iteration variable (can be used as array index)

    Example:
        ```python
        @QKernel
        def my_kernel(qubits: Array[Qubit, Literal[3]]) -> Array[Qubit, Literal[3]]:
            for i in qm.range(3):
                qubits[i] = h(qubits[i])
            return qubits

        @QKernel
        def my_kernel2(qubits: Array[Qubit, Literal[5]]) -> Array[Qubit, Literal[5]]:
            for i in qm.range(1, 4):  # i = 1, 2, 3
                qubits[i] = h(qubits[i])
            return qubits
        ```
    """
    parent_tracer = get_current_tracer()
    body_tracer = Tracer()

    # Create a UInt to represent the loop variable (can be used as array index)
    # Use the provided var_name so transpiler can identify this loop variable
    loop_var = UInt(
        value=Value(type=UIntType(), name=var_name),
        init_value=0,  # Placeholder, actual value is symbolic during tracing
    )

    with trace(body_tracer):
        yield loop_var

    # ForOperationを作成
    # operands: [start, stop, step]
    for_op = ForOperation(loop_var=var_name, operations=body_tracer.operations)
    for_op.operands.append(_value_to_ir_value(start, "start"))
    for_op.operands.append(_value_to_ir_value(stop, "stop"))
    for_op.operands.append(_value_to_ir_value(step, "step"))

    parent_tracer.add_operation(for_op)


def _create_handle_from_value(value: Value, template_handle: Handle) -> Handle:
    """Create an appropriate Handle type from a Value using a template."""
    if isinstance(template_handle, Qubit):
        return Qubit(value=value)
    elif isinstance(template_handle, UInt):
        return UInt(value=value)
    elif isinstance(template_handle, Float):
        return Float(value=value)
    elif isinstance(template_handle, Bit):
        return Bit(value=value)
    else:
        # Fallback: return a generic Handle
        return Handle(value=value)


def _value_to_ir_value(val: typing.Any, name_prefix: str = "const") -> Value:
    """Convert a Python value or Handle to an IR Value.

    Args:
        val: Python primitive, Handle, or Value to convert
        name_prefix: Prefix for generated value name

    Returns:
        IR Value object

    Raises:
        TypeError: If value type is not supported
    """
    # Already a Value
    if isinstance(val, Value):
        return val

    # Extract Value from Handle
    if hasattr(val, "value") and isinstance(val.value, Value):
        return val.value

    # Convert primitive to Value
    if isinstance(val, (int, float, bool)):
        if isinstance(val, bool):
            return Value(type=BitType(), name=name_prefix, params={"const": val})
        elif isinstance(val, float):
            return Value(type=FloatType(), name=name_prefix, params={"const": val})
        else:  # int
            return Value(type=UIntType(), name=name_prefix, params={"const": val})

    # Unsupported type
    raise TypeError(f"Cannot convert {type(val)} to IR Value")


def _create_phi_for_values(
    condition_value: Value,
    true_val: typing.Any,
    false_val: typing.Any,
    if_operation: IfOperation,
) -> typing.Tuple[Value, Handle]:
    """Create a Phi operation for merging branch values.

    Args:
        condition_value: The condition Value (from if statement)
        true_val: Value from true branch (Handle, Value, or primitive)
        false_val: Value from false branch (Handle, Value, or primitive)
        if_operation: The IfOperation to add Phi result to

    Returns:
        Tuple of (phi_output_value, merged_handle)
    """
    # Convert both values to IR Values
    true_v = _value_to_ir_value(true_val, "true_const")
    false_v = _value_to_ir_value(false_val, "false_const")

    # Create Phi output value
    phi_output = Value(type=true_v.type, name=f"{true_v.name}_phi")

    # Create PhiOp (operation is tracked via if_operation.results)
    _phi_op = PhiOp(operands=[condition_value, true_v, false_v], results=[phi_output])
    if_operation.results.append(phi_output)

    # Create appropriate Handle type for the merged value
    merged_handle = _create_handle_from_value(phi_output, true_val)

    return phi_output, merged_handle


def _trace_branch(
    branch_func: typing.Callable,
    variables: list,
) -> typing.Tuple[Tracer, tuple]:
    """Trace a conditional branch and return its tracer and results.

    Args:
        branch_func: Function to execute for this branch
        variables: List of variables passed to the function

    Returns:
        Tuple of (tracer, normalized_result_tuple)
    """
    tracer = Tracer()
    with trace(tracer):
        result = branch_func(*variables)

    # Normalize result to tuple
    if not isinstance(result, tuple):
        result = (result,) if result is not None else ()

    return tracer, result


def emit_if(
    cond_func: typing.Callable,
    true_func: typing.Callable,
    false_func: typing.Callable,
    variables: list,
) -> typing.Any:
    """Builder function for if-else conditional with Phi function merging.

    This function is called from AST-transformed code. The AST transformer
    converts:
        if condition:
            true_body
        else:
            false_body

    Into:
        def _cond_N(vars): return condition
        def _body_N(vars): true_body; return vars
        def _body_N+1(vars): false_body; return vars
        result = emit_if(_cond_N, _body_N, _body_N+1, [var_list])

    Args:
        cond_func: Function returning the condition (Bit or bool-like Handle)
        true_func: Function executing true branch, returns updated variables
        false_func: Function executing false branch, returns updated variables
        variables: List of variables used in the branches

    Returns:
        Merged variable values after conditional execution (using Phi functions)

    Example:
        ```python
        @qkernel
        def my_kernel(q: Qubit) -> Qubit:
            result = measure(q)
            if result:
                q = z(q)
            return q
        ```
    """
    parent_tracer = get_current_tracer()

    # 1. Evaluate condition to get Bit/condition Handle
    condition_result = cond_func(*variables)
    condition_value = (
        condition_result.value
        if hasattr(condition_result, "value")
        else condition_result
    )

    # 2. Trace both branches
    true_tracer, true_result = _trace_branch(true_func, variables)
    false_tracer, false_result = _trace_branch(false_func, variables)

    # 3. Create IfOperation
    if_op = IfOperation(
        true_operations=true_tracer.operations,
        false_operations=false_tracer.operations,
    )
    if_op.operands.append(condition_value)

    # 4. Create Phi functions for each variable to merge branches
    merged_results = []
    for true_val, false_val in zip(true_result, false_result):
        try:
            phi_output, merged_handle = _create_phi_for_values(
                condition_value, true_val, false_val, if_op
            )
            merged_results.append(merged_handle)
        except TypeError:
            # Skip phi creation for non-Value types
            merged_results.append(true_val)

    # 5. Add IfOperation to parent tracer
    parent_tracer.add_operation(if_op)

    # 6. Return merged results
    if len(merged_results) == 0:
        return None
    elif len(merged_results) == 1:
        return merged_results[0]
    else:
        return tuple(merged_results)


def range(
    stop_or_start: "int | UInt",
    stop: "int | UInt | None" = None,
    step: "int | UInt" = 1,
) -> typing.Iterator[UInt]:
    """Symbolic range for use in qkernel for-loops.

    This function accepts UInt (symbolic) values and is transformed
    by the AST transformer into for_loop() calls.

    Usage:
        for i in qmc.range(n):  # 0 to n-1
        for i in qmc.range(start, stop):  # start to stop-1
        for i in qmc.range(start, stop, step):

    Note:
        This function is a placeholder - the actual looping is handled by
        the AST transformer which converts range() calls to for_loop().
    """
    # This is a dummy implementation - AST transformer replaces this with for_loop()
    # The function signature accepts UInt for type checking purposes
    return iter([])
