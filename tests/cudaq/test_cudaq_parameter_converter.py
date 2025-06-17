# File: tests/qiskit/test_parameter_converter.py
import sys
import pytest

# Skip all tests if the platform is not Linux.
if sys.platform != "linux":
    pytest.skip("CUDA Quantum requires Linux", allow_module_level=True)


import cudaq

from qamomile.core.circuit import Parameter, BinaryOperator, Value, BinaryOpeKind
from qamomile.cudaq.parameter_converter import convert_parameter
from qamomile.cudaq.exceptions import QamomileCudaqTranspileError
from tests.mock import UnsupportedBinaryOpeKind, UnsupportedParam


def test_convert_parameter():
    """Convert a Qamomile parameter to a CUDA-Q parameter.

    Check if
    1. The conversion returns a CUDA-Q QuakeValue.
    2. The converted value matches the original parameter.
    """
    qamomile_param = Parameter("theta")
    _, cudaq_param = cudaq.make_kernel(float)
    param_map = {qamomile_param: cudaq_param}

    result = convert_parameter(qamomile_param, param_map)

    # 1. The conversion returns a CUDA-Q QuakeValue.
    assert isinstance(result, cudaq.QuakeValue)
    # 2. The converted value matches the original parameter.
    assert result == cudaq_param


def test_convert_value():
    """Convert a Qamomile Value to a float.

    Check if
    1. The conversion returns a float.
    2. The converted value matches the original value.
    """
    parameter_value_float = 2.5
    value = Value(parameter_value_float)
    param_map = {}

    result = convert_parameter(value, param_map)

    # 1. The conversion returns a float.
    assert isinstance(result, float)
    # 2. The converted value matches the original value.
    assert result == parameter_value_float


def test_convert_binary_operator_add():
    """Convert a Qamomile BinaryOperator with ADD operation to CUDA-Q.

    Check if
    1. The conversion returns a CUDA-Q QuakeValue.
    2. The converted value matches the expected CUDA-Q operation.
    """
    param1 = Parameter("theta")
    param2 = Parameter("phi")
    _, cudaq_param1, cudaq_param2 = cudaq.make_kernel(float, float)
    param_map = {param1: cudaq_param1, param2: cudaq_param2}

    qamomile_expr = BinaryOperator(param1, param2, BinaryOpeKind.ADD)
    result = convert_parameter(qamomile_expr, param_map)

    # 1. The conversion returns a CUDA-Q QuakeValue.
    assert isinstance(result, cudaq.QuakeValue)
    # 2. The converted value matches the expected CUDA-Q operation.
    #    Note that, cudaq_param1 + cudaq_param2 is not the same as result
    #    since apparently, cudaq stores the result of the operation in a different place every time.
    #    For example, "print(result)" shows something like Value(%0 = arith.addf %arg0, %arg1 : f64),
    #    meanwhile "print(cudaq_param1 + cudaq_param2)" shows something like Value(%1 = arith.addf %arg0, %arg1 : f64)
    #    Also, "arith.addf" is appaprently a way of representing addition in the Multi-Level Intermediate Representation (MLIR).
    assert r"arith.addf" in str(result)


def test_convert_binary_operator_mul():
    """Convert a Qamomile BinaryOperator with MUL operation to CUDA-Q.

    Check if
    1. The conversion returns a CUDA-Q QuakeValue.
    2. The converted value matches the expected CUDA-Q operation.
    """
    param = Parameter("theta")
    value = Value(2)
    _, cudaq_param = cudaq.make_kernel(float)
    param_map = {param: cudaq_param}

    qamomile_expr = BinaryOperator(param, value, BinaryOpeKind.MUL)
    result = convert_parameter(qamomile_expr, param_map)

    # 1. The conversion returns a CUDA-Q QuakeValue.
    assert isinstance(result, cudaq.QuakeValue)
    # 2. The converted value matches the expected CUDA-Q operation.
    #    Note that, cudaq_param * 2 is not the same as result
    #    since apparently, cudaq stores the result of the operation in a different place every time.
    #    For example, "print(result)" shows something like Value(%0 = arith.mulf %arg0, %cst : f64),
    #    meanwhile "print(cudaq_param1 * 2)" shows Value(%2 = arith.mulf %arg0, %1 : f64).
    #    Also, "arith.mulf" is appaprently a way of representing multiplication in the Multi-Level Intermediate Representation (MLIR).
    assert r"arith.mulf" in str(result)


def test_convert_binary_operator_div():
    """Convert a Qamomile BinaryOperator with DIV operation to CUDA-Q.

    Check if
    1. The conversion returns a CUDA-Q QuakeValue.
    2. The converted value matches the expected CUDA-Q operation.
    """
    param = Parameter("theta")
    value = Value(2)
    _, cudaq_param = cudaq.make_kernel(float)
    param_map = {param: cudaq_param}

    qamomile_expr = BinaryOperator(param, value, BinaryOpeKind.DIV)
    result = convert_parameter(qamomile_expr, param_map)

    # 1. The conversion returns a CUDA-Q QuakeValue.
    assert isinstance(result, cudaq.QuakeValue)
    # 2. The converted value matches the expected CUDA-Q operation.
    #    Note that, cudaq_param * 2 is not the same as result
    #    since apparently, cudaq stores the result of the operation in a different place every time.
    #    For example, "print(result)" shows something like Value(%0 = arith.divf %arg0, %cst : f64),
    #    meanwhile "print(cudaq_param1 * 2)" shows Value(%2 = arith.divf %arg0, %1 : f64).
    #    Also, "arith.mulf" is appaprently a way of representing multiplication in the Multi-Level Intermediate Representation (MLIR).
    assert r"arith.divf" in str(result)


def test_convert_all_binary_operators():
    """Convert a complex Qamomile BinaryOperator expression to CUDA-Q.
    Note that, CUDA-Q apparently does not show an intermediate computation.
    For example, if one runs "print(result)" in this test, it shows "Value(%2 = arith.divf %0, %1 : f64)".
    Also, if one runs "print((cudaq_param1 * cudaq_param2) / (cudaq_param1 + cudaq_param3))",
    then it shows "Value(%5 = arith.divf %3, %4 : f64)".
    Thus, we cannot check if the result is exactly what we need unless we actually run the kernel and check the output.

    Check if
    1.The conversion returns a CUDA-Q QuakeValue.
    """
    param1 = Parameter("theta")
    param2 = Parameter("phi")
    param3 = Parameter("delta")
    _, cudaq_param1, cudaq_param2, cudaq_param3 = cudaq.make_kernel(float, float, float)

    param_map = {param1: cudaq_param1, param2: cudaq_param2, param3: cudaq_param3}
    qamomile_expr = BinaryOperator(
        BinaryOperator(param1, param2, BinaryOpeKind.MUL),
        BinaryOperator(param1, param3, BinaryOpeKind.ADD),
        BinaryOpeKind.DIV,
    )

    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, cudaq.QuakeValue)


def test_unsupported_binary_operation():
    """Convert a Qamomile BinaryOperator with an unsupported operation to CUDA-Q.

    Check if
    1. The conversion raises a QamomileCudaqTranspileError.
    """
    param = Parameter("theta")
    _, cudaq_param = cudaq.make_kernel(float)
    param_map = {param: cudaq_param}

    unsupported_expr = BinaryOperator(param, param, UnsupportedBinaryOpeKind())

    # 1. The conversion raises a QamomileCudaqTranspileError.
    with pytest.raises(
        QamomileCudaqTranspileError, match="Unsupported binary operation"
    ):
        convert_parameter(unsupported_expr, param_map)


def test_unsupported_parameter_type():
    """Convert an unsupported parameter type to CUDA-Q.

    Check if
    1. The conversion raises a QamomileCudaqTranspileError.
    """

    unsupported_param = UnsupportedParam()
    param_map = {}

    # 1. The conversion raises a QamomileCudaqTranspileError.
    with pytest.raises(QamomileCudaqTranspileError, match="Unsupported parameter type"):
        convert_parameter(unsupported_param, param_map)
