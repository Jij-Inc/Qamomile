import pytest

import qamomile.core.circuit as qm_c
from qamomile.core.layer.parameter_context import ParameterContext


def test_initial_counter():
    context = ParameterContext()
    assert context.counter == 0

def test_get_next_parameter_default_symbol():
    context = ParameterContext()
    param = context.get_next_parameter()
    assert param.name == "θ_{0}"
    assert context.counter == 1

def test_get_next_parameter_custom_symbol():
    context = ParameterContext()
    param = context.get_next_parameter(symbol="α")
    assert param.name == "α_{0}"
    assert context.counter == 1

def test_get_next_parameter_multiple_calls():
    context = ParameterContext()
    param1 = context.get_next_parameter()
    param2 = context.get_next_parameter()
    assert param1.name == "θ_{0}"
    assert param2.name == "θ_{1}"
    assert context.counter == 2