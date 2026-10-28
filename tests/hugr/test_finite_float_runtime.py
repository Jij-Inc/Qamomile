"""Keep runtime Float inputs and provider outputs inside the finite f64 domain."""

from __future__ import annotations

import math
import sys
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from qamomile.circuit.ir.types import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler import (
    CompilationMetadata,
    CompiledProgram,
    CompletedExecutionHandle,
    ProgramABI,
)

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import tys
from hugr.build import Module
from hugr.package import Package
from hugr.std.float import FLOAT_T
from hugr.std.int import int_t

from qamomile.hugr import HugrExecutable, HugrExecutor
from qamomile.hugr._runtime import prepare_execution, validate_runtime_bindings

pytestmark = pytest.mark.hugr

_INVALID_FLOATS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(-float("inf"), id="negative-infinity"),
    pytest.param(np.float64("nan"), id="numpy-nan"),
    pytest.param(Decimal("NaN"), id="decimal-nan"),
    pytest.param(Decimal("sNaN"), id="decimal-signaling-nan"),
    pytest.param(Decimal("Infinity"), id="decimal-infinity"),
    pytest.param(Decimal("1e10000"), id="decimal-overflow"),
    pytest.param(10**1000, id="integer-overflow"),
    pytest.param(Fraction(10**1000, 3), id="fraction-overflow"),
]


@pytest.fixture(scope="module", params=["scalar", "nested"])
def float_abi(request):
    """Build independent scalar and tuple/dictionary/matrix identity carriers.

    Args:
        request (pytest.FixtureRequest): Scalar or nested carrier selection.

    Returns:
        SimpleNamespace: Compiled identity, value wrapper, leaf records, error
        path, and independently prepared execution schema.
    """
    if request.param == "scalar":
        value = Value(type=FloatType(), name="value")
        native = FLOAT_T

        def wrap(number):
            """Return the scalar binding unchanged.

            Args:
                number (Any): Candidate scalar value.

            Returns:
                Any: Unchanged scalar binding.
            """
            return number

        leaves = [0.5]
        path = "value"
    else:
        flag = Value(type=BitType(), name="flag")
        key = Value(type=UIntType(), name="key")
        matrix = ArrayValue(
            type=FloatType(),
            name="matrix",
            shape=tuple(
                Value(type=UIntType(), name=f"dim{index}").with_const(size)
                for index, size in enumerate((1, 2))
            ),
        )
        mapping = DictValue(name="mapping", entries=((key, matrix),))
        value = TupleValue(name="value", elements=(flag, mapping))
        native = tys.Tuple(
            tys.Bool, tys.Tuple(tys.Tuple(int_t(6), tys.Tuple(FLOAT_T, FLOAT_T)))
        )

        def wrap(number):
            """Place a candidate in a matrix within a dictionary and tuple.

            Args:
                number (Any): Candidate scalar value.

            Returns:
                tuple: Bit flag and dictionary containing the matrix binding.
            """
            return True, {7: ((0.125, number),)}

        leaves = [True, 7, 0.125, 0.5]
        path = "value[1][0][1][1]"
    module = Module()
    main = module.define_function("main", [native], [native], visibility="Public")
    main.set_outputs(*main.inputs())
    compiled = CompiledProgram(
        Package([module.hugr], []),
        ProgramABI(public_inputs={"value": value}, output_values=[value]),
        CompilationMetadata("hugr", "program_graph"),
    )
    prepared = prepare_execution(compiled, {"value": wrap(0.5)})
    return SimpleNamespace(
        compiled=compiled,
        wrap=wrap,
        leaves=leaves,
        path=path,
        prepared=prepared,
    )


@pytest.mark.parametrize("number", _INVALID_FLOATS)
@pytest.mark.parametrize("validate", [prepare_execution, validate_runtime_bindings])
def test_runtime_float_inputs_reject_nonfinite_values(float_abi, number, validate):
    """Both execution preparation and no-shot validation reject invalid Float leaves."""
    with pytest.raises(ValueError, match="must be finite") as error:
        validate(float_abi.compiled, {"value": float_abi.wrap(number)})
    assert repr(float_abi.path) in str(error.value)


@pytest.mark.parametrize("number", _INVALID_FLOATS)
@pytest.mark.parametrize("operation", ["run", "sample"])
def test_public_jobs_reject_nonfinite_provider_outputs(
    monkeypatch, float_abi, number, operation
):
    """Invalid tagged Float leaves fail before a run result or sample count escapes."""
    executor = HugrExecutor()
    leaves = [*float_abi.leaves[:-1], number]
    record = dict(zip(float_abi.prepared.output_tags, leaves, strict=True))
    monkeypatch.setattr(
        executor,
        "submit",
        lambda package, shots: CompletedExecutionHandle(
            [dict(record) for _ in range(shots)]
        ),
    )
    executable = HugrExecutable(float_abi.compiled)
    job = getattr(executable, operation)(
        executor, bindings={"value": float_abi.wrap(0.5)}, shots=3
    )
    with pytest.raises(ValueError, match="must be finite") as error:
        job.result()
    assert repr(float_abi.prepared.output_tags[-1]) in str(error.value)


@pytest.mark.parametrize(
    "number",
    [
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(sys.float_info.max, id="maximum-finite"),
        pytest.param(-sys.float_info.max, id="minimum-finite"),
        pytest.param(math.ulp(0.0), id="smallest-subnormal"),
        pytest.param(np.float32(0.25), id="numpy-float"),
        pytest.param(Decimal("0.125"), id="decimal"),
        pytest.param(Fraction(1, 2), id="fraction"),
        pytest.param(3, id="integer"),
    ],
)
def test_finite_float_inputs_and_outputs_preserve_values(
    monkeypatch, float_abi, number
):
    """Finite f64 boundaries and supported real scalars retain values and sample counts."""
    executor = HugrExecutor()
    leaves = [*float_abi.leaves[:-1], number]
    record = dict(zip(float_abi.prepared.output_tags, leaves, strict=True))
    monkeypatch.setattr(
        executor,
        "submit",
        lambda package, shots: CompletedExecutionHandle(
            [dict(record) for _ in range(shots)]
        ),
    )
    executable = HugrExecutable(float_abi.compiled)
    bindings = {"value": float_abi.wrap(number)}
    returned = executable.run(executor, bindings=bindings).result()
    sampled = executable.sample(executor, bindings=bindings, shots=3).result()
    assert sampled.shots == 3
    assert len(sampled.results) == 1
    sampled_value, count = sampled.results[0]
    assert count == 3
    for result in (returned, sampled_value):
        if len(leaves) == 1:
            normalized = result
        else:
            assert type(result) is tuple and len(result) == 2
            assert result[0] is True
            mapping = result[1]
            assert type(mapping) is dict and list(mapping) == [7]
            assert type(next(iter(mapping))) is int
            matrix = mapping[7]
            assert type(matrix) is tuple and len(matrix) == 1
            assert type(matrix[0]) is tuple and len(matrix[0]) == 2
            assert type(matrix[0][0]) is float
            np.testing.assert_allclose(matrix[0][0], 0.125, rtol=0, atol=0)
            normalized = matrix[0][1]
        assert type(normalized) is float
        np.testing.assert_allclose(normalized, float(number), rtol=0, atol=0)
        assert np.signbit(normalized) == np.signbit(float(number))
