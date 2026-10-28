"""Execute the HUGR QInt decoder without allocating a large quantum state."""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.circuit.ir.types import BitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler import CompilationMetadata, CompiledProgram
from qamomile.circuit.transpiler.segments import ProgramABI

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")
pytest.importorskip("selene_sim")

from hugr import ops, tys
from hugr.build import Module
from hugr.package import Package
from hugr.std.int import int_t

from qamomile.hugr import HugrExecutable, HugrExecutor, SeleneExecutionOptions
from qamomile.hugr.lowerer import HugrTarget, _decode_qint_bits

pytestmark = pytest.mark.hugr


def _decoder_program(width: int) -> CompiledProgram:
    """Expose the production decoder as a classical bit-vector-to-UInt function.

    Args:
        width (int): Number of input bits, between zero and 64 inclusive.

    Returns:
        CompiledProgram: Validated HUGR package and its bit-vector/UInt ABI.

    Raises:
        EmitError: If the decoder width exceeds the HUGR integer carrier limit.
        HugrCliError: If native validation rejects the generated package.
    """
    module = Module()
    bit_types = [tys.Bool] * width
    main = module.define_function(
        "main", [tys.Tuple(*bit_types)], [int_t(6)], visibility="Public"
    )
    [packed] = main.inputs()
    bits = list(main.add_op(ops.UnpackTuple(bit_types), packed))
    main.set_outputs(_decode_qint_bits(bits, main))
    package = Package.from_bytes(Package([module.hugr], []).to_bytes())
    HugrTarget().validate(package)
    return CompiledProgram(
        package,
        ProgramABI(
            public_inputs={
                "bits": ArrayValue(
                    type=BitType(),
                    name="bits",
                    shape=(Value(type=UIntType(), name="width").with_const(width),),
                )
            },
            output_values=[Value(type=UIntType(), name="decoded")],
        ),
        CompilationMetadata("hugr", "program_graph"),
    )


@pytest.mark.parametrize(
    "width,number",
    [
        # Boundary vectors distinguish an empty carrier, bit order, Float precision
        # loss above 2**53, the signed high bit, and the full unsigned range.
        (0, 0),
        (1, 0),
        (1, 1),
        (3, 3),
        (64, 0),
        (64, 2**53 + 1),
        (64, 2**63),
        (64, 2**63 + 1),
        (64, 2**64 - 1),
    ],
)
def test_qint_decoder_executes_exact_unsigned_arithmetic(tmp_path, width, number):
    """Selene preserves every bit, including bit 63, through integer decoding."""
    compiled = _decoder_program(width)
    executable = HugrExecutable(compiled)
    bindings = {"bits": [bool(number & (1 << index)) for index in range(width)]}
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))

    actual = executable.run(executor, bindings=bindings).result()
    sampled = executable.sample(executor, bindings=bindings, shots=2).result()

    assert type(actual) is int
    assert actual == number
    assert sampled.results == [(number, 2)]
    assert type(sampled.results[0][0]) is int
    operations = [str(data.op) for _, data in compiled.artifact.modules[0].nodes()]
    assert not any("float" in operation.lower() for operation in operations)
    assert not any("quantum" in operation.lower() for operation in operations)


@pytest.mark.parametrize("width", [2, 5, 17, 64])
def test_qint_decoder_executes_seeded_bit_patterns(tmp_path, width: int) -> None:
    """Random mixed bit patterns match their exact LSB-first integer sum."""
    rng = np.random.default_rng(83 + width)
    bits = [bool(bit) for bit in rng.integers(0, 2, size=width)]
    expected = sum(int(bit) << index for index, bit in enumerate(bits))
    executable = HugrExecutable(_decoder_program(width))
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))

    actual = executable.run(executor, bindings={"bits": bits}).result()
    sampled = executable.sample(executor, bindings={"bits": bits}, shots=2).result()

    assert type(actual) is int
    assert actual == expected
    assert sampled.results == [(expected, 2)]
    assert sampled.shots == 2
    assert type(sampled.results[0][0]) is int
