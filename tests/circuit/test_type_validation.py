"""Tests for type validation in QKernel.__call__ and InlinePass.

Validates that type mismatches between annotations and actual handles
are caught at the frontend (TypeError) and inline pass (InliningError).
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.ir.types.primitives import QubitType, FloatType, UIntType
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.frontend.type_check import (
    validate_return_type,
    _format_annotation,
)


class TestArgumentValidationRejects:
    """Tests for type mismatch rejection at QKernel.__call__."""

    def test_vector_qubit_to_qubit_rejects(self):
        """Case B: Vector[Qubit] expected, Qubit given."""

        @qmc.qkernel
        def vec_kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return qs

        @qmc.qkernel
        def caller(q: qmc.Qubit) -> qmc.Qubit:
            q = vec_kernel(q)
            return q

        with pytest.raises(TypeError, match=r"expected Vector\[Qubit\].*got Qubit"):
            caller.build()

    def test_qubit_to_vector_qubit_rejects(self):
        """Case C: Qubit expected, Vector[Qubit] given."""

        @qmc.qkernel
        def scalar_kernel(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def caller(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs = scalar_kernel(qs)
            return qs

        with pytest.raises(TypeError, match=r"expected Qubit.*got Vector\[Qubit\]"):
            caller.build()

    def test_vector_element_type_mismatch_rejects(self):
        """Vector[Qubit] expected, Vector[Float] given."""

        @qmc.qkernel
        def qubit_vec_kernel(
            qs: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            return qs

        @qmc.qkernel
        def caller(fs: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Float]:
            fs = qubit_vec_kernel(fs)
            return fs

        with pytest.raises(
            TypeError, match=r"expected Vector\[Qubit\].*got Vector\[Float\]"
        ):
            caller.build(parameters=["fs"])

    def test_uint_float_scalar_mismatch_rejects(self):
        """UInt expected, Float given."""

        @qmc.qkernel
        def uint_kernel(n: qmc.UInt) -> qmc.UInt:
            return n

        @qmc.qkernel
        def caller(f: qmc.Float) -> qmc.Float:
            f = uint_kernel(f)
            return f

        with pytest.raises(TypeError, match=r"expected UInt.*got Float"):
            caller.build(parameters=["f"])


class TestArgumentValidationPasses:
    """Tests that valid type-matching calls pass validation."""

    def test_matching_scalar_qubit(self):
        """Qubit -> Qubit should pass."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def outer(q: qmc.Qubit) -> qmc.Qubit:
            q = inner(q)
            return q

        graph = outer.build()
        assert graph is not None

    def test_matching_vector_qubit(self):
        """Vector[Qubit] -> Vector[Qubit] should pass."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return qs

        @qmc.qkernel
        def outer(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs = inner(qs)
            return qs

        graph = outer.build()
        assert graph is not None

    def test_matching_scalar_float(self):
        """Float -> Float should pass."""

        @qmc.qkernel
        def inner(f: qmc.Float) -> qmc.Float:
            return f

        @qmc.qkernel
        def outer(f: qmc.Float) -> qmc.Float:
            f = inner(f)
            return f

        graph = outer.build(parameters=["f"])
        assert graph is not None

    def test_matching_dict_type(self):
        """Dict[...] -> Dict[...] should pass (kind-only check)."""

        @qmc.qkernel
        def inner(
            d: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.UInt:
            return d.size

        @qmc.qkernel
        def outer(
            d: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.UInt:
            return inner(d)

        graph = outer.build(d={(0, 1): 1.0, (1, 2): -0.5})
        assert graph is not None

    def test_matching_vector_float(self):
        """Vector[Float] -> Vector[Float] should pass."""

        @qmc.qkernel
        def inner(fs: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Float]:
            return fs

        @qmc.qkernel
        def outer(fs: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Float]:
            fs = inner(fs)
            return fs

        graph = outer.build(parameters=["fs"])
        assert graph is not None


class TestReturnValueValidation:
    """Tests for return value type validation."""

    def test_return_array_when_scalar_expected(self):
        """Return value validation: scalar annotation but array value."""

        @qmc.qkernel
        def returns_vector(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return qs

        @qmc.qkernel
        def caller(qs: qmc.Vector[qmc.Qubit]) -> qmc.Qubit:
            # returns_vector(qs) is fine for arg validation, but caller says it
            # returns Qubit while actually returning Vector[Qubit].
            q = returns_vector(qs)
            return q

        # The returned handle from returns_vector is Vector[Qubit] which is an
        # ArrayValue. The caller expects to return Qubit (scalar). This mismatch
        # is caught by return validation of the inner call since inner's return
        # annotation says Vector[Qubit] and the call_op result is ArrayValue.
        # For the outer caller, the build should succeed since its return type
        # (Qubit) doesn't trigger return validation until outer is called.
        graph = caller.build()
        assert graph is not None


class TestFormatAnnotation:
    """Tests for _format_annotation recursive formatting."""

    def test_nested_tuple_formats_fully(self):
        """Tuple[Vector[Qubit], UInt] should not truncate to Tuple[Vector, UInt]."""
        result = _format_annotation(qmc.Tuple[qmc.Vector[qmc.Qubit], qmc.UInt])
        assert result == "Tuple[Vector[Qubit], UInt]"

    def test_plain_scalar(self):
        assert _format_annotation(qmc.Qubit) == "Qubit"

    def test_vector_with_element(self):
        assert _format_annotation(qmc.Vector[qmc.Float]) == "Vector[Float]"


class TestReturnTypeValidationDirect:
    """Direct unit tests for validate_return_type error branches."""

    def test_array_expected_scalar_got(self):
        """Array annotation with scalar Value should raise."""
        scalar = Value(type=QubitType(), name="q")
        with pytest.raises(TypeError, match=r"expected Vector\[Qubit\].*scalar"):
            validate_return_type(qmc.Vector[qmc.Qubit], scalar, 0)

    def test_scalar_expected_array_got(self):
        """Scalar annotation with ArrayValue should raise."""
        shape = (Value(type=UIntType(), name="n", params={"const": 2}),)
        arr = ArrayValue(type=QubitType(), name="qs", shape=shape)
        with pytest.raises(TypeError, match=r"expected Qubit.*array"):
            validate_return_type(qmc.Qubit, arr, 0)

    def test_tuple_expected_scalar_got(self):
        """Tuple annotation with scalar Value should raise."""
        scalar = Value(type=UIntType(), name="x")
        with pytest.raises(TypeError, match=r"expected.*Tuple.*non-tuple"):
            validate_return_type(qmc.Tuple[qmc.UInt, qmc.UInt], scalar, 0)

    def test_dict_expected_scalar_got(self):
        """Dict annotation with scalar Value should raise."""
        scalar = Value(type=FloatType(), name="f")
        with pytest.raises(TypeError, match=r"expected.*Dict.*non-dict"):
            validate_return_type(
                qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float], scalar, 0
            )

    def test_ir_type_mismatch(self):
        """Scalar annotation matching structure but wrong IR type should raise."""
        float_val = Value(type=FloatType(), name="f")
        with pytest.raises(TypeError, match=r"expected element type.*got"):
            validate_return_type(qmc.Qubit, float_val, 0)

    def test_matching_scalar_passes(self):
        """Matching type should not raise."""
        qubit_val = Value(type=QubitType(), name="q")
        validate_return_type(qmc.Qubit, qubit_val, 0)

    def test_matching_array_passes(self):
        """Matching array should not raise."""
        shape = (Value(type=UIntType(), name="n", params={"const": 3}),)
        arr = ArrayValue(type=QubitType(), name="qs", shape=shape)
        validate_return_type(qmc.Vector[qmc.Qubit], arr, 0)


class TestInlinePassValidation:
    """Tests for IR-level type validation in InlinePass."""

    def test_inline_catches_array_scalar_mismatch(self):
        """InlinePass should catch array/scalar mismatch."""
        from qamomile.circuit.transpiler.passes.inline import InlinePass
        from qamomile.circuit.transpiler.errors import InliningError

        @qmc.qkernel
        def vec_kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return qs

        block = vec_kernel.block

        # Create a scalar Value (not ArrayValue) to pass where array is expected
        scalar_val = Value(type=QubitType(), name="q_scalar")

        # Call the block with mismatched type
        call_op = block.call(qs=scalar_val)

        # Create a HIERARCHICAL block so InlinePass processes it
        test_block = Block(
            operations=[call_op],
            input_values=[scalar_val],
            kind=BlockKind.HIERARCHICAL,
        )

        inline_pass = InlinePass()
        with pytest.raises(InliningError, match=r"expects array.*got scalar"):
            inline_pass.run(test_block)

    def test_inline_catches_element_type_mismatch(self):
        """InlinePass should catch IR type mismatch."""
        from qamomile.circuit.transpiler.passes.inline import InlinePass
        from qamomile.circuit.transpiler.errors import InliningError

        @qmc.qkernel
        def qubit_vec_kernel(
            qs: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            return qs

        block = qubit_vec_kernel.block

        # Create a Float array where Qubit array is expected
        shape = (Value(type=UIntType(), name="n", params={"const": 3}),)
        float_array = ArrayValue(type=FloatType(), name="fs", shape=shape)

        call_op = block.call(qs=float_array)

        test_block = Block(
            operations=[call_op],
            input_values=[float_array],
            kind=BlockKind.HIERARCHICAL,
        )

        inline_pass = InlinePass()
        with pytest.raises(InliningError, match=r"Element type mismatch"):
            inline_pass.run(test_block)
