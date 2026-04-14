"""Tests for Vector parameter shape construction (Layer 1).

When a Vector parameter is declared in ``parameters=[...]`` at the top
level, its shape must be symbolic — not empty — so that ``.shape[i]``
returns a usable Value. This matches the behaviour of inner-kernel
tracing via ``func_to_block.create_dummy_input`` and is required for
``for i in qmc.range(gamma.shape[0])`` style code to build.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle.primitives import UInt
from qamomile.circuit.ir.value import ArrayValue


class TestVectorParameterShape:
    def test_top_level_vector_parameter_has_symbolic_shape(self):
        """Top-level Vector[Float] parameter gets a length-1 shape tuple."""

        @qmc.qkernel
        def circuit(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            return gamma[0]

        block = circuit.build(parameters=["gamma"])
        gamma_value = next(
            v for v in block.input_values if getattr(v, "name", None) == "gamma"
        )
        assert isinstance(gamma_value, ArrayValue)
        assert len(gamma_value.shape) == 1
        dim0 = gamma_value.shape[0]
        assert dim0.name == "gamma_dim0"

    def test_shape_dim_is_uint_handle_in_user_code(self):
        """``gamma.shape[0]`` inside a qkernel returns a UInt Handle."""

        captured: dict = {}

        @qmc.qkernel
        def probe(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            captured["dim0"] = gamma.shape[0]
            return gamma[0]

        probe.build(parameters=["gamma"])
        assert isinstance(captured["dim0"], UInt)
        assert captured["dim0"].value.name == "gamma_dim0"

    def test_flat_kernel_shape_loop_builds(self):
        """Flat kernel using ``gamma.shape[0]`` as a loop bound builds cleanly.

        Before Layer 1 this raised ``IndexError: tuple index out of range``
        because the top-level parameter had an empty ``_shape``.
        """

        @qmc.qkernel
        def flat(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            acc = gamma[0]
            num_iter = gamma.shape[0]
            for i in qmc.range(num_iter):
                acc = acc + gamma[i]
            return acc

        block = flat.build(parameters=["gamma"])
        assert block is not None

    def test_nested_kernel_shape_loop_builds(self):
        """Nested kernel using inner ``gamma.shape[0]`` still builds."""

        @qmc.qkernel
        def inner(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            acc = gamma[0]
            num_iter = gamma.shape[0]
            for i in qmc.range(num_iter):
                acc = acc + gamma[i]
            return acc

        @qmc.qkernel
        def outer(gamma: qmc.Vector[qmc.Float]) -> qmc.Float:
            return inner(gamma)

        block = outer.build(parameters=["gamma"])
        assert block is not None

    def test_uint_vector_parameter_has_symbolic_shape(self):
        """Vector[UInt] parameters also get symbolic shape."""

        @qmc.qkernel
        def circuit(idx: qmc.Vector[qmc.UInt]) -> qmc.UInt:
            return idx[0]

        block = circuit.build(parameters=["idx"])
        idx_value = next(
            v for v in block.input_values if getattr(v, "name", None) == "idx"
        )
        assert isinstance(idx_value, ArrayValue)
        assert len(idx_value.shape) == 1
        assert idx_value.shape[0].name == "idx_dim0"

    def test_parameter_array_rejects_qubit_element_type(self):
        """Vector[Qubit] is not a valid top-level scalar parameter."""

        @qmc.qkernel
        def circuit(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            return q

        with pytest.raises(TypeError, match="only float, int, UInt, and their arrays"):
            circuit.build(parameters=["q"])
