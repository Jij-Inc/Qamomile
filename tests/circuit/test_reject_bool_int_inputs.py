"""Frontend integer inputs reject Python ``bool`` (bool subclasses int).

``bool`` is a subclass of ``int`` in Python, so a bare ``isinstance(x, int)``
would accept ``True`` / ``False`` where a genuine integer is required and
silently treat them as ``1`` / ``0``. These tests pin that the frontend
integer inputs reject a bool via ``is_plain_int`` (or an equivalent explicit
guard) instead of silently coercing it:

- compile-time ``UInt`` / ``int`` qkernel bindings,
- ``uint(...)`` constructor,
- array element indexing / slice bounds (``qs[True]``, ``qs[True:3]``),
- ``control(..., num_controls=...)``.

This mirrors the serialization-side guards in ``tests/circuit/ir/test_serialize.py``
(``TestRejectBoolInIntFields``).
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.constructors import uint


@qmc.qkernel
def _phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single-qubit phase rotation used as a controlled-U body."""
    return qmc.p(q, theta)


@qmc.qkernel
def _bound_uint_entry(n: qmc.UInt) -> qmc.Bit:
    """Expose a UInt argument through the ordinary QKernel build path."""
    _ = n
    return qmc.measure(qmc.qubit("q"))


class TestUIntBindingRequiresIntegralScalar:
    """Compile-time UInt bindings reject coercive float and bool inputs."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(3.0, id="python-float"),
            pytest.param(np.float64(3.0), id="numpy-float"),
            pytest.param(True, id="python-bool"),
            pytest.param(np.bool_(True), id="numpy-bool"),
        ],
    )
    def test_qkernel_build_rejects_non_integral_uint_binding(self, value):
        """QKernel.build never truncates or boolean-coerces a UInt binding."""
        with pytest.raises(TypeError, match="requires an integer scalar"):
            _bound_uint_entry.build(n=value)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(3, id="python-int"),
            pytest.param(np.int64(3), id="numpy-int"),
        ],
    )
    def test_qkernel_build_accepts_integral_uint_binding(self, value):
        """Python and NumPy integer scalars remain valid UInt bindings."""
        block = _bound_uint_entry.build(n=value)
        assert block.param_slots[0].name == "n"
        assert int(block.param_slots[0].bound_value) == 3


class TestUintRejectsBool:
    """``uint()`` accepts a plain int / str but rejects a bool."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_uint_rejects_bool(self, flag):
        """``uint(True/False)`` raises instead of baking a bool-valued const."""
        with pytest.raises(TypeError, match="int or str"):
            uint(flag)

    def test_uint_accepts_int_and_str(self):
        """A plain int literal and a str parameter name still construct."""
        assert uint(3).init_value == 3
        assert uint("n").name == "n"


class TestControlNumControlsRejectsBool:
    """``control(num_controls=...)`` rejects a bool control count."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_num_controls_rejects_bool(self, flag):
        """``num_controls=True/False`` raises rather than acting as 1/0.

        Without the guard, ``num_controls=True`` was silently stored as a
        bool (acting as 1) and ``num_controls=False`` only raised via the
        downstream ``< 1`` check with a confusing message.
        """
        with pytest.raises(TypeError, match="num_controls must be"):
            qmc.control(_phase, num_controls=flag)

    def test_num_controls_accepts_int(self):
        """A valid integer control count still constructs."""
        assert qmc.control(_phase, num_controls=2)._num_controls == 2
        assert qmc.control(_phase)._num_controls == 1


class TestArrayIndexRejectsBool:
    """Array element indices and slice bounds reject a bool."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_element_index_rejects_bool(self, flag):
        """``qs[True]`` / ``qs[False]`` raises instead of aliasing ``qs[1]`` / ``qs[0]``."""

        @qmc.qkernel
        def kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs[flag] = qmc.h(qs[flag])
            return qs

        with pytest.raises(TypeError, match="must be a plain int"):
            kernel.block  # noqa: B018

    @pytest.mark.parametrize("flag", [True, False])
    def test_slice_start_bound_rejects_bool(self, flag):
        """A bool slice start (``qs[True:3]``) is rejected."""

        @qmc.qkernel
        def kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = qs[flag:3]
            view = qmc.h(view)
            qs[flag:3] = view
            return qs

        with pytest.raises(TypeError, match="must be a plain int"):
            kernel.block  # noqa: B018

    @pytest.mark.parametrize("flag", [True, False])
    def test_slice_stop_bound_rejects_bool(self, flag):
        """A bool slice stop (``qs[0:True]``) is rejected.

        The stop bound does not flow through ``_make_uint_index`` (it is
        handled via ``_compute_slice_length`` / ``_as_int_const``), so it
        needs its own guard; without it ``qs[0:True]`` silently became
        ``qs[0:1]``.
        """

        @qmc.qkernel
        def kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = qs[0:flag]
            view = qmc.h(view)
            qs[0:flag] = view
            return qs

        with pytest.raises(TypeError, match="must be a plain int"):
            kernel.block  # noqa: B018

    @pytest.mark.parametrize("flag", [True, False])
    def test_slice_step_bound_rejects_bool(self, flag):
        """A bool slice step (``qs[0:4:True]``) is rejected."""

        @qmc.qkernel
        def kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            view = qs[0:4:flag]
            view = qmc.h(view)
            qs[0:4:flag] = view
            return qs

        with pytest.raises(TypeError, match="must be a plain int"):
            kernel.block  # noqa: B018

    def test_integer_index_and_slice_still_work(self):
        """Plain integer indices and slices build without error."""

        @qmc.qkernel
        def kernel(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs[1] = qmc.h(qs[1])
            view = qs[0:2]
            view = qmc.h(view)
            qs[0:2] = view
            return qs

        assert kernel.block is not None
