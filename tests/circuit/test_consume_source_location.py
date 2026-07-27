"""Regression: affine-violation errors report user-source ``file:line`` locations.

``Handle.consume`` records the user-code location of the consuming call, and
the AST transform re-anchors traced kernel bodies to the original source file,
so a ``QubitConsumedError`` names both the first-use line and the reuse line.
These tests assert on this file's own line numbers, so the kernels below are
line-sensitive: renumbering requires updating the expected markers, which the
tests derive dynamically via ``inspect`` to stay robust.
"""

from __future__ import annotations

import inspect

import pytest

import qamomile.circuit as qm
from qamomile.circuit.transpiler.errors import QubitConsumedError


def _lines_of(func, *markers: str) -> list[int]:
    """Return absolute line numbers of marker comments inside ``func``.

    Args:
        func: Function whose source is scanned. May be a QKernel wrapper;
            its ``raw_func`` is used when present.
        *markers: Comment substrings to locate, one result per marker.

    Returns:
        list[int]: Absolute 1-based line numbers, in marker order.

    Raises:
        ValueError: If a marker does not appear in the function source.
    """
    target = getattr(func, "raw_func", func)
    source_lines, first = inspect.getsourcelines(target)
    result = []
    for marker in markers:
        for offset, line in enumerate(source_lines):
            if marker in line:
                result.append(first + offset)
                break
        else:
            raise ValueError(f"marker {marker!r} not found")
    return result


class TestConsumeSourceLocation:
    """QubitConsumedError carries first-use and reuse source locations."""

    def test_direct_reuse_reports_both_lines(self):
        """Straight-line reuse reports the h() line and the x() line."""

        @qm.qkernel
        def bad(q: qm.Qubit) -> qm.Qubit:
            q1 = qm.h(q)  # FIRST-USE
            q2 = qm.x(q)  # REUSE
            _ = q1
            return q2

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        message = str(excinfo.value)
        assert f"{__file__}:{first_line}" in message
        assert f"{__file__}:{reuse_line}" in message
        # The structured attribute carries the first-use location too.
        assert excinfo.value.first_use_location is not None
        assert excinfo.value.first_use_location.endswith(f":{first_line}")

    def test_branch_consumed_reuse_reports_branch_line(self):
        """Reuse after a branch-side consume points into the branch body."""

        @qm.qkernel
        def branchy(q: qm.Qubit, t: qm.Qubit) -> qm.Bit:
            t = qm.h(t)
            c = qm.measure(t)
            if c:
                _ = qm.measure(q)  # BRANCH-CONSUME
            q = qm.h(q)  # POST-IF-REUSE
            return qm.measure(q)

        branch_line, reuse_line = _lines_of(
            branchy, "# BRANCH-CONSUME", "# POST-IF-REUSE"
        )
        with pytest.raises(QubitConsumedError) as excinfo:
            branchy.build()
        message = str(excinfo.value)
        assert f"{__file__}:{branch_line}" in message
        assert f"{__file__}:{reuse_line}" in message

    def test_traceback_points_at_user_source(self):
        """Python errors raised inside a traced kernel body carry real lines.

        The AST transform compiles the traced body against the original file
        (not an opaque ``<qamomile-dsl>`` buffer), so an arbitrary exception's
        traceback names this file and the offending line.
        """

        @qm.qkernel
        def boom(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            raise RuntimeError("kernel body failure")  # BOOM
            return q

        (boom_line,) = _lines_of(boom, "# BOOM")
        with pytest.raises(RuntimeError, match="kernel body failure") as excinfo:
            boom.build()
        tb = excinfo.traceback
        assert any(
            str(entry.path) == __file__ and entry.lineno + 1 == boom_line
            for entry in tb
        ), f"traceback does not include {__file__}:{boom_line}"

    def test_fresh_handle_has_no_stale_location(self):
        """The handle returned by consume() starts with no recorded location."""
        from qamomile.circuit.frontend.handle import Qubit
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        q = Qubit(value=Value(type=QubitType(), name="q"))
        fresh = q.consume("h")
        assert q._consumed_at is not None
        assert fresh._consumed_at is None
        assert fresh._consumed is False


class TestVectorConsumeSourceLocation:
    """Vector / view QubitConsumedError sites carry file:line locations too.

    These mirror ``TestConsumeSourceLocation`` for the array-side raise
    sites: element access / element return on a consumed array, slice
    assignment onto a consumed root or a consumed LHS view, and passing an
    already-consumed view to a qkernel call.
    """

    def _assert_both_locations(self, excinfo, first_line: int, reuse_line: int):
        """Assert the error message and attribute carry both locations.

        Args:
            excinfo: ``pytest.ExceptionInfo`` for the raised
                ``QubitConsumedError``.
            first_line: Expected 1-based line of the first-consuming call.
            reuse_line: Expected 1-based line of the offending reuse.
        """
        message = str(excinfo.value)
        assert f"{__file__}:{first_line}" in message
        assert f"{__file__}:{reuse_line}" in message
        assert excinfo.value.first_use_location is not None
        assert excinfo.value.first_use_location.endswith(f":{first_line}")

    def test_element_access_after_measure_reports_both_lines(self):
        """Reading an element of a measured array names both source lines."""

        @qm.qkernel
        def bad() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            bits = qm.measure(qs)  # FIRST-USE
            _ = qs[0]  # REUSE
            return bits

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        self._assert_both_locations(excinfo, first_line, reuse_line)

    def test_element_return_after_measure_reports_both_lines(self):
        """Writing an element into a measured array names both source lines."""

        @qm.qkernel
        def bad() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(2, "qs")
            bits = qm.measure(qs)  # FIRST-USE
            qs[0] = qm.qubit("fresh")  # REUSE
            return bits

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        self._assert_both_locations(excinfo, first_line, reuse_line)

    def test_slice_assignment_target_consumed_reports_both_lines(self):
        """Slice-assigning into a measured root names both source lines."""

        @qm.qkernel
        def bad() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            v = qm.h(qs[0:2])
            qs[0:2] = v
            bits = qm.measure(qs)  # FIRST-USE
            w = qm.qubit_array(2, "w")
            qs[0:2] = w[0:2]  # REUSE
            return bits

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        self._assert_both_locations(excinfo, first_line, reuse_line)

    def test_slice_assignment_lhs_view_consumed_reports_both_lines(self):
        """Slice-assigning onto a released LHS view names both source lines."""

        @qm.qkernel
        def bad() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            outer = qs[0:4]
            inner = qm.h(outer[0:2])
            outer[0:2] = inner
            qs[0:4] = outer  # FIRST-USE
            w = qm.qubit_array(2, "w")
            outer[0:2] = w[0:2]  # REUSE
            return qm.measure(qs)

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        self._assert_both_locations(excinfo, first_line, reuse_line)

    def test_consumed_view_kernel_arg_reports_both_lines(self):
        """Passing a consumed view to a qkernel call names both source lines."""

        @qm.qkernel
        def sub(v: qm.Vector[qm.Qubit]) -> qm.Vector[qm.Qubit]:
            return qm.h(v)

        @qm.qkernel
        def bad() -> qm.Vector[qm.Bit]:
            qs = qm.qubit_array(4, "qs")
            v = qs[0:2]
            v2 = qm.h(v)  # FIRST-USE
            _ = sub(v)  # REUSE
            qs[0:2] = v2
            return qm.measure(qs)

        first_line, reuse_line = _lines_of(bad, "# FIRST-USE", "# REUSE")
        with pytest.raises(QubitConsumedError) as excinfo:
            bad.build()
        self._assert_both_locations(excinfo, first_line, reuse_line)

    def test_transfer_borrow_to_records_location(self):
        """The manual-consume transfer path records a first-use location.

        ``VectorView._transfer_borrow_to`` marks the view consumed without
        going through ``Handle.consume``, so it must record ``_consumed_at``
        itself for later reuse diagnostics to carry a ``file:line``.
        """
        from qamomile.circuit.frontend.constructors import qubit_array
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            qs = qubit_array(2, "qs")
            view = qs[0:2]
            successor = object.__new__(type(view))
            successor.value = view.value
            successor._borrowed_indices = {}
            successor._shape = view._shape
            successor.element_type = view.element_type
            successor.parent = None
            successor.indices = ()
            successor.name = None
            successor.id = "successor"
            successor._consumed = False
            successor._consumed_by = None
            successor._slice_parent = view._slice_parent
            successor._slice_start = view._slice_start
            successor._slice_step = view._slice_step
            successor._slice_outer_view = None
            view._transfer_borrow_to(successor, "pauli_evolve")
            assert view._consumed is True
            assert view._consumed_at is not None


class TestPseudoFilenameClassification:
    """Frame classification treats notebook cells as user code.

    Classic IPython compiles cells under ``<ipython-input-N-...>``
    pseudo-filenames. Only Qamomile-synthesized pseudo-files
    (``<qamomile...>``) are library-internal; skipping all ``<...>`` names
    lost the error location for every kernel defined in a notebook.
    """

    @staticmethod
    def _ref_via(filename: str):
        """Return the frame ref captured from code compiled as ``filename``.

        Args:
            filename: Pseudo-filename to compile the probe caller under.

        Returns:
            The ``_FrameRef | None`` observed by ``_user_code_frame_ref``
            from inside that frame.
        """
        from qamomile.circuit.frontend.handle.handle import _user_code_frame_ref

        namespace = {"_user_code_frame_ref": _user_code_frame_ref}
        source = "def caller():\n    return _user_code_frame_ref()\n"
        exec(compile(source, filename, "exec"), namespace)
        return namespace["caller"]()

    def test_ipython_input_frame_is_user_code(self):
        """An <ipython-input-...> frame is reported as the user location."""
        from qamomile.circuit.frontend.handle.handle import _format_frame_ref

        location = _format_frame_ref(self._ref_via("<ipython-input-7-deadbeef>"))
        assert location == "<ipython-input-7-deadbeef>:2"

    def test_qamomile_pseudo_frame_is_skipped(self):
        """A <qamomile-...> synthesized frame is skipped; the caller is used."""
        from qamomile.circuit.frontend.handle.handle import _format_frame_ref

        location = _format_frame_ref(self._ref_via("<qamomile-control-wrapper-x>"))
        assert location is not None
        assert "<qamomile" not in location
        assert __file__ in location
