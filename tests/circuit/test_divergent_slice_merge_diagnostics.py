"""Diagnostics for VectorView merges with divergent slices across if branches.

Merging different slices of a register across if/else branches never
miscompiles silently — every reachable divergent path fails loudly —
but the raw failures used to be opaque (a generic borrow conflict at
trace time; a generic value-resolution error at execution time). These
tests pin the targeted diagnostics that name the divergent merge as the
cause, and pin that the legal patterns (identical slices in both
branches; per-branch write-back) keep working.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import (
    ExecutionError,
    QubitBorrowConflictError,
)
from qamomile.circuit.transpiler.segments import MultipleQuantumSegmentsError

pytest.importorskip("qiskit")

from qamomile.qiskit import QiskitTranspiler  # noqa: E402


def _sample_outcomes(kernel, bindings=None, shots=200):
    """Transpile and sample a kernel, returning the outcome tuples.

    Args:
        kernel: A ``@qmc.qkernel`` decorated function.
        bindings: Optional compile-time bindings dict.
        shots: Number of shots to sample.

    Returns:
        set: The sampled outcome values (first element of each result
            entry).
    """
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel, bindings=bindings or {})
    result = executable.sample(transpiler.executor(), shots=shots).result()
    return {outcome for outcome, _count in result.results}


class TestDivergentSliceMergeDiagnostics:
    """The two loud failure modes name the divergent merge as the cause."""

    def test_post_if_use_of_divergent_merge_names_the_cause(self):
        """Element access on a divergent merged view explains the merge."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            probe = qmc.qubit("p")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            if bit:
                view = qs[0:2]
            else:
                view = qs[1:3]
            view[0] = qmc.x(view[0])
            return qmc.measure(view)

        with pytest.raises(
            QubitBorrowConflictError,
            match="merges different slices across if/else branches",
        ):
            QiskitTranspiler().transpile(kernel, bindings={"dummy": 0})

    @pytest.mark.parametrize("transform", ["reslice", "broadcast"])
    def test_transform_does_not_hide_divergent_merge(self, transform):
        """Ownership-preserving view transforms retain divergence metadata."""

        if transform == "reslice":

            @qmc.qkernel
            def kernel() -> qmc.Vector[qmc.Bit]:
                qs = qmc.qubit_array(3, "qs")
                probe = qmc.h(qmc.qubit("probe"))
                bit = qmc.measure(probe)
                if bit:
                    view = qs[0:2]
                else:
                    view = qs[1:3]
                view = view[:]
                view[0] = qmc.x(view[0])
                return qmc.measure(view)

        else:

            @qmc.qkernel
            def kernel() -> qmc.Vector[qmc.Bit]:
                qs = qmc.qubit_array(3, "qs")
                probe = qmc.h(qmc.qubit("probe"))
                bit = qmc.measure(probe)
                if bit:
                    view = qs[0:2]
                else:
                    view = qs[1:3]
                view = qmc.h(view)
                view[0] = qmc.x(view[0])
                return qmc.measure(view)

        with pytest.raises(
            QubitBorrowConflictError,
            match="merges different slices across if/else branches",
        ):
            _ = kernel.block

    def test_nested_merge_does_not_hide_existing_divergence(self):
        """A later view merge retains divergence inherited from its input."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            probe = qmc.h(qmc.qubit("probe"))
            bit = qmc.measure(probe)
            if bit:
                view = qs[0:2]
            else:
                view = qs[1:3]
            if True:
                view = view[:]
            else:
                view = qmc.h(view)
            view[0] = qmc.x(view[0])
            return qmc.measure(view)

        with pytest.raises(
            QubitBorrowConflictError,
            match="merges different slices across if/else branches",
        ):
            _ = kernel.block

    def test_classical_divergent_view_merge_names_the_cause(self):
        """Executing a divergent classical view merge explains the merge."""

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs[1] = qmc.x(qs[1])
            s = qmc.measure(qs)
            probe = qmc.qubit("p")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            if bit:
                out = s[0:2]
            else:
                out = s[1:3]
            return out

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(kernel, bindings={"dummy": 0})
        with pytest.raises(
            ExecutionError, match="merging different slices across if/else"
        ):
            executable.sample(transpiler.executor(), shots=50).result()

    def test_runtime_quantum_view_selection_is_rejected_by_nisq_plan(self):
        """A runtime branch cannot dynamically select a measured qubit view."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            probe = qmc.qubit("probe")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            work = qmc.qubit("work")
            if bit:
                work = qmc.x(work)
                view = qs[0:2]
            else:
                work = qmc.z(work)
                view = qs[1:3]
            return qmc.measure(view)

        with pytest.raises(
            MultipleQuantumSegmentsError,
            match="different physical qubit regions",
        ):
            QiskitTranspiler().transpile(kernel)

    def test_nested_runtime_quantum_view_selection_is_rejected(self):
        """A divergent view merge is found through nested runtime ifs."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            outer_q = qmc.qubit("outer")
            outer_q = qmc.h(outer_q)
            outer = qmc.measure(outer_q)
            inner_q = qmc.qubit("inner")
            inner_q = qmc.h(inner_q)
            inner = qmc.measure(inner_q)
            work = qmc.qubit("work")
            if outer:
                work = qmc.x(work)
                if inner:
                    view = qs[0:2]
                else:
                    view = qs[1:3]
            else:
                work = qmc.z(work)
                view = qs[0:2]
            return qmc.measure(view)

        with pytest.raises(
            MultipleQuantumSegmentsError,
            match="different physical qubit regions",
        ):
            QiskitTranspiler().transpile(kernel)


class TestSliceMergePatternsKeepWorking:
    """The legal slice-merge patterns stay accepted and correct."""

    def test_identical_slices_merge_usable_after_if(self):
        """Same slice in both branches keeps post-if use working.

        bit=1 applies X to view[0] (outcome (1, 0)); bit=0 applies X to
        view[1] (outcome (0, 1)). Both outcomes appear over enough
        shots, and no divergent-merge diagnostic fires.
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            probe = qmc.qubit("p")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            if bit:
                view = qs[0:2]
                view[0] = qmc.x(view[0])
            else:
                view = qs[0:2]
                view[1] = qmc.x(view[1])
            return qmc.measure(view)

        outcomes = _sample_outcomes(kernel, bindings={"dummy": 0})
        assert outcomes == {(1, 0), (0, 1)}

    def test_divergent_slices_with_write_back_keep_working(self):
        """Per-branch slice + write-back stays accepted and correct.

        bit=1 flips qs[0] via the [0:2] view (outcome (1, 0, 0)); bit=0
        applies Z to qs[1] via the [1:3] view, which leaves |0> alone
        (outcome (0, 0, 0)).
        """

        @qmc.qkernel
        def kernel(dummy: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            probe = qmc.qubit("p")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            if bit:
                view = qs[0:2]
                view[0] = qmc.x(view[0])
                qs[0:2] = view
            else:
                view = qs[1:3]
                view[0] = qmc.z(view[0])
                qs[1:3] = view
            return qmc.measure(qs)

        outcomes = _sample_outcomes(kernel, bindings={"dummy": 0})
        assert outcomes == {(1, 0, 0), (0, 0, 0)}

    def test_root_and_full_view_runtime_merge_share_one_region(self):
        """A root vector and its full view are the same physical region."""

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])
            probe = qmc.qubit("probe")
            probe = qmc.h(probe)
            bit = qmc.measure(probe)
            work = qmc.qubit("work")
            if bit:
                work = qmc.x(work)
                view = qs
            else:
                work = qmc.z(work)
                view = qs[:]
            return qmc.measure(view)

        assert _sample_outcomes(kernel) == {(1, 0)}

    def test_compile_time_divergent_view_selects_one_region(self):
        """A compile-time branch may select either concrete qubit view."""
        choose_tail = True

        @qmc.qkernel
        def kernel() -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(3, "qs")
            qs[1] = qmc.x(qs[1])
            if choose_tail:
                view = qs[1:3]
            else:
                view = qs[0:2]
            return qmc.measure(view)

        assert _sample_outcomes(kernel) == {(1, 0)}
