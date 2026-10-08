"""Regression tests: rebind-analyzer target-spelling coverage.

The decoration-time ``QuantumRebindAnalyzer`` protected exactly one spelling
of quantum rebinding — a plain ``ast.Name`` / flat all-``Name`` tuple target.
The same rebind written through a starred / list / nested-tuple target
decorated, transpiled, and executed — silently orphaning the original
register — while the plain spelling ``a = qm.x(b)`` was rejected at
decoration. Enforcement depended on syntax, not semantics.

The fix routes ``ast.List`` targets through the unpacking path and adds a
recursive target-tree walk for mixed literal-RHS shapes (starred, nested
tuple / list, and combinations). Every ``Name`` leaf meets the same rebind
rules as the plain spelling, while a pure-quantum permutation — a 1-to-1
swap of existing quantum origins, recognised across nesting levels and
around a star — is allowed (``a, (b,) = b, (a,)``; ``a, b, *rest = b, a,
pad``), matching the analyzer's documented permutation rule.

Trace-dead branch-suppression for ``try`` / ``match`` and a helper-local
scope model for nested ``def`` are intentionally OUT OF SCOPE: naive
suppression there creates unbackstopped false negatives (a handler or case
body that runs during tracing rebinds for real, and — unlike ``if`` / ``for``
/ ``while`` — no IR-layer discard check backstops it), so they keep the
pre-existing ``generic_visit`` behavior and are deferred to a follow-up.

Allowed patterns are executed with deterministic asserted outcomes, not just
decorated: a kernel that decorates but miscompiles would slip through a
decoration-only assertion.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import QubitRebindError
from qamomile.qiskit import QiskitTranspiler


def _sample_single(kernel, shots=100):
    """Transpile, execute on Qiskit, and return the single deterministic outcome.

    Args:
        kernel (QKernel): The qkernel to compile and run.
        shots (int): Number of shots. Defaults to 100.

    Returns:
        int | tuple[int, ...]: The unique measured value across all shots.
    """
    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(kernel)
    result = executable.sample(transpiler.executor(), shots=shots).result()
    assert len(result.results) == 1, f"expected deterministic outcome: {result}"
    return result.results[0][0]


class TestTargetSpellingMissesClosed:
    """Every spelling of the same rebind meets the same rejection."""

    def test_plain_rebind_rejected(self):
        """Control pin: the protected plain spelling keeps raising."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                a = qmc.x(b)
                return qmc.measure(a)

    def test_starred_target_rebind_rejected(self):
        """`a, *rest = x(b), pad` is the same rebind of `a` — now rejected."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                a, *rest = qmc.x(b), qmc.qubit("pad")  # noqa: F841
                return qmc.measure(a)

    def test_starred_suffix_rebind_rejected(self):
        """A rebind positioned AFTER the star pairs with the RHS tail."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                *rest, a = qmc.qubit("pad"), qmc.x(b)  # noqa: F841
                return qmc.measure(a)

    def test_list_target_rebind_rejected(self):
        """`[a] = [x(b)]` is the same rebind of `a` — now rejected."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                [a] = [qmc.x(b)]
                return qmc.measure(a)

    def test_nested_tuple_target_rebind_rejected(self):
        """`a, (m, n) = x(b), (c, d)` rebinds `a` — now rejected."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                c = qmc.qubit("c")
                d = qmc.qubit("d")
                a, (m, n) = qmc.x(b), (c, d)  # noqa: F841
                return qmc.measure(a)

    def test_fresh_rebind_via_list_target_rejected(self):
        """Fresh-allocation rebinds are caught through the new spellings too."""
        with pytest.raises(QubitRebindError, match="freshly allocated"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                [a] = [qmc.qubit("fresh")]
                return qmc.measure(a)


class TestLegalSpellingsStillDecorateAndExecute:
    """The target-tree walk must not reject legal unpacking shapes."""

    def test_starred_fresh_names_execute(self):
        """All-new names through a starred target stay legal: the X-prepared
        first qubit measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            first, *rest = qmc.x(qmc.qubit("a")), qmc.qubit("b")  # noqa: F841
            return qmc.measure(first)

        assert _sample_single(k) == 1

    def test_list_target_self_update_executes(self):
        """`[a] = [x(a)]` is a self-update, not a rebind: measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            a = qmc.qubit("a")
            [a] = [qmc.x(a)]
            return qmc.measure(a)

        assert _sample_single(k) == 1

    def test_nested_tuple_fresh_names_execute(self):
        """New names through a nested-tuple target stay legal: the X-prepared
        leaf measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            first, (m, n) = qmc.qubit("a"), (qmc.x(qmc.qubit("b")), qmc.qubit("c"))  # noqa: F841
            return qmc.measure(m)

        assert _sample_single(k) == 1

    def test_nested_permutation_swap_allowed(self):
        """The pure-quantum permutation escape carries into nested levels:
        ``(a, b), c = (b, a), c`` is a legal swap, not two rebinds. After
        the swap ``a`` holds the X-prepared qubit and measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            a = qmc.x(qmc.qubit("a"))
            b = qmc.qubit("b")
            c = qmc.qubit("c")
            (a, b), c = (b, a), c  # nested swap of a, b; c unchanged
            return qmc.measure(b)  # b now holds the X-prepared qubit

        assert _sample_single(k) == 1

    def test_cross_level_permutation_swap_allowed(self):
        """A 1-to-1 swap whose two halves sit at DIFFERENT nesting levels
        (``a, (b,) = b, (a,)``) is a legal permutation: the whole-subtree
        escape matches the ``{a, b}`` multiset across levels. ``b`` ends
        up holding the X-prepared qubit and measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            a = qmc.x(qmc.qubit("a"))
            b = qmc.qubit("b")
            a, (b,) = b, (a,)  # cross-level swap of a, b
            return qmc.measure(b)

        assert _sample_single(k) == 1

    def test_starred_permutation_swap_allowed(self):
        """A swap at a level that also carries a starred target
        (``a, b, *rest = b, a, pad``) keeps the permutation escape: the
        fixed ``a, b`` pair swaps and ``*rest`` binds a fresh list. ``b``
        holds the X-prepared qubit and measures 1."""

        @qmc.qkernel
        def k() -> qmc.Bit:
            a = qmc.x(qmc.qubit("a"))
            b = qmc.qubit("b")
            a, b, *rest = b, a, qmc.qubit("pad")  # noqa: F841
            return qmc.measure(b)

        assert _sample_single(k) == 1

    def test_swap_nested_in_non_permutation_still_flags_rebind(self):
        """A legal nested swap does not shield a genuine rebind elsewhere in
        the same statement: ``(a, b), c = (b, a), x(d)`` swaps a,b (legal)
        but rebinds ``c`` to d's value (illegal) — the rebind of c is
        still flagged."""
        with pytest.raises(QubitRebindError, match="different quantum variable"):

            @qmc.qkernel
            def k() -> qmc.Bit:
                a = qmc.qubit("a")
                b = qmc.qubit("b")
                c = qmc.qubit("c")
                d = qmc.qubit("d")
                (a, b), c = (b, a), qmc.x(d)
                return qmc.measure(c)
