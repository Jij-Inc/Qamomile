"""Tests for AST transform guards (loop-variable shadowing, while-condition checks, etc.)."""

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import qkernel


class TestLoopVariableShadowing:
    """Loop variable must not shadow function parameters."""

    def test_loop_var_shadows_parameter_raises(self):
        """for i in qm.range(n) where i is a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_circuit(i: qm.UInt) -> qm.UInt:
                for i in qm.range(3):
                    pass
                return i

    def test_loop_var_no_shadow_works(self):
        """Non-shadowing loop variable should work fine."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            for j in qm.range(3):
                qs[j] = qm.h(qs[j])
            q = qs[0]
            return q

        graph = good_circuit.build()
        assert graph is not None

    def test_items_key_shadow_raises(self):
        """items() key variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_items_key(i: qm.UInt) -> Qubit:
                qs = qubit_array(3, "qs")
                angles = qm.dict_input(qm.UInt, qm.Float, name="angles")
                for i, theta in qm.items(angles):
                    qs[i] = qm.rx(qs[i], theta)
                return qs[0]

    def test_items_value_shadow_raises(self):
        """items() value variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_items_value(theta: qm.Float) -> Qubit:
                qs = qubit_array(3, "qs")
                angles = qm.dict_input(qm.UInt, qm.Float, name="angles")
                for i, theta in qm.items(angles):
                    qs[i] = qm.rx(qs[i], theta)
                return qs[0]

    def test_items_tuple_key_shadow_raises(self):
        """items() tuple key variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_tuple_key(
                j: qm.UInt,
                ising: qm.Dict[qm.Tuple[qm.UInt, qm.UInt], qm.Float],
            ) -> Qubit:
                qs = qubit_array(3, "qs")
                for (i, j), Jij in qm.items(ising):
                    qs[i], qs[j] = qm.rzz(qs[i], qs[j], Jij)
                return qs[0]


class TestRuntimeLimitations:
    """Better diagnostics for runtime limitations."""

    def test_getsource_failure_descriptive_error(self):
        """Dynamically created function should give descriptive SyntaxError."""
        from qamomile.circuit.frontend.ast_transform import transform_control_flow

        ns = {"qm": qm, "Qubit": Qubit}
        exec(
            "def f(q: Qubit) -> Qubit:\n    q = qm.h(q)\n    return q\n",
            ns,
        )
        with pytest.raises(SyntaxError, match="Cannot retrieve source code"):
            transform_control_flow(ns["f"])

    def test_getsource_builtin_typeerror_gives_descriptive_error(self):
        """Passing a builtin (TypeError from getsource) should give descriptive SyntaxError."""
        from qamomile.circuit.frontend.ast_transform import transform_control_flow

        with pytest.raises(SyntaxError, match="Cannot retrieve source code"):
            transform_control_flow(len)

    def test_collect_quantum_rebind_violations_warns_on_source_unavailable(
        self, monkeypatch
    ):
        """Rebind analysis should warn and return empty list when source is unavailable."""
        import warnings

        import qamomile.circuit.frontend.ast_transform as ast_transform

        def _raise_oserror(_obj):
            raise OSError("source unavailable")

        monkeypatch.setattr(ast_transform.inspect, "getsource", _raise_oserror)

        def dummy(q: Qubit) -> Qubit:
            return q

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            violations = ast_transform.collect_quantum_rebind_violations(
                dummy, {"q"}
            )

        assert violations == []
        rebind_warnings = [
            w for w in ws if "Quantum rebind analysis skipped" in str(w.message)
        ]
        assert len(rebind_warnings) == 1

    def test_while_quantum_condition_raises(self):
        """while condition with quantum operation should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Quantum operation"):

            @qkernel
            def bad_circuit(q: Qubit) -> Qubit:
                while qm.measure(q):
                    q = qm.h(q)
                return q

    def test_while_classical_condition_works(self):
        """while condition with classical value should work."""

        @qkernel
        def good_circuit(q: Qubit, n: qm.UInt) -> Qubit:
            while n:
                q = qm.h(q)
            return q

        graph = good_circuit.build(n=1)
        assert graph is not None

    def test_while_qkernel_condition_raises(self):
        """while condition calling a @qkernel should raise SyntaxError."""

        @qkernel
        def cond_fn(q: Qubit) -> qm.Bit:
            return qm.measure(q)

        with pytest.raises(SyntaxError, match="Quantum kernel"):

            @qkernel
            def bad_indirect(q: Qubit) -> Qubit:
                while cond_fn(q):
                    q = qm.h(q)
                return q

    def test_while_classical_callable_same_name_as_quantum_op(self):
        """Classical function named 'measure' in while condition should be allowed."""

        def measure(n: int) -> bool:
            return n > 0

        @qkernel
        def circuit(q: Qubit, n: qm.UInt) -> Qubit:
            while measure(n):
                q = qm.h(q)
            return q

        # Should not raise — measure here is a classical function, not qm.measure
        graph = circuit.build(n=1)
        assert graph is not None


class TestEmptyClosureCell:
    """Empty closure cells (forward references) should not crash @qkernel."""

    def test_empty_cell_does_not_raise_valueerror(self):
        """@qkernel with forward-referenced freevar should not raise ValueError."""

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qm.h(q)
                return q

            def helper(q: Qubit) -> qm.Bit:
                return qm.measure(q)

            return circuit

        # Must not raise ValueError: Cell is empty
        kernel = factory()
        assert kernel is not None

    def test_empty_cell_triggers_fallback_warning(self):
        """Empty cell should trigger AST transformation fallback warning."""
        import warnings

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qm.h(q)
                return q

            def helper(q: Qubit) -> qm.Bit:
                return qm.measure(q)

            return circuit

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            factory()

        fallback_warnings = [
            w for w in ws if "AST transformation failed" in str(w.message)
        ]
        assert len(fallback_warnings) == 1

    def test_empty_cell_fallback_no_nameerror(self):
        """After fallback, build() must not raise NameError from missing freevars."""

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qm.h(q)
                return q

            def helper(q: Qubit) -> qm.Bit:
                return qm.measure(q)

            return circuit

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel = factory()

        # build() may fail for semantic reasons (e.g. QubitConsumedError),
        # but must NOT fail with NameError from missing freevar injection.
        try:
            kernel.build()
        except NameError:
            pytest.fail("build() raised NameError — freevar injection regression")
        except Exception:
            pass  # Other exceptions (semantic) are acceptable

    def test_bound_closure_still_works(self):
        """Closure with all cells bound should still work normally."""

        def factory():
            n_qubits = 3

            @qkernel
            def circuit() -> Qubit:
                qs = qubit_array(n_qubits, "qs")
                return qs[0]

            return circuit

        kernel = factory()
        graph = kernel.build()
        assert graph is not None


class TestQuantumOpsSpec:
    """Verify _QUANTUM_OPS matches the authoritative while-condition spec."""

    def test_quantum_ops_matches_spec(self):
        from qamomile.circuit.frontend.ast_transform import ControlFlowTransformer

        expected = frozenset(
            {
                "h",
                "x",
                "y",
                "z",
                "s",
                "t",
                "sdg",
                "tdg",
                "cx",
                "cz",
                "rx",
                "ry",
                "rz",
                "p",
                "cp",
                "swap",
                "ccx",
                "rzz",
                "measure",
                "expval",
            }
        )
        missing = expected - ControlFlowTransformer._QUANTUM_OPS
        extra = ControlFlowTransformer._QUANTUM_OPS - expected
        assert missing == set(), f"Missing from _QUANTUM_OPS: {missing}"
        assert extra == set(), f"Extra in _QUANTUM_OPS: {extra}"
