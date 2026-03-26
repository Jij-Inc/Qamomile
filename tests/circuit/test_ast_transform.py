"""Tests for AST transform guards (loop-variable shadowing, while-condition checks, etc.)."""

import ast
import textwrap
import warnings

import pytest

import qamomile.circuit as qm
import qamomile.circuit as qmc
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.ast_transform import (
    ControlFlowTransformer,
    VariableCollector,
)
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
            violations = ast_transform.collect_quantum_rebind_violations(dummy, {"q"})

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

    def test_while_classical_condition_builds(self):
        """while condition with classical value builds at AST level (rejected at transpile)."""

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


class TestVariableCollectorAttributeAccess:
    """Attribute access should distinguish user variables from module names."""

    def test_attribute_access_keeps_user_variable_live(self):
        """User-variable attribute access must count as a load."""
        tree = ast.parse(
            textwrap.dedent(
                """
                n = qs.shape[0]
                """
            )
        )

        collector = VariableCollector(global_names={"qm"})
        collector.visit(tree)

        assert "qs" in collector.vars
        assert "qs" in collector.load_vars
        assert collector._first_context["qs"] == "Load"

    def test_module_attribute_access_still_excludes_module_name(self):
        """Module attribute access must not treat the module name as a variable."""
        tree = ast.parse(
            textwrap.dedent(
                """
                q = qm.h(q)
                """
            )
        )

        collector = VariableCollector(global_names={"qm"})
        collector.visit(tree)

        assert "qm" not in collector.vars
        assert "qm" not in collector.load_vars
        assert "q" in collector.load_vars


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


class TestLoopBackedgeIfLiveness:
    """Loop body-entry liveness must be preserved across nested ifs."""

    @staticmethod
    def _transform_source(source: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(source))
        transformed = ControlFlowTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        func_def = transformed.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        return func_def

    @staticmethod
    def _find_loop_body(func_def: ast.FunctionDef) -> list[ast.stmt]:
        loop_stmt = next(stmt for stmt in func_def.body if isinstance(stmt, ast.With))
        return loop_stmt.body

    @staticmethod
    def _find_emit_if_stmt(loop_body: list[ast.stmt], index: int = 0) -> ast.stmt:
        emit_if_stmts = []
        for stmt in loop_body:
            value = None
            if isinstance(stmt, ast.Assign):
                value = stmt.value
            elif isinstance(stmt, ast.Expr):
                value = stmt.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "emit_if"
            ):
                emit_if_stmts.append(stmt)
        return emit_if_stmts[index]

    @staticmethod
    def _assignment_target_names(stmt: ast.stmt) -> list[str]:
        if not isinstance(stmt, ast.Assign):
            return []
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple):
            return [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        return []

    def test_for_next_iteration_only_live_if_is_assigned_back(self):
        func_def = self._transform_source(
            """
            def f(n, state):
                out = state
                for i in range(n):
                    out = state
                    if state:
                        state = 0
                    else:
                        state = 1
                return out
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def))
        assert isinstance(emit_if_stmt, ast.Assign)
        assert "state" in self._assignment_target_names(emit_if_stmt)

    def test_while_next_iteration_only_live_if_is_assigned_back(self):
        func_def = self._transform_source(
            """
            def f(run, step, state):
                out = state
                while run:
                    out = state
                    if state:
                        state = 0
                    else:
                        state = 1
                    if step:
                        run = 0
                    else:
                        run = 1
                        step = 1
                return out
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def), index=0)
        assert isinstance(emit_if_stmt, ast.Assign)
        assert "state" in self._assignment_target_names(emit_if_stmt)

    def test_for_mixed_path_loop_head_preserves_load_first_value(self):
        func_def = self._transform_source(
            """
            def f(n, flag, init, x):
                for i in range(n):
                    if flag:
                        x = init
                    else:
                        x = x + 1
                return 0
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def))
        assert "x" in self._assignment_target_names(emit_if_stmt)

    def test_while_mixed_path_loop_head_preserves_load_first_value(self):
        func_def = self._transform_source(
            """
            def f(run, flag, init, x):
                while run:
                    if flag:
                        x = init
                    else:
                        x = x + 1
                    run = 0
                return 0
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def))
        assert "x" in self._assignment_target_names(emit_if_stmt)

    def test_for_store_first_loop_head_keeps_dead_value_filtered(self):
        func_def = self._transform_source(
            """
            def f(n, theta, x):
                for i in range(n):
                    if theta:
                        x = 1
                    else:
                        x = 2
                return 0
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def))
        assert "x" not in self._assignment_target_names(emit_if_stmt)

    def test_while_store_first_loop_head_keeps_dead_value_filtered(self):
        func_def = self._transform_source(
            """
            def f(run, theta, x):
                while run:
                    if theta:
                        x = 1
                    else:
                        x = 2
                    run = 0
                return 0
            """
        )

        emit_if_stmt = self._find_emit_if_stmt(self._find_loop_body(func_def))
        assert "x" not in self._assignment_target_names(emit_if_stmt)


class TestLoopElseReject:
    """for ... else and while ... else must be rejected at AST transform time."""

    @staticmethod
    def _transform_source(source: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(source))
        result = ControlFlowTransformer().visit(tree)
        return result.body[0]

    def test_for_else_raises(self):
        with pytest.raises(
            SyntaxError, match="for ... else is not supported in @qkernel"
        ):
            self._transform_source(
                """
                def f(n, x):
                    for i in range(n):
                        x = x + 1
                    else:
                        x = 0
                    return x
                """
            )

    def test_while_else_raises(self):
        with pytest.raises(
            SyntaxError, match="while ... else is not supported in @qkernel"
        ):
            self._transform_source(
                """
                def f(cond, x):
                    while cond:
                        x = x + 1
                    else:
                        x = 0
                    return x
                """
            )

    def test_for_else_pass_raises(self):
        """Even `else: pass` is rejected -- loop-else syntax itself is unsupported."""
        with pytest.raises(
            SyntaxError, match="for ... else is not supported in @qkernel"
        ):
            self._transform_source(
                """
                def f(n, x):
                    for i in range(n):
                        x = x + 1
                    else:
                        pass
                    return x
                """
            )

    def test_while_else_pass_raises(self):
        """Even `else: pass` is rejected -- loop-else syntax itself is unsupported."""
        with pytest.raises(
            SyntaxError, match="while ... else is not supported in @qkernel"
        ):
            self._transform_source(
                """
                def f(cond, x):
                    while cond:
                        x = x + 1
                    else:
                        pass
                    return x
                """
            )

    def test_for_without_else_works(self):
        """Normal for loop (no else) must still work."""
        func_def = self._transform_source(
            """
            def f(n, x):
                for i in range(n):
                    x = x + 1
                return x
            """
        )
        assert isinstance(func_def, ast.FunctionDef)

    def test_while_without_else_works(self):
        """Normal while loop (no else) must still work."""
        func_def = self._transform_source(
            """
            def f(cond, x):
                while cond:
                    x = x + 1
                    cond = 0
                return x
            """
        )
        assert isinstance(func_def, ast.FunctionDef)

    def test_qkernel_for_else_raises(self):
        """@qkernel decorator propagates the SyntaxError."""
        with pytest.raises(
            SyntaxError, match="for ... else is not supported in @qkernel"
        ):

            @qkernel
            def bad(n: qm.UInt, x: qm.UInt) -> qm.UInt:
                for i in qm.range(n):
                    x = x + 1
                else:
                    x = qm.UInt(0)
                return x

    def test_qkernel_while_else_raises(self):
        """@qkernel decorator propagates the SyntaxError."""
        with pytest.raises(
            SyntaxError, match="while ... else is not supported in @qkernel"
        ):

            @qkernel
            def bad(cond: qm.Bit, x: qm.UInt) -> qm.UInt:
                while cond:
                    x = x + 1
                else:
                    x = qm.UInt(0)
                return x


class TestInvalidPlaceholderLoopTargets:
    """Invalid items()/range() targets must raise SyntaxError at decoration time.

    These patterns previously fell through to a warning + fallback that silently
    dropped the loop body, producing measure-only circuits.
    """

    def test_items_value_tuple_unpack_raises(self):
        """for _, (i, j) in qmc.items(edges) must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Value target in items"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for _, (i, j) in qmc.items(edges):
                        q[i] = qmc.h(q[i])
                    return q[0]

    def test_items_single_target_raises(self):
        """for pair in qmc.items(edges) must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="items.*requires.*for key, value"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for pair in qmc.items(edges):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_wrong_arity_raises(self):
        """for a, b, c in qmc.items(edges) must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="items.*requires.*for key, value"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for a, b, c in qmc.items(edges):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_list_key_target_raises(self):
        """for [i, j], w in qmc.items(edges) must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Key target in items"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for [i, j], w in qmc.items(edges):
                        q[i] = qmc.h(q[i])
                    return q[0]

    def test_items_nested_tuple_key_raises(self):
        """for (i, (j, k)), v in qmc.items(d) must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="Nested tuple unpacking in items"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    d: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.Tuple[qmc.UInt, qmc.UInt]], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for (i, (j, k)), v in qmc.items(d):
                        q[i] = qmc.h(q[i])
                    return q[0]

    def test_items_dotcall_single_target_raises(self):
        """for pair in edges.items() must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="items.*requires.*for key, value"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for pair in edges.items():
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_module_call_no_args_raises(self):
        """qmc.items() with no dict argument must raise SyntaxError."""
        with pytest.raises(SyntaxError, match="requires exactly one dict argument"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    d: qmc.Dict[qmc.UInt, qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for k, v in qmc.items():
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_method_call_with_args_raises(self):
        """d.items(1) must raise SyntaxError (dict.items takes no arguments)."""
        with pytest.raises(SyntaxError, match="d\\.items.*no arguments"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    d: qmc.Dict[qmc.UInt, qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for k, v in d.items(1):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_range_list_unpack_raises(self):
        """for [i, j] in qmc.range(n) must raise SyntaxError."""
        with pytest.raises(
            SyntaxError, match="qmc.range.*requires a single loop variable"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(n: qmc.UInt) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for [i, j] in qmc.range(n):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_no_warning_on_reject(self):
        """Invalid items() target must not produce a fallback warning."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with pytest.raises(SyntaxError):

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for pair in qmc.items(edges):
                        q[0] = qmc.h(q[0])
                    return q[0]

        fallback_warnings = [
            w for w in ws if "AST transformation failed" in str(w.message)
        ]
        assert len(fallback_warnings) == 0

    def test_range_no_warning_on_reject(self):
        """Invalid range() target must not produce a fallback warning."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with pytest.raises(SyntaxError):

                @qmc.qkernel
                def bad(n: qmc.UInt) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for [i, j] in qmc.range(n):
                        q[0] = qmc.h(q[0])
                    return q[0]

        fallback_warnings = [
            w for w in ws if "AST transformation failed" in str(w.message)
        ]
        assert len(fallback_warnings) == 0
