"""Tests for AST transform guards (loop-variable shadowing, while-condition checks, etc.)."""

import ast
import textwrap
import threading
import warnings

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.ast_transform import (
    ControlFlowTransformer,
    VariableCollector,
)
from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation import GateOperationType, InvokeOperation
from qamomile.circuit.transpiler.errors import FrontendTransformError

_LIVE_GLOBAL_ANGLE = 0.5


@qkernel
def _kernel_using_live_global(q: Qubit) -> Qubit:
    """Apply a rotation using a module global rebound after decoration."""
    return qmc.rz(q, _LIVE_GLOBAL_ANGLE)


_LIVE_GLOBAL_ANGLE = 1.25


@qkernel
def _kernel_using_forward_helper(q: Qubit) -> Qubit:
    """Call a helper whose module binding is defined later in this file."""
    return _forward_helper(q)


@qkernel
def _forward_helper(q: Qubit) -> Qubit:
    """Provide the forward-defined helper used by the preceding kernel."""
    return qmc.x(q)


class TestLoopVariableShadowing:
    """Loop variable must not shadow function parameters."""

    def test_loop_var_shadows_parameter_raises(self):
        """for i in qmc.range(n) where i is a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_circuit(i: qmc.UInt) -> qmc.UInt:
                for i in qmc.range(3):
                    pass
                return i

    def test_loop_var_no_shadow_works(self):
        """Non-shadowing loop variable should work fine."""

        @qkernel
        def good_circuit() -> Qubit:
            qs = qubit_array(3, "qs")
            for j in qmc.range(3):
                qs[j] = qmc.h(qs[j])
            q = qs[0]
            return q

        graph = good_circuit.build()
        assert graph is not None

    def test_items_key_shadow_raises(self):
        """items() key variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_items_key(i: qmc.UInt) -> Qubit:
                qs = qubit_array(3, "qs")
                angles = qmc.dict_input(qmc.UInt, qmc.Float, name="angles")
                for i, theta in qmc.items(angles):
                    qs[i] = qmc.rx(qs[i], theta)
                return qs[0]

    def test_items_value_shadow_raises(self):
        """items() value variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_items_value(theta: qmc.Float) -> Qubit:
                qs = qubit_array(3, "qs")
                angles = qmc.dict_input(qmc.UInt, qmc.Float, name="angles")
                for i, theta in qmc.items(angles):
                    qs[i] = qmc.rx(qs[i], theta)
                return qs[0]

    def test_items_tuple_key_shadow_raises(self):
        """items() tuple key variable shadowing a parameter should raise SyntaxError."""
        with pytest.raises(SyntaxError, match="shadows a function parameter"):

            @qkernel
            def bad_tuple_key(
                j: qmc.UInt,
                ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            ) -> Qubit:
                qs = qubit_array(3, "qs")
                for (i, j), Jij in qmc.items(ising):
                    qs[i], qs[j] = qmc.rzz(qs[i], qs[j], Jij)
                return qs[0]


class TestRuntimeLimitations:
    """Better diagnostics for runtime limitations."""

    def test_getsource_failure_descriptive_error(self):
        """Dynamically created function should give descriptive SyntaxError."""
        from qamomile.circuit.frontend.ast_transform import transform_control_flow

        ns = {"qmc": qmc, "Qubit": Qubit}
        exec(
            "def f(q: Qubit) -> Qubit:\n    q = qmc.h(q)\n    return q\n",
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
                while qmc.measure(q):
                    q = qmc.h(q)
                return q

    def test_while_classical_condition_builds(self):
        """while condition with classical value builds at AST level (rejected at transpile)."""

        @qkernel
        def good_circuit(q: Qubit, n: qmc.UInt) -> Qubit:
            while n:
                q = qmc.h(q)
            return q

        graph = good_circuit.build(n=1)
        assert graph is not None

    def test_while_qkernel_condition_raises(self):
        """while condition calling a @qkernel should raise SyntaxError."""

        @qkernel
        def cond_fn(q: Qubit) -> qmc.Bit:
            return qmc.measure(q)

        with pytest.raises(SyntaxError, match="Quantum kernel"):

            @qkernel
            def bad_indirect(q: Qubit) -> Qubit:
                while cond_fn(q):
                    q = qmc.h(q)
                return q

    def test_while_classical_callable_same_name_as_quantum_op(self):
        """Classical function named 'measure' in while condition should be allowed."""

        def measure(n: int) -> bool:
            return n > 0

        @qkernel
        def circuit(q: Qubit, n: qmc.UInt) -> Qubit:
            while measure(n):
                q = qmc.h(q)
            return q

        # Should not raise — measure here is a classical function, not qmc.measure
        graph = circuit.build(n=1)
        assert graph is not None


class TestFailLoudPythonSemantics:
    """Unsupported Python truth and control-flow semantics must fail loudly."""

    def test_symbolic_handle_has_no_python_truth_value(self):
        """Direct bool conversion of a symbolic handle raises TypeError."""
        with pytest.raises(TypeError, match="symbolic Qamomile handle"):
            bool(qmc.bit(False))

    def test_not_on_symbolic_handle_is_rejected(self):
        """Python ``not`` cannot silently fold a symbolic Bit to false."""

        @qkernel
        def bad_not() -> qmc.Bit:
            bit = qmc.bit(False)
            if not bit:
                bit = qmc.bit(True)
            return bit

        with pytest.raises(TypeError, match="use '&', '\\|', and '~'"):
            bad_not.build()

    def test_and_on_symbolic_handles_is_rejected(self):
        """Python ``and`` cannot silently discard its first symbolic operand."""

        @qkernel
        def bad_and() -> qmc.Bit:
            left = qmc.bit(True)
            right = qmc.bit(False)
            if left and right:
                right = qmc.bit(True)
            return right

        with pytest.raises(TypeError, match="symbolic Qamomile handle"):
            bad_and.build()

    def test_or_on_symbolic_handles_is_rejected(self):
        """Python ``or`` cannot silently discard its second symbolic operand."""

        @qkernel
        def bad_or() -> qmc.Bit:
            left = qmc.bit(False)
            right = qmc.bit(True)
            if left or right:
                right = qmc.bit(False)
            return right

        with pytest.raises(TypeError, match="symbolic Qamomile handle"):
            bad_or.build()

    def test_conditional_expression_on_symbolic_handle_is_rejected(self):
        """A symbolic conditional expression cannot choose a branch at trace time."""

        @qkernel
        def bad_conditional_expression() -> qmc.UInt:
            condition = qmc.bit(False)
            return qmc.uint(1) if condition else qmc.uint(0)

        with pytest.raises(TypeError, match="symbolic Qamomile handle"):
            bad_conditional_expression.build()

    def test_chained_comparison_on_symbolic_handles_is_rejected(self):
        """A chained comparison cannot discard its first symbolic comparison."""

        @qkernel
        def bad_chained_comparison(value: qmc.UInt) -> qmc.Bit:
            return qmc.uint(0) < value < qmc.uint(2)

        with pytest.raises(TypeError, match="symbolic Qamomile handle"):
            bad_chained_comparison.build(value=1)

    @pytest.mark.parametrize("loop_kind", ["for", "while"])
    def test_return_inside_loop_is_rejected(self, loop_kind):
        """A return that cannot preserve Python loop-exit semantics is rejected."""
        if loop_kind == "for":
            with pytest.raises(SyntaxError, match="return.*for loop"):

                @qkernel
                def bad_for_return() -> qmc.Bit:
                    q = qmc.qubit()
                    for _ in qmc.range(2):
                        q = qmc.x(q)
                        return qmc.measure(q)
                    return qmc.measure(q)

        else:
            with pytest.raises(SyntaxError, match="return.*while loop"):

                @qkernel
                def bad_while_return(run: qmc.UInt) -> qmc.Bit:
                    q = qmc.qubit()
                    while run:
                        q = qmc.x(q)
                        return qmc.measure(q)
                    return qmc.measure(q)

    @pytest.mark.parametrize("keyword", ["break", "continue"])
    @pytest.mark.parametrize("loop_kind", ["for", "while"])
    def test_loop_control_has_qkernel_specific_diagnostic(
        self,
        keyword: str,
        loop_kind: str,
    ) -> None:
        """Unsupported loop control never reports 'outside loop'."""
        source = f"""
        def invalid(run: qmc.UInt) -> qmc.UInt:
            {"for _ in qmc.range(2):" if loop_kind == "for" else "while run:"}
                {keyword}
            return run
        """
        with pytest.raises(
            SyntaxError, match=rf"{keyword}.*{loop_kind}.*not supported"
        ):
            ControlFlowTransformer().visit(ast.parse(textwrap.dedent(source)))

    def test_lazy_block_build_is_thread_safe(self, monkeypatch) -> None:
        """Concurrent first access shares one block without false recursion."""
        from qamomile.circuit.frontend import qkernel_block

        @qkernel
        def kernel() -> qmc.Bit:
            return qmc.measure(qmc.qubit("q"))

        original = qkernel_block.func_to_block
        started = threading.Event()
        release = threading.Event()
        build_count = 0

        def delayed_build(func):
            """Hold the sole block build until both workers overlap."""
            nonlocal build_count
            build_count += 1
            started.set()
            assert release.wait(timeout=5.0)
            return original(func)

        monkeypatch.setattr(qkernel_block, "func_to_block", delayed_build)
        blocks = []
        errors = []

        def access_block() -> None:
            """Read the shared lazy block from one worker."""
            try:
                blocks.append(kernel.block)
            except Exception as error:  # pragma: no cover - assertion payload
                errors.append(error)

        first = threading.Thread(target=access_block)
        second = threading.Thread(target=access_block)
        first.start()
        assert started.wait(timeout=5.0)
        second.start()
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)

        assert not first.is_alive() and not second.is_alive()
        assert errors == []
        assert len(blocks) == 2 and blocks[0] is blocks[1]
        assert build_count == 1

    @pytest.mark.parametrize("loop_kind", ["if", "while"])
    def test_walrus_in_condition_is_rejected(self, loop_kind):
        """A walrus target cannot be moved into a generated helper scope."""
        if loop_kind == "if":
            with pytest.raises(SyntaxError, match="Assignment expressions"):

                @qkernel
                def bad_if_walrus() -> qmc.Bit:
                    q = qmc.qubit()
                    bit = qmc.bit(False)
                    if bit := qmc.measure(q):
                        pass
                    return bit

        else:
            with pytest.raises(SyntaxError, match="Assignment expressions"):

                @qkernel
                def bad_while_walrus() -> qmc.Bit:
                    q = qmc.qubit()
                    bit = qmc.bit(True)
                    while bit := qmc.measure(q):
                        pass
                    return bit


class TestLivePythonNameResolution:
    """Transformed kernels retain Python's build-time name lookup behavior."""

    def test_forward_defined_module_helper_is_resolved_at_build(self):
        """A helper defined after its caller is available during lazy tracing."""
        block = _kernel_using_forward_helper.build()
        assert any(
            isinstance(op, InvokeOperation) and op.target.name == "_forward_helper"
            for op in block.operations
        )

    def test_rebound_module_global_is_refreshed_at_build(self):
        """A module constant rebound after decoration uses its current value."""
        block = _kernel_using_live_global.build()
        rotation = next(
            op
            for op in block.operations
            if getattr(op, "gate_type", None) == GateOperationType.RZ
        )
        assert rotation.operands[1].metadata.scalar.const_value == 1.25


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

        collector = VariableCollector(global_names={"qmc"})
        collector.visit(tree)

        assert "qs" in collector.vars
        assert "qs" in collector.load_vars
        assert collector._first_context["qs"] == "Load"

    def test_module_attribute_access_still_excludes_module_name(self):
        """Module attribute access must not treat the module name as a variable."""
        tree = ast.parse(
            textwrap.dedent(
                """
                q = qmc.h(q)
                """
            )
        )

        collector = VariableCollector(global_names={"qmc"})
        collector.visit(tree)

        assert "qmc" not in collector.vars
        assert "qmc" not in collector.load_vars
        assert "q" in collector.load_vars


class TestEmptyClosureCell:
    """Empty closure cells should now fail closed at decoration time."""

    def test_empty_cell_raises_frontend_transform_error(self):
        """Forward-referenced freevars should raise a dedicated transform error."""

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qmc.h(q)
                return q

            def helper(q: Qubit) -> qmc.Bit:
                return qmc.measure(q)

            return circuit

        with pytest.raises(
            FrontendTransformError, match="Closure variable 'helper' is not yet bound"
        ):
            factory()

    def test_empty_cell_does_not_fallback_with_warning(self):
        """Fail-closed mode should raise instead of warning-and-fallback."""

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qmc.h(q)
                return q

            def helper(q: Qubit) -> qmc.Bit:
                return qmc.measure(q)

            return circuit

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with pytest.raises(FrontendTransformError):
                factory()

        fallback_warnings = [
            w for w in ws if "AST transformation failed" in str(w.message)
        ]
        assert fallback_warnings == []

    def test_empty_cell_never_reaches_build(self):
        """The kernel should be rejected before any build-time NameError is possible."""

        def factory():
            @qkernel
            def circuit(q: Qubit) -> Qubit:
                while helper(q):
                    q = qmc.h(q)
                return q

            def helper(q: Qubit) -> qmc.Bit:
                return qmc.measure(q)

            return circuit

        with pytest.raises(
            FrontendTransformError, match="Closure variable 'helper' is not yet bound"
        ):
            factory()

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
        for stmt in func_def.body:
            if isinstance(stmt, ast.With):
                loop_stmt = stmt
                break
            if (
                isinstance(stmt, ast.If)
                and isinstance(stmt.test, ast.Call)
                and isinstance(stmt.test.func, ast.Name)
                and stmt.test.func.id == "should_trace_for_loop"
            ):
                loop_stmt = next(
                    inner for inner in stmt.body if isinstance(inner, ast.With)
                )
                break
        else:
            raise AssertionError("Expected transformed loop body")
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
            def bad(n: qmc.UInt, x: qmc.UInt) -> qmc.UInt:
                for i in qmc.range(n):
                    x = x + 1
                else:
                    x = qmc.UInt(0)
                return x

    def test_qkernel_while_else_raises(self):
        """@qkernel decorator propagates the SyntaxError."""
        with pytest.raises(
            SyntaxError, match="while ... else is not supported in @qkernel"
        ):

            @qkernel
            def bad(cond: qmc.Bit, x: qmc.UInt) -> qmc.UInt:
                while cond:
                    x = x + 1
                else:
                    x = qmc.UInt(0)
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
                    d: qmc.Dict[
                        qmc.Tuple[qmc.UInt, qmc.Tuple[qmc.UInt, qmc.UInt]], qmc.Float
                    ],
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

    def test_items_module_call_keyword_args_raises(self):
        """qmc.items(d=edges) must raise SyntaxError (keyword args not supported)."""
        with pytest.raises(SyntaxError, match="does not support keyword arguments"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for k, v in qmc.items(d=edges):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_items_module_call_extra_args_raises(self):
        """qmc.items(edges, extra) must raise SyntaxError (exactly 1 arg required)."""
        with pytest.raises(SyntaxError, match="requires exactly one dict argument"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(
                    edges: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
                    other: qmc.Dict[qmc.UInt, qmc.Float],
                ) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for k, v in qmc.items(edges, other):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_range_tuple_unpack_raises(self):
        """for (i, j) in qmc.range(n) must raise SyntaxError."""
        with pytest.raises(
            SyntaxError, match="qmc.range.*requires a single loop variable"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(n: qmc.UInt) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for i, j in qmc.range(n):
                        q[0] = qmc.h(q[0])
                    return q[0]

    def test_range_keyword_args_raises(self):
        """qmc.range(stop=n) must raise SyntaxError (keyword args not supported)."""
        with pytest.raises(SyntaxError, match="does not support keyword arguments"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")

                @qmc.qkernel
                def bad(n: qmc.UInt) -> qmc.Qubit:
                    q = qmc.qubit_array(2, "q")
                    for i in qmc.range(stop=n):
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
