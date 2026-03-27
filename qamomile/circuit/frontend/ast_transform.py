import ast
import inspect
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from qamomile.circuit.frontend.operation.control_flow import (
    emit_if,
    for_items,
    for_loop,
    while_loop,
)


class VariableCollector(ast.NodeVisitor):
    """ブロック内で使用・変更される変数を収集します。

    以下を除外:
    - 関数呼び出しの関数名 (func in Call)
    - 属性アクセスのグローバルなベースオブジェクト (value in Attribute)
    - グローバル変数（モジュール、組み込み関数など）
    """

    def __init__(self, global_names: set[str] | None = None):
        self.vars = set()
        self._exclude = set()  # 除外する名前
        self._global_names = global_names or set()
        self._first_context: dict[str, str] = {}  # name -> "Store" | "Load"
        self._load_names: set[str] = set()
        self._store_names: set[str] = set()

    def visit_Call(self, node: ast.Call):
        """関数呼び出しの関数名を除外"""
        if isinstance(node.func, ast.Name):
            # 直接関数呼び出し: func()
            self._exclude.add(node.func.id)
        # 引数は通常通り処理
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        # node.func が属性アクセスの場合は visit_Attribute で処理
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func)

    def visit_Attribute(self, node: ast.Attribute):
        """属性アクセスのベース名を記録する。

        モジュール名などグローバル名 (`qm.h`) は従来どおり除外し、
        ユーザー変数 (`qs.shape`) は Load として扱う。
        """
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self._global_names:
                # qm.x の qm を除外
                self._exclude.add(name)
            else:
                self.vars.add(name)
                self._load_names.add(name)
                if name not in self._first_context:
                    self._first_context[name] = "Load"
        else:
            # ネストした属性アクセス (a.b.c) の場合は再帰
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign):
        """右辺を先に走査し、Python の評価順序に合わせる。

        `q1 = qm.h(q1)` → 右辺 q1 (Load) が先 → first_context は "Load"
        `cond2 = qm.measure(q2)` → 右辺 q2 (Load) が先、左辺 cond2 (Store) が後
        """
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        """AugAssign (e.g. x += 1) は暗黙の Read-before-Write。

        右辺を先に走査し、Name ターゲットは Load + Store として記録する。
        first_context は "Load"（既存値の読み出しが先行するため）。
        """
        self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            name = target.id
            if name not in self._exclude and name not in self._global_names:
                self.vars.add(name)
                self._load_names.add(name)
                self._store_names.add(name)
                if name not in self._first_context:
                    self._first_context[name] = "Load"
        else:
            # Subscript / Attribute: visit normally to capture base/index loads
            self.visit(target)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """内部関数定義の走査をスキップする。"""
        pass

    def visit_Name(self, node: ast.Name):
        """変数名を収集（除外リストにないもののみ）"""
        if isinstance(node.id, str):
            if node.id not in self._exclude and node.id not in self._global_names:
                self.vars.add(node.id)
                if isinstance(node.ctx, ast.Load):
                    self._load_names.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    self._store_names.add(node.id)
                if node.id not in self._first_context:
                    if isinstance(node.ctx, ast.Store):
                        self._first_context[node.id] = "Store"
                    else:
                        self._first_context[node.id] = "Load"

    @property
    def locally_defined_vars(self) -> set[str]:
        """このスコープ内で初めて定義（Store）される変数。"""
        return {name for name, ctx in self._first_context.items() if ctx == "Store"}

    @property
    def incoming_vars(self) -> set[str]:
        """外部スコープから渡される必要がある変数（初回が Load）。"""
        return self.vars - self.locally_defined_vars

    @property
    def load_vars(self) -> set[str]:
        """Load コンテキストで参照される変数（実際に読まれる変数）。"""
        return self._load_names & self.vars

    @property
    def store_vars(self) -> set[str]:
        """Store コンテキストで代入される変数。"""
        return self._store_names & self.vars


class ControlFlowTransformer(ast.NodeTransformer):
    while_func_name = "emit_while"
    if_func_name = "emit_if"
    for_func_name = "emit_for"

    def __init__(
        self,
        global_names: set[str] | None = None,
        param_names: set[str] | None = None,
        namespace: dict[str, Any] | None = None,
    ) -> None:
        self.counter: int = 0
        # 変数名 -> 型注釈ノード(ast.AST) を保持する辞書
        self.type_registry: dict[str, ast.AST] = {}
        # グローバル変数名（モジュール、組み込み関数など）
        self._global_names = global_names or set()
        # Scope tracking for visit_If input/output separation
        self._outer_defined_vars: frozenset[str] = frozenset()
        self._after_stmt_read_vars: frozenset[str] = frozenset()
        self._after_stmt_load_vars: frozenset[str] = frozenset()
        # 関数パラメータ名（ループ変数シャドウイング検出用）
        self._param_names = param_names or set()
        # Namespace for resolving callables (used in while condition QKernel check)
        self._namespace = namespace or {}

    def _get_unique_name(self, prefix: str) -> str:
        name = f"_{prefix}_{self.counter}"
        self.counter += 1
        return name

    def _visit_body_with_tracking(
        self,
        body: list[ast.stmt],
        initial_defined: set[str],
        outer_after_loads: frozenset[str] = frozenset(),
    ) -> list[ast.stmt]:
        """Process a statement body sequentially, tracking defined variables.

        Before visiting each statement, sets ``_outer_defined_vars`` (variables
        defined before the statement) and ``_after_stmt_read_vars`` (variables
        referenced in subsequent statements) so that ``visit_If`` can compute
        input/output variable sets correctly.

        Args:
            outer_after_loads: Variables that are loaded after this entire body
                completes.  Seeded into the suffix-load array so that nested
                ``visit_If`` calls can see outer-scope liveness.
        """
        new_body: list[ast.stmt] = []
        defined_so_far = set(initial_defined)

        # Precompute per-statement read sets and locally-defined sets (O(n))
        per_stmt_collectors: list[VariableCollector] = []
        for stmt in body:
            c = VariableCollector(global_names=self._global_names)
            c.visit(stmt)
            per_stmt_collectors.append(c)

        # Build suffix-union of read vars in reverse (O(n))
        n = len(body)
        suffix_reads: list[frozenset[str]] = [frozenset()] * (n + 1)
        for j in range(n - 1, -1, -1):
            suffix_reads[j] = frozenset(
                per_stmt_collectors[j].vars | suffix_reads[j + 1]
            )

        # Build suffix liveness using the same statement-aware recurrence as
        # ``_stmt_live_in`` so control-flow statements preserve values that
        # may survive along one-sided or zero-iteration paths.
        suffix_loads: list[frozenset[str]] = [frozenset()] * (n + 1)
        suffix_loads[n] = outer_after_loads
        for j in range(n - 1, -1, -1):
            suffix_loads[j] = frozenset(
                self._stmt_live_in(body[j], set(suffix_loads[j + 1]))
            )

        for i, stmt in enumerate(body):
            # Set context for visit_If
            self._outer_defined_vars = frozenset(defined_so_far)
            self._after_stmt_read_vars = suffix_reads[i + 1]
            self._after_stmt_load_vars = suffix_loads[i + 1]

            # Visit the statement (may call visit_If, visit_For, etc.)
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)

            # Update defined vars from original statement
            defined_so_far |= per_stmt_collectors[i].locally_defined_vars

        return new_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process the function body with definition tracking.

        Collects parameter names as the initial set of defined variables
        and delegates to ``_visit_body_with_tracking`` for sequential
        statement processing.
        """
        param_names = {arg.arg for arg in node.args.args}
        node.body = self._visit_body_with_tracking(node.body, param_names)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """
        a: int = 0 のような型注釈付き代入を検出し、型情報を登録します。
        """
        if isinstance(node.target, ast.Name):
            self.type_registry[node.target.id] = node.annotation
        return self.generic_visit(node)

    def _collect_variables(self, nodes: list[ast.AST] | ast.AST) -> list[str]:
        collector = VariableCollector(global_names=self._global_names)
        if isinstance(nodes, list):
            for node in nodes:
                collector.visit(node)
        else:
            collector.visit(nodes)
        # 登録済みの型がある変数、または出現した変数を対象とする
        # グローバル変数（モジュール、組み込み関数など）は除外される
        return sorted(list(collector.vars))

    def _get_annotation(self, var_name: str) -> ast.AST | None:
        """登録済みの型情報を取得します（ディープコピーして返す）"""
        if var_name in self.type_registry:
            # ASTノードは使い回すとlocation情報などでバグりやすいためコピー推奨ですが、
            # ここでは簡易的に参照を返します（unparse時は問題なし）
            return self.type_registry[var_name]
        return None

    def _make_arguments(self, var_names: list[str]) -> ast.arguments:
        """型注釈付きの引数リストを作成"""
        args_list = []
        for name in var_names:
            # 型情報の取得
            annotation = self._get_annotation(name)
            args_list.append(ast.arg(arg=name, annotation=annotation))  # type: ignore

        return ast.arguments(
            posonlyargs=[],
            args=args_list,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

    def _create_return_annotation(self, var_names: list[str]) -> ast.AST | None:
        """戻り値の型注釈を作成 (例: int や tuple[int, int])"""
        if not var_names:
            return ast.Constant(value=None)

        # 変数が1つならその型を返す
        if len(var_names) == 1:
            return self._get_annotation(var_names[0])

        # 変数が複数の場合: tuple[Type1, Type2] を作成
        # ast.Subscript(value=Name(id='tuple'), slice=Tuple(elts=[...]))
        elts: list[ast.expr] = []
        for name in var_names:
            ann = self._get_annotation(name)
            # 型情報がない場合は Any として扱うか、Noneを入れる
            if ann is None:
                elts.append(ast.Name(id="Any", ctx=ast.Load()))
            else:
                # _get_annotation returns ast.AST but we know it's ast.expr in practice
                elts.append(ann)  # type: ignore[arg-type]

        return ast.Subscript(
            value=ast.Name(id="tuple", ctx=ast.Load()),
            slice=ast.Tuple(elts=elts, ctx=ast.Load()),
            ctx=ast.Load(),
        )

    def _create_return_node(self, var_names: list[str]) -> ast.Return:
        if len(var_names) == 0:
            return ast.Return(value=ast.Constant(value=None))
        elif len(var_names) == 1:
            return ast.Return(value=ast.Name(id=var_names[0], ctx=ast.Load()))
        else:
            return ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=v, ctx=ast.Load()) for v in var_names],
                    ctx=ast.Load(),
                )
            )

    def _create_assignment_node(
        self, var_names: list[str], value_node: ast.expr, lineno: int
    ) -> ast.stmt:
        if not var_names:
            return ast.Expr(value=value_node, lineno=lineno)

        target: ast.expr
        if len(var_names) == 1:
            target = ast.Name(id=var_names[0], ctx=ast.Store())
        else:
            target = ast.Tuple(
                elts=[ast.Name(id=v, ctx=ast.Store()) for v in var_names],
                ctx=ast.Store(),
            )
        return ast.Assign(targets=[target], value=value_node, lineno=lineno)

    def _create_inner_func(
        self,
        name_prefix: str,
        body_nodes: list[ast.stmt],
        var_names: list[str],
        lineno: int,
        return_var_names: list[str] | None = None,
        return_type_ast: ast.AST | None = None,
    ) -> tuple[ast.FunctionDef, str]:
        func_name = self._get_unique_name(name_prefix)
        func_args = self._make_arguments(var_names)

        ret_vars = return_var_names if return_var_names is not None else var_names
        new_body = list(body_nodes)
        new_body.append(self._create_return_node(ret_vars))

        # 戻り値の型注釈が指定されていなければ自動生成
        if return_type_ast is None:
            return_type_ast = self._create_return_annotation(ret_vars)

        func_def = ast.FunctionDef(
            name=func_name,
            args=func_args,
            body=new_body,
            decorator_list=[],
            returns=return_type_ast,  # type: ignore
            lineno=lineno,
            col_offset=0,
        )  # type: ignore
        return func_def, func_name

    # Authoritative spec: callable names prohibited in while conditions.
    # Must match implemented quantum operations in
    # qamomile/circuit/frontend/operation/{qubit_gates,measurement,expval}.py.
    _QUANTUM_OPS = frozenset(
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

    def _check_no_quantum_ops_in_condition(
        self, test_node: ast.expr, lineno: int
    ) -> None:
        """Detect quantum operations in while condition and raise SyntaxError.

        Uses resolved callable identity to distinguish Qamomile quantum
        operations from classical functions that happen to share the same name.
        Unresolvable callables are allowed (fail-open).
        """
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                # Resolve callable in namespace to check identity
                resolved = self._resolve_callable(node.func)
                if resolved is None:
                    # Unresolvable: fail-open (allow)
                    continue
                if self._is_qkernel(resolved):
                    display = func_name or "<callable>"
                    raise SyntaxError(
                        f"Quantum kernel '{display}' in while condition "
                        f"is not supported. @qkernel functions consume "
                        f"quantum arguments, which is not allowed in "
                        f"while conditions."
                    )
                if self._is_qamomile_quantum_op(resolved):
                    display = func_name or "<callable>"
                    raise SyntaxError(
                        f"Quantum operation '{display}' in while condition "
                        f"is not supported. Move the operation before the loop "
                        f"and use the result as condition."
                    )

    @staticmethod
    def _is_qamomile_quantum_op(obj: Any) -> bool:
        """Check if *obj* is a Qamomile quantum operation by module and name."""
        module = getattr(obj, "__module__", None)
        name = getattr(obj, "__name__", None)
        if module is None or name is None:
            return False
        return (
            module.startswith("qamomile.circuit.frontend.operation")
            and name in ControlFlowTransformer._QUANTUM_OPS
        )

    @staticmethod
    def _is_qkernel(obj: Any) -> bool:
        """Check if *obj* is a QKernel instance using type identity.

        Uses a lazy import to avoid circular dependency
        (qkernel.py imports transform_control_flow from this module).
        """
        from qamomile.circuit.frontend.qkernel import QKernel

        return isinstance(obj, QKernel)

    def _resolve_callable(self, func_node: ast.expr) -> Any | None:
        """Try to resolve a callable AST node in the namespace.

        Returns the resolved object, or None if unresolvable.
        """
        if not self._namespace:
            return None
        if isinstance(func_node, ast.Name):
            return self._namespace.get(func_node.id)
        if isinstance(func_node, ast.Attribute) and isinstance(
            func_node.value, ast.Name
        ):
            base = self._namespace.get(func_node.value.id)
            if base is not None:
                return getattr(base, func_node.attr, None)
        return None

    @staticmethod
    def _has_top_level_return(body_nodes: list[ast.stmt]) -> bool:
        """Check if any top-level statement in body_nodes is a Return with a value."""
        return any(
            isinstance(stmt, ast.Return) and stmt.value is not None
            for stmt in body_nodes
        )

    @staticmethod
    def _has_return_under_loop(body_nodes: list[ast.stmt]) -> bool:
        """Check if any return statement exists inside a for/while loop in body_nodes."""
        for stmt in body_nodes:
            if not isinstance(stmt, (ast.For, ast.While)):
                continue
            for node in ast.walk(stmt):
                if isinstance(node, ast.Return) and node.value is not None:
                    return True
        return False

    @staticmethod
    def _transform_returns_to_assignments(
        body_nodes: list[ast.stmt], ret_var_name: str
    ) -> list[ast.stmt]:
        """Replace top-level ``return expr`` with ``ret_var_name = expr``."""
        new_body: list[ast.stmt] = []
        for stmt in body_nodes:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                assign = ast.Assign(
                    targets=[ast.Name(id=ret_var_name, ctx=ast.Store())],
                    value=stmt.value,
                    lineno=stmt.lineno,
                    col_offset=getattr(stmt, "col_offset", 0),
                )
                new_body.append(assign)
            else:
                new_body.append(stmt)
        return new_body

    def _collect_load_vars_set(self, node: ast.AST) -> set[str]:
        collector = VariableCollector(global_names=self._global_names)
        collector.visit(node)
        return set(collector.load_vars)

    def _stmt_live_in(self, stmt: ast.stmt, live_out: set[str]) -> set[str]:
        """Compute variables that must be live before executing *stmt*."""
        if isinstance(stmt, ast.If):
            test_loads = self._collect_load_vars_set(stmt.test)
            true_live_in = self._block_live_in(stmt.body, live_out)
            false_live_in = (
                self._block_live_in(stmt.orelse, live_out)
                if stmt.orelse
                else set(live_out)
            )
            return test_loads | true_live_in | false_live_in

        if isinstance(stmt, ast.While):
            post_loop_live = (
                self._block_live_in(stmt.orelse, live_out)
                if stmt.orelse
                else set(live_out)
            )
            cond_loads = self._collect_load_vars_set(stmt.test)
            body_entry_live_in = self._loop_body_entry_live_in(
                stmt.body, post_loop_live | cond_loads
            )
            return post_loop_live | cond_loads | body_entry_live_in

        if isinstance(stmt, ast.For):
            post_loop_live = (
                self._block_live_in(stmt.orelse, live_out)
                if stmt.orelse
                else set(live_out)
            )
            iter_loads = self._collect_load_vars_set(stmt.iter)
            body_entry_live_in = self._loop_body_entry_live_in(
                stmt.body,
                post_loop_live,
                bound_vars=set(self._extract_tuple_vars(stmt.target)),
            )
            return post_loop_live | iter_loads | body_entry_live_in

        collector = VariableCollector(global_names=self._global_names)
        collector.visit(stmt)
        return set(collector.load_vars) | (set(live_out) - set(collector.store_vars))

    def _block_live_in(self, body: list[ast.stmt], live_out: set[str]) -> set[str]:
        live = set(live_out)
        for stmt in reversed(body):
            live = self._stmt_live_in(stmt, live)
        return live

    def _loop_body_entry_live_in(
        self,
        body: list[ast.stmt],
        exit_live: set[str],
        bound_vars: set[str] | None = None,
    ) -> set[str]:
        """Compute loop self-edge liveness at the start of the loop body."""
        bound = set(bound_vars or ())
        live = set()
        while True:
            next_live = self._block_live_in(body, set(exit_live) | live)
            next_live.difference_update(bound)
            if next_live == live:
                return next_live
            live = next_live

    def visit_While(self, node: ast.While) -> Any:
        if node.orelse:
            raise SyntaxError("while ... else is not supported in @qkernel")
        # Check for quantum operations in while condition
        self._check_no_quantum_ops_in_condition(node.test, node.lineno)
        # ネストされた制御フローを先に変換 (with definition tracking)
        saved_outer = self._outer_defined_vars
        saved_after = self._after_stmt_read_vars
        saved_after_load = self._after_stmt_load_vars
        # The while condition is re-evaluated each iteration, so its
        # load vars must be live at the end of each iteration body.
        # Also propagate loop back-edge liveness. VariableCollector.incoming_vars
        # is lexical and misses mixed-path Load-before-Store cases at loop entry.
        cond_collector = VariableCollector(global_names=self._global_names)
        cond_collector.visit(node.test)
        cond_loads = set(cond_collector.load_vars)
        body_entry_live_in = self._loop_body_entry_live_in(
            node.body, set(saved_after_load) | cond_loads
        )
        while_body_after = frozenset(
            set(saved_after_load) | cond_loads | body_entry_live_in
        )
        flattened_body = self._visit_body_with_tracking(
            node.body,
            set(self._outer_defined_vars),
            outer_after_loads=while_body_after,
        )
        self._outer_defined_vars = saved_outer
        self._after_stmt_read_vars = saved_after
        self._after_stmt_load_vars = saved_after_load

        # lambda: <condition> を作成
        lambda_node = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=node.test,
        )

        # while_loop(lambda: cond) コールを作成
        while_loop_call = ast.Call(
            func=ast.Name(id="while_loop", ctx=ast.Load()),
            args=[lambda_node],
            keywords=[],
        )

        # with文を作成
        with_item = ast.withitem(context_expr=while_loop_call, optional_vars=None)
        with_stmt = ast.With(
            items=[with_item],
            body=flattened_body,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return with_stmt

    def _is_range_call(self, node: ast.expr) -> bool:
        """Check if node is a range() or qm.range() call."""
        if not isinstance(node, ast.Call):
            return False

        # range(n) - built-in range
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            return True

        # qm.range(n) - qamomile's symbolic range
        if isinstance(node.func, ast.Attribute) and node.func.attr == "range":
            return True

        return False

    def _is_items_call(self, node: ast.expr) -> bool:
        """Check if node is an items() or qm.items() call."""
        if not isinstance(node, ast.Call):
            return False

        # items(d) - direct call
        if isinstance(node.func, ast.Name) and node.func.id == "items":
            return True

        # qm.items(d) or qmc.items(d) - module call
        if isinstance(node.func, ast.Attribute) and node.func.attr == "items":
            return True

        return False

    def _extract_tuple_vars(self, target: ast.expr) -> list[str]:
        """Extract variable names from tuple/list unpacking target.

        Examples:
            i -> ["i"]
            (i, j) -> ["i", "j"]
            ((i, j), k) -> ["i", "j", "k"]
            [i, j] -> ["i", "j"]
        """
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, (ast.Tuple, ast.List)):
            var_names = []
            for elt in target.elts:
                var_names.extend(self._extract_tuple_vars(elt))
            return var_names
        else:
            raise SyntaxError(
                f"Unsupported loop target type in @qkernel: {ast.dump(target)}"
            )

    def _validate_for_loop(self, node: ast.For) -> list[str]:
        """Validate a for-loop node and return binding variable names.

        Checks both the loop target and the iterator call arguments per
        loop kind (items / range / sequence).  Raises ``SyntaxError`` for
        invalid patterns so that ``QKernel.__init__`` propagates them
        instead of falling back to the original function.
        """
        if self._is_items_call(node.iter):
            call = node.iter  # type: ignore
            # from qamomile.circuit import items; items(d) form: require exactly 1 positional arg
            if isinstance(call.func, ast.Name):
                if call.keywords:
                    raise SyntaxError(
                        "items() does not support keyword arguments in @qkernel; "
                        "use items(d)."
                    )
                if len(call.args) != 1:
                    raise SyntaxError(
                        "items() requires exactly one dict argument: items(d)"
                    )
            # Attribute form: distinguish qmc.items(d) vs d.items() by receiver identity
            elif isinstance(call.func, ast.Attribute):
                func_value = call.func.value
                if (
                    isinstance(func_value, ast.Name)
                    and func_value.id in self._param_names
                ):
                    # d.items() form: dict.items() takes no arguments
                    if call.args or call.keywords:
                        raise SyntaxError(
                            "items() iteration over a dict parameter must use "
                            "'d.items()' with no arguments."
                        )
                elif (
                    isinstance(func_value, ast.Name)
                    and func_value.id in self._global_names
                ):
                    # module.items(d) form: require exactly 1 positional arg, no keywords
                    alias = func_value.id
                    if call.keywords:
                        raise SyntaxError(
                            f"items() does not support keyword arguments in @qkernel; "
                            f"use {alias}.items(d)."
                        )
                    if len(call.args) != 1:
                        raise SyntaxError(
                            f"items() requires exactly one dict argument: {alias}.items(d)"
                        )
                else:
                    # Non-global receiver (local variable, attribute, etc.):
                    # treat as method call like d.items().
                    if call.args or call.keywords:
                        raise SyntaxError(
                            "items() iteration over a dict must use "
                            "'d.items()' with no arguments."
                        )

            # items(): target must be a 2-element tuple (key, value)
            if not (isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2):
                raise SyntaxError(
                    "items() iteration requires 'for key, value in items(d)' or 'for key, value in d.items()' pattern. "
                    f"Got: {ast.dump(node.target)}"
                )
            key_target = node.target.elts[0]
            value_target = node.target.elts[1]
            # key may be a tuple like (i, j) or a simple Name
            if not isinstance(key_target, (ast.Name, ast.Tuple)):
                raise SyntaxError(
                    "Key target in items() iteration must be a variable or tuple of variables, "
                    f"got: {ast.dump(key_target)}"
                )
            # reject nested tuple unpacking in key, e.g. (i, (j, k))
            if isinstance(key_target, ast.Tuple) and not all(
                isinstance(e, ast.Name) for e in key_target.elts
            ):
                raise SyntaxError(
                    "Nested tuple unpacking in items() key is not supported. "
                    f"Use a flat tuple like (i, j) instead, got: {ast.dump(key_target)}"
                )
            # value must be a simple Name
            if not isinstance(value_target, ast.Name):
                raise SyntaxError(
                    "Value target in items() iteration must be a simple variable, "
                    f"got: {ast.dump(value_target)}"
                )
            binding_names = self._extract_tuple_vars(node.target)

        elif self._is_range_call(node.iter):
            range_call = node.iter  # type: ignore
            # Derive the caller name from the AST for error messages
            if isinstance(range_call.func, ast.Attribute) and isinstance(
                range_call.func.value, ast.Name
            ):
                range_label = f"{range_call.func.value.id}.range()"
            else:
                range_label = "range()"
            num_args = len(range_call.args)
            # Keyword arguments (e.g. range(stop=n)) are not consumed by
            # _transform_for_range; reject them explicitly to avoid silent loss.
            if getattr(range_call, "keywords", None):
                raise SyntaxError(
                    f"{range_label} does not support keyword arguments in @qkernel; "
                    "use positional arguments like range(stop) or range(start, stop, step)."
                )
            # range(): requires 1-3 positional arguments
            if num_args < 1 or num_args > 3:
                raise SyntaxError(
                    f"{range_label} requires 1-3 arguments: range(stop), range(start, stop), or range(start, stop, step)"
                )
            # range(): target must be a single variable
            if not isinstance(node.target, ast.Name):
                raise SyntaxError(
                    f"{range_label} iteration requires a single loop variable, "
                    f"got: {ast.dump(node.target)}"
                )
            binding_names = [node.target.id]

        else:
            # sequence: will be rejected by _transform_for_sequence
            binding_names = self._extract_tuple_vars(node.target)

        # Check for parameter shadowing (common to all for-loop variants)
        for var_name in binding_names:
            if var_name in self._param_names:
                raise SyntaxError(
                    f"Loop variable '{var_name}' shadows a function parameter. "
                    f"Use a different variable name to avoid silent value loss."
                )

        return binding_names

    def visit_For(self, node: ast.For) -> Any:
        if node.orelse:
            raise SyntaxError("for ... else is not supported in @qkernel")

        # Validate target per loop-kind BEFORE extracting binding names.
        # This raises SyntaxError for invalid placeholder-loop targets
        # instead of generic NotImplementedError that would be swallowed
        # by QKernel.__init__'s fallback.
        all_binding_names = self._validate_for_loop(node)

        # ネストされた制御フローを先に変換 (with definition tracking)
        saved_outer = self._outer_defined_vars
        saved_after = self._after_stmt_read_vars
        saved_after_load = self._after_stmt_load_vars
        initial_defined = set(self._outer_defined_vars)
        bound_vars = set(all_binding_names)
        initial_defined.update(bound_vars)
        body_entry_live_in = self._loop_body_entry_live_in(
            node.body, set(saved_after_load), bound_vars=bound_vars
        )
        flattened_body = self._visit_body_with_tracking(
            node.body,
            initial_defined,
            outer_after_loads=frozenset(set(saved_after_load) | body_entry_live_in),
        )
        self._outer_defined_vars = saved_outer
        self._after_stmt_read_vars = saved_after
        self._after_stmt_load_vars = saved_after_load

        # Check for items() iteration first
        if self._is_items_call(node.iter):
            return self._transform_for_items(node, flattened_body)

        # Check for range() iteration
        if self._is_range_call(node.iter):
            return self._transform_for_range(node, flattened_body)

        # Handle direct sequence iteration (for item in seq:)
        return self._transform_for_sequence(node, flattened_body)

    def _transform_for_range(
        self, node: ast.For, flattened_body: list[ast.stmt]
    ) -> ast.With:
        """Transform 'for i in range(...)' to 'with for_loop(...)'.

        Supports patterns:
            for i in range(stop):  ->  for_loop(0, stop, 1)
            for i in range(start, stop):  ->  for_loop(start, stop, 1)
            for i in range(start, stop, step):  ->  for_loop(start, stop, step)
        """

        # range の引数を取得 (_validate_for_loop で引数数は検証済み)
        num_args = len(node.iter.args)  # type: ignore
        # range(stop) -> for_loop(0, stop, 1)
        # range(start, stop) -> for_loop(start, stop, 1)
        # range(start, stop, step) -> for_loop(start, stop, step)
        if num_args == 1:
            start_arg = ast.Constant(value=0)
            stop_arg = node.iter.args[0]  # type: ignore
            step_arg = ast.Constant(value=1)
        elif num_args == 2:
            start_arg = node.iter.args[0]  # type: ignore
            stop_arg = node.iter.args[1]  # type: ignore
            step_arg = ast.Constant(value=1)
        else:  # num_args == 3
            start_arg = node.iter.args[0]  # type: ignore
            stop_arg = node.iter.args[1]  # type: ignore
            step_arg = node.iter.args[2]  # type: ignore

        # ループ変数名を取得 (_validate_for_loop で検証済み)
        loop_var_name = node.target.id

        # for_loop(start, stop, step, var_name) コールを作成
        for_loop_call = ast.Call(
            func=ast.Name(id="for_loop", ctx=ast.Load()),
            args=[start_arg, stop_arg, step_arg, ast.Constant(value=loop_var_name)],
            keywords=[],
        )

        # ループ変数を as i として設定
        with_item = ast.withitem(
            context_expr=for_loop_call,
            optional_vars=node.target,  # i をそのまま使用
        )

        with_stmt = ast.With(
            items=[with_item],
            body=flattened_body,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return with_stmt

    def _transform_for_sequence(
        self, node: ast.For, _flattened_body: list[ast.stmt]
    ) -> NoReturn:
        """Prohibit direct sequence iteration to prevent common bugs.

        Direct iteration like 'for item in vector:' doesn't support in-place
        modification in @qkernel contexts. This method raises an error to
        prevent silent bugs where reassignments don't affect the original array.

        Raises:
            SyntaxError: Always raised to prevent direct iteration.
        """
        # Get the source code representation if possible
        if isinstance(node.target, ast.Name):
            item_var = node.target.id
        else:
            item_var = "item"

        # Try to get a readable representation of the iterable
        iter_str = ast.unparse(node.iter) if hasattr(ast, "unparse") else "vector"

        raise SyntaxError(
            f"Direct iteration over sequences is not supported in @qkernel functions.\n"
            f"Line {node.lineno}: 'for {item_var} in {iter_str}:'\n\n"
            f"Direct iteration cannot modify elements in-place, leading to silent bugs.\n"
            f"Use explicit index-based iteration instead:\n\n"
            f"  # Incorrect (current code):\n"
            f"  for {item_var} in {iter_str}:\n"
            f"      {item_var} = qmc.operation({item_var})\n\n"
            f"  # Correct:\n"
            f"  n = {iter_str}.shape[0]\n"
            f"  for i in qmc.range(n):\n"
            f"      {iter_str}[i] = qmc.operation({iter_str}[i])\n"
        )

    def _transform_for_items(
        self, node: ast.For, flattened_body: list[ast.stmt]
    ) -> ast.With:
        """Transform 'for (k, v) in items(d)' or 'for (k, v) in d.items()' to 'with for_items(d, [...], "v")'.

        Supports patterns:
            for key, value in items(d):  ->  for_items(d, ["key"], "value")
            for key, value in d.items():  ->  for_items(d, ["key"], "value")
            for (i, j), value in items(d):  ->  for_items(d, ["i", "j"], "value")
            for (i, j), value in d.items():  ->  for_items(d, ["i", "j"], "value")
        """
        # Extract the dict argument from items(d) or d.items() call
        # (_validate_for_loop guarantees one of these patterns)
        if node.iter.args:  # type: ignore  # items(d) pattern
            dict_arg = node.iter.args[0]  # type: ignore
        else:  # d.items() pattern
            dict_arg = node.iter.func.value  # type: ignore

        # Parse the target pattern: (key, value) — validated by _validate_for_loop
        key_target = node.target.elts[0]
        value_target = node.target.elts[1]

        # Extract key variable names (may be tuple like (i, j))
        key_vars = self._extract_tuple_vars(key_target)

        # Extract value variable name (validated by _validate_for_loop)
        value_var = value_target.id

        # Create for_items(dict, key_vars, value_var) call
        for_items_call = ast.Call(
            func=ast.Name(id="for_items", ctx=ast.Load()),
            args=[
                dict_arg,
                ast.List(
                    elts=[ast.Constant(value=kv) for kv in key_vars],
                    ctx=ast.Load(),
                ),
                ast.Constant(value=value_var),
            ],
            keywords=[],
        )

        # Create with statement: with for_items(...) as (key, value):
        with_item = ast.withitem(
            context_expr=for_items_call,
            optional_vars=node.target,
        )

        with_stmt = ast.With(
            items=[with_item],
            body=flattened_body,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return with_stmt

    def visit_If(self, node: ast.If) -> Any:
        # 変換前のASTから変数を収集（generic_visit 後だと生成名が混入する）
        collector_test = VariableCollector(global_names=self._global_names)
        collector_test.visit(node.test)

        collector_body = VariableCollector(global_names=self._global_names)
        for stmt in node.body:
            collector_body.visit(stmt)

        collector_orelse = VariableCollector(global_names=self._global_names)
        if node.orelse:
            for stmt in node.orelse:
                collector_orelse.visit(stmt)

        # --- Input/output variable separation ---
        # input_vars: variables that exist before the if (passed to inner funcs)
        input_vars_set = (
            collector_test.vars
            | collector_body.incoming_vars
            | collector_orelse.incoming_vars
        )

        body_assigned = collector_body.locally_defined_vars
        orelse_assigned = collector_orelse.locally_defined_vars
        outer_defined = set(self._outer_defined_vars)
        has_else = bool(node.orelse)

        # Existing vars re-assigned in any branch may need phi outputs even
        # when the old value is dead and therefore should not be passed in.
        reassigned_existing = (body_assigned | orelse_assigned) & outer_defined

        # New locals defined in BOTH branches
        shared_new_locals = (
            ((body_assigned & orelse_assigned) - outer_defined) if has_else else set()
        )

        # One-sided new local rejection: reject only if read after the if
        if has_else:
            one_sided_new = (body_assigned ^ orelse_assigned) - outer_defined
        else:
            one_sided_new = body_assigned - outer_defined
        exposed_one_sided = one_sided_new & set(self._after_stmt_load_vars)
        if exposed_one_sided:
            raise SyntaxError(
                f"Variable(s) {sorted(exposed_one_sided)} defined in only one "
                f"branch of if-else at line {node.lineno} but used afterward.\n"
                f"Define in both branches, or move the definition outside the if."
            )

        after_loads = set(self._after_stmt_load_vars)
        live_in_vars = self._stmt_live_in(node, after_loads)
        live_existing_inputs = outer_defined & live_in_vars
        input_vars_set |= live_existing_inputs

        # Filter dead-and-modified variables from output.  A variable that is
        # stored (modified) in a branch but never loaded after the if would
        # generate an unnecessary PhiOp whose physical resources may differ
        # across branches, causing EmitError at emit time.
        # Variables that are dead but NOT stored in any branch pass through
        # unchanged (their phi has identical resources) and are harmless.
        stored_in_branches = collector_body.store_vars | collector_orelse.store_vars
        dead_modified = (stored_in_branches & input_vars_set) - after_loads
        live_shared = shared_new_locals & after_loads
        live_reassigned_existing = reassigned_existing & after_loads

        input_vars = sorted(input_vars_set)
        output_vars = sorted(
            (input_vars_set - dead_modified) | live_shared | live_reassigned_existing
        )

        # Detect return inside for/while in if-else branches (unsupported).
        # Must run before nested control flow transformation.
        if self._has_return_under_loop(node.body) or self._has_return_under_loop(
            node.orelse if node.orelse else []
        ):
            raise SyntaxError(
                "'return' inside for/while in @qkernel if-else is not supported"
            )

        # ネストされた制御フローを変換 (with definition tracking)
        saved_outer = self._outer_defined_vars
        saved_after = self._after_stmt_read_vars
        saved_after_load = self._after_stmt_load_vars
        inner_scope = set(input_vars)
        # Seed each branch with only the output vars that are actually stored
        # in that branch, plus the real downstream liveness from the enclosing
        # scope.  Pass-through variables (in output_vars but not stored in the
        # branch) don't need to flow through nested control-flow merges —
        # they're available as helper-function parameters.
        output_vars_set = set(output_vars)
        real_after = set(saved_after_load)
        outer_loads_body = frozenset((output_vars_set & body_assigned) | real_after)
        node.body = self._visit_body_with_tracking(
            node.body, inner_scope, outer_after_loads=outer_loads_body
        )
        if node.orelse:
            outer_loads_orelse = frozenset(
                (output_vars_set & orelse_assigned) | real_after
            )
            node.orelse = self._visit_body_with_tracking(
                node.orelse, inner_scope, outer_after_loads=outer_loads_orelse
            )
        self._outer_defined_vars = saved_outer
        self._after_stmt_read_vars = saved_after
        self._after_stmt_load_vars = saved_after_load

        # 明示的 return の検出と変換
        orelse_body = node.orelse if node.orelse else []
        body_has_return = self._has_top_level_return(node.body)
        orelse_has_return = self._has_top_level_return(orelse_body)

        # One-sided return is not supported: when only one branch returns,
        # the phi merge cannot pair the return value with a corresponding
        # value from the other branch, causing TypeError or silent bugs.
        if body_has_return != orelse_has_return:
            returning_branch = "if" if body_has_return else "else"
            raise SyntaxError(
                f"One-sided 'return' in @qkernel if-else is not supported.\n"
                f"Line {node.lineno}: only the '{returning_branch}' branch "
                f"has a 'return' statement.\n\n"
                f"In @qkernel functions, 'return' inside if-else must appear "
                f"in ALL branches or in NONE.\n"
                f"Either add 'return' to both branches, or move 'return' "
                f"after the if-else block."
            )

        has_return = body_has_return  # both are equal at this point

        ret_var_name: str | None = None
        if has_return:
            ret_var_name = self._get_unique_name("_if_ret")
            true_body = self._transform_returns_to_assignments(node.body, ret_var_name)
            false_body = self._transform_returns_to_assignments(
                orelse_body, ret_var_name
            )
            output_vars.append(ret_var_name)
        else:
            true_body = node.body
            false_body = orelse_body

        # Cond: returns bool (only needs input_vars)
        cond_name = self._get_unique_name("cond")
        cond_def = ast.FunctionDef(
            name=cond_name,
            args=self._make_arguments(input_vars),
            body=[ast.Return(value=node.test)],
            decorator_list=[],
            returns=ast.Name(id="bool", ctx=ast.Load()),
            lineno=node.lineno,
        )  # type: ignore

        # True Body: params=input_vars, return=output_vars
        true_def, true_name = self._create_inner_func(
            "body",
            true_body,
            input_vars,
            node.lineno,
            return_var_names=output_vars,
        )

        # False Body: params=input_vars, return=output_vars
        false_def, false_name = self._create_inner_func(
            "body",
            false_body,
            input_vars,
            node.lineno,
            return_var_names=output_vars,
        )

        # var_list: input_vars values (passed to emit_if)
        var_list = ast.List(
            elts=[ast.Name(id=v, ctx=ast.Load()) for v in input_vars],
            ctx=ast.Load(),
        )

        call_expr = ast.Call(
            func=ast.Name(id=self.if_func_name, ctx=ast.Load()),
            args=[
                ast.Name(id=cond_name, ctx=ast.Load()),
                ast.Name(id=true_name, ctx=ast.Load()),
                ast.Name(id=false_name, ctx=ast.Load()),
                var_list,
            ],
            keywords=[],
        )

        # Assignment: output_vars = emit_if(...)
        assign_node = self._create_assignment_node(output_vars, call_expr, node.lineno)

        result_stmts: list[ast.stmt] = [cond_def, true_def, false_def]

        if has_return:
            # _if_ret_N = None (初期化)
            init_stmt = ast.Assign(
                targets=[ast.Name(id=ret_var_name, ctx=ast.Store())],
                value=ast.Constant(value=None),
                lineno=node.lineno,
            )
            result_stmts.append(init_stmt)

        result_stmts.append(assign_node)

        if has_return:
            # return _if_ret_N
            outer_return = ast.Return(
                value=ast.Name(id=ret_var_name, ctx=ast.Load()),
                lineno=node.lineno,
            )
            result_stmts.append(outer_return)

        return result_stmts


def transform_control_flow(func: Callable):
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError) as e:
        func_name = getattr(func, "__name__", "<unknown>")
        raise SyntaxError(
            f"Cannot retrieve source code for '{func_name}'. "
            f"@qkernel functions must be defined in .py files "
            f"(not REPL/exec/eval)."
        ) from e
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # グローバル変数名を取得（モジュール、組み込み関数など）
    global_names = set(func.__globals__.keys())

    # クロージャ変数も除外する（VariableCollector が target_vars に含めないようにする）
    # クロージャの値は後で name_space に注入されるため、内部関数からアクセス可能
    if func.__closure__ is not None:
        global_names.update(func.__code__.co_freevars)

    # Extract parameter names for loop variable shadowing detection
    param_names = set(inspect.signature(func).parameters.keys())

    # Build resolver namespace for callable resolution (while condition QKernel check).
    # Includes globals and closure values available at function definition time.
    # Empty cells (forward references not yet bound) are skipped here; the resolver
    # treats unresolved callables as fail-open (see _resolve_callable).
    resolver_namespace: dict[str, Any] = dict(func.__globals__)
    if func.__closure__ is not None:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                resolver_namespace[name] = cell.cell_contents
            except ValueError:
                pass

    transformer = ControlFlowTransformer(
        global_names=global_names,
        param_names=param_names,
        namespace=resolver_namespace,
    )
    tree = transformer.visit(tree)

    # デコレータを削除（再帰を防ぐ）
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            node.decorator_list = []

    ast.fix_missing_locations(tree)

    # 元の関数のグローバル変数を継承
    name_space = func.__globals__.copy()
    name_space.update(
        {
            "while_loop": while_loop,
            "for_loop": for_loop,
            "for_items": for_items,
            "emit_if": emit_if,
            "Any": Any,  # For type annotations in generated code
        }
    )

    # クロージャ変数（関数内でインポートされた名前など）を追加
    # Empty cells indicate forward references not yet bound at definition time.
    # Skipping them would cause NameError at runtime, so we fail-closed here
    # and let QKernel.__init__ fall back to the original function.
    if func.__closure__ is not None:
        free_vars = func.__code__.co_freevars
        for name, cell in zip(free_vars, func.__closure__):
            try:
                name_space[name] = cell.cell_contents
            except ValueError:
                raise NotImplementedError(
                    f"Closure variable '{name}' is not yet bound (empty cell). "
                    f"This typically happens with forward references in nested functions."
                ) from None

    code_obj = compile(tree, filename="<qamomile-dsl>", mode="exec")
    exec(code_obj, name_space)

    return name_space[func.__name__]


# ---------------------------------------------------------------------------
# Quantum rebind analysis
# ---------------------------------------------------------------------------


@dataclass
class RebindViolation:
    """A detected forbidden quantum variable rebinding."""

    target_name: str
    source_name: str
    func_name: str | None
    lineno: int


class QuantumRebindAnalyzer(ast.NodeVisitor):
    """Detects forbidden quantum variable reassignment at the AST level.

    Forbidden patterns (target is an *existing* quantum variable):
      - ``a = b``        where b is quantum with a different origin
      - ``a = f(b, ...)`` where b is quantum with a different origin

    Allowed patterns:
      - ``a = f(a, ...)``  (self-update)
      - ``new = f(b, ...)`` (new binding – target was not quantum before)
      - ``alias = q``      (new alias – target was not quantum before)
    """

    def __init__(self, quantum_param_names: set[str]) -> None:
        # name → origin (the parameter name it traces back to)
        self.quantum_vars: dict[str, str] = {n: n for n in quantum_param_names}
        self.violations: list[RebindViolation] = []

    # ------------------------------------------------------------------
    # visitor
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self._check_single_assign(target.id, node.value, node.lineno)
            elif isinstance(target, ast.Tuple):
                self._check_tuple_assign(target, node.value, node.lineno)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # single assignment:  name = expr
    # ------------------------------------------------------------------

    def _check_single_assign(self, target: str, value: ast.expr, lineno: int) -> None:
        # Case 1: Name = Name  (direct alias / overwrite)
        if isinstance(value, ast.Name):
            source = value.id
            if source in self.quantum_vars:
                if target in self.quantum_vars:
                    # existing quantum target – check origin
                    if self.quantum_vars[target] != self.quantum_vars[source]:
                        self.violations.append(
                            RebindViolation(target, source, None, lineno)
                        )
                    # update origin regardless (may be same origin)
                    self.quantum_vars[target] = self.quantum_vars[source]
                else:
                    # new alias binding – allowed, propagate origin
                    self.quantum_vars[target] = self.quantum_vars[source]
            return

        # Case 2: Name = Subscript(...)  (array element alias / overwrite)
        # Example: a = qs[i]
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            source = value.value.id
            if source in self.quantum_vars:
                if target in self.quantum_vars:
                    # existing quantum target – check origin
                    if self.quantum_vars[target] != self.quantum_vars[source]:
                        self.violations.append(
                            RebindViolation(target, source, None, lineno)
                        )
                    # update origin regardless (may be same origin)
                    self.quantum_vars[target] = self.quantum_vars[source]
                else:
                    # new alias binding – allowed, propagate origin
                    self.quantum_vars[target] = self.quantum_vars[source]
            return

        # Case 3: Name = Call(...)
        if isinstance(value, ast.Call):
            # Calls like measure/expval return classical values.
            # If the target was tracked as quantum, clear it here so we don't
            # produce false-positive rebind violations across branches.
            if self._is_classical_returning_call(value):
                self.quantum_vars.pop(target, None)
                return

            quantum_args = self._extract_quantum_args(value)
            if not quantum_args:
                return

            if target in self.quantum_vars:
                target_origin = self.quantum_vars[target]
                has_self = any(
                    self.quantum_vars.get(a) == target_origin for a in quantum_args
                )
                if not has_self:
                    violating = quantum_args[0]
                    func_name = self._get_func_name(value)
                    self.violations.append(
                        RebindViolation(target, violating, func_name, lineno)
                    )

            # propagate quantum status from first quantum arg
            first_q = quantum_args[0]
            self.quantum_vars[target] = self.quantum_vars.get(first_q, first_q)
            return

    # ------------------------------------------------------------------
    # tuple assignment:  a, b = f(a, b)
    # ------------------------------------------------------------------

    def _check_tuple_assign(
        self, targets: ast.Tuple, value: ast.expr, lineno: int
    ) -> None:
        if not isinstance(value, ast.Call):
            return

        target_names = [elt.id for elt in targets.elts if isinstance(elt, ast.Name)]
        if not target_names:
            return

        if self._is_classical_returning_call(value):
            for tgt in target_names:
                self.quantum_vars.pop(tgt, None)
            return

        quantum_args = self._extract_quantum_args(value)
        if not quantum_args:
            return

        arg_origins = [self.quantum_vars.get(a, a) for a in quantum_args]

        for i, tgt in enumerate(target_names):
            mapped_source = (
                quantum_args[i] if i < len(quantum_args) else quantum_args[-1]
            )
            mapped_origin = self.quantum_vars.get(mapped_source, mapped_source)

            # Existing quantum targets keep their own origin when present anywhere
            # in call arguments (e.g. ctrl, tgt = gate(tgt, controls=(ctrl,))).
            if tgt in self.quantum_vars:
                tgt_origin = self.quantum_vars[tgt]
                if tgt_origin in arg_origins:
                    self.quantum_vars[tgt] = tgt_origin
                    continue
                func_name = self._get_func_name(value)
                self.violations.append(
                    RebindViolation(tgt, mapped_source, func_name, lineno)
                )

            # For new targets (or mismatched existing ones), track mapped origin.
            self.quantum_vars[tgt] = mapped_origin

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _extract_quantum_args(self, call: ast.Call) -> list[str]:
        """Recursively collect quantum variable names from args and kwargs.

        Uses a dict to preserve insertion (AST traversal) order while
        deduplicating, so that ``quantum_args[0]`` is deterministic.
        """
        seen: dict[str, None] = {}

        def _collect(expr: ast.expr) -> None:
            if isinstance(expr, ast.Name) and expr.id in self.quantum_vars:
                seen[expr.id] = None
            elif (
                isinstance(expr, ast.Subscript)
                and isinstance(expr.value, ast.Name)
                and expr.value.id in self.quantum_vars
            ):
                seen[expr.value.id] = None
            elif isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
                for e in expr.elts:
                    _collect(e)
            elif isinstance(expr, ast.Starred):
                _collect(expr.value)

        for arg in call.args:
            _collect(arg)
        for kw in call.keywords:
            if kw.value is not None:
                _collect(kw.value)

        return list(seen)

    @staticmethod
    def _get_func_name(call: ast.Call) -> str | None:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    @staticmethod
    def _is_classical_returning_call(call: ast.Call) -> bool:
        """Return True when the call is a known classical-returning frontend op."""
        if isinstance(call.func, ast.Name):
            return call.func.id in {"measure", "expval"}
        if isinstance(call.func, ast.Attribute):
            return call.func.attr in {"measure", "expval"}
        return False


def collect_quantum_rebind_violations(
    func: Callable,
    quantum_param_names: set[str],
) -> list[RebindViolation]:
    """Analyze *func* for forbidden quantum rebind patterns.

    Returns a (possibly empty) list of violations.  Never raises on
    analysis failure – returns ``[]`` instead.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as e:
        func_name = getattr(func, "__name__", "<unknown>")
        warnings.warn(
            f"Quantum rebind analysis skipped for '{func_name}': {e}",
            UserWarning,
            stacklevel=2,
        )
        return []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analyzer = QuantumRebindAnalyzer(quantum_param_names)
            for stmt in node.body:
                analyzer.visit(stmt)
            return analyzer.violations

    return []
