import ast
import inspect
import textwrap
from typing import Any, Callable


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
    - 属性アクセスのベースオブジェクト (value in Attribute)
    - グローバル変数（モジュール、組み込み関数など）
    """

    def __init__(self, global_names: set[str] | None = None):
        self.vars = set()
        self._exclude = set()  # 除外する名前
        self._global_names = global_names or set()
        self._first_context: dict[str, str] = {}  # name -> "Store" | "Load"

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
        """属性アクセスのベースオブジェクト（モジュール名）を除外"""
        if isinstance(node.value, ast.Name):
            # qm.x の qm を除外
            self._exclude.add(node.value.id)
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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """内部関数定義の走査をスキップする。"""
        pass

    def visit_Name(self, node: ast.Name):
        """変数名を収集（除外リストにないもののみ）"""
        if isinstance(node.id, str):
            if node.id not in self._exclude and node.id not in self._global_names:
                self.vars.add(node.id)
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


class ControlFlowTransformer(ast.NodeTransformer):
    while_func_name = "emit_while"
    if_func_name = "emit_if"
    for_func_name = "emit_for"

    def __init__(self, global_names: set[str] | None = None) -> None:
        self.counter: int = 0
        # 変数名 -> 型注釈ノード(ast.AST) を保持する辞書
        self.type_registry: dict[str, ast.AST] = {}
        # グローバル変数名（モジュール、組み込み関数など）
        self._global_names = global_names or set()

    def _get_unique_name(self, prefix: str) -> str:
        name = f"_{prefix}_{self.counter}"
        self.counter += 1
        return name

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
        return_type_ast: ast.AST | None = None,
    ) -> tuple[ast.FunctionDef, str]:
        func_name = self._get_unique_name(name_prefix)
        func_args = self._make_arguments(var_names)

        new_body = list(body_nodes)
        new_body.append(self._create_return_node(var_names))

        # 戻り値の型注釈が指定されていなければ自動生成
        if return_type_ast is None:
            return_type_ast = self._create_return_annotation(var_names)

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

    def visit_While(self, node: ast.While) -> Any:
        # ネストされた制御フローを先に変換
        new_body = [self.visit(stmt) for stmt in node.body]
        flattened_body = []
        for item in new_body:
            if isinstance(item, list):
                flattened_body.extend(item)
            else:
                flattened_body.append(item)

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
        """Extract variable names from tuple unpacking target.

        Examples:
            i -> ["i"]
            (i, j) -> ["i", "j"]
            ((i, j), k) -> ["i", "j", "k"]
        """
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, ast.Tuple):
            var_names = []
            for elt in target.elts:
                var_names.extend(self._extract_tuple_vars(elt))
            return var_names
        else:
            raise NotImplementedError(f"Unsupported target type: {type(target)}")

    def visit_For(self, node: ast.For) -> Any:
        # ネストされた制御フローを先に変換
        new_body = [self.visit(stmt) for stmt in node.body]
        flattened_body = []
        for item in new_body:
            if isinstance(item, list):
                flattened_body.extend(item)
            else:
                flattened_body.append(item)

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

        num_args = len(node.iter.args)  # type: ignore
        if num_args < 1 or num_args > 3:
            raise NotImplementedError(
                "range() requires 1-3 arguments: range(stop), range(start, stop), or range(start, stop, step)"
            )

        # range の引数を取得
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

        # ループ変数名を取得
        if isinstance(node.target, ast.Name):
            loop_var_name = node.target.id
        else:
            loop_var_name = "_loop_idx"

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
        self, node: ast.For, flattened_body: list[ast.stmt]
    ) -> ast.With:
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
        if node.iter.args:  # type: ignore  # items(d) pattern
            dict_arg = node.iter.args[0]  # type: ignore
        elif isinstance(node.iter.func, ast.Attribute):  # type: ignore  # d.items() pattern
            dict_arg = node.iter.func.value  # type: ignore
        else:
            raise NotImplementedError("items() requires a dict argument")

        # Parse the target pattern: (key, value) or just key, value
        if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
            key_target = node.target.elts[0]
            value_target = node.target.elts[1]
        else:
            raise NotImplementedError(
                "items() iteration requires 'for key, value in items(d)' pattern. "
                f"Got: {ast.dump(node.target)}"
            )

        # Extract key variable names (may be tuple like (i, j))
        key_vars = self._extract_tuple_vars(key_target)

        # Extract value variable name (must be a simple name)
        if not isinstance(value_target, ast.Name):
            raise NotImplementedError(
                "Value target in items() iteration must be a simple variable, "
                f"got: {ast.dump(value_target)}"
            )
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

        # incoming_vars のみ使用（ブランチ内で初定義される変数を除外）
        target_vars = sorted(
            collector_test.vars
            | collector_body.incoming_vars
            | collector_orelse.incoming_vars
        )

        # Detect return inside for/while in if-else branches (unsupported).
        # Must run before generic_visit which transforms loops into with-stmts.
        if self._has_return_under_loop(node.body) or self._has_return_under_loop(
            node.orelse if node.orelse else []
        ):
            raise SyntaxError(
                "'return' inside for/while in @qkernel if-else is not supported"
            )

        # ネストされた制御フローを変換
        self.generic_visit(node)

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
            target_vars.append(ret_var_name)
        else:
            true_body = node.body
            false_body = orelse_body

        # Cond: returns bool
        cond_name = self._get_unique_name("cond")
        cond_def = ast.FunctionDef(
            name=cond_name,
            args=self._make_arguments(target_vars),
            body=[ast.Return(value=node.test)],
            decorator_list=[],
            returns=ast.Name(id="bool", ctx=ast.Load()),
            lineno=node.lineno,
        )  # type: ignore

        # True Body
        true_def, true_name = self._create_inner_func(
            "body", true_body, target_vars, node.lineno
        )

        # False Body
        false_def, false_name = self._create_inner_func(
            "body", false_body, target_vars, node.lineno
        )

        var_list = ast.List(
            elts=[ast.Name(id=v, ctx=ast.Load()) for v in target_vars], ctx=ast.Load()
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

        assign_node = self._create_assignment_node(target_vars, call_expr, node.lineno)

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
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # グローバル変数名を取得（モジュール、組み込み関数など）
    global_names = set(func.__globals__.keys())

    # クロージャ変数も除外する（VariableCollector が target_vars に含めないようにする）
    # クロージャの値は後で name_space に注入されるため、内部関数からアクセス可能
    if func.__closure__ is not None:
        global_names.update(func.__code__.co_freevars)

    transformer = ControlFlowTransformer(global_names=global_names)
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
    if func.__closure__ is not None:
        free_vars = func.__code__.co_freevars
        for name, cell in zip(free_vars, func.__closure__):
            name_space[name] = cell.cell_contents

    code_obj = compile(tree, filename="<qamomile-dsl>", mode="exec")
    exec(code_obj, name_space)

    return name_space[func.__name__]
