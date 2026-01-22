import ast
import inspect
import textwrap
from typing import Any, Callable


from qamomile.circuit.frontend.operation.control_flow import (
    emit_if,
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

    def visit_Name(self, node: ast.Name):
        """変数名を収集（除外リストにないもののみ）"""
        if isinstance(node.id, str):
            if node.id not in self._exclude and node.id not in self._global_names:
                self.vars.add(node.id)


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

    def visit_For(self, node: ast.For) -> Any:
        # ネストされた制御フローを先に変換
        new_body = [self.visit(stmt) for stmt in node.body]
        flattened_body = []
        for item in new_body:
            if isinstance(item, list):
                flattened_body.extend(item)
            else:
                flattened_body.append(item)

        # for i in range(...) または for i in qm.range(...) の場合のみサポート
        if not self._is_range_call(node.iter):
            raise NotImplementedError(
                "Only 'for i in range(...)' or 'for i in qm.range(...)' is supported"
            )

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

    def visit_If(self, node: ast.If) -> Any:
        self.generic_visit(node)

        vars_test = self._collect_variables(node.test)
        vars_body = self._collect_variables(node.body)  # type: ignore
        vars_orelse = self._collect_variables(node.orelse) if node.orelse else []  # type: ignore
        target_vars = sorted(list(set(vars_test) | set(vars_body) | set(vars_orelse)))

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
            "body", node.body, target_vars, node.lineno
        )

        # False Body
        orelse_body = node.orelse if node.orelse else []
        false_def, false_name = self._create_inner_func(
            "body", orelse_body, target_vars, node.lineno
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
        return [cond_def, true_def, false_def, assign_node]


def transform_control_flow(func: Callable):
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # グローバル変数名を取得（モジュール、組み込み関数など）
    global_names = set(func.__globals__.keys())

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
