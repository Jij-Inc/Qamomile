"""Static control-flow signatures for explicit qkernel region lowering."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True, order=True)
class RegionLocation:
    """Identify one source-level structured control-flow region.

    Args:
        kind (str): Region kind: ``for``, ``while``, or ``if``.
        lineno (int): One-based source line in the original source file.
        col_offset (int): Zero-based source column.
    """

    kind: str
    lineno: int
    col_offset: int


@dataclasses.dataclass(frozen=True)
class RegionSignature:
    """Describe values crossing one structured region boundary.

    Args:
        inputs (tuple[str, ...]): Explicit values passed to the region.
        carried (tuple[str, ...]): Values updated across a loop back edge or
            merged across branches.
        captures (tuple[str, ...]): Read-only region inputs.
        results (tuple[str, ...]): Updated values live after the region.
    """

    inputs: tuple[str, ...]
    carried: tuple[str, ...]
    captures: tuple[str, ...]
    results: tuple[str, ...]


def analyze_function_regions(
    function: Callable[..., Any],
) -> dict[RegionLocation, RegionSignature]:
    """Analyze a Python function's explicit region interfaces.

    Args:
        function (Callable[..., Any]): Raw undecorated qkernel function whose
            source is available through :mod:`inspect`.

    Returns:
        dict[RegionLocation, RegionSignature]: Source locations mapped to
            deterministic region signatures.

    Raises:
        OSError: If Python source cannot be retrieved.
        SyntaxError: If source cannot be parsed or the definition is absent.
    """
    source = textwrap.dedent(inspect.getsource(function))
    module = ast.parse(source)
    ast.increment_lineno(module, function.__code__.co_firstlineno - 1)
    definition = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function.__name__
        ),
        None,
    )
    if definition is None:
        raise SyntaxError(f"Cannot locate function {function.__name__!r} in source")
    return analyze_region_signatures(definition)


def analyze_region_signatures(
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[RegionLocation, RegionSignature]:
    """Analyze structured interfaces in one parsed function definition.

    Args:
        definition (ast.FunctionDef | ast.AsyncFunctionDef): Parsed function
            body to analyze.

    Returns:
        dict[RegionLocation, RegionSignature]: Source region signatures.
    """
    arguments = {
        argument.arg
        for argument in (
            *definition.args.posonlyargs,
            *definition.args.args,
            *definition.args.kwonlyargs,
        )
    }
    if definition.args.vararg is not None:
        arguments.add(definition.args.vararg.arg)
    if definition.args.kwarg is not None:
        arguments.add(definition.args.kwarg.arg)
    analyzer = _RegionDataflowAnalyzer(arguments | _assigned_names(definition.body))
    analyzer.analyze_block(definition.body, set())
    return dict(analyzer.signatures)


class _RegionDataflowAnalyzer:
    """Compute backwards liveness and persistent-update region signatures."""

    def __init__(self, local_names: set[str]) -> None:
        """Initialize an empty signature registry.

        Args:
            local_names (set[str]): Function-local names that require SSA
                routing rather than global resolution.
        """
        self.signatures: dict[RegionLocation, RegionSignature] = {}
        self._local_names = local_names

    def analyze_block(
        self,
        statements: list[ast.stmt],
        live_out: set[str],
    ) -> set[str]:
        """Compute variables live at block entry.

        Args:
            statements (list[ast.stmt]): Statements in execution order.
            live_out (set[str]): Variables required after the block.

        Returns:
            set[str]: Variables required before the first statement.
        """
        live = set(live_out)
        for statement in reversed(statements):
            live = self._analyze_statement(statement, live)
        return live

    def _analyze_statement(self, statement: ast.stmt, live_out: set[str]) -> set[str]:
        """Apply one backwards dataflow transfer function.

        Args:
            statement (ast.stmt): Statement to analyze.
            live_out (set[str]): Variables live immediately afterwards.

        Returns:
            set[str]: Variables live immediately before ``statement``.
        """
        if isinstance(statement, ast.If):
            return self._analyze_if(statement, live_out)
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            return self._analyze_for(statement, live_out)
        if isinstance(statement, ast.While):
            return self._analyze_while(statement, live_out)
        uses, definitions = _simple_statement_flow(statement)
        return (uses & self._local_names) | (
            set(live_out) - (definitions & self._local_names)
        )

    def _analyze_if(self, statement: ast.If, live_out: set[str]) -> set[str]:
        """Analyze branch arguments and merged results.

        Args:
            statement (ast.If): Conditional statement.
            live_out (set[str]): Variables required after the conditional.

        Returns:
            set[str]: Variables live before the predicate.
        """
        true_live = self.analyze_block(statement.body, live_out)
        false_live = (
            self.analyze_block(statement.orelse, live_out)
            if statement.orelse
            else set(live_out)
        )
        predicate_uses = _loaded_names(statement.test) & self._local_names
        assigned = (
            _assigned_names(statement.body) | _assigned_names(statement.orelse)
        ) & self._local_names
        merged = assigned & set(live_out)
        branch_uses = (
            _loaded_names_in_statements(statement.body)
            | _loaded_names_in_statements(statement.orelse)
        ) & self._local_names
        inputs = (true_live | false_live) & (branch_uses | merged)
        self.signatures[
            RegionLocation("if", statement.lineno, statement.col_offset)
        ] = _signature(
            statement,
            inputs=inputs,
            carried=merged,
            captures=inputs - merged,
            results=merged,
        )
        return predicate_uses | inputs

    def _analyze_for(
        self,
        statement: ast.For | ast.AsyncFor,
        live_out: set[str],
    ) -> set[str]:
        """Analyze loop iter-args, captures, and post-loop results.

        Args:
            statement (ast.For | ast.AsyncFor): Iteration statement.
            live_out (set[str]): Variables required after the loop.

        Returns:
            set[str]: Variables live before iterator evaluation.
        """
        target_names, target_uses = _target_flow(statement.target)
        assigned = (_assigned_names(statement.body) & self._local_names) - target_names
        body_live = set(live_out)
        while True:
            next_live = self.analyze_block(statement.body, set(live_out) | body_live)
            next_live.difference_update(target_names)
            if next_live == body_live:
                break
            body_live = next_live
        carried = assigned & (body_live | set(live_out))
        body_uses = _loaded_names_in_statements(statement.body) & self._local_names
        captures = (body_live & body_uses) - carried
        results = carried & set(live_out)
        self.signatures[
            RegionLocation("for", statement.lineno, statement.col_offset)
        ] = _signature(
            statement,
            inputs=carried | captures,
            carried=carried,
            captures=captures,
            results=results,
        )
        else_live = (
            self.analyze_block(statement.orelse, live_out)
            if statement.orelse
            else set(live_out)
        )
        return (
            (_loaded_names(statement.iter) & self._local_names)
            | target_uses
            | body_live
            | else_live
            | set(live_out)
        ) - target_names

    def _analyze_while(self, statement: ast.While, live_out: set[str]) -> set[str]:
        """Analyze while state shared by condition and body regions.

        Args:
            statement (ast.While): While statement.
            live_out (set[str]): Variables required after the loop.

        Returns:
            set[str]: Variables live before the first condition evaluation.
        """
        predicate_uses = _loaded_names(statement.test) & self._local_names
        assigned = _assigned_names(statement.body) & self._local_names
        body_live = set(live_out) | predicate_uses
        while True:
            next_live = self.analyze_block(
                statement.body,
                set(live_out) | predicate_uses | body_live,
            )
            if next_live == body_live:
                break
            body_live = next_live
        carried = assigned & (body_live | predicate_uses | set(live_out))
        body_uses = _loaded_names_in_statements(statement.body) & self._local_names
        captures = (
            (body_live | predicate_uses) & (body_uses | predicate_uses)
        ) - carried
        results = carried & set(live_out)
        self.signatures[
            RegionLocation("while", statement.lineno, statement.col_offset)
        ] = _signature(
            statement,
            inputs=carried | captures,
            carried=carried,
            captures=captures,
            results=results,
        )
        else_live = (
            self.analyze_block(statement.orelse, live_out)
            if statement.orelse
            else set(live_out)
        )
        return set(live_out) | predicate_uses | body_live | else_live


def _signature(
    node: ast.AST,
    *,
    inputs: set[str],
    carried: set[str],
    captures: set[str],
    results: set[str],
) -> RegionSignature:
    """Build a deterministically ordered region signature.

    Args:
        node (ast.AST): Source region used to derive lexical name order.
        inputs (set[str]): Region input names.
        carried (set[str]): Carried or merged names.
        captures (set[str]): Read-only captured names.
        results (set[str]): Post-region result names.

    Returns:
        RegionSignature: Ordered immutable signature.
    """
    return RegionSignature(
        inputs=_ordered_names(node, inputs),
        carried=_ordered_names(node, carried),
        captures=_ordered_names(node, captures),
        results=_ordered_names(node, results),
    )


def _ordered_names(node: ast.AST, names: set[str]) -> tuple[str, ...]:
    """Order names by first lexical occurrence and then alphabetically.

    Args:
        node (ast.AST): Source subtree containing name occurrences.
        names (set[str]): Names to order.

    Returns:
        tuple[str, ...]: Deterministic source-shaped order.
    """
    ordered: list[str] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Name)
            and child.id in names
            and child.id not in ordered
        ):
            ordered.append(child.id)
    ordered.extend(sorted(names - set(ordered)))
    return tuple(ordered)


def _simple_statement_flow(statement: ast.stmt) -> tuple[set[str], set[str]]:
    """Return direct uses and definitions for a non-control statement.

    Args:
        statement (ast.stmt): Statement excluding structured if/for/while.

    Returns:
        tuple[set[str], set[str]]: Loaded and defined variable names.
    """
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        value = statement.value
        uses = _loaded_names(value) if value is not None else set()
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        definitions: set[str] = set()
        for target in targets:
            target_defs, target_uses = _target_flow(target)
            definitions.update(target_defs)
            uses.update(target_uses)
        return uses, definitions
    if isinstance(statement, ast.AugAssign):
        definitions, target_uses = _target_flow(statement.target)
        return target_uses | definitions | _loaded_names(statement.value), definitions
    if isinstance(statement, ast.Return):
        return _loaded_names(statement.value), set()
    if isinstance(statement, ast.Expr):
        return _loaded_names(statement.value), set()
    if isinstance(statement, ast.Delete):
        definitions: set[str] = set()
        uses: set[str] = set()
        for target in statement.targets:
            target_defs, target_uses = _target_flow(target)
            definitions.update(target_defs)
            uses.update(target_uses)
        return uses, definitions
    return _loaded_names(statement), set()


def _target_flow(target: ast.AST) -> tuple[set[str], set[str]]:
    """Return persistent definitions and reads for an assignment target.

    Args:
        target (ast.AST): Assignment target.

    Returns:
        tuple[set[str], set[str]]: Defined roots and names read by the update.
    """
    if isinstance(target, ast.Name):
        return {target.id}, set()
    if isinstance(target, (ast.Tuple, ast.List)):
        definitions: set[str] = set()
        uses: set[str] = set()
        for element in target.elts:
            element_defs, element_uses = _target_flow(element)
            definitions.update(element_defs)
            uses.update(element_uses)
        return definitions, uses
    if isinstance(target, ast.Starred):
        return _target_flow(target.value)
    if isinstance(target, ast.Subscript):
        root = _root_name(target.value)
        return ({root} if root is not None else set()), _loaded_names(target)
    if isinstance(target, ast.Attribute):
        root = _root_name(target.value)
        return ({root} if root is not None else set()), _loaded_names(target.value)
    return set(), _loaded_names(target)


def _root_name(expression: ast.AST) -> str | None:
    """Return the root variable updated by an attribute or subscript chain.

    Args:
        expression (ast.AST): Target base expression.

    Returns:
        str | None: Root name, or ``None`` for a non-name-rooted expression.
    """
    current = expression
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def _loaded_names(node: ast.AST | None) -> set[str]:
    """Collect loaded names without descending into nested definitions.

    Args:
        node (ast.AST | None): Syntax subtree to inspect.

    Returns:
        set[str]: Names occurring in load context.
    """
    if node is None:
        return set()
    collector = _NameCollector(collect_stores=False)
    collector.visit(node)
    return collector.names


def _assigned_names(statements: list[ast.stmt]) -> set[str]:
    """Collect root names assigned anywhere in a statement block.

    Args:
        statements (list[ast.stmt]): Statements to inspect recursively.

    Returns:
        set[str]: Assigned variable and persistent-update root names.
    """
    collector = _NameCollector(collect_stores=True)
    for statement in statements:
        collector.visit(statement)
    return collector.names


def _loaded_names_in_statements(statements: list[ast.stmt]) -> set[str]:
    """Collect names read anywhere in a statement block.

    Args:
        statements (list[ast.stmt]): Statements to inspect recursively.

    Returns:
        set[str]: Loaded variable names.
    """
    collector = _NameCollector(collect_stores=False)
    for statement in statements:
        collector.visit(statement)
    return collector.names


class _NameCollector(ast.NodeVisitor):
    """Collect either loaded names or persistent assignment roots."""

    def __init__(self, collect_stores: bool) -> None:
        """Initialize an empty name set.

        Args:
            collect_stores (bool): Collect assignment roots instead of loads.
        """
        self.collect_stores = collect_stores
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Collect a matching name occurrence.

        Args:
            node (ast.Name): Name expression to inspect.
        """
        if self.collect_stores == isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect assignment values or persistent target roots.

        Args:
            node (ast.Assign): Assignment statement.
        """
        if self.collect_stores:
            for target in node.targets:
                definitions, _ = _target_flow(target)
                self.names.update(definitions)
            self.visit(node.value)
            return
        self.visit(node.value)
        for target in node.targets:
            _, uses = _target_flow(target)
            self.names.update(uses)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect annotated assignment flow.

        Args:
            node (ast.AnnAssign): Annotated assignment statement.
        """
        if self.collect_stores:
            definitions, _ = _target_flow(node.target)
            self.names.update(definitions)
            if node.value is not None:
                self.visit(node.value)
            return
        if node.value is not None:
            self.visit(node.value)
        _, uses = _target_flow(node.target)
        self.names.update(uses)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Collect augmented assignment flow.

        Args:
            node (ast.AugAssign): Augmented assignment statement.
        """
        definitions, uses = _target_flow(node.target)
        if self.collect_stores:
            self.names.update(definitions)
            return
        self.names.update(definitions | uses)
        self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip nested function bodies.

        Args:
            node (ast.FunctionDef): Nested definition to ignore.
        """
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip nested async function bodies.

        Args:
            node (ast.AsyncFunctionDef): Nested definition to ignore.
        """
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Skip nested lambda bodies.

        Args:
            node (ast.Lambda): Nested lambda to ignore.
        """
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Skip nested class bodies.

        Args:
            node (ast.ClassDef): Nested class to ignore.
        """
        del node
