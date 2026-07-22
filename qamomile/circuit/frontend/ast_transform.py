import ast
import copy
import enum
import inspect
import textwrap
import types
import warnings
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from qamomile.circuit.frontend.operation.control_flow import (
    branch_rebind_pre_bindings,
    dead_rebind_binding,
    emit_if,
    explicit_loop_bindings,
    for_items,
    for_loop,
    loop_rebind_snapshot,
    loop_region_enter,
    loop_region_result,
    record_loop_rebinds,
    should_trace_for_loop,
    should_trace_items_loop,
    while_loop,
)
from qamomile.circuit.frontend.region_analysis import (
    RegionLocation,
    RegionSignature,
    analyze_region_signatures,
)


class VariableCollector(ast.NodeVisitor):
    """Collect variables used and mutated within a block.

    Excludes:
    - Function names in calls (func in Call)
    - Global base objects in attribute accesses (value in Attribute)
    - Global variables (modules, builtins, etc.)
    """

    def __init__(self, global_names: set[str] | None = None):
        """Initialize an empty variable/dataflow collector.

        Args:
            global_names (set[str] | None): Names treated as globals rather
                than function-local dataflow. Defaults to None.
        """
        self.vars = set()
        self._exclude = set()  # names to exclude
        self._global_names = global_names or set()
        self._first_context: dict[str, str] = {}  # name -> "Store" | "Load"
        self._load_names: set[str] = set()
        self._store_names: set[str] = set()
        self._store_order: list[str] = []

    def visit_Call(self, node: ast.Call):
        """Exclude the function name of a call."""
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            self._exclude.add(node.func.id)
        # Process arguments normally
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        # If node.func is an attribute access, handle it in visit_Attribute
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func)

    def visit_Attribute(self, node: ast.Attribute):
        """Record the base name of an attribute access.

        Global names such as module names (`qm.h`) are excluded as before,
        while user variables (`qs.shape`) are treated as Load.
        """
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self._global_names:
                # Exclude qm in qm.x
                self._exclude.add(name)
            else:
                self.vars.add(name)
                self._load_names.add(name)
                if name not in self._first_context:
                    self._first_context[name] = "Load"
        else:
            # Recurse for nested attribute accesses (a.b.c)
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign):
        """Visit the RHS first to match Python's evaluation order.

        `q1 = qm.h(q1)` → RHS q1 (Load) is first → first_context is "Load"
        `cond2 = qm.measure(q2)` → RHS q2 (Load) first, LHS cond2 (Store) after
        """
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit the RHS first, like ``visit_Assign``.

        `total: qmc.UInt = total + i` reads the RHS before storing, so
        the generic (target-first) traversal would misclassify the
        first context as "Store" and drop the read-before-write
        evidence the loop-carry candidate analysis needs. The
        annotation itself is type syntax, not dataflow, and is skipped.

        Args:
            node (ast.AnnAssign): Annotated assignment to visit.
        """
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        """Visit a named expression in Python evaluation order.

        A walrus assignment such as ``total := total + i`` evaluates the
        value before storing the target. Preserving that order keeps the
        read-before-write evidence used to identify loop-carry candidates.

        Args:
            node (ast.NamedExpr): The named-expression node to visit.
        """
        self.visit(node.value)
        self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign):
        """AugAssign (e.g. x += 1) is an implicit Read-before-Write.

        Visit the RHS first and record Name targets as both Load and Store.
        first_context is "Load" (the existing value is read first).

        Args:
            node (ast.AugAssign): Augmented assignment to visit in Python
                evaluation order.
        """
        self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            name = target.id
            if name not in self._exclude and name not in self._global_names:
                self.vars.add(name)
                self._load_names.add(name)
                if name not in self._store_names:
                    self._store_order.append(name)
                self._store_names.add(name)
                if name not in self._first_context:
                    self._first_context[name] = "Load"
        else:
            # Subscript / Attribute: visit normally to capture base/index loads
            self.visit(target)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Skip traversal of inner function definitions."""
        pass

    def visit_Name(self, node: ast.Name):
        """Collect a non-excluded variable name and its access context.

        Args:
            node (ast.Name): Name occurrence whose load/store context should
                contribute to variable dataflow.
        """
        if isinstance(node.id, str):
            if node.id not in self._exclude and node.id not in self._global_names:
                self.vars.add(node.id)
                if isinstance(node.ctx, ast.Load):
                    self._load_names.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self._store_names:
                        self._store_order.append(node.id)
                    self._store_names.add(node.id)
                if node.id not in self._first_context:
                    if isinstance(node.ctx, ast.Store):
                        self._first_context[node.id] = "Store"
                    else:
                        self._first_context[node.id] = "Load"

    @property
    def locally_defined_vars(self) -> set[str]:
        """Variables first defined (Store) within this scope."""
        return {name for name, ctx in self._first_context.items() if ctx == "Store"}

    @property
    def incoming_vars(self) -> set[str]:
        """Variables that must come from an outer scope (first use is Load)."""
        return self.vars - self.locally_defined_vars

    @property
    def load_vars(self) -> set[str]:
        """Variables referenced in Load context (actually read)."""
        return self._load_names & self.vars

    @property
    def store_vars(self) -> set[str]:
        """Variables assigned in Store context."""
        return self._store_names & self.vars

    @property
    def store_order(self) -> tuple[str, ...]:
        """Return assigned variable names in first-source-store order.

        Returns:
            tuple[str, ...]: Unique stored names in source order.
        """
        return tuple(name for name in self._store_order if name in self.vars)


class ControlFlowTransformer(ast.NodeTransformer):
    while_func_name = "emit_while"
    if_func_name = "emit_if"
    for_func_name = "emit_for"

    def __init__(
        self,
        global_names: set[str] | None = None,
        param_names: set[str] | None = None,
        namespace: dict[str, Any] | None = None,
        region_signatures: dict[RegionLocation, RegionSignature] | None = None,
    ) -> None:
        """Initialize source tracking and explicit region interfaces.

        Args:
            global_names (set[str] | None): Names resolved outside the qkernel
                local scope. Defaults to ``None``.
            param_names (set[str] | None): Function parameter names used for
                shadowing diagnostics. Defaults to ``None``.
            namespace (dict[str, Any] | None): Definition-time values used to
                resolve callable conditions. Defaults to ``None``.
            region_signatures (dict[RegionLocation, RegionSignature] | None):
                Static interfaces for structured source regions. Defaults to
                ``None``.
        """
        self.counter: int = 0
        # Dict mapping variable name -> type annotation node (ast.AST)
        self.type_registry: dict[str, ast.AST] = {}
        # Global variable names (modules, builtins, etc.)
        self._global_names = global_names or set()
        # Scope tracking for visit_If input/output separation
        self._outer_defined_vars: frozenset[str] = frozenset()
        self._after_stmt_read_vars: frozenset[str] = frozenset()
        self._after_stmt_load_vars: frozenset[str] = frozenset()
        # Function-wide accumulation of every name defined so far at any
        # nesting level (Python function scope), consulted only by the
        # if-rebind record candidate computation in ``visit_If``.
        self._lexical_defined_vars: set[str] = set()
        # Function parameter names (for loop-variable shadowing detection)
        self._param_names = param_names or set()
        # Namespace for resolving callables (used in while condition QKernel check)
        self._namespace = namespace or {}
        self._region_signatures = region_signatures or {}

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
            # Function-wide lexical accumulation for rebind-record
            # candidates: unlike ``_outer_defined_vars`` (deliberately
            # narrowed to each branch's inputs so dead old values never
            # force merge inputs), Python variables have function scope, so
            # a branch assignment to a name defined anywhere earlier in
            # the function rebinds THAT variable. This set is consulted
            # only by the rebind-record candidate computation and never
            # feeds the input/output/merge machinery.
            self._lexical_defined_vars |= defined_so_far

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
        """Detect annotated assignments such as ``a: int = 0`` and register
        the type information."""
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
        # Targets variables that have a registered type or that appear in the nodes.
        # Global variables (modules, builtins, etc.) are excluded.
        return sorted(list(collector.vars))

    def _get_annotation(self, var_name: str) -> ast.AST | None:
        """Return the registered type annotation (returning a deep copy is preferred)."""
        if var_name in self.type_registry:
            # Reusing AST nodes is bug-prone due to location metadata, so copying
            # is preferred; here we return the reference directly (safe for unparse).
            return self.type_registry[var_name]
        return None

    def _make_arguments(self, var_names: list[str]) -> ast.arguments:
        """Build an argument list with type annotations."""
        args_list = []
        for name in var_names:
            # Look up the annotation
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
        """Build a return type annotation (e.g. ``int`` or ``tuple[int, int]``)."""
        if not var_names:
            return ast.Constant(value=None)

        # Single variable: return its annotation
        if len(var_names) == 1:
            return self._get_annotation(var_names[0])

        # Multiple variables: build tuple[Type1, Type2]
        # ast.Subscript(value=Name(id='tuple'), slice=Tuple(elts=[...]))
        elts: list[ast.expr] = []
        for name in var_names:
            ann = self._get_annotation(name)
            # Fall back to Any when no annotation is registered
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
        extra_return_exprs: list[ast.expr] | None = None,
    ) -> tuple[ast.FunctionDef, str]:
        """Build a generated helper function for a traced control-flow body.

        Args:
            name_prefix (str): Prefix for the unique generated name.
            body_nodes (list[ast.stmt]): Already-transformed body
                statements.
            var_names (list[str]): Parameter names of the generated
                function.
            lineno (int): Source line for the generated node.
            return_var_names (list[str] | None): Names returned by the
                final return statement. Defaults to None, meaning
                ``var_names``.
            return_type_ast (ast.AST | None): Explicit return annotation.
                Defaults to None (derived from the returned names, or
                ``Any`` when probe expressions are appended).
            extra_return_exprs (list[ast.expr] | None): Probe expressions
                appended after the ordinary return values (used by the
                if-rebind records for dead-after variables); forces the
                tuple return form so callers can slice the probe tail
                off positionally. Defaults to None.

        Returns:
            tuple[ast.FunctionDef, str]: The generated function node and
                its unique name.
        """
        func_name = self._get_unique_name(name_prefix)
        func_args = self._make_arguments(var_names)

        ret_vars = return_var_names if return_var_names is not None else var_names
        new_body = list(body_nodes)
        if extra_return_exprs:
            # Append probe expressions after the ordinary return values
            # (used by the if-rebind records for dead-after variables).
            # The result is always a tuple so the caller can slice the
            # probe tail off positionally.
            new_body.append(
                ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(id=v, ctx=ast.Load()) for v in ret_vars]
                        + list(extra_return_exprs),
                        ctx=ast.Load(),
                    )
                )
            )
            if return_type_ast is None:
                return_type_ast = ast.Name(id="Any", ctx=ast.Load())
        else:
            new_body.append(self._create_return_node(ret_vars))

        # Auto-generate the return type annotation if not specified
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
    def _has_return_in_region(body_nodes: list[ast.stmt]) -> bool:
        """Check whether a control-flow region contains a return statement.

        Nested function, lambda, and class bodies introduce independent Python
        scopes, so returns inside them do not exit the enclosing qkernel region
        and are deliberately ignored.

        Args:
            body_nodes (list[ast.stmt]): Statements in the control-flow region.

        Returns:
            bool: ``True`` when the region contains a return that would exit
                the enclosing qkernel.
        """

        def contains_return(node: ast.AST) -> bool:
            """Recursively inspect one AST node without crossing scopes.

            Args:
                node (ast.AST): Node to inspect.

            Returns:
                bool: ``True`` when ``node`` contains an in-scope return.
            """
            if isinstance(node, ast.Return):
                return True
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                return False
            return any(contains_return(child) for child in ast.iter_child_nodes(node))

        return any(contains_return(stmt) for stmt in body_nodes)

    @staticmethod
    def _reject_loop_control(body_nodes: list[ast.stmt], loop_kind: str) -> None:
        """Reject Python break/continue before loop AST rewriting.

        Args:
            body_nodes (list[ast.stmt]): Original loop body statements.
            loop_kind (str): User-facing loop kind for the diagnostic.

        Raises:
            SyntaxError: If the body contains ``break`` or ``continue``.
        """
        for statement in body_nodes:
            for nested in ast.walk(statement):
                if isinstance(nested, ast.Break):
                    raise SyntaxError(
                        f"'break' inside a {loop_kind} loop is not supported "
                        "in @qkernel; express termination in the loop bound "
                        "or while condition instead"
                    )
                if isinstance(nested, ast.Continue):
                    raise SyntaxError(
                        f"'continue' inside a {loop_kind} loop is not supported "
                        "in @qkernel; guard the remaining body with an if instead"
                    )

    @staticmethod
    def _reject_named_expression_in_condition(
        condition: ast.expr,
        *,
        construct: str,
    ) -> None:
        """Reject assignment expressions whose scope changes after lowering.

        Conditions are moved into generated helper functions or lambdas.
        A walrus target would therefore bind inside the helper instead of the
        surrounding qkernel, diverging from Python semantics.

        Args:
            condition (ast.expr): Condition expression to inspect.
            construct (str): User-visible control-flow construct name.

        Raises:
            SyntaxError: If the condition contains a named expression.
        """
        if any(isinstance(node, ast.NamedExpr) for node in ast.walk(condition)):
            raise SyntaxError(
                f"Assignment expressions (':=') in {construct} conditions are "
                "not supported in @qkernel. Assign the value on a separate "
                "line before the condition."
            )

    @staticmethod
    def _transform_returns_to_assignments(
        body_nodes: list[ast.stmt], ret_var_name: str
    ) -> list[ast.stmt]:
        """Replace top-level returns with assignments to one result name.

        Args:
            body_nodes (list[ast.stmt]): Statements to rewrite.
            ret_var_name (str): Synthetic local receiving each return value.

        Returns:
            list[ast.stmt]: Statements with value-return nodes replaced by
                assignments while all other nodes are preserved.
        """
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
        """Collect non-global variable names loaded by an AST node.

        Args:
            node (ast.AST): Syntax tree to inspect.

        Returns:
            set[str]: Loaded local or closure variable names.
        """
        collector = VariableCollector(global_names=self._global_names)
        collector.visit(node)
        return set(collector.load_vars)

    def _stmt_live_in(self, stmt: ast.stmt, live_out: set[str]) -> set[str]:
        """Compute variables live before one statement.

        Args:
            stmt (ast.stmt): Statement whose transfer function is applied.
            live_out (set[str]): Variables required after the statement.

        Returns:
            set[str]: Variables required before the statement, including
                control predicates and loop back-edge dependencies.
        """
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
        """Compute variables live at entry to a statement block.

        Args:
            body (list[ast.stmt]): Statements in execution order.
            live_out (set[str]): Variables required after the block.

        Returns:
            set[str]: Variables required before the first statement.
        """
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
        """Compute the loop-body entry liveness fixed point.

        Args:
            body (list[ast.stmt]): Loop-body statements.
            exit_live (set[str]): Variables required after loop exit or by the
                condition.
            bound_vars (set[str] | None): Names rebound by loop iteration and
                therefore excluded from the carried fixed point. Defaults to
                None.

        Returns:
            set[str]: Variables live on entry and across the back edge.
        """
        bound = set(bound_vars or ())
        live = set()
        while True:
            next_live = self._block_live_in(body, set(exit_live) | live)
            next_live.difference_update(bound)
            if next_live == live:
                return next_live
            live = next_live

    def _loop_rebind_candidates(
        self, body: list[ast.stmt], bound_names: set[str]
    ) -> tuple[list[str], list[str]]:
        """Compute loop-body variables that may carry a loop-carried rebind.

        A candidate is a variable that the (pre-transform) loop body
        assigns and that some enclosing scope already defines — excluding
        the loop's own binding variables (whose per-iteration rebinding is
        handled by the emit-time loop-variable binding). Pre-existing
        names are gated on the union of the current scope's defined set
        and the function-wide lexical set: inside an if branch the probe
        resolves pre-branch handles through the pre-binding stack, so
        lexical visibility, not branch-scope definedness, is the right
        bound. The probes resolve names tolerantly (unbound candidates
        drop out), so over-approximating here is safe.

        The classical subset contains candidates that either enter the
        loop body live (the ``total = total + i`` back-edge shape) or
        leave the loop live (a store-only value read after the loop).
        Both need formal loop results: the former threads values between
        iterations, while the latter preserves the last body value and
        the initializer on a zero-trip path. Quantum rebind records are
        collected for every candidate because a store-only quantum
        reassignment is a state discard, not a recomputation.

        Args:
            body (list[ast.stmt]): The ORIGINAL (untransformed) loop body
                statements.
            bound_names (set[str]): Names bound by the loop statement
                itself (``for`` target names, ``items`` key/value names).

        Returns:
            tuple[list[str], list[str]]: ``(candidates, classical_candidates)``,
                in first-store source order; ``classical_candidates`` is a
                subset of ``candidates``. Both are empty when the body
                reassigns no pre-existing variable.
        """
        collector = VariableCollector(global_names=self._global_names)
        for stmt in body:
            collector.visit(stmt)
        outer_defined = set(self._outer_defined_vars)
        pre_existing = outer_defined | self._lexical_defined_vars
        candidates = (collector.store_vars - bound_names) & pre_existing
        loop_entry_live = self._loop_body_entry_live_in(
            body, set(), bound_vars=bound_names
        )
        backedge_candidates = candidates & loop_entry_live
        live_out_candidates = (
            (collector.store_vars - bound_names)
            & set(self._after_stmt_load_vars)
            & outer_defined
        )
        classical_candidate_set = backedge_candidates | live_out_candidates
        ordered_candidates = [
            name for name in collector.store_order if name in candidates
        ]
        classical_candidates = [
            name for name in ordered_candidates if name in classical_candidate_set
        ]
        return ordered_candidates, classical_candidates

    def _reject_loop_local_escapes(
        self,
        body: list[ast.stmt],
        bound_names: set[str],
        lineno: int,
        loop_kind: str,
    ) -> None:
        """Reject names first defined in a loop and read after it.

        Qamomile has no formal loop result or zero-trip initializer for a name
        first defined in the body. Letting such a name escape would therefore
        leak the value produced by the one trace-time body invocation instead
        of modeling Python's conditionally-bound local.

        Args:
            body (list[ast.stmt]): Original loop body statements.
            bound_names (set[str]): Names bound by the loop target itself.
            lineno (int): Source line used in the diagnostic.
            loop_kind (str): Human-readable loop kind.

        Raises:
            SyntaxError: If a body-local name is live after the loop.
        """
        collector = VariableCollector(global_names=self._global_names)
        for stmt in body:
            collector.visit(stmt)
        escaping = (
            (collector.store_vars - bound_names) - set(self._outer_defined_vars)
        ) & set(self._after_stmt_load_vars)
        if escaping:
            name = next(
                candidate
                for candidate in collector.store_order
                if candidate in escaping
            )
            raise SyntaxError(
                f"Variable '{name}' is first defined inside a {loop_kind} loop "
                f"at line {lineno} and read after it. Qamomile cannot represent "
                "that body-local binding as a formal loop result with a "
                "zero-trip initializer, so initialize the variable before "
                "the loop."
            )

    def _wrap_body_with_rebind_probes(
        self,
        flattened_body: list[ast.stmt],
        candidates: list[str],
        classical_candidates: list[str],
        lineno: int,
        region_bind: bool = False,
        array_region_candidates: set[str] | None = None,
    ) -> list[ast.stmt]:
        """Wrap a transformed loop body with rebind snapshot/record probes.

        Prepends an explicit lazy binding snapshot and appends a matching
        explicit post-body binding map so the tracer can compare pre/post
        handle identities for each candidate. Lazy lexical resolvers preserve
        tolerant handling of store-only names without inspecting the Python
        frame. The promotion
        assignments DO load the name directly, which is safe for
        classical candidates: back-edge candidates are incoming values,
        while store-only live-outs are restricted to names already defined
        in the enclosing scope. All
        probes live inside the loop's ``with`` body, so the zero-trip
        trace guard skips them together with the body.

        When ``region_bind`` is true (structured loops), a
        ``name = loop_region_enter(_qm_rebind_snap_N, "name")``
        assignment is additionally injected after the snapshot for each
        read-before-write classical candidate, so the body's carried
        reads go through an explicit region argument (see
        ``loop_region_enter``). Runtime ``while`` carries use the same
        interface; target validation later rejects non-identity state when a
        circuit backend cannot thread it through a measurement-controlled
        loop.

        Args:
            flattened_body (list[ast.stmt]): The already-transformed loop
                body statements.
            candidates (list[str]): Candidate variable names from
                :meth:`_loop_rebind_candidates`.
            classical_candidates (list[str]): The read-before-write subset
                of ``candidates`` eligible for classical rebind records.
            lineno (int): Source line of the loop statement, used for the
                generated nodes.
            region_bind (bool): Inject region-argument entry assignments
                for the classical candidates. Defaults to False.
            array_region_candidates (set[str] | None): Names whose carry is a
                persistent element update rather than a whole-array rebind.
                Defaults to ``None``.

        Returns:
            list[ast.stmt]: The wrapped body; ``flattened_body`` itself
                when ``candidates`` is empty.
        """
        if not candidates:
            return flattened_body

        def _name_tuple(names: list[str]) -> ast.Tuple:
            """Build a ``("a", "b", ...)`` literal over candidate names.

            Args:
                names (list[str]): The names to embed.

            Returns:
                ast.Tuple: A fresh tuple literal node (nodes must not be
                    shared between the two probe calls).
            """
            return ast.Tuple(
                elts=[ast.Constant(value=name) for name in names],
                ctx=ast.Load(),
            )

        def _binding_resolvers(names: list[str]) -> ast.Call:
            """Build an explicit lazy lexical binding map.

            Args:
                names (list[str]): Source names to resolve.

            Returns:
                ast.Call: Call to ``explicit_loop_bindings``.
            """
            return ast.Call(
                func=ast.Name(id="explicit_loop_bindings", ctx=ast.Load()),
                args=[
                    ast.Tuple(
                        elts=[
                            ast.Tuple(
                                elts=[
                                    ast.Constant(value=name),
                                    ast.Lambda(
                                        args=ast.arguments(
                                            posonlyargs=[],
                                            args=[],
                                            kwonlyargs=[],
                                            kw_defaults=[],
                                            defaults=[],
                                        ),
                                        body=ast.Name(id=name, ctx=ast.Load()),
                                    ),
                                ],
                                ctx=ast.Load(),
                            )
                            for name in names
                        ],
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            )

        snap_name = self._get_unique_name("_qm_rebind_snap")
        snap_stmt = ast.Assign(
            targets=[ast.Name(id=snap_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="loop_rebind_snapshot", ctx=ast.Load()),
                args=[_binding_resolvers(candidates), _name_tuple(candidates)],
                keywords=[],
            ),
            lineno=lineno,
        )
        record_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="record_loop_rebinds", ctx=ast.Load()),
                args=[
                    ast.Name(id=snap_name, ctx=ast.Load()),
                    _binding_resolvers(candidates),
                    _name_tuple(candidates),
                    _name_tuple(classical_candidates),
                ],
                keywords=[],
            ),
            lineno=lineno,
        )
        entry_stmts: list[ast.stmt] = []
        if region_bind:
            entry_stmts = [
                ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="loop_region_enter", ctx=ast.Load()),
                        args=[
                            ast.Name(id=snap_name, ctx=ast.Load()),
                            ast.Constant(value=name),
                            ast.Constant(
                                value=name in (array_region_candidates or set())
                            ),
                        ],
                        keywords=[],
                    ),
                    lineno=lineno,
                )
                for name in classical_candidates
            ]
        return [snap_stmt, *entry_stmts, *flattened_body, record_stmt]

    @staticmethod
    def _named_region_values(names: list[str]) -> ast.Tuple:
        """Build a tuple of explicit ``(name, value)`` region bindings.

        Args:
            names (list[str]): Source variable names in interface order.

        Returns:
            ast.Tuple: Tuple literal containing named lexical values.
        """
        return ast.Tuple(
            elts=[
                ast.Tuple(
                    elts=[
                        ast.Constant(value=name),
                        ast.Name(id=name, ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
                for name in names
            ],
            ctx=ast.Load(),
        )

    @staticmethod
    def _lazy_region_bindings(names: list[str]) -> ast.Call:
        """Build a lazy lexical binding map without frame inspection.

        Args:
            names (list[str]): Source names to resolve.

        Returns:
            ast.Call: Call to ``explicit_loop_bindings`` with one resolver per
                source name.
        """
        return ast.Call(
            func=ast.Name(id="explicit_loop_bindings", ctx=ast.Load()),
            args=[
                ast.Tuple(
                    elts=[
                        ast.Tuple(
                            elts=[
                                ast.Constant(value=name),
                                ast.Lambda(
                                    args=ast.arguments(
                                        posonlyargs=[],
                                        args=[],
                                        kwonlyargs=[],
                                        kw_defaults=[],
                                        defaults=[],
                                    ),
                                    body=ast.Name(id=name, ctx=ast.Load()),
                                ),
                            ],
                            ctx=ast.Load(),
                        )
                        for name in names
                    ],
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        )

    def visit_While(self, node: ast.While) -> Any:
        """Transform a qkernel while loop into a traced context-manager body.

        Args:
            node (ast.While): While statement to validate and transform.

        Returns:
            Any: Replacement ``ast.With`` node invoking ``while_loop``.

        Raises:
            SyntaxError: If the loop has an ``else`` clause, its condition
                performs a quantum operation, or a body-local value escapes.
        """
        if node.orelse:
            raise SyntaxError("while ... else is not supported in @qkernel")
        if self._has_return_in_region(node.body):
            raise SyntaxError(
                "'return' inside a while loop is not supported in @qkernel"
            )
        self._reject_loop_control(node.body, "while")
        self._reject_named_expression_in_condition(
            node.test,
            construct="while",
        )
        # Check for quantum operations in while condition
        self._check_no_quantum_ops_in_condition(node.test, node.lineno)
        self._reject_loop_local_escapes(node.body, set(), node.lineno, "while")
        # Compute rebind-probe candidates on the ORIGINAL body, before
        # nested transforms rewrite assignments into emit_if calls.
        rebind_candidates, classical_rebind_candidates = self._loop_rebind_candidates(
            node.body, set()
        )
        # Transform nested control flow first (with definition tracking)
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
        signature = self._region_signatures.get(
            RegionLocation("while", node.lineno, node.col_offset)
        )
        if signature is not None:
            classical_rebind_candidates = [
                name for name in signature.carried if name not in cond_loads
            ]
            rebind_candidates = list(
                dict.fromkeys([*rebind_candidates, *signature.carried])
            )
        capture_names = list(signature.captures) if signature is not None else []
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

        flattened_body = self._wrap_body_with_rebind_probes(
            flattened_body,
            rebind_candidates,
            classical_rebind_candidates,
            node.lineno,
            region_bind=True,
        )

        # Build ``lambda: <condition>``
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

        # Build the ``while_loop(lambda: cond)`` call
        while_loop_call = ast.Call(
            func=ast.Name(id="while_loop", ctx=ast.Load()),
            args=[lambda_node],
            keywords=[
                ast.keyword(
                    arg="captures",
                    value=self._named_region_values(capture_names),
                )
            ],
        )

        # Build the with-statement
        with_item = ast.withitem(context_expr=while_loop_call, optional_vars=None)
        with_stmt = ast.With(
            items=[with_item],
            body=flattened_body,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return [
            with_stmt,
            *self._region_result_stmts(
                classical_rebind_candidates,
                node.lineno,
            ),
        ]

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

    def _validate_items_loop(self, node: ast.For) -> list[str]:
        """Validate an items() for-loop and return binding variable names.

        Checks the call-site arguments and target pattern for items() /
        module.items() / d.items() forms.
        """
        assert isinstance(node.iter, ast.Call)
        call = node.iter
        # from qamomile.circuit import items; items(d) form
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
        # Attribute form: distinguish module.items(d) vs d.items()
        elif isinstance(call.func, ast.Attribute):
            func_value = call.func.value
            # Resolve the receiver to check if it's a module object
            receiver_obj = (
                self._namespace.get(func_value.id)
                if isinstance(func_value, ast.Name)
                else None
            )
            if isinstance(receiver_obj, types.ModuleType):
                # module.items(d) form: require exactly 1 positional arg, no keywords
                assert isinstance(func_value, ast.Name)
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
                # d.items() form (parameter, global dict, local variable, etc.):
                # dict.items() takes no arguments.
                if call.args or call.keywords:
                    receiver_expr = ast.unparse(call.func.value)
                    raise SyntaxError(
                        "items() iteration over a dict must use "
                        f"'{receiver_expr}.items()' with no arguments."
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
        return self._extract_tuple_vars(node.target)

    def _validate_range_loop(self, node: ast.For) -> list[str]:
        """Validate a range() for-loop and return binding variable names.

        Checks the call-site arguments and target for range() /
        module.range() forms.
        """
        assert isinstance(node.iter, ast.Call)
        range_call = node.iter
        # Derive the caller name from the AST for error messages
        if isinstance(range_call.func, ast.Attribute) and isinstance(
            range_call.func.value, ast.Name
        ):
            range_callee = f"{range_call.func.value.id}.range"
        else:
            range_callee = "range"
        range_label = f"{range_callee}()"
        num_args = len(range_call.args)
        # Keyword arguments (e.g. range(stop=n)) are not consumed by
        # _transform_for_range; reject them explicitly to avoid silent loss.
        if getattr(range_call, "keywords", None):
            raise SyntaxError(
                f"{range_label} does not support keyword arguments in @qkernel; "
                f"use positional arguments like {range_callee}(stop) or "
                f"{range_callee}(start, stop, step)."
            )
        # range(): requires 1-3 positional arguments
        if num_args < 1 or num_args > 3:
            raise SyntaxError(
                f"{range_label} requires 1-3 arguments: "
                f"{range_callee}(stop), {range_callee}(start, stop), or "
                f"{range_callee}(start, stop, step)"
            )
        # range(): target must be a single variable
        if not isinstance(node.target, ast.Name):
            raise SyntaxError(
                f"{range_label} iteration requires a single loop variable, "
                f"got: {ast.dump(node.target)}"
            )
        return [node.target.id]

    def _validate_sequence_loop(self, node: ast.For) -> NoReturn:
        """Reject direct sequence iteration in @qkernel.

        Direct iteration like 'for item in vector:' doesn't support in-place
        modification in @qkernel contexts.  Always raises ``SyntaxError``.
        """
        if isinstance(node.target, ast.Name):
            item_var = node.target.id
        else:
            item_var = "item"

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

    def _validate_for_loop(self, node: ast.For) -> list[str]:
        """Validate a for-loop node and return binding variable names.

        Dispatches to kind-specific validators and checks parameter
        shadowing.  Raises ``SyntaxError`` for invalid patterns so that
        ``QKernel.__init__`` propagates them instead of falling back to
        the original function.

        Args:
            node (ast.For): Loop node to validate.

        Returns:
            list[str]: Names bound by the range or items loop target.

        Raises:
            SyntaxError: If the loop kind or target is unsupported, a
                binding shadows a parameter, or a loop binding is read
                after the loop.
        """
        if self._is_items_call(node.iter):
            binding_names = self._validate_items_loop(node)
        elif self._is_range_call(node.iter):
            binding_names = self._validate_range_loop(node)
        else:
            self._validate_sequence_loop(node)  # always raises

        # Check for parameter shadowing (common to all for-loop variants)
        for var_name in binding_names:
            if var_name in self._param_names:
                raise SyntaxError(
                    f"Loop variable '{var_name}' shadows a function parameter. "
                    f"Use a different variable name to avoid silent value loss."
                )
            if var_name in self._after_stmt_load_vars:
                raise SyntaxError(
                    f"Loop variable '{var_name}' is read after the loop. "
                    "Post-loop loop-target values are not supported in "
                    "@qkernel functions; assign the value you need to a "
                    "separate variable inside the loop."
                )

        self._reject_loop_local_escapes(
            node.body,
            set(binding_names),
            node.lineno,
            "for",
        )

        return binding_names

    def visit_For(self, node: ast.For) -> Any:
        """Transform a supported qkernel for loop and attach rebind probes.

        Args:
            node (ast.For): Range or items loop to validate and transform.

        Returns:
            Any: Guarded AST statement implementing the range or items loop.

        Raises:
            SyntaxError: If the loop form, target, shadowing, or escaping
                bindings violate qkernel loop rules.
            AssertionError: If validated dispatch reaches an unknown loop kind.
        """
        if node.orelse:
            raise SyntaxError("for ... else is not supported in @qkernel")
        if self._has_return_in_region(node.body):
            raise SyntaxError("'return' inside a for loop is not supported in @qkernel")
        self._reject_loop_control(node.body, "for")

        # Validate target per loop-kind BEFORE extracting binding names.
        # This raises SyntaxError for invalid placeholder-loop targets
        # instead of generic NotImplementedError that would be swallowed
        # by QKernel.__init__'s fallback.
        all_binding_names = self._validate_for_loop(node)

        # Compute rebind-probe candidates on the ORIGINAL body, before
        # nested transforms rewrite assignments into emit_if calls.
        rebind_candidates, classical_rebind_candidates = self._loop_rebind_candidates(
            node.body, set(all_binding_names)
        )
        legacy_classical_candidates = set(classical_rebind_candidates)
        signature = self._region_signatures.get(
            RegionLocation("for", node.lineno, node.col_offset)
        )
        capture_names = list(signature.captures) if signature is not None else []
        array_region_candidates: set[str] = set()
        if signature is not None:
            classical_rebind_candidates = [
                name for name in signature.carried if name not in all_binding_names
            ] + [
                name
                for name in classical_rebind_candidates
                if name not in signature.carried
            ]
            rebind_candidates = list(
                dict.fromkeys([*rebind_candidates, *signature.carried])
            )
            array_region_candidates = (
                set(signature.carried) - legacy_classical_candidates
            )

        # Transform nested control flow first (with definition tracking)
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

        flattened_body = self._wrap_body_with_rebind_probes(
            flattened_body,
            rebind_candidates,
            classical_rebind_candidates,
            node.lineno,
            region_bind=True,
            array_region_candidates=array_region_candidates,
        )

        # Dispatch to the appropriate transform.
        # Sequence iteration is already rejected by _validate_for_loop.
        if self._is_items_call(node.iter):
            return self._transform_for_items(
                node,
                flattened_body,
                classical_rebind_candidates,
                capture_names,
            )

        if self._is_range_call(node.iter):
            return self._transform_for_range(
                node,
                flattened_body,
                classical_rebind_candidates,
                capture_names,
            )

        # [FOR DEVELOPER]
        # Since _validate_for_loop should have rejected any non-items, non-range loops,
        # reaching this point indicates a bug in the validation logic.
        # (sequence pattern has been explicitly rejected by _validate_sequence_loop).
        # Raising an explicit error helps catch such issues during development.
        raise AssertionError(
            "Unreachable: _validate_for_loop should have rejected this loop. "
            f"iter={ast.dump(node.iter)}"
        )

    def _region_result_stmts(
        self, classical_candidates: list[str], lineno: int
    ) -> list[ast.stmt]:
        """Build post-loop ``loop_region_result`` rebinding assignments.

        One ``name = loop_region_result("name", name)`` statement per
        loop-carried classical candidate, to run immediately after
        the loop's ``with`` block (inside the zero-trip guard) so
        post-loop reads reference the loop operation's region-argument
        result value instead of the body's final yielded value.

        Args:
            classical_candidates (list[str]): Read-before-write and
                store-only-live-out classical candidate names of the loop.
            lineno (int): Source line of the loop statement.

        Returns:
            list[ast.stmt]: The rebinding assignments (empty when there
                are no candidates).
        """
        return [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="loop_region_result", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=name),
                        ast.Name(id=name, ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                lineno=lineno,
            )
            for name in classical_candidates
        ]

    def _transform_for_range(
        self,
        node: ast.For,
        flattened_body: list[ast.stmt],
        classical_candidates: list[str],
        capture_names: list[str],
    ) -> ast.stmt:
        """Transform a range loop into a guarded ``for_loop`` context.

        Supports patterns:
            for i in range(stop):  ->  for_loop(0, stop, 1)
            for i in range(start, stop):  ->  for_loop(start, stop, 1)
            for i in range(start, stop, step):  ->  for_loop(start, stop, step)

        When the body has classical rebind candidates, post-loop
        ``loop_region_result`` assignments follow the with-statement
        inside the zero-trip trace guard.

        Args:
            node (ast.For): Validated range-loop AST node.
            flattened_body (list[ast.stmt]): Transformed body statements.
            classical_candidates (list[str]): Candidate names that may
                receive formal loop results.
            capture_names (list[str]): Read-only region inputs in static
                interface order.

        Returns:
            ast.stmt: Guarded context-manager statement implementing the loop.
        """

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

        # Read the loop variable name (already validated in _validate_for_loop)
        assert isinstance(node.target, ast.Name)
        loop_var_name = node.target.id

        # Build the ``for_loop(start, stop, step, var_name)`` call
        for_loop_call = ast.Call(
            func=ast.Name(id="for_loop", ctx=ast.Load()),
            args=[start_arg, stop_arg, step_arg, ast.Constant(value=loop_var_name)],
            keywords=[
                ast.keyword(
                    arg="captures",
                    value=self._named_region_values(capture_names),
                )
            ],
        )

        # Bind the loop variable via ``as i``
        with_item = ast.withitem(
            context_expr=for_loop_call,
            optional_vars=node.target,  # reuse ``i`` as-is
        )

        with_stmt = ast.With(
            items=[with_item],
            body=flattened_body,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return ast.If(
            test=ast.Call(
                func=ast.Name(id="should_trace_for_loop", ctx=ast.Load()),
                args=[
                    copy.deepcopy(start_arg),
                    copy.deepcopy(stop_arg),
                    copy.deepcopy(step_arg),
                ],
                keywords=[],
            ),
            body=[
                with_stmt,
                *self._region_result_stmts(classical_candidates, node.lineno),
            ],
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def _transform_for_items(
        self,
        node: ast.For,
        flattened_body: list[ast.stmt],
        classical_candidates: list[str],
        capture_names: list[str],
    ) -> ast.stmt:
        """Transform 'for (k, v) in items(d)' or 'for (k, v) in d.items()' to 'with for_items(d, [...], "v")'.

        The with-statement is wrapped in ``if should_trace_items_loop(d):``
        so a compile-time-known EMPTY dict skips tracing the body, exactly
        like the ``qmc.range`` zero-trip guard. When the body has
        classical rebind candidates, post-loop ``loop_region_result``
        assignments follow the with-statement inside that guard.

        Supports patterns:
            for key, value in items(d):  ->  for_items(d, ["key"], "value")
            for key, value in d.items():  ->  for_items(d, ["key"], "value")
            for (i, j), value in items(d):  ->  for_items(d, ["i", "j"], "value")
            for (i, j), value in d.items():  ->  for_items(d, ["i", "j"], "value")

        Args:
            node (ast.For): Validated items-loop AST node.
            flattened_body (list[ast.stmt]): Transformed body statements.
            classical_candidates (list[str]): Candidate names that
                may receive loop carry results.
            capture_names (list[str]): Read-only region inputs in static
                interface order.

        Returns:
            ast.stmt: Guarded with-statement implementing the items loop.
        """
        # Extract the dict argument from items(d) or d.items() call
        # (_validate_for_loop guarantees one of these patterns)
        if node.iter.args:  # type: ignore  # items(d) pattern
            dict_arg = node.iter.args[0]  # type: ignore
        else:  # d.items() pattern
            dict_arg = node.iter.func.value  # type: ignore

        # Parse the target pattern: (key, value) — validated by _validate_for_loop
        assert isinstance(node.target, ast.Tuple)
        key_target = node.target.elts[0]
        value_target = node.target.elts[1]

        # Extract key variable names (may be tuple like (i, j))
        key_vars = self._extract_tuple_vars(key_target)

        # Extract value variable name (validated by _validate_for_loop)
        assert isinstance(value_target, ast.Name)
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
            keywords=[
                ast.keyword(
                    arg="captures",
                    value=self._named_region_values(capture_names),
                )
            ],
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

        # Mirror the qmc.range zero-trip guard: a compile-time-known
        # EMPTY dict must not trace the body at all, so post-loop code
        # keeps the pre-loop handles exactly like Python's zero-pass
        # iteration (and no loop op / rebind record is created).
        return ast.If(
            test=ast.Call(
                func=ast.Name(id="should_trace_items_loop", ctx=ast.Load()),
                args=[copy.deepcopy(dict_arg)],
                keywords=[],
            ),
            body=[
                with_stmt,
                *self._region_result_stmts(classical_candidates, node.lineno),
            ],
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

    def visit_If(self, node: ast.If) -> Any:
        region_signature = self._region_signatures.get(
            RegionLocation("if", node.lineno, node.col_offset)
        )
        self._reject_named_expression_in_condition(
            node.test,
            construct="if",
        )
        # Collect variables from the pre-transform AST (post generic_visit would include generated names)
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

        # Existing vars re-assigned in any branch may need merge outputs even
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
        # generate an unnecessary merge slot whose physical resources may
        # differ across branches, causing EmitError at emit time.
        # Variables that are dead but NOT stored in any branch pass through
        # unchanged (their merge has identical resources) and are harmless.
        stored_in_branches = collector_body.store_vars | collector_orelse.store_vars
        dead_modified = (stored_in_branches & input_vars_set) - after_loads
        live_shared = shared_new_locals & after_loads
        live_reassigned_existing = reassigned_existing & after_loads

        input_vars = sorted(input_vars_set)
        capture_indices = tuple(
            input_vars.index(name)
            for name in (
                region_signature.inputs if region_signature is not None else ()
            )
            if name in input_vars
        )
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

        # Transform nested control flow (with definition tracking)
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

        # Detect and transform explicit returns
        orelse_body = node.orelse if node.orelse else []
        body_has_return = self._has_top_level_return(node.body)
        orelse_has_return = self._has_top_level_return(orelse_body)

        # One-sided return is not supported: when only one branch returns,
        # the merge cannot pair the return value with a corresponding
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

        # Dead-rebind probes: a pre-existing variable reassigned in a
        # branch but never read after the if is dead-store-eliminated
        # from ``output_vars`` (and, when its old value is unread, from
        # ``input_vars`` too), so no merge ever sees it — yet the rebind
        # still drops the pre-branch quantum state on the rebinding
        # path, exactly like the top-level rebind the decoration-time
        # analyzer rejects. Each branch body therefore returns, after
        # the ordinary outputs, an unbound-safe probe of every such
        # candidate's post-branch binding (``dead_rebind_binding``
        # returns a sentinel when the name is not bound in the body), so
        # ``emit_if`` can record the rebind without changing the merged
        # output contract. Candidates come from the function-wide lexical
        # set, not the branch-narrowed ``outer_defined``: Python names
        # have function scope, so a nested branch assigning a name whose
        # definition sits outside the enclosing branch's inputs still
        # rebinds that variable (the pre-branch handle is resolved
        # through the enclosing emit_if captures at run time). The
        # candidate base is ``store_vars``, not ``locally_defined_vars``:
        # a read-before-write reassignment (``q = qmc.x(q)`` followed by
        # ``q = qmc.qubit("fresh")``) is not "locally defined" yet still
        # ends the branch on a different register. Extra candidates are
        # harmless — pure gate self-updates keep their logical_id and
        # produce no record.
        rebind_record_existing = (
            collector_body.store_vars | collector_orelse.store_vars
        ) & (outer_defined | self._lexical_defined_vars)
        branch_store_order = tuple(
            dict.fromkeys((*collector_body.store_order, *collector_orelse.store_order))
        )
        dead_rebind_candidates = [
            name
            for name in branch_store_order
            if name in rebind_record_existing - set(output_vars)
        ]

        def _dead_probe_exprs() -> list[ast.expr]:
            """Build fresh probe expressions for the dead candidates.

            Returns:
                list[ast.expr]: One ``dead_rebind_binding(bindings,
                    "name")`` call per candidate. A fresh list per call —
                    AST nodes must not be shared between the two body
                    functions.
            """
            return [
                ast.Call(
                    func=ast.Name(id="dead_rebind_binding", ctx=ast.Load()),
                    args=[
                        self._lazy_region_bindings([name]),
                        ast.Constant(value=name),
                    ],
                    keywords=[],
                )
                for name in dead_rebind_candidates
            ]

        # True Body: inputs=input_vars, return=output_vars (+ dead probes)
        true_def, true_name = self._create_inner_func(
            "body",
            true_body,
            input_vars,
            node.lineno,
            return_var_names=output_vars,
            extra_return_exprs=_dead_probe_exprs(),
        )

        # False Body: inputs=input_vars, return=output_vars (+ dead probes)
        false_def, false_name = self._create_inner_func(
            "body",
            false_body,
            input_vars,
            node.lineno,
            return_var_names=output_vars,
            extra_return_exprs=_dead_probe_exprs(),
        )

        # var_list: input_vars values (passed to emit_if)
        var_list = ast.List(
            elts=[ast.Name(id=v, ctx=ast.Load()) for v in input_vars],
            ctx=ast.Load(),
        )

        # Rebind-record probe arguments: pre-branch bindings of every
        # pre-existing variable reassigned in a branch. A reassigned
        # variable whose old value is dead is deliberately NOT in
        # ``input_vars`` (its old value must not force a merge input), so
        # ``emit_if`` cannot see its pre-branch binding through
        # ``var_list``; an explicit lazy binding map
        # captures it at the call site instead. The candidate analysis
        # is lexical (``reassigned_existing`` uses the outer-defined
        # set), so the helper skips names a preceding pure-store if left
        # unbound at runtime rather than referencing them directly.
        # ``output_names`` gives ``emit_if`` the positional mapping of
        # the branch-result tuples; ``dead_names`` names the probe tail
        # appended after the outputs (see the dead-rebind probes above).
        record_candidates = [
            name
            for name in branch_store_order
            if name in rebind_record_existing & set(output_vars)
        ]
        pre_binding_names = record_candidates + dead_rebind_candidates
        call_keywords: list[ast.keyword] = []
        if capture_indices:
            call_keywords.append(
                ast.keyword(
                    arg="capture_indices",
                    value=ast.Tuple(
                        elts=[ast.Constant(value=index) for index in capture_indices],
                        ctx=ast.Load(),
                    ),
                )
            )
        if pre_binding_names:
            call_keywords.append(
                ast.keyword(
                    arg="output_names",
                    value=ast.Tuple(
                        elts=[ast.Constant(value=v) for v in output_vars],
                        ctx=ast.Load(),
                    ),
                )
            )
            call_keywords.append(
                ast.keyword(
                    arg="rebind_pre_bindings",
                    value=ast.Call(
                        func=ast.Name(id="branch_rebind_pre_bindings", ctx=ast.Load()),
                        args=[
                            self._lazy_region_bindings(pre_binding_names),
                            ast.Tuple(
                                elts=[ast.Constant(value=v) for v in pre_binding_names],
                                ctx=ast.Load(),
                            ),
                        ],
                        keywords=[],
                    ),
                )
            )
        if dead_rebind_candidates:
            call_keywords.append(
                ast.keyword(
                    arg="dead_names",
                    value=ast.Tuple(
                        elts=[ast.Constant(value=v) for v in dead_rebind_candidates],
                        ctx=ast.Load(),
                    ),
                )
            )

        call_expr = ast.Call(
            func=ast.Name(id=self.if_func_name, ctx=ast.Load()),
            args=[
                ast.Name(id=cond_name, ctx=ast.Load()),
                ast.Name(id=true_name, ctx=ast.Load()),
                ast.Name(id=false_name, ctx=ast.Load()),
                var_list,
            ],
            keywords=call_keywords,
        )

        # Assignment: output_vars = emit_if(...)
        assign_node = self._create_assignment_node(output_vars, call_expr, node.lineno)

        result_stmts: list[ast.stmt] = [cond_def, true_def, false_def]

        if has_return:
            assert ret_var_name is not None
            # _if_ret_N = None (initialization)
            ret_name_store: ast.expr = ast.Name(id=ret_var_name, ctx=ast.Store())
            init_stmt = ast.Assign(
                targets=[ret_name_store],
                value=ast.Constant(value=None),
                lineno=node.lineno,
            )
            result_stmts.append(init_stmt)

        result_stmts.append(assign_node)

        if has_return:
            assert ret_var_name is not None
            # return _if_ret_N
            outer_return = ast.Return(
                value=ast.Name(id=ret_var_name, ctx=ast.Load()),
                lineno=node.lineno,
            )
            result_stmts.append(outer_return)

        return result_stmts


def transform_control_flow(
    func: Callable[..., Any],
    *,
    region_signatures: dict[RegionLocation, RegionSignature] | None = None,
) -> Callable[..., Any]:
    """Rewrite Python control flow into tracer-visible region builders.

    Args:
        func (Callable[..., Any]): Raw qkernel function.
        region_signatures (dict[RegionLocation, RegionSignature] | None):
            Precomputed explicit region interfaces. Defaults to ``None``.

    Returns:
        Callable[..., Any]: Transformed function executed by the tracer.

    Raises:
        SyntaxError: If source retrieval or parsing fails.
        NotImplementedError: If a referenced closure value is unavailable.
    """
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
    # Re-anchor the tree to the original source file before any transform
    # runs, so that (a) transform-time diagnostics report absolute line
    # numbers and (b) frames executing the traced kernel body carry the
    # user's real ``file:line``. ``getsource`` yields line numbers relative
    # to the decorator line, so shifting by ``co_firstlineno - 1`` restores
    # absolute positions (``dedent`` only changes columns, never lines).
    # Nodes synthesized by the transformer inherit these absolute positions
    # via ``fix_missing_locations`` below. This gives readable tracebacks
    # for any error raised while tracing — and lets trace-time diagnostics
    # such as ``Handle.consume`` report the offending source line — instead
    # of pointing into an opaque ``<qamomile-dsl>`` buffer.
    ast.increment_lineno(tree, func.__code__.co_firstlineno - 1)
    source_filename = inspect.getsourcefile(func) or "<qamomile-dsl>"

    if region_signatures is None:
        definition = next(
            (
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func.__name__
            ),
            None,
        )
        if definition is None:
            raise SyntaxError(f"Cannot locate function {func.__name__!r} in source")
        region_signatures = analyze_region_signatures(definition)

    # Collect global names (modules, builtins, etc.)
    global_names = set(func.__globals__.keys())

    # Exclude closure variables too (so VariableCollector does not add them to target_vars).
    # Closure values are injected into name_space later, so inner functions can still access them.
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
        region_signatures=region_signatures,
    )
    tree = transformer.visit(tree)

    # Strip decorators (to prevent recursion)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            node.decorator_list = []

    ast.fix_missing_locations(tree)

    # Inherit the original function's globals
    name_space = func.__globals__.copy()
    generated_globals = {
        "while_loop": while_loop,
        "for_loop": for_loop,
        "should_trace_for_loop": should_trace_for_loop,
        "should_trace_items_loop": should_trace_items_loop,
        "for_items": for_items,
        "emit_if": emit_if,
        "explicit_loop_bindings": explicit_loop_bindings,
        "branch_rebind_pre_bindings": branch_rebind_pre_bindings,
        "dead_rebind_binding": dead_rebind_binding,
        "loop_rebind_snapshot": loop_rebind_snapshot,
        "loop_region_enter": loop_region_enter,
        "loop_region_result": loop_region_result,
        "record_loop_rebinds": record_loop_rebinds,
        "Any": Any,  # For type annotations in generated code
    }
    name_space.update(generated_globals)

    # Add closure variables (e.g. names imported inside the function).
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

    code_obj = compile(tree, filename=source_filename, mode="exec")
    exec(code_obj, name_space)

    transformed = name_space[func.__name__]
    transformed.__qamomile_generated_globals__ = generated_globals
    return transformed


# ---------------------------------------------------------------------------
# Quantum rebind analysis
# ---------------------------------------------------------------------------


class RebindSourceKind(enum.StrEnum):
    """Discriminator for the source of a detected rebind violation.

    Each value classifies *why* the analyzer believes an existing
    quantum binding is being silently discarded, and lets downstream
    error-message formatting render a domain-appropriate explanation
    instead of forcing a generic "different quantum variable" sentence
    onto, e.g., a fresh allocation.

    Members:
        DIRECT_ALIAS: ``q = other_q`` or ``q = qs[i]``.
        QUANTUM_ARG: ``q = f(other_q, ...)`` where ``other_q`` has a
            different origin than ``q``.
        FRESH_ALLOCATION: ``q = qm.qubit(...)`` /
            ``qm.qubit_array(...)`` — the original quantum state is
            silently discarded in favor of a freshly allocated one.
        UNKNOWN_CALL: ``q = some_func(...)`` where the call references
            no known quantum variable and is not a recognized quantum
            constructor; conservatively treated as a rebind because
            the original ``q`` is not threaded through the RHS.
        CHAINED_ASSIGNMENT: ``q1 = q2 = expr`` where at least one
            target is an existing quantum variable; chained binding
            semantics are too ambiguous to verify self-update.
    """

    DIRECT_ALIAS = "direct_alias"
    QUANTUM_ARG = "quantum_arg"
    FRESH_ALLOCATION = "fresh_allocation"
    UNKNOWN_CALL = "unknown_call"
    CHAINED_ASSIGNMENT = "chained_assignment"


@dataclass
class RebindViolation:
    """A detected forbidden quantum variable rebinding.

    Attributes:
        target_name (str): Name of the LHS variable being overwritten.
        source_name (str | None): Name of the underlying quantum
            variable the new value is derived from — for a
            ``Subscript`` source like ``a = qs[i]`` this is the base
            array name ``qs`` (the quantum binding being aliased
            from), not the full subscript expression. ``None`` when
            the source has no statically-known quantum name — e.g. a
            fresh allocation via ``qm.qubit(...)`` or an opaque call
            whose return type cannot be inferred.
        source_expr (str | None): Full RHS source string, used in the
            rendered error pattern when it differs from ``source_name``.
            Set via ``ast.unparse(value)`` for ``Subscript`` sources
            (``"qs[i]"``) so the message shows the actual code shape;
            ``None`` for plain ``Name`` sources where ``source_name``
            already matches the RHS text, and for kinds without a
            single-name source (fresh allocations, opaque calls,
            chained assignments).
        source_kind (RebindSourceKind): Discriminator for downstream
            error message formatting. See :class:`RebindSourceKind` for
            the meaning of each member.
        func_name (str | None): Name of the function/method invoked on
            the RHS, when the RHS is a ``Call``. ``None`` for direct
            aliases.
        lineno (int): 1-based source line of the offending assignment,
            counted from the first statement of the function body
            (i.e., the first body statement is line 1, regardless of
            blank lines or comments between the ``def`` line and that
            first statement). The analyzer walks
            ``ast.parse(textwrap.dedent(inspect.getsource(func)))`` so
            raw ``ast`` line numbers include the decorator / ``def``
            lines; ``collect_quantum_rebind_violations`` subtracts
            ``node.body[0].lineno - 1`` from every violation to produce
            this body-relative form.
    """

    target_name: str
    source_name: str | None
    source_kind: RebindSourceKind
    func_name: str | None
    lineno: int
    source_expr: str | None = None


# Recognized quantum-handle constructors. The analyzer treats the LHS of
# ``q = qubit(...)`` / ``q = qubit_array(...)`` (or their attribute
# forms ``qm.qubit(...)`` etc.) as a freshly-allocated quantum value:
# overwriting an existing quantum binding is a rebind violation, and a
# new binding starts a fresh origin so subsequent aliases can be tracked.
# Alias-imported constructors (``from ... import qubit as factory``) are
# out of scope for now — see ``_is_quantum_constructor_call`` for the
# four syntactic forms recognized.
_QUANTUM_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"qubit", "qubit_array"})

# Classical-returning frontend primitives. The analyzer models these as
# consuming their quantum arguments: every name in ``quantum_vars`` that
# shares an origin with one of the quantum args is removed after the
# call. This avoids false-positive rebind violations on subsequent
# fresh-allocation rebinds of a post-measurement variable.
_CLASSICAL_RETURNING_CALL_NAMES: frozenset[str] = frozenset({"measure", "expval"})


class QuantumRebindAnalyzer(ast.NodeVisitor):
    """Detects forbidden quantum variable reassignment at the AST level.

    Forbidden patterns (target is an *existing* quantum variable):

      - ``a = b``                 where ``b`` is quantum with a different origin
      - ``a = f(b, ...)``         where ``b`` is quantum with a different origin
      - ``a = qm.qubit("...")``   silently discarding ``a``
      - ``a = some_opaque_call()`` where the call has no known quantum source
      - ``a = b = expr``          chained assignment touching a quantum name
      - ``a: qm.Qubit = ...``     (annotated form of the above)

    Allowed patterns:

      - ``a = f(a, ...)``         (self-update)
      - ``new = f(b, ...)``       (new binding — target was not quantum before)
      - ``alias = q``             (new alias — target was not quantum before)
      - ``a, b = f(a, b)``        / ``a, b = (g(b), h(a))`` (1-to-1 quantum permutation)

    The analyzer is a single-pass ``ast.NodeVisitor`` and does not
    model Python control flow precisely. To keep compile-time-if
    dead-branch rebinds — which the IR's ``CompileTimeIfLoweringPass``
    will later resolve by selecting one branch and discarding the
    other — from being rejected at decoration time, ``visit_If`` /
    ``visit_For`` / ``visit_While`` route every branch through
    :meth:`_visit_branch_scope`, which snapshots ``quantum_vars``
    before and restores it after each ``body`` / ``orelse``, AND
    truncates any violations recorded inside the branch back to the
    pre-branch length. Top-level (non-branch-internal) rebinds are
    flagged as usual; branch-internal rebinds are deliberately not
    reported at decoration time. Runtime-branch and loop-body discards
    are instead rejected at the IR layer by
    ``reject_control_flow_quantum_discard`` (in
    ``qamomile.circuit.transpiler.passes.analyze``), which can tell
    compile-time branches from runtime ones.
    """

    def __init__(self, quantum_param_names: set[str]) -> None:
        """Initialize the analyzer with the kernel's quantum parameter names.

        Args:
            quantum_param_names (set[str]): Names of kernel parameters
                whose annotated type is a quantum handle (``Qubit`` /
                ``Vector[Qubit]``). Each is seeded into ``quantum_vars``
                as its own origin.
        """
        # name → origin (the parameter or fresh-allocation name it
        # traces back to). All aliases of the same physical quantum
        # value share an origin, which lets the analyzer recognize
        # self-update versus rebind across alias chains.
        self.quantum_vars: dict[str, str] = {n: n for n in quantum_param_names}
        self.violations: list[RebindViolation] = []

    # ------------------------------------------------------------------
    # visitor
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        """Dispatch ``a = expr`` / ``a, b = expr`` / ``a = b = expr``.

        Args:
            node (ast.Assign): The assignment statement.
        """
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self._check_single_assign(target.id, node.value, node.lineno)
            elif isinstance(target, (ast.Tuple, ast.List)):
                # ``[a] = [expr]`` is the same unpacking assignment as
                # ``a, = expr,`` — route list targets through the tuple
                # path so the spelling cannot dodge the rebind rules.
                self._check_tuple_assign(target, node.value, node.lineno)
        else:
            # Chained ``a = b = ...``. Conservative: flag if any target
            # is an existing quantum variable. A future PR can model
            # the chained semantics precisely.
            self._check_chained_assign(node.targets, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Dispatch ``q: qm.Qubit = expr`` through the single-assign path.

        Args:
            node (ast.AnnAssign): The annotated assignment statement.
                Annotation-only forms (``q: qm.Qubit`` with no RHS) are
                ignored — there is nothing to rebind.
        """
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._check_single_assign(node.target.id, node.value, node.lineno)
        self.generic_visit(node)

    def _visit_branch_scope(
        self,
        node: ast.If | ast.For | ast.While,
    ) -> None:
        """Visit a control-flow node's body / orelse with branch-local scope.

        Used by :meth:`visit_If`, :meth:`visit_For`, and
        :meth:`visit_While`. The analyzer is flow-insensitive: it walks
        the AST in source order without modeling branching. To prevent
        branch-local effects from leaking out, each branch is analyzed
        against a snapshot of both ``quantum_vars`` and the current
        ``violations`` length:

          - ``quantum_vars`` is restored after every branch body so a
            branch-local binding (e.g. ``q = qm.qubit("inner")`` inside
            an ``if``) does not leak into post-branch analysis and
            spuriously flag a later store-only rebind of the same name.
          - ``violations`` are truncated to their pre-branch length so
            that legitimate compile-time-if dead-branch patterns
            (``if flag: ... ; else: alt = qubit_array(...); q = alt``)
            are not rejected at decoration time. These patterns rely on
            the compile-time-if lowering pass selecting one branch and
            discarding the other; the AST analyzer cannot tell
            compile-time-if from runtime-if, so it conservatively
            allows branch-internal rebinds to keep the user's
            compile-time-if usage working.

        **Division of labor.** Suppressing branch-internal violations
        means a kernel that genuinely discards an outer quantum binding
        inside a runtime ``if`` / ``for`` / ``while`` branch (e.g. a
        runtime ``if cond: q = qm.qubit("fresh")``, or a loop body that
        rebinds ``q`` without consuming it) is NOT reported by
        ``collect_quantum_rebind_violations``. Those runtime cases are
        instead rejected at the IR layer by
        ``reject_control_flow_quantum_discard`` (in
        ``qamomile.circuit.transpiler.passes.analyze``): if branches are
        classified by resolving their conditions the same way
        ``CompileTimeIfLoweringPass`` does — exactly the compile-time /
        runtime distinction this single-pass AST analyzer cannot make —
        while loop bodies are checked unconditionally on live paths,
        since a loop-body discard fires on every iteration. Top-level
        (non-branch-internal) bypasses continue to be caught at
        decoration time.

        Args:
            node (ast.If | ast.For | ast.While): The control-flow node.
                ``body`` and ``orelse`` are each visited with the
                snapshot-restore protocol.
        """
        snapshot_vars = self.quantum_vars.copy()
        snapshot_violations = len(self.violations)
        for stmt in node.body:
            self.visit(stmt)
        self.quantum_vars = snapshot_vars.copy()
        del self.violations[snapshot_violations:]
        for stmt in node.orelse:
            self.visit(stmt)
        self.quantum_vars = snapshot_vars
        del self.violations[snapshot_violations:]
        # We intentionally do NOT call ``generic_visit`` here: the
        # children have already been visited in the per-branch loops
        # above, and falling through would re-visit them and
        # double-count branch state changes.

    def visit_If(self, node: ast.If) -> None:
        """Visit ``if``/``else`` body with branch-local scope.

        ``node.test`` is walked first (before the branch-scope
        snapshot) so that any consume effect inside the condition
        (e.g. ``if qm.measure(q):``) is reflected in the outer
        analyzer state, not silently rolled back when the branch
        scope restores. See :meth:`_visit_branch_scope` for the
        snapshot-restore protocol applied to ``body`` and ``orelse``.

        Args:
            node (ast.If): The ``if`` statement.
        """
        self.visit(node.test)
        self._visit_branch_scope(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit a ``for`` loop's body and ``else`` with branch-local scope.

        ``node.iter`` is walked first (before the branch-scope
        snapshot) so that any consume effect inside the iterable
        expression is reflected in the outer analyzer state. The
        loop target itself is not modeled — Qamomile's frontend
        rewrites ``qmc.range(...)`` loops via the control-flow
        transformer, so the iterator variable is a classical index
        and never quantum.

        Args:
            node (ast.For): The ``for`` statement.
        """
        self.visit(node.iter)
        self._visit_branch_scope(node)

    def visit_While(self, node: ast.While) -> None:
        """Visit a ``while`` loop's body and ``else`` with branch-local scope.

        Same protocol as :meth:`visit_If`. ``node.test`` is walked
        before the branch-scope snapshot.

        Args:
            node (ast.While): The ``while`` statement.
        """
        self.visit(node.test)
        self._visit_branch_scope(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Apply consume effects when a classical-returning call is seen.

        ``_check_single_assign`` / ``_check_tuple_assign`` already
        invoke :meth:`_consume_quantum_args` when a classical-returning
        call appears on the RHS of an assignment. This visitor handles
        the cases where the same call appears *outside* an assignment:
        as a bare expression statement (``qm.measure(q)``), inside an
        ``if`` / ``while`` condition, inside a ``for`` iterable, or
        nested inside any other expression visited via
        ``generic_visit``. Without this hook, those forms would leave
        ``q`` in ``quantum_vars`` and trip a false-positive
        ``FRESH_ALLOCATION`` violation on a later rebind of ``q``.

        Re-visiting from inside a covered assignment path is benign:
        ``_consume_quantum_args`` is idempotent — by the time
        ``visit_Call`` runs as part of ``generic_visit`` after the
        assignment dispatch, the relevant origins are already gone
        from ``quantum_vars`` and the second call is a no-op.

        Args:
            node (ast.Call): The call expression.
        """
        if self._is_classical_returning_call(node):
            self._consume_quantum_args(node)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # single assignment:  name = expr
    # ------------------------------------------------------------------

    def _check_single_assign(self, target: str, value: ast.expr, lineno: int) -> None:
        """Apply rebind rules to a single-name assignment.

        Args:
            target (str): Name being assigned to.
            value (ast.expr): RHS expression.
            lineno (int): Source line for diagnostic reporting.
        """
        # Python "throwaway" convention. ``_`` is the universally
        # recognized name for a binding the user intentionally does not
        # care about (``_ = qmc.qubit("ancilla")`` to allocate a qubit
        # without naming it). We do not register ``_`` into
        # ``quantum_vars`` and do not flag rebinds on it, so back-to-back
        # ``_ = ...`` assignments are not surfaced as fresh-allocation
        # violations. RHS consume effects are still propagated for
        # classical-returning calls so downstream analysis stays
        # consistent (e.g. ``_ = qm.measure(q)`` still consumes ``q``).
        if target == "_":
            if isinstance(value, ast.Call) and self._is_classical_returning_call(value):
                self._consume_quantum_args(value)
            return

        # Case 1: Name = Name  (direct alias / overwrite)
        if isinstance(value, ast.Name):
            source = value.id
            if source in self.quantum_vars:
                if target in self.quantum_vars:
                    if self.quantum_vars[target] != self.quantum_vars[source]:
                        self.violations.append(
                            RebindViolation(
                                target,
                                source,
                                RebindSourceKind.DIRECT_ALIAS,
                                None,
                                lineno,
                            )
                        )
                    self.quantum_vars[target] = self.quantum_vars[source]
                else:
                    # new alias binding – allowed, propagate origin
                    self.quantum_vars[target] = self.quantum_vars[source]
            return

        # Case 2: Name = Subscript(...)  (array element alias / overwrite)
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            source = value.value.id
            if source in self.quantum_vars:
                if target in self.quantum_vars:
                    if self.quantum_vars[target] != self.quantum_vars[source]:
                        self.violations.append(
                            RebindViolation(
                                target,
                                source,
                                RebindSourceKind.DIRECT_ALIAS,
                                None,
                                lineno,
                                source_expr=ast.unparse(value),
                            )
                        )
                    self.quantum_vars[target] = self.quantum_vars[source]
                else:
                    self.quantum_vars[target] = self.quantum_vars[source]
            return

        # Case 3: Name = Call(...)
        if isinstance(value, ast.Call):
            # Classical-returning frontend ops (measure / expval). The
            # call consumes its quantum args, so every alias of those
            # args' origins is removed from quantum_vars to model the
            # post-call state correctly. The target name is also dropped
            # if it was quantum, because the call's result is classical.
            if self._is_classical_returning_call(value):
                self.quantum_vars.pop(target, None)
                self._consume_quantum_args(value)
                return

            quantum_args = self._extract_quantum_args(value)
            is_constructor = self._is_quantum_constructor_call(value)

            if target in self.quantum_vars:
                target_origin = self.quantum_vars[target]
                has_self = any(
                    self.quantum_vars.get(a) == target_origin for a in quantum_args
                )
                if not has_self:
                    func_name = self._get_func_name(value)
                    if is_constructor:
                        self.violations.append(
                            RebindViolation(
                                target,
                                None,
                                RebindSourceKind.FRESH_ALLOCATION,
                                func_name,
                                lineno,
                            )
                        )
                    elif quantum_args:
                        self.violations.append(
                            RebindViolation(
                                target,
                                quantum_args[0],
                                RebindSourceKind.QUANTUM_ARG,
                                func_name,
                                lineno,
                            )
                        )
                    else:
                        self.violations.append(
                            RebindViolation(
                                target,
                                None,
                                RebindSourceKind.UNKNOWN_CALL,
                                func_name,
                                lineno,
                            )
                        )

            # Update tracking. A single-quantum-arg call produces a
            # value that physically corresponds to the consumed input,
            # so the target inherits its origin. A multi-quantum-arg
            # call (e.g. ``result = gate(q1, q2, q3)``) produces a
            # structured / tuple-like value where each component would
            # correspond to a different input — there is no single
            # origin to propagate, so we leave the target untracked
            # to avoid spurious aliasing claims on later
            # ``result[i]`` extractions. Fresh constructor allocations
            # seed a new origin so subsequent aliases of the freshly
            # allocated value can be detected.
            if len(quantum_args) == 1:
                first_q = quantum_args[0]
                self.quantum_vars[target] = self.quantum_vars.get(first_q, first_q)
            elif is_constructor:
                self.quantum_vars[target] = target
            return

    # ------------------------------------------------------------------
    # tuple assignment:  a, b = expr
    # ------------------------------------------------------------------

    def _check_tuple_assign(
        self, targets: ast.Tuple | ast.List, value: ast.expr, lineno: int
    ) -> None:
        """Apply rebind rules to an unpacking assignment.

        Args:
            targets (ast.Tuple | ast.List): The LHS tuple / list
                expression (Python unpacking semantics are identical).
            value (ast.expr): RHS expression. ``ast.Tuple`` /
                ``ast.List`` literals are paired element-wise —
                flat all-``Name`` shapes keep the pure-quantum
                permutation special case, and mixed shapes (nested
                tuples / lists, a starred target) go through the
                recursive target-tree walk. ``ast.Call`` keeps the
                existing permissive logic that preserves
                swap-via-call.
            lineno (int): Source line for diagnostic reporting.
        """
        target_names = [elt.id for elt in targets.elts if isinstance(elt, ast.Name)]
        flat_all_names = len(target_names) == len(targets.elts)
        if not target_names and flat_all_names:
            # Empty target list — nothing to check.
            return

        # Literal RHS: pair each target with the same-position RHS
        # element. Flat all-``Name`` shapes keep the dedicated
        # permutation-aware path; every other shape (nested tuple /
        # list targets, a starred target, non-``Name`` leaves) goes
        # through the recursive target-tree walk so a rebind cannot
        # dodge the rules by spelling — ``a, *rest = x(b), pad`` and
        # ``a, (m, n) = x(b), (c, d)`` are the same rebind of ``a``
        # as ``a = x(b)``.
        if isinstance(value, (ast.Tuple, ast.List)):
            if flat_all_names and len(targets.elts) == len(value.elts):
                self._check_tuple_literal_rhs(target_names, value.elts, lineno)
            else:
                self._check_target_tree_literal(
                    list(targets.elts), list(value.elts), lineno
                )
            return

        if not target_names:
            # Mixed non-literal shapes with no plain-``Name`` leaf at
            # this level (e.g. ``(m, n), obj.attr = f()``): nothing the
            # call path below can classify.
            return

        if not isinstance(value, ast.Call):
            return

        if self._is_classical_returning_call(value):
            for tgt in target_names:
                self.quantum_vars.pop(tgt, None)
            self._consume_quantum_args(value)
            return

        quantum_args = self._extract_quantum_args(value)
        is_constructor = self._is_quantum_constructor_call(value)

        if not quantum_args:
            # RHS call has no known quantum input. Every existing
            # quantum target is being silently discarded.
            func_name = self._get_func_name(value)
            kind = (
                RebindSourceKind.FRESH_ALLOCATION
                if is_constructor
                else RebindSourceKind.UNKNOWN_CALL
            )
            for tgt in target_names:
                if tgt in self.quantum_vars:
                    self.violations.append(
                        RebindViolation(tgt, None, kind, func_name, lineno)
                    )
            if is_constructor:
                for tgt in target_names:
                    if tgt == "_":
                        continue
                    self.quantum_vars[tgt] = tgt
            return

        arg_origins = [self.quantum_vars.get(a, a) for a in quantum_args]

        for i, tgt in enumerate(target_names):
            # Python "throwaway" convention. ``_`` positions are
            # ignored — we do not track them and do not flag rebinds on
            # them. See the matching skip at the head of
            # ``_check_single_assign`` for the rationale.
            if tgt == "_":
                continue

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
                    RebindViolation(
                        tgt,
                        mapped_source,
                        RebindSourceKind.QUANTUM_ARG,
                        func_name,
                        lineno,
                    )
                )

            # For new targets (or mismatched existing ones), track mapped origin.
            self.quantum_vars[tgt] = mapped_origin

    def _check_target_tree_literal(
        self,
        target_elts: list[ast.expr],
        value_elts: list[ast.expr],
        lineno: int,
    ) -> None:
        """Pair a mixed / nested target tree with a literal RHS, recursively.

        The target-tree walk for unpacking shapes the flat path cannot
        represent: starred targets (``a, *rest = x(b), pad``), nested
        tuple / list targets (``a, (m, n) = x(b), (c, d)``), and any
        combination thereof. Each ``Name`` leaf paired with its RHS
        element dispatches to :meth:`_check_single_assign`, so every
        spelling of ``a = x(b)`` meets the same rebind rules.

        Python grammar allows at most one starred target per level:
        prefix targets pair with prefix RHS elements, suffix targets
        with the RHS tail, and the starred name absorbs the middle as
        a plain Python list.

        Known non-flagged corners (conservative toward allowing):
        an *existing* quantum name as the starred target is untracked
        rather than flagged (it now holds a Python list — the heap
        boundary below); ``Subscript`` / ``Attribute`` leaves are heap
        or object stores the static analyzer cannot model (see the
        canonical in-place form ``qs[i] = qm.h(qs[i])``, which shares
        their syntax); a nested pattern unpacking a non-literal RHS
        element is left to runtime.

        Args:
            target_elts (list[ast.expr]): LHS elements at this nesting
                level.
            value_elts (list[ast.expr]): RHS literal elements at the
                same level.
            lineno (int): Source line for diagnostic reporting.
        """
        # Whole-subtree pure-quantum permutation escape FIRST: a 1-to-1
        # swap can span nesting levels and coexist with a star
        # (``a, (b,) = b, (a,)``; ``a, b, *rest = b, a, pad``), which a
        # per-level check cannot see. If every ``Name`` leaf of this
        # subtree maps to an existing quantum origin and the target and
        # value origin multisets match, it is a permutation — apply the
        # swap and stop, so a legal reordering is never flagged.
        if self._apply_permutation_escape(target_elts, value_elts):
            return

        star_idx = next(
            (i for i, t in enumerate(target_elts) if isinstance(t, ast.Starred)),
            None,
        )
        if star_idx is None:
            if len(target_elts) != len(value_elts):
                # Arity mismatch raises ValueError at runtime; there is
                # no sound positional pairing to classify.
                return
            pairs = list(zip(target_elts, value_elts))
        else:
            n_fixed = len(target_elts) - 1
            if len(value_elts) < n_fixed:
                # Runtime ValueError (not enough values to unpack).
                return
            suffix_len = n_fixed - star_idx
            pairs = list(zip(target_elts[:star_idx], value_elts[:star_idx]))
            if suffix_len:
                pairs += list(
                    zip(
                        target_elts[star_idx + 1 :],
                        value_elts[len(value_elts) - suffix_len :],
                    )
                )
            starred = target_elts[star_idx]
            if isinstance(starred, ast.Starred) and isinstance(starred.value, ast.Name):
                self.quantum_vars.pop(starred.value.id, None)

        for tgt, elt in pairs:
            if isinstance(tgt, ast.Name):
                self._check_single_assign(tgt.id, elt, lineno)
            elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(
                elt, (ast.Tuple, ast.List)
            ):
                # A nested level re-enters this method, so its own subtree
                # gets the permutation escape (a legal swap nested inside
                # a larger non-permutation statement stays legal) before
                # any element is flagged.
                self._check_target_tree_literal(list(tgt.elts), list(elt.elts), lineno)
            else:
                # Non-Name, non-recursable leaf (heap / object store, or
                # a nested pattern over a non-literal element) — out of
                # the static model, documented above.
                pass

    def _flat_leaf_pairs(
        self, target_elts: list[ast.expr], value_elts: list[ast.expr]
    ) -> list[tuple[ast.expr, ast.expr | None]] | None:
        """Structurally zip a target tree with a literal RHS into leaf pairs.

        Descends into matching nested ``Tuple`` / ``List`` levels so a
        cross-level permutation flattens to one leaf list. One starred
        target per level is allowed; it contributes a single
        ``(Starred, None)`` leaf (it binds a Python list, not a quantum
        value) and absorbs the middle RHS elements. Returns ``None`` when
        the shapes cannot be aligned (arity mismatch, or a ``Tuple`` /
        ``List`` target paired with a non-sequence RHS element) — the
        permutation escape then does not apply.

        Args:
            target_elts (list[ast.expr]): LHS elements at this level.
            value_elts (list[ast.expr]): RHS literal elements at this level.

        Returns:
            list[tuple[ast.expr, ast.expr | None]] | None: Flattened
                ``(target_leaf, value_leaf)`` pairs, or ``None`` if the
                structures do not align.
        """
        star_idx = next(
            (i for i, t in enumerate(target_elts) if isinstance(t, ast.Starred)),
            None,
        )
        result: list[tuple[ast.expr, ast.expr | None]] = []
        if star_idx is None:
            if len(target_elts) != len(value_elts):
                return None
            seq = list(zip(target_elts, value_elts))
        else:
            n_fixed = len(target_elts) - 1
            if len(value_elts) < n_fixed:
                return None
            suffix_len = n_fixed - star_idx
            seq = list(zip(target_elts[:star_idx], value_elts[:star_idx]))
            if suffix_len:
                seq += list(
                    zip(
                        target_elts[star_idx + 1 :],
                        value_elts[len(value_elts) - suffix_len :],
                    )
                )
            result.append((target_elts[star_idx], None))
        for tgt, elt in seq:
            if isinstance(tgt, (ast.Tuple, ast.List)):
                if not isinstance(elt, (ast.Tuple, ast.List)):
                    return None
                sub = self._flat_leaf_pairs(list(tgt.elts), list(elt.elts))
                if sub is None:
                    return None
                result.extend(sub)
            else:
                result.append((tgt, elt))
        return result

    def _apply_permutation_escape(
        self, target_elts: list[ast.expr], value_elts: list[ast.expr]
    ) -> bool:
        """Apply the swap and return ``True`` iff this subtree is a permutation.

        The subtree is a pure-quantum permutation when, across all
        flattened leaves, every non-starred target is an already-tracked
        quantum ``Name``, every paired RHS leaf contributes exactly one
        quantum origin, and the two origin multisets are equal. That is
        the generalisation of :meth:`_check_tuple_literal_rhs`'s flat
        escape to nested / starred shapes (``a, (b,) = b, (a,)``;
        ``a, b, *rest = b, a, pad``). On a match the swapped origins are
        assigned simultaneously and any starred quantum name is dropped
        (it now holds a Python list); nothing is flagged.

        Args:
            target_elts (list[ast.expr]): LHS elements at this level.
            value_elts (list[ast.expr]): RHS literal elements at this level.

        Returns:
            bool: ``True`` if a permutation was recognised and applied;
                ``False`` if the caller should fall back to element-wise
                classification.
        """
        pairs = self._flat_leaf_pairs(target_elts, value_elts)
        if pairs is None:
            return False
        name_pairs: list[tuple[str, str]] = []
        for tgt, elt in pairs:
            if isinstance(tgt, ast.Starred):
                continue
            if not isinstance(tgt, ast.Name) or elt is None:
                # A heap / object-store leaf (Subscript / Attribute) or a
                # missing RHS: not a clean 1-to-1 quantum permutation.
                return False
            target_origin = self.quantum_vars.get(tgt.id)
            value_origin = self._single_quantum_origin(elt)
            if target_origin is None or value_origin is None:
                return False
            name_pairs.append((tgt.id, value_origin))
        if not name_pairs:
            return False
        if sorted(o for _, o in name_pairs) != sorted(
            self.quantum_vars[n] for n, _ in name_pairs
        ):
            return False
        # Simultaneous swap: value_origins were computed before mutation.
        for name, value_origin in name_pairs:
            self.quantum_vars[name] = value_origin
        for tgt, _ in pairs:
            if isinstance(tgt, ast.Starred) and isinstance(tgt.value, ast.Name):
                self.quantum_vars.pop(tgt.value.id, None)
        return True

    def _check_tuple_literal_rhs(
        self, target_names: list[str], value_elts: list[ast.expr], lineno: int
    ) -> None:
        """Validate ``a, b = (e1, e2)`` element-wise with a permutation escape.

        The pure-quantum permutation special case allows ``q1, q2 =
        (q2, q1)`` and ``q1, q2 = (h(q2), h(q1))`` while still rejecting
        ``q1, q2 = (qm.qubit("fresh"), h(q1))`` (which a naive flatten
        approach would mis-accept by finding ``q1`` somewhere on the RHS).

        Args:
            target_names (list[str]): Flat list of target ``Name`` ids
                (non-``Name`` targets in the LHS tuple are skipped).
            value_elts (list[ast.expr]): RHS tuple elements; same length
                as ``targets.elts`` by precondition of the caller.
            lineno (int): Source line for diagnostic reporting.
        """
        elt_origins: list[str | None] = [
            self._single_quantum_origin(e) for e in value_elts
        ]
        target_origins: list[str | None] = [
            self.quantum_vars.get(t) for t in target_names
        ]

        # Permutation special case: every element contributes exactly
        # one quantum origin, every target is already quantum, and the
        # multisets match. Swap the origins.
        if (
            all(o is not None for o in elt_origins)
            and all(o is not None for o in target_origins)
            and sorted(t for t in elt_origins if t is not None)
            == sorted(t for t in target_origins if t is not None)
        ):
            for tgt, origin in zip(target_names, elt_origins):
                if origin is None:
                    # The all-not-None preconditions above guarantee
                    # this is unreachable; ``raise AssertionError``
                    # (not ``assert``) so the invariant holds under
                    # ``python -O``.
                    raise AssertionError(
                        f"permutation fast-path reached with None origin "
                        f"for target '{tgt}'"
                    )
                self.quantum_vars[tgt] = origin
            return

        # Element-wise fallback: dispatch each pair to the single-assign
        # path. This catches ``q1, q2 = (qm.qubit("a"), qm.qubit("b"))``
        # (fresh_allocation on existing quantum targets) and the mixed
        # case where one element is a swap-like self-update but another
        # is a fresh allocation.
        for tgt, elt in zip(target_names, value_elts):
            self._check_single_assign(tgt, elt, lineno)

    # ------------------------------------------------------------------
    # chained assignment:  a = b = expr
    # ------------------------------------------------------------------

    def _check_chained_assign(
        self, targets: list[ast.expr], value: ast.expr, lineno: int
    ) -> None:
        """Conservatively flag chained binding of an existing quantum name.

        Chained ``a = b = expr`` shares one RHS value across every
        target. Verifying self-update against a multi-target chain
        requires multi-name origin tracking that the single-pass
        analyzer does not currently model, so each existing quantum
        target is reported as a ``chained_assignment`` rebind. Targets
        that are not yet tracked as quantum are left alone — the
        chained form is rare enough that the resulting (mild) loss of
        alias tracking is acceptable.

        Args:
            targets (list[ast.expr]): The chain's target expressions.
            value (ast.expr): Shared RHS expression.
            lineno (int): Source line for diagnostic reporting.
        """
        func_name = self._get_func_name(value) if isinstance(value, ast.Call) else None
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id in self.quantum_vars:
                self.violations.append(
                    RebindViolation(
                        tgt.id,
                        None,
                        RebindSourceKind.CHAINED_ASSIGNMENT,
                        func_name,
                        lineno,
                    )
                )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _extract_quantum_args(self, call: ast.Call) -> list[str]:
        """Collect quantum variable names referenced in a call's arguments.

        Args:
            call (ast.Call): The call expression.

        Returns:
            list[str]: Quantum names appearing in positional or keyword
                arguments, in AST-traversal order and deduplicated.
                ``Tuple`` / ``List`` / ``Set`` / ``Starred`` arguments
                are walked recursively.
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

    def _consume_quantum_args(self, call: ast.Call) -> None:
        """Drop every name that shares an origin with the call's quantum args.

        Used after a classical-returning call (``measure`` / ``expval``)
        to model that the consumed quantum values are no longer
        available. Dropping by *origin* (not by name) covers the alias
        case: if ``alias = q`` and the call is ``bit = measure(alias)``,
        both ``alias`` and ``q`` are removed.

        Subscript-form arguments (``measure(qs[0])``) are intentionally
        NOT treated as a whole-array consume: a per-element measurement
        only consumes that element, and the array name ``qs`` must stay
        in ``quantum_vars`` so a subsequent rebind like
        ``qs = qm.qubit_array(...)`` is still flagged as silently
        discarding the unmeasured remaining elements. Only ``Name``-form
        arguments (and ``Name``s nested in ``Tuple`` / ``List`` / ``Set``
        / ``Starred``) drop the array origin.

        Args:
            call (ast.Call): The classical-returning call whose quantum
                arguments are being consumed.
        """
        whole_consumed: set[str] = set()

        def _collect_whole(expr: ast.expr) -> None:
            """Collect Name-form quantum args; deliberately skip Subscript."""
            if isinstance(expr, ast.Name) and expr.id in self.quantum_vars:
                whole_consumed.add(expr.id)
            elif isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
                for elt in expr.elts:
                    _collect_whole(elt)
            elif isinstance(expr, ast.Starred):
                _collect_whole(expr.value)
            # Intentionally skip ast.Subscript: per-element consumption
            # does not invalidate the parent array's binding.

        for arg in call.args:
            _collect_whole(arg)
        for kw in call.keywords:
            if kw.value is not None:
                _collect_whole(kw.value)

        consumed_origins = {self.quantum_vars[n] for n in whole_consumed}
        if not consumed_origins:
            return
        self.quantum_vars = {
            name: origin
            for name, origin in self.quantum_vars.items()
            if origin not in consumed_origins
        }

    def _single_quantum_origin(self, expr: ast.expr) -> str | None:
        """Return the single quantum origin contributed by ``expr``.

        Used by the tuple-literal RHS permutation special case to decide
        whether each RHS element passes through exactly one existing
        quantum value.

        Args:
            expr (ast.expr): RHS element expression.

        Returns:
            str | None: The single origin name when ``expr`` is a
                quantum-tracked ``Name``, ``Subscript`` of a quantum
                ``Name``, or ``Call`` with exactly one quantum argument.
                ``None`` for fresh allocations, classical-returning
                calls, multi-quantum-arg calls, or any other form.
        """
        if isinstance(expr, ast.Name) and expr.id in self.quantum_vars:
            return self.quantum_vars[expr.id]
        if (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.value, ast.Name)
            and expr.value.id in self.quantum_vars
        ):
            return self.quantum_vars[expr.value.id]
        if isinstance(expr, ast.Call):
            if self._is_quantum_constructor_call(expr):
                return None
            if self._is_classical_returning_call(expr):
                return None
            quantum_args = self._extract_quantum_args(expr)
            if len(quantum_args) == 1:
                return self.quantum_vars.get(quantum_args[0])
        return None

    @staticmethod
    def _get_func_name(call: ast.Call) -> str | None:
        """Return the syntactic function name from a call's ``func`` node.

        Args:
            call (ast.Call): The call expression.

        Returns:
            str | None: The bare name for ``Name`` callees, the
                attribute name for ``Attribute`` callees, or ``None``
                otherwise.
        """
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    @staticmethod
    def _is_classical_returning_call(call: ast.Call) -> bool:
        """Return True when the call is a known classical-returning frontend op.

        Recognizes ``measure(...)``, ``expval(...)`` and their attribute
        forms (e.g. ``qm.measure(q)``). Alias-imported forms are out of
        scope; matching is purely syntactic.

        Args:
            call (ast.Call): The call expression.

        Returns:
            bool: True for recognized classical-returning primitives.
        """
        if isinstance(call.func, ast.Name):
            return call.func.id in _CLASSICAL_RETURNING_CALL_NAMES
        if isinstance(call.func, ast.Attribute):
            return call.func.attr in _CLASSICAL_RETURNING_CALL_NAMES
        return False

    @staticmethod
    def _is_quantum_constructor_call(call: ast.Call) -> bool:
        """Return True when the call is a recognized fresh-qubit allocator.

        Matches the four syntactic forms ``qubit(...)``,
        ``qubit_array(...)``, ``<expr>.qubit(...)``, and
        ``<expr>.qubit_array(...)``. Alias imports (``from ... import
        qubit as factory``) are NOT recognized — that is tracked as a
        separate name-resolution improvement.

        Args:
            call (ast.Call): The call expression.

        Returns:
            bool: True for recognized fresh-quantum-handle constructors.
        """
        if isinstance(call.func, ast.Name):
            return call.func.id in _QUANTUM_CONSTRUCTOR_NAMES
        if isinstance(call.func, ast.Attribute):
            return call.func.attr in _QUANTUM_CONSTRUCTOR_NAMES
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
            # Normalize each violation's lineno so the FIRST body
            # statement becomes line 1, independent of any blank
            # lines / comments / multi-line signatures between the
            # ``def`` line and that first statement. Subtracting
            # ``node.body[0].lineno - 1`` achieves this; subtracting
            # ``node.lineno`` (the ``def`` line) would leave the
            # offset off by one or more whenever there are leading
            # blank lines in the body source.
            if node.body:
                body_offset = node.body[0].lineno - 1
                for v in analyzer.violations:
                    v.lineno -= body_offset
            return analyzer.violations

    return []
