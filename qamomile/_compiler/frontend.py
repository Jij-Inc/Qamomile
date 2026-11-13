"""Direct Python-AST frontend for the Qamomile source language.

The frontend never calls a decorated body and never imports a source module.
Python is parsed as concrete syntax, resolved into a closed typed HIR, and then
discarded before persistent QCore is produced.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypeAlias, cast

from ..effects import CallableKind
from ..errors import Diagnostic, FrontendError, TypeCheckError
from ..ir import PortableValue, SourceSpan, attributes, freeze_value
from ..operations import (
    GATES_BY_NAME,
    infer_classical_result_type,
    normalize_aggregate_slice,
)
from ..types import (
    ArrayType,
    Bit,
    Bits,
    Bool,
    Complex,
    Const,
    Dimension,
    FixedType,
    FunctionType,
    Int,
    Map,
    MapType,
    Matrix,
    Observable,
    Param,
    Probability,
    ProductType,
    Q,
    QuantumType,
    Real,
    RecordType,
    Shot,
    Tensor,
    TypeSpec,
    UFixed,
    UInt,
    Unit,
    Vector,
    WordType,
    aggregate_element_types,
    at_stage,
    bind_declared_function_arguments,
    coherent_leaf_widths,
    common_stage,
    infer_declared_function_application,
    infer_function_application,
    infer_literal_type,
    is_type_assignable,
    local_peer_types,
    logical_width,
    mapping_components,
    sequence_element_type,
    stage_of,
    strip_stage,
)
from .hir import (
    HIRBlock,
    HIRConstant,
    HIRExpression,
    HIRFunction,
    HIRModule,
    HIRParameter,
    HIRStatement,
)

MAX_INTENT_DOC_BYTES = 4000
"""UTF-8 budget for one captured docstring, under the native intent literal bound."""


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """One immutable source unit with a content-addressed URI.

    Attributes:
        text (str): Complete source text, never executed.
        display_name (str): Diagnostic-only basename.
        module_name (str): Stable semantic module name.
        uri (str): Content-addressed source identity used by spans.
        origins (tuple[tuple[str, str], ...]): Declared function name to the
            qualified symbol that originally declared it, for linked units.
        retain_docstrings (bool): Whether function docstrings become ``doc``
            intent rows on the lowered functions.
    """

    text: str
    display_name: str
    module_name: str
    uri: str
    origins: tuple[tuple[str, str], ...] = ()
    retain_docstrings: bool = True

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        module_name: str | None = None,
    ) -> SourceUnit:
        """Read a source file without importing or executing it."""

        resolved = Path(path)
        text = resolved.read_text(encoding="utf-8")
        return cls.from_text(
            text,
            display_name=resolved.name,
            module_name=module_name or resolved.stem,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        display_name: str = "module.py",
        module_name: str = "module",
        origins: Mapping[str, str] | None = None,
        retain_docstrings: bool = True,
    ) -> SourceUnit:
        """Create a content-addressed source unit from provided text.

        Args:
            text: Python source text containing Qamomile declarations.
            display_name: Diagnostic-only basename.
            module_name: Stable semantic module name.
            origins: Declared function name to the qualified symbol that
                originally declared it; functions without an entry carry no
                ``origin`` intent.
            retain_docstrings: Whether docstrings become ``doc`` intent rows.

        Returns:
            Immutable source unit whose URI is derived from ``text`` only.
        """

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        safe_name = Path(display_name).name.replace(" ", "_")
        return cls(
            text,
            safe_name,
            module_name,
            f"qmsrc://sha256/{digest}/{safe_name}",
            tuple(sorted((origins or {}).items())),
            bool(retain_docstrings),
        )


@dataclass(frozen=True, slots=True)
class _Signature:
    """A first-pass function declaration used for recursive name resolution."""

    symbol: str
    parameters: tuple[HIRParameter, ...]
    result_type: TypeSpec
    declared_kind: CallableKind | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    external: bool = False


@dataclass(frozen=True, slots=True)
class _BoundArguments:
    """Declaration-ordered arguments plus their source evaluation order."""

    arguments: tuple[HIRExpression, ...]
    evaluation_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RecordDeclaration:
    """One source-level immutable named product declaration."""

    source_name: str
    type: RecordType
    defaults: tuple[tuple[str, PortableValue], ...] = ()


_Environment: TypeAlias = dict[str, TypeSpec]


def parse_source(unit: SourceUnit) -> HIRModule:
    """Parse and type-check a source unit into ownership HIR.

    Args:
        unit: Content-addressed Python source unit.

    Returns:
        A closed typed HIR module.

    Raises:
        FrontendError: If syntax or constructs are outside the language.
        TypeCheckError: If values do not type-check.
    """

    try:
        tree = ast.parse(unit.text, filename=unit.display_name, type_comments=True)
    except SyntaxError as error:
        raise FrontendError(
            Diagnostic(
                "Q5001",
                error.msg,
                unit.uri,
                error.lineno,
                (error.offset or 1) - 1,
            )
        ) from error
    compiler = _Frontend(unit, tree)
    return compiler.compile()


class _Frontend:
    """Two-phase source declaration and body compiler."""

    def __init__(self, unit: SourceUnit, tree: ast.Module) -> None:
        """Initialize source-local symbol and constant tables.

        Args:
            unit (SourceUnit): Immutable source text and its diagnostic identity.
            tree (ast.Module): Parsed source syntax, never imported or executed.
        """

        self.unit = unit
        self.tree = tree
        self.qamomile_aliases = {"qamomile"}
        self.imported_qamomile_names: dict[str, str] = {}
        self.typing_aliases: set[str] = set()
        self.named_tuple_names: set[str] = set()
        self.constants: dict[str, tuple[TypeSpec, PortableValue]] = {}
        self.records: dict[str, _RecordDeclaration] = {}
        self.signatures: dict[str, _Signature] = {}
        self._loop_depth = 0
        self._slice_parameters: set[str] = set()
        self._dimension_parameters: set[str] = set()

    def compile(self) -> HIRModule:
        """Collect all signatures, then capture every declared body.

        Forward and recursive references use the same declared signatures.

        Returns:
            HIRModule: Typed source module for native semantic verification.

        Raises:
            FrontendError: A declaration or source construct is unsupported.
            TypeCheckError: An expression does not satisfy its declared type.
        """

        self._collect_imports_and_constants()
        self._declare_records()
        self._declare_functions()
        functions = tuple(
            self._compile_function(signature)
            for signature in sorted(
                self.signatures.values(),
                key=lambda value: value.symbol,
            )
        )
        return HIRModule(
            self.unit.module_name,
            functions,
            tuple(
                HIRConstant(name, value[0], value[1])
                for name, value in sorted(self.constants.items())
            ),
        )

    def _collect_imports_and_constants(self) -> None:
        """Collect declarative top-level names without evaluating Python."""

        for statement in self.tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname or alias.name
                    if alias.name == "qamomile":
                        self.qamomile_aliases.add(local)
                    elif alias.name == "typing":
                        self.typing_aliases.add(local)
                continue
            if isinstance(statement, ast.ImportFrom):
                module = statement.module or ""
                if module == "qamomile":
                    for alias in statement.names:
                        local = alias.asname or alias.name
                        self.imported_qamomile_names[local] = alias.name
                elif module == "typing":
                    for alias in statement.names:
                        if alias.name == "NamedTuple":
                            self.named_tuple_names.add(alias.asname or alias.name)
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                self._collect_constant(statement)

    def _collect_constant(self, statement: ast.Assign | ast.AnnAssign) -> None:
        """Collect a top-level literal constant when it is portable."""

        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                return
            name = statement.targets[0].id
            value_node = statement.value
            annotation = None
        else:
            if not isinstance(statement.target, ast.Name) or statement.value is None:
                return
            name = statement.target.id
            value_node = statement.value
            annotation = statement.annotation
        try:
            value = freeze_value(ast.literal_eval(value_node))
        except (ValueError, TypeError):
            return
        if annotation is not None:
            type_spec = self._parse_annotation(annotation)
            self._require_compatible(type_spec, infer_literal_type(value), value_node)
        else:
            try:
                type_spec = infer_literal_type(value)
            except TypeCheckError:
                return
        self.constants[name] = (at_stage(type_spec, "Const"), value)

    def _declare_records(self) -> None:
        """Declare standard ``typing.NamedTuple`` classes as immutable records.

        Quantum fields carry linear owners and require explicit construction.

        Raises:
            FrontendError: A declaration, field annotation, or default is invalid.
        """

        for statement in self.tree.body:
            if not isinstance(statement, ast.ClassDef) or not self._is_named_tuple(
                statement
            ):
                continue
            if statement.decorator_list or statement.keywords:
                self._error(
                    statement,
                    "Q5099",
                    "NamedTuple records cannot have decorators or class keywords",
                )
            if statement.name in self.records:
                self._error(statement, "Q5099", f"duplicate record {statement.name!r}")
            fields: list[tuple[str, TypeSpec]] = []
            defaults: list[tuple[str, PortableValue]] = []
            for item in statement.body:
                if (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ):
                    continue
                if not isinstance(item, ast.AnnAssign) or not isinstance(
                    item.target, ast.Name
                ):
                    self._error(
                        item,
                        "Q5099",
                        "NamedTuple records contain annotated fields only",
                    )
                if item.value is None and defaults:
                    self._error(
                        item,
                        "Q5099",
                        f"record field {item.target.id!r} without a default follows "
                        "a field with a default",
                    )
                if any(name == item.target.id for name, _ in fields):
                    self._error(
                        item,
                        "Q5099",
                        f"duplicate record field {item.target.id!r}",
                    )
                field_type = self._parse_annotation(item.annotation)
                if field_type.is_linear and item.value is not None:
                    self._error(
                        item,
                        "Q5099",
                        "quantum record fields cannot have defaults",
                    )
                if item.value is not None:
                    try:
                        default = freeze_value(ast.literal_eval(item.value))
                    except (ValueError, TypeError) as error:
                        self._error(
                            item.value,
                            "Q5099",
                            f"default for record field {item.target.id!r} is not "
                            "a portable literal",
                        )
                        raise AssertionError from error
                    self._require_compatible(
                        field_type,
                        infer_literal_type(default),
                        item.value,
                    )
                    defaults.append((item.target.id, default))
                fields.append((item.target.id, field_type))
            if not fields:
                self._error(statement, "Q5099", "NamedTuple records cannot be empty")
            record_type = RecordType(
                f"{self.unit.module_name}.{statement.name}",
                tuple(fields),
            )
            self.records[statement.name] = _RecordDeclaration(
                statement.name,
                record_type,
                tuple(defaults),
            )

    def _is_named_tuple(self, statement: ast.ClassDef) -> bool:
        """Return whether one class uses the supported NamedTuple base."""

        if len(statement.bases) != 1:
            return False
        base = statement.bases[0]
        if isinstance(base, ast.Name):
            return base.id in self.named_tuple_names
        return (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id in self.typing_aliases
            and base.attr == "NamedTuple"
        )

    def _declare_functions(self) -> None:
        """Declare all Qamomile symbols before compiling any body."""

        for statement in self.tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            declared_kind = self._declared_kind(statement)
            if declared_kind is _NOT_QAMOMILE:
                continue
            resolved_kind = cast(CallableKind | None, declared_kind)
            if isinstance(statement, ast.AsyncFunctionDef):
                self._error(
                    statement,
                    "Q5002",
                    "async functions are not part of the Qamomile source language",
                )
            parameters = self._parse_parameters(statement.args)
            if statement.returns is None:
                self._error(
                    statement,
                    "Q5003",
                    "Qamomile functions require a return annotation",
                )
            result_type = self._parse_annotation(statement.returns)
            if _contains_function_type(result_type):
                self._error(
                    statement.returns,
                    "Q5062",
                    "qkernels cannot return QKernel values",
                    "pass a QKernel as a direct static argument at its call site",
                )
            for parameter in parameters:
                parameter_type = strip_stage(parameter.type)
                if isinstance(parameter_type, FunctionType) and any(
                    _contains_function_type(item)
                    for item in (*parameter_type.inputs, parameter_type.output)
                ):
                    self._error(
                        statement,
                        "Q5062",
                        "nested QKernel types are unsupported",
                    )
                if _contains_function_type(parameter.type) and not isinstance(
                    parameter.type, FunctionType
                ):
                    self._error(
                        statement,
                        "Q5062",
                        "QKernel is allowed only as a direct static parameter type",
                    )
            symbol = f"{self.unit.module_name}.{statement.name}"
            if statement.name in self.records:
                self._error(
                    statement,
                    "Q5004",
                    f"function {statement.name!r} conflicts with a record",
                )
            if statement.name in self.signatures:
                self._error(
                    statement, "Q5004", f"duplicate function {statement.name!r}"
                )
            external = _is_external_declaration(statement)
            if external and resolved_kind is None:
                self._error(
                    statement,
                    "Q5079",
                    "an external declaration requires qm.qkernel(kind=...) with "
                    "an explicit denotation contract",
                )
            self.signatures[statement.name] = _Signature(
                symbol,
                parameters,
                result_type,
                resolved_kind,
                statement,
                external,
            )

    def _parse_parameters(self, arguments: ast.arguments) -> tuple[HIRParameter, ...]:
        """Parse the supported non-variadic parameter forms."""

        if arguments.vararg or arguments.kwarg:
            self._error(
                arguments,
                "Q5005",
                "variadic parameters are not supported",
            )
        positional = tuple(arguments.posonlyargs) + tuple(arguments.args)
        defaults: list[ast.expr | None] = [
            None for _ in range(len(positional) - len(arguments.defaults))
        ]
        defaults.extend(arguments.defaults)
        parameters: list[HIRParameter] = []
        for argument, default_node in zip(positional, defaults, strict=True):
            parameters.append(self._parse_parameter(argument, default_node))
        for argument, default_node in zip(
            arguments.kwonlyargs,
            arguments.kw_defaults,
            strict=True,
        ):
            parameters.append(self._parse_parameter(argument, default_node))
        return tuple(parameters)

    def _parse_parameter(
        self,
        argument: ast.arg,
        default_node: ast.expr | None,
    ) -> HIRParameter:
        """Parse one typed parameter and portable default."""

        if argument.annotation is None:
            self._error(
                argument,
                "Q5006",
                f"parameter {argument.arg!r} requires a logical type annotation",
            )
        type_spec = self._parse_annotation(argument.annotation)
        if mapping_components(type_spec) is not None and stage_of(type_spec) != "Const":
            self._error(
                argument,
                "Q5062",
                "Map parameters must use Const[Map[...]]",
                "mapping iteration and lookup are eliminated during specialization",
            )
        if default_node is None:
            return HIRParameter(argument.arg, type_spec)
        try:
            default = freeze_value(ast.literal_eval(default_node))
        except (ValueError, TypeError) as error:
            self._error(
                default_node,
                "Q5007",
                f"default for {argument.arg!r} is not a portable literal",
            )
            raise AssertionError from error
        self._require_compatible(
            type_spec,
            infer_literal_type(default),
            default_node,
        )
        return HIRParameter(argument.arg, type_spec, default, True)

    def _compile_function(self, signature: _Signature) -> HIRFunction:
        """Compile one declared function body against the complete symbol table.

        Args:
            signature (_Signature): Source declaration and its typed parameters.

        Returns:
            HIRFunction: Captured source body with explicit immutable bindings.

        Raises:
            FrontendError: If the body is unsupported or may omit a result.
            TypeCheckError: If a body expression violates its declared type.
        """

        intent = self._function_intent(signature.node)
        if signature.external:
            return HIRFunction(
                signature.symbol,
                signature.parameters,
                signature.result_type,
                HIRBlock(()),
                signature.declared_kind,
                self._span(signature.node),
                True,
                intent,
            )

        environment = {
            parameter.name: parameter.type for parameter in signature.parameters
        }
        self._slice_parameters = set(environment)
        rebound = _assigned_names(_function_body(signature.node))
        self._dimension_parameters = {
            parameter.name
            for parameter in signature.parameters
            if parameter.type == Const[Int] and parameter.name not in rebound
        }
        body, _, terminated = self._compile_block(
            _function_body(signature.node),
            environment,
            signature.result_type,
        )
        if not terminated:
            if signature.result_type != Unit:
                self._error(
                    signature.node,
                    "Q5008",
                    f"function {signature.node.name!r} may exit without returning "
                    f"{signature.result_type}",
                )
            body = HIRBlock(
                body.statements
                + (
                    HIRStatement(
                        "return",
                        HIRExpression(
                            "literal",
                            Unit,
                            attributes=attributes(value=None),
                            span=self._span(signature.node),
                        ),
                        span=self._span(signature.node),
                    ),
                )
            )
        return HIRFunction(
            signature.symbol,
            signature.parameters,
            signature.result_type,
            body,
            signature.declared_kind,
            self._span(signature.node),
            False,
            intent,
        )

    def _function_intent(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[tuple[str, PortableValue], ...]:
        """Collect the untrusted provenance rows one declaration carries.

        Args:
            node: Declared function syntax.

        Returns:
            ``origin`` when the unit records where the function was declared,
            then ``doc`` when docstrings are retained and one is present. The
            docstring is cut to ``MAX_INTENT_DOC_BYTES`` of UTF-8.
        """

        rows: list[tuple[str, PortableValue]] = []
        origin = dict(self.unit.origins).get(node.name)
        if origin is not None:
            rows.append(("origin", origin))
        if self.unit.retain_docstrings:
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                encoded = docstring.encode("utf-8")[:MAX_INTENT_DOC_BYTES]
                rows.append(("doc", encoded.decode("utf-8", errors="ignore")))
        return tuple(rows)

    def _compile_block(
        self,
        statements: list[ast.stmt],
        environment: _Environment,
        result_type: TypeSpec,
    ) -> tuple[HIRBlock, _Environment, bool]:
        """Compile a lexical block and return its post-environment."""

        compiled: list[HIRStatement] = []
        current = dict(environment)
        terminated = False
        for statement in statements:
            if terminated:
                self._error(
                    statement,
                    "Q5009",
                    "unreachable source statement",
                )
            item, current, terminated = self._compile_statement(
                statement,
                current,
                result_type,
            )
            if item is not None:
                compiled.append(item)
        return HIRBlock(tuple(compiled)), current, terminated

    def _compile_statement(
        self,
        statement: ast.stmt,
        environment: _Environment,
        result_type: TypeSpec,
    ) -> tuple[HIRStatement | None, _Environment, bool]:
        """Compile one source statement into immutable value bindings.

        Args:
            statement (ast.stmt): Source statement to capture.
            environment (_Environment): Available lexical names and types.
            result_type (TypeSpec): Declared function result type.

        Returns:
            tuple[HIRStatement | None, _Environment, bool]: Captured statement,
                updated lexical types, and whether control terminates.

        Raises:
            FrontendError: If the syntax or a target is unsupported.
            TypeCheckError: If an expression violates a native type equation.
        """

        if isinstance(statement, ast.Pass):
            return None, environment, False
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                self._error(statement, "Q5010", "chained assignment is not supported")
            expression = self._compile_expression(statement.value, environment)
            target = statement.targets[0]
            if isinstance(target, ast.Subscript):
                if not isinstance(target.value, ast.Name) or isinstance(
                    target.slice, (ast.Slice, ast.Tuple)
                ):
                    self._error(
                        target,
                        "Q5057",
                        "element assignment requires one local classical array and one index",
                    )
                collection = self._compile_expression(target.value, environment)
                if not isinstance(strip_stage(collection.type), (ArrayType, WordType)):
                    self._error(
                        target,
                        "Q5057",
                        "assignment mutates only local names or local classical array/word elements",
                    )
                index = self._compile_expression(target.slice, environment)
                operands = (collection, index, expression)
                expression = HIRExpression(
                    "classical.update",
                    self._classical_result_type("classical.update", operands, target),
                    operands,
                    span=self._span(statement),
                )
                targets = (target.value.id,)
                updated = dict(environment)
                updated[targets[0]] = expression.type
                return (
                    HIRStatement(
                        "assign", expression, targets, span=self._span(statement)
                    ),
                    updated,
                    False,
                )
            targets = self._assignment_targets(target, expression.type)
            destructure = isinstance(target, (ast.Tuple, ast.List))
            updated = self._bind_targets(
                self._after_linear_consumption(environment, expression),
                targets,
                expression.type,
                statement,
                destructure=destructure,
            )
            return (
                HIRStatement(
                    "assign",
                    expression,
                    targets,
                    span=self._span(statement),
                    result_types=self._assignment_result_types(
                        targets, expression.type, destructure=destructure
                    ),
                ),
                updated,
                False,
            )
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                self._error(
                    statement,
                    "Q5011",
                    "annotation-only local declarations are not supported",
                )
            expression = self._compile_expression(statement.value, environment)
            declared = self._parse_annotation(statement.annotation)
            self._require_compatible(declared, expression.type, statement.value)
            targets = self._assignment_targets(statement.target, expression.type)
            updated = self._bind_targets(
                self._after_linear_consumption(environment, expression),
                targets,
                expression.type,
                statement,
            )
            return (
                HIRStatement(
                    "assign",
                    expression,
                    targets,
                    span=self._span(statement),
                    result_types=self._assignment_result_types(
                        targets, expression.type
                    ),
                ),
                updated,
                False,
            )
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                self._error(
                    statement.target,
                    "Q5012",
                    "augmented assignment requires a local name",
                )
            left = self._compile_expression(statement.target, environment)
            right = self._compile_expression(statement.value, environment)
            expression = self._binary_expression(statement.op, left, right, statement)
            updated = self._bind_targets(
                self._after_linear_consumption(environment, expression),
                (statement.target.id,),
                expression.type,
                statement,
            )
            return (
                HIRStatement(
                    "assign",
                    expression,
                    (statement.target.id,),
                    span=self._span(statement),
                ),
                updated,
                False,
            )
        if isinstance(statement, ast.Expr):
            expression = self._compile_expression(statement.value, environment)
            if strip_stage(expression.type) != Unit:
                self._error(
                    statement,
                    "Q5066",
                    f"unused expression has non-Unit type {expression.type}",
                    "bind the returned value explicitly; quantum transforms are functional",
                )
            return (
                HIRStatement("expression", expression, span=self._span(statement)),
                self._after_linear_consumption(environment, expression),
                False,
            )
        if isinstance(statement, ast.Return):
            expression = (
                self._compile_expression(statement.value, environment)
                if statement.value is not None
                else HIRExpression(
                    "literal",
                    Unit,
                    attributes=attributes(value=None),
                    span=self._span(statement),
                )
            )
            self._require_compatible(result_type, expression.type, statement)
            return (
                HIRStatement("return", expression, span=self._span(statement)),
                environment,
                True,
            )
        if isinstance(statement, ast.If):
            return self._compile_if(statement, environment, result_type)
        if isinstance(statement, ast.For):
            return self._compile_for(statement, environment, result_type)
        if isinstance(statement, ast.While):
            return self._compile_while(statement, environment, result_type)
        if isinstance(statement, ast.Break):
            if self._loop_depth == 0:
                self._error(statement, "Q5013", "break appears outside a loop")
            return HIRStatement("break", span=self._span(statement)), environment, True
        if isinstance(statement, ast.Continue):
            if self._loop_depth == 0:
                self._error(statement, "Q5014", "continue appears outside a loop")
            return (
                HIRStatement("continue", span=self._span(statement)),
                environment,
                True,
            )
        self._error(
            statement,
            "Q5015",
            f"{type(statement).__name__} is not in the Qamomile statement subset",
        )
        raise AssertionError

    def _compile_if(
        self,
        statement: ast.If,
        environment: _Environment,
        result_type: TypeSpec,
    ) -> tuple[HIRStatement, _Environment, bool]:
        """Compile ordinary Python conditional syntax."""

        condition = self._compile_expression(statement.test, environment)
        self._require_condition(condition.type, statement.test, allow_quantum=False)
        true_block, true_env, true_terminated = self._compile_block(
            statement.body,
            dict(environment),
            result_type,
        )
        false_block, false_env, false_terminated = self._compile_block(
            statement.orelse,
            dict(environment),
            result_type,
        )
        if true_terminated and false_terminated:
            output_environment: _Environment = {}
        elif true_terminated:
            output_environment = false_env
        elif false_terminated:
            output_environment = true_env
        else:
            output_environment = self._merge_environments(
                true_env,
                false_env,
                statement,
            )
            changed = _assigned_names(statement.body) | _assigned_names(
                statement.orelse
            )
            output_environment = {
                name: _join_control_stage(type_spec, condition.type)
                if name in changed
                else type_spec
                for name, type_spec in output_environment.items()
            }
        if true_terminated != false_terminated:
            output_environment = {
                name: _join_control_stage(type_spec, condition.type)
                for name, type_spec in output_environment.items()
            }
        captures = tuple(sorted(environment))
        outputs = tuple(sorted(output_environment))
        return (
            HIRStatement(
                "if",
                condition,
                attributes=attributes(captures=captures, outputs=outputs),
                regions=(true_block, false_block),
                span=self._span(statement),
                result_types=tuple(output_environment[name] for name in outputs),
            ),
            output_environment,
            true_terminated and false_terminated,
        )

    def _compile_for(
        self,
        statement: ast.For,
        environment: _Environment,
        result_type: TypeSpec,
    ) -> tuple[HIRStatement, _Environment, bool]:
        """Compile a structured Python for loop."""

        iterable = self._compile_iterable(statement.iter, environment)
        target_types = aggregate_element_types(iterable.type)
        assert target_types is not None
        targets = self._for_targets(statement.target, iterable.opcode, target_types)
        if iterable.opcode == "range":
            target_types = (
                at_stage(
                    Int,
                    common_stage(*(operand.type for operand in iterable.operands)),
                ),
            )
            control_types = tuple(operand.type for operand in iterable.operands)
        else:
            control_types = ()
        changed = _assigned_names(statement.body) & environment.keys()
        post_loop = dict(environment)
        for _ in range(4):
            loop_environment = {**environment, **post_loop}
            loop_environment.update(zip(targets, target_types, strict=True))
            self._loop_depth += 1
            try:
                body, body_env, body_terminated = self._compile_block(
                    statement.body,
                    loop_environment,
                    result_type,
                )
            finally:
                self._loop_depth -= 1
            next_post_loop = {
                name: _join_loop_stage(
                    type_spec,
                    body_env.get(name, type_spec),
                    control_types if name in changed else (),
                )
                for name, type_spec in environment.items()
            }
            if next_post_loop == post_loop:
                break
            post_loop = next_post_loop
        else:
            self._error(statement, "Q5082", "for-loop stage inference did not converge")
        for name, type_spec in environment.items():
            if name in body_env:
                self._require_compatible(post_loop[name], body_env[name], statement)
        else_block, else_env, else_terminated = self._compile_block(
            statement.orelse,
            post_loop,
            result_type,
        )
        for name, type_spec in post_loop.items():
            if name in else_env:
                self._require_compatible(type_spec, else_env[name], statement)
        captures = tuple(sorted(environment))
        output_environment = self._merge_environments(post_loop, else_env, statement)
        all_paths_return = _for_guarantees_return(statement)
        return (
            HIRStatement(
                "for",
                iterable,
                targets=targets,
                attributes=attributes(
                    captures=captures,
                    outputs=captures,
                    body_terminated=body_terminated,
                    else_terminated=else_terminated,
                    all_paths_return=all_paths_return,
                ),
                regions=(body, else_block),
                span=self._span(statement),
                result_types=tuple(output_environment[name] for name in captures),
            ),
            output_environment,
            all_paths_return,
        )

    def _compile_while(
        self,
        statement: ast.While,
        environment: _Environment,
        result_type: TypeSpec,
    ) -> tuple[HIRStatement, _Environment, bool]:
        """Capture a while loop with a fixed carry topology and staged condition.

        Args:
            statement (ast.While): Source loop, including its optional else arm.
            environment (_Environment): Bindings entering the loop.
            result_type (TypeSpec): Enclosing callable's declared return type.

        Returns:
            tuple[HIRStatement, _Environment, bool]: Typed loop, outgoing bindings,
                and whether every terminating path returns.

        Raises:
            FrontendError: If the condition, carry types, or stages are invalid.
        """
        changed = _assigned_names(statement.body) & environment.keys()
        post_loop = dict(environment)
        for _ in range(4):
            condition = self._compile_expression(statement.test, post_loop)
            self._require_condition(condition.type, statement.test, allow_quantum=False)
            condition_env = self._after_linear_consumption(post_loop, condition)
            if condition_env.keys() != post_loop.keys():
                self._error(
                    statement.test, "Q5082", "while condition consumes a loop carry"
                )
            self._loop_depth += 1
            try:
                body, body_env, body_terminated = self._compile_block(
                    statement.body, dict(post_loop), result_type
                )
            finally:
                self._loop_depth -= 1
            next_post_loop = {
                name: _join_loop_stage(
                    type_spec,
                    body_env.get(name, type_spec),
                    (condition.type,) if name in changed else (),
                )
                for name, type_spec in environment.items()
            }
            if next_post_loop == post_loop:
                break
            post_loop = next_post_loop
        else:
            self._error(
                statement, "Q5082", "while-loop stage inference did not converge"
            )
        for name in environment:
            if name in body_env:
                self._require_compatible(post_loop[name], body_env[name], statement)
        else_block, else_env, _ = self._compile_block(
            statement.orelse, post_loop, result_type
        )
        output_environment = self._merge_environments(post_loop, else_env, statement)
        captures = tuple(sorted(environment))
        return (
            HIRStatement(
                "while",
                condition,
                attributes=attributes(
                    captures=captures,
                    outputs=captures,
                    all_paths_return=_while_guarantees_return(statement),
                ),
                regions=(body, else_block),
                span=self._span(statement),
                result_types=tuple(output_environment[name] for name in captures),
            ),
            output_environment,
            _while_guarantees_return(statement),
        )

    def _compile_iterable(
        self,
        expression: ast.expr,
        environment: _Environment,
    ) -> HIRExpression:
        """Compile a supported range or fixed aggregate iterable."""

        call_name = (
            self._call_name(expression.func)
            if isinstance(expression, ast.Call)
            else None
        )
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "items"
        ):
            if expression.args or expression.keywords:
                self._error(expression, "Q5086", "Map.items() accepts no arguments")
            collection = self._compile_expression(expression.func.value, environment)
            components = mapping_components(collection.type)
            if components is None:
                self._error(
                    expression,
                    "Q5086",
                    "items() is defined only for a typed Map",
                )
            key_type, value_type, _ = components
            stage = common_stage(collection.type)
            return HIRExpression(
                "map_items",
                RecordType(
                    "Iterable",
                    (
                        ("key", at_stage(key_type, stage)),
                        ("value", at_stage(value_type, stage)),
                    ),
                ),
                (collection,),
                span=self._span(expression),
            )
        if isinstance(expression, ast.Call) and call_name == "range":
            if expression.keywords or not 1 <= len(expression.args) <= 3:
                self._error(
                    expression,
                    "Q5017",
                    "range expects one to three positional arguments",
                )
            arguments = tuple(
                self._compile_expression(argument, environment)
                for argument in expression.args
            )
            for argument in arguments:
                self._require_integer(argument.type, expression)
            if len(arguments) == 1:
                zero = HIRExpression(
                    "literal",
                    Int,
                    attributes=attributes(value=0),
                    span=self._span(expression),
                )
                one = HIRExpression(
                    "literal",
                    Int,
                    attributes=attributes(value=1),
                    span=self._span(expression),
                )
                arguments = (zero, arguments[0], one)
            elif len(arguments) == 2:
                one = HIRExpression(
                    "literal",
                    Int,
                    attributes=attributes(value=1),
                    span=self._span(expression),
                )
                arguments = (*arguments, one)
            return HIRExpression(
                "range",
                RecordType("Range", (("index", Int),)),
                arguments,
                span=self._span(expression),
            )
        if isinstance(expression, ast.Call) and call_name == "enumerate":
            if len(expression.args) not in {1, 2}:
                self._error(
                    expression,
                    "Q5083",
                    "enumerate expects one iterable and an optional start",
                )
            if expression.keywords and (
                len(expression.keywords) != 1
                or expression.keywords[0].arg != "start"
                or len(expression.args) != 1
            ):
                self._error(
                    expression,
                    "Q5083",
                    "enumerate accepts only the start keyword",
                )
            collection = self._compile_fixed_iterable_value(
                expression.args[0], environment
            )
            if len(expression.args) == 2:
                start_node = expression.args[1]
            elif expression.keywords:
                start_node = expression.keywords[0].value
            else:
                start_node = ast.copy_location(ast.Constant(value=0), expression)
            start = self._compile_expression(start_node, environment)
            self._require_integer(start.type, start_node)
            induction_type = self._iterable_induction_type(collection.type, environment)
            index_type = self._classical_result_type(
                "classical.add",
                (
                    HIRExpression("name", induction_type, span=self._span(expression)),
                    start,
                ),
                expression,
            )
            return HIRExpression(
                "enumerate",
                RecordType(
                    "Iterable",
                    (
                        ("index", index_type),
                        ("value", self._iterable_element_type(collection, expression)),
                    ),
                ),
                (collection, start),
                span=self._span(expression),
            )
        if isinstance(expression, ast.Call) and call_name == "zip":
            if expression.keywords or not expression.args:
                self._error(
                    expression,
                    "Q5084",
                    "zip expects one or more positional fixed aggregates",
                )
            collections = tuple(
                self._compile_fixed_iterable_value(argument, environment)
                for argument in expression.args
            )
            extents = tuple(
                self._iterable_extent(collection.type) for collection in collections
            )
            if any(extent != extents[0] for extent in extents[1:]):
                self._error(
                    expression,
                    "Q5085",
                    "zip inputs must have identical leading extents",
                    "make the shared static shape explicit before iteration",
                )
            return HIRExpression(
                "zip",
                RecordType(
                    "Iterable",
                    tuple(
                        (
                            f"item_{index}",
                            self._iterable_element_type(collection, expression),
                        )
                        for index, collection in enumerate(collections)
                    ),
                ),
                collections,
                span=self._span(expression),
            )
        value = self._compile_fixed_iterable_value(expression, environment)
        return HIRExpression(
            "iterate",
            RecordType(
                "Iterable", (("value", self._iterable_element_type(value, expression)),)
            ),
            (value,),
            span=self._span(expression),
        )

    def _compile_fixed_iterable_value(
        self,
        expression: ast.expr,
        environment: _Environment,
    ) -> HIRExpression:
        """Compile and validate one statically shaped aggregate iterable."""

        value = self._compile_expression(expression, environment)
        base = strip_stage(value.type)
        if (
            not isinstance(base, (ArrayType, WordType))
            and aggregate_element_types(base) is None
        ):
            self._error(
                expression,
                "Q5018",
                "for expects range(...) or a statically shaped typed aggregate",
            )
        return value

    def _iterable_element_type(
        self,
        value: HIRExpression,
        node: ast.AST,
    ) -> TypeSpec:
        """Return the homogeneous element type of one fixed aggregate."""

        collection_type = strip_stage(value.type)
        if isinstance(collection_type, (ArrayType, WordType)):
            return sequence_element_type(value.type)
        element_types = aggregate_element_types(collection_type)
        assert element_types is not None
        if not element_types or any(
            item != element_types[0] for item in element_types[1:]
        ):
            self._error(
                node,
                "Q5080",
                "aggregate iteration requires one homogeneous element type",
            )
        return at_stage(element_types[0], common_stage(value.type))

    def _iterable_extent(self, type_spec: TypeSpec) -> Dimension:
        """Return the leading static extent of one iterable aggregate type."""

        collection_type = strip_stage(type_spec)
        if isinstance(collection_type, ArrayType):
            return collection_type.shape[0]
        if isinstance(collection_type, WordType):
            return collection_type.width
        element_types = aggregate_element_types(collection_type)
        assert element_types is not None
        return len(element_types)

    def _iterable_induction_type(
        self,
        type_spec: TypeSpec,
        environment: _Environment,
    ) -> TypeSpec:
        """Return the availability of an aggregate's zero-based index.

        Args:
            type_spec (TypeSpec): Aggregate whose extent determines iteration.
            environment (_Environment): Available direct dimension binders.

        Returns:
            TypeSpec: Direct binder stage, a Const Int for a derived shape, or a
                Value Int for a fixed extent resolved by native loop lowering.
        """

        extent = self._iterable_extent(type_spec)
        if isinstance(extent, str):
            return environment.get(extent, Const[Int])
        return Int

    def _for_targets(
        self,
        target: ast.expr,
        iterable_opcode: str,
        target_types: tuple[TypeSpec, ...],
    ) -> tuple[str, ...]:
        """Validate names bound by one structured source loop."""

        if iterable_opcode in {"enumerate", "zip", "map_items"}:
            if not isinstance(target, (ast.Tuple, ast.List)):
                self._error(
                    target,
                    "Q5016",
                    f"{iterable_opcode} iteration requires flat destructuring",
                )
            if len(target.elts) != len(target_types):
                self._error(target, "Q5055", "destructuring arity mismatch")
            if any(not isinstance(item, ast.Name) for item in target.elts):
                self._error(
                    target,
                    "Q5056",
                    "nested loop targets are unsupported",
                )
            return tuple(cast(ast.Name, item).id for item in target.elts)
        if not isinstance(target, ast.Name):
            self._error(
                target,
                "Q5016",
                "for-loop target must be one local name",
            )
        return (target.id,)

    def _compile_static_slice(
        self,
        value: HIRExpression,
        slice_node: ast.Slice,
        environment: _Environment,
    ) -> HIRExpression:
        """Capture a literal or Const-bounded array slice without executing source.

        Args:
            value (HIRExpression): Classical source array, evaluated once.
            slice_node (ast.Slice): Optional start, stop, and stride syntax.
            environment (_Environment): Available names and their stages.

        Returns:
            HIRExpression: Literal transient slice or a native dependent slice.

        Raises:
            FrontendError: If boundaries are not integer literals or direct Const
                Int parameters, or a literal slice is empty or excessive.
            TypeCheckError: If the native result-shape equation rejects the slice.
        """

        array_type = strip_stage(value.type)
        if not isinstance(array_type, ArrayType):
            self._error(
                slice_node,
                "Q5026",
                "slicing requires a classical array with a fixed leading extent",
            )

        bounds: list[PortableValue] = []
        operands: list[HIRExpression] = [value]
        for node, name in zip(
            (slice_node.lower, slice_node.upper, slice_node.step),
            ("start", "stop", "step"),
            strict=True,
        ):
            if node is None:
                bounds.append(None)
                continue
            bound = self._compile_expression(node, environment)
            raw = bound.attribute("value")
            if bound.opcode in {"literal", "module_constant"} and type(raw) is int:
                descriptor: PortableValue = raw
            elif isinstance(node, ast.UnaryOp):
                try:
                    descriptor = ast.literal_eval(node)
                except (ValueError, TypeError):
                    descriptor = None
                if type(descriptor) is not int:
                    self._error(
                        node,
                        "Q5026",
                        f"array slice {name} must be an exact integer literal or a Const Int parameter",
                    )
            elif (
                isinstance(node, ast.Name)
                and node.id in self._slice_parameters
                and bound.type == Const[Int]
            ):
                descriptor = node.id
            else:
                self._error(
                    node,
                    "Q5026",
                    f"array slice {name} must be an exact integer literal or a Const Int parameter",
                )
            bounds.append(descriptor)
            operands.append(bound)
        start, stop, step = bounds
        if any(isinstance(bound, str) for bound in bounds) or not isinstance(
            array_type.shape[0], int
        ):
            slice_attributes = attributes(start=start, stop=stop, step=step)
            return HIRExpression(
                "classical.slice",
                self._classical_result_type(
                    "classical.slice", tuple(operands), slice_node, slice_attributes
                ),
                tuple(operands),
                attributes=slice_attributes,
                span=self._span(slice_node),
            )
        assert start is None or type(start) is int
        assert stop is None or type(stop) is int
        assert step is None or type(step) is int
        try:
            positions = normalize_aggregate_slice(
                array_type.shape[0],
                start,
                stop,
                step,
            )
        except ValueError as error:
            self._error(slice_node, "Q5026", str(error))
        if not positions:
            self._error(
                slice_node,
                "Q5026",
                "an empty slice has no nonempty fixed-array type",
            )

        element_type = at_stage(
            strip_stage(sequence_element_type(value.type)),
            common_stage(value.type),
        )
        return HIRExpression(
            "static_slice",
            infer_classical_result_type(
                "array",
                (element_type,) * len(positions),
            ),
            (value,),
            attributes=attributes(
                start=start,
                step=step,
                stop=stop,
            ),
            span=self._span(slice_node),
        )

    def _compile_expression(
        self,
        expression: ast.expr,
        environment: _Environment,
    ) -> HIRExpression:
        """Compile and type one Python expression.

        Args:
            expression (ast.expr): Source expression to capture.
            environment (_Environment): Available lexical names and types.

        Returns:
            HIRExpression: Typed expression with source order and location.

        Raises:
            FrontendError: The syntax or operation is unsupported or ill-typed.
        """

        if isinstance(expression, ast.Constant):
            try:
                value = freeze_value(expression.value)
                type_spec = infer_literal_type(value)
            except (TypeError, TypeCheckError) as error:
                self._error(
                    expression,
                    "Q5019",
                    f"literal {expression.value!r} is not portable",
                )
                raise AssertionError from error
            return HIRExpression(
                "literal",
                type_spec,
                attributes=attributes(value=value),
                span=self._span(expression),
            )
        if isinstance(expression, ast.Name):
            if expression.id in environment:
                return HIRExpression(
                    "name",
                    environment[expression.id],
                    attributes=attributes(name=expression.id),
                    span=self._span(expression),
                )
            if expression.id in self.constants:
                type_spec, value = self.constants[expression.id]
                return HIRExpression(
                    "module_constant",
                    type_spec,
                    attributes=attributes(name=expression.id, value=value),
                    span=self._span(expression),
                )
            if expression.id in self.signatures:
                signature = self.signatures[expression.id]
                return HIRExpression(
                    "function_ref",
                    FunctionType(
                        tuple(parameter.type for parameter in signature.parameters),
                        signature.result_type,
                        tuple(parameter.name for parameter in signature.parameters),
                    ),
                    attributes=attributes(callee=signature.symbol),
                    span=self._span(expression),
                )
            self._error(
                expression,
                "Q5020",
                f"name {expression.id!r} is not a local, portable constant, or callable",
                "arbitrary Python globals and closure cells are not captured",
            )
        if isinstance(expression, ast.Attribute):
            record = self._compile_expression(expression.value, environment)
            record_type = strip_stage(record.type)
            if not isinstance(record_type, RecordType):
                self._error(
                    expression,
                    "Q5025",
                    f"field access is undefined for {record.type}",
                )
            if record_type.is_linear:
                self._error(
                    expression,
                    "Q5099",
                    "quantum records must be destructured as a whole before field use",
                )
            fields = tuple(name for name, _ in record_type.fields)
            try:
                position = fields.index(expression.attr)
            except ValueError as error:
                self._error(
                    expression,
                    "Q5025",
                    f"record {record_type.name!r} has no field {expression.attr!r}",
                )
                raise AssertionError from error
            projection_attributes = attributes(position=position)
            return HIRExpression(
                "classical.project",
                self._classical_result_type(
                    "classical.project",
                    (record,),
                    expression,
                    projection_attributes,
                ),
                (record,),
                attributes=projection_attributes,
                span=self._span(expression),
            )
        if isinstance(expression, ast.Tuple):
            operands = tuple(
                self._compile_expression(item, environment) for item in expression.elts
            )
            return HIRExpression(
                "tuple",
                ProductType(tuple(item.type for item in operands)),
                operands,
                span=self._span(expression),
            )
        if isinstance(expression, ast.List):
            operands = tuple(
                self._compile_expression(item, environment) for item in expression.elts
            )
            if not operands:
                self._error(expression, "Q5021", "empty list type cannot be inferred")
            first = operands[0].type
            for item in operands[1:]:
                self._require_compatible(first, item.type, expression)
            return HIRExpression(
                "array",
                self._classical_result_type("array", operands, expression),
                operands,
                span=self._span(expression),
            )
        if isinstance(expression, ast.BinOp):
            return self._binary_expression(
                expression.op,
                self._compile_expression(expression.left, environment),
                self._compile_expression(expression.right, environment),
                expression,
            )
        if isinstance(expression, ast.BoolOp):
            return self._compile_short_circuit_boolean(
                expression.values,
                environment,
                conjunction=isinstance(expression.op, ast.And),
                node=expression,
            )
        if isinstance(expression, ast.UnaryOp):
            operand = self._compile_expression(expression.operand, environment)
            if isinstance(expression.op, ast.Not):
                return HIRExpression(
                    "classical.not",
                    self._classical_result_type(
                        "classical.not", (operand,), expression
                    ),
                    (operand,),
                    span=self._span(expression),
                )
            if isinstance(expression.op, ast.Invert):
                return HIRExpression(
                    "classical.invert",
                    self._classical_result_type(
                        "classical.invert", (operand,), expression
                    ),
                    (operand,),
                    span=self._span(expression),
                )
            if isinstance(expression.op, (ast.USub, ast.UAdd)):
                opcode = "neg" if isinstance(expression.op, ast.USub) else "pos"
                return HIRExpression(
                    f"classical.{opcode}",
                    self._classical_result_type(
                        f"classical.{opcode}", (operand,), expression
                    ),
                    (operand,),
                    span=self._span(expression),
                )
        if isinstance(expression, ast.Compare):
            if len(expression.ops) > 1 and not all(
                _is_total_boolean_expression(item)
                for item in (expression.left, *expression.comparators)
            ):
                self._error(
                    expression,
                    "Q5069",
                    "comparison-chain operands must be total and side-effect-free",
                    "bind partial expressions before the comparison chain",
                )
            operands = (
                self._compile_expression(expression.left, environment),
                *(
                    self._compile_expression(item, environment)
                    for item in expression.comparators
                ),
            )
            if any(isinstance(value, (ast.Is, ast.IsNot)) for value in expression.ops):
                self._error(
                    expression,
                    "Q5059",
                    "object identity is not a portable Qamomile operation",
                    "use == or != for logical value comparison",
                )
            try:
                operators = tuple(
                    _COMPARE_NAMES[type(value)] for value in expression.ops
                )
            except KeyError as error:
                self._error(
                    expression,
                    "Q5060",
                    f"comparison {error.args[0].__name__} is unsupported",
                )
                raise AssertionError from error
            comparison_attributes = attributes(operators=operators)
            return HIRExpression(
                "classical.compare_chain",
                self._classical_result_type(
                    "classical.compare_chain",
                    operands,
                    expression,
                    comparison_attributes,
                ),
                operands,
                attributes=comparison_attributes,
                span=self._span(expression),
            )
        if isinstance(expression, ast.Subscript):
            current = self._compile_expression(expression.value, environment)
            mapping = mapping_components(current.type)
            if mapping is not None:
                key_type, value_type, _ = mapping
                key = self._compile_expression(expression.slice, environment)
                self._require_compatible(key_type, key.type, expression.slice)
                return HIRExpression(
                    "classical.map_get",
                    at_stage(value_type, common_stage(current.type, key.type)),
                    (current, key),
                    span=self._span(expression),
                )
            index_nodes = (
                tuple(expression.slice.elts)
                if isinstance(expression.slice, ast.Tuple)
                else (expression.slice,)
            )
            if not index_nodes:
                self._error(expression, "Q5025", "array access needs an index")
            for index_node in index_nodes:
                if isinstance(index_node, ast.Slice):
                    if len(index_nodes) != 1:
                        self._error(
                            index_node,
                            "Q5026",
                            "a fixed array slice must be the only subscript",
                        )
                    current = self._compile_static_slice(
                        current, index_node, environment
                    )
                    continue
                index = self._compile_expression(index_node, environment)
                self._require_integer(index.type, index_node)
                stripped = strip_stage(current.type)
                if isinstance(stripped, (ArrayType, WordType)):
                    raw_index = index.attribute("value")
                    extent = (
                        stripped.shape[0]
                        if isinstance(stripped, ArrayType)
                        else stripped.width
                    )
                    if (
                        index.opcode == "literal"
                        and isinstance(raw_index, int)
                        and not isinstance(raw_index, bool)
                        and isinstance(extent, int)
                        and not -extent <= raw_index < extent
                    ):
                        self._error(expression, "Q5024", "array index is out of bounds")
                    result = at_stage(
                        strip_stage(sequence_element_type(current.type)),
                        common_stage(current.type, index.type),
                    )
                elif isinstance(stripped, (ProductType, RecordType)):
                    if isinstance(stripped, RecordType) and stripped.is_linear:
                        self._error(
                            expression,
                            "Q5099",
                            "quantum records must be destructured as a whole before field use",
                        )
                    raw_index = index.attribute("value")
                    if index.opcode != "literal" or not isinstance(raw_index, int):
                        self._error(
                            expression,
                            "Q5023",
                            "product indexing requires a compile-time integer",
                        )
                    position = raw_index
                    projection_attributes = attributes(position=position)
                    try:
                        result = infer_classical_result_type(
                            "classical.project",
                            (current.type,),
                            attributes=projection_attributes,
                        )
                    except ValueError as error:
                        self._error(
                            expression, "Q5024", "product index is out of bounds"
                        )
                        raise AssertionError from error
                else:
                    self._error(expression, "Q5025", f"cannot index {current.type}")
                if isinstance(stripped, (ArrayType, WordType)):
                    current = HIRExpression(
                        "classical.index",
                        result,
                        (current, index),
                        span=self._span(expression),
                    )
                else:
                    current = HIRExpression(
                        "classical.project",
                        result,
                        (current,),
                        attributes=projection_attributes,
                        span=self._span(expression),
                    )
            return current
        if isinstance(expression, ast.IfExp):
            condition = self._compile_expression(expression.test, environment)
            self._require_condition(
                condition.type, expression.test, allow_quantum=False
            )
            when_true = self._compile_expression(expression.body, environment)
            when_false = self._compile_expression(expression.orelse, environment)
            return self._lazy_if_expression(
                condition,
                when_true,
                when_false,
                environment,
                expression,
            )
        if isinstance(expression, ast.Call):
            return self._compile_call(expression, environment)
        self._error(
            expression,
            "Q5026",
            f"{type(expression).__name__} is not in the Qamomile expression subset",
        )
        raise AssertionError

    def _compile_short_circuit_boolean(
        self,
        values: list[ast.expr],
        environment: _Environment,
        *,
        conjunction: bool,
        node: ast.BoolOp,
    ) -> HIRExpression:
        """Compile Boolean ``and`` or ``or`` as nested lazy regions."""

        first = self._compile_expression(values[0], environment)
        self._require_condition(first.type, values[0], allow_quantum=False)
        if len(values) == 1:
            return first
        remainder = self._compile_short_circuit_boolean(
            values[1:],
            environment,
            conjunction=conjunction,
            node=node,
        )
        constant = HIRExpression(
            "literal",
            Bool,
            attributes=attributes(value=not conjunction),
            span=self._span(node),
        )
        when_true, when_false = (
            (remainder, constant) if conjunction else (constant, remainder)
        )
        return self._lazy_if_expression(
            first,
            when_true,
            when_false,
            environment,
            node,
        )

    def _lazy_if_expression(
        self,
        condition: HIRExpression,
        when_true: HIRExpression,
        when_false: HIRExpression,
        environment: _Environment,
        node: ast.expr,
    ) -> HIRExpression:
        """Create one expression-valued branch over the ordinary region op."""

        function_peer, names_conflict = _common_function_peer(
            when_true.type,
            when_false.type,
        )
        if names_conflict:
            true_base = strip_stage(when_true.type)
            false_base = strip_stage(when_false.type)
            assert isinstance(true_base, FunctionType)
            assert isinstance(false_base, FunctionType)
            self._type_error(
                node,
                "conditional function branches expose incompatible parameter "
                f"names {true_base.parameter_names} and "
                f"{false_base.parameter_names}",
            )
        if (
            not local_peer_types(when_true.type, when_false.type)
            and function_peer is None
        ):
            self._type_error(
                node,
                f"conditional branches have incompatible types {when_true.type} and {when_false.type}",
            )
        captures = tuple(
            sorted(
                (_expression_names(when_true) | _expression_names(when_false))
                & environment.keys()
            )
        )
        result_base = function_peer or strip_stage(when_true.type)
        result_type = (
            result_base
            if result_base.is_quantum
            else at_stage(
                result_base,
                common_stage(condition.type, when_true.type, when_false.type),
            )
        )
        return HIRExpression(
            "lazy_if",
            result_type,
            (condition,),
            attributes=attributes(captures=captures),
            span=self._span(node),
            regions=(when_true, when_false),
        )

    def _compile_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Resolve a static function or versioned intrinsic call."""

        if any(keyword.arg is None for keyword in expression.keywords):
            self._error(
                expression, "Q5027", "dictionary argument expansion is unsupported"
            )
        name = self._call_name(expression.func)
        intrinsic = self._intrinsic_name(name)
        if intrinsic == "adjoint":
            return self._compile_adjoint_call(expression, environment)
        if intrinsic == "phase_by":
            return self._compile_phase_by_call(expression, environment)
        if intrinsic == "xor_by":
            return self._compile_xor_by_call(expression, environment)
        if intrinsic == "permute_by":
            return self._compile_permute_by_call(expression, environment)
        if intrinsic == "control":
            return self._compile_control_call(expression, environment)
        if intrinsic == "multiplex":
            return self._compile_multiplex_call(expression, environment)
        if name in environment and isinstance(
            strip_stage(environment[name]), FunctionType
        ):
            return self._compile_function_value_call(expression, environment)
        if not name:
            return self._compile_function_value_call(expression, environment)
        positional = tuple(
            self._compile_expression(item, environment) for item in expression.args
        )
        keywords = tuple(
            (keyword.arg or "", self._compile_expression(keyword.value, environment))
            for keyword in expression.keywords
        )
        if name in self.records:
            return self._compile_record_constructor(
                self.records[name],
                positional,
                keywords,
                expression,
            )
        if name in self.signatures:
            signature = self.signatures[name]
            bound = self._bind_call(signature, positional, keywords, expression)
            result_type = self._infer_declared_call_type(
                signature, bound.arguments, expression
            )
            return HIRExpression(
                "call",
                result_type,
                bound.arguments,
                attributes=self._evaluation_attributes(
                    bound.evaluation_order,
                    callee=signature.symbol,
                ),
                span=self._span(expression),
            )
        if intrinsic is not None:
            return self._compile_intrinsic(
                intrinsic,
                positional,
                keywords,
                expression,
            )
        if name in {"int", "float", "bool"}:
            if keywords or len(positional) != 1:
                self._error(expression, "Q5028", f"{name} expects one argument")
            target = {"int": Int, "float": Real, "bool": Bool}[name]
            source = strip_stage(positional[0].type)
            allowed = {Bool, Bit, Int, Real} if name == "bool" else {Int, Real}
            if source not in allowed:
                self._type_error(
                    expression,
                    f"{name} cast is undefined for {positional[0].type}",
                )
            return HIRExpression(
                "classical.cast",
                self._classical_result_type(
                    "classical.cast",
                    positional,
                    expression,
                    attributes(target=str(target)),
                ),
                positional,
                attributes=attributes(target=str(target)),
                span=self._span(expression),
            )
        if name == "len":
            if keywords or len(positional) != 1:
                self._error(expression, "Q5029", "len expects one argument")
            base = strip_stage(positional[0].type)
            if not isinstance(base, (ArrayType, WordType, ProductType, RecordType)):
                self._error(expression, "Q5030", f"len is undefined for {base}")
            return HIRExpression(
                "classical.len",
                self._classical_result_type("classical.len", positional, expression),
                positional,
                span=self._span(expression),
            )
        self._error(
            expression,
            "Q5031",
            f"call target {name!r} is not a Qamomile symbol or intrinsic",
            "arbitrary Python helpers must be declared as Qamomile functions",
        )
        raise AssertionError

    def _compile_function_value_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type one invocation of a closed first-class function value."""

        callee = self._compile_expression(expression.func, environment)
        function_type = strip_stage(callee.type)
        if not isinstance(function_type, FunctionType):
            self._type_error(
                expression.func, f"cannot call value of type {callee.type}"
            )
        positional = tuple(
            self._compile_expression(argument, environment)
            for argument in expression.args
        )
        keywords = tuple(
            (keyword.arg or "", self._compile_expression(keyword.value, environment))
            for keyword in expression.keywords
        )
        arguments = positional + tuple(value for _, value in keywords)
        keyword_names = tuple(name for name, _ in keywords)
        try:
            result_type = infer_function_application(
                callee.type,
                tuple(argument.type for argument in arguments),
                positional_count=len(positional),
                keyword_names=keyword_names,
            )
        except (TypeError, ValueError) as error:
            if str(error).startswith("Q4511:"):
                raise
            if not keywords and len(positional) != len(function_type.inputs):
                self._error(
                    expression,
                    "Q5099",
                    f"QKernel value expects {len(function_type.inputs)} positional arguments, "
                    f"got {len(positional)}",
                )
            if not keywords:
                for node, expected, argument in zip(
                    expression.args,
                    function_type.inputs,
                    positional,
                    strict=True,
                ):
                    if not is_type_assignable(expected, argument.type):
                        self._type_error(
                            node, f"expected {expected}, got {argument.type}"
                        )
                self._type_error(
                    expression,
                    "function argument dimensions do not match the selected signature",
                )
            reason = str(error).removeprefix("Q4510: invalid type-relation input: ")
            self._error(expression, "Q5099", reason)
        if not isinstance(result_type, TypeSpec):
            self._error(
                expression,
                "Q5099",
                "native function application inference returned an invalid type",
            )
        return HIRExpression(
            "invoke",
            result_type,
            (callee, *arguments),
            attributes=attributes(
                keyword_names=keyword_names,
                positional_count=len(positional),
            ),
            span=self._span(expression),
        )

    def _compile_record_constructor(
        self,
        declaration: _RecordDeclaration,
        positional: tuple[HIRExpression, ...],
        keywords: tuple[tuple[str, HIRExpression], ...],
        node: ast.Call,
    ) -> HIRExpression:
        """Bind one immutable NamedTuple constructor in declared field order."""

        fields = declaration.type.fields
        field_indices = {name: index for index, (name, _) in enumerate(fields)}
        if len(positional) > len(fields):
            self._error(node, "Q5031", f"too many values for {declaration.source_name}")
        bound = {
            name: value for (name, _), value in zip(fields, positional, strict=False)
        }
        for name, value in keywords:
            if name not in field_indices:
                self._error(
                    node,
                    "Q5031",
                    f"unknown {declaration.source_name} field {name!r}",
                )
            if name in bound:
                self._error(
                    node,
                    "Q5031",
                    f"{declaration.source_name} field {name!r} was supplied twice",
                )
            bound[name] = value
        default_values = dict(declaration.defaults)
        missing_required = tuple(
            name
            for name, _ in fields
            if name not in bound and name not in default_values
        )
        if missing_required:
            self._error(
                node,
                "Q5031",
                f"missing {declaration.source_name} field {missing_required[0]!r}",
            )
        default_indices: list[int] = []
        for index, (name, field_type) in enumerate(fields):
            if name in bound:
                continue
            bound[name] = HIRExpression(
                "literal",
                field_type,
                attributes=attributes(value=default_values[name]),
                span=self._span(node),
            )
            default_indices.append(index)
        ordered = tuple(bound[name] for name, _ in fields)
        for (_, field_type), value in zip(fields, ordered, strict=True):
            self._require_compatible(field_type, value.type, node)
        result_type = RecordType(
            declaration.type.name,
            tuple(
                (name, value.type)
                for (name, _), value in zip(fields, ordered, strict=True)
            ),
        )
        return HIRExpression(
            "record",
            result_type,
            ordered,
            attributes=self._evaluation_attributes(
                tuple(range(len(positional)))
                + tuple(field_indices[name] for name, _ in keywords)
                + tuple(default_indices)
            ),
            span=self._span(node),
        )

    def _compile_adjoint_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type an exact adjoint of a direct or later-specialized unitary.

        Args:
            expression (ast.Call): Source adjoint call.
            environment (_Environment): Types visible at this call site.

        Returns:
            HIRExpression: Direct adjoint call or transient callable invocation.

        Raises:
            FrontendError: If the callable or owned argument has invalid shape.
            TypeCheckError: If the argument types disagree with the callable.
        """

        if not expression.args:
            self._error(expression, "Q5080", "qm.adjoint expects a callable")
        callee_name = self._call_name(expression.args[0])
        signature = self.signatures.get(callee_name)
        if signature is None:
            candidate = environment.get(callee_name)
            candidate_type = strip_stage(candidate) if candidate is not None else None
            if not isinstance(candidate_type, FunctionType):
                self._error(
                    expression.args[0],
                    "Q5081",
                    "qm.adjoint requires a local Qamomile function reference",
                )
            if (
                len(candidate_type.inputs) != 1
                or not candidate_type.inputs[0].is_linear
                or not candidate_type.output.is_linear
                or not local_peer_types(candidate_type.inputs[0], candidate_type.output)
            ):
                self._error(
                    expression.args[0],
                    "Q5082",
                    "qm.adjoint requires one owned input and a shape-compatible owned result",
                )
            invocation = ast.Call(
                func=expression.args[0],
                args=expression.args[1:],
                keywords=expression.keywords,
            )
            ast.copy_location(invocation, expression)
            direct = self._compile_function_value_call(invocation, environment)
            return HIRExpression(
                "adjoint_invoke",
                direct.type,
                direct.operands,
                attributes=direct.attributes,
                span=self._span(expression),
            )
        self._require_unary_quantum_transform(
            signature,
            expression.args[0],
            intrinsic="adjoint",
        )
        positional = tuple(
            self._compile_expression(item, environment) for item in expression.args[1:]
        )
        keywords = tuple(
            (keyword.arg or "", self._compile_expression(keyword.value, environment))
            for keyword in expression.keywords
        )
        bound = self._bind_call(signature, positional, keywords, expression)
        result_type = self._infer_declared_call_type(
            signature, bound.arguments, expression
        )
        return HIRExpression(
            "adjoint_call",
            result_type,
            bound.arguments,
            attributes=self._evaluation_attributes(
                bound.evaluation_order,
                callee=signature.symbol,
                proof="explicit-adjoint",
            ),
            span=self._span(expression),
        )

    def _compile_phase_by_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type an exact phase lift of a local pure Boolean predicate."""

        if len(expression.args) != 2 or expression.keywords:
            self._error(
                expression,
                "Q5107",
                "qm.phase_by expects a predicate and one owned state",
            )
        predicate_name = self._call_name(expression.args[0])
        signature = self.signatures.get(predicate_name)
        if signature is None or signature.external:
            self._error(
                expression.args[0],
                "Q5108",
                "qm.phase_by requires a body-backed local Qamomile predicate",
            )
        if len(signature.parameters) != 1:
            self._error(
                expression.args[0],
                "Q5109",
                "qm.phase_by predicate must accept exactly one Bits[N] argument",
            )
        predicate_input = strip_stage(signature.parameters[0].type)
        if (
            not isinstance(predicate_input, WordType)
            or predicate_input.kind != "Bits"
            or not isinstance(predicate_input.width, int)
        ):
            self._error(
                expression.args[0],
                "Q5109",
                "qm.phase_by predicate input must be fixed-width Bits[N]",
            )
        if strip_stage(signature.result_type) != Bool:
            self._error(
                expression.args[0],
                "Q5110",
                "qm.phase_by predicate must return Bool",
            )
        state = self._compile_expression(expression.args[1], environment)
        state_type = strip_stage(state.type)
        if (
            not isinstance(state_type, QuantumType)
            or state_type.inner != predicate_input
        ):
            self._type_error(
                expression.args[1],
                f"qm.phase_by state must be {QuantumType(predicate_input)}",
            )
        return HIRExpression(
            "phase_by",
            state.type,
            (state,),
            attributes=attributes(predicate=signature.symbol),
            span=self._span(expression),
        )

    def _compile_xor_by_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type one exact Boolean XOR embedding over a packed owner product."""

        if len(expression.args) != 3 or expression.keywords:
            self._error(
                expression,
                "Q5111",
                "qm.xor_by expects a predicate, source state, and target state",
            )
        predicate_name = self._call_name(expression.args[0])
        signature = self.signatures.get(predicate_name)
        if signature is None or signature.external:
            self._error(
                expression.args[0],
                "Q5112",
                "qm.xor_by requires a body-backed local Qamomile predicate",
            )
        if len(signature.parameters) != 1:
            self._error(
                expression.args[0],
                "Q5113",
                "qm.xor_by predicate must accept exactly one Bits[N] argument",
            )
        predicate_input = strip_stage(signature.parameters[0].type)
        if (
            not isinstance(predicate_input, WordType)
            or predicate_input.kind != "Bits"
            or not isinstance(predicate_input.width, int)
        ):
            self._error(
                expression.args[0],
                "Q5113",
                "qm.xor_by predicate input must be fixed-width Bits[N]",
            )
        predicate_result = strip_stage(signature.result_type)
        if predicate_result == Bool:
            output_width = 1
        elif (
            isinstance(predicate_result, ProductType)
            and predicate_result.elements
            and all(element == Bool for element in predicate_result.elements)
        ):
            output_width = len(predicate_result.elements)
        else:
            self._error(
                expression.args[0],
                "Q5114",
                "qm.xor_by predicate must return Bool or a nonempty Product of Bool",
            )
        source = self._compile_expression(expression.args[1], environment)
        source_type = strip_stage(source.type)
        if (
            not isinstance(source_type, QuantumType)
            or source_type.inner != predicate_input
        ):
            self._type_error(
                expression.args[1],
                f"qm.xor_by source must be {QuantumType(predicate_input)}",
            )
        target = self._compile_expression(expression.args[2], environment)
        target_type = strip_stage(target.type)
        target_payload = WordType("Bits", output_width)
        if (
            not isinstance(target_type, QuantumType)
            or target_type.inner != target_payload
        ):
            self._type_error(
                expression.args[2],
                f"qm.xor_by target must be {QuantumType(target_payload)}",
            )
        product_type = ProductType((source.type, target.type))
        packed = HIRExpression(
            "tuple",
            product_type,
            (source, target),
            span=self._span(expression),
        )
        return HIRExpression(
            "xor_by",
            product_type,
            (packed,),
            attributes=attributes(predicate=signature.symbol),
            span=self._span(expression),
        )

    def _compile_permute_by_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type one proved in-place Boolean permutation."""

        if len(expression.args) != 2 or expression.keywords:
            self._error(
                expression,
                "Q5115",
                "qm.permute_by expects a function and one owned state",
            )
        function_name = self._call_name(expression.args[0])
        signature = self.signatures.get(function_name)
        if signature is None or signature.external:
            self._error(
                expression.args[0],
                "Q5116",
                "qm.permute_by requires a body-backed local Qamomile function",
            )
        if len(signature.parameters) != 1:
            self._error(
                expression.args[0],
                "Q5117",
                "qm.permute_by function must accept exactly one Bits[N] argument",
            )
        function_input = strip_stage(signature.parameters[0].type)
        if (
            not isinstance(function_input, WordType)
            or function_input.kind != "Bits"
            or not isinstance(function_input.width, int)
        ):
            self._error(
                expression.args[0],
                "Q5117",
                "qm.permute_by function input must be fixed-width Bits[N]",
            )
        function_result = strip_stage(signature.result_type)
        expected_width = function_input.width
        valid_result = (
            function_result == Bool
            if expected_width == 1
            else isinstance(function_result, ProductType)
            and len(function_result.elements) == expected_width
            and all(element == Bool for element in function_result.elements)
        )
        if not valid_result:
            self._error(
                expression.args[0],
                "Q5118",
                "qm.permute_by function must return one Bool per input bit",
            )
        state = self._compile_expression(expression.args[1], environment)
        state_type = strip_stage(state.type)
        if (
            not isinstance(state_type, QuantumType)
            or state_type.inner != function_input
        ):
            self._type_error(
                expression.args[1],
                f"qm.permute_by state must be {QuantumType(function_input)}",
            )
        return HIRExpression(
            "permute_by",
            state.type,
            (state,),
            attributes=attributes(predicate=signature.symbol),
            span=self._span(expression),
        )

    def _compile_control_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type an exact coherent application of one unitary qkernel.

        Args:
            expression (ast.Call): Source ``qm.control`` call to compile.
            environment (_Environment): Types of values visible at the call.

        Returns:
            HIRExpression: Typed static call or transient parameter invocation.

        Raises:
            TypeCheckError: If the callable, controls, arguments, or selected
                control state do not form a valid controlled unitary call.
        """

        if len(expression.args) < 3:
            self._error(
                expression,
                "Q5083",
                "qm.control expects a callable, controls, and target state",
            )
        callee_name = self._call_name(expression.args[0])
        signature = self.signatures.get(callee_name)
        dynamic_callee: HIRExpression | None = None
        if signature is None:
            function_type = strip_stage(environment.get(callee_name, Unit))
            if not isinstance(function_type, FunctionType):
                self._error(
                    expression.args[0],
                    "Q5084",
                    "qm.control requires a local qkernel or QKernel parameter",
                )
            parameter_names = function_type.parameter_names or tuple(
                f"argument_{index}" for index in range(len(function_type.inputs))
            )
            signature = _Signature(
                "",
                tuple(
                    HIRParameter(name, type_spec)
                    for name, type_spec in zip(
                        parameter_names,
                        function_type.inputs,
                        strict=True,
                    )
                ),
                function_type.output,
                None,
                cast(ast.FunctionDef, expression),
            )
            dynamic_callee = self._compile_expression(expression.args[0], environment)
        self._require_unary_quantum_transform(
            signature,
            expression.args[0],
            intrinsic="control",
        )
        controls = self._compile_expression(expression.args[1], environment)
        control_type = strip_stage(controls.type)
        control_width = logical_width(control_type)
        if not isinstance(control_type, QuantumType):
            self._type_error(
                expression.args[1],
                "qm.control controls must be Q[Bool], Q[Bit], or Q[Bits[N]] "
                "with a positive width",
            )
        control_state: int | None = None
        component: HIRExpression | None = None
        call_keywords: list[tuple[str, HIRExpression]] = []
        for keyword in expression.keywords:
            assert keyword.arg is not None
            if keyword.arg == "control_state":
                if (
                    not isinstance(keyword.value, ast.Constant)
                    or isinstance(keyword.value.value, bool)
                    or not isinstance(keyword.value.value, int)
                ):
                    self._error(
                        keyword.value,
                        "Q5085",
                        "control_state must be a compile-time integer literal",
                    )
                control_state = keyword.value.value
            elif keyword.arg == "at":
                if component is not None:
                    self._error(
                        keyword.value,
                        "Q5087",
                        "qm.control accepts at= only once",
                    )
                component = self._compile_expression(keyword.value, environment)
                self._require_integer(component.type, keyword.value)
                if component.opcode == "literal":
                    selected = component.attribute("value")
                    if isinstance(selected, bool) or not isinstance(selected, int):
                        self._error(
                            keyword.value,
                            "Q5087",
                            "qm.control at= must be an integer",
                        )
                    if selected < 0 or (
                        control_width is not None and selected >= control_width
                    ):
                        width = (
                            f"width {control_width}"
                            if control_width is not None
                            else "the control width"
                        )
                        self._error(
                            keyword.value,
                            "Q5087",
                            f"qm.control at={selected!r} is outside {width}",
                        )
            else:
                call_keywords.append(
                    (
                        keyword.arg,
                        self._compile_expression(keyword.value, environment),
                    )
                )
        effective_width = 1 if component is not None else control_width
        if control_state is not None and (
            control_state < 0
            or (effective_width is not None and control_state >= 1 << effective_width)
        ):
            domain = (
                f"{effective_width}-bit domain"
                if effective_width is not None
                else "the control domain"
            )
            self._error(
                expression,
                "Q5086",
                f"control_state {control_state} is outside {domain}",
            )
        positional = tuple(
            self._compile_expression(item, environment) for item in expression.args[2:]
        )
        bound = self._bind_call(
            signature,
            positional,
            tuple(call_keywords),
            expression,
        )
        result_type = self._infer_declared_call_type(
            signature, bound.arguments, expression
        )
        explicit_count = len(positional) + len(call_keywords)
        positional_order = bound.evaluation_order[: len(positional)]
        keyword_order = iter(bound.evaluation_order[len(positional) : explicit_count])
        default_order = bound.evaluation_order[explicit_count:]
        operand_prefix = 1 + int(component is not None)
        evaluation_order = [0]
        evaluation_order.extend(
            operand_prefix + parameter_index for parameter_index in positional_order
        )
        for keyword in expression.keywords:
            if keyword.arg == "at":
                evaluation_order.append(1)
            elif keyword.arg != "control_state":
                evaluation_order.append(operand_prefix + next(keyword_order))
        evaluation_order.extend(
            operand_prefix + parameter_index for parameter_index in default_order
        )
        call_attributes: dict[str, PortableValue] = {"control_state": control_state}
        if dynamic_callee is None:
            call_attributes["callee"] = signature.symbol
        if dynamic_callee is not None:
            evaluation_order = [0, *(index + 1 for index in evaluation_order)]
        return HIRExpression(
            "control_invoke" if dynamic_callee is not None else "control_call",
            ProductType((controls.type, result_type)),
            (
                *((dynamic_callee,) if dynamic_callee is not None else ()),
                controls,
                *((component,) if component is not None else ()),
                *bound.arguments,
            ),
            attributes=self._evaluation_attributes(
                tuple(evaluation_order),
                **call_attributes,
            ),
            span=self._span(expression),
        )

    def _compile_multiplex_call(
        self,
        expression: ast.Call,
        environment: _Environment,
    ) -> HIRExpression:
        """Type a total coherent selection over static unitary qkernels.

        Args:
            expression (ast.Call): Source ``qm.multiplex`` call to compile.
            environment (_Environment): Types of values visible at the call.

        Returns:
            HIRExpression: Typed selection that returns the index and successor
                target value.

        Raises:
            TypeCheckError: If the cases are not a complete, ABI-compatible
                static row of unitary qkernels.
        """

        if len(expression.args) < 3:
            self._error(
                expression,
                "Q5088",
                "qm.multiplex expects an index, cases, and a target state",
            )
        index = self._compile_expression(expression.args[0], environment)
        callable_node = expression.args[1]
        if not isinstance(callable_node, (ast.Tuple, ast.List)):
            self._error(
                callable_node,
                "Q5089",
                "qm.multiplex cases must be a static tuple or list",
            )
        signatures: list[_Signature] = []
        for item in callable_node.elts:
            name = self._call_name(item)
            signature = self.signatures.get(name)
            if signature is None:
                self._error(
                    item,
                    "Q5090",
                    "qm.multiplex requires local Qamomile function references",
                )
            self._require_unary_quantum_transform(
                signature,
                item,
                intrinsic="multiplex",
            )
            signatures.append(signature)
        if len(signatures) < 2:
            self._error(
                callable_node,
                "Q5091",
                "qm.multiplex requires at least two unitary cases",
            )
        prototype = signatures[0]
        index_type = strip_stage(index.type)
        index_width = logical_width(index_type)
        if not isinstance(index_type, QuantumType) or index_width is None:
            self._type_error(
                expression.args[0],
                "qm.multiplex index must be Q[Bool], Q[Bit], or fixed-width Q[Bits[N]]",
            )
        if len(signatures) != 1 << index_width:
            self._error(
                callable_node,
                "Q5093",
                f"qm.multiplex over a {index_width}-bit index requires exactly "
                f"{1 << index_width} cases",
            )
        positional = tuple(
            self._compile_expression(item, environment) for item in expression.args[2:]
        )
        keywords = tuple(
            (keyword.arg or "", self._compile_expression(keyword.value, environment))
            for keyword in expression.keywords
        )
        bound = self._bind_call(prototype, positional, keywords, expression)
        result_type = self._infer_declared_call_type(
            prototype, bound.arguments, expression
        )
        for signature in signatures[1:]:
            case_bound = self._bind_call(signature, positional, keywords, expression)
            case_result = self._infer_declared_call_type(
                signature,
                case_bound.arguments,
                expression,
            )
            if (
                case_bound.evaluation_order != bound.evaluation_order
                or not local_peer_types(result_type, case_result)
            ):
                self._error(
                    callable_node,
                    "Q5092",
                    "qm.multiplex cases must expose one compatible call ABI",
                )
        evaluation_order = (0,) + tuple(index + 1 for index in bound.evaluation_order)
        return HIRExpression(
            "select_call",
            ProductType((index.type, result_type)),
            (index, *bound.arguments),
            attributes=self._evaluation_attributes(
                evaluation_order,
                callees=tuple(signature.symbol for signature in signatures),
            ),
            span=self._span(expression),
        )

    def _require_unary_quantum_transform(
        self,
        signature: _Signature,
        node: ast.AST,
        *,
        intrinsic: str,
    ) -> HIRParameter:
        """Return the sole leading owner accepted by callable transforms."""

        linear = [
            parameter for parameter in signature.parameters if parameter.type.is_linear
        ]
        if (
            len(linear) != 1
            or not signature.parameters
            or signature.parameters[0] is not linear[0]
            or not signature.result_type.is_linear
            or not local_peer_types(linear[0].type, signature.result_type)
        ):
            self._error(
                node,
                "Q5082",
                f"qm.{intrinsic} requires one leading owned input and a "
                "shape-compatible owned result",
            )
        return linear[0]

    def _compile_intrinsic(
        self,
        name: str,
        positional: tuple[HIRExpression, ...],
        keywords: tuple[tuple[str, HIRExpression], ...],
        node: ast.Call,
    ) -> HIRExpression:
        """Type a versioned core intrinsic without invoking Python.

        Args:
            name (str): Public intrinsic name resolved from the ``qm``
                namespace.
            positional (tuple[HIRExpression, ...]): Positional arguments in
                source evaluation order.
            keywords (tuple[tuple[str, HIRExpression], ...]): Named arguments
                in source evaluation order.
            node (ast.Call): Source call used for precise diagnostics.

        Returns:
            HIRExpression: Typed intrinsic expression accepted by Rust HIR
                lowering.

        Raises:
            FrontendError: If the intrinsic name, arity, or keyword shape is
                unsupported.
            TypeCheckError: If Rust rejects an operand or result type.
        """

        operands = positional + tuple(value for _, value in keywords)
        if name in {"ceil", "log2", "sqrt"}:
            if len(positional) != 1 or keywords:
                self._error(node, "Q5105", f"qm.{name} expects one argument")
            if strip_stage(positional[0].type) not in {Int, Real}:
                self._type_error(node, f"qm.{name} expects Int or Real")
            opcode = f"classical.{name}"
            return HIRExpression(
                opcode,
                self._classical_result_type(opcode, positional, node),
                positional,
                span=self._span(node),
            )
        if name == "zero_observable":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5119", "qm.zero_observable expects one width")
            opcode = "classical.observable_zero"
            return HIRExpression(
                opcode,
                self._classical_result_type(opcode, positional, node),
                positional,
                span=self._span(node),
            )
        if name in {"X", "Y", "Z"}:
            if len(positional) != 1 or keywords:
                self._error(node, "Q5120", f"qm.{name} expects one logical index")
            opcode = "classical.observable_pauli"
            observable_attributes = attributes(axis=name)
            return HIRExpression(
                opcode,
                self._classical_result_type(
                    opcode,
                    positional,
                    node,
                    observable_attributes,
                ),
                positional,
                attributes=observable_attributes,
                span=self._span(node),
            )
        if name == "global_phase":
            if len(positional) != 2 or keywords:
                self._error(
                    node,
                    "Q5087",
                    "qm.global_phase expects an owned state and one angle",
                )
            if not isinstance(strip_stage(positional[0].type), QuantumType):
                self._type_error(node, "qm.global_phase expects an owned Q[T] value")
            if strip_stage(positional[1].type) not in {Int, Real}:
                self._type_error(node, "qm.global_phase angle must be Int or Real")
            return HIRExpression(
                "quantum.global_phase",
                positional[0].type,
                positional,
                span=self._span(node),
            )
        if name == "add_into":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5094", "qm.add_into expects one owned state")
            if not isinstance(strip_stage(positional[0].type), QuantumType):
                self._type_error(node, "qm.add_into expects an owned Q[T] value")
            return HIRExpression(
                "quantum.add_into",
                positional[0].type,
                positional,
                span=self._span(node),
            )
        if name == "mod_add":
            if len(positional) != 3 or keywords:
                self._error(node, "Q5094", "qm.mod_add expects target, addend, modulus")
            target, addend, modulus = positional
            width = _quantum_uint_width(target.type)
            if width is None:
                self._type_error(node, "qm.mod_add target must be Q[UInt[N]]")
            if all(
                _quantum_uint_width(value.type) == width for value in (addend, modulus)
            ):
                return HIRExpression(
                    "quantum.mod_add_values",
                    ProductType(tuple(value.type for value in positional)),
                    positional,
                    span=self._span(node),
                )
            if all(value.type in {Int, Const[Int]} for value in (addend, modulus)):
                return HIRExpression(
                    "quantum.mod_add_constant",
                    target.type,
                    positional,
                    span=self._span(node),
                )
            self._type_error(
                node,
                "qm.mod_add requires equally wide quantum addend and modulus, "
                "or two Const[Int] operands",
            )
        if name in {"mul_add", "less_than_xor"}:
            if len(positional) != 3 or keywords:
                self._error(node, "Q5094", f"qm.{name} expects three owned registers")
            first, left, right = positional
            width = _quantum_uint_width(left.type)
            if width is None or _quantum_uint_width(right.type) != width:
                self._type_error(node, f"qm.{name} requires equal Q[UInt[N]] operands")
            first_width = _quantum_uint_width(first.type)
            if name == "mul_add":
                if first_width is None or (
                    isinstance(width, int)
                    and isinstance(first_width, int)
                    and first_width != 2 * width
                ):
                    self._type_error(
                        node, "qm.mul_add accumulator must be Q[UInt[2*N]]"
                    )
            elif logical_width(first.type) != 1 or not isinstance(
                strip_stage(first.type), QuantumType
            ):
                self._type_error(node, "qm.less_than_xor flag must be one coherent bit")
            return HIRExpression(
                f"quantum.{name}_values",
                ProductType(tuple(value.type for value in positional)),
                positional,
                span=self._span(node),
            )
        if name == "mod_mul_add":
            if (
                len(positional) != 3
                or len(keywords) != 1
                or keywords[0][0] != "modulus"
            ):
                self._error(
                    node,
                    "Q5094",
                    "qm.mod_mul_add expects accumulator, left, right, modulus=...",
                )
            accumulator, left, right = positional
            width = _quantum_uint_width(accumulator.type)
            if width is None or any(
                _quantum_uint_width(value.type) != width for value in (left, right)
            ):
                self._type_error(
                    node,
                    "qm.mod_mul_add requires three equal Q[UInt[N]] registers",
                )
            modulus = keywords[0][1]
            if modulus.type not in {Int, Const[Int]}:
                self._type_error(node, "qm.mod_mul_add modulus must be Const[Int]")
            return HIRExpression(
                "quantum.mod_mul_add_values",
                ProductType(tuple(value.type for value in positional)),
                (*positional, modulus),
                span=self._span(node),
            )
        if name == "mod_mul":
            if (
                len(positional) != 1
                or len(keywords) != 2
                or {key for key, _ in keywords} != {"factor", "modulus"}
            ):
                self._error(
                    node,
                    "Q5094",
                    "qm.mod_mul expects target, factor=..., modulus=...",
                )
            if _quantum_uint_width(positional[0].type) is None:
                self._type_error(node, "qm.mod_mul target must be Q[UInt[N]]")
            named = dict(keywords)
            if any(
                named[key].type not in {Int, Const[Int]}
                for key in ("factor", "modulus")
            ):
                self._type_error(
                    node, "qm.mod_mul factor and modulus must be Const[Int]"
                )
            return HIRExpression(
                "quantum.mod_mul",
                positional[0].type,
                (positional[0], named["factor"], named["modulus"]),
                span=self._span(node),
            )
        if name == "mod_multi_pow":
            if (
                len(positional) != 2
                or len(keywords) != 4
                or {key for key, _ in keywords}
                != {"bases", "base_width", "modulus", "coordinate_width"}
            ):
                self._error(
                    node,
                    "Q5094",
                    "qm.mod_multi_pow expects coordinates, target, bases=..., base_width=..., "
                    "modulus=..., coordinate_width=...",
                )
            coordinates, target = positional
            coordinate_type = strip_stage(coordinates.type)
            if (
                not isinstance(coordinate_type, QuantumType)
                or not isinstance(coordinate_type.inner, WordType)
                or coordinate_type.inner.kind != "Bits"
            ):
                self._type_error(
                    node, "qm.mod_multi_pow coordinates must be Q[Bits[D*M]]"
                )
            if _quantum_uint_width(target.type) is None:
                self._type_error(node, "qm.mod_multi_pow target must be Q[UInt[N]]")
            named = dict(keywords)
            bases_type = strip_stage(named["bases"].type)
            if (
                stage_of(named["bases"].type) != "Const"
                or not isinstance(bases_type, ArrayType)
                or bases_type.element != Int
                or len(bases_type.shape) != 1
            ):
                self._type_error(
                    node, "qm.mod_multi_pow bases must be Const[Vector[Int, D]]"
                )
            if named["modulus"].type not in {Int, Const[Int]}:
                self._type_error(node, "qm.mod_multi_pow modulus must be Const[Int]")
            if any(
                named[key].type not in {Int, Const[Int]}
                for key in ("base_width", "coordinate_width")
            ):
                self._type_error(
                    node,
                    "qm.mod_multi_pow base_width and coordinate_width must be Const[Int]",
                )
            return HIRExpression(
                "quantum.mod_multi_pow_values",
                ProductType((coordinates.type, target.type)),
                (
                    coordinates,
                    target,
                    named["bases"],
                    named["base_width"],
                    named["modulus"],
                    named["coordinate_width"],
                ),
                span=self._span(node),
            )
        if name == "mod_pow":
            if (
                len(positional) != 2
                or len(keywords) != 2
                or {key for key, _ in keywords} != {"base", "modulus"}
            ):
                self._error(
                    node,
                    "Q5094",
                    "qm.mod_pow expects target, exponent, base=..., modulus=...",
                )
            target, exponent = positional
            if _quantum_uint_width(target.type) is None:
                self._type_error(node, "qm.mod_pow target must be Q[UInt[N]]")
            if _quantum_uint_width(exponent.type) is None:
                self._type_error(node, "qm.mod_pow exponent must be Q[UInt[P]]")
            named = dict(keywords)
            if any(
                named[key].type not in {Int, Const[Int]} for key in ("base", "modulus")
            ):
                self._type_error(node, "qm.mod_pow base and modulus must be Const[Int]")
            return HIRExpression(
                "quantum.mod_pow_values",
                ProductType((target.type, exponent.type)),
                (target, exponent, named["base"], named["modulus"]),
                span=self._span(node),
            )
        if name == "pauli_evolve":
            if len(positional) != 3 or keywords:
                self._error(
                    node,
                    "Q5094",
                    "qm.pauli_evolve expects a state, Hamiltonian, and time",
                )
            if not isinstance(strip_stage(positional[0].type), QuantumType):
                self._type_error(node, "qm.pauli_evolve expects an owned Q[T] value")
            if positional[1].type != Const[Observable]:
                self._type_error(
                    node,
                    "qm.pauli_evolve Hamiltonian must be Const[Observable]",
                )
            if strip_stage(positional[2].type) not in {Int, Real}:
                self._type_error(node, "qm.pauli_evolve time must be Int or Real")
            return HIRExpression(
                "quantum.pauli_evolution",
                positional[0].type,
                positional,
                span=self._span(node),
            )
        if name == "qubits":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5032", "qm.qubits expects one size argument")
            self._require_integer(positional[0].type, node)
            size: int | str
            if positional[0].opcode == "literal":
                raw_size = positional[0].attribute("value")
                if (
                    not isinstance(raw_size, int)
                    or isinstance(raw_size, bool)
                    or raw_size <= 0
                ):
                    self._error(node, "Q5033", "qubit count must be positive")
                size = raw_size
            elif positional[0].opcode == "name":
                size = str(positional[0].attribute("name"))
            elif positional[0].opcode == "module_constant":
                raw_size = positional[0].attribute("value")
                if not isinstance(raw_size, int) or raw_size <= 0:
                    self._error(node, "Q5034", "qubit count constant must be positive")
                size = raw_size
            else:
                polynomial = self._dimension_polynomial(positional[0])
                symbolic = (
                    None if polynomial is None else _polynomial_dimension(*polynomial)
                )
                if polynomial is not None and not polynomial[1] and symbolic is None:
                    self._error(node, "Q5033", "qubit count must be positive")
                size = "dynamic" if symbolic is None else symbolic
            return HIRExpression(
                "quantum.alloc",
                Q[Bits[size]],
                operands,
                span=self._span(node),
            )
        if name == "prepare":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5035", "qm.prepare expects one logical value")
            inner = strip_stage(positional[0].type)
            if isinstance(inner, QuantumType):
                self._error(node, "Q5036", "qm.prepare expects classical data")
            if positional[0].opcode in {"tuple", "array"} and all(
                item.opcode == "literal"
                and item.attribute("value") in {0, 1, False, True}
                for item in positional[0].operands
            ):
                inner = Bits[len(positional[0].operands)]
            if logical_width(inner) is None:
                self._error(
                    node,
                    "Q5077",
                    f"qm.prepare currently supports only logical bit data, got {inner}",
                )
            return HIRExpression(
                "quantum.prepare",
                Q[Bits[inner.width]]
                if isinstance(inner, WordType) and inner.kind == "Bits"
                else Q[inner],
                operands,
                span=self._span(node),
            )
        if name == "as_bits":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5105", "qm.as_bits expects exactly one value")
            state_type = strip_stage(positional[0].type)
            if not isinstance(state_type, QuantumType):
                self._type_error(
                    node, "qm.as_bits expects Q[UInt[N]] or Q[UFixed[I, F]]"
                )
            payload = state_type.inner
            if isinstance(payload, WordType) and payload.kind == "UInt":
                width = payload.width
            elif isinstance(payload, FixedType):
                integer_bits = payload.integer_bits
                fractional_bits = payload.fractional_bits
                if not isinstance(integer_bits, int) or not isinstance(
                    fractional_bits, int
                ):
                    self._type_error(
                        node,
                        "qm.as_bits currently requires concrete UFixed partitions",
                    )
                width = integer_bits + fractional_bits
            else:
                self._type_error(
                    node, "qm.as_bits expects Q[UInt[N]] or Q[UFixed[I, F]]"
                )
            return HIRExpression(
                "qcore.reshape",
                Q[Bits[width]],
                (positional[0],),
                span=self._span(node),
            )
        if name == "as_fixed":
            if len(positional) != 1:
                self._error(
                    node,
                    "Q5100",
                    "qm.as_fixed expects one state and optional integer_bits keyword",
                )
            if keywords and tuple(name for name, _ in keywords) != ("integer_bits",):
                self._error(
                    node,
                    "Q5102",
                    "qm.as_fixed accepts only the integer_bits keyword",
                )
            state_type = strip_stage(positional[0].type)
            if (
                not isinstance(state_type, QuantumType)
                or not isinstance(state_type.inner, WordType)
                or state_type.inner.kind != "Bits"
            ):
                self._type_error(node, "qm.as_fixed expects Q[Bits[N]]")
            width = state_type.inner.width
            partition = (
                keywords[0][1]
                if keywords
                else HIRExpression(
                    "literal",
                    Int,
                    attributes=attributes(value=0),
                    span=self._span(node),
                )
            )
            integer_bits = _static_integer_value(partition)
            if (
                integer_bits is None
                or isinstance(integer_bits, bool)
                or integer_bits < 0
                or (isinstance(width, int) and integer_bits > width)
                or (isinstance(width, str) and integer_bits != 0)
            ):
                self._error(
                    node,
                    "Q5104",
                    f"qm.as_fixed integer_bits is incompatible with width {width}",
                )
            fractional_bits: int | str
            if isinstance(width, int):
                fractional_bits = width - integer_bits
            else:
                fractional_bits = width
            return HIRExpression(
                "qcore.reshape",
                Q[UFixed[integer_bits, fractional_bits]],
                (positional[0],),
                span=self._span(node),
            )
        if name == "as_uint":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5105", "qm.as_uint expects exactly one state")
            state_type = strip_stage(positional[0].type)
            if (
                not isinstance(state_type, QuantumType)
                or not isinstance(state_type.inner, WordType)
                or state_type.inner.kind != "Bits"
            ):
                self._type_error(node, "qm.as_uint expects Q[Bits[N]]")
            return HIRExpression(
                "qcore.reshape",
                Q[UInt[state_type.inner.width]],
                (positional[0],),
                span=self._span(node),
            )
        if name == "split_at":
            if len(positional) != 2 or keywords:
                self._error(
                    node,
                    "Q5095",
                    "qm.split_at expects a state and boundary",
                )
            widths = _coherent_leaf_dimensions(positional[0].type)
            if widths is None or len(widths) != 1:
                self._type_error(node, "qm.split_at expects Q[Bits[N]]")
            width = widths[0]
            boundary_polynomial = self._dimension_polynomial(positional[1])
            if boundary_polynomial is None:
                self._error(
                    node,
                    "Q5096",
                    "qm.split_at boundary must be a compile-time integer or an "
                    "integer expression (+, -, *, // by an integer) over Const Int "
                    "parameters",
                )
            prefix = _polynomial_dimension(*boundary_polynomial)
            suffix = _polynomial_dimension(
                *_subtract_polynomials(
                    _dimension_to_polynomial(width), boundary_polynomial
                )
            )
            if prefix is None or suffix is None:
                bound = width if isinstance(width, int) else f"the width {width}"
                self._error(
                    node,
                    "Q5097",
                    f"qm.split_at boundary must satisfy 0 < boundary < {bound}",
                )
            return HIRExpression(
                "qcore.reshape",
                ProductType((Q[Bits[prefix]], Q[Bits[suffix]])),
                (positional[0],),
                span=self._span(node),
            )
        if name == "concat":
            if len(positional) < 2 or keywords:
                self._error(
                    node,
                    "Q5098",
                    "qm.concat expects at least two coherent values",
                )
            product = ProductType(tuple(value.type for value in positional))
            leaves = [_coherent_leaf_dimensions(value.type) for value in positional]
            if any(leaf is None for leaf in leaves):
                self._type_error(node, "qm.concat expects coherent values")
            total: tuple[int, dict[tuple[str, ...], int]] = (0, {})
            for leaf in leaves:
                assert leaf is not None
                for dimension in leaf:
                    total = _add_polynomials(total, _dimension_to_polynomial(dimension))
            combined = _polynomial_dimension(*total)
            if combined is None:
                self._type_error(node, "qm.concat width exceeds the dimension budget")
            packed = HIRExpression(
                "tuple",
                product,
                positional,
                span=self._span(node),
            )
            return HIRExpression(
                "qcore.reshape",
                Q[Bits[combined]],
                (packed,),
                span=self._span(node),
            )
        if name in GATES_BY_NAME:
            if not positional:
                self._error(node, "Q5037", f"qm.{name} expects an owned quantum state")
            state_type = positional[0].type
            if not isinstance(state_type, QuantumType):
                self._error(node, "Q5038", f"qm.{name} expects Q[T], got {state_type}")
            gate_operands, gate_keywords = self._bind_gate_arguments(
                name,
                positional,
                keywords,
                state_type,
                node,
            )
            return HIRExpression(
                f"quantum.{name}",
                state_type,
                gate_operands,
                attributes=attributes(keywords=gate_keywords),
                span=self._span(node),
            )
        if name == "measure":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5039", "qm.measure expects one owned quantum state")
            state_type = positional[0].type
            if not isinstance(state_type, QuantumType):
                self._error(node, "Q5040", f"qm.measure expects Q[T], got {state_type}")
            return HIRExpression(
                "quantum.measure",
                Shot[state_type.inner],
                operands,
                span=self._span(node),
            )
        if name in {"measure_x", "measure_y", "measure_z"}:
            if not positional:
                self._error(node, "Q5085", f"qm.{name} expects an owned state")
            state_type = positional[0].type
            if not isinstance(state_type, QuantumType):
                self._error(node, "Q5086", f"qm.{name} expects Q[T]")
            measurement_operands, measurement_keywords = self._bind_gate_arguments(
                name,
                positional,
                keywords,
                state_type,
                node,
            )
            return HIRExpression(
                f"quantum.{name}",
                ProductType((Shot[Bool], state_type)),
                measurement_operands,
                attributes=attributes(keywords=measurement_keywords),
                span=self._span(node),
            )
        if name == "measure_reset":
            if len(positional) != 1 or keywords:
                self._error(
                    node,
                    "Q5083",
                    "qm.measure_reset expects one owned quantum state",
                )
            state_type = positional[0].type
            if not isinstance(state_type, QuantumType):
                self._error(node, "Q5084", "qm.measure_reset expects Q[T]")
            return HIRExpression(
                "quantum.measure_reset",
                ProductType((Shot[state_type.inner], state_type)),
                operands,
                span=self._span(node),
            )
        if name == "discard":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5043", "qm.discard expects one owned quantum state")
            if not isinstance(positional[0].type, QuantumType):
                self._error(node, "Q5044", "qm.discard expects Q[T]")
            return HIRExpression(
                "quantum.discard",
                Unit,
                operands,
                span=self._span(node),
            )
        if name == "reset":
            if len(positional) != 1 or keywords:
                self._error(node, "Q5045", "qm.reset expects one owned quantum state")
            if not isinstance(positional[0].type, QuantumType):
                self._error(node, "Q5046", "qm.reset expects Q[T]")
            return HIRExpression(
                "quantum.reset",
                positional[0].type,
                operands,
                span=self._span(node),
            )
        self._error(node, "Q5047", f"intrinsic qm.{name} is not implemented")
        raise AssertionError

    def _bind_gate_arguments(
        self,
        name: str,
        positional: tuple[HIRExpression, ...],
        keywords: tuple[tuple[str, HIRExpression], ...],
        state_type: QuantumType,
        node: ast.Call,
    ) -> tuple[tuple[HIRExpression, ...], tuple[str, ...]]:
        """Bind and validate one primitive gate's source signature."""

        spec = GATES_BY_NAME.get(name)
        if spec is not None:
            parameter_names = spec.parameters
            required = frozenset({"angle"}) if spec.has_angle else frozenset()
        elif name in {"measure_x", "measure_y", "measure_z"}:
            parameter_names = ("target",)
            required = frozenset()
        else:
            self._error(node, "Q5047", f"intrinsic qm.{name} is not implemented")
        trailing = positional[1:]
        if len(trailing) > len(parameter_names):
            self._error(node, "Q5071", f"too many arguments for qm.{name}")
        bound = dict(zip(parameter_names, trailing, strict=False))
        for keyword, value in keywords:
            if keyword not in parameter_names:
                self._error(
                    node,
                    "Q5072",
                    f"unknown qm.{name} argument {keyword!r}",
                )
            if keyword in bound:
                self._error(
                    node,
                    "Q5073",
                    f"qm.{name} argument {keyword!r} was supplied twice",
                )
            bound[keyword] = value
        missing = required - bound.keys()
        if missing:
            self._error(
                node,
                "Q5074",
                f"qm.{name} is missing argument {next(iter(sorted(missing)))!r}",
            )
        if (
            name in {"measure_x", "measure_y", "measure_z"}
            and "target" not in bound
            and logical_width(state_type) != 1
        ):
            self._error(
                node,
                "Q5074",
                f"qm.{name} requires target= for a state whose width is not one",
            )
        if "angle" in bound and strip_stage(bound["angle"].type) not in {Int, Real}:
            self._type_error(node, f"qm.{name} angle must be Real")
        selector_names = tuple(
            parameter
            for parameter in parameter_names
            if parameter != "angle" and parameter in bound
        )
        for selector in selector_names:
            self._require_integer(bound[selector].type, node)
        self._validate_static_gate_indices(
            name,
            state_type,
            {selector: bound[selector] for selector in selector_names},
            node,
        )
        ordered = [positional[0]]
        if "angle" in bound:
            ordered.append(bound["angle"])
        ordered.extend(bound[selector] for selector in selector_names)
        return tuple(ordered), selector_names

    def _validate_static_gate_indices(
        self,
        name: str,
        state_type: QuantumType,
        selectors: dict[str, HIRExpression],
        node: ast.Call,
    ) -> None:
        """Reject statically known invalid component selectors at the source span."""

        width = logical_width(state_type)
        known: dict[str, int] = {}
        spec = GATES_BY_NAME.get(name)
        defaults = (
            dict(spec.default_indices)
            if spec is not None and not spec.broadcasts_when_unselected
            else {}
        )
        known.update(
            (selector, value)
            for selector, value in defaults.items()
            if selector not in selectors
        )
        for selector, expression in selectors.items():
            if expression.opcode != "literal":
                continue
            value = expression.attribute("value")
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            known[selector] = value
            if value < 0 or (width is not None and value >= width):
                self._error(
                    node,
                    "Q5075",
                    f"qm.{name} selector {selector}={value} is outside width {width}",
                )
        for selector, value in known.items():
            if selector in selectors:
                continue
            if value < 0 or (width is not None and value >= width):
                self._error(
                    node,
                    "Q5075",
                    f"qm.{name} default {selector}={value} is outside width {width}",
                )
        pairs = (
            ("control", "target"),
            ("control1", "control2"),
            ("control1", "target"),
            ("control2", "target"),
            ("left", "right"),
        )
        for left, right in pairs:
            if left in known and right in known and known[left] == known[right]:
                self._error(
                    node,
                    "Q5076",
                    f"qm.{name} requires distinct {left} and {right} selectors",
                )

    def _bind_call(
        self,
        signature: _Signature,
        positional: tuple[HIRExpression, ...],
        keywords: tuple[tuple[str, HIRExpression], ...],
        node: ast.Call,
    ) -> _BoundArguments:
        """Bind a static call through the native declaration-order authority."""

        function_type = FunctionType(
            tuple(parameter.type for parameter in signature.parameters),
            signature.result_type,
            tuple(parameter.name for parameter in signature.parameters),
        )
        raw = positional + tuple(value for _, value in keywords)
        try:
            plan = bind_declared_function_arguments(
                function_type,
                argument_count=len(raw),
                positional_count=len(positional),
                keyword_names=tuple(name for name, _ in keywords),
                has_defaults=tuple(
                    parameter.has_default for parameter in signature.parameters
                ),
            )
        except TypeError:
            raise
        except ValueError as error:
            diagnostic = str(error)
            for code in ("Q5048", "Q5049", "Q5050", "Q5051"):
                prefix = f"{code}: "
                if diagnostic.startswith(prefix):
                    self._error(node, code, diagnostic.removeprefix(prefix))
            reason = diagnostic.removeprefix(
                "Q4510: invalid function-argument binding input: "
            )
            self._error(node, "Q5099", reason)
        ordered: list[HIRExpression] = []
        source_order: list[int | None] = [None] * len(raw)
        defaults: list[int] = []
        for parameter_index, (parameter, raw_index) in enumerate(
            zip(signature.parameters, plan, strict=True)
        ):
            if raw_index is None:
                value = HIRExpression(
                    "literal",
                    parameter.type,
                    attributes=attributes(value=parameter.default),
                    span=self._span(node),
                )
                defaults.append(parameter_index)
            else:
                value = raw[raw_index]
                source_order[raw_index] = parameter_index
            ordered.append(value)
        if any(index is None for index in source_order):
            raise AssertionError("native argument binding returned an incomplete plan")
        return _BoundArguments(
            tuple(ordered),
            tuple(cast(int, index) for index in source_order) + tuple(defaults),
        )

    @staticmethod
    def _evaluation_attributes(
        evaluation_order: tuple[int, ...],
        **values: PortableValue,
    ) -> tuple[tuple[str, PortableValue], ...]:
        """Attach transient operand scheduling only for a nonidentity order."""

        if evaluation_order != tuple(range(len(evaluation_order))):
            values["evaluation_order"] = evaluation_order
        return attributes(**values)

    def _infer_declared_call_type(
        self,
        signature: _Signature,
        arguments: tuple[HIRExpression, ...],
        node: ast.Call,
    ) -> TypeSpec:
        """Infer one declared call result through the native type authority."""

        static_arguments = tuple(
            _static_dimension_candidate(argument) for argument in arguments
        )
        function_type = FunctionType(
            tuple(parameter.type for parameter in signature.parameters),
            signature.result_type,
            tuple(parameter.name for parameter in signature.parameters),
        )
        try:
            return infer_declared_function_application(
                function_type,
                tuple(argument.type for argument in arguments),
                static_arguments,
            )
        except (TypeError, ValueError) as error:
            if str(error).startswith("Q4511:"):
                raise
            for parameter, argument in zip(
                signature.parameters, arguments, strict=True
            ):
                if not is_type_assignable(parameter.type, argument.type):
                    self._type_error(
                        node,
                        f"expected {parameter.type}, got {argument.type}",
                    )
            reason = str(error).removeprefix("Q4510: invalid type-relation input: ")
            self._type_error(node, reason)

    def _binary_expression(
        self,
        operator: ast.operator,
        left: HIRExpression,
        right: HIRExpression,
        node: ast.AST,
    ) -> HIRExpression:
        """Type one Python binary expression.

        Coherent unsigned addition and subtraction are linear transitions:
        the left register is updated modulo its width and the right register
        is retained. Their product result requires explicit destructuring, as
        in ``left, right = left + right``.

        Args:
            operator (ast.operator): Parsed Python binary operator.
            left (HIRExpression): Evaluated left operand.
            right (HIRExpression): Evaluated right operand.
            node (ast.AST): Source node used for diagnostics.

        Returns:
            HIRExpression: Typed classical expression or coherent arithmetic
                transition.

        Raises:
            FrontendError: If the operator is unsupported.
            TypeCheckError: If coherent arithmetic operands disagree in type
                or width.
        """

        if isinstance(operator, (ast.Add, ast.Sub)) and (
            _quantum_uint_width(left.type) is not None
            or _quantum_uint_width(right.type) is not None
        ):
            left_width = _quantum_uint_width(left.type)
            right_width = _quantum_uint_width(right.type)
            if left_width is not None and right.type in {Int, Const[Int]}:
                operation = (
                    "add_constant"
                    if isinstance(operator, ast.Add)
                    else "subtract_constant"
                )
                return HIRExpression(
                    f"quantum.{operation}",
                    left.type,
                    (left, right),
                    span=self._span(node),
                )
            if left_width is None or right_width is None:
                self._type_error(
                    node,
                    "coherent arithmetic requires two Q[UInt[N]] operands",
                )
            if left_width != right_width:
                self._type_error(
                    node,
                    "coherent arithmetic requires equal UInt widths",
                )
            operation = "add_values" if isinstance(operator, ast.Add) else "sub_values"
            return HIRExpression(
                f"quantum.{operation}",
                ProductType((left.type, right.type)),
                (left, right),
                span=self._span(node),
            )

        opcode = _BINARY_NAMES.get(type(operator))
        if opcode is None:
            self._error(
                node, "Q5052", f"unsupported binary operator {type(operator).__name__}"
            )
        return HIRExpression(
            f"classical.{opcode}",
            self._classical_result_type(f"classical.{opcode}", (left, right), node),
            (left, right),
            span=self._span(node),
        )

    def _classical_result_type(
        self,
        opcode: str,
        operands: tuple[HIRExpression, ...],
        node: ast.AST,
        operation_attributes: tuple[tuple[str, PortableValue], ...] = (),
    ) -> TypeSpec:
        """Apply the shared classical type equation with source diagnostics."""

        try:
            return infer_classical_result_type(
                opcode,
                tuple(operand.type for operand in operands),
                attributes=operation_attributes,
            )
        except ValueError as error:
            self._type_error(node, str(error))
        raise AssertionError

    def _assignment_targets(
        self,
        target: ast.expr,
        value_type: TypeSpec,
    ) -> tuple[str, ...]:
        """Validate a local assignment target and product destructuring."""

        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            element_types = aggregate_element_types(value_type)
            if element_types is None:
                self._error(
                    target, "Q5054", "destructuring requires a product or record value"
                )
            if len(target.elts) != len(element_types):
                self._error(target, "Q5055", "destructuring arity mismatch")
            names: list[str] = []
            for element in target.elts:
                if not isinstance(element, ast.Name):
                    self._error(
                        element, "Q5056", "nested assignment targets are unsupported"
                    )
                names.append(element.id)
            return tuple(names)
        self._error(target, "Q5057", "assignment mutates only local names")
        raise AssertionError

    def _assignment_result_types(
        self,
        targets: tuple[str, ...],
        value_type: TypeSpec,
        *,
        destructure: bool = False,
    ) -> tuple[TypeSpec, ...]:
        """Return the explicit result row for aggregate destructuring.

        Args:
            targets (tuple[str, ...]): Validated target names in source order.
            value_type (TypeSpec): Type of the assigned expression.
            destructure (bool): Whether the source target explicitly unpacks a value,
                including an aggregate with one element.

        Returns:
            tuple[TypeSpec, ...]: Element types for unpacking, or an empty row
                for direct rebinding.
        """

        if len(targets) == 1 and not destructure:
            return ()
        element_types = aggregate_element_types(value_type)
        assert element_types is not None
        return element_types

    def _bind_targets(
        self,
        environment: _Environment,
        targets: tuple[str, ...],
        value_type: TypeSpec,
        node: ast.AST,
        *,
        destructure: bool = False,
    ) -> _Environment:
        """Return the lexical environment after immutable value rebinding.

        Args:
            environment (_Environment): Names available after the expression.
            targets (tuple[str, ...]): Validated target names in source order.
            value_type (TypeSpec): Type of the assigned expression.
            node (ast.AST): Source node used for rebinding diagnostics.
            destructure (bool): Whether the source target explicitly unpacks a value.

        Returns:
            _Environment: A fresh mapping with each target bound to its result type.

        Raises:
            FrontendError: A target would change an existing name's type.
        """

        updated = dict(environment)
        if len(targets) == 1 and not destructure:
            types = (value_type,)
        else:
            types = aggregate_element_types(value_type)
            assert types is not None
        for name, type_spec in zip(targets, types, strict=True):
            if name in updated:
                self._require_compatible(updated[name], type_spec, node)
            updated[name] = type_spec
        return updated

    def _after_linear_consumption(
        self,
        environment: _Environment,
        expression: HIRExpression,
    ) -> _Environment:
        """Remove source names consumed by one linear expression.

        Functional rebinding adds result targets only after this transition, so
        a name such as ``state`` remains available for ``state = gate(state)``
        while a destructured ``left, right = split_at(state)`` makes the old
        aggregate name unavailable immediately. Structured control therefore
        captures only live ownership versions rather than stale SSA aliases.

        Args:
            environment: Lexical names available before the expression.
            expression: Fully typed expression being evaluated.

        Returns:
            A fresh environment without consumed linear source names.
        """

        consumed: set[str] = set()

        def visit(current: HIRExpression) -> None:
            """Collect linear name leaves from one expression tree."""

            if current.opcode == "name" and current.type.is_linear:
                name = current.attribute("name")
                if isinstance(name, str):
                    consumed.add(name)
            for operand in current.operands:
                visit(operand)
            for branch in current.regions:
                visit(branch)

        visit(expression)
        return {
            name: type_spec
            for name, type_spec in environment.items()
            if name not in consumed
        }

    def _merge_environments(
        self,
        left: _Environment,
        right: _Environment,
        node: ast.AST,
    ) -> _Environment:
        """Compute values available after both conditional regions."""

        merged: _Environment = {}
        for name in sorted(left.keys() & right.keys()):
            function_peer, names_conflict = _common_function_peer(
                left[name],
                right[name],
            )
            if names_conflict:
                left_base = strip_stage(left[name])
                right_base = strip_stage(right[name])
                assert isinstance(left_base, FunctionType)
                assert isinstance(right_base, FunctionType)
                self._type_error(
                    node,
                    f"branches bind {name!r} with incompatible function "
                    f"parameter names {left_base.parameter_names} and "
                    f"{right_base.parameter_names}",
                )
            if not local_peer_types(left[name], right[name]) and function_peer is None:
                self._type_error(
                    node,
                    f"branches bind {name!r} as incompatible types "
                    f"{left[name]} and {right[name]}",
                )
            merged[name] = at_stage(
                function_peer or strip_stage(left[name]),
                common_stage(left[name], right[name]),
            )
        return merged

    def _parse_annotation(self, annotation: ast.expr | None) -> TypeSpec:
        """Parse one Qamomile type annotation without evaluating Python.

        Args:
            annotation (ast.expr | None): Source annotation syntax.

        Returns:
            TypeSpec: Parsed logical type or a transient function type.

        Raises:
            FrontendError: If the annotation is unsupported or malformed.
            TypeCheckError: If a nested type violates a logical type contract.
        """

        if annotation is None:
            return Unit
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return Unit
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval")
            except SyntaxError as error:
                self._error(annotation, "Q5058", "invalid quoted type annotation")
                raise AssertionError from error
            return self._parse_annotation(parsed.body)
        name = self._annotation_name(annotation)
        if name is not None:
            scalar = {
                "None": Unit,
                "Unit": Unit,
                "bool": Bool,
                "Bool": Bool,
                "Bit": Bit,
                "int": Int,
                "Int": Int,
                "float": Real,
                "Real": Real,
                "complex": Complex,
                "Complex": Complex,
                "Probability": Probability,
                "Observable": Observable,
            }.get(name)
            if scalar is not None:
                return scalar
            record = self.records.get(name)
            if record is not None:
                return record.type
            self._error(annotation, "Q5059", f"unknown logical type {name!r}")
        if isinstance(annotation, ast.Subscript):
            constructor = self._annotation_name(annotation.value)
            items = (
                tuple(annotation.slice.elts)
                if isinstance(annotation.slice, ast.Tuple)
                else (annotation.slice,)
            )
            if constructor in {"Bits", "UInt"}:
                if len(items) != 1:
                    self._error(
                        annotation, "Q5060", f"{constructor} expects one dimension"
                    )
                dimension = self._parse_dimension(items[0])
                return {"Bits": Bits, "UInt": UInt}[constructor][dimension]
            if constructor == "UFixed":
                if len(items) != 2:
                    self._error(
                        annotation,
                        "Q5060",
                        "UFixed expects integer and fractional bit counts",
                    )
                return UFixed[
                    self._parse_fixed_width(items[0]),
                    self._parse_fixed_width(items[1]),
                ]
            if constructor in {
                "Q",
                "Const",
                "Param",
                "Shot",
            }:
                if len(items) != 1:
                    self._error(annotation, "Q5061", f"{constructor} expects one type")
                inner = self._parse_annotation(items[0])
                table = {
                    "Q": Q,
                    "Const": Const,
                    "Param": Param,
                    "Shot": Shot,
                }
                return table[constructor][inner]
            if constructor == "QKernel":
                parsed_items = [self._parse_annotation(item) for item in items]
                if not parsed_items:
                    self._error(annotation, "Q5061", "QKernel expects a result type")
                return FunctionType(tuple(parsed_items[:-1]), parsed_items[-1])
            if constructor == "BlockEncoding":
                if len(items) != 2:
                    self._error(
                        annotation,
                        "Q5061",
                        "BlockEncoding expects signal and system widths",
                    )
                signal = self._parse_dimension(items[0])
                system = self._parse_dimension(items[1])
                if not isinstance(signal, int) or not isinstance(system, int):
                    self._error(
                        annotation,
                        "Q5062",
                        "BlockEncoding widths must be concrete at declaration",
                    )
                state = Q[Bits[signal + system]]
                return FunctionType((state,), state)
            if constructor == "tuple":
                return ProductType(
                    tuple(self._parse_annotation(item) for item in items)
                )
            if constructor == "Vector":
                if len(items) == 1:
                    return Vector[self._parse_dimension(items[0])]
                if len(items) == 2:
                    return Vector[
                        self._parse_annotation(items[0]),
                        self._parse_dimension(items[1]),
                    ]
            if constructor == "Matrix":
                if len(items) == 2:
                    return Matrix[
                        self._parse_dimension(items[0]),
                        self._parse_dimension(items[1]),
                    ]
                if len(items) == 3:
                    return Matrix[
                        self._parse_annotation(items[0]),
                        self._parse_dimension(items[1]),
                        self._parse_dimension(items[2]),
                    ]
            if constructor == "Tensor" and len(items) >= 4:
                return Tensor[
                    (
                        self._parse_annotation(items[0]),
                        *(self._parse_dimension(item) for item in items[1:]),
                    )
                ]
            if constructor == "Map" and len(items) == 3:
                return Map[
                    self._parse_annotation(items[0]),
                    self._parse_annotation(items[1]),
                    self._parse_dimension(items[2]),
                ]
            self._error(
                annotation, "Q5062", f"unsupported type constructor {constructor!r}"
            )
        self._error(annotation, "Q5063", "type annotation is not statically resolvable")
        raise AssertionError

    def _dimension_polynomial(
        self, expression: HIRExpression
    ) -> tuple[int, dict[tuple[str, ...], int]] | None:
        """Read a typed integer expression as a dimension polynomial.

        Only literals, module constants, and Const Int parameters that the
        function never rebinds can appear, combined with ``+``, ``-``, unary
        signs, and ``*``. Such parameters are the dependent dimension binders
        that specialization substitutes, so the resulting shape stays exact
        for every admitted binding.

        Args:
            expression (HIRExpression): Typed boundary or size expression.

        Returns:
            tuple[int, dict[tuple[str, ...], int]] | None: Constant and
                monomial coefficients, or ``None`` outside that fragment.
        """

        value = _static_integer_value(expression)
        if value is not None:
            return value, {}
        opcode = expression.opcode
        if opcode == "name":
            name = expression.attribute("name")
            if (
                isinstance(name, str)
                and name in self._dimension_parameters
                and expression.type == Const[Int]
            ):
                return 0, {(name,): 1}
            return None
        if opcode == "classical.pos" and len(expression.operands) == 1:
            return self._dimension_polynomial(expression.operands[0])
        if opcode == "classical.neg" and len(expression.operands) == 1:
            operand = self._dimension_polynomial(expression.operands[0])
            return None if operand is None else _subtract_polynomials((0, {}), operand)
        if opcode in {"classical.add", "classical.sub", "classical.mul"} and (
            len(expression.operands) == 2
        ):
            left = self._dimension_polynomial(expression.operands[0])
            right = self._dimension_polynomial(expression.operands[1])
            if left is None or right is None:
                return None
            if opcode == "classical.add":
                return _add_polynomials(left, right)
            if opcode == "classical.sub":
                return _subtract_polynomials(left, right)
            return _multiply_polynomials(left, right)
        if opcode == "classical.floordiv" and len(expression.operands) == 2:
            numerator = self._dimension_polynomial(expression.operands[0])
            divisor = _static_integer_value(expression.operands[1])
            if numerator is None or divisor is None:
                return None
            return _floor_quotient(numerator, divisor)
        return None

    def _parse_dimension(self, expression: ast.expr) -> int | str:
        """Parse and canonicalize one positive polynomial logical dimension.

        Args:
            expression (ast.expr): Dimension annotation expression.

        Returns:
            int | str: A positive literal or canonical affine spelling.

        Raises:
            FrontendError: If the expression is outside the bounded polynomial
                language or a concrete value is not positive.
        """

        constant, terms = self._parse_polynomial_dimension(expression)
        if not terms:
            if constant <= 0:
                self._error(expression, "Q5064", "dimension must be positive")
            return constant
        return _format_polynomial_dimension(constant, terms)

    def _parse_polynomial_dimension(
        self,
        expression: ast.expr,
    ) -> tuple[int, dict[tuple[str, ...], int]]:
        """Reduce one shape annotation to a bounded integer polynomial.

        Args:
            expression (ast.expr): Dimension expression to reduce.

        Returns:
            tuple[int, dict[tuple[str, ...], int]]: Constant and canonical
                monomial coefficients.

        Raises:
            FrontendError: If unsupported syntax, degree, term count, factor
                count, or coefficient magnitude exceeds the closed budget.
        """

        def add(
            left: tuple[int, dict[tuple[str, ...], int]],
            right: tuple[int, dict[tuple[str, ...], int]],
            scale: int,
        ) -> tuple[int, dict[tuple[str, ...], int]]:
            result = dict(left[1])
            for monomial, coefficient in right[1].items():
                result[monomial] = result.get(monomial, 0) + scale * coefficient
                if result[monomial] == 0:
                    del result[monomial]
            return left[0] + scale * right[0], result

        def multiply(
            left: tuple[int, dict[tuple[str, ...], int]],
            right: tuple[int, dict[tuple[str, ...], int]],
        ) -> tuple[int, dict[tuple[str, ...], int]]:
            left_terms = {(): left[0], **left[1]}
            right_terms = {(): right[0], **right[1]}
            result: dict[tuple[str, ...], int] = {}
            for left_monomial, left_coefficient in left_terms.items():
                for right_monomial, right_coefficient in right_terms.items():
                    monomial = tuple(sorted((*left_monomial, *right_monomial)))
                    result[monomial] = result.get(monomial, 0) + (
                        left_coefficient * right_coefficient
                    )
            constant = result.pop((), 0)
            return constant, {
                monomial: coefficient
                for monomial, coefficient in result.items()
                if coefficient
            }

        def parse(node: ast.expr) -> tuple[int, dict[tuple[str, ...], int]]:
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                if isinstance(node.value, bool):
                    self._error(node, "Q5065", "dimension must be polynomial")
                return node.value, {}
            if isinstance(node, ast.Name):
                if node.id in self.constants:
                    value = self.constants[node.id][1]
                    if isinstance(value, int) and not isinstance(value, bool):
                        return value, {}
                return 0, {(node.id,): 1}
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                value = parse(node.operand)
                return (
                    value if isinstance(node.op, ast.UAdd) else add((0, {}), value, -1)
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                return add(
                    parse(node.left),
                    parse(node.right),
                    -1 if isinstance(node.op, ast.Sub) else 1,
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                return multiply(parse(node.left), parse(node.right))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                divisor = parse(node.right)
                quotient = (
                    None
                    if divisor[1]
                    else _floor_quotient(parse(node.left), divisor[0])
                )
                if quotient is None:
                    self._error(
                        node,
                        "Q5065",
                        "dimension floor division needs an integer divisor of at least 2",
                    )
                return quotient
            self._error(
                node,
                "Q5065",
                "dimension supports only symbols, integers, +, -, multiplication, "
                "and floor division by an integer",
            )
            raise AssertionError

        constant, terms = parse(expression)
        if (
            len(terms) > 64
            or sum(len(monomial) for monomial in terms) > 64
            or any(abs(value) > 2**31 for value in (constant, *terms.values()))
        ):
            self._error(
                expression,
                "Q5065",
                "polynomial dimension exceeds its complexity budget",
            )
        return constant, terms

    def _parse_fixed_width(self, expression: ast.expr) -> int | str:
        """Parse one fixed or dependent fixed-point partition width.

        Args:
            expression (ast.expr): Annotation node naming a concrete width or
                dependent dimension.

        Returns:
            int | str: Nonnegative concrete width or symbolic width name.

        Raises:
            TypeCheckError: If the expression is not a valid partition width.
        """

        constant, terms = self._parse_affine_dimension(expression)
        if not terms:
            if constant < 0:
                self._error(
                    expression, "Q5105", "fixed-point width must be non-negative"
                )
            return constant
        return _format_affine_dimension(constant, terms)

    def _parse_affine_dimension(
        self,
        expression: ast.expr,
    ) -> tuple[int, dict[str, int]]:
        """Reduce an annotation expression to bounded affine coefficients.

        Args:
            expression (ast.expr): Dimension expression to reduce.

        Returns:
            tuple[int, dict[str, int]]: Constant term and nonzero symbol
                coefficients.

        Raises:
            FrontendError: If the expression uses nonlinear or unsupported
                syntax, or exceeds the affine-complexity budget.
        """

        def parse(node: ast.expr) -> tuple[int, dict[str, int]]:
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                if isinstance(node.value, bool):
                    self._error(node, "Q5065", "dimension must be affine")
                return node.value, {}
            if isinstance(node, ast.Name):
                if node.id in self.constants:
                    value = self.constants[node.id][1]
                    if isinstance(value, int) and not isinstance(value, bool):
                        return value, {}
                return 0, {node.id: 1}
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                constant, terms = parse(node.operand)
                scale = -1 if isinstance(node.op, ast.USub) else 1
                return scale * constant, {
                    name: scale * coefficient for name, coefficient in terms.items()
                }
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                left_constant, left_terms = parse(node.left)
                right_constant, right_terms = parse(node.right)
                scale = -1 if isinstance(node.op, ast.Sub) else 1
                combined = dict(left_terms)
                for name, coefficient in right_terms.items():
                    combined[name] = combined.get(name, 0) + scale * coefficient
                    if combined[name] == 0:
                        del combined[name]
                return left_constant + scale * right_constant, combined
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                left_constant, left_terms = parse(node.left)
                right_constant, right_terms = parse(node.right)
                if left_terms and right_terms:
                    self._error(
                        node,
                        "Q5065",
                        "dimension multiplication requires one integer operand",
                    )
                if left_terms:
                    scale, constant, terms = right_constant, left_constant, left_terms
                elif right_terms:
                    scale, constant, terms = left_constant, right_constant, right_terms
                else:
                    return left_constant * right_constant, {}
                return scale * constant, {
                    name: scale * coefficient for name, coefficient in terms.items()
                }
            self._error(
                node,
                "Q5065",
                "dimension supports only symbols, integers, +, -, and constant multiplication",
            )
            raise AssertionError

        constant, terms = parse(expression)
        if len(terms) > 64 or any(
            abs(value) > 2**31 for value in (constant, *terms.values())
        ):
            self._error(
                expression, "Q5065", "affine dimension exceeds its complexity budget"
            )
        return constant, terms

    def _declared_kind(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> CallableKind | None | object:
        """Resolve the single Qamomile function decorator and optional kind."""

        for decorator in function.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = self._call_name(target)
            intrinsic = self._intrinsic_name(name)
            if intrinsic != "qkernel":
                continue
            if not isinstance(decorator, ast.Call):
                return None
            if decorator.args or any(
                keyword.arg not in {"kind", "retain_docstrings"}
                for keyword in decorator.keywords
            ):
                self._error(
                    decorator,
                    "Q5096",
                    "qm.qkernel accepts only the optional kind= and "
                    "retain_docstrings= keywords",
                )
            kind_values = [
                keyword.value for keyword in decorator.keywords if keyword.arg == "kind"
            ]
            if not kind_values:
                return None
            if (
                len(kind_values) != 1
                or not isinstance(kind_values[0], ast.Constant)
                or not isinstance(kind_values[0].value, str)
            ):
                self._error(
                    decorator,
                    "Q5097",
                    "qm.qkernel kind must be a literal callable-kind string",
                )
            try:
                return CallableKind(kind_values[0].value)
            except ValueError:
                self._error(
                    kind_values[0],
                    "Q5098",
                    f"unknown Qamomile callable kind {kind_values[0].value!r}",
                )
        return _NOT_QAMOMILE

    def _annotation_name(self, expression: ast.expr) -> str | None:
        """Resolve an annotation name through Qamomile import aliases."""

        if isinstance(expression, ast.Name):
            return self.imported_qamomile_names.get(expression.id, expression.id)
        if isinstance(expression, ast.Attribute) and isinstance(
            expression.value, ast.Name
        ):
            if expression.value.id in self.qamomile_aliases:
                return expression.attr
        return None

    def _call_name(self, expression: ast.expr) -> str:
        """Return a shallow static call name or an empty marker."""

        if isinstance(expression, ast.Name):
            return expression.id
        if isinstance(expression, ast.Attribute) and isinstance(
            expression.value, ast.Name
        ):
            return f"{expression.value.id}.{expression.attr}"
        return ""

    def _intrinsic_name(self, name: str) -> str | None:
        """Resolve a call name to a Qamomile intrinsic spelling."""

        if name in self.imported_qamomile_names:
            return self.imported_qamomile_names[name]
        if "." in name:
            prefix, suffix = name.split(".", 1)
            if prefix in self.qamomile_aliases:
                return suffix
        return None

    def _require_compatible(
        self,
        expected: TypeSpec,
        actual: TypeSpec,
        node: ast.AST,
    ) -> None:
        """Require structural compatibility while allowing staged refinement."""

        if is_type_assignable(expected, actual):
            return
        self._type_error(node, f"expected {expected}, got {actual}")

    def _require_condition(
        self,
        type_spec: TypeSpec,
        node: ast.AST,
        *,
        allow_quantum: bool,
    ) -> None:
        """Require a classical or explicitly permitted coherent Boolean."""

        base = strip_stage(type_spec)
        if base in {Bool, Bit}:
            return
        if (
            allow_quantum
            and isinstance(base, QuantumType)
            and base.inner in {Bool, Bit}
        ):
            return
        self._type_error(node, f"control condition must be Bool/Bit, got {type_spec}")

    def _require_integer(self, type_spec: TypeSpec, node: ast.AST) -> None:
        """Require an integer-valued scalar."""

        if strip_stage(type_spec) != Int:
            self._type_error(node, f"expected Int, got {type_spec}")

    def _span(self, node: ast.AST) -> SourceSpan:
        """Convert a Python AST location to a normalized source span."""

        return SourceSpan(
            self.unit.uri,
            getattr(node, "lineno", 1),
            getattr(node, "col_offset", 0),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
        )

    def _error(
        self,
        node: ast.AST,
        code: str,
        message: str,
        note: str | None = None,
    ) -> NoReturn:
        """Raise a stable source-oriented frontend error."""

        span = self._span(node)
        raise FrontendError(
            Diagnostic(
                code,
                message,
                span.uri,
                span.start_line,
                span.start_column,
                note,
            )
        )

    def _type_error(self, node: ast.AST, message: str) -> NoReturn:
        """Raise a stable source-oriented type error."""

        span = self._span(node)
        raise TypeCheckError(
            Diagnostic(
                "Q5200",
                message,
                span.uri,
                span.start_line,
                span.start_column,
            )
        )


_NOT_QAMOMILE = object()


def _is_external_declaration(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether a declaration uses the canonical bodyless ``...`` form."""

    body = _function_body(function)
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )


def _function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    """Return executable statements after the optional Python docstring."""

    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _static_integer_value(expression: HIRExpression) -> int | None:
    """Evaluate the frontend's canonical static integer fragment.

    Args:
        expression: Typed HIR expression to inspect.

    Returns:
        Its integer value, or ``None`` when it is not statically known.
    """

    if expression.opcode in {"literal", "module_constant"}:
        value = expression.attribute("value")
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if expression.opcode in {"classical.neg", "classical.pos"}:
        value = _static_integer_value(expression.operands[0])
        if value is not None and expression.opcode == "classical.neg":
            return -value
        return value
    return None


def _static_dimension_candidate(expression: HIRExpression) -> Dimension | None:
    """Project source-visible static evidence without assigning it semantics."""

    value = _static_integer_value(expression)
    if value is not None:
        return value
    if expression.opcode != "name":
        return None
    name = expression.attribute("name")
    return name if isinstance(name, str) else None


def _quantum_uint_width(type_spec: TypeSpec) -> Dimension | None:
    """Return the width of one coherent unsigned integer.

    Args:
        type_spec (TypeSpec): Logical type to inspect.

    Returns:
        Dimension | None: Fixed or dependent width for ``Q[UInt[N]]``, or
            ``None`` for every other type.
    """

    base = strip_stage(type_spec)
    if (
        isinstance(base, QuantumType)
        and isinstance(base.inner, WordType)
        and base.inner.kind == "UInt"
    ):
        return base.inner.width
    return None


_Polynomial: TypeAlias = tuple[int, dict[tuple[str, ...], int]]
_MAX_DIMENSION_TERMS = 64
_MAX_DIMENSION_MAGNITUDE = 2**31


def _add_polynomials(
    left: _Polynomial, right: _Polynomial, scale: int = 1
) -> _Polynomial:
    """Return ``left + scale * right`` without zero coefficients."""

    terms = dict(left[1])
    for monomial, coefficient in right[1].items():
        terms[monomial] = terms.get(monomial, 0) + scale * coefficient
        if terms[monomial] == 0:
            del terms[monomial]
    return left[0] + scale * right[0], terms


def _subtract_polynomials(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    """Return ``left - right``."""

    return _add_polynomials(left, right, -1)


def _multiply_polynomials(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    """Return the product of two dimension polynomials."""

    left_terms = {(): left[0], **left[1]}
    right_terms = {(): right[0], **right[1]}
    product: dict[tuple[str, ...], int] = {}
    for left_monomial, left_coefficient in left_terms.items():
        for right_monomial, right_coefficient in right_terms.items():
            monomial = tuple(sorted((*left_monomial, *right_monomial)))
            product[monomial] = product.get(monomial, 0) + (
                left_coefficient * right_coefficient
            )
    constant = product.pop((), 0)
    return constant, {
        monomial: coefficient
        for monomial, coefficient in product.items()
        if coefficient
    }


def _floor_quotient(numerator: _Polynomial, divisor: int) -> _Polynomial | None:
    """Return ``numerator // divisor`` as a constant or one opaque atom.

    The canonical atom spelling is ``(name//d)`` or ``((polynomial)//d)``,
    matching the native dimension parser; the whole factor is parenthesized so
    products such as ``2*(n//2)`` read the same in Python and Rust.
    """

    if divisor < 2:
        return None
    constant, terms = numerator
    if not terms:
        return constant // divisor, {}
    rendered = _format_polynomial_dimension(constant, terms)
    atom = (
        f"({rendered}//{divisor})"
        if rendered.isidentifier()
        else f"(({rendered})//{divisor})"
    )
    return 0, {(atom,): 1}


def _dimension_to_polynomial(dimension: Dimension) -> _Polynomial:
    """Read a fixed width or canonical symbolic spelling as a polynomial.

    Args:
        dimension (Dimension): Positive width or canonical polynomial text.

    Returns:
        _Polynomial: Constant and monomial coefficients.

    Raises:
        TypeCheckError: If the spelling is outside the dimension language.
    """

    if isinstance(dimension, int):
        return dimension, {}
    try:
        tree = ast.parse(dimension, mode="eval").body
    except SyntaxError as error:
        raise TypeCheckError(f"invalid dimension spelling {dimension!r}") from error

    def read(node: ast.expr) -> _Polynomial:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value, {}
        if isinstance(node, ast.Name):
            return 0, {(node.id,): 1}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return _subtract_polynomials((0, {}), read(node.operand))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _add_polynomials(read(node.left), read(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return _subtract_polynomials(read(node.left), read(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return _multiply_polynomials(read(node.left), read(node.right))
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Constant)
            and type(node.right.value) is int
        ):
            quotient = _floor_quotient(read(node.left), node.right.value)
            if quotient is not None:
                return quotient
        raise TypeCheckError(f"invalid dimension spelling {dimension!r}")

    return read(tree)


def _polynomial_dimension(
    constant: int, terms: dict[tuple[str, ...], int]
) -> Dimension | None:
    """Return the canonical dimension of a polynomial, or ``None`` if invalid.

    A constant polynomial must be positive. A symbolic polynomial keeps its
    canonical spelling; specialization rejects bindings that make it
    nonpositive. Polynomials over the shared complexity budget are invalid.
    """

    if not terms:
        return constant if constant > 0 else None
    if (
        len(terms) > _MAX_DIMENSION_TERMS
        or sum(len(monomial) for monomial in terms) > _MAX_DIMENSION_TERMS
        or any(
            abs(value) > _MAX_DIMENSION_MAGNITUDE
            for value in (constant, *terms.values())
        )
    ):
        return None
    if all(coefficient < 0 for coefficient in terms.values()) and constant <= 0:
        return None
    return _format_polynomial_dimension(constant, terms)


def _coherent_leaf_dimensions(type_spec: TypeSpec) -> list[Dimension] | None:
    """Return the ordered fixed or symbolic widths of coherent leaves.

    Args:
        type_spec (TypeSpec): A quantum value or product of quantum values.

    Returns:
        list[Dimension] | None: Leaf widths, or ``None`` for other types.
    """

    base = strip_stage(type_spec)

    def payload(inner: TypeSpec) -> list[Dimension] | None:
        if isinstance(inner, WordType):
            return [inner.width]
        if inner in (Bit, Bool):
            return [1]
        if isinstance(inner, FixedType):
            total = _add_polynomials(
                _dimension_to_polynomial(inner.integer_bits),
                _dimension_to_polynomial(inner.fractional_bits),
            )
            width = _polynomial_dimension(*total)
            return None if width is None else [width]
        if isinstance(inner, ProductType):
            leaves: list[Dimension] = []
            for element in inner.elements:
                nested = payload(element)
                if nested is None:
                    return None
                leaves.extend(nested)
            return leaves
        return None

    if isinstance(base, QuantumType):
        return payload(base.inner)
    if isinstance(base, ProductType):
        leaves: list[Dimension] = []
        for element in base.elements:
            nested = _coherent_leaf_dimensions(element)
            if nested is None:
                return None
            leaves.extend(nested)
        return leaves
    widths = coherent_leaf_widths(type_spec)
    return None if widths is None else list(widths)


def _format_affine_dimension(constant: int, terms: dict[str, int]) -> str:
    """Return the canonical portable spelling of one affine dimension.

    Args:
        constant (int): Constant offset.
        terms (dict[str, int]): Nonzero symbol coefficients.

    Returns:
        str: Lexicographically ordered affine expression without whitespace.
    """

    pieces: list[tuple[int, str]] = []
    for name, coefficient in sorted(terms.items()):
        magnitude = abs(coefficient)
        term = name if magnitude == 1 else f"{magnitude}*{name}"
        pieces.append((coefficient, term))
    if constant:
        pieces.append((constant, str(abs(constant))))
    rendered = ""
    for index, (coefficient, term) in enumerate(pieces):
        if index == 0:
            rendered = f"-{term}" if coefficient < 0 else term
        else:
            rendered += ("-" if coefficient < 0 else "+") + term
    return rendered


def _format_polynomial_dimension(
    constant: int,
    terms: dict[tuple[str, ...], int],
) -> str:
    """Return the canonical portable spelling of one polynomial dimension.

    Args:
        constant (int): Constant offset.
        terms (dict[tuple[str, ...], int]): Nonzero canonical monomial
            coefficients.

    Returns:
        str: Ordered polynomial expression without whitespace.
    """

    pieces: list[tuple[int, str]] = []
    for monomial, coefficient in sorted(terms.items()):
        magnitude = abs(coefficient)
        symbols = "*".join(monomial)
        term = symbols if magnitude == 1 else f"{magnitude}*{symbols}"
        pieces.append((coefficient, term))
    if constant:
        pieces.append((constant, str(abs(constant))))
    rendered = ""
    for index, (coefficient, term) in enumerate(pieces):
        if index == 0:
            rendered = f"-{term}" if coefficient < 0 else term
        else:
            rendered += ("-" if coefficient < 0 else "+") + term
    return rendered


def _expression_names(expression: HIRExpression) -> set[str]:
    """Collect source-local names referenced by one HIR expression tree."""

    names: set[str] = set()
    if expression.opcode == "name":
        name = expression.attribute("name")
        if isinstance(name, str):
            names.add(name)
    for operand in expression.operands:
        names.update(_expression_names(operand))
    for branch in expression.regions:
        names.update(_expression_names(branch))
    return names


def _assigned_names(statements: list[ast.stmt]) -> frozenset[str]:
    """Collect local bindings, including functional element updates.

    Element assignment creates a successor aggregate and rebinds its local
    name, although Python represents that name as a load in the syntax tree.

    Args:
        statements (list[ast.stmt]): Statements in one structured source region.

    Returns:
        frozenset[str]: Local names possibly rebound in the region.
    """

    tree = ast.Module(body=statements, type_ignores=[])
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
        ):
            names.add(node.value.id)
    return frozenset(names)


def _join_control_stage(type_spec: TypeSpec, control: TypeSpec) -> TypeSpec:
    """Add one control dependency to a classical value type."""

    if type_spec.is_linear:
        return type_spec
    return at_stage(strip_stage(type_spec), common_stage(type_spec, control))


def _common_function_peer(
    left: TypeSpec,
    right: TypeSpec,
) -> tuple[FunctionType | None, bool]:
    """Join compatible callable labels without inventing keyword semantics.

    An unlabeled ``QKernel[...]`` annotation and a labeled direct kernel reference
    have the same positional denotation. Their join deliberately erases labels,
    while two distinct complete label rows remain an explicit source error.
    """

    left_base = strip_stage(left)
    right_base = strip_stage(right)
    if not isinstance(left_base, FunctionType) or not isinstance(
        right_base, FunctionType
    ):
        return None, False
    if left_base.inputs != right_base.inputs or left_base.output != right_base.output:
        return None, False
    if (
        left_base.parameter_names
        and right_base.parameter_names
        and left_base.parameter_names != right_base.parameter_names
    ):
        return None, True
    parameter_names = (
        left_base.parameter_names
        if left_base.parameter_names == right_base.parameter_names
        else ()
    )
    return (
        FunctionType(left_base.inputs, left_base.output, parameter_names),
        False,
    )


def _contains_function_type(type_spec: TypeSpec) -> bool:
    """Return whether a semantic type tree contains a qkernel value.

    Args:
        type_spec (TypeSpec): Semantic type tree to inspect.

    Returns:
        bool: Whether any node is a :class:`FunctionType`.

    Raises:
        TypeError: If ``type_spec`` is not a semantic type.
    """

    base = strip_stage(type_spec)
    if isinstance(base, FunctionType):
        return True
    if isinstance(base, ArrayType):
        return _contains_function_type(base.element)
    if isinstance(base, MapType):
        return _contains_function_type(base.key) or _contains_function_type(base.value)
    if isinstance(base, QuantumType):
        return _contains_function_type(base.inner)
    if isinstance(base, ProductType):
        return any(_contains_function_type(element) for element in base.elements)
    if isinstance(base, RecordType):
        return any(_contains_function_type(field) for _, field in base.fields)
    return False


def _join_loop_stage(
    initial: TypeSpec,
    body: TypeSpec,
    controls: tuple[TypeSpec, ...],
) -> TypeSpec:
    """Compute one monotone loop-carried availability fixed-point step."""

    if initial.is_linear:
        return initial
    return at_stage(strip_stage(initial), common_stage(initial, body, *controls))


def _is_total_boolean_expression(expression: ast.expr) -> bool:
    """Recognize expressions whose eager evaluation is observationally lazy.

    Qamomile classical primitives in this subset are immutable and total. Calls,
    indexing, and arithmetic are excluded because they may diverge or fail and
    therefore must retain Python short-circuit control as statements.

    Args:
        expression: Python expression to classify.

    Returns:
        Whether evaluating the expression has no observable partial behavior.
    """

    if isinstance(expression, (ast.Name, ast.Constant)):
        return True
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        return _is_total_boolean_expression(expression.operand)
    if isinstance(expression, ast.BoolOp):
        return all(_is_total_boolean_expression(item) for item in expression.values)
    if isinstance(expression, ast.Compare):
        return all(
            _is_total_boolean_expression(item)
            for item in (expression.left, *expression.comparators)
        )
    if isinstance(expression, ast.IfExp):
        return all(
            _is_total_boolean_expression(item)
            for item in (expression.test, expression.body, expression.orelse)
        )
    return False


def _block_guarantees_return(statements: list[ast.stmt]) -> bool:
    """Return whether every reachable path in a source block returns."""

    for statement in statements:
        if isinstance(statement, ast.Return):
            return True
        if (
            isinstance(statement, ast.If)
            and _block_guarantees_return(statement.body)
            and _block_guarantees_return(statement.orelse)
        ):
            return True
        if isinstance(statement, ast.For) and _for_guarantees_return(statement):
            return True
        if isinstance(statement, ast.While) and _while_guarantees_return(statement):
            return True
        if isinstance(statement, (ast.Break, ast.Continue)):
            return False
    return False


def _contains_loop_break(statements: list[ast.stmt]) -> bool:
    """Detect a break belonging to the current loop, excluding nested loops."""

    for statement in statements:
        if isinstance(statement, ast.Break):
            return True
        if isinstance(statement, (ast.For, ast.While)):
            if _contains_loop_break(statement.orelse):
                return True
            continue
        children = (
            (*statement.body, *statement.orelse)
            if isinstance(statement, ast.If)
            else ()
        )
        if children and _contains_loop_break(list(children)):
            return True
    return False


def _while_guarantees_return(statement: ast.While) -> bool:
    """Prove that every terminating exit reaches a returning else arm.

    Args:
        statement (ast.While): Source loop to inspect.

    Returns:
        bool: True when no break bypasses an else arm that always returns.
    """
    return (
        bool(statement.orelse)
        and not _contains_loop_break(statement.body)
        and _block_guarantees_return(statement.orelse)
    )


def _static_range_nonempty(expression: ast.expr) -> bool:
    """Prove that a literal Python range contains at least one element."""

    if not isinstance(expression, ast.Call) or not isinstance(
        expression.func, ast.Name
    ):
        return False
    if expression.func.id != "range" or not 1 <= len(expression.args) <= 3:
        return False
    if not all(
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, int)
        and not isinstance(argument.value, bool)
        for argument in expression.args
    ):
        return False
    values = cast(
        tuple[int, ...],
        tuple(cast(ast.Constant, item).value for item in expression.args),
    )
    try:
        return len(range(*values)) > 0
    except ValueError:
        return False


def _for_guarantees_return(statement: ast.For) -> bool:
    """Prove all-path return for a structured Python for statement."""

    if _static_range_nonempty(statement.iter) and _block_guarantees_return(
        statement.body
    ):
        return True
    return (
        bool(statement.orelse)
        and not _contains_loop_break(statement.body)
        and _block_guarantees_return(statement.orelse)
    )


_BINARY_NAMES: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.BitAnd: "and_bits",
    ast.BitOr: "or_bits",
    ast.BitXor: "xor",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
}

_COMPARE_NAMES: dict[type[ast.cmpop], str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.In: "in",
    ast.NotIn: "not_in",
}
