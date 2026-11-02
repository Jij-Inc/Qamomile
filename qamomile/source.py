"""Public source-module build API and non-tracing decorator handles."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, ParamSpec, TypeVar, cast, overload

from ._compiler.frontend import SourceUnit, parse_source
from ._compiler.lowering import lower_hir
from .effects import CallableKind
from .errors import Diagnostic, SourceUnavailableError, TypeCheckError
from .ir import Function, Module, Program

_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)
_COMPOSITE_PREFIX = "__qamomile_box_"


@dataclass(frozen=True, slots=True)
class ProgramFamily:
    """One unspecialized entrypoint and its closed source-module environment."""

    module: Module
    entrypoint: str

    def __post_init__(self) -> None:
        """Resolve the entrypoint eagerly so ambiguity cannot persist."""

        self.module.function(self.entrypoint)

    @property
    def function(self) -> Function:
        """Return the typed QCore callable representing this program family."""

        return self.module.function(self.entrypoint)

    @property
    def semantic_digest(self) -> str:
        """Return the stable family identity including its selected entrypoint."""

        return self.module.selected_digest(self.entrypoint)

    def specialize(
        self,
        **bindings: object,
    ) -> Program:
        """Specialize this family with Const values supplied by name.

        Args:
            **bindings (object): Const values keyed by source parameter name.

        Returns:
            Program: Closed verified program with Const values materialized.

        Raises:
            SpecializationError: If a binding is missing, unknown, or ill-typed.
        """

        return self.module.specialize(
            self.entrypoint,
            **bindings,
        )

    def inspect(self) -> str:
        """Render the unspecialized typed QCore callable."""

        return self.function.inspect()


def build(
    path: str | Path,
    *,
    module_name: str | None = None,
) -> Module:
    """Compile a Python source file without importing or executing it.

    Args:
        path (str | Path): Source file containing Qamomile declarations.
        module_name (str | None): Optional stable semantic module name.

    Returns:
        Module: A verified, serializable source module.

    Raises:
        QamomileError: If parsing, type checking, lowering, or native
            verification fails.
    """

    return lower_hir(parse_source(SourceUnit.from_path(path, module_name=module_name)))


def build_source(
    source: str,
    *,
    module_name: str = "module",
    display_name: str = "module.py",
    origins: Mapping[str, str] | None = None,
    retain_docstrings: bool = True,
) -> Module:
    """Compile provided source text without executing it.

    Args:
        source (str): Python source text containing Qamomile declarations.
        module_name (str): Stable semantic module name.
        display_name (str): Diagnostic-only basename.
        origins (Mapping[str, str] | None): Declared function name to the
            qualified symbol that originally declared it. Each listed function
            carries that symbol as its ``origin`` intent row.
        retain_docstrings (bool): Whether function docstrings persist as
            ``doc`` intent rows. Intent never changes program meaning or its
            semantic digest, but it does enter persisted module bytes.

    Returns:
        Module: A verified, serializable source module.

    Raises:
        QamomileError: If parsing, type checking, lowering, or native
            verification fails.
    """

    unit = SourceUnit.from_text(
        source,
        display_name=display_name,
        module_name=module_name,
        origins=origins,
        retain_docstrings=retain_docstrings,
    )
    return lower_hir(parse_source(unit))


@dataclass(frozen=True, slots=True)
class QKernel(Generic[_P, _R_co]):
    """A source-only quantum kernel and its captured module.

    The handle stores source text and symbol identity only. It never stores the
    Python callable and never runs its body. Persistent semantics begin at the
    resulting :class:`Module`. Its generic signature exists so calls made
    in another source declaration retain the original Python parameter and
    result types during static analysis; host execution remains prohibited.

    Attributes:
        module_name (str): Stable source module name.
        function_name (str): Declared function name inside the module.
        source_text (str): Isolated Python source captured for compilation.
        display_name (str): User-facing source filename for diagnostics.
        origins (tuple[tuple[str, str], ...]): Declared function name to the
            qualified symbol of the kernel that originally declared it. Linked
            dependencies keep their own symbols; the declared function maps to
            this kernel's symbol when nothing else is recorded.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.
    """

    module_name: str
    function_name: str
    source_text: str
    display_name: str
    origins: tuple[tuple[str, str], ...] = ()
    retain_docstrings: bool = True

    def __post_init__(self) -> None:
        """Record this kernel as the origin of its own declaration."""

        origins = dict(self.origins)
        origins.setdefault(self.function_name, self.symbol)
        object.__setattr__(self, "origins", tuple(sorted(origins.items())))

    def __class_getitem__(cls, items: object) -> object:
        """Construct a source-language qkernel signature annotation.

        Args:
            items (object): Zero or more input types followed by the result
                type.

        Returns:
            object: The corresponding immutable semantic function type.

        Raises:
            TypeCheckError: If an item is not a Qamomile logical type.
        """

        from .types import FunctionType, as_type

        type_items = list(items) if isinstance(items, tuple) else [items]
        if not type_items:
            raise TypeCheckError(
                "QKernel expects QKernel[result] or QKernel[input, ..., result]."
            )
        return FunctionType(
            tuple(as_type(input_type) for input_type in type_items[:-1]),
            as_type(type_items[-1]),
        )

    @property
    def symbol(self) -> str:
        """Return the stable semantic symbol.

        Returns:
            str: Fully qualified semantic symbol.
        """

        return f"{self.module_name}.{self.function_name}"

    def module(self) -> Module:
        """Compile and return the declaration's source module.

        Returns:
            Module: Verified source module.

        Raises:
            QamomileError: If source parsing, lowering, or verification fails.
        """

        return build_source(
            self.source_text,
            module_name=self.module_name,
            display_name=self.display_name,
            origins=dict(self.origins),
            retain_docstrings=self.retain_docstrings,
        )

    def family(self) -> ProgramFamily:
        """Compile this declaration into an unspecialized program family.

        Returns:
            ProgramFamily: Unspecialized entrypoint family.

        Raises:
            QamomileError: If source parsing, lowering, or verification fails.
        """

        return ProgramFamily(self.module(), self.function_name)

    def specialize(
        self,
        **bindings: object,
    ) -> Program:
        """Specialize this declaration using named static bindings.

        Args:
            **bindings (object): Const values or BlockEncoding descriptors
                keyed by source parameter name.

        Returns:
            Program: Closed or launch-parametric verified program.

        Raises:
            QamomileError: If bindings or specialization are invalid.
        """

        source, constants = self._bind_source_descriptors(bindings)
        return source.module().specialize(source.function_name, **constants)

    def _bind_source_descriptors(
        self, bindings: Mapping[str, object]
    ) -> tuple[QKernel[_P, _R_co], dict[str, object]]:
        """Resolve library descriptors before compiling a symbolic source family.

        Args:
            bindings (Mapping[str, object]): Source descriptors and Const bindings.

        Returns:
            tuple: Linked source and remaining native specialization bindings.

        Raises:
            TypeError: If a descriptor does not match its source annotation.
        """
        from .stdlib.block_encoding import BlockEncoding

        blocks = {
            name: value
            for name, value in bindings.items()
            if isinstance(value, BlockEncoding)
        }
        source = self.bind(**blocks) if blocks else self
        constants = {
            name: value for name, value in bindings.items() if name not in blocks
        }
        return source, constants

    def bind(self, **blocks: object) -> QKernel[_P, _R_co]:
        """Bind encoded blocks into source without compiling the algorithm.

        Args:
            **blocks (object): BlockEncoding values keyed by source parameter.

        Returns:
            QKernel[_P, _R_co]: Source declaration with closed block calls.

        Raises:
            TypeError: If a binding is not a BlockEncoding or does not name a
                block parameter.
            ValueError: If a block's signal or system width is incompatible.
            SourceUnavailableError: If linked declarations conflict.
        """

        return _bind_block_encodings(self, blocks)

    def inspect(self) -> str:
        """Compile and render the declaration's QCore function.

        Returns:
            str: Stable textual QCore inspection.

        Raises:
            QamomileError: If source compilation fails.
        """

        return self.module().function(self.function_name).inspect()

    def __call__(self, *_args: _P.args, **_kwargs: _P.kwargs) -> _R_co:
        """Reject host invocation of a semantic source declaration.

        Args:
            *_args (_P.args): Positional values supplied by host Python.
            **_kwargs (_P.kwargs): Keyword values supplied by host Python.

        Returns:
            _R_co: This method never returns.

        Raises:
            TypeError: Always, because qkernels execute only in Qamomile source.
        """

        raise TypeError(
            "Qamomile QKernel values are source declarations, not Python "
            "callables. Call them inside another Qamomile function or create a "
            "Program with .specialize(...)."
        )


def _bind_block_encodings(
    kernel: QKernel[_P, _R_co],
    blocks: Mapping[str, object],
) -> QKernel[_P, _R_co]:
    """Close block arguments at a source call site before native compilation.

    Args:
        kernel (QKernel[_P, _R_co]): Algorithm source declaration.
        blocks (Mapping[str, object]): Encoded blocks keyed by parameter name.

    Returns:
        QKernel[_P, _R_co]: A source-only declaration for the bound algorithm.

    Raises:
        TypeError: If a name is unknown or a value is not a block encoding.
        ValueError: If a block's declared widths do not match its descriptor.
        SourceUnavailableError: If the source declaration cannot be located or
            linked without conflicts.
    """

    from .stdlib.block_encoding import BlockEncoding

    tree = ast.parse(kernel.source_text)
    declarations = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == kernel.function_name
    ]
    if len(declarations) != 1:
        raise SourceUnavailableError(
            Diagnostic("Q5105", f"cannot locate declaration {kernel.function_name!r}")
        )
    declaration = declarations[0]
    if not blocks:
        raise TypeError("bind requires at least one BlockEncoding")
    if declaration.args.vararg or declaration.args.kwarg:
        raise TypeError("block binding does not support variadic parameters")
    positional = [*declaration.args.posonlyargs, *declaration.args.args]
    positions = {
        argument.arg: argument
        for argument in (*positional, *declaration.args.kwonlyargs)
    }
    checked: dict[str, BlockEncoding] = {}
    for name, block in blocks.items():
        argument = positions.get(name)
        if argument is None:
            raise TypeError(f"unknown block parameter {name!r}")
        if not isinstance(block, BlockEncoding):
            raise TypeError(f"{name!r} must be a BlockEncoding")
        checked[name] = block
        annotation = argument.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError as error:
                raise TypeError(
                    f"{name!r} has an invalid BlockEncoding annotation"
                ) from error
        if not (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, (ast.Name, ast.Attribute))
            and (
                annotation.value.id
                if isinstance(annotation.value, ast.Name)
                else annotation.value.attr
            )
            == "BlockEncoding"
        ):
            raise TypeError(f"{name!r} is not declared as BlockEncoding")
        widths: list[ast.expr] = (
            annotation.slice.elts
            if isinstance(annotation.slice, ast.Tuple)
            else [annotation.slice]
        )
        if len(widths) != 2:
            raise TypeError("BlockEncoding annotation requires two integer widths")
        signal_node, system_node = widths
        if (
            not isinstance(signal_node, ast.Constant)
            or not isinstance(system_node, ast.Constant)
            or isinstance(signal_node.value, bool)
            or isinstance(system_node.value, bool)
            or not isinstance(signal_node.value, int)
            or not isinstance(system_node.value, int)
        ):
            raise TypeError("BlockEncoding annotation requires two integer widths")
        if (signal_node.value, system_node.value) != (
            block.signal_width,
            block.system_width,
        ):
            raise ValueError(
                f"{name!r} requires BlockEncoding[{signal_node.value}, "
                f"{system_node.value}]"
            )

    wrapper = copy.deepcopy(declaration)
    separator: str = "\n"
    digest = hashlib.sha256(
        (
            kernel.source_text
            + "\n"
            + separator.join(
                f"{name}:{checked[name].unitary.source_text}"
                for name in sorted(checked)
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    wrapper.name = f"__bound_{kernel.function_name}_{digest}"
    original_defaults = (
        dict(
            zip(
                (
                    argument.arg
                    for argument in positional[-len(declaration.args.defaults) :]
                ),
                declaration.args.defaults,
                strict=True,
            )
        )
        if declaration.args.defaults
        else {}
    )
    wrapper.args.posonlyargs = [
        argument for argument in wrapper.args.posonlyargs if argument.arg not in checked
    ]
    wrapper.args.args = [
        argument for argument in wrapper.args.args if argument.arg not in checked
    ]
    retained_positional = [*wrapper.args.posonlyargs, *wrapper.args.args]
    wrapper.args.defaults = [
        original_defaults[argument.arg]
        for argument in retained_positional
        if argument.arg in original_defaults
    ]
    retained_keywords = [
        (argument, default)
        for argument, default in zip(
            wrapper.args.kwonlyargs, wrapper.args.kw_defaults, strict=True
        )
        if argument.arg not in checked
    ]
    wrapper.args.kwonlyargs = [argument for argument, _ in retained_keywords]
    wrapper.args.kw_defaults = [default for _, default in retained_keywords]
    wrapper.decorator_list = [
        ast.Attribute(
            value=ast.Name(id="qm", ctx=ast.Load()), attr="qkernel", ctx=ast.Load()
        )
    ]
    wrapper.body = [
        ast.Return(
            value=ast.Call(
                func=ast.Name(id=kernel.function_name, ctx=ast.Load()),
                args=[
                    ast.Name(
                        id=(
                            checked[argument.arg].unitary.function_name
                            if argument.arg in checked
                            else argument.arg
                        ),
                        ctx=ast.Load(),
                    )
                    for argument in positional
                ],
                keywords=[
                    ast.keyword(
                        arg=argument.arg,
                        value=ast.Name(
                            id=(
                                checked[argument.arg].unitary.function_name
                                if argument.arg in checked
                                else argument.arg
                            ),
                            ctx=ast.Load(),
                        ),
                    )
                    for argument in declaration.args.kwonlyargs
                ],
            )
        )
    ]
    ast.fix_missing_locations(wrapper)
    wrapper_source = "import qamomile as qm\n\n" + ast.unparse(wrapper) + "\n"
    linked = _link_source_modules(
        wrapper_source,
        (
            kernel.source_text,
            *(checked[name].unitary.source_text for name in sorted(checked)),
        ),
    )
    origins = dict(kernel.origins)
    for name in sorted(checked):
        origins.update(checked[name].unitary.origins)
    return QKernel(
        kernel.module_name,
        wrapper.name,
        linked,
        kernel.display_name,
        tuple(sorted(origins.items())),
        kernel.retain_docstrings,
    )


@overload
def qkernel(
    value: Callable[_P, _R_co],
    /,
    *,
    kind: str | None = None,
    retain_docstrings: bool = True,
) -> QKernel[_P, _R_co]:
    """Wrap one function as a Qamomile quantum kernel.

    Args:
        value (Callable[_P, _R_co]): Python function whose source declares the
            kernel body.
        kind (str | None): Optional explicit kernel kind.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.

    Returns:
        QKernel[_P, _R_co]: Source-backed quantum-kernel declaration.
    """

    ...


@overload
def qkernel(
    value: None = None,
    /,
    *,
    kind: str | None = None,
    retain_docstrings: bool = True,
) -> Callable[[Callable[_P, _R_co]], QKernel[_P, _R_co]]:
    """Return a decorator that declares a Qamomile quantum kernel.

    Args:
        value (None): Sentinel selecting decorator usage.
        kind (str | None): Optional explicit kernel kind.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.

    Returns:
        Callable[[Callable[_P, _R_co]], QKernel[_P, _R_co]]: Decorator for one
        Python source function.
    """

    ...


def qkernel(
    value: Callable[_P, _R_co] | None = None,
    /,
    *,
    kind: str | None = None,
    retain_docstrings: bool = True,
) -> QKernel[_P, _R_co] | Callable[[Callable[_P, _R_co]], QKernel[_P, _R_co]]:
    """Mark one Python definition as a Qamomile quantum kernel.

    Visible bodies normally omit ``kind`` because QCore derives their
    denotation. A bodyless external declaration must state its trusted boundary
    explicitly, for example ``kind="unitary"``.

    Every compiled function carries an ``origin`` intent row naming the kernel
    that declared it, and a ``doc`` row holding its docstring unless
    ``retain_docstrings=False``. Intent is untrusted provenance: it never
    changes meaning or the semantic digest, but it enters persisted bytes.

    Args:
        value (Callable[_P, _R_co] | None): Python definition whose source
            should be captured.
        kind (str | None): Optional trusted callable kind for a bodyless
            declaration.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.

    Returns:
        QKernel[_P, _R_co] | Callable: A source-only kernel handle, or a
        configured decorator.

    Raises:
        ValueError: If ``kind`` is not a recognized callable kind.
    """

    if kind is not None:
        try:
            CallableKind(kind)
        except ValueError as error:
            raise ValueError(f"unknown Qamomile callable kind {kind!r}") from error
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    lexical_scope = dict(caller.f_locals) if caller is not None else {}
    del frame
    return _decorator(
        value,
        lexical_scope=lexical_scope,
        retain_docstrings=bool(retain_docstrings),
    )


@overload
def composite_gate(
    value: QKernel[_P, _R_co] | Callable[_P, _R_co],
    /,
    *,
    name: str | None = None,
) -> QKernel[_P, _R_co]: ...


@overload
def composite_gate(
    value: None = None,
    /,
    *,
    name: str | None = None,
) -> Callable[[QKernel[_P, _R_co] | Callable[_P, _R_co]], QKernel[_P, _R_co]]: ...


def composite_gate(
    value: QKernel[_P, _R_co] | Callable[_P, _R_co] | None = None,
    /,
    *,
    name: str | None = None,
) -> (
    QKernel[_P, _R_co]
    | Callable[[QKernel[_P, _R_co] | Callable[_P, _R_co]], QKernel[_P, _R_co]]
):
    """Mark one qkernel call for named target realization.

    The callable body remains the sole semantic definition used for
    verification, execution, and resource analysis. Targets that advertise
    named-call support may preserve an invocation as one labeled composite
    instruction with the verified body attached as its decomposition. Other
    targets inline the same body normally.

    Args:
        value: Existing qkernel or Python definition to decorate immediately.
        name: Optional nonempty display label. The function name is used when
            omitted.

    Returns:
        A source-only qkernel carrying a portable named-call identity, or a
        configured decorator.

    Raises:
        TypeError: If the decorated value is not a qkernel or Python function.
        ValueError: If ``name`` is empty or contains a control character.
        SourceUnavailableError: If a Python definition has no recoverable
            declarative source.
    """

    if name is not None and (
        not name or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("composite gate name must be nonempty printable text")
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    lexical_scope = dict(caller.f_locals) if caller is not None else {}
    del frame

    def decorate(
        candidate: QKernel[_P, _R_co] | Callable[_P, _R_co],
    ) -> QKernel[_P, _R_co]:
        """Return one marker-renamed source declaration.

        Args:
            candidate: Existing qkernel or Python definition.

        Returns:
            Equivalent qkernel with a portable composite display identity.

        Raises:
            TypeError: If ``candidate`` is not supported.
            SourceUnavailableError: If source capture or renaming fails.
        """

        if isinstance(candidate, QKernel):
            kernel = candidate
        elif callable(candidate):
            kernel = _qkernel_ref(candidate, lexical_scope=lexical_scope)
        else:
            raise TypeError("composite_gate expects a QKernel or Python function")
        label = name or kernel.function_name
        marker = (
            f"{_COMPOSITE_PREFIX}{label.encode('utf-8').hex()}__{kernel.function_name}"
        )
        return QKernel(
            kernel.module_name,
            marker,
            _rename_declaration(kernel.source_text, kernel.function_name, marker),
            kernel.display_name,
            tuple(
                (marker if name == kernel.function_name else name, origin)
                for name, origin in kernel.origins
            ),
            kernel.retain_docstrings,
        )

    if value is None:
        return decorate
    return decorate(value)


def _decorator(
    value: Callable[_P, _R_co] | None,
    *,
    lexical_scope: Mapping[str, object],
    retain_docstrings: bool = True,
) -> QKernel[_P, _R_co] | Callable[[Callable[_P, _R_co]], QKernel[_P, _R_co]]:
    """Create one configured source-only decorator.

    Args:
        value (Callable[_P, _R_co] | None): Optional definition to decorate
            immediately.
        lexical_scope (Mapping[str, object]): Immutable capture candidates from
            the decorator's Python scope.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.

    Returns:
        QKernel[_P, _R_co] | Callable: Captured qkernel or configured decorator.

    Raises:
        SourceUnavailableError: If source for the definition is unavailable.
    """

    def decorate(
        python_function: Callable[_P, _R_co],
    ) -> QKernel[_P, _R_co]:
        """Capture source identity without executing the body.

        Args:
            python_function (Callable[_P, _R_co]): Definition to capture.

        Returns:
            QKernel[_P, _R_co]: Source-only qkernel handle.

        Raises:
            SourceUnavailableError: If source for the definition is unavailable.
        """

        return _qkernel_ref(
            python_function,
            lexical_scope=lexical_scope,
            retain_docstrings=retain_docstrings,
        )

    if value is None:
        return decorate
    return decorate(value)


def _qkernel_ref(
    function_value: Callable[_P, _R_co],
    *,
    lexical_scope: Mapping[str, object],
    retain_docstrings: bool = True,
) -> QKernel[_P, _R_co]:
    """Construct a source handle from a Python function definition.

    Args:
        function_value (Callable[_P, _R_co]): Python definition to isolate.
        lexical_scope (Mapping[str, object]): Capture candidates visible where
            the decorator was evaluated.
        retain_docstrings (bool): Whether docstrings persist as ``doc`` intent.

    Returns:
        QKernel[_P, _R_co]: Immutable source declaration handle.

    Raises:
        SourceUnavailableError: If source cannot be recovered or isolated.
        TypeError: If a referenced capture is not portable.
    """

    module_name = function_value.__module__.replace("<", "_").replace(">", "_")
    display_name = f"{function_value.__name__}.py"
    source_file = inspect.getsourcefile(function_value)
    is_top_level = "<locals>" not in function_value.__qualname__
    is_file_backed = bool(source_file and is_top_level and Path(source_file).is_file())
    if is_file_backed:
        assert source_file is not None
        source_text = _isolated_file_declaration(
            Path(source_file).read_text(encoding="utf-8"),
            function_value.__name__,
        )
        display_name = Path(source_file).name
    else:
        try:
            declaration = textwrap.dedent(inspect.getsource(function_value))
        except (OSError, TypeError) as error:
            raise SourceUnavailableError(
                Diagnostic(
                    "Q5101",
                    f"source is unavailable for {function_value.__qualname__!r}",
                    note="use qm.build_source(...) or a notebook source provider",
                )
            ) from error
        source_text = "import qamomile as qm\n\n" + declaration
    function_name = function_value.__name__
    annotation_names = _annotation_names(source_text, function_name)
    if is_file_backed:
        record_source, record_names = None, frozenset()
    else:
        record_source, record_names = _capture_annotation_records(
            annotation_names,
            lexical_scope,
        )
    dependencies, captures = _function_captures(
        function_value,
        lexical_scope=lexical_scope,
        annotation_names=annotation_names,
        record_names=record_names,
    )
    if dependencies or captures:
        source_text = _close_lexical_captures(
            source_text,
            function_name,
            captures,
        )
        source_text = _isolated_file_declaration(source_text, function_name)
    if record_source is not None:
        source_text = _link_source_modules(source_text, (record_source,))
    if not is_top_level:
        hygienic_name = (
            f"{function_name}__{hashlib.sha256(source_text.encode()).hexdigest()[:12]}"
        )
        source_text = _rename_declaration(source_text, function_name, hygienic_name)
        function_name = hygienic_name
    origins: dict[str, str] = {}
    if dependencies:
        source_text = _link_source_modules(
            source_text,
            tuple(
                dict.fromkeys(dependency.source_text for _, dependency in dependencies)
            ),
        )
        for _, dependency in dependencies:
            origins.update(dependency.origins)
    origins[function_name] = f"{module_name}.{function_name}"
    return QKernel(
        module_name,
        function_name,
        source_text,
        display_name,
        tuple(sorted(origins.items())),
        retain_docstrings,
    )


def _function_captures(
    function_value: Callable[..., object],
    *,
    lexical_scope: Mapping[str, object],
    annotation_names: frozenset[str],
    record_names: frozenset[str],
) -> tuple[tuple[tuple[str, QKernel], ...], dict[str, ast.expr]]:
    """Collect portable immutable expressions for closure conversion."""

    try:
        closure = inspect.getclosurevars(function_value)
    except TypeError:
        return (), {}
    dependencies: dict[str, QKernel] = {}
    expressions: dict[str, ast.expr] = {}
    captured = (*closure.globals.items(), *closure.nonlocals.items())
    nonlocal_names = closure.nonlocals.keys()
    for name, candidate in captured:
        if name in record_names:
            continue
        captured_expression = _capture_ast(candidate, name)
        if captured_expression is not None:
            expression, captured_dependencies = captured_expression
            expressions[name] = expression
            dependencies.update(captured_dependencies)
        elif name in nonlocal_names:
            raise SourceUnavailableError(
                Diagnostic(
                    "Q5104",
                    f"captured value {name!r} is not a portable immutable literal",
                    note=(
                        "pass it as Const/Param or capture QKernel values and "
                        "immutable scalar/product literals"
                    ),
                )
            )
    for name in sorted(annotation_names):
        if name in expressions or name not in lexical_scope:
            continue
        candidate = lexical_scope[name]
        captured_expression = _capture_ast(candidate, name)
        if captured_expression is not None:
            expression, captured_dependencies = captured_expression
            expressions[name] = expression
            dependencies.update(captured_dependencies)
    return tuple(sorted(dependencies.items())), expressions


def _capture_annotation_records(
    annotation_names: frozenset[str],
    lexical_scope: Mapping[str, object],
) -> tuple[str | None, frozenset[str]]:
    """Close notebook-local NamedTuple annotations into declarative source."""

    pending = sorted(annotation_names, reverse=True)
    classes: dict[str, ast.ClassDef] = {}
    dependencies: dict[str, set[str]] = {}
    constants: dict[str, ast.Assign] = {}
    while pending:
        name = pending.pop()
        if name in classes or name not in lexical_scope:
            continue
        declaration = _named_tuple_declaration(name, lexical_scope[name])
        if declaration is None:
            continue
        class_node, referenced_names = declaration
        classes[name] = class_node
        dependencies[name] = set()
        for referenced_name in sorted(referenced_names):
            if referenced_name not in lexical_scope:
                continue
            candidate = lexical_scope.get(referenced_name)
            if _is_typed_named_tuple(candidate):
                dependencies[name].add(referenced_name)
                if referenced_name not in classes:
                    pending.append(referenced_name)
                continue
            literal = _literal_ast(candidate)
            if literal is not None and referenced_name not in constants:
                constants[referenced_name] = ast.Assign(
                    targets=[ast.Name(id=referenced_name, ctx=ast.Store())],
                    value=literal,
                )
    if not classes:
        return None, frozenset()

    remaining = set(classes)
    ordered: list[ast.ClassDef] = []
    while remaining:
        ready = sorted(
            name for name in remaining if dependencies[name].isdisjoint(remaining)
        )
        if not ready:
            cycle = ", ".join(sorted(remaining))
            raise SourceUnavailableError(
                Diagnostic(
                    "Q5104",
                    f"captured record annotations form a cycle: {cycle}",
                    note="use qm.build_source(...) for mutually recursive declarations",
                )
            )
        for name in ready:
            ordered.append(classes[name])
            remaining.remove(name)

    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name="NamedTuple")],
                level=0,
            ),
            ast.Import(names=[ast.alias(name="qamomile", asname="qm")]),
            *[constants[name] for name in sorted(constants)],
            *ordered,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n", frozenset(classes)


def _is_typed_named_tuple(value: object) -> bool:
    """Return whether a host class exposes one complete typed tuple record."""

    if not inspect.isclass(value):
        return False
    try:
        if not issubclass(value, tuple):
            return False
    except TypeError:
        return False
    fields = getattr(value, "_fields", None)
    annotations = getattr(value, "__annotations__", None)
    defaults = getattr(value, "_field_defaults", None)
    return (
        isinstance(fields, tuple)
        and bool(fields)
        and all(isinstance(field, str) for field in fields)
        and isinstance(annotations, Mapping)
        and tuple(annotations) == fields
        and isinstance(defaults, Mapping)
    )


def _named_tuple_declaration(
    name: str,
    value: object,
) -> tuple[ast.ClassDef, frozenset[str]] | None:
    """Reconstruct syntax for one typed runtime NamedTuple without trusting it."""

    if not _is_typed_named_tuple(value):
        return None
    record_type = cast(type[object], value)
    try:
        runtime_source = textwrap.dedent(inspect.getsource(record_type))
    except (OSError, TypeError):
        runtime_source = ""
    if runtime_source:
        tree = ast.parse(runtime_source)
        matches = [
            statement
            for statement in tree.body
            if isinstance(statement, ast.ClassDef)
            and statement.name == record_type.__name__
        ]
        if len(matches) == 1:
            declaration = copy.deepcopy(matches[0])
            declaration.name = name
            source_referenced_names = frozenset(
                node.id
                for node in ast.walk(declaration)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
            return declaration, source_referenced_names
    fields = getattr(record_type, "_fields")
    annotations = getattr(record_type, "__annotations__")
    defaults = getattr(record_type, "_field_defaults")
    body: list[ast.stmt] = []
    referenced_names: set[str] = set()
    for field_name in fields:
        raw_annotation = annotations[field_name]
        annotation_source = getattr(raw_annotation, "__forward_arg__", None)
        if not isinstance(annotation_source, str):
            if isinstance(raw_annotation, str):
                annotation_source = raw_annotation
            else:
                raise SourceUnavailableError(
                    Diagnostic(
                        "Q5104",
                        f"captured record {name!r} lacks source-form annotations",
                        note=(
                            "use `from __future__ import annotations` in notebooks "
                            "or pass the declaration to qm.build_source(...)"
                        ),
                    )
                )
        try:
            annotation = ast.parse(annotation_source, mode="eval").body
        except SyntaxError as error:
            raise SourceUnavailableError(
                Diagnostic(
                    "Q5104",
                    f"captured record {name!r} has an invalid field annotation",
                    note="pass the declaration to qm.build_source(...)",
                )
            ) from error
        referenced_names.update(
            node.id
            for node in ast.walk(annotation)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        default = None
        if field_name in defaults:
            default = _literal_ast(defaults[field_name])
            if default is None:
                raise SourceUnavailableError(
                    Diagnostic(
                        "Q5104",
                        f"captured record {name!r} has a nonportable default",
                        note="record defaults must be immutable portable literals",
                    )
                )
        body.append(
            ast.AnnAssign(
                target=ast.Name(id=field_name, ctx=ast.Store()),
                annotation=annotation,
                value=default,
                simple=1,
            )
        )
    return (
        ast.ClassDef(
            name=name,
            bases=[ast.Name(id="NamedTuple", ctx=ast.Load())],
            keywords=[],
            body=body,
            decorator_list=[],
        ),
        frozenset(referenced_names),
    )


def _annotation_names(source: str, function_name: str) -> frozenset[str]:
    """Return names read by one declaration's parameter and result annotations."""

    tree = ast.parse(source)
    matches = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == function_name
    ]
    if len(matches) != 1:
        raise SourceUnavailableError(
            Diagnostic("Q5105", f"cannot locate declaration {function_name!r}")
        )
    declaration = matches[0]
    annotations = [
        argument.annotation
        for argument in (
            *declaration.args.posonlyargs,
            *declaration.args.args,
            *declaration.args.kwonlyargs,
        )
        if argument.annotation is not None
    ]
    if (
        declaration.args.vararg is not None
        and declaration.args.vararg.annotation is not None
    ):
        annotations.append(declaration.args.vararg.annotation)
    if (
        declaration.args.kwarg is not None
        and declaration.args.kwarg.annotation is not None
    ):
        annotations.append(declaration.args.kwarg.annotation)
    if declaration.returns is not None:
        annotations.append(declaration.returns)
    return frozenset(
        node.id
        for annotation in annotations
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _isolated_file_declaration(source: str, function_name: str) -> str:
    """Retain one function and the top-level types it references."""

    tree = ast.parse(source)
    declarations = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == function_name
    ]
    if len(declarations) != 1:
        raise SourceUnavailableError(
            Diagnostic(
                "Q5105",
                f"cannot uniquely locate declaration {function_name!r}",
            )
        )
    declaration = declarations[0]
    loaded_names = {
        node.id
        for node in ast.walk(declaration)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    classes = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, ast.ClassDef)
    }
    constants: dict[str, ast.Assign | ast.AnnAssign] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            constants[statement.targets[0].id] = statement
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            constants[statement.target.id] = statement
    selected_classes: dict[str, ast.ClassDef] = {}
    selected_constants: dict[str, ast.Assign | ast.AnnAssign] = {}
    pending = set(loaded_names) & (classes.keys() | constants.keys())
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in selected_classes or name in selected_constants:
            continue
        declaration_node: ast.AST
        if name in classes:
            selected_classes[name] = classes[name]
            declaration_node = classes[name]
        else:
            selected_constants[name] = constants[name]
            declaration_node = constants[name]
        referenced = {
            node.id
            for node in ast.walk(declaration_node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        loaded_names.update(referenced)
        available = classes.keys() | constants.keys()
        selected = selected_classes.keys() | selected_constants.keys()
        pending.update(referenced & (available - selected))
    imports: list[ast.Import | ast.ImportFrom] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            aliases = [
                alias
                for alias in statement.names
                if (alias.asname or alias.name.split(".", 1)[0]) in loaded_names
            ]
            if aliases:
                imports.append(ast.Import(names=aliases))
        elif isinstance(statement, ast.ImportFrom):
            aliases = [
                alias
                for alias in statement.names
                if alias.name == "annotations"
                and statement.module == "__future__"
                or (alias.asname or alias.name) in loaded_names
            ]
            if aliases:
                imports.append(
                    ast.ImportFrom(
                        module=statement.module,
                        names=aliases,
                        level=statement.level,
                    )
                )
    retained_declarations = [
        statement
        for statement in tree.body
        if (isinstance(statement, ast.ClassDef) and statement.name in selected_classes)
        or (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in selected_constants
        )
        or (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in selected_constants
        )
    ]
    isolated = ast.Module(
        body=[*imports, *retained_declarations, declaration],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    return ast.unparse(isolated) + "\n"


def _rename_declaration(source: str, old_name: str, new_name: str) -> str:
    """Assign one generated declaration a deterministic hygienic symbol."""

    tree = ast.parse(source)
    matches = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == old_name
    ]
    if len(matches) != 1:
        raise SourceUnavailableError(
            Diagnostic("Q5105", f"cannot uniquely rename declaration {old_name!r}")
        )
    matches[0].name = new_name
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


class _CaptureRewriter(ast.NodeTransformer):
    """Replace closed-over names by linked function names or literal syntax."""

    def __init__(
        self,
        expressions: Mapping[str, ast.expr],
    ) -> None:
        """Store the finite lexical environment being closed."""

        self.expressions = expressions

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Rewrite only reads; local bindings retain ordinary Python scope."""

        if not isinstance(node.ctx, ast.Load):
            return node
        expression = self.expressions.get(node.id)
        if expression is not None:
            return ast.copy_location(copy.deepcopy(expression), node)
        return node


def _close_lexical_captures(
    source: str,
    function_name: str,
    expressions: Mapping[str, ast.expr],
) -> str:
    """Close one declaration without executing it or changing persistent IR."""

    tree = ast.parse(source)
    rewritten = False
    transformer = _CaptureRewriter(expressions)
    body: list[ast.stmt] = []
    for statement in tree.body:
        if (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == function_name
        ):
            statement = transformer.visit(statement)
            rewritten = True
        body.append(statement)
    if not rewritten:
        raise SourceUnavailableError(
            Diagnostic("Q5105", f"cannot locate declaration {function_name!r}")
        )
    tree.body = body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _literal_ast(value: object) -> ast.expr | None:
    """Return safe syntax for one portable immutable capture."""

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return ast.Constant(value=value)
    if isinstance(value, tuple):
        elements = tuple(_literal_ast(item) for item in value)
        if any(item is None for item in elements):
            return None
        return ast.Tuple(
            elts=[item for item in elements if item is not None],
            ctx=ast.Load(),
        )
    return None


def _capture_ast(
    value: object,
    name: str,
) -> tuple[ast.expr, dict[str, QKernel]] | None:
    """Close one immutable host value into source syntax and dependencies."""

    if isinstance(value, QKernel):
        return ast.Name(id=value.function_name, ctx=ast.Load()), {name: value}
    literal = _literal_ast(value)
    if literal is not None:
        return literal, {}
    if not isinstance(value, tuple):
        return None
    elements: list[ast.expr] = []
    dependencies: dict[str, QKernel] = {}
    for index, item in enumerate(value):
        captured = _capture_ast(item, f"{name}_{index}")
        if captured is None:
            return None
        expression, nested_dependencies = captured
        elements.append(expression)
        dependencies.update(nested_dependencies)
    return ast.Tuple(elts=elements, ctx=ast.Load()), dependencies


def _link_source_modules(primary: str, dependencies: tuple[str, ...]) -> str:
    """Close imported QKernel modules into one deterministic source unit."""

    parsed = [ast.parse(source) for source in (*dependencies, primary)]
    imports: dict[str, ast.stmt] = {}
    constants: list[ast.stmt] = []
    classes: dict[str, ast.ClassDef] = {}
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for module in parsed:
        for statement in module.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                key = ast.dump(statement, include_attributes=False)
                imports[key] = statement
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                constants.append(statement)
            elif isinstance(statement, ast.ClassDef):
                existing_class = classes.get(statement.name)
                if existing_class is not None and ast.dump(
                    existing_class, include_attributes=False
                ) != ast.dump(statement, include_attributes=False):
                    raise SourceUnavailableError(
                        Diagnostic(
                            "Q5103",
                            f"linked Qamomile modules define conflicting class "
                            f"{statement.name!r}",
                        )
                    )
                classes[statement.name] = statement
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                existing = functions.get(statement.name)
                if existing is not None and ast.dump(
                    existing, include_attributes=False
                ) != ast.dump(statement, include_attributes=False):
                    raise SourceUnavailableError(
                        Diagnostic(
                            "Q5103",
                            f"linked Qamomile modules define conflicting function "
                            f"{statement.name!r}",
                        )
                    )
                functions[statement.name] = statement
    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    ordinary_imports = [
        statement
        for statement in imports.values()
        if not (
            isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
        )
    ]
    linked = ast.Module(
        body=[
            future,
            *ordinary_imports,
            *constants,
            *classes.values(),
            *functions.values(),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(linked)
    return ast.unparse(linked) + "\n"


def specialize(source: object, /, **bindings: object) -> Program:
    """Specialize a program family with named static keyword bindings.

    Args:
        source (object): QKernel, ProgramFamily, or already closed Program.
        **bindings (object): Const values, or BlockEncoding values for a
            source qkernel, keyed by parameter name.

    Returns:
        Program: Closed verified program with Const values materialized.

    Raises:
        TypeError: If ``source`` is unsupported or a closed Program receives
            bindings.
        SpecializationError: If a binding is missing, unknown, or ill-typed.
    """

    return _close_program(source, bindings)


def _close_program(
    source: object,
    bindings: Mapping[str, object] | None = None,
) -> Program:
    """Normalize one function-oriented public value to a closed Program.

    Args:
        source (object): QKernel, ProgramFamily, or already closed Program.
        bindings (Mapping[str, object] | None): Const values keyed by source
            parameter name.

    Returns:
        Program: Closed verified program.

    Raises:
        TypeError: If ``source`` is unsupported or a closed Program receives
            bindings.
        SpecializationError: If specialization bindings are invalid.
    """

    if isinstance(source, Program):
        if bindings:
            raise TypeError("a closed Program does not accept Const bindings")
        return source
    if isinstance(source, QKernel):
        return source.specialize(**dict(bindings or {}))
    if isinstance(source, ProgramFamily):
        return source.specialize(**dict(bindings or {}))
    raise TypeError("expected a QKernel, ProgramFamily, or Program")
