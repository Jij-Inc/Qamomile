"""Persistent region-typed linear SSA for Qamomile.

QCore is the semantic ownership boundary of the compiler.  Python AST and HIR
are frontend details; backend SDK objects never appear in these data classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .source import ProgramFamily

from ._native import compiler as native
from .effects import (
    CallableContract,
    CallableKind,
    Capability,
    Effect,
    Requirement,
)
from .errors import (
    Diagnostic,
    SpecializationError,
    VerificationError,
)
from .operations import (
    OPERATION_EFFECTS,
    STRUCTURED_OPCODES,
)
from .types import (
    TypeSpec,
    mapping_components,
)


@dataclass(frozen=True, slots=True)
class FrozenMapping:
    """A deterministic immutable mapping usable inside serialized attributes."""

    entries: tuple[tuple[str, PortableValue], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FrozenMapping:
        """Freeze a string-keyed mapping in lexical key order."""

        return cls(
            tuple(
                (key, freeze_value(item))
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            )
        )


PortableValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | bytes
    | tuple["PortableValue", ...]
    | FrozenMapping
)


def freeze_value(value: object) -> PortableValue:
    """Convert a Python value to QCore's closed portable literal domain.

    Args:
        value: Value to canonicalize.

    Returns:
        An immutable portable value.

    Raises:
        TypeError: If the value would require Python-specific serialization.
    """

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    from .observable import Hamiltonian

    if isinstance(value, Hamiltonian):
        return freeze_value(value.to_portable())
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, tuple):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return FrozenMapping.from_mapping(value)
    raise TypeError(
        f"{type(value).__name__} is not a portable Qamomile value; "
        "use literals, tuples, or string-keyed immutable records."
    )


def _freeze_typed_binding(
    value: object,
    type_spec: TypeSpec | None,
) -> PortableValue:
    """Freeze one host binding using source-only aggregate syntax.

    Args:
        value: Host value supplied for a typed parameter.
        type_spec: Declared semantic parameter type, when the name is known.

    Returns:
        PortableValue: Canonical structural value sent to Rust.

    Raises:
        TypeError: If a mapping parameter is not supplied as a mapping.
    """

    mapping = None if type_spec is None else mapping_components(type_spec)
    if mapping is None:
        return freeze_value(value)
    if not isinstance(value, Mapping):
        raise TypeError("Map bindings must be Python mappings")
    return tuple(
        FrozenMapping.from_mapping({"key": key, "value": item})
        for key, item in value.items()
    )


def thaw_value(value: PortableValue) -> object:
    """Convert an immutable portable value to ordinary Python containers."""

    if isinstance(value, FrozenMapping):
        return {key: thaw_value(item) for key, item in value.entries}
    if isinstance(value, tuple):
        return tuple(thaw_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A normalized source location retained across compilation hosts."""

    uri: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    @classmethod
    def synthetic(cls, label: str = "compiler") -> SourceSpan:
        """Return a stable location for compiler-generated IR."""

        return cls(f"qmi://{label}", 1, 0, 1, 0)


@dataclass(frozen=True, slots=True)
class RegionRef:
    """Ownership identity preserved across SSA state versions."""

    root: int
    projections: tuple[str, ...] = ()

    def project(self, component: str) -> RegionRef:
        """Return a logical aggregate projection."""

        return RegionRef(self.root, self.projections + (component,))


@dataclass(frozen=True, slots=True)
class Value:
    """One typed SSA definition."""

    id: int
    type: TypeSpec
    hint: str = ""
    region: RegionRef | None = None

    def label(self) -> str:
        """Return the stable textual SSA label."""

        return f"%{self.id}"


@dataclass(frozen=True, slots=True)
class Terminator:
    """The explicit control transfer at the end of a QCore block."""

    kind: str
    operands: tuple[int, ...] = ()
    attributes: tuple[tuple[str, PortableValue], ...] = ()
    span: SourceSpan = field(default_factory=SourceSpan.synthetic)

    def attribute(self, name: str, default: object = None) -> object:
        """Return one terminator attribute by name."""

        return dict(self.attributes).get(name, default)


@dataclass(frozen=True, slots=True)
class Block:
    """A sequence of SSA operations ending in one explicit terminator."""

    arguments: tuple[Value, ...]
    operations: tuple[Operation, ...]
    terminator: Terminator


@dataclass(frozen=True, slots=True)
class Region:
    """A first-class structured control-flow region."""

    blocks: tuple[Block, ...]

    @property
    def entry(self) -> Block:
        """Return the single entry block used by structured QCore operations."""

        if not self.blocks:
            raise VerificationError("Q1001: a QCore region has no entry block")
        return self.blocks[0]


@dataclass(frozen=True, slots=True)
class Operation:
    """One immutable QCore operation with optional nested regions.

    Attributes:
        intent (tuple[tuple[str, PortableValue], ...]): Untrusted provenance
            or tolerance rows. They are persisted, printed, and erased from the
            semantic digest; nothing in the compiler reads them.
    """

    opcode: str
    operands: tuple[int, ...]
    results: tuple[Value, ...]
    attributes: tuple[tuple[str, PortableValue], ...] = ()
    regions: tuple[Region, ...] = ()
    span: SourceSpan = field(default_factory=SourceSpan.synthetic)
    intent: tuple[tuple[str, PortableValue], ...] = ()

    def attribute(self, name: str, default: object = None) -> object:
        """Return one operation attribute by name."""

        return dict(self.attributes).get(name, default)

    @property
    def effects(self) -> frozenset[Effect]:
        """Derive direct effects from the operation's semantic identity."""

        return operation_effects(self)


@dataclass(frozen=True, slots=True)
class Parameter:
    """A named function input and its SSA definition."""

    name: str
    value: Value
    default: PortableValue = None
    has_default: bool = False


@dataclass(frozen=True, slots=True)
class Function:
    """A typed QCore callable definition.

    Attributes:
        intent (tuple[tuple[str, PortableValue], ...]): Untrusted provenance
            or tolerance rows such as ``origin`` and ``doc``. They are
            persisted, printed, and erased from the semantic digest; nothing in
            the compiler reads them.
    """

    symbol: str
    parameters: tuple[Parameter, ...]
    result_type: TypeSpec
    body: Block
    span: SourceSpan = field(default_factory=SourceSpan.synthetic)
    asserted_kind: CallableKind | None = None
    external: bool = False
    contract: CallableContract | None = field(default=None, compare=False, repr=False)
    intent: tuple[tuple[str, PortableValue], ...] = ()

    def _derived_contract(self) -> CallableContract:
        """Return the contract installed by the enclosing verified image."""

        if self.contract is None:
            raise VerificationError(
                f"Q1050: @{self.symbol} has not crossed a QCore module boundary"
            )
        return self.contract

    @property
    def kind(self) -> CallableKind:
        """Return the body-derived denotational kind."""

        return self._derived_contract().kind

    @property
    def effects(self) -> frozenset[Effect]:
        """Return transitive body-derived effects."""

        return self._derived_contract().effects

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Return transformations proved from the callable semantics."""

        return self._derived_contract().capabilities

    @property
    def requirements(self) -> frozenset[Requirement]:
        """Return target requirements derived from the closed call graph."""

        return self._derived_contract().requirements

    def inspect(self) -> str:
        """Return the stable human-readable QCore representation."""

        return _print_function(self)


@dataclass(frozen=True, slots=True)
class Constant:
    """A named portable module constant."""

    name: str
    type: TypeSpec
    value: PortableValue


@dataclass(frozen=True, slots=True)
class Module:
    """Represent an immutable, serializable typed Qamomile source module.

    Attributes:
        module_name (str): Stable semantic source-module name.
        functions (tuple[Function, ...]): Verified declarations in canonical
            order.
        constants (tuple[Constant, ...]): Portable module constants.
    """

    module_name: str
    functions: tuple[Function, ...]
    constants: tuple[Constant, ...] = ()

    def __post_init__(self) -> None:
        """Verify the module and install Rust-derived callable contracts."""

        from .serialization import _functions_with_native_contracts, encode_module

        inspect = getattr(native, "inspect_module", None)
        if not callable(inspect):
            raise VerificationError("Q4002: native module verification is unavailable")
        try:
            header = inspect(encode_module(self))
            functions = _functions_with_native_contracts(self.functions, header)
        except ValueError as error:
            raise VerificationError(str(error)) from error
        object.__setattr__(self, "functions", functions)

    def function(self, name: str) -> Function:
        """Resolve a local or fully qualified function symbol through Rust."""

        resolve = getattr(native, "resolve_module_entrypoint", None)
        if not callable(resolve):
            raise SpecializationError(
                "Q3000: native entrypoint resolution is unavailable"
            )
        try:
            symbol = resolve(self.to_bytes(), name)
        except ValueError as error:
            raise SpecializationError(str(error)) from error
        if type(symbol) is not str:
            raise VerificationError(
                "Q4002: native entrypoint resolution returned a malformed symbol"
            )
        matches = [function for function in self.functions if function.symbol == symbol]
        if len(matches) != 1:
            raise VerificationError(
                "Q4002: native entrypoint resolution returned an unknown symbol"
            )
        return matches[0]

    def __getitem__(self, entrypoint: str) -> ProgramFamily:
        """Select one unspecialized entrypoint as a ProgramFamily."""

        from .source import ProgramFamily

        return ProgramFamily(self, entrypoint)

    def specialize(
        self,
        entrypoint: str,
        /,
        **bindings: object,
    ) -> Program:
        """Specialize one entry point into a closed Program.

        Args:
            entrypoint (str): Local or qualified entrypoint symbol.
            **bindings (object): Values for parameters declared as
                ``Const[T]``.

        Returns:
            Program: Verified program with all Const defaults materialized.

        Raises:
            SpecializationError: If a binding is missing, unknown, or ill-typed.
        """

        provided = dict(bindings)
        if any(not isinstance(name, str) for name in provided):
            raise SpecializationError("Q3003: Const binding names must be strings")
        parameter_types = {
            parameter.name: parameter.value.type
            for parameter in self.function(entrypoint).parameters
        }
        frozen_provided: dict[str, PortableValue] = {}
        for name, value in provided.items():
            try:
                frozen_provided[name] = _freeze_typed_binding(
                    value,
                    parameter_types.get(name),
                )
            except TypeError as error:
                raise SpecializationError(
                    f"Q3005: Const binding {name!r} is not portable"
                ) from error
        from ._compiler.selection import prepare_selection

        return prepare_selection(
            self,
            entrypoint,
            tuple(sorted(frozen_provided.items())),
        )

    def inspect(self) -> str:
        """Return a stable textual rendering of the complete module."""

        return "\n\n".join(function.inspect() for function in self.functions)

    def to_bytes(self) -> bytes:
        """Serialize this Module with the canonical QCore wire schema."""

        from .serialization import encode_module

        return encode_module(self)

    @classmethod
    def from_bytes(cls, payload: bytes) -> Module:
        """Decode and verify a canonical Module.

        Args:
            payload (bytes): Canonical integrity-sealed Module bytes.

        Returns:
            Module: Verified source module.

        Raises:
            SerializationError: If the payload is malformed or noncanonical.
            VerificationError: If native verification rejects the Module.
        """

        from .serialization import decode_module

        return decode_module(payload)

    @property
    def semantic_digest(self) -> str:
        """Return the native SHA-256 identity after erasing debug metadata.

        Returns:
            str: Lowercase hexadecimal semantic module identity.

        Raises:
            VerificationError: If native identity derivation is unavailable or
                violates its result contract.
        """

        return _native_semantic_digest(
            self.to_bytes(),
            "module_semantic_digest",
        )

    def selected_digest(self, entrypoint: str) -> str:
        """Return the semantic identity of one selected program family.

        Args:
            entrypoint (str): Local or fully qualified callable selector.

        Returns:
            str: Lowercase hexadecimal family identity.

        Raises:
            SpecializationError: If the selector is absent or ambiguous.
            VerificationError: If native identity derivation is unavailable or
                violates its result contract.
        """

        return _native_semantic_digest(
            self.to_bytes(),
            "selected_module_semantic_digest",
            entrypoint,
            selection_errors=True,
        )


@dataclass(frozen=True, slots=True)
class Program:
    """Represent a closed entry point and transitive QCore dependency graph.

    Attributes:
        module_name (str): Stable semantic source-module name.
        functions (tuple[Function, ...]): Closed verified dependency graph.
        constants (tuple[Constant, ...]): Portable module constants.
        entrypoint (str): Fully qualified selected callable symbol.
        const_bindings (tuple[tuple[str, PortableValue], ...]): Canonical Const
            values materialized during specialization.
    """

    module_name: str
    functions: tuple[Function, ...]
    constants: tuple[Constant, ...]
    entrypoint: str
    const_bindings: tuple[tuple[str, PortableValue], ...] = ()

    _owner: Any = field(default=None, init=False, repr=False, compare=False)
    _owner_lock: Any = field(
        default_factory=RLock, init=False, repr=False, compare=False
    )

    def _native_program(self) -> Any:
        """Retain the Rust authority for this immutable source projection.

        Returns:
            Any: Native verified Program owner, shared within this facade.

        Raises:
            VerificationError: If verification or the required native API fails.
        """
        from .serialization import encode_program

        constructor = getattr(native, "VerifiedProgramHandle", None)
        if not callable(constructor):
            raise VerificationError(
                "Q4002: native verified Program ownership is unavailable"
            )
        with self._owner_lock:
            if self._owner is None:
                try:
                    owner = constructor(encode_program(self))
                except ValueError as error:
                    raise VerificationError(str(error)) from error
                object.__setattr__(self, "_owner", owner)
            return self._owner

    def __reduce__(self) -> tuple[object, tuple[bytes]]:
        """Persist canonical source while excluding native handles and locks.

        Returns:
            tuple: Portable reconstruction callable and source bytes.
        """
        return (type(self).from_bytes, (self.to_bytes(),))

    def __post_init__(self) -> None:
        """Verify a selected target-neutral Program through Rust."""

        from .serialization import (
            _functions_with_native_contracts,
            encode_specialization_request,
        )

        inspect = getattr(native, "inspect_specialization", None)
        if not callable(inspect):
            raise VerificationError(
                "Q4002: native specialization verification is unavailable"
            )
        try:
            header = inspect(encode_specialization_request(self))
            functions = _functions_with_native_contracts(self.functions, header)
        except ValueError as error:
            raise VerificationError(str(error)) from error
        object.__setattr__(self, "functions", functions)

    @property
    def entry(self) -> Function:
        """Return the closed entrypoint definition."""

        for function in self.functions:
            if function.symbol == self.entrypoint:
                return function
        raise VerificationError("Q1015: program entrypoint is missing")

    def inspect(self) -> str:
        """Return a stable textual rendering of the closed program."""

        bindings = ", ".join(f"{name}={value!r}" for name, value in self.const_bindings)
        header = f"program @{self.entrypoint}"
        if bindings:
            header += f" [{bindings}]"
        return (
            header
            + "\n\n"
            + "\n\n".join(function.inspect() for function in self.functions)
        )

    def to_bytes(self) -> bytes:
        """Export canonical source bytes from its retained verified owner.

        Returns:
            bytes: Integrity-sealed portable QCore; no process-local handles.
        """
        return self._native_program().to_bytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> Program:
        """Decode and verify a canonical closed Program.

        Args:
            payload (bytes): Canonical integrity-sealed Program bytes.

        Returns:
            Program: Verified closed program.

        Raises:
            SerializationError: If the payload is malformed or noncanonical.
            VerificationError: If native verification rejects the Program.
        """

        from .serialization import decode_program

        return decode_program(payload)

    @property
    def semantic_digest(self) -> str:
        """Return the native SHA-256 identity after erasing debug metadata.

        Returns:
            str: Lowercase hexadecimal semantic program identity.

        Raises:
            VerificationError: If native identity derivation is unavailable or
                violates its result contract.
        """

        return _native_semantic_digest(
            self._native_program(),
            "program_semantic_digest",
        )

    def with_intent(self, **values: PortableValue) -> Program:
        """Return an equal program whose entry carries additional intent rows.

        Intent is untrusted provenance such as ``problem``, ``variable_map``,
        ``tolerance.objective``, or ``tolerance.error``. A row with an existing
        name replaces that row; others are appended. The result is re-sealed
        and re-verified by Rust and keeps this program's ``semantic_digest``.

        Args:
            **values (PortableValue): Intent rows keyed by name.

        Returns:
            Program: Verified program with the merged entry intent.

        Raises:
            TypeError: If a value is not a portable literal.
            VerificationError: If the rows exceed the native intent bounds
                (``Q1060``).
        """

        from .serialization import replace_program_intent

        return replace_program_intent(self, values)


def _native_semantic_digest(
    payload: Any,
    entrypoint: str,
    *arguments: str,
    selection_errors: bool = False,
) -> str:
    """Derive one semantic identity through the required Rust compiler.

    Args:
        payload (Any): Verified native owner or canonical QCore image bytes.
        entrypoint (str): Required native identity function name.
        *arguments (str): Additional resolved semantic selectors.
        selection_errors: Whether Q3001 identifies source selection failure.

    Returns:
        str: Lowercase hexadecimal 32-byte identity.

    Raises:
        VerificationError: If the native entrypoint is unavailable, rejects
            the image, or returns a malformed digest.
    """

    derive = getattr(native, entrypoint, None)
    if not callable(derive):
        raise VerificationError(
            Diagnostic("Q4002", f"native {entrypoint} is unavailable")
        )
    try:
        digest = derive(payload, *arguments)
    except ValueError as error:
        if selection_errors and str(error).startswith("Q3001:"):
            raise SpecializationError(str(error)) from error
        raise VerificationError(str(error)) from error
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise VerificationError(
            Diagnostic("Q4008", f"native {entrypoint} returned a malformed digest")
        )
    return digest.hex()


def attributes(**values: object) -> tuple[tuple[str, PortableValue], ...]:
    """Create sorted immutable operation attributes."""

    return tuple((name, freeze_value(value)) for name, value in sorted(values.items()))


def coherent_call_argument_offset(
    operation: Operation,
    callee: Function,
) -> int | None:
    """Return the first callable-argument operand of a coherent transform.

    A selected call has one coherent selector prefix. A controlled call has
    either the same prefix or an additional classical component selector. The
    distinction is derived from the verified callable arity rather than stored
    as duplicated operation metadata.

    Args:
        operation: Coherent control or select operation.
        callee: Statically resolved callable declaration.

    Returns:
        Operand offset ``1`` or ``2``, or ``None`` for a malformed ABI.
    """

    prefix = len(operation.operands) - len(callee.parameters)
    if operation.opcode == "qcore.control_call" and prefix in {1, 2}:
        return prefix
    if operation.opcode == "qcore.select_call" and prefix == 1:
        return prefix
    return None


def operation_effects(operation: Operation) -> frozenset[Effect]:
    """Derive direct effects from an operation's semantic identity."""

    opcode = operation.opcode
    if opcode in STRUCTURED_OPCODES:
        stage = operation.attribute("stage")
        return (
            frozenset({Effect.SHOT_CONTROL})
            if stage == "Shot"
            else frozenset({Effect.HOST_CONTROL})
            if stage == "Param"
            else frozenset()
        )
    return OPERATION_EFFECTS.get(opcode, frozenset())


def _print_function(function: Function) -> str:
    """Render one function and its nested regions."""

    parameters = ", ".join(
        f"{parameter.value.label()}: {parameter.value.type} [{parameter.name}]"
        for parameter in function.parameters
    )
    contracts = [f"kind={function.kind.value}"]
    if function.effects:
        contracts.append("effects={" + ",".join(sorted(function.effects)) + "}")
    lines = [
        f"func @{function.symbol}({parameters}) -> {function.result_type} "
        f"[{', '.join(contracts)}]{_print_intent(function.intent)} {{"
    ]
    _print_block(function.body, lines, "  ")
    lines.append("}")
    return "\n".join(lines)


def _print_intent(intent: tuple[tuple[str, PortableValue], ...]) -> str:
    """Render intent rows after a function header or operation."""

    if not intent:
        return ""
    return " intent{" + ", ".join(f"{name}={value!r}" for name, value in intent) + "}"


def _print_block(block: Block, lines: list[str], indent: str) -> None:
    """Append a block and nested regions to a textual IR buffer."""

    if block.arguments:
        arguments = ", ".join(
            f"{value.label()}: {value.type}" for value in block.arguments
        )
        lines.append(f"{indent}^block({arguments}):")
        body_indent = indent + "  "
    else:
        body_indent = indent
    for operation in block.operations:
        results = ", ".join(
            f"{value.label()}: {value.type}" for value in operation.results
        )
        operands = ", ".join(f"%{operand}" for operand in operation.operands)
        prefix = f"{results} = " if results else ""
        attrs = ""
        if operation.attributes:
            attrs = (
                " {"
                + ", ".join(f"{name}={value!r}" for name, value in operation.attributes)
                + "}"
            )
        intent = _print_intent(operation.intent)
        lines.append(
            f"{body_indent}{prefix}{operation.opcode}({operands}){attrs}{intent}"
        )
        for index, region in enumerate(operation.regions):
            lines.append(f"{body_indent}  region {index} {{")
            _print_block(region.entry, lines, body_indent + "    ")
            lines.append(f"{body_indent}  }}")
    operands = ", ".join(f"%{operand}" for operand in block.terminator.operands)
    lines.append(f"{body_indent}{block.terminator.kind} {operands}".rstrip())
