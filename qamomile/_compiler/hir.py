"""Ephemeral ownership HIR between Python syntax and persistent QCore SSA."""

from __future__ import annotations

from dataclasses import dataclass

from ..effects import CallableKind
from ..ir import PortableValue, SourceSpan
from ..types import TypeSpec


@dataclass(frozen=True, slots=True)
class HIRParameter:
    """A resolved source parameter before SSA value assignment."""

    name: str
    type: TypeSpec
    default: PortableValue = None
    has_default: bool = False


@dataclass(frozen=True, slots=True)
class HIRExpression:
    """One typed source expression with no Python runtime dependency."""

    opcode: str
    type: TypeSpec
    operands: tuple[HIRExpression, ...] = ()
    attributes: tuple[tuple[str, PortableValue], ...] = ()
    span: SourceSpan = SourceSpan.synthetic("hir")
    regions: tuple[HIRExpression, ...] = ()

    def attribute(self, name: str, default: object = None) -> object:
        """Return one expression attribute by name."""

        return dict(self.attributes).get(name, default)


@dataclass(frozen=True, slots=True)
class HIRStatement:
    """One typed source statement or structured control construct."""

    opcode: str
    expression: HIRExpression | None = None
    targets: tuple[str, ...] = ()
    attributes: tuple[tuple[str, PortableValue], ...] = ()
    regions: tuple[HIRBlock, ...] = ()
    span: SourceSpan = SourceSpan.synthetic("hir")
    result_types: tuple[TypeSpec, ...] = ()

    def attribute(self, name: str, default: object = None) -> object:
        """Return one statement attribute by name."""

        return dict(self.attributes).get(name, default)


@dataclass(frozen=True, slots=True)
class HIRBlock:
    """A lexical source block containing typed HIR statements."""

    statements: tuple[HIRStatement, ...]


@dataclass(frozen=True, slots=True)
class HIRFunction:
    """A fully resolved source function before SSA conversion."""

    symbol: str
    parameters: tuple[HIRParameter, ...]
    result_type: TypeSpec
    body: HIRBlock
    declared_kind: CallableKind | None
    span: SourceSpan
    external: bool = False
    intent: tuple[tuple[str, PortableValue], ...] = ()


@dataclass(frozen=True, slots=True)
class HIRConstant:
    """A canonical source-module constant."""

    name: str
    type: TypeSpec
    value: PortableValue


@dataclass(frozen=True, slots=True)
class HIRModule:
    """A closed typed source unit ready for SSA lowering."""

    module_name: str
    functions: tuple[HIRFunction, ...]
    constants: tuple[HIRConstant, ...]
