"""Callable operation model for composite and oracle calls."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.block import Block

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.value import Value


class CallTransform(enum.Enum):
    """Describe the requested transform of a callable implementation."""

    DIRECT = "direct"
    INVERSE = "inverse"
    CONTROLLED = "controlled"


class CallPolicy(enum.Enum):
    """Describe the default lowering policy for a callable call."""

    INLINE = "inline"
    PRESERVE_BOX = "preserve_box"
    NATIVE_FIRST = "native_first"


def _policy_from_attrs(attrs: dict[str, Any]) -> CallPolicy:
    """Infer a default policy from serialized callable attributes.

    Args:
        attrs (dict[str, Any]): Invocation attributes.

    Returns:
        CallPolicy: Explicit ``default_policy`` when present; otherwise
            ``INLINE`` for qkernel calls and ``PRESERVE_BOX`` for boxed calls.
    """
    raw = attrs.get("default_policy")
    if isinstance(raw, str):
        try:
            return CallPolicy[raw]
        except KeyError:
            try:
                return CallPolicy(raw)
            except ValueError:
                pass
    if attrs.get("kind") in {"composite", "oracle"}:
        return CallPolicy.PRESERVE_BOX
    return CallPolicy.INLINE


class CompositeGateType(enum.Enum):
    """Classify standard boxed quantum callables."""

    QPE = "qpe"
    QFT = "qft"
    IQFT = "iqft"
    CUSTOM = "custom"


@dataclasses.dataclass(frozen=True)
class CallableRef:
    """Identify a callable independently of its Python object.

    Args:
        namespace (str): Stable namespace such as ``"qamomile.stdlib"`` or
            ``"user"``.
        name (str): Stable callable name within the namespace.
        version (str): Schema or behavior version for the callable.
    """

    namespace: str
    name: str
    version: str = "1"


@dataclasses.dataclass
class CallableBodyRef:
    """Reference a callable body that can be materialized later.

    Args:
        ref (CallableRef): Callable whose standard body is referenced.
        kind (str): Body-reference kind, such as ``"standard"`` or
            ``"symbolic_vector"``. Defaults to ``"standard"``.
        attrs (dict[str, Any]): Serializer-friendly body-materialization
            attributes. Defaults to an empty dict.
    """

    ref: CallableRef
    kind: str = "standard"
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CallableImplementation:
    """Describe one implementation candidate for a callable.

    Args:
        transform (CallTransform): Transform this implementation realizes.
        backend (str | None): Backend name for native implementations.
        strategy (str | None): Strategy name such as ``"standard"``.
        body (Block | None): IR implementation body. A transform-specific body
            realizes that transform completely; a controlled body therefore
            includes control operands in its signature.
        body_ref (CallableBodyRef | None): Reference to a body that should be
            materialized by a later resolver. Defaults to ``None``.
        emitter (Any): Backend-native emitter object.
        attrs (dict[str, Any]): Serializer-friendly implementation metadata.
    """

    transform: CallTransform = CallTransform.DIRECT
    backend: str | None = None
    strategy: str | None = None
    body: Block | None = None
    body_ref: CallableBodyRef | None = None
    emitter: Any = None
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CallableDef:
    """Describe a compiler-facing callable definition.

    Args:
        ref (CallableRef): Stable callable identity.
        signature (Signature | None): Optional callable signature.
        body (Block | None): Standard IR body, or ``None`` for opaque calls.
        body_ref (CallableBodyRef | None): Reference to a standard body that is
            intentionally deferred. Defaults to ``None``.
        implementations (list[CallableImplementation]): Alternative native or
            strategy-specific implementations.
        opaque_cost (Any | None): Explicit cost contract for a bodyless
            callable. Body-backed callables must leave this as ``None``.
        default_policy (CallPolicy): Default call lowering policy.
        attrs (dict[str, Any]): Serializer-friendly definition metadata.
    """

    ref: CallableRef
    signature: Signature | None = None
    body: Block | None = None
    body_ref: CallableBodyRef | None = None
    implementations: list[CallableImplementation] = dataclasses.field(
        default_factory=list
    )
    opaque_cost: Any | None = None
    default_policy: CallPolicy = CallPolicy.INLINE
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def implementation_for(
        self,
        *,
        transform: CallTransform = CallTransform.DIRECT,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> CallableImplementation | None:
        """Return the best matching implementation candidate.

        Args:
            transform (CallTransform): Requested call transform.
            backend (str | None): Requested backend name.
            strategy (str | None): Requested strategy name.

        Returns:
            CallableImplementation | None: Matching implementation, if any.
        """
        candidates = [
            impl
            for impl in self.implementations
            if impl.transform == transform
            and (
                (backend is None and impl.backend is None)
                or (backend is not None and impl.backend in (None, backend))
            )
            and (
                (strategy is None and impl.strategy is None)
                or (strategy is not None and impl.strategy in (None, strategy))
            )
        ]
        if not candidates:
            return None

        def score(impl: CallableImplementation) -> int:
            """Score exact backend and strategy matches above generic ones.

            Args:
                impl (CallableImplementation): Candidate implementation.

            Returns:
                int: Match score.
            """
            return int(impl.backend == backend) + int(impl.strategy == strategy)

        return max(candidates, key=score)


def signature_from_values(
    operands: Sequence["Value"],
    results: Sequence["Value"],
    *,
    operand_names: Sequence[str] | None = None,
    result_names: Sequence[str] | None = None,
) -> Signature:
    """Build a callable signature from concrete operand and result values.

    Args:
        operands (Sequence[Value]): Values consumed by the callable.
        results (Sequence[Value]): Values produced by the callable.
        operand_names (Sequence[str] | None): Optional names for operands.
            Missing entries fall back to ``arg_<index>``. Defaults to ``None``.
        result_names (Sequence[str] | None): Optional names for results.
            Missing entries fall back to ``result_<index>``. Defaults to
            ``None``.

    Returns:
        Signature: IR signature with typed parameter hints.
    """

    def name_at(names: Sequence[str] | None, index: int, fallback: str) -> str:
        """Return a provided name or a positional fallback.

        Args:
            names (Sequence[str] | None): Optional candidate names.
            index (int): Name index to read.
            fallback (str): Name to use when no candidate exists.

        Returns:
            str: Selected name.
        """
        if names is not None and index < len(names) and names[index]:
            return names[index]
        return fallback

    return Signature(
        operands=[
            ParamHint(name=name_at(operand_names, i, f"arg_{i}"), type=operand.type)
            for i, operand in enumerate(operands)
        ],
        results=[
            ParamHint(name=name_at(result_names, i, f"result_{i}"), type=result.type)
            for i, result in enumerate(results)
        ],
    )


def signature_from_block(block: Block) -> Signature:
    """Build a callable signature from a traced implementation block.

    Args:
        block (Block): Callable implementation block whose inputs and outputs
            define the signature.

    Returns:
        Signature: IR signature using ``Block.label_args`` and
        ``Block.output_names`` when available.
    """
    return signature_from_values(
        block.input_values,
        block.output_values,
        operand_names=block.label_args,
        result_names=block.output_names,
    )


@dataclasses.dataclass(init=False)
class InvokeOperation(Operation):
    """Represent a composite, stdlib, or oracle call.

    Args:
        operands (list[Value]): Input values consumed by the call.
        results (list[Value]): Output values produced by the call.
        target (CallableRef): Callable identity.
        transform (CallTransform): Direct, inverse, or controlled invocation.
        attrs (dict[str, Any]): Compile-time attributes for strategy, arity,
            and resource/lowering decisions. Values must be serializer-friendly.
        definition (CallableDef | None): Optional callable definition.
    """

    target: CallableRef = dataclasses.field(
        default_factory=lambda: CallableRef(namespace="user", name="anonymous")
    )
    transform: CallTransform = CallTransform.DIRECT
    attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
    definition: CallableDef | None = None

    def __init__(
        self,
        operands: list["Value"] | None = None,
        results: list["Value"] | None = None,
        *,
        target: CallableRef | None = None,
        transform: CallTransform = CallTransform.DIRECT,
        attrs: dict[str, Any] | None = None,
        definition: CallableDef | None = None,
    ) -> None:
        """Initialize an invocation operation.

        Args:
            operands (list[Value] | None): Input values consumed by the call.
                Defaults to ``None``, meaning no operands.
            results (list[Value] | None): Output values produced by the call.
                Defaults to ``None``, meaning no results.
            target (CallableRef | None): Callable identity. Defaults to an
                anonymous user callable when omitted.
            transform (CallTransform): Requested call transform. Defaults to
                ``CallTransform.DIRECT``.
            attrs (dict[str, Any] | None): Serializer-friendly call
                attributes. Defaults to an empty dict.
            definition (CallableDef | None): Callable definition. Defaults to
                ``None``, in which case one is created from ``target``.
        """
        self.operands = list(operands) if operands is not None else []
        self.results = list(results) if results is not None else []
        self.target = (
            target
            if target is not None
            else CallableRef(namespace="user", name="anonymous")
        )
        self.transform = transform
        self.attrs = dict(attrs) if attrs is not None else {}
        self.definition = definition
        self._ensure_definition()

    def _ensure_definition(self) -> None:
        """Ensure the invocation has a compiler-facing callable definition."""
        if self.definition is None:
            self.definition = CallableDef(
                ref=self.target,
                signature=(
                    signature_from_values(self.operands, self.results)
                    if self.operands or self.results
                    else None
                ),
                default_policy=_policy_from_attrs(self.attrs),
                attrs=dict(self.attrs),
            )
        else:
            self.target = self.definition.ref

    @property
    def body(self) -> Block | None:
        """Return the callable's default body from its definition.

        Returns:
            Block | None: The direct-call body stored on ``CallableDef``, or
            ``None`` for opaque callables.
        """
        if self.definition is None:
            return None
        return self.definition.body

    @body.setter
    def body(self, value: Block | None) -> None:
        """Set the callable's default body on its definition.

        Args:
            value (Block | None): Replacement default callable body.
        """
        if self.definition is None:
            self.definition = CallableDef(
                ref=self.target,
                signature=(
                    signature_from_values(self.operands, self.results)
                    if self.operands or self.results
                    else None
                ),
                default_policy=_policy_from_attrs(self.attrs),
                attrs=dict(self.attrs),
            )
        self.definition.body = value

    @property
    def body_ref(self) -> CallableBodyRef | None:
        """Return the callable's deferred body reference.

        Returns:
            CallableBodyRef | None: Selected implementation body reference, the
            definition-level body reference, or ``None`` when no deferred body is
            available.
        """
        impl = self.implementation_for()
        if impl is not None and impl.body_ref is not None:
            return impl.body_ref
        if self.definition is None:
            return None
        return self.definition.body_ref

    @property
    def name(self) -> str:
        """Return the display name for this invocation.

        Returns:
            str: The callable name, optionally prefixed for transforms.
        """
        if self.transform == CallTransform.INVERSE:
            return f"{self.target.name}†"
        return self.target.name

    @property
    def num_control_qubits(self) -> int:
        """Return the number of leading control-qubit operands.

        Returns:
            int: Control arity recorded in ``attrs``. Defaults to ``0``.
        """
        return int(self.attrs.get("num_control_qubits", 0))

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target-qubit operands.

        Returns:
            int: Target arity recorded in ``attrs``. Defaults to the operand
            count of quantum operands after controls.
        """
        default = sum(
            operand.type.is_quantum()
            for operand in self.operands[self.num_control_qubits :]
        )
        return int(self.attrs.get("num_target_qubits", default))

    @property
    def control_qubits(self) -> list["Value"]:
        """Return the control-qubit operands.

        Returns:
            list[Value]: Leading control operands.
        """
        return list(self.operands[: self.num_control_qubits])

    @property
    def target_qubits(self) -> list["Value"]:
        """Return the target-qubit operands.

        Returns:
            list[Value]: Quantum target operands after any controls, preserving
            their relative order even when classical parameters are interleaved
            in the callable signature. A vector target counts as one operand even when
            ``num_target_qubits`` records its scalar backend width.
        """
        start = self.num_control_qubits
        return [
            operand for operand in self.operands[start:] if operand.type.is_quantum()
        ]

    @property
    def parameters(self) -> list["Value"]:
        """Return non-qubit parameter operands.

        Returns:
            list[Value]: Classical/object operands after the control prefix,
            preserving their relative declaration order.
        """
        start = self.num_control_qubits
        return [
            operand
            for operand in self.operands[start:]
            if operand.type.is_classical() or operand.type.is_object()
        ]

    @property
    def gate_type(self) -> CompositeGateType:
        """Return the standard composite classification for this invocation.

        Returns:
            CompositeGateType: Type encoded in ``attrs["gate_type"]``. Unknown
            values map to ``CompositeGateType.CUSTOM``.
        """
        raw = str(self.attrs.get("gate_type", CompositeGateType.CUSTOM.name))
        try:
            return CompositeGateType[raw]
        except KeyError:
            return CompositeGateType.CUSTOM

    @property
    def custom_name(self) -> str:
        """Return the display name for custom callable boxes.

        Returns:
            str: ``attrs["custom_name"]`` when present, else target name.
        """
        return str(self.attrs.get("custom_name", self.target.name))

    @property
    def strategy_name(self) -> str | None:
        """Return the selected lowering/resource strategy name.

        Returns:
            str | None: Strategy name stored in attrs, or ``None``.
        """
        value = self.attrs.get("strategy_name")
        if value is None:
            return None
        return str(value)

    @property
    def default_policy(self) -> CallPolicy:
        """Return the callable's default lowering policy.

        Returns:
            CallPolicy: Policy from the callable definition, or ``INLINE``.
        """
        if self.definition is None:
            return CallPolicy.INLINE
        return self.definition.default_policy

    def implementation_for(
        self,
        *,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> CallableImplementation | None:
        """Return the selected implementation for this invocation.

        Args:
            backend (str | None): Backend name to match. Defaults to ``None``,
                which only selects backend-generic implementations.
            strategy (str | None): Strategy name to match. Defaults to
                ``None``, meaning the invocation's ``strategy_name`` attribute
                is used.

        Returns:
            CallableImplementation | None: Matching implementation candidate,
            or ``None`` when the callable definition has no match.
        """
        if self.definition is None:
            return None
        requested_strategy = self.strategy_name if strategy is None else strategy
        return self.definition.implementation_for(
            transform=self.transform,
            backend=backend,
            strategy=requested_strategy,
        )

    def effective_body(
        self,
        *,
        backend: str | None = None,
        strategy: str | None = None,
    ) -> Block | None:
        """Return the implementation body selected for this invocation.

        Args:
            backend (str | None): Backend name to match. Defaults to ``None``.
            strategy (str | None): Strategy name to match. Defaults to the
                invocation's ``strategy_name`` attribute.

        Returns:
            Block | None: Selected implementation body, or the callable's
            default body when no transform-specific implementation exists.
            A compiler may synthesize inverse or controlled behavior from this
            fallback body.
        """
        impl = self.implementation_for(backend=backend, strategy=strategy)
        if impl is not None and impl.body is not None:
            return impl.body
        return self.body

    @property
    def signature(self) -> Signature:
        """Return the operation signature.

        Returns:
            Signature: Callable definition signature when available,
            otherwise best-effort operand and result hints from concrete
            values.
        """
        if self.definition is not None and self.definition.signature is not None:
            return self.definition.signature
        return signature_from_values(self.operands, self.results)

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation kind.

        Returns:
            OperationKind: ``QUANTUM`` because callables in this model consume
            or produce quantum values.
        """
        return OperationKind.QUANTUM
