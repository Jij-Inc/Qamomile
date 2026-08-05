"""Frontend oracle callable."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast, overload

from qamomile.circuit.frontend.callable_signature import CallableSignature
from qamomile.circuit.frontend.handle.array import Vector, VectorView
from qamomile.circuit.frontend.handle.primitives import Qubit, UInt
from qamomile.circuit.frontend.qkernel_utils import reject_aliased_quantum_args
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CallTransform,
    InvokeOperation,
    signature_from_values,
)
from qamomile.circuit.ir.operation.control_value import normalize_control_value
from qamomile.circuit.ir.operation.operation import Signature
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.estimator import OpaqueCostContext, ResourceEstimate


def _normalize_nonnegative_integer(
    value: object,
    *,
    label: str = "num_control_qubits",
) -> int:
    """Return one nonnegative integral value as a Python integer.

    Args:
        value (object): Candidate Python or NumPy integer scalar.
        label (str): Parameter name used in validation errors. Defaults to
            ``"num_control_qubits"``.

    Returns:
        int: Equivalent nonnegative Python integer.

    Raises:
        TypeError: If ``value`` is boolean or not integral.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return normalized


@dataclasses.dataclass(eq=False)
class Oracle:
    """Represent an opaque oracle callable.

    Args:
        name (str): Human-readable oracle name.
        num_qubits (int | None): Number of target qubits consumed and returned
            by the oracle. ``None`` means the arity is provided by
            ``signature`` and may be vector-shaped. Python and NumPy integer
            scalars are accepted; booleans and negative values are rejected.
        num_control_qubits (int): Number of explicit control qubits required
            by scalar calls. Defaults to ``0``.
        signature (CallableSignature | None): Optional frontend signature for
            target operands only. It must not repeat the leading controls
            declared by ``num_control_qubits``; those controls are prefixed by
            the Oracle automatically. When omitted, a fixed-width
            scalar/vector-compatible oracle is created from ``num_qubits``.
        cost (ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None):
            Optional explicit cost for this bodyless callable. Both forms
            describe one ordinary application of the Oracle as declared,
            including ``num_control_qubits``. Controls added later with
            ``qmc.control`` are projected by resource estimation. This is a
            complete definition-level contract: the author must include any
            phase-relevant work that later coherent controls need. The
            estimator does not infer omitted global-phase overhead. An
            intrinsic nonidentity phase is represented as a logical primitive
            in the aggregate gate and arity counts. A one-qubit phase entry is
            an upper-bound representative for the target-free phase, not an
            angle-aware reconstruction; use a body-backed global phase when
            angle-specific classification is required. Defaults to ``None``.

    Raises:
        TypeError: If a supplied ``num_qubits`` or ``num_control_qubits`` is
            boolean or not an integral scalar.
        ValueError: If either width is negative, or neither ``num_qubits`` nor
            ``signature`` supplies enough target-arity information.
    """

    name: str
    num_qubits: int | None = None
    num_control_qubits: int = 0
    signature: CallableSignature | None = None
    cost: ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None = (
        None
    )

    def __init__(
        self,
        name: str,
        num_qubits: int | None = None,
        *,
        num_control_qubits: int = 0,
        signature: CallableSignature | None = None,
        cost: (
            ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None
        ) = None,
    ) -> None:
        """Initialize an opaque oracle callable.

        Args:
            name (str): Human-readable oracle name.
            num_qubits (int | None): Fixed scalar/vector width. Defaults to
                ``None`` when ``signature`` describes the callable. Python
                and NumPy integer scalars are accepted; booleans and negative
                values are rejected.
            num_control_qubits (int): Number of explicit scalar controls.
                Defaults to ``0``.
            signature (CallableSignature | None): Optional frontend signature
                for target operands only. Do not include controls declared by
                ``num_control_qubits``; the Oracle prefixes those controls to
                its internal callable signature. Defaults to ``None``.
            cost (ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None):
                Optional fixed or context-dependent opaque cost. The returned
                estimate describes one ordinary application of this Oracle
                definition, including its declared controls but excluding
                controls added by an outer transform. The result must be a
                complete definition-level contract, including phase-relevant
                work that an outer coherent control must transform. Represent
                an intrinsic nonidentity phase as a logical primitive in the
                aggregate gate and arity counts. A one-qubit phase entry is an
                upper-bound representative; use a body-backed global phase
                for angle-specific classification. Defaults to ``None``.

        Raises:
            TypeError: If a supplied ``num_qubits`` or
                ``num_control_qubits`` is not a non-boolean integral scalar.
            ValueError: If neither ``num_qubits`` nor ``signature`` supplies
                enough target-arity information, or if either width is
                negative.
        """
        normalized_control_qubits = _normalize_nonnegative_integer(num_control_qubits)
        if signature is not None and num_qubits is None:
            num_qubits = signature.scalar_qubit_input_count()
        if num_qubits is not None:
            num_qubits = _normalize_nonnegative_integer(
                num_qubits,
                label="num_qubits",
            )
        if num_qubits is None and not (
            signature is not None and signature.accepts_single_qubit_vector()
        ):
            raise ValueError(
                "Oracle requires either num_qubits or a single Vector[Qubit] "
                "CallableSignature."
            )
        self.name = name
        self.num_qubits = num_qubits
        self.num_control_qubits = normalized_control_qubits
        self.signature = signature
        self.cost = cost

    def _uses_vector_target_signature(self) -> bool:
        """Return whether this Oracle exposes one vector target operand.

        Returns:
            bool: Whether coherent controls cannot currently be prepended to
            this Oracle's target ABI.
        """
        return self.num_qubits is None or (
            self.signature is not None and self.signature.accepts_single_qubit_vector()
        )

    @overload
    def __call__(
        self,
        qubits: VectorView[Qubit],
        /,
        *,
        controls: tuple[()] = (),
        control_value: None = None,
    ) -> VectorView[Qubit]: ...

    @overload
    def __call__(
        self,
        qubits: Vector[Qubit],
        /,
        *,
        controls: tuple[()] = (),
        control_value: None = None,
    ) -> Vector[Qubit]: ...

    @overload
    def __call__(
        self,
        *qubits: Qubit,
        controls: Sequence[Qubit] = (),
        control_value: int | None = None,
    ) -> tuple[Qubit, ...]: ...

    def __call__(
        self,
        *qubits: Qubit | Vector[Qubit] | VectorView[Qubit],
        controls: Sequence[Qubit] = (),
        control_value: int | None = None,
    ) -> tuple[Qubit, ...] | Vector[Qubit] | VectorView[Qubit]:
        """Apply the oracle to scalar qubits or a vector register.

        Args:
            *qubits (Qubit | Vector[Qubit] | VectorView[Qubit]): Either a single
                vector register or view, or ``num_qubits`` scalar qubits.
            controls (Sequence[Qubit]): Explicit control qubits for scalar
                calls. Vector and vector-view calls currently require the
                default empty sequence.
            control_value (int | None): LSB-first activation value for scalar
                ``controls``. ``None`` uses the ordinary all-ones state. Vector
                and vector-view calls currently require ``None``. Defaults to
                ``None``.

        Returns:
            tuple[Qubit, ...] | Vector[Qubit] | VectorView[Qubit]: Oracle
                outputs with the same shape as the input form. Vector views
                remain vector views.

        Raises:
            ValueError: If the provided arity does not match ``num_qubits``
                or ``control_value`` does not fit the control width.
            TypeError: If the argument shape is not supported or
                ``control_value`` is invalid.
            QubitConsumedError: If an input was already consumed or two input
                roles overlap the same physical qubit.
            RuntimeError: If no qkernel tracer is active.
        """
        if len(qubits) == 1 and isinstance(qubits[0], Vector):
            if controls or control_value is not None:
                raise TypeError(
                    "Oracle vector calls do not accept controls or control_value yet."
                )
            if self.num_control_qubits:
                raise ValueError(
                    f"Oracle '{self.name}' requires {self.num_control_qubits} "
                    "explicit control qubits; use the scalar call form until "
                    "vector calls support a separate controls argument."
                )
            return self._call_vector(qubits[0])
        return self._call_scalars(
            *qubits,
            controls=controls,
            control_value=control_value,
        )

    def _call_vector(
        self,
        qubits: Vector[Qubit],
        *,
        _inverse: bool = False,
    ) -> Vector[Qubit]:
        """Apply the oracle to a vector-like register.

        Args:
            qubits (Vector[Qubit]): Register to consume.
            _inverse (bool): Whether this call applies the inverse Oracle
                transform. Defaults to ``False``.

        Returns:
            Vector[Qubit]: Next-version register.

        Raises:
            ValueError: If the vector has a concrete length different from
                ``num_qubits``.
            QubitConsumedError: If the register was already consumed or has
                an outstanding borrow.
            RuntimeError: If no tracer is active.
        """
        size_handle = qubits.shape[0] if qubits.shape else None
        if isinstance(size_handle, int):
            size = size_handle
        elif isinstance(size_handle, UInt) and size_handle.value.is_constant():
            size = size_handle.value.get_const()
        else:
            size = None
        if (
            self.num_qubits is not None
            and size is not None
            and int(size) != self.num_qubits
        ):
            raise ValueError(
                f"Oracle '{self.name}' requires {self.num_qubits} qubits, "
                f"got {int(size)}."
            )

        operation_name = f"Oracle[{self.name}]"
        tracer = get_current_tracer()
        qubits.validate_consumable(operation_name)
        result = qubits.value.next_version()
        oracle_ref = CallableRef(namespace="user.oracle", name=self.name)
        num_targets = self.num_qubits
        if num_targets is None and size is not None:
            num_targets = int(size)
        attrs = {
            "kind": "oracle",
            "num_control_qubits": 0,
            "num_declared_control_qubits": 0,
            "num_added_control_qubits": 0,
            "num_target_qubits": num_targets or 0,
            "custom_name": self.name,
            "gate_type": "CUSTOM",
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        signature = (
            self.signature.to_ir_signature()
            if self.signature is not None
            else signature_from_values(
                [qubits.value],
                [result],
                operand_names=["qubits"],
                result_names=["qubits"],
            )
        )
        op = InvokeOperation(
            operands=[qubits.value],
            results=[result],
            target=oracle_ref,
            transform=(CallTransform.INVERSE if _inverse else CallTransform.DIRECT),
            attrs=attrs,
            definition=CallableDef(
                ref=oracle_ref,
                signature=signature,
                opaque_cost=self.cost,
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )
        consumed = qubits.consume(operation_name=operation_name)
        tracer.add_operation(op)
        consumed_any: Any = consumed
        if isinstance(consumed_any, VectorView):
            new_view = VectorView._wrap_unregistered(
                parent=consumed_any._slice_parent,
                sliced_av=result,
                length=consumed_any.shape[0],
                start_uint=consumed_any._slice_start,
                step_uint=consumed_any._slice_step,
            )
            consumed_any._transfer_borrow_to(new_view, operation_name)
            return new_view
        return type(qubits)._create_from_value(value=result, shape=qubits.shape)

    def _call_scalars(
        self,
        *qubits: Qubit | Vector[Qubit],
        controls: Sequence[Qubit] = (),
        control_value: int | None = None,
        _added_num_control_qubits: int = 0,
        _inverse: bool = False,
    ) -> tuple[Qubit, ...]:
        """Apply the oracle to scalar qubits.

        Args:
            *qubits (Qubit | Vector[Qubit] | VectorView[Qubit]): Scalar qubit
                handles. Vector arguments are rejected here.
            controls (Sequence[Qubit]): Explicit scalar controls.
            control_value (int | None): LSB-first activation value for the
                explicit controls. Defaults to ``None``.
            _added_num_control_qubits (int): Number of leading controls added
                by an outer frontend transform rather than declared by this
                Oracle. Defaults to ``0``.
            _inverse (bool): Whether this call applies the inverse Oracle
                transform. Defaults to ``False``.

        Returns:
            tuple[Qubit, ...]: Next-version scalar qubits.

        Raises:
            TypeError: If any argument is not a scalar qubit or
                ``control_value`` is invalid, or an explicit signature type
                cannot be converted to IR.
            ValueError: If the number of scalar qubits is wrong or the
                activation value does not fit the control width, or an
                explicit signature disagrees with the scalar base ABI.
            QubitConsumedError: If an input was consumed or two roles overlap
                the same physical qubit.
            RuntimeError: If no tracer is active.
        """
        if self.num_qubits is None:
            raise TypeError(
                f"Oracle '{self.name}' was declared with a vector signature "
                "and does not accept scalar qubit arguments."
            )
        if len(qubits) != self.num_qubits:
            raise ValueError(
                f"Oracle '{self.name}' requires {self.num_qubits} qubits, "
                f"got {len(qubits)}."
            )
        expected_controls = self.num_control_qubits + _added_num_control_qubits
        if len(controls) != expected_controls:
            raise ValueError(
                f"Oracle '{self.name}' requires {expected_controls} "
                f"control qubits, got {len(controls)}."
            )
        if not all(isinstance(q, Qubit) for q in qubits):
            raise TypeError("Oracle scalar calls accept only Qubit arguments.")
        if not all(isinstance(c, Qubit) for c in controls):
            raise TypeError("Oracle controls accept only Qubit arguments.")
        normalized_control_value = (
            normalize_control_value(control_value, len(controls)) if controls else None
        )
        if not controls and control_value is not None:
            raise ValueError("Oracle control_value requires at least one control.")

        tracer = get_current_tracer()
        all_inputs = [*controls, *qubits]
        reject_aliased_quantum_args(
            self.name,
            {
                **{
                    f"control[{index}]": control
                    for index, control in enumerate(controls)
                },
                **{f"target[{index}]": qubit for index, qubit in enumerate(qubits)},
            },
            caller=f"Oracle[{self.name}]",
        )
        for handle in all_inputs:
            handle.validate_consumable(f"Oracle[{self.name}]")

        results = [q.value.next_version() for q in all_inputs]
        oracle_ref = CallableRef(namespace="user.oracle", name=self.name)
        attrs = {
            "kind": "oracle",
            "num_control_qubits": len(controls),
            "num_declared_control_qubits": self.num_control_qubits,
            "num_added_control_qubits": _added_num_control_qubits,
            "num_target_qubits": len(qubits),
            "custom_name": self.name,
            "gate_type": "CUSTOM",
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        if normalized_control_value is not None:
            attrs["control_value"] = normalized_control_value
        transform = (
            CallTransform.CONTROLLED_INVERSE
            if controls and _inverse
            else CallTransform.CONTROLLED
            if controls
            else CallTransform.INVERSE
            if _inverse
            else CallTransform.DIRECT
        )
        added_controls = _added_num_control_qubits
        base_inputs = [q.value for q in all_inputs[added_controls:]]
        base_results = results[added_controls:]
        base_attrs = {
            **attrs,
            "num_control_qubits": self.num_control_qubits,
            "num_declared_control_qubits": self.num_control_qubits,
            "num_added_control_qubits": 0,
        }
        base_attrs.pop("control_value", None)
        signature = self._scalar_definition_signature(
            base_inputs,
            base_results,
        )
        op = InvokeOperation(
            operands=[q.value for q in all_inputs],
            results=results,
            target=oracle_ref,
            transform=transform,
            attrs=attrs,
            definition=CallableDef(
                ref=oracle_ref,
                signature=signature,
                opaque_cost=self.cost,
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=base_attrs,
            ),
        )
        consumed_qubits = [
            handle.consume(
                operation_name=(
                    f"Oracle[{self.name}][control]"
                    if index < len(controls)
                    else f"Oracle[{self.name}][target]"
                )
            )
            for index, handle in enumerate(all_inputs)
        ]
        tracer.add_operation(op)

        outputs = []
        for result, qubit in zip(results, consumed_qubits, strict=True):
            output = Qubit(
                value=result,
                parent=qubit.parent,
                indices=qubit.indices,
            )
            qubit._handoff_direct_borrow_to(output)
            outputs.append(output)
        return tuple(outputs)

    def _scalar_definition_signature(
        self,
        base_inputs: Sequence[Value],
        base_results: Sequence[Value],
    ) -> Signature:
        """Build and validate the definition-level scalar Oracle signature.

        An explicit ``CallableSignature`` describes the target portion of the
        public Oracle API. Definition-declared controls are prefixed here,
        while controls added later by ``qmc.control`` remain outside the base
        ABI.

        Args:
            base_inputs (Sequence[Value]): Declared controls followed by target
                values for one ordinary Oracle application.
            base_results (Sequence[Value]): Result values in the same base-ABI
                order.

        Returns:
            Signature: Definition signature with declared controls and the
                preserved explicit target hints.

        Raises:
            TypeError: If an explicit frontend type cannot be converted to an
                IR value type.
            ValueError: If an explicit signature has a different target arity
                or value type from the scalar Oracle call.
        """
        declared_controls = self.num_control_qubits
        control_names = [f"control_{index}" for index in range(declared_controls)]
        target_count = len(base_inputs) - declared_controls
        if self.signature is None:
            target_names = [f"target_{index}" for index in range(target_count)]
            return signature_from_values(
                base_inputs,
                base_results,
                operand_names=[*control_names, *target_names],
                result_names=[*control_names, *target_names],
            )

        explicit = self.signature.to_ir_signature()
        target_inputs = base_inputs[declared_controls:]
        target_results = base_results[declared_controls:]
        if len(explicit.operands) != len(target_inputs) or len(explicit.results) != len(
            target_results
        ):
            raise ValueError(
                f"Oracle '{self.name}' explicit CallableSignature must describe "
                f"exactly {len(target_inputs)} scalar target input(s) and "
                f"{len(target_results)} target output(s); got "
                f"{len(explicit.operands)} input(s) and "
                f"{len(explicit.results)} output(s)."
            )
        for role, hints, values in (
            ("input", explicit.operands, target_inputs),
            ("output", explicit.results, target_results),
        ):
            for index, (hint, value) in enumerate(zip(hints, values, strict=True)):
                if hint is None or hint.type != value.type:
                    actual_type = None if hint is None else hint.type
                    raise ValueError(
                        f"Oracle '{self.name}' explicit CallableSignature "
                        f"target {role} {index} has type {actual_type!r}; "
                        f"expected {value.type!r}."
                    )

        control_signature = signature_from_values(
            base_inputs[:declared_controls],
            base_results[:declared_controls],
            operand_names=control_names,
            result_names=control_names,
        )
        return Signature(
            operands=[*control_signature.operands, *explicit.operands],
            results=[*control_signature.results, *explicit.results],
        )


@dataclasses.dataclass(frozen=True)
class TransformedOracle:
    """Represent composable inverse and controlled transforms of an Oracle.

    The wrapped ``Oracle`` remains the definition boundary: its explicit cost
    includes only controls declared by ``Oracle.num_control_qubits``. This
    wrapper records controls added later by ``qmc.control`` and whether the
    call is inverted, so the resulting ``InvokeOperation`` can apply those
    transforms exactly once.

    Args:
        oracle (Oracle): Definition-level opaque Oracle.
        added_num_control_qubits (int): Number of controls added outside the
            Oracle definition. Defaults to ``0``.
        added_control_value (int | None): LSB-first activation value for the
            added controls, or ``None`` for all ones. Defaults to ``None``.
        inverse (bool): Whether to apply the inverse Oracle. Defaults to
            ``False``.

    Raises:
        TypeError: If ``added_num_control_qubits`` is boolean or not integral,
            or a positive count is attached to a vector-signature Oracle.
        ValueError: If the added-control count is negative or its activation
            value is invalid for that width.
    """

    oracle: Oracle
    added_num_control_qubits: int = 0
    added_control_value: int | None = None
    inverse: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize the stored added-control condition.

        Raises:
            TypeError: If ``added_num_control_qubits`` is not a non-boolean
                integer, or controls are added to a vector-signature Oracle.
            ValueError: If the added-control count is negative or its
                activation value is invalid for that width.
        """
        count = _normalize_nonnegative_integer(
            self.added_num_control_qubits,
            label="added_num_control_qubits",
        )
        if count and self.oracle._uses_vector_target_signature():
            raise TypeError(
                "control(Oracle) supports fixed-width scalar oracles only; "
                "vector-signature oracles must be called directly."
            )
        object.__setattr__(self, "added_num_control_qubits", count)
        if count == 0:
            if self.added_control_value is not None:
                raise ValueError(
                    "added_control_value requires at least one added control."
                )
            return
        object.__setattr__(
            self,
            "added_control_value",
            normalize_control_value(self.added_control_value, count),
        )

    @property
    def name(self) -> str:
        """Return the source Oracle name.

        Returns:
            str: Human-readable Oracle name.
        """
        return self.oracle.name

    def controlled(
        self,
        num_controls: int,
        *,
        control_value: int | None = None,
    ) -> TransformedOracle:
        """Prepend another concrete control group.

        Args:
            num_controls (int): Number of newly added leading controls.
            control_value (int | None): LSB-first activation value for the new
                control group. ``None`` means all ones. Defaults to ``None``.

        Returns:
            TransformedOracle: Wrapper carrying the combined added-control
                condition.

        Raises:
            TypeError: If ``num_controls`` is not a non-boolean integer or
                ``control_value`` is not a Python integer or ``None``, or the
                source Oracle uses a vector target signature.
            ValueError: If ``num_controls`` is not positive or
                ``control_value`` does not fit its width.
        """
        if self.oracle._uses_vector_target_signature():
            raise TypeError(
                "control(Oracle) supports fixed-width scalar oracles only; "
                "vector-signature oracles must be called directly."
            )
        num_controls = _normalize_nonnegative_integer(
            num_controls,
            label="num_controls",
        )
        if num_controls == 0:
            raise ValueError("num_controls must be >= 1, got 0.")
        outer_value = normalize_control_value(control_value, num_controls)
        outer_pattern = (1 << num_controls) - 1 if outer_value is None else outer_value
        existing_count = self.added_num_control_qubits
        existing_pattern = (
            (1 << existing_count) - 1
            if self.added_control_value is None
            else self.added_control_value
        )
        combined_count = num_controls + existing_count
        combined_pattern = outer_pattern | (existing_pattern << num_controls)
        return TransformedOracle(
            oracle=self.oracle,
            added_num_control_qubits=combined_count,
            added_control_value=combined_pattern,
            inverse=self.inverse,
        )

    def inverted(self) -> Oracle | TransformedOracle:
        """Toggle inverse application while preserving added controls.

        Returns:
            Oracle | TransformedOracle: The original Oracle when every
                transform cancels, otherwise a transformed wrapper.
        """
        if self.inverse and self.added_num_control_qubits == 0:
            return self.oracle
        return dataclasses.replace(self, inverse=not self.inverse)

    def __call__(
        self,
        *qubits: Qubit | Vector[Qubit],
        controls: Sequence[Qubit] = (),
        control_value: int | None = None,
        declared_control_value: int | None = None,
    ) -> tuple[Qubit, ...] | Vector[Qubit]:
        """Apply the transformed Oracle at the current trace site.

        An inverse-only wrapper retains the source Oracle's call shape. Once
        controls have been added with ``qmc.control``, positional arguments are
        the added controls, the Oracle-declared controls, and the targets, in
        that order.

        Args:
            *qubits (Qubit | Vector[Qubit]): Oracle arguments. For a controlled
                wrapper, these are the complete leading control prefix followed
                by fixed-width scalar targets.
            controls (Sequence[Qubit]): Definition-declared controls for an
                inverse-only call. Defaults to an empty sequence.
            control_value (int | None): Activation value for ``controls`` on
                an inverse-only call. Defaults to ``None``.
            declared_control_value (int | None): LSB-first activation value
                for the Oracle-declared portion of a controlled wrapper's
                positional control prefix. Defaults to ``None`` (all ones).

        Returns:
            tuple[Qubit, ...] | Vector[Qubit]: Transformed Oracle outputs in
                input order.

        Raises:
            TypeError: If vector arguments, ``controls``, or ``control_value``
                are used after controls were added, or if
                ``declared_control_value`` is used before controls are added
                or has an invalid Python type.
            ValueError: If a controlled wrapper receives too few positional
                qubits, ``declared_control_value`` is used without declared
                controls or does not fit their width, or the source Oracle
                arity validation fails.
            QubitConsumedError: If an input was consumed or input roles
                overlap.
            RuntimeError: If no qkernel tracer is active.
        """
        added_controls = self.added_num_control_qubits
        if added_controls == 0:
            if declared_control_value is not None:
                raise TypeError(
                    "declared_control_value is only used after qmc.control(); "
                    "pass control_value for an ordinary or inverse-only "
                    "Oracle call."
                )
            if len(qubits) == 1 and isinstance(qubits[0], Vector):
                if controls or control_value is not None:
                    raise TypeError(
                        "Oracle vector calls do not accept controls or "
                        "control_value yet."
                    )
                if self.oracle.num_control_qubits:
                    raise ValueError(
                        f"Oracle '{self.oracle.name}' requires "
                        f"{self.oracle.num_control_qubits} explicit control "
                        "qubits; use the scalar call form."
                    )
                return self.oracle._call_vector(qubits[0], _inverse=self.inverse)
            return self.oracle._call_scalars(
                *qubits,
                controls=controls,
                control_value=control_value,
                _inverse=self.inverse,
            )

        if controls or control_value is not None:
            raise TypeError(
                "A controlled Oracle takes its complete control prefix "
                "positionally; controls= and control_value= apply only before "
                "qmc.control()."
            )
        if self.oracle.num_qubits is None:
            raise TypeError(
                "control(Oracle) supports fixed-width scalar oracles only. "
                "Vector-signature oracles should be called directly."
            )
        total_controls = self.oracle.num_control_qubits + added_controls
        if len(qubits) < total_controls:
            raise ValueError(
                f"Controlled Oracle '{self.oracle.name}' requires "
                f"{total_controls} control qubits, got {len(qubits)}."
            )
        if not all(isinstance(qubit, Qubit) for qubit in qubits):
            raise TypeError(
                "Controlled Oracle calls accept only scalar Qubit arguments."
            )
        control_prefix = cast(tuple[Qubit, ...], qubits[:total_controls])
        targets = qubits[total_controls:]
        added_pattern = (
            (1 << added_controls) - 1
            if self.added_control_value is None
            else self.added_control_value
        )
        declared_controls = self.oracle.num_control_qubits
        if declared_controls:
            normalized_declared_value = normalize_control_value(
                declared_control_value,
                declared_controls,
            )
            declared_pattern = (
                (1 << declared_controls) - 1
                if normalized_declared_value is None
                else normalized_declared_value
            )
        elif declared_control_value is not None:
            raise ValueError(
                "declared_control_value requires an Oracle with at least one "
                "declared control."
            )
        else:
            declared_pattern = 0
        combined_pattern = added_pattern | (declared_pattern << added_controls)
        return self.oracle._call_scalars(
            *targets,
            controls=control_prefix,
            control_value=combined_pattern,
            _added_num_control_qubits=added_controls,
            _inverse=self.inverse,
        )


def opaque(
    name: str,
    num_qubits: int | None = None,
    *,
    num_control_qubits: int = 0,
    signature: CallableSignature | None = None,
    cost: (
        ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None
    ) = None,
) -> Oracle:
    """Create an opaque callable for top-down circuit design.

    Args:
        name (str): Human-readable callable name.
        num_qubits (int | None): Number of target qubits consumed and returned
            by the callable. Defaults to ``None`` when ``signature`` carries
            the target shape contract. Python and NumPy integer scalars are
            accepted; booleans and negative values are rejected.
        num_control_qubits (int): Number of explicit scalar control qubits
            required by scalar calls. Defaults to ``0``.
        signature (CallableSignature | None): Optional frontend signature for
            target operands only. It must exclude controls declared by
            ``num_control_qubits`` because the Oracle prefixes those controls
            automatically. Defaults to ``None``.
        cost (ResourceEstimate | Callable[[OpaqueCostContext], ResourceEstimate] | None):
            Optional fixed or context-dependent opaque cost. Both forms
            describe one ordinary application of this Oracle definition,
            including declared controls and any phase-relevant work that later
            coherent controls must transform. The estimator treats this as a
            complete definition-level contract and does not add hidden
            global-phase overhead. A one-qubit phase entry is an upper-bound
            representative rather than an angle-aware reconstruction.
            Defaults to ``None``.

    Returns:
        Oracle: Opaque callable backed by ``InvokeOperation`` with no body.

    Raises:
        TypeError: If a supplied ``num_qubits`` or ``num_control_qubits`` is
            not a non-boolean integral scalar.
        ValueError: If neither ``num_qubits`` nor ``signature`` supplies
            enough target-arity information, or if either width is negative.
    """
    return Oracle(
        name=name,
        num_qubits=num_qubits,
        num_control_qubits=num_control_qubits,
        signature=signature,
        cost=cost,
    )
