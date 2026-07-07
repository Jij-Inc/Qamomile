"""Frontend oracle callable."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from qamomile.circuit.frontend.callable_signature import CallableSignature
from qamomile.circuit.frontend.handle.array import Vector, VectorView
from qamomile.circuit.frontend.handle.primitives import Qubit, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CallTransform,
    InvokeOperation,
    ResourceModelBinding,
    signature_from_values,
)


@dataclasses.dataclass
class Oracle:
    """Represent an opaque oracle callable.

    Args:
        name (str): Human-readable oracle name.
        num_qubits (int | None): Number of target qubits consumed and returned
            by the oracle. ``None`` means the arity is provided by
            ``signature`` and may be vector-shaped.
        num_control_qubits (int): Number of explicit control qubits required
            by scalar calls. Defaults to ``0``.
        signature (CallableSignature | None): Optional frontend signature.
            When omitted, a fixed-width scalar/vector-compatible oracle is
            created from ``num_qubits``.
        resource_model (Any | None): Optional context-aware resource model used
            by symbolic resource estimation. Defaults to ``None``.
    """

    name: str
    num_qubits: int | None = None
    num_control_qubits: int = 0
    signature: CallableSignature | None = None
    resource_model: Any | None = None

    def __init__(
        self,
        name: str,
        num_qubits: int | None = None,
        *,
        num_control_qubits: int = 0,
        signature: CallableSignature | None = None,
        resource_model: Any | None = None,
    ) -> None:
        """Initialize an opaque oracle callable.

        Args:
            name (str): Human-readable oracle name.
            num_qubits (int | None): Fixed scalar/vector width. Defaults to
                ``None`` when ``signature`` describes the callable.
            num_control_qubits (int): Number of explicit scalar controls.
                Defaults to ``0``.
            signature (CallableSignature | None): Optional frontend signature.
                Defaults to ``None``.
            resource_model (Any | None): Optional context-aware resource model.
                Defaults to ``None``.

        Raises:
            ValueError: If neither ``num_qubits`` nor ``signature`` supplies
                enough arity information.
        """
        if signature is not None and num_qubits is None:
            num_qubits = signature.scalar_qubit_input_count()
        if num_qubits is None and not (
            signature is not None and signature.accepts_single_qubit_vector()
        ):
            raise ValueError(
                "Oracle requires either num_qubits or a single Vector[Qubit] "
                "CallableSignature."
            )
        self.name = name
        self.num_qubits = num_qubits
        self.num_control_qubits = num_control_qubits
        self.signature = signature
        self.resource_model = resource_model

    def __call__(
        self,
        *qubits: Qubit | Vector[Qubit],
        controls: Sequence[Qubit] = (),
    ) -> tuple[Qubit, ...] | Vector[Qubit]:
        """Apply the oracle to scalar qubits or a vector register.

        Args:
            *qubits (Qubit | Vector[Qubit]): Either a single vector register
                or ``num_qubits`` scalar qubits.
            controls (Sequence[Qubit]): Explicit control qubits. Defaults to
                an empty sequence.

        Returns:
            tuple[Qubit, ...] | Vector[Qubit]: Oracle outputs with the same
            shape as the input form.

        Raises:
            ValueError: If the provided arity does not match ``num_qubits``.
            TypeError: If the argument shape is not supported.
        """
        if len(qubits) == 1 and isinstance(qubits[0], Vector):
            if controls:
                raise TypeError("Oracle vector calls do not accept controls yet.")
            return self._call_vector(qubits[0])
        return self._call_scalars(*qubits, controls=controls)

    def _call_vector(
        self,
        qubits: Vector[Qubit],
    ) -> Vector[Qubit]:
        """Apply the oracle to a vector-like register.

        Args:
            qubits (Vector[Qubit]): Register to consume.

        Returns:
            Vector[Qubit]: Next-version register.

        Raises:
            ValueError: If the vector has a concrete length different from
                ``num_qubits``.
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

        consumed = qubits.consume(operation_name=f"Oracle[{self.name}]")
        result = consumed.value.next_version()
        oracle_ref = CallableRef(namespace="user.oracle", name=self.name)
        num_targets = self.num_qubits
        if num_targets is None and size is not None:
            num_targets = int(size)
        attrs = {
            "kind": "oracle",
            "num_control_qubits": 0,
            "num_target_qubits": num_targets or 0,
            "custom_name": self.name,
            "gate_type": "CUSTOM",
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        signature = (
            self.signature.to_ir_signature()
            if self.signature is not None
            else signature_from_values(
                [consumed.value],
                [result],
                operand_names=["qubits"],
                result_names=["qubits"],
            )
        )
        op = InvokeOperation(
            operands=[consumed.value],
            results=[result],
            target=oracle_ref,
            attrs=attrs,
            definition=CallableDef(
                ref=oracle_ref,
                signature=signature,
                resource_models=_oracle_resource_models(self.resource_model),
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )
        get_current_tracer().add_operation(op)
        consumed_any: Any = consumed
        if isinstance(consumed_any, VectorView):
            new_view = VectorView._wrap_unregistered(
                parent=consumed_any._slice_parent,
                sliced_av=result,
                length=consumed_any.shape[0],
                start_uint=consumed_any._slice_start,
                step_uint=consumed_any._slice_step,
            )
            consumed_any._transfer_borrow_to(new_view, f"Oracle[{self.name}]")
            return new_view
        return type(qubits)._create_from_value(value=result, shape=qubits.shape)

    def _call_scalars(
        self,
        *qubits: Qubit | Vector[Qubit],
        controls: Sequence[Qubit] = (),
    ) -> tuple[Qubit, ...]:
        """Apply the oracle to scalar qubits.

        Args:
            *qubits (Qubit | Vector[Qubit] | VectorView[Qubit]): Scalar qubit
                handles. Vector arguments are rejected here.
            controls (Sequence[Qubit]): Explicit scalar controls.

        Returns:
            tuple[Qubit, ...]: Next-version scalar qubits.

        Raises:
            TypeError: If any argument is not a scalar qubit.
            ValueError: If the number of scalar qubits is wrong.
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
        if len(controls) != self.num_control_qubits:
            raise ValueError(
                f"Oracle '{self.name}' requires {self.num_control_qubits} "
                f"control qubits, got {len(controls)}."
            )
        if not all(isinstance(q, Qubit) for q in qubits):
            raise TypeError("Oracle scalar calls accept only Qubit arguments.")
        if not all(isinstance(c, Qubit) for c in controls):
            raise TypeError("Oracle controls accept only Qubit arguments.")

        consumed_controls = [
            c.consume(operation_name=f"Oracle[{self.name}][control]") for c in controls
        ]
        consumed = [
            q.consume(operation_name=f"Oracle[{self.name}][target]") for q in qubits
        ]
        results = [q.value.next_version() for q in [*consumed_controls, *consumed]]
        oracle_ref = CallableRef(namespace="user.oracle", name=self.name)
        attrs = {
            "kind": "oracle",
            "num_control_qubits": len(consumed_controls),
            "num_target_qubits": len(consumed),
            "custom_name": self.name,
            "gate_type": "CUSTOM",
            "default_policy": CallPolicy.PRESERVE_BOX.name,
        }
        transform = (
            CallTransform.CONTROLLED if consumed_controls else CallTransform.DIRECT
        )
        signature = (
            self.signature.to_ir_signature()
            if self.signature is not None
            else signature_from_values(
                [q.value for q in [*consumed_controls, *consumed]],
                results,
                operand_names=[
                    *[f"control_{i}" for i in range(len(consumed_controls))],
                    *[f"target_{i}" for i in range(len(consumed))],
                ],
                result_names=[
                    *[f"control_{i}" for i in range(len(consumed_controls))],
                    *[f"target_{i}" for i in range(len(consumed))],
                ],
            )
        )
        op = InvokeOperation(
            operands=[q.value for q in [*consumed_controls, *consumed]],
            results=results,
            target=oracle_ref,
            transform=transform,
            attrs=attrs,
            definition=CallableDef(
                ref=oracle_ref,
                signature=signature,
                resource_models=_oracle_resource_models(self.resource_model),
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )
        get_current_tracer().add_operation(op)

        consumed_qubits = [*consumed_controls, *consumed]
        return tuple(
            Qubit(value=result, parent=qubit.parent, indices=qubit.indices)
            for result, qubit in zip(results, consumed_qubits, strict=True)
        )


def opaque(
    name: str,
    num_qubits: int | None = None,
    *,
    num_control_qubits: int = 0,
    signature: CallableSignature | None = None,
    resource_model: Any | None = None,
) -> Oracle:
    """Create an opaque callable for top-down circuit design.

    Args:
        name (str): Human-readable callable name.
        num_qubits (int | None): Number of target qubits consumed and returned
            by the callable. Defaults to ``None`` when ``signature`` carries
            the shape contract.
        num_control_qubits (int): Number of explicit scalar control qubits
            required by scalar calls. Defaults to ``0``.
        signature (CallableSignature | None): Optional frontend signature.
            Defaults to ``None``.
        resource_model (Any | None): Optional context-aware resource model used
            by symbolic resource estimation. Defaults to ``None``.

    Returns:
        Oracle: Opaque callable backed by ``InvokeOperation`` with no body.
    """
    return Oracle(
        name=name,
        num_qubits=num_qubits,
        num_control_qubits=num_control_qubits,
        signature=signature,
        resource_model=resource_model,
    )


def _oracle_resource_models(model: Any | None) -> list[ResourceModelBinding]:
    """Normalize an optional oracle resource model.

    Args:
        model (Any | None): Model object implementing ``estimate(ctx)``.

    Returns:
        list[ResourceModelBinding]: Callable definition resource models.
    """
    if model is None:
        return []
    return [ResourceModelBinding(model=model)]
