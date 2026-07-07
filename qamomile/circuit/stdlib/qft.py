"""Implement Quantum Fourier Transform stdlib callables.

The public qkernel-facing entry points are :func:`qft` and :func:`iqft`. They
emit named callables that carry a standard Qamomile body and a native-first
lowering policy. The ``QFT`` and ``IQFT`` classes are the
advanced strategy-backed implementation objects behind those functions; custom
user-defined named operations should normally use
``qamomile.circuit.composite_gate`` rather than subclassing these patterns
directly.

Multiple decomposition strategies are available:
    - "standard": Full precision QFT with O(n^2) gates
    - "approximate": Truncated rotations with O(n*k) gates (default k=3)
    - "approximate_k2": Truncated rotations with k=2

Example:
    import qamomile.circuit as qmc

    @qmc.qkernel
    def my_algorithm(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        qubits = qmc.qft(qubits)
        # ... some operations ...
        return qmc.iqft(qubits)

    # Advanced strategy selection remains available through the class API.
    qft_gate = QFT(3)
    result = qft_gate(q0, q1, q2, strategy="approximate")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Qubit, Vector, VectorView
from qamomile.circuit.frontend.handle.utils import get_size as _get_size
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableBodyRef,
    CallableDef,
    CallableRef,
    CallPolicy,
    CompositeGateType,
    InvokeOperation,
    signature_from_values,
)

# Import strategies
from qamomile.circuit.stdlib.qft_strategies import (
    ApproximateIQFTStrategy,
    ApproximateQFTStrategy,
    StandardIQFTStrategy,
    StandardQFTStrategy,
)

if TYPE_CHECKING:
    pass


class QFT(CompositeGate):
    """Quantum Fourier Transform composite gate.

    The QFT is the quantum analog of the discrete Fourier transform.
    It's a key component of many quantum algorithms.

    Available strategies:
        - "standard": Full precision QFT (default)
        - "approximate": Truncated rotations (k=3)
        - "approximate_k2": Truncated rotations (k=2)

    Example:
        # Create QFT for 3 qubits
        qft_gate = QFT(3)

        # Apply to qubits (uses default strategy)
        result = qft_gate(q0, q1, q2)

        # Apply with specific strategy
        result = qft_gate(q0, q1, q2, strategy="approximate")

        # Or use the factory function
        qubits = qft(qubit_vector)
    """

    gate_type = CompositeGateType.QFT
    custom_name = "qft"

    # Initialize strategy registry for QFT
    _strategies = {
        "standard": StandardQFTStrategy(),
        "approximate": ApproximateQFTStrategy(truncation_depth=3),
        "approximate_k2": ApproximateQFTStrategy(truncation_depth=2),
        "approximate_k4": ApproximateQFTStrategy(truncation_depth=4),
    }
    _default_strategy = "standard"

    def __init__(self, num_qubits: int):
        """Initialize QFT gate.

        Args:
            num_qubits: Number of qubits for the QFT
        """
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target qubits."""
        return self._num_qubits

    def _decompose(
        self,
        qubits: Vector[Qubit] | tuple[Qubit, ...],
    ) -> Vector[Qubit] | tuple[Qubit, ...] | None:
        """Decompose QFT into elementary gates.

        Args:
            qubits: Tuple of input qubits

        Returns:
            Tuple of output qubits after QFT transformation
        """
        n = self._num_qubits
        qubits_list = list(qubits)

        # Apply QFT rotations (from highest index to lowest)
        for j in range(n - 1, -1, -1):
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

            # Apply controlled phase rotations
            for k in range(j - 1, -1, -1):
                angle = math.pi / (2 ** (j - k))
                qubits_list[j], qubits_list[k] = qmc.cp(
                    qubits_list[j], qubits_list[k], angle
                )

        # Swap qubits to reverse order
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        return tuple(qubits_list)

class IQFT(CompositeGate):
    """Inverse Quantum Fourier Transform composite gate.

    The IQFT is the inverse of the QFT. It's a key component of:
    - Quantum Phase Estimation (QPE)
    - Shor's algorithm
    - Quantum counting

    Available strategies:
        - "standard": Full precision IQFT (default)
        - "approximate": Truncated rotations (k=3)
        - "approximate_k2": Truncated rotations (k=2)

    Example:
        # Create IQFT for 3 qubits
        iqft_gate = IQFT(3)

        # Apply to qubits
        result = iqft_gate(q0, q1, q2)

        # Apply with specific strategy
        result = iqft_gate(q0, q1, q2, strategy="approximate")

        # Or use the factory function
        qubits = iqft(qubit_vector)
    """

    gate_type = CompositeGateType.IQFT
    custom_name = "iqft"

    # Initialize strategy registry for IQFT
    _strategies = {
        "standard": StandardIQFTStrategy(),
        "approximate": ApproximateIQFTStrategy(truncation_depth=3),
        "approximate_k2": ApproximateIQFTStrategy(truncation_depth=2),
        "approximate_k4": ApproximateIQFTStrategy(truncation_depth=4),
    }
    _default_strategy = "standard"

    def __init__(self, num_qubits: int):
        """Initialize IQFT gate.

        Args:
            num_qubits: Number of qubits for the IQFT
        """
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        """Return the number of target qubits."""
        return self._num_qubits

    def _decompose(
        self,
        qubits: Vector[Qubit] | tuple[Qubit, ...],
    ) -> Vector[Qubit] | tuple[Qubit, ...] | None:
        """Decompose IQFT into elementary gates.

        Args:
            qubits: Tuple of input qubits

        Returns:
            Tuple of output qubits after IQFT transformation
        """
        n = self._num_qubits
        qubits_list = list(qubits)

        # Swap qubits to reverse order
        for j in range(n // 2):
            qubits_list[j], qubits_list[n - j - 1] = qmc.swap(
                qubits_list[j], qubits_list[n - j - 1]
            )

        # Apply inverse QFT rotations (from lowest index to highest)
        for j in range(n):
            # Apply inverse controlled phase rotations first
            for k in range(j):
                angle = -math.pi / (2 ** (j - k))
                qubits_list[j], qubits_list[k] = qmc.cp(
                    qubits_list[j], qubits_list[k], angle
                )
            # Apply H gate
            qubits_list[j] = qmc.h(qubits_list[j])

        return tuple(qubits_list)

@overload
def qft(qubits: VectorView[Qubit]) -> VectorView[Qubit]: ...
@overload
def qft(qubits: Vector[Qubit]) -> Vector[Qubit]: ...


def _emit_vector_qft_invoke(
    qubits: Vector[Qubit],
    *,
    gate_type: CompositeGateType,
    name: str,
) -> Vector[Qubit]:
    """Emit a vector-width QFT/IQFT invoke for symbolic-shape registers.

    Args:
        qubits (Vector[Qubit]): Vector register whose length is not concrete
            during the current trace.
        gate_type (CompositeGateType): The stdlib callable classification to
            record on the invocation.
        name (str): Stable stdlib callable name, such as ``"qft"`` or
            ``"iqft"``.

    Returns:
        Vector[Qubit]: Next-version vector handle backed by the invoke result.
    """
    consumed = qubits.consume(operation_name=f"{name}[target]")
    result = consumed.value.next_version()
    callable_ref = CallableRef(namespace="qamomile.stdlib", name=name)
    body_ref = CallableBodyRef(
        ref=callable_ref,
        kind="symbolic_vector",
        attrs={
            "gate_type": gate_type.name,
            "width_source": "operand_shape",
        },
    )
    attrs = {
        "kind": "composite",
        "gate_type": gate_type.name,
        "num_control_qubits": 0,
        "num_target_qubits": 0,
        "custom_name": name,
        "strategy_name": None,
        "default_policy": CallPolicy.NATIVE_FIRST.name,
    }
    op = InvokeOperation(
        operands=[consumed.value],
        results=[result],
        target=callable_ref,
        attrs=attrs,
        definition=CallableDef(
            ref=callable_ref,
            signature=signature_from_values(
                [consumed.value],
                [result],
                operand_names=["qubits"],
                result_names=["qubits"],
            ),
            body=None,
            body_ref=body_ref,
            default_policy=CallPolicy.NATIVE_FIRST,
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
        consumed_any._transfer_borrow_to(new_view, name)
        return new_view
    return type(qubits)._create_from_value(value=result, shape=qubits.shape)


def qft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply Quantum Fourier Transform to a vector of qubits.

    This is a convenience factory function that creates a QFT gate
    and applies it to the qubits.

    When *qubits* has a concrete (compile-time known) shape this emits
    the standard ``O(n^2)`` QFT decomposition. When *qubits* is a
    sub-kernel parameter whose shape is still symbolic at trace time,
    the function emits a boxed stdlib ``InvokeOperation`` over the
    vector operand. Later emit/resource passes resolve the vector
    width from the call-site shape instead of silently dropping the
    operation.

    Args:
        qubits (Vector[Qubit]): Vector of qubits to transform.

    Returns:
        Vector[Qubit]: Transformed qubits (same vector, modified in
            place).

    Example:
        @qmc.qkernel
        def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = qft(qubits)
            return qubits
    """
    try:
        n = _get_size(qubits)
    except ValueError:
        return _emit_vector_qft_invoke(
            qubits,
            gate_type=CompositeGateType.QFT,
            name="qft",
        )
    qft_gate = QFT(n)

    # Get individual qubits from vector
    qubit_list = [qubits[i] for i in range(n)]

    # Apply QFT gate
    result = qft_gate(*qubit_list)

    # Write results back to vector
    for i in range(n):
        qubits[i] = result[i]

    return qubits


@overload
def iqft(qubits: VectorView[Qubit]) -> VectorView[Qubit]: ...
@overload
def iqft(qubits: Vector[Qubit]) -> Vector[Qubit]: ...
def iqft(qubits: Vector[Qubit]) -> Vector[Qubit]:
    """Apply Inverse Quantum Fourier Transform to a vector of qubits.

    This is a convenience factory function that creates an IQFT gate
    and applies it to the qubits.

    The same symbolic-shape contract as :func:`qft` applies here:
    when *qubits* has no compile-time-known shape, the function emits
    a boxed stdlib ``InvokeOperation`` over the vector operand so the
    operation remains visible to later compiler stages.

    Args:
        qubits (Vector[Qubit]): Vector of qubits to transform.

    Returns:
        Vector[Qubit]: Transformed qubits (same vector, modified in
            place).

    Example:
        @qmc.qkernel
        def my_algorithm(qubits: Vector[Qubit]) -> Vector[Qubit]:
            qubits = iqft(qubits)
            return qubits
    """
    try:
        n = _get_size(qubits)
    except ValueError:
        return _emit_vector_qft_invoke(
            qubits,
            gate_type=CompositeGateType.IQFT,
            name="iqft",
        )
    iqft_gate = IQFT(n)

    # Get individual qubits from vector
    qubit_list = [qubits[i] for i in range(n)]

    # Apply IQFT gate
    result = iqft_gate(*qubit_list)

    # Write results back to vector
    for i in range(n):
        qubits[i] = result[i]

    return qubits
