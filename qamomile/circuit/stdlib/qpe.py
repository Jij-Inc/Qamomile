"""Quantum Phase Estimation implementation.

Example:
    @qmc.qkernel
    def p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
        return qmc.p(q, theta)

    @qmc.qkernel
    def circuit(theta: float) -> qmc.Float:
        counting = qmc.qubit_array(3, name="counting")
        target = qmc.qubit(name="target")
        target = qmc.x(target)

        phase = qmc.qpe(target, counting, p_gate, theta=theta)
        return qmc.measure(phase)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import QFixed, Qubit, Vector
from qamomile.circuit.frontend.operation.control_flow import for_loop
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CompositeGateType,
    InvokeOperation,
    signature_from_values,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.types import QFixedType, QubitType
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel_like import QKernelLike


def qpe(
    target: Qubit,
    counting: Vector[Qubit],
    unitary: "QKernelLike",
    **params: Any,
) -> QFixed:
    """Quantum Phase Estimation.

    Estimates the phase φ where U|ψ> = e^{2πiφ}|ψ>.

    Args:
        target: The eigenstate |ψ> of the unitary U
        counting: Vector of qubits for phase estimation result
        unitary: The unitary operation U as a qkernel-like callable
        **params: Parameters to pass to the unitary

    Returns:
        QFixed: Phase register as quantum fixed-point number
    """
    n = counting.shape[0]  # UInt handle (symbolic or concrete)
    controlled_u = cast(Callable[..., tuple[Any, ...]], qmc.control(unitary))

    # 1. Hadamard gates on counting qubits
    with for_loop(0, n, var_name="i") as i:
        counting[i] = qmc.h(counting[i])

    # 2. Controlled-U^(2^k) operations
    # Uses power parameter to emit Controlled(U^(2^k)), NOT Controlled(U)^(2^k)
    with for_loop(0, n, var_name="k") as k:
        counting[k], target = controlled_u(counting[k], target, power=2**k, **params)

    # 3. Inverse QFT (emit as InvokeOperation for native backend support)
    # Returns QFixed directly (bypassing cast) to ensure correct UUID mapping
    return _emit_iqft_and_cast_to_qfixed(counting)


def _emit_iqft_and_cast_to_qfixed(qubits: Vector[Qubit]) -> QFixed:
    """Emit IQFT and create QFixed directly with correct UUID mapping.

    This function combines IQFT emission and QFixed creation to ensure
    the qubit UUIDs are correctly tracked through the transformation.

    Instead of calling the iqft qkernel, this emits a single InvokeOperation
    that the backend can render as a native inverse QFT when available.
    """
    n = qubits.shape[0]

    # Get concrete size for the IQFT invoke metadata.  If the size is
    # symbolic, keep the operation as a vector-operand callable and cast
    # the resulting vector as a deferred QFixed alias instead of falling
    # back to UInt.init_value (which is 0 for symbolic handles).
    concrete_n = _get_concrete_size(n)
    if concrete_n is None:
        from qamomile.circuit.frontend.operation.cast import cast
        from qamomile.circuit.stdlib.qft import iqft

        transformed = iqft(qubits)
        return cast(transformed, QFixed, int_bits=0)

    # Collect input qubit values - borrow elements once
    qubit_handles = [qubits[i] for i in range(concrete_n)]
    operands = [h.value for h in qubit_handles]

    # Create result values (new versions) - these will be the qubit UUIDs for QFixed
    iqft_results = [v.next_version() for v in operands]

    # Return borrowed elements by assigning back to the array
    for i, h in enumerate(qubit_handles):
        qubits[i] = h

    iqft_body = None
    if concrete_n > 0:
        from qamomile.circuit.stdlib.qft import IQFT

        dummy_qubits = tuple(
            Qubit(value=Value(type=QubitType(), name=f"_iqft_q{i}"))
            for i in range(concrete_n)
        )
        iqft_body = IQFT(concrete_n)._build_decomposition_block(dummy_qubits)

    # Create InvokeOperation for IQFT
    iqft_ref = CallableRef(namespace="qamomile.stdlib", name="iqft")
    attrs = {
        "kind": "composite",
        "gate_type": CompositeGateType.IQFT.name,
        "num_control_qubits": 0,
        "num_target_qubits": concrete_n,
        "custom_name": "iqft",
        "strategy_name": None,
        "default_policy": CallPolicy.NATIVE_FIRST.name,
    }
    iqft_op = InvokeOperation(
        operands=operands,
        results=iqft_results,
        target=iqft_ref,
        attrs=attrs,
        definition=CallableDef(
            ref=iqft_ref,
            signature=signature_from_values(
                operands,
                iqft_results,
                operand_names=[f"qubit_{i}" for i in range(len(operands))],
                result_names=[f"qubit_{i}" for i in range(len(iqft_results))],
            ),
            body=iqft_body,
            default_policy=CallPolicy.NATIVE_FIRST,
            attrs=attrs,
        ),
    )

    tracer = get_current_tracer()
    tracer.add_operation(iqft_op)

    # Now create QFixed directly using the IQFT result UUIDs
    int_bits = 0
    qubit_uuids = [r.uuid for r in iqft_results]
    qubit_logical_ids = [r.logical_id for r in iqft_results]

    result_type = QFixedType()
    result_value = (
        Value(
            type=result_type,
            name=f"{qubits.value.name}_as_qfixed",
        )
        .with_cast_metadata(
            source_uuid=qubits.value.uuid,
            source_logical_id=qubits.value.logical_id,
            qubit_uuids=qubit_uuids,
            qubit_logical_ids=qubit_logical_ids,
        )
        .with_qfixed_metadata(
            qubit_uuids=qubit_uuids,
            num_bits=concrete_n,
            int_bits=int_bits,
        )
    )

    # Create CastOperation (for IR completeness, though we've already set up UUIDs)
    cast_op = CastOperation(
        operands=[qubits.value],
        results=[result_value],
        source_type=qubits.value.type,
        target_type=result_type,
        qubit_mapping=qubit_uuids,
    )
    tracer.add_operation(cast_op)

    return QFixed(value=result_value)


def _get_concrete_size(size: int | Any) -> int | None:
    """Get array size as Python int for IQFT invocation metadata.

    Args:
        size: Array size, either an int or a handle with value information.

    Returns:
        int | None: Concrete integer size, or ``None`` when the size is
        genuinely symbolic.
    """
    if isinstance(size, int):
        return size
    if hasattr(size, "value") and size.value.is_constant():
        val = size.value.get_const()
        if val is not None:
            return int(val)
    return None
