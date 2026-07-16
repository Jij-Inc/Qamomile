"""Frontend ``qmc.select``: the quantum multiplexer (SELECT) gate.

``qmc.select([U_0, U_1, ...], num_index_qubits=...)`` builds a
:class:`SelectGate` that applies ``U_i`` to a shared target register when an
index (select) register reads the integer ``i``::

    sel = qmc.select([qmc.x, qmc.y, qmc.z, qmc.h])
    idx_out, tgt_out = sel(index_register, target)

With the default ``num_index_qubits=None``, the width is inferred as
``ceil(log2(len(cases)))``. An explicit ``int`` may be wider, leaving the
extra index states as identity, while ``UInt`` defers the width to transpile
time. Leading positional ``Qubit`` / ``Vector`` / ``VectorView`` arguments
form the index prefix; the shared case signature identifies the trailing
target and parameter arguments. Index bit order follows Qamomile's LSB-first
convention: the first flattened index qubit is bit zero.

The frontend reuses ``qmc.control``'s operand / result machinery (so every
control-prefix and target handle pattern is supported identically) but emits a
single :class:`SelectOperation`. Circuit-family lowering preserves the SELECT
identity as a reusable call and keeps each case as a controlled fallback call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from qamomile.circuit.frontend.handle.primitives import UInt
from qamomile.circuit.frontend.qkernel_specialization import select_specialized_block
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation, ResetOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
)
from qamomile.circuit.ir.operation.select import SelectOperation

from .control import ControlledGate, _qkernel_for_callable

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


def _num_index_qubits_for(num_cases: int) -> int:
    """Return the minimal index-qubit count addressing ``num_cases`` cases.

    Args:
        num_cases (int): Number of selectable cases (``>= 2``).

    Returns:
        int: ``ceil(log2(num_cases))`` — the number of index qubits whose
            ``2 ** k`` basis states cover all cases.

    Raises:
        ValueError: If ``num_cases < 2``.
    """
    if num_cases < 2:
        raise ValueError(
            f"qmc.select requires at least 2 cases, got {num_cases}. A "
            f"single case is just an unconditional unitary — apply it "
            f"directly instead of wrapping it in a select."
        )
    return (num_cases - 1).bit_length()


def _normalize_num_index_qubits(
    num_cases: int,
    num_index_qubits: int | UInt | None,
) -> int | UInt:
    """Normalize an inferred, concrete, or symbolic SELECT index width.

    Args:
        num_cases (int): Number of SELECT cases, which must be at least two.
        num_index_qubits (int | UInt | None): Requested index width. ``None``
            infers the minimal width, an ``int`` may be wider, and ``UInt``
            defers width validation until transpilation.

    Returns:
        int | UInt: Minimal or explicitly requested index width.

    Raises:
        TypeError: If the requested width is neither a plain ``int``,
            ``UInt``, nor ``None``.
        ValueError: If a concrete width cannot address every case.
    """
    minimum = _num_index_qubits_for(num_cases)
    if num_index_qubits is None:
        return minimum
    if isinstance(num_index_qubits, bool) or not isinstance(
        num_index_qubits,
        (int, UInt),
    ):
        raise TypeError(
            "num_index_qubits must be a Python int, UInt, or None; "
            f"got {type(num_index_qubits).__name__}."
        )
    if isinstance(num_index_qubits, int) and num_index_qubits < minimum:
        raise ValueError(
            f"num_index_qubits={num_index_qubits} cannot address "
            f"{num_cases} cases; at least {minimum} index qubit(s) are "
            f"required."
        )
    return num_index_qubits


def _signature_key(qkernel: "Any") -> tuple[tuple[str, Any, Any], ...]:
    """Return a comparable signature key for a wrapped case kernel.

    Two cases are select-compatible only when their parameter lists match
    in name, type, order, **and Python default value**, because
    ``qmc.select`` binds the call once (against the first case, applying
    *its* defaults) and forwards the resulting argument values to every
    case block. If two cases declared the same parameter with different
    defaults, a caller that omitted that parameter would silently get the
    first case's default applied to all cases — so differing defaults are
    rejected up front by including them in the key.

    Args:
        qkernel (Any): A wrapped case kernel exposing ``input_types`` and
            an ``inspect.Signature`` ``signature``.

    Returns:
        tuple[tuple[str, Any, Any], ...]: The ``(name, annotation,
            default)`` triples of the kernel's inputs, in declaration
            order. The default is ``inspect.Parameter.empty`` for
            parameters without one.
    """
    signature = qkernel.signature
    return tuple(
        (name, annotation, signature.parameters[name].default)
        for name, annotation in qkernel.input_types.items()
    )


def _default_values_equal(left: Any, right: Any) -> bool:
    """Compare Python parameter defaults without ambiguous array truth values.

    Args:
        left (Any): Default value from the reference case.
        right (Any): Default value from another case.

    Returns:
        bool: Whether the defaults are structurally equal.
    """
    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        try:
            return bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left, right))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _default_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_default_values_equal(left[key], right[key]) for key in left)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _signature_keys_equal(
    left: tuple[tuple[str, Any, Any], ...],
    right: tuple[tuple[str, Any, Any], ...],
) -> bool:
    """Compare case signature keys with structural default-value equality.

    Args:
        left (tuple[tuple[str, Any, Any], ...]): Reference signature key.
        right (tuple[tuple[str, Any, Any], ...]): Candidate signature key.

    Returns:
        bool: Whether names, annotations, order, and defaults all match.
    """
    if len(left) != len(right):
        return False
    return all(
        left_name == right_name
        and left_annotation == right_annotation
        and _default_values_equal(left_default, right_default)
        for (
            left_name,
            left_annotation,
            left_default,
        ), (
            right_name,
            right_annotation,
            right_default,
        ) in zip(left, right, strict=True)
    )


def _quantum_logical_ids(values: Sequence[Any]) -> tuple[Any, ...]:
    """Return the ordered logical IDs of quantum values in a sequence.

    Args:
        values (Sequence[Any]): ``Value`` s (a case block's inputs or
            outputs). Each carries a ``logical_id`` identifying the logical
            qubit wire it belongs to, stable across SSA re-versioning.

    Returns:
        tuple[Any, ...]: The ``logical_id`` of every quantum ``Value`` in
            positional order (scalar ``Qubit`` and whole ``Vector[Qubit]``
            wires alike). Classical values are ignored.
    """
    return tuple(value.logical_id for value in values if value.type.is_quantum())


def _validate_case_target_footprint(case_blocks: Sequence[Any]) -> None:
    """Reject cases that do not return exactly the target register they got.

    A SELECT applies each case to the same shared target register, so every
    case must be a unitary on that register: it must return exactly the
    quantum wires it received, neither dropping a target nor allocating and
    returning fresh quantum state. ``_signature_key`` (checked in
    ``__init__``) only pins the quantum *inputs*, so a case could still
    return a different quantum footprint — e.g. allocate an ancilla and
    return it while dropping the input target — which keeps the same output
    *count* yet miswires against the SELECT's shared result list.

    Each case is checked against ITS OWN inputs by comparing the ordered
    quantum logical IDs: a proper unitary threads each input position through
    to the matching output position (re-versioned, same ``logical_id``).
    Positional equality rejects duplicates, dropped/fresh wires, and a bare
    return-value permutation such as ``return q1, q0``. The latter is a free
    handle relabel for an ordinary qkernel call but cannot be made conditional
    on a quantum SELECT index; callers must express it as an explicit SWAP
    gate instead. Internal ancillas that are allocated and uncomputed (not
    returned) are rejected today because the controlled fallback has no
    operation-owned ancilla contract.

    Args:
        case_blocks (Sequence[Any]): The specialized case blocks, in index
            order, each exposing ``input_values`` and ``output_values``.

    Returns:
        None: This function returns nothing; it raises on mismatch.

    Raises:
        ValueError: If any case block's quantum output wires are not exactly
            the quantum input wires it received, or if its reachable body
            contains non-unitary behavior or internal ancilla allocation.
    """
    for position, block in enumerate(case_blocks):
        quantum_inputs = [
            value for value in block.input_values if value.type.is_quantum()
        ]
        if len(block.output_values) != len(quantum_inputs) or any(
            not value.type.is_quantum() for value in block.output_values
        ):
            raise ValueError(
                f"qmc.select case {position} is not a unitary on the target "
                f"register: it must return exactly its {len(quantum_inputs)} "
                f"quantum input wire(s) and no classical outputs, but returns "
                f"{len(block.output_values)} value(s)."
            )
        input_ids = _quantum_logical_ids(block.input_values)
        output_ids = _quantum_logical_ids(block.output_values)
        if input_ids != output_ids:
            raise ValueError(
                f"qmc.select case {position} is not a unitary on the target "
                f"register: its ordered quantum outputs do not match its "
                f"ordered quantum inputs ({len(output_ids)} output value(s), "
                f"{len(input_ids)} input value(s)). Every case must thread "
                f"each target position through unchanged — allocate/drop/"
                f"replace patterns and bare return-value permutations are "
                f"not valid selectable unitaries; use explicit gates such "
                f"as qmc.swap for physical permutations."
            )
        _validate_case_operations_are_unitary(block, position)


def _validate_case_operations_are_unitary(block: Block, position: int) -> None:
    """Reject non-unitary behavior and internal ancillas in a SELECT case.

    Classical arithmetic and control-flow nodes are allowed because a
    compile-time-resolvable branch may parameterize an otherwise unitary case;
    the normal partial-evaluation pipeline lowers it before emission. Hybrid
    operations and reset are intrinsically non-unitary and cannot appear in a
    quantum multiplexer case. Internal allocation is also rejected because a
    SELECT case has no operation-owned ancilla ABI. Operation-owned
    callable/control/inverse/select bodies are checked recursively so boxing
    cannot hide a violation.

    Args:
        block (Block): Case block to inspect recursively.
        position (int): Case index used in diagnostics.

    Returns:
        None: The function returns nothing when the case is unitary.

    Raises:
        ValueError: If a reachable operation is hybrid, resets a qubit, or
            allocates an internal ancilla.
    """
    seen_blocks: set[int] = set()

    def visit_block(candidate: Block) -> None:
        """Inspect one block once and recurse through owned bodies.

        Args:
            candidate (Block): Block to inspect.

        Returns:
            None: The function updates ``seen_blocks`` and raises on failure.

        Raises:
            ValueError: If a non-unitary or unsupported operation is found.
        """
        block_id = id(candidate)
        if block_id in seen_blocks:
            return
        seen_blocks.add(block_id)
        visit_operations(candidate.operations)

    def visit_operations(operations: Sequence[Operation]) -> None:
        """Inspect an operation list and every nested operation/body.

        Args:
            operations (Sequence[Operation]): Operations to inspect.

        Returns:
            None: The function raises on the first non-unitary operation.

        Raises:
            ValueError: If a non-unitary or unsupported operation is found.
        """
        for operation in operations:
            if isinstance(operation, QInitOperation):
                raise ValueError(
                    f"qmc.select case {position} contains internal "
                    f"QInitOperation; SELECT cases cannot allocate ancillas."
                )
            if operation.operation_kind is OperationKind.HYBRID or isinstance(
                operation, ResetOperation
            ):
                raise ValueError(
                    f"qmc.select case {position} contains non-unitary "
                    f"{type(operation).__name__}; SELECT cases may contain "
                    f"only unitary quantum behavior."
                )
            if isinstance(operation, HasNestedOps):
                for nested in operation.nested_op_lists():
                    visit_operations(nested)
            if isinstance(operation, InvokeOperation):
                definition = operation.definition
                if definition is not None:
                    if definition.body is not None:
                        visit_block(definition.body)
                    for implementation in definition.implementations:
                        if implementation.body is not None:
                            visit_block(implementation.body)
            if (
                isinstance(operation, ControlledUOperation)
                and operation.block is not None
            ):
                visit_block(operation.block)
            if isinstance(operation, InverseBlockOperation):
                if operation.source_block is not None:
                    visit_block(operation.source_block)
                if operation.implementation_block is not None:
                    visit_block(operation.implementation_block)
            if isinstance(operation, SelectOperation):
                for case_block in operation.case_blocks:
                    visit_block(case_block)

    visit_block(block)


class SelectGate:
    """Callable wrapper for a quantum multiplexer over a list of unitaries.

    Created by :func:`select`. Calling the instance applies case ``i`` to
    the target register controlled on the index register reading ``i``.

    Args:
        cases (Sequence[QKernel | Callable[..., Any]]): Case unitaries in
            ascending index order. Every case must expose the same signature.
        num_index_qubits (int | UInt | None): Number of leading index qubits.
            ``None`` infers the minimal width. Defaults to ``None``.

    Attributes:
        num_index_qubits (int | UInt): LSB-first index-register width.
        num_cases (int): Number of selectable unitary cases.

    Example:
        >>> import qamomile.circuit as qm
        >>> @qm.qkernel
        ... def demo() -> qm.Bit:
        ...     idx = qm.qubit_array(1, name="idx")
        ...     t = qm.qubit(name="t")
        ...     idx, t = qm.select([qm.x, qm.h])(idx, t)
        ...     return qm.measure(t)
    """

    def __init__(
        self,
        cases: Sequence["QKernel | Callable[..., Any]"],
        num_index_qubits: int | UInt | None = None,
    ) -> None:
        """Wrap and validate the case unitaries.

        Args:
            cases (Sequence[QKernel | Callable[..., Any]]): The case
                unitaries in ascending index order. Each may be a
                ``@qmc.qkernel`` function, a qkernel-backed composite gate,
                or a built-in gate callable (``qmc.x``, ``qmc.ry``, ...).
                All cases must share the same parameter signature (name,
                type, and order).
            num_index_qubits (int | UInt | None): Number of index qubits.
                ``None`` infers ``ceil(log2(len(cases)))``. A wider concrete
                value leaves unassigned basis states as identity. ``UInt``
                defers the width and flattened-prefix check to transpilation.
                Defaults to ``None``.

        Raises:
            ValueError: If fewer than two cases are supplied, a concrete
                width is too small, or the cases do not all share an identical
                parameter signature. Case-body footprint and unitarity are
                validated when the gate is called.
            TypeError: If the width has an unsupported type or a case cannot
                be wrapped into a qkernel.
        """
        case_list = list(cases)
        self._num_index_qubits = _normalize_num_index_qubits(
            len(case_list),
            num_index_qubits,
        )

        wrapped = [_qkernel_for_callable(c, caller="select") for c in case_list]
        reference_key = _signature_key(wrapped[0])
        for position, kernel in enumerate(wrapped[1:], start=1):
            candidate_key = _signature_key(kernel)
            if not _signature_keys_equal(candidate_key, reference_key):
                raise ValueError(
                    f"qmc.select requires every case to share the same "
                    f"parameter signature; case {position} has signature "
                    f"{candidate_key!r} which differs from case 0's "
                    f"{reference_key!r}. Selected unitaries act on the same "
                    f"target register and receive the same forwarded "
                    f"arguments, so their signatures must match."
                )
        self._cases = wrapped
        self._driver = ControlledGate(
            wrapped[0],
            num_controls=self._num_index_qubits,
        )

    @property
    def num_index_qubits(self) -> int | UInt:
        """Number of index (select) qubits this multiplexer expects.

        Returns:
            int | UInt: Inferred/concrete width or the symbolic width handle.
        """
        return self._num_index_qubits

    @property
    def num_cases(self) -> int:
        """Number of selectable cases.

        Returns:
            int: The count of wrapped case unitaries.
        """
        return len(self._cases)

    def __call__(self, *args: Any, **params: Any) -> tuple[Any, ...]:
        """Apply the multiplexer.

        Args:
            *args (Any): The index register followed by the target
                argument(s). For a concrete width, leading quantum arguments
                must contribute exactly ``num_index_qubits`` qubits on an
                argument boundary. For a symbolic width, the shared case
                signature identifies the trailing arguments and every earlier
                ``Qubit`` / ``Vector`` / ``VectorView`` forms the index prefix;
                its flattened width is checked during transpilation. The first
                flattened index qubit is bit zero (LSB).
            **params (Any): Classical parameters forwarded identically to
                every case unitary.

        Returns:
            tuple[Any, ...]: One output handle per input handle, in the
                order ``(index..., targets...)``, each of the same runtime
                kind as its input (scalar ``Qubit`` -> ``Qubit``,
                ``Vector`` -> ``Vector``, ``VectorView`` -> ``VectorView``).

        Raises:
            RuntimeError: If no qkernel tracer is active.
            ValueError: If the index prefix is missing or malformed, concrete
                splitting crosses an argument boundary, no target argument is
                given, or a specialized case is not a supported unitary on
                exactly the shared target register.
            TypeError: On unknown / mistyped forwarded parameters.
            QubitConsumedError: If an index or target qubit was already
                consumed by an earlier operation.
            QubitBorrowConflictError: If index and target array views overlap
                or otherwise conflict in the borrow tracker.
        """
        tracer = get_current_tracer()

        # Reuse ``qmc.control``'s concrete prepare choreography: the index
        # register plays the role of the control prefix and the targets /
        # params play the role of the sub-kernel arguments. The Select /
        # index labels flow into consume tags and boundary errors so misuse
        # diagnostics name the API the caller actually used.
        driver = self._driver
        num_index_qubits = self._num_index_qubits
        if isinstance(num_index_qubits, UInt):
            symbolic_prep = driver._prepare_symbolic(
                args,
                params,
                None,
                operation_label="Select",
                control_role="index",
                allow_single_scalar_prefix=True,
            )
            case_blocks = [
                select_specialized_block(case, symbolic_prep.sub_args_resolved)
                for case in self._cases
            ]
            _validate_case_target_footprint(case_blocks)

            op = SelectOperation(
                operands=symbolic_prep.operands,
                results=symbolic_prep.results,
                num_index_qubits=num_index_qubits.value,
                num_index_args=len(symbolic_prep.prefix_entries),
                case_blocks=case_blocks,
            )
            driver._commit_control_entries(
                symbolic_prep.prefix_entries,
                "Select[index]",
            )
            driver._commit_control_entries(
                symbolic_prep.target_entries,
                "Select[target]",
            )
            tracer.add_operation(op)
            return driver._wrap_symbolic_results_by_input_kind(
                symbolic_prep,
                operation_label="Select",
                control_role="index",
            )

        assert isinstance(num_index_qubits, int)
        prep = driver._prepare_concrete(
            args,
            params,
            num_index_qubits,
            operation_label="Select",
            control_role="index",
        )

        # Specialize every case block for this concrete call site (so a
        # shape-dependent stdlib unitary used as a case does not no-op on
        # its cached symbolic block).
        case_blocks = [
            select_specialized_block(case, prep.sub_args_resolved)
            for case in self._cases
        ]

        # Every case must be a unitary on the SAME target register: it must
        # return exactly the quantum wires it received. ``_signature_key``
        # (checked in ``__init__``) already guarantees matching quantum
        # *inputs*, but a case that allocates and returns an ancilla — or
        # drops a target and returns fresh state of the same count — would
        # keep the same inputs yet return a different quantum footprint,
        # silently miswiring against the SELECT's shared result list (built
        # from case 0). Reject that here.
        _validate_case_target_footprint(case_blocks)

        op = SelectOperation(
            operands=prep.operands,
            results=prep.results,
            num_index_qubits=num_index_qubits,
            case_blocks=case_blocks,
        )
        driver._commit_control_entries(
            prep.control_entries,
            "Select[index]",
        )
        driver._commit_control_entries(
            prep.target_entries,
            "Select[target]",
        )
        tracer.add_operation(op)

        return driver._wrap_results_by_input_kind(
            prep.control_entries,
            prep.target_entries,
            prep.results,
            operation_name="Select",
            control_role="index",
        )


def select(
    cases: Sequence["QKernel | Callable[..., Any]"],
    num_index_qubits: int | UInt | None = None,
) -> SelectGate:
    """Create a quantum multiplexer (SELECT) over a list of unitaries.

    The returned gate applies ``cases[i]`` to a shared target register
    when the index register reads the integer ``i`` with index qubit zero as
    the least-significant bit. ``len(cases)`` need not be a power of
    two; index values ``>= len(cases)`` apply no operation.

    A scalar ``Qubit`` case called with a ``Vector[Qubit]`` or
    ``VectorView[Qubit]`` target is applied independently to every element.
    This is the same tensor-product unitary as an explicit per-element loop,
    including one copy of the scalar case's global phase per element. A phase
    intended for the complete register belongs on a case whose parameter is
    itself ``Vector[Qubit]``.

    Circuit-family lowering retains the abstract SELECT identity while its
    portable fallback invokes each case under the corresponding mixed
    ``0``/``1`` (anti-/normal) index pattern.

    Args:
        cases (Sequence[QKernel | Callable[..., Any]]): The case unitaries
            in ascending index order. Each may be a ``@qmc.qkernel``
            function, a qkernel-backed composite gate, or a built-in gate
            callable. All cases must share the same parameter signature
            and act on the same target register.
        num_index_qubits (int | UInt | None): Number of leading index qubits.
            ``None`` infers the minimal width from the case count. A wider
            concrete value leaves its unassigned index states as identity.
            ``UInt`` defers the width check to transpilation. Defaults to
            ``None``.

    Returns:
        SelectGate: A callable applied as ``sel(index, *targets, **params)``.

    Raises:
        ValueError: If fewer than two cases are supplied, a concrete width is
            too small, or the cases do not share an identical parameter
            signature. Case-body footprint and unitarity are validated when
            the returned gate is called.
        TypeError: If the width has an unsupported type or a case cannot be
            wrapped into a qkernel.

    Example:
        >>> import qamomile.circuit as qm
        >>> @qm.qkernel
        ... def pick() -> qm.Vector[qm.Bit]:
        ...     idx = qm.qubit_array(2, name="idx")
        ...     idx = qm.h(idx)
        ...     t = qm.qubit(name="t")
        ...     idx, t = qm.select([qm.x, qm.y, qm.z, qm.h])(idx, t)
        ...     return qm.measure(idx)
    """
    return SelectGate(cases, num_index_qubits=num_index_qubits)
