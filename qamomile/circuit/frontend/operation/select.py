"""Frontend ``qmc.select``: the quantum multiplexer (SELECT) gate.

``qmc.select([U_0, U_1, ...])`` builds a :class:`SelectGate` that applies
``U_i`` to a shared target register when an index (select) register reads
the integer ``i``::

    sel = qmc.select([qmc.x, qmc.y, qmc.z, qmc.h])
    idx_out, tgt_out = sel(index_register, target)

The leading argument is the index register (a ``Vector[Qubit]`` /
``VectorView[Qubit]`` of length ``ceil(log2(len(cases)))``, or that many
individual ``Qubit`` arguments). Everything after it is forwarded to every
case unitary, exactly like the target / parameter arguments of
``qmc.control``. Index bit order is big-endian: the first index qubit is
the most-significant bit.

The frontend reuses ``qmc.control``'s operand / result machinery (so every
control-prefix and target handle pattern is supported identically) but
emits a single :class:`SelectOperation`, leaving the native-vs-decomposed
choice to the backend emit pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.select import SelectOperation

from .control import ControlledGate, _qkernel_for_callable, _specialized_block_for_call

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


class SelectGate:
    """Callable wrapper for a quantum multiplexer over a list of unitaries.

    Created by :func:`select`. Calling the instance applies case ``i`` to
    the target register controlled on the index register reading ``i``.

    Example:
        >>> import qamomile.circuit as qm
        >>> @qm.qkernel
        ... def demo() -> qm.Bit:
        ...     idx = qm.qubit_array(1, name="idx")
        ...     t = qm.qubit(name="t")
        ...     idx, t = qm.select([qm.x, qm.h])(idx, t)
        ...     return qm.measure(t)
    """

    def __init__(self, cases: Sequence["QKernel | Callable[..., Any]"]) -> None:
        """Wrap and validate the case unitaries.

        Args:
            cases (Sequence[QKernel | Callable[..., Any]]): The case
                unitaries in ascending index order. Each may be a
                ``@qmc.qkernel`` function, a qkernel-backed composite gate,
                or a built-in gate callable (``qmc.x``, ``qmc.ry``, ...).
                All cases must share the same parameter signature (name,
                type, and order).

        Raises:
            ValueError: If fewer than two cases are supplied, or the cases
                do not all share an identical parameter signature.
            TypeError: If a case cannot be wrapped into a kernel (missing
                annotations, unsupported types, or no qubit parameter).
        """
        case_list = list(cases)
        self._num_index_qubits = _num_index_qubits_for(len(case_list))

        wrapped = [_qkernel_for_callable(c, caller="select") for c in case_list]
        reference_key = _signature_key(wrapped[0])
        for position, kernel in enumerate(wrapped[1:], start=1):
            if _signature_key(kernel) != reference_key:
                raise ValueError(
                    f"qmc.select requires every case to share the same "
                    f"parameter signature; case {position} has signature "
                    f"{_signature_key(kernel)!r} which differs from case 0's "
                    f"{reference_key!r}. Selected unitaries act on the same "
                    f"target register and receive the same forwarded "
                    f"arguments, so their signatures must match."
                )
        self._cases = wrapped

    @property
    def num_index_qubits(self) -> int:
        """Number of index (select) qubits this multiplexer expects.

        Returns:
            int: ``ceil(log2(num_cases))``.
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
                argument(s). The leading ``num_index_qubits`` qubits form
                the index (supplied as one ``Vector[Qubit]`` /
                ``VectorView`` of that length, or that many individual
                ``Qubit`` arguments); the rest are bound to every case's
                quantum parameters. The first index qubit is the
                most-significant bit.
            **params (Any): Classical parameters forwarded identically to
                every case unitary.

        Returns:
            tuple[Any, ...]: One output handle per input handle, in the
                order ``(index..., targets...)``, each of the same runtime
                kind as its input (scalar ``Qubit`` -> ``Qubit``,
                ``Vector`` -> ``Vector``, ``VectorView`` -> ``VectorView``).

        Raises:
            ValueError: If the leading qubits cannot supply exactly
                ``num_index_qubits`` index qubits on an argument boundary,
                or no target argument is given.
            TypeError: On unknown / mistyped forwarded parameters.
            QubitConsumedError / QubitBorrowConflictError: On aliasing
                between the index and target arguments.
        """
        # Reuse ``qmc.control``'s concrete prepare choreography: the index
        # register plays the role of the control prefix and the targets /
        # params play the role of the sub-kernel arguments.
        driver = ControlledGate(self._cases[0], num_controls=self._num_index_qubits)
        prep = driver._prepare_concrete(args, params, self._num_index_qubits)

        # Specialize every case block for this concrete call site (so a
        # shape-dependent stdlib unitary used as a case does not no-op on
        # its cached symbolic block).
        case_blocks = [
            _specialized_block_for_call(case, prep.sub_args_resolved)
            for case in self._cases
        ]

        op = SelectOperation(
            operands=prep.operands,
            results=prep.results,
            num_index_qubits=self._num_index_qubits,
            case_blocks=case_blocks,
        )
        get_current_tracer().add_operation(op)

        return driver._wrap_results_by_input_kind(
            prep.consumed_controls,
            prep.consumed_sub_quantum,
            prep.results,
            operation_name="Select",
        )


def select(cases: Sequence["QKernel | Callable[..., Any]"]) -> SelectGate:
    """Create a quantum multiplexer (SELECT) over a list of unitaries.

    The returned gate applies ``cases[i]`` to a shared target register
    when the index register reads the integer ``i`` (big-endian, first
    index qubit most-significant). ``len(cases)`` need not be a power of
    two; index values ``>= len(cases)`` apply no operation.

    At emit time a backend with a native multiplexer primitive can emit a
    single instruction; otherwise the op is decomposed into one
    controlled-U per case, each controlled on the index register with a
    mixed ``0``/``1`` (anti-/normal) control pattern.

    Args:
        cases (Sequence[QKernel | Callable[..., Any]]): The case unitaries
            in ascending index order. Each may be a ``@qmc.qkernel``
            function, a qkernel-backed composite gate, or a built-in gate
            callable. All cases must share the same parameter signature
            and act on the same target register.

    Returns:
        SelectGate: A callable applied as ``sel(index, *targets, **params)``.

    Raises:
        ValueError: If fewer than two cases are supplied or the cases do
            not share an identical parameter signature.
        TypeError: If a case cannot be wrapped into a kernel.

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
    return SelectGate(cases)
