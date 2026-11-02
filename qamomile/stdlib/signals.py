"""Selected quantum processes and composable fixed-basis phase operations.

Regions use source-ordered patterns: ``0`` and ``1`` fix a lane, while ``*``
leaves it free. Signal descriptors stay on the host; their processes and phase
operations compile to ordinary Qamomile functions. A Signal is a reusable
recipe, never a live quantum state or a certificate for an external matrix.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.effects import CallableKind
from qamomile.types import (
    FunctionType,
    infer_declared_function_application,
    is_type_assignable,
    strip_stage,
)

from ._structure import adjoint_unary, lift_slice
from .block_encoding import BlockEncoding
from .lcu import _with_phase
from .state_preparation import amplitude_encoding


@dataclass(frozen=True, slots=True)
class Region:
    """Embed free source coordinates in a fixed-basis subspace.

    The first pattern character describes source lane zero, not the most
    significant character of an integer. For example, ``"1*0"`` leaves lane
    one free and fixes lanes zero and two. ``"***"`` selects the full space;
    ``"010"`` selects a one-dimensional subspace without a zero-width register.

    Args:
        pattern (str): Between 1 and 4096 characters drawn from ``0``, ``1``,
            and ``*``.

    Raises:
        TypeError: If the pattern is not text or native support is unavailable.
        ValueError: If the native checker rejects the pattern or width.
    """

    pattern: str
    _region: native.SignalRegion = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the pattern through the required native region constructor.

        Raises:
            TypeError: If the input or native projection has an invalid type.
            ValueError: If the region is empty, excessive, or malformed.
        """
        if not isinstance(self.pattern, str):
            raise TypeError("region pattern must be a string")
        derive = getattr(native, "signal_region", None)
        region_type = getattr(native, "SignalRegion", None)
        if not callable(derive) or not isinstance(region_type, type):
            raise TypeError("Q7733: native signal region construction is unavailable")
        region = derive(self.pattern)
        if not isinstance(region, region_type) or region.pattern != self.pattern:
            raise TypeError("Q7734: malformed native signal region")
        object.__setattr__(self, "_region", region)

    @property
    def width(self) -> int:
        """Return the ambient coherent width.

        Returns:
            int: Positive number of source lanes.
        """
        return self._region.width

    @property
    def payload_indices(self) -> tuple[int, ...]:
        """Return the ordered free coordinates of the logical input or output.

        Returns:
            tuple[int, ...]: Free source lanes; empty for a rank-one region.
        """
        return tuple(self._region.payload_indices)


@dataclass(frozen=True, slots=True, init=False)
class Signal:
    """Retain a selected process without changing its unitary implementation.

    Create values with :func:`select` or :meth:`from_block_encoding`.
    ``process`` accepts one leading owned bit register and optionally classical
    arguments available no later than Param. Input and output regions define
    the selected map ``J_out.adjoint() U J_in``. A fresh selection uses scale
    one and zero reconstruction error by definition. Imported block encodings
    retain their declared normalization and error, without upgrading an
    assertion about an external matrix into a proof.

    Attributes:
        process (qm.QKernel): Complete coherent process, including global phase.
        input_region (Region): Selected input embedding.
        output_region (Region): Selected output embedding.
        normalization (float): Positive reconstruction scale.
        error_bound (float): Imported reconstruction error, or zero for selection.
    """

    process: qm.QKernel
    input_region: Region
    output_region: Region
    normalization: float
    error_bound: float
    _contract: native.SignalContract = field(repr=False, compare=False)
    _label: str = field(repr=False, compare=False)

    def __init__(self) -> None:
        """Require construction through a checked selection boundary.

        Raises:
            TypeError: Always; use ``select`` or ``from_block_encoding``.
        """
        raise TypeError("use select() or Signal.from_block_encoding()")

    @classmethod
    def _checked(
        cls,
        process: qm.QKernel,
        input_region: Region,
        output_region: Region,
        normalization: float,
        error_bound: float,
    ) -> Signal:
        """Freeze an already checked process and its source contract.

        Args:
            process (qm.QKernel): Verified compatible process.
            input_region (Region): Checked input embedding.
            output_region (Region): Checked output embedding.
            normalization (float): Checked positive scale.
            error_bound (float): Checked nonnegative reconstruction error.

        Returns:
            Signal: Immutable host composition data.
        """
        result = object.__new__(cls)
        object.__setattr__(result, "process", process)
        object.__setattr__(result, "input_region", input_region)
        object.__setattr__(result, "output_region", output_region)
        object.__setattr__(result, "normalization", normalization)
        object.__setattr__(result, "error_bound", error_bound)
        derive = getattr(native, "signal_contract", None)
        if not callable(derive):
            raise TypeError("Q7733: native signal algebra is unavailable")
        contract = derive(
            input_region.pattern, output_region.pattern, normalization, error_bound
        )
        if not isinstance(contract, native.SignalContract):
            raise TypeError("Q7734: malformed native signal contract")
        object.__setattr__(result, "_contract", contract)
        object.__setattr__(
            result,
            "_label",
            f"{normalization} * select({process.function_name}, {input_region.pattern!r} -> {output_region.pattern!r}); imported errors remain assumptions",
        )
        return result

    @classmethod
    def _from_contract(
        cls, process: qm.QKernel, contract: native.SignalContract, label: str
    ) -> Signal:
        """Attach a native-derived algebra contract to its ordinary source witness.

        Args:
            process (qm.QKernel): Generated complete unitary.
            contract (native.SignalContract): Checked layout and reconstruction data.
            label (str): Bounded description of the construction, not a proof.

        Returns:
            Signal: Immutable construction recipe.

        Raises:
            TypeError: If the native projection has the wrong type.
        """
        if not isinstance(contract, native.SignalContract):
            raise TypeError("Q7734: malformed native signal contract")
        result = cls._checked(
            process,
            Region(contract.input),
            Region(contract.output),
            contract.normalization,
            contract.error_bound,
        )
        object.__setattr__(result, "_contract", contract)
        object.__setattr__(
            result,
            "_label",
            label if len(label) <= 4096 else label[:4096] + " ... [display truncated]",
        )
        return result

    def __matmul__(self, right: Signal) -> Signal:
        """Encode the selected product using independent success workspaces.

        Args:
            right (Signal): Map applied first to the logical input.

        Returns:
            Signal: Selected target ``self @ right``.

        Raises:
            TypeError: If either process is not a closed unary callable.
            ValueError: If spaces, scalar ranges, or construction budgets disagree.
        """
        _closed(self)
        _closed(right)
        contract, swaps = native.signal_product_plan(self._contract, right._contract)
        width = len(contract.input)
        first = lift_slice(
            total_width=width,
            start=0,
            width=right.ambient_width,
            transform=right.process,
        )
        transfer = _route(width, tuple(swaps))
        last = lift_slice(
            total_width=width,
            start=right.ambient_width,
            width=self.ambient_width,
            transform=self.process,
        )
        return Signal._from_contract(
            sequence(first, transfer, last, width=width),
            contract,
            f"({self._label}) @ ({right._label}); independent success workspaces",
        )

    def adjoint(self) -> Signal:
        """Reverse the selected map and its complete coherent witness.

        Returns:
            Signal: Adjoint target with input/output embeddings exchanged.

        Raises:
            TypeError: If the process still has unbound classical parameters.
        """
        _closed(self)
        return Signal._from_contract(
            adjoint_unary(width=self.ambient_width, transform=self.process),
            self._contract.adjoint(),
            f"adjoint({self._label})",
        )

    def hermitian_part(self) -> Signal:
        """Construct a structurally Hermitian target, not a claimed symmetry.

        Returns:
            Signal: Encoding of ``(A + A.adjoint()) / 2``.

        Raises:
            TypeError: If a process is not closed.
            ValueError: If the logical input and output spaces differ.
        """
        self._contract.hermitian_part()
        result = interfere(((0.5, self), (0.5, self.adjoint())))
        return Signal._from_contract(
            result.process,
            result._contract.hermitian_part(),
            f"HermitianPart({self._label})",
        )

    @classmethod
    def from_block_encoding(cls, block: BlockEncoding) -> Signal:
        """Import the complete zero-signal layout of an existing block encoding.

        Args:
            block (BlockEncoding): Existing unitary and reconstruction contract.

        Returns:
            Signal: Selection retaining the declared scale and error.

        Raises:
            TypeError: If the block or required native boundary is invalid.
            ValueError: If the complete layout exceeds the native width budget.
        """
        if not isinstance(block, BlockEncoding):
            raise TypeError("block must be a BlockEncoding")
        limit = getattr(native, "MAX_SIGNAL_REGION_WIDTH", None)
        if type(limit) is not int or limit <= 0:
            raise TypeError("Q7733: native signal region construction is unavailable")
        if block.total_width > limit:
            raise ValueError(f"Q7730: signal region width exceeds {limit}")
        region = Region("0" * block.signal_width + "*" * block.system_width)
        _validate_process(block.unitary, region.width)
        return cls._checked(
            block.unitary, region, region, block.normalization, block.error_bound
        )

    @property
    def ambient_width(self) -> int:
        """Return the width of the complete process register.

        Returns:
            int: Common ambient width of the two embeddings.
        """
        return self.input_region.width

    def inspect(self) -> str:
        """Describe the selected map without inventing response or success bounds.

        Returns:
            str: Process, embeddings, declared scale/error, and unknown success.
        """
        return (
            f"Signal({self.process.function_name})\n"
            f"construction: {self._label}\n"
            f"input: {self.input_region.pattern}\n"
            f"output: {self.output_region.pattern}\n"
            f"normalization: {self.normalization}\n"
            f"reconstruction error (declared): {self.error_bound}\n"
            f"recipe nodes/depth: {self._contract.nodes}/{self._contract.depth}\n"
            f"structural Hermitian target: {self._contract.hermitian}\n"
            "success probability: input-dependent; not established"
        )


def select(process: qm.QKernel, *, input: Region, output: Region) -> Signal:
    """Select input and output regions of a complete coherent process.

    The result defines a new target at scale one. It does not preserve a
    polynomial or external-matrix claim from a previously selected process.
    Classical parameters stay in the ordinary callable ABI; selection does
    not bind their values or change their availability.

    Args:
        process (qm.QKernel): Bit-register process with optional classical inputs.
        input (Region): Input embedding in the complete register.
        output (Region): Output embedding in that same register.

    Returns:
        Signal: Immutable selected-process recipe.

    Raises:
        TypeError: If inputs, owned ABI, stage, or unitary contract are invalid.
        ValueError: If the input and output ambient widths differ.
        qm.QamomileError: If compiling the source process fails.
    """
    if not isinstance(input, Region) or not isinstance(output, Region):
        raise TypeError("input and output must be Regions")
    if input.width != output.width:
        raise ValueError("Q7735: signal regions must have the same ambient width")
    _validate_process(process, input.width)
    return Signal._checked(process, input, output, 1.0, 0.0)


def _validate_process(process: qm.QKernel, width: int) -> None:
    """Check a selected callable through its native-derived type and contract.

    Args:
        process (qm.QKernel): Source process to inspect.
        width (int): Required coherent input/output width.

    Raises:
        TypeError: If the ABI is not one owned register plus classical inputs,
            the process is not unitary, or an argument requires Shot data.
        qm.QamomileError: If native source compilation fails.
    """
    if not isinstance(process, qm.QKernel):
        raise TypeError("process must be a Qamomile QKernel")
    function = process.module().function(process.function_name)
    parameters = tuple(parameter.value.type for parameter in function.parameters)
    if not parameters or function.kind is not CallableKind.UNITARY:
        raise TypeError("Q7735: signal process must preserve one owned unitary state")
    for parameter in parameters[1:]:
        if parameter.is_quantum or not is_type_assignable(
            qm.Param[strip_stage(parameter)], parameter
        ):
            raise TypeError(
                "Q7735: signal arguments must be classical and at most Param"
            )
    state_type = qm.Q[qm.Bits[width]]
    try:
        result = infer_declared_function_application(
            FunctionType(
                parameters,
                function.result_type,
                tuple(parameter.name for parameter in function.parameters),
            ),
            (state_type, *parameters[1:]),
            (None,) * len(parameters),
        )
    except ValueError as error:
        raise TypeError(
            "Q7735: signal process has an incompatible register ABI"
        ) from error
    if not is_type_assignable(state_type, result):
        raise TypeError("Q7735: signal process must return its complete bit register")


@qm.qkernel
def _phase_one(state: qm.Q[qm.Bits[1]], angle: qm.Real) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
    """Apply the selected target phase inside an ordinary controlled call.

    Args:
        state (qm.Q[qm.Bits[1]]): Owned target lane.
        angle (qm.Real): Phase in radians.

    Returns:
        qm.Q[qm.Bits[1]]: Phase-rotated target.
    """
    return qm.p(state, angle)


def turn(region: Region) -> qm.QKernel:
    """Construct ``I + (exp(i angle) - 1) P`` for a fixed-basis region.

    The returned function takes ``(state, angle)``. It retains full-space
    scalar phase, supports Param angles, and introduces neither ancillas nor
    a region opcode. Its adjoint and coherent control use ordinary transforms.

    Args:
        region (Region): Checked source-ordered phase selection.

    Returns:
        qm.QKernel: Ordinary phase function over the complete bit register.

    Raises:
        TypeError: If region has the wrong host type.
    """
    if not isinstance(region, Region):
        raise TypeError("region must be a Region")
    width = region.width
    target = region._region.target
    invert = region._region.target_is_zero
    indices = tuple(region._region.controls)
    control_state = int(region._region.control_state)

    if target is None:

        @qm.qkernel
        def full(state: qm.Q[qm.Bits[width]], angle: qm.Real) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Phase the complete ambient space.

            Args:
                state (qm.Q[qm.Bits[width]]): Complete owned register.
                angle (qm.Real): Scalar phase in radians.

            Returns:
                qm.Q[qm.Bits[width]]: Register with its scalar phase retained.
            """
            return qm.global_phase(state, angle)

        return full

    if not indices:

        @qm.qkernel
        def single(state: qm.Q[qm.Bits[width]], angle: qm.Real) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Phase a single constrained lane while retaining all free lanes.

            Args:
                state (qm.Q[qm.Bits[width]]): Complete owned register.
                angle (qm.Real): Selected phase in radians.

            Returns:
                qm.Q[qm.Bits[width]]: Register after the region phase.
            """
            if invert:
                state = qm.x(state, target=target)
            state = qm.p(state, angle, target=target)
            if invert:
                state = qm.x(state, target=target)
            return state

        return single

    routing = _route(width, tuple(region._region.routing_swaps))
    target = region._region.phase_width - 1
    if target == width - 1:

        @qm.qkernel
        def trailing(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
            angle: qm.Real,  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Apply selected controls to the final phase target.

            Args:
                state (qm.Q[qm.Bits[width]]): Complete owned register.
                angle (qm.Real): Selected phase in radians.

            Returns:
                qm.Q[qm.Bits[width]]: Reassembled register after the phase.
            """
            state = routing(state)
            if invert:
                state = qm.x(state, target=target)
            controls, pivot = qm.split_at(state, target)
            controls, pivot = qm.control(
                _phase_one,
                controls,
                pivot,
                angle,
                control_state=control_state,
            )
            state = qm.concat(controls, pivot)
            if invert:
                state = qm.x(state, target=target)
            return qm.adjoint(routing, state)

        return trailing

    @qm.qkernel
    def interior(state: qm.Q[qm.Bits[width]], angle: qm.Real) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Route an interior phase target while preserving source lane order.

        Args:
            state (qm.Q[qm.Bits[width]]): Complete owned register.
            angle (qm.Real): Selected phase in radians.

        Returns:
            qm.Q[qm.Bits[width]]: Register restored to its source layout.
        """
        state = routing(state)
        if invert:
            state = qm.x(state, target=target)
        prefix, tail = qm.split_at(state, target)
        pivot, suffix = qm.split_at(tail, 1)
        prefix, pivot = qm.control(
            _phase_one, prefix, pivot, angle, control_state=control_state
        )
        state = qm.concat(prefix, pivot, suffix)
        if invert:
            state = qm.x(state, target=target)
        return qm.adjoint(routing, state)

    return interior


def _route(width: int, swaps: tuple[tuple[int, int], ...]) -> qm.QKernel:
    """Compose a bounded native coordinate-routing witness as an ordinary loop.

    Args:
        width (int): Checked ambient width.
        swaps (tuple[tuple[int, int], ...]): Native-validated coordinate exchanges.

    Returns:
        qm.QKernel: Routing process whose ordinary adjoint restores source order.
    """
    if not swaps:

        @qm.qkernel
        def unchanged(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Keep an already packed register with a concrete callable width.

            Args:
                state (qm.Q[qm.Bits[width]]): Complete owned register.

            Returns:
                qm.Q[qm.Bits[width]]: The unchanged register.
            """
            return state

        return unchanged

    @qm.qkernel
    def route(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Route fixed coordinates without building a chain of host closures.

        Args:
            state (qm.Q[qm.Bits[width]]): Complete owned register.

        Returns:
            qm.Q[qm.Bits[width]]: Register with fixed coordinates in its prefix.
        """
        for pair in swaps:
            state = qm.swap(state, left=pair[0], right=pair[1])
        return state

    return route


def _closed(signal: Signal) -> None:
    """Require the static unary slice of selected-map algebra.

    Args:
        signal (Signal): Host recipe to validate.

    Raises:
        TypeError: If the recipe or its callable ABI is unsupported.
    """
    if not isinstance(signal, Signal):
        raise TypeError("expected a Signal")
    function = signal.process.module().function(signal.process.function_name)
    if len(function.parameters) != 1:
        raise TypeError(
            "Q7745: close classical parameters in source before composing signals"
        )


def sequence(*processes: qm.QKernel, width: int) -> qm.QKernel:
    """Compose complete processes in source order, retaining failure amplitudes.

    Args:
        processes (qm.QKernel): At most 256 closed unary processes.
        width (int): Positive common ambient width, at most 4096.

    Returns:
        qm.QKernel: Ordinary flat source sequence, or a concrete identity.

    Raises:
        TypeError: If a process has an incompatible callable ABI.
        ValueError: If construction limits are exceeded.
    """
    if type(width) is not int or not 0 < width <= native.MAX_SIGNAL_REGION_WIDTH:
        _invalid_width()
    Region("*" * width)
    if len(processes) > native.MAX_SIGNAL_BRANCHES:
        raise ValueError("Q7742: too many sequence processes")
    steps = tuple(processes)
    for process in steps:
        _validate_process(process, width)
        if len(process.module().function(process.function_name).parameters) != 1:
            raise TypeError("Q7745: sequence requires closed unary processes")
    if not steps:
        return _route(width, ())

    @qm.qkernel
    def composed(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Execute a finite source sequence.

        Args:
            state (qm.Q[qm.Bits[width]]): Complete owned register.

        Returns:
            qm.Q[qm.Bits[width]]: State after all source-ordered calls.
        """
        for operation in steps:
            state = operation(state)
        return state

    # Emit a bounded flat row of ordinary source calls. Function-valued arrays
    # are not a portable QCore value; do not ask the compiler to execute one.
    tree = ast.parse(composed.source_text)
    entry = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == composed.function_name
    )
    loop = next(item for item in entry.body if isinstance(item, ast.For))
    if not isinstance(loop.iter, ast.Tuple) or len(loop.iter.elts) != len(steps):
        raise TypeError("Q7745: source capture did not retain the finite call row")
    calls = [
        ast.Assign(
            targets=[ast.Name(id="state", ctx=ast.Store())],
            value=ast.Call(
                func=callee, args=[ast.Name(id="state", ctx=ast.Load())], keywords=[]
            ),
        )
        for callee in loop.iter.elts
    ]
    position = entry.body.index(loop)
    entry.body[position : position + 1] = calls
    return qm.QKernel(
        composed.module_name,
        composed.function_name,
        ast.unparse(ast.fix_missing_locations(tree)),
        composed.display_name,
    )


def _invalid_width() -> None:
    """Reject an invalid ambient construction width.

    Raises:
        ValueError: Always, for an invalid width request.
    """
    raise ValueError("Q7730: ambient width must be between 1 and 4096")


def _embedding(region: Region) -> qm.QKernel:
    """Map a canonical zero-prefix embedding into the requested coordinates.

    Args:
        region (Region): Native-checked target embedding.

    Returns:
        qm.QKernel: Complete reversible coordinate and fixed-bit transformation.
    """
    swaps, raw_flips = native.signal_embedding_plan(region.pattern)
    routing = _route(region.width, tuple(swaps))
    flips = tuple(raw_flips)
    width = region.width
    if not flips:
        return routing

    @qm.qkernel
    def embedded(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Route payload and initialize fixed-one coordinates reversibly.

        Args:
            state (qm.Q[qm.Bits[width]]): Canonical owned register.

        Returns:
            qm.Q[qm.Bits[width]]: Register in the requested basis embedding.
        """
        state = routing(state)
        for lane in flips:
            state = qm.x(state, target=lane)
        return state

    return embedded


def _aligned(signal: Signal, width: int) -> qm.QKernel:
    """Align both logical coordinate rows to a common trailing payload space.

    Args:
        signal (Signal): Closed selected process.
        width (int): Native-planned common ambient width.

    Returns:
        qm.QKernel: Aligned unitary with zero-prefix input/output selections.
    """
    inner = sequence(
        _embedding(signal.input_region),
        signal.process,
        adjoint_unary(
            width=signal.ambient_width, transform=_embedding(signal.output_region)
        ),
        width=signal.ambient_width,
    )
    return lift_slice(
        total_width=width,
        start=width - signal.ambient_width,
        width=signal.ambient_width,
        transform=inner,
    )


def interfere(terms: Sequence[tuple[complex, Signal]]) -> Signal:
    """Coherently sum selected maps with their reconstruction scales included.

    Args:
        terms (Sequence[tuple[complex, Signal]]): Nonempty bounded coefficient/map row.
            Exact cancellation is grouped only for identical complete recipes.

    Returns:
        Signal: Encoding of the weighted mathematical targets, with all success
        workspaces retained. A cancelled sum has scale one and zero success.

    Raises:
        TypeError: If a term or closed process ABI is invalid.
        ValueError: If spaces, numeric domains, or construction budgets disagree.
    """
    if not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of coefficient/Signal pairs")
    if not 1 <= len(terms) <= native.MAX_SIGNAL_BRANCHES:
        raise ValueError("Q7742: interference requires between 1 and 256 terms")
    groups: dict[Signal, int] = {}
    operands: list[Signal] = []
    rows: list[tuple[int, float, float]] = []
    for coefficient, signal in terms:
        _closed(signal)
        if isinstance(coefficient, bool) or not isinstance(
            coefficient, (int, float, complex)
        ):
            raise TypeError("coefficient must be a complex scalar")
        value = complex(coefficient)
        if signal not in groups:
            groups[signal] = len(operands)
            operands.append(signal)
        rows.append((groups[signal], value.real, value.imag))
    contract, workspace, selector_width, native_terms = native.signal_sum_plan(
        [s._contract for s in operands], rows
    )
    width = len(contract.input)
    if not native_terms:
        # Flip a selector, so the complete projected block is exactly zero.
        @qm.qkernel
        def zero(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Move a success selector outside its zero state.

            Args:
                state (qm.Q[qm.Bits[width]]): Complete owned register.

            Returns:
                qm.Q[qm.Bits[width]]: Register with an orthogonal selector.
            """
            return qm.x(state, target=0)

        return Signal._from_contract(zero, contract, "zero selected map")
    case_count = 1 << selector_width
    preparation = amplitude_encoding(
        tuple(row[1] for row in native_terms)
        + (0.0,) * (case_count - len(native_terms))
    )
    cases = tuple(
        _with_phase(_aligned(operands[index], workspace), width=workspace, phase=phase)
        for index, _, phase in native_terms
    )
    cases += (_route(workspace, ()),) * (case_count - len(cases))

    @qm.qkernel
    def summed(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Execute scale-aware PREPARE, SELECT, and unpreparation.

        Args:
            state (qm.Q[qm.Bits[width]]): Selector plus aligned workspace.

        Returns:
            qm.Q[qm.Bits[width]]: Complete coherent sum witness.
        """
        selector, workspace_state = qm.split_at(state, selector_width)
        selector = preparation(selector)
        selector, workspace_state = qm.multiplex(selector, cases, workspace_state)
        selector = qm.adjoint(preparation, selector)
        return qm.concat(selector, workspace_state)

    description = " + ".join(
        f"({coefficient}) * ({signal._label})" for coefficient, signal in terms
    )
    return Signal._from_contract(summed, contract, "coherent sum: " + description)


@dataclass(frozen=True, slots=True, init=False)
class Response:
    """An editable real Chebyshev target in normalized spectral coordinates.

    Construct with ``Response.chebyshev``. The target is ``q(A / alpha)``;
    ``scale`` is the requested reconstruction scale, never silently increased.
    Creating a target does not certify phase synthesis or an approximation to
    an independently intended function such as an inverse.

    Attributes:
        coefficients (tuple[float, ...]): Ascending first-kind Chebyshev values.
        scale (float): Positive requested reconstruction scale.
    """

    coefficients: tuple[float, ...]
    scale: float
    _spec: native.SignalResponse = field(repr=False, compare=False)

    def __init__(self) -> None:
        """Require the checked target constructor.

        Raises:
            TypeError: Always; use ``Response.chebyshev``.
        """
        raise TypeError("use Response.chebyshev()")

    @classmethod
    def chebyshev(
        cls, coefficients: Sequence[float], *, scale: float = 1.0
    ) -> Response:
        """Check explicit real, definite-parity polynomial coordinates.

        Args:
            coefficients (Sequence[float]): Nonempty ascending coefficients,
                at most degree 256. Trailing zeros are canonicalized by Rust.
            scale (float): Finite positive reconstruction scale; default one.

        Returns:
            Response: Immutable target, not yet a realization certificate.

        Raises:
            TypeError: If host values or required native support are invalid.
            ValueError: If parity, domains, or degree budgets are invalid.
        """
        if (
            not isinstance(coefficients, Sequence)
            or len(coefficients) > native.MAX_CERTIFIED_QSVT_DEGREE + 1
        ):
            raise ValueError("Q7742: response requires at most 257 coefficients")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or any(
                isinstance(c, bool) or not isinstance(c, (int, float))
                for c in coefficients
            )
        ):
            raise TypeError("response coordinates and scale must be real scalars")
        derive = getattr(native, "signal_response", None)
        if not callable(derive):
            raise TypeError("Q7733: native response construction is unavailable")
        spec = derive(list(coefficients), float(scale))
        if not isinstance(spec, native.SignalResponse):
            raise TypeError("Q7734: malformed native response")
        result = object.__new__(cls)
        object.__setattr__(result, "coefficients", tuple(spec.coefficients))
        object.__setattr__(result, "scale", spec.scale)
        object.__setattr__(result, "_spec", spec)
        return result

    def evaluate(self, x: float) -> float:
        """Evaluate the requested target, not a measured quantum response.

        Args:
            x (float): Normalized real coordinate in ``[-1, 1]``.

        Returns:
            float: Pointwise requested polynomial value, before scaling.

        Raises:
            ValueError: If the coordinate or result is nonfinite or out of range.
        """
        return self._spec.evaluate(x)


def response(
    signal: Signal,
    target: Response,
    *,
    kind: str = "singular",
    side: str | None = None,
    max_error: float = 1e-7,
    max_iterations: int = 512,
    method: str = "qsvt",
) -> Signal:
    """Realize a certified polynomial using two equally oriented QSVT branches.

    Odd singular responses map input to output. Even responses act on the
    explicitly chosen input or output space, including its nullspace. Positive
    and negative phase schedules share the same U/U-adjoint orientation; using
    the adjoint of a completed branch would be incorrect for general matrices.
    Imported input-error claims remain premises of the returned error bound.

    Args:
        signal (Signal): Closed selected map with reconstruction scale alpha.
        target (Response): Target ``q(A/alpha)`` and explicit output scale beta.
        kind (str): ``singular`` or ``eigenvalue``; the latter requires a
            structurally constructed Hermitian target.
        side (str | None): Required ``input`` or ``output`` for even degree;
            must be omitted for odd degree.
        max_error (float): Positive total reconstruction-error tolerance.
        max_iterations (int): Bounded native synthesis iteration limit.
        method (str): ``qsvt`` for a single synthesized schedule or explicit
            ``chebyshev_lcu`` for independently realized Chebyshev terms. The
            latter requires scale at least the coefficient one-norm, uses more
            queries/workspace, and extends beyond the Newton solver region.

    Returns:
        Signal: Complete selected polynomial block at the requested scale.

    Raises:
        TypeError: If host types, native projection, or a process ABI is invalid.
        ValueError: If synthesis, premises, spaces, tolerances, or budgets fail.
    """
    _closed(signal)
    if not isinstance(target, Response):
        raise TypeError("target must be a Response")
    if method == "chebyshev_lcu" and target._spec.degree > 0:
        return _lcu_response(signal, target, kind, side, max_error, max_iterations)
    if method not in ("qsvt", "chebyshev_lcu"):
        raise ValueError("method must be qsvt or chebyshev_lcu")
    if isinstance(max_iterations, bool) or type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer")
    plan = native.signal_response_plan(
        target._spec, signal._contract, kind, side or "", max_error, max_iterations
    )
    if not isinstance(plan, native.SignalResponsePlan):
        raise TypeError("Q7734: malformed native response plan")
    oriented = signal.adjoint() if plan.reverse else signal
    positive = _response_branch(oriented, tuple(plan.phases), False)
    negative = _response_branch(oriented, tuple(plan.phases), True)
    # Both complete witnesses use the original ambient basis and identical
    # selected input/output embeddings. Only their phase coordinates differ.
    width = signal.ambient_width + 1
    half = amplitude_encoding((2**-0.5, 2**-0.5))
    cases = (positive, negative)

    @qm.qkernel
    def extracted(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Interfere opposite phase schedules without transposing the target.

        Args:
            state (qm.Q[qm.Bits[width]]): Extraction selector and source ambient state.

        Returns:
            qm.Q[qm.Bits[width]]: Full complex polynomial extraction witness.
        """
        selector, workspace = qm.split_at(state, 1)
        selector = half(selector)
        selector, workspace = qm.multiplex(selector, cases, workspace)
        selector = qm.adjoint(half, selector)
        return qm.concat(selector, workspace)

    label = f"{kind} Chebyshev{target.coefficients} on ({signal._label}) / {signal.normalization}; side={side}; scale={target.scale}; synthesis error={plan.synthesis_error}; conditional input error={plan.input_error}"
    return Signal._from_contract(extracted, plan.contract, label)


def _lcu_response(
    signal: Signal,
    target: Response,
    kind: str,
    side: str | None,
    max_error: float,
    max_iterations: int,
) -> Signal:
    """Realize explicit Chebyshev terms with a native-planned scale margin.

    Args:
        signal (Signal): Closed selected source.
        target (Response): Checked polynomial target and requested scale.
        kind (str): Singular or structurally Hermitian eigenvalue semantics.
        side (str | None): Space selection for even responses.
        max_error (float): Total target-unit error tolerance.
        max_iterations (int): Per-branch synthesis work budget.

    Returns:
        Signal: Coherent polynomial sum at the explicit requested scale.

    Raises:
        ValueError: If native work, scale, synthesis, or final error checks fail.
    """
    coordinates, margin, tolerance = target._spec.lcu_terms(signal._contract, max_error)
    terms: list[tuple[complex, Signal]] = []
    for degree, coefficient in coordinates:
        term = Response.chebyshev(
            (0.0,) * degree + (coefficient,), scale=abs(coefficient)
        )
        realized = response(
            signal,
            term,
            kind=kind,
            side=side,
            max_error=tolerance,
            max_iterations=max_iterations,
        )
        terms.append((1.0, realized))
    if margin > 0:
        # Cancellation of the same complete witness preserves even/odd spaces.
        zero = interfere(((1.0, terms[0][1]), (-1.0, terms[0][1])))
        terms.append((margin, zero))
    combined = interfere(tuple(terms))
    contract = combined._contract.rescale(target.scale, max_error)
    return Signal._from_contract(
        combined.process,
        contract,
        "explicit Chebyshev-term LCU response; input-error assumptions retained",
    )


def _response_branch(
    signal: Signal, phases: tuple[tuple[float, float], ...], negate: bool
) -> qm.QKernel:
    """Project native QSVT coordinates to ordinary selected and scalar phases.

    Args:
        signal (Signal): Oriented source encoding.
        phases (tuple[tuple[float, float], ...]): Native phase/global-phase rows.
        negate (bool): Whether to conjugate the scalar response coordinates.

    Returns:
        qm.QKernel: Complete branch with unchanged singular input/output orientation.
    """
    width = signal.ambient_width
    unitary = signal.process
    input_turn = turn(signal.input_region)
    output_turn = turn(signal.output_region)
    first = phases[0]
    steps = tuple((i % 2 == 1, row[0], row[1]) for i, row in enumerate(phases[1:], 1))
    sign = -1.0 if negate else 1.0
    if not steps:

        @qm.qkernel
        def constant(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Retain the constant response on the complete selected space.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned source state.

            Returns:
                qm.Q[qm.Bits[width]]: State with the complete scalar phase.
            """
            state = input_turn(state, sign * first[0])
            return qm.global_phase(state, sign * first[1])

        return constant

    @qm.qkernel
    def branch(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Alternate source queries and correctly oriented projector phases.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned source state.

        Returns:
            qm.Q[qm.Bits[width]]: Complete phase-sequence output.
        """
        state = input_turn(state, sign * first[0])
        state = qm.global_phase(state, sign * first[1])
        for step in steps:
            if step[0]:
                state = unitary(state)
                state = output_turn(state, sign * step[1])
            else:
                state = qm.adjoint(unitary, state)
                state = input_turn(state, sign * step[1])
            state = qm.global_phase(state, sign * step[2])
        return state

    return branch


from ._signal_workflows import (  # noqa: E402
    SuccessEstimate,
    amplify,
    estimate_success,
    inspect_schedule,
    sample,
)

__all__ = [
    "SuccessEstimate",
    "amplify",
    "estimate_success",
    "inspect_schedule",
    "sample",
    "Region",
    "Signal",
    "select",
    "turn",
    "sequence",
    "interfere",
    "Response",
    "response",
]
