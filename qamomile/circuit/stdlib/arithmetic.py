"""Constant modular arithmetic building blocks with resource models.

This module provides :func:`modmul_const`, the reversible primitive
``|x> -> |a*x mod N>`` used by Shor order finding. Every call is one named,
executable composite with an attached primitive-level resource model. The
estimator therefore derives an algorithm's cost by walking its real call graph;
algorithms do not need to restate an aggregate resource formula.

The default model uses the literature calibration ``0.15 n^2`` per controlled
modular multiplication, while ``ResourcePolicy.ASYMPTOTIC`` leaves the
coefficient symbolic. Small concrete instances have executable SWAP-network or
basis-permutation bodies for cross-backend correctness checks.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from qamomile.circuit.estimator import (
    CallResources,
    DepthResources,
    GateResources,
    ResourceAssumption,
    ResourceContext,
    ResourceEstimate,
    WidthResources,
)
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.ir.operation.callable import (
    ResourceModelBinding,
)

# One constant modular multiplication costs ~c * n**2 non-Clifford (Toffoli)
# gates on an n-bit register. The literature prefactor is chosen so that an
# order-finding run — 2n controlled modular multiplications, one per exponent
# bit — totals ~0.3 * n**3 non-Clifford gates, matching the book's RSA-2048
# figure (Section 6.1: "roughly 0.3 n^3 ~= 3e9 non-Clifford gates"). This
# prefactor is NOT an independently derived gate count; it is the book's
# optimized empirical aggregate (0.3 n^3) divided over the 2n multiplications.
_LITERATURE_NONCLIFFORD_COEFF = sp.Rational(15, 100)

# The asymptotic model states only the shape O(n^2) and leaves the prefactor as
# an explicit free symbol, so callers see it is undetermined rather than a
# reverse-engineered constant. ``ResourcePolicy.ASYMPTOTIC`` selects this model;
# the default and ``ResourcePolicy.LITERATURE`` select the calibrated one.
_ASYMPTOTIC_NONCLIFFORD_COEFF = sp.Symbol("c_modmul", positive=True)

_MODMUL_NAME = "modmul_const"


def _register_bits(ctx: ResourceContext) -> sp.Expr:
    """Return the modular register width ``n`` for a resource context.

    Reads the concrete ``register_bits`` attr first (set when the width is
    statically known), then the single array operand shape (the register), then
    falls back to a fresh symbol. Only one array operand is expected; a scalar
    control does not appear in ``operand_shapes``.

    Args:
        ctx (ResourceContext): Call-site context supplied by the estimator.

    Returns:
        sp.Expr: Symbolic or concrete register width.
    """
    attr_bits = ctx.attrs.get("register_bits")
    if attr_bits is not None:
        return sp.sympify(attr_bits)
    array_widths = [sp.sympify(width) for width in ctx.operand_shapes.values()]
    if len(array_widths) == 1:
        return array_widths[0]
    if array_widths:
        # Defensive: sum shapes if a future signature grows a second array
        # operand, so the width is at least not silently dropped.
        return sp.Add(*array_widths)
    return sp.Symbol("n", positive=True, integer=True)


def _modmul_model_estimate(
    ctx: ResourceContext,
    coefficient: sp.Expr,
    assumption: ResourceAssumption,
) -> ResourceEstimate:
    """Estimate one constant modular multiplication with a given prefactor.

    Reports ``coefficient * n**2`` non-Clifford (Toffoli) gates on an ``n``-bit
    register and records the ``O(n)`` scratch of the arithmetic as borrowable
    ``dirty_ancilla_qubits`` — machine-readable rather than buried in prose —
    so an order-finding run reports ``~3n`` algorithmic logical qubits while
    still carrying the scratch requirement as data. Depth assumes fully-serial
    execution of the non-Clifford layers.

    Args:
        ctx (ResourceContext): Call-site context supplied by the estimator.
        coefficient (sp.Expr): Non-Clifford ``n**2`` prefactor.
        assumption (ResourceAssumption): Modeling assumption to attach.

    Returns:
        ResourceEstimate: Gate/width/call resources for one modular
        multiplication.
    """
    n = _register_bits(ctx)
    nonclifford = coefficient * n**2
    serial_depth = ResourceAssumption(
        "modular multiplication depth assumes fully-serial non-Clifford layers",
        source=_MODMUL_NAME,
    )
    # The 0.15 n^2 calibration is per *controlled* modular multiplication (the
    # book's 0.3 n^3 over 2n controlled multiplications). A control adds only
    # lower-order overhead against the O(n^2) leading term, so the count is the
    # same either way — but say honestly whether this call is controlled.
    if ctx.own_controls:
        control_coverage = ResourceAssumption(
            "cost includes the single control of a controlled multiplication "
            "(the calibration is per controlled modmul, as in order finding); "
            "additional surrounding controls are not included",
            source=_MODMUL_NAME,
        )
    else:
        control_coverage = ResourceAssumption(
            "calibration is per *controlled* modular multiplication (as in order "
            "finding); this uncontrolled call is priced at the controlled cost — "
            "a slight overestimate, since the control adds only lower-order "
            "overhead against the O(n^2) leading term",
            source=_MODMUL_NAME,
        )
    return ResourceEstimate(
        width=WidthResources(dirty_ancilla_qubits=n),
        gates=GateResources(
            total=nonclifford,
            multi_qubit=nonclifford,
            toffoli=nonclifford,
            non_clifford=nonclifford,
        ),
        depth=DepthResources(depth=nonclifford, non_clifford_depth=nonclifford),
        calls=CallResources(
            calls_by_name={_MODMUL_NAME: sp.Integer(1)},
            queries_by_name={_MODMUL_NAME: sp.Integer(1)},
        ),
        assumptions=(assumption, serial_depth, control_coverage),
    )


def _modmul_asymptotic_model(ctx: ResourceContext) -> ResourceEstimate:
    """Estimate one constant modular multiplication with an open prefactor.

    Reports ``c_modmul * n**2`` non-Clifford gates, leaving the prefactor as a
    free symbol so the ``O(n^2)`` shape is explicit and undetermined.

    Args:
        ctx (ResourceContext): Call-site context supplied by the estimator.

    Returns:
        ResourceEstimate: Gate/call resources for one modular multiplication.
    """
    return _modmul_model_estimate(
        ctx,
        _ASYMPTOTIC_NONCLIFFORD_COEFF,
        ResourceAssumption(
            "constant modular multiplication is O(n^2) non-Clifford; prefactor "
            "'c_modmul' left symbolic (implementation-dependent)",
            source=_MODMUL_NAME,
        ),
    )


def _modmul_literature_model(ctx: ResourceContext) -> ResourceEstimate:
    """Estimate one constant modular multiplication at literature level.

    Reports ``0.15 n^2`` non-Clifford gates. This prefactor is back-derived from
    the reference's aggregate ``0.3 n^3`` over ``2n`` multiplications — it is a
    literature calibration, not an independent gate count.

    Args:
        ctx (ResourceContext): Call-site context supplied by the estimator.

    Returns:
        ResourceEstimate: Gate/call resources for one modular multiplication.
    """
    return _modmul_model_estimate(
        ctx,
        _LITERATURE_NONCLIFFORD_COEFF,
        ResourceAssumption(
            "literature RSA-2048 calibration: prefactor 0.15 n^2 per modular "
            "multiplication is the reference's aggregate 0.3 n^3 divided over "
            "2n multiplications (book Section 6.1, optimized surface-code "
            "estimate) — not an independent schoolbook count",
            source=_MODMUL_NAME,
        ),
    )


def _modmul_resource_models() -> list[ResourceModelBinding]:
    """Return the resource-model bindings attached to every ``modmul_const``.

    The default policy selects the literature (calibrated) model because the box
    pins ``default_estimate_kind="literature"`` in its attrs, so reproducing the
    book figures does not depend on binding order.
    ``ResourcePolicy.ASYMPTOTIC`` selects the open-prefactor model;
    ``ResourcePolicy.LITERATURE`` selects the calibrated one explicitly.

    Returns:
        list[ResourceModelBinding]: Literature and asymptotic constant modular
        multiplication models.
    """
    return [
        ResourceModelBinding(
            model=_ModmulModel(_modmul_literature_model),
            estimate_kind="literature",
        ),
        ResourceModelBinding(
            model=_ModmulModel(_modmul_asymptotic_model),
            estimate_kind="asymptotic",
        ),
    ]


class _ModmulModel:
    """Adapt a ``ctx -> ResourceEstimate`` function to the model protocol.

    Args:
        func (Any): Callable taking a ``ResourceContext`` and returning a
            ``ResourceEstimate``.
    """

    def __init__(self, func: Any) -> None:
        """Store the wrapped estimation function.

        Args:
            func (Any): Callable taking a ``ResourceContext``.
        """
        self._func = func

    def estimate(self, ctx: ResourceContext) -> ResourceEstimate:
        """Estimate resources by delegating to the wrapped function.

        Args:
            ctx (ResourceContext): Call-site context supplied by the estimator.

        Returns:
            ResourceEstimate: Estimate returned by the wrapped function.
        """
        return self._func(ctx)


def _concrete_size(reg: Vector[Qubit]) -> int | None:
    """Return the concrete length of a qubit register, or ``None``.

    Args:
        reg (Vector[Qubit]): Register handle to inspect.

    Returns:
        int | None: Concrete register length when statically known, else
        ``None`` for symbolic-width registers.
    """
    if not reg.shape:
        return None
    dim = reg.shape[0]
    if isinstance(dim, int):
        return dim
    value = getattr(dim, "value", None)
    if value is not None and value.is_constant():
        const = value.get_const()
        if const is not None:
            return int(const)
    return None


def _is_cyclic_shift(multiplier: int, modulus: int, n: int) -> int | None:
    """Return the rotation amount if ``x -> multiplier*x mod modulus`` rotates.

    Multiplication by ``2**j`` modulo ``2**n - 1`` is a left cyclic rotation of
    the ``n``-bit register by ``j`` positions, realizable purely with SWAP
    gates. This detects that special case so a correct executable body can be
    attached for small concrete sizes.

    Args:
        multiplier (int): Constant multiplier ``a``.
        modulus (int): Modulus ``N``.
        n (int): Register width in bits.

    Returns:
        int | None: Rotation amount ``j`` in ``[0, n)`` when the map is a cyclic
        shift, else ``None``.
    """
    if modulus != (1 << n) - 1:
        return None
    m = multiplier % modulus
    for j in range(n):
        if (1 << j) % modulus == m:
            return j
    return None


def _emit_cyclic_shift_body(
    reg: Vector[Qubit],
    rotation: int,
    control: Qubit | None = None,
) -> tuple[Qubit | None, Vector[Qubit]]:
    """Emit a SWAP-network cyclic multiply-by-``2**rotation`` of ``reg``.

    Args:
        reg (Vector[Qubit]): Register handle to rotate in place.
        rotation (int): Number of positions to rotate (multiply by
            ``2**rotation``).
        control (Qubit | None): Optional control qubit. When provided the
            rotation is conditioned on it via controlled swaps (Fredkin gates).
            Defaults to ``None``.

    Returns:
        tuple[Qubit | None, Vector[Qubit]]: The (possibly updated) control
        handle and the rotated register.
    """
    import qamomile.circuit as qmc

    n = _concrete_size(reg)
    assert n is not None
    rotation %= n
    cswap = qmc.control(qmc.swap, num_controls=1) if control is not None else None
    for _ in range(rotation):
        # Multiply by 2: bit i -> bit i+1, with the top bit wrapping to bit 0.
        # Realized by adjacent swaps sweeping downward so each qubit's content
        # advances one position and reg[n-1] wraps into reg[0].
        for i in range(n - 1, 0, -1):
            if control is None or cswap is None:
                reg[i], reg[i - 1] = qmc.swap(reg[i], reg[i - 1])
            else:
                control, reg[i], reg[i - 1] = cswap(control, reg[i], reg[i - 1])
    return control, reg


def _permutation_transpositions(
    multiplier: int,
    modulus: int,
    num_qubits: int,
) -> tuple[tuple[int, int], ...]:
    """Decompose constant modular multiplication into basis transpositions.

    Basis states below ``modulus`` follow ``x -> multiplier*x mod modulus``;
    states outside that range are fixed so the map is a permutation over the
    complete register Hilbert space.

    Args:
        multiplier (int): Coprime multiplier.
        modulus (int): Arithmetic modulus.
        num_qubits (int): Register width.

    Returns:
        tuple[tuple[int, int], ...]: Basis-state swaps in circuit order.
    """
    dimension = 1 << num_qubits
    permutation = [
        (multiplier * value) % modulus if value < modulus else value
        for value in range(dimension)
    ]
    visited: set[int] = set()
    transpositions: list[tuple[int, int]] = []
    for start in range(dimension):
        if start in visited:
            continue
        cycle: list[int] = []
        value = start
        while value not in visited:
            visited.add(value)
            cycle.append(value)
            value = permutation[value]
        for target in cycle[1:]:
            transpositions.append((cycle[0], target))
    return tuple(transpositions)


def _gray_path(start: int, stop: int) -> tuple[int, ...]:
    """Return a Hamming-distance-one path between two basis states.

    Args:
        start (int): Initial basis-state integer.
        stop (int): Final basis-state integer.

    Returns:
        tuple[int, ...]: Path including both endpoints.
    """
    path = [start]
    current = start
    differing = start ^ stop
    bit = 0
    while differing:
        if differing & 1:
            current ^= 1 << bit
            path.append(current)
        differing >>= 1
        bit += 1
    return tuple(path)


def _emit_adjacent_basis_swap(
    reg: Vector[Qubit],
    left: int,
    right: int,
    num_qubits: int,
) -> None:
    """Swap two basis states that differ in exactly one bit.

    Args:
        reg (Vector[Qubit]): Register to update.
        left (int): First basis state.
        right (int): Second basis state.
        num_qubits (int): Register width.

    Raises:
        ValueError: If the states are not Hamming-distance one.
    """
    import qamomile.circuit as qmc

    difference = left ^ right
    if difference == 0 or difference & (difference - 1):
        raise ValueError("Adjacent basis swap requires Hamming distance one.")
    target = difference.bit_length() - 1
    control_indices = [index for index in range(num_qubits) if index != target]
    zero_controls = [index for index in control_indices if not (left >> index) & 1]
    for index in zero_controls:
        reg[index] = qmc.x(reg[index])
    if control_indices:
        controlled_x = qmc.control(qmc.x, num_controls=len(control_indices))
        outputs = controlled_x(
            *(reg[index] for index in control_indices),
            reg[target],
        )
        for index, output in zip(
            (*control_indices, target),
            outputs,
            strict=True,
        ):
            reg[index] = output
    else:
        reg[target] = qmc.x(reg[target])
    for index in reversed(zero_controls):
        reg[index] = qmc.x(reg[index])


def _emit_basis_transposition(
    reg: Vector[Qubit],
    left: int,
    right: int,
    num_qubits: int,
) -> None:
    """Swap two arbitrary computational basis states.

    Args:
        reg (Vector[Qubit]): Register to update.
        left (int): First basis state.
        right (int): Second basis state.
        num_qubits (int): Register width.
    """
    path = _gray_path(left, right)
    adjacent = list(zip(path, path[1:]))
    for start, stop in adjacent:
        _emit_adjacent_basis_swap(reg, start, stop, num_qubits)
    for start, stop in reversed(adjacent[:-1]):
        _emit_adjacent_basis_swap(reg, start, stop, num_qubits)


def _emit_permutation_body(
    reg: Vector[Qubit],
    transpositions: tuple[tuple[int, int], ...],
    num_qubits: int,
) -> Vector[Qubit]:
    """Emit a reversible basis-permutation implementation.

    Args:
        reg (Vector[Qubit]): Register to update.
        transpositions (tuple[tuple[int, int], ...]): Basis swaps in circuit
            order.
        num_qubits (int): Register width.

    Returns:
        Vector[Qubit]: Updated register.
    """
    for left, right in transpositions:
        _emit_basis_transposition(reg, left, right, num_qubits)
    return reg


def modmul_const(
    reg: Vector[Qubit],
    *,
    multiplier: int,
    modulus: int,
    control: Qubit | None = None,
) -> Vector[Qubit] | tuple[Qubit, Vector[Qubit]]:
    """Apply constant modular multiplication ``|x> -> |a*x mod N>``.

    This is a resource-modeled composite. At estimate time the attached models
    report one modular multiplication's cost (``~0.15 n^2`` non-Clifford gates
    on an ``n``-bit register under the default literature model), so iterating
    it reproduces the book's order-finding scaling symbolically.

    Every call has an executable body. Cyclic Mersenne multiplications use a
    compact SWAP network; other constants use an ancilla-free basis-permutation
    synthesis intended for small correctness tests. The attached literature and
    asymptotic models remain available for scalable estimation.

    Args:
        reg (Vector[Qubit]): Little-endian register to multiply in place. Its
            width must be known when the qkernel is traced.
        multiplier (int): Positive constant multiplier ``a``, coprime to
            ``modulus``.
        modulus (int): Integer modulus ``N`` of at least two.
        control (Qubit | None): Optional control qubit. When provided the
            multiplication is applied conditionally (as Shor's order finding
            conditions each modular multiplication on an exponent qubit).
            Defaults to ``None``.

    Returns:
        Vector[Qubit] | tuple[Qubit, Vector[Qubit]]: The register after modular
        multiplication, or ``(control, register)`` when a control qubit is
        supplied.

    Raises:
        ValueError: If the modulus or multiplier is invalid, or the register
            width is symbolic.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import modmul_const
        >>> @qmc.qkernel
        ... def mul(reg: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        ...     return modmul_const(reg, multiplier=2, modulus=15)
    """
    if modulus < 2:
        raise ValueError(f"modulus must be >= 2, got {modulus}.")
    if multiplier <= 0 or sp.igcd(multiplier, modulus) != 1:
        raise ValueError(
            "multiplier must be a positive integer coprime to modulus; "
            f"got multiplier={multiplier}, modulus={modulus}."
        )

    size = _concrete_size(reg)
    if size is None:
        raise ValueError(
            "modmul_const requires a compile-time-known register width so its "
            "reversible implementation can be synthesized. Bind the width before "
            "calling or transpiling this kernel."
        )
    if modulus > 1 << size:
        raise ValueError(f"modulus {modulus} does not fit in a {size}-qubit register.")

    import qamomile.circuit as qmc

    rotation = _is_cyclic_shift(multiplier, modulus, size)
    if rotation is not None:

        @qmc.composite_gate(
            name=_MODMUL_NAME,
            resource_models=_modmul_resource_models(),
            default_estimate_kind="literature",
        )
        def implementation(register: Vector[Qubit]) -> Vector[Qubit]:
            """Apply the cyclic-shift implementation.

            Args:
                register (Vector[Qubit]): Register to rotate.

            Returns:
                Vector[Qubit]: Rotated register.
            """
            return _emit_cyclic_shift_body(register, rotation)[1]

    else:
        transpositions = _permutation_transpositions(multiplier, modulus, size)

        @qmc.composite_gate(
            name=_MODMUL_NAME,
            resource_models=_modmul_resource_models(),
            default_estimate_kind="literature",
        )
        def implementation(register: Vector[Qubit]) -> Vector[Qubit]:
            """Apply the general basis-permutation implementation.

            Args:
                register (Vector[Qubit]): Register to permute.

            Returns:
                Vector[Qubit]: Permuted register.
            """
            return _emit_permutation_body(register, transpositions, size)

    if control is None:
        return implementation(reg)
    controlled = qmc.control(implementation)
    control_out, reg_out = controlled(control, reg)
    return control_out, reg_out


__all__ = ["modmul_const"]
