"""Constant modular arithmetic building blocks with resource models.

This module provides :func:`modmul_const`, the constant modular multiplication
primitive ``|x> -> |a*x mod N>`` that Shor's order-finding algorithm iterates.
It is a resource-modeled composite: at estimate time the attached models supply
the cost of one modular multiplication (``O(n^2)`` non-Clifford gates on an
``n``-bit register), so a symbolic order-finding kernel reproduces the book's
``O(n^3)`` non-Clifford / ``O(n)`` qubit scaling without unrolling. The default
(literature) model uses the calibrated prefactor ``0.15 n^2`` — back-derived
from the reference's aggregate ``0.3 n^3`` over ``2n`` multiplications, not an
independent gate count — while ``ResourcePolicy.ASYMPTOTIC`` leaves the
prefactor as a free symbol. ``modmul_const`` can be called *abstractly* (no
specific ``multiplier``/``modulus``) — the honest choice for a
schedule-independent estimate — or with a concrete constant; only the concrete
cyclic-rotation case (``modulus == 2**n - 1`` and ``multiplier`` a power of two)
gains an executable SWAP-network body, so every other form is estimation-only
and raises at transpile time rather than emitting a silently-wrong circuit.

The design follows the project's IR-abstraction principle: ``modmul_const`` is a
single abstract box carrying an optional body and a resource model, never
pre-expanded into per-bit arithmetic at IR level. Backends and the estimator
choose the concrete realization.
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
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
    ResourceModelBinding,
    signature_from_values,
)
from qamomile.circuit.ir.value import Value

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


def modmul_const(
    reg: Vector[Qubit],
    *,
    multiplier: int | None = None,
    modulus: int | None = None,
    control: Qubit | None = None,
) -> Vector[Qubit] | tuple[Qubit, Vector[Qubit]]:
    """Apply constant modular multiplication ``|x> -> |a*x mod N>``.

    This is a resource-modeled composite. At estimate time the attached models
    report one modular multiplication's cost (``~0.15 n^2`` non-Clifford gates
    on an ``n``-bit register under the default literature model), so iterating
    it reproduces the book's order-finding scaling symbolically.

    Three modes, all sharing the same resource models:

    - **Abstract** (``multiplier`` and ``modulus`` both omitted): "some constant
      modular multiplication of this width", carrying no specific constant. This
      is the honest choice for a schedule-independent resource estimate (the
      cost is the same for every constant). It has no executable body and raises
      at transpile time — estimation only.
    - **Concrete cyclic** (``modulus == 2**n - 1`` with ``multiplier`` a power of
      two at a concrete width): a real SWAP-network cyclic rotation with an
      executable body that runs on backends.
    - **Concrete general** (any other coprime constant): an estimation-only box
      like the abstract mode, but recording the specific constant in attrs.

    Args:
        reg (Vector[Qubit]): Little-endian register to multiply in place. May
            have a symbolic width for estimation.
        multiplier (int | None): Constant multiplier ``a``; must be coprime to
            ``modulus`` to be reversible. ``None`` (with ``modulus`` also
            ``None``) selects abstract mode. Defaults to ``None``.
        modulus (int | None): Modulus ``N``. ``None`` (with ``multiplier`` also
            ``None``) selects abstract mode. Defaults to ``None``.
        control (Qubit | None): Optional control qubit. When provided the
            multiplication is applied conditionally (as Shor's order finding
            conditions each modular multiplication on an exponent qubit).
            Defaults to ``None``.

    Returns:
        Vector[Qubit] | tuple[Qubit, Vector[Qubit]]: The register after modular
        multiplication, or ``(control, register)`` when a control qubit is
        supplied.

    Raises:
        ValueError: If exactly one of ``multiplier`` / ``modulus`` is given, or
            (in concrete mode) ``modulus`` is not at least 2, or ``multiplier``
            is not a positive integer coprime to ``modulus``.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import modmul_const
        >>> @qmc.qkernel
        ... def mul(reg: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        ...     return modmul_const(reg, multiplier=2, modulus=15)
    """
    if (multiplier is None) != (modulus is None):
        raise ValueError(
            "modmul_const requires multiplier and modulus to be both given "
            "(concrete mode) or both omitted (abstract mode); got "
            f"multiplier={multiplier}, modulus={modulus}."
        )
    abstract = multiplier is None
    if not abstract:
        assert modulus is not None and multiplier is not None
        if modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {modulus}.")
        if multiplier <= 0 or sp.igcd(multiplier, modulus) != 1:
            raise ValueError(
                f"multiplier must be a positive integer coprime to modulus; "
                f"got multiplier={multiplier}, modulus={modulus}."
            )

    size = _concrete_size(reg)
    rotation = None
    if not abstract and size is not None:
        assert multiplier is not None and modulus is not None
        rotation = _is_cyclic_shift(multiplier, modulus, size)

    if rotation is not None:
        control_out, reg_out = _emit_cyclic_shift_body(reg, rotation, control)
        if control is None:
            return reg_out
        assert control_out is not None
        return control_out, reg_out

    # Abstract, symbolic width, or non-rotation constant: emit a resource-modeled
    # estimation-only box (no executable body).
    return _emit_modmul_box(
        reg,
        multiplier=multiplier,
        modulus=modulus,
        size=size,
        control=control,
    )


def _emit_modmul_box(
    reg: Vector[Qubit],
    *,
    multiplier: int | None,
    modulus: int | None,
    size: int | None,
    control: Qubit | None = None,
) -> Vector[Qubit] | tuple[Qubit, Vector[Qubit]]:
    """Emit a bodyless resource-modeled ``modmul_const`` invocation.

    Args:
        reg (Vector[Qubit]): Register consumed and returned by the box.
        multiplier (int | None): Constant multiplier ``a`` recorded in attrs,
            or ``None`` in abstract mode.
        modulus (int | None): Modulus ``N`` recorded in attrs, or ``None`` in
            abstract mode.
        size (int | None): Concrete register width when known, recorded as
            ``register_bits`` so models can read it without an operand shape.
        control (Qubit | None): Optional control qubit consumed and returned
            alongside the register. Defaults to ``None``.

    Returns:
        Vector[Qubit] | tuple[Qubit, Vector[Qubit]]: Next-version register
        handle, or ``(control, register)`` when controlled.
    """
    label = "abstract" if multiplier is None else f"a={multiplier}"
    consumed = reg.consume(operation_name=f"modmul_const[{label}]")
    result = consumed.value.next_version()
    ref = CallableRef(namespace="qamomile.stdlib", name=_MODMUL_NAME)
    num_controls = 0 if control is None else 1
    attrs: dict[str, Any] = {
        "kind": "composite",
        "gate_type": CompositeGateType.CUSTOM.name,
        "num_control_qubits": num_controls,
        "num_target_qubits": size or 0,
        "custom_name": _MODMUL_NAME,
        "strategy_name": None,
        "default_policy": CallPolicy.PRESERVE_BOX.name,
        # This box has resource models but no executable body, so it is
        # estimation-only; the emit guard rejects transpiling it.
        "estimation_only": True,
        # Pin the default (literature) model explicitly so the default policy is
        # not silently dependent on binding order.
        "default_estimate_kind": "literature",
        "multiplier": multiplier,
        "modulus": modulus,
        "register_bits": size,
    }
    operands: list[Value]
    results: list[Value]
    if control is None:
        operands = [consumed.value]
        results = [result]
        operand_names = ["reg"]
        transform = CallTransform.DIRECT
    else:
        consumed_control = control.consume(operation_name="modmul_const[control]")
        control_result = consumed_control.value.next_version()
        operands = [consumed_control.value, consumed.value]
        results = [control_result, result]
        operand_names = ["control", "reg"]
        transform = CallTransform.CONTROLLED
    op = InvokeOperation(
        operands=operands,
        results=results,
        target=ref,
        transform=transform,
        attrs=attrs,
        definition=CallableDef(
            ref=ref,
            signature=signature_from_values(
                operands,
                results,
                operand_names=operand_names,
                result_names=operand_names,
            ),
            resource_models=_modmul_resource_models(),
            default_policy=CallPolicy.PRESERVE_BOX,
            attrs=attrs,
        ),
    )
    get_current_tracer().add_operation(op)
    reg_out = type(reg)._create_from_value(value=result, shape=reg.shape)
    if control is None:
        return reg_out
    control_out = Qubit(
        value=control_result,
        parent=consumed_control.parent,
        indices=consumed_control.indices,
    )
    return control_out, reg_out


__all__ = ["modmul_const"]
