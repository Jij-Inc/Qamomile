"""Regression tests for exact complete-pair transform depth scheduling."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.errors import AffineTypeError


@qmc.qkernel
def _forward_complete_pair_transform(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply diagonal-before complete pairs followed by a mirror layer."""
    size = register.shape[0]
    for outer in qmc.range(size - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(size // 2):
        mirror = size - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _backward_complete_pair_transform(
    register: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply a mirror layer followed by complete pairs and diagonal gates."""
    size = register.shape[0]
    for index in qmc.range(size // 2):
        mirror = size - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    for outer in qmc.range(size):
        for inner in qmc.range(outer):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                -0.125,
            )
        register[outer] = qmc.h(register[outer])
    return register


@qmc.qkernel
def _forward_transform_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate and apply the forward complete-pair transform."""
    return _forward_complete_pair_transform(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _backward_transform_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate and apply the backward complete-pair transform."""
    return _backward_complete_pair_transform(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _stdlib_qft_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate a register and apply the public standard QFT."""
    return qmc.qft(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _stdlib_iqft_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate a register and apply the public standard inverse QFT."""
    return qmc.iqft(qmc.qubit_array(n, "register"))


@qmc.qkernel
def _unrolled_forward_four() -> qmc.Vector[qmc.Qubit]:
    """Apply the four-wire forward pattern without loops or helper calls."""
    register = qmc.qubit_array(4, "register")
    register[3] = qmc.h(register[3])
    register[3], register[2] = qmc.rzz(register[3], register[2], 0.125)
    register[3], register[1] = qmc.rzz(register[3], register[1], 0.125)
    register[3], register[0] = qmc.rzz(register[3], register[0], 0.125)
    register[2] = qmc.h(register[2])
    register[2], register[1] = qmc.rzz(register[2], register[1], 0.125)
    register[2], register[0] = qmc.rzz(register[2], register[0], 0.125)
    register[1] = qmc.h(register[1])
    register[1], register[0] = qmc.rzz(register[1], register[0], 0.125)
    register[0] = qmc.h(register[0])
    register[0], register[3] = qmc.swap(register[0], register[3])
    register[1], register[2] = qmc.swap(register[1], register[2])
    return register


@qmc.qkernel
def _unrolled_backward_four() -> qmc.Vector[qmc.Qubit]:
    """Apply the four-wire backward pattern without loops or helper calls."""
    register = qmc.qubit_array(4, "register")
    register[0], register[3] = qmc.swap(register[0], register[3])
    register[1], register[2] = qmc.swap(register[1], register[2])
    register[0] = qmc.h(register[0])
    register[1], register[0] = qmc.rzz(register[1], register[0], -0.125)
    register[1] = qmc.h(register[1])
    register[2], register[0] = qmc.rzz(register[2], register[0], -0.125)
    register[2], register[1] = qmc.rzz(register[2], register[1], -0.125)
    register[2] = qmc.h(register[2])
    register[3], register[0] = qmc.rzz(register[3], register[0], -0.125)
    register[3], register[1] = qmc.rzz(register[3], register[1], -0.125)
    register[3], register[2] = qmc.rzz(register[3], register[2], -0.125)
    register[3] = qmc.h(register[3])
    return register


@qmc.qkernel
def _skipped_pair_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Skip adjacent partners so the complete-pair proof cannot apply."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 2, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _reversed_inner_order_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Visit every suffix pair in the unsupported reverse inner order."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(n - 1, outer, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _wrong_diagonal_side_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Place the diagonal gate after a suffix pair row."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n):
        for inner in qmc.range(outer + 1, n):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
        register[outer] = qmc.h(register[outer])
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _incomplete_mirror_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Omit the outer mirror pair after an otherwise matching core."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(1, n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _shifted_mirror_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Use the right mirror count with the wrong endpoint pairing."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        shifted = index + 1
        mirror = n - index - 1
        register[shifted], register[mirror] = qmc.swap(
            register[shifted], register[mirror]
        )
    return register


@qmc.qkernel
def _swapped_result_mapping_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Store pair-gate successors at the opposite source-level indices."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            left, right = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
            register[inner] = left
            register[outer] = right
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _conditional_extra_gate_transform(
    n: qmc.UInt,
    flag: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Add conditional quantum work to an otherwise matching outer row."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        if flag:
            register[outer] = qmc.x(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _workspace_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate body-local workspace so the compact proof must reject."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        _scratch = qmc.qubit("scratch")
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _inner_workspace_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate workspace inside each pair row so the proof must reject."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            _scratch = qmc.qubit("scratch")
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _primitive_only_outer_loop(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply two primitive gates per outer iteration without a nested loop."""
    register = qmc.qubit_array(n, "register")
    for index in qmc.range(n):
        register[index] = qmc.h(register[index])
        register[index] = qmc.rz(register[index], 0.125)
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(register[index], register[mirror])
    return register


@qmc.qkernel
def _two_nested_loops_without_diagonal(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply two nested gate loops without a diagonal primitive."""
    register = qmc.qubit_array(n, "register")
    for outer in qmc.range(n):
        for left in qmc.range(outer):
            register[left] = qmc.h(register[left])
        for right in qmc.range(outer):
            register[right] = qmc.x(register[right])
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(register[index], register[mirror])
    return register


@qmc.qkernel
def _carried_angle_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Carry a classical gate parameter between complete-pair rows."""
    register = qmc.qubit_array(n, "register")
    angle = qmc.float_(0.125)
    for outer in qmc.range(n - 1, -1, -1):
        angle = angle + 0.125
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            register[outer], register[inner] = qmc.rzz(
                register[outer],
                register[inner],
                angle,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _controlled_pair_transform(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Share an explicit coherent control across every pair operation."""
    control = qmc.qubit("control")
    register = qmc.qubit_array(n, "register")
    controlled_rzz = qmc.control(qmc.rzz)
    for outer in qmc.range(n - 1, -1, -1):
        register[outer] = qmc.h(register[outer])
        for inner in qmc.range(outer - 1, -1, -1):
            control, register[outer], register[inner] = controlled_rzz(
                control,
                register[outer],
                register[inner],
                0.125,
            )
    for index in qmc.range(n // 2):
        mirror = n - index - 1
        register[index], register[mirror] = qmc.swap(
            register[index],
            register[mirror],
        )
    return register


@qmc.qkernel
def _prior_first_wire() -> qmc.Vector[qmc.Qubit]:
    """Advance the first forward traversal wire before the transform."""
    register = qmc.qubit_array(4, "register")
    register[3] = qmc.x(register[3])
    return _forward_complete_pair_transform(register)


@qmc.qkernel
def _prior_middle_wire() -> qmc.Vector[qmc.Qubit]:
    """Advance a non-frontier traversal wire before the transform."""
    register = qmc.qubit_array(4, "register")
    register[1] = qmc.x(register[1])
    return _forward_complete_pair_transform(register)


@qmc.qkernel
def _prior_mixed_wires() -> qmc.Vector[qmc.Qubit]:
    """Advance frontier and non-frontier wires in one prior event."""
    register = qmc.qubit_array(4, "register")
    register[3], register[1] = qmc.rzz(register[3], register[1], 0.25)
    return _forward_complete_pair_transform(register)


@qmc.qkernel
def _prior_disjoint_wire() -> qmc.Vector[qmc.Qubit]:
    """Keep unrelated prior work parallel with the complete-pair transform."""
    unrelated = qmc.h(qmc.qubit("unrelated"))
    register = _forward_complete_pair_transform(qmc.qubit_array(4, "register"))
    unrelated = qmc.x(unrelated)
    return register


@qmc.qkernel
def _uniform_prior_layer() -> qmc.Vector[qmc.Qubit]:
    """Synchronize every transform wire with one disjoint gate layer."""
    register = qmc.qubit_array(4, "register")
    for index in qmc.range(4):
        register[index] = qmc.x(register[index])
    return _forward_complete_pair_transform(register)


@qmc.qkernel
def _downstream_early_wire() -> qmc.Vector[qmc.Qubit]:
    """Use an early-finishing wire after the compact transform."""
    register = _forward_complete_pair_transform(qmc.qubit_array(4, "register"))
    register[1] = qmc.x(register[1])
    return register


@qmc.qkernel
def _transform_on_strided_view() -> qmc.Vector[qmc.Qubit]:
    """Map the complete-pair proof through a nonunit-stride view."""
    register = qmc.qubit_array(9, "register")
    _forward_complete_pair_transform(register[1:9:2])
    return register


@qmc.qkernel
def _prior_outside_strided_view() -> qmc.Vector[qmc.Qubit]:
    """Keep work outside a transformed view independent."""
    register = qmc.qubit_array(9, "register")
    register[0] = qmc.x(register[0])
    _forward_complete_pair_transform(register[1:9:2])
    return register


@qmc.qkernel
def _prior_inside_strided_view() -> qmc.Vector[qmc.Qubit]:
    """Advance one root wire covered by a transformed view."""
    register = qmc.qubit_array(9, "register")
    register[3] = qmc.x(register[3])
    _forward_complete_pair_transform(register[1:9:2])
    return register


@qmc.qkernel
def _repeat_transform_helper() -> qmc.Vector[qmc.Qubit]:
    """Apply the same nonuniform complete-pair helper twice."""
    register = qmc.qubit_array(4, "register")
    register = _forward_complete_pair_transform(register)
    register = _forward_complete_pair_transform(register)
    return register


@qmc.qkernel
def _sum_transform_helper() -> qmc.Vector[qmc.Qubit]:
    """Apply the helper twice through an enclosing range sum."""
    register = qmc.qubit_array(4, "register")
    for iteration in qmc.range(2):
        _scratch = qmc.qubit_array(iteration, "scratch")
        register = _forward_complete_pair_transform(register)
    return register


@qmc.qkernel
def _controlled_transform_helper() -> qmc.Vector[qmc.Qubit]:
    """Control a matched transform through the public control wrapper."""
    control = qmc.qubit("control")
    register = qmc.qubit_array(4, "register")
    control, register = qmc.control(_forward_complete_pair_transform)(
        control,
        register,
    )
    return register


@qmc.qkernel
def _conditional_transform_helper(flag: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Apply the matched helper only on one compile-time branch."""
    register = qmc.qubit_array(4, "register")
    if flag:
        register = _forward_complete_pair_transform(register)
    return register


def _expected_depth(n: int) -> qmc.DepthResources:
    """Build the exact depth profile of either transform direction.

    Args:
        n (int): Number of transformed wires.

    Returns:
        qmc.DepthResources: Exact field-by-field critical path.
    """
    mirror_depth = int(n >= 2)
    return qmc.DepthResources(
        depth=max(0, 2 * n - 1) + mirror_depth,
        clifford_depth=n + mirror_depth,
        rotation_depth=max(0, 2 * n - 3),
        non_clifford_depth=max(0, 2 * n - 3),
        gate_depth=max(0, 2 * n - 1) + mirror_depth,
    )


def _expected_gates(n: int) -> qmc.GateResources:
    """Build the exact gate profile of either transform direction.

    Args:
        n (int): Number of transformed wires.

    Returns:
        qmc.GateResources: Exact field-by-field gate counts.
    """
    pairs = n * (n - 1) // 2
    mirrors = n // 2
    return qmc.GateResources(
        total=n + pairs + mirrors,
        single_qubit=n,
        two_qubit=pairs + mirrors,
        clifford=n + mirrors,
        rotation=pairs,
        non_clifford=pairs,
    )


@pytest.mark.parametrize(
    "kernel",
    [_forward_transform_circuit, _backward_transform_circuit],
    ids=["forward", "backward"],
)
@pytest.mark.parametrize("n", range(9))
def test_complete_pair_transform_direct_and_substitution_are_exact(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
    n: int,
) -> None:
    """Direct inputs and late substitution agree for every small boundary."""
    symbolic = kernel.estimate_resources()
    direct = kernel.estimate_resources(inputs={"n": n})
    substituted = symbolic.substitute(n=n)

    assert direct.gates == _expected_gates(n)
    assert direct.depth == _expected_depth(n)
    assert direct.gates == substituted.gates
    assert direct.depth == substituted.depth
    assert direct.width == substituted.width
    assert direct.quality is qmc.EstimateQuality.EXACT
    assert substituted.quality is qmc.EstimateQuality.EXACT
    assert direct.assumptions == ()
    assert substituted.assumptions == ()


@pytest.mark.parametrize(
    "kernel",
    [_forward_transform_circuit, _backward_transform_circuit],
    ids=["forward", "backward"],
)
def test_complete_pair_transform_symbolic_profile_is_exact(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """The unspecialized proof remains exact and retains only public inputs."""
    estimate = kernel.estimate_resources()

    assert set(estimate.parameters) == {"n"}
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()
    for n in range(9):
        specialized = estimate.substitute(n=n)
        assert specialized.gates == _expected_gates(n)
        assert specialized.depth == _expected_depth(n)


@pytest.mark.parametrize(
    "kernel",
    [_stdlib_qft_circuit, _stdlib_iqft_circuit],
    ids=["qft", "iqft"],
)
def test_stdlib_fourier_transforms_use_the_exact_compound_schedule(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Public Fourier kernels are exact for every specialization route."""
    symbolic = kernel.estimate_resources()

    assert symbolic.quality is qmc.EstimateQuality.EXACT
    assert symbolic.assumptions == ()
    for n in range(9):
        direct = kernel.estimate_resources(inputs={"n": n})
        substituted = symbolic.substitute(n=n)
        assert direct.gates == _expected_gates(n)
        assert direct.depth == _expected_depth(n)
        assert direct.gates == substituted.gates
        assert direct.depth == substituted.depth
        assert direct.quality is qmc.EstimateQuality.EXACT
        assert substituted.quality is qmc.EstimateQuality.EXACT
        assert direct.assumptions == ()
        assert substituted.assumptions == ()


@pytest.mark.parametrize(
    ("looped", "unrolled"),
    [
        (_forward_transform_circuit, _unrolled_forward_four),
        (_backward_transform_circuit, _unrolled_backward_four),
    ],
    ids=["forward", "backward"],
)
def test_complete_pair_transform_matches_explicit_unroll(
    looped: Callable[..., qmc.Vector[qmc.Qubit]],
    unrolled: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """The compact theorem agrees with an independently scheduled circuit."""
    compact = looped.estimate_resources(inputs={"n": 4})
    explicit = unrolled.estimate_resources()

    assert compact.gates == explicit.gates
    assert compact.depth == explicit.depth
    assert compact.depth == _expected_depth(4)
    assert compact.quality is qmc.EstimateQuality.EXACT
    assert explicit.quality is qmc.EstimateQuality.EXACT


@pytest.mark.parametrize(
    "kernel",
    [
        _skipped_pair_transform,
        _conditional_extra_gate_transform,
        _workspace_transform,
        _inner_workspace_transform,
        _carried_angle_transform,
    ],
    ids=[
        "bounds",
        "condition",
        "outer-workspace",
        "inner-workspace",
        "classical-carry",
    ],
)
def test_near_miss_structure_keeps_the_safe_fallback(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Unsupported structure cannot inherit the compact exact certificate."""
    estimate = kernel.estimate_resources()

    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.assumptions


def test_reverse_inner_order_is_not_mistaken_for_the_proven_schedule() -> None:
    """The same pair set in a different order has a longer critical path."""
    estimate = _reversed_inner_order_transform.estimate_resources(inputs={"n": 4})

    assert estimate.gates == _expected_gates(4)
    assert estimate.depth.depth == 11
    assert estimate.depth.gate_depth == 11


@pytest.mark.parametrize(
    "kernel",
    [_wrong_diagonal_side_transform, _incomplete_mirror_transform],
    ids=["diagonal-side", "mirror-bounds"],
)
def test_near_match_is_never_labeled_with_the_full_transform_formula(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """A near match may fall back, but cannot claim the wrong exact depth."""
    estimate = kernel.estimate_resources(inputs={"n": 4})

    assert not (
        estimate.quality is qmc.EstimateQuality.EXACT
        and estimate.depth == _expected_depth(4)
    )


def test_wrong_mirror_endpoints_reject_the_compound_schedule() -> None:
    """The mirror count alone cannot prove reversal endpoint coverage."""
    estimate = _shifted_mirror_transform.estimate_resources(inputs={"n": 5})

    assert estimate.gates == _expected_gates(5)
    assert not (
        estimate.quality is qmc.EstimateQuality.EXACT
        and estimate.depth == _expected_depth(5)
    )


def test_mismatched_pair_results_fail_before_resource_scheduling() -> None:
    """Affine validation rejects result remapping before pattern analysis."""
    with pytest.raises(AffineTypeError, match="different symbolic index"):
        _swapped_result_mapping_transform.estimate_resources(inputs={"n": 4})


@pytest.mark.parametrize(
    ("kernel", "expected_gates"),
    [
        (_primitive_only_outer_loop, 10),
        (_two_nested_loops_without_diagonal, 14),
    ],
    ids=["two-gates", "two-loops"],
)
def test_outer_loop_without_gate_and_nested_loop_fails_closed(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
    expected_gates: int,
) -> None:
    """A candidate must contain exactly one gate and one nested loop."""
    estimate = kernel.estimate_resources(inputs={"n": 4})

    assert estimate.gates.total == expected_gates
    assert not (
        estimate.quality is qmc.EstimateQuality.EXACT
        and estimate.depth == _expected_depth(4)
    )


def test_explicit_shared_control_rejects_the_pair_schedule() -> None:
    """One coherent control serializes pair gates across outer iterations."""
    estimate = _controlled_pair_transform.estimate_resources(inputs={"n": 4})

    assert estimate.depth.depth > _expected_depth(4).depth
    assert estimate.width.allocated_qubits > 4


@pytest.mark.parametrize(
    "kernel",
    [_prior_first_wire, _prior_middle_wire, _prior_mixed_wires],
    ids=["first", "middle", "mixed"],
)
def test_overlapping_prior_work_activates_the_entry_premise(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """Any nonuniform prior overlap keeps the aggregate composition safe."""
    estimate = kernel.estimate_resources()

    assert estimate.gates.total == 13
    assert estimate.depth.depth == 9
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        assumption.source == "dependency scheduler"
        for assumption in estimate.assumptions
    )


def test_disjoint_prior_work_remains_parallel_with_the_transform() -> None:
    """Work on another allocation does not violate the entry premise."""
    estimate = _prior_disjoint_wire.estimate_resources()

    assert estimate.gates.total == 14
    assert estimate.depth.depth == 8
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


def test_uniform_prior_layer_resynchronizes_the_complete_coverage() -> None:
    """A uniform full write discharges earlier entry synchronization needs."""
    estimate = _uniform_prior_layer.estimate_resources()

    assert estimate.gates.total == 16
    assert estimate.depth.depth == 9
    assert estimate.depth.clifford_depth == 6
    assert estimate.depth.rotation_depth == 5
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


def test_downstream_early_wire_discloses_nonuniform_completion() -> None:
    """A later early-wire use cannot consume a falsely uniform completion."""
    estimate = _downstream_early_wire.estimate_resources()

    assert estimate.gates.total == 13
    assert estimate.depth.depth == 9
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert any(
        "aggregate latency" in assumption.message for assumption in estimate.assumptions
    )


def test_helper_maps_complete_pair_coverage_through_a_strided_view() -> None:
    """An exact affine view preserves the compact transform proof."""
    estimate = _transform_on_strided_view.estimate_resources()

    assert estimate.width.allocated_qubits == 9
    assert estimate.gates == _expected_gates(4)
    assert estimate.depth == _expected_depth(4)
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_prior_work_outside_a_strided_view_stays_disjoint() -> None:
    """Full proof coverage excludes a concrete root wire outside the view."""
    estimate = _prior_outside_strided_view.estimate_resources()

    assert estimate.gates.total == 13
    assert estimate.depth.depth == 8
    assert estimate.quality is qmc.EstimateQuality.EXACT


def test_prior_work_inside_a_strided_view_violates_entry_synchrony() -> None:
    """The mapped coverage detects a prior root-wire overlap."""
    estimate = _prior_inside_strided_view.estimate_resources()

    assert estimate.gates.total == 13
    assert estimate.depth.depth == 9
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE


@pytest.mark.parametrize(
    "kernel",
    [_repeat_transform_helper, _sum_transform_helper],
    ids=["explicit-repeat", "range-sum"],
)
def test_repeated_nonuniform_transform_keeps_a_conservative_bound(
    kernel: Callable[..., qmc.Vector[qmc.Qubit]],
) -> None:
    """A second transform cannot assume the first one's outputs are uniform."""
    estimate = kernel.estimate_resources()

    assert estimate.gates.total == 24
    assert estimate.depth.depth == 16
    assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
    assert estimate.assumptions


def test_surrounding_control_does_not_reuse_the_uncontrolled_formula() -> None:
    """A controlled helper uses its control model instead of compact depth."""
    estimate = _controlled_transform_helper.estimate_resources()

    assert estimate.depth.depth > _expected_depth(4).depth
    assert estimate.width.allocated_qubits > 4


@pytest.mark.parametrize(("flag", "expected_gates"), [(0, 0), (1, 12)])
def test_conditional_composition_guards_the_compound_schedule(
    flag: int,
    expected_gates: int,
) -> None:
    """Conditional composition preserves the exact active compound branch."""
    estimate = _conditional_transform_helper.estimate_resources(inputs={"flag": flag})

    assert estimate.gates.total == expected_gates
    assert estimate.depth.depth == (8 if flag else 0)
    assert estimate.quality is qmc.EstimateQuality.EXACT
    assert estimate.assumptions == ()


def test_public_repeat_and_sum_preserve_nonuniform_completion_disclosure() -> None:
    """Public algebra cannot relabel a compact nonuniform result as exact."""
    once = _forward_complete_pair_transform.estimate_resources(inputs={"register": 4})
    iteration = sp.Dummy("iteration", integer=True, nonnegative=True)

    repeated = once.repeat(2)
    summed = once.sum_over(iteration, 0, 2)

    for estimate in (repeated, summed):
        assert estimate.gates.total == 24
        assert estimate.depth.depth == 16
        assert estimate.quality is qmc.EstimateQuality.CONSERVATIVE
        assert estimate.assumptions
