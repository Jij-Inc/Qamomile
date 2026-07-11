"""Argument-type validation for qkernel calls (plain and ``qmc.control``).

A scalar ``Qubit`` bound to a ``Vector[Qubit]`` parameter, a quantum handle
bound to a classical parameter, and similar arity mismatches must fail fast
with a clear ``TypeError`` at the call site instead of leaking an
``AttributeError`` from ``get_size`` (``'Qubit' object has no attribute
'shape'``) or silently miscompiling (a ``Qubit`` bound to a ``Float``
parameter emitting ``Rx(0.0)``).

The one asymmetry is intentional: a quantum array bound to a *scalar*
``Qubit`` parameter is a legitimate per-element broadcast in the control
path (one controlled application per target qubit, exercised end-to-end by
``controlled_native_broadcast_target`` in ``test_frontend_cross_backend_
execution.py``), so it is accepted there. The same shape on a plain qkernel
call does not broadcast -- it would silently drop qubits -- so it is
rejected.
"""

import pytest

import qamomile.circuit as qmc

pytest.importorskip("qiskit")
from qamomile.qiskit import QiskitTranspiler  # noqa: E402


# --------------------------------------------------------------------------
# Call targets with distinct parameter declarations.
# --------------------------------------------------------------------------
@qmc.qkernel
def _vec_target(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Declare a ``Vector[Qubit]`` register parameter."""
    return qmc.x(qs)


@qmc.qkernel
def _scalar_target(q: qmc.Qubit) -> qmc.Qubit:
    """Declare a scalar ``Qubit`` parameter."""
    return qmc.x(q)


@qmc.qkernel
def _rx_target(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Declare a scalar ``Qubit`` plus a classical ``Float`` parameter."""
    return qmc.rx(q, theta)


@qmc.qkernel
def _bit_target(b: qmc.Bit) -> qmc.Bit:
    """Declare a scalar ``Bit`` parameter."""
    return b


@qmc.qkernel
def _bitvec_target(bits: qmc.Vector[qmc.Bit]) -> qmc.Vector[qmc.Bit]:
    """Declare a ``Vector[Bit]`` parameter."""
    return bits


@qmc.qkernel
def _obsvec_target(q: qmc.Qubit, obs: qmc.Vector[qmc.Observable]) -> qmc.Float:
    """Declare a ``Vector[Observable]`` parameter alongside a qubit."""
    return qmc.expval(q, obs[0])


_c_vec = qmc.control(_vec_target, num_controls=1)
_c_scalar = qmc.control(_scalar_target, num_controls=1)
_c_rx = qmc.control(_rx_target, num_controls=1)


# --------------------------------------------------------------------------
# Mismatched-call kernels -- each must raise a clear ``TypeError`` at build.
# --------------------------------------------------------------------------
@qmc.qkernel
def _err_normal_scalar_into_vec() -> qmc.Bit:
    """P1 (plain): scalar ``Qubit`` passed to a ``Vector[Qubit]`` parameter."""
    q = qmc.qubit("q")
    out = _vec_target(q)
    return qmc.measure(out[0])


@qmc.qkernel
def _err_normal_vec_into_scalar() -> qmc.Bit:
    """P2 (plain): whole ``Vector[Qubit]`` passed to a scalar ``Qubit``."""
    qs = qmc.qubit_array(2, "qs")
    out = _scalar_target(qs)
    return qmc.measure(out)


@qmc.qkernel
def _err_normal_view_into_scalar() -> qmc.Bit:
    """P2 (plain): a ``VectorView`` passed to a scalar ``Qubit`` parameter."""
    qs = qmc.qubit_array(3, "qs")
    out = _scalar_target(qs[0:2])
    return qmc.measure(out)


@qmc.qkernel
def _err_normal_qubit_into_float() -> qmc.Bit:
    """P3 (plain): a quantum ``Qubit`` passed to a classical ``Float``."""
    qa = qmc.qubit("qa")
    qb = qmc.qubit("qb")
    qa = _rx_target(qa, qb)
    return qmc.measure(qa)


@qmc.qkernel
def _err_normal_vec_into_float() -> qmc.Bit:
    """P3 variant (plain): a ``Vector[Qubit]`` passed to a ``Float``."""
    qa = qmc.qubit("qa")
    qs = qmc.qubit_array(2, "qs")
    qa = _rx_target(qa, qs)
    return qmc.measure(qa)


@qmc.qkernel
def _err_normal_float_into_qubit() -> qmc.Bit:
    """Reverse P3 (plain): a classical ``Float`` passed to a ``Qubit``."""
    theta = qmc.float_(0.5)
    out = _scalar_target(theta)
    return qmc.measure(out)


@qmc.qkernel
def _err_normal_qubit_into_bit() -> qmc.Bit:
    """A quantum ``Qubit`` passed to a classical ``Bit`` parameter."""
    q = qmc.qubit("q")
    return _bit_target(q)


@qmc.qkernel
def _err_normal_float_into_bit() -> qmc.Bit:
    """A classical ``Float`` passed to a ``Bit`` parameter (wrong classical kind)."""
    f = qmc.float_(0.5)
    return _bit_target(f)


@qmc.qkernel
def _err_normal_qubit_into_obsvec() -> qmc.Float:
    """A quantum ``Qubit`` passed to a ``Vector[Observable]`` parameter."""
    a = qmc.qubit("a")
    q = qmc.qubit("q")
    return _obsvec_target(a, q)


@qmc.qkernel
def _err_control_scalar_into_vec() -> qmc.Bit:
    """P1 (control): scalar ``Qubit`` target into a ``Vector[Qubit]`` sub-kernel."""
    c = qmc.qubit("c")
    t = qmc.qubit("t")
    c = qmc.h(c)
    c, t = _c_vec(c, t)
    return qmc.measure(c)


@qmc.qkernel
def _err_control_qubit_into_float() -> qmc.Bit:
    """P3 (control): a quantum ``Qubit`` passed as the classical ``theta``."""
    c = qmc.qubit("c")
    t = qmc.qubit("t")
    qb = qmc.qubit("qb")
    c, t = _c_rx(c, t, theta=qb)
    return qmc.measure(c)


@qmc.qkernel
def _err_control_float_into_qubit() -> qmc.Bit:
    """Reverse P3 (control): a classical ``Float`` passed as a quantum target."""
    c = qmc.qubit("c")
    f = qmc.float_(0.5)
    c, out = _c_scalar(c, f)
    return qmc.measure(c)


# --------------------------------------------------------------------------
# Type-matching kernels -- each must build and transpile cleanly.
# --------------------------------------------------------------------------
@qmc.qkernel
def _ok_normal_scalar() -> qmc.Bit:
    """Plain call: scalar ``Qubit`` into a scalar ``Qubit`` parameter."""
    q = qmc.qubit("q")
    q = _scalar_target(q)
    return qmc.measure(q)


@qmc.qkernel
def _ok_normal_vec() -> qmc.Bit:
    """Plain call: ``Vector[Qubit]`` into a ``Vector[Qubit]`` parameter."""
    qs = qmc.qubit_array(2, "qs")
    qs = _vec_target(qs)
    return qmc.measure(qs[0])


@qmc.qkernel
def _ok_normal_singleton_vec() -> qmc.Bit:
    """Plain call: a length-1 register satisfies a ``Vector[Qubit]`` parameter."""
    qs = qmc.qubit_array(1, "qs")
    qs = _vec_target(qs)
    return qmc.measure(qs[0])


@qmc.qkernel
def _ok_normal_bitvec() -> qmc.Vector[qmc.Bit]:
    """Plain call: a measurement-derived ``Vector[Bit]`` into a ``Vector[Bit]`` parameter."""
    qs = qmc.qubit_array(2, "qs")
    bits = qmc.measure(qs)
    return _bitvec_target(bits)


@qmc.qkernel
def _ok_normal_bit() -> qmc.Bit:
    """Plain call: a ``Bit`` into a ``Bit`` parameter (classical-only, build path)."""
    b = qmc.bit(True)
    return _bit_target(b)


@qmc.qkernel
def _ok_normal_obsvec(obs: qmc.Vector[qmc.Observable]) -> qmc.Float:
    """Plain call: a ``Vector[Observable]`` forwarded to a ``Vector[Observable]`` parameter."""
    q = qmc.qubit("q")
    return _obsvec_target(q, obs)


@qmc.qkernel
def _ok_control_scalar() -> qmc.Bit:
    """Control: scalar ``Qubit`` target into a scalar ``Qubit`` sub-kernel."""
    c = qmc.qubit("c")
    t = qmc.qubit("t")
    c, t = _c_scalar(c, t)
    return qmc.measure(c)


@qmc.qkernel
def _ok_control_vec() -> qmc.Bit:
    """Control: ``Vector[Qubit]`` target into a ``Vector[Qubit]`` sub-kernel."""
    c = qmc.qubit("c")
    qs = qmc.qubit_array(2, "qs")
    c, qs = _c_vec(c, qs)
    return qmc.measure(c)


@qmc.qkernel
def _ok_control_broadcast_wholevec() -> qmc.Vector[qmc.Bit]:
    """Control broadcast: a whole ``Vector[Qubit]`` into a scalar target."""
    c = qmc.qubit("c")
    qs = qmc.qubit_array(2, "qs")
    c = qmc.x(c)
    c, qs = _c_scalar(c, qs)
    return qmc.measure(qs)


@qmc.qkernel
def _ok_control_broadcast_view() -> qmc.Vector[qmc.Bit]:
    """Control broadcast: a ``VectorView`` into a scalar target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[0], tail = _c_scalar(q[0], q[1:3])
    q[1:3] = tail
    return qmc.measure(q)


# Higher-rank quantum registers (``Matrix[Qubit]`` / ``Tensor[Qubit]``) are
# rejected at construction time: ``qubit_array((2, 2), ...)`` and a
# ``Matrix[Qubit]`` kernel parameter both raise ``NotImplementedError``
# because the qubit addressing path is rank-1 and a higher-rank register
# would silently alias distinct elements onto the same physical qubit. They
# therefore cannot serve as rank-mismatch vehicles for the call-site
# argument-type validation exercised here -- the rejection fires before the
# call is ever reached. That rejection is covered end-to-end in
# ``tests/circuit/test_multidim_quantum_rejection.py``.


# ``qmc.inverse`` binds caller handles to the wrapped kernel's declared
# parameters through its own path, so it must run the same argument
# validation as the plain and controlled calls.
_inv_rx = qmc.inverse(_rx_target)
_inv_vec = qmc.inverse(_vec_target)


@qmc.qkernel
def _err_inverse_qubit_into_float() -> qmc.Bit:
    """P3 (inverse): a quantum ``Qubit`` passed as an inverse gate's ``theta``."""
    qa = qmc.qubit("qa")
    qb = qmc.qubit("qb")
    qa = _inv_rx(qa, qb)
    return qmc.measure(qa)


@qmc.qkernel
def _ok_inverse_vec() -> qmc.Bit:
    """Inverse: a ``Vector[Qubit]`` into a matching ``Vector[Qubit]`` inverse target."""
    qs = qmc.qubit_array(2, "qs")
    qs = _inv_vec(qs)
    return qmc.measure(qs[0])


_ERROR_CASES = [
    ("normal_scalar_into_vec", _err_normal_scalar_into_vec),
    ("normal_vec_into_scalar", _err_normal_vec_into_scalar),
    ("normal_view_into_scalar", _err_normal_view_into_scalar),
    ("normal_qubit_into_float", _err_normal_qubit_into_float),
    ("normal_vec_into_float", _err_normal_vec_into_float),
    ("normal_float_into_qubit", _err_normal_float_into_qubit),
    ("normal_qubit_into_bit", _err_normal_qubit_into_bit),
    ("normal_float_into_bit", _err_normal_float_into_bit),
    ("normal_qubit_into_obsvec", _err_normal_qubit_into_obsvec),
    ("control_scalar_into_vec", _err_control_scalar_into_vec),
    ("control_qubit_into_float", _err_control_qubit_into_float),
    ("control_float_into_qubit", _err_control_float_into_qubit),
    ("inverse_qubit_into_float", _err_inverse_qubit_into_float),
]

_OK_CASES = [
    ("normal_scalar", _ok_normal_scalar),
    ("normal_vec", _ok_normal_vec),
    ("normal_singleton_vec", _ok_normal_singleton_vec),
    ("normal_bitvec", _ok_normal_bitvec),
    ("control_scalar", _ok_control_scalar),
    ("control_vec", _ok_control_vec),
    ("control_broadcast_wholevec", _ok_control_broadcast_wholevec),
    ("control_broadcast_view", _ok_control_broadcast_view),
    ("inverse_vec", _ok_inverse_vec),
]

# Classical-only valid cases that have no quantum content (Bit pass-through)
# or carry an unbound Observable parameter, so they exercise the build path
# (where the argument validation fires) but are not parametrized through the
# no-bindings transpile path.
_OK_BUILD_ONLY_CASES = [
    ("normal_bit", _ok_normal_bit),
    ("normal_obsvec", _ok_normal_obsvec),
]

_ERROR_IDS = [case_id for case_id, _ in _ERROR_CASES]
_ERROR_KERNELS = [kernel for _, kernel in _ERROR_CASES]
_OK_IDS = [case_id for case_id, _ in _OK_CASES]
_OK_KERNELS = [kernel for _, kernel in _OK_CASES]


@pytest.mark.parametrize("kernel", _ERROR_KERNELS, ids=_ERROR_IDS)
def test_arg_type_mismatch_raises_on_build(kernel):
    """A mismatched-argument qkernel raises ``TypeError`` from ``build``."""
    with pytest.raises(TypeError, match="parameter"):
        kernel.build(parameters=None)


@pytest.mark.parametrize("kernel", _ERROR_KERNELS, ids=_ERROR_IDS)
def test_arg_type_mismatch_raises_on_transpile(kernel):
    """A mismatched-argument qkernel raises ``TypeError`` from ``transpile``."""
    with pytest.raises(TypeError, match="parameter"):
        QiskitTranspiler().transpile(kernel)


@pytest.mark.parametrize("kernel", _OK_KERNELS, ids=_OK_IDS)
def test_arg_type_match_builds(kernel):
    """A type-correct qkernel (including control broadcast) builds cleanly."""
    block = kernel.build(parameters=None)
    assert block is not None


@pytest.mark.parametrize("kernel", _OK_KERNELS, ids=_OK_IDS)
def test_arg_type_match_transpiles(kernel):
    """A type-correct qkernel (including control broadcast) transpiles cleanly."""
    executable = QiskitTranspiler().transpile(kernel)
    assert executable is not None


@pytest.mark.parametrize(
    "kernel",
    [k for _, k in _OK_BUILD_ONLY_CASES],
    ids=[i for i, _ in _OK_BUILD_ONLY_CASES],
)
def test_classical_arg_type_match_builds(kernel):
    """A matching Bit / Vector[Observable] argument builds without a false positive.

    Covers the classical types beyond Float/UInt -- a Bit into a Bit
    parameter and a Vector[Observable] into a Vector[Observable] parameter --
    so the broadened classical coverage does not over-reject valid handles.
    """
    block = kernel.build(parameters=None)
    assert block is not None


def test_qubit_into_float_does_not_silently_emit_rx_zero():
    """P3 regression: a ``Qubit`` bound to a ``Float`` must raise, not emit Rx(0.0).

    This is the worst-case mismatch -- before the fix it transpiled silently
    into an ``Rx(0.0)`` gate -- so it gets its own explicit guard separate
    from the parametrized sweep.
    """
    with pytest.raises(TypeError, match="classical parameter but received quantum"):
        QiskitTranspiler().transpile(_err_normal_qubit_into_float)


def test_control_broadcast_emits_one_application_per_target_qubit():
    """A scalar-target controlled gate broadcast over a 2-qubit register.

    Confirms the accepted (non-error) broadcast path is not a silent no-op:
    the controlled sub-block is applied once per target qubit, so the emitted
    circuit carries two controlled instructions.
    """
    executable = QiskitTranspiler().transpile(_ok_control_broadcast_wholevec)
    circuit = executable.get_first_circuit()
    controlled_op_count = sum(
        instruction.operation.num_ctrl_qubits == 1
        for instruction in circuit.data
        if hasattr(instruction.operation, "num_ctrl_qubits")
    )
    assert controlled_op_count == 2
