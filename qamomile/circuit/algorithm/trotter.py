"""Suzuki-Trotter time evolution as a self-recursive ``@qkernel``.

The public :func:`trotterized_time_evolution` wrapper validates the
``hamiltonian`` length and ``order`` and then delegates to the
``@qkernel`` :func:`_trotter_evolve`, which slices the evolution into
``step`` Trotter steps and applies a single-step operator from
:func:`_suzuki_trotter_step`.  That step kernel branches on ``order``:

* ``order == 1`` — Lie-Trotter forward sweep (base case).
* ``order == 2`` — Strang splitting with the palindrome center
  merged (base case).
* ``order >= 4`` (even) — Suzuki's fractal recursion

    .. math::

        S_{2k}(\\Delta t) = S_{2k-2}(p_k \\Delta t)^2
                            S_{2k-2}((1 - 4 p_k)\\Delta t)
                            S_{2k-2}(p_k \\Delta t)^2,
        \\quad p_k = \\frac{1}{4 - 4^{1/(2k-1)}}.

  The recursion calls back into :func:`_suzuki_trotter_step` with
  ``order - 2``.  Qamomile's transpiler resolves the self-call by
  iterating inline + partial-eval under the concrete ``order`` binding,
  so the emitted circuit is flat regardless of recursion depth.

``order`` must be bound to a compile-time constant at transpile time —
without it the base-case ``if`` never folds and the unroll loop has
nothing to terminate on.  Only ``order == 1`` or even orders
``2, 4, 6, ...`` are accepted; other values raise ``ValueError`` at
the call site.  ``hamiltonian`` must contain at least two terms.

Example::

    import qamomile.circuit as qmc
    import qamomile.observable as qm_o
    from qamomile.circuit.algorithm.trotter import (
        trotterized_time_evolution,
    )

    @qmc.qkernel
    def my_circuit(
        Hs: qmc.Vector[qmc.Observable],
        gamma: qmc.Float,
        order: qmc.UInt,
        step: qmc.UInt,
    ) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(1, name="q")
        q = trotterized_time_evolution(q, Hs, order, gamma, step)
        return q

    Hs = [qm_o.Z(0), qm_o.X(0)]  # list of qamomile.observable.Hamiltonian
"""

from __future__ import annotations

from typing import Sequence

import qamomile.circuit as qmc
from qamomile.observable import Hamiltonian


@qmc.qkernel
def _suzuki_trotter_step(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    dt: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply one Suzuki-Trotter step of the given order at time ``dt``.

    Self-recursive: for ``order >= 4`` the body calls itself with
    ``order - 2`` five times per the Suzuki fractal formula.  The
    transpiler unrolls the recursion under a concrete ``order`` binding.
    """
    n = hamiltonian.shape[0]
    if order == 1:
        # Lie-Trotter base case
        for i in qmc.range(n):
            q = qmc.pauli_evolve(q, hamiltonian[i], dt)
    else:
        if order == 2:
            # Strang base case (palindrome center merged)
            last = n - 1
            half_dt = 0.5 * dt
            for i in qmc.range(last):
                q = qmc.pauli_evolve(q, hamiltonian[i], half_dt)
            q = qmc.pauli_evolve(q, hamiltonian[last], dt)
            for j in qmc.range(last):
                rev = last - 1 - j
                q = qmc.pauli_evolve(q, hamiltonian[rev], half_dt)
        else:
            # Suzuki recursion: S_{2k} built from five S_{2k-2} blocks
            p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
            w = 1.0 - 4.0 * p
            q = _suzuki_trotter_step(q, hamiltonian, order - 2, p * dt)
            q = _suzuki_trotter_step(q, hamiltonian, order - 2, p * dt)
            q = _suzuki_trotter_step(q, hamiltonian, order - 2, w * dt)
            q = _suzuki_trotter_step(q, hamiltonian, order - 2, p * dt)
            q = _suzuki_trotter_step(q, hamiltonian, order - 2, p * dt)
    return q


@qmc.qkernel
def _trotter_evolve(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Vector[qmc.Observable],
    order: qmc.UInt,
    gamma: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Slice the evolution into ``step`` Suzuki-Trotter steps."""
    dt = gamma / step
    for _ in qmc.range(step):
        q = _suzuki_trotter_step(q, hamiltonian, order, dt)
    return q


# ======================================================================
# Argument resolution and validation
# ======================================================================


def _resolve_order(order: int | qmc.UInt) -> int | None:
    """Return ``order`` as a Python int when known, else ``None``.

    A symbolic ``UInt`` (no compile-time constant) returns ``None`` so
    that validation silently defers to the next re-trace with bindings.
    """
    if isinstance(order, qmc.UInt):
        if order.value.is_constant():
            return int(order.value.get_const())
        return None
    if isinstance(order, int):
        return order
    return None


def _resolve_hamiltonian_len(
    hamiltonian: qmc.Vector[qmc.Observable] | Sequence[Hamiltonian],
) -> int | None:
    """Return the number of sub-Hamiltonian terms when known.

    A bound ``Vector[qmc.Observable]`` (the qkernel-side handle for a
    vector of ``qamomile.observable.Hamiltonian`` objects) stores
    ``_shape`` as an int tuple (see ``qkernel._create_bound_input``);
    cache-trace Vectors carry a symbolic ``UInt`` in shape[0] and
    return ``None``.
    """
    if isinstance(hamiltonian, qmc.Vector):
        dim = hamiltonian.shape[0]
        if isinstance(dim, int):
            return dim
        if isinstance(dim, qmc.UInt) and dim.value.is_constant():
            return int(dim.value.get_const())
        return None
    if isinstance(hamiltonian, (list, tuple)):
        return len(hamiltonian)
    return None


def trotterized_time_evolution(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Vector[qmc.Observable] | Sequence[Hamiltonian],
    order: int | qmc.UInt,
    gamma: float | qmc.Float,
    step: int | qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Suzuki-Trotter time evolution ``exp(-i gamma H)`` to *q*.

    ``H = sum_k hamiltonian[k]``.  The evolution is split into ``step``
    Trotter slices of size ``dt = gamma / step``; each slice applies
    the ``order``-th Suzuki-Trotter formula via :func:`_trotter_evolve`.

    Args:
        q: Qubit register handle.
        hamiltonian: ``qmc.Vector[qmc.Observable]`` when called from a
            ``@qkernel`` (the handle type for a vector of Hamiltonians),
            or a Python list of ``qamomile.observable.Hamiltonian``
            objects.  At least two sub-Hamiltonian terms are required.
        order: Approximation order — ``1`` or a positive even integer
            (``2``, ``4``, ``6``, …).  Must be a compile-time constant.
        gamma: Total evolution time.
        step: Number of Trotter slices.

    Returns:
        The evolved qubit register.

    Raises:
        ValueError: If ``hamiltonian`` has fewer than two terms, if
            ``order`` is a ``bool``, or if ``order`` is not ``1`` or a
            positive even integer.  When these arguments are still
            symbolic (e.g. the enclosing kernel has not been re-traced
            with bindings yet) validation silently defers.
    """
    # ``bool`` is a subclass of ``int``; reject it before numeric checks
    # so ``order=True`` does not silently satisfy ``order == 1``.
    if isinstance(order, bool):
        raise ValueError(f"order must be int or qmc.UInt, got bool ({order})")

    o = _resolve_order(order)
    if o is not None and not (o == 1 or (o >= 2 and o % 2 == 0)):
        raise ValueError(f"order must be 1 or a positive even integer, got {o}")

    n = _resolve_hamiltonian_len(hamiltonian)
    if n is not None and n < 2:
        raise ValueError(f"hamiltonian must contain at least 2 terms, got {n}")

    return _trotter_evolve(q, hamiltonian, order, gamma, step)


__all__ = ["trotterized_time_evolution"]
