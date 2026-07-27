"""Grover search building blocks with query-complexity resource estimation.

This module provides :func:`grover_search`, the amplitude-amplification loop
from the book's Section 4.1 ("Search algorithms a la Grover"), and helpers to
compute the optimal iteration count. Grover search over ``N = 2**n`` items with
``m`` marked solutions needs ``O(sqrt(N/m))`` iterations, each applying one
oracle query and one diffusion (reflection about the uniform superposition).

The oracle is supplied by the caller as a costed opaque box (e.g.
``qmc.opaque(..., cost=...)``), so the total gate cost is
``iterations x (oracle_cost + diffusion_cost)`` while the *query* complexity is
the universal ``O(sqrt(N/m))``. Leaving the iteration count symbolic makes
``estimate_resources`` report that scaling directly; :func:`grover_iteration_count`
returns the concrete or symbolic ``floor((pi/4) sqrt(N/m))``. Plug it in with the
``inputs`` argument to get the optimal query complexity
straight from one estimate call::

    est = kernel.estimate_resources(
        inputs={"iterations": grover_iteration_count(n, m)}
    )
    est.calls.queries_by_name[oracle_name]  # floor((pi/4) sqrt(2^n / m))
"""

from __future__ import annotations

import numbers
from typing import Any, Callable, cast

import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.operation.control_flow import for_loop
from qamomile.circuit.frontend.qkernel_like import QKernelLike


def grover_iteration_count(
    num_qubits: int | sp.Expr,
    num_marked: int | sp.Expr = 1,
) -> int | sp.Expr:
    """Return the optimal Grover iteration count ``floor((pi/4) sqrt(N/m))``.

    Args:
        num_qubits (int | sp.Expr): Number of search qubits ``n`` (search space
            ``N = 2**n``). May be symbolic.
        num_marked (int | sp.Expr): Number of marked solutions ``m``. Defaults
            to ``1``.

    Returns:
        int | sp.Expr: Concrete iteration count when both arguments are concrete
        integers (Python or NumPy), otherwise the symbolic expression
        ``floor((pi/4) sqrt(2**n / m))``.

    Raises:
        ValueError: If concrete ``num_qubits`` or ``num_marked`` is not
            positive.

    Example:
        >>> grover_iteration_count(4, 1)
        3
    """
    # Normalize NumPy integer scalars (np.int64, ...) to Python ints so they
    # take the concrete, positivity-validated path rather than falling through
    # to the symbolic branch. Route through ``Any`` so the numeric-tower check
    # is not narrowed away by the declared ``int | sp.Expr`` annotation.
    n_in: Any = num_qubits
    m_in: Any = num_marked
    n_val = int(n_in) if isinstance(n_in, numbers.Integral) else n_in
    m_val = int(m_in) if isinstance(m_in, numbers.Integral) else m_in
    if isinstance(n_val, int) and isinstance(m_val, int):
        if n_val <= 0 or m_val <= 0:
            raise ValueError("num_qubits and num_marked must be positive.")
        expression = (sp.pi / 4) * sp.sqrt(sp.Rational(2**n_val, m_val))
        # Evaluate with precision proportional to the result magnitude. This
        # avoids both float overflow at n>=1024 and loss of integer accuracy
        # for much smaller but still large search spaces.
        decimal_digits = max(50, n_val // 6 + 30)
        return int(sp.floor(sp.N(expression, decimal_digits)))
    n = sp.sympify(n_val)
    m = sp.sympify(m_val)
    return sp.floor((sp.pi / 4) * sp.sqrt(sp.Integer(2) ** n / m))


def _diffusion(reg: Vector[Qubit]) -> Vector[Qubit]:
    """Apply the Grover diffusion operator ``2|s><s| - I`` in place.

    Implements the reflection about the uniform superposition as
    ``H^n X^n (multi-controlled-Z) X^n H^n`` using a real, backend-emittable
    body. The multi-controlled Z is realized as ``H . MCX . H`` on the last
    qubit.

    Args:
        reg (Vector[Qubit]): Search register of concrete width ``n``.

    Returns:
        Vector[Qubit]: Register after diffusion.
    """
    n = reg.shape[0]
    with for_loop(0, n, var_name="i") as i:
        reg[i] = qmc.h(reg[i])
    with for_loop(0, n, var_name="i") as i:
        reg[i] = qmc.x(reg[i])
    # Multi-controlled Z on the top qubit, controlled by the lower n-1 qubits.
    top = n - 1
    reg[top] = qmc.h(reg[top])
    reg[0:top], reg[top] = qmc.mcx(reg[0:top], reg[top])
    reg[top] = qmc.h(reg[top])
    with for_loop(0, n, var_name="i") as i:
        reg[i] = qmc.x(reg[i])
    with for_loop(0, n, var_name="i") as i:
        reg[i] = qmc.h(reg[i])
    return reg


def grover_search(
    reg: Vector[Qubit],
    oracle: QKernelLike,
    iterations: int | qmc.UInt,
) -> Vector[Qubit]:
    """Run the Grover amplitude-amplification loop on ``reg``.

    Prepares the uniform superposition, then applies ``iterations`` rounds of
    ``oracle`` followed by the diffusion operator. Leaving ``iterations``
    symbolic makes resource estimation report the universal ``O(sqrt(N/m))``
    query complexity: the oracle's opaque cost contributes the per-query gate
    cost and the diffusion contributes ``O(n)`` gates, both summed over the
    symbolic iteration count.

    Args:
        reg (Vector[Qubit]): Search register in the all-zero state on entry.
        oracle (QKernelLike): Phase oracle marking the solution(s). Supply a
            costed opaque box (e.g. ``qmc.opaque(..., cost=...)``)
            so the estimator can cost each query.
        iterations (int | qmc.UInt): Number of Grover iterations. Use
            :func:`grover_iteration_count` to obtain the optimal value; leave it
            as an unbound ``UInt`` parameter for symbolic estimation.

    Returns:
        Vector[Qubit]: Register after amplitude amplification.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.stdlib import grover_search, grover_iteration_count
        >>> mark = qmc.opaque("mark", num_qubits=3)
        >>> @qmc.qkernel
        ... def search(reg: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
        ...     return grover_search(reg, mark, grover_iteration_count(3))
    """
    n = reg.shape[0]
    with for_loop(0, n, var_name="i") as i:
        reg[i] = qmc.h(reg[i])
    if isinstance(iterations, int):
        # Concrete iteration count: unroll so the emitted circuit contains no
        # runtime control flow (statevector-friendly, backend-portable).
        for _ in range(iterations):
            reg = _grover_step(reg, oracle)
    else:
        # Symbolic iteration count: keep a runtime loop so resource estimation
        # sums the per-iteration cost over the symbolic bound.
        with for_loop(0, iterations, var_name="g"):
            reg = _grover_step(reg, oracle)
    return reg


def _grover_step(reg: Vector[Qubit], oracle: QKernelLike) -> Vector[Qubit]:
    """Apply one Grover iteration: an oracle query then the diffusion operator.

    Args:
        reg (Vector[Qubit]): Search register.
        oracle (QKernelLike): Phase oracle marking the solution(s).

    Returns:
        Vector[Qubit]: Register after one amplitude-amplification step.
    """
    result = cast(Callable[..., Any], oracle)(reg)
    reg = result if isinstance(result, Vector) else result[0]
    return _diffusion(reg)


__all__ = ["grover_search", "grover_iteration_count"]
