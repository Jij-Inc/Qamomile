"""Shared numerical tolerances for interpreting Hamiltonian coefficients.

These tolerances decide how a Hamiltonian's coefficients are treated when an
``exp(-i * gamma * H)`` (``PauliEvolveOp``) is lowered to gates. They live in
their own module — rather than inside any single emit pass — so every path
that emits Pauli evolution shares one source of truth: the uncontrolled
``pauli_evolve_emission.emit_pauli_evolve`` and the controlled
``controlled_emission.emit_controlled_pauli_evolve`` both import them, and any
future emit path can do the same without depending on an unrelated module.
"""

# A Hamiltonian coefficient (a Pauli term's coefficient or the Hamiltonian's
# constant offset) whose magnitude is at or below this is treated as zero and
# skipped: no gate is emitted for a negligible term. The floating-point slack
# keeps coefficients that cancel to ~0 during Hamiltonian arithmetic from
# emitting spurious gates.
PAULI_TERM_ZERO_ATOL = 1e-15

# A coefficient whose imaginary part exceeds this fails the Hermiticity
# requirement: ``exp(-i * gamma * H)`` is unitary only for a Hermitian ``H``
# (real coefficients). The slack absorbs floating-point imaginary residue left
# by complex arithmetic.
HERMITIAN_IMAG_ATOL = 1e-10
