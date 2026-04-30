"""Pauli Correlation Encoding (PCE) converter.

This module provides :class:`PCEConverter`, which encodes the binary
variables of a combinatorial optimization problem into expectation values
of :math:`k`-body Pauli correlators acting on a small number of qubits.

PCE mirrors the QRAC family's API shape — ``num_qubits``,
``get_encoded_pauli_list``, ``encoder``, and a sign-rounding decoder — but
:class:`PCEConverter` does **not** inherit from
:class:`~qamomile.optimization.qrao.base_converter.QRACConverterBase` (or
:class:`~qamomile.optimization.converter.MathematicalProblemConverter`).
The reason is structural:

- PCE has no single cost Hamiltonian. The classical cost is a polynomial
  in the per-variable expectations :math:`\\langle P_i \\rangle`, so the
  abstract ``get_cost_hamiltonian`` contract from the QRAC base would be
  a misleading promise. Rather than carry a stub, we drop the inheritance
  entirely.
- PCE decodes from a list of Pauli expectations, not a measurement
  :class:`~qamomile.circuit.transpiler.job.SampleResult`. The inherited
  ``decode(SampleResult)`` does not apply.

The actual enumeration of :math:`k`-body Pauli correlators lives on the
standalone :class:`PCEEncoder`, mirroring how QRAC factors its encoders
out of the converter so users can construct and inspect the encoding
without going through the converter.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet, VarType
from qamomile.optimization.qrao.rounding import SignRounder

__all__ = ["PCEConverter", "PCEEncoder", "SignRounder"]


class PCEEncoder:
    """Pauli Correlation Encoding (PCE) encoder.

    Builds the deterministic mapping from each binary variable to a
    distinct :math:`k`-body Pauli correlator on :math:`n` qubits, where
    :math:`n` is the smallest integer satisfying
    :math:`\\binom{n}{k} \\cdot 3^k \\ge \\text{num\\_vars}`.

    The enumeration first iterates over all :math:`\\binom{n}{k}` qubit
    combinations in lexicographic order, then over the :math:`3^k`
    assignments of ``X``, ``Y``, ``Z`` on those qubits. The first
    ``spin_model.num_bits`` correlators are assigned to variables in index
    order; the remaining correlators are unused.

    Args:
        spin_model (BinaryModel): The problem in SPIN vartype. Must not
            contain higher-order (HUBO) terms.
        k (int): Compression rate — the body (weight) of each Pauli
            correlator. Must be a positive integer.

    Raises:
        ValueError: If ``k`` is not a positive integer, if ``spin_model``
            is not in SPIN vartype, or if ``spin_model`` contains HUBO
            terms.

    Example:
        >>> from qamomile.optimization.binary_model import BinaryModel, VarType
        >>> spin = BinaryModel(...).change_vartype(VarType.SPIN)
        >>> encoder = PCEEncoder(spin, k=2)
        >>> encoder.num_qubits
        4
        >>> encoder.pauli_encoding[0]  # Hamiltonian with one Pauli string
    """

    def __init__(self, spin_model: BinaryModel, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if spin_model.vartype != VarType.SPIN:
            raise ValueError("PCEEncoder requires a SPIN-type BinaryModel.")
        if spin_model.higher:
            raise ValueError(
                "PCEEncoder does not support higher-order (HUBO) terms. "
                "All interaction terms must be at most quadratic."
            )

        self.spin_model: BinaryModel = spin_model
        self.k: int = k
        self._num_qubits: int = PCEEncoder.min_num_qubits(spin_model.num_bits, k)
        self._pauli_encoding: dict[int, qm_o.Hamiltonian] = self._build_encoding()

    @staticmethod
    def min_num_qubits(num_vars: int, k: int) -> int:
        """Return the smallest ``n`` with ``C(n, k) * 3**k >= num_vars``.

        Args:
            num_vars (int): Number of variables to encode. Must be
                non-negative.
            k (int): Compression rate (number of non-identity Paulis per
                correlator). Must be a positive integer.

        Returns:
            int: The minimum number of qubits required to host
            ``num_vars`` distinct :math:`k`-body Pauli correlators.

        Raises:
            ValueError: If ``k`` is not a positive integer or ``num_vars``
                is negative.

        Example:
            >>> PCEEncoder.min_num_qubits(num_vars=10, k=2)
            3
        """
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if num_vars < 0:
            raise ValueError(f"num_vars must be non-negative, got {num_vars}.")
        if num_vars == 0:
            return k

        n = k
        while math.comb(n, k) * (3**k) < num_vars:
            n += 1
        return n

    def _build_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Enumerate :math:`k`-body Pauli correlators and assign one per variable.

        Returns:
            dict[int, qm_o.Hamiltonian]: Mapping from variable index to a
            single-term Hamiltonian on ``self.num_qubits`` qubits with
            coefficient ``1.0``.
        """
        num_vars = self.spin_model.num_bits
        n = self._num_qubits
        pauli_choices = (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z)

        encoding: dict[int, qm_o.Hamiltonian] = {}
        var_idx = 0
        for qubit_indices in itertools.combinations(range(n), self.k):
            if var_idx >= num_vars:
                break
            for pauli_assignment in itertools.product(pauli_choices, repeat=self.k):
                if var_idx >= num_vars:
                    break
                h = qm_o.Hamiltonian(num_qubits=n)
                pauli_ops = tuple(
                    qm_o.PauliOperator(p, q)
                    for p, q in zip(pauli_assignment, qubit_indices)
                )
                h.add_term(pauli_ops, 1.0)
                encoding[var_idx] = h
                var_idx += 1
        return encoding

    @property
    def num_qubits(self) -> int:
        """Number of qubits required by the encoding."""
        return self._num_qubits

    @property
    def pauli_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Mapping from variable index to its Pauli correlator Hamiltonian."""
        return self._pauli_encoding


class PCEConverter:
    """Converter for Pauli Correlation Encoding (PCE).

    PCE compresses :math:`N` optimization variables into the expectation
    values of :math:`k`-body Pauli correlators on :math:`n` qubits, where
    :math:`n` is chosen as the smallest integer satisfying
    :math:`\\binom{n}{k} \\cdot 3^k \\ge N`. Each variable :math:`i` is
    associated with a distinct correlator :math:`P_i`, and the decoded spin
    value is :math:`s_i = \\operatorname{sgn}\\langle P_i \\rangle`.

    PCE does not prescribe a specific ansatz — users provide their own
    variational circuit via :meth:`transpile`. The classical cost that the
    outer optimizer should minimize is

    .. math::
        C(\\langle P \\rangle) = \\sum_i h_i \\langle P_i \\rangle
            + \\sum_{i<j} J_{ij} \\langle P_i \\rangle \\langle P_j \\rangle
            + \\ldots

    evaluated from the per-variable expectation values produced by the
    user's ansatz. Use :meth:`get_encoded_pauli_list` to obtain the
    observables to feed into the backend's estimator.

    The encoding is built once at construction (parametrized by ``k``) and
    cached on the encoder. To re-encode with a different ``k``, construct a
    new :class:`PCEConverter`.

    Although the public API mirrors
    :class:`~qamomile.optimization.qrao.base_converter.QRACConverterBase`
    (``num_qubits``, ``encoder``, ``get_encoded_pauli_list``), this class
    deliberately does **not** inherit from it. PCE has no single cost
    Hamiltonian, so the QRAC base's abstract ``get_cost_hamiltonian``
    contract would be a misleading promise. PCE also decodes from
    expectation values rather than a sample set, which is incompatible
    with ``MathematicalProblemConverter.decode``.

    Attributes:
        instance (ommx.v1.Instance | None): The original OMMX instance if
            one was supplied, otherwise ``None``.
        original_vartype (VarType): The vartype of the input problem. The
            decoded sample set is returned in this vartype.
        spin_model (BinaryModel): The problem rewritten in SPIN form,
            which is the natural domain for sign rounding.

    Example:
        >>> converter = PCEConverter(instance, k=2)
        >>> observables = converter.get_encoded_pauli_list()
        >>> executable = converter.transpile(my_ansatz, transpiler)
        >>> # ... evaluate <P_i> for each i via executable ...
        >>> sampleset = converter.decode([0.7, -0.2, 0.9, -0.4])
    """

    def __init__(
        self,
        instance: ommx.v1.Instance | BinaryModel,
        k: int,
    ) -> None:
        """Initialize the converter from an OMMX instance or BinaryModel.

        The problem is internally converted to SPIN vartype, while the
        original vartype is remembered so that :meth:`decode` can return
        results in the user's preferred representation. The PCE encoding
        is built immediately so :attr:`num_qubits` and
        :meth:`get_encoded_pauli_list` are usable right after
        construction.

        Args:
            instance (ommx.v1.Instance | BinaryModel): The combinatorial
                optimization problem. ``ommx.v1.Instance`` inputs are
                converted via ``to_qubo()``; ``BinaryModel`` inputs retain
                their declared vartype as the target output vartype.
            k (int): Compression rate — the body (weight) of each Pauli
                correlator. Must be a positive integer.

        Raises:
            TypeError: If ``instance`` is neither an ``ommx.v1.Instance``
                nor a :class:`BinaryModel`.
            ValueError: If ``k`` is not a positive integer or if the
                problem contains higher-order (HUBO) terms (raised by
                :class:`PCEEncoder`).
        """
        if isinstance(instance, BinaryModel):
            self.instance: ommx.v1.Instance | None = None
            self.original_vartype: VarType = instance.vartype
            self.spin_model: BinaryModel = instance.change_vartype(VarType.SPIN)
        elif isinstance(instance, ommx.v1.Instance):
            self.instance = instance
            self.original_vartype = VarType.BINARY
            instance_copy = ommx.v1.Instance.from_bytes(instance.to_bytes())
            qubo, constant = instance_copy.to_qubo()
            self.spin_model = BinaryModel.from_qubo(qubo, constant).change_vartype(
                VarType.SPIN
            )
        else:
            raise TypeError("instance must be ommx.v1.Instance or BinaryModel")

        self._encoder: PCEEncoder = PCEEncoder(self.spin_model, k=k)

    @property
    def encoder(self) -> PCEEncoder:
        """The PCE encoder used by this converter."""
        return self._encoder

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits required by the PCE encoding."""
        return self._encoder.num_qubits

    @property
    def k(self) -> int:
        """Compression rate of the PCE encoding."""
        return self._encoder.k

    @property
    def pauli_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Mapping from variable index to its Pauli correlator Hamiltonian."""
        return self._encoder.pauli_encoding

    @staticmethod
    def min_num_qubits(num_vars: int, k: int) -> int:
        """Return the smallest ``n`` with ``C(n, k) * 3**k >= num_vars``.

        Thin forwarder to :meth:`PCEEncoder.min_num_qubits` so callers can
        size a problem before constructing the converter.

        Args:
            num_vars (int): Number of variables to encode. Must be
                non-negative.
            k (int): Compression rate. Must be a positive integer.

        Returns:
            int: The minimum number of qubits required.

        Raises:
            ValueError: If ``k`` is not a positive integer or ``num_vars``
                is negative.

        Example:
            >>> PCEConverter.min_num_qubits(num_vars=10, k=2)
            3
        """
        return PCEEncoder.min_num_qubits(num_vars, k)

    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]:
        """Return the per-variable Pauli correlator observables.

        Returns the encoding as a list indexed by variable, suitable for
        passing to a backend estimator to obtain :math:`\\langle P_i \\rangle`
        for each variable.

        Returns:
            list[qm_o.Hamiltonian]: One Hamiltonian per variable, in
            variable-index order. Each Hamiltonian contains a single Pauli
            string with coefficient ``1.0`` on ``self.num_qubits`` qubits.
        """
        encoding = self._encoder.pauli_encoding
        return [encoding[i] for i in range(self.spin_model.num_bits)]

    def transpile(
        self,
        circuit: QKernel,
        transpiler: Transpiler,
        *,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> ExecutableProgram:
        """Transpile a user-provided ansatz ``@qkernel`` to an executable.

        PCE does not prescribe a fixed ansatz; this method simply forwards
        the user's circuit through the given backend transpiler. The
        optional ``bindings`` and ``parameters`` arguments mirror
        :meth:`Transpiler.transpile` so that compile-time bindings can be
        supplied while reserving selected kernel arguments (typically the
        variational angles) as runtime parameters.

        Args:
            circuit (QKernel): The user-supplied ``@qkernel`` ansatz to
                compile. Any non-trivial kernel accepted by the backend
                transpiler is valid.
            transpiler (Transpiler): The backend transpiler to use (e.g.
                ``QiskitTranspiler`` or ``QuriPartsTranspiler``).
            bindings (dict[str, Any] | None): Compile-time parameter
                bindings passed through to the transpiler. Defaults to
                ``None``, meaning no bindings.
            parameters (list[str] | None): Names of kernel parameters to
                preserve as runtime backend parameters. Defaults to
                ``None``, meaning every unbound parameter is treated as a
                compile-time input by the transpiler.

        Returns:
            ExecutableProgram: The compiled program ready for execution
            against the backend's estimator (to obtain the per-variable
            Pauli expectations needed by :meth:`decode`).

        Raises:
            Any exception raised by the underlying ``transpiler.transpile``
            call — the PCE converter adds no validation beyond delegation.
        """
        return transpiler.transpile(
            circuit,
            bindings=bindings,
            parameters=parameters,
        )

    def decode(self, expectations: list[float]) -> BinarySampleSet:
        """Decode Pauli expectation values into a binary sample set.

        Uses :class:`SignRounder` to convert each expectation
        :math:`\\langle P_i \\rangle` into a spin value
        :math:`s_i \\in \\{+1, -1\\}`, computes the energy of the resulting
        assignment from the stored SPIN model, and returns a single-sample
        :class:`BinarySampleSet` in the original vartype of the input
        problem.

        Args:
            expectations (list[float]): Expectation values of the encoding
                Pauli correlators, in variable-index order. Each value is
                expected to lie in :math:`[-1, 1]`, though the sign
                rounder does not enforce that range — only the sign is
                consulted.

        Returns:
            BinarySampleSet: A sample set containing one sample
            (``num_occurrences=[1]``) labelled with the computed energy,
            expressed in the original vartype (``BINARY`` or ``SPIN``) of
            the input problem.

        Raises:
            ValueError: If the length of ``expectations`` does not match
                the number of variables in the stored spin model.

        Example:
            >>> # A problem with 3 variables.
            >>> sampleset = converter.decode([0.4, -0.1, 0.8])
            >>> sampleset.samples
            [{0: 1, 1: -1, 2: 1}]  # if original_vartype is SPIN
        """
        num_vars = self.spin_model.num_bits
        if len(expectations) != num_vars:
            raise ValueError(
                f"Expected {num_vars} expectation values, got {len(expectations)}."
            )

        rounder = SignRounder()
        spins = rounder.round(expectations)

        idx_map = self.spin_model.index_new_to_origin
        spin_sample: dict[int, int] = {
            idx_map[new_idx]: spins[new_idx] for new_idx in range(len(spins))
        }
        energy = self.spin_model.calc_energy(spins)

        if self.original_vartype == VarType.BINARY:
            binary_sample = {i: (1 - s) // 2 for i, s in spin_sample.items()}
            return BinarySampleSet(
                samples=[binary_sample],
                num_occurrences=[1],
                energy=[energy],
                vartype=VarType.BINARY,
            )

        return BinarySampleSet(
            samples=[spin_sample],
            num_occurrences=[1],
            energy=[energy],
            vartype=VarType.SPIN,
        )
