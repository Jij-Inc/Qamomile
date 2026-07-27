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

import ommx.v1

import qamomile.observable as qm_o
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet, VarType
from qamomile.optimization.converter import (
    binary_sampleset_to_ommx_samples,
    evaluate_original_instance,
    normalize_problem_input,
)
from qamomile.optimization.qrao.rounding import SignRounder

__all__ = ["PCEConverter", "PCEEncoder"]


class PCEEncoder:
    """Pauli Correlation Encoding (PCE) encoder.

    Builds the deterministic mapping from each binary variable to a
    distinct :math:`k`-body Pauli correlator on :math:`n` qubits. By
    default, :math:`n` is the smallest integer satisfying
    :math:`\\binom{n}{k} \\cdot 3^k \\ge \\text{num\\_vars}`; callers may
    instead pass an explicit ``num_qubits`` to use a larger register
    (e.g., to match a hardware topology), provided it is at least the
    computed minimum.

    The enumeration first iterates over all :math:`\\binom{n}{k}` qubit
    combinations in lexicographic order, then over the :math:`3^k`
    assignments of ``X``, ``Y``, ``Z`` on those qubits. The first
    ``spin_model.num_bits`` correlators are assigned to variables in index
    order; the remaining correlators are unused. Because the lexicographic
    enumeration depends on ``num_qubits``, the per-variable correlator
    assignment may differ when ``num_qubits`` is set above the minimum.

    Args:
        spin_model (BinaryModel): The problem in SPIN vartype. Must not
            contain higher-order (HUBO) terms.
        correlator_order (int): Compression rate — the body (weight) of
            each Pauli correlator (denoted :math:`k` in the math). Must
            be a positive integer.
        num_qubits (int | None): Number of qubits to host the encoding.
            Defaults to ``None``, meaning the encoder uses
            :meth:`min_num_qubits` — the smallest :math:`n` satisfying
            the capacity inequality. When given, must be at least that
            minimum.

    Raises:
        ValueError: If ``correlator_order`` is not a positive integer, if
            ``spin_model`` is not in SPIN vartype, if ``spin_model``
            contains HUBO terms, or if ``num_qubits`` is below the
            minimum required for the problem at the given
            ``correlator_order``.

    Example:
        >>> from qamomile.optimization.binary_model import BinaryModel, VarType
        >>> spin = BinaryModel(...).change_vartype(VarType.SPIN)
        >>> encoder = PCEEncoder(spin, correlator_order=2)
        >>> encoder.num_qubits
        4
        >>> # Override to use a larger register:
        >>> wide_encoder = PCEEncoder(spin, correlator_order=2, num_qubits=6)
        >>> wide_encoder.num_qubits
        6
        >>> encoder.pauli_encoding[0]  # Hamiltonian with one Pauli string
    """

    def __init__(
        self,
        spin_model: BinaryModel,
        correlator_order: int,
        num_qubits: int | None = None,
    ) -> None:
        if correlator_order < 1:
            raise ValueError(
                f"correlator_order must be a positive integer, got {correlator_order}."
            )
        if spin_model.vartype != VarType.SPIN:
            raise ValueError("PCEEncoder requires a SPIN-type BinaryModel.")
        if spin_model.higher:
            raise ValueError(
                "PCEEncoder rejects higher-order (HUBO) terms. The Pauli-"
                "correlator enumeration itself does not require quadratic "
                "structure, but PCE as published targets quadratic Ising — "
                "its tanh-relaxed cost surrogate and Edwards–Erdős "
                "regularizer assume pairwise interactions. Extending PCE "
                "to HUBO would require re-deriving the surrogate; we "
                "reject HUBO inputs here to make that boundary explicit."
            )

        min_n = PCEEncoder.min_num_qubits(spin_model.num_bits, correlator_order)
        if num_qubits is None:
            n = min_n
        else:
            if num_qubits < min_n:
                raise ValueError(
                    f"num_qubits={num_qubits} is too small to host "
                    f"{spin_model.num_bits} variables at "
                    f"correlator_order={correlator_order}; "
                    f"need at least {min_n}."
                )
            n = num_qubits

        self.spin_model: BinaryModel = spin_model
        self.correlator_order: int = correlator_order
        self._num_qubits: int = n
        self._pauli_encoding: dict[int, qm_o.Hamiltonian] = self._build_encoding()

    @staticmethod
    def min_num_qubits(num_vars: int, correlator_order: int) -> int:
        """Return the smallest ``n >= k`` with ``C(n, k) * 3**k >= num_vars``.

        Here ``k`` denotes the ``correlator_order`` argument. The result
        is always at least ``k`` because a :math:`k`-body correlator
        requires at least ``k`` qubits (``C(n, k) = 0`` for ``n < k``).
        When ``num_vars == 0`` the method returns ``k`` directly.

        Args:
            num_vars (int): Number of variables to encode. Must be
                non-negative.
            correlator_order (int): Compression rate (number of
                non-identity Paulis per correlator; :math:`k` in the
                math). Must be a positive integer.

        Returns:
            int: The minimum number of qubits ``n`` (with
            ``n >= correlator_order``) required to host ``num_vars``
            distinct :math:`k`-body Pauli correlators.

        Raises:
            ValueError: If ``correlator_order`` is not a positive integer
                or ``num_vars`` is negative.

        Example:
            >>> PCEEncoder.min_num_qubits(num_vars=10, correlator_order=2)
            3
            >>> # Lower bound: result is always >= correlator_order
            >>> PCEEncoder.min_num_qubits(num_vars=0, correlator_order=3)
            3
        """
        if correlator_order < 1:
            raise ValueError(
                f"correlator_order must be a positive integer, got {correlator_order}."
            )
        if num_vars < 0:
            raise ValueError(f"num_vars must be non-negative, got {num_vars}.")
        if num_vars == 0:
            return correlator_order

        n = correlator_order
        while math.comb(n, correlator_order) * (3**correlator_order) < num_vars:
            n += 1
        return n

    def _build_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Enumerate :math:`k`-body Pauli correlators and assign one per variable.

        Returns:
            dict[int, qm_o.Hamiltonian]: Mapping from a :class:`BinaryModel`
            **new** variable index (the contiguous 0-based index into
            ``spin_model``, *not* the caller's original variable IDs;
            see :attr:`BinaryModel.index_new_to_origin` /
            :attr:`BinaryModel.index_origin_to_new` for the round-trip)
            to a single-term Hamiltonian on ``self.num_qubits`` qubits
            with coefficient ``1.0``. Mapping decoded keys back to the
            user's original variable IDs is the job of :meth:`decode`,
            which uses :attr:`BinaryModel.index_new_to_origin`.
        """
        num_vars = self.spin_model.num_bits
        n = self._num_qubits
        pauli_choices = (qm_o.Pauli.X, qm_o.Pauli.Y, qm_o.Pauli.Z)

        encoding: dict[int, qm_o.Hamiltonian] = {}
        var_idx = 0
        for qubit_indices in itertools.combinations(range(n), self.correlator_order):
            if var_idx >= num_vars:
                break
            for pauli_assignment in itertools.product(
                pauli_choices, repeat=self.correlator_order
            ):
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
        """Number of qubits used by the PCE encoding.

        Equal to the ``num_qubits`` argument passed to :meth:`__init__`
        when given, or to :meth:`min_num_qubits` (the smallest ``n >= k``
        satisfying :math:`\\binom{n}{k} \\cdot 3^k \\ge \\text{num\\_vars}`)
        when ``num_qubits`` was left at its default. Every Pauli
        correlator in :attr:`pauli_encoding` acts on exactly this many
        qubits.

        Returns:
            int: The qubit count used by every correlator Hamiltonian in
            :attr:`pauli_encoding`.
        """
        return self._num_qubits

    @property
    def pauli_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Mapping from variable index to its Pauli correlator Hamiltonian.

        Returns:
            dict[int, qm_o.Hamiltonian]: Dictionary keyed by variable index
            ``i`` in ``range(spin_model.num_bits)``. Each value is a
            single-term Hamiltonian on :attr:`num_qubits` qubits whose
            Pauli string is the :math:`k`-body correlator
            :math:`P_i` assigned to that variable, with coefficient
            ``1.0``.
        """
        return self._pauli_encoding


class PCEConverter:
    """Converter for Pauli Correlation Encoding (PCE).

    PCE compresses :math:`N` optimization variables into the expectation
    values of :math:`k`-body Pauli correlators on :math:`n` qubits. By
    default, :math:`n` is chosen as the smallest integer satisfying
    :math:`\\binom{n}{k} \\cdot 3^k \\ge N`; callers may instead pass an
    explicit ``num_qubits`` (at least the minimum) to host the encoding
    on a larger register. Each variable :math:`i` is associated with a
    distinct correlator :math:`P_i`, and the decoded spin value is
    :math:`s_i = \\operatorname{sgn}\\langle P_i \\rangle`.

    PCE does not prescribe a specific ansatz — users build their own
    variational circuit and transpile it directly with their backend's
    :class:`~qamomile.circuit.transpiler.transpiler.Transpiler` (this
    converter does not wrap that step). The classical cost that the
    outer optimizer should minimize is

    .. math::
        C(\\langle P \\rangle) = \\sum_i h_i \\langle P_i \\rangle
            + \\sum_{i<j} J_{ij} \\langle P_i \\rangle \\langle P_j \\rangle
            + \\ldots

    evaluated from the per-variable expectation values produced by the
    user's ansatz. Use :meth:`get_encoded_pauli_list` to obtain the
    observables to feed into the backend's estimator.

    The encoding is built once at construction (parametrized by
    ``correlator_order``) and cached on the encoder. To re-encode with a
    different ``correlator_order``, construct a new :class:`PCEConverter`.

    Although the public API mirrors
    :class:`~qamomile.optimization.qrao.base_converter.QRACConverterBase`
    (``num_qubits``, ``encoder``, ``get_encoded_pauli_list``), this class
    deliberately does **not** inherit from it. PCE has no single cost
    Hamiltonian, so the QRAC base's abstract ``get_cost_hamiltonian``
    contract would be a misleading promise. PCE also decodes from
    expectation values rather than a sample set, which is incompatible
    with ``MathematicalProblemConverter.decode``.

    The polymorphic return type of :meth:`decode`, however, is aligned
    with the rest of the converter family: an OMMX-backed converter
    returns an :class:`ommx.v1.SampleSet` (so feasibility, original
    objective, and per-constraint diagnostics are available through
    OMMX's own API), while a :class:`BinaryModel`-backed converter
    returns a :class:`BinarySampleSet` in the model's original vartype.

    Attributes:
        instance (ommx.v1.Instance | None): The original OMMX instance if
            one was supplied, otherwise ``None``.
        original_vartype (VarType): The vartype of the input problem. The
            decoded sample set is returned in this vartype.
        spin_model (BinaryModel): The problem rewritten in SPIN form,
            which is the natural domain for sign rounding.

    Example:
        >>> converter = PCEConverter(instance, correlator_order=2)
        >>> observables = converter.get_encoded_pauli_list()
        >>> # Transpile the user's ansatz directly with the backend
        >>> # transpiler (PCEConverter does not wrap this step):
        >>> executable = transpiler.transpile(
        ...     my_ansatz,
        ...     bindings={"P": observables[0]},
        ...     parameters=["thetas"],
        ... )
        >>> # ... evaluate <P_i> for each i via executable ...
        >>> sampleset = converter.decode([0.7, -0.2, 0.9, -0.4])
    """

    def __init__(
        self,
        instance: ommx.v1.Instance | BinaryModel,
        correlator_order: int,
        num_qubits: int | None = None,
        *,
        uniform_penalty_weight: float | None = None,
        penalty_weights: dict[int, float] | None = None,
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
            correlator_order (int): Compression rate — the body (weight)
                of each Pauli correlator (denoted :math:`k` in the math).
                Must be a positive integer.
            num_qubits (int | None): Number of qubits to host the
                encoding. Defaults to ``None``, meaning the converter
                uses :meth:`min_num_qubits` — the smallest :math:`n`
                large enough to host every variable's correlator. When
                given, must be at least that minimum; supplying a larger
                value is supported for callers that want to match a
                specific hardware register width.
            uniform_penalty_weight (float | None): Uniform OMMX constraint
                penalty. ``None`` delegates weight selection to OMMX.
            penalty_weights (dict[int, float] | None): Optional per-constraint
                penalty weights keyed by constraint ID.

        Raises:
            TypeError: If ``instance`` is neither an ``ommx.v1.Instance``
                nor a :class:`BinaryModel`.
            ValueError: If ``correlator_order`` is not a positive integer,
                if ``num_qubits`` is below the minimum required for the
                problem at the given ``correlator_order``, or if the
                problem contains higher-order (HUBO) terms (raised by
                :class:`PCEEncoder`).
        """
        self.original_instance: ommx.v1.Instance | None
        self.instance: ommx.v1.Instance | None
        self.original_vartype: VarType
        self.spin_model: BinaryModel
        (
            self.original_instance,
            self.instance,
            self.original_vartype,
            self.spin_model,
        ) = normalize_problem_input(
            instance,
            uniform_penalty_weight=uniform_penalty_weight,
            penalty_weights=penalty_weights,
        )
        self._encoder: PCEEncoder = PCEEncoder(
            self.spin_model,
            correlator_order=correlator_order,
            num_qubits=num_qubits,
        )

    @property
    def encoder(self) -> PCEEncoder:
        """The :class:`PCEEncoder` instance built at construction time.

        Exposed so callers can inspect the raw enumeration
        (:attr:`PCEEncoder.pauli_encoding`) or compute encoding-related
        quantities without going through the converter.

        Returns:
            PCEEncoder: The encoder built from the SPIN-form problem and
            the user-supplied ``correlator_order``.
        """
        return self._encoder

    @property
    def num_qubits(self) -> int:
        """Number of physical qubits used by the PCE encoding.

        Equal to :attr:`PCEEncoder.num_qubits`: either the
        ``num_qubits`` argument supplied to :meth:`__init__`, or — when
        that was left at its default — the smallest ``n >= k`` (with
        :math:`k =` :attr:`correlator_order`) satisfying
        :math:`\\binom{n}{k} \\cdot 3^k \\ge N`, where ``N`` is the
        variable count of the SPIN-form problem.

        Returns:
            int: The number of qubits acted on by every observable
            returned by :meth:`get_encoded_pauli_list`.
        """
        return self._encoder.num_qubits

    @property
    def correlator_order(self) -> int:
        """Compression rate (body weight) of the PCE encoding.

        Returns:
            int: The number of non-identity Pauli factors in each
            correlator :math:`P_i` (the :math:`k` in :math:`k`-body),
            as supplied to :meth:`__init__`.
        """
        return self._encoder.correlator_order

    @property
    def pauli_encoding(self) -> dict[int, qm_o.Hamiltonian]:
        """Mapping from variable index to its Pauli correlator Hamiltonian.

        Thin pass-through to :attr:`PCEEncoder.pauli_encoding` for callers
        that hold only the converter.

        Returns:
            dict[int, qm_o.Hamiltonian]: Dictionary keyed by variable
            index ``i`` in ``range(spin_model.num_bits)``. Each value is a
            single-term Hamiltonian on :attr:`num_qubits` qubits whose
            Pauli string is the :math:`k`-body correlator
            :math:`P_i` assigned to that variable, with coefficient
            ``1.0``.
        """
        return self._encoder.pauli_encoding

    @staticmethod
    def min_num_qubits(num_vars: int, correlator_order: int) -> int:
        """Return the smallest ``n`` with ``C(n, k) * 3**k >= num_vars``.

        Here ``k`` denotes ``correlator_order``. Thin forwarder to
        :meth:`PCEEncoder.min_num_qubits` so callers can size a problem
        before constructing the converter.

        Args:
            num_vars (int): Number of variables to encode. Must be
                non-negative.
            correlator_order (int): Compression rate (:math:`k` in the
                math). Must be a positive integer.

        Returns:
            int: The minimum number of qubits required.

        Raises:
            ValueError: If ``correlator_order`` is not a positive integer
                or ``num_vars`` is negative.

        Example:
            >>> PCEConverter.min_num_qubits(num_vars=10, correlator_order=2)
            3
        """
        return PCEEncoder.min_num_qubits(num_vars, correlator_order)

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

    def decode(
        self,
        expectations: list[float],
    ) -> BinarySampleSet | ommx.v1.SampleSet:
        """Decode Pauli expectation values into a sample set.

        Uses :class:`SignRounder` to convert each expectation
        :math:`\\langle P_i \\rangle` into a spin value
        :math:`s_i \\in \\{+1, -1\\}` and computes the energy of the
        resulting assignment from the stored SPIN model. The return type
        tracks the input that built this converter — matching the
        polymorphic behaviour of
        :meth:`MathematicalProblemConverter.decode`:

        * Built from an :class:`ommx.v1.Instance` — returns an
          :class:`ommx.v1.SampleSet` evaluated against the stored OMMX
          instance, so feasibility, original objective, and per-constraint
          violations are available through OMMX's own API
          (``.summary``, ``.best_feasible``, ``.feasible``,
          ``.objectives``, ...).
        * Built from a :class:`BinaryModel` — returns a single-sample
          :class:`BinarySampleSet` (``num_occurrences=[1]``) labelled with
          the computed energy, expressed in the original vartype
          (``BINARY`` or ``SPIN``) of the input problem.

        Args:
            expectations (list[float]): Expectation values of the encoding
                Pauli correlators, in variable-index order. Each value is
                expected to lie in :math:`[-1, 1]`, though the sign
                rounder does not enforce that range — only the sign is
                consulted.

        Returns:
            BinarySampleSet | ommx.v1.SampleSet: see method description.

        Raises:
            ValueError: If the length of ``expectations`` does not match
                the number of variables in the stored spin model.

        Example:
            >>> # A BinaryModel-backed converter with 3 SPIN variables.
            >>> sampleset = converter.decode([0.4, -0.1, 0.8])
            >>> sampleset.samples
            [{0: 1, 1: -1, 2: 1}]
            >>> # An OMMX-backed converter returns an ommx SampleSet.
            >>> sample_set = ommx_converter.decode([0.4, -0.1, 0.8])
            >>> sample_set.best_feasible.objective
        """
        num_vars = self.spin_model.num_bits
        if len(expectations) != num_vars:
            raise ValueError(
                f"Expected {num_vars} expectation values, got {len(expectations)}."
            )

        spins = SignRounder().round(expectations)

        idx_map = self.spin_model.index_new_to_origin
        spin_sample: dict[int, int] = {
            idx_map[new_idx]: spins[new_idx] for new_idx in range(len(spins))
        }
        energy = self.spin_model.calc_energy(spins)
        binary_sample = {i: (1 - s) // 2 for i, s in spin_sample.items()}

        # OMMX path: build a BINARY single-sample BinarySampleSet, route it
        # through the shared helper, and let evaluate_samples report the
        # original (un-penalized) objective and feasibility.
        if self.instance is not None:
            assert self.original_instance is not None
            binary_sampleset = BinarySampleSet(
                samples=[binary_sample],
                num_occurrences=[1],
                energy=[energy],
                vartype=VarType.BINARY,
            )
            ommx_samples = binary_sampleset_to_ommx_samples(binary_sampleset)
            return evaluate_original_instance(
                self.original_instance,
                self.instance,
                ommx_samples,
            )

        # BinaryModel path: return in the model's original vartype.
        if self.original_vartype == VarType.BINARY:
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
