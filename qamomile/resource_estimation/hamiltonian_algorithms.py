"""Estimate logical resources for Hamiltonian phase-estimation workloads."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy as sp

from qamomile.circuit.estimator import GateCount, ResourceEstimate
from qamomile.resource_estimation._common import (
    _as_expr,
    _SympyLike,
    _validate_nonnegative,
    _validate_positive,
)
from qamomile.resource_estimation.hamiltonian import (
    PauliHamiltonianResource,
    summarize_openfermion_qubit_operator,
)
from qamomile.resource_estimation.workload import HamiltonianWorkloadMixin


class HamiltonianRepresentation(enum.StrEnum):
    """Select a Hamiltonian representation for logical QPE estimates.

    Attributes:
        SPARSE_PAULI_LCU: Sparse Pauli-LCU representation.
        SINGLE_FACTORIZATION: Single-factorized tensor representation.
        DOUBLE_FACTORIZATION: Double-factorized tensor representation.
        TENSOR_HYPERCONTRACTION: Tensor-hypercontracted representation.
        SYMMETRY_COMPRESSED_DF: Symmetry-compressed double factorization.
        UNITARY_WEIGHT_CONCENTRATION: Early-FTQC unitary-weight concentration.

    Example:
        >>> HamiltonianRepresentation("sparse_pauli_lcu")
        <HamiltonianRepresentation.SPARSE_PAULI_LCU: 'sparse_pauli_lcu'>
    """

    SPARSE_PAULI_LCU = "sparse_pauli_lcu"
    SINGLE_FACTORIZATION = "single_factorization"
    DOUBLE_FACTORIZATION = "double_factorization"
    TENSOR_HYPERCONTRACTION = "tensor_hypercontraction"
    SYMMETRY_COMPRESSED_DF = "symmetry_compressed_df"
    UNITARY_WEIGHT_CONCENTRATION = "unitary_weight_concentration"


@dataclass(frozen=True)
class HamiltonianQPEWorkload(HamiltonianWorkloadMixin):
    """Describe a Hamiltonian workload for phase-estimation resource estimates.

    Attributes:
        hamiltonian (PauliHamiltonianResource): Generic Pauli Hamiltonian
            resource summary.
        walk_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            qubitized walk operator call.
        representation (str | HamiltonianRepresentation): Hamiltonian
            representation used to choose default logical-qubit scaling.
        sparsity (sp.Expr | int | float | None): Sparse-method nonzero term
            count. Defaults to ``hamiltonian.n_pauli_terms`` for sparse LCU.
        second_factor_rank (sp.Expr | int | float | None): Average second
            factorization rank for double-factorized methods.
        logical_qubits (sp.Expr | int | float | None): Explicit logical-qubit
            count before ``qpe_register_qubits`` are added. Defaults to the
            representation-specific scaling model.
        representation_error (sp.Expr | int | float): Hamiltonian
            representation error budget.
        description (str): Reader-facing model label.
        qpe_register_qubits (sp.Expr | int | float): Optional QPE readout
            register qubits.

    Raises:
        TypeError: If ``hamiltonian`` is not a ``PauliHamiltonianResource``.
        ValueError: If any positive-valued quantity is non-positive or if
            ``representation_error`` is negative.

    Example:
        >>> summary = PauliHamiltonianResource(
        ...     n_qubits=4,
        ...     n_pauli_terms=10,
        ...     lambda_norm=20,
        ...     max_locality=2,
        ... )
        >>> workload = HamiltonianQPEWorkload(
        ...     summary,
        ...     walk_cost_toffoli=100,
        ...     representation=HamiltonianRepresentation.SPARSE_PAULI_LCU,
        ... )
        >>> workload.effective_sparsity
        10
    """

    hamiltonian: PauliHamiltonianResource
    walk_cost_toffoli: _SympyLike
    representation: str | HamiltonianRepresentation = (
        HamiltonianRepresentation.DOUBLE_FACTORIZATION
    )
    sparsity: _SympyLike | None = None
    second_factor_rank: _SympyLike | None = None
    logical_qubits: _SympyLike | None = None
    representation_error: _SympyLike = 0
    description: str = ""
    qpe_register_qubits: _SympyLike = 0

    _POSITIVE_FIELDS = ("walk_cost_toffoli",)
    _OPTIONAL_POSITIVE_FIELDS = ("sparsity", "second_factor_rank", "logical_qubits")
    _NONNEGATIVE_FIELDS = ("representation_error", "qpe_register_qubits")

    def __post_init__(self) -> None:
        """Validate workload fields after dataclass construction.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource``.
            ValueError: If any positive-valued quantity is non-positive, if
                ``representation_error`` is negative, or if
                ``representation`` is unknown.
        """
        self._validate_workload_fields()
        _normalize_representation(self.representation)

    @property
    def normalized_representation(self) -> str:
        """Return the normalized Hamiltonian representation key.

        Returns:
            str: Registered representation key. Built-in keys compare equal
                to their ``HamiltonianRepresentation`` members.
        """
        return _normalize_representation(self.representation)

    @property
    def effective_sparsity(self) -> sp.Expr | None:
        """Return sparse-method term count with a Hamiltonian fallback.

        Returns:
            sp.Expr | None: Explicit sparsity, Hamiltonian term count for the
                sparse representation, or None for non-sparse representations.
        """
        if self.sparsity is not None:
            return _as_expr(self.sparsity, "sparsity")
        if self.normalized_representation == HamiltonianRepresentation.SPARSE_PAULI_LCU:
            return self.hamiltonian.n_pauli_terms
        return None

    def _own_resource_values(self) -> dict[str, sp.Expr]:
        """Return the qubitized-QPE workload's algorithm-level values.

        The Hamiltonian's ``n_pauli_terms`` is always the true term count of
        the problem; the representation-level term count assumed by sparse
        methods is reported separately under the ``sparsity`` key.

        Returns:
            dict[str, sp.Expr]: Walk cost, error budget, QPE register
                qubits, and ``sparsity`` when the workload carries a
                sparse-method term count.
        """
        values = {
            "walk_cost_toffoli": _as_expr(
                self.walk_cost_toffoli,
                "walk_cost_toffoli",
            ),
            "representation_error": _as_expr(
                self.representation_error,
                "representation_error",
            ),
            "qpe_register_qubits": _as_expr(
                self.qpe_register_qubits,
                "qpe_register_qubits",
            ),
        }
        if self.effective_sparsity is not None:
            values["sparsity"] = self.effective_sparsity
        return values

    def _dict_overrides(self) -> dict[str, Any]:
        """Return serialization overrides for :meth:`to_dict`.

        Returns:
            dict[str, Any]: The ``sparsity`` entry serialized from
                ``effective_sparsity`` so the sparse-LCU fallback is visible
                in serialized form.
        """
        return {
            "sparsity": (
                None
                if self.effective_sparsity is None
                else str(self.effective_sparsity)
            ),
        }

    @classmethod
    def from_block_encoding(
        cls,
        hamiltonian: PauliHamiltonianResource,
        block_encoding: Any,
        *,
        representation: str | HamiltonianRepresentation = (
            HamiltonianRepresentation.DOUBLE_FACTORIZATION
        ),
        second_factor_rank: _SympyLike | None = None,
        qpe_register_qubits: _SympyLike = 0,
        representation_error: _SympyLike = 0,
        description: str | None = None,
    ) -> HamiltonianQPEWorkload:
        """Build a QPE workload from a block-encoding contract.

        The returned workload uses the block-encoding normalization as the
        Hamiltonian normalization, the walk cost from the PREPARE/SELECT
        contract, the block-encoding footprint as the base logical-qubit
        count, and optional QPE readout qubits as separate algorithm metadata.

        Args:
            hamiltonian (PauliHamiltonianResource): Problem-level Hamiltonian
                summary whose term and locality metadata should be preserved.
            block_encoding (Any): ``BlockEncodingResource`` describing the
                algorithm-specific block encoding.
            representation (str | HamiltonianRepresentation): Hamiltonian
                representation label for the resulting workload. Defaults to
                ``HamiltonianRepresentation.DOUBLE_FACTORIZATION``.
            second_factor_rank (sp.Expr | int | float | None): Optional
                second-factor rank metadata for factorized representations.
                Defaults to None.
            qpe_register_qubits (sp.Expr | int | float): Optional QPE readout
                register qubits added to the block-encoding footprint.
                Defaults to 0.
            representation_error (sp.Expr | int | float): Hamiltonian
                representation error consumed before phase estimation.
                Defaults to 0.
            description (str | None): Optional reader-facing label. Defaults
                to the block-encoding name.

        Returns:
            HamiltonianQPEWorkload: Workload backed by the block-encoding
            resource contract.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource`` or ``block_encoding`` is not a
                ``BlockEncodingResource``.
            ValueError: If ``qpe_register_qubits`` is negative or any
                workload quantity is invalid.
        """
        from qamomile.resource_estimation.block_encoding import BlockEncodingResource

        if not isinstance(hamiltonian, PauliHamiltonianResource):
            raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
        if not isinstance(block_encoding, BlockEncodingResource):
            raise TypeError("block_encoding must be a BlockEncodingResource.")

        qpe_qubits = _as_expr(qpe_register_qubits, "qpe_register_qubits")
        _validate_nonnegative(qpe_qubits, "qpe_register_qubits")
        block_values = block_encoding.resource_values()
        encoded_hamiltonian = PauliHamiltonianResource(
            n_qubits=block_values["system_qubits"],
            n_pauli_terms=hamiltonian.n_pauli_terms,
            lambda_norm=block_values["lambda_norm"],
            max_locality=hamiltonian.max_locality,
            constant=hamiltonian.constant,
            constant_included=hamiltonian.constant_included,
            source=block_encoding.name,
        )
        return cls(
            hamiltonian=encoded_hamiltonian,
            walk_cost_toffoli=block_encoding.walk_cost_toffoli,
            representation=representation,
            second_factor_rank=second_factor_rank,
            logical_qubits=block_values["logical_qubits"],
            representation_error=representation_error,
            qpe_register_qubits=qpe_qubits,
            description=block_encoding.name if description is None else description,
        )


@dataclass(frozen=True)
class TrotterQPEWorkload(HamiltonianWorkloadMixin):
    """Describe a product-formula QPE workload for resource reviews.

    The workload keeps product-formula and unitary-weight assumptions symbolic
    so recent low-qubit chemistry proposals can be compared against
    qubitized-QPE candidates without prematurely lowering either method to a
    backend circuit.

    Attributes:
        hamiltonian (PauliHamiltonianResource): Generic Pauli Hamiltonian
            resource summary.
        trotter_steps_per_sample (sp.Expr | int | float): Product-formula
            steps per sampled time-evolution segment.
        samples (sp.Expr | int | float): Number of sampled time points or
            signal-processing shots.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            Hamiltonian-weight reduction. Values below one model unitary
            weight concentration.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            product-formula cost factor from randomized evolution.
        rotation_synthesis_t_gates (sp.Expr | int | float): T-gate cost for
            one Pauli rotation.
        logical_qubits (sp.Expr | int | float | None): Explicit logical-qubit
            count. Defaults to data qubits plus one Hadamard-test ancilla.
        representation_error (sp.Expr | int | float): Energy error consumed
            before phase estimation.
        description (str): Reader-facing workload label.

    Raises:
        TypeError: If ``hamiltonian`` is not a ``PauliHamiltonianResource``.
        ValueError: If a positive-valued quantity is non-positive or if a
            reduction/error quantity is negative.

    Example:
        >>> summary = PauliHamiltonianResource(
        ...     n_qubits=4,
        ...     n_pauli_terms=10,
        ...     lambda_norm=20,
        ...     max_locality=2,
        ... )
        >>> workload = TrotterQPEWorkload(
        ...     summary,
        ...     trotter_steps_per_sample=2,
        ...     samples=5,
        ...     unitary_weight_factor=sp.Rational(1, 4),
        ... )
        >>> workload.effective_lambda_norm
        5
    """

    hamiltonian: PauliHamiltonianResource
    trotter_steps_per_sample: _SympyLike
    samples: _SympyLike
    unitary_weight_factor: _SympyLike = 1
    randomized_compilation_factor: _SympyLike = 1
    rotation_synthesis_t_gates: _SympyLike = 1
    logical_qubits: _SympyLike | None = None
    representation_error: _SympyLike = 0
    description: str = ""

    _POSITIVE_FIELDS = (
        "trotter_steps_per_sample",
        "samples",
        "rotation_synthesis_t_gates",
    )
    _OPTIONAL_POSITIVE_FIELDS = ("logical_qubits",)
    _NONNEGATIVE_FIELDS = (
        "unitary_weight_factor",
        "randomized_compilation_factor",
        "representation_error",
    )

    @property
    def effective_lambda_norm(self) -> sp.Expr:
        """Return Hamiltonian normalization after weight concentration.

        Returns:
            sp.Expr: ``lambda_norm * unitary_weight_factor``.
        """
        return sp.simplify(
            self.hamiltonian.lambda_norm
            * _as_expr(self.unitary_weight_factor, "unitary_weight_factor")
        )

    @classmethod
    def from_effective_lambda_norm(
        cls,
        hamiltonian: PauliHamiltonianResource,
        effective_lambda_norm: _SympyLike,
        *,
        trotter_steps_per_sample: _SympyLike,
        samples: _SympyLike,
        randomized_compilation_factor: _SympyLike = 1,
        rotation_synthesis_t_gates: _SympyLike = 1,
        logical_qubits: _SympyLike | None = None,
        representation_error: _SympyLike = 0,
        description: str = "",
    ) -> TrotterQPEWorkload:
        """Build a Trotter workload from a reported effective lambda norm.

        Recent chemistry resource estimates often report the Hamiltonian
        weight after concentration or randomized time-evolution analysis,
        rather than the multiplicative factor that produced it. This
        constructor preserves the original Hamiltonian summary and derives the
        unitary-weight factor from the requested effective lambda norm.

        Args:
            hamiltonian (PauliHamiltonianResource): Original Pauli
                Hamiltonian summary before weight concentration.
            effective_lambda_norm (sp.Expr | int | float): Hamiltonian
                normalization after weight concentration. Must be positive.
            trotter_steps_per_sample (sp.Expr | int | float): Product-formula
                steps per sampled time-evolution segment.
            samples (sp.Expr | int | float): Number of sampled time points or
                signal-processing shots.
            randomized_compilation_factor (sp.Expr | int | float):
                Multiplicative product-formula cost factor from randomized
                evolution. Defaults to 1.
            rotation_synthesis_t_gates (sp.Expr | int | float): T-gate cost
                for one Pauli rotation. Defaults to 1.
            logical_qubits (sp.Expr | int | float | None): Explicit logical
                qubit count. Defaults to data qubits plus one Hadamard-test
                ancilla.
            representation_error (sp.Expr | int | float): Energy error
                consumed before phase estimation. Defaults to 0.
            description (str): Reader-facing workload label.

        Returns:
            TrotterQPEWorkload: Workload whose ``unitary_weight_factor`` is
            ``effective_lambda_norm / hamiltonian.lambda_norm``.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource``.
            ValueError: If either lambda norm is non-positive or any workload
                quantity is invalid.

        Example:
            >>> summary = PauliHamiltonianResource(
            ...     n_qubits=4,
            ...     n_pauli_terms=10,
            ...     lambda_norm=20,
            ...     max_locality=2,
            ... )
            >>> workload = TrotterQPEWorkload.from_effective_lambda_norm(
            ...     summary,
            ...     effective_lambda_norm=5,
            ...     trotter_steps_per_sample=2,
            ...     samples=5,
            ... )
            >>> workload.unitary_weight_factor
            1/4
        """
        if not isinstance(hamiltonian, PauliHamiltonianResource):
            raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
        original_lambda = _as_expr(hamiltonian.lambda_norm, "lambda_norm")
        effective_lambda = _as_expr(effective_lambda_norm, "effective_lambda_norm")
        _validate_positive(original_lambda, "lambda_norm")
        _validate_positive(effective_lambda, "effective_lambda_norm")
        return cls(
            hamiltonian=hamiltonian,
            trotter_steps_per_sample=trotter_steps_per_sample,
            samples=samples,
            unitary_weight_factor=sp.simplify(effective_lambda / original_lambda),
            randomized_compilation_factor=randomized_compilation_factor,
            rotation_synthesis_t_gates=rotation_synthesis_t_gates,
            logical_qubits=logical_qubits,
            representation_error=representation_error,
            description=description,
        )

    def _own_resource_values(self) -> dict[str, sp.Expr]:
        """Return the product-formula workload's algorithm-level values.

        Returns:
            dict[str, sp.Expr]: Effective normalization, sampling, weight,
                synthesis, and error-budget quantities. ``samples`` is
                exposed under the canonical ``trotter_samples`` key.
        """
        return {
            "effective_lambda_norm": self.effective_lambda_norm,
            "trotter_steps_per_sample": _as_expr(
                self.trotter_steps_per_sample,
                "trotter_steps_per_sample",
            ),
            "trotter_samples": _as_expr(self.samples, "samples"),
            "unitary_weight_factor": _as_expr(
                self.unitary_weight_factor,
                "unitary_weight_factor",
            ),
            "randomized_compilation_factor": _as_expr(
                self.randomized_compilation_factor,
                "randomized_compilation_factor",
            ),
            "rotation_synthesis_t_gates": _as_expr(
                self.rotation_synthesis_t_gates,
                "rotation_synthesis_t_gates",
            ),
            "representation_error": _as_expr(
                self.representation_error,
                "representation_error",
            ),
        }


def estimate_qubitized_qpe_resources(
    n_qubits: sp.Expr | int,
    lambda_norm: _SympyLike,
    precision: _SympyLike,
    walk_cost_toffoli: sp.Expr | int,
    *,
    representation: str | HamiltonianRepresentation = (
        HamiltonianRepresentation.DOUBLE_FACTORIZATION
    ),
    sparsity: sp.Expr | int | None = None,
    second_factor_rank: sp.Expr | int | None = None,
    logical_qubits: sp.Expr | int | None = None,
    qpe_register_qubits: _SympyLike = 0,
) -> ResourceEstimate:
    """Estimate logical qubitized-QPE resources for a Hamiltonian.

    This models the walk-based QPE workload with explicit walk costs and
    Hamiltonian representations. For a coarse textbook-level QPE bound
    parameterized by precision bits, see ``estimate_qpe`` in this package.

    Args:
        n_qubits (sp.Expr | int): Number of encoded Hamiltonian qubits.
        lambda_norm (sp.Expr | int | float): LCU block-encoding
            normalization, often the Hamiltonian 1-norm.
        precision (sp.Expr | int | float): Target phase-estimation energy
            precision in a consistent energy unit.
        walk_cost_toffoli (sp.Expr | int): Toffoli cost for one qubitized
            walk operator call.
        representation (str | HamiltonianRepresentation): Hamiltonian
            representation used to choose a default logical-qubit model.
        sparsity (sp.Expr | int | None): Number of nonzero Pauli or LCU
            terms for sparse Pauli-LCU estimates. Defaults to None.
        second_factor_rank (sp.Expr | int | None): Average rank for
            double-factorized methods. Defaults to symbolic ``Xi`` when
            needed.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            When omitted, a representation-level scaling model is used. This
            footprint excludes ``qpe_register_qubits``.
        qpe_register_qubits (sp.Expr | int | float): Optional QPE readout
            register qubits added to the logical footprint. Defaults to 0.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        ValueError: If a finite-set representation is unknown, a required
            sparse parameter is missing, or a positive-valued input is
            non-positive.
        TypeError: If a value cannot be converted into a SymPy expression.
    """
    representation_key = _normalize_representation(representation)
    n_expr = _as_expr(n_qubits, "n_qubits")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    walk_expr = _as_expr(walk_cost_toffoli, "walk_cost_toffoli")
    qpe_qubits = _as_expr(qpe_register_qubits, "qpe_register_qubits")

    _validate_positive(n_expr, "n_qubits")
    _validate_positive(lambda_expr, "lambda_norm")
    _validate_positive(precision_expr, "precision")
    _validate_positive(walk_expr, "walk_cost_toffoli")
    _validate_nonnegative(qpe_qubits, "qpe_register_qubits")

    if logical_qubits is None:
        logical_expr = _default_logical_qubits(
            representation_key,
            n_expr,
            sparsity=sparsity,
            second_factor_rank=second_factor_rank,
        )
    else:
        logical_expr = _as_expr(logical_qubits, "logical_qubits")
        _validate_positive(logical_expr, "logical_qubits")
    logical_expr = sp.simplify(logical_expr + qpe_qubits)

    qpe_iterations = sp.simplify(lambda_expr / precision_expr)
    toffoli_gates = sp.simplify(qpe_iterations * walk_expr)
    return _build_logical_estimate(
        logical_qubits=logical_expr,
        total_gates=toffoli_gates,
        multi_qubit_gates=toffoli_gates,
        t_gates=sp.Integer(0),
        clifford_gates=sp.Integer(0),
        rotation_gates=sp.Integer(0),
        qpe_iterations=qpe_iterations,
    )


def estimate_qubitized_qpe_resources_from_workload(
    workload: HamiltonianQPEWorkload,
    precision: _SympyLike,
) -> ResourceEstimate:
    """Estimate logical qubitized-QPE resources from a workload object.

    Args:
        workload (HamiltonianQPEWorkload): Hamiltonian workload carrying
            lambda norm, sparsity/rank metadata, walk cost, QPE readout
            registers, and representation error.
        precision (sp.Expr | int | float): Total target energy precision
            budget. ``workload.representation_error`` is subtracted before
            estimating QPE iterations.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``workload`` is not a ``HamiltonianQPEWorkload``.
        ValueError: If ``precision`` is non-positive or if the representation
            error consumes the full precision budget.
    """
    if not isinstance(workload, HamiltonianQPEWorkload):
        raise TypeError("workload must be a HamiltonianQPEWorkload instance.")
    return estimate_qubitized_qpe_resources(
        n_qubits=workload.hamiltonian.n_qubits,
        lambda_norm=workload.hamiltonian.lambda_norm,
        precision=workload.algorithmic_precision(precision),
        walk_cost_toffoli=_as_expr(workload.walk_cost_toffoli, "walk_cost_toffoli"),
        representation=workload.normalized_representation,
        sparsity=workload.effective_sparsity,
        second_factor_rank=(
            None
            if workload.second_factor_rank is None
            else _as_expr(workload.second_factor_rank, "second_factor_rank")
        ),
        logical_qubits=(
            None
            if workload.logical_qubits is None
            else _as_expr(workload.logical_qubits, "logical_qubits")
        ),
        qpe_register_qubits=_as_expr(
            workload.qpe_register_qubits,
            "qpe_register_qubits",
        ),
    )


def qubitized_qpe_workload_from_openfermion(
    openfermion_operator: Any,
    *,
    walk_cost_toffoli: _SympyLike,
    representation: str | HamiltonianRepresentation = (
        HamiltonianRepresentation.SPARSE_PAULI_LCU
    ),
    n_qubits: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "openfermion_qubit_operator",
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
    logical_qubits: _SympyLike | None = None,
    representation_error: _SympyLike = 0,
    description: str = "",
    qpe_register_qubits: _SympyLike = 0,
) -> HamiltonianQPEWorkload:
    """Build a qubitized-QPE workload from an OpenFermion operator.

    This helper preserves Qamomile's resource-estimation layer boundary:
    OpenFermion-like data is summarized as a Pauli Hamiltonian, then packaged
    as an algorithm workload without constructing a backend circuit.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like object
            exposing a ``terms`` mapping.
        walk_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            qubitized walk-operator call.
        representation (str | HamiltonianRepresentation): Hamiltonian
            representation used for logical-qubit scaling. Defaults to sparse
            Pauli LCU.
        n_qubits (sp.Expr | int | float | None): Optional encoded qubit-count
            override. Defaults to the operator-inferred width.
        include_constant (bool): Whether to include the identity coefficient
            in the Hamiltonian normalization. Defaults to False.
        source (str): Human-readable Hamiltonian source label.
        sparsity (sp.Expr | int | float | None): Optional sparse-method
            nonzero term count. Defaults to the Hamiltonian term count for
            sparse Pauli LCU.
        second_factor_rank (sp.Expr | int | float | None): Optional
            second-factor rank metadata for factorized representations.
        logical_qubits (sp.Expr | int | float | None): Explicit base logical
            qubit count before QPE readout registers are added.
        representation_error (sp.Expr | int | float): Hamiltonian
            representation error consumed before phase estimation. Defaults
            to 0.
        description (str): Reader-facing workload label. Defaults to an empty
            string.
        qpe_register_qubits (sp.Expr | int | float): Optional QPE readout
            register qubits. Defaults to 0.

    Returns:
        HamiltonianQPEWorkload: Workload backed by the OpenFermion-style
        Hamiltonian summary.

    Raises:
        TypeError: If the OpenFermion object is malformed or any symbolic
            input cannot be converted to SymPy.
        ValueError: If a Pauli label, representation, or resource quantity is
            invalid.
    """
    summary = summarize_openfermion_qubit_operator(
        openfermion_operator,
        n_qubits=n_qubits,
        include_constant=include_constant,
        source=source,
    )
    return HamiltonianQPEWorkload(
        hamiltonian=summary,
        walk_cost_toffoli=walk_cost_toffoli,
        representation=representation,
        sparsity=sparsity,
        second_factor_rank=second_factor_rank,
        logical_qubits=logical_qubits,
        representation_error=representation_error,
        description=description,
        qpe_register_qubits=qpe_register_qubits,
    )


def estimate_trotter_qpe_resources(
    n_qubits: sp.Expr | int,
    n_pauli_terms: sp.Expr | int,
    lambda_norm: _SympyLike,
    precision: _SympyLike,
    *,
    trotter_steps_per_sample: sp.Expr | int,
    samples: sp.Expr | int,
    unitary_weight_factor: _SympyLike = 1,
    randomized_compilation_factor: _SympyLike = 1,
    rotation_synthesis_t_gates: sp.Expr | int = 1,
    logical_qubits: sp.Expr | int | None = None,
) -> ResourceEstimate:
    """Estimate logical single-ancilla Trotter-QPE resources.

    The returned ``rotation_gates`` counts every Pauli rotation executed
    across all QPE iterations, matching the total-count semantics of the
    other ``GateCount`` fields, so ``t_gates == rotation_gates *
    rotation_synthesis_t_gates`` always holds.

    This prices a QPE workload driven by a normalization/precision budget.
    To price a single approximate time evolution e^(iHt) at fixed time and
    error instead, use ``estimate_trotter`` from this package.

    Args:
        n_qubits (sp.Expr | int): Number of encoded Hamiltonian qubits.
        n_pauli_terms (sp.Expr | int): Number of Pauli terms.
        lambda_norm (sp.Expr | int | float): Original Hamiltonian 1-norm.
        precision (sp.Expr | int | float): Target energy precision.
        trotter_steps_per_sample (sp.Expr | int): Product-formula steps per
            Hadamard-test sample.
        samples (sp.Expr | int): Number of sampled time points or shots in
            the signal-processing routine.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            reduction in Hamiltonian weight. Values below one model cost
            reduction. Defaults to one.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            cost factor for partially randomized compilation. Defaults to one.
        rotation_synthesis_t_gates (sp.Expr | int): T-gate cost per small
            Pauli rotation. Defaults to one symbolic T-equivalent unit.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            Defaults to ``n_qubits + 1`` for data plus Hadamard-test ancilla.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        ValueError: If any positive-valued input is non-positive or either
            multiplicative reduction factor is negative.
        TypeError: If a value cannot be converted into a SymPy expression.
    """
    n_expr = _as_expr(n_qubits, "n_qubits")
    terms_expr = _as_expr(n_pauli_terms, "n_pauli_terms")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    steps_expr = _as_expr(trotter_steps_per_sample, "trotter_steps_per_sample")
    samples_expr = _as_expr(samples, "samples")
    weight_factor = _as_expr(unitary_weight_factor, "unitary_weight_factor")
    randomized_factor = _as_expr(
        randomized_compilation_factor,
        "randomized_compilation_factor",
    )
    rotation_t = _as_expr(rotation_synthesis_t_gates, "rotation_synthesis_t_gates")

    for name, expr in [
        ("n_qubits", n_expr),
        ("n_pauli_terms", terms_expr),
        ("lambda_norm", lambda_expr),
        ("precision", precision_expr),
        ("trotter_steps_per_sample", steps_expr),
        ("samples", samples_expr),
        ("rotation_synthesis_t_gates", rotation_t),
    ]:
        _validate_positive(expr, name)
    _validate_nonnegative(weight_factor, "unitary_weight_factor")
    _validate_nonnegative(randomized_factor, "randomized_compilation_factor")

    logical_expr = (
        n_expr + 1
        if logical_qubits is None
        else _as_expr(logical_qubits, "logical_qubits")
    )
    _validate_positive(logical_expr, "logical_qubits")

    effective_lambda = sp.simplify(lambda_expr * weight_factor)
    qpe_iterations = sp.simplify(effective_lambda / precision_expr)
    rotations_per_iteration = sp.simplify(
        samples_expr * steps_expr * terms_expr * randomized_factor
    )
    pauli_rotations = sp.simplify(qpe_iterations * rotations_per_iteration)
    t_gates = sp.simplify(pauli_rotations * rotation_t)
    return _build_logical_estimate(
        logical_qubits=logical_expr,
        total_gates=t_gates,
        multi_qubit_gates=sp.Integer(0),
        t_gates=t_gates,
        clifford_gates=sp.Integer(0),
        rotation_gates=pauli_rotations,
        qpe_iterations=qpe_iterations,
    )


def estimate_trotter_qpe_resources_from_workload(
    workload: TrotterQPEWorkload,
    precision: _SympyLike,
) -> ResourceEstimate:
    """Estimate logical Trotter-QPE resources from a workload object.

    Args:
        workload (TrotterQPEWorkload): Product-formula workload carrying
            Hamiltonian, sampling, unitary-weight, and synthesis assumptions.
        precision (sp.Expr | int | float): Total target energy precision
            budget. ``workload.representation_error`` is subtracted before
            estimating QPE iterations.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``workload`` is not a ``TrotterQPEWorkload``.
        ValueError: If ``precision`` is non-positive or exhausted by the
            workload representation error.
    """
    if not isinstance(workload, TrotterQPEWorkload):
        raise TypeError("workload must be a TrotterQPEWorkload instance.")
    return estimate_trotter_qpe_resources(
        n_qubits=workload.hamiltonian.n_qubits,
        n_pauli_terms=workload.hamiltonian.n_pauli_terms,
        lambda_norm=workload.hamiltonian.lambda_norm,
        precision=workload.algorithmic_precision(precision),
        trotter_steps_per_sample=_as_expr(
            workload.trotter_steps_per_sample,
            "trotter_steps_per_sample",
        ),
        samples=_as_expr(workload.samples, "samples"),
        unitary_weight_factor=_as_expr(
            workload.unitary_weight_factor,
            "unitary_weight_factor",
        ),
        randomized_compilation_factor=_as_expr(
            workload.randomized_compilation_factor,
            "randomized_compilation_factor",
        ),
        rotation_synthesis_t_gates=_as_expr(
            workload.rotation_synthesis_t_gates,
            "rotation_synthesis_t_gates",
        ),
        logical_qubits=(
            None
            if workload.logical_qubits is None
            else _as_expr(workload.logical_qubits, "logical_qubits")
        ),
    )


def estimate_trotter_qpe_resources_from_hamiltonian(
    hamiltonian: PauliHamiltonianResource,
    precision: _SympyLike,
    *,
    trotter_steps_per_sample: sp.Expr | int,
    samples: sp.Expr | int,
    unitary_weight_factor: _SympyLike = 1,
    randomized_compilation_factor: _SympyLike = 1,
    rotation_synthesis_t_gates: sp.Expr | int = 1,
    logical_qubits: sp.Expr | int | None = None,
) -> ResourceEstimate:
    """Estimate logical Trotter-QPE resources from a Hamiltonian summary.

    Args:
        hamiltonian (PauliHamiltonianResource): Pauli-LCU Hamiltonian summary.
        precision (sp.Expr | int | float): Target energy precision.
        trotter_steps_per_sample (sp.Expr | int): Product-formula steps per
            Hadamard-test sample.
        samples (sp.Expr | int): Number of sampled time points or shots.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            reduction in Hamiltonian weight. Defaults to one.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            cost factor for randomized compilation. Defaults to one.
        rotation_synthesis_t_gates (sp.Expr | int): T-gate cost per Pauli
            rotation. Defaults to one.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            Defaults to ``hamiltonian.n_qubits + 1``.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``hamiltonian`` is not a ``PauliHamiltonianResource``.
        ValueError: If any resource quantity is invalid.
    """
    if not isinstance(hamiltonian, PauliHamiltonianResource):
        raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
    return estimate_trotter_qpe_resources(
        n_qubits=hamiltonian.n_qubits,
        n_pauli_terms=hamiltonian.n_pauli_terms,
        lambda_norm=hamiltonian.lambda_norm,
        precision=precision,
        trotter_steps_per_sample=trotter_steps_per_sample,
        samples=samples,
        unitary_weight_factor=unitary_weight_factor,
        randomized_compilation_factor=randomized_compilation_factor,
        rotation_synthesis_t_gates=rotation_synthesis_t_gates,
        logical_qubits=logical_qubits,
    )


def trotter_qpe_workload_from_openfermion(
    openfermion_operator: Any,
    *,
    trotter_steps_per_sample: _SympyLike,
    samples: _SympyLike,
    effective_lambda_norm: _SympyLike | None = None,
    unitary_weight_factor: _SympyLike = 1,
    randomized_compilation_factor: _SympyLike = 1,
    rotation_synthesis_t_gates: _SympyLike = 1,
    logical_qubits: _SympyLike | None = None,
    representation_error: _SympyLike = 0,
    n_qubits: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "openfermion_qubit_operator",
    description: str = "",
) -> TrotterQPEWorkload:
    """Build a Trotter-QPE workload from an OpenFermion operator.

    Use ``effective_lambda_norm`` when a chemistry resource-estimation table
    reports the Hamiltonian weight after concentration. Otherwise provide a
    multiplicative ``unitary_weight_factor`` directly.

    Args:
        openfermion_operator (Any): OpenFermion ``QubitOperator``-like object
            exposing a ``terms`` mapping.
        trotter_steps_per_sample (sp.Expr | int | float): Product-formula
            steps per sampled time-evolution segment.
        samples (sp.Expr | int | float): Number of sampled time points or
            signal-processing shots.
        effective_lambda_norm (sp.Expr | int | float | None): Optional
            Hamiltonian normalization after concentration. When provided, the
            workload derives ``unitary_weight_factor`` from it. Defaults to
            None.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            Hamiltonian-weight reduction used when ``effective_lambda_norm``
            is not provided. Defaults to 1.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            product-formula cost factor from randomized evolution. Defaults
            to 1.
        rotation_synthesis_t_gates (sp.Expr | int | float): T-gate cost for
            one Pauli rotation. Defaults to 1.
        logical_qubits (sp.Expr | int | float | None): Explicit logical qubit
            count. Defaults to data qubits plus one Hadamard-test ancilla.
        representation_error (sp.Expr | int | float): Energy error consumed
            before phase estimation. Defaults to 0.
        n_qubits (sp.Expr | int | float | None): Optional encoded qubit-count
            override. Defaults to the operator-inferred width.
        include_constant (bool): Whether to include the identity coefficient
            in the Hamiltonian normalization. Defaults to False.
        source (str): Human-readable Hamiltonian source label.
        description (str): Reader-facing workload label.

    Returns:
        TrotterQPEWorkload: Workload backed by the OpenFermion-style
        Hamiltonian summary.

    Raises:
        TypeError: If the OpenFermion object is malformed or any symbolic
            input cannot be converted to SymPy.
        ValueError: If a Pauli label or resource quantity is invalid, or if
            ``effective_lambda_norm`` is provided together with a non-unit
            ``unitary_weight_factor``.
    """
    summary = summarize_openfermion_qubit_operator(
        openfermion_operator,
        n_qubits=n_qubits,
        include_constant=include_constant,
        source=source,
    )
    weight_factor = _as_expr(unitary_weight_factor, "unitary_weight_factor")
    if effective_lambda_norm is not None:
        if sp.simplify(weight_factor - 1) != 0:
            raise ValueError(
                "effective_lambda_norm and a non-unit unitary_weight_factor "
                "cannot both be provided."
            )
        return TrotterQPEWorkload.from_effective_lambda_norm(
            summary,
            effective_lambda_norm,
            trotter_steps_per_sample=trotter_steps_per_sample,
            samples=samples,
            randomized_compilation_factor=randomized_compilation_factor,
            rotation_synthesis_t_gates=rotation_synthesis_t_gates,
            logical_qubits=logical_qubits,
            representation_error=representation_error,
            description=description,
        )

    return TrotterQPEWorkload(
        hamiltonian=summary,
        trotter_steps_per_sample=trotter_steps_per_sample,
        samples=samples,
        unitary_weight_factor=weight_factor,
        randomized_compilation_factor=randomized_compilation_factor,
        rotation_synthesis_t_gates=rotation_synthesis_t_gates,
        logical_qubits=logical_qubits,
        representation_error=representation_error,
        description=description,
    )


def _sparse_pauli_lcu_qubits(
    n_qubits: sp.Expr,
    *,
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
) -> sp.Expr:
    """Return sparse Pauli-LCU logical-qubit scaling.

    Args:
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | float | None): Sparse-method nonzero term
            count. Required for this representation.
        second_factor_rank (sp.Expr | int | float | None): Unused by this
            representation.

    Returns:
        sp.Expr: ``n + sqrt(sparsity)``.

    Raises:
        ValueError: If ``sparsity`` is missing or non-positive.
    """
    if sparsity is None:
        raise ValueError("sparsity is required for sparse Pauli-LCU QPE estimates.")
    sparsity_expr = _as_expr(sparsity, "sparsity")
    _validate_positive(sparsity_expr, "sparsity")
    return sp.simplify(n_qubits + sp.sqrt(sparsity_expr))


def _single_factorization_qubits(
    n_qubits: sp.Expr,
    *,
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
) -> sp.Expr:
    """Return single-factorization logical-qubit scaling.

    Args:
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | float | None): Unused by this
            representation.
        second_factor_rank (sp.Expr | int | float | None): Unused by this
            representation.

    Returns:
        sp.Expr: ``n ** (3/2)``.
    """
    return sp.simplify(n_qubits ** sp.Rational(3, 2))


def _double_factorization_qubits(
    n_qubits: sp.Expr,
    *,
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
) -> sp.Expr:
    """Return double-factorization logical-qubit scaling.

    Args:
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | float | None): Unused by this
            representation.
        second_factor_rank (sp.Expr | int | float | None): Average second
            factorization rank. Defaults to the symbolic ``Xi`` when None.

    Returns:
        sp.Expr: ``n * sqrt(second_factor_rank)``.

    Raises:
        ValueError: If ``second_factor_rank`` is provably non-positive.
    """
    rank_expr = (
        sp.Symbol("Xi", positive=True)
        if second_factor_rank is None
        else _as_expr(second_factor_rank, "second_factor_rank")
    )
    _validate_positive(rank_expr, "second_factor_rank")
    return sp.simplify(n_qubits * sp.sqrt(rank_expr))


def _tensor_hypercontraction_qubits(
    n_qubits: sp.Expr,
    *,
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
) -> sp.Expr:
    """Return tensor-hypercontraction logical-qubit scaling.

    Args:
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | float | None): Unused by this
            representation.
        second_factor_rank (sp.Expr | int | float | None): Unused by this
            representation.

    Returns:
        sp.Expr: ``n``.
    """
    return n_qubits


def _unitary_weight_concentration_qubits(
    n_qubits: sp.Expr,
    *,
    sparsity: _SympyLike | None = None,
    second_factor_rank: _SympyLike | None = None,
) -> sp.Expr:
    """Return unitary-weight-concentration logical-qubit scaling.

    Args:
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | float | None): Unused by this
            representation.
        second_factor_rank (sp.Expr | int | float | None): Unused by this
            representation.

    Returns:
        sp.Expr: ``n + 1`` for the data register plus one ancilla.
    """
    return n_qubits + 1


_LogicalQubitsModel = Callable[..., sp.Expr]

_REPRESENTATION_MODELS: dict[str, _LogicalQubitsModel] = {
    HamiltonianRepresentation.SPARSE_PAULI_LCU.value: _sparse_pauli_lcu_qubits,
    HamiltonianRepresentation.SINGLE_FACTORIZATION.value: (
        _single_factorization_qubits
    ),
    HamiltonianRepresentation.DOUBLE_FACTORIZATION.value: (
        _double_factorization_qubits
    ),
    HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF.value: (
        _double_factorization_qubits
    ),
    HamiltonianRepresentation.TENSOR_HYPERCONTRACTION.value: (
        _tensor_hypercontraction_qubits
    ),
    HamiltonianRepresentation.UNITARY_WEIGHT_CONCENTRATION.value: (
        _unitary_weight_concentration_qubits
    ),
}


def register_hamiltonian_representation(
    name: str,
    logical_qubits: _LogicalQubitsModel,
) -> str:
    """Register a custom Hamiltonian representation scaling model.

    Hamiltonian representations are open: any registered name is accepted by
    ``HamiltonianQPEWorkload`` and ``estimate_qubitized_qpe_resources``.
    ``HamiltonianRepresentation`` enumerates the built-ins; this function
    extends the set without editing the package.

    Args:
        name (str): Representation key, used as the ``representation``
            argument of workloads and estimators.
        logical_qubits (Callable[..., sp.Expr]): Scaling model called as
            ``logical_qubits(n_qubits, *, sparsity=None,
            second_factor_rank=None)`` and returning the representation's
            logical-qubit count expression. Re-registering the identical
            callable is a no-op.

    Returns:
        str: The registered representation key.

    Raises:
        ValueError: If ``name`` is empty or already registered with a
            different model.

    Example:
        >>> def _flat_qubits(n_qubits, *, sparsity=None, second_factor_rank=None):
        ...     return n_qubits + 3
        >>> register_hamiltonian_representation("flat_encoding", _flat_qubits)
        'flat_encoding'
    """
    key = str(name)
    if not key:
        raise ValueError("Representation names must be non-empty strings.")
    existing = _REPRESENTATION_MODELS.get(key)
    if existing is not None and existing is not logical_qubits:
        raise ValueError(
            f"Hamiltonian representation {key!r} is already registered with a "
            "different model."
        )
    _REPRESENTATION_MODELS[key] = logical_qubits
    return key


def _default_logical_qubits(
    representation: str,
    n_qubits: sp.Expr,
    *,
    sparsity: sp.Expr | int | None,
    second_factor_rank: sp.Expr | int | None,
) -> sp.Expr:
    """Return representation-level logical-qubit scaling.

    Args:
        representation (str): Normalized Hamiltonian representation key.
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | None): Sparse-method nonzero term count.
        second_factor_rank (sp.Expr | int | None): Average rank for
            double-factorized methods.

    Returns:
        sp.Expr: Symbolic logical-qubit estimate.

    Raises:
        ValueError: If the representation's model rejects its inputs, such
            as a sparse representation without ``sparsity``.
    """
    model = _REPRESENTATION_MODELS[_normalize_representation(representation)]
    return model(
        n_qubits,
        sparsity=sparsity,
        second_factor_rank=second_factor_rank,
    )


def _build_logical_estimate(
    *,
    logical_qubits: sp.Expr,
    total_gates: sp.Expr,
    multi_qubit_gates: sp.Expr,
    t_gates: sp.Expr,
    clifford_gates: sp.Expr,
    qpe_iterations: sp.Expr,
    rotation_gates: sp.Expr,
) -> ResourceEstimate:
    """Create a logical estimate and collect free symbolic parameters.

    Args:
        logical_qubits (sp.Expr): Logical qubit count.
        total_gates (sp.Expr): Total logical gate-count proxy.
        multi_qubit_gates (sp.Expr): Multi-qubit gate-count proxy.
        t_gates (sp.Expr): T count.
        clifford_gates (sp.Expr): Clifford count.
        qpe_iterations (sp.Expr): QPE iteration count.
        rotation_gates (sp.Expr): Rotation gate-count proxy.

    Returns:
        ResourceEstimate: Logical estimate with collected parameters.
    """
    estimate = ResourceEstimate(
        qubits=logical_qubits,
        gates=GateCount(
            total=total_gates,
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=multi_qubit_gates,
            t_gates=t_gates,
            clifford_gates=clifford_gates,
            rotation_gates=rotation_gates,
            oracle_calls={"qpe_iterations": qpe_iterations},
            oracle_queries={},
        ),
    )
    return estimate.simplify()


def _normalize_representation(
    representation: str | HamiltonianRepresentation,
) -> str:
    """Normalize a public representation value to its registered key.

    Args:
        representation (str | HamiltonianRepresentation): User-provided
            representation. Built-in enum members and registered custom
            keys are both accepted.

    Returns:
        str: Normalized representation key.

    Raises:
        ValueError: If ``representation`` is not a registered
            representation.
    """
    key = str(representation)
    if key not in _REPRESENTATION_MODELS:
        valid = ", ".join(sorted(_REPRESENTATION_MODELS))
        raise ValueError(
            f"Unknown Hamiltonian representation {representation!r}; valid: {valid}."
        )
    return key
