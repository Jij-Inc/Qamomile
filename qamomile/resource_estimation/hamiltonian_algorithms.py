"""Estimate logical resources for Hamiltonian phase-estimation workloads."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import sympy as sp

from qamomile.circuit.estimator import GateCount, ResourceEstimate
from qamomile.resource_estimation.hamiltonian import PauliHamiltonianResource

_SympyLike = sp.Expr | int | float


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
class HamiltonianQPEWorkload:
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
            count. Defaults to the representation-specific scaling model.
        representation_error (sp.Expr | int | float): Hamiltonian
            representation error budget.
        description (str): Reader-facing model label.

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

    def __post_init__(self) -> None:
        """Validate workload fields after dataclass construction.

        Raises:
            TypeError: If ``hamiltonian`` is not a
                ``PauliHamiltonianResource``.
            ValueError: If any positive-valued quantity is non-positive or if
                ``representation_error`` is negative.
        """
        if not isinstance(self.hamiltonian, PauliHamiltonianResource):
            raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
        _validate_positive(
            _as_expr(self.walk_cost_toffoli, "walk_cost_toffoli"),
            "walk_cost_toffoli",
        )
        if self.sparsity is not None:
            _validate_positive(_as_expr(self.sparsity, "sparsity"), "sparsity")
        if self.second_factor_rank is not None:
            _validate_positive(
                _as_expr(self.second_factor_rank, "second_factor_rank"),
                "second_factor_rank",
            )
        if self.logical_qubits is not None:
            _validate_positive(
                _as_expr(self.logical_qubits, "logical_qubits"),
                "logical_qubits",
            )
        _validate_nonnegative(
            _as_expr(self.representation_error, "representation_error"),
            "representation_error",
        )
        _normalize_representation(self.representation)

    @property
    def normalized_representation(self) -> HamiltonianRepresentation:
        """Return the normalized Hamiltonian representation.

        Returns:
            HamiltonianRepresentation: Normalized finite-set representation.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workload to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: String-valued workload metadata.
        """
        return {
            "hamiltonian": self.hamiltonian.to_dict(),
            "representation": self.normalized_representation.value,
            "walk_cost_toffoli": str(
                _as_expr(self.walk_cost_toffoli, "walk_cost_toffoli")
            ),
            "sparsity": (
                None
                if self.effective_sparsity is None
                else str(self.effective_sparsity)
            ),
            "second_factor_rank": (
                None
                if self.second_factor_rank is None
                else str(_as_expr(self.second_factor_rank, "second_factor_rank"))
            ),
            "logical_qubits": (
                None
                if self.logical_qubits is None
                else str(_as_expr(self.logical_qubits, "logical_qubits"))
            ),
            "representation_error": str(
                _as_expr(self.representation_error, "representation_error")
            ),
            "description": self.description,
        }

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the workload.

        Returns:
            dict[str, sp.Expr]: Hamiltonian summary values plus algorithm
            workload parameters.
        """
        values = self.hamiltonian.resource_values()
        values["walk_cost_toffoli"] = _as_expr(
            self.walk_cost_toffoli,
            "walk_cost_toffoli",
        )
        values["representation_error"] = _as_expr(
            self.representation_error,
            "representation_error",
        )
        if self.effective_sparsity is not None:
            values["n_pauli_terms"] = self.effective_sparsity
        return values

    def algorithmic_precision(self, precision: _SympyLike) -> sp.Expr:
        """Return precision remaining after representation error.

        Args:
            precision (sp.Expr | int | float): Total target energy precision
                budget.

        Returns:
            sp.Expr: Precision budget available to phase estimation after
                subtracting ``representation_error``.

        Raises:
            ValueError: If ``precision`` is non-positive or if the remaining
                algorithmic precision is provably non-positive.
            TypeError: If ``precision`` cannot be converted into a SymPy
                expression.
        """
        precision_expr = _as_expr(precision, "precision")
        _validate_positive(precision_expr, "precision")
        remaining = sp.simplify(
            precision_expr - _as_expr(self.representation_error, "representation_error")
        )
        _validate_positive(remaining, "algorithmic_precision")
        return remaining

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
        contract, and the block-encoding footprint plus optional QPE readout
        qubits as the explicit logical-qubit count.

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
            logical_qubits=sp.simplify(block_values["logical_qubits"] + qpe_qubits),
            representation_error=representation_error,
            description=block_encoding.name if description is None else description,
        )


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
) -> ResourceEstimate:
    """Estimate logical qubitized-QPE resources for a Hamiltonian.

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
            When omitted, a representation-level scaling model is used.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        ValueError: If a finite-set representation is unknown, a required
            sparse parameter is missing, or a positive-valued input is
            non-positive.
        TypeError: If a value cannot be converted into a SymPy expression.
    """
    representation_enum = _normalize_representation(representation)
    n_expr = _as_expr(n_qubits, "n_qubits")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    walk_expr = _as_expr(walk_cost_toffoli, "walk_cost_toffoli")

    _validate_positive(n_expr, "n_qubits")
    _validate_positive(lambda_expr, "lambda_norm")
    _validate_positive(precision_expr, "precision")
    _validate_positive(walk_expr, "walk_cost_toffoli")

    if logical_qubits is None:
        logical_expr = _default_logical_qubits(
            representation_enum,
            n_expr,
            sparsity=sparsity,
            second_factor_rank=second_factor_rank,
        )
    else:
        logical_expr = _as_expr(logical_qubits, "logical_qubits")
        _validate_positive(logical_expr, "logical_qubits")

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
            lambda norm, sparsity/rank metadata, and walk cost.
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
    pauli_rotations = sp.simplify(
        samples_expr * steps_expr * terms_expr * randomized_factor
    )
    logical_depth = sp.simplify(qpe_iterations * pauli_rotations)
    t_gates = sp.simplify(logical_depth * rotation_t)
    return _build_logical_estimate(
        logical_qubits=logical_expr,
        total_gates=t_gates,
        multi_qubit_gates=sp.Integer(0),
        t_gates=t_gates,
        clifford_gates=sp.Integer(0),
        rotation_gates=pauli_rotations,
        qpe_iterations=qpe_iterations,
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


def _default_logical_qubits(
    representation: HamiltonianRepresentation,
    n_qubits: sp.Expr,
    *,
    sparsity: sp.Expr | int | None,
    second_factor_rank: sp.Expr | int | None,
) -> sp.Expr:
    """Return representation-level logical-qubit scaling.

    Args:
        representation (HamiltonianRepresentation): Hamiltonian representation.
        n_qubits (sp.Expr): Encoded Hamiltonian qubit count.
        sparsity (sp.Expr | int | None): Sparse-method nonzero term count.
        second_factor_rank (sp.Expr | int | None): Average rank for
            double-factorized methods.

    Returns:
        sp.Expr: Symbolic logical-qubit estimate.

    Raises:
        ValueError: If the sparse representation lacks ``sparsity``.
    """
    n = n_qubits
    match representation:
        case HamiltonianRepresentation.SPARSE_PAULI_LCU:
            if sparsity is None:
                raise ValueError(
                    "sparsity is required for sparse Pauli-LCU QPE estimates."
                )
            sparsity_expr = _as_expr(sparsity, "sparsity")
            _validate_positive(sparsity_expr, "sparsity")
            return sp.simplify(n + sp.sqrt(sparsity_expr))
        case HamiltonianRepresentation.SINGLE_FACTORIZATION:
            return sp.simplify(n ** sp.Rational(3, 2))
        case (
            HamiltonianRepresentation.DOUBLE_FACTORIZATION
            | HamiltonianRepresentation.SYMMETRY_COMPRESSED_DF
        ):
            rank_expr = (
                sp.Symbol("Xi", positive=True)
                if second_factor_rank is None
                else _as_expr(second_factor_rank, "second_factor_rank")
            )
            _validate_positive(rank_expr, "second_factor_rank")
            return sp.simplify(n * sp.sqrt(rank_expr))
        case HamiltonianRepresentation.TENSOR_HYPERCONTRACTION:
            return n
        case HamiltonianRepresentation.UNITARY_WEIGHT_CONCENTRATION:
            return n + 1
        case _:
            raise ValueError(f"Unhandled Hamiltonian representation: {representation}")


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
        qubits=sp.simplify(logical_qubits),
        gates=GateCount(
            total=sp.simplify(total_gates),
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=sp.simplify(multi_qubit_gates),
            t_gates=sp.simplify(t_gates),
            clifford_gates=sp.simplify(clifford_gates),
            rotation_gates=sp.simplify(rotation_gates),
            oracle_calls={"qpe_iterations": sp.simplify(qpe_iterations)},
            oracle_queries={},
        ),
    )
    return estimate.simplify()


def _normalize_representation(
    representation: str | HamiltonianRepresentation,
) -> HamiltonianRepresentation:
    """Normalize a public representation value.

    Args:
        representation (str | HamiltonianRepresentation): User-provided
            representation.

    Returns:
        HamiltonianRepresentation: Normalized enum value.

    Raises:
        ValueError: If ``representation`` is not known.
    """
    try:
        return HamiltonianRepresentation(representation)
    except ValueError as exc:
        valid = ", ".join(item.value for item in HamiltonianRepresentation)
        raise ValueError(
            f"Unknown Hamiltonian representation {representation!r}; valid: {valid}."
        ) from exc


def _as_expr(value: _SympyLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    Args:
        value (sp.Expr | int | float): Value to convert.
        name (str): Field name used in error messages.

    Returns:
        sp.Expr: Converted SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be sympified.
    """
    try:
        return sp.sympify(value)
    except (TypeError, sp.SympifyError) as exc:
        raise TypeError(f"{name} must be a numeric or SymPy expression.") from exc


def _validate_positive(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is positive when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is not positive.
    """
    if expr.is_positive is False:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative(expr: sp.Expr, name: str) -> None:
    """Validate that an expression is nonnegative when decidable.

    Args:
        expr (sp.Expr): Expression to validate.
        name (str): Field name used in error messages.

    Raises:
        ValueError: If SymPy can prove that ``expr`` is negative.
    """
    if expr.is_nonnegative is False:
        raise ValueError(f"{name} must be nonnegative.")
