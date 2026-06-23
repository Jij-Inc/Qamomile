"""Estimate logical resources for quantum-chemistry FTQC workflows."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import sympy as sp

from qamomile.resource_estimation import GateCount, ResourceEstimate

_SympyLike = sp.Expr | int | float
_CoefficientLike = _SympyLike | complex


class ChemistryQPEMethod(enum.StrEnum):
    """Select a chemistry Hamiltonian representation for QPE estimates.

    Attributes:
        SPARSE: Sparse Pauli-LCU representation.
        SINGLE_FACTORIZATION: Single-factorized two-electron tensor.
        DOUBLE_FACTORIZATION: Double-factorized two-electron tensor.
        TENSOR_HYPERCONTRACTION: Tensor-hypercontracted representation.
        SYMMETRY_COMPRESSED_DF: Symmetry-compressed double factorization.
        UNITARY_WEIGHT_CONCENTRATION: Early-FTQC unitary-weight concentration.

    Example:
        >>> ChemistryQPEMethod("tensor_hypercontraction")
        <ChemistryQPEMethod.TENSOR_HYPERCONTRACTION: 'tensor_hypercontraction'>
    """

    SPARSE = "sparse"
    SINGLE_FACTORIZATION = "single_factorization"
    DOUBLE_FACTORIZATION = "double_factorization"
    TENSOR_HYPERCONTRACTION = "tensor_hypercontraction"
    SYMMETRY_COMPRESSED_DF = "symmetry_compressed_df"
    UNITARY_WEIGHT_CONCENTRATION = "unitary_weight_concentration"


@dataclass(frozen=True)
class PauliHamiltonianResource:
    """Summarize a Pauli-LCU Hamiltonian for chemistry resource estimates.

    Attributes:
        n_spin_orbitals (sp.Expr): Number of spin orbitals or encoded qubits.
        n_pauli_terms (sp.Expr): Number of non-identity Pauli terms.
        lambda_norm (sp.Expr): Sum of absolute non-identity Pauli
            coefficients used as the LCU normalization proxy.
        max_locality (sp.Expr): Maximum number of non-identity Pauli factors
            in any Hamiltonian term.
        constant (sp.Expr): Constant energy shift stored on the Hamiltonian.
        constant_included (bool): Whether ``constant`` was included in
            ``lambda_norm``.
        source (str): Human-readable source label.

    Raises:
        ValueError: If a positive-valued quantity is non-positive or if a
            nonnegative-valued quantity is negative.

    Example:
        >>> import qamomile.observable as qm_o
        >>> hamiltonian = 0.5 * qm_o.Z(0) + 0.25 * qm_o.X(1)
        >>> summary = summarize_pauli_hamiltonian(hamiltonian)
        >>> summary.n_pauli_terms
        2
    """

    n_spin_orbitals: sp.Expr
    n_pauli_terms: sp.Expr
    lambda_norm: sp.Expr
    max_locality: sp.Expr
    constant: sp.Expr = sp.Integer(0)
    constant_included: bool = False
    source: str = "pauli_lcu"

    def __post_init__(self) -> None:
        """Validate summary fields after dataclass construction.

        Raises:
            ValueError: If a positive-valued quantity is non-positive or if a
                nonnegative-valued quantity is negative.
        """
        _validate_positive(self.n_spin_orbitals, "n_spin_orbitals")
        _validate_nonnegative(self.n_pauli_terms, "n_pauli_terms")
        _validate_nonnegative(self.lambda_norm, "lambda_norm")
        _validate_nonnegative(self.max_locality, "max_locality")
        _validate_nonnegative(sp.Abs(self.constant), "constant")

    def with_lambda_scale(
        self,
        scale: _SympyLike,
        *,
        source: str | None = None,
    ) -> PauliHamiltonianResource:
        """Return a copy with the Hamiltonian normalization rescaled.

        Args:
            scale (sp.Expr | int | float): Multiplicative scale applied to
                ``lambda_norm``. Values below one model transformations that
                reduce the effective Hamiltonian weight.
            source (str | None): Optional replacement source label. Defaults
                to preserving ``self.source``.

        Returns:
            PauliHamiltonianResource: New summary with rescaled
                ``lambda_norm``.

        Raises:
            ValueError: If ``scale`` is negative.
        """
        scale_expr = _as_expr(scale, "scale")
        _validate_nonnegative(scale_expr, "scale")
        return PauliHamiltonianResource(
            n_spin_orbitals=self.n_spin_orbitals,
            n_pauli_terms=self.n_pauli_terms,
            lambda_norm=sp.simplify(self.lambda_norm * scale_expr),
            max_locality=self.max_locality,
            constant=self.constant,
            constant_included=self.constant_included,
            source=self.source if source is None else source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Hamiltonian summary to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: String-valued resource summary.
        """
        return {
            "n_spin_orbitals": str(self.n_spin_orbitals),
            "n_pauli_terms": str(self.n_pauli_terms),
            "lambda_norm": str(self.lambda_norm),
            "max_locality": str(self.max_locality),
            "constant": str(self.constant),
            "constant_included": self.constant_included,
            "source": self.source,
        }


@dataclass(frozen=True)
class ChemistryQPEModel:
    """Describe a chemistry representation used by QPE resource estimates.

    Attributes:
        hamiltonian (PauliHamiltonianResource): Pauli-LCU Hamiltonian summary.
        walk_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            qubitized walk operator call.
        method (str | ChemistryQPEMethod): QPE representation or optimization
            strategy used to choose default logical-qubit scaling.
        sparsity (sp.Expr | int | float | None): Sparse-method nonzero term
            count. Defaults to ``hamiltonian.n_pauli_terms`` for sparse QPE.
        second_factor_rank (sp.Expr | int | float | None): Average second
            factorization rank for double-factorized methods.
        logical_qubits (sp.Expr | int | float | None): Explicit logical-qubit
            count. Defaults to the method-specific scaling model.
        truncation_error (sp.Expr | int | float): Hamiltonian representation
            error budget.
        description (str): Reader-facing model label.

    Raises:
        ValueError: If any positive-valued quantity is non-positive or if
            ``truncation_error`` is negative.

    Example:
        >>> summary = PauliHamiltonianResource(
        ...     n_spin_orbitals=4,
        ...     n_pauli_terms=10,
        ...     lambda_norm=20,
        ...     max_locality=2,
        ... )
        >>> model = ChemistryQPEModel(
        ...     summary,
        ...     walk_cost_toffoli=100,
        ...     method=ChemistryQPEMethod.SPARSE,
        ... )
        >>> model.effective_sparsity
        10
    """

    hamiltonian: PauliHamiltonianResource
    walk_cost_toffoli: _SympyLike
    method: str | ChemistryQPEMethod = ChemistryQPEMethod.DOUBLE_FACTORIZATION
    sparsity: _SympyLike | None = None
    second_factor_rank: _SympyLike | None = None
    logical_qubits: _SympyLike | None = None
    truncation_error: _SympyLike = 0
    description: str = ""

    def __post_init__(self) -> None:
        """Validate model fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued quantity is non-positive or if
                ``truncation_error`` is negative.
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
            _as_expr(self.truncation_error, "truncation_error"),
            "truncation_error",
        )
        _normalize_method(self.method)

    @property
    def normalized_method(self) -> ChemistryQPEMethod:
        """Return the normalized QPE method.

        Returns:
            ChemistryQPEMethod: Normalized finite-set method.
        """
        return _normalize_method(self.method)

    @property
    def effective_sparsity(self) -> sp.Expr | None:
        """Return sparse-method term count with a Hamiltonian fallback.

        Returns:
            sp.Expr | None: Explicit sparsity, Hamiltonian term count for the
                sparse method, or None for non-sparse methods.
        """
        if self.sparsity is not None:
            return _as_expr(self.sparsity, "sparsity")
        if self.normalized_method == ChemistryQPEMethod.SPARSE:
            return self.hamiltonian.n_pauli_terms
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: String-valued model metadata.
        """
        return {
            "hamiltonian": self.hamiltonian.to_dict(),
            "method": self.normalized_method.value,
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
            "truncation_error": str(
                _as_expr(self.truncation_error, "truncation_error")
            ),
            "description": self.description,
        }


def summarize_pauli_hamiltonian(
    hamiltonian: Any,
    *,
    n_spin_orbitals: _SympyLike | None = None,
    include_constant: bool = False,
    source: str = "pauli_lcu",
) -> PauliHamiltonianResource:
    """Summarize a Qamomile Pauli Hamiltonian for chemistry estimates.

    Args:
        hamiltonian (Any): ``qamomile.observable.Hamiltonian`` instance to
            summarize.
        n_spin_orbitals (sp.Expr | int | float | None): Override for the
            encoded active-space size. Defaults to ``hamiltonian.num_qubits``.
        include_constant (bool): Whether to include the Hamiltonian constant
            term in ``lambda_norm``. Defaults to False, modeling the constant
            as a classical energy shift.
        source (str): Human-readable source label for the summary.

    Returns:
        PauliHamiltonianResource: Hamiltonian summary containing term count,
            lambda norm, constant, and max locality.

    Raises:
        TypeError: If ``hamiltonian`` is not a Qamomile Hamiltonian.
        ValueError: If the orbital count or derived norm is invalid.
    """
    import qamomile.observable as qm_o

    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        raise TypeError(
            "hamiltonian must be a qamomile.observable.Hamiltonian instance."
        )

    n_expr = (
        _as_expr(hamiltonian.num_qubits, "n_spin_orbitals")
        if n_spin_orbitals is None
        else _as_expr(n_spin_orbitals, "n_spin_orbitals")
    )
    lambda_norm = sp.Integer(0)
    max_locality = 0
    for operators, coeff in hamiltonian:
        lambda_norm += _abs_as_expr(coeff)
        max_locality = max(max_locality, len(operators))
    constant = _as_expr(hamiltonian.constant, "constant")
    if include_constant:
        lambda_norm += _abs_as_expr(hamiltonian.constant)

    return PauliHamiltonianResource(
        n_spin_orbitals=n_expr,
        n_pauli_terms=sp.Integer(len(hamiltonian)),
        lambda_norm=sp.simplify(lambda_norm),
        max_locality=sp.Integer(max_locality),
        constant=constant,
        constant_included=include_constant,
        source=source,
    )


def estimate_qubitized_chemistry_qpe(
    n_spin_orbitals: sp.Expr | int,
    lambda_norm: _SympyLike,
    precision: _SympyLike,
    walk_cost_toffoli: sp.Expr | int,
    *,
    method: str | ChemistryQPEMethod = ChemistryQPEMethod.DOUBLE_FACTORIZATION,
    sparsity: sp.Expr | int | None = None,
    second_factor_rank: sp.Expr | int | None = None,
    logical_qubits: sp.Expr | int | None = None,
) -> ResourceEstimate:
    """Estimate logical qubitized QPE resources for molecular Hamiltonians.

    Args:
        n_spin_orbitals (sp.Expr | int): Number of spin orbitals in the
            active-space Hamiltonian.
        lambda_norm (sp.Expr | int | float): LCU block-encoding
            normalization, often the representation-dependent Hamiltonian
            1-norm.
        precision (sp.Expr | int | float): Target phase-estimation energy
            precision in Hartree or another consistent energy unit.
        walk_cost_toffoli (sp.Expr | int): Toffoli cost for one qubitized
            walk operator call.
        method (str | ChemistryQPEMethod): Hamiltonian representation used
            to choose a default logical-qubit model. Defaults to double
            factorization.
        sparsity (sp.Expr | int | None): Number of nonzero Pauli or LCU
            terms for the sparse method. Required only when using the sparse
            default logical-qubit model.
        second_factor_rank (sp.Expr | int | None): Average second
            factorization rank for double-factorized methods. Defaults to a
            symbolic ``Xi``.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            When omitted, a representation-level scaling model is used.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        ValueError: If a finite-set method is unknown, a required sparse
            parameter is missing, or a positive-valued input is non-positive.
        TypeError: If a value cannot be converted into a SymPy expression.
    """
    method_enum = _normalize_method(method)
    n_expr = _as_expr(n_spin_orbitals, "n_spin_orbitals")
    lambda_expr = _as_expr(lambda_norm, "lambda_norm")
    precision_expr = _as_expr(precision, "precision")
    walk_expr = _as_expr(walk_cost_toffoli, "walk_cost_toffoli")

    _validate_positive(n_expr, "n_spin_orbitals")
    _validate_positive(lambda_expr, "lambda_norm")
    _validate_positive(precision_expr, "precision")
    _validate_positive(walk_expr, "walk_cost_toffoli")

    if logical_qubits is None:
        logical_expr = _default_logical_qubits(
            method_enum,
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


def estimate_qubitized_chemistry_qpe_from_model(
    model: ChemistryQPEModel,
    precision: _SympyLike,
) -> ResourceEstimate:
    """Estimate logical qubitized QPE resources from a chemistry model.

    Args:
        model (ChemistryQPEModel): Hamiltonian representation model carrying
            lambda norm, sparsity/rank metadata, and walk cost.
        precision (sp.Expr | int | float): Target phase-estimation energy
            precision.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``model`` is not a ``ChemistryQPEModel``.
        ValueError: If ``precision`` is non-positive.
    """
    if not isinstance(model, ChemistryQPEModel):
        raise TypeError("model must be a ChemistryQPEModel instance.")
    return estimate_qubitized_chemistry_qpe(
        n_spin_orbitals=model.hamiltonian.n_spin_orbitals,
        lambda_norm=model.hamiltonian.lambda_norm,
        precision=precision,
        walk_cost_toffoli=_as_expr(model.walk_cost_toffoli, "walk_cost_toffoli"),
        method=model.normalized_method,
        sparsity=model.effective_sparsity,
        second_factor_rank=(
            None
            if model.second_factor_rank is None
            else _as_expr(model.second_factor_rank, "second_factor_rank")
        ),
        logical_qubits=(
            None
            if model.logical_qubits is None
            else _as_expr(model.logical_qubits, "logical_qubits")
        ),
    )


def estimate_single_ancilla_trotter_qpe(
    n_spin_orbitals: sp.Expr | int,
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
    """Estimate logical early-FTQC single-ancilla Trotter QPE resources.

    This estimator models the style of early-FTQC chemistry proposals that
    combine single-ancilla QPE, partially randomized product formulas, and
    Hamiltonian-weight reduction such as unitary weight concentration.

    Args:
        n_spin_orbitals (sp.Expr | int): Number of spin orbitals.
        n_pauli_terms (sp.Expr | int): Number of Pauli LCU terms.
        lambda_norm (sp.Expr | int | float): Original Hamiltonian 1-norm.
        precision (sp.Expr | int | float): Target energy precision.
        trotter_steps_per_sample (sp.Expr | int): Product-formula steps per
            Hadamard-test sample.
        samples (sp.Expr | int): Number of sampled time points or shots in
            the signal-processing routine.
        unitary_weight_factor (sp.Expr | int | float): Multiplicative
            reduction in Hamiltonian weight after spectrally invariant
            transformations. Values below one model cost reduction. Defaults
            to one.
        randomized_compilation_factor (sp.Expr | int | float): Multiplicative
            cost factor for partially randomized compilation. Defaults to one.
        rotation_synthesis_t_gates (sp.Expr | int): T-gate cost per small
            Pauli rotation. Defaults to one symbolic T-equivalent unit.
        logical_qubits (sp.Expr | int | None): Explicit logical-qubit count.
            Defaults to ``n_spin_orbitals + 1`` for the data register plus
            the Hadamard-test ancilla.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        ValueError: If any positive-valued input is non-positive or either
            multiplicative reduction factor is negative.
        TypeError: If a value cannot be converted into a SymPy expression.
    """
    n_expr = _as_expr(n_spin_orbitals, "n_spin_orbitals")
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
        ("n_spin_orbitals", n_expr),
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


def estimate_single_ancilla_trotter_qpe_from_hamiltonian(
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
    """Estimate logical Trotter QPE resources from a Hamiltonian summary.

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
            Defaults to ``hamiltonian.n_spin_orbitals + 1``.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``hamiltonian`` is not a ``PauliHamiltonianResource``.
        ValueError: If any resource quantity is invalid.
    """
    if not isinstance(hamiltonian, PauliHamiltonianResource):
        raise TypeError("hamiltonian must be a PauliHamiltonianResource.")
    return estimate_single_ancilla_trotter_qpe(
        n_spin_orbitals=hamiltonian.n_spin_orbitals,
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
    method: ChemistryQPEMethod,
    n_spin_orbitals: sp.Expr,
    *,
    sparsity: sp.Expr | int | None,
    second_factor_rank: sp.Expr | int | None,
) -> sp.Expr:
    """Return representation-level logical-qubit scaling.

    Args:
        method (ChemistryQPEMethod): Hamiltonian representation.
        n_spin_orbitals (sp.Expr): Number of spin orbitals.
        sparsity (sp.Expr | int | None): Sparse-method nonzero term count.
        second_factor_rank (sp.Expr | int | None): Average rank for
            double-factorized methods.

    Returns:
        sp.Expr: Symbolic logical-qubit estimate.

    Raises:
        ValueError: If the sparse method lacks ``sparsity``.
    """
    n = n_spin_orbitals
    match method:
        case ChemistryQPEMethod.SPARSE:
            if sparsity is None:
                raise ValueError("sparsity is required for sparse QPE estimates.")
            sparsity_expr = _as_expr(sparsity, "sparsity")
            _validate_positive(sparsity_expr, "sparsity")
            return sp.simplify(n + sp.sqrt(sparsity_expr))
        case ChemistryQPEMethod.SINGLE_FACTORIZATION:
            return sp.simplify(n ** sp.Rational(3, 2))
        case (
            ChemistryQPEMethod.DOUBLE_FACTORIZATION
            | ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF
        ):
            rank_expr = (
                sp.Symbol("Xi", positive=True)
                if second_factor_rank is None
                else _as_expr(second_factor_rank, "second_factor_rank")
            )
            _validate_positive(rank_expr, "second_factor_rank")
            return sp.simplify(n * sp.sqrt(rank_expr))
        case ChemistryQPEMethod.TENSOR_HYPERCONTRACTION:
            return n
        case ChemistryQPEMethod.UNITARY_WEIGHT_CONCENTRATION:
            return n + 1
        case _:
            raise ValueError(f"Unhandled chemistry QPE method: {method}")


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


def _normalize_method(method: str | ChemistryQPEMethod) -> ChemistryQPEMethod:
    """Normalize a public method value to ``ChemistryQPEMethod``.

    Args:
        method (str | ChemistryQPEMethod): User-provided method.

    Returns:
        ChemistryQPEMethod: Normalized enum value.

    Raises:
        ValueError: If ``method`` is not a known chemistry QPE method.
    """
    try:
        return ChemistryQPEMethod(method)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ChemistryQPEMethod)
        raise ValueError(
            f"Unknown chemistry QPE method {method!r}; valid: {valid}."
        ) from exc


def _as_expr(value: _CoefficientLike, name: str) -> sp.Expr:
    """Convert a numeric or symbolic value to a SymPy expression.

    Args:
        value (sp.Expr | int | float | complex): Value to convert.
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


def _abs_as_expr(value: _CoefficientLike) -> sp.Expr:
    """Return the symbolic absolute value of a numeric coefficient.

    Args:
        value (sp.Expr | int | float | complex): Coefficient to convert.

    Returns:
        sp.Expr: Absolute value represented as a SymPy expression.

    Raises:
        TypeError: If ``value`` cannot be converted into a SymPy expression.
    """
    expr = _as_expr(value, "coefficient")
    if expr.is_number:
        return sp.Abs(expr)
    return sp.Abs(expr)


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
