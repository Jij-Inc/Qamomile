"""Fault-tolerant resource estimates for quantum chemistry algorithms."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

_SympyLike = sp.Expr | int | float


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
class FTQCCostModel:
    """Describe an architecture-level FTQC cost model.

    Attributes:
        physical_qubits_per_logical (sp.Expr | int | float): Physical qubit
            overhead for one data or ancilla logical qubit.
        logical_cycle_time_seconds (sp.Expr | int | float): Time for one
            logical layer or logical cycle, in seconds.
        factory_qubits (sp.Expr | int | float): Physical qubits reserved for
            magic-state factories or equivalent non-Clifford support.
        toffoli_throughput_per_second (sp.Expr | int | float): Sustainable
            Toffoli or non-Clifford-equivalent throughput. Runtime uses the
            larger of logical-depth time and factory-throughput time.

    Raises:
        ValueError: If any positive-valued field is non-positive or if
            ``factory_qubits`` is negative.

    Example:
        >>> model = FTQCCostModel(
        ...     physical_qubits_per_logical=1000,
        ...     logical_cycle_time_seconds=1e-6,
        ...     factory_qubits=20000,
        ...     toffoli_throughput_per_second=1e5,
        ... )
        >>> model.physical_qubits_per_logical
        1000
    """

    physical_qubits_per_logical: _SympyLike = field(
        default_factory=lambda: sp.Symbol(
            "physical_qubits_per_logical",
            positive=True,
        )
    )
    logical_cycle_time_seconds: _SympyLike = field(
        default_factory=lambda: sp.Symbol("logical_cycle_time", positive=True)
    )
    factory_qubits: _SympyLike = field(
        default_factory=lambda: sp.Symbol("factory_qubits", nonnegative=True)
    )
    toffoli_throughput_per_second: _SympyLike = field(
        default_factory=lambda: sp.Symbol("toffoli_throughput", positive=True)
    )

    def __post_init__(self) -> None:
        """Validate cost-model fields after dataclass construction.

        Raises:
            ValueError: If any positive-valued field is non-positive or if
                ``factory_qubits`` is negative.
        """
        _validate_positive(
            _as_expr(
                self.physical_qubits_per_logical,
                "physical_qubits_per_logical",
            ),
            "physical_qubits_per_logical",
        )
        _validate_positive(
            _as_expr(
                self.logical_cycle_time_seconds,
                "logical_cycle_time_seconds",
            ),
            "logical_cycle_time_seconds",
        )
        _validate_nonnegative(
            _as_expr(self.factory_qubits, "factory_qubits"),
            "factory_qubits",
        )
        _validate_positive(
            _as_expr(
                self.toffoli_throughput_per_second,
                "toffoli_throughput_per_second",
            ),
            "toffoli_throughput_per_second",
        )


@dataclass(frozen=True)
class FTQCResourceEstimate:
    """Represent algorithm-level FTQC resource estimates.

    Attributes:
        algorithm (str): Human-readable algorithm or representation name.
        logical_qubits (sp.Expr): Logical qubits required by the algorithm.
        physical_qubits (sp.Expr): Physical qubits under the selected
            architecture model.
        toffoli_gates (sp.Expr): Toffoli gate count or Toffoli-equivalent
            non-Clifford count.
        t_gates (sp.Expr): T gate count when it is distinct from Toffoli
            count. Defaults to zero for Toffoli-native estimates.
        clifford_gates (sp.Expr): Clifford gate estimate when available.
        qpe_iterations (sp.Expr): Number of phase-estimation walk or
            time-evolution calls.
        logical_depth (sp.Expr): Logical circuit depth proxy.
        runtime_seconds (sp.Expr): Runtime estimate in seconds.
        parameters (dict[str, sp.Symbol]): Free symbols appearing in the
            estimate, keyed by display name.
        assumptions (dict[str, str]): Reader-facing notes about model choices.

    Example:
        >>> n, lam, eps, walk = sp.symbols("n lambda eps C_W", positive=True)
        >>> est = estimate_qubitized_chemistry_qpe(n, lam, eps, walk)
        >>> est.qpe_iterations
        lambda/eps
    """

    algorithm: str
    logical_qubits: sp.Expr
    physical_qubits: sp.Expr
    toffoli_gates: sp.Expr
    t_gates: sp.Expr
    clifford_gates: sp.Expr
    qpe_iterations: sp.Expr
    logical_depth: sp.Expr
    runtime_seconds: sp.Expr
    parameters: dict[str, sp.Symbol] = field(default_factory=dict)
    assumptions: dict[str, str] = field(default_factory=dict)

    def substitute(self, **values: int | float) -> FTQCResourceEstimate:
        """Substitute concrete values into all symbolic fields.

        Args:
            **values (int | float): Mapping from symbol name to concrete
                value. Unknown names are accepted and converted into SymPy
                symbols so callers can substitute ad hoc expressions.

        Returns:
            FTQCResourceEstimate: New estimate with substitutions applied.
        """
        substitutions: dict[Any, Any] = {}
        for name, value in values.items():
            substitutions[self.parameters.get(name, sp.Symbol(name))] = value

        return self._map_exprs(lambda expr: expr.subs(substitutions).doit())

    def simplify(self) -> FTQCResourceEstimate:
        """Simplify every symbolic field.

        Returns:
            FTQCResourceEstimate: New estimate with simplified expressions.
        """
        return self._map_exprs(sp.simplify)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the estimate to string-valued dictionaries.

        Returns:
            dict[str, Any]: JSON-friendly dictionary containing all resource
                fields, parameters, and assumptions.
        """
        return {
            "algorithm": self.algorithm,
            "logical_qubits": str(self.logical_qubits),
            "physical_qubits": str(self.physical_qubits),
            "toffoli_gates": str(self.toffoli_gates),
            "t_gates": str(self.t_gates),
            "clifford_gates": str(self.clifford_gates),
            "qpe_iterations": str(self.qpe_iterations),
            "logical_depth": str(self.logical_depth),
            "runtime_seconds": str(self.runtime_seconds),
            "parameters": {
                name: str(symbol) for name, symbol in self.parameters.items()
            },
            "assumptions": dict(self.assumptions),
        }

    def _map_exprs(self, fn: Callable[[sp.Expr], sp.Expr]) -> FTQCResourceEstimate:
        """Apply a function to each symbolic resource field.

        Args:
            fn (Callable[[sp.Expr], sp.Expr]): Callable accepting and
                returning a SymPy expression.

        Returns:
            FTQCResourceEstimate: New estimate with mapped expressions.
        """
        return FTQCResourceEstimate(
            algorithm=self.algorithm,
            logical_qubits=fn(self.logical_qubits),
            physical_qubits=fn(self.physical_qubits),
            toffoli_gates=fn(self.toffoli_gates),
            t_gates=fn(self.t_gates),
            clifford_gates=fn(self.clifford_gates),
            qpe_iterations=fn(self.qpe_iterations),
            logical_depth=fn(self.logical_depth),
            runtime_seconds=fn(self.runtime_seconds),
            parameters=self.parameters,
            assumptions=self.assumptions,
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
    cost_model: FTQCCostModel | None = None,
) -> FTQCResourceEstimate:
    """Estimate qubitized QPE resources for molecular Hamiltonians.

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
        cost_model (FTQCCostModel | None): Architecture model used to lift
            logical estimates to physical qubits and runtime. Defaults to a
            symbolic model.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If a finite-set method is unknown, a required sparse
            parameter is missing, or a positive-valued input is non-positive.
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

    model = cost_model or FTQCCostModel()
    qpe_iterations = sp.simplify(lambda_expr / precision_expr)
    toffoli_gates = sp.simplify(qpe_iterations * walk_expr)
    logical_depth = toffoli_gates
    physical_qubits = sp.simplify(
        logical_expr * model.physical_qubits_per_logical + model.factory_qubits
    )
    runtime_seconds = sp.simplify(
        sp.Max(
            logical_depth * model.logical_cycle_time_seconds,
            toffoli_gates / model.toffoli_throughput_per_second,
        )
    )
    assumptions = {
        "qpe_iterations": "Uses lambda_norm / precision as the walk-call proxy.",
        "method": method_enum.value,
        "walk_cost_toffoli": (
            "Caller supplies the per-walk Toffoli model so Qamomile does not "
            "bake in one chemistry factorization implementation."
        ),
    }
    return _build_estimate(
        algorithm=f"qubitized_qpe:{method_enum.value}",
        logical_qubits=logical_expr,
        physical_qubits=physical_qubits,
        toffoli_gates=toffoli_gates,
        t_gates=sp.Integer(0),
        clifford_gates=sp.Integer(0),
        qpe_iterations=qpe_iterations,
        logical_depth=logical_depth,
        runtime_seconds=runtime_seconds,
        assumptions=assumptions,
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
    cost_model: FTQCCostModel | None = None,
) -> FTQCResourceEstimate:
    """Estimate early-FTQC single-ancilla Trotter QPE resources.

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
        cost_model (FTQCCostModel | None): Architecture model used to lift
            logical estimates to physical qubits and runtime.

    Returns:
        FTQCResourceEstimate: Symbolic FTQC resource estimate.

    Raises:
        ValueError: If any positive-valued input is non-positive or either
            multiplicative reduction factor is negative.
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
    toffoli_gates = sp.Integer(0)
    model = cost_model or FTQCCostModel()
    physical_qubits = sp.simplify(
        logical_expr * model.physical_qubits_per_logical + model.factory_qubits
    )
    runtime_seconds = sp.simplify(
        sp.Max(
            logical_depth * model.logical_cycle_time_seconds,
            t_gates / model.toffoli_throughput_per_second,
        )
    )
    assumptions = {
        "effective_lambda": "lambda_norm * unitary_weight_factor.",
        "qpe_style": "Single-ancilla Hadamard-test QPE with product-formula evolution.",
        "randomization": "randomized_compilation_factor rescales Pauli-rotation work.",
    }
    return _build_estimate(
        algorithm="single_ancilla_trotter_qpe:unitary_weight_concentration",
        logical_qubits=logical_expr,
        physical_qubits=physical_qubits,
        toffoli_gates=toffoli_gates,
        t_gates=t_gates,
        clifford_gates=sp.Integer(0),
        qpe_iterations=qpe_iterations,
        logical_depth=logical_depth,
        runtime_seconds=runtime_seconds,
        assumptions=assumptions,
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


def _build_estimate(
    *,
    algorithm: str,
    logical_qubits: sp.Expr,
    physical_qubits: sp.Expr,
    toffoli_gates: sp.Expr,
    t_gates: sp.Expr,
    clifford_gates: sp.Expr,
    qpe_iterations: sp.Expr,
    logical_depth: sp.Expr,
    runtime_seconds: sp.Expr,
    assumptions: dict[str, str],
) -> FTQCResourceEstimate:
    """Create an estimate and collect its free symbolic parameters.

    Args:
        algorithm (str): Algorithm name.
        logical_qubits (sp.Expr): Logical qubit count.
        physical_qubits (sp.Expr): Physical qubit count.
        toffoli_gates (sp.Expr): Toffoli count.
        t_gates (sp.Expr): T count.
        clifford_gates (sp.Expr): Clifford count.
        qpe_iterations (sp.Expr): QPE iteration count.
        logical_depth (sp.Expr): Logical depth proxy.
        runtime_seconds (sp.Expr): Runtime estimate in seconds.
        assumptions (dict[str, str]): Notes about model assumptions.

    Returns:
        FTQCResourceEstimate: Estimate with collected parameters.
    """
    expressions = [
        logical_qubits,
        physical_qubits,
        toffoli_gates,
        t_gates,
        clifford_gates,
        qpe_iterations,
        logical_depth,
        runtime_seconds,
    ]
    symbols: set[sp.Symbol] = set()
    for expr in expressions:
        for symbol in expr.free_symbols:
            if isinstance(symbol, sp.Symbol):
                symbols.add(symbol)
    return FTQCResourceEstimate(
        algorithm=algorithm,
        logical_qubits=sp.simplify(logical_qubits),
        physical_qubits=sp.simplify(physical_qubits),
        toffoli_gates=sp.simplify(toffoli_gates),
        t_gates=sp.simplify(t_gates),
        clifford_gates=sp.simplify(clifford_gates),
        qpe_iterations=sp.simplify(qpe_iterations),
        logical_depth=sp.simplify(logical_depth),
        runtime_seconds=sp.simplify(runtime_seconds),
        parameters={str(symbol): symbol for symbol in sorted(symbols, key=str)},
        assumptions=assumptions,
    )


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
