from __future__ import annotations

import abc

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet, VarType


def normalize_problem_input(
    instance: ommx.v1.Instance | BinaryModel,
    *,
    uniform_penalty_weight: float | None = None,
    penalty_weights: dict[int, float] | None = None,
) -> tuple[
    ommx.v1.Instance | None,
    ommx.v1.Instance | None,
    VarType,
    BinaryModel,
]:
    """Normalize a problem into evaluation, transformed, and spin forms.

    Shared canonical entry point used by every converter that consumes a
    combinatorial optimization problem expressed either as an OMMX
    :class:`ommx.v1.Instance` or a Qamomile :class:`BinaryModel`. The
    :class:`MathematicalProblemConverter` base class and
    :class:`~qamomile.optimization.pce.PCEConverter` (which deliberately
    does not inherit from that base) both delegate to this helper so the
    OMMX deep-copy / ``to_hubo`` semantics live in exactly one place.

    For OMMX inputs, the instance is deep-copied via a bytes round-trip
    before ``to_hubo`` is called: ``to_hubo`` mutates the instance it is
    called on (it appends slack decision variables for non-binary vars
    and absorbs constraints into the objective via the penalty method).
    Mutating the caller's instance silently is surprising; copying first
    leaves the caller's view untouched. Separate original and transformed
    copies let downstream decoding reconstruct substituted variables with the
    transformed instance, then evaluate the untouched objective and
    constraints with the original instance.

    ``to_hubo`` is used instead of ``to_qubo`` so that both purely
    quadratic (QUBO) and higher-order (HUBO) instances are handled
    uniformly. For quadratic-only instances the two paths produce
    equivalent optimization problems (same coefficients and vartype).

    Args:
        instance (ommx.v1.Instance | BinaryModel): The combinatorial
            optimization problem. ``ommx.v1.Instance`` inputs are
            deep-copied and converted via ``to_hubo()``; ``BinaryModel``
            inputs retain their declared vartype as the target output
            vartype.
        uniform_penalty_weight (float | None): Uniform weight passed to
            :meth:`ommx.v1.Instance.to_hubo`. ``None`` delegates weight
            selection to OMMX. Ignored for :class:`BinaryModel` inputs only
            when left at its default.
        penalty_weights (dict[int, float] | None): Per-constraint penalty
            weights passed to :meth:`ommx.v1.Instance.to_hubo`. Keys are
            constraint IDs. Defaults to no explicit overrides.

    Returns:
        tuple[ommx.v1.Instance | None, ommx.v1.Instance | None, VarType,
        BinaryModel]: A tuple ``(original_instance, transformed_instance,
        original_vartype, spin_model)``. The two instances are ``None`` for a
        :class:`BinaryModel` input. The original copy is reserved for
        unpenalized result evaluation; the transformed copy owns the binary
        substitutions and penalty metadata needed to reconstruct samples.

    Raises:
        TypeError: If ``instance`` is neither an :class:`ommx.v1.Instance`
            nor a :class:`BinaryModel`.
        ValueError: If penalty options are supplied for a
            :class:`BinaryModel`, which has no OMMX constraints to absorb.
    """
    if isinstance(instance, BinaryModel):
        if uniform_penalty_weight is not None or penalty_weights:
            raise ValueError(
                "Penalty weights can only be used with an ommx.v1.Instance"
            )
        return (
            None,
            None,
            instance.vartype,
            instance.change_vartype(VarType.SPIN),
        )
    if isinstance(instance, ommx.v1.Instance):
        original = ommx.v1.Instance.from_bytes(instance.to_bytes())
        transformed = ommx.v1.Instance.from_bytes(instance.to_bytes())
        hubo, constant = transformed.to_hubo(
            uniform_penalty_weight=uniform_penalty_weight,
            penalty_weights=dict(penalty_weights or {}),
        )
        spin_model = BinaryModel.from_hubo(hubo, constant).change_vartype(VarType.SPIN)
        return original, transformed, VarType.BINARY, spin_model
    raise TypeError("instance must be ommx.v1.Instance or BinaryModel")


def evaluate_original_instance(
    original_instance: ommx.v1.Instance,
    transformed_instance: ommx.v1.Instance,
    samples: ommx.v1.Samples,
) -> ommx.v1.SampleSet:
    """Evaluate transformed binary samples against the original problem.

    The transformed instance first reconstructs substituted integer and slack
    variables. Only the caller's original decision-variable IDs are then
    copied into a new sample container and evaluated against the untouched
    original instance. This keeps feasibility and objective values free of
    penalty terms while retaining OMMX's binary-substitution semantics.

    Args:
        original_instance (ommx.v1.Instance): Untouched problem copy used for
            objective and constraint evaluation.
        transformed_instance (ommx.v1.Instance): Post-``to_hubo`` problem copy
            used to reconstruct original decision-variable values.
        samples (ommx.v1.Samples): Binary samples in the transformed problem's
            variable space.

    Returns:
        ommx.v1.SampleSet: Samples evaluated against the original objective
        and constraints.
    """
    reconstructed = transformed_instance.evaluate_samples(samples)
    original_samples = ommx.v1.Samples({})
    original_ids = [variable.id for variable in original_instance.decision_variables]
    for sample_id in reconstructed.sample_ids:
        solution = reconstructed.get(sample_id)
        values = solution.decision_variables_df["value"]
        state = ommx.v1.State(
            {
                variable_id: float(values.loc[variable_id])
                for variable_id in original_ids
            }
        )
        original_samples.append([sample_id], state)
    return original_instance.evaluate_samples(original_samples)


def binary_sampleset_to_ommx_samples(
    binary_sampleset: BinarySampleSet,
) -> ommx.v1.Samples:
    """Convert a BINARY sample set into an OMMX ``Samples`` container.

    Each unique sample state is appended once with a list of sample IDs of
    length ``num_occurrences``, so OMMX-side aggregation reflects the
    original shot counts without duplicating the underlying state. States
    with ``num_occurrences == 0`` are skipped.

    This is the canonical bridge from Qamomile's
    :class:`BinarySampleSet` to OMMX's
    :class:`ommx.v1.Samples`. It is the helper that
    :meth:`MathematicalProblemConverter.decode` and
    :meth:`~qamomile.optimization.pce.PCEConverter.decode` use to feed
    samples into :meth:`ommx.v1.Instance.evaluate_samples` for feasibility
    and original-objective evaluation.

    Args:
        binary_sampleset (BinarySampleSet): A sample set with
            ``vartype=VarType.BINARY``. SPIN sample sets must be converted
            to BINARY first because OMMX expects 0/1 decision-variable
            values.

    Returns:
        ommx.v1.Samples: An OMMX Samples object with
        ``sum(num_occurrences)`` sample IDs, where IDs sharing the same
        state are grouped together. The returned object is empty when the
        input has no samples.

    Raises:
        ValueError: If ``binary_sampleset.vartype`` is not
            ``VarType.BINARY``. SPIN sample sets must be converted first.
        ValueError: If the lengths of ``binary_sampleset.samples`` and
            ``binary_sampleset.num_occurrences`` are inconsistent (and,
            when present, ``binary_sampleset.energy`` as well).
    """
    # Compare via the enum's str value: bound TypeVar VT can be narrowed
    # to the default (BINARY) by static type-checkers, which then flag
    # the guard's non-BINARY branch as unreachable. The str(...) cast
    # bypasses that narrowing without changing runtime semantics.
    if str(binary_sampleset.vartype) != str(VarType.BINARY):
        raise ValueError(
            "binary_sampleset_to_ommx_samples requires vartype=BINARY; "
            f"got {binary_sampleset.vartype}. Convert to BINARY first."
        )

    n_samples = len(binary_sampleset.samples)
    n_occ = len(binary_sampleset.num_occurrences)
    if n_samples != n_occ:
        raise ValueError(
            "binary_sampleset.samples and binary_sampleset.num_occurrences "
            f"must have the same length; got {n_samples} samples and "
            f"{n_occ} num_occurrences."
        )
    n_energies = len(binary_sampleset.energy)
    if n_samples != n_energies:
        raise ValueError(
            "binary_sampleset.samples and binary_sampleset.energy "
            f"must have the same length; got {n_samples} samples and "
            f"{n_energies} energies."
        )

    ommx_samples = ommx.v1.Samples({})
    next_id = 0
    for sample, occ in zip(binary_sampleset.samples, binary_sampleset.num_occurrences):
        if occ <= 0:
            continue
        sample_ids = list(range(next_id, next_id + occ))
        next_id += occ
        state = ommx.v1.State({idx: float(val) for idx, val in sample.items()})
        ommx_samples.append(sample_ids, state)
    return ommx_samples


class MathematicalProblemConverter(abc.ABC):
    def __init__(
        self,
        instance: ommx.v1.Instance | BinaryModel,
        *,
        uniform_penalty_weight: float | None = None,
        penalty_weights: dict[int, float] | None = None,
    ) -> None:
        """Initialize a converter from an OMMX instance or binary model.

        Args:
            instance (ommx.v1.Instance | BinaryModel): Optimization problem.
            uniform_penalty_weight (float | None): Uniform constraint penalty
                passed to OMMX. ``None`` delegates selection to OMMX.
            penalty_weights (dict[int, float] | None): Optional per-constraint
                penalty weights keyed by constraint ID.
        """
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
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the cost Hamiltonian.

        Subclasses must implement this method to build the appropriate
        Hamiltonian for their specific algorithm (e.g., Pauli-Z for QAOA,
        QRAC-encoded for QRAO). Oracle-based converters that do not use a cost
        Hamiltonian (e.g., ``GASConverter``) should raise ``NotImplementedError``.

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.

        Raises:
            NotImplementedError: If the converter does not expose a cost
                Hamiltonian.
        """
        ...

    def decode(
        self,
        samples: SampleResult[list[int]],
    ) -> BinarySampleSet | ommx.v1.SampleSet:
        """Decode quantum measurement results.

        The return type tracks the input that built this converter:

        * Built from an :class:`ommx.v1.Instance` — returns an
          :class:`ommx.v1.SampleSet` evaluated against the original
          (un-penalized) instance, so feasibility, objective, and
          per-constraint violations are available through OMMX's own
          API (``.summary``, ``.summary_with_constraints``,
          ``.best_feasible``, ``.feasible``, ``.objectives``).
        * Built from a :class:`BinaryModel` — returns a
          :class:`BinarySampleSet` with samples in the model's original
          vartype (BINARY 0/1 or SPIN ±1), energies, and shot counts.

        .. note::
           For OMMX-backed converters, ``Instance.to_qubo`` was applied
           to a deep copy of the caller's instance during
           construction — it mutates the instance it is called on
           (penalty form, slack decision variables for non-binary vars).
           The stored copy retains original-constraint metadata
           internally, so feasibility on the returned SampleSet is
           reported against the user's *original* constraints, not the
           absorbed-penalty form. Slack bits added by ``to_qubo`` are
           reconstructed back into the original decision variables
           (e.g., integers rebuilt from log-encoded binary slack bits)
           by ``evaluate_samples`` automatically.

        Args:
            samples: Raw quantum measurement results from
                ``ExecutableProgram.sample(...).result()``.

        Returns:
            BinarySampleSet | ommx.v1.SampleSet: see method description.

        See Also:
            :meth:`decode_to_binary_sampleset`: always returns a
            :class:`BinarySampleSet`. Use it when you need the
            QUBO-domain (penalty-included) ``energy`` — e.g. to drive a
            classical optimizer that must penalize infeasibility.

        Example:
            >>> # OMMX in → OMMX out
            >>> converter = QAOAConverter(ommx_instance)
            >>> exe = converter.transpile(QiskitTranspiler(), p=2)
            >>> result = exe.sample(QiskitTranspiler().executor(),
            ...                     shots=1024,
            ...                     bindings={"gammas": gs, "betas": bs}).result()
            >>> sample_set = converter.decode(result)
            >>> sample_set.best_feasible.objective
        """
        binary_sampleset = self.decode_to_binary_sampleset(samples)
        if self.instance is not None:
            assert self.original_instance is not None
            ommx_samples = binary_sampleset_to_ommx_samples(binary_sampleset)
            return evaluate_original_instance(
                self.original_instance,
                self.instance,
                ommx_samples,
            )
        return binary_sampleset

    def decode_to_binary_sampleset(
        self,
        samples: SampleResult[list[int]],
    ) -> BinarySampleSet:
        """Decode samples into a :class:`BinarySampleSet`.

        Always returns a :class:`BinarySampleSet`, regardless of whether
        this converter was constructed with an :class:`ommx.v1.Instance`
        or a :class:`BinaryModel`. Use this when you need:

        * The QUBO-domain ``energy`` (penalty-included), e.g. as the cost
          driving a classical optimizer like COBYLA — :meth:`decode` on
          OMMX-backed converters returns the *un-penalized* OMMX
          objective which won't penalize infeasibility.
        * The per-state ``samples`` / ``num_occurrences`` /
          ``vartype`` views from :class:`BinarySampleSet`.

        For most usage — feasibility, original-objective evaluation,
        per-constraint diagnostics — prefer the polymorphic
        :meth:`decode`, which returns an :class:`ommx.v1.SampleSet` for
        OMMX-backed converters.

        Args:
            samples: Raw quantum measurement results from
                ``ExecutableProgram.sample(...).result()``.

        Returns:
            BinarySampleSet: keyed by the SPIN model's original variable
            indices (the QUBO variable IDs for OMMX-backed converters)
            in the converter's ``original_vartype`` — BINARY for
            OMMX-backed converters, the :class:`BinaryModel`'s declared
            vartype otherwise.
        """
        # First decode in SPIN domain
        spin_sampleset = self.spin_model.decode_from_sampleresult(samples)

        # If original problem was BINARY, convert back
        if self.original_vartype == VarType.BINARY:
            binary_samples = []
            for spin_sample in spin_sampleset.samples:
                # Convert SPIN (+/-1) to BINARY (0/1): x = (1 - s) / 2
                binary_sample = {
                    idx: (1 - spin_val) // 2 for idx, spin_val in spin_sample.items()
                }
                binary_samples.append(binary_sample)

            return BinarySampleSet(
                samples=binary_samples,
                num_occurrences=spin_sampleset.num_occurrences,
                energy=spin_sampleset.energy,
                vartype=VarType.BINARY,
            )
        else:
            # Already in SPIN, return as-is
            return spin_sampleset
