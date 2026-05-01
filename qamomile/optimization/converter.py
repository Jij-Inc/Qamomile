import abc

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet, VarType


def normalize_problem_input(
    instance: ommx.v1.Instance | BinaryModel,
) -> tuple[ommx.v1.Instance | None, VarType, BinaryModel]:
    """Normalize a problem input into ``(stored_instance, original_vartype, spin_model)``.

    Shared canonical entry point used by every converter that consumes a
    combinatorial optimization problem expressed either as an OMMX
    :class:`ommx.v1.Instance` or a Qamomile :class:`BinaryModel`. The
    :class:`MathematicalProblemConverter` base class and
    :class:`~qamomile.optimization.pce.PCEConverter` (which deliberately
    does not inherit from that base) both delegate to this helper so the
    OMMX deep-copy / ``to_qubo`` semantics live in exactly one place.

    For OMMX inputs, the instance is deep-copied via a bytes round-trip
    before ``to_qubo`` is called: ``to_qubo`` mutates the instance it is
    called on (it appends slack decision variables for non-binary vars
    and absorbs constraints into the objective via the penalty method).
    Mutating the caller's instance silently is surprising; copying first
    leaves the caller's view untouched. The deep copy retains
    original-constraint metadata internally, so downstream
    ``evaluate_samples`` reports feasibility against the user's
    *original* constraints, and slack bits added by ``to_qubo`` are
    reconstructed back into the original decision variables (e.g.,
    integers rebuilt from log-encoded slack bits) by ``evaluate_samples``
    automatically.

    Args:
        instance (ommx.v1.Instance | BinaryModel): The combinatorial
            optimization problem. ``ommx.v1.Instance`` inputs are
            deep-copied and converted via ``to_qubo()``; ``BinaryModel``
            inputs retain their declared vartype as the target output
            vartype.

    Returns:
        tuple[ommx.v1.Instance | None, VarType, BinaryModel]: A triple
        ``(stored_instance, original_vartype, spin_model)``.

        * ``stored_instance``: the deep-copied ``Instance`` for OMMX
          inputs (post ``to_qubo``), or ``None`` for ``BinaryModel``
          inputs.
        * ``original_vartype``: the declared vartype of the input — used
          by converters' ``decode`` paths to return results in the
          caller's preferred representation.
        * ``spin_model``: the problem rewritten in SPIN form, which is
          the natural domain for sign rounding and Pauli encoding.

    Raises:
        TypeError: If ``instance`` is neither an :class:`ommx.v1.Instance`
            nor a :class:`BinaryModel`.
    """
    if isinstance(instance, BinaryModel):
        return None, instance.vartype, instance.change_vartype(VarType.SPIN)
    if isinstance(instance, ommx.v1.Instance):
        stored = ommx.v1.Instance.from_bytes(instance.to_bytes())
        qubo, constant = stored.to_qubo()
        spin_model = BinaryModel.from_qubo(qubo, constant).change_vartype(VarType.SPIN)
        return stored, VarType.BINARY, spin_model
    raise TypeError("instance must be ommx.v1.Instance or BinaryModel")


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
    ) -> None:
        self.instance, self.original_vartype, self.spin_model = normalize_problem_input(
            instance
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        """Construct the cost Hamiltonian.

        Subclasses must implement this method to build the appropriate
        Hamiltonian for their specific algorithm (e.g., Pauli-Z for QAOA,
        QRAC-encoded for QRAO).

        Returns:
            qm_o.Hamiltonian: The cost Hamiltonian.
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
            ommx_samples = binary_sampleset_to_ommx_samples(binary_sampleset)
            return self.instance.evaluate_samples(ommx_samples)
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
