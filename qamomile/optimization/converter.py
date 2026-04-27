import abc

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinaryModel, BinarySampleSet, VarType


class MathematicalProblemConverter(abc.ABC):
    def __init__(
        self,
        instance: ommx.v1.Instance | BinaryModel,
    ) -> None:
        if isinstance(instance, BinaryModel):
            self.instance = None
            self.original_vartype = instance.vartype
            self.spin_model = instance.change_vartype(VarType.SPIN)
        elif isinstance(instance, ommx.v1.Instance):
            # Deep-copy via bytes round-trip before to_qubo: to_qubo mutates
            # the instance it is called on (appends slack decision variables
            # for non-binary vars, absorbs constraints into the objective via
            # the penalty method). Mutating the caller's instance silently is
            # surprising; copy first so the caller keeps an untouched view.
            # The deep copy retains original-constraint metadata internally,
            # so evaluate_samples on it still reports feasibility against the
            # user's original constraints.
            self.instance = ommx.v1.Instance.from_bytes(instance.to_bytes())
            self.original_vartype = VarType.BINARY  # OMMX uses BINARY
            qubo, constant = self.instance.to_qubo()
            self.spin_model = BinaryModel.from_qubo(qubo, constant).change_vartype(
                VarType.SPIN
            )
        else:
            raise TypeError("instance must be ommx.v1.Instance or BinaryModel")

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
            ommx_samples = binary_sampleset.to_ommx_samples()
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
