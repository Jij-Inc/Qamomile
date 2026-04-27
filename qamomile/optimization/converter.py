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
            self.instance = instance
            self.original_vartype = VarType.BINARY  # OMMX uses BINARY
            qubo, constant = instance.to_qubo()
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
    ) -> BinarySampleSet:
        """Decode quantum measurement results.

        Returns results in the original vartype (BINARY or SPIN) that was
        provided when constructing the converter.
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

    def decode_to_ommx_sampleset(
        self,
        samples: SampleResult[list[int]],
    ) -> ommx.v1.SampleSet:
        """Decode quantum measurement results into an OMMX ``SampleSet``.

        Closes the OMMX round-trip: when this converter was constructed with
        an ``ommx.v1.Instance``, the returned ``SampleSet`` evaluates the
        original (un-penalized) objective and constraints against the OMMX
        instance, exposing feasibility, objective values, and per-constraint
        violations through OMMX's own API surface.

        Args:
            samples: Raw quantum measurement results from
                ``ExecutableProgram.sample(...).result()``. The bitstrings
                are decoded into the QUBO variable space and forwarded to
                ``Instance.evaluate_samples``, which performs the inverse
                mapping (including reconstruction of integer / continuous
                decision variables from slack bits) internally.

        Returns:
            ommx.v1.SampleSet: An OMMX SampleSet containing one sample ID per
            shot occurrence, with objective/feasibility evaluated against the
            original OMMX instance. Use ``.summary``,
            ``.summary_with_constraints``, ``.best_feasible``,
            ``.feasible``, and ``.objectives`` to inspect results.

        Raises:
            ValueError: If this converter was constructed from a
                ``BinaryModel`` rather than an ``ommx.v1.Instance``. In that
                case there is no OMMX instance to evaluate against; use
                :meth:`decode` to obtain a ``BinarySampleSet`` instead.

        Example:
            >>> converter = QAOAConverter(ommx_instance)
            >>> exe = converter.transpile(QiskitTranspiler(), p=2)
            >>> result = exe.sample(QiskitTranspiler().executor(),
            ...                     shots=1024,
            ...                     bindings={"gammas": gs, "betas": bs}).result()
            >>> sample_set = converter.decode_to_ommx_sampleset(result)
            >>> best = sample_set.best_feasible
            >>> best.objective, best.feasible
        """
        if self.instance is None:
            raise ValueError(
                "decode_to_ommx_sampleset requires the converter to have been "
                "constructed with an ommx.v1.Instance; this converter was built "
                "from a BinaryModel. Use decode() to obtain a BinarySampleSet "
                "instead."
            )

        binary_sampleset = self.decode(samples)
        ommx_samples = binary_sampleset.to_ommx_samples()
        return self.instance.evaluate_samples(ommx_samples)
