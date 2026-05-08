"""qBraid backend executor for Qamomile.

This module provides QBraidExecutor, which bridges Qamomile's compiled
Qiskit circuits to qBraid-supported quantum devices via the qBraid runtime.

Example:
    from qamomile.qbraid import QBraidExecutor

    executor = QBraidExecutor(device_id="qbraid_qir_simulator")
    counts = executor.execute(circuit, shots=1000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.executable import ParameterMetadata, QuantumExecutor

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    import qamomile.observable as qm_o


class QBraidExecutor(QuantumExecutor["QuantumCircuit"]):
    """Quantum executor that runs Qiskit circuits on qBraid-supported devices.

    This executor implements the ``QuantumExecutor[QuantumCircuit]`` contract,
    allowing ``ExecutableProgram.sample()``, measured ``run()``, and
    expectation-value ``run()`` to work with any qBraid-accessible backend.

    The ``estimate()`` method uses a counts-based measurement approach. It
    only supports circuits with ``num_clbits == 0`` (no pre-existing classical
    bits). Circuits with existing classical bits are rejected with an
    ``ExecutionError`` to prevent silent wrong results caused by qBraid's
    counts normalization removing register separators.

    Endian convention:
        ``execute()`` returns canonical Qiskit-style bitstring keys after
        normalization. Keys are big-endian classical-bit strings: the
        leftmost character is the highest classical-bit index and the
        rightmost character is classical bit 0. ``QBraidExecutor`` never
        reverses or permutes the bit order; normalization only removes spaces
        and zero-pads under-width keys.

        ``estimate()`` uses the same bitstring convention when reconstructing
        expectation values from counts. Since it applies ``measure_all()`` to a
        circuit with no pre-existing classical bits, classical bit ``i``
        measures qubit ``i``. Therefore, in ``estimate()`` the rightmost count
        character corresponds to qubit 0 and the leftmost character to the
        highest qubit index.

    Args:
        device: A pre-configured qBraid ``QuantumDevice``. Mutually exclusive
            with ``device_id``, ``provider``, and ``api_key``.
        device_id: qBraid device identifier (e.g., ``"qbraid_qir_simulator"``).
        provider: A ``QbraidProvider`` instance for device lookup.
        api_key: qBraid API key, used to create a ``QbraidProvider`` when
            ``provider`` is not given.
        expval_shots: Number of shots for each basis-rotation circuit in
            ``estimate()``. Defaults to 4096.
        timeout: Timeout in seconds for ``wait_for_final_state()``.
            ``None`` means wait indefinitely.
        poll_interval: Polling interval in seconds for
            ``wait_for_final_state()``. Defaults to 5.
        run_kwargs: Extra keyword arguments forwarded to
            ``device.run()``. Must not contain ``"shots"`` — use the
            ``shots`` parameter of ``execute()`` instead.

    Raises:
        ValueError: If constructor arguments are inconsistent (e.g., ``device``
            combined with ``device_id``).

    Example (device_id + api_key)::

        executor = QBraidExecutor(
            device_id="qbraid_qir_simulator",
            api_key="your-api-key",
        )

    Example (pre-configured device)::

        from qbraid import QbraidProvider
        provider = QbraidProvider(api_key="...")
        device = provider.get_device("qbraid_qir_simulator")
        executor = QBraidExecutor(device=device)
    """

    def __init__(
        self,
        device: Any | None = None,
        *,
        device_id: str | None = None,
        provider: Any | None = None,
        api_key: str | None = None,
        expval_shots: int = 4096,
        timeout: int | None = None,
        poll_interval: int = 5,
        run_kwargs: dict[str, Any] | None = None,
    ):
        # Validate: device is mutually exclusive with device_id/provider/api_key
        if device is not None:
            if device_id is not None or provider is not None or api_key is not None:
                raise ValueError(
                    "When 'device' is provided, 'device_id', 'provider', and "
                    "'api_key' must not be specified. Pass either a pre-configured "
                    "device or use device_id to resolve one."
                )
            self.device = device
        elif device_id is not None:
            self.device = self._resolve_device(device_id, provider, api_key)
        else:
            raise ValueError(
                "Either 'device' or 'device_id' must be provided to "
                "identify the target quantum device."
            )

        self.expval_shots = expval_shots
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.run_kwargs = dict(run_kwargs) if run_kwargs else {}
        self._validate_run_kwargs(self.run_kwargs)

    _RESERVED_RUN_KWARGS = frozenset({"shots"})

    @staticmethod
    def _validate_run_kwargs(kwargs: dict[str, Any]) -> None:
        """Validate that run_kwargs does not contain reserved keys.

        ``shots`` is managed by the executor's ``execute()`` and
        ``estimate()`` methods and must not be passed via ``run_kwargs``.

        Args:
            kwargs: The run_kwargs dict to validate.

        Raises:
            ValueError: If a reserved key is found.
        """
        reserved = QBraidExecutor._RESERVED_RUN_KWARGS & kwargs.keys()
        if reserved:
            raise ValueError(
                f"run_kwargs contains reserved key(s): {sorted(reserved)}. "
                f"'shots' must be passed via execute() or estimate(), "
                f"not through run_kwargs."
            )

    @staticmethod
    def _resolve_device(
        device_id: str,
        provider: Any | None,
        api_key: str | None,
    ) -> Any:
        """Resolve a qBraid device from the given identifiers.

        Args:
            device_id: qBraid device identifier.
            provider: Optional pre-configured ``QbraidProvider``.
            api_key: Optional API key for creating a new ``QbraidProvider``.

        Returns:
            A qBraid ``QuantumDevice``.
        """
        if provider is not None:
            return provider.get_device(device_id)

        from qbraid import (
            QbraidProvider as _QbraidProvider,  # type: ignore[attr-defined]
        )

        if api_key is not None:
            p = _QbraidProvider(api_key=api_key)
        else:
            p = _QbraidProvider()
        return p.get_device(device_id)

    def _submit_and_wait(
        self,
        circuit: "QuantumCircuit",
        shots: int,
    ) -> dict[str, int]:
        """Submit a circuit, wait for completion, and return counts.

        This is the shared wait helper used by both ``execute()`` and
        ``estimate()``. It enforces the contract that
        ``wait_for_final_state()`` is called with ``poll_interval`` before
        ``result()`` is invoked.

        Args:
            circuit: The quantum circuit to run.
            shots: Number of measurement shots.

        Returns:
            Bitstring counts from the job result.

        Raises:
            ValueError: If ``run_kwargs`` was mutated after construction
                to include reserved keys.
        """
        self._validate_run_kwargs(self.run_kwargs)
        job = self.device.run(circuit, shots=shots, **self.run_kwargs)
        job.wait_for_final_state(
            timeout=self.timeout,
            poll_interval=self.poll_interval,
        )
        result = job.result()
        raw_counts = result.data.get_counts()
        return self._normalize_counts(raw_counts, circuit.num_clbits)

    @staticmethod
    def _normalize_counts(
        raw_counts: dict[str, int], num_clbits: int
    ) -> dict[str, int]:
        """Normalize qBraid raw counts to canonical bitstrings.

        qBraid backends may return under-width keys (leading zeros stripped)
        or space-separated register keys. This method flattens spaces and
        zero-pads short keys to match the expected classical bit width.
        It does not reverse or permute bit order.

        Args:
            raw_counts: Raw counts from the qBraid job result.
            num_clbits: Expected number of classical bits in the circuit.

        Returns:
            Normalized counts with canonical big-endian classical-bit keys.

        Raises:
            ExecutionError: If a key contains non-binary characters after
                space removal.
        """
        normalized: dict[str, int] = {}
        for key, count in raw_counts.items():
            flat = key.replace(" ", "")
            if not flat:
                raise ExecutionError(
                    f"qBraid returned an empty bitstring key (raw: {key!r})."
                )
            if not all(c in "01" for c in flat):
                raise ExecutionError(
                    f"qBraid returned a non-binary bitstring key: {key!r}."
                )
            if len(flat) < num_clbits:
                flat = flat.zfill(num_clbits)
            normalized[flat] = normalized.get(flat, 0) + count
        return normalized

    def execute(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        If the circuit has no classical bits, ``measure_all()`` is added
        automatically (on a copy).

        Returned keys use canonical Qiskit-style big-endian classical-bit
        order: the leftmost character is the highest classical-bit index and
        the rightmost character is classical bit 0. ``execute()`` does not
        reinterpret those keys as qubit-ordered strings.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.

        Returns:
            Dictionary mapping canonical big-endian classical-bit strings
            to counts.
        """
        circuit_with_meas = self._ensure_measurements(circuit)
        return self._submit_and_wait(circuit_with_meas, shots)

    def bind_parameters(
        self,
        circuit: "QuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "QuantumCircuit":
        """Bind parameter values to the Qiskit circuit.

        Uses the same indexed-binding semantics as ``QiskitExecutor``.

        Args:
            circuit: The parameterized circuit.
            bindings: Dict mapping parameter names to values.
            parameter_metadata: Metadata about circuit parameters.

        Returns:
            New circuit with parameters bound.
        """
        qiskit_bindings = {}
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                qiskit_bindings[param_info.backend_param] = bindings[param_info.name]
        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: "QuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of a Hamiltonian using counts.

        This method decomposes the Hamiltonian into Pauli terms, groups them
        by measurement basis, applies basis-rotation gates, and reconstructs
        the expectation value from measurement counts.

        Only circuits with ``num_clbits == 0`` are supported. Circuits with
        pre-existing classical bits are rejected because qBraid's counts
        normalization removes register separators, making it impossible to
        reliably identify which measured bits correspond to which qubits.

        Count bitstrings are interpreted using the same big-endian convention
        as Qiskit counts: the leftmost character is the highest measured qubit
        index and the rightmost character is qubit 0. This follows directly
        from calling ``measure_all()`` on a circuit whose classical-bit indices
        match its qubit indices.

        Args:
            circuit: The state-preparation circuit (no measurements).
            hamiltonian: The Hamiltonian whose expectation value is computed.
            params: Optional parameter values for parametric circuits.
                Values are bound positionally in Qiskit circuit parameter
                order.  If ``None``, the circuit must already have all
                parameters bound.

        Returns:
            The estimated real-valued expectation value.

        Raises:
            ExecutionError: If the circuit has existing classical bits,
                if the Hamiltonian references qubit indices outside the
                circuit width, if the circuit has unbound parameters
                after binding, or if the result has a non-negligible
                imaginary part.
        """
        if circuit.num_clbits > 0:
            raise ExecutionError(
                "QBraidExecutor.estimate() does not support circuits with "
                "existing classical bits (num_clbits > 0). This restriction "
                "prevents silent wrong results caused by qBraid's counts "
                "normalization removing register boundary information. "
                "Use a circuit without pre-existing measurements or classical "
                "registers for expectation value estimation."
            )

        # Bind parameters if provided (positional semantics, same as Qiskit).
        if params is not None:
            circuit = circuit.assign_parameters(list(params))

        # Reject circuits with unresolved parameters before remote submission.
        if circuit.parameters:
            raise ExecutionError(
                f"Circuit has {len(circuit.parameters)} unbound parameter(s) "
                f"({', '.join(p.name for p in circuit.parameters)}). "
                "Provide values via the `params` argument or call "
                "`circuit.assign_parameters(...)` before `estimate()`."
            )

        # Validate Hamiltonian qubit indices are within circuit width.
        num_qubits = circuit.num_qubits
        max_idx = -1
        for operators, _ in hamiltonian:
            for op in operators:
                if op.index > max_idx:
                    max_idx = op.index
        if max_idx >= num_qubits:
            raise ExecutionError(
                f"Hamiltonian references qubit index {max_idx} but circuit "
                f"has only {num_qubits} qubit(s)."
            )

        from qamomile.observable import Pauli

        # Group Pauli terms by compatible measurement basis assignment.
        # Terms that only differ by identities can share one measurement run.
        basis_groups: list[
            tuple[dict[int, Pauli], list[tuple[tuple[Any, ...], complex]]]
        ] = []

        for operators, coeff in hamiltonian:
            basis_assignment = {
                op.index: op.pauli for op in operators if op.pauli != Pauli.I
            }

            for group_basis, group_terms in basis_groups:
                if all(
                    group_basis.get(qubit_idx, pauli_type) == pauli_type
                    for qubit_idx, pauli_type in basis_assignment.items()
                ):
                    group_basis.update(basis_assignment)
                    group_terms.append((operators, coeff))
                    break
            else:
                basis_groups.append((dict(basis_assignment), [(operators, coeff)]))

        total_expval: complex = hamiltonian.constant

        for basis_assignment, terms in basis_groups:
            # Build the rotated circuit
            rotated = circuit.copy()

            for qubit_idx, pauli_type in sorted(basis_assignment.items()):
                if pauli_type == Pauli.X:
                    rotated.h(qubit_idx)
                elif pauli_type == Pauli.Y:
                    rotated.sdg(qubit_idx)
                    rotated.h(qubit_idx)
                # Z and I need no rotation

            rotated.measure_all()

            counts = self._submit_and_wait(rotated, self.expval_shots)

            # Compute parity expectation for each term in this group
            total_shots = sum(counts.values())

            for operators, coeff in terms:
                # Determine which qubits contribute to parity for this term
                parity_qubits = [op.index for op in operators if op.pauli != Pauli.I]

                if not parity_qubits:
                    # Pure identity term (should have been absorbed into constant)
                    total_expval += coeff
                    continue

                parity_sum = 0.0
                for bitstring, count in counts.items():
                    # bitstring is big-endian: leftmost bit = highest qubit index
                    bits = bitstring.replace(" ", "")
                    n_bits = len(bits)
                    parity = 0
                    for q in parity_qubits:
                        # big-endian: bit for qubit q is at position (n_bits - 1 - q)
                        bit_pos = n_bits - 1 - q
                        if 0 <= bit_pos < n_bits:
                            parity ^= int(bits[bit_pos])
                    # parity 0 -> +1, parity 1 -> -1
                    parity_sum += ((-1) ** parity) * count

                term_expval = parity_sum / total_shots
                total_expval += coeff * term_expval

        # Check imaginary part
        if abs(total_expval.imag) > 1e-6:
            raise ExecutionError(
                f"Expectation value has non-negligible imaginary part: "
                f"{total_expval.imag:.6e}. This indicates an error in the "
                f"Hamiltonian or circuit."
            )

        return float(total_expval.real)

    @staticmethod
    def _ensure_measurements(circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Ensure circuit has measurements, adding measure_all if needed."""
        if circuit.num_clbits > 0:
            return circuit
        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        return circuit_copy
