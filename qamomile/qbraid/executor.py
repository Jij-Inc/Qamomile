"""qBraid backend executor for Qamomile.

This module provides QBraidExecutor, which bridges Qamomile's compiled
Qiskit circuits to qBraid-supported quantum devices via the qBraid runtime.

Example:
    from qamomile.qbraid import QBraidExecutor

    executor = QBraidExecutor(device_id="qbraid_qir_simulator")
    counts = executor.execute(circuit, shots=1000)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.executable import ParameterMetadata, QuantumExecutor
from qamomile.circuit.transpiler.execution_capability import ExecutionCapabilities
from qamomile.circuit.transpiler.execution_handle import (
    CompletedExecutionHandle,
    CompositeExecutionHandle,
    ExecutionHandle,
    MappedExecutionHandle,
)
from qamomile.circuit.transpiler.execution_request import (
    CircuitInvocation,
    EstimateRequest,
    Exact,
    SampleRequest,
    ShotBased,
    TargetPrecision,
)
from qamomile.qbraid.execution import QBraidExecutionHandle

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
    ) -> None:
        """Initialize qBraid device access and execution policy.

        Args:
            device (Any | None): Pre-configured qBraid device. Mutually
                exclusive with identifier and credential arguments.
            device_id (str | None): qBraid device identifier. Defaults to
                ``None``.
            provider (Any | None): Provider used to resolve ``device_id``.
                Defaults to ``None``.
            api_key (str | None): API key used when constructing a provider.
                Defaults to ``None``.
            expval_shots (int): Positive shots per expectation basis group.
                Defaults to 4096.
            timeout (int | None): Default result wait timeout in seconds.
                Defaults to no timeout.
            poll_interval (int): Positive provider polling interval in seconds.
                Defaults to five.
            run_kwargs (dict[str, Any] | None): Additional qBraid submission
                options. Defaults to ``None``.

        Raises:
            ValueError: If device arguments conflict, required identifiers are
                missing, numeric wait options are invalid, or ``run_kwargs``
                contains an executor-owned key.
        """
        if isinstance(expval_shots, bool) or expval_shots <= 0:
            raise ValueError("expval_shots must be a positive integer")
        if isinstance(poll_interval, bool) or poll_interval <= 0:
            raise ValueError("poll_interval must be a positive integer")
        if timeout is not None and (isinstance(timeout, bool) or timeout <= 0):
            raise ValueError("timeout must be a positive integer or None")
        self._device_id = device_id
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

        self.expval_shots = int(expval_shots)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.run_kwargs = dict(run_kwargs) if run_kwargs else {}
        self._validate_run_kwargs(self.run_kwargs)

    @property
    def capabilities(self) -> ExecutionCapabilities:
        """Describe qBraid features implemented by this executor.

        Returns:
            ExecutionCapabilities: Sampling lifecycle and estimation support.
        """
        return ExecutionCapabilities(
            supports_async_sampling=True,
            supports_async_estimation=True,
            supports_estimation=True,
            supports_cancellation=True,
            estimation_accuracy=frozenset({ShotBased}),
        )

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

        try:
            from qbraid import (
                QbraidProvider as _QbraidProvider,  # type: ignore[attr-defined]
            )
        except ModuleNotFoundError as error:
            raise ImportError(
                "qBraid execution requires the optional qbraid dependencies. "
                "Install them with `pip install 'qamomile[qbraid]'` or "
                "`uv sync --extra qbraid`."
            ) from error

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
        return self._submit(circuit, shots).result()

    def _submit(
        self,
        circuit: "QuantumCircuit",
        shots: int,
    ) -> QBraidExecutionHandle:
        """Submit one qBraid sampling job without waiting for completion.

        Args:
            circuit (QuantumCircuit): Measured Qiskit circuit to submit.
            shots (int): Positive number of measurement shots.

        Returns:
            QBraidExecutionHandle: Lazy provider-backed sampling handle.

        Raises:
            ValueError: If mutable run options contain a reserved key.
            Exception: Any qBraid submission failure.
        """
        self._validate_run_kwargs(self.run_kwargs)
        job = self.device.run(circuit, shots=shots, **self.run_kwargs)

        def decode(result: Any) -> dict[str, int]:
            """Normalize counts from one qBraid result.

            Args:
                result (Any): Native qBraid result object.

            Returns:
                dict[str, int]: Canonical big-endian counts.
            """
            return self._normalize_counts(
                result.data.get_counts(),
                circuit.num_clbits,
            )

        return QBraidExecutionHandle(
            job,
            decode,
            target=self._target_id(),
            timeout=self.timeout,
            poll_interval=self.poll_interval,
        )

    def _target_id(self) -> str | None:
        """Return a stable qBraid device identifier when available.

        Returns:
            str | None: Configured or SDK-provided device identifier.
        """
        if self._device_id:
            return self._device_id
        for attribute in ("id", "device_id"):
            value = getattr(self.device, attribute, None)
            value = value() if callable(value) else value
            if isinstance(value, str) and value:
                return value
        return None

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
        request_circuit = self._ensure_measurements(circuit)
        return self._submit_and_wait(request_circuit, shots)

    def submit_sample(
        self,
        request: SampleRequest["QuantumCircuit"],
    ) -> ExecutionHandle[dict[str, int]]:
        """Submit qBraid sampling without waiting for the remote result.

        Args:
            request (SampleRequest[QuantumCircuit]): Circuit invocation and
                requested shot count.

        Returns:
            ExecutionHandle[dict[str, int]]: Lazy qBraid sampling handle.

        Raises:
            ValueError: If runtime bindings are incomplete or run options
                contain a reserved key.
            Exception: Any qBraid submission failure.
        """
        circuit = self.bind_invocation(request.invocation)
        return self._submit(self._ensure_measurements(circuit), request.shots)

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
        if params is not None:
            circuit = circuit.assign_parameters(list(params))
        invocation = CircuitInvocation(circuit, {}, ParameterMetadata())
        request = EstimateRequest(
            invocation,
            hamiltonian,
            ShotBased(self.expval_shots),
        )
        return self.submit_estimate(request).result()

    def submit_estimate(
        self,
        request: EstimateRequest["QuantumCircuit"],
    ) -> ExecutionHandle[float]:
        """Submit counts-based expectation tasks without waiting.

        Args:
            request (EstimateRequest[QuantumCircuit]): Circuit invocation,
                Hamiltonian, and optional shot policy.

        Returns:
            ExecutionHandle[float]: Lazy aggregate expectation handle.

        Raises:
            ValueError: If exact or target-precision estimation is requested.
            ExecutionError: If the circuit or Hamiltonian cannot be estimated
                safely through normalized counts.
        """
        shots = self.expval_shots
        if isinstance(request.accuracy, ShotBased):
            shots = request.accuracy.shots
        elif isinstance(request.accuracy, Exact):
            raise ValueError("qBraid counts execution does not support Exact")
        elif isinstance(request.accuracy, TargetPrecision):
            raise ValueError("qBraid counts execution does not support TargetPrecision")

        circuit = self.bind_invocation(request.invocation)
        circuits, decoder = self._prepare_estimate_tasks(
            circuit,
            request.hamiltonian,
        )
        if not circuits:
            return CompletedExecutionHandle(decoder(()))
        executions = CompositeExecutionHandle(
            tuple(self._submit(task, shots) for task in circuits)
        )
        return MappedExecutionHandle(executions, decoder)

    def _prepare_estimate_tasks(
        self,
        circuit: "QuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
    ) -> tuple[
        tuple["QuantumCircuit", ...],
        Callable[[tuple[dict[str, int], ...]], float],
    ]:
        """Build basis-rotation circuits and their aggregate decoder.

        Args:
            circuit (QuantumCircuit): Bound state-preparation circuit without
                pre-existing classical bits.
            hamiltonian (qm_o.Hamiltonian): Observable to estimate.

        Returns:
            tuple[tuple[QuantumCircuit, ...], Callable[[tuple[dict[str, int], ...]], float]]:
                Ordered measurement circuits and a callable that converts their
                counts to one expectation.

        Raises:
            ExecutionError: If the circuit has classical bits or unbound
                parameters, or the Hamiltonian exceeds the circuit width.
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

        if circuit.parameters:
            raise ExecutionError(
                f"Circuit has {len(circuit.parameters)} unbound parameter(s) "
                f"({', '.join(p.name for p in circuit.parameters)}). "
                "Provide values via the `params` argument or call "
                "`circuit.assign_parameters(...)` before `estimate()`."
            )

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

        circuits = []
        for basis_assignment, terms in basis_groups:
            rotated = circuit.copy()
            for qubit_idx, pauli_type in sorted(basis_assignment.items()):
                if pauli_type == Pauli.X:
                    rotated.h(qubit_idx)
                elif pauli_type == Pauli.Y:
                    rotated.sdg(qubit_idx)
                    rotated.h(qubit_idx)
            rotated.measure_all()
            circuits.append(rotated)

        def decode(results: tuple[dict[str, int], ...]) -> float:
            """Combine measurement-group counts into one expectation value.

            Args:
                results (tuple[dict[str, int], ...]): Counts in basis-group
                    order.

            Returns:
                float: Real Hamiltonian expectation value.

            Raises:
                ExecutionError: If results are missing, contain no shots, or
                    produce a non-negligible imaginary value.
            """
            if len(results) != len(basis_groups):
                raise ExecutionError(
                    "qBraid expectation result count does not match submitted "
                    f"basis groups: expected={len(basis_groups)}, "
                    f"actual={len(results)}"
                )
            total_expval: complex = hamiltonian.constant
            for (_, terms), counts in zip(basis_groups, results, strict=True):
                total_expval += self._decode_basis_group(counts, terms, Pauli.I)
            if abs(total_expval.imag) > 1e-6:
                raise ExecutionError(
                    "Expectation value has non-negligible imaginary part: "
                    f"{total_expval.imag:.6e}. This indicates an error in the "
                    "Hamiltonian or circuit."
                )
            return float(total_expval.real)

        return tuple(circuits), decode

    @staticmethod
    def _decode_basis_group(
        counts: dict[str, int],
        terms: list[tuple[tuple[Any, ...], complex]],
        identity: Any,
    ) -> complex:
        """Decode all compatible Pauli terms from one counts dictionary.

        Args:
            counts (dict[str, int]): Big-endian measurement counts.
            terms (list[tuple[tuple[Any, ...], complex]]): Pauli terms sharing
                the submitted measurement basis.
            identity (Any): Backend-independent identity Pauli enum value.

        Returns:
            complex: Sum of coefficient-weighted term expectations.

        Raises:
            ExecutionError: If the provider returns an empty counts mapping.
        """
        total_shots = sum(counts.values())
        if total_shots <= 0:
            raise ExecutionError("qBraid returned no shots for expectation decoding")
        group_expval = 0j
        for operators, coeff in terms:
            parity_qubits = [op.index for op in operators if op.pauli != identity]
            if not parity_qubits:
                group_expval += coeff
                continue

            parity_sum = 0.0
            for bitstring, count in counts.items():
                bits = bitstring.replace(" ", "")
                n_bits = len(bits)
                parity = 0
                for qubit in parity_qubits:
                    bit_position = n_bits - 1 - qubit
                    if 0 <= bit_position < n_bits:
                        parity ^= int(bits[bit_position])
                parity_sum += (-1 if parity else 1) * count
            group_expval += coeff * (parity_sum / total_shots)
        return group_expval

    @staticmethod
    def _ensure_measurements(circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Ensure circuit has measurements, adding measure_all if needed."""
        if circuit.num_clbits > 0:
            return circuit
        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        return circuit_copy
