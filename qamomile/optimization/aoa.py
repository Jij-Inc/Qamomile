"""Converter for the Alternating Operator Ansatz (AOA).

AOA reuses the same Ising cost Hamiltonian as QAOA but replaces the standard
X mixer with a constraint-aware XY mixer. The XY mixer schedule is encoded as a
matrix of qubit pairs, which can either be supplied explicitly or generated
from a named built-in schedule.
"""

from __future__ import annotations

import enum
import warnings

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.aoa import (
    aoa_state_basis_state,
    aoa_state_dicke,
    aoa_state_superposition,
    hubo_aoa_state_basis_state,
    hubo_aoa_state_dicke,
    hubo_aoa_state_superposition,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule

from .qaoa import QAOAConverter


class InitialState(enum.StrEnum):
    """Initial state selector for :class:`AOAConverter`.

    Attributes:
        UNIFORM: Uniform superposition prepared by a Hadamard layer.
        DICKE: Product of Dicke states prepared via the Bartschi-Eidenbenz
            SCS construction.
        SINGLE_BASIS_STATE: Computational basis state with the requested
            Hamming weight per block (last ``k`` qubits of each block set
            to ``|1>``).
    """

    UNIFORM = "uniform"
    DICKE = "dicke"
    SINGLE_BASIS_STATE = "single_basis_state"


class MixerName(enum.StrEnum):
    """Built-in XY mixer schedule selector for :class:`AOAConverter`.

    Attributes:
        RING: Parity-style ring schedule — adjacent pairs in two alternating
            layers, plus a wrap-around pair when ``block_size > 2``.
        FULLY_CONNECTED: All pairs within each block, greedily partitioned
            into non-overlapping batches.
    """

    RING = "ring"
    FULLY_CONNECTED = "fully-connected"


class AOAConverter(QAOAConverter):
    """Converter for Alternating Operator Ansatz (AOA).

    Extends :class:`QAOAConverter` by keeping the same Ising cost Hamiltonian
    while replacing the X mixer with a configurable XY mixer schedule.

    The mixer schedule can either be selected from built-in options such as a
    ring mixer or a fully connected mixer, or passed explicitly as a matrix of
    qubit index pairs.

    Example:
        >>> from qamomile.optimization.binary_model import BinaryModel
        >>> from qamomile.qiskit.transpiler import QiskitTranspiler
        >>> model = BinaryModel.from_hubo({(0, 1, 2): 1.0, (0,): -2.0})
        >>> converter = AOAConverter(model)
        >>> executable = converter.transpile(
        ...     QiskitTranspiler(),
        ...     p=2,
        ...     mixer="ring",
        ...     block_size=3,
        ... )
    """

    def _normalize_pair_indices(self, pair_indices: np.ndarray) -> np.ndarray:
        """Validate and normalize a user-provided pair schedule.

        Args:
            pair_indices (np.ndarray): Array-like schedule of qubit pairs.

        Returns:
            numpy.ndarray: Pair schedule converted to ``uint64`` with shape
            ``(num_pairs, 2)``.

        Raises:
            ValueError: If the provided schedule does not have shape
                ``(num_pairs, 2)``.
                If the schedule contains negative qubit indices.
                If the schedule contains self-pairs (``(i, i)``).
                If the schedule contains qubit indices >= num_bits.
        """
        signed = np.asarray(pair_indices, dtype=np.int64)
        if signed.ndim != 2 or signed.shape[1] != 2:
            raise ValueError("pair_indices must have shape (num_pairs, 2).")
        negative_rows = np.flatnonzero((signed < 0).any(axis=1))
        if negative_rows.size:
            offending = signed[negative_rows].tolist()
            raise ValueError(
                f"pair_indices contains negative qubit indices at rows "
                f"{negative_rows.tolist()}: {offending}."
            )
        normalized = signed.astype(np.uint64)
        self_pair_rows = np.flatnonzero(normalized[:, 0] == normalized[:, 1])
        if self_pair_rows.size:
            offending = normalized[self_pair_rows].tolist()
            raise ValueError(
                f"pair_indices contains self-pairs at rows {self_pair_rows.tolist()}: "
                f"{offending}. XY mixer pairs must connect two distinct qubits."
            )

        n = self.spin_model.num_bits
        out_of_range_rows = np.flatnonzero((normalized >= n).any(axis=1))
        if out_of_range_rows.size:
            offending = normalized[out_of_range_rows].tolist()
            raise ValueError(
                f"pair_indices contains qubit indices >= num_bits ({n}) at rows "
                f"{out_of_range_rows.tolist()}: {offending}."
            )

        return normalized

    def _build_ring_pair_indices(self, num_blocks: int, block_size: int) -> np.ndarray:
        """Build the parity-style ring XY mixer schedule.

        Args:
            num_blocks (int): Number of one-hot blocks.
            block_size (int): Number of qubits per block.

        Returns:
            numpy.ndarray: Flattened list of qubit pairs implementing the ring
            schedule over all blocks.
        """
        pairs: list[tuple[int, int]] = []
        for block in range(num_blocks):
            start = block * block_size
            for i in range(0, block_size - 1, 2):
                pairs.append((start + i, start + i + 1))
            for i in range(1, block_size - 1, 2):
                pairs.append((start + i, start + i + 1))
            if block_size > 2:
                pairs.append((start + block_size - 1, start))
        return np.asarray(pairs, dtype=np.uint64)

    def _partition_pairs(self, pair_indices: np.ndarray) -> list[list[tuple[int, int]]]:
        """Greedily partition pairs into non-overlapping batches.

        Args:
            pair_indices (np.ndarray): Array of qubit pairs with shape ``(num_pairs, 2)``.

        Returns:
            list[list[tuple[int, int]]]: List of partitions where pairs inside
            each partition do not share any qubit.
        """
        partitions: list[list[tuple[int, int]]] = []
        used_nodes_per_partition: list[set[int]] = []

        for raw_left, raw_right in pair_indices:
            left, right = int(raw_left), int(raw_right)
            for partition, used_nodes in zip(partitions, used_nodes_per_partition):
                if left not in used_nodes and right not in used_nodes:
                    partition.append((left, right))
                    used_nodes.update((left, right))
                    break
            else:
                partitions.append([(left, right)])
                used_nodes_per_partition.append({left, right})

        return partitions

    def _build_fully_connected_pair_indices(
        self,
        num_blocks: int,
        block_size: int,
    ) -> np.ndarray:
        """Build a partitioned fully connected XY mixer schedule.

        The method first enumerates all pairs inside each one-hot block, then
        greedily partitions overlapping pairs into non-overlapping batches and
        finally flattens those batches into a single execution order.

        Args:
            num_blocks (int): Number of one-hot blocks.
            block_size (int): Number of qubits per block.

        Returns:
            numpy.ndarray: Flattened list of qubit pairs implementing the
            fully connected schedule.
        """
        pairs: list[tuple[int, int]] = []
        for block in range(num_blocks):
            start = block * block_size
            for i in range(block_size):
                for j in range(i + 1, block_size):
                    pairs.append((start + i, start + j))

        partitions = self._partition_pairs(np.asarray(pairs, dtype=np.uint64))
        flattened_pairs = [pair for partition in partitions for pair in partition]
        return np.asarray(flattened_pairs, dtype=np.uint64)

    def resolve_pair_indices(
        self,
        *,
        mixer: str | MixerName,
        pair_indices: np.ndarray | None,
        block_size: int | None,
    ) -> np.ndarray:
        """Resolve the XY mixer schedule into a validated pair matrix.

        Args:
            mixer (str | MixerName): Name of the built-in mixer schedule to
                use when explicit pair indices are not supplied.
            pair_indices (np.ndarray | None): Optional explicit array of qubit
                pairs with shape ``(num_pairs, 2)``.
            block_size (int | None): Size of each one-hot block used by the
                built-in mixer schedule builders.

        Returns:
            numpy.ndarray: A validated ``uint64`` array of qubit pairs.

        Raises:
            ValueError: If ``mixer`` is not a recognised :class:`MixerName`
                value (and ``pair_indices`` is ``None``); if ``block_size`` is
                ``None`` when ``pair_indices`` is not provided; if
                ``block_size <= 1``; if ``spin_model.num_bits`` is not
                divisible by ``block_size``; or if ``pair_indices`` has an
                invalid shape, contains self-pairs, or references qubit
                indices outside the model.
        """
        if pair_indices is not None:
            try:
                mixer_name = MixerName(mixer)
                if mixer_name != MixerName.RING:
                    warnings.warn(
                        f"pair_indices_mixer was provided; the mixer={mixer!r} argument is ignored.",
                        UserWarning,
                        stacklevel=3,
                    )
            except ValueError:
                warnings.warn(
                    f"pair_indices_mixer was provided; the unrecognised mixer={mixer!r} is ignored.",
                    UserWarning,
                    stacklevel=3,
                )
            return self._normalize_pair_indices(pair_indices)

        mixer_name = MixerName(mixer)

        if block_size is None:
            raise ValueError(
                "block_size is required when pair_indices is not provided."
            )

        if block_size <= 1:
            raise ValueError("block_size must be greater than 1.")

        if self.spin_model.num_bits % block_size != 0:
            raise ValueError(
                "spin_model.num_bits must be divisible by block_size to build "
                "a built-in AOA mixer schedule."
            )

        num_blocks = self.spin_model.num_bits // block_size
        match mixer_name:
            case MixerName.RING:
                return self._build_ring_pair_indices(num_blocks, block_size)
            case MixerName.FULLY_CONNECTED:
                return self._build_fully_connected_pair_indices(num_blocks, block_size)
            case _:
                raise RuntimeError(f"unreachable: unhandled MixerName {mixer_name!r}")

    def compute_dicke_composition_schedule(
        self,
        hamming_weight: int,
        block_size: int,
    ) -> tuple[np.ndarray, dict[tuple[int, int, int], float]]:
        """Build the global Dicke state preparation schedule for a product of blocks.

        Delegates to :func:`dicke_state_composition_schedule` using the total
        number of qubits from the spin model.

        Args:
            hamming_weight (int): Dicke Hamming weight ``k`` applied to every block.
            block_size (int): Number of qubits per block.

        Returns:
            tuple[np.ndarray, dict[tuple[int, int, int], float]]:
            ``(initial_ones, schedule_dicke)`` — global qubit indices and the
            ordered SCS gate schedule for the full register.  Pair entries in
            ``schedule_dicke`` satisfy ``key[1] == key[2]``; triplet entries
            satisfy ``key[1] != key[2]``.

        Raises:
            ValueError: If ``block_size <= 0``; if ``spin_model.num_bits`` is
                not divisible by ``block_size``; or if ``hamming_weight`` is
                outside ``[0, block_size]``.
        """
        n_qubits = self.spin_model.num_bits
        return dicke_state_composition_schedule(n_qubits, block_size, hamming_weight)

    def compute_basis_state_initial_ones(
        self,
        hamming_weight: int,
        block_size: int,
    ) -> np.ndarray:
        """Build basis-state ``|1>`` indices per block for a target Hamming weight.

        Args:
            hamming_weight (int): Number of ``|1>`` qubits per block.
            block_size (int): Number of qubits per block.

        Returns:
            numpy.ndarray: Global qubit indices initialized in ``|1>``.

        Raises:
            ValueError: If ``block_size <= 0``; if ``n_qubits`` is not
                divisible by ``block_size``; or if ``hamming_weight`` is
                outside ``[0, block_size]``.
        """
        if block_size <= 0:
            raise ValueError("block_size must be > 0.")

        n_qubits = self.spin_model.num_bits
        if n_qubits % block_size != 0:
            raise ValueError("n_qubits must be divisible by block_size.")

        if not (0 <= hamming_weight <= block_size):
            raise ValueError("Require 0 <= hamming_weight <= block_size.")

        num_blocks = n_qubits // block_size

        initial_ones: list[int] = []
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            for local_idx in range(block_size - hamming_weight, block_size):
                initial_ones.append(start + local_idx)

        return np.asarray(initial_ones, dtype=np.uint32)

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        initial_state: str | InitialState = InitialState.DICKE,
        hamming_weight: int = 1,
        mixer: str | MixerName = MixerName.RING,
        pair_indices_mixer: np.ndarray | None = None,
        block_size: int | None = None,
    ) -> ExecutableProgram:
        """Transpile the model into an executable AOA circuit.

        Dispatches to the quadratic-only fast path when no higher-order terms
        are present, otherwise uses the HUBO path with phase-gadget
        decomposition. The XY mixer schedule can be selected through a named
        built-in mixer or provided directly as explicit qubit pairs.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            initial_state (str | InitialState): Initial state for the AOA
                circuit. ``InitialState.UNIFORM`` (``"uniform"``) uses the
                equal superposition; ``InitialState.DICKE`` (``"dicke"``)
                prepares a product of Dicke states via the Bartschi-Eidenbenz
                SCS construction; ``InitialState.SINGLE_BASIS_STATE``
                (``"single_basis_state"``) prepares one computational basis
                state per block with the requested Hamming weight. Plain
                strings are accepted and normalised
                automatically.
            hamming_weight (int): Target Hamming weight per block. Used when
                ``initial_state`` is ``InitialState.DICKE`` or
                ``InitialState.SINGLE_BASIS_STATE``.
            mixer (str | MixerName): Named built-in mixer schedule to use when
                ``pair_indices_mixer`` is not provided. Plain strings such as
                ``"ring"`` or ``"fully-connected"`` are accepted and
                normalised automatically.
            pair_indices_mixer (numpy.ndarray | None): Explicit mixer schedule
                as an array of shape ``(num_pairs, 2)``. When provided,
                ``mixer`` is ignored.
            block_size (int | None): Size of each block on which the Dicke
                state is prepared and on which the XY mixer acts. Must divide
                ``spin_model.num_bits`` and be ``> 1`` when a built-in mixer
                schedule is used. If omitted, defaults to the full register
                size (single block).

        Returns:
            ExecutableProgram: The compiled circuit program.

        Raises:
            ValueError: If ``initial_state`` is not a recognised
                :class:`InitialState` member; if ``mixer`` is not a recognised
                :class:`MixerName` member; if ``pair_indices_mixer`` has the
                wrong shape; if ``block_size`` is ``<= 1`` or does not divide
                ``spin_model.num_bits`` when a built-in mixer is requested; or
                if ``hamming_weight`` is outside ``[0, block_size]``.
        """
        initial_state = InitialState(initial_state)

        if (
            pair_indices_mixer is not None
            and block_size is None
            and initial_state in {InitialState.DICKE, InitialState.SINGLE_BASIS_STATE}
        ):
            warnings.warn(
                "pair_indices_mixer was provided without block_size; Dicke and "
                "basis-state initialisation will treat the full register as a single "
                f"block (block_size = {self.spin_model.num_bits}). Pass block_size "
                "explicitly to initialise per-block Dicke states.",
                UserWarning,
                stacklevel=2,
            )

        effective_block_size = (
            self.spin_model.num_bits if block_size is None else block_size
        )

        resolved_pair_indices_mixer = self.resolve_pair_indices(
            mixer=mixer,
            pair_indices=pair_indices_mixer,
            block_size=effective_block_size,
        )

        quadratic = not self.spin_model.higher

        if initial_state == InitialState.DICKE:
            (
                initial_ones,
                schedule_dicke,
            ) = self.compute_dicke_composition_schedule(
                hamming_weight=hamming_weight,
                block_size=effective_block_size,
            )
        elif initial_state == InitialState.SINGLE_BASIS_STATE:
            initial_ones = self.compute_basis_state_initial_ones(
                hamming_weight=hamming_weight,
                block_size=effective_block_size,
            )
        else:
            # UNIFORM: no extra basis-state setup required
            assert initial_state == InitialState.UNIFORM

        match (quadratic, initial_state):
            case (True, InitialState.UNIFORM):
                return self._transpile_aoa_quadratic(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                )
            case (True, InitialState.DICKE):
                return self._transpile_aoa_quadratic_dicke(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                    initial_ones=initial_ones,
                    schedule_dicke=schedule_dicke,
                )
            case (True, InitialState.SINGLE_BASIS_STATE):
                return self._transpile_aoa_quadratic_basis_state(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                    initial_ones=initial_ones,
                )
            case (False, InitialState.UNIFORM):
                return self._transpile_aoa_hubo(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                )
            case (False, InitialState.DICKE):
                return self._transpile_aoa_hubo_dicke(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                    initial_ones=initial_ones,
                    schedule_dicke=schedule_dicke,
                )
            case (False, InitialState.SINGLE_BASIS_STATE):
                return self._transpile_aoa_hubo_basis_state(
                    transpiler,
                    p=p,
                    pair_indices_mixer=resolved_pair_indices_mixer,
                    initial_ones=initial_ones,
                )
            case _:
                raise RuntimeError(
                    f"unreachable: quadratic={quadratic}, initial_state={initial_state!r}"
                )

    def _transpile_aoa_quadratic(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model using the AOA circuit.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Explicit mixer schedule as an array of
                shape ``(num_pairs, 2)``.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = aoa_state_superposition(
                p=p,
                quad=quad,
                linear=linear,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_aoa_quadratic_dicke(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
        initial_ones: np.ndarray,
        schedule_dicke: dict[tuple[int, int, int], float],
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model using the AOA circuit with Dicke state preparation.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Explicit mixer schedule as an array of
                shape ``(num_pairs, 2)``.
            initial_ones (numpy.ndarray): Indices of qubits initialized in ``|1>``.
            schedule_dicke (dict): Ordered SCS gate schedule from
                :func:`~qamomile.optimization.schedules.dicke.dicke_state_composition_schedule`.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling_dicke(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
            initial_ones: qmc.Vector[qmc.UInt],
            schedule_dicke: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = aoa_state_dicke(
                p=p,
                quad=quad,
                linear=linear,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
                initial_ones=initial_ones,
                schedule_dicke=schedule_dicke,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling_dicke,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
                "initial_ones": initial_ones,
                "schedule_dicke": schedule_dicke,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_aoa_quadratic_basis_state(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
        initial_ones: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model with basis-state initialization.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Explicit mixer schedule as an array of
                shape ``(num_pairs, 2)``.
            initial_ones (numpy.ndarray): Indices of qubits initialized in ``|1>``.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling_basis_state(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
            initial_ones: qmc.Vector[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = aoa_state_basis_state(
                p=p,
                quad=quad,
                linear=linear,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
                initial_ones=initial_ones,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling_basis_state,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
                "initial_ones": initial_ones,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_aoa_hubo(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a HUBO model using the AOA circuit with uniform superposition.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Mixer schedule as an array of
                shape ``(num_pairs, 2)``.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling_hubo(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = hubo_aoa_state_superposition(
                p=p,
                quad=quad,
                linear=linear,
                higher=higher,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling_hubo,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "higher": self.spin_model.higher,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_aoa_hubo_dicke(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
        initial_ones: np.ndarray,
        schedule_dicke: dict[tuple[int, int, int], float],
    ) -> ExecutableProgram:
        """Transpile a HUBO model using the AOA circuit with Dicke state preparation.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Mixer schedule as an array of
                shape ``(num_pairs, 2)``.
            initial_ones (numpy.ndarray): Indices of qubits initialized in ``|1>``.
            schedule_dicke (dict): Ordered SCS gate schedule from
                :func:`~qamomile.optimization.schedules.dicke.dicke_state_composition_schedule`.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling_hubo_dicke(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
            initial_ones: qmc.Vector[qmc.UInt],
            schedule_dicke: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = hubo_aoa_state_dicke(
                p=p,
                quad=quad,
                linear=linear,
                higher=higher,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
                initial_ones=initial_ones,
                schedule_dicke=schedule_dicke,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling_hubo_dicke,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "higher": self.spin_model.higher,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
                "initial_ones": initial_ones,
                "schedule_dicke": schedule_dicke,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_aoa_hubo_basis_state(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices_mixer: np.ndarray,
        initial_ones: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a HUBO model with basis-state initialization.

        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices_mixer (numpy.ndarray): Mixer schedule as an array of
                shape ``(num_pairs, 2)``.
            initial_ones (numpy.ndarray): Indices of qubits initialized in ``|1>``.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """

        @qmc.qkernel
        def aoa_sampling_hubo_basis_state(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
            n: qmc.UInt,
            pair_indices_mixer: qmc.Matrix[qmc.UInt],
            initial_ones: qmc.Vector[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = hubo_aoa_state_basis_state(
                p=p,
                quad=quad,
                linear=linear,
                higher=higher,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices_mixer=pair_indices_mixer,
                initial_ones=initial_ones,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling_hubo_basis_state,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "higher": self.spin_model.higher,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices_mixer": pair_indices_mixer,
                "initial_ones": initial_ones,
            },
            parameters=["gammas", "betas"],
        )
