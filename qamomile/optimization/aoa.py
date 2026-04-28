"""Converter for the Alternating Operator Ansatz (AOA).

AOA reuses the same Ising cost Hamiltonian as QAOA but replaces the standard
X mixer with a constraint-aware XY mixer. The XY mixer schedule is encoded as a
matrix of qubit pairs, which can either be supplied explicitly or generated
from a named built-in schedule.
"""

from __future__ import annotations

import typing as typ
import warnings

import numpy as np

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.aoa import aoa_state, hubo_aoa_state
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.transpiler import Transpiler

from .qaoa import QAOAConverter

MixerName = typ.Literal["ring", "fully-connected"]


class AOAConverter(QAOAConverter):
    """Converter for Alternating Operator Ansatz (AOA).

    Extends :class:`QAOAConverter` by keeping the same Ising cost Hamiltonian
    while replacing the X mixer with a configurable XY mixer schedule.

    The mixer schedule can either be selected from built-in options such as a
    ring mixer or a fully connected mixer, or passed explicitly as a matrix of
    qubit index pairs.

    Example:
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
            pair_indices: Array-like schedule of qubit pairs.

        Returns:
            numpy.ndarray: Pair schedule converted to ``uint64`` with shape
            ``(num_pairs, 2)``.

        Raises:
            ValueError: If the provided schedule does not have shape
                ``(num_pairs, 2)``.
        """
        normalized = np.asarray(pair_indices, dtype=np.uint64)
        if normalized.ndim != 2 or normalized.shape[1] != 2:
            raise ValueError("pair_indices must have shape (num_pairs, 2).")
        return normalized

    def _build_ring_pair_indices(self, num_blocks: int, block_size: int) -> np.ndarray:
        """Build the parity-style ring XY mixer schedule.

        Args:
            num_blocks: Number of one-hot blocks.
            block_size: Number of qubits per block.

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
            pair_indices: Array of qubit pairs with shape ``(num_pairs, 2)``.

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
            num_blocks: Number of one-hot blocks.
            block_size: Number of qubits per block.

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

    def _resolve_pair_indices(
        self,
        *,
        mixer: MixerName,
        pair_indices: np.ndarray | None,
        block_size: int | None,
    ) -> np.ndarray:
        """Resolve the XY mixer schedule into a validated pair matrix.

        Args:
            mixer: Name of the built-in mixer schedule to use when explicit
                pair indices are not supplied.
            pair_indices: Optional explicit array of qubit pairs with shape
                ``(num_pairs, 2)``.
            block_size: Size of each one-hot block used by the built-in mixer
                schedule builders.

        Returns:
            numpy.ndarray: A validated ``uint64`` array of qubit pairs.

        Raises:
            ValueError: If neither explicit pairs nor a valid built-in mixer
                configuration is provided.
        """
        if pair_indices is not None:
            if mixer != "ring":
                warnings.warn(
                    f"pair_indices was provided; the mixer={mixer!r} argument is ignored.",
                    stacklevel=3,
                )
            return self._normalize_pair_indices(pair_indices)

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
        if mixer == "ring":
            return self._build_ring_pair_indices(num_blocks, block_size)
        if mixer == "fully-connected":
            return self._build_fully_connected_pair_indices(num_blocks, block_size)
        raise ValueError(
            f"Unknown mixer {mixer!r}. Must be 'ring' or 'fully-connected' "
            "when pair_indices is not provided."
        )

    def transpile(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        mixer: MixerName = "ring",
        pair_indices: np.ndarray | None = None,
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
            mixer (Literal["ring", "fully-connected"]): Named built-in mixer
                schedule to use when ``pair_indices`` is not provided.
            pair_indices (numpy.ndarray | None): Explicit mixer schedule as an
                array of shape ``(num_pairs, 2)``. When provided, ``mixer`` is
                ignored.
            block_size (int | None): Size of each one-hot block. Required when
                ``pair_indices`` is not provided because built-in schedules need
                the block structure.

        Returns:
            ExecutableProgram: The compiled circuit program.
        """
        resolved_pair_indices = self._resolve_pair_indices(
            mixer=mixer,
            pair_indices=pair_indices,
            block_size=block_size,
        )

        if not self.spin_model.higher:
            return self._transpile_quadratic(
                transpiler,
                p=p,
                pair_indices=resolved_pair_indices,
            )
        return self._transpile_hubo(
            transpiler,
            p=p,
            pair_indices=resolved_pair_indices,
        )

    def _transpile_quadratic(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a quadratic-only model using the AOA circuit.
        
        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices (numpy.ndarray): Explicit mixer schedule as an array of
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
            pair_indices: qmc.Matrix[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = aoa_state(
                p=p,
                quad=quad,
                linear=linear,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices=pair_indices,
            )
            return qmc.measure(q)

        return transpiler.transpile(
            aoa_sampling,
            bindings={
                "linear": self.spin_model.linear,
                "quad": self.spin_model.quad,
                "n": self.spin_model.num_bits,
                "p": p,
                "pair_indices": pair_indices,
            },
            parameters=["gammas", "betas"],
        )

    def _transpile_hubo(
        self,
        transpiler: Transpiler,
        *,
        p: int,
        pair_indices: np.ndarray,
    ) -> ExecutableProgram:
        """Transpile a model with higher-order terms using the AOA circuit.
        
        Args:
            transpiler (Transpiler): Backend transpiler to use.
            p (int): Number of AOA layers.
            pair_indices (numpy.ndarray): Explicit mixer schedule as an array of
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
            pair_indices: qmc.Matrix[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = hubo_aoa_state(
                p=p,
                quad=quad,
                linear=linear,
                higher=higher,
                n=n,
                gammas=gammas,
                betas=betas,
                pair_indices=pair_indices,
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
                "pair_indices": pair_indices,
            },
            parameters=["gammas", "betas"],
        )