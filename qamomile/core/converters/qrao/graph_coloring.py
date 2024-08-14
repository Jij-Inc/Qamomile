from __future__ import annotations
import typing as typ


def greedy_graph_coloring(
    graph: typ.Iterable[tuple[int, int]],
    max_color_group_size: int,
    init_coloring: typ.Optional[dict[int, int]] = None,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """graph coloring for QRAC

    Args:
        graph (typ.Iterable[tuple[int, int]]): _description_
        max_color_group_size (int): if you want to use for the qrac31, set 3.
        init_coloring (typ.Optional[dict[int, int]], optional): initial coloring. Defaults to None.

    Returns:
        tuple[dict[int, int], dict[int, list[int]]]: coloring, color_group


    Examples:
        >>> graph = [(0, 1), (1, 2), (2, 0), (0, 4)]
        >>> greedy_graph_coloring(graph, 2)
        ({0: 0, 1: 1, 2: 2, 4: 1}, {0: [0], 1: [1, 4], 2: [2]})

    """
    coloring = {}
    if init_coloring:
        coloring.update(init_coloring)

    max_color = max(coloring.values()) if coloring else -1

    adj_matrix: dict[int, list[int]] = {}
    for i, j in graph:
        if i == j:
            continue
        if i not in adj_matrix:
            adj_matrix[i] = []
        if j not in adj_matrix:
            adj_matrix[j] = []
        adj_matrix[i].append(j)
        adj_matrix[j].append(i)

    color_group: dict[int, list[int]] = {}
    for index, color in coloring.items():
        if color not in coloring:
            color_group[color] = []
        color_group[color].append(index)

    for i, neigborhoors in adj_matrix.items():
        if i in coloring:
            continue

        neighbor_colors = [coloring[k] for k in neigborhoors if k in coloring]

        done_coloring = False
        for color in range(max_color + 1):
            if color not in color_group or color not in neighbor_colors:
                if color not in color_group or max_color_group_size > len(
                    color_group[color]
                ):
                    coloring[i] = color
                    done_coloring = True
                    if color not in color_group:
                        color_group[color] = []
                    color_group[color].append(i)
                    break
        if not done_coloring:
            coloring[i] = max_color + 1
            color_group[max_color + 1] = [i]
            max_color += 1

    return coloring, color_group


def check_linear_term(
    color_group: dict[int, list[int]],
    linear_term_index: list[int],
    max_color_group_size: int,
) -> dict[int, list[int]]:
    """Search for items within the index of linear term that have not been assigned to the color_group, and add them.

    Args:
        color_group (dict[int, list[int]]): color_group
        linear_term_index (list[int]): list of index of linear term
        max_color_group_size (int): the maximum number of encoding qubits. if you want to use for the qrac31, set 3.

    Returns:
        dict[int, list[int]]: color_group which added items within the index of linear term that have not been assigned to the color_group.
    """
    idx_in_color_group = []
    for v in color_group.values():
        idx_in_color_group.extend(v)

    # We're adding a bit to the next index of the 'quad' qubit, affecting only the linear term.
    # If the Ising Hamiltonian has only linear terms, making 'quad' empty and causing a 'max' error, we return index 0 to avoid this.
    qubit_index_for_linear = max(color_group.keys()) + 1 if color_group else 0
    bits_in_qubits_counter = 1
    for idx in linear_term_index:
        if idx not in idx_in_color_group:
            if bits_in_qubits_counter == 1:
                color_group[qubit_index_for_linear] = []

            color_group[qubit_index_for_linear].append(idx)
            bits_in_qubits_counter += 1

            if bits_in_qubits_counter > max_color_group_size:
                qubit_index_for_linear += 1
                bits_in_qubits_counter = 1

    return color_group
