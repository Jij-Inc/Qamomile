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
        for color in range(max_color+1):
            if color not in color_group or color not in neighbor_colors:
                if color not in color_group or max_color_group_size > len(color_group[color]):
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
