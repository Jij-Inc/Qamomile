core.converters.qrao.graph_coloring
===================================

.. py:module:: core.converters.qrao.graph_coloring


Functions
---------

.. autoapisummary::

   core.converters.qrao.graph_coloring.greedy_graph_coloring
   core.converters.qrao.graph_coloring.check_linear_term


Module Contents
---------------

.. py:function:: greedy_graph_coloring(graph: Iterable[tuple[int, int]], max_color_group_size: int, init_coloring: Optional[dict[int, int]] = None) -> tuple[dict[int, int], dict[int, list[int]]]

   graph coloring for QRAC

   :param graph: _description_
   :type graph: typ.Iterable[tuple[int, int]]
   :param max_color_group_size: if you want to use for the qrac31, set 3.
   :type max_color_group_size: int
   :param init_coloring: initial coloring. Defaults to None.
   :type init_coloring: typ.Optional[dict[int, int]], optional

   :returns: coloring, color_group
   :rtype: tuple[dict[int, int], dict[int, list[int]]]

   .. rubric:: Examples

   >>> graph = [(0, 1), (1, 2), (2, 0), (0, 4)]
   >>> greedy_graph_coloring(graph, 2)
   ({0: 0, 1: 1, 2: 2, 4: 1}, {0: [0], 1: [1, 4], 2: [2]})


.. py:function:: check_linear_term(color_group: dict[int, list[int]], linear_term_index: list[int], max_color_group_size: int) -> dict[int, list[int]]

   Search for items within the index of linear term that have not been assigned to the color_group, and add them.

   :param color_group: color_group
   :type color_group: dict[int, list[int]]
   :param linear_term_index: list of index of linear term
   :type linear_term_index: list[int]
   :param max_color_group_size: the maximum number of encoding qubits. if you want to use for the qrac31, set 3.
   :type max_color_group_size: int

   :returns: color_group which added items within the index of linear term that have not been assigned to the color_group.
   :rtype: dict[int, list[int]]


