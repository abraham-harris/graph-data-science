import networkx as nx
from typing import Tuple, Hashable, Set


def test_hw7_problem_2() -> None:
    """
    Problem 2: Create a graph and a partition of the nodes such that Q >= 0.95.
    Graph must have >= 5 vertices, >= 2 edges, and partition must have >= 2 sets.
    """

    # Build graph
    G: nx.Graph = nx.Graph()
    # TODO: Add vertices
    G.add_nodes_from([i for i in range(1, 41)])
    # TODO: Add edges
    G.add_edges_from([
        (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), 
        (15,16), (17,18), (19,20), (21,22), (23,24), (25,26),
        (27,28), (29,30), (31,32), (33,34), (35,36), (37,38),
        (39,40)
    ])
    # TODO: Define partition
    partition: Tuple[Set[Hashable], ...] = (
        set([1, 2]), set([3,4]), set([5,6]), set([7,8]), set([9,10]), 
        set([11,12]), set([13,14]), set([15,16]), set([17,18]),
        set([19,20]), set([21,22]), set([23,24]), set([25,26]),
        set([27,28]), set([29,30]), set([31,32]), set([33,34]),
        set([35,36]), set([37,38]), set([39,40])
    )

    # Basic structural checks
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() >= 5
    assert G.number_of_edges() >= 2

    # Partition validity checks
    assert len(partition) >= 2
    assert all(len(group) > 0 for group in partition)
    union = set().union(*partition)
    assert union == set(G.nodes())
    assert sum(len(group) for group in partition) == len(union)

    # Modularity check
    q = nx.community.modularity(G, partition)
    assert q >= 0.95    # Q >= 0.95