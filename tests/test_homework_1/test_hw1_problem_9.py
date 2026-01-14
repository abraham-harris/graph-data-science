from network_utilities import adjacency_list_to_graph
import networkx as nx

def test_homework_problem_9() -> None:
    # What I expect
    desired_number_nodes: int = 6
    desired_minimum_density: float = 0.42
    desired_maximum_density: float = 0.46

    # when
    ## FIX THIS ADJACENCY LIST
    adjacency_list: dict[int, set[int]] = {1: {2, 3, 4, 5, 6, 7, 8, 9},
                                           2: {1, 3, 4},
                                           3: {1, 2, 4},
                                           4: {1, 2, 3, 5},
                                           5: {1, 4, 6},
                                           6: {1, 5, 7},
                                           7: {1, 6, 8},
                                           8: {1, 7, 9},
                                           9: {1, 8}}
    G = adjacency_list_to_graph(adjacency_list)

    # then
    assert nx.is_connected(G)
    assert len(G.nodes()) >= desired_number_nodes
    assert nx.density(G) >= desired_minimum_density
    assert nx.density(G) <= desired_maximum_density