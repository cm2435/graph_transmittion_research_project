import adj_matrix_gen
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
import collections

class TransitionRule(ABC):
    @abstractmethod
    def query(self, node_index: int):
        pass
class ProbabilisticTransition(TransitionRule):
    def __init__(self, probability: float):
        self.probability = probability
    def query(self, node_index: int):
        return True
"""
Old API:

def infect_till_saturation(
        self,
        structure_name: str,
        infection_probability: float = 1,
        max_iters: int = 15000,
        modality: str = "irreversable",
        sample_giant : bool = True,
        store_infectivity_list : bool = True,
        verbose: bool = True,
    ) """


def saturator(structure: nx.Graph, trans: TransitionRule, max_iters: int = 15000):
    num_nodes = structure.number_of_nodes()
    edges = np.array(structure.edges)
    #print(edges.shape)

    graph = nx.Graph(structure)
    # Important! The node IDs are preserved but the edge-structure is removed entirely
    graph.clear_edges()

    saturation_state = np.zeros(shape=num_nodes, dtype=np.bool8)

    def sample_edges(how_many = 5):
        return edges[np.random.choice(edges.shape[0], how_many)]

    def sample_nonzero(how_many = 5):
        return np.random.choice(np.where(saturation_state == 0)[0], how_many)

    saturation_proportion = []

    # Initial seeding
    initial_nodes = sample_nonzero()

    # print(initial_nodes)
    saturation_state[initial_nodes] = 1
    for iter_idx in range(max_iters):
        if np.all(saturation_state):
            # Fully saturated
            break
        sampled_edges = sample_edges()

        for (u, v) in sampled_edges:
            graph.add_edge(u, v)

        test_these = np.nonzero(saturation_state)[0]
        dq = collections.deque(test_these)
        while len(dq) > 0:
            import random
            node = dq.pop()
            if random.random() < 0.9:
                saturation_state[node] = 1
                nbors = graph.neighbors(node)
                for v in nbors:
                    #print(f"{node} -> {v}")
                    # Don't push if already on
                    if not saturation_state[v]:
                        dq.appendleft(v)

        saturation_proportion.append(np.nonzero(saturation_state)[0].__len__() / num_nodes)
    return saturation_proportion

if __name__ == "__main__":
    print("Testing new saturator")
    ba = adj_matrix_gen.BarabasiAlbert(num_nodes=5000)
    raw_graph = ba.generate_graph()

    graph = raw_graph.subgraph(max(nx.connected_components(raw_graph), key=len))
    #nx.draw(raw_graph)
    import matplotlib.pyplot as plt
    #plt.show()
    print(sum([d for k, d in graph.degree()]) / graph.number_of_nodes())
    print(raw_graph)
    print(graph)

    trans = ProbabilisticTransition

    result = saturator(graph, trans)
    print(len(result))
    # nx.draw(result)
    # plt.show()
