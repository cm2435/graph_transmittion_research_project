import numpy as np
import random
import networkx as nx 


class GraphStructureGenerator(object):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 5):
        self.structure_name = structure_name
        self.num_nodes = num_nodes
        self.allowed_structures = ["fully_connected", "random_sparse"]

    @property
    def adj_matrix(self):
        """ """
        graph_mapping = {
            "fully_connected": self._generate_fully_connected_graph,
            "random_sparse": self._generate_sparse_graph,
            "composite_model" : self._generate_composite_graph
        }
        return graph_mapping[self.structure_name]()

    def _generate_fully_connected_graph(self):
        """ """
        return np.ones((self.num_nodes, self.num_nodes))

    def _generate_sparse_graph(self, num_edges: int = 5) -> np.ndarray:
        """
        Generate a random num_node X num_node adjacency matrix that is seeded with
        num_timestep_edges connections in another wise sparse graph.
        """
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(num_edges):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(
                0, self.num_nodes - 1
            )
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph
    
    def _generate_composite_graph(self):
        """
        """
        sequence = nx.random_powerlaw_tree_sequence(self.num_nodes, tries=5000)
        print(sequence)
        G = nx.configuration_model(sequence)

        return nx.adjacency_matrix(G).todense()

    

if __name__ == "__main__": 
    x = GraphStructureGenerator("composite_model")
    print(x.adj_matrix)