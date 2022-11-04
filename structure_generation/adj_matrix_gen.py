import numpy as np 
import random 

class GraphStructureGenerator(object):
    """
    """
    def __init__(self, structure_name : str, num_nodes : int = 20):
        self.structure_name = structure_name
        self.num_nodes = num_nodes
        self.allowed_structures = ["fully_connected", "random_sparse"]

    @property
    def adj_matrix(self):
        '''
        '''
        graph_mapping = {
            "fully_connected" : self.generate_fully_connected_graph,
            "random_sparse" : self.generate_sparse_graph
        }
        return graph_mapping[self.structure_name]()
    
    def generate_fully_connected_graph(self): 
        """
        """
        return np.ones((self.num_nodes, self.num_nodes))

    def generate_sparse_graph(self, num_edges : int = 5) -> np.ndarray:
        """
        Generate a random num_node by num_node adjacency matrix that is seeded with
        num_timestep_edges connections in an otherwise sparse graph.
        """
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(num_edges):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(
                0, self.num_nodes - 1
            )
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph