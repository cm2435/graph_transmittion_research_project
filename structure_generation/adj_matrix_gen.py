import numpy as np
import random
import abc
import networkx as nx
from typing import Type

class GraphGenerator(abc.ABC):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 20):
        self.structure_name: str = structure_name
        self.num_nodes: int = num_nodes

    @abc.abstractmethod
    def generate_adj_matrix(self):
        pass

    @staticmethod
    def get_graph_names():
        return [x.name for x in GraphGenerator.__subclasses__()]

    @staticmethod
    def from_string(name: str):
        try:
            return next(
                iter([x for x in GraphGenerator.__subclasses__() if x.name == name])
            )
        except:
            assert False


class DebugStatic(GraphGenerator):
    name = "debug_static"

    def adj_matrix(self):
        debug_graph = np.zeros((self.num_nodes, self.num_nodes))
        debug_graph[1, 2] = 1
        return debug_graph


class CycleGraph(GraphGenerator):
    name = "cycle_generator"

    def adj_matrix(self):
        import networkx as nx

        graph = nx.cycle_graph(self.num_nodes)
        return nx.to_numpy_array(graph)


class BarabasiAlbert(GraphGenerator):
    '''
    '''
    name = "barabasi_albert"
    def __init__(self, num_nodes : int, structure_name : str = "barabasi_albert"):
        super(BarabasiAlbert, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import networkx as nx
        graph = nx.barabasi_albert_graph(self.num_nodes, 1)
        return nx.to_numpy_array(graph)


class ConfigurationGraph(GraphGenerator):
    '''
    '''
    name = "configuration"
    def __init__(self, num_nodes : int, structure_name : str = "configuration"):
        super(ConfigurationGraph, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.initial_adj_matrix = self.generate_adj_matrix()
    
    def generate_adj_matrix(self) -> np.ndarray:
        import networkx as nx
        sequence = nx.random_powerlaw_tree_sequence(self.num_nodes,tries=50000)
        graph = nx.configuration_model(sequence)
        return nx.to_numpy_array(graph)


class RandomSparse(GraphGenerator):
    '''
    '''
    name = "random_sparse"
    def __init__(self, num_nodes : int, structure_name : str = "random_sparse"):
        super(RandomSparse, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self, num_edges : int = 5) -> np.ndarray:
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

class FullyConnected(GraphGenerator):
    '''
    '''
    name = "fully_connected"
    def __init__(self, num_nodes: int, structure_name : str = "fully_connected"): 
        super(FullyConnected, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(nx.complete_graph(self.num_nodes))


class RandomGeometric(GraphGenerator):
    '''
    '''
    name = "random_geometric"
    def __init__(self, num_nodes: int, structure_name : str = "random_geometric"): 
        super(RandomGeometric, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.node_mean = 0 
        self.node_std = 2
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import random
        pos = {i: 
            (random.gauss(self.node_mean, self.node_std), random.gauss(self.node_mean, self.node_std))
            for i in range(self.num_nodes)
        }

        return nx.to_numpy_array(nx.random_geometric_graph(self.num_nodes, 0.2, pos=pos))

class SparseErdos(GraphGenerator):
    '''
    '''
    name = "sparse_erdos"
    def __init__(self, num_nodes: int, structure_name : str = "sparse_erdos"): 
        super(SparseErdos, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.edge_prob = 500 / self.num_nodes ** 2
        self.node_std = 2

        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import random
        return nx.to_numpy_array(nx.fast_gnp_random_graph(self.num_nodes, self.edge_prob, seed=None, directed=False))

class GraphStructureGenerator(object):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 20):
        self.structure_name: str = structure_name
        self.num_nodes: int = num_nodes
        self.allowed_structures: list[str] = ["fully_connected", "random_sparse", "barabasi_albert", "configuration", "random_geometric", "sparse_erdos"]
        self.initial_adj_matrix = self.get_graph_structure().initial_adj_matrix

    def get_graph_structure(self) -> np.ndarray:
        """ """
        structure_name = self.structure_name
        graph_mapping = {
            "fully_connected": FullyConnected,
            "random_sparse":  RandomSparse,
            "barabasi_albert" : BarabasiAlbert,
            "configuration" : ConfigurationGraph,
            "random_geometric" : RandomGeometric,
            "sparse_erdos" : SparseErdos,
        }
        return graph_mapping[structure_name](num_nodes = self.num_nodes)


#if __name__ == "__main__":
#    x = RandomSparse(structure_name= "random_sparse", num_nodes= 50)
