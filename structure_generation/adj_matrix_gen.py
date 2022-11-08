import numpy as np
import random
import abc
import networkx as nx


class GraphGenerator(abc.ABC):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 20):
        self.structure_name = structure_name
        self.num_nodes = num_nodes

    @abc.abstractmethod
    def adj_matrix(self):
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


class FullyConnected(GraphGenerator):
    name = "fully_connected"

    def adj_matrix(self):
        return nx.to_numpy_array(nx.complete_graph(self.num_nodes))


class RandomSparse(GraphGenerator):
    name = "random_sparse"

    def adj_matrix(self):
        num_edges = 5
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


class DebugStatic(GraphGenerator):
    name = "debug_static"

    def adj_matrix(self):
        debug_graph = np.zeros((self.num_nodes, self.num_nodes))
        debug_graph[1, 2] = 1
        return debug_graph


class BarabasiAlbert(GraphGenerator):
    name = "barabasi_albert"

    def adj_matrix(self):
        import networkx as nx

        graph = nx.barabasi_albert_graph(self.num_nodes, 2)
        return nx.to_numpy_array(graph)


class CycleGraph(GraphGenerator):
    name = "cycle_generator"

    def adj_matrix(self):
        import networkx as nx

        graph = nx.cycle_graph(self.num_nodes)
        return nx.to_numpy_array(graph)


class GraphStructureGenerator(object):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 20):
        self.structure_name = structure_name
        self.num_nodes = num_nodes
        self.allowed_structures = ["fully_connected", "random_sparse"]

    @property
    def adj_matrix(self):
        """ """
        graph_mapping = {
            "fully_connected": self.generate_fully_connected_graph,
            "random_sparse": self.generate_sparse_graph,
        }
        return graph_mapping[self.structure_name]()

    def generate_fully_connected_graph(self):
        """ """
        return np.ones((self.num_nodes, self.num_nodes))

    def generate_sparse_graph(self, num_edges: int = 5) -> np.ndarray:
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
