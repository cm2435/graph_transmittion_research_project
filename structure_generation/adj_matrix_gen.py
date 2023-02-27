import numpy as np
import random
import abc
import networkx as nx
from typing import Type


class GraphGenerator(abc.ABC):
    """ """

    def __init__(self, structure_name: str, num_nodes: int):
        self.structure_name: str = structure_name
        self.num_nodes: int = num_nodes

    @abc.abstractmethod
    def generate_graph(self):
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
    
    @staticmethod
    def _find_mean_degree(graph : nx.classes.graph.Graph):
        return np.average([val for (node, val) in graph.degree()])
    

class ConfigurationGraph(GraphGenerator):
    """ """

    name = "configuration"

    def __init__(self, structure_name: str = "configuration", num_nodes: int = 50):
        super().__init__(structure_name=structure_name, num_nodes=num_nodes)
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import networkx as nx

        sequence = nx.random_powerlaw_tree_sequence(self.num_nodes, tries=50000)
        graph = nx.configuration_model(sequence)
        return nx.to_numpy_array(graph)


class BarabasiAlbert(GraphGenerator):
    """ """

    name = "barabasi_albert"

    def __init__(
        self,
        structure_name: str = "barabasi_albert",
        num_nodes: int = 50,
        target_mean_pathlength: float = None,
        target_mean_degree : float = None 
    ):
        super().__init__(structure_name=structure_name, num_nodes=num_nodes)
        self.target_mean_pathlength = target_mean_pathlength
        self.target_mean_degree = target_mean_degree
        
        self.optim_metric = "target_mean_degree" if target_mean_degree is not None else "target_mean_pathlength"
        self.initial_adj_matrix = nx.to_numpy_array(self.generate_calibrated_graph())

    def generate_graph(self, num_attachment_nodes : int = 10):
        graph = nx.barabasi_albert_graph(
            self.num_nodes, num_attachment_nodes
        )
        return graph
    
    def generate_calibrated_graph(self, 
        update_val : int = 15,
        max_iters : int = 25, 
        accepted_precision_percentage : float =  0.0005,
        connecting_edges : float = 20,
    ): 
        #This is slow and I would like to optimise it 
        if self.optim_metric == "target_mean_degree":
            graph = self.generate_graph(connecting_edges)
            for i in range(max_iters): 
                graph = self.generate_graph(np.absolute(int(connecting_edges)))
                graph_mean_degree = self._find_mean_degree(graph)
                update_val = update_val/1.5
                if graph_mean_degree < self.target_mean_degree:
                    connecting_edges += update_val
                elif graph_mean_degree > self.target_mean_degree:
                    connecting_edges -= update_val
                    
                if np.absolute(self.target_mean_degree - graph_mean_degree) < accepted_precision_percentage * self.num_nodes: 
                    return graph
            return graph 
    
    def generate_adj_matrix(self) -> np.ndarray:
        if isinstance(self.node_degree, tuple):
            self.node_degree = self.node_degree[0]
        graph = nx.barabasi_albert_graph(
            self.num_nodes, self.node_degree if self.node_degree is not None else 3
        )
        return nx.to_numpy_array(graph)

class RandomGeometric(GraphGenerator):
    """ """

    name = "random_geometric"

    def __init__(
        self,
        structure_name: str = "random_geometric",
        num_nodes: int = 50,
        target_mean_pathlength: float = None,
        target_mean_degree : float = None 

    ):
        super().__init__(structure_name=structure_name, num_nodes=num_nodes)
        self.target_mean_pathlength = target_mean_pathlength
        self.target_mean_degree = target_mean_degree

        self.optim_metric = "target_mean_degree" if target_mean_degree is not None else "target_mean_pathlength"
        self.initial_adj_matrix = nx.to_numpy_array(self.generate_calibrated_graph())

    def generate_graph(self, graph_edge_radius : float = 0.5) -> np.ndarray:
        
        return nx.random_geometric_graph(
            self.num_nodes, graph_edge_radius
        )
        
    def generate_calibrated_graph(self, 
        update_val : int = 5,
        max_iters : int = 50, 
        accepted_precision_percentage : float =  0.00025,
        graph_radius : float = 10,
    ): 
        #This is slow and I would like to optimise it 
        if self.optim_metric == "target_mean_degree":
            graph = self.generate_graph(graph_radius)
            for i in range(max_iters): 
                graph = self.generate_graph(np.absolute(graph_radius))
                graph_mean_degree = self._find_mean_degree(graph)
                if np.absolute(self.target_mean_degree - graph_mean_degree) < accepted_precision_percentage * self.num_nodes: 
                    return graph
                
                update_val = update_val/1.5
                if graph_mean_degree < self.target_mean_degree:
                    graph_radius += update_val
                elif graph_mean_degree > self.target_mean_degree:
                    graph_radius -= update_val
            return graph 

class SparseErdos(GraphGenerator):
    """ """

    name = "sparse_erdos"

    def __init__(self, structure_name: str = "sparse_erdos", num_nodes: int = 50):
        super().__init__(structure_name=structure_name, num_nodes=num_nodes)
        self.edge_prob = 0.01
        assert self.edge_prob >= 0 and self.edge_prob <= 1
        self.initial_adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import random

        return nx.to_numpy_array(
            nx.fast_gnp_random_graph(
                self.num_nodes, self.edge_prob, seed=None, directed=False
            )
        )


class GraphStructureGenerator(object):
    """ """

    def __init__(self, 
        structure_name: str, 
        num_nodes: int = 50, 
        **kwargs):

        self.num_nodes: int = num_nodes
        self.structure_name = structure_name

        self.target_mean_degree = kwargs.get("target_mean_degree", None)
        self.average_path_length = kwargs.get("average_path_length", None)
        
        assert len(kwargs)==1, \
            f"please pass only one from [node_degree, average_path_length] to build a random graph, {len(kwargs)} was passed"
        assert self.target_mean_degree is not None or self.average_path_length is not None,\
            f"to set internal graph parameters, a (single) metric from [target_mean_degree, average_path_length] should be supplied. {kwargs} was passed"
    
        self.initial_adj_matrix = self.get_graph_structure().initial_adj_matrix

    def get_graph_structure(self) -> np.ndarray:
        """ """
        structure_name = self.structure_name
        graph_mapping = {
            "barabasi_albert": BarabasiAlbert,
            #"configuration": ConfigurationGraph,
            "random_geometric": RandomGeometric,
            #"sparse_erdos": SparseErdos,
        }
        return graph_mapping[structure_name](
            num_nodes=self.num_nodes, 
            target_mean_pathlength = self.average_path_length, 
            target_mean_degree = self.target_mean_degree
        )


if __name__ == "__main__":
    #x = GraphStructureGenerator(structure_name="random_sparse", num_nodes=50, node_degree = 50)
    x = GraphStructureGenerator(structure_name = "barabasi_albert", num_nodes= 5000, target_mean_degree = 10)
    print(nx.from_numpy_array(x.initial_adj_matrix))