from typing import Tuple
import numpy as np
import random
import tqdm
import multiprocessing
import scipy
import pandas as pd
import itertools
from typing import List 

#from structure_generation.adj_matrix_gen import GraphStructureGenerator
#from ..structure_generation.adj_matrix_gen import GraphStructureGenerator

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
        self.adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import networkx as nx
        graph = nx.barabasi_albert_graph(self.num_nodes, 5)
        return nx.to_numpy_array(graph)


class ConfigurationGraph(GraphGenerator):
    '''
    '''
    name = "configuration"
    def __init__(self, num_nodes : int, structure_name : str = "configuration"):
        super(ConfigurationGraph, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.adj_matrix = self.generate_adj_matrix()
    
    def generate_adj_matrix(self) -> np.ndarray:
        import networkx as nx
        sequence = nx.random_powerlaw_tree_sequence(self.num_nodes,tries=5000)
        graph = nx.configuration_model(sequence)
        return nx.to_numpy_array(graph)


class RandomSparse(GraphGenerator):
    '''
    '''
    name = "random_sparse"
    def __init__(self, num_nodes : int, structure_name : str = "random_sparse"):
        super(RandomSparse, self).__init__(structure_name= structure_name, num_nodes = num_nodes)
        self.structure_name = structure_name
        self.adj_matrix = self.generate_adj_matrix()

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
        self.adj_matrix = self.generate_adj_matrix()

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
        self.adj_matrix = self.generate_adj_matrix()

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

        self.adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self) -> np.ndarray:
        import random
        return nx.to_numpy_array(nx.fast_gnp_random_graph(self.num_nodes, self.edge_prob, seed=None, directed=False))

class GraphStructureGenerator(object):
    """ """

    def __init__(self, structure_name: str, num_nodes: int = 20):
        self.structure_name: str = structure_name
        self.num_nodes: int = num_nodes
        self.allowed_structures: list[str] = ["fully_connected", "random_sparse", "barabasi_albert", "configuration", "random_geometric", "sparse_erdos"]

    @property
    def adj_matrix(self) -> np.ndarray:
        """ """
        graph_mapping = {
            "fully_connected": FullyConnected(num_nodes= self.num_nodes),
            "random_sparse":  RandomSparse(num_nodes= self.num_nodes),
            "barabasi_albert" : BarabasiAlbert(num_nodes= self.num_nodes),
            "configuration" : ConfigurationGraph(num_nodes= self.num_nodes),
            "random_geometric" : RandomGeometric(num_nodes= self.num_nodes),
            "sparse_erdos" : SparseErdos(num_nodes= self.num_nodes)
        }
        return graph_mapping[self.structure_name].adj_matrix

#if __name__ == "__main__":
#    x = RandomSparse(structure_name= "random_sparse", num_nodes= 50)


class ErdosGraphSimulator(object):
    """ """

    def __init__(
        self, num_nodes: int = 250, 
        num_agents: int = 1,
        num_timestep_edges: int = 5,
        structure_name : str = "sparse_erdos"
    ):

        self.num_nodes: int = num_nodes
        self.num_agents: int = num_agents
        self.num_timestep_edges: int = num_timestep_edges
        self.graph_generator: GraphStructureGenerator = GraphStructureGenerator(structure_name= structure_name, num_nodes= num_nodes)


    @property
    def infection_matrix(self) -> np.ndarray:
        """
        Generates a 1D array of the length of the number of nodes and seeds it
        with num_agents number of initial infections
        """
        infection_array = np.zeros(self.num_nodes)
        infected_nodes = []

        while len(infected_nodes) < self.num_agents:
            random_idx = random.randint(0, self.num_nodes - 1)
            if random_idx not in infected_nodes:
                infected_nodes.append(random_idx)
                infection_array[random_idx] = 1

        return infection_array


    def infect_till_saturation(
        self, infection_probability: float = 1
    ) -> Tuple[List[np.ndarray], int, List[float]]:
        """
        Procedure to measure time to infection saturation for a given set of initial conditions
        in a graph structure.

        Procedure:
        1. Randomly sample an adjacency matrix
        2. If any node edge pairs in the adj matrix are infected, infect their pair with p = infection_probability
        3. Update the infection matrix with any newly infected nodes by index and update timestep

        4. Iterate 1,2,3 untill infection matrix is saturated, then log the number of timesteps needed

        """
        infection_matrix_list = [self.infection_matrix]
        timesteps_to_full_saturation = 0
        fraction_infected = []
        while (
            np.array_equal(infection_matrix_list[-1], np.ones(self.num_nodes)) is False
        ):
            timesteps_to_full_saturation += 1
            current_infection_matrix = infection_matrix_list[-1]

            #adj_matrix = self.generate_adj_matrix()
            adj_matrix = self.graph_generator.adj_matrix
            nodepair_list = np.dstack(np.where(adj_matrix == 1))[0]

            for pair in nodepair_list:
                if (
                    current_infection_matrix[pair[0]]
                    or current_infection_matrix[pair[1]] == 1
                ):
                    # Do not always guarrentee infection
                    infection_outcome = np.random.choice(
                        [0, 1], p=[1 - infection_probability, infection_probability]
                    )
                    if infection_outcome == 1:
                        (
                            current_infection_matrix[pair[0]],
                            current_infection_matrix[pair[1]],
                        ) = (1, 1)

            infection_matrix_list.append(current_infection_matrix)
            fraction_infected.append(np.count_nonzero(current_infection_matrix == 1) / len(current_infection_matrix))
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected 


    def infect_until_saturation_variable_timesteps(
        self, infection_probability: float = 1
    ) -> Tuple[List[np.ndarray], int, List[float]]:
        """
        Procedure to measure time to infection saturation for a given set of initial conditions
        in a graph structure.

        Procedure:
        1. Randomly sample an adjacency matrix
        2. If any node edge pairs in the adj matrix are infected, infect their pair with p = infection_probability
        3. Update the infection matrix with any newly infected nodes by index and update timestep

        4. Iterate 1,2,3 untill infection matrix is saturated, then log the number of timesteps needed

        """
        initial_infection_matrix = np.zeros((self.num_nodes, self.num_nodes))
        #print(initial_infection_matrix)
        infection_matrix_list = [self.infection_matrix]
        timesteps_to_full_saturation = 0
        fraction_infected = []
        while (
            np.array_equal(infection_matrix_list[-1], np.ones(self.num_nodes)) is False
        ):
            timesteps_to_full_saturation += 1
            current_infection_matrix = infection_matrix_list[-1]
            print(current_infection_matrix[-1] == current_infection_matrix[-2])
            #adj_matrix = self.generate_adj_matrix()
            adj_matrix = self.graph_generator.adj_matrix
            nodepair_list = np.dstack(np.where(adj_matrix == 1))[0]

            for pair in nodepair_list:
                if (
                    current_infection_matrix[pair[0]]
                    or current_infection_matrix[pair[1]] == 1
                ):
                    # Do not always guarrentee infection
                    infection_outcome = np.random.choice(
                        [0, 1], p=[1 - infection_probability, infection_probability]
                    )
                    if infection_outcome == 1:
                        (
                            current_infection_matrix[pair[0]],
                            current_infection_matrix[pair[1]],
                        ) = (1, 1)

            infection_matrix_list.append(current_infection_matrix)
            fraction_infected.append(np.count_nonzero(current_infection_matrix == 1) / len(current_infection_matrix))
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected 



if __name__ == "__main__":
    x = ErdosGraphSimulator()
    t,q,r = x.infect_until_saturation_variable_timesteps()
    print(q)