from adj_matrix_gen import GraphStructureGenerator
import numpy as np 
import pandas as pd 
import random 
import tqdm 
from typing import List, Tuple, Optional
import networkx as nx 

class ProceduralGraphGenerator(object):
    '''
    '''
    def __init__(self, initial_structure : np.ndarray, num_nodes : int = 100, num_agents : int = 1):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.initial_structure = initial_structure

    def _make_infection_array(self, largest_subcomponent : np.ndarray) -> np.ndarray:
        """
        Generates a 1D array of the length of the number of nodes and seeds it
        with num_agents number of initial infections
        """
        infection_array = np.zeros(self.num_nodes)

        infected_nodes = []
        nodepair_list = np.dstack(np.where(largest_subcomponent == 1))[0]
        while len(infected_nodes) < self.num_agents:

            infection_node = nodepair_list[random.randint(0, len(nodepair_list) - 1)][1]
            infected_nodes.append(infection_node)
            infection_array[infection_node] = 1

        return infection_array

    def _next_structure(self, updating_graph : np.ndarray) -> np.ndarray: 
        '''
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to 
        '''
        nodepair_list = np.dstack(np.where(self.initial_structure == 1))[0]
        nodepair_x, nodepair_y = nodepair_list[random.randint(0, len(nodepair_list) - 1)]
        updating_graph[[nodepair_x, nodepair_y]], updating_graph[[nodepair_y, nodepair_x]] = 1, 1
        return updating_graph


    def infect_till_saturation(
        self, infection_probability: float = 1, max_iters = 50
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

        graph = nx.from_numpy_array(self.initial_structure)
        nx_giant_graph = graph.subgraph(max(nx.connected_components(graph), key=len))
        giant_graph = nx.to_numpy_array(nx_giant_graph)
        print(self._make_infection_array(giant_graph))



        infection_matrix_list = [self.infection_matrix]
        timesteps_to_full_saturation = 0
        fraction_infected = []
        
        while (
            np.array_equal(infection_matrix_list[-1], np.ones(self.num_nodes)) is False
        ):
            timesteps_to_full_saturation += 1
            current_infection_matrix = infection_matrix_list[-1]
            
            #If dynamic graph structure like random sparse, get new adj matrix. If static, stay with the same
            if self.structure_name in self.dynamic_structures:
                adj_matrix = self.graph_generator.get_graph_structure().initial_adj_matrix
            else: 
                adj_matrix = self.graph_generator.initial_adj_matrix

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

            if timesteps_to_full_saturation == max_iters:
                break
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected 

if __name__ == "__main__":
    
    graphgen = GraphStructureGenerator(structure_name= "sparse_erdos", num_nodes= 100)
    graph = graphgen.initial_adj_matrix
    graph_rand = graphgen.get_graph_structure().initial_adj_matrix
    
    x = ProceduralGraphGenerator(graph)

    #print(graph_rand)
    modded_structure = x._next_structure(graph_rand)
    t = np.vstack(np.where(graph == 1))[0]
    y = np.vstack(np.where(modded_structure == 1))[0]

    assert len(t) < len(y)
    x.infect_till_saturation()

    graph = nx.from_numpy_array(x.initial_structure)
    print(type(graph))
    largest_cc = max(nx.connected_components(graph), key=len)
    print(graph)
    print(graph.subgraph(largest_cc))
