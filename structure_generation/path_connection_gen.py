from .adj_matrix_gen import GraphStructureGenerator
import numpy as np
import pandas as pd
import random
import tqdm
from typing import List, Tuple, Optional
import networkx as nx
import gc
import os
from scipy.optimize import curve_fit
from scipy.stats import kstest, chisquare, ks_2samp, epps_singleton_2samp


class StatsUtils(object):
    """
    """
    def __init__(self):
        pass

    def chisquared_reduced(x, y, degrees_freedom : int = 3):
        '''
        '''
        chisquare_score = chisquare(x, y)
        reduced_chisquared = chisquare_score / (len(x) - degrees_freedom -1) # 3 degrees of freedom for the quartic fit hence -3 - 1 
        return reduced_chisquared


class GraphStructureMutator(object):
    """
    Parameters:
        node_structure : a stack structure storing the lifetime of all edges in a given network so they can be removed
            after a given time. of shape [(adj_matrix_x_cord, adj_matrix_y_cord, timesteps_left_to_live)...]
    """

    def __init__(self, initial_structure: np.ndarray):
        self.initial_structure: str = initial_structure
        self.edge_structure: List[Tuple[int, int, int]] = []

    def _remove_stale_edges(
        self, updating_adj_matrix: np.ndarray, update_timestep: bool = True
    ) -> np.ndarray:
        """ """
        if update_timestep:
            self.edge_structure = list(
                map(lambda x: (x[0], x[1], x[2] - 1), self.edge_structure)
            )

        edges_to_pop = [x for x in self.edge_structure if x[2] == 0]
        for edge_pair in edges_to_pop:
            # TODO figure out why the alternate doesn't work as intended, sets saturation behaviour to zero
            updating_adj_matrix[edge_pair[0]][edge_pair[1]] = 0
            updating_adj_matrix[edge_pair[1]][edge_pair[0]] = 0

        self.edge_structure = [x for x in self.edge_structure if x[2] != 0]
        return updating_adj_matrix

    def _next_structure(
        self,
        sampling_graph: np.ndarray,
        updating_graph: np.ndarray,
        num_new_edges_per_timestep: int = 2,
        generated_edge_lifespan: int = 100 ,
        modality: str = "saturation",
    ) -> np.ndarray:
        """
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to
        """
        assert modality in [
            "saturation",
            "causal",
        ], f"Invalid structure modality passed : {modality}. Allowed types are saturation, causal"
        if modality == "saturation":
            generated_edge_lifespan = 2147483647

        nodepair_list = np.dstack(np.where(sampling_graph == 1))[0]
        for _ in range(num_new_edges_per_timestep):
            nodepair_x, nodepair_y = nodepair_list[
                random.randint(0, len(nodepair_list) - 1)
            ]
            (
                updating_graph[nodepair_x][nodepair_y],
                updating_graph[nodepair_y][[nodepair_x]],
            ) = (1, 1)
            self.edge_structure.append(
                (nodepair_x, nodepair_y, generated_edge_lifespan)
            )
        updating_graph = self._remove_stale_edges(updating_adj_matrix=updating_graph)

        return updating_graph


class ProceduralGraphGenerator(object):
    """
    Class to generate a graph structure, infect it with agents, and spread the infection till saturation.

    Parameters:
        initial_structure (np.ndarray): 
            2D array representing the adjacency matrix of the initial graph structure.
        num_nodes (int): 
            Number of nodes in the graph structure. Default is 200.
        num_agents (int): 
            Number of agents that are initially infected. Default is 1.
        
    Attributes:
        num_nodes (int): 
            Number of nodes in the graph structure.
        num_agents (int): 
            Number of agents that are initially infected.
        initial_structure (np.ndarray): 
            2D array representing the adjacency matrix of the initial graph structure.
        structure_mutator (GraphStructureMutator): 
            Instance of GraphStructureMutator with the `initial_structure` as the initial structure.
    """

    def __init__(
        self, initial_structure: np.ndarray, num_nodes: int = 500, num_agents: int = 1
    ):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.initial_structure = initial_structure
        self.structure_mutator = GraphStructureMutator(
            initial_structure=initial_structure
        )

    @staticmethod
    def _find_giant_structure(
        graph_structure: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        """
        Helper method to find the giant graph of a adj matrix represented graph structure.
        
        Parameters:
            graph_structure (np.ndarray): 2D array representing the adjacency matrix of the input graph structure.
            verbose (bool): Flag to enable/disable printing of information. Default is True.
            
        Returns:
            np.ndarray: 2D array representing the adjacency matrix of the giant graph.
            
        """
        graph = nx.from_numpy_array(graph_structure)
        nx_giant_graph = graph.subgraph(max(nx.connected_components(graph), key=len))
        if verbose:
            print(f"""graph structure properties : {nx_giant_graph} average degree {np.average([val for (node, val) in nx_giant_graph.degree()])}""")
        giant_graph = nx.to_numpy_array(nx_giant_graph)

        return giant_graph, np.average([val for (node, val) in nx_giant_graph.degree()])

    def _make_initial_structure(self, giant_graph: np.ndarray) -> np.ndarray:
        """
        Method to generate the initial graph structure by adding a random edge to the `giant_graph`.
        
        Parameters:
            giant_graph (np.ndarray): 2D array representing the adjacency matrix of the giant graph.
            
        Returns:
            np.ndarray: 2D array representing the adjacency matrix of the initial graph.
            
        """
        initial_graph = np.zeros((self.num_nodes, self.num_nodes))
        edges = np.dstack(np.where(giant_graph == 1))[0]
        random_edge_x, random_edge_y = edges[random.randint(0, len(edges) - 1)]
        (
            initial_graph[random_edge_x][random_edge_y],
            initial_graph[random_edge_y][random_edge_x],
        ) = (1, 1)

        return initial_graph

    def _find_reachability_matrix(self,
        input_graph : np.ndarray, 
    ) -> np.ndarray: 
        """
        """
        graph = nx.from_numpy_array(input_graph)
        assert input_graph.shape[0] == input_graph.shape[1]
        
        reachability_arrays = []
        for _ in range(input_graph.shape[0]):
            reachability_array = np.zeros(input_graph.shape[0])   
            for reachable_path_idx, path_length in nx.single_target_shortest_path_length(graph, _, cutoff=None):
                reachability_array[reachable_path_idx] = path_length
            reachability_arrays.append(reachability_array)
    
        final_matrix = np.vstack(reachability_arrays)
        final_matrix[final_matrix == 0] = np.inf

        np.fill_diagonal(final_matrix, 0)
        return final_matrix

    def _make_infection_array(self, largest_subcomponent: np.ndarray) -> np.ndarray:
        """
        Generates a 1D array of the length of the number of nodes and seeds it
        with num_agents number of initial infections with the agents in the largest
        """
        infected_nodes = []
        nodepair_list = np.dstack(np.where(largest_subcomponent == 1))[0]
        infection_arr = np.zeros(largest_subcomponent.shape[0])
        fully_saturated_arr = np.ones(largest_subcomponent.shape[0])

        while len(infected_nodes) < self.num_agents:
            infection_node = nodepair_list[random.randint(0, len(nodepair_list) - 1)][1]
            infected_nodes.append(infection_node)
            infection_arr[infection_node] = 1

        return infection_arr, fully_saturated_arr

    def infect_till_saturation(
        self,
        infection_probability: float = 1,
        max_iters: int = 2000,
        modality: str = "saturation",
        verbose : bool = True
    ) -> Tuple[List[np.ndarray], int, List[float]]:
        """
        Method to spread the infection till saturation in the graph.

        Parameters:
            infection_probability (float): Probability of infection spread between nodes.
            num_iterations (int): Number of iterations to run the infection spread.
            verbose (bool): Flag to enable/disable printing of information. Default is True.
            
        Returns:
            tuple:
                infection_iteration_array (list): 
                    List of dictionaries with keys as the node indices and values as the infection status (0 or 1) at each iteration.
                fully_saturated_iteration_array (list):
                    List of dictionaries with keys as the node indices and values as 1, indicating that the node is present in the graph at each iteration.
                success_iteration (int): 
                    The iteration number where the infection spread has reached saturation or -1 if the infection did not reach saturation.
        """
        fraction_infected, infection_matrix_list, average_reachability, timesteps_to_full_saturation = (
            [],
            [],
            [],
            0,
        )
        if verbose: 
            pbar = tqdm.tqdm(total=max_iters)
        #Generate the giant graph as our initial structure from our 'choosing' structure
        #Generate the infected nodes list and the initial infection graph structure. 
        giant_graph, average_degree = self._find_giant_structure(self.initial_structure)
        infection_arr, fully_saturated_arr = self._make_infection_array(giant_graph)
        initial_graph = self._make_initial_structure(giant_graph)

        infection_arr_list = [infection_arr]
        while np.array_equal(infection_arr_list[-1], fully_saturated_arr) is False:
            #Update timesteps and take current infection array
            timesteps_to_full_saturation += 1
            current_infection_arr = infection_arr_list[-1]
            #print(current_infection_arr.shape, current_infection_arr, np.count_nonzero(current_infection_arr == 1))
            import time
            #time.sleep(0.1)
            #Update the graph structure to infect a new node
            graph_structure = self.structure_mutator._next_structure(
                sampling_graph=giant_graph,
                updating_graph=initial_graph,
                modality=modality,
            )
            nodepair_list = np.dstack(np.where(graph_structure == 1))[0]
            for pair in nodepair_list:
                if (
                    current_infection_arr[pair[0]]
                    or current_infection_arr[pair[1]] == 1
                ):
                    # Do not always guarrentee infection
                    infection_outcome = np.random.choice(
                        [0, 1], p=[1 - infection_probability, infection_probability]
                    )
                    if infection_outcome == 1:
                        (
                            current_infection_arr[pair[0]],
                            current_infection_arr[pair[1]],
                        ) = (1, 1)

            infection_matrix_list.append(current_infection_arr)
            fraction_infected.append(
                np.count_nonzero(current_infection_arr == 1)
                / len(current_infection_arr)
            )
            if verbose: 
                pbar.update(1)
            if timesteps_to_full_saturation == max_iters:
                break

        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected, average_degree
