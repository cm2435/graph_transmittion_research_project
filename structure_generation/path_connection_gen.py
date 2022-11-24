from adj_matrix_gen import GraphStructureGenerator
import numpy as np
import pandas as pd
import random
import tqdm
from typing import List, Tuple, Optional
import networkx as nx
import gc
import os


class GraphStructureMutator(object):
    """
    Parameters:
        node_structure : a stack structure storing the lifetime of all edges in a given network so they can be removed
            after a given time. of shape [(adj_matrix_x_cord, adj_matrix_y_cord, timesteps_left_to_live)...]
    """

    def __init__(self, initial_structure: np.ndarray):
        self.initial_structure = initial_structure
        self.edge_structure: List[Tuple[int, int, int]] = []

    def _remove_stale_edges(self, update_timestep: bool = True):
        """ """
        if update_timestep:
            self.edge_structure = list(
                map(lambda x: (x[0], x[1], x[2] - 1), self.edge_structure)
            )

        self.edge_structure = [x for x in self.edge_structure if x[2] != 0]
        pass

    def _next_structure_saturation(
        self,
        sampling_graph: np.ndarray,
        updating_graph: np.ndarray,
        num_new_edges_per_timestep: int = 1,
    ) -> np.ndarray:
        """
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to
        """
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
                (nodepair_x, nodepair_y, 2147483647)
            )  # For saturation model, timesteps to edge removal should be ostensibly inf

        self._remove_stale_edges()
        return updating_graph

    def _next_structure_causal(
        self,
        sampling_graph: np.ndarray,
        updating_graph: np.ndarray,
        num_new_edges_per_timestep: int = 1,
        generated_edge_lifespan: int = 5,
    ) -> np.ndarray:
        """
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to
        """
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
            )  # For saturation model, timesteps to edge removal should be ostensibly inf

        self._remove_stale_edges()
        return updating_graph


class ProceduralGraphGenerator(object):
    """ """

    def __init__(
        self, initial_structure: np.ndarray, num_nodes: int = 200, num_agents: int = 1
    ):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.initial_structure = initial_structure

    def _make_initial_structure(self, giant_graph: np.ndarray):
        """ """
        initial_graph = np.zeros((self.num_nodes, self.num_nodes))
        edges = np.dstack(np.where(giant_graph == 1))[0]
        random_edge_x, random_edge_y = edges[random.randint(0, len(edges) - 1)]
        (
            initial_graph[random_edge_x][random_edge_y],
            initial_graph[random_edge_y][random_edge_x],
        ) = (1, 1)

        return initial_graph

    def _make_infection_array(self, 
        largest_subcomponent: np.ndarray
        ) -> np.ndarray:
        """
        Generates a 1D array of the length of the number of nodes and seeds it
        with num_agents number of initial infections with the agents in the largest
        """
        infected_nodes = []
        nodepair_list = np.dstack(np.where(largest_subcomponent == 1))[0]
        infection_arr = {k: 0 for k in set([x[0] for x in nodepair_list])}
        fully_saturated_arr = {k: 1 for k in set([x[0] for x in nodepair_list])}

        while len(infected_nodes) < self.num_agents:
            infection_node = nodepair_list[random.randint(0, len(nodepair_list) - 1)][1]
            infected_nodes.append(infection_node)
            infection_arr[infection_node] = 1

        return infection_arr, fully_saturated_arr

    def _next_structure_saturation(
        self,
        sampling_graph: np.ndarray,
        updating_graph: np.ndarray,
        num_new_edges_per_timestep: int = 1,
    ) -> np.ndarray:
        """
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to
        """
        nodepair_list = np.dstack(np.where(sampling_graph == 1))[0]

        for _ in range(num_new_edges_per_timestep):
            nodepair_x, nodepair_y = nodepair_list[
                random.randint(0, len(nodepair_list) - 1)
            ]
            (
                updating_graph[nodepair_x][nodepair_y],
                updating_graph[nodepair_y][[nodepair_x]],
            ) = (1, 1)

        return updating_graph

    def infect_till_saturation(
        self, 
        infection_probability: float = 0.05,
        max_iters=2000
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
        print(f"graph structure with properties{nx_giant_graph}")
        giant_graph = nx.to_numpy_array(nx_giant_graph)
        infection_dict, fully_saturated_dict = self._make_infection_array(giant_graph)

        infection_dict_list = [infection_dict]
        timesteps_to_full_saturation = 0
        fraction_infected, infection_matrix_list = [], []

        # Make the initial edge structure
        initial_graph = self._make_initial_structure(giant_graph)

        while infection_dict_list[-1] != fully_saturated_dict:
            timesteps_to_full_saturation += 1
            current_infection_dict = infection_dict_list[-1]

            graph_structure = self._next_structure_saturation(
                sampling_graph=giant_graph,
                updating_graph=initial_graph,
            )
            nodepair_list = np.dstack(np.where(graph_structure == 1))[0]
            for pair in nodepair_list:
                if (
                    current_infection_dict[pair[0]]
                    or current_infection_dict[pair[1]] == 1
                ):
                    # Do not always guarrentee infection
                    infection_outcome = np.random.choice(
                        [0, 1], p=[1 - infection_probability, infection_probability]
                    )
                    if infection_outcome == 1:
                        (
                            current_infection_dict[pair[0]],
                            current_infection_dict[pair[1]],
                        ) = (1, 1)

            infection_matrix_list.append(current_infection_dict)
            fraction_infected.append(
                sum(value == 1 for value in current_infection_dict.values())
                / len(current_infection_dict)
            )

            # print(graph_structure)
            if timesteps_to_full_saturation == max_iters:
                break
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected


if __name__ == "__main__":
    global num_edges_per_timestep
    num_edges_per_timestep = 10

    # for structure_name in ["fully_connected", "random_sparse", "barabasi_albert", "configuration", "random_geometric", "sparse_erdos"]:
    for structure_name in ["configuration"]:
        import matplotlib.pyplot as plt

        graphgen = GraphStructureGenerator(structure_name=structure_name, num_nodes=200)
        graph = graphgen.initial_adj_matrix
        graph_rand = graphgen.get_graph_structure().initial_adj_matrix

        mutator = GraphStructureMutator(graph)
        x = ProceduralGraphGenerator(graph)

        mutator._next_structure_saturation(graph, graph_rand)

        test = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]

        print(list(map(lambda x: (x[0], x[1], x[2] - 1), test)))
        # q,r, t = x.infect_till_saturation()

        """fig, ax = plt.subplots()
        ax.plot([x for x in range(len(t))], t)

        fp = f"/home/cm2435/Desktop/university_final_year_cw/data/figures_sequential_choose_{num_edges_per_timestep}"
        if os.path.isdir(fp) is False: 
            os.makedirs(fp)
        fig.savefig(f"{fp}/{structure_name}.png")

        del plt
        gc.collect()"""
