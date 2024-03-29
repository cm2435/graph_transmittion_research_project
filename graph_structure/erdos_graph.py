from typing import Tuple
import numpy as np
import random
import tqdm
import multiprocessing
import scipy
import pandas as pd
import itertools
from typing import List

# from structure_generation.adj_matrix_gen import GraphStructureGenerator
from structure_generation.adj_matrix_gen import GraphStructureGenerator

import numpy as np
import random
import abc
import networkx as nx
from typing import Type


class ErdosGraphSimulator(object):
    """ """

    def __init__(
        self,
        num_nodes: int = 250,
        num_agents: int = 1,
        num_timestep_edges: int = 5,
        structure_name: str = "fully_connected",
    ):

        self.structure_name = structure_name
        self.num_nodes: int = num_nodes
        self.num_agents: int = num_agents
        self.num_timestep_edges: int = num_timestep_edges
        self.graph_generator: GraphStructureGenerator = GraphStructureGenerator(
            structure_name=structure_name, num_nodes=num_nodes
        )
        self.dynamic_structures = ["random_sparse"]

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
        self, infection_probability: float = 1, max_iters=50
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

            # If dynamic graph structure like random sparse, get new adj matrix. If static, stay with the same
            if self.structure_name in self.dynamic_structures:
                adj_matrix = (
                    self.graph_generator.get_graph_structure().initial_adj_matrix
                )
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
            fraction_infected.append(
                np.count_nonzero(current_infection_matrix == 1)
                / len(current_infection_matrix)
            )

            if timesteps_to_full_saturation == max_iters:
                break
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected


if __name__ == "__main__":
    x = ErdosGraphSimulator(structure_name="random_geometric")
    print(x.structure_name)
    t, q, r = x.infect_till_saturation()
    print(q)
