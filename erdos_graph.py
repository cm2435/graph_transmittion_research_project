from typing import Tuple
import numpy as np
import random
import tqdm
import multiprocessing


class ErdosGraphSimulator(object):
    """ """

    def __init__(
        self, num_nodes: int = 20, num_agents: int = 3, num_timestep_edges: int = 4
    ):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.num_timestep_edges = num_timestep_edges

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

    def generate_matrix(self, graph_type : str):
        graph_types = {
            "random" : self.infection_matrix()
        }  
        return graph_types

    def generate_adj_matrix(self) -> np.ndarray:
        """
        Generate a random num_node X num_node adjacency matrix that is seeded with
        num_timestep_edges connections in another wise sparse graph.
        """
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(self.num_timestep_edges):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(
                0, self.num_nodes - 1
            )
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph

    def infect_till_saturation(
        self, infection_probability: float = 1
    ) -> Tuple[np.ndarray, int]:
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
        timesteps = 0

        while (
            np.array_equal(infection_matrix_list[-1], np.ones(self.num_nodes)) == False
        ):
            timesteps += 1
            current_infection_matrix = infection_matrix_list[-1]
            
            adj_matrix = self.generate_adj_matrix()
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

        return infection_matrix_list, timesteps


def simulate_saturation(_=1):
    # Global function just for the sake of making multiprocessing nice and simple
    x = ErdosGraphSimulator()
    _, iterations = x.infect_till_saturation()
    return iterations


if __name__ == "__main__":
    convergence_steps = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2 - 1) as p:
        with tqdm.tqdm(total=10000) as pbar:
            for _ in p.imap_unordered(simulate_saturation, range(0, 10000)):
                pbar.update()
                convergence_steps.append(_)
    
    ##print(convergence_steps, "\n")
    print(np.average(convergence_steps))



