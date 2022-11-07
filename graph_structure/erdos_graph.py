from typing import Tuple
import numpy as np
import random
import tqdm
import multiprocessing
import scipy
import pandas as pd
import itertools


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
        Generate a random num_node X num_node adjacency matrix that is seeded with
        num_timestep_edges connections in another wise sparse graph.
        """
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(num_edges):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(
                0, self.num_nodes - 1
            )
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph


class ErdosGraphSimulator(object):
    """ """

    def __init__(
        self,
        num_nodes: int = 100,
        num_agents: int = 3,
        num_timestep_edges: int = 4,
        structure_name: str = "fully_connected",
    ):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.num_timestep_edges = num_timestep_edges

        self.graph_generator = GraphStructureGenerator(
            structure_name=structure_name, num_nodes=num_nodes
        )

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
        timesteps_to_full_saturation = 0
        fraction_infected = []
        while (
            np.array_equal(infection_matrix_list[-1], np.ones(self.num_nodes)) is False
        ):
            timesteps_to_full_saturation += 1
            current_infection_matrix = infection_matrix_list[-1]

            # adj_matrix = self.generate_adj_matrix()
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
            fraction_infected.append(
                np.count_nonzero(current_infection_matrix == 1)
                / len(current_infection_matrix)
            )
        return infection_matrix_list, timesteps_to_full_saturation, fraction_infected


def simulate_saturation(_=1):
    # Global function just for the sake of making multiprocessing nice and simple
    x = ErdosGraphSimulator(
        num_nodes=num_nodes,
        num_agents=num_initial_agents,
        structure_name=structure_name,
    )
    _, iterations, fraction_infected = x.infect_till_saturation()
    return iterations, fraction_infected


if __name__ == "__main__":
    final_dicts = []
    global num_initial_agents, num_nodes, structure_name
    num_initial_agents, num_nodes, structure_name = 1, 20, "fully_connected"
    simulation_output = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2 - 1) as p:
        num_simulation_steps = 1000
        with tqdm.tqdm(total=num_simulation_steps) as pbar:
            for _ in p.imap_unordered(
                simulate_saturation, range(0, num_simulation_steps)
            ):
                pbar.update()
                simulation_output.append(_)

    convergence_steps = [x[0] for x in simulation_output]
    saturation_fractions = [x[1] for x in simulation_output]
    print(max([len(x) for x in saturation_fractions]))

    padded_list = list(zip(*itertools.zip_longest(*saturation_fractions, fillvalue=1)))
    saturation_timestep = np.mean(padded_list, axis=0)

    print(saturation_timestep)

    stats_dict = {
        "mean": np.average(convergence_steps),
        "variance": np.var(convergence_steps),
        "skew": scipy.stats.skew(convergence_steps),
        "kurtosis": scipy.stats.kurtosis(convergence_steps),
        "num_nodes": num_nodes,
    }
    print(stats_dict)

    """df = pd.DataFrame.from_dict({f"num_agents_{num_initial_agents}": stats_dict})
    df.to_csv(
        f"/home/cm2435/Desktop/university_final_year_cw/data/stats/{num_initial_agents}.csv"
    )"""
