from  .adj_matrix_gen import GraphStructureGenerator
import numpy as np
import random
import tqdm
from typing import List, Tuple, Optional, Union
import networkx as nx
from scipy.stats import kstest, chisquare
import gc

class StatsUtils(object):
    """ """

    def __init__(self):
        pass

    def chisquared_reduced(x, y, degrees_freedom: int = 3):
        """ """
        chisquare_score = chisquare(x, y)
        reduced_chisquared = chisquare_score / (
            len(x) - degrees_freedom - 1
        )  # 3 degrees of freedom for the quartic fit hence -3 - 1
        return reduced_chisquared


class GraphStructureMutator(object):
    """
    Parameters:
        node_structure : a stack structure storing the lifetime of all edges in a given network so they can be removed
            after a given time. of shape [(adj_matrix_x_cord, adj_matrix_y_cord, timesteps_left_to_live)...]
    """

    def __init__(self, initial_structure: np.ndarray, edge_lifespan_mean : int = 10, use_probabilistic_edgelife : bool = False):
        self.initial_structure: str = initial_structure
        self.edge_lifespan_mean = edge_lifespan_mean
        self.use_probabilistic_edgelife = use_probabilistic_edgelife
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
        modality: str = "irreversable",
    ) -> np.ndarray:
        """
        sampling graph : graph to sample new edges from (largest component of structure)
        updating graph : graph to add new randomly chosen edges to
        """
        assert modality in [
            "irreversable",
            "reversable",
        ], f"Invalid structure modality passed : {modality}. Allowed types are irreversable, reversable"
        generated_edge_lifespan = self.edge_lifespan_mean

        if modality == "irreversable":
            generated_edge_lifespan = 1000000

        nodepair_list = np.dstack(np.where(sampling_graph == 1))[0]
        for _ in range(num_new_edges_per_timestep):
            if self.use_probabilistic_edgelife: 
                generated_edge_lifespan = np.random.normal(loc = generated_edge_lifespan, scale = generated_edge_lifespan/0.25)
        
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

    def __init__(self, initial_graph, num_nodes: int = 1000, num_agents: int = 1, edge_lifespan_mean : int = 10,use_probabilistic_edgelife : bool = False):
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.initial_graph = initial_graph
        self.structure_mutator = GraphStructureMutator(initial_structure=initial_graph, edge_lifespan_mean = edge_lifespan_mean, use_probabilistic_edgelife = use_probabilistic_edgelife)
        random.seed(1234)

    @staticmethod
    def _find_giant_structure(
        graph_structure: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        Helper method to find the giant graph of a adj matrix represented graph structure.

        Parameters:
            graph_structure (np.ndarray): 2D array representing the adjacency matrix of the input graph structure.
            verbose (bool): Flag to enable/disable printing of information. Default is True.

        Returns:
            np.ndarray: 2D array representing the adjacency matrix of the giant graph.

        """
        if isinstance(graph_structure, np.ndarray):
            graph_structure = nx.from_numpy_array(graph_structure)

        nx_giant_graph = graph_structure.subgraph(
            max(nx.connected_components(graph_structure), key=len)
        )
        if verbose:
            print(
                f"""graph structure properties : {nx_giant_graph} average degree {np.average([val for (node, val) in nx_giant_graph.degree()])}"""
            )
        degree_list = [val for (node, val) in nx_giant_graph.degree()]
        return nx_giant_graph, np.average(degree_list), degree_list

    @staticmethod
    def _find_network_closeness_centralities(
        graph: Union[np.ndarray, nx.classes.graph.Graph]
    ) -> dict:
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph)
        return graph.degree()


    @staticmethod
    def _find_network_node_positions(graph) -> dict:
        return nx.get_node_attributes(graph, "pos")

    @staticmethod
    def _find_farness(graph, node): 
        pass

    @staticmethod
    def _generate_network_statistics(
        graph: Union[np.ndarray, nx.classes.graph.Graph]
    ) -> dict:
        """ """
        info_dict = {}
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph)
        info_dict["clustering_coefficient"] = nx.average_clustering(graph)
        info_dict["degree_assortivity"] = nx.degree_pearson_correlation_coefficient(
            graph
        )
        info_dict["mean_shortest_pathlength"] = nx.average_shortest_path_length(graph)

        return info_dict

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

    def _find_reachability_matrix(
        self,
        input_graph: np.ndarray,
    ) -> np.ndarray:
        """ """
        graph = nx.from_numpy_array(input_graph)
        assert input_graph.shape[0] == input_graph.shape[1]

        reachability_arrays = []
        for _ in range(input_graph.shape[0]):
            reachability_array = np.zeros(input_graph.shape[0])
            for (
                reachable_path_idx,
                path_length,
            ) in nx.single_target_shortest_path_length(graph, _, cutoff=None):
                reachability_array[reachable_path_idx] = path_length
            reachability_arrays.append(reachability_array)

        final_matrix = np.vstack(reachability_arrays)
        final_matrix[final_matrix == 0] = np.inf

        np.fill_diagonal(final_matrix, 0)
        return final_matrix

    def _make_infection_array(
        self,
        largest_subcomponent,
        structure_type: str,
        desired_agent_degree: int = 5,
    ) -> np.ndarray:
        """
        Generates a 1D array of the length of the number of nodes and seeds it
        with num_agents number of initial infections with the agents in the largest
        """
        subcomponent_adj_matrix = nx.to_numpy_array(largest_subcomponent)
        infection_arr = np.zeros(subcomponent_adj_matrix.shape[0])
        fully_saturated_arr = np.ones(subcomponent_adj_matrix.shape[0])

        # Compute the degree of all nodes in network, reorder closeness dict by closest to desired_agent_closeness_centrality and chose first N
        if structure_type == "barabasi_albert":
            node_degree = self._find_network_closeness_centralities(
                largest_subcomponent
            )
            reordered_list = sorted(node_degree,key=lambda x: x[1], reverse=True)
            infection_nodes = [x[0] for x in reordered_list[:self.num_agents]]
        # Find position of all nodes in network, find the closest to the 'centre', seed the initial agents as those closest to 0,0
        
        elif structure_type == "random_geometric":
            node_positions = self._find_network_node_positions(largest_subcomponent)
            distances = []
            for n in node_positions:
                x, y = node_positions[n]
                distances.append((x - 0.5) ** 2 + (y - 0.5) ** 2)
            infection_nodes = np.argsort(distances).tolist()[: self.num_agents]

        for node in infection_nodes:
            infection_arr[node] = 1

        return infection_arr, fully_saturated_arr

    def infect_till_saturation(
        self,
        structure_name: str,
        infection_probability: float = 1,
        max_iters: int = 15000,
        modality: str = "irreversable",
        sample_giant : bool = True,
        store_infectivity_list : bool = True,
        verbose: bool = True,
        return_components: bool = False,

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
        (
            fraction_infected,
            infection_matrix_list,
            average_reachability,
            timesteps_to_full_saturation,
        ) = ([], [], [], 0)
        largest_active_component = []
        largest_inactive_component = []
        if verbose:
            pbar = tqdm.tqdm(total=max_iters)
        # Generate the giant graph as our initial structure from our 'choosing' structure
        # Generate the infected nodes list and the initial infection graph structure.
        giant_graph, average_degree, degree_list = self._find_giant_structure(
            self.initial_graph, verbose=verbose
        )
        infection_arr, fully_saturated_arr = self._make_infection_array(
            giant_graph, structure_name
        )
        initial_graph = self._make_initial_structure(
            giant_graph=nx.to_numpy_array(giant_graph)
        )

        giant_graph = nx.to_numpy_array(giant_graph)
        infection_arr_list = [infection_arr]
        while np.array_equal(infection_arr_list[-1], fully_saturated_arr) is False:
            if verbose:
                pbar.update(1)
            if timesteps_to_full_saturation == max_iters:
                break            # Update timesteps and take current infection array
            timesteps_to_full_saturation += 1
            current_infection_arr = infection_arr_list[-1]
            # Update the graph structure to infect a new node
            if sample_giant:
                graph_structure = self.structure_mutator._next_structure(
                    sampling_graph=giant_graph,
                    updating_graph=initial_graph,
                    modality=modality,
                )
            else:
                graph_structure = giant_graph

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

            if store_infectivity_list:
                infection_matrix_list.append(current_infection_arr)
            if return_components:
                # nice and slow
                comps = list(nx.connected_components(nx.from_numpy_array(graph_structure)))
                on = []
                off = []
                for c in comps:
                    c = list(c)
                    if all(map(lambda x: current_infection_arr[x], c)):
                        on.append(c)
                    elif not any(map(lambda x: current_infection_arr[x], c)):
                        off.append(c)
                largest_active_component.append(len(max(on, key=len)) / self.num_nodes if len(on) else 0.0)
                largest_inactive_component.append(len(max(off, key=len)) / self.num_nodes if len(off) else 0.0)
                # Size of largest component that has all nodes infected
                #largest_active_component.append()
            fraction_infected.append(
                np.count_nonzero(current_infection_arr == 1)
                / len(current_infection_arr)
            )
        info_dict = {
            "average_degree": average_degree,
            "num_nodes": len(current_infection_arr),
            "modality": modality,
            "degree_list": degree_list,
        }
        info_dict.update(self._generate_network_statistics(giant_graph))
        
        if return_components:
            return (
            infection_matrix_list,
            timesteps_to_full_saturation,
            fraction_infected,
            info_dict,
            largest_active_component,
            largest_inactive_component
            )
        else:
            return (
                infection_matrix_list,
                timesteps_to_full_saturation,
                fraction_infected,
                info_dict,
            )


if __name__ == "__main__":
    graphgen = GraphStructureGenerator(
        structure_name="barabasi_albert",
        num_nodes=500,
        target_mean_degree = 5.0
    )
    graph = graphgen.initial_graph
    print(graph)
    x = ProceduralGraphGenerator(graph)
    q = x.infect_till_saturation("barabasi_albert", sample_giant= False, infection_probability=1.0, store_infectivity_list = False, verbose=True, modality="irreversable", return_components=False)
    print(q)
    #print(q[-1])
    #print(q[2])
