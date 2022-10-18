import numpy as np
import random


class GraphHolder(object):
    """
    Simple class to hold logic for the generation of graph structures
    and methods to model various virology effects on them

    Attrs:
        num_nodes - int:
            initial number of nodes for simulation with which to generate the adjacency matrix
        num_initial_infections - int:
            the number of nodes in the matrix chosen at random for t=0 to set = 1

    """

    def __init__(self, num_nodes: int = 20, num_initial_infections: int = 5):
        self.num_nodes = num_nodes
        self.num_initial_infections = num_initial_infections
        self.num_iterations = 0

    @property
    def initial_graph(self):
        """
        initial_graph class property - np.ndarray:
            a square adjacency matrix representation of the first time step of our graph.
        """
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(self.num_initial_infections):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(
                0, self.num_nodes - 1
            )
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph

    def sample_single_graph(self, graph: np.array):
        """
        Method to sample one time step of the graph evolution.

        Current method is to take all neibours above, below left and right of the adj
        matrix positive values and set them equal to 1 with 100% probability.

        Returns:
            a new numpy adj matrix with updated values
        """
        initial_nodepair_list = np.dstack(np.where(graph == 1))[0]

        node_pairs = []
        for node_pair in initial_nodepair_list:
            node_pairs.append(node_pair)
            if node_pair[0] != 0:
                node_pairs.append(np.array((node_pair[0] - 1, node_pair[1])))
            if node_pair[0] < self.num_nodes - 1:
                node_pairs.append(np.array((node_pair[0] + 1, node_pair[1])))
            if node_pair[1] != 0:
                node_pairs.append(np.array((node_pair[0], node_pair[1] - 1)))
            if node_pair[1] < self.num_nodes - 1:
                node_pairs.append(np.array((node_pair[0], node_pair[1] + 1)))

        for augmented_nodepair in node_pairs:
            graph[augmented_nodepair[0], augmented_nodepair[1]] = 1
        return graph

    def sample_graph_iterative(self):
        """
        Iteratively call the method 'sample_single_graph' till the graph reaches
        it's terminal state and measure the time required for this.

        Returns:
            iterations - int:
                the number of iterations of sampling simulation for the graph to reach its
                terminal state
        """
        graph, iterations = self.initial_graph, 0
        final_infection_graph = np.ones((20, 20))

        while np.array_equal(graph, final_infection_graph) == False:
            graph = self.sample_single_graph(graph)
            iterations += 1

        return iterations


if __name__ == "__main__":
    ##TODO abstract simulation variables for cls and simulation to yaml file.
    import tqdm 

    total_iters = []
    for i in tqdm.tqdm(range(100)):
        x = GraphHolder()
        total_iters.append(x.sample_graph_iterative())

    print(total_iters)
