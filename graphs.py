import numpy as np
import random 

class GraphHolder(object):
    '''
    '''
    def __init__(self, 
                num_nodes : int = 20, 
                num_initial_infections : int = 5
                ):
        self.num_nodes = num_nodes
        self.num_initial_infections = num_initial_infections
        self.num_iterations = 0 

    @property
    def initial_graph(self):
        uninfected_graph = np.zeros((self.num_nodes, self.num_nodes))
        for _ in range(self.num_initial_infections):
            random_i, random_j = random.randint(0, self.num_nodes - 1), random.randint(0, self.num_nodes - 1)
            uninfected_graph[random_i][random_j] = 1

        return uninfected_graph

    def sample_single_graph(self, graph : np.array): 
        '''
        '''
        initial_nodepair_list = np.dstack(np.where(graph == 1))[0]
        
        node_pairs = []
        for node_pair in initial_nodepair_list: 
            node_pairs.append(node_pair)
            if node_pair[0] !=0: 
                node_pairs.append(np.array((node_pair[0]-1, node_pair[1])))
            if node_pair[0] < self.num_nodes - 1: 
                node_pairs.append(np.array((node_pair[0]+1, node_pair[1])))
            if node_pair[1] != 0:
                node_pairs.append(np.array((node_pair[0], node_pair[1]-1)))
            if node_pair[1] < self.num_nodes - 1: 
                node_pairs.append(np.array((node_pair[0], node_pair[1]+1)))

        for augmented_nodepair in node_pairs:
            graph[augmented_nodepair[0], augmented_nodepair[1]] = 1 
        return graph 


    def sample_graph_iterative(self) : 
        graph, iterations = self.initial_graph, 0 
        final_infection_graph = np.ones((20,20))

        while np.array_equal(graph, final_infection_graph) == False:
            graph = self.sample_single_graph(graph)
            iterations += 1 
        
        return iterations

if __name__ == "__main__": 
    total_iters = []
    for i in range(100):
        x = GraphHolder()
        total_iters.append(x.sample_graph_iterative())
    
    print(total_iters)