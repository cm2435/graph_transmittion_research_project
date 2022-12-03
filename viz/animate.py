import numpy as np 
import networkx as nx 
import pandas as pd 
import random
import matplotlib.pyplot as plt 
from matplotlib import animation
from typing import List 

class NetworkAnimationBase(object):
    '''
    '''
    def __init__(self, num_nodes : int = 10):
        self.num_nodes = num_nodes

    @staticmethod
    def simple_update(
        num_iters, 
        giant_graph,
        sampled_graph_structure : List[np.ndarray],
        infection_graph_list : List[np.ndarray],
        layout, 
        ax,
        ):

        ax.clear()
        #Draw the giant graph as dashed lines
        nx.draw(giant_graph, pos=layout, ax=ax, style = "--", alpha = 0.3)

        #Draw the sampled graph for given timestep as normal line
        nx.draw(nx.from_numpy_array(sampled_graph_structure[num_iters]), pos=layout, node_color=infection_graph_list[num_iters], ax=ax)
        ax.set_title("Frame {}".format(num_iters))
    

    def simple_animation(self,
        giant_graph_adj_matrix : np.ndarray,
        adj_matrix_list : List[np.ndarray],
        infection_matrix_list : List[np.ndarray],
        num_iters : int = 20,
        ) -> None:

        assert num_iters == len(adj_matrix_list), f"the number of iterations must match the number of timesteps of graphs generated in the \
            simulation. Number of timesteps passed was {num_iters}, but number of adj arrays was {len(adj_matrix_list)}"
        
        giant_networkx_graph = nx.from_numpy_array(giant_graph_adj_matrix)
        fig, ax = plt.subplots(figsize=(6,4))
        layout = nx.spring_layout(giant_networkx_graph)
        print(giant_networkx_graph)

        ani = animation.FuncAnimation(fig, self.simple_update, frames=num_iters, interval = 750, fargs=(giant_networkx_graph, adj_matrix_list, infection_matrix_list, layout, ax, ))
        ani.save('animation_1.gif', writer='imagemagick')

        plt.show()



if __name__ == "__main__": 
    #random.seed(5)
    grapher = np.ones((10, 10))

    random_graphs = []
    for i in range(20):
        graph = np.zeros((10, 10), dtype=int)
        for i in range(5):
            random_x, random_y = random.randint(0, 9), random.randint(0, 9)
            graph[random_x][random_y] = 1 
            graph[random_y][random_x] = 1 
        np.fill_diagonal(graph, 0)
        random_graphs.append(graph)

    infectors = []
    for i in range(20):
        infection_matrix = np.zeros(10)
        for i in range(4):
            random_num = random.randint(0, 9)
            infection_matrix[random_num] = 1 
        infectors.append(infection_matrix)

    animator_new = NetworkAnimationBase()
    animator_new.simple_animation(grapher, random_graphs, infectors)