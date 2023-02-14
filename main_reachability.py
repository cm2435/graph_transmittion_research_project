import numpy as np
import pandas as pd
import random
import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
from pathlib import Path

from util.functions import logistic
from structure_generation.path_connection_gen import ProceduralGraphGenerator, StatsUtils, GraphStructureGenerator

if __name__ == "__main__":
    conf = yaml.safe_load(Path('config.yml').read_text())['reachability']

    average_degrees = []
    residuals_dict = {k : [] for k in conf['graph_edge_radii']}
    raw_vals_dict = {}
    for graph_rad in tqdm.tqdm(conf['graph_edge_radii']):
        mean_degree_simulation_runs = []
        for repeat in range(int(conf['num_simulation_repeats'])): 
            graphgen = GraphStructureGenerator(structure_name=conf['structure_name'], num_nodes=int(conf['nodes']), graph_edge_radius = float(graph_rad))
            graph = graphgen.initial_adj_matrix
            graph_rand = graphgen.get_graph_structure().initial_adj_matrix

            x = ProceduralGraphGenerator(graph)
            infection_matrix_list, timesteps_saturation, fraction_infected_list, average_degree = x.infect_till_saturation(
                modality="saturation", verbose= False
            )
            timesteps = [x for x in range(timesteps_saturation)]
            try: 
                #Fit a logistic curve to the simulated infection data for one simulation run, generate data with this logistic, use to find residuals in fit
                p, cov = curve_fit(logistic, timesteps, fraction_infected_list)
                logistic_curve_data = logistic(timesteps, *p)
                residuals_dict[graph_rad].extend(fraction_infected_list - logistic_curve_data)
                mean_degree_simulation_runs.append(average_degree)
            except RuntimeError as e:
                print(e)
                pass 
        average_degrees.append(np.mean(mean_degree_simulation_runs))
    
    #Update keys of dictionary so that keys are the (averaged) mean degree of the simulations networks
    residuals_dict = {k : v for k,v in list(zip(average_degrees, residuals_dict.values()))}
    least_noisy_data = residuals_dict
    for key in residuals_dict.keys(): 
        sns.kdeplot(residuals_dict[key], label=f"mean degree : {round(key, 2)}")
    plt.legend()
    plt.show()

