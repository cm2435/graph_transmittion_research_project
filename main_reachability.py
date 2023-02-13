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
import matplotlib.pyplot as plt
import seaborn as sns 

from structure_generation.path_connection_gen import ProceduralGraphGenerator, StatsUtils, GraphStructureGenerator

def logistic(x, A, k, x0):
    return A / (1.0 + np.exp(-k * (x - x0)))
    

if __name__ == "__main__":
    import configparser
    from scipy import stats
    import argparse
    configuration = configparser.ConfigParser()

    parser = argparse.ArgumentParser(
        prog="Graph Reachabbility",
        description="Simulates information propagation on networks",
        epilog="Written by Charlie Masters and Max Haughton",
    )
    parser.add_argument("--csv-dir", dest="csv_dir", help="Where to write a .csv file")
    parser.add_argument(
        "--config",
        dest="config_file",
        help="Path to a configuration file, default is config.ini",
        default="config.ini",
    )
    subParsers = parser.add_subparsers(
        title="Some sub-utilities are available", dest="cmd"
    )
    graphDebug = subParsers.add_parser(
        "gen-graph", help="Generate an example graph using each available method"
    )
    parsedArgs = parser.parse_args()
    configuration.read(parsedArgs.config_file)
    final_dicts = []
    conf = {k:v for k,v in configuration["reachability"].items()}
    if parsedArgs.cmd is not None:
        parsedArgs.func(parsedArgs, configuration)
        exit()

    average_degrees, results_dict = [], {k : [] for k in conf['graph_edge_radii']}
    for graph_rad in tqdm.tqdm(conf['graph_edge_radii']):
        mean_degree_simulation_runs = []
        for repeat in range(conf['num_simulation_repeats']): 
            graphgen = GraphStructureGenerator(structure_name=conf['structure_name'], num_nodes=conf['nodes'], graph_edge_radius = conf['graph_edge_radii'])
            graph = graphgen.initial_adj_matrix
            graph_rand = graphgen.get_graph_structure().initial_adj_matrix
            x = ProceduralGraphGenerator(graph)

            infection_matrix_list, timesteps_saturation, fraction_infected_list, average_degree = x.infect_till_saturation(
                modality="causal", verbose= False
            )
            timesteps = [x for x in range(timesteps_saturation)]
            try: 
                p, cov = curve_fit(logistic, timesteps, fraction_infected_list)
                logistic_curve_data = logistic(timesteps, *p)
                results_dict[graph_rad].extend(fraction_infected_list - logistic_curve_data)
                mean_degree_simulation_runs.append(average_degree)
            except RuntimeError as e:
                print(e)
                pass 
        average_degrees.append(np.mean(mean_degree_simulation_runs))
    
    results_dict = {k : v for k,v in list(zip(average_degrees, results_dict.values()))}
    for key in results_dict.keys(): 
        sns.kdeplot(results_dict[key])
    plt.show()

