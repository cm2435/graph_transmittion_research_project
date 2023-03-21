import numpy as np
import pandas as pd
import random
import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
from pathlib import Path
import sys 
import os 
import multiprocessing
import itertools

sys.path.append(os.path.dirname(os.getcwd()))
from util.functions import logistic
from structure_generation.path_connection_gen import ProceduralGraphGenerator, StatsUtils, GraphStructureGenerator
import time

def run_single_simulation(
    passed_inputs, 
    verbose : bool =  False 
    ):
    """
    Function to run one iteration of the simulation. Need to pass a list instead of arguments to function 
    is so that it can work with pmap.unordered
    """
    np.random.seed(random.randint(0,10000))
    mean_degree, structure_name, number_nodes = passed_inputs
    results_dict_irreversable, results_dict_reversable, results_dict_probability = {}, {}, {}
    if verbose: 
        print(f"simulation run for degree {mean_degree}")
    graphgen = GraphStructureGenerator(
        structure_name=structure_name, 
        num_nodes=number_nodes, 
        target_mean_degree = mean_degree
    )
    graph = graphgen.initial_graph  

    x = ProceduralGraphGenerator(graph, num_nodes= graph.number_of_nodes())

    infection_matrix_list_prob, timesteps_saturation_prob, fraction_infected_list_prob, info_dict_prob = x.infect_till_saturation(
        structure_name = structure_name, modality="irreversable", verbose= False, infection_probability= 0.01, sample_giant= False
    )
    infection_matrix_list_irreversable, timesteps_saturation_irreversable, fraction_infected_list_irreversable, info_dict_irreversable = x.infect_till_saturation(
        structure_name = structure_name, modality="irreversable", verbose= False, sample_giant= True
    )


    infection_matrix_list_reversable, timesteps_saturation_reversable, fraction_infected_list_reversable, info_dict_reversable = x.infect_till_saturation(
        structure_name = structure_name, modality="reversable", verbose= False
    )


    results_dict_irreversable["infection_matrix"] = infection_matrix_list_irreversable
    results_dict_irreversable["timesteps_saturation"] = timesteps_saturation_irreversable
    results_dict_irreversable["fraction_infected_list"] = fraction_infected_list_irreversable
    results_dict_irreversable["info_dict"] = info_dict_irreversable

    results_dict_reversable["infection_matrix"] = infection_matrix_list_reversable
    results_dict_reversable["timesteps_saturation"] = timesteps_saturation_reversable
    results_dict_reversable["fraction_infected_list"] = fraction_infected_list_reversable
    results_dict_reversable["info_dict"] = info_dict_reversable

    results_dict_probability["infection_matrix"] = infection_matrix_list_prob
    results_dict_probability["timesteps_saturation"] = timesteps_saturation_prob
    results_dict_probability["fraction_infected_list"] = fraction_infected_list_prob
    results_dict_probability["info_dict"] = info_dict_prob

    return results_dict_irreversable,results_dict_reversable, results_dict_probability
    
def run_simulation(mean_degree : int, structure_name : str) -> list: 
    geometric_graph_conf = yaml.safe_load(Path('config.yml').read_text())['reachability']

    num_runs = {
        "random_geometric" : geometric_graph_conf['num_simulation_runs'],
        "barabasi_albert" : geometric_graph_conf['num_simulation_runs'],

    }[structure_name]
    simulation_run_reversable, simulation_run_irreversable, simulation_run_prob = [], [], []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()- 1) as p:
        iterThis = itertools.repeat(
            [
                mean_degree, 
                structure_name,
                geometric_graph_conf['nodes']
            ],
            num_runs,
        )
        with tqdm.tqdm(total=num_runs) as pbar:
            for _ in p.imap_unordered(run_single_simulation, iterThis):
                pbar.update()
                simulation_run_reversable.append(_[0])
                simulation_run_irreversable.append(_[1])
                simulation_run_prob.append(_[2])

    final_dict_reversable = {k : [] for k in list(_[0].keys())}
    for key in final_dict_reversable.keys():
        for simulation_dict in simulation_run_reversable: 
            final_dict_reversable[key].append(simulation_dict[key])

    final_dict_irreversable = {k : [] for k in list(_[0].keys())}
    for key in final_dict_irreversable.keys():
        for simulation_dict in simulation_run_irreversable: 
            final_dict_irreversable[key].append(simulation_dict[key])

    final_dict_prob = {k : [] for k in list(_[0].keys())}
    for key in final_dict_prob.keys():
        for simulation_dict in simulation_run_irreversable: 
            final_dict_prob[key].append(simulation_dict[key])


    return final_dict_reversable, final_dict_irreversable, final_dict_prob

def plot_results(results_dict : dict, structure_name : str, xlim_range = None):
    num_nodes_run = [x['num_nodes'] for x in results_dict['info_dict']]

    average_nodes_final = []
    for i, iter in enumerate(results_dict['fraction_infected_list']):
        if 1==1:
        # if average_node_num-(std_node_num/2) <= results_dict['info_dict'][i]['num_nodes'] <= average_node_num+(std_node_num/2):
            timesteps_list = [x for x in range(len(iter))]
            plt.plot(timesteps_list, iter,)#label = f"average simulated saturation across {len(result['timesteps_saturation'])} runs")
            average_nodes_final.append(results_dict['info_dict'][i]['num_nodes'])
        else:
            print(f"dropping run number {i} due to insufficient nodes in the giant component")
    
    print("average num nodes", np.average(average_nodes_final), "std : ", np.std(average_nodes_final))
    if xlim_range: 
        plt.xlim(xlim_range)
    plt.title(f"saturation curve for {structure_name}     modality: {results_dict['info_dict'][0]['modality']}    mean degree: {round(results_dict['info_dict'][0]['average_degree'], 2)}")
    plt.xlabel("number simulation timesteps")
    plt.ylabel("fraction of giant graph infected")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    graphgen = GraphStructureGenerator(
        structure_name="random_geometric", 
        num_nodes=100, 
        target_mean_degree = 5
    )
    graph = graphgen.initial_graph  

    x = ProceduralGraphGenerator(graph, num_nodes= graph.number_of_nodes())

    infection_matrix_list_prob, timesteps_saturation_prob, fraction_infected_list_prob, info_dict_prob = x.infect_till_saturation(
        structure_name = "random_geometric", modality="irreversable", verbose= False, infection_probability= 1, sample_giant= True
    )
    print(fraction_infected_list_prob)