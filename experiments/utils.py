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
from structure_generation.path_connection_gen import (
    ProceduralGraphGenerator,
    StatsUtils,
    GraphStructureGenerator,
)
import time


def _run_single_simulation(
    passed_inputs,
):
    np.random.seed(random.randint(0, 10000))
    mean_degree, structure_name = passed_inputs
    geometric_graph_conf = yaml.safe_load(Path("config.yml").read_text())[
        "reachability"
    ]
    graphgen = GraphStructureGenerator(
        structure_name=structure_name,
        num_nodes=geometric_graph_conf["nodes"],
        target_mean_degree=mean_degree,
    )
    graph = graphgen.initial_graph

    graph_generator = ProceduralGraphGenerator(graph, num_nodes=graph.number_of_nodes())
    #First is irreversable, then reversable, then probability
    final_results_dict = {}
    saturation_types = {"irreversable": ("irreversable", True), "reversable" : ("reversable", True), "probability" : ("irreversable", False)}
    for key in saturation_types.keys():
        (
            infection_matrix_list,
            timesteps_to_saturation,
            fraction_infected_list,
            info_dict,
        ) = graph_generator.infect_till_saturation(
            structure_name=structure_name,
            modality=saturation_types[key][0],
            verbose=False,
            infection_probability=0.01,
            sample_giant=saturation_types[key][1],
        )
        results_dict = {}
        results_dict["infection_matrix"] = infection_matrix_list
        results_dict["timesteps_saturation"] = timesteps_to_saturation
        results_dict["fraction_infected_list"] = fraction_infected_list
        results_dict["info_dict"] = info_dict
        final_results_dict[key] = results_dict

    return final_results_dict


def run_simulation(mean_degree: int, structure_name: str) -> list:
    geometric_graph_conf = yaml.safe_load(Path("config.yml").read_text())[
        "reachability"
    ]
    num_runs = {
        "random_geometric": geometric_graph_conf["num_simulation_runs"],
        "barabasi_albert": geometric_graph_conf["num_simulation_runs"],
    }[structure_name]

    output_dictionary = {"reversable": [], "irreversable": [], "probability": []}
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
        iterThis = itertools.repeat(
            [mean_degree, structure_name],
            num_runs,
        )
        with tqdm.tqdm(total=num_runs) as pbar:
            for _ in p.imap_unordered(_run_single_simulation, iterThis):
                pbar.update()
                output_dictionary["irreversable"].append(_['irreversable'])
                output_dictionary["reversable"].append(_['reversable'])
                output_dictionary["probability"].append(_['probability'])

    for saturation_type in output_dictionary.keys():
        results_list = output_dictionary[saturation_type]
        results_dict = {k: [] for k in list(results_list[0].keys())}
        for key in results_dict.keys():
            for simulation_dict in results_list:
                results_dict[key].append(simulation_dict[key])

        output_dictionary[saturation_type] = results_dict

    return output_dictionary


def plot_results(results_dict: dict, structure_name: str, xlim_range=None):
    num_nodes_run = [x["num_nodes"] for x in results_dict["info_dict"]]

    average_nodes_final = []
    for i, iter in enumerate(results_dict["fraction_infected_list"]):
        if 1 == 1:
            # if average_node_num-(std_node_num/2) <= results_dict['info_dict'][i]['num_nodes'] <= average_node_num+(std_node_num/2):
            timesteps_list = [x for x in range(len(iter))]
            plt.plot(
                timesteps_list,
                iter,
            )  # label = f"average simulated saturation across {len(result['timesteps_saturation'])} runs")
            average_nodes_final.append(results_dict["info_dict"][i]["num_nodes"])
        else:
            print(
                f"dropping run number {i} due to insufficient nodes in the giant component"
            )

    print(
        "average num nodes",
        np.average(average_nodes_final),
        "std : ",
        np.std(average_nodes_final),
    )
    if xlim_range:
        plt.xlim(xlim_range)
    plt.title(
        f"saturation curve for {structure_name}     modality: {results_dict['info_dict'][0]['modality']}    mean degree: {round(results_dict['info_dict'][0]['average_degree'], 2)}"
    )
    plt.xlabel("number simulation timesteps")
    plt.ylabel("fraction of giant graph infected")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """graphgen = GraphStructureGenerator(
        structure_name="random_geometric",
        num_nodes=100,
        target_mean_degree = 5
    )
    graph = graphgen.initial_graph

    x = ProceduralGraphGenerator(graph, num_nodes= graph.number_of_nodes())

    infection_matrix_list_prob, timesteps_saturation_prob, fraction_infected_list_prob, info_dict_prob = x.infect_till_saturation(
        structure_name = "random_geometric", modality="irreversable", verbose= False, infection_probability= 1, sample_giant= True
    )
    print(fraction_infected_list_prob)"""
    x = run_simulation(5, "random_geometric")
    for key in x.keys():
        print(x[key]["timesteps_saturation"])
        print("\n\n")
