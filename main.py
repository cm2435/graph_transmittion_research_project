#!/usr/bin/env python3
import itertools
import numpy
import random
import multiprocessing
import tqdm
import scipy
import argparse
from graph_structure.erdos_graph import ErdosGraphSimulator
from structure_generation.adj_matrix_gen import *
from collections import namedtuple
from typing import Optional, Tuple, List 
from viz.graph_plot import plot_saturation
import gc

def simulate_saturation(params) -> Tuple[List[float], List[float]]:
    # I hate this
    num_nodes, num_initial_agents, structure_name, transmittion_prob, max_iters = params
    x = ErdosGraphSimulator(
        num_nodes=num_nodes,
        num_agents=num_initial_agents,
        structure_name=structure_name,
    )
    _, iterations, fraction_infected = x.infect_till_saturation(
        infection_probability= transmittion_prob, max_iters = max_iters
    )
    del x, _ 
    gc.collect()
    return iterations, fraction_infected


# run main.py gen-graph
def genAndViz(args, conf) -> None:
    import subprocess
    from multiprocessing.pool import ThreadPool as Pool
    def job(structure):
        RecClass = GraphGenerator.from_string(structure)
        gen = RecClass(structure, int(conf["RUN"]["nodes"]))
        mat = gen.generate_adj_matrix()
        import viz.draw
        import os

        full = os.path.join(conf["VIZ"]["output_dir"], structure + ".png")
        viz.draw.draw_graph(mat, path=full, label_name=structure)
    if args.graph_name is None:
        with Pool() as pool:
            pool.map(job, GraphGenerator.get_graph_names())
    else:
        job(args.graph_name)
    return None 


if __name__ == "__main__":
    """
    At some point:
    Have a config file which defines defaults for this, that, and the other
    but allow those defaults to be overridden with a CLI argument.
    """
    # Config parser uses .ini
    import configparser
    from scipy import stats

    configuration = configparser.ConfigParser()

    parser = argparse.ArgumentParser(
        prog="GraphTransmission",
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
    graphDebug.set_defaults(func=genAndViz)
    graphDebug.add_argument("--graph", dest="graph_name",
                            help="Sample & draw a graph only with this name")
    parsedArgs = parser.parse_args()
    configuration.read(parsedArgs.config_file)

    final_dicts = []
    conf = configuration["RUN"]
    num_initial_agents = int(conf["initial_agents"])
    num_nodes = int(conf["nodes"])
    structure_name = conf["structure"]
    simulation_iters = int(conf['simulation_iterations'])
    transmittion_prob = float(conf['transmittion_prob'])
    max_iters = int(conf['max_iterations'])
    if parsedArgs.cmd is not None:
        parsedArgs.func(parsedArgs, configuration)
        exit()

    simulation_output = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2 - 1) as p:
        iterThis = itertools.repeat(
            (num_nodes, num_initial_agents, structure_name, transmittion_prob, max_iters), simulation_iters
        )
        with tqdm.tqdm(total=simulation_iters) as pbar:
            for _ in p.imap_unordered(simulate_saturation, iterThis):
                pbar.update()
                simulation_output.append(_)

    convergence_steps = [x[0] for x in simulation_output]
    saturation_fractions = [x[1] for x in simulation_output]

    # Pad the list to ones to the longest saturation length, find the mean across all simulations and the std at each timestep
    padded_list = np.array(
        list(zip(*itertools.zip_longest(*saturation_fractions, fillvalue=1)))
    )
    saturation_timestep = np.mean(padded_list, axis=0)
    saturation_timestep_std = [
        np.std(padded_list[:, i]) for i in range(len(padded_list[0]))
    ]

    stats_dict = {
        "mean": np.average(convergence_steps),
        "variance": np.var(convergence_steps),
        "skew": scipy.stats.skew(convergence_steps),
        "kurtosis": scipy.stats.kurtosis(convergence_steps),
        "num_nodes": num_nodes,
    }

    plot_saturation(
            saturation_fraction_mean= saturation_timestep,
            saturation_fraction_std= saturation_timestep_std,
            graph_type= structure_name,
            save_filename= True
            )

    import pandas as pd

    if parsedArgs.csv_dir is not None:
        df = pd.DataFrame.from_dict({f"num_agents_{num_initial_agents}": stats_dict})
        import os

        df.to_csv(
            os.path.join(parsedArgs.csv_dir, f"{num_initial_agents}.csv")
            # f"/home/cm2435/Desktop/university_final_year_cw/data/stats/{num_initial_agents}.csv"
        )/home/cm2435/Desktop/graph_transmittion_research_project/data/figures