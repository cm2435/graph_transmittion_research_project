import itertools 
import numpy 
import random 
import multiprocessing
import tqdm 
import scipy 
import argparse
from graph_structure.erdos_graph import ErdosGraphSimulator
from structure_generation.adj_matrix_gen import *


def simulate_saturation(_=1):
    # Global function just for the sake of making multiprocessing nice and simple
    x = ErdosGraphSimulator(num_nodes=num_nodes, num_agents=num_initial_agents, structure_name= structure_name)
    _, iterations, fraction_infected = x.infect_till_saturation(infection_probability= 0.01)
    return iterations, fraction_infected


if __name__ == "__main__":
    """
    At some point:
    Have a config file which defines defaults for this, that, and the other
    but allow those defaults to be overridden with a CLI argument.
    """
    # Config parser uses .ini
    import configparser
    configuration = configparser.ConfigParser()

    parser = argparse.ArgumentParser(
                    prog = 'GraphTransmission',
                    description = 'Simulates information propagation on networks',
                    epilog = 'Written by Charlie Masters and Max Haughton')
    parser.add_argument("--csv-dir", dest="csv_dir", help="Where to write a .csv file")
    parser.add_argument("--config", dest="config_file", help="Path to a configuration file, default is config.ini", default="config.ini")
    parsedArgs = parser.parse_args()

    configuration.read(parsedArgs.config_file)

    final_dicts = []
    global num_initial_agents, num_nodes, structure_name
    conf = configuration["RUN"]
    num_initial_agents = int(conf["initial_agents"])
    num_nodes = int(conf["nodes"])
    structure_name = conf["structure"]
    # num_initial_agents, num_nodes, structure_name = 1, 20, "fully_connected"
    simulation_output = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() * 2 - 1) as p:
        num_simulation_steps = 1000
        with tqdm.tqdm(total=num_simulation_steps) as pbar:
            for _ in p.imap_unordered(simulate_saturation, range(0, num_simulation_steps)):
                pbar.update()
                simulation_output.append(_)

    convergence_steps = [x[0] for x in simulation_output]
    saturation_fractions = [x[1] for x in simulation_output]

    #Pad the list to ones to the longest saturation length, find the mean across all simulations and the std at each timestep 
    padded_list = np.array(list(zip(*itertools.zip_longest(*saturation_fractions, fillvalue=1))))
    saturation_timestep = np.mean(padded_list, axis = 0)
    saturation_timestep_std = [np.std(padded_list[:, i]) for i in range(len(padded_list[0]))]

    stats_dict = {
        "mean": np.average(convergence_steps),
        "variance": np.var(convergence_steps),
        "skew": scipy.stats.skew(convergence_steps),
        "kurtosis": scipy.stats.kurtosis(convergence_steps),
        "num_nodes": num_nodes,
    }
    print(stats_dict)
    import pandas as pd
    if parsedArgs.csv_dir is not None:
        df = pd.DataFrame.from_dict({f"num_agents_{num_initial_agents}": stats_dict})
        import os
        df.to_csv(
            os.path.join(parsedArgs.csv_dir, f"{num_initial_agents}.csv")
            #f"/home/cm2435/Desktop/university_final_year_cw/data/stats/{num_initial_agents}.csv"
        )
