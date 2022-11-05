# Utilities for drawing pictures of the graphs
from cProfile import label
import numpy as np
from pathlib import Path
# Draw a picture of an adjacency matrix. No customization, aims to do the right thing.
#   mat: adjacency matrix
def draw_graph(mat: np.array, path: str, label_name = ""):
    import os
    label_name = Path(path).stem if label_name == "" else label_name
    """
    For the unitiated, this is building a (graphviz) dot file.
    The for an undirected graph syntax is:
    graph {
        N1 -- N2
    }
    """
    graphString = "graph {"
    # neato supports non-overlapping w/ splines
    graphString += "graph [layout=neato, overlap=false, splines=true]"

    # If the adjacency matrix is symmetric then it
    # is assumed that is an undirected graph.

    assumeUndirected = (mat == mat.T).all()
    # If we think it's undirected then we only iterate the
    # matrix elements above the diagonal
    explain = "(assumed undirected)" if assumeUndirected else ""
    graphString += f"label = \"{label_name} {explain} \" \n"
    spliced = np.triu(mat) if assumeUndirected else mat
    it = np.nditer(spliced, flags=['multi_index'])
    for x in it:
        idx = it.multi_index
        if x == 1:
            graphString += f"{idx[0]} -- {idx[1]}\n"
        else:
            graphString += f"{idx[0]}\n"
    graphString += "}"
    import subprocess
    subprocess.run(["dot", "-Tpng", "-o", path], input=graphString.encode())
    print (f"See {path}")