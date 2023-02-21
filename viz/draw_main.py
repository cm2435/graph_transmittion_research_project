# Utilities for drawing pictures of the graphs
from cProfile import label
import numpy as np
from pathlib import Path


def find_components(mat: np.array) -> np.array:
    """
    Given an input adjacency matrix, the connected components of the graph
    are found and returned as a list of component numbers indexed by the node.
    """
    out = np.zeros(shape=mat.shape[0], dtype=np.int8)
    from scipy.cluster.hierarchy import DisjointSet

    set = DisjointSet(range(mat.shape[0]))
    tri = np.triu(mat)
    it = np.nditer(mat, flags=["multi_index"])
    # To draw a directed graph properly in DOT a different symbol is needed
    for x in it:
        idx = it.multi_index
        if x == 1:
            # This is a connection between nodes
            set.merge(idx[0], idx[1])
    for idx, subset in enumerate(set.subsets()):
        for i in subset:
            out[i] = idx
    return out


# Draw a picture of an adjacency matrix. No customization, aims to do the right thing.
#   mat: adjacency matrix


def draw_graph(mat: np.array, path: str, label_name=""):
    import os

    label_name = Path(path).stem if label_name == "" else label_name
    """
    For the unitiated, this is building a (graphviz) dot file.
    The for an undirected graph syntax is:
    graph {
        N1 -- N2
    }
    """

    assumeUndirected = (mat == mat.T).all()
    components = find_components(mat)
    graphString = "graph { " if assumeUndirected else "digraph { \n"
    # neato supports non-overlapping w/ splines
    graphString += "graph [layout=neato, overlap=false, splines=true]"

    # If the adjacency matrix is symmetric then it
    # is assumed that is an undirected graph.

    # If we think it's undirected then we only iterate the
    # matrix elements above the diagonal
    explain = "(assumed undirected)" if assumeUndirected else ""
    from datetime import datetime

    graphString += f'label = "{label_name} {explain} - {datetime.now()}" \n'
    spliced = np.triu(mat) if assumeUndirected else mat
    it = np.nditer(spliced, flags=["multi_index"])
    # To draw a directed graph properly in DOT a different symbol is needed
    conString = "--" if assumeUndirected else "->"
    for idx in range(mat.shape[0]):
        graphString += f'{idx} [label="{components[idx]}"]\n'
    for x in it:
        idx = it.multi_index
        if x == 1:
            graphString += f"{idx[0]} {conString} {idx[1]}\n"
        else:
            graphString += f"{idx[0]}\n"
    graphString += "}"
    import subprocess

    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=300", "-o", path], input=graphString.encode()
    )
    print(f"See {path}")
