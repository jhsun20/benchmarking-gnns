import time
import os
import pickle
import dgl
import torch

import networkx as nx
from scipy import sparse as sp
import numpy as np

# node density graph size graphs
# add label balance graphs
# add average number of solutions graphs


def analyze_graphs(dataset):
    # Initialize lists to store statistics
    densities = []
    degree_distribution = []

    # Iterate over each graph in the dataset
    for graph in dataset:
        # Calculate density of the graph
        density = nx.density(graph)
        densities.append(density)

        # Calculate degree distribution
        degrees = [degree for node, degree in graph.degree()]
        degree_distribution.extend(degrees)

    # Calculate statistics
    avg_density = np.mean(densities)
    std_density = np.std(densities)
    avg_degree = np.mean(degree_distribution)
    std_degree = np.std(degree_distribution)

    return avg_density, std_density, avg_degree, std_degree, degree_distribution


# Example dataset (replace this with your actual dataset)
with open("data/CO/train/MIS_train.pkl", 'rb') as f:
    dataset = pickle.load(f)

# Analyze the dataset
avg_density, std_density, avg_degree, std_degree, degree_distribution = analyze_graphs(dataset)

# Print results
print("Average density:", avg_density)
print("Standard deviation of density:", std_density)
print("Average degree:", avg_degree)
print("Standard deviation of degree:", std_degree)
