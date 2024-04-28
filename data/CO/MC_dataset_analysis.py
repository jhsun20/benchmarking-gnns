import time
import os
import pickle
import dgl
import torch
import pandas as pd

import networkx as nx
from scipy import sparse as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter



# node density graph size graphs
# add label balance graphs
# add average number of solutions graphs


def analyze_graphs_list(list_of_datasets):
    all_densities = []
    all_degrees = []
    for dataset in list_of_datasets:
        densities = []
        degrees = []
        for graph in dataset:
            density = nx.density(graph)
            densities.append(density)
            graph_degrees = [degree for node, degree in graph.degree()]
            degrees.extend(graph_degrees)
        all_densities.append(densities)
        all_degrees.append(degrees)
    return all_densities, all_degrees


def analyze_labels_list(list_of_labels):
    average_objs = []
    average_sols = []
    for dataset in list_of_labels:
        dataset_objs = []
        dataset_sols = []
        for graph in dataset:
            dataset_objs.append(len(graph[0]))
            dataset_sols.append(len(graph))
        mean = np.mean(dataset_objs)
        std_dev = np.std(dataset_objs)
        sol_mean = np.mean(dataset_sols)
        sol_std_dev = np.std(dataset_sols)
        average_objs.append([mean, std_dev, sol_mean, sol_std_dev])
    return average_objs


MC_graphs = []
MC_labels = []

with open("test/MC_test.pkl", 'rb') as f:
    dataset = pickle.load(f)
    graphs_list = [data[0] for data in dataset]
    labels_list = [data[1] for data in dataset]
    MC_graphs.append(graphs_list)
    MC_labels.append(labels_list)

with open("test/MC_30_0.6_test.pkl", 'rb') as f:
    dataset = pickle.load(f)
    graphs_list = [data[0] for data in dataset]
    labels_list = [data[1] for data in dataset]
    MC_graphs.append(graphs_list)
    MC_labels.append(labels_list)

with open("test/MC_90_0.2_test.pkl", 'rb') as f:
    dataset = pickle.load(f)
    graphs_list = [data[0] for data in dataset]
    labels_list = [data[1] for data in dataset]
    MC_graphs.append(graphs_list)
    MC_labels.append(labels_list)

with open("test/MC_90_0.6_test.pkl", 'rb') as f:
    dataset = pickle.load(f)
    graphs_list = [data[0] for data in dataset]
    labels_list = [data[1] for data in dataset]
    MC_graphs.append(graphs_list)
    MC_labels.append(labels_list)

with open("test/MC_180_0.5_test.pkl", 'rb') as f:
    dataset = pickle.load(f)
    graphs_list = [data[0] for data in dataset]
    labels_list = [data[1] for data in dataset]
    MC_graphs.append(graphs_list)
    MC_labels.append(labels_list)


# CLASS IMBALANCE DATA
all_average_objs = analyze_labels_list(MC_labels)
print('[average opt value, std dev, average num of opt sols, std]')
print(all_average_objs)

# Analyze the dataset
all_densities, all_degrees = analyze_graphs_list(MC_graphs)

# Convert the list of densities to a long-form DataFrame for Seaborn
dataset_names = ['30, sparse', '30, dense', '90, sparse', '90, dense', '180, varied']

# Convert the list of densities to a long-form DataFrame for Seaborn
densities_data = []
for i, densities in enumerate(all_densities):
    for density in densities:
        densities_data.append({'Dataset': dataset_names[i], 'Density': density})
densities_df = pd.DataFrame(densities_data)

# Create boxplot for densities using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(data=densities_df, y='Density', x='Dataset', color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('Density')
plt.title('Densities for Each Test Dataset (MC)')
plt.show()

# Convert the list of densities to a long-form DataFrame for Seaborn
degrees_data = []
for i, degrees in enumerate(all_degrees):
    for degree in degrees:
        degrees_data.append({'Dataset': dataset_names[i], 'Degree': degree})
degrees_df = pd.DataFrame(degrees_data)

# Create boxplot for densities using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(data=degrees_df, y='Degree', x='Dataset', color='skyblue')
plt.xlabel('Dataset')
plt.ylabel('Degree')
plt.title('Degrees for Each Test Dataset (MC)')
plt.show()

