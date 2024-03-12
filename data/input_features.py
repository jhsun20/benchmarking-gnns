import time
import os
import pickle
import numpy as np
import dgl
import torch
import networkx as nx

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g


def basic_features(nxgraph):
    """
    input: networkx graph
    adds basic structural features for each node: node degree, pagerank, etc.
    output: tensor of shape (num nodes, input dim)
    """
    node_degrees = dict(nxgraph.degree())
    degree_centrality = nx.degree_centrality(nxgraph)
    betweenness_centrality = nx.betweenness_centrality(nxgraph)
    closeness_centrality = nx.closeness_centrality(nxgraph)
    pagerank_centrality = nx.pagerank(nxgraph)
    harmonic_centrality = nx.harmonic_centrality(nxgraph)
    load_centrality = nx.load_centrality(nxgraph)
    clustering_coefficient = nx.clustering(nxgraph)
    # make it into an array
    features_array = np.array([
        list(node_degrees.values()),
        list(degree_centrality.values()),
        list(betweenness_centrality.values()),
        list(closeness_centrality.values()),
        list(pagerank_centrality.values()),
        list(harmonic_centrality.values()),
        list(load_centrality.values()),
        list(clustering_coefficient.values())])
    features_array = features_array.T
    features_tensor = torch.tensor(features_array)
    return features_tensor.to(dtype=torch.float32)
