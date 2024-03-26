import torch
import torch.nn as nn
import math
import dgl
import torch.nn.functional as F
import numpy as np
import time
import heapq
import itertools
import networkx as nx
from tqdm.auto import tqdm


def solution_construction(model, device, data_loader, beam_width, time_limit):
    model.eval()
    pred_sols = []
    pred_objs = []
    opt_objs = []
    opt_gaps = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels_list) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            graph_size = batch_graphs.number_of_nodes()
            try:
                batch_labels_list = batch_labels_list.view(1, 100, graph_size)
                label = batch_labels_list[0][0].cpu().numpy()
            except:
                label = batch_labels_list.cpu().numpy()
            nx_graph = batch_graphs.cpu().to_networkx()
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            # get optimal obj value

            opt_obj = np.sum(label)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            scores = F.softmax(batch_scores)
            scores = scores.detach().cpu().numpy()
            probs = scores[:, 1]
            pred_sol, _ = beam_search(nx_graph, probs, beam_width=beam_width, time_limit=time_limit)
            assert (is_independent_set(nx_graph, pred_sol))
            pred_sols.append(pred_sol)
            pred_objs.append(len(pred_sol))
            opt_objs.append(opt_obj)
            opt_gaps.append((opt_obj - len(pred_sol))/len(pred_sol))
    return pred_objs, opt_objs, opt_gaps


class BeamNode:
    def __init__(self, ind_set, score):
        self.ind_set = ind_set  # current cover
        self.ind_set_size = len(self.ind_set)
        self.score = score  # score of current cover according to GNN

    def __lt__(self, other):
        # Override less than operator for min-heap comparison
        return self.score > other.score  # Higher score is better


def is_independent_set(graph, nodes):
    # Iterate through each pair of nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Check if the pair of nodes are adjacent
            if graph.has_edge(nodes[i], nodes[j]):
                # If adjacent, return False as it's not an independent set
                return False
    # If no adjacent nodes found, return True indicating it's an independent set
    return True


def beam_search(nxgraph, probabilities, beam_width=1, time_limit=10):
    start_time = time.time()

    # get sorted copy of probs (indices of the probabilities sorted from max to min)
    sorted_indices = np.flip(np.argsort(probabilities))
    # initialize the initial nodes for all beams
    beam = []
    for i in range(beam_width):
        initial_ind_set = [sorted_indices[i]]
        beam.append(BeamNode(initial_ind_set, probabilities[sorted_indices[i]]))

    while time.time() - start_time < time_limit and beam:
        # Expand each beam with highest scored nodes
        new_beam = []
        for beam_node in beam:
            for new_node in sorted_indices:
                if new_node not in beam_node.ind_set and is_independent_set(nxgraph, beam_node.ind_set + [new_node]):
                    new_ind_set = beam_node.ind_set + [new_node]
                    new_score = beam_node.score + probabilities[new_node]
                    new_beam.append(BeamNode(new_ind_set, new_score))

        # Update beam with top-k solutions or break if theres no new cliques formed
        if not new_beam:
            break
        else:
            beam = heapq.nlargest(beam_width, new_beam)

    # Return the best solution found
    return beam[0].ind_set, beam[0].score
