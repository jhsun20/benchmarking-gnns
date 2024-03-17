"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
import numpy as np

from train.metrics import accuracy_CO as accuracy, binary_f1_score as f1_score

"""
    For GCNs
"""


def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_f1 = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        epoch_train_f1 += f1_score(batch_scores, batch_labels)

    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    epoch_train_f1 /= (iter + 1)

    return epoch_loss, epoch_train_acc, epoch_train_f1, optimizer


def train_epoch_all_optimal(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_f1 = 0
    current_batch_graphs = []
    current_batch_nodes = []
    current_batch_edges = []
    current_batch_labels = []
    batches = len(data_loader) // batch_size
    last_batch_size = len(data_loader) % batch_size
    current_batch = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        graph_size = batch_graphs.number_of_nodes()
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        # save the batch graphs until they reach batch size
        current_batch_graphs.append(batch_graphs)
        current_batch_nodes.append(batch_x)
        current_batch_edges.append(batch_e)
        current_batch_labels.append(batch_labels)
        if (current_batch == batches and len(current_batch_graphs) == last_batch_size) or (len(current_batch_graphs) == batch_size):
            if current_batch == batches:
                current_batch_size = last_batch_size
            else:
                current_batch_size = batch_size
            # CONCATE ALL THE TENSORS FIRST
            batch_graphs = dgl.batch(current_batch_graphs)
            batch_x = torch.cat(current_batch_nodes)
            batch_e = torch.cat(current_batch_edges)
            batch_labels = torch.cat(current_batch_labels)
            # CLEAR ALL THE LISTS
            current_batch_graphs = []
            current_batch_nodes = []
            current_batch_edges = []
            current_batch_labels = []
            # UPDATE BATCH
            current_batch += 1
            # forward step
            optimizer.zero_grad()
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # GET THE BEST LABELS FOR EACH GRAPH
            best_batch_labels = []
            separate_batch_scores = batch_scores.view(current_batch_size, graph_size, 2)
            separate_batch_labels = batch_labels.view(current_batch_size, 100, graph_size)
            model.eval()
            for i in range(len(separate_batch_scores)):
                scores = separate_batch_scores[i]
                labels_list = separate_batch_labels[i]
                current_loss = 1000
                for labels in labels_list:
                    if torch.all(labels == 0):
                        continue
                    loss = model.loss(scores, labels)
                    if loss < current_loss:
                        current_loss = loss
                        current_label = labels
                best_batch_labels.append(current_label)
            batch_labels = torch.cat(best_batch_labels)
            # learn
            model.train()
            loss = model.loss(batch_scores, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_acc += accuracy(batch_scores, batch_labels)
            epoch_train_f1 += f1_score(batch_scores, batch_labels)
    epoch_loss /= (batches + 1)
    epoch_train_acc /= (batches + 1)
    epoch_train_f1 /= (batches + 1)

    return epoch_loss, epoch_train_acc, epoch_train_f1, optimizer


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
    predicted_obj = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            #batch_x = batch_x.flatten(0)
            batch_labels = batch_labels.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            y_true = batch_labels.cpu().numpy()
            y_pred = batch_scores.argmax(dim=1).cpu().numpy()
            predicted_obj.append([np.sum(y_true), np.sum(y_pred)])
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            epoch_test_f1 += f1_score(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
    #print('\n', predicted_obj)
    return epoch_test_loss, epoch_test_acc, epoch_test_f1


def evaluate_network_all_optimal(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
    predicted_obj = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels_list) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            graph_size = batch_graphs.number_of_nodes()
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels_list = batch_labels_list.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # within each graph instance
            batch_labels_list = batch_labels_list.view(1, 100, batch_graphs.number_of_nodes())
            for labels_list in batch_labels_list:
                # we need to check prediction against ALL optimal solutions and keep one with highest acc
                current_acc = 0
                current_loss = 1000
                current_f1 = 0
                batch_scores = batch_scores.view(graph_size, 2)
                y_true = labels_list[0].cpu().numpy()
                y_pred = batch_scores.argmax(dim=1).cpu().numpy()
                predicted_obj.append([np.sum(y_true), np.sum(y_pred)])
                for labels in labels_list:
                    # first check if it's a dummy filler label
                    if torch.all(labels == 0):
                        continue
                    loss = model.loss(batch_scores, labels)
                    acc = accuracy(batch_scores, labels)
                    f1 = f1_score(batch_scores, labels)
                    # MAYBE USE LOWEST LOSS INSTEAD OF LOWEST F1?
                    if acc >= current_acc:
                        current_f1 = f1
                        current_loss = loss
                        current_acc = acc
                epoch_test_loss += current_loss.detach().item()
                epoch_test_acc += current_acc
                epoch_test_f1 += current_f1
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
    #print('\n', predicted_obj)
    return epoch_test_loss, epoch_test_acc, epoch_test_f1


def solution_construction(model, device, data_loader):
    model.eval()
    pred_objs = []
    opt_objs = []
    opt_gaps = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels_list) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            nx_graph = batch_graphs.to_networkx
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            # get optimal obj value
            label = batch_labels_list[0][0].cpu().numpy()
            opt_obj = np.sum(label)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            scores = nn.Softmax(batch_scores)
            scores = scores.detach().cpu().numpy()
            probs = scores[:, 1]
            pred_sol = heuristic_search(nx_graph, probs, problem="MVC", time_limit=10)
            pred_objs.append(pred_sol)
            opt_objs.append(opt_obj)
            opt_gaps.append((opt_obj - pred_sol)/pred_sol)
    return pred_objs, opt_objs, opt_gaps


def heuristic_search(nxgraph, probabilities, problem, time_limit=10):
    return 2