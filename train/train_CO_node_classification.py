"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

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
        #batch_x = batch_x.flatten(0)
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


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
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

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            epoch_test_f1 += f1_score(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        epoch_test_f1 /= (iter + 1)

    return epoch_test_loss, epoch_test_acc, epoch_test_f1


def evaluate_network_all_optimal(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    epoch_test_f1 = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels_list) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            #batch_x = batch_x.flatten(0)
            batch_labels_list = batch_labels_list.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # within each graph instance

            for labels_list in batch_labels_list:
                # we need to check prediction against ALL optimal solutions and keep one with highest acc
                current_acc = 0
                current_loss = 0
                current_f1 = 0
                for labels in labels_list:
                    # first check if it's a dummy filler label
                    if torch.all(labels == 0):
                        continue
                    loss = model.loss(batch_scores, labels)
                    acc = accuracy(batch_scores, labels)
                    f1 = f1_score(batch_scores, labels, split="test")
                    # MAYBE USE LOWEST LOSS INSTEAD OF LOWEST F1?
                    if f1 > current_f1:
                        current_f1 = f1
                        current_loss = loss
                        current_acc = acc
                epoch_test_loss += current_loss
                epoch_test_acc += current_acc
                epoch_test_f1 += current_f1
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        epoch_test_f1 /= (iter + 1)

    return epoch_test_loss, epoch_test_acc, epoch_test_f1

