import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


"""
    PNA: Principal Neighbourhood Aggregation
    Principal Neighbourhood Aggregation for Graph Nets (Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković, NeurIPS 2020)
    https://arxiv.org/abs/2004.05718
    For more information regarding the code and hyperparameter ranges:
    https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.conv.PNAConv.html
"""

from layers.pna_layer import PNALayer
from layers.mlp_readout_layer import MLPReadout


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        batch_norm = net_params['batch_norm']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        residual = net_params['residual']
        n_classes = net_params['n_classes']
        aggregators = net_params['aggregators']
        scalers = net_params['scalers']
        delta = net_params['delta']
        num_towers = net_params['num_towers']

        self.loss_weight = net_params['loss_weight']
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.feature = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.layers = nn.ModuleList([PNALayer(hidden_dim, hidden_dim, aggregators, scalers, delta, num_towers, batch_norm, dropout, residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(PNALayer(hidden_dim, out_dim, aggregators, scalers, delta, num_towers, batch_norm, dropout, residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):

        # input embedding
        h = self.feature(h)

        # PNA
        for conv in self.layers:
            h = conv(g, h)

        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label, loss_fn='weighted_ce'):
        if loss_fn == 'ce':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, label)
            return loss
        elif loss_fn == 'weighted_ce':
            # calculating label weights for weighted loss computation
            # weight = compute_class_weight(class_weight="balanced", classes=np.unique(label.detach().cpu().numpy()), y=label.detach().cpu().numpy())
            # weight = torch.tensor(weight).to(pred.device)

            weight = torch.tensor(self.loss_weight).to(pred.device)
            weight = weight.to(dtype=torch.float32)
            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)
            return loss

