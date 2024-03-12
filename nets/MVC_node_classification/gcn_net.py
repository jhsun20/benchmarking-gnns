import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
    For more information regarding the code and hyperparameter ranges:
    https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.conv.GraphConv.html
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout

class GCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        n_classes = net_params['n_classes']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.loss_weight = net_params['loss_weight']
        
        self.feature = nn.Linear(in_dim, hidden_dim) # node feat is an integer
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, self.batch_norm, dropout, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, self.batch_norm, dropout, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):

        # input embedding
        h = self.feature(h)
        
        # GCN
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











