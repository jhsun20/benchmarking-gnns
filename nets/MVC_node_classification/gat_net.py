import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

"""
    GAT: Graph Attention Network (v2)
    Based on Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
    But Improved by How Attentive are Graph Attention Networks? (Brody et al., ICLR 2022)
    https://arxiv.org/abs/2105.14491
    For more information regarding the code and hyperparameter ranges:
    https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.conv.GATv2Conv.html

"""
from layers.gat_layer import GATLayer, GATv2Layer
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        batch_norm = net_params['batch_norm']
        attn_drop = net_params['attn_drop']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        neg_slope = net_params['neg_slope']
        residual = net_params['residual']
        n_classes = net_params['n_classes']
        self.loss_weight = net_params['loss_weight']
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.feature = nn.Linear(in_dim_node, hidden_dim * num_heads) # node feat is an integer
        self.layers = nn.ModuleList([GATv2Layer(hidden_dim * num_heads, hidden_dim, num_heads, batch_norm, dropout, attn_drop, neg_slope, residual) for _ in range(n_layers-1)])
        self.layers.append(GATv2Layer(hidden_dim * num_heads, out_dim, 1, batch_norm, dropout, attn_drop, neg_slope,
                                    residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):

        # input embedding
        h = self.feature(h)

        # GAT
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

