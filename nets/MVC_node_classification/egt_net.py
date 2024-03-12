import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

"""
    EGT: l Edge-augmented Graph Transformer
    Global Self-Attention as a Replacement for Graph Convolution (Md Shamim Hussain, Mohammed J. Zaki, Dharmashankar Subramanian, ACM SIGKDD 2022)
    https://arxiv.org/pdf/2108.03348.pdf
    For more information regarding the code and hyperparameter ranges:
    https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.gt.EGTLayer.html
"""
from layers.egt_layer import EGTLayer
from layers.mlp_readout_layer import MLPReadout


class EGTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        in_dim_edge = net_params['edge_dim']
        node_hidden_dim = net_params['node_hidden_dim'] # must be divisible by num heads
        edge_hidden_dim = net_params['edge_hidden_dim']
        num_heads = net_params['n_heads']
        num_virtual_nodes = net_params['num_virtual_nodes']
        attn_drop = net_params['attn_drop']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        n_classes = net_params['n_classes']
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.loss_weight = net_params['loss_weight']
        self.batch_size = net_params['batch_size']
        self.graph_size = net_params['graph_size']
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        self.edge_update = net_params['edge_update']
        self.nfeature = nn.Linear(in_dim_node, node_hidden_dim)  # node feat is an integer
        self.efeature = nn.Linear(in_dim_edge, edge_hidden_dim)
        self.layers = nn.ModuleList([EGTLayer(node_hidden_dim, edge_hidden_dim, num_heads, num_virtual_nodes, dropout, attn_drop, edge_update=self.edge_update)
                                     for _ in range(n_layers)])
        self.MLP_layer = MLPReadout(self.node_hidden_dim, n_classes)

    def forward(self, g, h, e):

        # input embedding
        h = self.nfeature(h)
        if self.edge_update:
            e = self.efeature(e)

        e = torch.ones((h.shape[0] // self.graph_size, self.graph_size, self.graph_size, self.edge_hidden_dim)).to( self.device)
        h = h.view(h.shape[0] // self.graph_size, self.graph_size, self.node_hidden_dim)

        # EGT
        for conv in self.layers:
            if self.edge_update:
                h, e = conv(h, e)
            else:
                h = conv(h, e)

        # output
        h = h.flatten(0, 1)

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

