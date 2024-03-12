import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
    For more information regarding the code and hyperparameter ranges:
    https://docs.dgl.ai/en/2.0.x/generated/dgl.nn.pytorch.conv.SAGEConv.html
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator'] # mean, gcn, pool, lstm
        n_layers = net_params['L']   
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.loss_weight = net_params['loss_weight']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, aggregator_type, dropout, batch_norm, residual=residual)
                                     for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, aggregator_type, dropout, batch_norm, residual=residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e):

        # input embedding
        h = self.embedding_h(h)

        # graphsage
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



        
