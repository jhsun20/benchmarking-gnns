import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_u(u='h', out='m')
reduce = fn.mean('m', 'h')

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, batch_norm, in_feat_dropout, residual=False, activ=F.elu):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activ
        self.dropout = nn.Dropout(in_feat_dropout)
        self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)

    def forward(self, g, feature):

        h = self.dropout(feature)
        h_in = h   # to be used for residual connection
        h = self.conv(g, h)

        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h # residual connection


        return h