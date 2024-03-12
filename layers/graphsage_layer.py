import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import SAGEConv

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""


class GraphSageLayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type, dropout, batch_norm, residual=False, activation=F.leaky_relu):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation
        
        if in_feats != out_feats:
            self.residual = False

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
        else:
            self.batchnorm_h = None

        self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type, dropout, norm=self.batchnorm_h, activation=self.activation)

    def forward(self, g, h):
        h_in = h              # for residual connection

        h = self.sageconv(g, h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
