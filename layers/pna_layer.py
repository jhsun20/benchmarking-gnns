import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import PNAConv

"""
    PNA: Principal Neighbourhood Aggregation
    Principal Neighbourhood Aggregation for Graph Nets (Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković, NeurIPS 2020)
    https://arxiv.org/abs/2004.05718
"""


class PNALayer(nn.Module):
    """
    [!] code adapted from dgl implementation of PNAConv
    """

    def __init__(self, in_dim, out_dim, aggregators, scalers, delta, num_towers, dropout, batch_norm, residual=False, activation=F.leaky_relu):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation

        if in_dim != out_dim:
            self.residual = False

        self.bn_node_h = nn.BatchNorm1d(out_dim)

        self.pnaconv = PNAConv(in_dim, out_dim, aggregators, scalers, delta, dropout, num_towers, residual=False)

    def forward(self, g, h):
        h_in = h  # for residual connection
        h = self.pnaconv(g, h)

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection

        return h
