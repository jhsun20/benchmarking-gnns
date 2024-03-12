import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import EGTLayer as EGTConv

"""
    EGT: l Edge-augmented Graph Transformer
    Global Self-Attention as a Replacement for Graph Convolution (Md Shamim Hussain, Mohammed J. Zaki, Dharmashankar Subramanian, ACM SIGKDD 2022)
    https://arxiv.org/pdf/2108.03348.pdf
"""


class EGTLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of PNAConv
    """

    def __init__(self, in_dim, edge_dim, num_heads, num_virtual_nodes, dropout, attn_dropout, edge_update=False):
        super().__init__()
        self.edge_update = edge_update
        self.egtconv = EGTConv(in_dim, edge_dim, num_heads, num_virtual_nodes, dropout, attn_dropout, edge_update=False)

    def forward(self, h, e):
        if self.edge_update:
            h, e = self.egtconv(h, e)
        else:

            h = self.egtconv(h, e)

        if self.edge_update:
            return h, e
        else:
            return h
