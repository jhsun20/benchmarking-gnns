import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
from dgl.nn.pytorch import GMMConv
"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

class GMMLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GMMConv

    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    dim : 
        Dimensionality of pseudo-coordinte.
    kernel : 
        Number of kernels :math:`K`.
    aggr_type : 
        Aggregator type (``sum``, ``mean``, ``max``).
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    bias : 
        If True, adds a learnable bias to the output. Default: ``True``.
    
    """
    def __init__(self, in_dim, out_dim, dim, kernel, aggr_type, dropout,
                 batch_norm, residual=False, activation=F.leaky_relu):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.kernel = kernel
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.aggr_type = aggr_type
        self.gmmconv = GMMConv(in_dim, out_dim, dim, kernel, aggr_type, residual=False, bias=True,  allow_zero_in_degree=True)
        self.bn_node_h = nn.BatchNorm1d(out_dim)
        
        if in_dim != out_dim:
            self.residual = False
    
    def forward(self, g, h, pseudo):
        h_in = h
        h = self.gmmconv(g, h, pseudo)

        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  

        if self.activation:
            h = self.activation(h)

        if self.dropout:
            h = self.dropout(h)

        if self.residual:
            h = h + h_in

        return h
