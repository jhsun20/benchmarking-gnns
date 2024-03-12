"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.MVC_node_classification.gated_gcn_net import GatedGCNNet
from nets.MVC_node_classification.gcn_net import GCNNet
from nets.MVC_node_classification.gat_net import GATNet
from nets.MVC_node_classification.graphsage_net import GraphSageNet
from nets.MVC_node_classification.mlp_net import MLPNet
from nets.MVC_node_classification.gin_net import GINNet
from nets.MVC_node_classification.gmm_net import GMMNet
from nets.MVC_node_classification.pna_net import PNANet
from nets.MVC_node_classification.egt_net import EGTNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GMM(net_params):
    return GMMNet(net_params)

def PNA(net_params):
    return PNANet(net_params)

def EGT(net_params):
    return EGTNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'GMM': GMM,
        'PNA': PNA,
        'EGT': EGT
    }
        
    return models[MODEL_NAME](net_params)
