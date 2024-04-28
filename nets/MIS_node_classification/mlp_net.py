import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout


class MLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.loss_weight = net_params['loss_weight']
        self.input = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        
        feat_mlp_modules = [
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(nn.LeakyReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        self.readout_mlp = MLPReadout(hidden_dim, n_classes)


    def forward(self, g, h, e):
        # MLP
        h = self.input(h)
        h = self.feat_mlp(h)
        
        # output
        h_out = self.readout_mlp(h)

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



        
