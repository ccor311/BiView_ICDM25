import torch
from models.BiViewGATv2.GATv2Mask import GATv2Conv
from models.BiViewGATv2.GATv2Mask import GATv2Conv
from models.BiViewGATv2.HeterophilyAttentionGATv2 import HeterophilyAttentionGATv2
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class BiViewCompatibilityWeightedGATv2(torch.nn.Module):
    def __init__(self, args):
        super(BiViewCompatibilityWeightedGATv2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.dropout_ratio = args.dropout_ratio

        # bi-view conv lists
        self.hom_conv_layers = torch.nn.ModuleList()
        self.het_conv_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            in_feats = self.num_features if i == 0 else self.nhid
            self.hom_conv_layers.append(GATv2Conv(self.nhid, self.nhid, edge_dim=1))
            self.het_conv_layers.append(HeterophilyAttentionGATv2(self.nhid, self.nhid, edge_dim=1))

        self.pre_lin = torch.nn.Linear(self.num_features, self.nhid)

        # final MLP
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid * 2)
        self.lin2 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data, last_epoch=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hom_mask, het_mask = data.homophily_mask, data.heterophily_mask
        compat = data.hom_compatibility.view(-1, 1).to(x)  # shape [num_nodes, 1]
        if x is None:  #Handle case when there are no node features
            x = torch.ones((data.num_nodes, 1)).to(self.args.cuda)  #Initialize features as identity vectors
        x = x.float()  #Convert x to float
        readouts = []

        x = self.pre_lin(x)

        for i in range(self.num_layers):
            h_hom = F.relu(self.hom_conv_layers[i](x, edge_index, mask=hom_mask))
            h_het = F.relu(self.het_conv_layers[i](x, edge_index, mask=het_mask))
            # weighted sum using compat
            x = x + compat * h_hom + (1.0 - compat) * h_het
            readouts.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        # sum readouts from all layers
        x = sum(readouts)

        # MLP classifier
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        return F.log_softmax(self.lin3(x), dim=-1)
