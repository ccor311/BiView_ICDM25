import torch
#from torch_geometric.nn import GATConv
from torch_geometric.nn import FAConv
from models.BiViewFAGCN.FAMaskConv import FAMaskConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric import utils
from torch_sparse import mul

class BiViewFAGCN(torch.nn.Module):
    def __init__(self, args):
        super(BiViewFAGCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.which_g = 0
        self.num_layers = args.num_layers

        self.hom_conv_layers = torch.nn.ModuleList()
        self.het_conv_layers = torch.nn.ModuleList()    
        self.pre_lin = torch.nn.Linear(self.num_features, self.nhid)
        for i in range(self.num_layers):
            if i == 0:
                if self.num_features > 0:
                    in_features = self.num_features
                else:
                    in_features = 1
            else:
                #in_features = self.nhid
                in_features = self.nhid * 2

            out_features = self.nhid
            #Conv Layer
            self.hom_conv_layers.append(FAMaskConv(channels=out_features))

            #Heterophily Conv Layer
            self.het_conv_layers.append(FAMaskConv(channels=out_features))

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
        
    def forward(self, data, last_epoch):
        x, edge_index, batch, hom_mask, het_mask = data.x, data.edge_index, data.batch, data.homophily_mask, data.heterophily_mask
        x = x.float()  # Convert x to float

        if x is None:  #Handle case when there are no node features
            x = torch.ones((data.num_nodes, 1)).to(self.args.cuda)  #Initialize features as identity vectors
        readout_list = []
        #Begin forward pass
        x = self.pre_lin(x)

        x_hom_out = F.relu(self.hom_conv_layers[0](x, x, edge_index, mask=hom_mask))
        x_het_out = F.relu(self.het_conv_layers[0](x, x, edge_index, mask=het_mask))
        #Sum
        x_out = x_hom_out + x_het_out
        readout_list.append(torch.cat([gmp(x_out, batch), gap(x_out, batch)], dim=1))

        for i in range(1,self.args.num_layers):
            hom_conv_layer = self.hom_conv_layers[i]
            het_conv_layer = self.het_conv_layers[i]
            x_hom_out = F.relu(hom_conv_layer(x_hom_out, x, edge_index, mask=hom_mask))
            x_het_out = F.relu(het_conv_layer(x_het_out, x, edge_index, mask=het_mask))
            x_out = x_hom_out + x_het_out
            readout_list.append(torch.cat([gmp(x_out, batch), gap(x_out, batch)], dim=1))

        #Add readout layers
        x_out = sum(readout_list)

        x_out = F.relu(self.lin1(x_out))
        x_out = F.dropout(x_out, p=self.dropout_ratio, training=self.training)
        x_out = F.relu(self.lin2(x_out))
        x_out = F.log_softmax(self.lin3(x_out), dim=-1)

        return x_out