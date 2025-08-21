import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric import utils
from torch_sparse import mul
from models.BiViewBernNet.Bern import BernConv
from torch.nn import Parameter

class BiViewBernNet(torch.nn.Module):
    def __init__(self, args, K=10):
        super(BiViewBernNet, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.which_g = 0
        self.num_layers = args.num_layers

        #Initial linear layer
        self.feature_transform = torch.nn.Linear(self.num_features, self.nhid)
        self.coe = Parameter(torch.Tensor(K+1))


        self.hom_conv_layers = torch.nn.ModuleList()
        self.het_conv_layers = torch.nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                if self.num_features > 0:
                    in_features = self.num_features
                else:
                    in_features = 1
            else:
                in_features = self.nhid * 2

            out_features = self.nhid 

            self.hom_conv_layers.append(BernConv(in_features, out_features, K))#, edge_dim=1))
            self.het_conv_layers.append(BernConv(in_features, out_features, K))#, edge_dim=1))
        self.lin1 = torch.nn.Linear(self.nhid * 4, self.nhid * 2)
        self.lin2 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)
        

    def forward(self, data, last_epoch):
        x, edge_index, batch, hom_mask, het_mask = data.x, data.edge_index, data.batch, data.homophily_mask, data.heterophily_mask
        x = x.float()  # Convert x to float

        if x is None:  #Handle case when there are no node features
            x = torch.ones((data.num_nodes, 1)).to(self.args.cuda)  #Initialize features as identity vectors
        readout_list = []
        #Begin forward pass
        #x = self.pre_lin(x)

        x_hom = F.relu(self.hom_conv_layers[0](x, edge_index, self.coe, mask=hom_mask))
        x_het = F.relu(self.het_conv_layers[0](x, edge_index, self.coe, mask=het_mask))
        #Sum
        x = torch.cat((x_hom, x_het), dim=1)
        readout_list.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        for i in range(1,self.args.num_layers):
            hom_conv_layer = self.hom_conv_layers[i]
            het_conv_layer = self.het_conv_layers[i]
            x_hom = F.relu(hom_conv_layer(x, edge_index, self.coe, mask=hom_mask))
            x_het = F.relu(het_conv_layer(x, edge_index, self.coe, mask=het_mask))
            x = torch.cat((x_hom, x_het), dim=1)
            readout_list.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            
        #Add readout layers
        x = sum(readout_list)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x