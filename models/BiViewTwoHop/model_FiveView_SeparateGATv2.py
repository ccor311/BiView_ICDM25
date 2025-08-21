import torch
from torch_geometric.nn import GATv2Conv
from models.BiViewTwoHop.GATv2Mask import GATv2Conv
from models.BiViewTwoHop.HeterophilyAttentionGATv2 import HeterophilyAttentionGATv2
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

class FiveViewGATv2(torch.nn.Module):
    def __init__(self, args):
        super(FiveViewGATv2, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        # Initial feature transform
        self.feature_transform = torch.nn.Linear(self.num_features, self.nhid)

        # One-hop message passing
        self.graph_hom_layers = torch.nn.ModuleList()
        self.graph_het_layers = torch.nn.ModuleList()
        # Two-hop message passing (reuse hom/het/mixed convs)
        self.hom_layers = torch.nn.ModuleList()
        self.het_layers = torch.nn.ModuleList()
        self.mixed_layers = torch.nn.ModuleList()

        # Layer 0 convs
        self.graph_hom_layers.append(GATv2Conv(self.nhid, self.nhid))
        self.graph_het_layers.append(HeterophilyAttentionGATv2(self.nhid, self.nhid))
        self.hom_layers.append(GATv2Conv(self.nhid, self.nhid))
        self.het_layers.append(HeterophilyAttentionGATv2(self.nhid, self.nhid))
        self.mixed_layers.append(HeterophilyAttentionGATv2(self.nhid, self.nhid))

        # Additional layers
        for _ in range(1, self.num_layers+1):
            in_ch = self.nhid
            out_ch = self.nhid
            # two-hop
            self.hom_layers.append(GATv2Conv(in_ch, out_ch))
            self.het_layers.append(HeterophilyAttentionGATv2(in_ch, out_ch))
            self.mixed_layers.append(HeterophilyAttentionGATv2(in_ch, out_ch))

        # Classification head: three readouts (one-hop, two-hop, one-hop) each 2*nhid, total=6*nhid
        self.lin1 = torch.nn.Linear(self.nhid * 4, self.nhid * 2)
        self.lin2 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data, last_epoch):
        x = data.x
        edge_index = data.edge_index
        two_hop_edge_index = data.two_hop_edge_index
        batch = data.batch

        # one-hop masks
        hom_mask = data.homophily_mask
        het_mask = data.heterophily_mask
        # two-hop masks
        hh_mask = data.hom_hom_mask
        mm_mask = data.mixed_mask
        tt_mask = data.het_het_mask

        if self.args.collection == 'ogb':
            x = x.float()
        if x is None:
            x = torch.ones((data.num_nodes, 1), device=edge_index.device)

        x = self.feature_transform(x)
        readout_list = [None] * 2

        # --- Layer 0: one-hop conv ---
        x1 = self.graph_convs(x, edge_index, (hom_mask, het_mask), index=0)
        #if self.args.norm:
        x1 = F.normalize(x1, p=2, dim=1)
        #readout_list[0] = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        # --- Layer 0: two-hop conv ---
        x2 = self.two_path_graph_convs(x1, two_hop_edge_index, (hh_mask, tt_mask, mm_mask), index=0)
        #if self.args.norm:
        x2 = F.normalize(x2, p=2, dim=1)
        readout_list[0] = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)


        # --- Remaining one-hop layers ---
        for i in range(1, self.num_layers):
            x3 = self.two_path_graph_convs(x2, two_hop_edge_index, (hh_mask, tt_mask, mm_mask), index=i)
            #if self.args.norm:
            x3 = F.normalize(x3, p=2, dim=1)
            readout_list[1] = torch.cat([gmp(x3, batch), gap(x3, batch)], dim=1)

        # --- Classification head ---
        x_cat = torch.cat(readout_list, dim=1)
        x = F.relu(self.lin1(x_cat))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        return F.log_softmax(self.lin3(x), dim=-1)

    def graph_convs(self, x, edge_index, masks, index):
        hm, ht = masks
        y1 = F.relu(self.graph_hom_layers[index](x, edge_index, mask=hm))
        y2 = F.relu(self.graph_het_layers[index](x, edge_index, mask=ht))
        if self.args.skip_connections:
            return torch.cat([x + y1 + y2], dim=1)
        else:
            return torch.cat([y1 + y2], dim=1)

    def two_path_graph_convs(self, x, edge_index, masks, index):
        hm, ht, mm = masks
        y1 = F.relu(self.hom_layers[index](x, edge_index, mask=hm))
        y2 = F.relu(self.het_layers[index](x, edge_index, mask=ht))
        y3 = F.relu(self.mixed_layers[index](x, edge_index, mask=mm))
        if self.args.skip_connections:
            return torch.cat([x + y1 + y2 + y3], dim=1)
        else:
            return torch.cat([y1 + y2 + y3], dim=1)
