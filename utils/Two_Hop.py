import torch
from torch_sparse import coalesce, spspmm

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops

class TwoHop(BaseTransform):
    r"""Adds a new attribute two_hop_edge_index with only the two-hop edges to the data object."""
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        # Calculate two-hop edges
        two_hop_index, two_hop_value = spspmm(edge_index, value, edge_index, value, N, N, N)
        two_hop_value.fill_(0)
        two_hop_index, two_hop_value = remove_self_loops(two_hop_index, two_hop_value)

        # Remove one-hop edges from the two-hop edges
        mask = torch.ones(two_hop_index.size(1), dtype=torch.bool)
        one_hop_edges_set = set(map(tuple, edge_index.t().tolist()))
        for i in range(two_hop_index.size(1)):
            if tuple(two_hop_index[:, i].tolist()) in one_hop_edges_set:
                mask[i] = False
        
        two_hop_index = two_hop_index[:, mask]

        # Set two_hop_edge_index attribute
        data.two_hop_edge_index = two_hop_index

        return data