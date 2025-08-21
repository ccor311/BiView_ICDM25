from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from models.BiViewMixHop.gcn_norm import gcn_norm

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class MixHopMaskConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        powers: Optional[List[int]] = None,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if powers is None:
            powers = [0, 1, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.powers = powers
        self.add_self_loops = add_self_loops

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False)
            if p in powers else torch.nn.Identity()
            for p in range(max(powers) + 1)
        ])

        if bias:
            #self.bias = Parameter(torch.empty(len(powers) * out_channels))
            self.bias = Parameter(torch.empty(1 * out_channels))

        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                mask: OptTensor = None,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight, mask = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype, mask=mask)
        elif isinstance(edge_index, SparseTensor):
            edge_index, mask = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype, mask=mask)

        outs = [self.lins[0](x)]

        for lin in self.lins[1:]:
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, mask=mask)

            outs.append(lin.forward(x))

        #out = torch.cat([outs[p] for p in self.powers], dim=-1)
        out = sum(outs[p] for p in self.powers)
        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor, mask: OptTensor) -> Tensor:
        mask = mask[:, None]
        x_j  = x_j * mask
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    # def propagate(self, edge_index, size=None, **kwargs):
    #     # Custom propagate method to include the mask
    #     mask = kwargs.get('mask', None)
        
    #     return super(MixHopMaskConv, self).propagate(edge_index, size=size, mask=mask, **kwargs)


    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, powers={self.powers})')
