# Bi-View

Official Code for Bi-View, a message-passing architecture for graph classification under homophily and heterophily.

<img src="BiViewOverview3.PNG" alt="BiView" width="1000"/>

## Requirements:

cudatoolkit==11.8.0

python==3.9

pytorch==2.2.2

torch-geometric==2.4.0

pytorch-scatter==2.1.2

pytorch-sparse==0.6.18

numpy==1.26.4

## How to run:

### TUDatasets:

    python main_BiView.py --dataset={dataset} --collection='tud' --model='BiView'

### OGB Datasets:

    python main_BiView.py --dataset={dataset} --collection='ogb' --feature_as_label=6 --model='BiView'


## Datasets
**TUDatasets** include **NCI1**, **NCI109**, **MUTAG**, **Mutagenicity**, **ER_MD**, **COX2_MD**, **PROTEINS** and **DD**.

**OGB Datasets** include **ogbg-molbace** and **ogbg-molhiv**.


## Baseline Message-Passing models:

**A-DGN**:

Retrieved from https://github.com/gravins/Anti-SymmetricDGN and https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.AntiSymmetricConv.html#torch_geometric.nn.conv.AntiSymmetricConv

**BernNet**: 

Retrieved from https://github.com/ivam-he/BernNet

**FAGCN**:

Retrieved From https://github.com/bdy9527/FAGCN and https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.FAConv.html#torch_geometric.nn.conv.FAConv

**GATv2**: 

Retrieved from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html#torch_geometric.nn.conv.GATv2Conv

**GCN**: 

Retrieved from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv

**MixHop**: 

Retrieved from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MixHopConv.html#torch_geometric.nn.conv.MixHopConv

**SGC**:

Retrieved from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SGConv.html#torch_geometric.nn.conv.SGConv
