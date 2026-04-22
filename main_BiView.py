import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from models.BiViewGATv2.model_BiView_HeterophilyAttention_GATv2_concat import BiView
from models.BiViewADGN.model_BiView_ADGN import BiViewADGN
from models.BiViewSGC.model_BiView_SGC import BiViewSGC
from models.BiViewBernNet.model_BiView_BernNet import BiViewBernNet
from models.BiViewFAGCN.model_BiView_FAGCN import BiViewFAGCN
from models.BiViewGCN.model_BiView_GCN import BiViewGCN
from models.BiViewMixHop.model_BiView_MixHop import BiViewMixHop
from models.BiViewGATv2.model_BiView_HeterophilyAttention_GATv2_degree_normalized import BiViewDegreeNormalized
from models.BiViewTwoHop.model_BiView_TwoHop import BiViewTwoHop
from models.BiViewTwoHop.model_BiView_TwoHop_sum import BiViewTwoHopSum
from models.BiViewGATv2.model_BiView_GATv2_node_wise_compatibility_combine import BiViewCompatibilityWeightedGATv2
from models.BiViewTwoHop.model_FiveView_SeparateGATv2 import FiveViewGATv2
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from datetime import datetime
import random
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score
from utils.ogb_mask_generators import generate_homophily_mask, generate_heterophily_mask, generate_random_masks, ogb_generate_homophily_mask, ogb_generate_heterophily_mask
from utils.Two_Hop import TwoHop
from torch_geometric.utils import to_undirected

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of pooling layers')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden dimension')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout')
parser.add_argument('--dataset', type=str, default='ogbg-molbace',
                    help='COX2_MD/ER_MD/DD/MUTAG/Mutagenicity/NCI1/NCI109/PROTEINS/ogbg-molhiv/ogbg-molbace')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200,
                    help='max epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience')
parser.add_argument('--cuda', type=str, default='cuda:0',
                    help='which gpu')
parser.add_argument('--seed', type=int, default=10,
                    help='seed')
parser.add_argument('--model_save', type=str, default='latest.pth',
                    help='model save file name')
parser.add_argument('--feature_as_label', type=int, default='6',
                    help='which feature to use as label, set between 0-8, only for ogb datasets')
parser.add_argument('--collection', type=str, default='ogb',
                    help='Dataset source: set to tud OR ogb')
parser.add_argument('--model', type=str, default='BiView',
                    help='eg. BiView')
parser.add_argument('--norm', type=bool, default=False,
                    help='Whether to normalize node embeddings after each MP layer')
parser.add_argument('--skip_connections', type=bool, default=False,
                    help='Whether to use skip connections')

torch.set_printoptions(precision=2, sci_mode=False)
args = parser.parse_args()
args.device = args.cuda
torch.cuda.set_device(int(args.device[-1]))  #Set the current device to cuda:1
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.collection == 'tud':
    dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
    samples = 20
elif args.collection == 'ogb':
    dataset = PygGraphPropPredDataset(name = args.dataset, root = 'dataset/')
    samples = 10
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)


def test_roc(model, loader):
    model.eval()
    correct = 0.
    total_loss = 0.
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            out = model(data, False) 
            
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            
            all_labels.append(data.y.cpu()) 
            all_preds.append(out.cpu()) 

            #Loss
            total_loss += F.nll_loss(out, data.y, reduction='sum').item()

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    #Convert log probabilities back to normal probabilities
    probs = torch.exp(all_preds)

    #If binary classification, extract the probability of the positive class
    if probs.size(1) == 2:  
        positive_probs = probs[:, 1]  
    else: 
        positive_probs = probs

    #Compute ROC-AUC
    if probs.size(1) == 2: 
        roc_auc = roc_auc_score(all_labels.cpu(), positive_probs.cpu())
    else: 
        all_labels_one_hot = F.one_hot(all_labels, num_classes=probs.size(1))
        roc_auc = roc_auc_score(all_labels_one_hot.cpu(), positive_probs.cpu(), multi_class='ovr')

    #Return accuracy, loss, and ROC-AUC
    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return accuracy, avg_loss, roc_auc

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data, False)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def validate(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data, False)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()

    return correct / len(loader.dataset),loss / len(loader.dataset)


def compute_edge_degrees(data):
    #Compute node degrees from the existing edges
    node_degrees = torch.bincount(data.edge_index[0], minlength=data.x.shape[0]).float()

    #Create per-edge degree tensor (same shape as edge_index)
    source_degrees = node_degrees[data.edge_index[0]]  #Degree of source node
    target_degrees = node_degrees[data.edge_index[1]]  #Degree of target node
    edge_degrees = torch.stack([source_degrees, target_degrees], dim=1)


    #Self-loop degrees: The degree of each node is used for both source and target
    self_loop_degrees = torch.stack([node_degrees, node_degrees], dim=1)

    #Append self-loop degrees to edge_degrees
    edge_degrees = torch.cat([edge_degrees, self_loop_degrees], dim=0)

    return edge_degrees

def compute_hom_het_edge_degrees(data, collection, feature_as_label=None):
    """Computes the homophily and heterophily degrees for each node and assigns
    them to the corresponding edges."""
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    #Extract node labels based on dataset type
    if collection == 'tud':
        node_labels = data.x.argmax(dim=1)  
    elif collection == 'ogb' and feature_as_label is not None:
        node_labels = data.x[:, feature_as_label].long() 
    else:
        raise ValueError("Unknown dataset collection type or missing feature_as_label if OGB dataset.")

    #Compute total node degrees
    #node_degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

    #Identify homophilous and heterophilous edges
    same_label = node_labels[edge_index[0]] == node_labels[edge_index[1]]
    different_label = ~same_label

    #Compute homophily and heterophily degrees per node
    homophily_degrees = torch.zeros(num_nodes, device=data.x.device).scatter_add_(
        0, edge_index[0][same_label], torch.ones_like(edge_index[0][same_label], dtype=torch.float)
    ) + 1  #Add 1 to account for self-loop

    heterophily_degrees = torch.zeros(num_nodes, device=data.x.device).scatter_add_(
        0, edge_index[0][different_label], torch.ones_like(edge_index[0][different_label], dtype=torch.float)
    ) + 1  #Add 1 to account for self-loop

    #Assign per-edge homophily and heterophily degrees
    hom_edge_degrees = torch.stack([homophily_degrees[edge_index[0]], homophily_degrees[edge_index[1]]], dim=1)
    het_edge_degrees = torch.stack([heterophily_degrees[edge_index[0]], heterophily_degrees[edge_index[1]]], dim=1)

    #Handle self-loops
    self_loop_degrees_hom = torch.stack([homophily_degrees, homophily_degrees], dim=1)
    self_loop_degrees_het = torch.stack([heterophily_degrees, heterophily_degrees], dim=1)

    #Append self-loop degrees to edge_degrees
    hom_edge_degrees = torch.cat([hom_edge_degrees, self_loop_degrees_hom], dim=0)
    het_edge_degrees = torch.cat([het_edge_degrees, self_loop_degrees_het], dim=0)

    return hom_edge_degrees, het_edge_degrees

def node_hom_compatibility(data, args):
    # … your existing mask code …

    # ------------- new block begins -------------
    # Compute per-graph class compatibility and assign diagonal entries to nodes
    # 1) Extract node labels
    try:
        node_labels = data.x.argmax(dim=1)
    except:
        node_labels = data.x[:, args.feature_as_label].long()
    num_classes = int(node_labels.max().item()) + 1

    # 2) Build class-to-class edge counts
    compat_counts = torch.zeros((num_classes, num_classes), device=data.x.device)
    src, dst = data.edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        cu = node_labels[u].item()
        cv = node_labels[v].item()
        compat_counts[cu, cv] += 1

    # 3) Row-normalize to get compatibility proportions
    row_sums = compat_counts.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    compat_matrix = compat_counts / row_sums  # shape [C, C]

    # 4) Extract diagonal: homophily per class
    compat_diag = compat_matrix.diag()  # shape [C]

    # 5) Assign to each node the compatibility of its class
    data.hom_compatibility = compat_diag[node_labels]  # shape [num_nodes]
    return compat_diag[node_labels]


#Train and test model
modified_dataset = []
hom_mask_sum = 0
het_mask_sum = 0
for i in range(len(dataset)):
    data = dataset[i]
    #Generate Masks
    if data.y.dim() > 1: 
        data.y = data.y[0]
    if args.model == 'BiViewTwoHop' or args.model == 'BiViewTwoHopSum':
        two_hop = TwoHop()
        data = two_hop(data)
        if args.collection == 'ogb':
            data.homophily_mask = ogb_generate_homophily_mask(data.edge_index, data.x, args.feature_as_label)
            data.heterophily_mask = ogb_generate_heterophily_mask(data.edge_index, data.x, args.feature_as_label)
            data.two_hop_homophily_mask = ogb_generate_homophily_mask(data.two_hop_edge_index, data.x, args.feature_as_label)
            data.two_hop_heterophily_mask = ogb_generate_heterophily_mask(data.two_hop_edge_index, data.x, args.feature_as_label)
        else:
            data.homophily_mask = generate_homophily_mask(data.edge_index, data.x)
            data.heterophily_mask = generate_heterophily_mask(data.edge_index, data.x)
            data.two_hop_homophily_mask = generate_homophily_mask(data.two_hop_edge_index, data.x)
            data.two_hop_heterophily_mask = generate_heterophily_mask(data.two_hop_edge_index, data.x)
    
    else:
        if args.collection == 'ogb':
            data.homophily_mask = ogb_generate_homophily_mask(data.edge_index, data.x, args.feature_as_label)
            data.heterophily_mask = ogb_generate_heterophily_mask(data.edge_index, data.x, args.feature_as_label)
        else:
            data.homophily_mask = generate_homophily_mask(data.edge_index, data.x)
            data.heterophily_mask = generate_heterophily_mask(data.edge_index, data.x)

    #Random Mask
    #data.homophily_mask, data.heterophily_mask = generate_random_masks(data.edge_index)
    data.hom_compatibility = node_hom_compatibility(data, args)

    #Add degrees
    data.edge_degrees = compute_edge_degrees(data)
    #Compute homophily and heterophily edge degrees
    data.hom_edge_degrees, data.het_edge_degrees = compute_hom_het_edge_degrees(data, args.collection, args.feature_as_label)
    # 2-hop masks
    if args.model == 'FiveViewGATv2':
        src, dst = data.edge_index
        hom_mask = data.homophily_mask
        two_hop_src, two_hop_dst = [], []
        hom_hom, mixed, het_het = [], [], []
        for e1 in range(src.size(0)):
            i_node = int(src[e1]); j_node = int(dst[e1])
            type1 = bool(hom_mask[e1])
            # find edges i->j->k
            mask2 = (src == j_node)
            idx2 = mask2.nonzero(as_tuple=False).view(-1)
            for e2 in idx2:
                k_node = int(dst[int(e2)])
                if k_node == i_node:
                    continue
                type2 = bool(hom_mask[int(e2)])
                two_hop_src.append(i_node)
                two_hop_dst.append(k_node)
                if type1 and type2:
                    hom_hom.append(True); mixed.append(False); het_het.append(False)
                elif (not type1) and (not type2):
                    hom_hom.append(False); mixed.append(False); het_het.append(True)
                else:
                    hom_hom.append(False); mixed.append(True); het_het.append(False)
        data.two_hop_edge_index = torch.stack([
            torch.tensor(two_hop_src, dtype=torch.long),
            torch.tensor(two_hop_dst, dtype=torch.long)
        ], dim=0)
        data.hom_hom_mask = torch.tensor(hom_hom, dtype=torch.bool)
        data.mixed_mask = torch.tensor(mixed, dtype=torch.bool)
        data.het_het_mask = torch.tensor(het_het, dtype=torch.bool)
    modified_dataset.append(data)

#Use ogb pre-defined splits
if args.collection == 'ogb':
    split_idx = dataset.get_idx_split() 
    train_indices = split_idx["train"].tolist()
    valid_indices = split_idx["valid"].tolist()
    test_indices = split_idx["test"].tolist()

    train_loader = DataLoader([modified_dataset[i] for i in train_indices], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([modified_dataset[i] for i in valid_indices], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader([modified_dataset[i] for i in test_indices], batch_size=1, shuffle=False)
#tud split
else:
    training_set, validation_set, test_set = random_split(modified_dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)


if args.model == 'BiView':
    model = BiView(args).to(args.device)
elif args.model == 'BiViewDegreeNormalized':
    model = BiViewDegreeNormalized(args).to(args.device)
elif args.model == 'BiViewTwoHop':
    model = BiViewTwoHop(args).to(args.device)
elif args.model == 'BiViewTwoHopSum':
    model = BiViewTwoHopSum(args).to(args.device)
elif args.model == 'BiViewADGN':
    model = BiViewADGN(args).to(args.device)
elif args.model == 'BiViewSGC':
    model = BiViewSGC(args).to(args.device)
elif args.model == 'BiViewBernNet':
    model = BiViewBernNet(args).to(args.device)
elif args.model == 'BiViewFAGCN':
    model = BiViewFAGCN(args).to(args.device)
elif args.model == 'BiViewGCN':
    model = BiViewGCN(args).to(args.device)
elif args.model == 'BiViewMixHop':
    model = BiViewMixHop(args).to(args.device)
elif args.model == 'BiViewCompatibilityWeightedGATv2':
    model = BiViewCompatibilityWeightedGATv2(args).to(args.device)
elif args.model == 'FiveViewGATv2':
    model = FiveViewGATv2(args).to(args.device)   
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
min_loss = 1e10
patience = 0
for epoch in range(args.epochs):
    if epoch == 0:
        torch.cuda.reset_peak_memory_stats(args.device)
    print(epoch)
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data, False)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = validate(model,val_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),args.model_save)
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        model.train()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            out = model(data, False)
            loss = F.nll_loss(out, data.y)
            print("Training loss:{}".format(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc, val_loss = validate(model, val_loader)
        print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), args.model_save)
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
        break

if args.model == 'BiView':
    model = BiView(args).to(args.device)
elif args.model == 'BiViewDegreeNormalized':
    model = BiViewDegreeNormalized(args).to(args.device)
elif args.model == 'BiViewTwoHop':
    model = BiViewTwoHop(args).to(args.device)
elif args.model == 'BiViewTwoHopSum':
    model = BiViewTwoHopSum(args).to(args.device)
elif args.model == 'BiViewADGN':
    model = BiViewADGN(args).to(args.device)
elif args.model == 'BiViewSGC':
    model = BiViewSGC(args).to(args.device)
elif args.model == 'BiViewBernNet':
    model = BiViewBernNet(args).to(args.device)
elif args.model == 'BiViewFAGCN':
    model = BiViewFAGCN(args).to(args.device)
elif args.model == 'BiViewGCN':
    model = BiViewGCN(args).to(args.device)
elif args.model == 'BiViewMixHop':
    model = BiViewMixHop(args).to(args.device)
elif args.model == 'BiViewCompatibilityWeightedGATv2':
    model = BiViewCompatibilityWeightedGATv2(args).to(args.device)
elif args.model == 'FiveViewGATv2':
    model = FiveViewGATv2(args).to(args.device)  
model.load_state_dict(torch.load(args.model_save))

if args.collection == 'tud':
    test_acc,test_loss = test(model,test_loader)
    print("Test accuracy:{}".format(test_acc))
elif args.collection == 'ogb':
    test_acc, test_loss, roc_auc = test_roc(model, test_loader)
    print("ROC-AUC:{}".format(roc_auc))