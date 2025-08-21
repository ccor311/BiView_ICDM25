import torch
import numpy as np



def ogb_generate_homophily_mask(edge_index, node_labels, index):
    #Convert one-hot labels to class labels
    node_labels = node_labels[:,index]
    row, col = edge_index
    mask = node_labels[row] == node_labels[col]
    return mask
    
def ogb_generate_heterophily_mask(edge_index, node_labels, index):
    #Convert one-hot labels to class labels
    node_labels = node_labels[:,index]
    row, col = edge_index
    mask = node_labels[row] != node_labels[col]
    return mask

#Create sparse COO homophily matrix to act as mask from sparse COO edge_index
def generate_homophily_mask(edge_index, node_labels):
    #Convert one-hot labels to class labels
    node_labels = np.argmax(node_labels, axis=1)
    row, col = edge_index
    mask = node_labels[row] == node_labels[col]
    return mask
    
def generate_heterophily_mask(edge_index, node_labels):
    #Convert one-hot labels to class labels
    node_labels = np.argmax(node_labels, axis=1)
    row, col = edge_index
    mask = node_labels[row] != node_labels[col]
    return mask

def generate_random_masks(edge_index):
    #Generate random mask of Trues and Falses for edge_index as tensor
    row, col = edge_index
    mask1 = np.random.randint(2, size=len(row))
    mask1 = mask1.astype(bool)
    mask1 = torch.tensor(mask1, dtype=torch.bool)
    #Generate opposite of mask 1
    mask2 = torch.logical_not(mask1)
    return mask1, mask2
