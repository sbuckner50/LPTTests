import networkx as nx
import pickle as pickle
import os
from pdb import set_trace as debug
from torch_geometric.data import Data
import numpy as np
import torch
    
def get_torchgeometric_data(fname):
    with open(fname, 'rb') as f:
        data_input = pickle.load(f)
        
    gt_inds = data_input[0]
    G = data_input[1]

    num_nodes = len(list(G.nodes))
    num_edges = len(list(G.edges))

    # Instantiate a pytorch_geometric Data object to be filled in
    data = Data()
    data.x = torch.zeros((num_nodes,0)) # TBD, need pixel images
    data.edge_index = torch.zeros((2,num_edges))
    data.edge_attr = torch.zeros((num_edges,0)) # TBD, need ReID info to fill this in
    data.ground_truth = torch.zeros((num_nodes,6))
    data.y = torch.zeros(num_edges)

    # Other preprocessing
    gt_inds_flat = np.array([]).reshape(2,0)
    for inds in gt_inds.values():
        nds = np.array(inds[:-2]).reshape((1,-1))
        cons = np.array(inds[1:-1]).reshape((1,-1))
        pairs = np.vstack((nds,cons))
        gt_inds_flat = np.hstack((gt_inds_flat, pairs))

    # Obtain node feature vectors
    for k in range(num_nodes):
        fv = []
        fv.append(G.nodes[k]['time_ndx'])
        fv.append(G.nodes[k]['det_id'])
        fv.extend(list(G.nodes[k]['feats_mvg']))
        data.ground_truth[k,:] = torch.tensor(fv)

    # Obtain edge associations and labels
    count = 0
    for nd in G.adj:
        for con in G.adj[nd]:
            data.edge_index[:,count] = torch.tensor([nd,con])
            count += 1
            for k in range(gt_inds_flat.shape[1]):
                if (np.array([nd,con]) == gt_inds_flat[:,k]).all():
                    data.y[count] = 1.
                    gt_inds_flat = np.delete(gt_inds_flat, (k), axis=1)
                    break

    return data

def reduce_window_size(data, window=10):
    # Obtain node IDs that need to be removed
    ground_truth = np.concatenate((data.ground_truth, np.arange(data.ground_truth.shape[0])[:, None]), axis=1)
    timestamps = ground_truth[:, 0].astype(int)
    nodes_keep = np.where(timestamps <= window)[0]
    assert timestamps.max() >= window
    
    # Remove stuff
    data = data.clone()
    data.x = data.x[nodes_keep,:]
    data.ground_truth = data.ground_truth[nodes_keep,:]
    edges_keep = []
    for k in range(data.edge_index.shape[1]):
        src_node = int(data.edge_index[0,k])
        dst_node = int(data.edge_index[1,k])
        if src_node in nodes_keep and dst_node in nodes_keep:
            edges_keep.append(k)
    data.edge_index = data.edge_index[:,edges_keep]
    data.edge_attr = data.edge_attr[edges_keep,:]
    data.y = data.y[edges_keep]
    
    return data