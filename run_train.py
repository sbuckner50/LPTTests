import os, time
import cv2, random
import pickle, joblib
import sklearn.metrics
import numpy as np
np.set_printoptions(suppress=True)
import gurobipy as gp

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
from torch_geometric.data import Data

from lib.tracking import Tracker
# from lib.qpthlocal.qp import QPFunction, QPSolvers
from qpth.qp import QPFunction, QPSolvers
from lib.utils import getIoU, interpolateTrack, interpolateTracks
from pdb import set_trace as debug

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

train_data_list_full, val_data_list_full = [], []
for file in os.listdir('data/train_data/'):
    file_name = 'data/train_data/' + file
    with open(file_name, 'rb') as f:
        data_list = pickle.load(f)
         
    if file.startswith('MOT16-09') or file.startswith('MOT16-13'):
        val_data_list_full = val_data_list_full + data_list
    else:
        train_data_list_full = train_data_list_full + data_list
        
print('Total {} samples in training set, {} in validation set'.format(
                     len(train_data_list_full),len(val_data_list_full)))

train_data_list, val_data_list = [], []
for ind in range(len(train_data_list_full)):
    if train_data_list_full[ind].x.shape[0] < 200: #Avoid too big graph
        train_data_list.append(train_data_list_full[ind])
    else:
        continue
        
for ind in range(len(val_data_list_full)):
    if val_data_list_full[ind].x.shape[0] < 200:
        val_data_list.append(val_data_list_full[ind])
    else:
        continue
        
print('Used {} samples in training set, {} in validation set'.format(len(train_data_list), len(val_data_list)))

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(6,6), nn.ReLU(), nn.Linear(6,1))
    def forward(self, data):
        x = self.fc(data.edge_attr)
        x = nn.Sigmoid()(x)
        return x
    
net = Net()
tracker = Tracker(net)

gamma = 0.1
train_list, val_list = [], [] #Used for login loss, AUC, etc.
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))

num_epochs = 10
for epoch in range(num_epochs):
    np.random.shuffle(train_data_list)
    for itr in range(len(train_data_list)):
        data = train_data_list[itr]
        A_eq, b_eq, A_ub, b_ub, x_gt = tracker.build_constraint_training(data)
        
        A, b = torch.from_numpy(A_eq).float(), torch.from_numpy(b_eq).float().flatten()
        G, h = torch.from_numpy(A_ub).float(), torch.from_numpy(b_ub).float().flatten()

        num_nodes = A.shape[0] // 2
        Q = gamma * torch.eye(A.shape[1])

        prob = net(data)
        prob = torch.clamp(prob, min=1e-7, max=1-1e-7)                       #Predicted matching probability
        prob_numpy = prob.detach().numpy()                                   #Predicted matching probability in np
        auc = sklearn.metrics.roc_auc_score(x_gt[num_nodes*3: ], prob_numpy) #Area under the ROC Curve

        c_det, c_entry, c_exit = -1 * torch.ones(num_nodes), torch.ones(num_nodes), torch.ones(num_nodes)
        c_pred = -1 * torch.log(prob).squeeze()
        c_pred = torch.cat([c_det, c_entry, c_exit, c_pred])

        x = QPFunction(verbose=False, solver=QPSolvers.CVXPY, maxIter=50)(Q, c_pred, G, h, A, b)
        # model_params_quad = tracker.make_gurobi_model_tracking(G.numpy(),h.numpy(),A.numpy(),b.numpy(),Q.numpy())
        # x = QPFunction(verbose=False, solver=QPSolvers.CVXPY, maxIter=50, 
        #                model_params=model_params_quad)(Q, c_pred, G, h, A, b)
        
        loss = nn.MSELoss()(x.flatten(), torch.from_numpy(x_gt))
        loss_edge = nn.MSELoss()(x[:, num_nodes*3:].flatten(), torch.from_numpy(x_gt[num_nodes*3:]).float())

        const_cost = 1
        c_ = const_cost * (1 - x_gt[num_nodes*3:])
        c_gt = torch.cat([c_pred[:num_nodes*3], torch.from_numpy(c_).float()]) #Ground truth cost

        obj_gt = c_gt @ torch.from_numpy(x_gt).float()               #Ground truth objective value
        obj_pred = c_pred @ x.squeeze() #Predicted objective value, should be close to GT objective value after training

        bce = nn.BCELoss()(prob.flatten(), torch.from_numpy(x_gt[num_nodes*3:]).float())
        x_sol = tracker.linprog(c_pred.detach().numpy(), A_eq, b_eq, A_ub, b_ub)
        debug()
        ham_loss = sklearn.metrics.hamming_loss(x_gt, x_sol)

        train_list.append((loss.item(), loss_edge.item(), auc, bce.item()))
        print('epoch {} it [{}/{}] pr [{:.2f}-{:.2f}] obj [{:.2f}/{:.2f}] mse {:.4f} mse edge {:.4f} ce {:.3f} \
auc {:.3f} ham {:.3f}'.format(epoch, itr, len(train_data_list), prob.min(), prob.max(), 
                              obj_pred, obj_gt, loss, loss_edge, bce, auc, ham_loss))
        
        optimizer.zero_grad()
        loss_edge.backward()
        #bce.backward()
        optimizer.step()
        
    print('Saving epoch {} ...\n'.format(epoch))
    torch.save(net.state_dict(), 'ckpt/qp/epoch-{}.pth'.format(epoch))
    
    np.random.shuffle(val_data_list)
    for itr in range(len(val_data_list)):
        
        val_data = val_data_list[itr]
        A_eq, b_eq, A_ub, b_ub, x_gt = tracker.build_constraint_training(val_data)
        
        A, b = torch.from_numpy(A_eq).float(), torch.from_numpy(b_eq).float().flatten()
        G, h = torch.from_numpy(A_ub).float(), torch.from_numpy(b_ub).float().flatten()

        num_nodes = A.shape[0] // 2
        Q = gamma * torch.eye(A.shape[1])
        
        with torch.no_grad():
            prob = net(val_data)
            prob = torch.clamp(prob, min=1e-7, max=1-1e-7)
        
        prob_numpy = prob.detach().numpy() #matching probability in numpy
        auc = sklearn.metrics.roc_auc_score(x_gt[num_nodes*3: ], prob_numpy) #Area under the ROC Curve

        c_det, c_entry, c_exit = -1 * torch.ones(num_nodes), torch.ones(num_nodes), torch.ones(num_nodes)
        c_pred = -1 * torch.log(prob).squeeze()
        c_pred = torch.cat([c_det, c_entry, c_exit, c_pred])

        model_params_quad = tracker.make_gurobi_model_tracking(G.numpy(),h.numpy(),A.numpy(),b.numpy(),Q.numpy())
        x = QPFunction(verbose=False, solver=QPSolvers.CVXPY, maxIter=50)(Q, c_pred, G, h, A, b)
        # x = QPFunction(verbose=False, solver=QPSolvers.CVXPY, maxIter=50, 
        #                model_params=model_params_quad)(Q, c_pred, G, h, A, b)

        loss = nn.MSELoss()(x.flatten(), torch.from_numpy(x_gt))
        loss_edge = nn.MSELoss()(x[:, num_nodes*3:].flatten(), torch.from_numpy(x_gt[num_nodes*3:]).float())

        const_cost = 1
        c_ = const_cost * (1 - x_gt[num_nodes*3:])
        c_gt = torch.cat([c_pred[:num_nodes*3], torch.from_numpy(c_).float()]) #Ground truth cost

        obj_gt = c_gt @ torch.from_numpy(x_gt).float()               #Ground truth objective value
        obj_pred = c_pred @ x.squeeze() #Predicted objective value, should be close to GT objective value after training

        bce = nn.BCELoss()(prob.flatten(), torch.from_numpy(x_gt[num_nodes*3:]).float())
        x_sol = tracker.linprog(c_pred.detach().numpy(), A_eq, b_eq, A_ub, b_ub)
        ham_loss = sklearn.metrics.hamming_loss(x_gt, x_sol)
        val_list.append((loss.item(), loss_edge.item(), auc, bce.item()))
        print('vepo {} it [{}/{}] pr [{:.2f}-{:.2f}] obj [{:.2f}/{:.2f}] mse {:.4f} mse edge {:.4f} ce {:.3f} \
auc {:.3f} ham {:.3f}'.format(epoch, itr, len(val_data_list), prob.min(), prob.max(), 
                              obj_pred, obj_gt, loss, loss_edge, bce, auc, ham_loss))