import os, time
import cv2, random
import pickle, joblib
import sklearn.metrics
import numpy as np
np.set_printoptions(suppress=True)
import gurobipy as gp
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
from torch_geometric.data import Data

from lib.tracking import Tracker
# from lib.qpthlocal.qp import QPFunction, QPSolvers
from qpth.qp import QPFunction, QPSolvers
from lib.utils import getIoU, interpolateTrack, interpolateTracks
from processing import get_torchgeometric_data, reduce_window_size
from pdb import set_trace as debug

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

data_folder = 'data_sspgnn/'
data_list = []
for file in os.listdir(data_folder):
    filename = data_folder + file
    data = get_torchgeometric_data(filename)
    data_list.append(data)
    
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
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
windows = [3,6,10,15,20]

time_results = np.zeros((len(windows),len(data_list)))
for witr,window in enumerate(windows):
    for ditr,data in enumerate(data_list):
        data = reduce_window_size(data, window=window)
        A_eq, b_eq, A_ub, b_ub, x_gt = tracker.build_constraint_training(data)
        num_nodes = A_eq.shape[0] // 2
        c_det, c_entry, c_exit = -1 * np.ones(num_nodes), np.ones(num_nodes), np.ones(num_nodes)
        c_prob_gt = 1 - x_gt[num_nodes*3:]
        c_gt = np.concatenate((c_det, c_entry, c_exit, c_prob_gt))
        x_sol, elapsed_time = tracker.linprog(c_gt, A_eq, b_eq, A_ub, b_ub)
        time_results[witr,ditr] = elapsed_time

means = time_results.mean(1)
perc25 = np.percentile(time_results, 0.25, axis=1).reshape((1,-1))
perc75 = np.percentile(time_results, 0.75, axis=1).reshape((1,-1))
perc = np.vstack((perc25,perc75))
# ulim = time_results.max()
# llim = time_results.min()

plt.figure()
plt.rcParams['text.usetex'] = True
fontsize = 12
ax = plt.errorbar(windows, means, yerr=perc, color='black', solid_capstyle='projecting', capsize=5)
plt.xticks(windows)
# plt.yscale('log')
plt.xlabel(r'Window size', fontsize=fontsize)
plt.ylabel(r'Elapsed solver time [$s$]', fontsize=fontsize)
plt.savefig('figures/timing_result.pdf', bbox_inches='tight')