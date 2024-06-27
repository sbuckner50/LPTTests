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

# string parsing utility fcns
get_between = lambda s,s_,_s : s[s.find(s_) + len(s_) : s.find(_s)]
get_after = lambda s,s_ : s[s.find(s_) + len(s_):]

# parse data folder
data_folder = 'data_sspgnn/'
dataset = {}
k_set = []
fa_set = []
T_set = []
n_samples = 1e6
max_window = 1e6
# max_window = 50
for subfolder in os.listdir(data_folder):
    k = int(get_between(subfolder,'k','_fa'))
    fa = int(get_between(subfolder,'_fa','_T'))
    T = int(get_after(subfolder,'_T'))
    if T < max_window:
        print('Loading: ' + subfolder)
        dataset[(k,fa,T)] = []
        for file in os.listdir(data_folder+subfolder):
            filename = data_folder + subfolder + '/' + file
            data,n_nodes,n_edges = get_torchgeometric_data(filename)
            dataset[(k,fa,T)].append((data,n_nodes,n_edges))
        k_set.append(k)
        fa_set.append(fa)
        T_set.append(T)
        n_samples = min(n_samples,len(dataset[(k,fa,T)]))

# remove duplicates
k_set = list(set(k_set))
fa_set = list(set(fa_set))
T_set = list(set(T_set))
k_set.sort()
fa_set.sort()
T_set.sort()
    
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

n_recomps = 5
time_results = np.zeros((len(k_set),len(fa_set),len(T_set),n_samples,n_recomps))
node_counts = np.zeros((len(k_set),len(fa_set),len(T_set),n_samples))
edge_counts = np.zeros((len(k_set),len(fa_set),len(T_set),n_samples))
for kitr,k in enumerate(k_set):
    for faitr,fa in enumerate(fa_set):
        for Titr,T in enumerate(T_set):
            print('Testing: k='+str(k)+', fa='+str(fa)+', T='+str(T))
            for ditr,data in enumerate(dataset[(k,fa,T)]):
                for recitr in range(n_recomps):
                    data_raw = data[0]
                    num_nodes = data[1]
                    num_edges = data[2]
                    A_eq, b_eq, A_ub, b_ub, x_gt = tracker.build_constraint_training(data_raw)
                    num_nodes_ = A_eq.shape[0] // 2
                    assert num_nodes_ == num_nodes
                    c_det, c_entry, c_exit = -1 * np.ones(num_nodes), np.ones(num_nodes), np.ones(num_nodes)
                    c_prob_gt = 1 - x_gt[num_nodes*3:]
                    c_gt = np.concatenate((c_det, c_entry, c_exit, c_prob_gt))
                    x_sol, elapsed_time = tracker.linprog(c_gt, A_eq, b_eq, A_ub, b_ub)
                    time_results[kitr,faitr,Titr,ditr,recitr] = elapsed_time
                    node_counts[kitr,faitr,Titr,ditr] = num_nodes
                    edge_counts[kitr,faitr,Titr,ditr] = num_edges

means = time_results.mean(axis=(3,4))
perc25 = np.percentile(time_results, 25, axis=(3,4))
perc75 = np.percentile(time_results, 75, axis=(3,4))
cstack = lambda x1,x2 : np.vstack((x1.reshape((1,-1)),x2.reshape((1,-1))))

means_samp = time_results.mean(axis=4)
perc25_samp = np.percentile(time_results, 25, axis=4)
perc75_samp = np.percentile(time_results, 75, axis=4)

# Plotting config
colors = ('#264653','#2A9D8F','#F4A261','#E76F51') # colors across fa and k enumerations (len = n_fa*n_k)
kfa_label = lambda k,fa : r'$k='+str(k)+',\:n_{FA}='+str(fa)+'$'
fontsize = 12

# Plot compare against window size
plt.figure()
plt.rcParams['text.usetex'] = True
colcount = 0
for faitr,fa in enumerate(fa_set):
    for kitr,k in enumerate(k_set):
        mean = means[kitr,faitr,:]
        p25 = perc25[kitr,faitr,:]
        p75 = perc75[kitr,faitr,:]
        try:
            plt.errorbar(T_set, mean, yerr=cstack(mean-p25,p75-mean), color=colors[colcount], solid_capstyle='projecting', linewidth=1, capsize=5, alpha=.5, zorder=2)
            plt.fill_between(T_set, p25, p75, fc=colors[colcount], alpha=.2, zorder=1)
            plt.text(T_set[-1] + (T_set[-1] - T_set[0]) * 0.02, mean[-1], kfa_label(k,fa), ha='left', va='center', color=colors[colcount])
        except:
            debug()
        colcount += 1
plt.xticks(T_set)
xlims = plt.xlim()
plt.xlim((xlims[0], xlims[1]*1.3)) # hack to include text
# plt.yscale('log')
plt.xlabel(r'Window size', fontsize=fontsize)
plt.ylabel(r'Elapsed solver time [$s$]', fontsize=fontsize)
plt.savefig('figures/timing_compare_window.pdf', bbox_inches='tight')

# Plot compare against node and edge count
file_names = ['nodes','edges']
labels_xaxis = [r'Node Count', r'Edge Count']
data_xaxis = [node_counts, edge_counts]
for j in range(len(data_xaxis)):
    plt.figure()
    plt.rcParams['text.usetex'] = True
    colcount = 0
    for faitr,fa in enumerate(fa_set):
        for kitr,k in enumerate(k_set):
            xaxis = data_xaxis[j][kitr,faitr,:,:]
            mean = means_samp[kitr,faitr,:,:]
            p25 = perc25_samp[kitr,faitr,:,:]
            p75 = perc75_samp[kitr,faitr,:,:]
            xaxis_sort_idx = np.argsort(xaxis.reshape(-1))
            xaxis_sort = xaxis.reshape(-1)[xaxis_sort_idx]
            mean_sort = mean.reshape(-1)[xaxis_sort_idx]
            coefs = np.polyfit(xaxis_sort, mean_sort, 2)
            poly = np.poly1d(coefs)
            xaxis_fine = np.linspace(xaxis_sort[0], xaxis_sort[-1], 1000)
            plt.scatter(xaxis, mean, color=colors[colcount], alpha = 0.5)
            plt.plot(xaxis_fine, poly(xaxis_fine), color=colors[colcount], label=kfa_label(k,fa))
            colcount += 1
    plt.legend()
    # plt.yscale('log')
    plt.xlabel(labels_xaxis[j], fontsize=fontsize)
    plt.ylabel(r'Elapsed solver time [$s$]', fontsize=fontsize)
    plt.savefig('figures/timing_compare_'+file_names[j]+'.pdf', bbox_inches='tight')