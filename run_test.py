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
from torch_geometric.data import Data

from lib.tracking import Tracker
from lib.utils import getIoU, computeBoxFeatures, interpolateTrack, interpolateTracks

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(6,6), nn.ReLU(), nn.Linear(6,1))
    def forward(self, data):
        x = self.fc(data.edge_attr)
        x = nn.Sigmoid()(x)
        return x
    
net = Net()
net.load_state_dict(torch.load('ckpt/qp/epoch_8.pth'))
tracker = Tracker(net)

def get_trans_probs(tracker, curr_dets, curr_app_feats, app_thresh, max_frame_gap = 5):
    """
    Inputs: tracker: an instance of the Tracker.
            curr_dets: frame, x1, y1, x2, y2, det_confidence, node_ind.
            curr_app_feats: normalized appearance features for curr_dets.
            max_frame_gap: frame gap used to connect detections.
    Return: transition probabilities for LP that handles false negatives(missing detections).
    """
    edge_ind = 0
    edge_feats, lifted_probs = [], []
    edge_type = [] #1:base edge 2:lifted edge-1:pruned lifted edge
    
    cos_sim_mat = np.dot(curr_app_feats, curr_app_feats.T)
    linkIndexGraph = np.zeros((curr_dets.shape[0], curr_dets.shape[0]), dtype=np.int32)
    for i in range(curr_dets.shape[0]):
        for j in range(curr_dets.shape[0]):
            frame_gap = curr_dets[j][0] - curr_dets[i][0]
            cos_sim = cos_sim_mat[i, j]

            if frame_gap == 1: #base edge
                edge_type.append(1)
                feats = computeBoxFeatures(curr_dets[i, 1:5], curr_dets[j, 1:5])
                iou = getIoU(curr_dets[i, 1:5], curr_dets[j, 1:5])
                feats.extend((iou, cos_sim))
                edge_feats.append(feats)
                edge_ind += 1
                linkIndexGraph[i, j] = edge_ind

            elif frame_gap > 1 and frame_gap <= max_frame_gap: #lifted edge
                if cos_sim > app_thresh:
                    edge_type.append(2)
                    time_weight = 0.9 ** frame_gap
                    lifted_probs.append(cos_sim * time_weight)
                else:
                    edge_type.append(-1)

                edge_ind += 1
                linkIndexGraph[i, j] = edge_ind
                
    edge_type = np.array(edge_type)
    edge_feats = torch.Tensor(edge_feats)
    with torch.no_grad():
        logits = tracker.net.fc(edge_feats)
        prob = nn.Sigmoid()(logits)
        prob = torch.clamp(prob, min=1e-7, max=1-1e-7).flatten().numpy()
        
    probs = np.zeros(edge_ind)
    base_inds = np.where(edge_type == 1)[0]
    lifted_inds = np.where(edge_type == 2)[0]
    pruned_lifted_inds = np.where(edge_type == -1)[0]
    probs[base_inds] = prob            #base probs
    probs[lifted_inds] = lifted_probs  #lifted probs
    return linkIndexGraph, probs

app_thresh = 0.75 #0.7, 0.8
nms_thresh, eps = 0.3, 1e-7

#for sequence in ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']:
for seq in ['MOT17-01']:
    
    #Static camera, moving camera
    if seq in ['MOT17-03']:
        batch_size, dist_thresh, prune_len = 50, 50, 2 #tracklets less than 2 are pruned
    else:
        batch_size, dist_thresh, prune_len = 100, 100, 3
        
    if seq == 'MOT17-06':
        img_Height, img_Width = 480, 640
    else:
        img_Height, img_Width = 1080, 1920
        
    #for detector in ['DPM','FRCNN','SDP']:
    for detector in ['DPM']:
        print('Sequence {}, {} detection, app thresh {}, dist thresh {}, retain length {}'.format(
            seq, detector, app_thresh, dist_thresh, prune_len))
        
        det_file = './data/{}/det_{}.txt'.format(seq, detector)
        app_file = './data/{}/app_det_{}.npy'.format(seq, detector)
        dets = np.loadtxt(det_file, delimiter=',')
        app_feats = np.load(app_file)
        assert dets.shape[0] == app_feats.shape[0], 'Shape mismatch'

        batch_overlap = 5                  #Number of frames to overlap between 2 batches
        num_frames = int(dets[:, 0].max()) #Number of frames for this video
        tracks_list, assignments_list, features_list, nms_list = [],[],[],[]
        
        for start_frame in range(1, num_frames+1, batch_size-batch_overlap):
            end_frame = start_frame + batch_size - 1
            if end_frame >= num_frames:
                end_frame = num_frames
                
            print('Tracking from frame %d to %d'%(start_frame, end_frame))
            curr_ind = np.logical_and(dets[:, 0] >= start_frame, dets[:, 0] <= end_frame)
            curr_dets = np.concatenate([dets[curr_ind, 0][:, None], dets[curr_ind, 2:7],
                                        np.arange(dets[curr_ind].shape[0])[:, None]], axis=1)

            curr_dets[:, 3:5] = curr_dets[:, 3:5] + curr_dets[:, 1:3] # convert to frame,x1,y1,x2,y2,conf,node_ind
            curr_app_feats = app_feats[curr_ind]
            curr_app_feats = curr_app_feats / np.linalg.norm(curr_app_feats, axis=1, keepdims=True)
            for iteration in range(2):
                if iteration == 0:
                    print('%d-th iteration'%iteration)
                    linkIndexGraph, probs = get_trans_probs(tracker, curr_dets, curr_app_feats, 
                                                            app_thresh, max_frame_gap = 5)
                    trans_cost = - np.log(probs + eps) #np.log((1 - probs + eps)/(probs + eps))
                    det_cost = - curr_dets[:, -2]
                    entry_cost = 0.5 * np.ones(det_cost.shape[0])
                    exit_cost = entry_cost
                    cost = np.concatenate((det_cost, entry_cost, exit_cost, trans_cost))

                    A_eq, b_eq, A_ub, b_ub = tracker.build_constraint(linkIndexGraph)
                    sol = tracker.linprog(c=cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)
                    
                    tracklets = tracker.recoverTracklets(curr_dets, sol, linkIndexGraph, prune_len=prune_len)    
                    tracklets_ = np.delete(tracklets, -1, axis=1)
                    interpolated_tracklets = interpolateTracks(tracklets_)
                    
                else:
                    print('%d-th iteration'%iteration)
                    assignment_list, feature_list = tracker.clusterSkipTracklets(tracklets, curr_app_feats, 
                                                                                 dist_thresh, app_thresh)
                    tracks = tracker.recoverClusteredTracklets(tracklets, assignment_list)
                    tracks = interpolateTracks(tracks)

                    assignments_list.append(assignment_list)
                    feature_array = np.stack(feature_list)
                    feature_array = feature_array / np.linalg.norm(feature_array, axis=1, keepdims=True)
                    
            tracks_list.append(tracks)
            features_list.append(feature_array)
            
        final_tracks = tracker.stitchTracklets(tracks_list, features_list)
        save_file = 'BYTE_Results/MOT17-{}-{}.txt'.format(seq.split('-')[1], detector)
        print('Finished tracking, saving to {}'.format(save_file))
        np.savetxt(save_file, final_tracks, fmt='%d',delimiter=',')
        
#seq = 'MOT17-03'
detector = 'DPM'

# for seq in ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']:
for seq in ['MOT17-01']:
    save_dir = 'BYTE_Results/{}-{}'.format(seq, detector)
    print('save dir {}'.format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    
    #tracks: frame, ID, x, y, w, h, -1, -1, -1, -1
    #dets:   frame, -1, x, y, w, h, conf, -1, -1, -1
    tracks = np.loadtxt('BYTE_Results/MOT17-01-DPM.txt', delimiter=',')
    tracks = tracks.astype(np.int32)
    
    colors = np.random.rand(1000,3)
    resize_scale = 0.5
    for frame in range(tracks[:, 0].min(), tracks[:, 0].max()+1):
        if frame % 100 == 0:
            print('Processing frame {}'.format(frame))
        
        img_file = os.path.join('/home/lishuai/Experiment/MOT/MOT17/test/{}-{}/img1/{:06d}.jpg'.format(seq,detector,frame))
        img = cv2.imread(img_file)
        img = cv2.resize(img, (int(resize_scale*img.shape[1]), int(resize_scale*img.shape[0])))
        cv2.putText(img, '{:04}'.format(frame), (0,50) ,cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), thickness=2)
        bboxes = tracks[tracks[:, 0] == frame, 1:6]
        
        if bboxes.shape[0] != 0:
            #detections = dets[dets[:, 0] == frame, 2:7]
            for i in range(bboxes.shape[0]):
                ID = int(bboxes[i][0])
                x, y = int(resize_scale*(bboxes[i][1])), int(resize_scale*(bboxes[i][2]))
                w, h = int(resize_scale*(bboxes[i][3])), int(resize_scale*(bboxes[i][4]))
                cv2.rectangle(img, (x,y),(x+w,y+h), 255*colors[ID], thickness=2)
                cv2.putText(img, str(ID), (x,y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255*colors[ID], thickness=2)

    #         for i in range(detections.shape[0]):
    #             x, y = int(resize_scale*(detections[i][0])), int(resize_scale*(detections[i][1]))
    #             w, h = int(resize_scale*(detections[i][2])), int(resize_scale*(detections[i][3]))
    #             score = detections[i][4] 
    #             drawrect(img,(x,y),(x+w,y+h),(0,0,255),2,'dotted')
        cv2.imwrite(save_dir+'/'+'{:06d}.jpg'.format(frame), img)