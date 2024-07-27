import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

import open3d as o3d

import time
import logging
import os
import sys
import h5py
# add current directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hegn.utils.vn_dgcnn_util import get_graph_feature
from hegn.models.hegn import HEGN, HEGN_Loss
from utils.eval_utils import DeepGMRDataSet, RMSE


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
deepgmrdataset = DeepGMRDataSet('/export/home/werbya/dll/deepgmr/data/test/modelnet_unseen.h5')
print(len(deepgmrdataset))
dataloader = DataLoader(deepgmrdataset, batch_size=batch_size, shuffle=True)

# make sure that args are the same as in the training script
class Args:
    def __init__(self):
        self.device = 'cuda'
        self.vngcnn_in =  [2, 64, 64, 128]
        self.vngcnn_out = [32, 32, 64, 32]
        self.n_knn = [20, 20, 16, 16]
        self.topk = [4, 4, 2, 2]
        self.num_blocks = len(self.vngcnn_in)
        
args = Args()
model = HEGN(args=args).to(device)

# Define loss function and optimizer
criterion = HEGN_Loss()


# Load the model from the checkpoint
model = HEGN(args=args).to(device)
model.load_state_dict(torch.load('checkpoints/hegn_100e_nobatch.pth'))
# model.load_state_dict(torch.load('checkpoints/hegn_100e_512.pth'))
model.eval()


# test loop
with torch.no_grad():
    running_loss = 0.0
    running_loss_reg = 0.0
    running_loss_chm = 0.0
    RMSE_loss = 0.0
    time_per_batch = []
    for i, batch in enumerate(dataloader):
        x = batch[0].transpose(2, 1).to(device).to(torch.float32)
        y = batch[1].transpose(2, 1).to(device).to(torch.float32)
        transform_gt = batch[2].to(device).to(torch.float32)
        t_gt = transform_gt[:, :3, 3]
        R_gt = transform_gt[:, :3, :3]
        S_gt = torch.eye(3).to(device).to(torch.float32).unsqueeze(0).repeat(x.size(0), 1, 1)
        curr_batch_size = x.size(0)
        # find centroids of both x and y
        x_centroid = x.mean(dim=2, keepdim=True)
        y_centroid = y.mean(dim=2, keepdim=True)    
        x_par = x - x_centroid
        y_par = y - y_centroid
        # stop the logging
        start_time = time.time()
        R, S = model(x_par, y_par)
        # R = R_gt
        S = S_gt
        t = y_centroid - torch.matmul(R, x_centroid)
        x_aligned = torch.matmul(R, S @ x) + t
        x_aligned_gt = torch.matmul(R_gt, S_gt @ x) + t_gt.unsqueeze(-1)
        end_time = time.time()
        # show the point cloud original and transformed
        pc = o3d.geometry.PointCloud()
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(y[0].cpu().numpy().T)
        pc.points = o3d.utility.Vector3dVector(x_aligned[0].cpu().numpy().T)
        pc1.paint_uniform_color([0, 0, 1])
        pc.paint_uniform_color([1, 0, 0])
        before_folder = 'output_DeepGMR_data/before'
        after_folder = 'output_DeepGMR_data/after'
        if not os.path.exists(before_folder):
            os.makedirs(before_folder)
        if not os.path.exists(after_folder):
            os.makedirs(after_folder)
        o3d.io.write_point_cloud(f'{after_folder}/pc_{i}.ply', pc1+pc)
        pc.points = o3d.utility.Vector3dVector(x[0].cpu().numpy().T)
        o3d.io.write_point_cloud(f'{before_folder}/pc_{i}.ply', pc1+pc)
        
        if curr_batch_size == batch_size:
            time_per_batch.append(end_time - start_time)
        loss, loss_reg, loss_chm = criterion(x_aligned, x_aligned_gt, R, S, t, R_gt, S_gt, t_gt)
        # calculate RMSE
        RMSE_loss += RMSE(x_par, R, S, t, R_gt, S_gt, t_gt)
        running_loss += loss.item()
        running_loss_reg += loss_reg.item()
        running_loss_chm += loss_chm.item()


print(f'Loss: {running_loss/len(dataloader)}')
print(f'Reg Loss: {running_loss_reg/len(dataloader)}')
print(f'RMSE: {RMSE_loss/len(dataloader)}')
print(f'Chm Loss: {running_loss_chm/len(dataloader)}')
print(f'Average time per batch: {sum(time_per_batch)/len(time_per_batch)}, FPS: {1/(sum(time_per_batch)/len(time_per_batch))}')
print(f'Max time per batch: {max(time_per_batch)}')
print(f'Min time per batch: {min(time_per_batch)}')
print(f'Average time per sample: {sum(time_per_batch)/len(dataloader)/batch_size}, FPS: {1/(sum(time_per_batch)/len(dataloader)/batch_size)}')
