import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

import open3d as o3d

import time
import logging
import os
import sys
# add current directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hegn.utils.vn_dgcnn_util import get_graph_feature
from hegn.models.hegn import HEGN, HEGN_Loss
from hegn.dataloader.dataloader import ModelNetHdf
from hegn.dataloader.transforms import (
                        Resampler,
                        FixedResampler,
                        RandomJitter,
                        RandomCrop,
                        RandomTransformSE3
                    )

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

# Create dataset and dataloader
transform = Compose([
    Resampler(1024),
    RandomJitter(scale=0.01, clip=0.05),
    RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=(0.5, 1.5)),
])
torch.cuda.memory._record_memory_history(True)

dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048',
                      subset='test', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
model.eval()


# test loop
with torch.no_grad():
    running_loss = 0.0
    running_loss_reg = 0.0
    running_loss_chm = 0.0
    time_per_batch = []
    for i, batch in enumerate(dataloader):
        x = batch['points'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        y = batch['points_ts'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        t_gt = batch['T'].to(device).to(torch.float32)
        R_gt = batch['R'].to(device).to(torch.float32)
        S_gt = batch['S'].to(device).to(torch.float32)
        curr_batch_size = x.size(0)
        # find centroids of both x and y
        x_centroid = x.mean(dim=2, keepdim=True)
        y_centroid = y.mean(dim=2, keepdim=True)    
        x_par = x - x_centroid
        y_par = y - y_centroid
        # stop the logging
        start_time = time.time()
        R, S = model(x_par, y_par)
        t = y_centroid - torch.matmul(R, x_centroid)
        S = torch.diag_embed(S)
        x_aligned = torch.matmul(R, S @ x_par) + t
        end_time = time.time()
        if curr_batch_size == batch_size:
            time_per_batch.append(end_time - start_time)
        loss, loss_reg, loss_chm = criterion(x_aligned, y, R, S, t, R_gt, S_gt, t_gt)
        running_loss += loss.item()
        running_loss_reg += loss_reg.item()
        running_loss_chm += loss_chm.item()


print(f'Loss: {running_loss/len(dataloader)}')
print(f'Reg Loss MSE: {running_loss_reg/len(dataloader)}')
print(f'Reg loss RMSE: {running_loss_reg/len(dataloader)**0.5}')
print(f'Chm Loss: {running_loss_chm/len(dataloader)}')
print(f'Average time per batch: {sum(time_per_batch)/len(time_per_batch)}, FPS: {1/(sum(time_per_batch)/len(time_per_batch))}')
print(f'Max time per batch: {max(time_per_batch)}')
print(f'Min time per batch: {min(time_per_batch)}')
print(f'Average time per sample: {sum(time_per_batch)/len(dataloader)/batch_size}, FPS: {1/(sum(time_per_batch)/len(dataloader)/batch_size)}')
