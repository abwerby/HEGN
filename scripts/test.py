import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

import open3d as o3d

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

class Args:
    def __init__(self):
        self.device = 'cuda'
        self.vngcnn_in =  [2, 64, 64, 256]
        self.vngcnn_out = [32, 32, 128, 64]
        self.n_knn = [20, 20, 16, 16]
        self.topk = [4, 4, 2, 2]
        self.pooling = 'mean'
        self.num_blocks = 4
        
args = Args()
model = HEGN(args=args).to(device)

# Define loss function and optimizer
criterion = HEGN_Loss()


# Load the model and test it
model = HEGN(args=args).to(device)
model.load_state_dict(torch.load('checkpoints/hegn.pth'))
model.eval()


# test loop
with torch.no_grad():
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        x = batch['points'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        y = batch['points_ts'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        T = batch['transform'].to(device).to(torch.float32)
        R_gt = T[:,:,:3]
        t_gt = T[:,:,3]
        S_gt = R_gt.diagonal(dim1=1, dim2=2)
        # find centroids of both x and y
        x_centroid = x.mean(dim=2, keepdim=True)
        y_centroid = y.mean(dim=2, keepdim=True)    
        x_par = x - x_centroid
        y_par = y - y_centroid
        # stop the logging
        R, S = model(x_par, y_par)
        t = y_centroid - torch.matmul(R, x_centroid)
        x_aligned = torch.matmul(R, x_par) + t
        loss = criterion(x_aligned, y, R, S, t, R_gt, S_gt, t_gt)
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"[{i+1}/{len(dataloader)}] loss: {running_loss/10}")
            running_loss = 0.0
print('Finished Testing')