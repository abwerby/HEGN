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
import h5py
# add current directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.eval_utils import RMSE, to_h5
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
    Resampler(1024, resample_both=True),
    RandomJitter(scale=0.01, clip=0.05),
    # RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=(0.5, 1.5)),
    RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=None),
])
torch.cuda.memory._record_memory_history(True)

dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048',
                      subset='test', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# use only 50% of the dataset
# dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(range(0, len(dataset), 2)))

# write the dataset to an h5 file to test DeepGMR 
to_h5(dataset, 'data/test_9dof.h5')
print('Dataset written to test.h5')
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
    RMSE_loss = 0.0
    CHM_loss = 0.0
    time_per_batch = []
    for i, batch in enumerate(dataloader):
        x = batch['points'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        y = batch['points_ts'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        t_gt = batch['T'].to(device).to(torch.float32)
        R_gt = batch['R'].to(device).to(torch.float32)
        S_gt = batch['S'].to(device).to(torch.float32)
        transform_gt = batch['transform'].to(device).to(torch.float32)
        # S_gt = torch.diag_embed(batch['transform'][:, :3, :3].det()).to(device).to(torch.float32)
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
        # S = torch.diag_embed(S)
        S = S_gt
        x_aligned = torch.matmul(R, S @ x) + t
        x_aligned_gt = torch.matmul(R_gt, S_gt @ x) + t_gt.unsqueeze(-1)
        end_time = time.time()
        # save the point cloud original and transformed
        pc = o3d.geometry.PointCloud()
        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector(y[0].cpu().numpy().T)
        pc.points = o3d.utility.Vector3dVector(x_aligned[0].cpu().numpy().T)
        pc1.paint_uniform_color([0, 0, 1])
        pc.paint_uniform_color([1, 0, 0])
        before_folder = 'output_HEGN_data/before'
        after_folder = 'output_HEGN_data/after'
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
