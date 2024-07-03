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

# Define hyperparameters
learning_rate = 1e-3
batch_size = 8
num_epochs = 10

# Create dataset and dataloader
transform = Compose([
    Resampler(1024),
    RandomJitter(scale=0.01, clip=0.05),
    RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=(0.5, 1.5)),
])
torch.cuda.memory._record_memory_history(True)

dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048',
                      subset='train', categories=['airplane'], transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# select only 10% of the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(range(0, int(0.15*len(dataset)))))
# Create your model and move it to the device
class Args:
    def __init__(self):
        self.device = 'cuda'
        self.n_knn = 20
        self.topk = [1024//16, 1024//16, 1024//16, 1024//16]
        self.pooling = 'mean'
        self.num_blocks = 4

args = Args()
model = HEGN(args=args).to(device)

logging.info(f"number of parameters in HEGEN: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
logging.info(f"dataloader length: {len(dataloader)}")

# logging.info(f"Cuda memory allocated at this batch: {torch.cuda.memory_allocated(device='cuda')/1e6} Mb")


# Define loss function and optimizer
criterion = HEGN_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

# Training loop
model.train()
for epoch in range(num_epochs):
    batches_loss = 0
    for batch_idx, batch in enumerate(dataloader):
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
        optimizer.zero_grad()
        # stop the logging
        R, S = model(x_par, y_par)
        t = y_centroid - torch.matmul(R, x_centroid)
        x_aligned = torch.matmul(R, x_par) + t
        loss = criterion(x_aligned, y, R, S, t, R_gt, S_gt, t_gt)
        batches_loss += loss.item()
        loss.backward()
        optimizer.step()
        logging.disable(logging.NOTSET)
    logging.info(f"epoch {epoch} loss: {batches_loss/len(dataloader)}")

# Save the model
torch.save(model.state_dict(), 'checkpoints/hegn.pth')

# Load the model and test it
model = HEGN(args=args).to(device)
model.load_state_dict(torch.load('checkpoints/hegn.pth'))
model.eval()

# select random batch from the dataset
batch = next(iter(dataloader))
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
R, S = model(x_par, y_par)
t = y_centroid - torch.matmul(R, x_centroid)
print(f"R_gt: {R_gt[0]}")
print(f"S_gt: {S_gt[0]}")
print(f"t_gt: {t_gt[0]}")
print(f"--------------------------------")
print(f"R: {R[0]}")
print(f"S: {S[0]}")
print(f"t: {t[0]}")
# apply scaling to x
x_aligned[0] = torch.matmul(torch.diag(S[0]), x_aligned[0])
x_aligned[0] = torch.matmul(R[0], x_par[0]) + t[0]
print(f"x_aligned: {x_aligned.size()}")
# visualize the point clouds
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(x[0].cpu().detach().numpy().transpose(1, 0))
pcd2.points = o3d.utility.Vector3dVector(y[0].cpu().detach().numpy().transpose(1, 0))
pcd1.paint_uniform_color([1, 0, 0])
pcd2.paint_uniform_color([0, 1, 0])
o3d.io.write_point_cloud("before.ply", pcd1 + pcd2)
pcd1.points = o3d.utility.Vector3dVector(x_aligned[0].cpu().detach().numpy().transpose(1, 0))
o3d.io.write_point_cloud("after.ply", pcd1 + pcd2)
