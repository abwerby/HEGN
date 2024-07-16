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

import wandb
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
batch_size = 32
num_epochs = 100
optimizer_name = 'adam'

# Create dataset and dataloader
transform = Compose([
    Resampler(1024),
    RandomJitter(scale=0.01, clip=0.05),
    RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=(0.5, 1.5)),
])

dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048',
                      subset='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# select only 10% of the dataset
# dataloader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(range(0, int(0.1*len(dataset)))))
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

logging.info(f"number of parameters in HEGN: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
logging.info(f"dataloader length: {len(dataloader)}")
logging.info(f"sample in one batch: {next(iter(dataloader))['points'].size()}")


# Define loss function and optimizer
criterion = HEGN_Loss()

if optimizer_name == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.99))
elif optimizer_name == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)


# Initialize wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="HEGN",
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "optimizer": optimizer_name,
    "dataset": 'ModelNet40',
    "model args": args.__dict__
    }
)


# Training loop
model.train()
for epoch in range(num_epochs):
    batches_loss = 0
    batches_loss_reg = 0
    batches_loss_chm = 0
    for batch_idx, batch in enumerate(dataloader):
        x = batch['points'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        y = batch['points_ts'][:,:,:3].transpose(2, 1).to(device).to(torch.float32)
        t_gt = batch['T'].to(device).to(torch.float32)
        R_gt = batch['R'].to(device).to(torch.float32)
        S_gt = batch['S'].to(device).to(torch.float32)
        # skip if the batch size is less than the required batch size
        # if x.size(0) < batch_size:
        #     continue
        # find centroids of both x and y
        x_centroid = x.mean(dim=2, keepdim=True)
        y_centroid = y.mean(dim=2, keepdim=True)
        x_par = x - x_centroid
        y_par = y - y_centroid
        optimizer.zero_grad()
        R, S = model(x_par, y_par)
        t = y_centroid - torch.matmul(R, x_centroid)
        S = torch.diag_embed(S)
        # uncomment the following line to use ground truth S, R, t
        # S, R, t = S_gt, R_gt, t_gt.unsqueeze(-1)
        x_aligned = torch.matmul(R, S @ x_par) + t
        loss, loss_reg, loss_chm = criterion(x_aligned, y, R, S, t, R_gt, S_gt, t_gt)
        
        batches_loss += loss.item()
        batches_loss_reg += loss_reg.item()
        batches_loss_chm += loss_chm.item()
        loss.backward()
        optimizer.step()
        logging.disable(logging.NOTSET)
        # logging.info(f"epoch {epoch} batch {batch_idx} loss: {loss.item()}")
        # wandb.log({"batch idx": batch_idx, "batch loss": loss.item()})
    wandb.log({
            "epoch": epoch,
            "epoch loss": batches_loss/len(dataloader),
            "reg loss": batches_loss_reg/len(dataloader),
            "chm loss": batches_loss_chm/len(dataloader)
            })
    logging.info(f"epoch {epoch} loss: {batches_loss/len(dataloader)}, reg loss: {batches_loss_reg/len(dataloader)}, chm loss: {batches_loss_chm/len(dataloader)}")

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
t_gt = batch['T'].to(device).to(torch.float32)
R_gt = batch['R'].to(device).to(torch.float32)
S_gt = batch['S'].to(device).to(torch.float32)
# find centroids of both x and y
x_centroid = x.mean(dim=2, keepdim=True)
y_centroid = y.mean(dim=2, keepdim=True)
x_par = x - x_centroid
y_par = y - y_centroid
R, S = model(x_par, y_par)
S = torch.diag_embed(S)
t = y_centroid - torch.matmul(R, x_centroid)
print(f"R: {R.size()}")
print(f"S: {S.size()}")
print(f"t: {t.size()}")
print(f"R_gt: {R_gt.size()}")
print(f"S_gt: {S_gt.size()}")
print(f"t_gt: {t_gt.size()}")
print(f"S: {S[0]}")
# S, R, t = S_gt, R_gt, t_gt.unsqueeze(-1)
x_aligned = torch.matmul(R, S @ x_par) + t
print(f"x_aligned: {x_aligned.size()}")
print(f"loss: {criterion(x_aligned, y, R, S, t, R_gt, S_gt, t_gt)}")
# print(f"loss: {criterion(R, t, S, R_gt, t_gt, S_gt)}")
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


wandb.finish()