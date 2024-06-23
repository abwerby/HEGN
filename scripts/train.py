import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

import os
import sys
# add current directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hegn.utils.vn_dgcnn_util import get_graph_feature
from hegn.models.hegn import HEGN
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
learning_rate = 10e-3
batch_size = 8
num_epochs = 10

# Create dataset and dataloader
transform = Compose([
    Resampler(1024),
])
torch.cuda.memory._record_memory_history(True)

dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048',
                      subset='train', categories=['airplane'], transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create your model and move it to the device
class Args:
    def __init__(self):
        self.n_knn = 20
        self.pooling = 'mean'

args = Args()
model = HEGN(args=args).to(device)
print(f"number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        data = batch['points'][:,:,:3].transpose(2, 1)
        if data.size(0) < batch_size:
            continue
        data = data.to(device)

        outputs = model(data)
    break
