import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py


class DeepGMRDataSet(Dataset):
    def __init__(self, path):
        super(DeepGMRDataSet, self).__init__()
        with h5py.File(path, 'r') as f:
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]
        print(self.source.shape, self.target.shape, self.transform.shape)
        self.n_points = 1024

    def __getitem__(self, index):
        pcd1 = self._resample(self.source[index], self.n_points)
        pcd2 = self._resample(self.target[index], self.n_points)
        transform = self.transform[index]
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def _resample(self, pcd, n_points):
        """
            randomly sample n_points from the point cloud
        """
        if n_points < pcd.shape[0]:
            idx = np.random.choice(pcd.shape[0], n_points, replace=False)
            pcd = pcd[idx]
        return pcd

    def __len__(self):
        return self.transform.shape[0]
    
def to_h5(dataset, filename):
    """
    Save the dataset to an h5 file
    """
    with h5py.File(filename, 'w') as f:
        source_data = [entry['points'][:, :3] for entry in dataset]
        target_data = [entry['points_ts'][:, :3] for entry in dataset]
        transform_data = [entry['transform'] for entry in dataset]

        # Write datasets to the file
        f.create_dataset('source', data=source_data)
        f.create_dataset('target', data=target_data)
        f.create_dataset('transform', data=transform_data)

def from_h5(filename):
    """
    Read the dataset from an h5 file
    """
    with h5py.File(filename, 'r') as f:
        source_data = f['source'][...]
        target_data = f['target'][...]
        transform_data = f['transform'][...]
    return source_data, target_data, transform_data      

def RMSE(x, R, S, t, R_gt, S_gt, t_gt):
    """
        Compute the root mean squared error between predicted and ground truth transformation
        on the same point cloud
    """
    x_aligned = torch.matmul(R, S @ x) + t
    x_gt_aligned = torch.matmul(R_gt, S_gt @ x) + t_gt.unsqueeze(-1)
    rmse =  torch.norm(x_aligned - x_gt_aligned, dim=1).mean(dim=1)
    return rmse.mean()