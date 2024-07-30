import numpy as np
import torch
import sys
from torch.utils.data import DataLoader, Dataset
import h5py
import open3d as o3d

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
        # pcd1 = self.source[index][:self.n_points]
        # pcd2 = self.target[index][:self.n_points]
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
        return self.source.shape[0]
    
def to_h5(dataset, filename):
    """
    Save the dataset to an h5 file
    """
    # create dataloader with batch size 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with h5py.File(filename, 'w') as f:
        source = f.create_dataset('source', (len(dataset), 1024, 6), dtype='f')
        target = f.create_dataset('target', (len(dataset), 1024, 6), dtype='f')
        transform = f.create_dataset('transform', (len(dataset), 4, 4), dtype='f')
        for i, batch in enumerate(dataloader):
            source[i] = batch['points'][0]
            target[i] = batch['points_ts'][0]
            transform[i] = batch['transform'][0]
        print('Dataset written to hdf5 file')

def RMSE(x, R, S, t, R_gt, S_gt, t_gt):
    """
        Compute the root mean squared error between predicted and ground truth transformation
        on the same point cloud
    """
    x_aligned = torch.matmul(R, S @ x) + t # Shape: [B, 3, N]
    x_gt_aligned = torch.matmul(R_gt, S_gt @ x) + t_gt.unsqueeze(-1) # Shape: [B, 3, N]
    rmse =  torch.norm(x_aligned - x_gt_aligned, dim=1).mean(dim=1) # Shape: [B]
    return rmse.mean()


if __name__ == '__main__':
    # test the dataset  
    batch_size = 32
    # deepgmrdataset = DeepGMRDataSet('/export/home/werbya/dll/deepgmr/data/test/modelnet_unseen.h5')
    deepgmrdataset = DeepGMRDataSet('/export/home/werbya/dll/HEGN/data/test_6dof.h5')
    print(len(deepgmrdataset))
    dataloader = DataLoader(deepgmrdataset, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(dataloader):
        x = batch[0][:,:,:3].transpose(2, 1).to(torch.float32)
        y = batch[1][:,:,:3].transpose(2, 1).to(torch.float32)
        transform_gt = batch[2].to(torch.float32)
        t_gt = transform_gt[:, :3, 3]
        R_gt = transform_gt[:, :3, :3]
        x_aligned_gt = torch.matmul(R_gt, x) + t_gt.unsqueeze(-1)
        # show the point cloud original and transformed
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(y[10].numpy().T)
        pcd2.points = o3d.utility.Vector3dVector(x_aligned_gt[10].numpy().T)
        pcd1.paint_uniform_color([1, 0, 0])
        pcd2.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd1, pcd2])
        break
    
