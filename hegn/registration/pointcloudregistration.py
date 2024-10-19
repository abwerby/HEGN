import torch
import torch.nn as nn
from torchvision.transforms import Compose
import open3d as o3d
import numpy as np
from typing import Tuple, Optional

from hegn.models.hegn import HEGN
from hegn.dataloader.transforms import Resampler, RandomJitter, RandomTransformSE3
from hegn.dataloader.dataloader import ScanObjetDataLoader, ModelNetHdf


class HEGNRegistration:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        class Args:
            def __init__(self):
                self.device = device
                self.vngcnn_in = [2, 64, 64, 64]
                self.vngcnn_out = [32, 32, 32, 32]
                self.n_knn = [20, 20, 16, 16]
                self.topk = [4, 4, 2, 2]
                self.num_blocks = len(self.vngcnn_in)
        args = Args()
        self.model = HEGN(args=args).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
    
    def _resample(self, points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


    def preprocess(self, points: np.ndarray, num_points: int = 1024) -> torch.Tensor:
        """Preprocess the input point cloud."""
        if not isinstance(points, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if points.shape[1] != 3:
            raise ValueError("Input must have shape (N, 3)")
        # sample uniformly to num_points
        points = self._resample(points, num_points)
        
        # Convert to torch tensor
        points = torch.from_numpy(points).float().to(self.device)
        
        # Transpose to (3, N) as expected by the model
        points = points.transpose(1, 0).unsqueeze(0)
        return points

    def register(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Register source point cloud to target point cloud.
        
        Args:
            source (np.ndarray): Source point cloud of shape (N, 3)
            target (np.ndarray): Target point cloud of shape (M, 3)
        
        Returns:
            Tuple containing:
            - aligned_source (np.ndarray): Aligned source point cloud
            - R (np.ndarray): Rotation matrix
            - S (np.ndarray): Scale matrix
            - t (np.ndarray): Translation vector
        """
        with torch.no_grad():
            x = self.preprocess(source)
            y = self.preprocess(target)

            x_centroid = x.mean(dim=2, keepdim=True)
            y_centroid = y.mean(dim=2, keepdim=True)
            x_par = x - x_centroid
            y_par = y - y_centroid

            R, S = self.model(x_par, y_par)
            t = y_centroid - torch.matmul(R, x_centroid)
            S = torch.diag_embed(S)

            x_aligned = torch.matmul(R, S @ x) + t

            # Convert to numpy arrays
            aligned_source = x_aligned.squeeze().cpu().numpy().T
            R = R.squeeze().cpu().numpy()
            S = S.squeeze().cpu().numpy()
            t = t.squeeze().cpu().numpy()

        return aligned_source, R, S, t

    def visualize(self, source: np.ndarray, target: np.ndarray, aligned_source: Optional[np.ndarray] = None):
        """
        Visualize the point clouds using Open3D.
        
        Args:
            source (np.ndarray): Source point cloud
            target (np.ndarray): Target point cloud
            aligned_source (np.ndarray, optional): Aligned source point cloud
        """
        def create_point_cloud(points, color, voxel_size=0.05):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(color)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            return pcd

        source_pcd = create_point_cloud(source, [1, 0, 0])  # Red
        target_pcd = create_point_cloud(target, [0, 1, 0])  # Green

        if aligned_source is not None:
            aligned_pcd = create_point_cloud(aligned_source, [0, 0, 1])  # Blue
            # o3d.visualization.draw_geometries([source_pcd, target_pcd, aligned_pcd])
            o3d.io.write_point_cloud("aligned.ply", target_pcd + aligned_pcd)
        else:
            # o3d.visualization.draw_geometries([source_pcd, target_pcd])
            o3d.io.write_point_cloud("aligned.ply", source_pcd + target_pcd)

# Example usage
if __name__ == "__main__":
    # Initialize the PointCloudRegistration class
    pcr = HEGNRegistration(checkpoint_path="/export/home/werbya/thesis/HEGN/checkpoints/hegn.pth")

    # Load your point clouds (replace with your actual data)
    # source = o3d.io.read_point_cloud("/export/home/werbya/thesis/before.ply")
    # target = o3d.io.read_point_cloud("/export/home/werbya/thesis/after.ply")
    
    # downsample the point clouds
    # source = source.voxel_down_sample(voxel_size=0.05)
    # target = target.voxel_down_sample(voxel_size=0.05)
    
    # source = np.asarray(source.points)
    # target = np.asarray(target.points)
    
    # load the point clouds from the data loader
    loader = ScanObjetDataLoader()
    h5_file = '/export/home/werbya/thesis/HEGN/data/scanobjectnn/main_split/test_objectdataset.h5'
    h5_data, h5_labels = loader.load_h5(h5_file)
    num_points = 1024
    h5_batch_pcs, h5_batch_labels = loader.get_current_data_h5(h5_data, h5_labels, num_points)
    source = h5_batch_pcs[10]
    target = h5_batch_pcs[10]
    
        
    # apply random translation to the source point cloud
    # t_rand = np.random.uniform(-0.5, 0.5, size=(3,))
    # source += t_rand
    
    transform = Compose([
        RandomTransformSE3(rot_mag=180, trans_mag=0.5, scale_range=(0.5, 1.5)),
        Resampler(1024),
        RandomJitter(scale=0.01, clip=0.05),
    ])

    pc = dict(points=source, points_ts=target)
    sample = transform(pc)
    source = sample['points']
    target = sample['points_ts']

    # save both point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    o3d.io.write_point_cloud("source.ply", source_pcd)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    o3d.io.write_point_cloud("target.ply", target_pcd)

    

    # dataset = ModelNetHdf(dataset_path='data/modelnet40_ply_hdf5_2048', subset='train', categories=['airplane'],
    #                       transform=transform)
    # sample = dataset[0]
    # pcd = o3d.geometry.PointCloud()
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(sample['points'][:, :3])
    # pcd.points = o3d.utility.Vector3dVector(sample['points_ts'][:, :3])

    # source = np.asarray(pcd1.points)
    # target = np.asarray(pcd.points)
    
    print("Source shape:", source.shape)
    print("Target shape:", target.shape)

    # Perform registration
    aligned_source, R, S, t = pcr.register(source, target)

    # Visualize the results
    pcr.visualize(source, target, aligned_source)

    print("Rotation matrix:")
    print(R)
    print("\nScale matrix:")
    print(S)
    print("\nTranslation vector:")
    print(t)