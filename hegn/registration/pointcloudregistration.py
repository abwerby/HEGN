import torch
import torch.nn as nn
from torchvision.transforms import Compose
import open3d as o3d
import numpy as np
from typing import Tuple, Optional

from hegn.models.hegn import HEGN
from hegn.dataloader.transforms import Resampler, RandomJitter

class PointCloudRegistration:
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

        self.model = HEGN(args=Args()).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # Initialize transforms
        self.transform = Compose([
            Resampler(1024, resample_both=True),
            RandomJitter(scale=0.01, clip=0.05)
        ])

    def preprocess(self, points: np.ndarray) -> torch.Tensor:
        """Preprocess the input point cloud."""
        if not isinstance(points, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if points.shape[1] != 3:
            raise ValueError("Input must have shape (N, 3)")
        
        # Convert to torch tensor
        points = torch.from_numpy(points).float().to(self.device)
        
        # Apply transforms
        # points = self.transform({'points': points})['points']
        
        # Transpose to (3, N) as expected by the model
        return points.transpose(1, 0).unsqueeze(0)

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
            aligned_source = x_aligned.squeeze().transpose(1, 0).cpu().numpy()
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
        def create_point_cloud(points, color):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color(color)
            return pcd

        source_pcd = create_point_cloud(source, [1, 0, 0])  # Red
        target_pcd = create_point_cloud(target, [0, 1, 0])  # Green

        if aligned_source is not None:
            aligned_pcd = create_point_cloud(aligned_source, [0, 0, 1])  # Blue
            o3d.visualization.draw_geometries([source_pcd, target_pcd, aligned_pcd])
        else:
            o3d.visualization.draw_geometries([source_pcd, target_pcd])

# Example usage
if __name__ == "__main__":
    # Initialize the PointCloudRegistration class
    pcr = PointCloudRegistration(checkpoint_path="/export/home/werbya/thesis/HEGN/checkpoints/hegn.pth")

    # Load your point clouds (replace with your actual data)
    source = np.random.rand(1000, 3)
    target = np.random.rand(1000, 3)

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