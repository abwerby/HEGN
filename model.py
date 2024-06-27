import torch
import torch.nn as nn
from layers.Equivariant_Feature_Extraction import EquivariantFeatureExtraction
from layers.Invariant_Node_Pooling import InvariantNodePooling
from layers.Hierarchical_9DoF_Registration import Hierarchical_9DoF_Registration
import gc
import torch.autograd.profiler as profiler


class CenterPointCloud(nn.Module):
    def __init__(self):
        super(CenterPointCloud, self).__init__()

    def forward(self, point_cloud):
        """
        Center the point cloud by subtracting its centroid.

        Args:
            point_cloud : Input point cloud of shape (B,3,N).

        Returns:
            centered_point_cloud : Centered point cloud of shape (B, 3, N).
            centroid : Centroid of the point cloud of shape ().
        """
        centroid = torch.mean(point_cloud, dim=2)
        centered_point_cloud = point_cloud - centroid[..., None]
        return centered_point_cloud, centroid


class HEGN(nn.Module):
    def __init__(self, N=1024):
        super(HEGN, self).__init__()
        self.center_point_cloud = CenterPointCloud()

        self.feature_extraction1 = EquivariantFeatureExtraction(k=20)
        self.feature_extraction2 = EquivariantFeatureExtraction(k=20)
        self.feature_extraction3 = EquivariantFeatureExtraction(k=16)
        self.feature_extraction4 = EquivariantFeatureExtraction(k=16)

        self.invariant_node_pooling1 = InvariantNodePooling(128, K=N // 4)
        self.invariant_node_pooling2 = InvariantNodePooling(128, K=N // 4)
        self.invariant_node_pooling3 = InvariantNodePooling(128, K=N // 2)
        self.invariant_node_pooling4 = InvariantNodePooling(128, K=N // 2)

        self.hierarchical_9DoF_registration = Hierarchical_9DoF_Registration(global_descriptor_size=8)

    def forward(self, x, y):
        x, mu_x = self.center_point_cloud(x)
        y, mu_y = self.center_point_cloud(y)

        # Add more layers and operations as needed
        f_X_all = []
        f_Y_all = []

        x1, y1 = self.feature_extraction1(x, y)
        x1, y1 = self.invariant_node_pooling1(x1, y1)
        f_X_all.append(x1)
        f_Y_all.append(y1)

        del x1, y1

        gc.collect()
        torch.cuda.empty_cache()

        x2, y2 = self.feature_extraction2(x, y)
        x2, y2 = self.invariant_node_pooling2(x2, y2)
        f_X_all.append(x2)
        f_Y_all.append(y2)

        del x2, y2

        gc.collect()
        torch.cuda.empty_cache()

        x3, y3 = self.feature_extraction3(x, y)
        x3, y3 = self.invariant_node_pooling3(x3, y3)
        f_X_all.append(x3)
        f_Y_all.append(y3)

        del x3, y3

        gc.collect()
        torch.cuda.empty_cache()

        x4, y4 = self.feature_extraction4(x, y)
        x4, y4 = self.invariant_node_pooling4(x4, y4)
        f_X_all.append(x4)
        f_Y_all.append(y4)

        del x4, y4

        gc.collect()
        torch.cuda.empty_cache()

        R, t, s = self.hierarchical_9DoF_registration(f_X_all, f_Y_all, mu_x, mu_y)

        del f_X_all, f_Y_all

        gc.collect()
        torch.cuda.empty_cache()

        return R, t, s

    def predict(self, x, y):
        self.eval()  # Set the layers to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            return self.forward(x, y)
