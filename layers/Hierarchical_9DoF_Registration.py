import torch.nn as nn
import torch
from layers.VNTransformer import VNMLP


class GlobalPooling(nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()

    def forward(self, f_x_array):
        """
        Perform mean pooling along the node dimension.
        :param x:
        :return:
        """
        # Perform mean pooling along the node dimension
        x = [torch.mean(f_X, dim=-1) for f_X in f_x_array]
        return x


class HierarchicalAggregation(nn.Module):
    def __init__(self, global_descriptor_size=512):
        super(HierarchicalAggregation, self).__init__()
        self.vn_mlp = VNMLP(global_descriptor_size, 512)

    def forward(self, x):
        """
        Perform hierarchical aggregation of features.
        :param x:
        :return:
        """
        # Concatenate the pooled features
        x = torch.cat(x, dim=1)  # Concatenate along the C dimension
        x = self.vn_mlp(x)
        return x


class nine_DoFAlignment(nn.Module):
    def __init__(self):
        super(nine_DoFAlignment, self).__init__()

    def forward(self, x, mu_x, y, mu_y):
        # Calculate R  , t and s
        # Step 1: Compute the mean-centered features
        x = x - mu_x.unsqueeze(1)
        y = y - mu_y.unsqueeze(1)

        # Step 2: Compute the covariance matrix
        cov_matrix = torch.einsum('bcd,bce->bde', x, y)  # B*D*D

        # Step 3: Perform SVD on the covariance matrix
        U, S, Vt = torch.linalg.svd(cov_matrix)

        # Step 4: Compute the rotation matrix R
        R = torch.matmul(U, Vt)

        # Step 5: Compute the scale factor S
        x = torch.norm(x, dim=(1, 2))  # Norm over C, D, and N dimensions: B
        y = torch.norm(y, dim=(1, 2))  # Norm over C, D, and N dimensions: B
        scale_factor_S = (y / (x + 1e-6))  # Avoid division by zero

        # Step 6: Compute the translation vector t
        mu_X_rotated = torch.einsum('bij,bj->bi', R, mu_x)  # Apply the rotation matrix: B*D
        t = mu_y - mu_X_rotated  # Compute the translation vector: B*D
        return R, t, scale_factor_S


class Hierarchical_9DoF_Registration(nn.Module):
    def __init__(self, global_descriptor_size=512):
        super(Hierarchical_9DoF_Registration, self).__init__()
        self.global_pooling_x = GlobalPooling()
        self.global_pooling_y = GlobalPooling()
        self.hierarchical_aggregation = HierarchicalAggregation(global_descriptor_size=global_descriptor_size)
        self.nine_DofAlighnment = nine_DoFAlignment()

    def forward(self, x, y, mu_x, mu_y):
        x = self.global_pooling_x(x)
        y = self.global_pooling_y(y)

        x = self.hierarchical_aggregation(x)
        y = self.hierarchical_aggregation(y)

        return self.nine_DofAlighnment(x, mu_x, y, mu_y)
