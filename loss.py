import torch
import torch.nn as nn


class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super(ChamferDistanceLoss, self).__init__()

    def forward(self, x, y):
        # Expand dimensions
        x_expanded = x.unsqueeze(2)  # Shape: [B, N, 1, 3]
        y_expanded = y.unsqueeze(1)  # Shape: [B, 1, N, 3]

        # Calculate pairwise distances
        distances = torch.norm(x_expanded - y_expanded, dim=-1)  # Shape: [B, N, N]

        # Find the minimum distance from each point in x to y and from y to x
        min_distances_x_to_y, _ = torch.min(distances, dim=2)  # Shape: [B, N]
        min_distances_y_to_x, _ = torch.min(distances, dim=1)  # Shape: [B, N]

        # Calculate the Chamfer distance
        chamfer_dist = torch.mean(min_distances_x_to_y, dim=1) + torch.mean(min_distances_y_to_x, dim=1)  # Shape: [B]

        return torch.mean(chamfer_dist)


class RegistrationLoss(nn.Module):
    def __init__(self):
        super(RegistrationLoss, self).__init__()

    def forward(self, R, t, S, R_g, t_g, S_g):
        # Calculate loss for Batch of predicted R, t, S and ground truth R, t, S
        # L_R = ||R_g^T * R - I||^2 + ||t_g - t||^2 + ||S_g - S||^2

        batch_size = R.shape[0]
        identity_matrix = torch.eye(3).to(R.device).unsqueeze(0).repeat(batch_size, 1, 1)
        L_R = torch.norm(torch.bmm(R_g.transpose(1, 2), R) - identity_matrix, dim=(1, 2)) ** 2
        L_t = torch.norm(t_g - t, dim=1) ** 2
        L_S = (S_g - S) ** 2
        return torch.mean(L_R + L_t + L_S)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.chamfer_loss = ChamferDistanceLoss()
        self.registration_loss = RegistrationLoss()

    def forward(self, R, t, S, R_g, t_g, S_g, T_X, Y):
        L_R = self.registration_loss(R, t, S, R_g, t_g, S_g)
        L_CD = self.chamfer_loss(T_X, Y)
        return L_R + L_CD, L_R, L_CD
