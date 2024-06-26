from torch import nn
import torch
import torch.nn.functional as F
from src.models.vn_dgcnn import VNDGCNN


def chamfer_distance(x, y):
    raise NotImplementedError


class RegistrationLoss(nn.Module):
    def __init__(self):
        super(RegistrationLoss, self).__init__()

    def forward(self, Rg, Rp, tg, tp, Sg, Sp):
        identity_matrix = torch.eye(3).to(Rg.device)
        rotation_loss = F.mse_loss(
            torch.matmul(Rg.transpose(1, 2), Rp), identity_matrix
        )
        translation_loss = F.mse_loss(tg, tp)
        scale_loss = F.mse_loss(Sg, Sp)
        return rotation_loss + translation_loss + scale_loss


class HEGNLoss(nn.Module):
    def __init__(self):
        super(HEGNLoss, self).__init__()
        self.registration_loss = RegistrationLoss()

    def forward(self, Rg, Rp, tg, tp, Sg, Sp, T_X, Y):
        L_R = self.registration_loss(Rg, Rp, tg, tp, Sg, Sp)
        L_CD = chamfer_distance(T_X, Y)
        return L_R + L_CD


class HEGN(nn.Module):
    def __init__(self, n_knn):
        super(HEGN, self).__init__()
        self.n_knn = n_knn
        self.vn_dgcnn = VNDGCNN(
            args=type("args", (), {"n_knn": n_knn, "pooling": "mean"})
        )

    def forward(self, x, y):
        f_X = self.vn_dgcnn(x)
        f_Y = self.vn_dgcnn(y)
        return f_X, f_Y
