import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from layers.VN_DGCNN import VNLinearLeakyReLU, get_graph_feature


def channel_equi_vec_normalize(x):
    # B,C,3,...
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    x_dir = F.normalize(x, dim=2)
    x_norm = x.norm(dim=2, keepdim=True)
    x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
    y = x_dir * x_normalized_norm
    return y


class VNMLP(nn.Module):
    def __init__(self, in_channels, out_channel, dim=3):
        super(VNMLP, self).__init__()
        # self.conv1 = VNLinearLeakyReLU(in_channels, out_channel // 16, dim=dim)
        # self.conv2 = VNLinearLeakyReLU(out_channel // 16, out_channel // 8, dim=dim)
        # self.conv3 = VNLinearLeakyReLU(out_channel // 8, out_channel // 4, dim=dim)
        # self.conv4 = VNLinearLeakyReLU(out_channel // 4, out_channel // 2, dim=dim)
        # self.conv5 = VNLinearLeakyReLU(out_channel // 2, out_channel, dim=dim)
        self.conv1 = VNLinearLeakyReLU(in_channels, out_channel, dim=dim)

    def forward(self, x):
        # Apply VN-MLP layers
        x = self.conv1(x)

        # x = self.conv2(x)
        #
        # x = self.conv3(x)
        #
        # x = self.conv4(x)
        #
        # x = self.conv5(x)

        return x


class VecDGCNNAtten(nn.Module):
    def __init__(self, in_channels, atten_multi_head_c=16, k=20):
        super(VecDGCNNAtten, self).__init__()

        self.atten_multi_head_c = atten_multi_head_c
        self.k = k
        self.vn_mlp = VNMLP(1, in_channels)
        self.vn_mlp_2 = VNMLP(2, in_channels, dim=5)

    def forward(self, x, y):
        # Compute Q, V, K
        # x_reshaped = x.repeat(1, 1, 2, 1, 1).reshape(x.size(0), x.size(1) * 2, x.size(2), x.size(3))
        x = x.unsqueeze(1)
        q_x = channel_equi_vec_normalize(self.vn_mlp(x)).contiguous()  # B,C,3,N ### Q_x

        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = get_graph_feature(y, k=self.k)
        y = self.vn_mlp_2(y).contiguous()  # v_y

        # K_y * Q_x
        qk = (channel_equi_vec_normalize(y) * q_x[..., None]).sum(2)  # B,C,N,K
        B, C, N, K = qk.shape
        N_head = C // self.atten_multi_head_c
        qk_c = qk.reshape(B, N_head, self.atten_multi_head_c, N, K)
        atten = qk_c.sum(2, keepdim=True) / np.sqrt(3 * self.atten_multi_head_c)
        atten = torch.softmax(atten, dim=-1)
        atten = atten.expand(-1, -1, self.atten_multi_head_c, -1, -1)
        atten = atten.reshape(qk.shape).unsqueeze(2)  # B,C,1,N,K
        return (atten * y).sum(-1)
        # del qk, qk_c, atten, y
        # return x


class GlobalContext(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalContext, self).__init__()

        self.in_channels = in_channels
        self.vn_mlp = VNMLP(in_channels * 2, out_channels, dim=4)

    def forward(self, x):
        F_x = torch.mean(x, dim=-1, keepdim=True).expand(
            x.size())  # Mean along the node dimension to obtain global features -> B*C*3
        x = torch.cat((x, F_x), dim=1)  # Concatenate along the C dimension -> B*(2*C)*3*N
        x = self.vn_mlp(x)
        return x
