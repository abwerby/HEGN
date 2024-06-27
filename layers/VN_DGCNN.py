import torch
import torch.nn as nn

EPS = 1e-6


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:  # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear mapping for features
        x_t = x.transpose(1, -1)  # [B, ..., N_samples, 3, N_feat]
        p = self.map_to_feat(x_t).transpose(1, -1)  # [B, N_feat, 3, N_samples, ...]

        # BatchNorm
        p = self.batchnorm(p)

        # Linear mapping for directions
        d = self.map_to_dir(x_t).transpose(1, -1)  # [B, N_feat, 1, N_samples, ...] or [B, N_feat, 3, N_samples, ...]

        # LeakyReLU operation
        dotprod = (p * d).sum(2, keepdim=True)  # [B, N_feat, 1, N_samples, ...]
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)  # [B, N_feat, 1, N_samples, ...]

        # Combine results
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class SharedVN_DGCNNLayer(nn.Module):
    def __init__(self, k=20, out_channel_ratio=3):
        super(SharedVN_DGCNNLayer, self).__init__()

        self.k = k

        self.conv1 = VNLinearLeakyReLU(2, 64 // out_channel_ratio)
        # self.conv2 = VNLinearLeakyReLU(64 // out_channel_ratio * 2, 64 // out_channel_ratio)
        # self.conv3 = VNLinearLeakyReLU(64 // out_channel_ratio * 2, 128 // out_channel_ratio)
        # self.conv4 = VNLinearLeakyReLU(128 // out_channel_ratio * 2, 256 // out_channel_ratio)

        #
        # self.conv5 = VNLinearLeakyReLU(
        #     256 // out_channel_ratio + 128 // out_channel_ratio + 64 // out_channel_ratio * 2,
        #     1024 // out_channel_ratio, dim=4, share_nonlinearity=True)

        self.pool1 = mean_pool
        self.pool2 = mean_pool
        self.pool3 = mean_pool
        self.pool4 = mean_pool

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = get_graph_feature(x, k=self.k)  # B*C*D*N*K
        x = self.conv1(x)
        x = self.pool1(x)

        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv2(x)
        # x2 = self.pool2(x)
        #
        # x = get_graph_feature(x2, k=self.k)
        # x = self.conv3(x)
        # x3 = self.pool3(x)
        #
        # x = get_graph_feature(x3, k=self.k)
        # x = self.conv4(x)
        # x4 = self.pool4(x)
        #
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        return x
