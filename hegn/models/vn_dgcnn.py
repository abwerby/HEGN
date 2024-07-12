import sys
import logging
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


from hegn.models.vn_layers import VNLinearLeakyReLU, VNMaxPool, mean_pool, VNLinearAndLeakyReLU
from hegn.utils.vn_dgcnn_util import get_graph_feature

class VNDGCNN(nn.Module):
    def __init__(self, in_feat, out_feat, k, pooling='mean'):
        super(VNDGCNN, self).__init__()     
        self.n_knn = k
        self.in_feat = in_feat
        self.conv1 = VNLinearAndLeakyReLU(in_feat, out_feat, dim=5)

        if pooling == 'max':
            self.pool1 = VNMaxPool(self.in_feat)
        elif pooling == 'mean':
            self.pool1 = mean_pool

    def forward(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.pool1(x)
        return x
