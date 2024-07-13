import sys
import logging
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


from hegn.models.vn_layers import VNLinearLeakyReLU, VNMaxPool, mean_pool
from hegn.utils.vn_dgcnn_util import get_graph_feature

class VNDGCNN(nn.Module):
    def __init__(self, in_feat, out_feat, k, pooling='mean'):
        super(VNDGCNN, self).__init__()     
        self.n_knn = k
        self.in_feat = in_feat
        self.conv1 = VNLinearLeakyReLU(in_feat, out_feat)
        self.conv2 = VNLinearLeakyReLU(out_feat, out_feat)

        if pooling == 'max':
            self.pool = VNMaxPool(self.in_feat)
        elif pooling == 'mean':
            self.pool = mean_pool
            

    def forward(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
